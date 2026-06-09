import colorsys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import slicer
from scipy import ndimage as ndi

from .Models import ModelConfig
from .fragment_separation import separate_touching_fragments


@dataclass
class SeparationResult:
    instances: np.ndarray      # 0 = background, 1..N = fragments (volume axis order k,j,i)
    nFragments: int
    dilationMm: float          # dilation radius that produced this result


class FragmentSeparationLogic:
    """Separate touching bone fragments of a loaded segmentation using the
    fracture-line class as a cut. Wraps the pure-numpy `separate_touching_fragments`
    function (imported as a library, no external process)."""

    def __init__(self):
        # Escalation schedule for the automatic mode (mm).
        self.minDilationMm = 0.25
        self.maxDilationMm = 3.0
        self.dilationStepMm = 0.25
        self.minFragmentVolumeMm3 = 25.0
        # Split fragments that the distance-reclaim merged into spatially
        # disconnected pieces (face connectivity) into separate fragments.
        self.splitDisconnected = True

    # -- input extraction -----------------------------------------------------
    @staticmethod
    def _segmentId(label: int) -> str:
        # SlicerNNUNetLib loads segments as Segment_<labelValue>.
        return f"Segment_{label}"

    def buildSemanticLabelmap(
        self,
        segmentationNode,
        model: ModelConfig,
        referenceVolumeNode,
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """Compose an integer labelmap (bone=boneLabel, fracture=fractureLabel) from
        the bone and fracture segments, on the reference volume's voxel grid.
        Returns (labelmap, spacing) with spacing in array axis order (k,j,i)."""
        boneArr = slicer.util.arrayFromSegmentBinaryLabelmap(
            segmentationNode, self._segmentId(model.boneLabel), referenceVolumeNode)
        fracArr = slicer.util.arrayFromSegmentBinaryLabelmap(
            segmentationNode, self._segmentId(model.fractureLabel), referenceVolumeNode)
        if boneArr is None or fracArr is None:
            raise RuntimeError(
                "Could not read bone and/or fracture segment from the segmentation. "
                f"Expected segments {self._segmentId(model.boneLabel)} and "
                f"{self._segmentId(model.fractureLabel)}.")

        labelmap = np.zeros(boneArr.shape, dtype=np.int16)
        labelmap[boneArr > 0] = model.boneLabel
        labelmap[fracArr > 0] = model.fractureLabel  # fracture wins on overlap

        spacing = tuple(float(s) for s in referenceVolumeNode.GetSpacing()[::-1])
        return labelmap, spacing

    # -- separation -----------------------------------------------------------
    def separate(
        self,
        labelmap: np.ndarray,
        model: ModelConfig,
        spacing: Tuple[float, float, float],
        dilationMm: float,
    ) -> SeparationResult:
        instances = separate_touching_fragments(
            labelmap,
            fracture_label=model.fractureLabel,
            bone_labels=model.boneLabel,
            spacing=spacing,
            fracture_dilation_mm=dilationMm,
            min_fragment_volume_mm3=self.minFragmentVolumeMm3,
            reclaim_cut_voxels=True,
            reclaim_method="distance",
            sort_by_size=True,
        )
        if self.splitDisconnected:
            instances = self._splitDisconnectedFragments(instances, spacing)
        return SeparationResult(instances=instances,
                                nFragments=int(instances.max()),
                                dilationMm=dilationMm)

    def _splitDisconnectedFragments(self, instances: np.ndarray,
                                    spacing: Tuple[float, float, float]) -> np.ndarray:
        """Ensure every output fragment is a single connected region. The distance
        reclaim does not guarantee connectivity, so a fragment may end up as two
        spatially separate blobs under one label; split those into separate
        fragments. Sub-threshold specks (only when a fragment actually breaks up)
        are dropped as noise. Result is relabelled 1..N by descending size."""
        structure = ndi.generate_binary_structure(instances.ndim, 1)  # faces only
        voxelVolume = float(np.prod(spacing))
        minVoxels = max(1, int(round(self.minFragmentVolumeMm3 / voxelVolume)))

        pieces: List[np.ndarray] = []
        for label in range(1, int(instances.max()) + 1):
            mask = instances == label
            if not mask.any():
                continue
            components, n = ndi.label(mask, structure=structure)
            if n <= 1:
                pieces.append(mask)
                continue
            for c in range(1, n + 1):
                piece = components == c
                if piece.sum() >= minVoxels:
                    pieces.append(piece)

        out = np.zeros_like(instances)
        for newLabel, piece in enumerate(sorted(pieces, key=lambda m: int(m.sum()),
                                                reverse=True), start=1):
            out[piece] = newLabel
        return out

    def autoSeparate(
        self,
        labelmap: np.ndarray,
        model: ModelConfig,
        spacing: Tuple[float, float, float],
        progressCallback=None,
    ) -> SeparationResult:
        """Increase the dilation radius from minDilationMm until at least two
        fragments result (or maxDilationMm is reached). Returns the best result."""
        radii = np.arange(self.minDilationMm,
                          self.maxDilationMm + 1e-9,
                          self.dilationStepMm)
        last: Optional[SeparationResult] = None
        for radius in radii:
            radius = round(float(radius), 4)
            if progressCallback:
                progressCallback(f"Separating fragments at {radius:.2f} mm dilation...")
            last = self.separate(labelmap, model, spacing, radius)
            if last.nFragments >= 2:
                return last
        return last

    def separateCarryForward(
        self,
        labelmap: np.ndarray,
        model: ModelConfig,
        spacing: Tuple[float, float, float],
        dilationMm: float,
        previous: SeparationResult,
    ) -> SeparationResult:
        """Re-run separation while carrying forward already-separated fragments.

        Each fragment from `previous` is re-separated on its OWN bone territory at the
        new radius: a fragment that still straddles a fracture line splits further,
        while a fragment without an internal fracture bridge is kept unchanged (it is
        NOT eroded by the larger radius). This avoids over-dilating already-good
        fragments when the user manually increases the radius.

        `labelmap` is re-read from the (possibly hand-edited) segments on every call,
        so manual edits to the bone or fracture segments are honoured: previous
        fragments are clipped to the current bone, and bone the edits added that no
        previous fragment covers is reclaimed to the nearest fragment afterwards."""
        boneMask = labelmap == model.boneLabel
        fractureMask = labelmap == model.fractureLabel
        out = np.zeros(previous.instances.shape, dtype=np.int32)
        nextLabel = 1
        for frag in range(1, previous.nFragments + 1):
            fragMask = (previous.instances == frag) & boneMask  # clip to current bone
            if not fragMask.any():
                continue
            sub = np.zeros_like(labelmap)
            sub[fragMask] = model.boneLabel
            sub[fractureMask] = model.fractureLabel  # fracture lines do the cutting
            subInstances = separate_touching_fragments(
                sub,
                fracture_label=model.fractureLabel,
                bone_labels=model.boneLabel,
                spacing=spacing,
                fracture_dilation_mm=dilationMm,
                min_fragment_volume_mm3=self.minFragmentVolumeMm3,
                reclaim_cut_voxels=True,
                reclaim_method="distance",
                sort_by_size=True,
            )
            if self.splitDisconnected:
                subInstances = self._splitDisconnectedFragments(subInstances, spacing)
            nSub = int(subInstances.max())
            if nSub >= 2:
                for s in range(1, nSub + 1):
                    out[subInstances == s] = nextLabel
                    nextLabel += 1
            else:
                out[fragMask] = nextLabel  # frozen: kept exactly, not eroded
                nextLabel += 1

        # Reclaim bone the edits added that no previous fragment covered.
        leftover = boneMask & (out == 0)
        seeds = out > 0
        if leftover.any() and seeds.any():
            _, inds = ndi.distance_transform_edt(~seeds, sampling=spacing,
                                                 return_indices=True)
            nearest = out[tuple(inds)]
            out[leftover] = nearest[leftover]

        out = self._relabelBySize(out)
        return SeparationResult(instances=out, nFragments=int(out.max()),
                                dilationMm=dilationMm)

    @staticmethod
    def _relabelBySize(instances: np.ndarray) -> np.ndarray:
        counts = np.bincount(instances.ravel())
        if counts.size <= 1:
            return instances
        order = np.argsort(counts[1:])[::-1] + 1
        remap = np.zeros(counts.size, dtype=np.int32)
        for newLabel, oldLabel in enumerate(order, start=1):
            remap[oldLabel] = newLabel
        return remap[instances]

    # -- output ---------------------------------------------------------------
    @staticmethod
    def _fragmentColors(n: int) -> List[Tuple[float, float, float]]:
        return [colorsys.hsv_to_rgb((i / max(n, 1)) % 1.0, 0.55, 0.95) for i in range(n)]

    def writeFragments(
        self,
        result: SeparationResult,
        segmentationNode,
        referenceVolumeNode,
        model: ModelConfig,
        namePrefix: str = "Fragment",
    ) -> List[str]:
        """Replace the bone segment with one segment per fragment. The fracture-line
        and other anatomy segments are left untouched. Returns new segment ids."""
        segmentation = segmentationNode.GetSegmentation()

        # Remove any fragment segments from a previous run.
        existingIds = [segmentation.GetNthSegmentID(i)
                       for i in range(segmentation.GetNumberOfSegments())]
        for sid in existingIds:
            if sid.startswith("Fragment_"):
                segmentation.RemoveSegment(sid)

        # Hide the original bone segment (fragments replace it visually).
        boneSegmentId = self._segmentId(model.boneLabel)
        if segmentation.GetSegment(boneSegmentId) is not None:
            segmentationNode.GetDisplayNode().SetSegmentVisibility(boneSegmentId, False)

        colors = self._fragmentColors(result.nFragments)
        newIds: List[str] = []
        for frag in range(1, result.nFragments + 1):
            mask = (result.instances == frag).astype(np.uint8)
            segmentId = f"Fragment_{frag}"
            segmentation.AddEmptySegment(segmentId, f"{namePrefix} {frag}")
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                mask, segmentationNode, segmentId, referenceVolumeNode)
            segmentation.GetSegment(segmentId).SetColor(*colors[frag - 1])
            newIds.append(segmentId)
        return newIds
