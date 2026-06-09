"""
fragment_separation.py

Separate touching bone fragments from a multi-class semantic segmentation by
using a segmented fracture-line class as a cut, then reclaiming the cut voxels
either by nearest-fragment distance or by a marker-controlled watershed.

Design intent
-------------
The numeric core (`separate_touching_fragments`) depends only on numpy + scipy
for the default "distance" reclaim, both of which ship inside 3D Slicer's bundled
Python. The optional "watershed" reclaim additionally uses SimpleITK, which is
also bundled with Slicer. The core takes arrays in and returns arrays out and
knows nothing about Slicer, VTK, or file formats, so it stays unit-testable and
drops straight into a Slicer module's Logic class later.

Two thin, OPTIONAL adapters live at the bottom:
  * a SimpleITK/NIfTI loader+saver for offline CLI testing on nnU-Net output, and
  * a Slicer adapter showing how to pull arrays from / push results back into nodes.
Both are guarded so importing this file never fails when those packages are absent.

Axis-order contract
--------------------
`labelmap` is indexed [axis0, axis1, axis2] and `spacing` MUST be given in the
SAME axis order, as positive physical sizes (e.g. mm). Watch out:
  * SimpleITK:  sitk.GetArrayFromImage -> (z, y, x); img.GetSpacing() -> (x, y, z),
                so pass spacing = img.GetSpacing()[::-1].
  * Slicer:     slicer.util.arrayFromVolume -> (k, j, i) ~ (z, y, x);
                node.GetSpacing() -> (i, j, k) ~ (x, y, z), so reverse it too.
The adapters below handle this reversal for you.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Optional, Sequence, Union

import numpy as np
from scipy import ndimage as ndi

LabelSpec = Optional[Union[int, Iterable[int]]]


def _physical_dilate(mask: np.ndarray, radius_mm: float,
                     spacing: Sequence[float]) -> np.ndarray:
    """Dilate a binary mask by an isotropic physical radius, respecting anisotropic
    voxel spacing. Implemented via the EDT of the complement, so no structuring
    element needs to be built and anisotropy is handled exactly."""
    if radius_mm <= 0:
        return mask
    dist_outside = ndi.distance_transform_edt(~mask, sampling=spacing)
    return dist_outside <= radius_mm


def _warn_geometry_mismatch_sitk(label_img, intensity_img, atol: float = 1e-4) -> None:
    """Warn if a SimpleITK label and intensity image are not on the same voxel grid.
    A mismatch puts the gradient ridge in the wrong place for watershed reclaim."""
    issues = []
    if label_img.GetSize() != intensity_img.GetSize():
        issues.append(f"size {label_img.GetSize()} vs {intensity_img.GetSize()}")
    if not np.allclose(label_img.GetSpacing(), intensity_img.GetSpacing(), atol=atol):
        issues.append(f"spacing {label_img.GetSpacing()} vs {intensity_img.GetSpacing()}")
    if not np.allclose(label_img.GetOrigin(), intensity_img.GetOrigin(), atol=atol):
        issues.append(f"origin {label_img.GetOrigin()} vs {intensity_img.GetOrigin()}")
    if not np.allclose(label_img.GetDirection(), intensity_img.GetDirection(), atol=atol):
        issues.append("direction differs")
    if issues:
        warnings.warn(
            "Label and intensity volumes are not on the same grid; watershed reclaim "
            "may place the gradient ridge incorrectly. Resample onto a common grid "
            "first. Mismatches: " + "; ".join(issues),
            RuntimeWarning, stacklevel=3)


def _warn_geometry_mismatch_slicer(label_node, intensity_node, atol: float = 1e-4) -> None:
    """Warn if two Slicer volume nodes differ in dimensions, spacing, or IJKToRAS."""
    import vtk

    issues = []
    dims_a = label_node.GetImageData().GetDimensions()
    dims_b = intensity_node.GetImageData().GetDimensions()
    if dims_a != dims_b:
        issues.append(f"dimensions {dims_a} vs {dims_b}")
    if not np.allclose(label_node.GetSpacing(), intensity_node.GetSpacing(), atol=atol):
        issues.append(f"spacing {label_node.GetSpacing()} vs {intensity_node.GetSpacing()}")
    m_a, m_b = vtk.vtkMatrix4x4(), vtk.vtkMatrix4x4()
    label_node.GetIJKToRASMatrix(m_a)
    intensity_node.GetIJKToRASMatrix(m_b)
    a = np.array([[m_a.GetElement(i, j) for j in range(4)] for i in range(4)])
    b = np.array([[m_b.GetElement(i, j) for j in range(4)] for i in range(4)])
    if not np.allclose(a, b, atol=atol):
        issues.append("IJKToRAS differs")
    if issues:
        warnings.warn(
            "Label and intensity volumes are not on the same grid; watershed reclaim "
            "may place the gradient ridge incorrectly. Resample onto a common grid "
            "first. Mismatches: " + "; ".join(issues),
            RuntimeWarning, stacklevel=3)


def _reclaim_distance(full_bone: np.ndarray, instances: np.ndarray,
                      spacing: Sequence[float]) -> np.ndarray:
    """Assign every unlabelled bone voxel to the nearest fragment by physical
    Euclidean distance (a feature-transform Voronoi fill). numpy + scipy only."""
    seeds = instances > 0
    to_fill = full_bone & ~seeds
    if to_fill.any() and seeds.any():
        _, inds = ndi.distance_transform_edt(~seeds, sampling=spacing,
                                             return_indices=True)
        nearest = instances[tuple(inds)]
        instances = instances.copy()
        instances[to_fill] = nearest[to_fill]
    return instances


def _reclaim_watershed(full_bone: np.ndarray, instances: np.ndarray,
                       spacing: Sequence[float], image: Optional[np.ndarray] = None,
                       gradient_sigma_mm: float = 0.5) -> np.ndarray:
    """Assign unlabelled bone voxels with a marker-controlled watershed.

    The surviving fragments are the markers. The flood follows a height image:
      * if `image` (the grayscale CT/CBCT) is given, the height is its smoothed
        gradient magnitude, so the watershed line settles on the true fracture
        surface (the intensity ridge), and the strong bone/air gradient naturally
        walls the flood inside the bone, or
      * otherwise the height is the inverted distance transform of the bone, so
        fragment interiors are basins and the medial surface between them a ridge.

    Marker voxels keep their labels; only the gap (zeros inside bone) is filled.
    Requires SimpleITK (bundled with Slicer)."""
    try:
        import SimpleITK as sitk
    except ImportError as e:
        raise ImportError(
            "reclaim_method='watershed' needs SimpleITK (bundled with 3D Slicer; "
            "offline: pip install SimpleITK)."
        ) from e

    sitk_spacing = tuple(float(s) for s in spacing[::-1])  # (z,y,x) -> (x,y,z)

    if image is not None:
        img = np.asarray(image)
        if img.shape != instances.shape:
            raise ValueError("intensity image shape must match labelmap shape")
        feat = sitk.GetImageFromArray(img.astype(np.float32))
        feat.SetSpacing(sitk_spacing)
        feat = sitk.GradientMagnitudeRecursiveGaussian(feat,
                                                       sigma=float(gradient_sigma_mm))
    else:
        dist = ndi.distance_transform_edt(full_bone, sampling=spacing).astype(np.float32)
        height = -dist
        height[~full_bone] = float(height.max()) + 1.0  # discourage outside flooding
        feat = sitk.GetImageFromArray(height)
        feat.SetSpacing(sitk_spacing)

    markers = sitk.GetImageFromArray(instances.astype(np.uint16))
    markers.SetSpacing(sitk_spacing)

    ws = sitk.MorphologicalWatershedFromMarkers(
        feat, markers, markWatershedLine=False, fullyConnected=False)
    ws_arr = sitk.GetArrayFromImage(ws).astype(np.int32)
    ws_arr[~full_bone] = 0  # keep labels only inside the bone region
    return ws_arr


def separate_touching_fragments(
    labelmap: np.ndarray,
    fracture_label: int,
    bone_labels: LabelSpec = None,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    fracture_dilation_mm: float = 0.0,
    fracture_is_bone: bool = False,
    min_fragment_volume_mm3: float = 50.0,
    connectivity: int = 1,
    reclaim_cut_voxels: bool = True,
    reclaim_method: str = "distance",
    image: Optional[np.ndarray] = None,
    gradient_sigma_mm: float = 0.5,
    sort_by_size: bool = True,
) -> np.ndarray:
    """Split touching bone fragments into separate instance labels.

    Parameters
    ----------
    labelmap : np.ndarray
        Integer semantic segmentation. Background is assumed to be 0.
    fracture_label : int
        Class index of the fracture line/surface. A (dilated) copy of it is the cut.
        By default it is treated as NON-bone, so the fracture voxels remain an
        unlabelled gap and are never assigned to a fragment (see `fracture_is_bone`).
    bone_labels : int | iterable[int] | None
        Which label(s) count as bone. If None, every nonzero label except
        `fracture_label` is treated as bone (the common nnU-Net 3-class setup).
    spacing : sequence of float
        Voxel spacing in the SAME axis order as `labelmap` (see module docstring).
    fracture_dilation_mm : float
        Grow the fracture line by this physical radius before cutting. When the
        fracture label is NON-bone (the default), the bone fragments often touch
        directly and the bare fracture voxels are not part of the bone, so a raw
        subtraction removes nothing: you MUST set this > 0 (e.g. 0.3-1.0 mm) to eat
        into the bone and break the bridge. The bone the dilation cuts away is what
        gets reclaimed; the fracture voxels themselves stay a gap.
    fracture_is_bone : bool
        If False (default): the fracture label is a non-bone gap. Final instances
        cover only `bone_labels`; the fracture voxels are left unlabelled, and only
        the dilation margin (the cut bone overlapping the fracture) is reclaimed.
        If True: the fracture voxels are part of the bone and get reclaimed into
        fragments too (use this only if your fracture label sits inside the bone).
    min_fragment_volume_mm3 : float
        Connected components smaller than this (after the cut) are discarded as
        noise before reclaim. Set to 0 to keep everything.
    connectivity : int
        ndi connectivity for labelling: 1 = faces (conservative), 2 = +edges,
        3 = +corners. Lower is safer against diagonal re-bridging.
    reclaim_cut_voxels : bool
        If True, the cut-away bone is assigned back to the nearest fragment, giving
        a gap-free instance map over the bone. If False, it is left as background.
        Never reclaims fracture voxels unless `fracture_is_bone` is True.
    reclaim_method : {"distance", "watershed"}
        "distance": nearest-fragment Euclidean fill (numpy+scipy, fast, ignores
        intensity). Good for cleanly displaced fragments.
        "watershed": marker-controlled watershed via SimpleITK, following the
        intensity gradient when `image` is supplied. Better for interlocked or
        compressed comminuted fragments where the true surface is irregular.
    image : np.ndarray | None
        Grayscale CT/CBCT volume, same shape/axis-order as `labelmap`. Used only by
        the watershed reclaim to build the gradient height image. If None, the
        watershed falls back to a distance-based height.
    gradient_sigma_mm : float
        Gaussian sigma (physical mm) for the gradient-magnitude height image.
    sort_by_size : bool
        If True, relabel so 1 = largest fragment, 2 = next, etc. Reproducible
        ordering and a free "main vs minor fragment" notion.

    Returns
    -------
    instances : np.ndarray (int32)
        0 = background, 1..N = fragments. Same shape as input.
    """
    labelmap = np.asarray(labelmap)
    spacing = tuple(float(s) for s in spacing)
    if len(spacing) != labelmap.ndim:
        raise ValueError(f"spacing {spacing} does not match labelmap ndim {labelmap.ndim}")
    if reclaim_method not in ("distance", "watershed"):
        raise ValueError("reclaim_method must be 'distance' or 'watershed'")

    # 1. Bone mask. By default the fracture label is NOT bone.
    if bone_labels is None:
        bone_mask = (labelmap != 0) & (labelmap != fracture_label)
    else:
        if np.isscalar(bone_labels):
            bone_labels = [int(bone_labels)]
        bone_mask = np.isin(labelmap, list(bone_labels))
    fracture_mask = labelmap == fracture_label

    # Region the final instances should cover. With a non-bone fracture label the
    # fracture voxels stay an unlabelled gap, so only the bone is reclaimed; the
    # dilation margin that the cut removes is bone and therefore reclaimed.
    full_bone = (bone_mask | fracture_mask) if fracture_is_bone else bone_mask

    if (not fracture_is_bone) and reclaim_cut_voxels and fracture_dilation_mm <= 0:
        warnings.warn(
            "fracture_is_bone=False with fracture_dilation_mm<=0: the fracture label "
            "is not part of the bone, so the cut removes no bone and directly touching "
            "fragments will not be separated. Set fracture_dilation_mm > 0.",
            RuntimeWarning, stacklevel=2)

    # 2. Cut: remove the (optionally dilated) fracture line to break bridges.
    cut = _physical_dilate(fracture_mask, fracture_dilation_mm, spacing)
    cut_bone = bone_mask & ~cut

    # 3. Connected components -> candidate fragments.
    structure = ndi.generate_binary_structure(labelmap.ndim, connectivity)
    instances, n = ndi.label(cut_bone, structure=structure)
    if n == 0:
        return np.zeros_like(labelmap, dtype=np.int32)

    # 4. Drop small components as noise.
    voxel_volume = float(np.prod(spacing))
    if min_fragment_volume_mm3 > 0:
        min_voxels = max(1, int(round(min_fragment_volume_mm3 / voxel_volume)))
        counts = np.bincount(instances.ravel())
        too_small = np.where(counts < min_voxels)[0]
        too_small = too_small[too_small != 0]
        if too_small.size:
            instances[np.isin(instances, too_small)] = 0

    instances, n = ndi.label(instances > 0, structure=structure)  # relabel 1..N
    if n == 0:
        return np.zeros_like(labelmap, dtype=np.int32)

    # 5. Reclaim the cut voxels.
    if reclaim_cut_voxels:
        if reclaim_method == "distance":
            instances = _reclaim_distance(full_bone, instances, spacing)
        else:
            instances = _reclaim_watershed(full_bone, instances, spacing,
                                           image=image,
                                           gradient_sigma_mm=gradient_sigma_mm)

    instances = instances.astype(np.int32)

    # 6. Optional size ordering (largest fragment = 1).
    if sort_by_size:
        counts = np.bincount(instances.ravel())
        if counts.size > 1:
            order = np.argsort(counts[1:])[::-1] + 1
            remap = np.zeros(counts.size, dtype=np.int32)
            for new_id, old_id in enumerate(order, start=1):
                remap[old_id] = new_id
            instances = remap[instances]

    return instances


# ---------------------------------------------------------------------------
# OPTIONAL adapter A: SimpleITK / NIfTI, for offline testing on nnU-Net output.
# ---------------------------------------------------------------------------
def separate_from_nifti(in_path: str, out_path: str, fracture_label: int,
                        intensity_path: Optional[str] = None, **kwargs):
    """Load an nnU-Net .nii.gz labelmap, separate fragments, save the instance map.
    Pass `intensity_path` (the matching grayscale volume) to enable gradient-guided
    watershed reclaim. Preserves geometry. Requires SimpleITK."""
    import SimpleITK as sitk

    img = sitk.ReadImage(in_path)
    arr = sitk.GetArrayFromImage(img)              # (z, y, x)
    spacing = tuple(img.GetSpacing()[::-1])        # (x,y,z) -> (z,y,x)

    intensity = None
    if intensity_path is not None:
        intensity_img = sitk.ReadImage(intensity_path)
        _warn_geometry_mismatch_sitk(img, intensity_img)
        intensity = sitk.GetArrayFromImage(intensity_img)

    instances = separate_touching_fragments(arr, fracture_label, spacing=spacing,
                                            image=intensity, **kwargs)

    out = sitk.GetImageFromArray(instances)
    out.CopyInformation(img)                       # keep spacing/origin/direction
    sitk.WriteImage(out, out_path)
    return int(instances.max())


# ---------------------------------------------------------------------------
# OPTIONAL adapter B: 3D Slicer. Sketch of how a scripted module's Logic would
# call the core. Imports are local so this file loads fine outside Slicer.
# ---------------------------------------------------------------------------
def separate_in_slicer(labelmap_volume_node, fracture_label: int,
                       intensity_volume_node=None,
                       output_segmentation_node=None, **kwargs):
    """Run separation on a vtkMRMLLabelMapVolumeNode and return an instance array;
    optionally write each fragment back as its own segment. Pass the source/master
    grayscale volume node as `intensity_volume_node` to use watershed reclaim.

    In a real module this body lives in YourModuleLogic.process(...)."""
    import slicer

    arr = slicer.util.arrayFromVolume(labelmap_volume_node)   # (k,j,i) ~ (z,y,x)
    spacing = tuple(labelmap_volume_node.GetSpacing()[::-1])  # (i,j,k) -> (z,y,x)

    intensity = None
    if intensity_volume_node is not None:
        _warn_geometry_mismatch_slicer(labelmap_volume_node, intensity_volume_node)
        intensity = slicer.util.arrayFromVolume(intensity_volume_node)

    instances = separate_touching_fragments(arr, fracture_label, spacing=spacing,
                                            image=intensity, **kwargs)

    out_volume = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLLabelMapVolumeNode", "FragmentInstances")
    out_volume.CopyOrientation(labelmap_volume_node)
    out_volume.SetSpacing(labelmap_volume_node.GetSpacing())
    slicer.util.updateVolumeFromArray(out_volume, instances)

    if output_segmentation_node is not None:
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            out_volume, output_segmentation_node)

    return instances


def _run_selftest():
    """Quick synthetic smoke-test (two scenarios from the original __main__)."""
    lm = np.zeros((40, 40, 40), dtype=np.int16)
    lm[5:35, 5:35, 5:35] = 1
    lm[5:35, 5:35, 19:21] = 2
    img = np.full((40, 40, 40), -1000.0, dtype=np.float32)
    img[lm == 1] = 1000.0
    img[lm == 2] = -200.0
    common = dict(bone_labels=1, spacing=(0.3, 0.3, 0.3),
                  fracture_dilation_mm=0.3, min_fragment_volume_mm3=1.0)
    fracture = lm == 2
    bone = lm == 1
    for method, image in (("distance", None), ("watershed", img), ("watershed/no-image", None)):
        m = "watershed" if method.startswith("watershed") else "distance"
        try:
            out = separate_touching_fragments(lm, fracture_label=2,
                                              reclaim_method=m, image=image, **common)
        except ImportError as e:
            print(f"[{method}] skipped:", e)
            continue
        print(f"[{method:18}] fragments:", int(out.max()),
              "| bone holes:", int((bone & (out == 0)).sum()),
              "| fracture voxels labelled as bone:", int((fracture & (out > 0)).sum()))

    lm2 = np.zeros((40, 40, 40), dtype=np.int16)
    lm2[5:35, 5:35, 5:35] = 1
    lm2[5:35, 5:33, 19] = 2
    bd0 = separate_touching_fragments(lm2, fracture_label=2, bone_labels=1,
                                      spacing=(0.3, 0.3, 0.3), fracture_dilation_mm=0.0,
                                      min_fragment_volume_mm3=1.0, reclaim_cut_voxels=False)
    bd1 = separate_touching_fragments(lm2, fracture_label=2, bone_labels=1,
                                      spacing=(0.3, 0.3, 0.3), fracture_dilation_mm=0.7,
                                      min_fragment_volume_mm3=1.0)
    print("[bridge, no dilation] fragments:", int(bd0.max()), "(expected 1: bridge intact)")
    print("[bridge, 0.7 mm cut ] fragments:", int(bd1.max()), "(expected 2: bridge cut)",
          "| fracture voxels labelled:", int(((lm2 == 2) & (bd1 > 0)).sum()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _process_one(src: "Path", dst: "Path", fracture_label: int,
                 bone_labels, intensity_dir: "Optional[Path]",
                 kwargs: dict) -> tuple[str, int, str]:
    """Worker for one file. Returns (filename, n_fragments, error_or_empty)."""
    try:
        import SimpleITK as sitk

        img = sitk.ReadImage(str(src))
        arr = sitk.GetArrayFromImage(img)           # (z, y, x)
        spacing = tuple(img.GetSpacing()[::-1])     # (x,y,z) → (z,y,x)

        intensity = None
        if intensity_dir is not None:
            # try exact name match (e.g. CASE_0000.nrrd), else CASE.nrrd
            candidates = [
                intensity_dir / src.name,
                intensity_dir / (src.stem + "_0000.nrrd"),
            ]
            for c in candidates:
                if c.exists():
                    int_img = sitk.ReadImage(str(c))
                    _warn_geometry_mismatch_sitk(img, int_img)
                    intensity = sitk.GetArrayFromImage(int_img)
                    break

        instances = separate_touching_fragments(
            arr,
            fracture_label=fracture_label,
            bone_labels=bone_labels,
            spacing=spacing,
            image=intensity,
            **kwargs,
        )

        out = sitk.GetImageFromArray(instances)
        out.CopyInformation(img)
        dst.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(out, str(dst), useCompression=True)
        return src.name, int(instances.max()), ""

    except Exception as exc:
        return src.name, 0, str(exc)


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description=(
            "Separate touching bone fragments in nnU-Net NRRD segmentations.\n"
            "Uses the fracture-line class as a cut, then reclaims cut voxels by\n"
            "nearest-neighbour distance or marker-controlled watershed.\n"
            "Input can be a single .nrrd file or a directory of .nrrd files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── I/O ──────────────────────────────────────────────────────────────────
    parser.add_argument("-i", "--input", required="--selftest" not in __import__("sys").argv,
                        help="Input .nrrd file OR directory of .nrrd files")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file (single input) or directory (batch). "
                             "Default: <input>_instances.nrrd or <input_dir>_instances/")
    parser.add_argument("--intensity-dir", default=None,
                        help="Directory with grayscale CT volumes for watershed reclaim "
                             "(matched by filename or <stem>_0000.nrrd). Optional.")

    # ── algorithm ────────────────────────────────────────────────────────────
    parser.add_argument("--fracture-label", type=int, default=2,
                        help="Semantic label of the fracture line (default: 2)")
    parser.add_argument("--bone-labels", type=int, nargs="+", default=None,
                        help="Bone label(s). Default: all non-zero except fracture-label.")
    parser.add_argument("--fracture-dilation-mm", type=float, default=0.5,
                        help="Grow fracture mask by this radius before cutting (default: 0.5 mm)")
    parser.add_argument("--fracture-is-bone", action="store_true",
                        help="Treat fracture voxels as bone (reclaim them into fragments)")
    parser.add_argument("--min-fragment-volume-mm3", type=float, default=50.0,
                        help="Drop components smaller than this (default: 50 mm³)")
    parser.add_argument("--connectivity", type=int, default=1, choices=[1, 2, 3],
                        help="Connected-components connectivity: 1=faces, 2=+edges, "
                             "3=+corners (default: 1)")
    parser.add_argument("--no-reclaim", action="store_true",
                        help="Do not reclaim cut-away bone voxels")
    parser.add_argument("--reclaim-method", default="distance",
                        choices=["distance", "watershed"],
                        help="Reclaim strategy (default: distance)")
    parser.add_argument("--gradient-sigma-mm", type=float, default=0.5,
                        help="Gaussian sigma for watershed gradient height (default: 0.5 mm)")
    parser.add_argument("--no-sort-by-size", action="store_true",
                        help="Do not relabel fragments by descending size")

    # ── execution ────────────────────────────────────────────────────────────
    parser.add_argument("--jobs", type=int, default=-1,
                        help="Parallel workers for batch mode (-1 = all cores)")
    parser.add_argument("--selftest", action="store_true",
                        help="Run synthetic smoke-tests and exit")

    args = parser.parse_args()

    if args.selftest:
        _run_selftest()
        return

    from pathlib import Path
    from joblib import Parallel, delayed

    input_path = Path(args.input)
    intensity_dir = Path(args.intensity_dir) if args.intensity_dir else None

    # ── collect files ─────────────────────────────────────────────────────────
    if input_path.is_file():
        src_files = [input_path]
        default_output = input_path.parent / (input_path.stem + "_instances.nrrd")
        single_mode = True
    elif input_path.is_dir():
        src_files = sorted(input_path.glob("*.nrrd"))
        default_output = input_path.parent / (input_path.name + "_instances")
        single_mode = False
    else:
        parser.error(f"Input not found: {input_path}")

    if not src_files:
        print(f"No .nrrd files found in {input_path}")
        return

    # ── output paths ──────────────────────────────────────────────────────────
    output_path = Path(args.output) if args.output else default_output

    def dst_for(src: Path) -> Path:
        if single_mode:
            return output_path
        return output_path / src.name

    # ── shared kwargs ─────────────────────────────────────────────────────────
    sep_kwargs = dict(
        fracture_dilation_mm=args.fracture_dilation_mm,
        fracture_is_bone=args.fracture_is_bone,
        min_fragment_volume_mm3=args.min_fragment_volume_mm3,
        connectivity=args.connectivity,
        reclaim_cut_voxels=not args.no_reclaim,
        reclaim_method=args.reclaim_method,
        gradient_sigma_mm=args.gradient_sigma_mm,
        sort_by_size=not args.no_sort_by_size,
    )

    print(f"\n{'='*70}")
    print("FRAGMENT SEPARATION")
    print(f"{'='*70}")
    print(f"Input:              {input_path}")
    print(f"Output:             {output_path}")
    print(f"Files:              {len(src_files)}")
    print(f"Fracture label:     {args.fracture_label}")
    print(f"Bone labels:        {args.bone_labels or 'all non-zero except fracture'}")
    print(f"Fracture dilation:  {args.fracture_dilation_mm} mm")
    print(f"Min fragment vol:   {args.min_fragment_volume_mm3} mm³")
    print(f"Reclaim method:     {'none' if args.no_reclaim else args.reclaim_method}")
    if intensity_dir:
        print(f"Intensity dir:      {intensity_dir}")
    print(f"{'='*70}\n")

    results = Parallel(n_jobs=args.jobs if not single_mode else 1, verbose=1)(
        delayed(_process_one)(
            src, dst_for(src), args.fracture_label, args.bone_labels,
            intensity_dir, sep_kwargs
        )
        for src in src_files
    )

    print(f"\n{'='*70}")
    errors = [(n, e) for n, _, e in results if e]
    ok = [(n, f) for n, f, e in results if not e]

    for name, n_frag, err in results:
        if err:
            print(f"  ERROR  {name}: {err}")
        else:
            print(f"  OK     {name}: {n_frag} fragment(s)")

    print(f"\nSuccessful: {len(ok)}/{len(src_files)}")
    if errors:
        print(f"Errors:     {len(errors)}")
    print(f"Output:     {output_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
