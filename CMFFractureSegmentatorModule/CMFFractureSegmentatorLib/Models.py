from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

RGB = Tuple[float, float, float]


@dataclass
class SegmentDef:
    """Display definition for one semantic label of a model."""
    label: int          # nnU-Net label value (also the loaded Segment_<label> id suffix)
    name: str           # human-readable segment name
    color: RGB          # display color, RGB floats 0..1
    opacity: float = 1.0


@dataclass
class ModelConfig:
    """One selectable nnU-Net model and how to display / post-process its result."""
    key: str                       # short identifier
    displayName: str               # text shown in the model dropdown
    datasetFolder: str             # Resources/ML/<datasetFolder>/<configFolder>
    configFolder: str
    folds: str                     # fallback fold spec; "" means auto-detect (see detectFolds)
    segments: List[SegmentDef]
    boneLabel: int                 # label used as bone for fragment separation
    fractureLabel: int             # label used as the fracture line/cut
    downloadUrl: str = ""          # release asset (zip) to fetch if weights are missing
    downloadSizeMb: int = 0        # approximate archive size, shown in the download prompt

    def modelPath(self, mlRoot: Path) -> Path:
        """Configuration folder expected by SlicerNNUNetLib.Parameter.modelPath."""
        return mlRoot / self.datasetFolder / self.configFolder

    def detectFolds(self, mlRoot: Path, checkpoint: str = "checkpoint_final.pth") -> str:
        """Detect which folds are present on disk and return the nnU-Net `-f` string.
        Prefers a model trained on all data (`fold_all`), otherwise lists the numbered
        cross-validation folds (`fold_0`..). Falls back to `self.folds` or `"0"`."""
        cfg = self.modelPath(mlRoot)
        if (cfg / "fold_all" / checkpoint).exists():
            return "all"
        nums = [i for i in range(10) if (cfg / f"fold_{i}" / checkpoint).exists()]
        if nums:
            return ",".join(str(i) for i in nums)
        return self.folds or "0"

    def segmentDef(self, label: int) -> SegmentDef:
        for seg in self.segments:
            if seg.label == label:
                return seg
        raise KeyError(f"No segment defined for label {label} in model {self.key}")


# Shared colors
_BONE_LIGHT: RGB = (0.93, 0.87, 0.73)   # mandible
_BONE_DARK: RGB = (0.80, 0.72, 0.55)    # maxilla / upper skull
_TOOTH_UPPER: RGB = (0.96, 0.96, 0.91)
_TOOTH_LOWER: RGB = (0.90, 0.89, 0.80)
_CANAL: RGB = (0.62, 0.35, 0.80)
_FRACTURE: RGB = (0.90, 0.10, 0.10)

_CONFIG = "nnUNetTrainer__nnUNetPlans__3d_fullres"


MODEL_3CLASS = ModelConfig(
    key="3class",
    displayName="3 classes (mandible + fracture lines)",
    datasetFolder="Dataset056_MandibleFractureBig_3class_dilate3",
    configFolder=_CONFIG,
    folds="",  # auto-detected at runtime (fold_all or fold_0..N)
    segments=[
        SegmentDef(1, "Mandible", _BONE_LIGHT, opacity=0.6),
        SegmentDef(2, "Fracture Lines", _FRACTURE, opacity=1.0),
    ],
    boneLabel=1,
    fractureLabel=2,
    downloadUrl="https://zenodo.org/records/20617996/files/56_3class_big.zip",
    downloadSizeMb=220,
)

MODEL_7CLASS = ModelConfig(
    key="7class",
    displayName="7 classes (full anatomy + fracture lines)",
    datasetFolder="Dataset057_MandibleFractureBig_7class_dilate3",
    configFolder=_CONFIG,
    folds="",  # auto-detected at runtime (fold_all or fold_0..N)
    segments=[
        SegmentDef(1, "Maxilla and Upper Skull", _BONE_DARK, opacity=0.5),
        SegmentDef(2, "Mandible", _BONE_LIGHT, opacity=0.6),
        SegmentDef(3, "Upper Teeth", _TOOTH_UPPER, opacity=1.0),
        SegmentDef(4, "Lower Teeth", _TOOTH_LOWER, opacity=1.0),
        SegmentDef(5, "Mandibular canal left", _CANAL, opacity=1.0),
        SegmentDef(6, "Mandibular canal right", _CANAL, opacity=1.0),
        SegmentDef(7, "Fracture Lines", _FRACTURE, opacity=1.0),
    ],
    boneLabel=2,
    fractureLabel=7,
    # TODO: set to the GitHub Release (or Zenodo) asset URL of 57_7class_dilate3.zip
    downloadUrl="",
    downloadSizeMb=220,
)


MODELS: List[ModelConfig] = [MODEL_3CLASS, MODEL_7CLASS]


def modelByKey(key: str) -> ModelConfig:
    for model in MODELS:
        if model.key == key:
            return model
    raise KeyError(f"Unknown model key: {key}")
