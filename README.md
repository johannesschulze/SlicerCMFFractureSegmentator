# CMFFractureSegmentator

A [3D Slicer](https://www.slicer.org/) extension for **automatic segmentation of fractured
mandibles** from CT/CBCT scans, followed by **separation of the touching bone fragments**
along the segmented fracture lines.

Segmentation is performed by a dedicated [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) model;
the fragments are then split apart by a self-contained NumPy/SciPy post-processing step that
uses the fracture-line class as a cut.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20617995.svg)](https://doi.org/10.5281/zenodo.20617995)

<p align="center">
  <img src="CMFFractureSegmentator.png" alt="CMFFractureSegmentator" width="260">
</p>

---

## Features

- One-click pipeline: **segment → recolor → separate fragments → show in 3D**.
- Two selectable models (see [Models](#models)).
- Inference via the [SlicerNNUNet](https://github.com/lassoan/SlicerNNUNet) extension
  (CUDA / CPU / MPS), including support for `fold_all` models.
- Automatic fragment separation: the fracture-line dilation radius is increased from
  0.25 mm until at least two fragments result.
- Interactive refinement: re-run separation with a chosen radius, a one-click
  **"Re-run (+0.25 mm)"** step, and an adjustable minimum fragment volume.
- **Carry-forward re-run**: already-separated fragments are kept untouched; only the
  still-fused parts are re-cut at the new radius. This avoids over-eroding good fragments
  and improves the separation of small fragments. Manual edits (scissors etc.) to the
  bone / fracture segments are picked up on each re-run.
- Model weights are **downloaded on demand** (kept out of the repository).
- Hand-off button into the **Segment Editor** for manual touch-ups.

---

## Requirements

- **3D Slicer 5.8** or newer.
- The **PyTorch** extension (`SlicerPyTorch`).
- The **NNUNet** extension (`SlicerNNUNet`) — provides `nnUNetv2` inference inside Slicer.
- A CUDA-capable GPU is strongly recommended; CPU inference works but is slow.

Install the PyTorch and NNUNet extensions from the Slicer **Extensions Manager** and
restart Slicer before running a segmentation. On the first run, the NNUNet extension
installs the required Python packages (`torch`, `nnunetv2`) automatically.

---

## Installation

Until the extension is available in the Slicer Extensions Index, install it manually:

1. Clone or download this repository.
2. In Slicer: **Edit → Application Settings → Modules → Additional module paths**,
   add the path to `CMFFractureSegmentatorModule/`.
3. Restart Slicer. The module appears under **Segmentation → CMF Fracture Segmentator**.

---

## Usage

1. **Input volume** — select a volume already loaded in the scene, or use
   *Load volume from file…* to load one.
2. **Model** — choose the 3-class or 7-class model (see below).
3. **Device** — `cuda`, `cpu`, or `mps`.
4. Click **Run Segmentation!**
   - If the model weights are not present locally, you are offered to download them
     (~220 MB) from [Zenodo](https://zenodo.org/).
   - The volume is exported, nnU-Net runs the prediction, and the result is loaded,
     renamed, and recolored.
   - Fragments are separated automatically and shown as a 3D surface, centered in the
     3D view.
5. **Fragment separation** (collapsible section)
   - *Dilation radius* — the cut width used along the fracture line.
   - *Min. fragment volume* — fragments smaller than this (mm³) are discarded as noise.
   - *Re-run separation* / *Re-run (+0.25 mm)* — recompute with the chosen / next radius,
     carrying forward fragments that are already cleanly separated.
6. **Open in Segment Editor** — continue with manual editing if needed.

> To influence the separation result, edit the **Mandible** (bone) or **Fracture Lines**
> segments and re-run. The `Fragment_*` segments are outputs and are overwritten by a
> re-run, so make final manual touch-ups to them after the last separation.

---

## Models

Both models use the nnU-Net `3d_fullres` configuration and are hosted on Zenodo. They are
downloaded into `CMFFractureSegmentatorModule/Resources/ML/` on first use.

| Model | Classes | Bone / fracture label | Weights |
|-------|---------|-----------------------|---------|
| **3 classes** | background, mandible, fracture lines | 1 / 2 | [Zenodo](https://zenodo.org/records/20617996) |
| **7 classes** | background, maxilla & upper skull, mandible, upper teeth, lower teeth, mandibular canal (L/R), fracture lines | 2 / 7 | *(hosting pending)* |

The mandible is used as the bone to be separated; the fracture-line class is used as the cut.

---

## How fragment separation works

The numeric core lives in
[`fragment_separation.py`](CMFFractureSegmentatorModule/CMFFractureSegmentatorLib/fragment_separation.py)
and depends only on NumPy and SciPy (both bundled with Slicer):

1. The (optionally dilated) fracture-line class is removed from the bone mask, breaking the
   bridges between fragments.
2. The remaining bone is split into connected components; components below the minimum
   volume are dropped as noise.
3. The cut-away bone is reclaimed to the nearest fragment (a distance-based Voronoi fill),
   giving a gap-free instance map.
4. A connectivity pass ensures every output fragment is a single connected region.

The module wraps this as a library (no external process) in
[`FragmentSeparationLogic.py`](CMFFractureSegmentatorModule/CMFFractureSegmentatorLib/FragmentSeparationLogic.py).

---

## For developers

```
CMFFractureSegmentatorModule/
├── CMFFractureSegmentatorModule.py        # thin ScriptedLoadableModule wrapper
├── CMFFractureSegmentatorLib/
│   ├── SegmentationWidget.py              # UI and workflow
│   ├── FragmentSeparationLogic.py         # fragment-separation wrapper
│   ├── fragment_separation.py             # numeric core (NumPy/SciPy)
│   ├── WeightsDownloader.py               # on-demand weight download
│   ├── Models.py                          # model definitions (labels, colors, URLs)
│   └── Utils.py
└── Resources/
    ├── Icons/
    └── ML/                                # downloaded weights (not in git)
```

When initializing git, exclude the downloaded weights:

```gitignore
CMFFractureSegmentatorModule/Resources/ML/
```

---

## Acknowledgements

Developed at the **German Armed Forces Military Hospital Ulm**, Germany.
Inference relies on [SlicerNNUNet](https://github.com/lassoan/SlicerNNUNet); the module
structure is inspired by the [DentalSegmentator](https://github.com/gaudot/SlicerDentalSegmentator)
extension, which was also used for the initial generation of the training label maps.

This work builds on:

- Dot G, et al. *DentalSegmentator: robust open source deep learning-based CT and CBCT
  image segmentation.* Journal of Dentistry (2024). doi:[10.1016/j.jdent.2024.105130](https://doi.org/10.1016/j.jdent.2024.105130)
- Isensee F, et al. *nnU-Net: a self-configuring method for deep learning-based biomedical
  image segmentation.* Nat Methods. 2021;18(2):203-211. doi:[10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z)

The authors acknowledge support by the state of Baden-Württemberg through
[bwHPC](https://www.bwhpc.de).

## Citation

If you use this extension or its models in your research, please cite the archived
model weights:

> Schulze, J. *CMFFractureSegmentator — nnU-Net models for mandibular fracture
> segmentation.* Zenodo. https://doi.org/10.5281/zenodo.20617995

```bibtex
@dataset{schulze_cmffracturesegmentator,
  author    = {Schulze, Johannes},
  title     = {CMFFractureSegmentator --- nnU-Net models for mandibular fracture segmentation},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20617995},
  url       = {https://doi.org/10.5281/zenodo.20617995}
}
```

## Contributors

- Johannes Schulze (German Armed Forces Military Hospital Ulm, Germany)

## License

Distributed under the same permissive (BSD-style) terms as 3D Slicer — the
*3D Slicer Contribution and Software License Agreement*. See the [LICENSE](LICENSE) file.
