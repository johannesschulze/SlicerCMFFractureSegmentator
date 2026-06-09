import zipfile
from pathlib import Path

import slicer

from .Models import ModelConfig


class WeightsDownloader:
    """Ensure an nnU-Net model's weights are present in Resources/ML, downloading and
    extracting a release archive on demand. Keeps the multi-hundred-MB weights out of
    the git repository (the archive root already contains the DatasetXXX/ folder, so it
    extracts straight into the ML root). Mirrors the runtime-download approach used by
    the DentalSegmentator extension."""

    def __init__(self, progressCallback=None):
        self._progress = progressCallback or (lambda msg: None)

    @staticmethod
    def isModelPresent(model: ModelConfig, mlRoot: Path) -> bool:
        """True if the config folder has the metadata and at least one fold checkpoint."""
        cfg = model.modelPath(mlRoot)
        hasMeta = (cfg / "dataset.json").exists() and (cfg / "plans.json").exists()
        hasWeights = any(cfg.glob("fold_*/checkpoint_final.pth"))
        return hasMeta and hasWeights

    def ensureModel(self, model: ModelConfig, mlRoot: Path) -> bool:
        """Return True if the weights are available, downloading them if necessary and
        possible. Returns False if missing and no download URL is configured."""
        if self.isModelPresent(model, mlRoot):
            return True
        if not model.downloadUrl:
            return False
        return self._download(model, mlRoot)

    def _download(self, model: ModelConfig, mlRoot: Path) -> bool:
        import requests

        mlRoot.mkdir(parents=True, exist_ok=True)
        zipPath = mlRoot / f"{model.datasetFolder}.zip"
        sizeHint = f" (~{model.downloadSizeMb} MB)" if model.downloadSizeMb else ""
        self._progress(f"Downloading weights for {model.displayName}{sizeHint}...")
        try:
            with requests.get(model.downloadUrl, stream=True, timeout=30) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                downloaded = 0
                nextReport = 0
                with open(zipPath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        slicer.app.processEvents()
                        if total and downloaded >= nextReport:
                            self._progress(
                                f"  {downloaded * 100 // total}%  "
                                f"({downloaded // (1024 * 1024)}/{total // (1024 * 1024)} MB)")
                            nextReport = downloaded + total // 10  # report ~ every 10%
            self._progress("Extracting weights...")
            slicer.app.processEvents()
            with zipfile.ZipFile(zipPath, "r") as archive:
                archive.extractall(mlRoot)
        except Exception as exc:  # noqa
            self._progress(f"Weight download failed: {exc}")
            return False
        finally:
            if zipPath.exists():
                try:
                    zipPath.unlink()
                except OSError:
                    pass
        return self.isModelPresent(model, mlRoot)
