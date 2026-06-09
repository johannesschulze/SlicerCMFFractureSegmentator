import slicer
from slicer.ScriptedLoadableModule import *

from CMFFractureSegmentatorLib import SegmentationWidget


class CMFFractureSegmentatorModule(ScriptedLoadableModule):
    """nnU-Net based automatic segmentation and fragment separation of mandibular fractures."""

    def __init__(self, parent):
        from slicer.i18n import tr, translate

        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = tr("CMF Fracture Segmentator")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []  # SlicerNNUNet is imported lazily at runtime
        self.parent.contributors = [
            "Johannes Schulze (German Armed Forces Military Hospital Ulm, Germany)"
        ]
        self.parent.helpText = tr(
            "Automatic segmentation of fractured mandibles with a dedicated nnU-Net model, "
            "followed by separation of the touching bone fragments along the segmented "
            "fracture lines."
        )
        self.parent.acknowledgementText = tr(
            "Developed at the German Armed Forces Military Hospital Ulm. "
            "Uses the SlicerNNUNet extension for inference."
        )


class CMFFractureSegmentatorModuleWidget(ScriptedLoadableModuleWidget):
    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = None

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        widget = SegmentationWidget()
        self.logic = widget.logic
        self.layout.addWidget(widget)
        self.layout.addStretch()

    def onReload(self) -> None:
        """Reload this module AND its CMFFractureSegmentatorLib subpackage.

        Slicer's default reload only re-executes this file. The library submodules
        stay cached in sys.modules, so edits to SegmentationWidget, Models, etc. would
        not take effect. Purging them first forces a fresh import when this file is
        re-executed by reloadScriptedModule.
        """
        import sys

        packageName = "CMFFractureSegmentatorLib"
        for name in [m for m in list(sys.modules)
                     if m == packageName or m.startswith(packageName + ".")]:
            del sys.modules[name]

        super().onReload()


class CMFFractureSegmentatorModuleTest(ScriptedLoadableModuleTest):
    def runTest(self):
        slicer.util.delayDisplay("No automated tests defined for CMFFractureSegmentator yet.")
