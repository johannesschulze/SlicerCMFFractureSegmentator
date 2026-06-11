from pathlib import Path
from typing import Optional

import numpy as np
import qt
import slicer

from .FragmentSeparationLogic import FragmentSeparationLogic, SeparationResult
from .Models import MODELS, ModelConfig, modelByKey
from .Utils import addInCollapsibleLayout, createButton


class SegmentationWidget(qt.QWidget):
    """End-to-end UI: pick/load a volume, choose an nnU-Net model, segment via
    SlicerNNUNet, recolor segments, separate fracture fragments, and hand off to
    the Segment Editor."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.segmentationLogic = self._createSegmentationLogic()
        self.logic = self.segmentationLogic  # exposed to the top-level module wrapper
        self.fragmentLogic = FragmentSeparationLogic()

        self._currentSegmentationNode = None
        self._referenceVolumeNode = None
        self._currentModel: Optional[ModelConfig] = None
        self._semanticLabelmap = None
        self._semanticSpacing = None
        self._lastResult = None
        self._defaultMinVolumeMm3 = self.fragmentLogic.minFragmentVolumeMm3
        self._restoring = False  # guard against writing while restoring controls
        self.isStopping = False
        self.fullInfoLogs = []

        self._buildGui()
        self._connectSegmentationLogic()
        self.onInputChanged()

        # Restore persisted state and keep it in sync when a scene is loaded;
        # reset the module when the scene is closed (Ctrl-W).
        self._sceneImportObserver = slicer.mrmlScene.AddObserver(
            slicer.mrmlScene.EndImportEvent, self._onSceneEndImport)
        self._sceneCloseObserver = slicer.mrmlScene.AddObserver(
            slicer.mrmlScene.EndCloseEvent, self._onSceneEndClose)
        self._connectParameterPersistence()
        self._restoreParameters()

    def __del__(self):
        for observer in (getattr(self, "_sceneImportObserver", None),
                         getattr(self, "_sceneCloseObserver", None)):
            try:
                slicer.mrmlScene.RemoveObserver(observer)
            except Exception:
                pass
        super().__del__()

    # ------------------------------------------------------------------ GUI --
    def _buildGui(self):
        layout = qt.QVBoxLayout(self)

        # --- Logo ------------------------------------------------------------
        logoLabel = qt.QLabel()
        logoPath = (Path(__file__).parent / ".." / "Resources" / "Icons"
                    / "CMFFractureSegmentatorModule_wide.png").resolve()
        pixmap = qt.QPixmap(str(logoPath))
        if not pixmap.isNull():
            logoLabel.setPixmap(pixmap.scaledToWidth(300, qt.Qt.SmoothTransformation))
        logoLabel.setAlignment(qt.Qt.AlignCenter)
        layout.addWidget(logoLabel)
        layout.addSpacing(30)

        # --- Input -----------------------------------------------------------
        self.inputSelector = slicer.qMRMLNodeComboBox(self)
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.showHidden = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputChanged)

        # Output segmentation: empty selection means "create a new one on Run".
        self.segmentationSelector = slicer.qMRMLNodeComboBox(self)
        self.segmentationSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.segmentationSelector.selectNodeUponCreation = True
        self.segmentationSelector.addEnabled = True
        self.segmentationSelector.removeEnabled = True
        self.segmentationSelector.renameEnabled = True
        self.segmentationSelector.noneEnabled = True
        self.segmentationSelector.noneDisplay = "Create new segmentation on Run"
        self.segmentationSelector.showHidden = False
        self.segmentationSelector.setMRMLScene(slicer.mrmlScene)
        self.segmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)",
                                          self._onSegmentationSelectorChanged)

        self.modelComboBox = qt.QComboBox(self)
        for model in MODELS:
            self.modelComboBox.addItem(model.displayName, model.key)

        self.deviceComboBox = qt.QComboBox(self)
        self.deviceComboBox.addItems(["cuda", "cpu", "mps"])

        self.inputWidget = qt.QWidget(self)
        inputLayout = qt.QFormLayout(self.inputWidget)
        inputLayout.setContentsMargins(0, 0, 0, 0)
        inputLayout.addRow("Input volume:", self.inputSelector)
        inputLayout.addRow(createButton("Load volume from file...", callback=self.onLoadVolume,
                                        toolTip="Load a volume from disk into the scene."))
        inputLayout.addRow("Segmentation:", self.segmentationSelector)
        inputLayout.addRow("Model:", self.modelComboBox)
        inputLayout.addRow("Device:", self.deviceComboBox)
        layout.addWidget(self.inputWidget)

        # --- Apply / Stop ----------------------------------------------------
        self.applyButton = createButton("Run Segmentation!", callback=self.onApplyClicked,
                                        toolTip="Run nnU-Net segmentation and fragment separation.")
        layout.addWidget(self.applyButton)

        self.stopButton = createButton("Stop", callback=self.onStopClicked,
                                       toolTip="Stop the running segmentation.")
        self.infoTextEdit = qt.QTextEdit()
        self.infoTextEdit.setReadOnly(True)
        self.infoTextEdit.setLineWrapMode(qt.QTextEdit.NoWrap)
        self.stopWidget = qt.QWidget(self)
        stopLayout = qt.QVBoxLayout(self.stopWidget)
        stopLayout.setContentsMargins(0, 0, 0, 0)
        stopLayout.addWidget(self.stopButton)
        stopLayout.addWidget(self.infoTextEdit)
        self.stopWidget.setVisible(False)
        layout.addWidget(self.stopWidget)

        # --- Fragment separation --------------------------------------------
        self.separationWidget = qt.QWidget(self)
        sepLayout = qt.QFormLayout(self.separationWidget)
        sepLayout.setContentsMargins(0, 0, 0, 0)

        self.fragmentCountLabel = qt.QLabel("-")
        sepLayout.addRow("Fragments found:", self.fragmentCountLabel)

        self.dilationSpinBox = qt.QDoubleSpinBox()
        self.dilationSpinBox.setRange(0.0, 10.0)
        self.dilationSpinBox.setSingleStep(self.fragmentLogic.dilationStepMm)
        self.dilationSpinBox.setDecimals(2)
        self.dilationSpinBox.setSuffix(" mm")
        self.dilationSpinBox.setValue(self.fragmentLogic.minDilationMm)
        sepLayout.addRow("Dilation radius:", self.dilationSpinBox)

        self.minVolumeSpinBox = qt.QDoubleSpinBox()
        self.minVolumeSpinBox.setRange(0.0, 10000.0)
        self.minVolumeSpinBox.setSingleStep(5.0)
        self.minVolumeSpinBox.setDecimals(1)
        self.minVolumeSpinBox.setSuffix(" mm³")
        self.minVolumeSpinBox.setValue(self.fragmentLogic.minFragmentVolumeMm3)
        self.minVolumeSpinBox.setToolTip(
            "Fragments smaller than this are discarded as noise. Lower it to keep small "
            "fragments. Note: a given voxel count corresponds to a different mm³ per spacing.")
        sepLayout.addRow("Min. fragment volume:", self.minVolumeSpinBox)

        reRunWidget = qt.QWidget()
        reRunLayout = qt.QHBoxLayout(reRunWidget)
        reRunLayout.setContentsMargins(0, 0, 0, 0)
        reRunLayout.addWidget(createButton(
            "Re-run separation", callback=self.onReSeparate,
            toolTip="Re-run fragment separation with the chosen dilation radius "
                    "(keeps already-separated fragments unchanged)."))
        reRunLayout.addWidget(createButton(
            "Re-run (+0.25 mm)", callback=self.onReSeparatePlus,
            toolTip="Increase the dilation radius by one step and re-run."))
        sepLayout.addRow(reRunWidget)
        self.separationWidget.setEnabled(False)
        addInCollapsibleLayout(self.separationWidget, layout, "Fragment separation", isCollapsed=False)

        # --- Segment Editor hand-off ----------------------------------------
        self.segmentEditorButton = createButton("Open in Segment Editor",
                                                callback=self.onOpenSegmentEditor,
                                                toolTip="Switch to the Segment Editor with this result.")
        self.segmentEditorButton.setEnabled(False)
        layout.addWidget(self.segmentEditorButton)

        layout.addStretch()

    # ----------------------------------------------------------- input flow --
    def getCurrentVolumeNode(self):
        return self.inputSelector.currentNode()

    def onInputChanged(self, *_):
        self.applyButton.setEnabled(self.getCurrentVolumeNode() is not None)

    def onLoadVolume(self):
        volumeNode = slicer.util.openAddVolumeDialog()
        # openAddVolumeDialog returns True/False; pick the most recent volume as current.
        if volumeNode:
            nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
            if nodes:
                self.inputSelector.setCurrentNode(nodes[-1])

    # ---------------------------------------------------------- apply / run --
    def onApplyClicked(self, *_):
        if not self._isNNUNetModuleInstalled() or self.segmentationLogic is None:
            slicer.util.errorDisplay(
                "This module depends on the NNUNet extension. "
                "Please install it (Extensions Manager) and restart Slicer.")
            return

        model = self._selectedModel()
        self.infoTextEdit.clear()
        self._setRunning(True)

        if not self._installNNUNetIfNeeded():
            self._setRunning(False)
            return

        if not self._ensureWeights(model):
            self._setRunning(False)
            return

        self._currentModel = model
        self._referenceVolumeNode = self.getCurrentVolumeNode()
        self._runSegmentation(model)

    def _ensureWeights(self, model: ModelConfig) -> bool:
        """Make sure the model weights are present, offering to download them if not.
        Validates with SlicerNNUNetLib before returning True."""
        from .WeightsDownloader import WeightsDownloader

        if not WeightsDownloader.isModelPresent(model, self._mlRoot()):
            if not model.downloadUrl:
                slicer.util.errorDisplay(
                    f"Weights for '{model.displayName}' are not installed and no download "
                    f"URL is configured.\n\nExpected at:\n{model.modelPath(self._mlRoot())}")
                return False
            sizeHint = f" (~{model.downloadSizeMb} MB)" if model.downloadSizeMb else ""
            if qt.QMessageBox.question(
                    self, "Download model weights",
                    f"The weights for '{model.displayName}' are not installed.\n"
                    f"Download them now{sizeHint}?") != qt.QMessageBox.Yes:
                return False
            if not WeightsDownloader(self.onProgressInfo).ensureModel(model, self._mlRoot()):
                slicer.util.errorDisplay(
                    f"Failed to download or extract weights for '{model.displayName}'.")
                return False

        ok, reason = self._validateModelWeights(model)
        if not ok:
            slicer.util.errorDisplay(
                f"Weights for '{model.displayName}' are invalid.\n\n{reason}")
        return ok

    def _makeParameter(self, model: ModelConfig, device: str = "cuda"):
        """Build a SlicerNNUNetLib.Parameter with auto-detected folds. SlicerNNUNetLib's
        Parameter._foldsAsList does int(fold), which crashes on 'all'; patch the instance
        to support a fold_all model trained on all data."""
        from SlicerNNUNetLib import Parameter

        folds = model.detectFolds(self._mlRoot())
        parameter = Parameter(folds=folds,
                              modelPath=model.modelPath(self._mlRoot()),
                              device=device)
        if folds.strip() == "all":
            parameter._foldsAsList = lambda: ["all"]
        return parameter, folds

    def _runSegmentation(self, model: ModelConfig):
        parameter, folds = self._makeParameter(model, self.deviceComboBox.currentText)
        self.onProgressInfo(f"Using fold(s): {folds}")

        if not parameter.isSelectedDeviceAvailable():
            deviceName = parameter.device.upper()
            ret = qt.QMessageBox.question(
                self, f"{deviceName} not available",
                f"Selected device ({deviceName}) is not available and will fall back to CPU.\n"
                "Running on CPU may take a long time.\nProceed?")
            if ret == qt.QMessageBox.No:
                self._setRunning(False)
                return

        slicer.app.processEvents()
        self.segmentationLogic.setParameter(parameter)
        self.segmentationLogic.startSegmentation(self._referenceVolumeNode)

    def onStopClicked(self):
        self.isStopping = True
        self.segmentationLogic.stopSegmentation()
        self.segmentationLogic.waitForSegmentationFinished()
        slicer.app.processEvents()
        self.isStopping = False
        self._setRunning(False)

    # --------------------------------------------------------- run callbacks --
    def onInferenceFinished(self, *_):
        if self.isStopping:
            self._setRunning(False)
            return
        try:
            self.onProgressInfo("Loading segmentation result...")
            self._loadAndDisplayResult()
            self.onProgressInfo("Separating fracture fragments...")
            self._autoSeparate()
            self.onProgressInfo("Done.")
        except Exception as e:  # noqa
            slicer.util.errorDisplay(f"Failed to process segmentation result:\n{e}")
            self.onProgressInfo(f"Error: {e}")
        finally:
            self._setRunning(False)

    def onInferenceError(self, errorMsg):
        if self.isStopping:
            return
        self._setRunning(False)
        slicer.util.errorDisplay("Error during inference:\n" + str(errorMsg))

    # ------------------------------------------------------------- results ---
    def _loadAndDisplayResult(self):
        loaded = self.segmentationLogic.loadSegmentation()
        loaded.SetReferenceImageGeometryParameterFromVolumeNode(self._referenceVolumeNode)

        target = self.segmentationSelector.currentNode()
        if target is not None and target is not loaded:
            # Reuse the pre-selected segmentation node (keep its identity/references).
            name = target.GetName()
            target.Copy(loaded)
            target.SetName(name)
            slicer.mrmlScene.RemoveNode(loaded)
            segmentationNode = target
        else:
            segmentationNode = loaded
            segmentationNode.SetName(self._referenceVolumeNode.GetName() + "_Segmentation")

        if not segmentationNode.GetDisplayNode():
            segmentationNode.CreateDefaultDisplayNodes()
        # Remember which model produced this result so a re-run is possible after reload.
        segmentationNode.SetAttribute(self._ATTR_MODEL, self._currentModel.key)
        self._currentSegmentationNode = segmentationNode
        # Select the result node (also persists it via the selector handler).
        self.segmentationSelector.setCurrentNode(segmentationNode)
        self._applyModelDisplay()
        segmentationNode.SetDisplayVisibility(True)
        slicer.app.processEvents()

    def _applyModelDisplay(self):
        segmentation = self._currentSegmentationNode.GetSegmentation()
        displayNode = self._currentSegmentationNode.GetDisplayNode()
        for segDef in self._currentModel.segments:
            segmentId = f"Segment_{segDef.label}"
            segment = segmentation.GetSegment(segmentId)
            if segment is None:
                continue
            segment.SetName(segDef.name)
            segment.SetColor(*segDef.color)
            displayNode.SetSegmentOpacity3D(segmentId, segDef.opacity)

    def _autoSeparate(self):
        self.fragmentLogic.minFragmentVolumeMm3 = self.minVolumeSpinBox.value
        self._semanticLabelmap, self._semanticSpacing = self.fragmentLogic.buildSemanticLabelmap(
            self._currentSegmentationNode, self._currentModel, self._referenceVolumeNode)
        result = self.fragmentLogic.autoSeparate(
            self._semanticLabelmap, self._currentModel, self._semanticSpacing,
            progressCallback=self.onProgressInfo)
        self._writeSeparationResult(result)

    def onReSeparate(self):
        self._reSeparate(self.dilationSpinBox.value)

    def onReSeparatePlus(self):
        self._reSeparate(self.dilationSpinBox.value + self.fragmentLogic.dilationStepMm)

    def _reSeparate(self, dilationMm: float):
        """Manual re-run. Re-reads the (possibly hand-edited) bone + fracture segments,
        then carries forward already-separated fragments: only fragments that still
        straddle a fracture line are split further at the new radius."""
        if self._currentSegmentationNode is None or self._currentModel is None:
            return
        self.dilationSpinBox.setValue(dilationMm)
        self.fragmentLogic.minFragmentVolumeMm3 = self.minVolumeSpinBox.value
        with slicer.util.tryWithErrorDisplay("Fragment separation failed.", waitCursor=True):
            # Pick up manual edits (scissors etc.) to the Mandible / Fracture Lines segments.
            self._semanticLabelmap, self._semanticSpacing = self.fragmentLogic.buildSemanticLabelmap(
                self._currentSegmentationNode, self._currentModel, self._referenceVolumeNode)
            if self._lastResult is not None:
                result = self.fragmentLogic.separateCarryForward(
                    self._semanticLabelmap, self._currentModel, self._semanticSpacing,
                    dilationMm, self._lastResult)
            else:
                result = self.fragmentLogic.separate(
                    self._semanticLabelmap, self._currentModel, self._semanticSpacing,
                    dilationMm)
            self._writeSeparationResult(result)

    def _writeSeparationResult(self, result):
        self.fragmentLogic.writeFragments(
            result, self._currentSegmentationNode, self._referenceVolumeNode, self._currentModel)
        self._lastResult = result
        self.fragmentCountLabel.setText(
            f"{result.nFragments} (at {result.dilationMm:.2f} mm dilation)")
        self.dilationSpinBox.setValue(result.dilationMm)  # also persists via valueChanged
        self.separationWidget.setEnabled(True)
        self.segmentEditorButton.setEnabled(True)
        self._show3D()

    def _show3D(self):
        """Build the closed-surface (3D) representation and centre the 3D view on it."""
        node = self._currentSegmentationNode
        if node is None:
            return
        node.CreateClosedSurfaceRepresentation()
        displayNode = node.GetDisplayNode()
        if displayNode:
            displayNode.SetVisibility3D(True)
        layoutManager = slicer.app.layoutManager()
        threeDWidget = layoutManager.threeDWidget(0) if layoutManager else None
        if threeDWidget:
            threeDWidget.threeDView().rotateToViewAxis(3)  # anterior
        slicer.util.resetThreeDViews()

    def onOpenSegmentEditor(self):
        if self._currentSegmentationNode is None:
            return
        slicer.util.selectModule("SegmentEditor")
        editorWidget = slicer.modules.segmenteditor.widgetRepresentation().self().editor
        editorWidget.setSegmentationNode(self._currentSegmentationNode)
        editorWidget.setSourceVolumeNode(self._referenceVolumeNode)

    # ------------------------------------------------------- helpers / deps --
    def _selectedModel(self) -> ModelConfig:
        key = self.modelComboBox.currentData
        return next(m for m in MODELS if m.key == key)

    @classmethod
    def _mlRoot(cls) -> Path:
        return (Path(__file__).parent / ".." / "Resources" / "ML").resolve()

    # --------------------------------------------------- persisted parameters --
    # State is kept in a singleton vtkMRMLScriptedModuleNode so it is stored with the
    # scene and survives a module reload (the node is not destroyed by Reload, and a
    # freshly created widget restores from it in __init__).
    _MODULE_NAME = "CMFFractureSegmentator"
    _PARAM_MODEL = "Model"
    _PARAM_DEVICE = "Device"
    _PARAM_DILATION = "LastDilationMm"
    _PARAM_MIN_VOLUME = "MinFragmentVolumeMm3"
    _REF_SEGMENTATION = "SegmentationNodeRef"  # node reference role (remapped on load)
    _ATTR_MODEL = "CMFFractureSegmentator.Model"  # model key stored on the segmentation node

    def _moduleParameterNode(self):
        """Get-or-create a singleton scripted-module node, stored with the scene."""
        node = slicer.mrmlScene.GetSingletonNode(self._MODULE_NAME, "vtkMRMLScriptedModuleNode")
        if node is None:
            node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScriptedModuleNode")
            node.UnRegister(None)  # owned by the scene after AddNode
            node.SetSingletonTag(self._MODULE_NAME)
            node.SetAttribute("ModuleName", self._MODULE_NAME)
            node.SetName(self._MODULE_NAME)
            node = slicer.mrmlScene.AddNode(node)
        return node

    def _connectParameterPersistence(self):
        """Persist the relevant UI state to the parameter node on every change."""
        self.modelComboBox.connect("currentIndexChanged(int)", self._onParameterChanged)
        self.deviceComboBox.connect("currentIndexChanged(int)", self._onParameterChanged)
        self.dilationSpinBox.connect("valueChanged(double)", self._onParameterChanged)
        self.minVolumeSpinBox.connect("valueChanged(double)", self._onParameterChanged)

    def _onParameterChanged(self, *_):
        if not self._restoring:
            self._storeParameters()

    def _onSegmentationSelectorChanged(self, *_):
        node = self.segmentationSelector.currentNode()
        self._currentSegmentationNode = node
        self.segmentEditorButton.setEnabled(node is not None)
        if not self._restoring:
            self._moduleParameterNode().SetNodeReferenceID(
                self._REF_SEGMENTATION, node.GetID() if node else None)

    def _storeParameters(self):
        node = self._moduleParameterNode()
        node.SetParameter(self._PARAM_MODEL, str(self.modelComboBox.currentData or ""))
        node.SetParameter(self._PARAM_DEVICE, self.deviceComboBox.currentText)
        node.SetParameter(self._PARAM_DILATION, f"{self.dilationSpinBox.value:.4f}")
        node.SetParameter(self._PARAM_MIN_VOLUME, f"{self.minVolumeSpinBox.value:.4f}")

    def _restoreParameters(self):
        """Restore controls from the parameter node (after reload or scene load)."""
        node = self._moduleParameterNode()
        self._restoring = True
        try:
            modelKey = node.GetParameter(self._PARAM_MODEL)
            if modelKey:
                idx = self.modelComboBox.findData(modelKey)
                if idx >= 0:
                    self.modelComboBox.setCurrentIndex(idx)
            device = node.GetParameter(self._PARAM_DEVICE)
            if device:
                idx = self.deviceComboBox.findText(device)
                if idx >= 0:
                    self.deviceComboBox.setCurrentIndex(idx)
            for key, spinBox in ((self._PARAM_DILATION, self.dilationSpinBox),
                                 (self._PARAM_MIN_VOLUME, self.minVolumeSpinBox)):
                value = node.GetParameter(key)
                if value:
                    try:
                        spinBox.setValue(float(value))
                    except ValueError:
                        pass
            self.segmentationSelector.setCurrentNode(node.GetNodeReference(self._REF_SEGMENTATION))
            self._reattachFromSegmentation()
        finally:
            self._restoring = False

    def _reattachFromSegmentation(self):
        """After a reload / scene load, rebuild enough state from the selected
        segmentation to allow a fragment re-run: the model (from a node attribute),
        the reference volume (from the segmentation's reference geometry), and the
        previous fragment instances (from the Fragment_* segments)."""
        segNode = self.segmentationSelector.currentNode()
        if segNode is None:
            return

        modelKey = segNode.GetAttribute(self._ATTR_MODEL)
        model = None
        if modelKey:
            try:
                model = modelByKey(modelKey)
            except KeyError:
                model = None
        refVolume = segNode.GetNodeReference(
            slicer.vtkMRMLSegmentationNode.GetReferenceImageGeometryReferenceRole())
        if model is None or refVolume is None:
            return  # not a result this module can re-run

        self._currentModel = model
        self._referenceVolumeNode = refVolume
        idx = self.modelComboBox.findData(model.key)
        if idx >= 0:
            self.modelComboBox.setCurrentIndex(idx)

        lastResult = self._rebuildLastResult(segNode, refVolume)
        if lastResult is not None:
            self._lastResult = lastResult
            self.fragmentCountLabel.setText(f"{lastResult.nFragments} (restored)")
            self.separationWidget.setEnabled(True)
        self.segmentEditorButton.setEnabled(True)

    def _rebuildLastResult(self, segNode, referenceVolume):
        """Reconstruct the fragment instance map from the Fragment_* segments."""
        segmentation = segNode.GetSegmentation()
        allIds = [segmentation.GetNthSegmentID(i)
                  for i in range(segmentation.GetNumberOfSegments())]
        fragIds = sorted((sid for sid in allIds if sid.startswith("Fragment_")),
                         key=lambda s: int(s.split("_")[1]))
        if not fragIds:
            return None
        instances = None
        for label, sid in enumerate(fragIds, start=1):
            arr = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, sid, referenceVolume)
            if arr is None:
                continue
            if instances is None:
                instances = np.zeros(arr.shape, dtype=np.int32)
            instances[arr > 0] = label
        if instances is None:
            return None
        return SeparationResult(instances=instances, nFragments=len(fragIds),
                                dilationMm=self.dilationSpinBox.value)

    def _onSceneEndImport(self, *_):
        self._restoreParameters()

    def _onSceneEndClose(self, *_):
        """Reset the module to its initial state when the scene is closed (Ctrl-W)."""
        self._currentSegmentationNode = None
        self._referenceVolumeNode = None
        self._currentModel = None
        self._semanticLabelmap = None
        self._semanticSpacing = None
        self._lastResult = None
        self.infoTextEdit.clear()
        self.fragmentCountLabel.setText("-")
        self._restoring = True  # the scene (and its parameter node) is gone; just reset UI
        try:
            self.dilationSpinBox.setValue(self.fragmentLogic.minDilationMm)
            self.minVolumeSpinBox.setValue(self._defaultMinVolumeMm3)
            self.segmentationSelector.setCurrentNode(None)
        finally:
            self._restoring = False
        self.separationWidget.setEnabled(False)
        self.segmentEditorButton.setEnabled(False)
        self._setRunning(False)
        self.onInputChanged()

    def _validateModelWeights(self, model: ModelConfig):
        parameter, _ = self._makeParameter(model)
        return parameter.isValid()

    def _setRunning(self, isRunning: bool):
        self.applyButton.setVisible(not isRunning)
        self.stopWidget.setVisible(isRunning)
        self.inputWidget.setEnabled(not isRunning)

    @staticmethod
    def _isNNUNetModuleInstalled() -> bool:
        try:
            import SlicerNNUNetLib  # noqa
            return True
        except ImportError:
            return False

    def _installNNUNetIfNeeded(self) -> bool:
        from SlicerNNUNetLib import InstallLogic
        logic = InstallLogic()
        logic.progressInfo.connect(self.onProgressInfo)
        return logic.setupPythonRequirements()

    def _createSegmentationLogic(self):
        if not self._isNNUNetModuleInstalled():
            return None
        from SlicerNNUNetLib import SegmentationLogic
        return SegmentationLogic()

    def _connectSegmentationLogic(self):
        if self.segmentationLogic is None:
            return
        self.segmentationLogic.progressInfo.connect(self.onProgressInfo)
        self.segmentationLogic.errorOccurred.connect(self.onInferenceError)
        self.segmentationLogic.inferenceFinished.connect(self.onInferenceFinished)

    def onProgressInfo(self, infoMsg):
        infoMsg = str(infoMsg)
        self.infoTextEdit.insertPlainText(infoMsg + "\n")
        self.infoTextEdit.verticalScrollBar().setValue(
            self.infoTextEdit.verticalScrollBar().maximum)
        self.fullInfoLogs.append(infoMsg)
        slicer.app.processEvents()
