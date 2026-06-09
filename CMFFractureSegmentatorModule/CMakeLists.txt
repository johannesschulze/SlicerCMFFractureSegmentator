#-----------------------------------------------------------------------------
set(MODULE_NAME CMFFractureSegmentatorModule)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  CMFFractureSegmentatorLib/__init__.py
  CMFFractureSegmentatorLib/SegmentationWidget.py
  CMFFractureSegmentatorLib/FragmentSeparationLogic.py
  CMFFractureSegmentatorLib/WeightsDownloader.py
  CMFFractureSegmentatorLib/Models.py
  CMFFractureSegmentatorLib/Utils.py
  CMFFractureSegmentatorLib/fragment_separation.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
