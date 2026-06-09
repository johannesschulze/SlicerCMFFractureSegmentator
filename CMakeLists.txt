cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(CMFFractureSegmentator)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/CMFFractureSegmentator")
set(EXTENSION_CONTRIBUTORS "Johannes Schulze (German Armed Forces Military Hospital Ulm, Germany)")
set(EXTENSION_DESCRIPTION "nnU-Net base automatic segmentatation of mandibular fractures")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/CMFFractureSegmentator.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/CMFFractureSegmentator/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(CMFFractureSegmentatorModule)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
