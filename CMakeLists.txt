cmake_minimum_required(VERSION 3.8)
project(drims_die_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/DieIdentification3D.srv"
  "srv/DieIdentification.srv"
  DEPENDENCIES geometry_msgs builtin_interfaces
)

# rosidl_generator_py already sets up python generation for ${PROJECT_NAME}.
# Install our Python library modules alongside the generated interfaces:
install(DIRECTORY drims_die_detection
  DESTINATION "${PYTHON_INSTALL_DIR}"
)

install(PROGRAMS
  scripts/die_detector_node.py
  scripts/run_die_detector.py
  scripts/rgbd_mock_publisher.py
  scripts/tune_die_detector_gui.py
  scripts/detection_optimization.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  rviz
  calibration_data
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()

