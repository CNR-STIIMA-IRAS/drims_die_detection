from setuptools import setup, find_packages
import os
from glob import glob

package_name = "drims_die_detection"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["tests*"]),
    data_files=[
        # ament resource index
        ("share/ament_index/resource_index/packages",
         [f"resource/{package_name}"]),
        # package.xml
        (f"share/{package_name}", ["package.xml"]),
        # launch files
        (os.path.join("share", package_name, "launch"),
         glob("launch/*.launch.py")),
        # config files
        (os.path.join("share", package_name, "config"),
         glob("config/*.yaml")),
        # rviz config
        (os.path.join("share", package_name, "rviz"),
         glob("rviz/*.rviz")),
        # test images
        (os.path.join("share", package_name, "test_images", "rgb"),
         glob("test_images/rgb/*")),
        (os.path.join("share", package_name, "test_images", "depth"),
         glob("test_images/depth/*")),
        # calibration data
        (os.path.join("share", package_name, "calibration_data"),
         glob("calibration_data/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Michele Ferrari",
    maintainer_email="micheleferrari1@cnr.it",
    description="3D die detection and 6-DOF pose estimation for angled RGBD cameras.",
    license="MIT",
    tests_require=["pytest"],
    scripts=[
        "scripts/die_detector_node.py",
        "scripts/run_die_detector.py",
        "scripts/rgbd_mock_publisher.py",
        "scripts/tune_die_detector_gui.py",
        "scripts/detection_optimization.py",
    ],
)
