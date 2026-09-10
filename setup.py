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
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="DRIMS Team",
    maintainer_email="drims@example.com",
    description="3D die detection and 6-DOF pose estimation for angled RGBD cameras.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "die_detector_node = scripts.die_detector_node:main",
            "run_die_detector  = scripts.run_die_detector:main",
        ],
    },
)
