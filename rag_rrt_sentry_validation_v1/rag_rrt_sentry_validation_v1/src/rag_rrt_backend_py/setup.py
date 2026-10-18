from setuptools import find_packages, setup
from pathlib import Path

package_name = "rag_rrt_backend_py"

setup(
    name=package_name,
    version="0.7.0",
    packages=find_packages(include=["rag_rrt_backend_py", "rag_rrt_backend_py.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/backend.yaml"]),
    ] + [("share/" + package_name + "/" + str(p.parent), [str(p)])
           for p in sorted(Path("algorithm").rglob("*")) if p.is_file()],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="RAG-RRT Maintainer",
    maintainer_email="maintainer@example.com",
    description="Bundled Python algorithm backend for ROS 2 robot validation.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "planner_backend = rag_rrt_backend_py.backend_node:main",
        ],
    },
)
