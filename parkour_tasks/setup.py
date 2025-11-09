# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab_tasks' python package."""

import ast
from pathlib import Path

from setuptools import setup


def load_package_metadata(extension_dir: Path) -> dict:
    """Extract the ``[package]`` table from the extension.toml file without third-party deps."""
    package_data = {}
    in_package_section = False
    config_path = extension_dir / "config" / "extension.toml"

    with config_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            # Drop inline comments and surrounding whitespace.
            line = raw_line.split("#", 1)[0].strip()

            if not line:
                continue

            # Detect section headers.
            if line.startswith("["):
                in_package_section = line == "[package]"
                continue

            if in_package_section and "=" in line:
                key, value = line.split("=", 1)
                package_data[key.strip()] = ast.literal_eval(value.strip())

    if not package_data:
        raise RuntimeError(f"Failed to load '[package]' metadata from {config_path}")

    return package_data


# Obtain the extension data from the extension.toml file
EXTENSION_PATH = Path(__file__).resolve().parent
PACKAGE_METADATA = load_package_metadata(EXTENSION_PATH)

# Installation operation
setup(
    name="parkour_tasks",
    author="Isaac Lab Project Developers",
    maintainer="Isaac Lab Project Developers",
    url=PACKAGE_METADATA["repository"],
    version=PACKAGE_METADATA["version"],
    description=PACKAGE_METADATA["description"],
    keywords=PACKAGE_METADATA["keywords"],
    include_package_data=True,
    python_requires=">=3.10",
    packages=["parkour_tasks"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.5.0",
    ],
    zip_safe=False,
    install_requires=['fast_simplification']
)
