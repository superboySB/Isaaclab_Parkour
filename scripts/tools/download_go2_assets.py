#!/usr/bin/env python3
# Copyright (c) 2025
#
# SPDX-License-Identifier: BSD-3-Clause
"""Download the full Unitree Go2 asset folder from the Omniverse public bucket."""

from __future__ import annotations

import argparse
import os
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


BASE_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
ASSET_PREFIXES = {
    "go2": "Assets/Isaac/4.5/Isaac/IsaacLab/Robots/Unitree/Go2/",
    "skybox": "Assets/Isaac/4.5/Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k",
    "tiles_marble": "Assets/Isaac/4.5/Isaac/IsaacLab/Materials/TilesMarbleSpiderWhiteBrickBondHoned/",
    "ui_elements": "Assets/Isaac/4.5/Isaac/Props/UIElements/",
}
XML_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_objects(prefix: str):
    """Yield every key under the specified prefix."""
    continuation_token = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if continuation_token:
            params["continuation-token"] = continuation_token
        url = f"{BASE_URL}/?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url) as response:
            xml_text = response.read()
        root = ET.fromstring(xml_text)
        for contents in root.findall("s3:Contents", XML_NS):
            key = contents.find("s3:Key", XML_NS).text
            if key.endswith("/"):
                continue
            yield key

        is_truncated = root.find("s3:IsTruncated", XML_NS).text.lower() == "true"
        if not is_truncated:
            break
        continuation_token = root.find("s3:NextContinuationToken", XML_NS).text


def download_key(key: str, output_root: Path, force: bool = False):
    """Download a single S3 key into the output root."""
    assert key.startswith("Assets/")
    relative = key[len("Assets/") :]
    destination = output_root / relative
    if destination.exists() and not force:
        print(f"[SKIP] {relative} (exists)")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{urllib.parse.quote(key)}"
    print(f"[DOWNLOAD] {relative}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as out_file:
        chunk = response.read(1024 * 1024)
        while chunk:
            out_file.write(chunk)
            chunk = response.read(1024 * 1024)


def main():
    parser = argparse.ArgumentParser(
        description="Download the Unitree Go2 USD and supporting assets (sky textures, materials, UI arrows) for offline use."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "assets" / "nucleus",
        help="Directory where the 'Isaac/4.5/…' tree will be created.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        choices=sorted(ASSET_PREFIXES.keys()),
        default=sorted(ASSET_PREFIXES.keys()),
        help="Subset of asset groups to download. Defaults to all.",
    )
    args = parser.parse_args()

    try:
        for asset_name in args.assets:
            prefix = ASSET_PREFIXES[asset_name]
            print(f"[INFO] Downloading asset group '{asset_name}' (prefix: {prefix})")
            for key in list_objects(prefix):
                download_key(key, args.output, force=args.force)
    except KeyboardInterrupt:
        print("\n[INFO] Download interrupted by user.", file=sys.stderr)
        sys.exit(1)

    print(
        "\n[INFO] Completed. Point configs to "
        f"{(args.output / 'Isaac' / '4.5' / 'Isaac' / 'IsaacLab' / 'Robots' / 'Unitree' / 'Go2').as_posix()}."
    )


if __name__ == "__main__":
    main()
