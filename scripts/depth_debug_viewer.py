#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone viewer process that receives PNG-encoded depth grids over stdin and displays them."""

import os
import sys

import cv2
import numpy as np


def main():
    buffer = sys.stdin.buffer
    window_name = "depth_images_grid"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 900)
    scale = float(os.environ.get("DEPTH_DEBUG_VIEWER_SCALE", "8"))
    scale = max(scale, 1.0)
    try:
        while True:
            length_bytes = buffer.read(4)
            if not length_bytes:
                break
            length = int.from_bytes(length_bytes, byteorder="little", signed=False)
            data = buffer.read(length)
            if not data:
                break
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if scale != 1.0:
                img = cv2.resize(
                    img,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_NEAREST,
                )
            cv2.imshow(window_name, img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()
