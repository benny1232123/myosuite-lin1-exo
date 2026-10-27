"""Assemble the offscreen PNG frame sequence into an MP4.

The acceptance script writes one PNG every 10 control steps (10 fps at 100 Hz), so a
1500-step episode becomes a 150-frame, 15 s clip.

    python scripts/make_walk_video.py                       # frames/ -> walk_exo.mp4
    python scripts/make_walk_video.py --src walk_frames2    # a fresh render
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

SRC = r"E:\exo algorithm\frames"
OUT = r"E:\exo algorithm\walk_exo.mp4"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=SRC, help="directory holding frame_*.png")
    ap.add_argument("--out", default=OUT, help="output mp4 path")
    args = ap.parse_args()
    files = sorted(Path(args.src).glob("frame_*.png"))
    if not files:
        raise SystemExit(f"no frames under {args.src}")
    print(f"{len(files)} frames, {files[0]} .. {files[-1]}")
    writer = imageio.get_writer(args.out, fps=10, codec="libx264", quality=8,
                                macro_block_size=None, ffmpeg_log_level="error")
    try:
        for path in files:
            # PIL reads the PNGs (imageio has no PNG read backend in this venv); imageio's
            # ffmpeg plugin does the encoding
            writer.append_data(np.asarray(Image.open(path).convert("RGB")))
    finally:
        writer.close()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
