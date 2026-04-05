import os
import subprocess

SRC = "data/raw/rainfall/chirps_all"
DST = "data/raw/rainfall/chirps_india_daily"
SHAPE = "data/external/boundary/gadm41_IND_1.shp"

os.makedirs(DST, exist_ok=True)

files = [f for f in os.listdir(SRC) if f.endswith(".tif")]

for f in files:

    src_file = os.path.join(SRC, f)
    dst_file = os.path.join(DST, f)

    if os.path.exists(dst_file):
        continue

    print("Clipping", f)

    cmd = [
        "gdalwarp",
        "-cutline", SHAPE,
        "-crop_to_cutline",
        src_file,
        dst_file
    ]

    subprocess.run(cmd)