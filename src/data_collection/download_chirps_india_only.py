import os
import subprocess
from datetime import datetime, timedelta
import multiprocessing as mp

TMP = "data/temp/chirps_global"
OUT = "data/raw/rainfall/chirps_india_daily"

SHAPE = "data/external/boundary/gadm41_IND_1.shp"

START = datetime(2000, 1, 1)
END = datetime(2025, 12, 31)

WORKERS = 8   # best for 10 core CPU

os.makedirs(TMP, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def process_day(date):

    y = date.strftime("%Y")
    m = date.strftime("%m")
    d = date.strftime("%d")

    name = f"chirps-v2.0.{y}.{m}.{d}.tif.gz"

    url = f"https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/{y}/{name}"

    gz_path = os.path.join(TMP, name)
    tif_path = gz_path.replace(".gz", "")

    out_path = os.path.join(
        OUT,
        f"chirps_india_{y}_{m}_{d}.tif"
    )

    if os.path.exists(out_path):
        return

    try:

        subprocess.run(
            ["wget", "-q", "-c", url, "-O", gz_path],
            check=True
        )

        subprocess.run(
            ["gunzip", "-f", gz_path],
            check=True
        )

        subprocess.run([
            "gdalwarp",
            "-cutline", SHAPE,
            "-crop_to_cutline",
            tif_path,
            out_path
        ], check=True)

        os.remove(tif_path)

    except:
        print("Error:", name)


def main():

    dates = []

    d = START
    while d <= END:
        dates.append(d)
        d += timedelta(days=1)

    print("Total days:", len(dates))
    print("Workers:", WORKERS)

    with mp.Pool(WORKERS) as pool:
        pool.map(process_day, dates)


if __name__ == "__main__":
    main()