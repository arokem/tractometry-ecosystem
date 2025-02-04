
# Set environment to point to local data directory for dowonloading templates:

import os
import os.path as op
import zipfile
import requests
from tqdm import tqdm
import wget
import templateflow.api as tflow
tractometry_dir = op.join(op.expanduser("~"), "data_", "tractometry")

import afqinsight.datasets
afqinsight.datasets._DATA_DIR = op.join(tractometry_dir, "afq-insight")

from afqinsight.datasets import download_weston_havens, download_sarica, download_hbn

os.environ["TEMPLATEFLOW_HOME"] = tractometry_dir
os.environ["DIPY_HOME"] = tractometry_dir
os.environ["AFQ_HOME"] = tractometry_dir

os.makedirs(tractometry_dir, exist_ok=True)
tracometry_zip_f = tractometry_dir + ".zip"


# These imports have to happen after setting "AFQ_HOME":
from AFQ.data.fetch import (
        read_templates,
        read_pediatric_templates,
        read_callosum_templates,
        read_cp_templates,
        read_or_templates,
        read_ar_templates)


def download_data():
    read_templates()
    read_pediatric_templates()
    read_callosum_templates()
    read_cp_templates()
    read_or_templates()
    read_ar_templates()
    download_weston_havens()
    download_sarica()
    download_hbn()


if not op.exists(tracometry_zip_f):
    figshare_path = "https://figshare.com/ndownloader/files/52115198"
    with requests.get(figshare_path, stream=True) as response:
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(tracometry_zip_f, "wb") as file, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))

# Extract the ZIP file
with zipfile.ZipFile(tracometry_zip_f, 'r') as zip_ref:
    for file_ in tqdm(zip_ref.namelist(), desc="Unzipping"):
        zip_ref.extract(file_, tractometry_dir)

os.remove(tracometry_zip_f)

# Templates:
tflow.get('MNI152NLin2009cAsym',
          resolution=1,
          desc='brain',
          suffix='mask')

download_data()
