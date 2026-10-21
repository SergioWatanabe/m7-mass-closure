#!/bin/sh
# Fetch the Harder et al. (2022) M7 emulation dataset (Zenodo record 6583397).
# ~4.6 GB download, ~5.5 GB unpacked into ./aerosol_emulation_data/
set -e
cd "$(dirname "$0")/.."
if [ -d aerosol_emulation_data ]; then echo "aerosol_emulation_data/ already exists"; exit 0; fi
wget -c https://zenodo.org/record/6583397/files/aerosol_emulation_data.zip
unzip -q aerosol_emulation_data.zip
rm aerosol_emulation_data.zip
