
# relabel atlas label image with label from 1 to N
import os, sys, shutil, time, subprocess, inspect, math, glob, re
import nibabel as nib
import argparse
import numpy as np
import json
import pandas as pd


main_dir = '/data/cye_code/BIDS-dmrirecon/atlases'
atlas_input = os.path.join(main_dir, 'AAL3_MNI.nii.gz')
atlas_input_nii = nib.load(atlas_input)
sgm_lut_path = os.path.join(main_dir, 'AAL3_MNI.csv')
sgm_lut = pd.read_csv(sgm_lut_path)

if not (sgm_lut.Index == sgm_lut.Intensity).all():
    atlas_input_nii_img = atlas_input_nii.get_fdata()
    atlas_output_path = os.path.join(main_dir, 'AAL3_MNI_relabel.nii.gz')
    # if label index != intensity, relabel it
    for item in range(len(sgm_lut)):
        if sgm_lut.Index[item] != sgm_lut.Intensity[item]:
            np.place(atlas_input_nii_img, atlas_input_nii_img == sgm_lut.Intensity[item], sgm_lut.Index[item])
    new_label = nib.Nifti1Image(np.int16(atlas_input_nii_img), atlas_input_nii.affine, atlas_input_nii.header)
    nib.save(new_label, atlas_output_path)

print('OK')