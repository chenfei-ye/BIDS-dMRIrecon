

# BIDS-dMRIrecon

`BIDS-dmrirecon` is developed to perform dwi quantitative mapping/tractography based on CSD (constrained spherical deconvolution) estimation from [MRtrix3](https://www.mrtrix.org/). Main functions include:
- CSD estimation (*dwi2fod*)
- DTI mapping (*dwi2tensor*)
- Fiber tracking (*https://github.com/MIC-DKFZ/TractSeg*)
- DKI mapping (*https://github.com/m-ama/PyDesigner*)
- NODDI mapping (*https://github.com/daducci/AMICO*)
- Network establishment (*tck2connectome*)
- Visualization (*vtk/vtp files*)

The input data should be arranged according to [BIDS format](https://bids.neuroimaging.io/). Input image modalities must include 3D-T1w and dMRI data. 

[BIDS-dMRIrecon 中文说明](resources/README_Chs.md)

Check details of [brain atlases](https://github.com/chenfei-ye/BIDS-fMRIpost/blob/main/resources/atlases.md)

Check bids-dmriprep version history in [Change Log](resources/CHANGELOG.md)

## Contents
* [Install](#Install)
* [Before Running](#before-running)
* [Running](#running)
* [Input Argument](#input-argument)
* [Output Explanation](#output-explanation)

## Install
### install by pulling (recommend)
```
docker pull mindsgo-sz-docker.pkg.coding.net/neuroimage_analysis/base/bids-dmrirecon:latest
docker tag  mindsgo-sz-docker.pkg.coding.net/neuroimage_analysis/base/bids-dmrirecon:latest  bids-dmrirecon:latest
```

### or install by docker build
```
cd BIDS-dmrirecon
docker build -t bids-dmrirecon:latest .
```
## Before Running
use [sMRIPrep](https://github.com/chenfei-ye/BIDS-sMRIprep) for T1w data preprocessing
```
docker run -it --rm -v <bids_root>:/bids_dataset bids-smriprep:latest python /run.py /bids_dataset --participant_label 01 02 03 -MNInormalization -fsl_5ttgen -cleanup
```

use [dMRIPrep](https://github.com/chenfei-ye/BIDS-dMRIprep) for dMRI data preprocessing
```
docker run -it --rm --gpus all -v <bids_root>:/bids_dataset bids-dmriprep:latest python /run.py /bids_dataset /bids_dataset/derivatives/dmri_prep participant --participant_label 01 02 03 -mode complete
```

## Running
### default running
```
docker run -it --rm -v <bids_root>:/bids_dataset bids-dmrirecon:latest python /run.py /bids_dataset /bids_dataset/derivatives/dmri_recon participant --participant_label 01 02 03 -mode tract,dti_para,connectome -bundle_json /scripts/bundle_list_all72.json -wholebrain_fiber 5000000 -atlases AAL3_MNI desikan_T1w -cleanup
```

### optional: summarize quantitative mapping metrics across all participants (post-hoc)
```
docker run -it --rm -v <bids_root>:/bids_dataset bids-dmrirecon:latest python /scripts/json2csv.py /bids_dataset participant
```

## Input Argument
####   positional argument:
-   `/bids_dataset`: The root folder of a BIDS valid dataset (sub-XX folders should be found at the top level in this folder).
-   `/bids_dataset/derivatives/dmri_recon`: output path
- `participant`: process on participant level

####   optional argument:
-   `--participant_label [str]`：A space delimited list of participant identifiers or a single identifier (the sub- prefix can be removed)
-   `--session_label [str]`：A space delimited list of session identifiers or a single identifier (the ses- prefix can be removed)
-  `-tracking [prob|det]`：tracking mode, probabilistic or deterministic.  default = probabilistic 
-  `-odf [tom|peaks]`：orientation distribution function (ODF, the probability of diffusion in a given direction). tom refers to Tract Orientation Maps,  default = tom.
-  `-wholebrain_fiber_num [int]`：number of streamline for whole brain.  default = 10000000
-  `-fiber_num [int]`：number of streamline for each tract, default = 2000
-  `-bundle_json [str]`：tract list to be tracked.  default = `/scripts/bundle_list.json`
-  `-atlases [str]`：A space delimited list of brain atlases. e.g. `-atlases AAL3_MNI hcpmmp_T1w`. FreeSurfer should be ran first for atlases ends with `_T1w`. See `atlases/atlas_config_docker.json` for details.
- `-resume`：resume running based on the temporary output generated by last run. 
- `-v`：check version 
- `-cleanup`: remove temporary files.


## Output explanation
-   `DTI_mapping`: DTI metrics, `FA`/`MD`/`RD`/`AD`/`DEC-map(directionally-encoded colour map)`
-   `fiber_tracts/bundle_segmentations`: bundle mask
-   `fiber_tracts/Fibers`: tck file for each tract 
-   `fiber_tracts/fiber_streamline.json`: summary for each tract
-   `DKI_mapping`: DTI metrics, `MK`/`RK`/`AK`/`KA`
-   `NODDI_mapping`: NODDI metrics, `FIT_ICVF.nii.gz`/`FIT_OD.nii.gz`/`FIT_ISOVF.nii.gz`/`FIT_dir.nii.gz`
-   `connectome`: brain structural networks. see [structural-connectome-metric-options](https://mrtrix.readthedocs.io/en/latest/reference/commands/tck2connectome.html#structural-connectome-metric-options) for details. 
-   `visualization`: vtk/vtp files for each tract


## Copyright
Copyright © chenfei.ye@foxmail.com
Please make sure that your usage of this code is in compliance with the code license.

