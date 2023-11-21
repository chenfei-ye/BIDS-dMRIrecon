# -*- coding: utf-8 -*-

"""
@author: Chenfei
@contact:chenfei.ye@foxmail.com
@version: 4.0
@file: run.py
@time: 2023/11/21

"""

import os, sys, shutil, time, subprocess, inspect, math, glob, re
import nibabel as nib
import argparse
import numpy as np
import json
import pandas as pd
import object_visualization

from designer.fitting import dwipy as dp
from designer.preprocessing import util
from designer.postprocessing import filters


DWIFile = util.DWIFile
__version__ = 'v4.0'
colourClear = '\033[0m'
colourConsole = '\033[03;32m'
colourError = '\033[01;31m'
colourExec = '\033[03;36m'
colourWarn = '\033[00;31m'


def read_json_file(json_file):
    """
    read json file
    :param json_file:
    :return:
    """
    if not json_file or not os.path.exists(json_file):
        print('json file %s not exist' % json_file)
        return None

    with open(json_file, 'r', encoding='utf-8') as fp:
        out_dict = json.load(fp)

    return out_dict


def write_json_file(file, content, ensure_ascii=True, sort_keys=True):
    """
    write dictionary into a json file
    :param file:
    :param content:
    :param ensure_ascii:
    :param sort_keys:
    :return:
    """
    with open(file, 'w+', encoding='utf-8') as fp:
        try:
            fp.write(json.dumps(content, indent=2, ensure_ascii=ensure_ascii, sort_keys=sort_keys))
        except TypeError:
            use_content = dict()
            for key, value in content.items():
                use_content[str(key)] = value

            fp.write(json.dumps(use_content, indent=2, ensure_ascii=ensure_ascii, sort_keys=sort_keys))


def makeTempDir():
    import random, string
    global tempDir, workingDir
    if tempDir:
        app_error('Script error: Cannot use multiple temporary directories')

    random_string = ''.join(random.choice(string.ascii_uppercase + string.digits) for x in range(6))
    tempDir = os.path.join(workingDir, 'tmp-' + random_string) + os.sep
    os.makedirs(tempDir)
    app_console('Generated temporary directory: ' + tempDir)
    with open(os.path.join(tempDir, 'cwd.txt'), 'w') as outfile:
        outfile.write(workingDir + '\n')
    with open(os.path.join(tempDir, 'command.txt'), 'w') as outfile:
        outfile.write(' '.join(sys.argv) + '\n')
    open(os.path.join(tempDir, 'log.txt'), 'w').close()


def gotoTempDir():
    global tempDir
    if not tempDir:
        app_error('Script error: No temporary directory location set')
    else:
        app_console('Changing to temporary directory (' + tempDir + ')')
        os.chdir(tempDir)


def app_complete():
    global cleanup, tempDir, workingDir
    global colourClear, colourConsole, colourWarn
    if cleanup and tempDir:
        app_console('Deleting temporary directory ' + tempDir)
        shutil.rmtree(tempDir)
    elif tempDir:
        if os.path.isfile(os.path.join(tempDir, 'error.txt')):
            with open(os.path.join(tempDir, 'error.txt'), 'r') as errortext:
                sys.stderr.write(os.path.basename(sys.argv[0]) + ': ' + colourWarn + 
                                'Script failed while executing the command: ' + errortext.readline().rstrip() + colourClear + '\n')
                sys.stderr.write(os.path.basename(sys.argv[0]) + ': ' + colourWarn + 
                                'For debugging, inspect contents of temporary directory: ' + tempDir + colourClear + '\n')
        else:
            sys.stderr.write(os.path.basename(sys.argv[0]) + ': ' + colourConsole + 
                                'Contents of temporary directory kept, location: ' + tempDir + colourClear + '\n')
            sys.stderr.flush()


def app_console(text):
    global colourClear, colourConsole
    sys.stderr.write(os.path.basename(sys.argv[0]) + ': ' + colourConsole + text + colourClear + '\n')
    if tempDir:
        with open(os.path.join(tempDir, 'log.txt'), 'a') as outfile:
            outfile.write(text + '\n')


def app_warn(text):
    global colourClear, colourWarn
    sys.stderr.write(os.path.basename(sys.argv[0]) + ': ' + colourWarn + '[WARNING] ' + text + colourClear + '\n')
    if tempDir:
        with open(os.path.join(tempDir, 'log.txt'), 'a') as outfile:
            outfile.write(text + '\n')


def app_error(text):
    global colourClear, colourError, cleanup
    sys.stderr.write(os.path.basename(sys.argv[0]) + ': ' + colourError + '[ERROR] ' + text + colourClear + '\n')
    if tempDir:
        with open(os.path.join(tempDir, 'log.txt'), 'a') as outfile:
            outfile.write(text + '\n')
    cleanup = False
    app_complete()
    sys.exit(1)


def mrinfo(image_path, field):
    command_mrinfo = ['mrinfo', image_path, '-' + field ]
    app_console('Command: \'' + ' '.join(command_mrinfo) + '\' (piping data to local storage)')
    proc = subprocess.Popen(command_mrinfo, stdout=subprocess.PIPE, stderr=None)
    result, dummy_err = proc.communicate()
    result = result.rstrip().decode('utf-8')
    return result


def command(cmd):
    global _processes, app_verbosity, tempDir, cleanup
    global colourClear, colourError, colourConsole
    _env = os.environ.copy()

    return_stdout = ''
    return_stderr = ''
    sys.stderr.write(colourExec + 'Command:' + colourClear + '  ' + cmd + '\n')
    sys.stderr.flush()

    # process = subprocess.Popen(cmd, env=_env, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process = subprocess.run(cmd, env=_env, shell=True)
    # return_stdout += process.stdout.readline()

    if process.returncode:
        cleanup = False
        caller = inspect.getframeinfo(inspect.stack()[1][0])
        script_name = os.path.basename(sys.argv[0])
        app_console('')
        try:
            filename = caller.filename
            lineno = caller.lineno
        except AttributeError:
            filename = caller[1]
            lineno = caller[2]
        sys.stderr.write(script_name + ': ' + colourError + '[ERROR] Command failed: ' + cmd + colourClear + colourConsole + ' (' + os.path.basename(filename) + ':' + str(lineno) + ')' + colourClear + '\n')
        sys.stderr.write(script_name + ': ' + colourConsole + 'Output of failed command:' + colourClear + '\n')

        app_console('')
        sys.stderr.flush()
        if tempDir:
            # with open(os.path.join(tempDir, 'error.txt'), 'w') as outfile:
            #     outfile.write(cmd + '\n\n' + process.stderr.readline() + '\n')
            # return_stderr += process.stderr.readline()
            app_complete()
            sys.exit(1)
        else:
            app_warn('Command failed: ' + cmd)

    if tempDir:
        with open(os.path.join(tempDir, 'log.txt'), 'a') as outfile:
            outfile.write(cmd + '\n')

    # return return_stdout, return_stderr


# Class for importing header information from an image file for reading
class Header(object):
    def __init__(self, image_path):
        filename = 'img_header.json'
        command = ['mrinfo', image_path, '-json_all', filename]
        app_console(str(command))
        result = subprocess.call(command, stdout=None, stderr=None)
        if result:
            app_error('Could not access header information for image \'' + image_path + '\'')
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except UnicodeDecodeError:
            with open(filename, 'r') as f:
                data = json.loads(f.read().decode('utf-8', errors='replace'))
        os.remove(filename)
        try:
            # self.__dict__.update(data)
            # Load the individual header elements manually, for a couple of reasons:
            # - So that pylint knows that they'll be there
            # - Write to private members, and give read-only access
            self._name = data['name']
            self._size = data['size']
            self._spacing = data['spacing']
            self._strides = data['strides']
            self._format = data['format']
            self._datatype = data['datatype']
            self._intensity_offset = data['intensity_offset']
            self._intensity_scale = data['intensity_scale']
            self._transform = data['transform']
            if not 'keyval' in data or not data['keyval']:
                self._keyval = {}
            else:
                self._keyval = data['keyval']
        except:
            app_error('Error in reading header information from file \'' + image_path + '\'')
        app_console(str(vars(self)))

    def name(self):
        return self._name
    def size(self):
        return self._size
    def spacing(self):
        return self._spacing
    def strides(self):
        return self._strides
    def format(self):
        return self._format
    def datatype(self):
        return self.datatype
    def intensity_offset(self):
        return self._intensity_offset
    def intensity_scale(self):
        return self._intensity_scale
    def transform(self):
        return self._transform
    def keyval(self):
        return self._keyval

# Computes image statistics using mrstats.
# Return will be a list of ImageStatistics instances if there is more than one volume
#   and allvolumes=True is not set; a single ImageStatistics instance otherwise
from collections import namedtuple
ImageStatistics = namedtuple('ImageStatistics', 'mean median std std_rv min max count')
IMAGE_STATISTICS = ['mean', 'median', 'std', 'std_rv', 'min', 'max', 'count' ]

def img_statistics(image_path, **kwargs):

    mask = kwargs.pop('mask', None)
    allvolumes = kwargs.pop('allvolumes', False)
    ignorezero = kwargs.pop('ignorezero', False)
    if kwargs:
        raise TypeError('Unsupported keyword arguments passed to image.statistics(): ' + str(kwargs))

    command = ['mrstats', image_path ]
    for stat in IMAGE_STATISTICS:
        command.extend([ '-output', stat ])
    if mask:
        command.extend([ '-mask', mask ])
    if allvolumes:
        command.append('-allvolumes')
    if ignorezero:
        command.append('-ignorezero')

    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None)
    stdout = proc.communicate()[0]
    if proc.returncode:
        raise OSError('Error trying to calculate statistics from image \'' + image_path + '\'')
    stdout_lines = [line.strip() for line in stdout.decode('cp437').splitlines() ]
    result = []
    for line in stdout_lines:
        line = line.replace('N/A', 'nan').split()
        assert len(line) == len(IMAGE_STATISTICS)
        result.append(ImageStatistics(float(line[0]), float(line[1]), float(line[2]), float(line[3]),
                                      float(line[4]), float(line[5]), int(line[6])))
    if len(result) == 1:
        result = result[0]
    return result


def get_image_spacing(img_path):
    img = nib.load(img_path)
    # affine = img.affine
    # return str(abs(round(affine[0, 0], 2)))
    return str(abs(np.min(img.header.get_zooms()[0:3])))


def fiber_statistics(bundle, metric, file_path):
    if not os.path.isfile(file_path):
        app_error('metric files are lacking for ' + bundle)
    else:
        metrics_file = open(file_path, "r")
        lines = metrics_file.read().split('\n')
        metrics_file.close()

        # added by Chenfei: adapt to MRtrix3 new version where # command history was written to the 1st line 
        if len(lines) == 3:
            lines.pop(0)

        lines.pop(-1)
        lines = list(map(float, lines))
        if len(lines):
            bundle_dict = {'count': len(lines), 'mean': np.nanmean(lines), 'median': np.nanmedian(lines),
                           'min': np.nanmin(lines), 'max': np.nanmax(lines), 'standard deviation': np.nanstd(lines)}
        else:
            app_error('Empty ' + metric + ' file for ' + bundle)
            bundle_dict = []
    return bundle_dict


def DTI_statistics(bundle, metric, file_path):
    if not os.path.isfile(file_path):
        app_error('metric files are lacking for ' + bundle)
    else:
        metrics_file = open(file_path, "r")
        lines = metrics_file.read().split('\n')
        metrics_file.close()

        # added by Chenfei: adapt to MRtrix3 new version where # command history was written to the 1st line 
        if len(lines) == 3:
            lines.pop(0)

        lines.pop(-1)
        try:
            lines = list(map(float, lines[0].split(' ')))
        except IndexError:
            app_error('Empty ' + metric + ' file for ' + bundle)
            bundle_dict = []
        else:
            bundle_dict = {'mean': np.nanmean(lines), 'median': np.nanmedian(lines), 'min': np.nanmin(lines),
                           'max': np.nanmax(lines), 'standard deviation': np.nanstd(lines)}
    return bundle_dict


def statistic_summary_fiber(metric, bundle_ls, output_dir, **kwargs):
    map_dir = kwargs.pop('map_dir', None)
    bundle_dir = kwargs.pop('bundle_dir', None)
    CSD_all_dic = {}
    for row, bundle in enumerate(bundle_ls): 
        if bundle_dir != None: # statistics on bundle_mask image
            bundle_stats = img_statistics(os.path.join(map_dir, metric + '.nii.gz'), 
                                    mask=os.path.join(bundle_dir, bundle + '.nii.gz'))
            CSD_all_dic[bundle] = {'mean': bundle_stats.mean, 'median': bundle_stats.median, 'min': bundle_stats.min,
                           'max': bundle_stats.max, 'standard deviation': bundle_stats.std}
            with open(os.path.join(output_dir, metric + '_bundle.json'), 'w+') as result_file:
                json.dump(CSD_all_dic, result_file, sort_keys=True, indent=4, separators=(',', ': '))

        else: 
            if metric == 'fiber': # statistics on tck files (fiber streamline)
                file_path = os.path.join(output_dir, bundle + '.txt')
                CSD_all_dic[bundle] = fiber_statistics(bundle, metric, file_path)
            else: # statistics on parametric images
                file_path = os.path.join(output_dir, metric + '_' + bundle + '.txt')
                CSD_all_dic[bundle] = DTI_statistics(bundle, metric, file_path)
            with open(os.path.join(output_dir, metric + '_streamline.json'), 'w+') as result_file:
                json.dump(CSD_all_dic, result_file, sort_keys=True, indent=4, separators=(',', ': '))
                

def statistic_summary_parcel(dmri_map_path, t1_label_path, lookup_table_path, stats_type="intensity", json_name = 'volume_L4_regions'):
    t1_L4_label_nii = nib.load(t1_label_path)
    lookup = pd.read_csv(lookup_table_path)
    parcel_num = lookup.shape[0]
    statistics_dict = {}
    if stats_type == "volume":
        voxel_volumn = t1_L4_label_nii.header['pixdim'][1] * t1_L4_label_nii.header['pixdim'][2] * t1_L4_label_nii.header['pixdim'][3]
        output_dir = os.path.dirname(t1_label_path)
        for j in range(parcel_num):
            region_idx = j+1
            # volume of each WM regions
            region_name = 'volume_' + lookup.query('label==@region_idx').loc[:,'label_name'].tolist()[0]
            statistics_dict[region_name] = np.sum(t1_L4_label_nii.get_fdata() == region_idx) * voxel_volumn
        statistics_json_path = os.path.join(output_dir, json_name + '.json')
        write_json_file(statistics_json_path, statistics_dict)
        del statistics_dict, statistics_json_path

    elif stats_type == "intensity":
        dmri_map = nib.load(dmri_map_path)
        output_dir = os.path.dirname(dmri_map_path)
        for j in range(parcel_num):
            region_idx = j+1
            # volume of each WM regions
            region_name = 'MeanValue_' + lookup.query('label==@region_idx').loc[:,'label_name'].tolist()[0]
            roi_idx = np.in1d(t1_L4_label_nii.get_fdata(), region_idx).reshape(t1_L4_label_nii.get_fdata().shape)
            intensity_roi = dmri_map.get_fdata()[roi_idx]
            statistics_dict[region_name] = np.nanmean(intensity_roi[np.nonzero(intensity_roi)])
        statistics_json_path = os.path.join(output_dir, json_name + '.json')
        write_json_file(statistics_json_path, statistics_dict)
        del statistics_dict, statistics_json_path


def runSubject(args, subject_label):
    global workingDir, tempDir, cleanup, resume
    label = 'sub-' + subject_label
    output_dir = os.path.join(args.output_dir, label)
    if os.path.exists(output_dir):
        app_warn('Output directory for subject \'' + label + '\' already exists. May override output files if new output creates')
        status_override = True
    else:
        status_override = False
    
    dmriprep_dir = os.path.join(args.bids_dir, 'derivatives', 'dmri_prep', label)
    if not os.path.exists(dmriprep_dir):
        app_console('dmri_dir: '+ dmriprep_dir)
        app_error('Failed to detect output folder of BIDS-dmriprep for subject' + label)
    
    # if -resume, find the last modified temp folder
    if resume:
        tmp_folder_find = glob.glob(os.path.join(workingDir, 'tmp-*'))
        if len(tmp_folder_find) == 0:
            app_warn('Found no tmp folder in resume mode, create a new tmp folder')
            makeTempDir()
            gotoTempDir()
        else:
            modified_time_ls = [os.path.getmtime(path) for path in tmp_folder_find]
            max_idx = modified_time_ls.index(max(modified_time_ls))
            tempDir = tmp_folder_find[max_idx]
            os.chdir(tempDir)
    else:
        makeTempDir()
        gotoTempDir()
    app_console('working directory: ' + os.getcwd())

    if status_override:
        app_warn('copy existing dmrirecon files into temp dir')
        if not os.path.exists(os.path.join(tempDir, 'DTI_mapping')) and os.path.exists(os.path.join(output_dir, 'DTI_mapping')):
            shutil.copytree(os.path.join(output_dir, 'DTI_mapping'), os.path.join(tempDir, 'DTI_mapping'))
        if not os.path.exists(os.path.join(tempDir, 'DKI_mapping')) and os.path.exists(os.path.join(output_dir, 'DKI_mapping')):
            shutil.copytree(os.path.join(output_dir, 'DKI_mapping'), os.path.join(tempDir, 'DKI_mapping'))
        if not os.path.exists(os.path.join(tempDir, 'NODDI_mapping')) and os.path.exists(os.path.join(output_dir, 'NODDI_mapping')):
            shutil.copytree(os.path.join(output_dir, 'NODDI_mapping'), os.path.join(tempDir, 'NODDI_mapping'))
        if not os.path.exists(os.path.join(tempDir, 'connectome')) and os.path.exists(os.path.join(output_dir, 'connectome')):
            shutil.copytree(os.path.join(output_dir, 'connectome'), os.path.join(tempDir, 'connectome'))

    app_console('Launching participant-level analysis for subject \'' + label + '\'')

    #-----------------------------------------------------------------
    # Step 0: input initialization
    #-----------------------------------------------------------------
    modes_ls = args.mode.split(",")
    lookup_table_path = '/WMH_lookup_table.csv'
    no_vtp = args.no_vtp

    start = time.time()
    t1prep_dir = os.path.join(args.bids_dir, 'derivatives', 'smri_prep', label)
    input_dwi_nii = os.path.join(dmriprep_dir, 'dwi.nii.gz')
    input_dwi_bval = os.path.join(dmriprep_dir, 'dwi.bval')
    input_dwi_bvec = os.path.join(dmriprep_dir, 'dwi.bvec')
    input_dwi_json = os.path.join(dmriprep_dir, 'dwi.json')
    input_dwi_b0_nii = os.path.join(dmriprep_dir, 'dwi_bzero.nii.gz')
    input_dwi_mask_nii = os.path.join(dmriprep_dir, 'dwi_mask.nii.gz')
    input_dwi_mif = os.path.join(dmriprep_dir, 'dwi.mif')
    input_dwi_mni_nii = os.path.join(dmriprep_dir, 'dwi_mni.nii.gz')
    input_dwi_mni_bval = os.path.join(dmriprep_dir, 'dwi_mni.bval')
    input_dwi_mni_bvec = os.path.join(dmriprep_dir, 'dwi_mni.bvec')
    input_dwi_mni_mask_nii = os.path.join(dmriprep_dir, 'dwi_mni_mask.nii.gz')
    input_dwi_mni_b0_nii = os.path.join(dmriprep_dir, 'dwi_mni_bzero.nii.gz')
    input_dwi_mni_fslmat = os.path.join(dmriprep_dir, 'dwi_mni_fsl.mat')
    input_dwi_mni_antsmat = os.path.join(dmriprep_dir, 'dwi_mni_ants.mat')


    input_dwi_to_t1_ANTs_mat = os.path.join(dmriprep_dir, 'dwi_t1_ants.mat')
    input_t1_5tt_nii = os.path.join(t1prep_dir, 'T1w_5tt.nii.gz')    
    input_t1_bet_mask_nii = os.path.join(t1prep_dir, 'T1w_bet_mask.nii.gz')    
    input_mni_to_t1_ANTs_nii = os.path.join(t1prep_dir, 'composite_warp_mni_to_t1.nii.gz')    

    if not os.path.exists(input_t1_bet_mask_nii):
        app_error('failed to find file: T1w_bet_mask.nii.gz in t1_prep directory')

    if not os.path.exists(input_dwi_mif):
        command('mrconvert ' + input_dwi_nii + ' ' + input_dwi_mif + ' -json_import ' +
                    input_dwi_json + ' -fslgrad ' + input_dwi_bvec + ' ' + input_dwi_bval)
    
    # Determine whether we are working with single-shell or multi-shell data
    bvalues = [int(round(float(value)))
        for value in mrinfo(input_dwi_mif, 'shell_bvalues').strip().split()]
    multishell = (len(bvalues) > 2)

    #-----------------------------------------------------------------
    # Step 1: DTI parameter mapping
    #-----------------------------------------------------------------
    if any(item in modes_ls for item in ['tract', 'dti_para']):
        app_console('run Step 1: DTI parameter mapping on tck files')

        dti_mapping_dir = 'DTI_mapping'
        
        if args.resume and os.path.exists(dti_mapping_dir):  # if -resume, check if output file exist or not
            app_console('DTI_mapping folder found, skip this step')
        else:
            os.mkdir(os.path.join(tempDir, dti_mapping_dir))
    
            command('dwi2tensor ' + input_dwi_mif + ' ' + dti_mapping_dir + '/tensor.mif')
            command('tensor2metric -fa ' + dti_mapping_dir + '/FA.nii.gz ' + dti_mapping_dir + '/tensor.mif' +
                    ' -mask ' + input_dwi_mask_nii)
            command('tensor2metric -adc ' + dti_mapping_dir + '/MD.nii.gz ' + dti_mapping_dir + '/tensor.mif' +
                    ' -mask ' + input_dwi_mask_nii)
            command('tensor2metric -ad ' + dti_mapping_dir + '/AD.nii.gz ' + dti_mapping_dir + '/tensor.mif' +
                    ' -mask ' + input_dwi_mask_nii)
            command('tensor2metric -rd ' + dti_mapping_dir + '/RD.nii.gz ' + dti_mapping_dir + '/tensor.mif' +
                    ' -mask ' + input_dwi_mask_nii)    
            command('tensor2metric ' + dti_mapping_dir + '/tensor.mif' +
                    ' -vec ' + dti_mapping_dir + '/dec.nii.gz' +
                    ' -mask ' + input_dwi_mask_nii)     

            # make sure all img values are finite
            command('mrcalc ' + dti_mapping_dir + '/FA.nii.gz -finite ' + dti_mapping_dir + '/FA.nii.gz 0.0 -if ' + dti_mapping_dir + '/FA.nii.gz -force') 
            command('mrcalc ' + dti_mapping_dir + '/MD.nii.gz -finite ' + dti_mapping_dir + '/MD.nii.gz 0.0 -if ' + dti_mapping_dir + '/MD.nii.gz -force') 
            command('mrcalc ' + dti_mapping_dir + '/AD.nii.gz -finite ' + dti_mapping_dir + '/AD.nii.gz 0.0 -if ' + dti_mapping_dir + '/AD.nii.gz -force') 
            command('mrcalc ' + dti_mapping_dir + '/RD.nii.gz -finite ' + dti_mapping_dir + '/RD.nii.gz 0.0 -if ' + dti_mapping_dir + '/RD.nii.gz -force') 
            command('mrcalc ' + dti_mapping_dir + '/dec.nii.gz -finite ' + dti_mapping_dir + '/dec.nii.gz 0.0 -if ' + dti_mapping_dir + '/dec.nii.gz -force') 
            
            # filter abnormal voxel intensity
            command('mrcalc ' + dti_mapping_dir + '/FA.nii.gz 1 -le ' + dti_mapping_dir + 
                    '/FA.nii.gz 1 -if - | mrcalc - 0.0 -gt - 0.0 -if ' + dti_mapping_dir + '/FA.nii.gz -force')
            command('mrcalc ' + dti_mapping_dir + '/MD.nii.gz 0.005 -le ' + dti_mapping_dir + 
                    '/MD.nii.gz 0.005 -if - | mrcalc - 0.0 -gt - 0.0 -if ' + dti_mapping_dir + '/MD.nii.gz -force')
            command('mrcalc ' + dti_mapping_dir + '/AD.nii.gz 0.005 -le ' + dti_mapping_dir + 
                    '/AD.nii.gz 0.005 -if - | mrcalc - 0.0 -gt - 0.0 -if ' + dti_mapping_dir + '/AD.nii.gz -force')
            command('mrcalc ' + dti_mapping_dir + '/RD.nii.gz 0.005 -le ' + dti_mapping_dir + 
                    '/RD.nii.gz 0.005 -if - | mrcalc - 0.0 -gt - 0.0 -if ' + dti_mapping_dir + '/RD.nii.gz -force')
            

    #-----------------------------------------------------------------
    # Step 2: TractSeg
    #-----------------------------------------------------------------
    if 'tract' in modes_ls:
        app_console('run Step 2: fiber tract segmentation')
        if args.bundle_json:
            try:
                with open(args.bundle_json, 'r') as f:
                    bundle_dic = json.load(f)
                bundle_ls = bundle_dic['bundle_ls']
            except:
                app_error('the external bundle json file is not correct')
        else:
            bundle_ls = args.bundle_list.split(",")

        dwi_fiber_dir = 'fiber_tracking'
        if not os.path.exists(os.path.join(tempDir, dwi_fiber_dir)):
            os.mkdir(os.path.join(tempDir, dwi_fiber_dir))

        dwi_fiber_mni_dir = 'fiber_tracking_MNI'
        if not os.path.exists(os.path.join(tempDir, dwi_fiber_mni_dir)):
            os.mkdir(os.path.join(tempDir, dwi_fiber_mni_dir))

        # generate peaks file in MNI space
        bundle_segmentations_MNI_dir = os.path.join(dwi_fiber_mni_dir, 'bundle_segmentations')
        bundle_segmentations_dir = os.path.join(dwi_fiber_dir, 'bundle_segmentations')
        if not os.path.exists(bundle_segmentations_dir):
            os.mkdir(bundle_segmentations_dir)
        app_console('generate bundle_segmentations in MNI space')
        if args.resume and os.path.exists(bundle_segmentations_MNI_dir): # if -resume, check if output file exist or not
            app_console('bundle_segmentations folder found, skip this step')
        else:
            command('TractSeg -i ' + input_dwi_mni_nii + ' --bvals ' + input_dwi_mni_bval +
                    ' --bvecs ' + input_dwi_mni_bvec + ' --raw_diffusion_input -o ' +
                    dwi_fiber_mni_dir + ' --brain_mask ' + input_dwi_mni_mask_nii)

        # generate Tract Orientation Maps (TOMs) in MNI space
        app_console('generate Tract Orientation Maps (TOMs) in MNI space')
        if args.resume and os.path.exists(os.path.join(dwi_fiber_mni_dir, 'TOM')):  # if -resume, check if output file exist or not
            app_console('TOM folder found, skip this step')
        else:
            command('TractSeg -i ' + dwi_fiber_mni_dir + '/peaks.nii.gz -o ' + dwi_fiber_mni_dir + ' --output_type TOM')

        # generate endings_segmentations in MNI space
        endings_segmentations_MNI_dir = os.path.join(dwi_fiber_mni_dir, 'endings_segmentations')
        endings_segmentations_dir = os.path.join(dwi_fiber_dir, 'endings_segmentations')
        if not os.path.exists(endings_segmentations_dir):
            os.mkdir(endings_segmentations_dir)
        app_console('generate endings_segmentations in MNI space')
        if args.resume and os.path.exists(os.path.join(dwi_fiber_mni_dir, 'endings_segmentations')): # if -resume, check if output file exist or not
            app_console('endings_segmentation folder found, skip this step')
        else:
            command('TractSeg -i ' + dwi_fiber_mni_dir + '/peaks.nii.gz -o ' + dwi_fiber_mni_dir + ' --output_type endings_segmentation')

        # fiber tracking for specific tracts
        app_console('fiber tracking for specific tracts')
        tck_output_dir = 'Fibers'
        tck_native_dir = os.path.join(dwi_fiber_dir, 'Fibers')
        if args.resume and os.path.exists(tck_native_dir):  # if -resume, check if output file exist or not
            app_console('Fibers_native folder found, skip this step')
        else:
            if args.no_endmask_filtering:
                app_console('tracking type: FACT tracking on TOM (without filtering results by tract mask and endpoint masks)')
                for i in bundle_ls:
                    command('Tracking -i ' + dwi_fiber_mni_dir + '/peaks.nii.gz -o ' + dwi_fiber_mni_dir + ' --tracking_dir ' + tck_output_dir +
                    ' --nr_fibers ' + str(args.fiber_num) + ' --no_filtering_by_endpoints --tracking_format tck --bundles ' + i)
            else:
                if args.tracking == 'prob' and args.odf == 'tom':
                    app_console('tracking type: probabilistic tracking on TOM (TractSeg prob, not iFOD2)')
                    for i in bundle_ls:
                        command('Tracking -i ' + dwi_fiber_mni_dir + '/peaks.nii.gz -o ' + dwi_fiber_mni_dir + ' --tracking_dir ' + tck_output_dir +
                        ' --nr_fibers ' + str(args.fiber_num) + ' --track_FODs False --algorithm prob --tracking_format tck --bundles ' + i)
                    
                elif args.tracking == 'fact' and args.odf == 'tom':
                    app_console('tracking type: deterministic tracking on TOM (FACT)')
                    for i in bundle_ls:
                        command('Tracking -i ' + dwi_fiber_mni_dir + '/peaks.nii.gz -o ' + dwi_fiber_mni_dir + ' --tracking_dir ' + tck_output_dir +
                        ' --nr_fibers ' + str(args.fiber_num) + ' --track_FODs False --algorithm det --tracking_format tck --bundles ' + i)
                    
                elif args.tracking == 'sd_stream' and args.odf == 'peaks':
                    app_console('tracking type: deterministic tracking on peaks (SD_STREAM)')
                    for i in bundle_ls:
                        command('Tracking -i ' + dwi_fiber_mni_dir + '/peaks.nii.gz -o ' + dwi_fiber_mni_dir + ' --tracking_dir ' + tck_output_dir +
                        ' --nr_fibers ' + str(args.fiber_num) + ' --track_FODs iFOD2 --tracking_format tck --bundles ' + i)
                
                elif args.tracking == 'fact' and args.odf == 'peaks':
                    app_console('tracking type: deterministic tracking on peaks (FACT)')
                    for i in bundle_ls:
                        command('Tracking -i ' + dwi_fiber_mni_dir + '/peaks.nii.gz -o ' + dwi_fiber_mni_dir + ' --tracking_dir ' + tck_output_dir +
                        ' --nr_fibers ' + str(args.fiber_num) + ' --track_FODs iFOD2 --algorithm det --tracking_format tck --bundles ' + i)
                else:
                    app_error('Script error: such tracking configuration is not currently supported yet')
            
            # transform bundle segmentations back to dwi native space
            for i in bundle_ls:
                command('antsApplyTransforms -d 3 -i ' + os.path.join(bundle_segmentations_MNI_dir, i) + '.nii.gz -r ' + 
                        input_dwi_mask_nii + ' -t [' + input_dwi_mni_antsmat +
                        ',1] -o ' + os.path.join(bundle_segmentations_dir, i) + '.nii.gz ' + ' -v -n GenericLabel[Linear] -f 0')
            
            # transform endings segmentations back to dwi native space
            for i in bundle_ls:
                command('antsApplyTransforms -d 3 -i ' + os.path.join(endings_segmentations_MNI_dir, i) + '_b.nii.gz -r ' + 
                        input_dwi_mask_nii + ' -t [' + input_dwi_mni_antsmat +
                        ',1] -o ' + os.path.join(endings_segmentations_dir, i) + '_b.nii.gz ' + ' -v -n GenericLabel[Linear] -f 0')
                command('antsApplyTransforms -d 3 -i ' + os.path.join(endings_segmentations_MNI_dir, i) + '_e.nii.gz -r ' + 
                        input_dwi_mask_nii + ' -t [' + input_dwi_mni_antsmat +
                        ',1] -o ' + os.path.join(endings_segmentations_dir, i) + '_e.nii.gz ' + ' -v -n GenericLabel[Linear] -f 0')

            # transform tck files back to dwi native space
            app_console('transform tck files back to dwi native space')
            if not os.path.exists(os.path.join(tempDir, tck_native_dir)):
                os.mkdir(os.path.join(tempDir, tck_native_dir))
            
            command('warpinit ' + input_dwi_b0_nii + ' ' + dwi_fiber_mni_dir + '/flirt.nii.gz -force')
            command('transformconvert ' + input_dwi_mni_fslmat + ' ' + input_dwi_b0_nii + ' ' + 
                    input_dwi_mni_b0_nii + ' flirt_import ' + dwi_fiber_mni_dir + '/native_2_mni.mrtrix  -force')
            command('mrtransform ' + dwi_fiber_mni_dir + '/flirt.nii.gz -linear ' + dwi_fiber_mni_dir + '/native_2_mni.mrtrix ' +
                    dwi_fiber_mni_dir + '/flirt2tck.mif -force')

            bundle_valid_ls = []
            for i in bundle_ls:
                # generate tck in dwi native space
                command('tcktransform -i ' + os.path.join(dwi_fiber_mni_dir, tck_output_dir, i + '.tck') + ' ' +  
                        dwi_fiber_mni_dir + '/flirt2tck.mif ' + os.path.join(tck_native_dir, i + '.tck'))
                # generate fiber streamline statistics        
                command('tckstats ' + os.path.join(tck_native_dir, i + '.tck') + 
                        ' -dump ' + os.path.join(tck_native_dir, i + '.txt'))
                # check if any bundle tracking fails
                if os.path.getsize(os.path.join(tck_native_dir, i + '.txt')) == 0:
                    app_warn('Warning: Bundle ' + i + ' has no streamline tracked!')
                    os.remove(os.path.join(tck_native_dir, i + '.txt'))
                else:
                    bundle_valid_ls.append(i)
            # export missing bundle names to tempDir 
            if len(bundle_ls) != len(bundle_valid_ls):
                missing_bundles = {}
                missing_bundles['missing_bundles'] = list(set(bundle_ls).difference(set(bundle_valid_ls)))
                write_json_file(os.path.join(tempDir, 'missing_bundles.json'), missing_bundles)
            bundle_ls = bundle_valid_ls

        app_console('DTI mapping on tractseg results')
        dti_metrics_dir = os.path.join(dwi_fiber_mni_dir, 'DTI_metrics_on_tck')
        if args.resume and os.path.exists(dti_metrics_dir):  # if -resume, check if output file exist or not
            app_console('DTI_metrics_on_tck folder found, skip this step')
        else:
            if not os.path.exists(os.path.join(tempDir, dti_metrics_dir)):
                os.mkdir(os.path.join(tempDir, dti_metrics_dir))

            for i in bundle_ls:
                command('tcksample ' + os.path.join(dwi_fiber_dir, tck_output_dir, i + '.tck') + ' ' + 
                        dti_mapping_dir + '/FA.nii.gz ' + os.path.join(dti_metrics_dir, 'FA_' + i + '.txt') + ' -stat_tck mean -nthreads 0')
                command('tcksample ' + os.path.join(dwi_fiber_dir, tck_output_dir, i + '.tck') + ' ' + 
                        dti_mapping_dir + '/MD.nii.gz ' + os.path.join(dti_metrics_dir, 'MD_' + i + '.txt') + ' -stat_tck mean -nthreads 0')
                command('tcksample ' + os.path.join(dwi_fiber_dir, tck_output_dir, i + '.tck') + ' ' + 
                        dti_mapping_dir + '/AD.nii.gz ' + os.path.join(dti_metrics_dir, 'AD_' + i + '.txt') + ' -stat_tck mean -nthreads 0')
                command('tcksample ' + os.path.join(dwi_fiber_dir, tck_output_dir, i + '.tck') + ' ' + 
                        dti_mapping_dir + '/RD.nii.gz ' + os.path.join(dti_metrics_dir, 'RD_' + i + '.txt') + ' -stat_tck mean -nthreads 0')

            # save statistics as json
            statistic_summary_fiber('fiber', bundle_ls, tck_native_dir)
            statistic_summary_fiber('FA', bundle_ls, dti_metrics_dir)
            statistic_summary_fiber('MD', bundle_ls, dti_metrics_dir)
            statistic_summary_fiber('AD', bundle_ls, dti_metrics_dir)
            statistic_summary_fiber('RD', bundle_ls, dti_metrics_dir)
            
            shutil.move(os.path.join(tck_native_dir, 'fiber_streamline.json'), 
                            os.path.join(tempDir, dwi_fiber_dir))
            shutil.copyfile(os.path.join(dti_metrics_dir, 'FA_streamline.json'), 
                            os.path.join(dti_mapping_dir, 'FA_streamline.json'))
            shutil.copyfile(os.path.join(dti_metrics_dir, 'MD_streamline.json'), 
                            os.path.join(dti_mapping_dir, 'MD_streamline.json'))
            shutil.copyfile(os.path.join(dti_metrics_dir, 'AD_streamline.json'), 
                            os.path.join(dti_mapping_dir, 'AD_streamline.json'))
            shutil.copyfile(os.path.join(dti_metrics_dir, 'RD_streamline.json'), 
                            os.path.join(dti_mapping_dir, 'RD_streamline.json'))
            
            app_console('DTI parameter mapping on bundle mask')
            dti_metrics_bundle_dir = os.path.join(dwi_fiber_mni_dir, 'DTI_metrics_on_bundle')
            if not os.path.exists(os.path.join(tempDir, dti_metrics_bundle_dir)):
                os.mkdir(os.path.join(tempDir, dti_metrics_bundle_dir))

            statistic_summary_fiber('FA', bundle_ls, dti_mapping_dir, map_dir=dti_mapping_dir, 
                        bundle_dir=bundle_segmentations_dir) 
            statistic_summary_fiber('MD', bundle_ls, dti_mapping_dir, map_dir=dti_mapping_dir, 
                        bundle_dir=bundle_segmentations_dir)
            statistic_summary_fiber('AD', bundle_ls, dti_mapping_dir, map_dir=dti_mapping_dir, 
                        bundle_dir=bundle_segmentations_dir)
            statistic_summary_fiber('RD', bundle_ls, dti_mapping_dir, map_dir=dti_mapping_dir, 
                        bundle_dir=bundle_segmentations_dir)   

        app_console('complete Step 2: fiber tract segmentation')
        app_console('-------------------------------------')


    #-----------------------------------------------------------------
    # Step 4: DKI parameter mapping 
    #-----------------------------------------------------------------
    if 'dki_para' in modes_ls:
        app_console('Step 4: DKI parameter mapping ')
        dki_mapping_dir = 'DKI_mapping'
        
        img = dp.DWI(imPath=input_dwi_nii, bvecPath = input_dwi_bvec, 
                        bvalPath = input_dwi_bval, mask = input_dwi_mask_nii)
        if args.resume and os.path.exists(dki_mapping_dir):  # if -resume, check if output file exist or not
            app_console('DKI_mapping folder found, skip this step')
        elif not img.isdki(): 
            app_warn('DWI data is not based on DKI scanning protocols, skip this step')
        else:
            # modified from pydesigner https://github.com/m-ama/PyDesigner
            
            # Define filenames
            fn_dti_md = 'dti_md'
            fn_dti_rd = 'dti_rd'
            fn_dti_ad = 'dti_ad'
            fn_dti_fa = 'dti_fa'
            fn_dki_mk = 'dki_mk'
            fn_dki_rk = 'dki_rk'
            fn_dki_ak = 'dki_ak'
            fn_dki_kfa = 'dki_kfa'
            fn_ext = '.nii.gz'

            fit_constraints = np.fromstring('0,1,0', dtype=int, sep=',')
            # irlls noise remove
            outliers, dt_est = img.irlls(mode='DKI')
            img.fit(fit_constraints, reject=outliers)

            tensorType = 'dki'
            DT, KT = img.tensorReorder(tensorType)
            os.mkdir(os.path.join(tempDir, dki_mapping_dir))

            md, rd, ad, fa = img.extractDTI()
            dp.writeNii(md, img.hdr, os.path.join(dki_mapping_dir, fn_dti_md + fn_ext))
            dp.writeNii(rd, img.hdr, os.path.join(dki_mapping_dir, fn_dti_rd + fn_ext))
            dp.writeNii(ad, img.hdr, os.path.join(dki_mapping_dir, fn_dti_ad + fn_ext))
            dp.writeNii(fa, img.hdr, os.path.join(dki_mapping_dir, fn_dti_fa + fn_ext))

            for x in [fn_dti_md, fn_dti_rd, fn_dti_ad, fn_dti_fa]:
                filters.median(
                    input=os.path.join(dki_mapping_dir, x + fn_ext),
                    output=os.path.join(dki_mapping_dir, x + fn_ext),
                    mask=input_dwi_mask_nii)

            mk, rk, ak, kfa, mkt = img.extractDKI()
            
            # naive implementation of writing these variables
            dp.writeNii(mk, img.hdr, os.path.join(dki_mapping_dir, fn_dki_mk + fn_ext))
            dp.writeNii(rk, img.hdr, os.path.join(dki_mapping_dir, fn_dki_rk + fn_ext))
            dp.writeNii(ak, img.hdr, os.path.join(dki_mapping_dir, fn_dki_ak + fn_ext))
            dp.writeNii(kfa, img.hdr, os.path.join(dki_mapping_dir, fn_dki_kfa + fn_ext))
            
            # median filter
            for x in [fn_dki_mk, fn_dki_rk, fn_dki_ak, fn_dki_kfa]:
                filters.median(
                    input=os.path.join(dki_mapping_dir, x + fn_ext),
                    output=os.path.join(dki_mapping_dir, x + fn_ext),
                    mask=input_dwi_mask_nii)
            
            # filter abnormal voxel intensity
            command('mrcalc ' + os.path.join(dki_mapping_dir, fn_dki_mk + fn_ext) + ' 5 -le ' + os.path.join(dki_mapping_dir, fn_dki_mk + fn_ext) + 
                    ' 5 -if - | mrcalc - 0.0 -gt - 0.0 -if ' + os.path.join(dki_mapping_dir, fn_dki_mk + fn_ext) + ' -force')
            command('mrcalc ' + os.path.join(dki_mapping_dir, fn_dki_rk + fn_ext) + ' 5 -le ' + os.path.join(dki_mapping_dir, fn_dki_rk + fn_ext) + 
                    ' 5 -if - | mrcalc - 0.0 -gt - 0.0 -if ' + os.path.join(dki_mapping_dir, fn_dki_rk + fn_ext) + ' -force')
            command('mrcalc ' + os.path.join(dki_mapping_dir, fn_dki_ak + fn_ext) + ' 5 -le ' + os.path.join(dki_mapping_dir, fn_dki_ak + fn_ext) + 
                    ' 5 -if - | mrcalc - 0.0 -gt - 0.0 -if ' + os.path.join(dki_mapping_dir, fn_dki_ak + fn_ext) + ' -force')
            command('mrcalc ' + os.path.join(dki_mapping_dir, fn_dki_kfa + fn_ext) + ' 1 -le ' + os.path.join(dki_mapping_dir, fn_dki_kfa + fn_ext) + 
                    ' 1 -if - | mrcalc - 0.0 -gt - 0.0 -if ' + os.path.join(dki_mapping_dir, fn_dki_kfa + fn_ext) + ' -force')

            if 'tract' in modes_ls:
                app_console('DKI parameter mapping on bundle mask') 
                statistic_summary_fiber('dki_mk', bundle_ls, dki_mapping_dir, map_dir=dki_mapping_dir, 
                                    bundle_dir=bundle_segmentations_dir) 
                statistic_summary_fiber('dki_rk', bundle_ls, dki_mapping_dir, map_dir=dki_mapping_dir, 
                                    bundle_dir=bundle_segmentations_dir) 
                statistic_summary_fiber('dki_ak', bundle_ls, dki_mapping_dir, map_dir=dki_mapping_dir, 
                                    bundle_dir=bundle_segmentations_dir) 
                statistic_summary_fiber('dki_kfa', bundle_ls, dki_mapping_dir, map_dir=dki_mapping_dir, 
                                    bundle_dir=bundle_segmentations_dir) 
            

        app_console('complete Step 4: DKI parameter mapping')
        app_console('-------------------------------------')


    #-----------------------------------------------------------------
    # Step 5: NODDI parameter mapping 
    #-----------------------------------------------------------------
    if 'noddi_para' in modes_ls:
        app_console('Step 5: NODDI parameter mapping ')
        noddi_mapping_dir = os.path.join(tempDir, label, 'AMICO', 'NODDI')
        img = dp.DWI(imPath=input_dwi_nii, bvecPath = input_dwi_bvec, 
                        bvalPath = input_dwi_bval, mask = input_dwi_mask_nii)
        if args.resume and os.path.exists(noddi_mapping_dir):  # if -resume, check if output file exist or not
            app_console('NODDI_mapping folder found, skip this step')
        elif not img.isdki(): 
            app_warn('DWI data is not based on NODDI scanning protocols, skip this step')
        else:
            import amico
            amico.core.setup()

            ae = amico.Evaluation(tempDir, os.path.basename(dmriprep_dir))
            amico.util.fsl2scheme(input_dwi_bval, input_dwi_bvec)
            step2_output_dwi_scheme = os.path.join(os.path.dirname(input_dwi_bval), 'dwi.scheme')
            if not os.path.exists(step2_output_dwi_scheme):
                app_error('failed to create scheme file from bvals/bvecs files')
            ae.load_data(dwi_filename = input_dwi_nii, 
                        scheme_filename = step2_output_dwi_scheme, 
                        mask_filename = input_dwi_mask_nii, b0_thr = 0)
            ae.set_model("NODDI")
            ae.generate_kernels()
            ae.load_kernels()
            ae.fit()
            ae.save_results()

            if 'tract' in modes_ls:
                app_console('NODDI parameter mapping on bundle mask') 
                statistic_summary_fiber('FIT_ICVF', bundle_ls, noddi_mapping_dir, map_dir=noddi_mapping_dir, 
                                    bundle_dir=bundle_segmentations_dir) 
                statistic_summary_fiber('FIT_ISOVF', bundle_ls, noddi_mapping_dir, map_dir=noddi_mapping_dir, 
                                    bundle_dir=bundle_segmentations_dir) 
                statistic_summary_fiber('FIT_OD', bundle_ls, noddi_mapping_dir, map_dir=noddi_mapping_dir, 
                                    bundle_dir=bundle_segmentations_dir) 

            app_console('complete Step 5: NODDI parameter mapping')
            app_console('-------------------------------------')

    #-----------------------------------------------------------------
    # Step 6: structural connectome generation
    #-----------------------------------------------------------------
    if 'connectome' in modes_ls:
        app_console('run Step 6: structural connectome generation')
        if not os.path.exists(input_dwi_to_t1_ANTs_mat):
            app_error("DWI to T1 affine matrix is not detected in the input directory")
        if not os.path.exists(input_mni_to_t1_ANTs_nii):
            app_error("MNI to T1 warpping map is not detected in the input directory. Please rerun t1prep pipeline with -MNInormalization mode")
        
    
        atlas_config = read_json_file(args.atlas_config)

        connectome_dir = os.path.join(tempDir, 'connectome')
        num_tracks = args.wholebrain_fiber_num
        brain_tck = os.path.join(os.path.join(connectome_dir, 'brain_raw.tck'))
        WM_fods_path = os.path.join(os.path.join(connectome_dir, 'FOD_WM.mif'))

        if os.path.exists(connectome_dir):  # if -resume, check if output file exist or not
            app_console('connectome_dir folder found, skip this step')
        else:
            os.mkdir(connectome_dir)
                
            if multishell:
                command('dwi2response dhollander ' + input_dwi_mif + ' ' +
                        os.path.join(connectome_dir, 'response_wm.txt') + ' ' +
                        os.path.join(connectome_dir, 'response_gm.txt') + ' ' +
                        os.path.join(connectome_dir, 'response_csf.txt') +
                        ' -mask ' + input_dwi_mask_nii)
                command('dwi2fod msmt_csd ' + input_dwi_mif + ' ' +
                        os.path.join(connectome_dir, 'response_wm.txt') + ' ' + os.path.join(connectome_dir, 'FOD_WM.mif') + ' ' +
                        os.path.join(connectome_dir, 'response_gm.txt') + ' ' + os.path.join(connectome_dir, 'FOD_GM.mif') + ' ' +
                        os.path.join(connectome_dir, 'response_csf.txt') + ' ' + os.path.join(connectome_dir, 'FOD_CSF.mif') +
                        ' -lmax 10,0,0  -mask ' + input_dwi_mask_nii)
                command('mrconvert ' + os.path.join(connectome_dir, 'FOD_WM.mif') + ' - -coord 3 0 | '
                        'mrcat ' + os.path.join(connectome_dir, 'FOD_CSF.mif') + ' ' +
                        os.path.join(connectome_dir, 'FOD_GM.mif') + ' - ' + os.path.join(connectome_dir, 'tissues.mif') + ' -axis 3')
            else:
                command('dwi2response tournier ' +  input_dwi_mif + ' ' + os.path.join(connectome_dir, 'response_wm.txt') + \
                        ' -mask ' + input_dwi_mask_nii )
                command('dwi2fod csd ' + input_dwi_mif + ' ' + os.path.join(connectome_dir, 'response_wm.txt') + ' ' + \
                        os.path.join(connectome_dir, 'FOD_WM.mif') + ' -mask ' + input_dwi_mask_nii)
                    
            if os.path.exists(input_t1_5tt_nii):
                app_console('Yes! The 5TT file detected, start to generate whole-brain tracks with ACT..')
                command('tckgen ' + WM_fods_path + ' ' + brain_tck + ' -act ' + input_t1_5tt_nii + \
                    ' -maxlength 300 -cutoff 0.05 ' + \
                    '-select ' + str(num_tracks) + ' -seed_dynamic ' + WM_fods_path) # do not need a brainmask here, ACT already provided
                command('tcksift2 ' + brain_tck + ' ' + WM_fods_path + ' ' + os.path.join(connectome_dir, 'weights.csv') + \
                    ' -fd_thresh 0.05 -act ' + input_t1_5tt_nii)
            else:
                app_console('Sorry! The 5TT file was not detected, start to generate whole-brain tracks without ACT..')
                command('tckgen ' + WM_fods_path + ' ' + brain_tck + ' -mask ' + input_dwi_mask_nii + 
                        ' -maxlength 300 -cutoff 0.05 -select ' + str(num_tracks) + ' -seed_dynamic ' + WM_fods_path)
                command('tcksift2 ' + brain_tck + ' ' + WM_fods_path + ' ' + os.path.join(connectome_dir, 'weights.csv') + \
                    ' -fd_thresh 0.05')
        

        # process for atlases
        freesurfer_path = os.path.join(args.bids_dir, 'derivatives', 'freesurfer', label)
        mrtrix_lut_dir = os.path.join('/mrtrix3', 'labelconvert')
        atlases_ls = args.atlases
        for atlas_name in atlases_ls:
            ## for each brain atlas
            # convert from FreeSurfer Space Back to Native Anatomical Space (https://surfer.nmr.mgh.harvard.edu/fswiki/FsAnat-to-NativeAnat)
            if atlas_name == 'desikan_T1w':
                parc_native_path = os.path.join(freesurfer_path, 'mri', 'aparc+aseg.mgz')
                parc_lut_file = os.path.join(mrtrix_lut_dir, 'FreeSurferColorLUT.txt')
                mrtrix_lut_file = os.path.join(mrtrix_lut_dir, 'fs_default.txt')
                parc_T1w_nii_path = os.path.join(connectome_dir, label + '_T1w_desikan.nii.gz')
            elif atlas_name == 'destrieux_T1w':
                parc_native_path = os.path.join(freesurfer_path, 'mri', 'aparc.a2009s+aseg.mgz')
                parc_lut_file = os.path.join(mrtrix_lut_dir, 'FreeSurferColorLUT.txt')
                mrtrix_lut_file = os.path.join(mrtrix_lut_dir, 'fs_a2009s.txt')
                parc_T1w_nii_path = os.path.join(connectome_dir, label + '_T1w_destrieux.nii.gz')
            elif atlas_name == 'hcpmmp_T1w':
                parc_native_path = os.path.join(freesurfer_path, 'mri', 'aparc.HCPMMP1+aseg.mgz')
                if not os.path.exists(parc_native_path):
                    app_error('Failed to detect ' + parc_native_path + ', should run docker image bids-freesurfer /hcpmmp_conv.py first')

                parc_lut_file = os.path.join(mrtrix_lut_dir, 'hcpmmp1_original.txt')
                mrtrix_lut_file = os.path.join(mrtrix_lut_dir, 'hcpmmp1_ordered.txt')
                parc_T1w_nii_path = os.path.join(connectome_dir, label + '_T1w_hcpmmp.nii.gz')

            if atlas_name.split('_')[1] == 'T1w':
                if not os.path.exists(freesurfer_path):
                    app_error("Failed to detect /derivatives/freesurfer for subject " + label)
                if os.path.exists(parc_T1w_nii_path):
                    app_console("found results of freesurfer post-processing, jump this step")
                else:
                    app_console("start running freesurfer post-processing")
                    command('labelconvert ' + parc_native_path + ' ' + parc_lut_file + ' ' + mrtrix_lut_file + ' ' + parc_T1w_nii_path)
                atlas_config[atlas_name]['parc_nii'] = parc_T1w_nii_path
            
            # load LUT
            sgm_lut_path = atlas_config[atlas_name]['sgm_lut']
            if not os.path.exists(sgm_lut_path):
                app_error('The following sgm_lut not exists: ' + sgm_lut_path)
            else:
                sgm_lut = pd.read_csv(sgm_lut_path)

            if atlas_name.split('_')[-1] == 'MNI':
                # inverse warp atlas to dwi native space
                command('antsApplyTransforms -d 3 -i ' + atlas_config[atlas_name]['parc_nii'] + ' -r ' + input_t1_bet_mask_nii + 
                        ' -o ' + os.path.join(connectome_dir, atlas_name+'_dwispace.nii.gz') +  ' -t [' + 
                        input_dwi_to_t1_ANTs_mat + ',1] -t ' + input_mni_to_t1_ANTs_nii + ' -n GenericLabel[Linear]')
                atlas_config[atlas_name]['parc_nii'] = os.path.join(connectome_dir, atlas_name+'_dwispace.nii.gz')

            # if label index != intensity, relabel it
            if not (sgm_lut.Index == sgm_lut.Intensity).all():
                atlas_input_nii = nib.load(atlas_config[atlas_name]['parc_nii'])
                atlas_input_nii_img = atlas_input_nii.get_fdata()
                atlas_output_path = os.path.join(connectome_dir, atlas_name + '_relabel.nii.gz')
                for item in range(len(sgm_lut)):
                    if sgm_lut.Index[item] != sgm_lut.Intensity[item]:
                        np.place(atlas_input_nii_img, atlas_input_nii_img == sgm_lut.Intensity[item], sgm_lut.Index[item])
                new_label = nib.Nifti1Image(np.int16(atlas_input_nii_img), atlas_input_nii.affine, atlas_input_nii.header)
                nib.save(new_label, atlas_output_path)
                atlas_config[atlas_name]['parc_nii'] = atlas_output_path

            # connectome generation
            app_console('Combining whole-brain tractogram with grey matter parcellation to produce the connectome')

            atlas_dir = os.path.join(connectome_dir, atlas_name)
            if not os.path.exists(atlas_dir):
                os.mkdir(atlas_dir)
            command('tck2connectome ' + brain_tck + ' ' + atlas_config[atlas_name]['parc_nii'] + 
                    ' ' + os.path.join(atlas_dir, atlas_name + '_connectome.csv') + ' -tck_weights_in ' + os.path.join(connectome_dir, 'weights.csv') +
                    ' -symmetric -zero_diagonal -force')
            command('tck2connectome ' + brain_tck + ' ' + atlas_config[atlas_name]['parc_nii'] + 
                    ' ' + os.path.join(atlas_dir, atlas_name + '_meanlength.csv') + ' -tck_weights_in ' + os.path.join(connectome_dir, 'weights.csv') +
                    ' -scale_length -symmetric -zero_diagonal -stat_edge mean -force')
            command('tck2connectome ' + brain_tck + ' ' + atlas_config[atlas_name]['parc_nii'] + 
                    ' ' + os.path.join(atlas_dir, atlas_name + '_invnodevol.csv') + ' -tck_weights_in ' + os.path.join(connectome_dir, 'weights.csv') +
                    ' -scale_invnodevol -symmetric -zero_diagonal -force')      
        
        
        
    #-----------------------------------------------------------------
    # Step 7: move files to output directory
    #-----------------------------------------------------------------
    app_console('Step 7: move files to output directory')
    if os.path.exists(output_dir):
        app_warn('Found output directory existing, delete it and create a new one')
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    
    if 'tract' in modes_ls:
        output_tract_dir = os.path.join(output_dir, 'fiber_tracts')
        shutil.copytree(os.path.join(tempDir, dwi_fiber_dir), output_tract_dir)

    if any(item in modes_ls for item in ['tract', 'dti_para']):
        shutil.copytree(os.path.join(tempDir, dti_mapping_dir), os.path.join(output_dir, dti_mapping_dir))
    
    if 'dki_para' in modes_ls and os.path.exists(os.path.join(tempDir, dki_mapping_dir)):
        shutil.copytree(os.path.join(tempDir, dki_mapping_dir), os.path.join(output_dir, dki_mapping_dir))

    if 'noddi_para' in modes_ls and os.path.exists(noddi_mapping_dir):
        shutil.copytree(noddi_mapping_dir, os.path.join(output_dir, 'NODDI_mapping'))

    if 'connectome' in modes_ls and os.path.exists(connectome_dir):
        shutil.copytree(connectome_dir, os.path.join(output_dir, 'connectome'))

    #-----------------------------------------------------------------
    # Step 8: Visualization conversion
    #-----------------------------------------------------------------
    if not no_vtp:
        app_console('run Step 8: visualization conversion')
        output_vis_Dir = os.path.join(output_dir, 'visualization')
        if not os.path.exists(output_vis_Dir):
            os.mkdir(output_vis_Dir)
        vtk_native_dir = os.path.join(output_vis_Dir, 'Fibers_vtk')
        vtp_native_dir = os.path.join(output_vis_Dir, 'Fibers_vtp')
        bundle_seg_native_dir = os.path.join(output_vis_Dir, 'Fibers_bundlemask_vtp')
        endings_seg_native_dir = os.path.join(output_vis_Dir, 'Fibers_endingmask_vtp')

        if not os.path.exists(vtk_native_dir):
            os.mkdir(vtk_native_dir)
        
        if not os.path.exists(vtp_native_dir):
            os.mkdir(vtp_native_dir)

        if not os.path.exists(bundle_seg_native_dir):
            os.mkdir(bundle_seg_native_dir)

        if not os.path.exists(endings_seg_native_dir):
            os.mkdir(endings_seg_native_dir)

        if 'tract' in modes_ls:
            object_visualization.tck2vtk_batch(tck_native_dir, vtk_native_dir, input_dwi_b0_nii, bundle_ls)
            object_visualization.vtk2vtp_batch(vtk_native_dir, vtp_native_dir, bundle_ls)
            for i in bundle_ls:
                command('/Roi3D/VtpRoi3DGenerator ' + os.path.join(bundle_segmentations_dir, i + '.nii.gz') + 
                    ' /Roi3D/mask_lookup_table.json ' + bundle_seg_native_dir + '/' + i)
                command('/Roi3D/VtpRoi3DGenerator ' + os.path.join(endings_segmentations_dir, i + '_b.nii.gz') + 
                    ' /Roi3D/mask_lookup_table.json ' + endings_seg_native_dir + '/' + i + '_b')
                command('/Roi3D/VtpRoi3DGenerator ' + os.path.join(endings_segmentations_dir, i + '_e.nii.gz') + 
                    ' /Roi3D/mask_lookup_table.json ' + endings_seg_native_dir + '/' + i + '_e')

    else:
        app_console('jump visualization conversion step')

    end = time.time()
    Execution_time = str(round((end - start)/60, 2))
    app_console('dwi recon finished. Execution time: ' + Execution_time)

    if cleanup:
        shutil.rmtree(tempDir)
    tempDir = ''


if __name__ == "__main__":
    global cleanup, resume, tempDir, workingDir
    tempDir = ''

    parser = argparse.ArgumentParser(description="diffusion MRI data analysis on participant level",
                                     epilog="author chenfei.ye@foxmail.com")

    parser.add_argument('bids_dir', help='The directory with the input dataset '
                        'formatted according to the BIDS standard.')
    parser.add_argument('output_dir', help='The directory where the output files '
                        'should be stored. If you are running group level analysis '
                        'this folder should be prepopulated with the results of the'
                        'participant level analysis.')
    parser.add_argument('analysis_level', help='Level of the analysis that will be performed. '
                        'Multiple participant level analyses can be run independently '
                        '(in parallel) using the same output_dir.',
                        choices=['participant', 'group'])
    parser.add_argument('--participant_label', help='The label(s) of the participant(s) that should be analyzed. The label '
                        'corresponds to sub-<participant_label> from the BIDS spec '
                        '(so it does not include "sub-"). If this parameter is not '
                        'provided all subjects should be analyzed. Multiple '
                        'participants can be specified with a space separated list.',
                        nargs="+")
    parser.add_argument('--session_label', help='The label of the session that should be analyzed. The label '
                        'corresponds to ses-<session_label> from the BIDS spec '
                        '(so it does not include "ses-"). If this parameter is not '
                        'provided, all sessions should be analyzed. Multiple '
                        'sessions can be specified with a space separated list.',
                        nargs="+")
    parser.add_argument("-mode", metavar="tract|dti_para|dki_para|noddi_para|connectome",
                        help="Which type of dMRI analysis mode to run.\n"
                             "'tract' [DEFAULT]: fiber tracking for predefined tracts.\n"
                             "'dti_para': DTI parameter mapping, generating FA/MD/AD/RD. \n"
                             "'dki_para': DKI parameter mapping, generating MK/RK/AK/KFA/MKT. \n"
                             "'noddi_para': NODDI parameter mapping, generating ICVF/IVF/ODI. \n"
                             "'connectome': structural network creation. \n"
                             "multiple modes can be switched on simultaneously by seperating by comma",
                        default="tract")

    parser.add_argument("-fiber_num", metavar="num", dest="fiber_num", type=int,
                        help="number of fiber for each tract (default: 2000)",
                        default=2000)

    parser.add_argument('-odf', metavar="peaks|tom", choices=["peaks", "tom"],
                        help="Select which orientation distribution function (ODF) to use. "
                             "'peaks' is MRtrix-based file"
                             "'tom' is tract_seg-based Tract Orientation Map (TOM). Simplified version of peaks (default)",
                        default="tom")

    parser.add_argument('-tracking', metavar="prob|det", choices=["prob", "fact", "sd_stream"],
                        help="Select which tractography method to use. "
                             "'prob' is probabilistic method (iFOD2, default)"
                             "'fact' is deterministic method (FACT). very fast."
                             "'sd_stream' is deterministic method (sd_stream). slow.",
                        default="prob")

    parser.add_argument("-no_endmask_filtering", action="store_true",
                        help="Run tracking on TOMs without filtering results by "
                        "tract mask and endpoint masks. MRtrix FACT tracking "
                        "will be forced.",
                        default=False)
    parser.add_argument('-atlas_config', help="path of atlas_config.json", default='/atlases/atlas_config_docker.json')
    parser.add_argument('-atlases', 
                        help="Select predefined atlases to use. ",
                        nargs="+",
                        default="AAL3_MNI")

    parser.add_argument("-wholebrain_tract", action="store_true",
                        help="whole brain tractogrpahy",
                        default=False)

    parser.add_argument("-wholebrain_fiber_num", metavar="num", dest="wholebrain_fiber_num", type=int,
                        help="number of fiber for wholebrain (default: 10000000)",
                        default=10000000)

    parser.add_argument("-bundle_list", metavar="bundle-name", dest="bundle_list",
                        help="Comma separated list (without spaces) of bundles (default: CC,CST_left,CST_right)",
                        default='CC,CST_left,CST_right')

    parser.add_argument("-bundle_json", metavar="json_path", dest="bundle_json",
                        help="json file containing bundle names of interests. See the full list in /scripts/bundle_list_all72.json")

    parser.add_argument("-resume", action="store_true",
                        help="resume the uncompleted process, for debug only",
                        default=False)
    
    parser.add_argument("-cleanup", action="store_true",
                        help="remove temp folder after finish",
                        default=False)
    
    parser.add_argument("-no_vtp", action="store_true",
                        help="do not convert files to vtp format",
                        default=False)

    parser.add_argument('-v', '--version', action='version',
                        version='BIDS-App example version {}'.format(__version__))

    args = parser.parse_args()
    
    workingDir = args.output_dir
    resume = args.resume
    cleanup = args.cleanup
    start = time.time()
    
    subjects_to_analyze = []
    # only for a subset of subjects
    if args.participant_label:
        subjects_to_analyze = args.participant_label
    # for all subjects
    else:
        subject_dirs = glob.glob(os.path.join(args.bids_dir, "sub-*"))
        subjects_to_analyze = [subject_dir.split("-")[-1] for subject_dir in subject_dirs]
    subjects_to_analyze.sort()
    
   # running participant level
    if args.analysis_level == "participant":
        # find all T1s 
        for subject_label in subjects_to_analyze:
            runSubject(args, subject_label)

    # running group level
    elif args.analysis_level == "group":
        if args.participant_label:
            app_error('Cannot use --participant_label option when performing group analysis')
            app_console('Warning: the group analysis is still in development')
            # runGroup(os.path.abspath(args.output_dir))

    app_complete()
    end = time.time()
    running_time = end - start
    print('running time: {:.0f}min {:.0f}sec'.format(running_time//60, running_time % 60))




