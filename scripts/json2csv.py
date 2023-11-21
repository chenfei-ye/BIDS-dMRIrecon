
import os, sys, shutil, time, glob
import nibabel as nib
import argparse
import numpy as np
import json
import pandas as pd
from dppd import dppd 
dp, X = dppd() 

def read_json_file(json_file, encoding='utf-8'):
    """
    read json file
    :param json_file:
    :param encoding:
    :return:
    """
    if not os.path.exists(json_file):
        print('json file %s not exist' % json_file)
        return None

    with open(json_file, 'r', encoding=encoding) as fp:
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

def summarize_streamline_para(sub_path_ls, json_name):
    """
    pool individual streamline DTI parameter json files into a group-level dataframe(num_sub * num_fiber)
    :param sub_path_ls: glob.glob(os.path.join(main_path, 'sub-*'))
    :param json_name: filename of input json file
    :return: a group-level dataframe(num_sub * num_fiber)
    """
    df_all= pd.DataFrame()
    for i in range(len(sub_path_ls)):
        sub_path = sub_path_ls[i]
        subname = os.path.basename(sub_path)
        json_path = os.path.join(sub_path, 'DTI_mapping', json_name)
        json_file = read_json_file(json_path)
        df_sub = pd.DataFrame.from_dict(json_file)
        df_all[subname] = df_sub.loc['mean',:]

    df_all = dp(df_all).transpose().pd
    return df_all

def summarize_parcel_para(sub_path_ls, json_name):
    """
    pool individual parcel DTI parameter json files into a group-level dataframe(num_sub * num_fiber)
    :param sub_path_ls: glob.glob(os.path.join(main_path, 'sub-*'))
    :param json_name: filename of input json file
    :return: a group-level dataframe(num_sub * num_fiber)
    """
    df_all= pd.DataFrame()
    for i in range(len(sub_path_ls)):
        sub_path = sub_path_ls[i]
        subname = os.path.basename(sub_path)
        json_path = os.path.join(sub_path, 'DTI_mapping', json_name)
        json_file = read_json_file(json_path)
        df_sub = pd.DataFrame.from_dict(json_file,orient='index')
        df_all[subname] = df_sub

    df_all = dp(df_all).transpose().pd
    return df_all


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="pool individual DTI parameter json files into a group-level dataframe",
                                     epilog="Copyright © 2016 - 2021 MindsGo Life Science and Technology Co. Ltd."
                                            " All Rights Reserved")

    parser.add_argument('bids_dir', help='The directory with the input dataset '
                        'formatted according to the BIDS standard.')

    parser.add_argument('analysis_level', help='Level of the analysis that will be performed. '
                        'Multiple participant level analyses can be run independently '
                        '(in parallel) using the same output_dir.',
                        choices=['participant', 'group'])
    
    args = parser.parse_args()
    start = time.time()

    dmri_recon_dir = os.path.join(args.bids_dir, 'derivatives', 'dmri_recon')
    sub_path_ls = glob.glob(os.path.join(dmri_recon_dir, 'sub-*'))
    sub_path_ls.sort()
    num_sub = len(sub_path_ls)

    df_all_streamline_FA= summarize_streamline_para(sub_path_ls, 'FA_streamline.json')
    df_all_streamline_MD= summarize_streamline_para(sub_path_ls, 'MD_streamline.json')
    df_all_streamline_AD= summarize_streamline_para(sub_path_ls, 'AD_streamline.json')
    df_all_streamline_RD= summarize_streamline_para(sub_path_ls, 'RD_streamline.json')

    df_all_bundle_FA= summarize_streamline_para(sub_path_ls, 'FA_bundle.json')
    df_all_bundle_MD= summarize_streamline_para(sub_path_ls, 'MD_bundle.json')
    df_all_bundle_AD= summarize_streamline_para(sub_path_ls, 'AD_bundle.json')
    df_all_bundle_RD= summarize_streamline_para(sub_path_ls, 'RD_bundle.json')

    # df_all_parcel_FA = summarize_parcel_para(sub_path_ls, 'FA_parcel.json')
    # df_all_parcel_MD = summarize_parcel_para(sub_path_ls, 'MD_parcel.json')
    # df_all_parcel_AD = summarize_parcel_para(sub_path_ls, 'AD_parcel.json')
    # df_all_parcel_RD = summarize_parcel_para(sub_path_ls, 'RD_parcel.json')
    
    df_all_streamline_FA.to_csv(os.path.join(dmri_recon_dir, 'Group_FA_streamline.csv'))
    df_all_streamline_MD.to_csv(os.path.join(dmri_recon_dir, 'Group_MD_streamline.csv'))
    df_all_streamline_AD.to_csv(os.path.join(dmri_recon_dir, 'Group_AD_streamline.csv'))
    df_all_streamline_RD.to_csv(os.path.join(dmri_recon_dir, 'Group_RD_streamline.csv'))

    df_all_bundle_FA.to_csv(os.path.join(dmri_recon_dir, 'Group_FA_bundle.csv'))
    df_all_bundle_MD.to_csv(os.path.join(dmri_recon_dir, 'Group_MD_bundle.csv'))
    df_all_bundle_AD.to_csv(os.path.join(dmri_recon_dir, 'Group_AD_bundle.csv'))
    df_all_bundle_RD.to_csv(os.path.join(dmri_recon_dir, 'Group_RD_bundle.csv'))

    # df_all_parcel_FA.to_csv(os.path.join(dmri_recon_dir, 'Group_FA_parcel.csv'))
    # df_all_parcel_MD.to_csv(os.path.join(dmri_recon_dir, 'Group_MD_parcel.csv'))
    # df_all_parcel_AD.to_csv(os.path.join(dmri_recon_dir, 'Group_AD_parcel.csv'))
    # df_all_parcel_RD.to_csv(os.path.join(dmri_recon_dir, 'Group_RD_parcel.csv'))


    end = time.time()
    running_time = end - start
    print('running time: {:.0f}min {:.0f}sec'.format(running_time//60, running_time % 60))