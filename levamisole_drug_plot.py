#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Weds Feb 2 09:42:17 2022

@author: tobrien

Script for looking at cat-1 and N2 strains dosed with:
    -Dopamine
    -Carbidopa
    -Levodopa
    -Pramipexole

"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import chain
from tierpsytools.preprocessing.preprocess_features import impute_nan_inf
from tierpsytools.read_data.hydra_metadata import (align_bluelight_conditions,
                                                   read_hydra_metadata)

sys.path.insert(0, '/Users/tobrien/analysis_repo/DiseaseModelling/hydra_screen/phenotype_summary')
from helper import (read_disease_data,
                    select_strains,
                    filter_features,
                    make_colormaps,
                    find_window,
                    # long_featmap,
                    BLUELIGHT_WINDOW_DICT,
                    DATES_TO_DROP,
                    STIMULI_ORDER)
from plotting_helper import  (plot_colormap,
                              plot_cmap_text,
                              feature_box_plots,
                              average_feature_box_plots,
                              clipped_feature_box_plots,
                              average_feat_swarm,
                              window_errorbar_plots,
                              CUSTOM_STYLE,
                              clustered_barcodes)
from ts_helper import (MODECOLNAMES,
                       plot_frac_by_mode,
                       short_plot_frac_by_mode)
# Set paths for data
ROOT_DIR = Path('/Users/tobrien/Documents/Imperial : MRC/AE_Disease_Model_Strains/drug_analysis/gaba_drugs')
FEAT_FILE =  Path('/Users/tobrien/Documents/Imperial : MRC/AE_Disease_Model_Strains/drug_analysis/gaba_drugs/Results/features_summary_tierpsy_plate_20220204_103445.csv') 
FNAME_FILE = Path('/Users/tobrien/Documents/Imperial : MRC/AE_Disease_Model_Strains/drug_analysis/gaba_drugs/Results/filenames_summary_tierpsy_plate_20220204_103445.csv')
METADATA_FILE = Path('/Users/tobrien/Documents/Imperial : MRC/AE_Disease_Model_Strains/drug_analysis/gaba_drugs/AuxiliaryFiles/final_metadata.csv')
SAVE_PATH = Path('/Users/tobrien/Documents/Imperial : MRC/AE_Disease_Model_Strains/drug_analysis/gaba_drugs/Analysis')
# Select features to be plotted
FEATURES = [
    'motion_mode_paused_fraction_prestim',
    'motion_mode_forward_fraction_prestim',
    'motion_mode_backward_fraction_prestim',
    'motion_mode_paused_fraction_poststim',
    'motion_mode_forward_fraction_poststim',
    'motion_mode_backward_fraction_poststim',
                    ]
# Select timepoint of interest and drug of interest
TIME_POINTS = '4'
DRUG_TYPE = 'levamisole'
# Define a different save path for final figures    
PAPER_FIG = True
# Strains/drug concentrations to ignore during analysis
IGNORE = [
    'N2_solvent_control_4h',
    'hlb-1_solvent_control_4h',
    '500uM_levamisole_hlb-1_4h',
    '50uM_levamisole_hlb-1_4h',
    '5uM_levamisole_hlb-1_4h',
    '500uM_levamisole_N2_4h',
    '50uM_levamisole_N2_4h',
    '5uM_levamisole_N2_4h',
    'N2_4h',
    'hlb-1_4h',
    'N2_null',
    'hlb-1_null',
      ]
# Set control strain and strain of interest
CONTROL_STRAIN = 'N2'
STRAIN = 'hlb-1'
#%%
if __name__ == '__main__':
    #set style for all figures
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
    saveto = SAVE_PATH / STRAIN / DRUG_TYPE
    
    # Only select plates and information of the drug we are interested in
    if DRUG_TYPE == 'aldicarb':
        plates_to_drop = [
                'carbachol_high_01',
                'carbachol_high_02',
                'carbachol_low_01',
                'carbachol_low_02',
                'levamisole_high_01',
                'levamisole_high_02',
                'levamisole_low_01',
                'levamisole_low_02',
                'pilocarpine_high_01',
                'pilocarpine_high_02',
                'pilocarpine_low_01',
                'pilocarpine_low_02'
                ]
        
    elif DRUG_TYPE == 'carbachol':
        plates_to_drop = [
                'levamisole_high_01',
                'levamisole_high_02',
                'levamisole_low_01',
                'levamisole_low_02',
                'aldicarb_high_01',
                'aldicarb_high_02',
                'aldicarb_low_01',
                'aldicarb_low_02',
                'pilocarpine_high_01',
                'pilocarpine_high_02',
                'pilocarpine_low_01',
                'pilocarpine_low_02'
                ]
    
    elif DRUG_TYPE == 'levamisole':
        plates_to_drop = [
                'carbachol_high_01',
                'carbachol_high_02',
                'carbachol_low_01',
                'carbachol_low_02',
                'aldicarb_high_01',
                'aldicarb_high_02',
                'aldicarb_low_01',
                'aldicarb_low_02',
                'pilocarpine_high_01',
                'pilocarpine_high_02',
                'pilocarpine_low_01',
                'pilocarpine_low_02'
                ]
        
    elif DRUG_TYPE == 'pilocarpine':
        plates_to_drop = [
                'levamisole_high_01',
                'levamisole_high_02',
                'levamisole_low_01',
                'levamisole_low_02',
                'aldicarb_high_01',
                'aldicarb_high_02',
                'aldicarb_low_01',
                'aldicarb_low_02',
                'carbachol_high_01',
                'carbachol_high_02',
                'carbachol_low_01',
                'carbachol_low_02',
                ]        
    
    elif DRUG_TYPE == 'all':
        plates_to_drop = []
    
    # Read in metadata using tierpsy tools function
    feat, meta = read_hydra_metadata(
        FEAT_FILE,
        FNAME_FILE,
        METADATA_FILE)
    # Using tierpsy tools function to align by bluelight 
    feat, meta = align_bluelight_conditions(feat, meta, how='inner')
    # Replacing incorrect name in metadata file
    meta = meta.replace('pramipexole', 'pilocarpine')
    # Filter out bad worms annotated in metadata file, also filtering out
    # strains we're not interested in for this study
    strain_drop = ['BAD',
                   'gpb-2']
    meta = meta[~meta['worm_gene'].isin(strain_drop)]
    feat = feat.loc[meta.index]
    # Drop plates to only look at one data set at a time
    meta = meta[~meta['imaging_plate_id'].isin(plates_to_drop)]
    feat = feat.loc[meta.index]
    # Filter out nan's within specified columns and print .csv of these    
    nan_worms = meta[meta.worm_gene.isna()][['featuresN_filename',
                                             'well_name',
                                             'imaging_plate_id',
                                             'instrument_name',
                                             'date_yyyymmdd']]
    nan_worms.to_csv(
        METADATA_FILE.parent / 'nan_worms.csv', index=False)
    print('{} nan worms'.format(nan_worms.shape[0]))
    feat = feat.drop(index=nan_worms.index)
    meta = meta.drop(index=nan_worms.index)
    # Print out number of features processed
    feat_filters = [line for line in open(FNAME_FILE) 
                     if line.startswith("#")]
    print ('Features summaries were processed \n {}'.format(
           feat_filters))
    
    # Make summary .txt file of feats
    with open(ROOT_DIR / 'feat_filters_applied.txt', 'w+') as fid:
        fid.writelines(feat_filters)
    # Set date to date-time format for plotting purposes
    meta['date_yyyymmdd'] = pd.to_datetime(
    meta['date_yyyymmdd'], format='%Y%m%d').dt.date
    # Merge strain, drug, conc and exposure info into one column and rename as
    # appropiate for plots and analysis
    meta['analysis'] = meta['imaging_plate_drug_concentration_uM'].astype(str) + 'uM' + '_' + meta['drug_type'] + '_' + meta['worm_gene'] + '_' + meta['exposure_time']
    meta.analysis.replace({
                           '0uM_no_compound_N2_1h': 'N2_null',
                           '0uM_no_compound_N2_4h': 'N2',
                           '0uM_solvent_control_N2_1h': 'N2_null',
                           '0uM_solvent_control_N2_4h': 'N2_null',
                           '0uM_no_compound_hlb-1_1h': 'hlb-1_null',
                           '0uM_no_compound_hlb-1_4h': 'hlb-1',
                            '0uM_solvent_control_hlb-1_1h': 'hlb-1_null',
                            '0uM_solvent_control_hlb-1_4h': 'hlb-1_null',
                           }, inplace=True)      
    meta['worm_gene'] = meta['analysis']
    # Select only timepoints of interest
    if TIME_POINTS == '1':
        mask = meta['worm_gene'].str.endswith('4h')
        meta = meta[~mask]
        feat = feat[~mask]
    if TIME_POINTS == '4':
        mask = meta['worm_gene'].str.endswith('1h')
        meta = meta[~mask]
        feat = feat[~mask]
    # Drop strains present in 'ignore' list from analysis
    genes = [g for g in meta.worm_gene.unique() if g != CONTROL_STRAIN]
    genes = list(set(genes) - set(IGNORE))
    genes.sort()
    STRAINS = genes
    # Setting final save path
    if PAPER_FIG == True:
        saveto = saveto / 'paper_figs'
    #%%
    # Select strains, stim and filter features using tierpsy functions
    feat_df, meta_df, idx, gene_list = select_strains(STRAINS,
                                                    CONTROL_STRAIN,
                                                    feat_df=feat,
                                                    meta_df=meta)
    feat_df, meta_df, featsets = filter_features(feat_df,
                                                 meta_df)
    strain_lut, stim_lut, feat_lut = make_colormaps(gene_list,
                                                    featlist=featsets['all'],
                                                    idx=idx,
                                                    candidate_gene=STRAINS
                                                    )
    # Setting colour map and order of plot
    strain_lut = {'N2': (0.6, 0.6, 0.6),
                  'hlb-1':(0.4498107260398470086, 0.4779114713477097820, 0.7644583735059833351),
                  '1uM_levamisole_hlb-1_4h':(5.625403702604810929e-01, 4.930128535855061722e-01, 8.110720492272496251e-01),
                  '1uM_levamisole_N2_4h':(7.332316564779161050e-01, 5.430012014091432082e-01, 8.331814925220115686e-01),
                  '10uM_levamisole_hlb-1_4h':(8.578575836301945978e-01, 6.167295569758052265e-01, 8.124207621880670249e-01),
                  '10uM_levamisole_N2_4h':(9.348790605721730707e-01, 7.141566770453827706e-01, 7.755791415598958238e-01),
                  '100uM_levamisole_hlb-1_4h':(9.548328907488411454e-01, 8.123776332007711654e-01, 7.581010612365786105e-01),
                  '100uM_levamisole_N2_4h':(9.429422622250018815e-01, 8.942894451415310808e-01, 7.843740204486435719e-01)
                  }
    # Make and plot colorbars to map colors to strains
    plot_colormap(strain_lut)
    plt.savefig(saveto / 'strain_cmap.png')
    plot_cmap_text(strain_lut)
    plt.savefig(saveto / 'strain_cmap_text.png')

    plot_colormap(stim_lut, orientation='horizontal')
    plt.savefig(saveto / 'stim_cmap.png')
    plot_cmap_text(stim_lut)
    plt.savefig(saveto / 'stim_cmap_text.png')

    plt.close('all')

    #%% Plotting raw data and averaged day data points and box plots
    for t in TIME_POINTS:   
            for f in  FEATURES:
                feature_box_plots(f,
                                  feat_df,
                                  meta_df,
                                  strain_lut,
                                  show_raw_data='date',
                                  add_stats=False)
                
                plt.savefig(saveto / 'boxplots' / f'{t}h_{f}_date_boxplot.png',
                            bbox_inches='tight',
                            dpi=200)
            plt.close('all')

            
            for f in  FEATURES:
                average_feature_box_plots(f,
                                  feat_df,
                                  meta_df,
                                  strain_lut,
                                  show_raw_data='date',
                                  add_stats=False)
                
                plt.savefig(saveto / 'average_boxplots' / f'{t}h_{f}_date_boxplot.png',
                            bbox_inches='tight',
                            dpi=200)
            plt.close('all')