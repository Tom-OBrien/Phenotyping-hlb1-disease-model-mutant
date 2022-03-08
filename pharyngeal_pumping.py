#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 09:53:33 2022

Script for performing stats and making swarm plot of pharangeal pumping assay

@author: tobrien
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

from tierpsytools.analysis.statistical_tests import (univariate_tests,
                                                     _multitest_correct)
# Set paths for data
DATA = Path('//Users/tobrien/Documents/Imperial : MRC/AE_Disease_Model_Strains/Analysis/pharangeal_pumping_assay/hlb-1/hlb-1_pharangeal_pumping_assay_results_and_metadata.csv')
CUSTOM_STYLE = '/Users/tobrien/analysis_repo/DiseaseModelling/hydra_screen/phenotype_summary/gene_cards.mplstyle'
plt.style.use(CUSTOM_STYLE)

# Set control and strain for analysis
CONTROL_STRAIN = 'N2'
STRAIN = 'hlb-1'

# Hardcoded colour maps to keep figures in line with existing paper
strain_lut = {'N2':(0.6, 0.6, 0.6),
              # 'cat-1':(0.89259172, 0.65021834, 0.79918302),
              'hlb-1':(0.44981072, 0.47791147, 0.76445837),
              }
#%% 
if __name__ == '__main__':
    saveto = Path('/Users/tobrien/Documents/Imperial : MRC/AE_Disease_Model_Strains/Analysis/pharangeal_pumping_assay')
    saveto = saveto / STRAIN
    df = pd.read_csv(DATA)

    # Convert dates into pandas date-time format (for plotting nicely)
    df['date_yyyymmdd'] = pd.to_datetime(df['date_yyyymmdd'], 
                                         format='%Y%m%d').dt.date
    
    # Select unique genes for analysis from the metadata
    gene_list = [g for g in df.worm_gene.unique()]
    genes = [g for g in df.worm_gene.unique() if g != CONTROL_STRAIN]

#%% Calculate mean and standard deviation of pumping rates
    strain_df = df.worm_gene==CONTROL_STRAIN
    strain_df = df[~strain_df]
    N2_df = df.worm_gene==STRAIN
    N2_df = df[~N2_df]
    
    N2_mean = N2_df['pumps_per_min'].mean()
    N2_std = N2_df['pumps_per_min'].std()
    strain_mean = strain_df['pumps_per_min'].mean()
    strain_std = strain_df['pumps_per_min'].std()    
    foo = pd.DataFrame([[N2_mean, N2_std, strain_mean, strain_std]],
                 columns=['N2_mean', 'N2_std', '{}_mean'.format(STRAIN), '{}_std'.format(STRAIN)])
    foo.to_csv(saveto / 'mean_ppm_of_{}_compared_to_N2.csv'.format(STRAIN), index=False)

#%% Use Tierpsy tools permutation tets to calculate stats between N2 and strain
    _, unc_pvals, unc_reject = univariate_tests(
                            X=df['pumps_per_min'], 
                            y=df['worm_gene'],
                            control=CONTROL_STRAIN,
                            test='t-test',
                            comparison_type='binary_each_group',
                            multitest_correction=None,
                            n_permutation_test=10000,
                            perm_blocks=df['date_yyyymmdd'],
                            )
    reject, pvals = _multitest_correct(
                            unc_pvals, 'fdr_by', 0.05)
    unc_pvals = unc_pvals.T
    pvals = pvals.T
    reject = reject.T   
    
    unc_pvals.to_csv(saveto/'uncorrected_pvals.csv', index=False)
    pvals.to_csv(saveto/'fdrby_pvals.csv')

    # Saving pvalues as .csv file
    bhP_values = pvals.copy(deep=True)
    bhP_values.loc[:, ~reject.iloc[0, :]] = np.nan
    bhP_values['worm_gene'] = STRAIN
    bhP_values.index = ['p<0.05']
    assert reject.sum().sum() == bhP_values.notna().sum().sum()-1
    bhP_values.to_csv(saveto / '{}_pumping_rate_stats.csv'.format(STRAIN),
                                      index=False)

#%% Plot combined boxp and swarm plot of pumping rates
    label_format = '{0:.3g}'
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')
    plt.tight_layout()

    plt.figure(figsize=(5,10))
    ax = sns.boxplot(
                y=df['pumps_per_min'],
                x=df['worm_gene'],
                data=df,
                order=strain_lut.keys(),
                palette=strain_lut.values(),
                showfliers=False)
    plt.tight_layout()

    sns.swarmplot(
                  y=df['pumps_per_min'],
                  x=df['worm_gene'],
                  data=df,
                order=strain_lut.keys(),
                hue=df['date_yyyymmdd'],
                palette='Greys',
                alpha=0.6)
    ax.set_ylabel(fontsize=22, ylabel='Pharyngeal pumps per min')
    ax.set_yticklabels(labels=[label_format.format(x) for x in ax.get_yticks()])#, fontsize=16) #labels = ax.get_yticks(),
    ax.set_xlabel('')
    ax.set_xticklabels(labels = strain_lut.keys())
    plt.tight_layout()
    plt.savefig(saveto / '{}_pharyngeal_pumping_rate.png'.format(STRAIN), dpi=200)
    
    