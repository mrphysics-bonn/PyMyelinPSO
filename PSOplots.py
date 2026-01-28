#! /usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Exemplary methods for visualization of PSO results from single/joint inversion
of MRI invivo data (T1, T2, T2S, CT2S).

Author: Martin Kobe
Contact: martin.kobe@ufz.de; martin.kobe@email.de
Status: January 2026
Project affiliation: JIMM / JIMM2 (DZNE Bonn, UFZ Leipzig)
"""

import os, matplotlib, time, copy
import matplotlib.pyplot       as plt
import numpy                   as np
import imageio.v2              as imageio
from   matplotlib.lines        import Line2D


def comp_atlas_invivo(inv_data: np.array, sys_param: object, corridor=float,
                      save_path=None, save_format=None, save_dpi=300, save=False):
    
    """Comparison of atlas-derived true MWF with atlas-based inverted MWF."""

    # complex signal needs to be labeled as T2S, as PSO returns it like this
    if sys_param.inv_SI:
        decay     = sys_param.decay_types[0]
        dec_label = 'CT2S' if sys_param.inv_CT2S and sys_param.inv_T2S else decay
    if sys_param.inv_JI:
        decay      = sys_param.decay_types[0]
        sig1, sig2 = sys_param.decay_types
        dec_label  = 'JI'
   
    # figure and color map specification
    fig, ax     = plt.subplots(3,4, figsize=(10, 9), gridspec_kw={'hspace': 0.05, 'wspace': 0.05})    
    cmap        = matplotlib.colormaps.get_cmap('viridis')
    cmap.set_bad(color='black')
    
    # figure axis labels
    labels      = ['Run 1', 'Run 2', f'Best MWF ({sys_param.n_pso_cycles} runs)']
    
    # mwf atlas data for comparison
    mask        = sys_param.masks[decay][..., 0]
    MWF_atlas   = sys_param.obs_data['MWF_ATLAS'][...,90]*mask
    MWF_copy_a  = copy.deepcopy(MWF_atlas)
    MWF_copy_b  = copy.deepcopy(MWF_atlas)
    
    # mask handling/adjustment
    mask_atlas  = MWF_atlas <= 0.025
    MWF_copy_a[~mask_atlas] = np.nan
    MWF_copy_b[mask_atlas]  = np.nan

    # index of mwf and fit ind the results array as returned from PSOmain
    # n_comp=2 --> 0:m1, 1:m1_sig, 2:m2, 3:m2_sig, 4:int2, 5:mwf, (6:mw_f, 7:fw_f, 8:phi), -1:fit
    # n_comp=3 --> 0:m1, 1:m1_sig, 2:m2, 3:m2_sig, 4:m3, 5:m3_sig, 6:int2, 7:int3, (8:mw_f, 9:ew_f, 10:aw_f, 11:phi), -1:fit
    ind_mwf     = 5 if sys_param.n_comp == 2 else 8
    ind_fit     = -1

    # get inversion results
    if sys_param.inv_SI:
        mwf_results_array = inv_data[ind_mwf]
        fit_results_array = inv_data[ind_fit]
    if sys_param.inv_JI:
        mwf_results_array = np.mean(np.stack([inv_data[sig1][ind_mwf], 
                                              inv_data[sig2][ind_mwf]], axis=0), axis=0)
        fit_results_array = inv_data[sig1][ind_fit] 
    
    # plot
    for i,j in enumerate([0,1,-1]):
        PSO_results             = np.nan_to_num(mwf_results_array[:, :, j], nan=0.0)
        diff_map                = MWF_atlas - PSO_results
        PSO_results[mask == 0]  = np.nan
    
        # 1st column: MWF Atlas map
        im0 = ax[i,0].imshow(np.rot90(MWF_copy_b),cmap=cmap,vmin=0,vmax=0.35)    
    
        # 2nd column: MWF map from PSO signal
        im1 = ax[i,1].imshow(np.rot90(PSO_results),cmap=cmap,vmin=0,vmax=0.35)
    
        # 3rd column: Differences map MWF Atlas - MWF PSO
        im2 = ax[i,2].imshow(np.rot90(diff_map), cmap='bwr', vmin=-0.07, vmax=0.07)
        ax[i,2].add_patch(plt.Rectangle((0, 0), PSO_results.shape[0], PSO_results.shape[1],
                                        edgecolor='black', facecolor='none', lw=0.001))
        
        # 4th column: True versus estimated MWF values, scattered 
        ax[i,3].scatter(MWF_atlas, MWF_atlas, s=0.5, c='r')
        ax[i,3].scatter(MWF_copy_a, MWF_copy_a, s=0.6, c='w')
        ax[i,3].scatter(MWF_atlas, PSO_results, s=0.5, c='g')
        ax[i,3].text(0.05, 0.9, rf'diff$_{{max}}$={np.max(np.abs(diff_map)):.3f}',
                    transform=ax[i, 3].transAxes, fontsize=8)
    
        # gray scattered: uncertainty corridor of 3 percent for best solution
        if j == -1:
            corridor_range = 1 + corridor/100
            MWF_results = np.nan_to_num(mwf_results_array, nan=0.0)
            FIT_results = np.nan_to_num(fit_results_array, nan=0.0)
            min_val = np.min(FIT_results, axis=2)
    
            min_per_pixel = np.min(FIT_results, axis=2, keepdims=True)
            _mask = FIT_results <= min_per_pixel * corridor_range
    
            corridor_fits = np.where(_mask, FIT_results, np.nan)
            corridor_pso  = np.where(_mask, MWF_results, np.nan)
    
            for k in range(corridor_pso.shape[-1]):
                if k == corridor_pso.shape[-1]-1:
                    ax[i,3].scatter(MWF_atlas, MWF_atlas, s=0.5, c='r')
                    ax[i,3].scatter(MWF_copy_a, MWF_copy_a, s=0.6, c='w')
                
                color = 'g' if k == corridor_pso.shape[-1]-1 else 'lightgray'
                ax[2,3].scatter(MWF_copy_b[~np.isnan(corridor_pso[...,k])], corridor_pso[...,k][~np.isnan(corridor_pso[...,k])], s=0.5, c=color)
            
        # Layout: Axis and titles
        ax[i,0].set_ylabel(labels[i], fontsize=10, labelpad=10)
        
        for p in range(3):
            ax[i,p].set_xticks([])
            ax[i,p].set_yticks([])
    
        ax[i,3].tick_params(axis='x', direction='in', top=False, bottom=True, labelbottom=False, length=3)
        ax[i,3].tick_params(axis='y', direction='in', left=False, right=True, labelleft=False, labelright=True, length=3)
    
        ax[i,3].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax[i,3].set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax[i,3].set_yticklabels(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5'], fontsize=8)
    
        if i == 2:
            ax[i,3].tick_params(axis='x', direction='in', top=False, bottom=True, labelbottom=True, length=3)
            ax[i,3].set_xticklabels(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5'], fontsize=8)  
        
        if i == 0:
            ax[i,0].set_title(r'MWF from Atlas', fontsize=10)
            ax[i,1].set_title(f'MWF$_{{{dec_label}}}$ from PSO', fontsize=10)
            ax[i,2].set_title(f'MWF$_{{{dec_label}}}$ - MWF$_{{true}}$',fontsize=10)
            ax[i,3].set_title(f'MWF$_{{{dec_label}}}$ vs MWF$_{{true}}$',fontsize=10)

    # colorbars
    cbar1 = fig.colorbar(im1, ax=ax[:, :2], orientation='horizontal', shrink=0.8, pad=0.02, aspect=45)
    cbar1.set_label('MWF', fontsize=10)
    
    cbar2 = fig.colorbar(im2, ax=ax[:, 2], orientation='horizontal', shrink=0.8, pad=0.02, aspect=20)
    cbar2.set_label('Difference', fontsize=10)
    
    # adjust Column 4 to the size of the other columns
    for i in range(3):
        box1 = ax[i, 0].get_position()
        box3 = ax[i, 3].get_position()
        ax[i, 3].set_position([box3.x0, box1.y0, box3.width, box1.height])
    
    # shared horizontal legend below column 3
    handles1 = [Line2D([], [], marker='o', markersize=3, color='r', linestyle='', label='y=x'),
                Line2D([], [], marker='o', markersize=3, color='g', linestyle='', label=f'MWF$_{{{dec_label}}}$')]
    
    handles2 = [Line2D([], [], marker='o', markersize=3, color='lightgray', linestyle='', label=r'Uncertainty distribution')]
    
    fig.legend(handles=handles1, loc='lower center', bbox_to_anchor=(0.81, 0.185), 
               handletextpad=0.2, columnspacing=0.2, ncol=2, fontsize=10, frameon=False)
    fig.legend(handles=handles2, loc='lower center', bbox_to_anchor=(0.80, 0.155), 
               handletextpad=0.2, columnspacing=0.2, ncol=2, fontsize=10, frameon=False)
    
    fig.set_facecolor('w')
    plt.show() 
    
    # make the destination directory and save figure
    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)      
        fig.savefig(save_path, dpi=save_dpi, format=save_format.lower(), bbox_inches='tight')

def comp_preanalysis_states(inv_data: np.array, param: list(), inv_type='single',
                            save_path=None, save_format=None, save_dpi=300, save=False):
    
    """Comparison of MWF from PSO inversion across three pre-analysis states of three datasets."""
        
    # figure and color map specification
    fig, ax = plt.subplots(3, 3, figsize=(7, 6), gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
    cmap    = matplotlib.colormaps.get_cmap('viridis')
    cmap.set_bad(color='black')

    # plot
    for i, ds in enumerate(list(inv_data.keys())):
        preproc_status = list(inv_data[list(inv_data.keys())[0]].keys())
        titles         = [k.capitalize() for k in preproc_status]
        labels         = ['Subject 1', 'Subject 2', 'Subject 3']
        label_cbar     = 'MWF' if param[1] == 'mwf' else 'Misfit'
        vmin, vmax     = [0,0.35] if param[1] == 'mwf' else [0,0.01]
        
        for j, pp in enumerate(preproc_status):

            if inv_type == 'single':
                im = ax[i, j].imshow(np.rot90(inv_data[ds][pp][param[0], :, :, -1]),
                                     cmap=cmap, vmin=vmin, vmax=vmax)
            
            if inv_type == 'joint':
                
                sig1, sig2 = list(inv_data[ds][pp].keys())
                
                if param[1] == 'mwf':
                    results_array = np.mean(np.stack([inv_data[ds][pp][sig1][param[0]], 
                                                      inv_data[ds][pp][sig2][param[0]]], axis=0), axis=0)
                elif param[1] == 'fit':
                    results_array = inv_data[ds][pp][sig1][param[0]]
                    vmin, vmax = 0.0, 0.05
                
                else:
                    import sys
                    sys.exit('Only yet implemented for plotting MWF and Misfit.')
                
                im = ax[i, j].imshow(np.rot90(results_array[:, :, -1]),
                                     cmap=cmap, vmin=vmin, vmax=vmax)
                
            if i == 0:
                ax[i, j].set_title(titles[j], fontsize=10, pad=4)
            if j == 0:
                ax[i, j].set_ylabel(labels[i], fontsize=10, labelpad=10)
            
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    # colorbars
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.025, pad=0.02)
    cbar.set_label(label_cbar, rotation=90, labelpad=5, fontsize=10)
   
    fig.set_facecolor('w')
    plt.show()
    
    # make the destination directory and save figure
    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)      
        fig.savefig(save_path, dpi=save_dpi, format=save_format.lower(), bbox_inches='tight')
    
def pareto_pixel(inv_data: np.array, sys_param: object, num_val: int, corridor=[False, float],
                 save_path=None, save_format=None, save_dpi=300, save=False):
    
    """
    Plot of the best `num_val` Pareto-optimal PSO solutions for a single pixel,
    optionally highlighting a percentage-based corridor of top solutions.
    """
    
    # get data for plot
    decay    = sys_param.decay_types[0]
    ind_mwf  = 5 if sys_param.n_comp == 2 else 8
    MWF      = np.array([r[f"mod{decay}"][ind_mwf] for r in inv_data])
    fits     = np.array([r["fit"]                  for r in inv_data])
    
    # mask for best n values
    idx_sorted = np.argsort(fits)
    idx_best   = idx_sorted[:num_val]
    idx_worst  = idx_sorted[num_val:]
    
    # best fit MWF
    MWF_of_best_fit = MWF[idx_sorted[np.argmin(fits[idx_sorted])]]
    
    if not corridor[0]:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharey=False)
        
        ax.plot(fits[idx_best]*1e3, MWF[idx_best], markersize=2, linestyle='none', marker='o',
                markeredgewidth=0.5, color='r', markerfacecolor='lightpink',
                label='10e3 best results')
        ax.plot(np.min(fits[idx_best])*1e3, MWF_of_best_fit,
                markersize=5, linestyle='none', marker='o', markeredgewidth=0.5,
                color='b', label=f'Best MWF: {np.round(MWF_of_best_fit,3)}\nBest FIT: {np.round(np.min(fits),5)}')
        
        # legend specifications
        leg = ax.legend(loc='upper right', fontsize=14)
        for handle in leg.get_lines():
            handle.set_markersize(7)
        
        # axis specification
        ax.set_ylabel('Global best MWF []', fontsize=16)    
        ax.set_xlabel(r'Global best Fit [$\times 10^{-3}$]', fontsize=16)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
        ax.tick_params(axis='both', labelsize=13)
        
        plt.tight_layout()    
        fig.set_facecolor('w')
        plt.show()
        
        # make the destination directory and save figure
        if save:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)      
            fig.savefig(save_path, dpi=save_dpi, format=save_format.lower(), bbox_inches='tight')
    
    if corridor[0]:
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    
        # subplot 1 showing pareto curve for all values
        ax1.plot(fits[idx_worst]*1e3, MWF[idx_worst], markersize=2, linestyle='none', marker='o',
                 markeredgewidth=0.5, color='gray', markerfacecolor='lightgray',
                 label='20e3 PSO runs')    
        
        ax1.plot(fits[idx_best]*1e3, MWF[idx_best], markersize=2, linestyle='none', marker='o',
                 markeredgewidth=0.5, color='r', markerfacecolor='lightpink',
                 label='10e3 best results')      
    
        # corridor for best fit +3% of best fit
        corridor_range = 1 + corridor[1]/100
        low, high = np.min(fits)*1e3, np.min(fits)*1e3*corridor_range
        ax1.axvspan(low, high, color="lightblue", alpha=0.5)
        ax1.plot([], [], color="lightblue", linewidth=8, alpha=0.5,
                 label='Uncertainty spread')  # label='corridor: bestFit +3% * BF'
    
        # legend specifications
        leg = ax1.legend(loc='upper right', fontsize=14)
        for handle in leg.get_lines():
            handle.set_markersize(7)
    
        # axis specifications
        ax1.set_xlabel(r'Global best Fit [$\times 10^{-3}$]', fontsize=16)
        ax1.set_ylabel('Global best MWF []', fontsize=16)    
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
        ax1.tick_params(axis='both', labelsize=14)
        
        # subplot 2 showing pareto curve for best n values
        ax2.plot(fits[idx_best]*1e3, MWF[idx_best], markersize=2, linestyle='none', marker='o',
                markeredgewidth=0.5, color='r', markerfacecolor='lightpink',
                label='10e3 best results')
        
        ax2.plot(np.min(fits[idx_best])*1e3, MWF_of_best_fit,
                markersize=5, linestyle='none', marker='o', markeredgewidth=0.5,
                color='b', label=f'Best MWF: {np.round(MWF_of_best_fit,3)}')
    
        # legend specifications
        leg2 = ax2.legend(loc='upper right', fontsize=14)
        for handle in leg2.get_lines():
            handle.set_markersize(7)
    
        # axis specification
        ax2.set_ylabel('Global best MWF []', fontsize=16)    
        ax2.set_xlabel(r'Global best Fit [$\times 10^{-3}$]', fontsize=16)
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
        ax2.tick_params(axis='both', labelsize=13)
    
        plt.tight_layout()    
        plt.show()
    
        # make the destination directory and save figure
        if save:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)      
            fig.savefig(save_path, dpi=save_dpi, format=save_format.lower(), bbox_inches='tight')

def mwf_slice(inv_data: np.array, sys_param: object, pixel_list=None, scatter=False,
              save_path=None, save_format=None, save_dpi=300, save=False):
    
    """
    Display an MWF map for a single slice, optionally overlaying and labeling
    selected MWF scatter points.
    """

    # figure and color map specification
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = matplotlib.colormaps.get_cmap('viridis')
    cmap.set_bad(color='black')
    
    # index of mwf and fit ind the results array as returned from PSOmain
    # n_comp=2 --> 0:m1, 1:m1_sig, 2:m2, 3:m2_sig, 4:int2, 5:mwf, (6:mw_f, 7:fw_f, 8:phi), -1:fit
    # n_comp=3 --> 0:m1, 1:m1_sig, 2:m2, 3:m2_sig, 4:m3, 5:m3_sig, 6:int2, 7:int3, (8:mw_f, 9:ew_f, 10:aw_f, 11:phi), -1:fit
    ind_mwf     = 5 if sys_param.n_comp == 2 else 8
    
    image     = inv_data[ind_mwf, :, :, -1]
    image_rot = np.rot90(image)
    
    # after rotation: (x, y) → (new_x, new_y) = (y, width - 1 - x)
    height, width = image_rot.shape

    # plot
    im = ax.imshow(image_rot, cmap=cmap, vmin=0, vmax=0.35)
    
    # scatter plot option
    if scatter:
        for x, y in pixel_list:
            ax.scatter(y, width-1-x, color='red', s=10)
            plt.text(y-6, width-1-x-3, 
                     f'MWF: {image_rot[height-1-y,x]:.3f}', 
                     color='red', fontsize=6.5, fontweight='bold', 
                     va='center', ha='left',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    
    ax.axis('off')
    ax.set_title(f'Subject 1: Degibbsed ({sys_param.n_pso_cycles} runs)', fontsize=9, pad=4)
    ax.set_xticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
    ax.set_xticklabels(['0.0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35'], fontsize=1)
    
    # colorbars
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.01, shrink=0.7, aspect=50)
    cbar.set_label('MWF', rotation=0, labelpad=6, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    plt.show()

    # make the destination directory and save figure
    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)      
        fig.savefig(save_path, dpi=save_dpi, format=save_format.lower(), bbox_inches='tight')
    
class PSOgrafics():

    def __init__(self):
        pass   
        
    def plotIterTest(self,
                     MWFdata:  object,
                     PSOclass: object,
                     position: tuple,
                     savepath: str,
                     string:   str):

        sig           = PSOclass.decay_types[0]
        yy,xx,noSlice = position[0], position[1], position[2]
        savepath      = f'{savepath}/{string[0]}/'
        
        fig, ax       = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), tight_layout=True)
        
        ax[0,0].imshow(MWFdata, cmap=matplotlib.colormaps.get_cmap('viridis'), vmin=0, vmax=np.max(MWFdata))
        ax[0,0].scatter(xx, yy, color='red', s=10)
        ax[0,0].set_xticks([]); ax[0,0].set_yticks([])
        ax[0,0].set_title(f'Calculated MWF map for {PSOclass.decay_types[0]} signal on slice {noSlice}')
        [sp.set_visible(False) for sp in ax[0,0].spines.values()]

        if sig == 'T2': 
            timesteps = np.arange(0,24)
        if sig == 'T2S': 
            timesteps = np.arange(0,32)
        
        ax[0,1].plot(timesteps, PSOclass.obsData[sig], 'k', linewidth=2)
        ax[0,1].plot(timesteps, PSOclass.glob_syn_data[sig], markersize=2, linestyle='-', 
                     marker='o', markeredgewidth=0.5, color='red', linewidth=2, markerfacecolor='lightpink')

        ax[0,1].set_title(f'{sig}: observed (black) vs. synthetic (red) signal')
        ax[0,1].set_ylim(np.min(PSOclass.obsData[sig]-0.1), np.max(PSOclass.obsData[sig])+0.1)

        ax[1,0].plot(np.arange(0, len(PSOclass.glob_ind_list[sig]), 1), PSOclass.glob_ind_list[sig], 
                     markersize=2, linestyle='None', marker='o', color='b')
        ax[1,0].set_ylim(0.5,  PSOclass.n_part+0.5)
        ax[1,0].set_xlim(-0.5, PSOclass.n_iter+0.5)
        ax[1,0].set_xticks(np.arange(0, PSOclass.n_iter+1, int(PSOclass.n_iter/5)), 
                           np.arange(0, PSOclass.n_iter+1, int(PSOclass.n_iter/5), dtype=int))
        ax[1,0].set_yticks(np.arange(0,PSOclass.n_part+1,PSOclass.n_part/10))
        ax[1,0].set_title('global best particle')   
        
        ylim_min = np.min(PSOclass.glob_fit_list[sig])-np.min(PSOclass.glob_fit_list[sig])/3
        ylim_max = np.max(PSOclass.glob_fit_list[sig])+np.min(PSOclass.glob_fit_list[sig])/3

        ax[1,1].plot(np.arange(0, len(PSOclass.glob_fit_list[sig]), 1), 
                     PSOclass.glob_fit_list[sig], markersize=2, linestyle='-', marker='o', color='b')
        ax[1,1].set_ylim(ylim_min, ylim_max)
        ax[1,1].set_xlim(-5, PSOclass.n_iter+5)
        ax[1,1].set_xticks(np.arange(0, PSOclass.n_iter+1, int(PSOclass.n_iter/5)),
                           np.arange(0, PSOclass.n_iter+1, int(PSOclass.n_iter/5), dtype=int))       
        # ax[1,1].set_yticks(np.arange(ylim_min, ylim_max, steps))

        ax[1,1].set_title('global best fit')
             
        value = PSOclass.glob_fit[sig]; n = 0
        while abs(value) < 1:
            value *= 10
            n -= 1
        rounded_number = round(value, 3)
        value = f'{rounded_number} * 10e{n}'

        text  = (f'glob best [px: fit]:\n{PSOclass.glob_ind[sig]+1}: {value}\n\n'
                  f'MWFcalc:{np.round(PSOclass.glob_mod[sig][-1], 4)}')
        
        x_lim = (PSOclass.n_iter+1)/10*7
        ax[1,1].text(x_lim, ylim_max/10*9.5, text, va='top')

        os.makedirs(savepath, exist_ok=True)
        plt.savefig(f'{savepath}pix_y-{yy}_x-{xx}_iter{string[1]}.jpg', dpi=300, format='jpg')                
    
        plt.close()

###############################################################################

    def plotSlice(self, 
                  PSOclass:  object,
                  PSOresult: dict,
                  index:     int,
                  saveFig:   tuple,
                  string =   '',
                  **kwargs):
        
        if PSOclass.decay_types[0] == 'T2S': upper = 0.5
        if PSOclass.decay_types[0] == 'T2':  upper = 0.35
        
        param     = kwargs.get('limit', (object,False))[0]
        paramBool = kwargs.get('limit', (object,False))[1]
        performance_test = kwargs.get('performance_test', False)
        lim       = {sig:[] for sig in PSOclass.decay_types}
        
        if paramBool == False:            
            if PSOclass.n_comp == 2:
                for sig in PSOclass.decay_types:                    
                    lim[sig] = [getattr(getattr(PSOclass, sig), "TwoComponentParams").m1,
                                getattr(getattr(PSOclass, sig), "TwoComponentParams").m1_sig,
                                getattr(getattr(PSOclass, sig), "TwoComponentParams").m2,
                                getattr(getattr(PSOclass, sig), "TwoComponentParams").m2_sig,
                                getattr(getattr(PSOclass, sig), "TwoComponentParams").int2,
                                (0,upper), (0,0.005)]
            
            if PSOclass.n_comp == 3:                
                for sig in PSOclass.decay_types:                    
                    lim[sig] = [getattr(getattr(PSOclass, sig), "ThreeComponentParams").m1,
                                getattr(getattr(PSOclass, sig), "ThreeComponentParams").m1_sig,
                                getattr(getattr(PSOclass, sig), "ThreeComponentParams").m2,
                                getattr(getattr(PSOclass, sig), "ThreeComponentParams").m2_sig,
                                getattr(getattr(PSOclass, sig), "ThreeComponentParams").m3,
                                getattr(getattr(PSOclass, sig), "ThreeComponentParams").m3_sig,
                                getattr(getattr(PSOclass, sig), "ThreeComponentParams").int2,
                                getattr(getattr(PSOclass, sig), "ThreeComponentParams").int3,
                                (0,upper), (0,0.005)]
        
        if paramBool == True:
            if PSOclass.n_comp == 2:              
                for sig in PSOclass.decay_types:                    
                    lim[sig] = [param[f'SynData{sig}']['m1'],(0,1),
                                param[f'SynData{sig}']['m2'],param[f'SynData{sig}']['m2_sig'],
                                param[f'SynData{sig}']['integ2'],(0,0.35), (0,0.005)]  
        
            if PSOclass.n_comp == 3:             
                for sig in PSOclass.decay_types:                    
                    lim[sig] = [param[f'SynData{sig}']['m1'],param[f'SynData{sig}']['m1_sig'],
                                param[f'SynData{sig}']['m2'],param[f'SynData{sig}']['m2_sig'],
                                param[f'SynData{sig}']['m3'],param[f'SynData{sig}']['m3_sig'],
                                param[f'SynData{sig}']['integ2'],param[f'SynData{sig}']['integ3'],
                                (0,0.35), (0,0.005)]        

        cmap = matplotlib.colormaps.get_cmap('viridis')
        cmap.set_bad(color='black')
        
        for sig in PSOclass.decay_types:
            
            PSOres  = PSOresult[sig]  
            _lim    = lim[sig]
            ind     = index
            
            if PSOclass.n_comp == 2:
                fig, ax = plt.subplots(2, 4, figsize=(12, 6))
            
            if PSOclass.n_comp == 3:
                fig, ax = plt.subplots(3, 4, figsize=(14, 6))
        
        # GAUSS results - 0: m1   | 1: m1_sig | 2:  m2  | 3:  m2_sig | 4: int2 | 5: MWF | 6: MW_f | 7: FW_f | 8: phi | -1: misfit
        # DIRAC results - 0: m1   | 1: m1_sig | 2:  m2  | 3:  m2_sig | 4: m3   | 5: m3_sig | 
        #                 6: int2 | 7: int3   | -2: MWF | -1: misfit
        
            # im0 = ax[0,0].imshow(PSOres[5,:,:,ind],cmap=cmap,vmin=_lim[5][0],vmax=_lim[5][1])
            # ax[0,0].set_title('MWF')
            # fig.colorbar(im0, ax=ax[0,0])
            
            # if performance_test == False:
            #     ax[0,0].scatter(35, 35, color='red', s=6)
            #     ax[0,0].scatter(55, 45, color='red', s=6)
        
            im1 = ax[0,1].imshow(PSOres[0,:,:,ind],cmap=cmap,vmin=_lim[0][0],vmax=_lim[0][1])
            ax[0,1].set_title('m1')
            fig.colorbar(im1, ax=ax[0,1])
        
            im2 = ax[0,2].imshow(PSOres[2,:,:,ind],cmap=cmap,vmin=_lim[2][0],vmax=_lim[2][1])
            ax[0,2].set_title('m2')
            fig.colorbar(im2, ax=ax[0,2])
            
            im3 = ax[1,0].imshow(PSOres[-1,:,:,ind],cmap=cmap,vmin=_lim[-1][0],vmax=_lim[-1][1])
            ax[1,0].set_title('misfit')
            fig.colorbar(im3, ax=ax[1,0])

            im4 = ax[1,1].imshow(PSOres[1,:,:,ind],cmap=cmap,vmin=_lim[1][0],vmax=_lim[1][1])
            ax[1,1].set_title('m1_sig')
            fig.colorbar(im4, ax=ax[1,1])
            
            im5 = ax[1,2].imshow(PSOres[3,:,:,ind],cmap=cmap,vmin=_lim[3][0],vmax=_lim[3][1])
            ax[1,2].set_title('m2_sig')
            fig.colorbar(im5, ax=ax[1,2])

            if PSOclass.n_comp == 2:
                im0 = ax[0,0].imshow(PSOres[5,:,:,ind],cmap=cmap,vmin=_lim[5][0],vmax=_lim[5][1])
                ax[0,0].set_title('MWF')
                fig.colorbar(im0, ax=ax[0,0])
                
                im6 = ax[1,3].imshow(PSOres[4,:,:,ind],cmap=cmap,vmin=_lim[4][0],vmax=_lim[4][1])
                ax[1,3].set_title('int2')
                fig.colorbar(im6, ax=ax[1,3])
                
            if PSOclass.n_comp == 3:
                im0 = ax[0,0].imshow(PSOres[8,:,:,ind],cmap=cmap,vmin=_lim[8][0],vmax=_lim[8][1])
                ax[0,0].set_title('MWF')
                fig.colorbar(im0, ax=ax[0,0])
                
                im6 = ax[0,3].imshow(PSOres[4,:,:,ind],cmap=cmap,vmin=_lim[4][0],vmax=_lim[4][1])
                ax[0,3].set_title('m3')
                fig.colorbar(im6, ax=ax[0,3])

                im7 = ax[1,3].imshow(PSOres[5,:,:,ind],cmap=cmap,vmin=_lim[5][0],vmax=_lim[5][1])
                ax[1,3].set_title('m3_sig')
                fig.colorbar(im7, ax=ax[1,3])

                im8 = ax[2,2].imshow(PSOres[6,:,:,ind],cmap=cmap,vmin=_lim[6][0],vmax=_lim[6][1])
                ax[2,2].set_title('int2')
                fig.colorbar(im8, ax=ax[2,2])
                
                im9 = ax[2,3].imshow(PSOres[7,:,:,ind],cmap=cmap,vmin=_lim[7][0],vmax=_lim[7][1])
                ax[2,3].set_title('int3')
                fig.colorbar(im9, ax=ax[2,3])
                            
            for ii in range(ax.shape[0]*ax.shape[1]):
                ax.flat[ii].axis('off')
                ax.flat[ii].set_facecolor('b')
                fig.set_facecolor('w')
        
            if ind == -1: ind = 'bfit'

            dist = 'three-component' if PSOclass.n_comp == 3 else 'two-component'
            
            fig.suptitle(f'Calculated {sig} parameter maps using a {dist} relaxation-model. PSO cycle: {str(ind).zfill(2)}')
        
            if saveFig[1]==True:
                if performance_test==True:
                    fig.savefig(f'{saveFig[0]}.png', dpi=300, format='png')
                else:
                    savepath = f'{saveFig[0]}{string}/'
                    os.makedirs(savepath, exist_ok=True)
                    fig.savefig(f'{savepath}{sig}_ID_{str(ind).zfill(2)}.png', dpi=300, format='png')

            plt.close()

###############################################################################

    def plotHist(self,
                 PSOclass:     object,
                 PSOresult:    np.array,
                 PSOresultCUT: np.array,
                 position:     tuple,
                 noBins:       int,
                 saveFig:      tuple,
                 string =      ''):
        
        yy,xx    = position[0], position[1]
        res2     = np.copy(PSOresultCUT)
        
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(14, 10), tight_layout=True)
        cmap = matplotlib.colormaps.get_cmap('viridis')
        cmap.set_bad(color='black')
        
        im0=ax[0,0].imshow(PSOresult[-2,:,:,-1],cmap=cmap,vmin=0,vmax=0.35)
        ax[0,0].scatter(yy,xx, color='red',marker='o')
        ax[0,0].set_title(f'MWF (best fit: {np.round(PSOresult[-2,yy,xx,-1],3)})')
        fig.colorbar(im0, ax=ax[0,0])

        hist, bins = np.histogram(PSOresult[-2, yy, xx, :-1], bins=noBins, density=False)
        ax[0,1].hist(PSOresult[-2, yy, xx, :-1], bins=noBins, edgecolor='black')
        ax[0,1].vlines(np.max(res2[-2]),ymin=0,ymax=np.max(hist), linestyle='dashed',
                       linewidth=1, color='r', label=f'MWF cut: {np.round(np.max(res2[-2]),3)}')
        ax[0,1].legend(loc='upper right')
        
        ax[0,2].hist(res2[-2,:-1], bins=noBins, edgecolor='black',label='noVal')
        # ax[0,2].legend(loc='upper right')
                
        value = PSOresult[-1,yy,xx,-1]; n = 0
        while abs(value) < 1:
            value *= 10
            n -= 1
        rounded_number = round(value, 3)
        value = f'{rounded_number} * 10e{n}'
                
        im1=ax[1,0].imshow(PSOresult[-1,:,:,-1],cmap=cmap,vmin=0,vmax=0.005)
        ax[1,0].scatter(yy,xx, color='red',marker='o')
        ax[1,0].set_title(f'misfit (best fit: {value})')
        fig.colorbar(im1, ax=ax[1,0])

        hist, bins = np.histogram(PSOresult[-1, yy, xx, :-1], bins=noBins, density=False)
        
        value = np.max(res2[-1]); n = 0
        while abs(value) < 1:
            value *= 10
            n -= 1
        rounded_number = round(value, 3)
        value = f'{rounded_number} * 10e{n}'
        
        ax[1,1].hist(PSOresult[-1, yy, xx, :-1], bins=noBins, edgecolor='black',label=f'no.Values: {res2.shape[-1]}|{PSOresult.shape[-1]-1}')
        ax[1,1].vlines(bins[1],ymin=0,ymax=np.max(hist), linestyle='dashed',
                        linewidth=1, color='r', label=f'misfit cut: {value}')
        ax[1,1].legend(loc='upper right')
        
        ax[1,2].hist(res2[-1,:-1], bins=noBins, edgecolor='black',label=f'no.Values: {res2.shape[-1]}|{PSOresult.shape[-1]-1}')
        # ax[1,2].legend(loc='upper right')
        
        if PSOclass.n_comp == 2:
            param_class      = "TwoComponentParams"
        else:
            param_class      = "ThreeComponentParams"
            
        vmin=getattr(getattr(PSOclass, PSOclass.decay_types[0]), param_class).m1[0]
        vmax=getattr(getattr(PSOclass, PSOclass.decay_types[0]), param_class).m1[1]
        
        im2=ax[2,0].imshow(PSOresult[0,:,:,-1],cmap=cmap,vmin=vmin,vmax=vmax)
        ax[2,0].scatter(yy,xx, color='red',marker='o')
        ax[2,0].set_title(f'm1 (best fit: {np.round(PSOresult[0,yy,xx,-1],1)})')
        fig.colorbar(im2, ax=ax[2,0])
        
        hist, bins = np.histogram(PSOresult[0, yy, xx, :-1], bins=noBins, density=False)
        ax[2,1].hist(PSOresult[0, yy, xx, :-1], bins=noBins, edgecolor='black')
        ax[2,1].vlines(np.max(res2[0]),ymin=0,ymax=np.max(hist), linestyle='dashed', 
                       linewidth=1, color='r', label=f'm1 cut: {np.round(np.max(res2[0]),3)}') 
        ax[2,1].legend(loc='upper right')
        ax[2,1].set_xlim(vmin, vmax)
        
        ax[2,2].hist(res2[0,:-1], bins=noBins, edgecolor='black',label='ID bfit[0]')
        # ax[2,2].legend(loc='upper right')
        
        ax.flat[0].axis('off')
        ax.flat[0].set_facecolor('b')
        ax.flat[3].axis('off')
        ax.flat[3].set_facecolor('b')
        ax.flat[6].axis('off')
        ax.flat[6].set_facecolor('b')
        fig.set_facecolor('w')

        fig.suptitle(f'Pix.[{yy},{xx}]: Calculated parameters for signal {PSOclass.decay_types[0]} | {PSOclass.n_comp} components',fontsize=14)
        
        if saveFig[1]==True:
            savepath = f'{saveFig[0]}{string}/'
            os.makedirs(savepath, exist_ok=True) 
            plt.savefig(f'{savepath}pix_y-{yy}_x-{xx}.png', dpi=300, format='png')
            
        plt.close()


###############################################################################

    def plotMWFvsFIT(self,
                     PSOclass:     object,
                     PSOresultPIX: np.array,
                     PSOresultSLI: np.array,
                     position:     tuple,
                     saveFig:      tuple,
                     string =      '',
                     **kwargs):
        
        '''
        Plot function for n PSO inversion results of invivo MRI data.\n
        2-window grafic with slice overview (left) and MWF versus misfit (right).
        
        Input parameters:
            PSOclass     - initialized class instance of the PSO workflow\n
            PSOresultPIX - numpy array for PSO results calculated on one pixel\n
                           (shape: np.array([x,PSO.noPSOIterPix]), x=7 (GAUSS), x=10 (DIRAC))\n
            PSOresultSLI - numpy array for PSO results calculated on a complete slice\n
                           (shape: np.array([x,lengthY(slice),lengthX(slice),PSO.noPSOIterSli+1]))\n
            position     - tuple of pixel location (yy,xx,sliceNumber)\n
            saveFig      - tuple of savepath and boolean (path, True/False)\n
            string       - string for new save directory in the working directory if filled\n
        
        kwargs:
            cutPercentile - cut MWF and FIT array for a chosen percentile, boolean\n
            valPercentile - cut percentile parameters, tuple: (low,high)\n
            cutOutliers   - cut MWF and FIT array for a chosen standard deviation, boolean\n
            valOutliers   - cut outlier parameters, tuple: (low,high)
            '''

        # get keyword arguments from the function call
        cutPercentile  = kwargs.get('cutPercentile', False)
        valPercentile  = kwargs.get('valPercentile', None)
        
        cutOutliers    = kwargs.get('cutOutliers', False)
        valOutliers    = kwargs.get('valOutliers', None)

        cmap = matplotlib.colormaps.get_cmap('viridis')
        cmap.set_bad(color='black')
        
        for sig in PSOclass.decay_types:
        
            # pixel position        
            yy,xx        = position[0],position[1]        
            MWFarray     = PSOresultPIX[sig][-2,:-1]
            FITarray     = PSOresultPIX[sig][-1,:-1]

            # plot raw
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6), tight_layout=True)       
            fig.gca().xaxis.set_major_formatter('{:.3f}'.format)
        
            _lim = (0,0.35) if PSOclass.n_comp == 2 else (0,0.25)
            
            im0 = ax[0].imshow(PSOresultSLI[sig][-2,:,:,-1],cmap=cmap,vmin=_lim[0],vmax=_lim[1])
            ax[0].set_title('MWF map with best misfit'); ax[0].scatter(position[1], position[0], color='red')
            ax.flat[0].axis('off'); ax.flat[0].set_facecolor('b') 
            fig.set_facecolor('w'); #fig.colorbar(im0, ax=ax[0],orientation='horizontal')
            
            ax[1].plot(FITarray, MWFarray, markersize=2, linestyle='none', marker='o', 
                       markeredgewidth=0.5, color='r', markerfacecolor='lightpink',
                       label='MWF vs. misfit')

            ax[1].plot(np.min(FITarray), MWFarray[FITarray==np.min(FITarray)],
                       markersize=5, linestyle='none', marker='o', markeredgewidth=0.5, 
                       color='b', label=f'bestMWF: {np.round(MWFarray[FITarray==np.min(FITarray)][0],3)}')
            
            ax[1].legend(loc='upper right')

            ax[1].set_title(f'{sig}(y{yy}x{xx}) | {PSOclass.n_iter}Iter,'
                            f'{PSOclass.n_part}Part,{PSOclass.n_pso_cycles}PSO | {PSOclass.n_comp} components')
        
            ax[1].set_ylabel('global best MWF []'); ax[1].set_xlabel('global best Fit []')
            
            if saveFig[1]==True:
                yy, xx   = str(yy).zfill(2), str(xx).zfill(2)
                savepath = f'{saveFig[0]}{string}/'
                os.makedirs(savepath, exist_ok=True)
                fig.savefig(f'{savepath}y{yy}x{xx}_{sig}.png', dpi=300, format='png')
                
            plt.close()

            # cut outliers and percentile
            if cutPercentile == True:
                    
                # calculate percentiles and plot the curve of global bestMWF vs. global bestFit
                outliers = np.where((FITarray > np.percentile(FITarray, valPercentile[1])))
                
                MWFarray = np.delete(MWFarray, outliers)
                FITarray = np.delete(FITarray, outliers)        
            
            
            if cutOutliers == True:            
                median       = np.median(FITarray)
                stdv         = np.std(FITarray)
        
                MWFarray     = MWFarray[FITarray<median+stdv*valOutliers[1]] # upper              
                FITarray     = FITarray[FITarray<median+stdv*valOutliers[1]] # upper
    
            # plot cut            
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6), tight_layout=True)
            fig.gca().xaxis.set_major_formatter('{:.3f}'.format)

            im0 = ax[0].imshow(PSOresultSLI[sig][-2,:,:,-1],cmap=cmap,vmin=_lim[0],vmax=_lim[1])
            ax[0].set_title('MWF map with best misfit'); ax[0].scatter(position[1], position[0], color='red')
            ax.flat[0].axis('off'); ax.flat[0].set_facecolor('b')
            fig.set_facecolor('w'); #fig.colorbar(im0, ax=ax[0], orientation='horizontal')
            
            ax[1].plot(FITarray, MWFarray, markersize=2, linestyle='none', marker='o', 
                       markeredgewidth=0.5, color='r', markerfacecolor='lightpink',
                       label='MWF vs. misfit')
            
            ax[1].plot(np.min(FITarray), MWFarray[FITarray==np.min(FITarray)],
                       markersize=5, linestyle='none', marker='o', markeredgewidth=0.5, 
                       color='b', label=f'bestMWF: {np.round(MWFarray[FITarray==np.min(FITarray)][0],3)}')
            
            ax[1].legend(loc='upper right')
            
            ax[1].set_title(f'{sig}(y{yy}x{xx}) | {PSOclass.n_iter}Iter,'
                          f'{PSOclass.n_part}Part,{PSOclass.n_pso_cycles}PSO | {PSOclass.n_comp} components')
            
            ax[1].set_ylabel('global best MWF []'); ax[1].set_xlabel('global best Fit []')
            
            if saveFig[1]==True:
                yy, xx   = str(yy).zfill(2), str(xx).zfill(2)
                savepath = f'{saveFig[0]}{string}/'
                os.makedirs(savepath, exist_ok=True)
                fig.savefig(f'{savepath}y{yy}x{xx}_{sig}_cut.png', dpi=300, format='png')
                
            plt.close()
            

###############################################################################

    def log(self, startTime, string='', dim='sek', boolean=False):
        
        '''
        Parameters
        ----------
        startTime : time.time() object

        string : string

        dim : output dimension
              ms  - milli seconds
              mus - micro seconds
              MS  - min:sec (default)
              HMS - hr:min:sec.

        Returns : None
        '''
            
        T_now       = time.time()
        T_elapsed   = T_now - startTime
        
        if dim=='HMS':
            TT = time.strftime('%H:%M:%S', time.gmtime(T_elapsed))
            print(f'{string}: {TT} hrs')
            
        if dim=='MS':
            TT = time.strftime('%M:%S', time.gmtime(T_elapsed))
            print(f'{string}: {TT} min')
        
        if dim=='ms':
            t_ms        = round(T_elapsed*1000, 2)
            print(f'{string}: {t_ms} ms')
        
        if dim=='mus':
            t_mus        = round(T_elapsed*1000 * 1000, 2)
            print(f'{string}: {t_mus} \u03BCs')

###############################################################################

class PSOvideos():    
    
    def __init__(self):
        
        import imageio.v2        as imageio
        
        pass  
    
    def build_gif(self, filelist: list, savepath: str, fps=1, loops=0):        

        images = []
        
        for filename in sorted(filelist):
            images.append(imageio.imread(filename))

        imageio.mimsave(f'{savepath}', images, duration=fps, loop=loops)