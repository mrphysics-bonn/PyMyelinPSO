# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# Author:
#   Martin Kobe (Helmholtz Centre for Environmental Research - UFZ)
#
# This file is part of the PyMRI_PSO software.
# See the LICENSE file in the project root for full license information.

"""
Exemplary methods for visualization of PSO results from single/joint inversion
of MRI invivo data (T2, T2S, CT2S).
"""

import os, matplotlib, copy
import matplotlib.pyplot       as plt
import numpy                   as np
import imageio.v2              as imageio
from   matplotlib.lines        import Line2D

class PSOPlotter():

    def __init__(self):
        pass

    def comp_atlas_invivo(self, inv_data: np.array, sys_param: object, signal:str, corridor=float,
                          save_path=None, save_format=None, save_dpi=300, save=False, verbose=False):
        
        """
        Comparison of atlas-derived true MWF with atlas-based inverted MWF.
        
        Args:
            inv_data: inverted pso model vector array with shape
                [x, y, n_parameters, n_pso_cycles + 1]
            sys_param: instance of PSO_preparation_SI or PSO_preparation_JI class\n
            signal: decay type ['T2', 'T2S']\n
            corridor: percentage range behind the best fit to highlight
        """
    
        # complex signal needs to be labeled as T2S, as PSO returns it like this
        if sys_param.inv_SI:
            decay      = signal
            dec_label  = 'CT2S' if sys_param.inv_CT2S and sys_param.inv_T2S else decay
        if sys_param.inv_JI:
            decay      = signal
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
        mwf_results_array = inv_data[:,:,ind_mwf]
        fit_results_array = inv_data[:,:,ind_fit]

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
        
            # gray scattered: uncertainty corridor of n percent for best solution
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
        
        # make the destination directory and save figure
        if save:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)      
            fig.savefig(save_path, dpi=save_dpi, format=save_format.lower(), bbox_inches='tight')
            if verbose:
                print(f'Saved "{os.path.basename(save_path)}" to {os.path.dirname(save_path)} ...')

        plt.show()
        
    def comp_preanalysis_states(self, inv_data: np.array, param: list(), inv_type='single',
                                save_path=None, save_format=None, save_dpi=300, save=False, verbose=False):
        
        """
        Display the best MWF or best misfit maps across three pre-analysis states
        for three datasets.
        
        Args:
            inv_data: inverted pso model vector array with shape
                [x, y, n_parameters, n_pso_cycles + 1]
            param: (index, name), where index corresponds to the
                    parameter name position in the model vector array
        """

        # troubleshooting
        if param[1] not in ['fit', 'mwf']:
            import sys
            sys.exit('Only yet implemented for plotting MWF and Misfit.')
                        
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
            vmin_mwf, vmax_mwf = [0,0.35] if param[1] == 'mwf' and inv_type == 'single' else [0,0.5]
            vmin_fit, vmax_fit = [0,0.01] if param[1] == 'fit' and inv_type == 'single' else [0,0.1]
            
            vmin, vmax = [vmin_mwf, vmax_mwf] if param[1] == 'mwf' else [vmin_fit, vmax_fit]
            
            for j, pp in enumerate(preproc_status):

                im = ax[i, j].imshow(np.rot90(inv_data[ds][pp][:, :, param[0], -1]),
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
        
        # make the destination directory and save figure
        if save:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)      
            fig.savefig(save_path, dpi=save_dpi, format=save_format.lower(), bbox_inches='tight')
            if verbose: 
                print(f'Saved "{os.path.basename(save_path)}" to {os.path.dirname(save_path)}')
        
        plt.show()
            
    def mwf_map(self, inv_data: np.array, pso_class: object, pixel_list=[[0,0]], 
                scatter=False, save_path=None, save_format=None, save_dpi=300, save=False, verbose=False):
        
        """
        Display best fitting MWF map for a single slice, with optional scatter overlay.
    
        Args:
            inv_data: inverted pso model vector array with shape
                [x, y, n_parameters, n_pso_cycles + 1]
            pso_class: instance of the core ParticleSwarmOptimizer class\n
            pixel_list: pixel coordinates to overlay as scatter points
        """
    
        # figure and color map specification
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = matplotlib.colormaps.get_cmap('viridis')
        cmap.set_bad(color='black')
        
        # index of mwf and fit ind the results array as returned from PSOmain
        # n_comp=2 --> 0:m1, 1:m1_sig, 2:m2, 3:m2_sig, 4:int2, 5:mwf, (6:mw_f, 7:fw_f, 8:phi), -1:fit
        # n_comp=3 --> 0:m1, 1:m1_sig, 2:m2, 3:m2_sig, 4:m3, 5:m3_sig, 6:int2, 7:int3, 8:mwf, (9:mw_f, 10:ew_f, 11:aw_f, 12:phi), -1:fit
        ind_mwf     = 5 if pso_class.n_comp == 2 else 8
        
        image     = inv_data[:, :, ind_mwf, -1]
        image_rot = np.rot90(image)
        
        # after rotation: (x, y) → (new_x, new_y) = (y, width - 1 - x)
        height, width = image_rot.shape
    
        # plot
        #vmin, vmax = [0,0.35] if pso_class.inv_T2 and pso_class.inv_SI else[0, 0.5]
        vmin, vmax = 0, 0.35
        im = ax.imshow(image_rot, cmap=cmap, vmin=vmin, vmax=vmax)
        
        # scatter plot option
        # scatter location before rotation: arr[y,x] → scatter(x,y) #PLOT
        # scatter location after rotation:  scatter(new_x, new_y) = (y, width-1-x) #PLOT
        # array location before rotation: arr[y,x) → row, column
        # array location after rotation: arr[new_y, new_x] = (height-1-x,y) #VALUES
        # NOTE: MWF could be slightly different to the pareto plos due to a different PSO cycle number
        #       --> for the very same values use a collector like best_MWF list
        if scatter:
            for y,x in pixel_list:
                ax.scatter(y, width-1-x, color='red', s=10)
                plt.text(y-6, width-1-x-3, 
                         f'MWF: {image_rot[height-1-x,y]:.3f}',
                         color='red', fontsize=6.5, fontweight='bold', 
                         va='center', ha='left',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        
        ax.axis('off')
        ax.set_title(f'Subject 1: Degibbsed ({pso_class.n_pso_cycles} runs)', fontsize=9, pad=4)
        ax.set_xticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
        ax.set_xticklabels(['0.0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35'], fontsize=1)
        
        # colorbars
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.01, shrink=0.7, aspect=50)
        cbar.set_label('MWF', rotation=0, labelpad=6, fontsize=9)
        cbar.ax.tick_params(labelsize=8)
           
        # make the destination directory and save figure
        if save:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)      
            fig.savefig(save_path, dpi=save_dpi, format=save_format.lower(), bbox_inches='tight')
            if verbose:
                print(f'Saved "{os.path.basename(save_path)}" to {os.path.dirname(save_path)} ...')
        
        plt.show()

    def param_map_single(self):
        '''planned'''
        
    def param_map_multi(self, inv_data: np.array, pso_class: object, signal: str, idx: int,
                        save_path=None, save_format=None, save_dpi=300, save=False, show=False):
        
        """
        Display PSO model parameter maps for one slice (m1/m2/..., MWF, misfit etc.).
        
        Args:
            inv_data: inverted pso model vector array with shape
                [x, y, n_parameters, n_pso_cycles + 1]
            pso_class: instance of the core ParticleSwarmOptimizer class\n
            signal: decay type ['T2', 'T2S']\n
            idx: index of the PSO solution to plot
                use -1 to display the best-fitting parameter maps\n
        """
        
        # color map specification
        cmap = matplotlib.colormaps.get_cmap('viridis')
        cmap.set_bad(color='black')
        
        # vmin and vmax of MWF and misfit are dependent on signal type
        upper_mwf = 0.35
        upper_mwf = 0.5  if signal == 'T2S' or pso_class.inv_JI else upper_mwf 
        upper_fit = 0.05 if pso_class.inv_CT2S else 0.005
        
        # limits for parameter maps        
        if pso_class.n_comp == 2:
            lim = [getattr(getattr(pso_class, signal), "TwoComponentParams").m1,
                   getattr(getattr(pso_class, signal), "TwoComponentParams").m1_sig,
                   getattr(getattr(pso_class, signal), "TwoComponentParams").m2,
                   getattr(getattr(pso_class, signal), "TwoComponentParams").m2_sig,
                   getattr(getattr(pso_class, signal), "TwoComponentParams").int2,
                   (0,upper_mwf), (0,upper_fit)]
        
        elif pso_class.n_comp == 3:                
            lim = [getattr(getattr(pso_class, signal), "ThreeComponentParams").m1,
                   getattr(getattr(pso_class, signal), "ThreeComponentParams").m1_sig,
                   getattr(getattr(pso_class, signal), "ThreeComponentParams").m2,
                   getattr(getattr(pso_class, signal), "ThreeComponentParams").m2_sig,
                   getattr(getattr(pso_class, signal), "ThreeComponentParams").m3,
                   getattr(getattr(pso_class, signal), "ThreeComponentParams").m3_sig,
                   getattr(getattr(pso_class, signal), "ThreeComponentParams").int2,
                   getattr(getattr(pso_class, signal), "ThreeComponentParams").int3,
                   (0,upper_mwf), (0,upper_fit)]

        # index of mwf and fit ind the results array as returned from PSOmain
        # n_comp=2 --> 0:m1, 1:m1_sig, 2:m2, 3:m2_sig, 4:int2, 5:mwf, (6:mw_f, 7:fw_f, 8:phi), -1:fit
        # n_comp=3 --> 0:m1, 1:m1_sig, 2:m2, 3:m2_sig, 4:m3, 5:m3_sig, 6:int2, 7:int3, 8:mwf, (9:mw_f, 10:ew_f, 11:aw_f, 12:phi), -1:fit

        # figure specification
        if pso_class.n_comp == 2: fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        if pso_class.n_comp == 3: fig, ax = plt.subplots(3, 4, figsize=(14, 6))
        
        # MWF map
        ind_mwf = 5 if pso_class.n_comp == 2 else 8  
        data = np.rot90(inv_data[:,:,ind_mwf,idx])
        im0  = ax[0,0].imshow(data,cmap=cmap,vmin=lim[ind_mwf][0],vmax=lim[ind_mwf][1])
        ax[0,0].set_title('MWF')
        fig.colorbar(im0, ax=ax[0,0])

        # m1 and m2 map
        data = np.rot90(inv_data[:,:,0,idx])
        im1  = ax[0,1].imshow(data,cmap=cmap,vmin=lim[0][0],vmax=lim[0][1])            
        ax[0,1].set_title('m1')
        fig.colorbar(im1, ax=ax[0,1])
        
        data = np.rot90(inv_data[:,:,2,idx])
        im2 = ax[0,2].imshow(data,cmap=cmap,vmin=lim[2][0],vmax=lim[2][1])
        ax[0,2].set_title('m2')
        fig.colorbar(im2, ax=ax[0,2])            
                        
        # misfit map
        data = np.rot90(inv_data[:,:,-1,idx])
        im3  = ax[1,0].imshow(data,cmap=cmap,vmin=lim[-1][0],vmax=lim[-1][1])
        ax[1,0].set_title('misfit')
        fig.colorbar(im3, ax=ax[1,0])

        # m1 and m2 sigma map
        data = np.rot90(inv_data[:,:,1,idx])
        im4  = ax[1,1].imshow(data,cmap=cmap,vmin=lim[1][0],vmax=lim[1][1])
        ax[1,1].set_title('m1_sig')
        fig.colorbar(im4, ax=ax[1,1])
        
        data = np.rot90(inv_data[:,:,3,idx])
        im5  = ax[1,2].imshow(data,cmap=cmap,vmin=lim[3][0],vmax=lim[3][1])
        ax[1,2].set_title('m2_sig')
        fig.colorbar(im5, ax=ax[1,2])

        if pso_class.n_comp == 2:  
            
            # map for integral around m2
            data = np.rot90(inv_data[:,:,4,idx])
            im6  = ax[1,3].imshow(data,cmap=cmap,vmin=lim[4][0],vmax=lim[4][1])
            ax[1,3].set_title('int2')
            fig.colorbar(im6, ax=ax[1,3])
                
        elif pso_class.n_comp == 3:
            
            # m3 map
            data = np.rot90(inv_data[:,:,4,idx])
            im6  = ax[0,3].imshow(data,cmap=cmap,vmin=lim[4][0],vmax=lim[4][1])
            ax[0,3].set_title('m3')
            fig.colorbar(im6, ax=ax[0,3])

            # m3 sigma map
            data = np.rot90(inv_data[:,:,5,idx])
            im7  = ax[1,3].imshow(data,cmap=cmap,vmin=lim[5][0],vmax=lim[5][1])
            ax[1,3].set_title('m3_sig')
            fig.colorbar(im7, ax=ax[1,3])

            # maps for integral around m2 and m3
            data = np.rot90(inv_data[:,:,6,idx])
            im8  = ax[2,2].imshow(data,cmap=cmap,vmin=lim[6][0],vmax=lim[6][1])
            ax[2,2].set_title('int2')
            fig.colorbar(im8, ax=ax[2,2])
            
            data = np.rot90(inv_data[:,:,7,idx])
            im9 = ax[2,3].imshow(data,cmap=cmap,vmin=lim[7][0],vmax=lim[7][1])
            ax[2,3].set_title('int3')
            fig.colorbar(im9, ax=ax[2,3])
            
        # set axis of                            
        for ii in range(ax.shape[0]*ax.shape[1]):
            ax.flat[ii].axis('off')
            ax.flat[ii].set_facecolor('b')
            fig.set_facecolor('w')
        
        # figure title
        if idx == -1: ind = 'best fit'
        dist = '3-component' if pso_class.n_comp == 3 else '2-component'
        sig  = 'CT2S' if pso_class.inv_CT2S and signal == 'T2S' else signal
            
        fig.suptitle(f'Calculated {sig} parameter maps using a {dist} relaxation-model. PSO cycle: {str(ind).zfill(2)}')
        
        # make the destination directory and save figure
        if save:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)      
            fig.savefig(save_path, dpi=save_dpi, format=save_format.lower(), bbox_inches='tight')
        
        if show:
            plt.show()
        if not show:
            plt.close()
                    
    def pareto_pixel_single(self, inv_data: dict, pso_class: object, num_val: int, 
                            corridor=[False, float], save_path=None, save_format=None, 
                            save_dpi=300, save=False, verbose=False):
        
        """
        Plot the best `num_val` pareto-optimal pso solutions for a single pixel
        from a single inversion, with optional percentage-based corridor behind the best fit.
        
        Args:
            inv_data: dictionary of inverted pso results as returned from the main function\n
            pso_class: instance of the core ParticleSwarmOptimizer\n
            num_val: number of best solutions to display\n
            corridor: percentage range behind the best fit to highlight
        """
        
        # complex signal needs to be labeled as T2S, as PSO returns it like this
        decay = pso_class.decay_types[0]  

        # index of mwf and fit ind the results array as returned from PSOmain
        # n_comp=2 --> 0:m1, 1:m1_sig, 2:m2, 3:m2_sig, 4:int2, 5:mwf, (6:mw_f, 7:fw_f, 8:phi), -1:fit
        # n_comp=3 --> 0:m1, 1:m1_sig, 2:m2, 3:m2_sig, 4:m3, 5:m3_sig, 6:int2, 7:int3, (8:mw_f, 9:ew_f, 10:aw_f, 11:phi), -1:fit
        ind_mwf     = 5 if pso_class.n_comp == 2 else 8
        
        # get inversion results
        MWF  = np.array([r[f"mod{decay}"][ind_mwf] for r in inv_data])
        fits = np.array([r["fit"]                  for r in inv_data])

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
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
            ax.tick_params(axis='both', labelsize=12)
            
            plt.tight_layout()    
            fig.set_facecolor('w')
            plt.show()
            
            # make the destination directory and save figure
            if save:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)      
                fig.savefig(save_path, dpi=save_dpi, format=save_format.lower(), bbox_inches='tight')
                if verbose: 
                    print(f'Saved "{os.path.basename(save_path)}" to {os.path.dirname(save_path)} ...')
        
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
            leg1 = ax1.legend(loc='upper right', fontsize=14)
            for handle in leg1.get_lines():
                handle.set_markersize(7)
        
            # axis specifications
            ax1.set_xlabel(r'Global best Fit [$\times 10^{-3}$]', fontsize=16)
            ax1.set_ylabel('Global best MWF []', fontsize=16)    
            ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
            ax1.tick_params(axis='both', labelsize=12)
            
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
            ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
            ax2.tick_params(axis='both', labelsize=12)
        
            plt.tight_layout()    
        
            # make the destination directory and save figure
            if save:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)      
                fig.savefig(save_path, dpi=save_dpi, format=save_format.lower(), bbox_inches='tight')
                if verbose: 
                    print(f'Saved "{os.path.basename(save_path)}" to {os.path.dirname(save_path)} ...')

            plt.show()
            
    def pareto_pixel_joint(self, inv_data: dict, pso_class: object, num_val: int, 
                           corridor=[False, float], save_path=None, save_format=None, 
                           save_dpi=300, save=False, verbose=False):
        
        """
        Plot the best `num_val` pareto-optimal pso solutions for a single pixel
        from a joint inversion, with optional percentage-based corridor behind the best fit.
        
        Args:
            inv_data: dictionary of inverted pso results as returned from the main function\n
            pso_class: instance of the core ParticleSwarmOptimizer\n
            num_val: number of best solutions to display\n
            corridor: percentage range behind the best fit to highlight
        """
        
        # complex signal needs to be labeled as T2S, as PSO returns it like this
        sig1, sig2 = pso_class.decay_types         

        # index of mwf and fit ind the results array as returned from PSOmain
        # n_comp=2 --> 0:m1, 1:m1_sig, 2:m2, 3:m2_sig, 4:int2, 5:mwf, (6:mw_f, 7:fw_f, 8:phi), -1:fit
        # n_comp=3 --> 0:m1, 1:m1_sig, 2:m2, 3:m2_sig, 4:m3, 5:m3_sig, 6:int2, 7:int3, (8:mw_f, 9:ew_f, 10:aw_f, 11:phi), -1:fit
        ind_mwf    = 5 if pso_class.n_comp == 2 else 8
        
        MWF_sig1   = np.array([r[f"mod{sig1}"][ind_mwf] for r in inv_data])
        MWF_sig2   = np.array([r[f"mod{sig2}"][ind_mwf] for r in inv_data])    
        fits       = np.array([r["fit"] for r in inv_data])
        
        # mask for best n values
        idx_sorted = np.argsort(fits)
        idx_best   = idx_sorted[:num_val]
        idx_worst  = idx_sorted[num_val:]
        
        # best fit MWF
        MWF_sig1_best_fit = MWF_sig1[idx_sorted[np.argmin(fits[idx_sorted])]]
        MWF_sig2_best_fit = MWF_sig2[idx_sorted[np.argmin(fits[idx_sorted])]]
        
        if not corridor[0]:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharey=False)
            label = (f'Best MWF {sig1}: {MWF_sig1_best_fit:.3f}\n'
                     f'Best MWF {sig2}: {MWF_sig2_best_fit:.3f}\n'
                     f'Best FIT: {np.round(np.min(fits),5)}')
            
            # plot of all solutions per signal
            ax.plot(fits[idx_best]*1e3, MWF_sig1[idx_best], markersize=2, linestyle='none', marker='o',
                    markeredgewidth=0.5, color='r', markerfacecolor='lightpink', label=f'{sig1}: 10e3 best results')            
            ax.plot(fits[idx_best]*1e3, MWF_sig2[idx_best], markersize=2, linestyle='none', marker='o',
                    markeredgewidth=0.5, color='b', markerfacecolor='lightblue', label=f'{sig2}: 10e3 best results')
            
            # plot of the best solution per signal
            ax.plot(np.min(fits[idx_best])*1e3, MWF_sig1_best_fit,
                    markersize=5, linestyle='none', marker='o', markeredgewidth=0.5,
                    color='b', label=label)
            ax.plot(np.min(fits[idx_best])*1e3, MWF_sig2_best_fit,
                    markersize=5, linestyle='none', marker='o', markeredgewidth=0.5,
                    color='b')
            
            # legend specifications
            leg = ax.legend(loc='upper right', fontsize=14)
            for handle in leg.get_lines():
                handle.set_markersize(7)
            
            # axis specification
            ax.set_ylabel('Global best MWF []', fontsize=16)    
            ax.set_xlabel(r'Global best Fit [$\times 10^{-3}$]', fontsize=16)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
            ax.tick_params(axis='both', labelsize=12)
            
            plt.tight_layout()    
            fig.set_facecolor('w')
            plt.show()
            
            # make the destination directory and save figure
            if save:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)      
                fig.savefig(save_path, dpi=save_dpi, format=save_format.lower(), bbox_inches='tight')
                if verbose: 
                    print(f'Saved "{os.path.basename(save_path)}" to {os.path.dirname(save_path)} ...')
        
        if corridor[0]:            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
            label = (f'Best MWF {sig1}: {MWF_sig1_best_fit:.3f}\n'
                     f'Best MWF {sig2}: {MWF_sig2_best_fit:.3f}\n'
                     f'Best FIT: {np.round(np.min(fits),5)}')
        
            # plot of all solutions per signal
            ax1.plot(fits[idx_worst]*1e3, MWF_sig1[idx_worst], markersize=2, linestyle='none', marker='o',
                     markeredgewidth=0.5, color='gray', markerfacecolor='lightgray', label='20e3 PSO runs')    
            ax1.plot(fits[idx_worst]*1e3, MWF_sig2[idx_worst], markersize=2, linestyle='none', marker='o',
                     markeredgewidth=0.5, color='gray', markerfacecolor='lightgray')    
    
            ax1.plot(fits[idx_best]*1e3, MWF_sig1[idx_best],markersize=2, linestyle='none', marker='o',
                     markeredgewidth=0.5, color='r', markerfacecolor='lightpink',label=f'{sig1} - 10e3 best')    
            ax1.plot(fits[idx_best]*1e3, MWF_sig1[idx_best],markersize=2, linestyle='none', marker='o',
                     markeredgewidth=0.5, color='b', markerfacecolor='lightblue',label=f'{sig2} - 10e3 best')     
        
            # corridor for best fit +3% of best fit
            corridor_range = 1 + corridor[1]/100
            low, high = np.min(fits)*1e3, np.min(fits)*1e3*corridor_range
            ax1.axvspan(low, high, color="lightblue", alpha=0.5)
            ax1.plot([], [], color="lightblue", linewidth=8, alpha=0.5,
                     label='Uncertainty spread')  # label='corridor: bestFit +3% * BF'
        
            # legend specifications
            leg1 = ax1.legend(loc='upper right', fontsize=14)
            for handle in leg1.get_lines():
                handle.set_markersize(7)
        
            # axis specifications
            ax1.set_xlabel(r'Global best Fit [$\times 10^{-3}$]', fontsize=16)
            ax1.set_ylabel('Global best MWF []', fontsize=16)    
            ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
            ax1.tick_params(axis='both', labelsize=12)
            
            # subplot 2 showing pareto curve for best n values
            ax2.plot(fits[idx_best]*1e3, MWF_sig1[idx_best], markersize=2, linestyle='none', marker='o',
                     markeredgewidth=0.5, color='r', markerfacecolor='lightpink',label=f'{sig1} best 10e3')
            ax2.plot(fits[idx_best]*1e3, MWF_sig2[idx_best], markersize=2, linestyle='none', marker='o',
                     markeredgewidth=0.5, color='b', markerfacecolor='lightblue',label=f'{sig2} best 10e3')
            
            ax2.plot(np.min(fits[idx_best])*1e3, MWF_sig1_best_fit,
                     markersize=5, linestyle='none', marker='o', markeredgewidth=0.5,color='k', label=label)
            ax2.plot(np.min(fits[idx_best])*1e3, MWF_sig2_best_fit,
                     markersize=5, linestyle='none', marker='o', markeredgewidth=0.5,color='k')
        
            # legend specifications
            leg2 = ax2.legend(loc='upper right', fontsize=14)
            for handle in leg2.get_lines():
                handle.set_markersize(7)
        
            # axis specification
            ax2.set_ylabel('Global best MWF []', fontsize=16)    
            ax2.set_xlabel(r'Global best Fit [$\times 10^{-3}$]', fontsize=16)
            ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
            ax2.tick_params(axis='both', labelsize=12)
        
            plt.tight_layout()    
        
            # make the destination directory and save figure
            if save:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)      
                fig.savefig(save_path, dpi=save_dpi, format=save_format.lower(), bbox_inches='tight')
                if verbose: 
                    print(f'Saved "{os.path.basename(save_path)}" to {os.path.dirname(save_path)} ...')

            plt.show()
                
    def _iter_test(self, mwf_data: object, pso_class: object, position: tuple, n_iter: int,
                   save_path=None, save_format=None, save_dpi=300, save=False, verbose=False):

        '''
        Display diagnostic plots for a single inverted mwf pixel during pso execution.
        
        Features:
            a) Brain slice with a scatter marker indicating the current location
            b) Synthetic vs. observed decay curve (magnitude)
            c) Global best pixel distribution across iterations
            d) Global best fit evolution across iterations
    
        Args:
            mwf_data: runtime [x, y]
            pso_class: instance of the core ParticleSwarmOptimizer\n
            position: [x,y,n_slice]
            n_iter: number of iterations until the threshold is met
            
        Note:
            Triggered during the execution of the pso iteration test.         
        '''
        
        sig       = pso_class.decay_types[0]
        thresh    = pso_class.config.PSO_spec.comp_mode.iter_test.thresh
        sig_label = 'CT2S' if pso_class.inv_CT2S and sig == 'T2S' else sig
        yy,xx     = position[0], position[1]
        
        fig, ax   = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), tight_layout=True)
        
        ax[0,0].imshow(mwf_data, cmap=matplotlib.colormaps.get_cmap('viridis'), vmin=0, vmax=np.nanmax(mwf_data))
        ax[0,0].scatter(xx, yy, color='red', s=10)
        ax[0,0].set_xticks([]); ax[0,0].set_yticks([])
        ax[0,0].set_title(f'2D map of {sig} signal summed over echoes')
        [sp.set_visible(False) for sp in ax[0,0].spines.values()]

        if sig == 'T2': 
            timesteps = np.arange(0,24)
        if sig == 'T2S': 
            timesteps = np.arange(0,32)
        
        if sig_label != 'CT2S':
            ax[0,1].plot(timesteps, pso_class.obs_decay[sig], 'k', linewidth=2)
            ax[0,1].plot(timesteps, pso_class.glob_syn_data[sig], markersize=2, linestyle='-', 
                         marker='o', markeredgewidth=0.5, color='red', linewidth=2, markerfacecolor='lightpink')    
            ax[0,1].set_title(f'{sig_label}: observed (black) vs. synthetic (red) signal')
            ax[0,1].set_ylim(np.min(pso_class.obs_decay[sig]-0.1), np.max(pso_class.obs_decay[sig])+0.1)        
        else:
            ax[0,1].plot(timesteps, np.abs(pso_class.obs_decay[sig]), 'k', linewidth=2)
            ax[0,1].plot(timesteps, np.abs(pso_class.glob_syn_data[sig]), markersize=2, linestyle='-', 
                         marker='o', markeredgewidth=0.5, color='red', linewidth=2, markerfacecolor='lightpink')    
            ax[0,1].set_title(f'{sig_label}: observed (black) vs. synthetic (red) signal')
            ax[0,1].set_ylim(np.min(np.abs(pso_class.obs_decay[sig]-0.1)), np.max(np.abs(pso_class.obs_decay[sig])+0.1))

        ax[1,0].plot(np.arange(0, len(pso_class.glob_ind_list[sig]), 1), pso_class.glob_ind_list[sig], 
                     markersize=2, linestyle='None', marker='o', color='b')
        ax[1,0].set_ylim(0.5,  pso_class.n_part+0.5)
        ax[1,0].set_xlim(-0.5, pso_class.n_iter+0.5)
        ax[1,0].set_xticks(np.arange(0, pso_class.n_iter+1, int(pso_class.n_iter/5)), 
                           np.arange(0, pso_class.n_iter+1, int(pso_class.n_iter/5), dtype=int))
        ax[1,0].set_yticks(np.arange(0,pso_class.n_part+1,pso_class.n_part/10))
        ax[1,0].set_title('global best particle')   
        
        ylim_min = np.min(pso_class.glob_fit_list[sig])-np.min(pso_class.glob_fit_list[sig])/3
        ylim_max = np.max(pso_class.glob_fit_list[sig])+np.min(pso_class.glob_fit_list[sig])/3

        ax[1,1].plot(np.arange(0, len(pso_class.glob_fit_list[sig]), 1), 
                     pso_class.glob_fit_list[sig], markersize=2, linestyle='-', marker='o', color='b')
        ax[1,1].set_ylim(ylim_min, ylim_max)
        ax[1,1].set_xlim(-5, pso_class.n_iter+5)
        ax[1,1].set_xticks(np.arange(0, pso_class.n_iter+1, int(pso_class.n_iter/5)),
                           np.arange(0, pso_class.n_iter+1, int(pso_class.n_iter/5), dtype=int))       
        # ax[1,1].set_yticks(np.arange(ylim_min, ylim_max, steps))

        ax[1,1].set_title('global best fit')
             
        value = pso_class.glob_fit[sig]; n = 0
        while abs(value) < 1:
            value *= 10
            n -= 1
        rounded_number = round(value, 3)
        value = f'{rounded_number} * 10e{n}'

        text  = (f'glob best [px: fit]:\n{pso_class.glob_ind[sig]+1}: {value}\n\n'
                 f'MWFcalc: {np.round(pso_class.glob_mod[sig][-1], 4)}\n'
                 f'IterThresh ({thresh}%): {n_iter}')
        
        x_lim = (pso_class.n_iter+1)/10*7
        ax[1,1].text(x_lim, ylim_max/10*9.5, text, va='top')

        plt.tight_layout()    
            
        # make the destination directory and save figure
        if save:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)    
            fig.savefig(save_path, dpi=save_dpi, format=save_format.lower())  
            if verbose:     
                print(f'Saved "{os.path.basename(save_path)}" to {os.path.dirname(save_path)} ...')

        plt.show()
        
###############################################################################

class PSOvideos():    
    
    def __init__(self): 
        pass  
    
    def build_gif(self, filelist: list, savepath: str, fps=1, loops=0):        

        images = []
        
        for filename in sorted(filelist):
            images.append(imageio.imread(filename))

        imageio.mimsave(f'{savepath}', images, duration=fps, loop=loops)