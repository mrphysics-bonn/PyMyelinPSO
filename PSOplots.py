# -*- coding: utf-8 -*-
"""
Class and methods for visualizing PSO-derived results on invivo MRI data.

@author: Martin Kobe, martin.kobe@ufz.de, martin.kobe@email.de

@status: 03.2024; part of the JIMM Project (DZNE Bonn & UFZ Leipzig)

@ideas: Clustering using for example DBscan (from scikit.cluster import DBSCAN)
"""

import os, matplotlib, time
import matplotlib.pyplot as plt
import numpy             as np
import imageio.v2        as imageio


class PSOgrafics():

    def __init__(self):
        pass   
        
    def plotIterTest(self,
                     MWFdata:  object,
                     PSOclass: object,
                     position: tuple,
                     savepath: str,
                     string:   str):

        sig           = PSOclass.signType[0]
        yy,xx,noSlice = position[0], position[1], position[2]
        savepath      = f'{savepath}/{string[0]}/'
        
        fig, ax       = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), tight_layout=True)
        
        ax[0,0].imshow(MWFdata, cmap=matplotlib.colormaps.get_cmap('viridis'), vmin=0, vmax=np.max(MWFdata))
        ax[0,0].scatter(xx, yy, color='red', s=10)
        ax[0,0].set_xticks([]); ax[0,0].set_yticks([])
        ax[0,0].set_title(f'Calculated MWF map for {PSOclass.signType[0]} signal on slice {noSlice}')
        [sp.set_visible(False) for sp in ax[0,0].spines.values()]

        if sig == 'T2': 
            timesteps = np.arange(0,24)
        if sig == 'T2S': 
            timesteps = np.arange(0,32)
        
        ax[0,1].plot(timesteps, PSOclass.obsData[sig], 'k', linewidth=2)
        ax[0,1].plot(timesteps, PSOclass.globSynDat[sig], markersize=2, linestyle='-', 
                     marker='o', markeredgewidth=0.5, color='red', linewidth=2, markerfacecolor='lightpink')

        ax[0,1].set_title(f'{sig}: observed (black) vs. synthetic (red) signal')
        ax[0,1].set_ylim(np.min(PSOclass.obsData[sig]-0.1), np.max(PSOclass.obsData[sig])+0.1)

        ax[1,0].plot(np.arange(0, len(PSOclass.globInd_list[sig]), 1), PSOclass.globInd_list[sig], 
                     markersize=2, linestyle='None', marker='o', color='b')
        ax[1,0].set_ylim(0.5,  PSOclass.noPart+0.5)
        ax[1,0].set_xlim(-0.5, PSOclass.noIter+0.5)
        ax[1,0].set_xticks(np.arange(0, PSOclass.noIter+1, int(PSOclass.noIter/5)), 
                           np.arange(0, PSOclass.noIter+1, int(PSOclass.noIter/5), dtype=int))
        ax[1,0].set_yticks(np.arange(0,PSOclass.noPart+1,PSOclass.noPart/10))
        ax[1,0].set_title('global best particle')   
        
        ylim_min = np.min(PSOclass.globFit_list[sig])-np.min(PSOclass.globFit_list[sig])/3
        ylim_max = np.max(PSOclass.globFit_list[sig])+np.min(PSOclass.globFit_list[sig])/3

        ax[1,1].plot(np.arange(0, len(PSOclass.globFit_list[sig]), 1), 
                     PSOclass.globFit_list[sig], markersize=2, linestyle='-', marker='o', color='b')
        ax[1,1].set_ylim(ylim_min, ylim_max)
        ax[1,1].set_xlim(-5, PSOclass.noIter+5)
        ax[1,1].set_xticks(np.arange(0, PSOclass.noIter+1, int(PSOclass.noIter/5)),
                           np.arange(0, PSOclass.noIter+1, int(PSOclass.noIter/5), dtype=int))       
        # ax[1,1].set_yticks(np.arange(ylim_min, ylim_max, steps))

        ax[1,1].set_title('global best fit')
             
        value = PSOclass.globFit[sig]; n = 0
        while abs(value) < 1:
            value *= 10
            n -= 1
        rounded_number = round(value, 3)
        value = f'{rounded_number} * 10e{n}'

        text  = (f'glob best [px: fit]:\n{PSOclass.globIndex[sig]+1}: {value}\n\n'
                  f'MWFcalc:{np.round(PSOclass.globMod[sig][-1], 4)}')
        
        x_lim = (PSOclass.noIter+1)/10*7
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
        
        if PSOclass.signType[0] == 'T2S': upper = 0.5
        if PSOclass.signType[0] == 'T2':  upper = 0.35
        
        param     = kwargs.get('limit', (object,False))[0]
        paramBool = kwargs.get('limit', (object,False))[1]
        performance_test = kwargs.get('performance_test', False)
        lim       = {sig:[] for sig in PSOclass.signType}
        
        if paramBool == False:            
            if PSOclass.noPeaks == 'GAUSS':                
                for sig in PSOclass.signType:                    
                    lim[sig] = [getattr(getattr(PSOclass, sig), PSOclass.noPeaks).m1,
                                getattr(getattr(PSOclass, sig), PSOclass.noPeaks).m1_sig,
                                getattr(getattr(PSOclass, sig), PSOclass.noPeaks).m2,
                                getattr(getattr(PSOclass, sig), PSOclass.noPeaks).m2_sig,
                                getattr(getattr(PSOclass, sig), PSOclass.noPeaks).int2,
                                (0,upper), (0,0.005)]
            
            if PSOclass.noPeaks == 'DIRAC':                
                for sig in PSOclass.signType:                    
                    lim[sig] = [getattr(getattr(PSOclass, sig), PSOclass.noPeaks).m1,
                                getattr(getattr(PSOclass, sig), PSOclass.noPeaks).m1_sig,
                                getattr(getattr(PSOclass, sig), PSOclass.noPeaks).m2,
                                getattr(getattr(PSOclass, sig), PSOclass.noPeaks).m2_sig,
                                getattr(getattr(PSOclass, sig), PSOclass.noPeaks).m3,
                                getattr(getattr(PSOclass, sig), PSOclass.noPeaks).m3_sig,
                                getattr(getattr(PSOclass, sig), PSOclass.noPeaks).int2,
                                getattr(getattr(PSOclass, sig), PSOclass.noPeaks).int3,
                                (0,upper), (0,0.005)]
        
        if paramBool == True:
            if PSOclass.noPeaks == 'GAUSS':                
                for sig in PSOclass.signType:                    
                    lim[sig] = [param[f'SynData{sig}']['m1'],(0,1),
                                param[f'SynData{sig}']['m2'],param[f'SynData{sig}']['m2_sig'],
                                param[f'SynData{sig}']['integ2'],(0,0.35), (0,0.005)]  
        
            if PSOclass.noPeaks == 'DIRAC':                
                for sig in PSOclass.signType:                    
                    lim[sig] = [param[f'SynData{sig}']['m1'],param[f'SynData{sig}']['m1_sig'],
                                param[f'SynData{sig}']['m2'],param[f'SynData{sig}']['m2_sig'],
                                param[f'SynData{sig}']['m3'],param[f'SynData{sig}']['m3_sig'],
                                param[f'SynData{sig}']['integ2'],param[f'SynData{sig}']['integ3'],
                                (0,0.35), (0,0.005)]        

        cmap = matplotlib.colormaps.get_cmap('viridis')
        cmap.set_bad(color='black')
        
        for sig in PSOclass.signType:
            
            PSOres  = PSOresult[sig]  
            _lim    = lim[sig]
            ind     = index
            
            if PSOclass.noPeaks=='GAUSS':
                fig, ax = plt.subplots(2, 4, figsize=(12, 6))
            
            if PSOclass.noPeaks=='DIRAC':
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

            if PSOclass.noPeaks == 'GAUSS':
                im0 = ax[0,0].imshow(PSOres[5,:,:,ind],cmap=cmap,vmin=_lim[5][0],vmax=_lim[5][1])
                ax[0,0].set_title('MWF')
                fig.colorbar(im0, ax=ax[0,0])
                
                im6 = ax[1,3].imshow(PSOres[4,:,:,ind],cmap=cmap,vmin=_lim[4][0],vmax=_lim[4][1])
                ax[1,3].set_title('int2')
                fig.colorbar(im6, ax=ax[1,3])
                
            if PSOclass.noPeaks == 'DIRAC':
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

            dist = '3 peaks (DIRAC)' if PSOclass.noPeaks == 'DIRAC' else '2 peaks (GAUSS)'
        
            fig.suptitle(f'Calculated {sig} map for {dist}. PSO iteration: {str(ind).zfill(2)}')
        
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
        
        vmin=getattr(getattr(PSOclass, PSOclass.signType[0]), PSOclass.noPeaks).m1[0]
        vmax=getattr(getattr(PSOclass, PSOclass.signType[0]), PSOclass.noPeaks).m1[1]
        
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

        fig.suptitle(f'Pix.[{yy},{xx}]: Calculated parameters for signal {PSOclass.signType[0]} | {PSOclass.noPeaks}',fontsize=14)
        
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
        
        for sig in PSOclass.signType:
        
            # pixel position        
            yy,xx        = position[0],position[1]        
            MWFarray     = PSOresultPIX[sig][-2,:-1]
            FITarray     = PSOresultPIX[sig][-1,:-1]

            # plot raw
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6), tight_layout=True)       
            fig.gca().xaxis.set_major_formatter('{:.3f}'.format)
        
            _lim = (0,0.35) if PSOclass.noPeaks=='GAUSS' else (0,0.25)
            
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

            ax[1].set_title(f'{sig}(y{yy}x{xx}) | {PSOclass.noIter}Iter,'
                            f'{PSOclass.noPart}Part,{PSOclass.noPSOIter}PSO | {PSOclass.noPeaks}')
        
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
            
            ax[1].set_title(f'{sig}(y{yy}x{xx}) | {PSOclass.noIter}Iter,'
                          f'{PSOclass.noPart}Part,{PSOclass.noPSOIter}PSO | {PSOclass.noPeaks}')
            
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