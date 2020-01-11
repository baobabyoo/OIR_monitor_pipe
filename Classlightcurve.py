## environment terada2019
# conda create --name terada2019 python=3.7
# pip install --upgrade pip
# pip install astropy scipy
# pip install photutils
# pip install jupyter matplotlib h5py aplpy pyregion PyAVM healpy
# pip install astroquery
# pip install pandas
# pip install -U statsmodels
# pip install -U scikit-learn

import os
import sys
import time

import pandas as pd
import numpy as np
from astropy.stats import mad_std
from astropy.stats import sigma_clip
from astropy.convolution import convolve, Box1DKernel
from astropy.stats import biweight_location, biweight_scale

import scipy
from scipy import signal
from scipy import optimize
from scipy import stats
import scipy.fftpack
from scipy.optimize import minimize

import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

from sklearn.preprocessing import MinMaxScaler

import matplotlib
import matplotlib.pyplot as plt




class lightcurve:
    '''
    Light curve class.

    '''

    def __init__(self, filename='', time_col_label='', data_col_label='', verbose=False,
                 plot_preview=True, previewfig_name='preview.pdf',
                 xlim = [], ylim = []
                ):
        '''
        Initializer.

        Keywords:

        filename               [str]:  name of the input CSV file.

        time_col_label         [str]:  Time column name of the input CSV table.

        data_col_label         [str]:  Data column name of the input CSV table.

        plot_preview    [True/False]:  Generate preview figure if true.

        previewfig_name        [str] : Name of the output preview figure.

        xlim, ylim  [list of float] : X and y ranges for plotting

        '''

        self.verbose = verbose
        self.time_col_label = time_col_label
        self.data_col_label = data_col_label

        self.meantimestep   = 0.0
        self.mintimestep    = 0.0
        self.maxtimestep    = 0.0
        self.stddevtimestep = 0.0
        self.mintime        = 0.0
        self.maxtime        = 0.0

        self.period         = []
        self.period_error   = []

        self.Q              = 0.0
        self.M              = 0.0

        try:
            intable = pd.read_csv(filename)
            self.time    = intable[time_col_label]
            self.data    = intable[data_col_label]
            self.mintime = np.min(self.time)
            self.maxtime = np.max(self.time)

            self.time_plot = self.time
            self.data_plot = self.data
            self.x_label   = time_col_label
            self.y_label   = data_col_label

        except:
            if (verbose == True):
                print("Warning. Failed to open input file. Quit")

        if (plot_preview==True):
            self.plotdata(previewfig_name, label="Input data", xlim=xlim, ylim=ylim)

        self.get_timestep()


    def __del__(self):
        pass




    ##### Functions #######################################################
    def plotdata(self, outfigname, label='None', xlim=[], ylim=[],
                figsize=[10,6], plot_range=[0.1, 0.1, 0.85, 0.85]):
        '''
        Function to plot data as figure.

        Keyword:

        outfigname      [str]       : Name of the output figure.

        xlim, ylim  [list of float] : X and y ranges for plotting

        '''
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(plot_range)

        ax.plot(self.time_plot, self.data_plot, 'o',  markersize=2, color=(0.2,0.2,0.2,1) )
        plt.tick_params(labelsize=14)
        plt.xlabel(self.x_label, fontsize=14)
        plt.ylabel(self.y_label, fontsize=14)

        if (label != 'None'):
            ax.text(0.02, 0.92, label, transform=ax.transAxes,
                     color=(0,0,0,1),
                     fontsize=14, horizontalalignment='left')

        if ( len(xlim) > 1 ):
            plt.xlim(xlim[0], xlim[1])

        if ( len(ylim) > 1 ):
            plt.ylim(ylim[0], ylim[1])

        plt.show()
        plt.savefig(outfigname)



    def fft(self, verbose=False,
            plot_preview=False, previewfig_name='preview_fft.pdf',
            xlim=[], ylim=[]
           ):
        '''
        Doing FFT and return the frequency and power-spectrum of the data

        plot_preview    [True/False] :  Generate preview figure if true.

        previewfig_name        [str] :  Name of the output preview figure.

        xlim, ylim  [list of float]  : X and y ranges for plotting
        '''

        # obtaining power-spectrum
        w = scipy.fftpack.rfft(self.data)
        spectrum = abs(w)**2

        # obtaining frequency
        freq = scipy.fftpack.rfftfreq( np.size(self.time), self.meantimestep)

        if (plot_preview==True):
            self.time_plot = freq
            self.data_plot = spectrum
            self.x_label   = 'Frequency'
            self.y_label   = 'Power'
            self.plotdata(previewfig_name, label='FFT', xlim=xlim, ylim=ylim)

        return freq, spectrum



    def boxcar_smooth(self, boxsize=0.0):
        '''
        Perform boxcar smoothing and return the boxcar-smoothed light curve.


        boxsize             [float]  : size of the box in the same unit with
                                       the input time.

        '''

        if (boxsize > 0.0):
            box_pts = int( round( boxsize / self.meantimestep) )
            box     = np.ones(box_pts) / box_pts
            smoothed_data = np.convolve(self.data, box, mode='same')
            return smoothed_data
        else:
            return self.data



    ##### Methods #########################################################
    def get_timestep(self):
        '''
        Need to obtain the following information:
        The minimum time stamp, the maximum time stamp,
        the mean, minimum, and maximum time stemps.

        It will also update the information of timestep accordingly.
        '''

        timestep_list = []
        for i in range(0, len(self.time)-1 ):
            timestep_list.append( self.time[i+1] - self.time[i] )
        self.meantimestep   = np.mean(timestep_list)
        self.mintimestep    = np.min(timestep_list)
        self.maxtimestep    = np.max(timestep_list)
        self.stddevtimestep = np.std(timestep_list)

        if (self.verbose==True):
            if ( len( set(timestep_list) ) > 1 ):
                print( "Warning. Time step is not uniform. \n")
                print( "Identified time steps : ", set(timestep_list), "\n" )



    def interpolate(self, time_grid = np.array([]),
                    plot_preview=False, previewfig_name='preview_interpolate.pdf',
                    xlim = [], ylim = []
                   ):
        '''
        Remove NaN from the data array and then interpolate the data onto the regular
        time grids as specified by input.


        Keywords:

        time_grid      [numpy array] :  Time grid to interpolate data onto.
                                        In the same unit with the input time.

        plot_preview    [True/False] :  Generate preview figure if true.

        previewfig_name        [str] :  Name of the output preview figure.

        xlim, ylim  [list of float]  : X and y ranges for plotting

        '''

        if ( np.size(time_grid) > 0 ):

            # removing NaNs
            self.time = self.time[np.isfinite(self.data)]
            self.data = self.data[np.isfinite(self.data)]

            # interpolation
            self.data = np.interp(time_grid, self.time, self.data)
            self.time = time_grid
            timestep  = abs( time_grid[1] - time_grid[0] )
            self.meantimestep = timestep
            self.mintimeste   = timestep
            self.maxtimestep  = timestep

        if (plot_preview==True):
            self.time_plot = self.time
            self.data_plot = self.data
            self.plotdata(previewfig_name, label='Interpolated', xlim=xlim, ylim=ylim)



    def get_period(self, method='fft'):
        '''

        Evaluating the period assuming that the source has periodic variation.

        Keywords:

        method           [str]  : Methods to derive period.
            fft:  Using FFT to find periods from the input data.
                  The derived periods are sorted according to the strength of power spectrum.

        '''

        if (method == 'fft'):

            # evaluate period
            freq, spectrum = self.fft()
            self.period =  1.0 / freq[ np.argmax(spectrum) ]

            # evaluate error bar of period
            halfmax_spectrum = np.max(spectrum) / 2.0

            spectrum = abs(spectrum - halfmax_spectrum)
            index    = np.argsort(spectrum)
            freq_err = abs( freq[ index[0] ] - freq[ index[1] ] ) / 2.35
            self.period_error = self.period - 1.0 / ( 1.0/self.period - freq_err )

            if (self.verbose==True):
                print('Found periods derived using FFT : ',  self.period, "+/-", self.period_error)



    def phase_fold(self, residual=False, timebin=0.0,
                   plot_preview=False, previewfig_name='preview_phasefold.pdf',
                   xlim = [], ylim = []):
        '''
        Phase-fold the data (in time dimension).
        This procedure only affect plotting (i.e., self.data_plot, self.time_plot).
        It does not update self.data or self.time.


        Keywords:

        residual       [True]/False] : If True, evaluate the residual after
                                       phase folding and subtracting the value of boxcar smoothed data.
                                       Otherwise, only do phase-folding.

        timebin              [float] : timebin used for boxcar smoothing.


        plot_preview    [True/False] :  Generate preview figure if true.

        previewfig_name        [str] :  Name of the output preview figure.

        xlim, ylim  [list of float]  : X and y ranges for plotting

        '''

        if ( self.period == 0.0 ):
            print("Warning. Period is not yet derived. Nothing is done.")
            return

        sort_idx = np.argsort( self.time % self.period )
        self.time_plot = (self.time % self.period)[sort_idx]
        self.data_plot = self.data[sort_idx]

        if ( (residual == True) and (timebin>0.0) ):
            box_pts = int( round(self.period / self.meantimestep) )
            box     = np.ones(box_pts) / box_pts
            smoothed_data = np.convolve(self.data_plot, box, mode='same')
            self.data_plot = self.data_plot - smoothed_data


        if (plot_preview==True):
            if (residual == True):
                label='Phase-folded residual'
            else:
                label='Phase-folded'

            self.x_label   = 'Phase-folded time'
            self.y_label   = self.data_col_label
            self.plotdata(previewfig_name, label=label, xlim=xlim, ylim=ylim)


    def get_M(self, sigma_clip=5.0):
        '''
        Evalute the M parameter, which was defined in Cody et al. (2014), ApJ
        This task use the self.data_plot data instead of the self.data data.
        Becareful.

        sigma_clip     [float] : clipping data above and below sigma_clip times 1-sigma,
                                 before analyzing M. (sigma_clip=5.0 in Cody+14)
        '''

        sigma = biweight_scale( self.data_plot )
        data  = self.data_plot[ (self.data_plot >= (-1.0 * sigma_clip * sigma) )
                              & (self.data_plot <= sigma_clip * sigma) ]

        sigma_d  = np.sqrt(np.mean(data**2))
        d_med    = np.median(data)
        num_data = len(data)
        d_10per_upp  = np.sort(data)[ (num_data-int(round(num_data*0.1))):num_data]
        d_10per_low  = np.sort(data)[0:int(round(num_data*0.1))]
        mean_d_10per = (np.mean(d_10per_upp) + np.mean(d_10per_low) )/ 2.0
        self.M       = (mean_d_10per - d_med) / sigma_d



    def get_Q(self, sigma=0.0, timebin=0.0):
        '''
        Evalute the Q parameter, which was defined in Cody et al. (2014), ApJ
        This task use the self.data_plot data instead of the self.data data.
        Becareful.


        Keyword :

        timebin              [float] : timebin used for boxcar smoothing.

        sigma                [float] : Estimated uncertainty of data
        '''

        self.phase_fold(timebin=timebin, residual=True)

        residual = self.data_plot
        rms_residual = np.sqrt(np.mean(residual**2))
        rms_raw      = np.sqrt(np.mean(self.data**2))
        self.Q       = ( rms_residual**2 - sigma**2 ) / (rms_raw**2 - sigma**2)



