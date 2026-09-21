
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.patches import Circle
import numpy as np
from numpy.fft import fft2, fftshift
from maoppy.utils import circavg
from maoppy.instrument import Instrument
from maoppy.psfmodel import Psfao, Turbulent
from maoppy.psffit import psffit
from scipy.ndimage import gaussian_filter

from scipy.signal import welch
from scipy.signal import convolve2d

from scipy.ndimage import maximum_filter

import h5py


def Gaussian(ref_image, sampling):
    N = ref_image.shape[-1]

    x = np.linspace(-N/2, N/2 - 1, N)
    x,y = np.meshgrid(x,x)

    sigma = sampling / 2 * np.sqrt(2 * np.log(2))

    gauss = np.exp(-(x ** 2 + y ** 2) / 2 / sigma ** 2)

    return gauss / np.sum(gauss)

def ComputePSFWeightingFunction(psf_series, sampling):
    average_frame = np.mean(psf_series, axis = 0)
    gaussian = Gaussian(average_frame, sampling)
    return convolve2d(average_frame, gaussian, mode = 'same')


def ComputeSingleFrameWCoG(psf, weighting_map):
    W, H = psf.shape
    x = np.linspace(-W/2, W/2 - 1, W)
    x,y = np.meshgrid(x,x)

    denominator = (psf * weighting_map).sum()

    x_numerator = (psf * weighting_map * x).sum()
    y_numerator = (psf * weighting_map * y).sum()

    x_wcog = x_numerator / denominator
    y_wcog = y_numerator / denominator

    return x_wcog, y_wcog

def ComputeWCoGGain(psf_series, weighting_map):

    avg_psf = psf_series.mean(axis = 0)

    test_frame = np.zeros_like(avg_psf)
    test_frame[1:,1:] = avg_psf[:-1,:-1]

    xcog_1, y_cog1 = ComputeSingleFrameWCoG(avg_psf, weighting_map)
    xcog_2, y_cog2 = ComputeSingleFrameWCoG(test_frame, weighting_map)

    gain = 0.5 * (xcog_2 - xcog_1) + 0.5 * (y_cog2 - y_cog1)

    return gain


def ComputeWCoG(psf_series, sampling):

    N, W, H = psf_series.shape
    weighting_map = ComputePSFWeightingFunction(psf_series=psf_series, sampling=sampling)

    x = np.linspace(-W/2, W/2 - 1, W)
    x,y = np.meshgrid(x,x)

    x = np.expand_dims(x, 0)
    y = np.expand_dims(y, 0)
    weighting_map = np.expand_dims(weighting_map, 0)

    WCoG_gain = ComputeWCoGGain(psf_series, weighting_map)
    

    denominator = (psf_series * weighting_map).sum(axis = (1,2))

    x_numerator = (psf_series * weighting_map * x).sum(axis = (1,2))
    y_numerator = (psf_series * weighting_map * y).sum(axis = (1,2))

    x_wcog = x_numerator / denominator / sampling / WCoG_gain
    y_wcog = y_numerator / denominator / sampling / WCoG_gain

    x_wcog = x_wcog - np.mean(x_wcog)
    y_wcog = y_wcog - np.mean(y_wcog)

    return x_wcog, y_wcog

def GetSignalPSD(signal, period):
    return welch(signal, 1/period, nperseg=500)

def dead_pixel_map(bkg, maxi=0.7):
    """Compute the map of dead pixels"""
    bkg = bkg - np.median(bkg)
    return bkg > (bkg.max()*maxi)


def filter_dead_pixel(img, dead_pix_map):
    """Filter dead pixels"""
    img[np.where(dead_pix_map==1)] = np.median(img)
    return img


def filter_dead_pixel_full_img(img):
    footprint = np.ones((3, 3))  # 3x3 neighborhood
    local_max = (img == maximum_filter(img, footprint=footprint))
    bright_pixels = local_max
    img[local_max] = np.median(img)
    return img

def progressive_center_crop(img, final_size, step_sizes=None):
    """
    Progressively center an image on its CoG, shrinking it to `final_size`.
    
    Parameters
    ----------
    img : 2D array
        Input image.
    final_size : int
        Size of the output square image (MxM).
    step_sizes : list of int, optional
        Sequence of intermediate crop sizes. 
        If None, automatically halves the image until reaching final_size.
    
    Returns
    -------
    img_cropped : 2D array
        Centered image of size final_size x final_size.
    cx, cy : int
        Final CoG coordinates relative to the original image.
    """
    current_img = img.copy()
    cx_total, cy_total = None, None
    
    # If step_sizes not provided, generate a decreasing sequence
    if step_sizes is None:
        sz = current_img.shape[0]
        step_sizes = []
        while sz > final_size:
            step_sizes.append(sz)
            sz = max(sz // 2, final_size)
    
    # Iterate over progressively smaller windows
    for sz in step_sizes:
        cx, cy = compute_cog(current_img, integer=True)
        half = sz // 2
        y_min = max(cy - half, 0)
        y_max = min(cy + half, current_img.shape[0])
        x_min = max(cx - half, 0)
        x_max = min(cx + half, current_img.shape[1])
        
        current_img = current_img[y_min:y_max, x_min:x_max]
        
        # Keep track of CoG relative to original image
        if cx_total is None:
            cx_total, cy_total = x_min, y_min
        else:
            cx_total += x_min
            cy_total += y_min

    # Final crop to exactly final_size if needed
    cx, cy = compute_cog(current_img, integer=True)
    half = final_size // 2
    y_min = max(cy - half, 0)
    y_max = min(cy + half, current_img.shape[0])
    x_min = max(cx - half, 0)
    x_max = min(cx + half, current_img.shape[1])

    cx_total += x_min + half
    cy_total += y_min + half

    # Go back to original image and crop one more time
    y_min = max(cy_total - half, 0)
    y_max = min(cy_total + half, img.shape[0])
    x_min = max(cx_total - half, 0)
    x_max = min(cx_total + half, img.shape[1])

    img_cropped = img[y_min:y_max, x_min:x_max]
    cx, cy = compute_cog(img_cropped, integer=True)

    cx_total += cx - half
    cy_total += cy - half

    y_min = max(cy_total - half, 0)
    y_max = min(cy_total + half, img.shape[0])
    x_min = max(cx_total - half, 0)
    x_max = min(cx_total + half, img.shape[1])

    img_cropped = img[y_min:y_max, x_min:x_max]

    return img_cropped, cx_total, cy_total
    


def compute_cog(img, integer=False, low=0.1):
    """
    Compute the center of gravity of an image.
    Threshold pixels lower than `low*np.max(img)`
    """
    x = np.arange(0,img.shape[1])
    y = np.arange(0,img.shape[0])
    X,Y = np.meshgrid(x,y)
    img_filtered = (img) - np.median(img)
    img_filtered = np.clip(img_filtered - low*np.max(img_filtered), 0, None)
    cx = np.sum(img_filtered*X)/np.sum(img_filtered)
    cy = np.sum(img_filtered*Y)/np.sum(img_filtered)
    if integer:
        cx = int(np.round(cx))
        cy = int(np.round(cy))
    return cx, cy

def center_cog(img, nx):
    """Center an image on its CoG"""
    cx,cy = compute_cog(img, integer=True)
    m_center = img[cy-nx//2:cy+nx//2,cx-nx//2:cx+nx//2]

    return m_center, cx, cy


def normalize(m):
    return m / np.sum(m)

def get_otf(psf):
    otf = np.abs(fftshift(fft2(fftshift(psf / np.sum(psf)))))
    return otf/np.max(otf)

def strehl_ratio(psf, sampling, is_sky):
    nx = psf.shape[0]
    x  = (np.arange(nx)-nx/2)*sampling/nx
    xx,yy = np.meshgrid(x,x)
    rr = np.sqrt(xx**2+yy**2)
    if is_sky:
        pup = np.where((rr<=0.5)*(rr>(0.27/2)), 1, 0)
    else:
        pup = np.where(rr<=0.5, 1, 0)
    pup_tf = fftshift(fft2(fftshift(pup))) / np.sum(pup)
    psf_diff = np.abs(pup_tf)**2
    otf_diff = get_otf(psf_diff)
    otf_exp = get_otf(psf)
    noise_filtering = (otf_diff>1e-3)
    sr = np.sum(otf_exp*noise_filtering) / np.sum(otf_diff)
    return sr, otf_diff


class PSF_Processing:
    def __init__(self, file_name, batch_size):
        self.file_name = file_name
        self.batch_start = 0
        self.batch_size = batch_size

        with h5py.File(file_name, "r") as file:
            
            self.is_closed_loop = True# file.attrs['Is_Closed_Loop']
            science_grp = file['Science']
            # self.target_name = science_grp.attrs['Target']
            self.elevation = science_grp.attrs['Elevation']
            self.time_stamps = science_grp.attrs['PSF_TimeStamps'][:]

            self.is_sky = True
            science_frames_dset = science_grp['Science_PSFs']
            self.exposure_time = science_frames_dset.attrs['Exposure_Time']
            self.fps = science_frames_dset.attrs['FPS']
            self.period = 1 / self.fps
            self.gain = science_frames_dset.attrs['Gain']
            self.sampling_calib = science_frames_dset.attrs['Sampling']

            calibration_grp = file['Calibration']
            self.actuators_in_dm_diameter = calibration_grp.attrs['Actuators_in_diameter']
            self.total_number_of_actuators = calibration_grp.attrs['Total_Number_Of_Actuators']
            self.total_number_of_controlled_modes = calibration_grp.attrs['Total_Number_Of_Controlled_Modes']
            self.SkyCalibPupilRatio = calibration_grp.attrs['SkyCalibPupilRatio']
            self.Diameter = calibration_grp.attrs['Diameter']
            self.Obstruction_ratio = calibration_grp.attrs['Obstruction_ratio']
            
            self.wvl_calib = calibration_grp.attrs['Science_Calibration_Wavelength']
            self.wvl_sky = science_frames_dset.attrs['Wavelength']
            self.instrument = Instrument(D=self.Diameter, 
                                         occ=self.Obstruction_ratio,
                                         res = self.sampling_calib * self.wvl_calib / self.Diameter)

            self.rad2arcsec = 206265

            self.nx = min(int(science_frames_dset.shape[-1]*0.9)//2*2, int((3 * self.sampling_calib / (self.wvl_sky / self.Diameter * self.rad2arcsec)) // 2) * 2)
            self.nx_cog = self.nx // 2
            self.cx = None
            self.cy = None

            

            self.dark = 0#science_grp['Dark'][:].squeeze().astype(np.float32)
            self.number_of_frames = file['Science']['Science_PSFs'].shape[0]

            self.cogs_x = np.zeros(science_frames_dset.shape[0])
            self.cogs_y = np.zeros(science_frames_dset.shape[0])

            self.iteration_times = []
            self.LoadData()
        

    def ComputeCenterOfGravity(self, display = False):
        
        frames_in = (self.science_frames - np.expand_dims(self.dark,0))
        self.frames_in = frames_in[:, self.cy-self.nx_cog//2:self.cy+self.nx_cog//2,self.cx-self.nx_cog//2:self.cx+self.nx_cog//2]

        self.xcog, self.ycog = ComputeWCoG(self.frames_in, self.sampling)

        if display:
            fig, ax = plt.subplots(1,2, figsize = (12,6))

            ax[0].plot(self.T, self.xcog, label = 'x-cog')
            ax[0].plot(self.T, self.ycog, label = 'y-cog')
            ax[0].set_title('Temporal evolution of the CoG')
            ax[0].set_xlabel('Time (s)')
            ax[0].set_ylabel('Position ($\\lambda/D$)')
            ax[0].legend()

            ax[1].hist(self.xcog, label = 'x-cog')
            ax[1].hist(self.ycog, label = 'y-cog')
            ax[1].set_title('Histogram of the CoG')
            ax[1].set_xlabel('Position ($\\lambda/D$)')
            ax[1].set_ylabel('Frequency of repetition')
            ax[1].legend()
            


    def ComputeCoGPSD(self, display = False):
        if self.xcog is None or self.ycog is None:
            self.ComputeCenterOfGravity()
        
        _, self.x_psd = GetSignalPSD(self.cogs_x, self.period)
        self.f, self.y_psd = GetSignalPSD(self.cogs_y, self.period)

        if display:
            fig, ax = plt.subplots()
            ax.loglog(self.f, self.x_psd, label = 'x-cog')
            ax.loglog(self.f, self.y_psd, label = 'y-cog')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('PSD ( $(\\lambda/D)^2/Hz$ )')
            ax.legend()

    def ComputePSFStatistics(self):
        self.ComputeCenterOfGravity(display=True)

        self.ComputeCoGPSD(display=True)

    def LoadData(self):
        with h5py.File(self.file_name, "r") as file:
            self.science_frames = file['Science']['Science_PSFs'][self.batch_start:self.batch_start + self.batch_size].astype(np.float32)
            self.time_stamp = self.time_stamps[self.batch_start + self.batch_size // 2]


    def ProcessDark(self):
        # self.dead_pix_map = dead_pixel_map(self.dark)
        # self.bkg_dp = filter_dead_pixel(self.dark, self.dead_pix_map)
        self.bkg_dp = self.dark

    def ProcessPSF(self, psf):
        temp = np.copy(psf)
        # img_dp = filter_dead_pixel(temp, self.dead_pix_map)
        # post_processed_img = filter_dead_pixel_full_img(img_dp - self.bkg_dp)
        post_processed_img = temp - self.dark
        test, cx,cy = progressive_center_crop(post_processed_img, self.nx, step_sizes=None)
        return test, cx,cy


    def SetSkyOrCalibContiditons(self):
        if self.is_sky:
            self.sampling = self.sampling_calib / self.SkyCalibPupilRatio * self.wvl_sky/self.wvl_calib
            self.instrument.occ = 0.3
            self.wvl = self.wvl_sky
        else:
            self.sampling = self.sampling_calib
            self.instrument.occ = 0
            self.wvl = self.wvl_calib

    def SetPSFModel(self):
        nb_act_lin = self.actuators_in_dm_diameter
        self.instrument.Nact = round(nb_act_lin * self.SkyCalibPupilRatio * np.sqrt(self.total_number_of_controlled_modes/self.total_number_of_actuators))

        if self.is_closed_loop:
            self.psfmodel = Psfao((self.nx,self.nx), system=self.instrument, samp=self.sampling)
            self.psfparam_guess = [0.09, 1e-4, 0.4, 0.5, 1, 0, 1.5]
            self.fixed = [False, ]*7
        else:
            self.psfmodel = Turbulent((self.nx,self.nx), system=self.instrument, samp=self.sampling)
            self.psfparam_guess = [0.09, 30]
            self.fixed = [False, ]*2


    def ProcessPSFBatch(self):

        self.long_exp = np.mean(self.science_frames, axis = 0)     
        self.long_exp, cx, cy = self.ProcessPSF(self.long_exp)

        if self.cx is None: 
            self.cx = cx
            self.cy = cy

        self.ComputeCenterOfGravity()

        self.cogs_x[self.batch_start:self.batch_start + self.batch_size] = self.xcog
        self.cogs_y[self.batch_start:self.batch_start + self.batch_size] = self.ycog

        self.jitter.append([np.std(self.xcog), np.std(self.ycog)])

        r0, sr_otf, sr_fit, psf_norm, psf_model, dxdy = self.FitPSFModel(self.long_exp)

        self.long_exp_r0 = r0
        self.long_exp_sr_otf = sr_otf
        self.long_exp_sr_fit = sr_fit
        self.long_exp_psf_norm = psf_norm
        self.long_exp_psf_model = psf_model

    def AnalyzeAllTheFile(self):
        print('#####################')
        print('Analysing PSFs')
        print('#####################')
        self.long_exp_r0_list = []
        self.long_exp_sr_otf_list = []
        self.long_exp_sr_fit_list = []
        self.long_exp_psf_norm_list = []
        self.long_exp_psf_model_list = []
        self.jitter = []

        while (self.batch_start + self.batch_size) <= self.number_of_frames:
            
            self.ProcessPSFBatch()
            self.batch_start += self.batch_size
            self.LoadData()
            self.long_exp_r0_list.append(self.long_exp_r0)
            self.long_exp_sr_otf_list.append(self.long_exp_sr_otf)
            self.long_exp_sr_fit_list.append(self.long_exp_sr_fit)
            self.long_exp_psf_norm_list.append(self.long_exp_psf_norm)
            self.long_exp_psf_model_list.append(self.long_exp_psf_model)
            print(f'{self.batch_start} out of {self.number_of_frames} frames processed')


        self.long_exp_r0_list = np.array(self.long_exp_r0_list)
        self.long_exp_sr_otf_list = np.array(self.long_exp_sr_otf_list)
        self.long_exp_sr_fit_list = np.array(self.long_exp_sr_fit_list)
        self.long_exp_psf_norm_list = np.array(self.long_exp_psf_norm_list)
        self.long_exp_psf_model_list = np.array(self.long_exp_psf_model_list)
        self.jitter = np.array(self.jitter)

        self.SaveAnalysis()


    def FitPSFModel(self, psf, display = False):
        ron = 0
        weights = 1/(gaussian_filter(np.abs(psf), 2)+ron**2)
        out = psffit(psf, self.psfmodel, self.psfparam_guess, weights=weights, fixed=self.fixed, max_nfev=30)
        otf_fit_avg = circavg(get_otf(out.psf), center=(self.nx//2,self.nx//2))

        psf_norm = (psf-out.flux_bck[1])/out.flux_bck[0]

        otf = get_otf(psf_norm)
        otf_avg = circavg(otf, center=(self.nx//2,self.nx//2))
        sr, otf_diff = strehl_ratio(psf_norm, self.sampling, self.is_sky)
        otf_diff_avg = circavg(otf_diff, center=(self.nx//2,self.nx//2))

        r0_zenith = out.x[0]/np.cos(np.pi/2-self.elevation*np.pi/180)**(3/5)
        r0_V0 = r0_zenith * (550e-9 / self.wvl) ** (6/5)
        seeing = self.rad2arcsec*550e-9/r0_V0

        sr_OTF = 100*sr
        sr_fit = 100*self.psfmodel.strehlOTF(out.x)

        if display:


            print('Strehl : %.1f %% (from OTF)'%(100*sr))
            print('Strehl : %.1f %% (from fit)'%(100*self.psfmodel.strehlOTF(out.x)))
            print('r0 @ 550 nm : %.2f cm'%(r0_V0 * 100))

            
            pix_scale = self.wvl/self.Diameter / self.sampling
            axis = np.linspace(-self.nx//2, self.nx//2, self.nx) * pix_scale * self.rad2arcsec

            def setplt(tab, cmap = 'Spectral_r', norm = LogNorm(vmin=1e-3, vmax=1)):
                maxi = np.max(psf_norm)
                im1 = plt.imshow(tab/maxi, norm=norm, cmap=cmap, extent=[axis[0],axis[-1],axis[0],axis[-1]])
                plt.colorbar(im1, fraction=0.046, pad=0.04)
                corr_zone = Circle([0,0], self.wvl/self.Diameter*self.instrument.Nact/2*self.rad2arcsec, fc='none', ec='k', ls=':')
                plt.gca().add_artist(corr_zone)
                plt.xlabel('[arcsec]')
                plt.ylabel('[arcsec]')
                hfov = 2
                plt.xlim(-hfov, hfov)
                plt.ylim(-hfov, hfov)

                

            plt.figure(1, figsize=(9,4))
            plt.clf()
            #plt.suptitle('Strehl = %.1f %%\nSeeing = %.1f\"'%(100*sr, seeing))
            plt.subplot(131)
            plt.title('data')
            setplt(psf_norm)

            plt.subplot(132)
            plt.title('fit')
            setplt(out.psf)

            plt.subplot(133)
            plt.title('fit - data')
            setplt(out.psf-psf_norm, cmap='RdBu', norm=SymLogNorm(1e-3, vmin=-1, vmax=1))


            
            plt.figure(2, figsize=(8,4))
            plt.clf()
            plt.suptitle(f'Strehl = {100*self.psfmodel.strehlOTF(out.x):.1f}%    Seeing = {seeing:.1f}"')

            plt.subplot(121)
            plt.title('PSF')
            plt.semilogy(circavg(self.psfmodel.psfDiffraction, center=(self.nx//2,self.nx//2)), label='diffraction', c='k')
            plt.semilogy(circavg(psf_norm, center=(self.nx//2,self.nx//2)), label='data')
            plt.semilogy(circavg(out.psf, center=(self.nx//2,self.nx//2)), label='fit')
            plt.axhline(out.flux_bck[1]/out.flux_bck[0], c='C1', ls='--', label='bck fit')
            plt.axvline(self.instrument.Nact/2*self.sampling, c='k', ls=':', label='AO')
            plt.grid()
            plt.xlim(0, 60)
            plt.ylim(1e-7, 0.1)
            plt.xlabel('Position [pix]')
            plt.legend()

            plt.subplot(122)
            plt.title('OTF')
            plt.loglog(otf_diff_avg, label='diffraction', c='k')
            plt.loglog(otf_avg, label='data')
            plt.loglog(otf_fit_avg, label='fit')
            plt.xlabel('Frequency [1/pix]')
            plt.ylim(1e-3,2)
            plt.xlim(right=self.nx//2)
            plt.grid()
            plt.legend()

            plt.tight_layout()

        return r0_V0, sr_OTF, sr_fit, psf_norm, out.psf, out.dxdy
    
    def SaveAnalysis(self):
        with h5py.File(self.file_name, "a") as file:
            def write_or_replace_group(grp, name):
                if name not in grp:
                    return grp.create_group(name)
                return grp[name]
            def write_or_replace(grp, name, data):
                if name in grp:
                    dset = grp[name]
                    # overwrite only if shape is compatible
                    if dset.shape == data.shape:
                        dset[:] = data
                    else:
                        del grp[name]
                        dset = grp.create_dataset(name, data=data)
                else:
                    dset = grp.create_dataset(name, data=data)


            sci_grp = file['Science']
            
            analysis_grp = write_or_replace_group(sci_grp, 'Analysis')
            analysis_grp_se = write_or_replace_group(analysis_grp, 'Short_Exposure')
            analysis_grp_le = write_or_replace_group(analysis_grp, 'Long_Exposure')


            write_or_replace(analysis_grp_se,'CoG-X', data = self.cogs_x)
            write_or_replace(analysis_grp_se,'CoG-Y', data = self.cogs_y)
            write_or_replace(analysis_grp_se,'Jitter', data = self.jitter)
            

            write_or_replace(analysis_grp_le,'r0', data = self.long_exp_r0_list)
            write_or_replace(analysis_grp_le,'sr_otf', data = self.long_exp_sr_otf_list)
            write_or_replace(analysis_grp_le,'sr_fit', data = self.long_exp_sr_fit_list)
            write_or_replace(analysis_grp_le,'psf_stack', data = self.long_exp_psf_norm_list)
            write_or_replace(analysis_grp_le,'psf_model', data = self.long_exp_psf_model_list)
            write_or_replace(analysis_grp_le, "Iteration_Times", np.array(self.iteration_times))
