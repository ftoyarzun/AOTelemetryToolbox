import numpy as np
import pylab as plt
import math
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import welch

import h5py



def RadialOrder(nModes):
    index = 2
    jump = 3
    radial_order_matrix = []
    while index < nModes:
        radial_order_matrix.append(index)
        index += jump
        jump += 1
    
    return np.array(radial_order_matrix)

def RadialOrderArray(nModes):
    index = 2
    number_of_modes_in_radial_order = 3
    index_matrix = []
    while index < nModes:
        local_index_matrix = []
        for ii in range(number_of_modes_in_radial_order):
            local_index_matrix.append(index)
            index += 1
            if index == nModes:
                break
        index_matrix.append(local_index_matrix)
        number_of_modes_in_radial_order +=1
    
    return index_matrix


def ComputeRadialOrderMean(Z_variance, radialOrderModes):
    return np.mean(Z_variance[radialOrderModes], axis = 0)

def ComputeRadialOrderMeans(Z_Series, radial_mode_arrays):
    return np.array([ComputeRadialOrderMean(Z_Series, modes) for modes in radial_mode_arrays])

def ZernikeTurbulenceVariance(r0, L0, D, n):
    radial_order_variance = []
    variance_n1 = 2.34e-2 * (D/r0)**(5/3) * (1 - 0.39 * (2 * np.pi * D / L0)**2 + 0.27 * (2 * np.pi * D / L0)**(7/3))
    
    radial_order_variance.append(variance_n1)

    for i in range(3,n+2):
        variance_n = 0.756 * (i+1) * math.gamma(i - 5/6)/math.gamma(i + 23/6) * (D/r0)**(5/3) * (1 - 0.38/((i - 11/6)*(i+23/6))*(2 * np.pi * D / L0)**2)
        radial_order_variance.append(variance_n)

    return np.array(radial_order_variance)

def ProjectPhaseSeries(phaseSeries, basis):
    phaseSeries_flatten = phaseSeries.reshape(phaseSeries.shape[0], -1)
    Z_modes = (basis.T @ phaseSeries_flatten.T).T

    return Z_modes

def NollMatrix(D, r0, n):
    return (D / r0) ** (5./3.) * 0.15337 * (n + 1)* math.gamma(14/3)*math.gamma(n - 5/6) / (math.gamma(17/6)**2 * math.gamma(n + 23/6))


def ComputeCoherenceTime(D_tau, dt):
    
    N, T = D_tau.shape
    
    tau_vec = np.zeros((N, 1))
    
    lags = np.arange(0, T) * dt
    
    for i in range(N):
        over_1_rad = np.where(D_tau[i] > 1)[0]
        if len(over_1_rad) == 0:
            tau = np.nan
        else:
            i1 = over_1_rad[0]
            if i1 == 0:
                tau = lags[0]
            else:
                # Interpolate for better precision
                f = interp1d(D_tau[i][i1 - 1:i1 + 1], lags[i1 - 1:i1 + 1], kind='linear')
                tau = f(1)
                
        tau_vec[i] = tau

    if np.sum(np.isnan(tau_vec)) > 5:
        print("Some intervals have larger coherence time than requested, increase the lookahead time!")
    return tau_vec[~np.isnan(tau_vec)]


def ComputeR0L0(Z_timeSeries, D, display = False):

    nModes = Z_timeSeries.shape[1]
    radial_mode_arrays = RadialOrderArray(nModes)
    radial_order = np.arange(2,2 + len(radial_mode_arrays))

    Z_variance = np.var(Z_timeSeries, axis = 0)
    measured_variances = ComputeRadialOrderMeans(Z_variance, radial_mode_arrays)

    scale = np.linspace(1,10,len(radial_mode_arrays))

    # Wrap the model to fix dm_psd
    model = lambda x, r0, L0: np.log(np.abs(ZernikeTurbulenceVariance(r0, L0, D = D, n = len(radial_mode_arrays)))) / scale

    # Fit only r0
    A_fit, _ = curve_fit(model, radial_order, np.log(measured_variances) / scale, p0=[0.1, 150], bounds=(0.001, np.inf))

    predicted_variances = ZernikeTurbulenceVariance(r0 = A_fit[0], L0 = A_fit[1], D = D, n = len(radial_mode_arrays))

    if display:
        plt.figure()
        plt.semilogy(np.array(measured_variances), label = 'DM Radial PSD')
        plt.semilogy(predicted_variances, label = 'Prediction PSD')
        plt.legend()

    return A_fit

def PhaseTemporalStructureFunction(delay, wind_speed, r0):
    return 6.88 * (wind_speed * delay / r0) ** (5/3)


def GetEstimatedAverageWindSpeedFittingStructureFunction(D_tau, dt, r0):
    N, T = D_tau.shape
    wind_speed_vec = np.zeros((N, 1))
    lags = np.arange(0, T) * dt

    model = lambda delay, wind_speed: PhaseTemporalStructureFunction(delay, wind_speed, r0 = r0)

    for i in range(N):
        A_fit, _ = curve_fit(model, lags, D_tau[i], p0=[10], bounds=(0.001, np.inf))
        wind_speed_vec[i] = A_fit[0]

    return np.array(wind_speed_vec)


def temporal_structure_function(images,diameter, max_lag):
    T = images.shape[0]
    D = np.zeros(max_lag + 1)

    for tau in range(max_lag + 1):
        diff = images[:T-tau] - images[tau:T]
        D[tau] = np.mean(diff**2) * diameter**2

    return D


## NEW WIND ESTIMATION

def loglog_derivative(f, psd):
    d1 = np.gradient(psd, f) * f / psd
    return d1

def find_min_derivative(f, d1):
    idx_min = np.argmin(d1)
    return f[idx_min], idx_min

def third_derivative(f, psd):

    d1 = loglog_derivative(f, psd)
    d2 = np.gradient(d1, f)
    d3 = np.gradient(d2, f)
    return d3

def last_zero_before_min(f, d3, idx_min):
    # sign changes indicate zero crossings
    sign = np.sign(d3)
    zero_crossings = np.where(np.diff(sign))[0]

    # only crossings before min
    zero_before = zero_crossings[zero_crossings < idx_min]

    if len(zero_before) == 0:
        return None, None

    idx = zero_before[-1]
    return f[idx], idx

def find_inertial_range_start(f, psd):
    d1 = loglog_derivative(f, psd)
    f_min, idx_min = find_min_derivative(f, d1)

    d3 = third_derivative(f, psd)
    f_zero, idx_zero = last_zero_before_min(f, d3, idx_min)

    return f_zero

def GetEstimatedAverageWindspeedZernikeTemporalCutoff(signal, period, leak, radial_mode_arrays, diameter, display = False):
    f, Zernike_psds = GetRadialOrderTemporalPSD(signal, period, radial_mode_arrays)
    optimize_mask = f > 0
    params = np.array(GetZernikeTemporalCutoffFrequency(f[optimize_mask], Zernike_psds[:, optimize_mask], period, leak))

    N, _ = Zernike_psds.shape

    # cutoffs = np.zeros(params.shape[1])
    # for i in range(N):
    #     model_psd = TotalTransferFunction(f, period, *params[:,i])
    #     cutoffs[i] = find_inertial_range_start(f, model_psd)

    # print(cutoffs)

    n_array = np.arange(2, 2 + N)
    wind_speed_estimation = diameter * np.sum(params[2,:].T * (n_array + 1)) / np.sum(0.3 * (n_array + 1)**2)
    effective_gain = np.mean(params[-4])
    effective_delay = np.mean(params[-2])
    #
    if display:
        
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, N))
        fig, ax = plt.subplots()
        
        for i in range(N):
            ax.loglog(f, Zernike_psds[i], color = colors[i], linestyle = '-')
            ax.loglog(f, TotalTransferFunction(f, period, *params[:,i]), color = colors[i], linestyle = '--')

    return wind_speed_estimation, effective_gain, effective_delay


def LowPassTransferFunction(f,omega, alpha):
    return 1 / (1 + (f/omega) ** (alpha))

def GetRadialOrderTemporalPSD(signal, period, radial_mode_arrays):
    f, Zernike_psds = GetSignalPSD(signal.T, period)
    Zernike_psds = ComputeRadialOrderMeans(Zernike_psds, radial_mode_arrays)
    return f, Zernike_psds

def AOTransferFunctions(f, period, ki, leak, NFramesDelay, noise):
    s = 2 * np.pi * 1j * f
    sT = s * period

    Hdm = (1 - np.exp(-sT)) / sT
    Hwfs = (1 - np.exp(-sT)) / sT
    Hdelay = np.exp(- NFramesDelay * sT)
    Hcontroller = ki / (leak - np.exp(-sT))
    Hnoise = np.abs(sT * 0 + noise)

    return Hdm, Hdelay, Hcontroller, Hwfs, Hnoise

def TotalTransferFunction(f, period, amp1,alpha, omega1, alpha1, ki, leak, Ndelay, noise):
    Hdm, Hdelay, Hcontroller, Hwfs, Hnoise = AOTransferFunctions(f + f[1], period, ki, leak, Ndelay, noise)

    signal_tf = np.abs((Hdm * Hdelay * Hcontroller)/(1 + Hwfs * Hdm * Hdelay * Hcontroller))**2
    atm_tf = amp1 * (f + f[1]) ** (-alpha) * LowPassTransferFunction(f, omega1, alpha1) * np.abs(Hwfs)**2 + Hnoise


    return signal_tf * atm_tf

def AtmosphereTransferFunction(f, period, amp, alpha, ki, leak, Ndelay, noise):
    Hdm, Hdelay, Hcontroller, Hwfs, Hnoise = AOTransferFunctions(f + f[1], period, ki, leak, Ndelay, noise)

    signal_tf = np.abs((Hdm * Hdelay * Hcontroller)/(1 + Hwfs * Hdm * Hdelay * Hcontroller))**2
    atm_tf = amp / ((f + f[1]) ** alpha) * np.abs(Hwfs)**2 + Hnoise

    return signal_tf * atm_tf



def GetZernikeTemporalCutoffFrequency(f, Zernike_psds, period, leak):

    N, _ = Zernike_psds.shape
    amplitudes = np.zeros((N,1))
    cutoffFrequencies = np.zeros((N,1))
    alphas = np.zeros((N,1))
    alphas1 = np.zeros((N,1))
    kis = np.zeros((N,1))
    Ndelays = np.zeros((N,1))
    noises = np.zeros((N,1))
    leaks = np.ones((N,1)) * leak


    scale = np.linspace(1,10,f.shape[0])



    model = lambda f,amp1,alpha, omega1, alpha1, ki, Ndelay, noise : np.log(TotalTransferFunction(f + 1e-8, period, amp1,alpha, omega1, alpha1, ki, leak, Ndelay, noise)) / scale

    lower = [0, 0.,    0,   5/3, 0.1, 1., 0]
    upper = [np.inf,1, 200, 23/3, 0.8, 3.5, 1]


    for i in range(N):
        A_fit, _ = curve_fit(model, f, np.log(Zernike_psds[i]) / scale, p0=[5e-1, 0.5, 5, 11/3, 0.44, 1.8, 0.5e-6], bounds=(lower, upper), maxfev=1500)
        amplitudes[i] = A_fit[0]
        alphas[i] = A_fit[1]
        cutoffFrequencies[i] = A_fit[2]
        alphas1[i] = A_fit[3]
        kis[i] = A_fit[4]
        Ndelays[i] = A_fit[5]
        noises[i] = A_fit[6]

    return amplitudes, alphas, cutoffFrequencies, alphas1, kis, leaks, Ndelays, noises

def GetAtmosphereDynamicParameters(f, psd, period, leak):

    scale = np.linspace(1,10,f.shape[0])


    model = lambda f,amp1, alpha1, ki, Ndelay, noise : np.log(AtmosphereTransferFunction(f + 1e-8, period, amp1, alpha1, ki, leak, Ndelay, noise)) / scale

    lower = [0,        6/3, 0.1, 0.5, 0]
    upper = [np.inf, 23/3, 0.8, 3.5, 1]

    A_fit, _ = curve_fit(model, f, np.log(psd) / scale, p0=[1.4, 11/3, 0.44, 1.8, 0.5e-6], bounds=(lower, upper))

    A_fit.insert(3, leak)
    return A_fit

def GetSignalPSD(signal, period):
    return welch(signal, 1/period, nperseg=1000)


class Atmosphere_Characterization:
    def __init__(self, file_name, batch_size, filter_TT):
        with h5py.File(file_name, "r") as file:
            self.file_name = file_name
            self.batch_start = 0
            self.batch_size = batch_size
            self.number_of_frames = file['WFS']['DM_commands'].shape[0]
            self.time_stamp = file['WFS']['DM_TimeStamps'][0]

            self.filter_TT = filter_TT

            self.gain = file['WFS'].attrs['Loop_Gain']
            self.wavelength = file['Calibration'].attrs["AO_Calibration_Wavelength"]
            self.freq = file['WFS'].attrs['Loop_Freq']
            self.period = 1 / self.freq
            self.loop_gain = file['WFS'].attrs['Loop_Gain']
            self.loop_leak = file['WFS'].attrs['Loop_Leak']
            


            self.dm_modes = file['Calibration']['DM_modes'][:] # / self.wavelength * 2 * np.pi
            # self.Z_fullRes = file['Calibration']['Z_full_resolution']
            self.M2C = file['Calibration']['M2C'][:]
            self.C2Z = file['Calibration']['C2Z'][:]
            self.Diameter = file['Calibration'].attrs['Diameter']

            self.nModes = 27

            #self.Z = self.Z_fullRes[:self.nModes]

            
        
        # self.Z_flatten = self.Z.reshape(self.Z.shape[0], -1)
        # self.Z_inv = np.linalg.pinv(self.Z_flatten, 0.05)

        self.iteration_times = []
        self.LoadData()


    def AnalyzeAllTheFile(self):
        print('#####################')
        print('Analysing AO Telemetry')
        print('#####################')
        self.r0_list = []
        self.tau0_list = []
        self.V0_list = []
        self.V0_Zernike_list = []
        self.effective_gain_list = []
        self.effective_delay_list = []

        while (self.batch_start + self.batch_size) <= self.number_of_frames:
            
            self.ComputeAtmospericParameters(display=False)
            self.ZernikeWindEstimation()
            self.batch_start += self.batch_size // 2
            self.LoadData()
            self.r0_list.append(self.r0 * 100)
            self.tau0_list.append(self.tau0 * 1000)
            self.V0_list.append(self.V0)
            self.V0_Zernike_list.append(self.zernike_wind_speed)
            self.effective_gain_list.append(self.effective_gain)
            self.effective_delay_list.append(self.effective_delay)
            print(f'{self.batch_start} out of {self.number_of_frames} frames processed')

        self.r0_list = np.array(self.r0_list)
        self.tau0_list = np.array(self.tau0_list)
        self.V0_list = np.array(self.V0_list)
        self.V0_Zernike_list = np.array(self.V0_Zernike_list)
        self.effective_gain_list = np.array(self.effective_gain_list)
        self.effective_delay_list = np.array(self.effective_delay_list)

        self.SaveAnalysis()

    def LoadData(self):
        with h5py.File(self.file_name, "r") as file:

            self.dm_commands = file['WFS']['DM_commands'][self.batch_start:self.batch_start + self.batch_size]
            # self.wfs_measurements = file['WFS']['WFS_measurements'][self.batch_start:self.batch_start + self.batch_size].squeeze()
            # self.wfs_coefs = self.wfs_measurements @ self.M2C.T
            # #POL
            # self.dm_commands = self.dm_commands[2:] - self.wfs_coefs[:-2]

            if self.filter_TT:
                TT_modes = np.linalg.pinv(self.M2C[:,:2])
                TT_proj = self.dm_commands @ TT_modes.T
                TT_commands = TT_proj @ self.M2C[:,:2].T
                self.dm_commands -= TT_commands
            
        
        self.dm_commands = self.dm_commands
        self.dm_commands -= self.dm_commands.mean(axis = 1, keepdims = True)

        self.full_dm_map = np.tensordot(self.dm_commands, self.dm_modes, axes=(1, 0))
        self.dm_flat = np.mean(self.full_dm_map, axis = 0)
        self.full_dm_map -= np.expand_dims(self.dm_flat,0)
        self.full_dm_map *= 2 * np.pi / self.wavelength

        #self.dm_Z_modes = ProjectPhaseSeries(self.full_dm_map, self.Z_inv)
        self.dm_Z_modes = self.dm_commands @ self.C2Z.T
        self.iteration_times.append(self.time_stamp)
        self.time_stamp += self.batch_size//2 *self.period

    def RecreateDMCommandsFromWFS(self, startFromMode):
        dm_commands_list = np.zeros_like(self.wfs_measurements)
        dm_commands = np.zeros_like(self.wfs_measurements[0])


        for i, signal in enumerate(self.wfs_measurements):
            dm_commands = dm_commands * self.loop_leak - signal * self.loop_gain
            dm_commands_list[i] = dm_commands

        dm_commands_list = dm_commands_list.squeeze()
        dm_commands_filtered = dm_commands_list[:, startFromMode:] @ self.M2C[:,startFromMode:].T


        dm_commands_filtered = dm_commands_filtered
        dm_commands_filtered -= dm_commands_filtered.mean(axis = 0, keepdims = True)

        return dm_commands_filtered


    def ComputeR0(self,display = False):
        self.r0, self.L0 = ComputeR0L0(self.dm_Z_modes, self.Diameter, display = display)

    def ComputeTau0(self, display = False):
        estimated_lookahead = 3
        self.D_tau = temporal_structure_function(self.full_dm_map,self.Diameter, estimated_lookahead)
        while self.D_tau.max() < 2:
            estimated_lookahead += 10
            self.D_tau = temporal_structure_function(self.full_dm_map,self.Diameter, estimated_lookahead)

        self.tau0 = ComputeCoherenceTime(np.array(self.D_tau)[None,:], self.period)

        if display:
            plt.figure()
            T = np.arange(0,self.D_tau.shape[0]) * self.period
            plt.plot(T, self.D_tau)
            plt.plot([0, T.max()], [1, 1], 'k:')

    def ComputeV0(self, display = False):
        self.V0 = GetEstimatedAverageWindSpeedFittingStructureFunction(np.array([self.D_tau]), self.period, self.r0)[0]

        if display:
            plt.figure()
            T = np.arange(0,self.D_tau.shape[0]) * self.period
            plt.plot(T, self.D_tau)
            plt.plot(T, PhaseTemporalStructureFunction(T, self.V0, self.r0))

    def ComputeAtmospericParameters(self, display = False):
        self.ComputeR0(display=display)
        
        self.ComputeTau0(display=display)
        
        self.ComputeV0(display = display)
        

        if display:
            print(f'r0 = {self.r0 * 100:.1f} cm, L0 = {self.L0:.1f} m')
            print(f'tau0 = {np.mean(self.tau0) * 1000:.2f} ms')
            print(f'V0 = {np.mean(self.V0):.1f} m/s')
    
    def ZernikeWindEstimation(self):

        radial_mode_arrays = RadialOrderArray(self.nModes)

        self.zernike_wind_speed, self.effective_gain, self.effective_delay = GetEstimatedAverageWindspeedZernikeTemporalCutoff(self.dm_Z_modes, self.period, self.loop_leak, radial_mode_arrays, self.Diameter, display = False)

        
    def SaveAnalysis(self):
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

        with h5py.File(self.file_name, "a") as file:
            wfs_grp = file['WFS']
            if "Analysis" not in wfs_grp:
                analysis_grp = file['WFS'].create_group('Analysis')
            else:
                analysis_grp = wfs_grp['Analysis']

            write_or_replace(analysis_grp, "r0",   self.r0_list)
            write_or_replace(analysis_grp, "tau0", self.tau0_list)
            write_or_replace(analysis_grp, "V0",   self.V0_list)
            write_or_replace(analysis_grp, "Iteration_Times", np.array(self.iteration_times))

            write_or_replace(analysis_grp, "V0_Zernike",   self.V0_Zernike_list)
            write_or_replace(analysis_grp, "Effective_Gain", self.effective_gain_list)
            write_or_replace(analysis_grp, "Measured_Loop_Delay",   self.effective_delay_list)

            analysis_grp['r0'].attrs['Units'] = 'cm'
            analysis_grp['tau0'].attrs['Units'] = 'ms'
            analysis_grp['V0'].attrs['Units'] = 'm/s'