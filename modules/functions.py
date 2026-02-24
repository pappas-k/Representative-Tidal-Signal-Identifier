import uptide
import numpy as np
from statistics import mean, stdev, median
import datetime
from scipy.interpolate import interp1d
import math
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import stats

def deg_to_dms(deg, pretty_print=None, ndp=4):
    """Convert from decimal degrees to degrees, minutes, seconds."""

    m, s = divmod(abs(deg)*3600, 60)
    d, m = divmod(m, 60)
    if deg < 0:
        d = -d
    d, m = int(d), int(m)

    if pretty_print:
        if pretty_print=='latitude':
            hemi = 'N' if d>=0 else 'S'
        elif pretty_print=='longitude':
            hemi = 'E' if d>=0 else 'W'
        else:
            hemi = '?'
        return '{d:d}° {m:d}′ {s:.{ndp:d}f}″ {hemi:1s}'.format(
                    d=abs(d), m=m, s=s, hemi=hemi, ndp=ndp)
    return d, m, s




def timestamp_to_datetime(timestamp):
    dt_object = datetime.datetime(2002, 1, 1, 0, 0, 0)  + datetime.timedelta(seconds=timestamp)
    return  dt_object

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

def relative_difference(value1, value2):
    rel_diff= ((value1-value2)/value2)*100
    return rel_diff

def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    a=int(h) * 3600 + int(m) * 60 + int(s)
    return a

def iqr(data):
     return np.percentile(data, 75) - np.percentile(data, 25)

def rmse_reconstructed_vs_recorded_data():
    print('---------------------------------------------------------')
    new_eta_2, new_eta_4, new_eta_8, new_eta_12= points_interpolation()
    constituents=[2,4,8,12]
    rmse_list = [rmse(new_eta_2,eta), rmse(new_eta_4,eta), rmse(new_eta_8,eta), rmse(new_eta_12,eta) ]
    nrmse_list = [rmse(new_eta_2,eta)/ stdev(new_eta_2), rmse(new_eta_4,eta)/ stdev(new_eta_4), rmse(new_eta_8,eta)/ stdev(new_eta_8), rmse(new_eta_12,eta)/ stdev(new_eta_12)]
    pd_rmse= pd.DataFrame({"CONSTITUENTS": constituents, "RMSE": rmse_list, "NRMSE": nrmse_list })
    nrmse_2 = rmse(new_eta_2, eta) / stdev(new_eta_2)
    nrmse_4 = rmse(new_eta_4, eta) / stdev(new_eta_4)
    nrmse_8 = rmse(new_eta_8, eta) / stdev(new_eta_8)
    nrmse_12 = rmse(new_eta_12,eta)/ stdev(new_eta_12)
    print(pd_rmse)
    return   nrmse_12


def extract_constituents_from_tidegauge_file(tidegauge_file='inputs/Mumbles_20010101_20010131.csv', start_date=datetime.datetime(2001, 1, 1, 0, 0, 0)):
    file = open(tidegauge_file)
    print(file)
    numline = len(file.readlines()) #number of rows of the tidegauge_file
  
    t = np.arange(0, (numline - 2) * 15 * 60, 15 * 60)
    eta = np.loadtxt(tidegauge_file, skiprows=2, usecols=(11,), dtype=float, delimiter=',')
    QC_flag = np.loadtxt(tidegauge_file, skiprows=2, usecols=(12,), dtype=str, delimiter=',') #quality control flag of data, if non empty data entries must be skipped
    #print("ETA=",eta)
    #print("QC=",QC_flag)
    #print("------------------------------------------------------")
    pd0 = pd.DataFrame({"Time": t, "Elevation": eta, "QC flag": QC_flag  })
    # filter all rows for which the elevation has errors, (drop negative-wrong values)
    pd0_filtered = pd0[(~pd0['QC flag'].str.contains('N')) & (~pd0['QC flag'].str.contains('M')) & (~pd0['QC flag'].str.contains('T')) ]
    pd0_filtered = pd0_filtered[pd0_filtered["Elevation"] > -15]
    pd0_filtered = pd0_filtered[pd0_filtered["Elevation"] < 15]
    #print(pd0)
    #print(pd0_filtered)

    t=pd0_filtered['Time'].values
    eta = pd0_filtered['Elevation'].values

    #print("Availability of signal over a nodal cycle (%):", int(len(eta)*100 / ((int(18.61 * 365.25 * 24 * 60 * 60 / 900)))))

  
    constituents = ['Q1', 'O1', 'P1', 'S1', 'K1', 'J1', 'M1', '2N2', 'MU2', 'N2', 'NU2', 'M2', 'L2', 'T2', 'S2', 'K2',
                     'LAMBDA2', 'EPS2', 'R2', 'ETA2', 'MSN2', 'MNS2', '3M2S2', '2SM2', 'MKS2', 'MK3', 'MO3', 'MS4',
                     'MN4', 'N4', 'M4', 'S4', '2MK6', '2MS6','M3', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12']

    tide = uptide.Tides(constituents)
    tide.set_initial_time(start_date)
    thetis_amplitudes, thetis_phases = uptide.analysis.harmonic_analysis(tide, eta, t)
    #thetis_phases=np.degrees(thetis_phases)
    #thetis_phases = np.remainder(thetis_phases,2*math.pi)*360/(2*math.pi)
    #thetis_phases = np.remainder(thetis_phases, 2*math.pi) * 360 / (2 * math.pi)
    #print(constituents,thetis_amplitudes,thetis_phases)
    pd1 = pd.DataFrame({"Constituents": constituents,"Amplitude":thetis_amplitudes , "Phase": thetis_phases, })
    pd1 = pd1.sort_values(by=['Amplitude'], ignore_index=True, ascending=False)  # rearange dataframe in descending order of amplitude
    print(pd1)
    #print('Minimum Rayleigh Period=', tide.get_minimum_Rayleigh_period() / 86400.)
    #print("Constituents to be resolved in ONE year of data=", uptide.select_constituents(constituents, 365 * 86400))
    #print(f"Mean eta = {mean(eta)}")
    eta = eta - mean(eta)
    return pd1, t , eta


def signal_reconstruction(dataset,
                          dt=108,
                          constituents=["M2", "S2"],
                          signal_duration=365.25 * 24 * 3600,
                          start_date=datetime.datetime(2002, 1, 1, 0, 0, 0),
                          time_series_start_time=0):
    #     Create a time series
    time_series = np.arange(time_series_start_time, signal_duration, dt) #time_series_start_time initially was 0, now we put the timestamp we want or not?

    #     Extract amplitudes and phases at location
    #amplitudes, phases = extract_constituent_data(dataset, location=location, constituents=constituents)
    amplitudes, phases = dataset['Amplitude'].values, dataset['Phase'].values
    amplitudes, phases= amplitudes[:len(constituents)], phases[:len(constituents)]
    # Conduct signal reconstruction
    tide = uptide.Tides(constituents)
    tide.set_initial_time(start_date)
    tide_elevs = tide.from_amplitude_phase(amplitudes, phases, time_series)

    return np.column_stack((time_series, tide_elevs))


def plot_signals(t,eta,signal_2, signal_4, signal_8,signal_12,days):
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 40})
    rc('text', usetex=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.figure(figsize=(10, 4))
    #plt.plot(t, eta, label='Gauge data', lw=0.5)
    plt.plot(t/3600, eta, 'bo', ms=3.0, label='Gauge data' )
    plt.plot(signal_12[:, 0] / 3600, signal_12[:, 1], label='12 Cons', lw=1.2)
    plt.plot(signal_8[:, 0] / 3600, signal_8[:, 1], label='8 Cons', lw=1.2)
    plt.plot(signal_4[:, 0] / 3600, signal_4[:, 1], label='4 Cons', lw=1.2)
    plt.plot(signal_2[:, 0]/3600, signal_2[:, 1], label='2 Cons', lw=1.2)
    #plt.plot(signal_4[:, 0]/3600, signal_4[:, 1], label='4 Cons.', lw=0.9)
    #plt.plot(signal_2[:, 0]/3600, signal_2[:, 1], label='2 Cons.', lw=0.9)
    #plt.plot(t/3600, eta-signal_12[:, 1][len(t)], label='residual')
    plt.legend(ncol=2, loc="upper left", prop={"size":33.5})
    plt.ylabel("$\eta$ (m)")
    plt.xlabel("$t$ (hrs) ")
    plt.ylim([-7, 16])
    plt.yticks(np.arange(-6, 8, step=2))
    plt.xlim([0, days * 24  ])
    plt.xticks(np.arange(0,days * 24 +1,2))
    #plt.xlim([0,10 * 24 * 3600])
    #plt.ylim([-10, 10])
    # plt.xlim([0,27.6 * 24 * 3600])
    #
    props = dict(boxstyle='round', facecolor='none', alpha=0.25)
    plt.annotate("Location: Avonmouth \n Lat/Lon: 51.51089, -2.71497 \n Start Date: 01/01/2002 00:00:00", xy=(1, 1), xytext=(-460, -26), fontsize=33.5,
                 xycoords='axes fraction', textcoords='offset points',
                 bbox=props,
                 horizontalalignment='left', verticalalignment='top')
    plt.annotate("(a)",
                 xy=(0, 1),
                 xytext=(-115, +50),
                 xycoords='axes fraction', textcoords='offset points',
                  fontsize=45,
                 horizontalalignment='left', verticalalignment='top')
    plt.show()


def plot_signals_with_residual(days):
    fig, axs = plt.subplots(2)

    #fig.suptitle('Vertically stacked subplots')
    axs[0].plot(t/3600, eta, 'bo', ms=1.2, label='Gauge data')

    axs[0].plot(signal_12[:, 0]/3600, signal_12[:, 1], "r", label='12 Cons.', lw=0.9)
    axs[0].plot(signal_8[:, 0] / 3600 , signal_30[:, 1], label='8 Cons.', lw=0.9)
    axs[0].plot(signal_4[:, 0] / 3600, signal_4[:, 1] , label='4 Cons.', lw=0.9)
    axs[0].plot(signal_2[:, 0] / 3600, signal_2[:, 1], label='2 Cons.', lw=0.9)
    axs[0].plot(t/3600, eta, "g", label='Gauge data', lw=0.9)
    axs[0].set_title("Elevation")
    axs[0].set_xlim([0, days * 24])
    axs[0].legend(loc="upper right")

    #Residual plot
    new_eta_2 = np.interp(t, signal_2[:, 0], signal_2[:, 1])
    new_eta_4 = np.interp(t, signal_4[:, 0], signal_4[:, 1])
    new_eta_8 = np.interp(t, signal_8[:, 0], signal_8[:, 1])
    new_eta_12 = np.interp(t, signal_12[:, 0], signal_12[:, 1])
    residual_2 = eta - new_eta_2
    residual_4 = eta - new_eta_4
    residual_8 = eta - new_eta_8
    residual_12 = eta - new_eta_12
    axs[1].set_title("Residual")
    axs[1].plot(t / 3600, residual_2, label='Residual 2 cons ')
    axs[1].plot(t / 3600, residual_4, label='Residual 4 cons ')
    axs[1].plot(t / 3600, residual_8, label='Residual 8 cons')
    axs[1].plot(t / 3600, residual_12, label='Residual 12 cons')
    axs[1].set_xlim([0, days * 24])
    axs[1].legend(loc="upper right")
    axs[1].set_title("Residual")

    plt.xlabel("Time [h]")
    plt.show()



def points_interpolation():
    new_eta_2 = np.interp(t, signal_2[:, 0], signal_2[:, 1])
    new_eta_4 = np.interp(t, signal_4[:, 0], signal_4[:, 1])
    new_eta_8 = np.interp(t, signal_8[:, 0], signal_8[:, 1])
    new_eta_12 = np.interp(t, signal_12[:, 0], signal_12[:, 1])

    #plt.plot(t, eta, 'bo', ms=1.2, label='Gauge data')
    #plt.plot(t, new_eta_2, 'ro', ms=1, label='Interpolated points from reconstructed signal 2 constituents')
    #plt.plot(t, new_eta_4, 'yo', ms=1, label='4')
    #plt.plot(t, new_eta_8, 'go', ms=1, label='8')
    #plt.plot(t, new_eta_12, 'mo', ms=1, label='12')
    #plt.xlim([1422000, 30 * 24 * 3600])
    #plt.legend(loc="lower right")
    #plt.xlim([0, 100 * 24 * 3600])
    #plt.show()
    return new_eta_2, new_eta_4, new_eta_8, new_eta_12



def find_tidal_peaks( rel_times, tide_elevs, peak_type):
    """"
    Function which finds HW or LW peaks and their locations. Averages every two as per the methodology from NOAA,
    rel_times is the time in seconds that the tidal elevations tide_elevs occurs. peak_type = HW for high water,
    and = LW for low water. rel_times is the time in s.
    rel_times:
    """
    from scipy import signal
    # if peak_type is low water (LW), will make tidal elevations negative to fine 'peaks' using negative multiplier
    if peak_type == 'LW':
        multiplier = -1
    # if peak_type is high water (HW), will use original elevation data i.e. multiply by 1
    else:
        multiplier = 1
    # determine index at which the peaks occur - multiplier determines if HW (1) or LW (-1)
    peak_idx = signal.find_peaks(tide_elevs * multiplier)[0]
    # use peak index to determine real times (datetime objects) at which peaks occur
    #peak_real_times = real_times[peak_idx]

    peak_rel_times = rel_times[peak_idx]
    # use peak index to determine tidal elevations at which peaks occur
    peak_elevs = tide_elevs[peak_idx]
    # print('----------------------------------------------')
    # print('Peak Elevations (m)=', peak_elevs)
    # print('Number of Peaks =', len(peak_elevs))
    # print('Peak Relatives Times (s)=', peak_rel_times)
    # print('----------------------------------------------')
    return peak_rel_times, peak_elevs


def tidal_ranges_from_peaks(peak_real_times_HW, peak_real_times_LW, peak_elevs_HW, peak_elevs_LW):
    """"
    Function which finds tidal ranges based on peaks.
    """
    if peak_real_times_HW[0] < peak_real_times_LW[0]:
        #print('HIGH WATER OCCURS FIRST')
        #print('TIME OF HIGH WATER PEAK =', peak_real_times_HW[0])
        try:
            tidal_ranges_from_HW = peak_elevs_HW - peak_elevs_LW
            tidal_ranges_from_LW = abs(peak_elevs_LW[:-1] - peak_elevs_HW[1:])
        except:
            tidal_ranges_from_HW = peak_elevs_HW[:-1] - peak_elevs_LW         # remove last element to match list of peaks and times
            tidal_ranges_from_LW = abs(peak_elevs_LW[:] - peak_elevs_HW[1:])
        tidal_ranges_all = []
        rel_times_all = []
        # Since HW occurs first we append to a list he HW value and the LW one and so on
        for i in range(0, len(tidal_ranges_from_LW)):
            tidal_ranges_all.append(tidal_ranges_from_HW[i])
            tidal_ranges_all.append(tidal_ranges_from_LW[i])
            rel_times_all.append(peak_real_times_HW[i])
            rel_times_all.append(peak_real_times_LW[i])
    elif peak_real_times_HW[0] > peak_real_times_LW[0]:
        #print('LOW WATER OCCURS FIRST')
        #print('TIME OF LOW WATER PEAK =', peak_real_times_LW[0])
        try:
            tidal_ranges_from_LW = abs(peak_elevs_LW[:] - peak_elevs_HW[:])
            tidal_ranges_from_HW = peak_elevs_HW[:-1] - peak_elevs_LW[1:]
        except:
            tidal_ranges_from_LW = abs(peak_elevs_LW[:-1] - peak_elevs_HW[:])
            tidal_ranges_from_HW = peak_elevs_HW[:] - peak_elevs_LW[1:]
        tidal_ranges_all = []
        rel_times_all = []
        for i in range(0, len(tidal_ranges_from_HW)):
            tidal_ranges_all.append(tidal_ranges_from_LW[i])
            tidal_ranges_all.append(tidal_ranges_from_HW[i])
            rel_times_all.append(peak_real_times_LW[i])
            rel_times_all.append(peak_real_times_HW[i])

    return tidal_ranges_all, rel_times_all





def PE(signal):
    return ((1021 * 9.81 / (signal[:,0][-1]-signal[:,0][0])) * np.trapz(signal[:, 1] ** 2, signal[:, 0]))/3.6e+3 #in Wh/m^2
    #return (1021 * 9.81 /(x*12.42*60*60)) * np.trapz(signal[:, 1] ** 2, signal[:, 0])



def Hm0(signal):
    return 4 * np.std(signal[:,1])





def metric_1_ranges(signal):
    metric_1_all = []
    metric_1_time_step =[]
    metric_1_rel_times = []
    P50_contribution=[]
    IQR_contribution=[]
    IQR_all=[]
    P50_all=[]
    rel_time, tide_elevs = signal[:, 0], signal[:, 1]
    peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
    peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
    tidal_ranges_nodal, rel_times_nodal = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW,peak_elevs_HW, peak_elevs_LW)
    #print("mean tidal range over a nodal cycle:", mean(tidal_ranges_nodal))
    IQR_nodal = np.percentile(tidal_ranges_nodal, 75) - np.percentile(tidal_ranges_nodal, 25)
    P50_nodal = np.percentile(tidal_ranges_nodal, 50)
    #print("IQR nodal", IQR_nodal)
    #print("P50 nodal", P50_nodal)
    for j in np.arange(0, 13134 - 57, 1):                    # 18.61*365.25*24/12.42 = 13134   the number of 12.42 hrs cycles over the nodal period
        i = j * 414                                          # 12.42*60*60/108 = 414  the number of points per cycle of 12.42 hrs

        rel_time, tide_elevs = signal[:, 0][ i:(414 * 58 + i)], signal[:, 1][ i:(414  * 58 + i)]
        peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
        peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
        tidal_ranges, rel_times = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW)

        P50 = np.percentile(tidal_ranges, 50)
        IQR = np.percentile(tidal_ranges, 75) - np.percentile(tidal_ranges, 25)
        IQR_all.append(IQR)
        P50_all.append(P50)
        metric_1 = 0.5* rmse(P50,P50_nodal) + 0.5 * rmse(IQR, IQR_nodal)
        metric_1_all.append(metric_1)
        metric_1_time_step.append(i)
        metric_1_rel_times.append(rel_time[0])
        P50_contribution.append(rmse(P50, P50_nodal) * 100 * 0.5 / metric_1)
        IQR_contribution.append(rmse(IQR, IQR_nodal) * 100 * 0.5 / metric_1)
        #print(" lunar month ", j)
        #print("P50 contribution:", rmse(P50, P50_nodal) * 100 * 0.5 / metric_1)
        #print("IQR contribution:", rmse(IQR, IQR_nodal) * 100 * 0.5 / metric_1)
    #print(IQR_all)

    # time_series = pd.Series(P50_contribution)
    # time_series.plot(style='k-')
    # plt.title('Original Series')
    # plt.show()

    #print(P50_all)
    # print("min outlier P50", np.percentile(P50_contribution,25)  - 1.5*(np.percentile(P50_contribution,75)-np.percentile(P50_contribution,25))    )
    # print("max outlier P50", np.percentile(P50_contribution, 75) + 1.5*(np.percentile(P50_contribution, 75) - np.percentile(P50_contribution, 25))   )
    # print("min outlier IQR", np.percentile(IQR_contribution, 25) - 1.5 * (np.percentile(IQR_contribution, 75) - np.percentile(IQR_contribution, 25)))
    # print("max outlier IQR", np.percentile(IQR_contribution, 75) + 1.5 * (np.percentile(IQR_contribution, 75) - np.percentile(IQR_contribution, 25)))
    # print("average constibution of P50", mean(P50_contribution))
    # print("average constibution of IQR", mean(IQR_contribution))
    # print("max contribution of P50:", max(P50_contribution))
    # print("min contribution of P50:", min(P50_contribution))

    # pd_1=pd.DataFrame({"rmse(P50) % :":P50_contribution, "rmse(IQR(H)) % :":IQR_contribution})     #We compute how much rmse(Hm0) and rmse(IQR(H)) contibute to the metric
    # print(pd_1)
    IQR_all_percentages= (IQR_all-IQR_nodal)*100/IQR_nodal
    P50_all_percentages=(P50_all-P50_nodal)*100/P50_nodal
    pd_metric_1=pd.DataFrame({"TIME STEP":metric_1_time_step, "START TIME":metric_1_rel_times, "METRIC_1":metric_1_all, "IQR":IQR_all, "IQR %":IQR_all_percentages, "P50": P50_all, "P50 %":P50_all_percentages })
    a=pd_metric_1.sort_values(["METRIC_1"])
    #print("DATAFRAME METRIC 1:\n",pd_metric_1.sort_values(["METRIC_1"]).to_string())
    #print(a[0:1000].to_string())
    return pd_metric_1


def metric_2_ranges(pd_metric_1, signal):
    Hm0_lunar_all=[]
    metric_2_time_step=[]
    metric_2_rel_times = []
    metric_2_all = []

    Hm0_nodal = Hm0(signal)
    #We take the values within the 5th percentile of metric 1:

    pd_metric_1_5th_percentile= pd_metric_1[pd_metric_1['METRIC_1'] <= np.percentile(pd_metric_1['METRIC_1'], 5)].reset_index()
    #pd_metric_1_5th_percentile = pd_metric_1[pd_metric_1['METRIC_1'] <= np.percentile(pd_metric_1, 5)].reset_index()
    #print("DATAFRAME METRIC 1 5th percentile:\n",pd_metric_1_5th_percentile.to_string())

    for i in pd_metric_1_5th_percentile["TIME STEP"]:
        Hm0_lunar=Hm0(signal[i:(414 * 58 + i)])
        Hm0_lunar_all.append(Hm0_lunar)

        metric_2 = rmse(Hm0_lunar, Hm0_nodal)
        metric_2_all.append(metric_2)

        metric_2_time_step.append(i)
        metric_2_rel_times.append(i*108)

    pd_metric_2 = pd.DataFrame({"TIME STEP": metric_2_time_step, "START TIME":metric_2_rel_times, "HM0 LUNAR": Hm0_lunar_all, "METRIC_2": metric_2_all})
    pd_metric_2=pd_metric_2.sort_values(["METRIC_2"])
    #print("HM0 NODAL=", Hm0_nodal)
    #print("DATAFRAME METRIC 2:\n", pd_metric_2)
    #print("DATAFRAME METRIC 2:\n", pd_metric_2.sort_values(["METRIC_2"]))
    return pd_metric_2



def metric_1_energy(signal):
    metric_1_all = []
    metric_1_time_step = []
    metric_1_all_times = []
    Emax_all=[]
    rho=1021
    grav= 9.81

    rel_time, tide_elevs = signal[:, 0], signal[:, 1]
    peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
    peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
    tidal_ranges_nodal, rel_times_nodal = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW,peak_elevs_HW, peak_elevs_LW)

    emax_nodal= 0.5*rho*grav*np.square(tidal_ranges_nodal)
    Emax_nodal=np.sum(emax_nodal)

    IQR_nodal = np.percentile(emax_nodal, 75) - np.percentile(emax_nodal, 25)
    P50_nodal = np.percentile(emax_nodal, 50)

    for j in np.arange(0, 13134-57, 1):         # 18.61*365.25*24/12.42 = 13134   the number of 12.42 hrs cycles over the nodal period,
        i = j * 414                             # 12.42*60*60/108 = 414           the number of points per cycle of 12.42 hrs

        rel_time, tide_elevs = signal[:, 0][i:(414 * 57 + i)], signal[:, 1][ i:(414  * 57 + i)]
        peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
        peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
        tidal_ranges, rel_times = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW)

        emax = 0.5 * rho * grav * np.square(tidal_ranges)
        Emax_lunar = np.sum(emax)
        Emax_all.append(Emax_lunar)

        P50 = np.percentile(emax, 50)
        IQR = np.percentile(emax, 75) - np.percentile(emax, 25)

        metric_1 = 0.5* rmse(P50,P50_nodal) + 0.5 * rmse(IQR, IQR_nodal)

        metric_1_all.append(metric_1)
        metric_1_time_step.append(i)
        metric_1_all_times.append(rel_time[0])
    pd_metric_1=pd.DataFrame({"TIME STEP":metric_1_time_step, "START TIME":metric_1_all_times, "METRIC_1":metric_1_all})
    #print("DATAFRAME METRIC 1:\n", pd_metric_1)
    return pd_metric_1


def metric_2_energy(pd_metric_1, signal):
    PE_lunar_all=[]
    metric_2_time_step=[]
    metric_2_all = []
    metric_2_rel_times = []

    PE_nodal=PE(signal)

    pd_metric_1_5th_percentile = pd_metric_1[pd_metric_1['METRIC_1'] <= np.percentile(pd_metric_1['METRIC_1'], 5)].reset_index()
    #pd_metric_1_5th_percentile = pd_metric_1[pd_metric_1['METRIC_1'] <= np.percentile(pd_metric_1, 5)].reset_index()
    #print("DATAFRAME METRIC 1 5th percentile:\n", pd_metric_1_5th_percentile)
    for i in pd_metric_1_5th_percentile["TIME STEP"]:
        PE_lunar=PE(signal[i:(414 * 58 + i)])
        PE_lunar_all.append(PE_lunar)

        metric_2 = rmse(PE_lunar, PE_nodal)

        metric_2_all.append(metric_2)
        metric_2_time_step.append(i)
        metric_2_rel_times.append(i * 108)

    pd_metric_2 = pd.DataFrame({"TIME STEP": metric_2_time_step, "START TIME":metric_2_rel_times, "PE LUNAR": PE_lunar_all, "METRIC_2": metric_2_all})
    pd_metric_2=pd_metric_2.sort_values(["METRIC_2"])
    print("PE nodal", PE_nodal)
    print("DATAFRAME METRIC 2:\n", pd_metric_2)
    print("DATAFRAME METRIC 2:\n", pd_metric_2.sort_values(["METRIC_2"]))
    return pd_metric_2



def Emax_all_years(signal):
    "Function that calculates the annual Emax over 18 years and the average yearly Emax over the nodal cycle of 18.61 years"
    Emax_times=[]        #list of the start time of each lunar months (after 12.42hrs) to caclulate Emax for a lunar month
    Emax_year_all = []
    rho = 1021
    grav = 9.81

    rel_time, tide_elevs = signal[:, 0], signal[:, 1]
    peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
    peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
    tidal_ranges_nodal, rel_times_nodal = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW)

    emax_nodal = 0.5 * rho * grav * np.square(tidal_ranges_nodal)
    Emax_nodal = np.sum(emax_nodal)
    Emax_nodal_year_average = (Emax_nodal / 18.61)/3.6e+6  # 18.61*365.25/29.53

    Emax_nodal_median = np.percentile(emax_nodal, 50)
   # print("Median Emax over nodal cycle", Emax_nodal_median)
    IQR_nodal = np.percentile(emax_nodal, 75) - np.percentile(emax_nodal, 25)

    for j in np.arange(0, 18, 1):  # 18.61*365.25*24/12.42 = 13134   the number of 12.42 hrs cycles over the nodal period,
        i = j * 414 *706 # 12.42*60*60/108 = 414           the number of points per cycle of 12.42 hrs

        rel_time, tide_elevs = signal[:, 0][i:(414 * 706 + i)], signal[:, 1][i:(414 * 706 + i)]
        peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
        peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
        tidal_ranges, rel_times = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW,peak_elevs_LW)
        #print(len(tidal_ranges))
        emax = 0.5 * rho * grav * np.square(tidal_ranges)
        # PE_lunar = PE(signal[(0 + i):(414 * 57 + i)])
        Emax_year = np.sum(emax)/3.6e+6
        # A.append(Emax_lunar/PE_lunar)
        #time.append(rel_time[i])
        Emax_times.append(rel_time[0])
        Emax_year_all.append(Emax_year)
    return Emax_times, Emax_year_all, Emax_nodal_year_average





def IQR_all_monthhs(signal):
    "Function that calculates all IQR in all lunar months over the nodal cycle of 18.61 years and the average PE over nodal cycle"
    IQR_times=[]        #list of the start time of each lunar months (after 12.42hrs) to caclulate Emax for a lunar month
    IQR_all_months = []
    rel_time, tide_elevs = signal[:, 0], signal[:, 1]
    peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
    peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
    tidal_ranges_nodal, rel_times_nodal = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW)
    IQR_nodal=iqr(tidal_ranges_nodal)
    print("IQR(R) nodal", "%.2f" % IQR_nodal)
    for j in np.arange(0, 13134 - 58, 1):                         # 18.61*365.25*24/12.42 = 13134   the number of 12.42 hrs cycles over the nodal period,
        i = j * 414                                               # 12.42*60*60/108 = 414           the number of points per cycle of 12.42 hrs
        rel_time, tide_elevs = signal[:, 0][i:(414 * 58 + i)], signal[:, 1][i:(414 * 58 + i)]
        peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
        peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
        tidal_ranges, rel_times = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW)
        IQR_lunar= iqr(tidal_ranges)
        IQR_times.append(rel_time[0])
        IQR_all_months.append(IQR_lunar)
    #print("IQR tidal ranges all months", IQR_all_months)
    return IQR_times, IQR_all_months, IQR_nodal


# def P50_all_monthhs(signal):
#     "Function that calculates all IQR in all lunar months over the nodal cycle of 18.61 years and the average PE over nodal cycle"
#     P50_times=[]        #list of the start time of each lunar months (after 12.42hrs) to caclulate Emax for a lunar month
#     P50_all_months = []
#     rel_time, tide_elevs = signal[:, 0], signal[:, 1]
#     peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
#     peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
#     tidal_ranges_nodal, rel_times_nodal = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW)
#     P50R_nodal = np.percentile(tidal_ranges_nodal, 50)
#     print("IQR(R) nodal", "%.2f" % P50_nodal)
#     for j in np.arange(0, 13134 - 58, 1):                         # 18.61*365.25*24/12.42 = 13134   the number of 12.42 hrs cycles over the nodal period,
#         i = j * 414                                               # 12.42*60*60/108 = 414           the number of points per cycle of 12.42 hrs
#         rel_time, tide_elevs = signal[:, 0][i:(414 * 58 + i)], signal[:, 1][i:(414 * 58 + i)]
#         peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
#         peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
#         tidal_ranges, rel_times = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW)
#         P50_lunar= np.percentile(tidal_ranges, 50)
#         P50_times.append(rel_time[0])
#         P50_all_months.append(P50_lunar)
#     #print("IQR tidal ranges all months", IQR_all_months)
#     return IQR_times, IQR_all_months, IQR_nodal


def P50_all_monthhs(signal):
    "Function that calculates all IQR in all lunar months over the nodal cycle of 18.61 years and the average PE over nodal cycle"
    P50_times=[]        #list of the start time of each lunar months (after 12.42hrs) to caclulate Emax for a lunar month
    P50_all_months = []
    rel_time, tide_elevs = signal[:, 0], signal[:, 1]
    peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
    peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
    tidal_ranges_nodal, rel_times_nodal = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW)
    P50_nodal=np.percentile(tidal_ranges_nodal,50)
    print("P50(R) nodal", "%.2f" % P50_nodal)
    for j in np.arange(0, 13134 - 58, 1):                         # 18.61*365.25*24/12.42 = 13134   the number of 12.42 hrs cycles over the nodal period,
        i = j * 414                                               # 12.42*60*60/108 = 414           the number of points per cycle of 12.42 hrs
        rel_time, tide_elevs = signal[:, 0][i:(414 * 58 + i)], signal[:, 1][i:(414 * 58 + i)]
        peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
        peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
        tidal_ranges, rel_times = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW)
        P50_lunar= np.percentile(tidal_ranges,50)
        P50_times.append(rel_time[0])
        P50_all_months.append(P50_lunar)
    #print("IQR tidal ranges all months", IQR_all_months)
    return P50_times, P50_all_months, P50_nodal






# def P50_all_monthhs(signal):
#     "Function that calculates all PE in all lunar months over the nodal cycle of 18.61 years and the average PE over nodal cycle"
#     median_times=[]        #list of the start time of each lunar months (after 12.42hrs) to caclulate Emax for a lunar month
#     P50_all_months = []
#     rel_time, tide_elevs = signal[:, 0], signal[:, 1]
#     peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
#     peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
#     tidal_ranges_nodal, rel_times_nodal = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW)
#     P50_nodal=median(tidal_ranges_nodal)
#     print("P50 nodal", P50_nodal)
#     for j in np.arange(0, 13134 - 58, 1):  # 18.61*365.25*24/12.42 = 13134   the number of 12.42 hrs cycles over the nodal period,
#         i = j * 414  # 12.                  # 42*60*60/108 = 414           the number of points per cycle of 12.42 hrs
#
#         rel_time, tide_elevs = signal[:, 0][i:(414 * 58 + i)], signal[:, 1][i:(414 * 58 + i)]
#         peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
#         peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
#         tidal_ranges, rel_times = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW)
#         median_lunar= median(tidal_ranges)
#         median_times.append(rel_time[0])
#         P50_all_months.append(median_lunar)
#     print("P50 all months:", P50_all_months)
#     return median_times, P50_all_months, P50_nodal

def P50_energy_all_monthhs(signal):
    rho = 1021
    grav = 9.81

    "Function that calculates all IQR in all lunar months over the nodal cycle of 18.61 years and the average PE over nodal cycle"
    P50_times=[]        #list of the start time of each lunar months (after 12.42hrs) to caclulate Emax for a lunar month
    P50_all_months = []
    rel_time, tide_elevs = signal[:, 0], signal[:, 1]
    peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
    peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
    tidal_ranges_nodal, rel_times_nodal = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW)
    emax_nodal = 0.5 * rho * grav * np.square(tidal_ranges_nodal) / 3.6e+3
    P50_nodal=np.percentile(emax_nodal,50)
    print("P50(R) nodal", "%.2f" % P50_nodal)
    for j in np.arange(0, 13134 - 58, 1):                         # 18.61*365.25*24/12.42 = 13134   the number of 12.42 hrs cycles over the nodal period,
        i = j * 414                                               # 12.42*60*60/108 = 414           the number of points per cycle of 12.42 hrs
        rel_time, tide_elevs = signal[:, 0][i:(414 * 58 + i)], signal[:, 1][i:(414 * 58 + i)]
        peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
        peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
        tidal_ranges, rel_times = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW)
        emax_lunar = 0.5 * rho * grav * np.square(tidal_ranges) / 3.6e+3
        P50_lunar= np.percentile(emax_lunar,50)
        P50_times.append(rel_time[0])
        P50_all_months.append(P50_lunar)
    #print("IQR tidal ranges all months", IQR_all_months)
    return P50_times, P50_all_months, P50_nodal

def IQR_energy_all_monthhs(signal):
    "Function that calculates all IQR in all lunar months over the nodal cycle of 18.61 years and the average PE over nodal cycle"
    rho = 1021
    grav = 9.81

    IQR_times=[]        #list of the start time of each lunar months (after 12.42hrs) to caclulate Emax for a lunar month
    IQR_all_months = []
    rel_time, tide_elevs = signal[:, 0], signal[:, 1]
    peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
    peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
    tidal_ranges_nodal, rel_times_nodal = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW)
    emax_nodal= 0.5*rho*grav*np.square(tidal_ranges_nodal)/3.6e+3
    IQR_nodal=iqr(emax_nodal)
    print("IQR(E) nodal", "%.2f" % IQR_nodal)
    for j in np.arange(0,13134 - 58, 1):                         # 18.61*365.25*24/12.42 = 13134   the number of 12.42 hrs cycles over the nodal period,
        i = j * 414                                               # 12.42*60*60/108 = 414           the number of points per cycle of 12.42 hrs
        rel_time, tide_elevs = signal[:, 0][i:(414 * 58 + i)], signal[:, 1][i:(414 * 58 + i)]
        peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
        peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
        tidal_ranges, rel_times = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW)
        #print(j)
        #print("Len tidal ranges:",len(tidal_ranges))
        emax_lunar = 0.5 * rho * grav * np.square(tidal_ranges)/3.6e+3
        IQR_lunar= iqr(emax_lunar)
        IQR_times.append(rel_time[0])
        IQR_all_months.append(IQR_lunar)
    #print("IQR tidal ranges all months", IQR_all_months)
    return IQR_times, IQR_all_months, IQR_nodal


def IQR_energy_all_monthhs(signal):
    "Function that calculates all IQR in all lunar months over the nodal cycle of 18.61 years and the average PE over nodal cycle"
    rho = 1021
    grav = 9.81

    IQR_times=[]        #list of the start time of each lunar months (after 12.42hrs) to caclulate Emax for a lunar month
    IQR_all_months = []
    rel_time, tide_elevs = signal[:, 0], signal[:, 1]
    peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
    peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
    tidal_ranges_nodal, rel_times_nodal = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW)
    emax_nodal= 0.5*rho*grav*np.square(tidal_ranges_nodal)/3.6e+3
    IQR_nodal=iqr(emax_nodal)
    print("IQR(E) nodal", "%.2f" % IQR_nodal)
    for j in np.arange(0,13134 - 58, 1):                         # 18.61*365.25*24/12.42 = 13134   the number of 12.42 hrs cycles over the nodal period,
        i = j * 414                                               # 12.42*60*60/108 = 414           the number of points per cycle of 12.42 hrs
        rel_time, tide_elevs = signal[:, 0][i:(414 * 58 + i)], signal[:, 1][i:(414 * 58 + i)]
        peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
        peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
        tidal_ranges, rel_times = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW)
        #print(j)
        #print("Len tidal ranges:",len(tidal_ranges))
        emax_lunar = 0.5 * rho * grav * np.square(tidal_ranges)/3.6e+3
        IQR_lunar= iqr(emax_lunar)
        IQR_times.append(rel_time[0])
        IQR_all_months.append(IQR_lunar)
    #print("IQR tidal ranges all months", IQR_all_months)
    return IQR_times, IQR_all_months, IQR_nodal


def Dn(signal):
    tidal_ranges_nodal = ranges(signal)
    KS = []  # stores the Dn of lunar-montly tidal range distribution with starting point each lunar month
    # months=np.arange(0, 230, 1)
    months = np.arange(0, 13134 - 58, 1)
    for j in months:  # 18.61*365.25*24/12.42 = 13134   the number of 12.42 hrs cycles over the nodal period
        i = j * 414  # * 57                                  # 414*57 (57 lunar cycles in a lunar month) (12.42*60*60/108 = 414  the number of points per cycle of 12.42 hrs)
        tidal_ranges =ranges(signal[i:(414 * 58 + i)])
        #print(j)
        #print(len(tidal_ranges))
        ks, p = stats.ks_2samp(tidal_ranges, tidal_ranges_nodal)
        KS.append(ks)
    return KS

def PE_all_monthhs(signal):
    "Function that calculates all PE in all lunar months over the nodal cycle of 18.61 years and the average PE over nodal cycle"
    PE_times=[]        #list of the start time of each lunar months (after 12.42hrs) to caclulate Emax for a lunar month
    PE_all_months = []
    PE_nodal=PE(signal)
    print("PE nodal", "%.2f" % PE_nodal)
    for j in np.arange(0, 13134 - 58, 1):  # 18.61*365.25*24/12.42 = 13134   the number of 12.42 hrs cycles over the nodal period,
        i = j * 414  # 12.42*60*60/108 = 414           the number of points per cycle of 12.42 hrs
        rel_time, tide_elevs = signal[:, 0][i:(414 * 58 + i)], signal[:, 1][i:(414 * 58 + i)]

        PE_lunar= PE(signal[i:(414 * 58 + i)])
        PE_times.append(rel_time[0])
        PE_all_months.append(PE_lunar)
    return PE_times, PE_all_months, PE_nodal

def Hm0_all_monthhs(signal):
    "Function that calculates all Hm0 in all lunar months over the nodal cycle of 18.61 years and the  Hm0 over nodal cycle"
    Hm0_times=[]        #list of the start time of each lunar months (after 12.42hrs) to caclulate Emax for a lunar month
    Hm0_all_months = []
    Hm0_nodal = Hm0(signal)
    print("-----------------------------------")
    print("Hm0 nodal", "%.2f" % Hm0_nodal)
    for j in np.arange(0, 13134 - 58, 1):  # 18.61*365.25*24/12.42 = 13134   the number of 12.42 hrs cycles over the nodal period,
        i = j * 414  # 12.42*60*60/108 = 414           the number of points per cycle of 12.42 hrs
        rel_time, tide_elevs = signal[:, 0][i:((414 * 58) + i)], signal[:, 1][i:((414 * 58) + i)]
        Hm0_lunar= Hm0(signal[i:(414 * 58 + i)])
        Hm0_times.append(rel_time[0])
        Hm0_all_months.append(Hm0_lunar)
    return Hm0_times, Hm0_all_months, Hm0_nodal

def Emax_all_monthhs(signal):
    "Function that calculates all Emax in all luanr months over the nodal cycle of 18.61 years and the average lunar monthly Emax over nodal cycle"
    Emax_times=[]        #list of the start time of each lunar months (after 12.42hrs) to caclulate Emax for a lunar month
    Emax_lunar_all_months = []
    rho = 1021
    grav = 9.81

    rel_time, tide_elevs = signal[:, 0], signal[:, 1]
    peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
    peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
    tidal_ranges_nodal, rel_times_nodal = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW,peak_elevs_LW)

    emax_nodal = 0.5 * rho * grav * np.square(tidal_ranges_nodal)/3.6e+6
    Emax_nodal = np.sum(emax_nodal)
    Emax_nodal_average = (Emax_nodal / 230)/3.6e+6  # 18.61*365.25/29.53
    Emax_nodal_median = np.percentile(emax_nodal, 50)
    IQR_nodal = np.percentile(emax_nodal, 75) - np.percentile(emax_nodal, 25)
    a=tidal_ranges_nodal
    b=np.square(tidal_ranges_nodal)
    print(np.stack((a, b), axis=-1))
    for j in np.arange(0, 13134 - 58, 1):  # 18.61*365.25*24/12.42 = 13134   the number of 12.42 hrs cycles over the nodal period,
        i = j * 414  # 12.42*60*60/108 = 414           the number of points per cycle of 12.42 hrs
        rel_time, tide_elevs = signal[:, 0][i:(414 * 58 + i)], signal[:, 1][i:(414 * 58 + i)]
        peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
        peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
        tidal_ranges, rel_times = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW,peak_elevs_LW)
        #print(len(tidal_ranges))
        emax = 0.5 * rho * grav * np.square(tidal_ranges)
        Emax_lunar = np.sum(emax)/3.6e+6
        Emax_times.append(rel_time[0])
        Emax_lunar_all_months.append(Emax_lunar)
    return Emax_times, Emax_lunar_all_months, emax_nodal


def Emax_all_monthhs_for_each_range(signal):
    rho = 1021
    grav =9.81

    rel_time, tide_elevs = signal[:, 0], signal[:, 1]
    peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
    peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')

    tidal_ranges_nodal, rel_times_nodal = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW,peak_elevs_LW)
    Emax_all =0.5 * rho * grav * np.square(tidal_ranges_nodal)/3.6e3
    return rel_times_nodal, Emax_all

def ranges(signal):
    rel_time, tide_elevs = signal[:, 0], signal[:, 1]
    peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
    peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
    tidal_ranges, rel_times = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW,peak_elevs_LW)
    return tidal_ranges


def PE_vs_Emax(signal):
    "Function that calculates all Emax in all luanr months over the nodal cycle of 18.61 years and the average lunar monthly Emax over nodal cycle"
    Emax_times=[]        #list of the start time of each lunar months (after 12.42hrs) to caclulate Emax for a lunar month
    Emax_lunar_all_months = []
    PE_lunar_all_months = []
    rho = 1021
    grav = 9.81
    A=[]
    rel_time, tide_elevs = signal[:, 0], signal[:, 1]
    peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
    peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
    tidal_ranges_nodal, rel_times_nodal = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW, peak_elevs_LW)

    emax_nodal = 0.5 * rho * grav * np.square(tidal_ranges_nodal)
    print(len(emax_nodal))
    Emax_nodal = np.sum(emax_nodal)
    PE_nodal=PE(signal)
    print("Emax nodal:", Emax_nodal/(3.6e+6*26268))
    print("PE nodal:", PE_nodal/3.6e+6)
    print(Emax_nodal/PE_nodal)
    Emax_nodal_average = (Emax_nodal / 230)/3.6e+6  # 18.61*365.25/29.53

    Emax_nodal_median = np.percentile(emax_nodal, 50)
    IQR_nodal = np.percentile(emax_nodal, 75) - np.percentile(emax_nodal, 25)

    for j in np.arange(0, 13134 - 58, 1):  # 18.61*365.25*24/12.42 = 13134   the number of 12.42 hrs cycles over the nodal period,
        i = j * 414  # 12.42*60*60/108 = 414           the number of points per cycle of 12.42 hrs

        rel_time, tide_elevs = signal[:, 0][i:(414 * 58 + i)], signal[:, 1][i:(414 * 58 + i)]
        peak_rel_times_HW, peak_elevs_HW = find_tidal_peaks(rel_time, tide_elevs, peak_type='HW')
        peak_rel_times_LW, peak_elevs_LW = find_tidal_peaks(rel_time, tide_elevs, peak_type='LW')
        tidal_ranges, rel_times = tidal_ranges_from_peaks(peak_rel_times_HW, peak_rel_times_LW, peak_elevs_HW,peak_elevs_LW)

        emax = 0.5 * rho * grav * np.square(tidal_ranges)
        Emax_lunar = np.sum(emax)/(3.6e+6*58)
        PE_lunar = PE(signal[(0 + i):(414 * 58 + i)])/3.6e+6

        A.append(Emax_lunar/PE_lunar)

        Emax_lunar_all_months.append(Emax_lunar)
        PE_lunar_all_months.append(PE_lunar)
        df=pd.DataFrame({"Emax": Emax_lunar_all_months,"PE": PE_lunar_all_months, "Emax/PE":A })

    return df


def spatial_values(file,file2,N):
    df = pd.read_csv(file)
    location_all = df["LOCATION"][0:N]

    pd_metrics = pd.read_csv(file2)

    IQRE_12 = []
    PE_12 = []
    for loc in location_all:
        min_diff_PE_12_all = []
        max_diff_PE_12_all = []
        diff_PE_2_all = []
        diff_PE_4_all = []
        diff_PE_8_all = []
        diff_PE_12_all = []
        diff_PE_Dn_12_all = []

        PE_2_all = []
        PE_4_all = []
        PE_8_all = []
        PE_12_all = []

        IQRE_2_all = []
        IQRE_4_all = []
        IQRE_8_all = []
        IQRE_12_all = []

        min_diff_IQRE_12_all = []
        max_diff_IQRE_12_all = []
        diff_IQRE_2_all = []
        diff_IQRE_4_all = []
        diff_IQRE_8_all = []
        diff_IQRE_12_all = []
        diff_IQRE_Dn_12_all = []
        A_12 = []

        for location in location_all:
            # location="AVONMOUTH"
            print('---------------------' + location + '-----------------------')
            # Get representative times for energy and ranges based on location:
            rep_time_2_ranges = df[df['LOCATION'] == loc]['START TIME 2 CONS RANGES'].values[0]
            rep_time_4_ranges = df[df['LOCATION'] == loc]['START TIME 4 CONS RANGES'].values[0]
            rep_time_8_ranges = df[df['LOCATION'] == loc]['START TIME 8 CONS RANGES'].values[0]
            rep_time_12_ranges = df[df['LOCATION'] == loc]['START TIME 12 CONS RANGES'].values[0]

            rep_time_2_energy = df[df['LOCATION'] == loc]['START TIME 2 CONS ENERGY'].values[0]
            rep_time_4_energy = df[df['LOCATION'] == loc]['START TIME 4 CONS ENERGY'].values[0]
            rep_time_8_energy = df[df['LOCATION'] == loc]['START TIME 8 CONS ENERGY'].values[0]
            rep_time_12_energy = df[df['LOCATION'] == loc]['START TIME 12 CONS ENERGY'].values[0]

            #rep_time_12_energy_Dn = df2[df2['LOCATION'] == location]['START TIME 12 CONS ENERGY'].values[0]
            # We find the corresponding PE based on the time we set and compare to the median PE of nodal cycle for 12 constituents to see how the representative period of other constituents match
            diff_PE_2_ranges = \
            pd_metrics[pd_metrics["TIME"] == rep_time_2_ranges][location + " (PE_m - PE_n)/PE_n"].values[0]
            diff_PE_4_ranges = \
            pd_metrics[pd_metrics["TIME"] == rep_time_4_ranges][location + " (PE_m - PE_n)/PE_n"].values[0]
            diff_PE_8_ranges = \
            pd_metrics[pd_metrics["TIME"] == rep_time_8_ranges][location + " (PE_m - PE_n)/PE_n"].values[0]
            diff_PE_12_ranges = \
            pd_metrics[pd_metrics["TIME"] == rep_time_12_ranges][location + " (PE_m - PE_n)/PE_n"].values[0]

            diff_Hm0_2_ranges = \
            pd_metrics[pd_metrics["TIME"] == rep_time_2_ranges][location + " (Hm0_m - Hm0_n)/Hm0_n"].values[0]
            diff_Hm0_4_ranges = \
            pd_metrics[pd_metrics["TIME"] == rep_time_4_ranges][location + " (Hm0_m - Hm0_n)/Hm0_n"].values[0]
            diff_Hm0_8_ranges = \
            pd_metrics[pd_metrics["TIME"] == rep_time_8_ranges][location + " (Hm0_m - Hm0_n)/Hm0_n"].values[0]
            diff_Hm0_12_ranges = \
            pd_metrics[pd_metrics["TIME"] == rep_time_12_ranges][location + " (Hm0_m - Hm0_n)/Hm0_n"].values[0]

            diff_IQRR_2_ranges = \
            pd_metrics[pd_metrics["TIME"] == rep_time_2_ranges][location + " (IQRR_m - IQRR_n)/IQRR_n"].values[0]
            diff_IQRR_4_ranges = \
            pd_metrics[pd_metrics["TIME"] == rep_time_4_ranges][location + " (IQRR_m - IQRR_n)/IQRR_n"].values[0]
            diff_IQRR_8_ranges = \
            pd_metrics[pd_metrics["TIME"] == rep_time_8_ranges][location + " (IQRR_m - IQRR_n)/IQRR_n"].values[0]
            diff_IQRR_12_ranges = \
            pd_metrics[pd_metrics["TIME"] == rep_time_12_ranges][location + " (IQRR_m - IQRR_n)/IQRR_n"].values[0]

            IQRE_2_energy = pd_metrics[pd_metrics["TIME"] == rep_time_2_energy][location + " IQRE_m"].values[0]
            IQRE_4_energy = pd_metrics[pd_metrics["TIME"] == rep_time_4_energy][location + " IQRE_m"].values[0]
            IQRE_8_energy = pd_metrics[pd_metrics["TIME"] == rep_time_8_energy][location + " IQRE_m"].values[0]
            IQRE_12_energy = pd_metrics[pd_metrics["TIME"] == rep_time_12_energy][location + " IQRE_m"].values[0]

            diff_IQRE_2_ranges = \
            pd_metrics[pd_metrics["TIME"] == rep_time_2_ranges][location + " (IQRE_m - IQRE_n)/IQRE_n"].values[0]
            diff_IQRE_4_ranges = \
            pd_metrics[pd_metrics["TIME"] == rep_time_4_ranges][location + " (IQRE_m - IQRE_n)/IQRE_n"].values[0]
            diff_IQRE_8_ranges = \
            pd_metrics[pd_metrics["TIME"] == rep_time_8_ranges][location + " (IQRE_m - IQRE_n)/IQRE_n"].values[0]
            diff_IQRE_12_ranges = \
            pd_metrics[pd_metrics["TIME"] == rep_time_12_ranges][location + " (IQRE_m - IQRE_n)/IQRE_n"].values[0]

            # Energy rep month

            PE_2_energy = pd_metrics[pd_metrics["TIME"] == rep_time_2_energy][location + " PE_m"].values[0]
            PE_4_energy = pd_metrics[pd_metrics["TIME"] == rep_time_4_energy][location + " PE_m"].values[0]
            PE_8_energy = pd_metrics[pd_metrics["TIME"] == rep_time_8_energy][location + " PE_m"].values[0]
            PE_12_energy = pd_metrics[pd_metrics["TIME"] == rep_time_12_energy][location + " PE_m"].values[0]

            diff_PE_2_energy = \
            pd_metrics[pd_metrics["TIME"] == rep_time_2_energy][location + " (PE_m - PE_n)/PE_n"].values[0]
            diff_PE_4_energy = \
            pd_metrics[pd_metrics["TIME"] == rep_time_4_energy][location + " (PE_m - PE_n)/PE_n"].values[0]
            diff_PE_8_energy = \
            pd_metrics[pd_metrics["TIME"] == rep_time_8_energy][location + " (PE_m - PE_n)/PE_n"].values[0]
            diff_PE_12_energy = \
            pd_metrics[pd_metrics["TIME"] == rep_time_12_energy][location + " (PE_m - PE_n)/PE_n"].values[0]

            diff_Hm0_2_energy = \
            pd_metrics[pd_metrics["TIME"] == rep_time_2_energy][location + " (Hm0_m - Hm0_n)/Hm0_n"].values[0]
            diff_Hm0_4_energy = \
            pd_metrics[pd_metrics["TIME"] == rep_time_4_energy][location + " (Hm0_m - Hm0_n)/Hm0_n"].values[0]
            diff_Hm0_8_energy = \
            pd_metrics[pd_metrics["TIME"] == rep_time_8_energy][location + " (Hm0_m - Hm0_n)/Hm0_n"].values[0]
            diff_Hm0_12_energy = \
            pd_metrics[pd_metrics["TIME"] == rep_time_12_energy][location + " (Hm0_m - Hm0_n)/Hm0_n"].values[0]

            diff_IQRR_2_energy = \
            pd_metrics[pd_metrics["TIME"] == rep_time_2_energy][location + " (IQRR_m - IQRR_n)/IQRR_n"].values[0]
            diff_IQRR_4_energy = \
            pd_metrics[pd_metrics["TIME"] == rep_time_4_energy][location + " (IQRR_m - IQRR_n)/IQRR_n"].values[0]
            diff_IQRR_8_energy = \
            pd_metrics[pd_metrics["TIME"] == rep_time_8_energy][location + " (IQRR_m - IQRR_n)/IQRR_n"].values[0]
            diff_IQRR_12_energy = \
            pd_metrics[pd_metrics["TIME"] == rep_time_12_energy][location + " (IQRR_m - IQRR_n)/IQRR_n"].values[0]

            diff_IQRE_2_energy = \
            pd_metrics[pd_metrics["TIME"] == rep_time_2_energy][location + " (IQRE_m - IQRE_n)/IQRE_n"].values[0]
            diff_IQRE_4_energy = \
            pd_metrics[pd_metrics["TIME"] == rep_time_4_energy][location + " (IQRE_m - IQRE_n)/IQRE_n"].values[0]
            diff_IQRE_8_energy = \
            pd_metrics[pd_metrics["TIME"] == rep_time_8_energy][location + " (IQRE_m - IQRE_n)/IQRE_n"].values[0]
            diff_IQRE_12_energy = \
            pd_metrics[pd_metrics["TIME"] == rep_time_12_energy][location + " (IQRE_m - IQRE_n)/IQRE_n"].values[0]

            # Find min/max values
            min_diff_PE_12 = pd_metrics[
                pd_metrics[location + " (PE_m - PE_n)/PE_n"] == pd_metrics[location + " (PE_m - PE_n)/PE_n"].min()][
                location + " (PE_m - PE_n)/PE_n"].values[0]
            max_diff_PE_12 = pd_metrics[
                pd_metrics[location + " (PE_m - PE_n)/PE_n"] == pd_metrics[location + " (PE_m - PE_n)/PE_n"].max()][
                location + " (PE_m - PE_n)/PE_n"].values[0]

            min_diff_IQRE_12 = pd_metrics[pd_metrics[location + " (IQRE_m - IQRE_n)/IQRE_n"] == pd_metrics[
                location + " (IQRE_m - IQRE_n)/IQRE_n"].min()][location + " (IQRE_m - IQRE_n)/IQRE_n"].values[0]
            max_diff_IQRE_12 = pd_metrics[pd_metrics[location + " (IQRE_m - IQRE_n)/IQRE_n"] == pd_metrics[
                location + " (IQRE_m - IQRE_n)/IQRE_n"].max()][location + " (IQRE_m - IQRE_n)/IQRE_n"].values[0]

            # diff_IQRE_Dn_12_energy = \
            # pd_metrics[pd_metrics["TIME"] == rep_time_12_energy_Dn][location + " (IQRE_m - IQRE_n)/IQRE_n"].values[0]
            # diff_PE_Dn_12_energy = \
            # pd_metrics[pd_metrics["TIME"] == rep_time_12_energy_Dn][location + " (PE_m - PE_n)/PE_n"].values[0]

            min_diff_IQRE_12_all.append(min_diff_IQRE_12)
            max_diff_IQRE_12_all.append(max_diff_IQRE_12)
            diff_IQRE_2_all.append(diff_IQRE_2_energy)
            diff_IQRE_4_all.append(diff_IQRE_4_energy)
            diff_IQRE_8_all.append(diff_IQRE_8_energy)
            diff_IQRE_12_all.append(diff_IQRE_12_energy)
           # diff_IQRE_Dn_12_all.append(diff_IQRE_Dn_12_energy)

            IQRE_2_all.append(IQRE_2_energy)
            IQRE_4_all.append(IQRE_4_energy)
            IQRE_8_all.append(IQRE_8_energy)
            IQRE_12_all.append(IQRE_12_energy)

            min_diff_PE_12_all.append(min_diff_PE_12)
            max_diff_PE_12_all.append(max_diff_PE_12)
            diff_PE_2_all.append(diff_PE_2_energy)
            diff_PE_4_all.append(diff_PE_4_energy)
            diff_PE_8_all.append(diff_PE_8_energy)
            diff_PE_12_all.append(diff_PE_12_energy)
           # diff_PE_Dn_12_all.append(diff_PE_Dn_12_energy)

            PE_2_all.append(PE_2_energy)
            PE_4_all.append(PE_4_energy)
            PE_8_all.append(PE_8_energy)
            PE_12_all.append(PE_12_energy * 1000)

        IQRE_12.append(diff_IQRE_12_all)
        PE_12.append(diff_PE_12_all)

        return PE_12,IQRE_12,  min_diff_PE_12_all, max_diff_PE_12, min_diff_IQRE_12_all, max_diff_IQRE_12_all
