import bisect
from pathlib import Path
from statsmodels.tsa import vector_ar
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from mne.io import read_raw_edf

# import mne.io.raw
from mne import *
import mne
import matplotlib
import numpy as np
from statsmodels import *
import statsmodels
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

import warnings
warnings.filterwarnings("ignore")

def adf_test(timeseries):
    # print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    # print(dftest)
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    # for key, value in dftest[4].items():
    #     dfoutput["Critiical Value (%s)" % key] = value
    # print(dfoutput)

    return dfoutput["p-value"]
def read(path, elec, pre_ictal, end):

    edf = read_raw_edf(path, preload=False, stim_channel=None, verbose=False)
    xx = edf.ch_names
    index = xx.index(elec)
    fs = edf.info['sfreq']
    fs = int(fs)
    signal_input = edf[index]
    signal = signal_input[0]
    signal_input = signal[0]

    signal_input = signal_input[pre_ictal:end]

    return signal_input
def stat(array):

    dftest = adfuller(array, autolag="AIC")

    if (dftest[1] > 0.05):

        # print("is not stationary")
        array = np.diff(array)
        stat(array)

    # print("lenght of the result", len(array))
    return array
def stat_signal(signal, wind):
    start = 0
    end = wind * fs
    output = []

    while (True):
        # print("start computing")

        sig = signal[start:end]


        si = stat(sig)

        # print("lenght of si", len(si))

        output = np.append(output, si)

        start = start + (wind * fs)
        end = end + (wind * fs)

        # print("how many minute left", ((len(signal) - end) /60/200))

        # if (((len(signal) - end) /60/200) == 0):
        if (end > len(signal)):
            break

    return output
def causal_relation(array, wind, elec1, elec2):

    start = 0
    end = wind * fs
    output = []

    while (True):

        sig = array[start:end]

        # print("computing the lag order")

        model = VAR(sig)
        x = model.select_order(maxlags=50)
        selected_order = x.aic

        # print("Granger causality computing")

        gc_res = grangercausalitytests(sig[[elec1, elec2]], [int(selected_order)], verbose=False)

        pp = list(gc_res.keys())[0]
        xx = gc_res[pp]
        yy = xx[0]; zz = yy["ssr_ftest"]

        output.append(zz[1])

        start = start + (wind * fs)
        end = end + (wind * fs)

        if (end > len(array)):

            break

    return output
def signal_preparation(EEG, ECG, wind, electrod, electrode):

    print("starting stationarity tests")
    EEG_s = stat_signal(EEG, wind)
    ECG_s = stat_signal(ECG, wind)

    arrayx = []
    arrayx.append(EEG_s)
    arrayx.append(ECG_s)

    # == cvreating the Data frame
    d = pd.DataFrame(arrayx)
    d1 = d.transpose()
    d1.columns = [electrod, electrode]

    d1[electrod] = d1[electrod].replace(np.nan, 0)
    d1[electrode] = d1[electrode].replace(np.nan, 0)

    return d1
def elmenet_count(cc):

    elm = sum(1 for e in cc if e <= 0.05)
    per = (elm / len(cc)) * 100
    # print("elmenet_count T6 for EEG --> ECG\t", per)

    return per

def do (file_path, pre_ictal, end, wind, ecg, eeg, eeg1):


    # == reading the electrode

    ECGR = read(file_path, ecg, pre_ictal, end)
    signal = ECGR
    # ===============================================
    # == working on T3
    EEG_T3 = read(file_path, eeg, pre_ictal, end)

    d1 = signal_preparation(EEG_T3, signal, wind, eeg, ecg)

    cc = causal_relation(d1, wind, ecg, eeg)
    print("elmenet_count " + eeg + " for EEG --> ECG\t", elmenet_count(cc))

    cc = 0
    # d1 = d1[[ecg, eeg]]
    cc = causal_relation(d1, wind, eeg, ecg)
    print("elmenet_count " + eeg + " for ECG --> EEG\t", elmenet_count(cc))

    # ===============================================
    # == working on T5
    EEG_T5 = read(file_path, eeg1, pre_ictal, end)

    d1 = signal_preparation(EEG_T5, signal, wind, eeg1, ecg)

    cc = causal_relation(d1, wind, ecg, eeg1)
    print("elmenet_count " + eeg1 + " for EEG --> ECG\t", elmenet_count(cc))

    cc = causal_relation(d1, wind, eeg1, ecg)
    print("elmenet_count " + eeg1 + " for ECG --> EEG\t", elmenet_count(cc))

    # ===========================================================
    # == reading the electrode
    d1 = signal_preparation(EEG_T3, EEG_T5, wind, eeg, eeg1)

    cc = causal_relation(d1, wind, eeg, eeg1)
    print("elmenet_count " + eeg + " --> " + eeg1 + "\t", elmenet_count(cc))

    cc = causal_relation(d1, wind, eeg1, eeg)
    print("elmenet_count" + eeg1 + " --> " + eeg + "\t", elmenet_count(cc))

    return True
def do_mod (file_path, pre_ictal, end, wind, ecg, eeg, eeg1):

    # == reading the electrode

    ECGR = read(file_path, ecg, pre_ictal, end)
    signal = ECGR
    # ===============================================
    # == working on T3

    EEG_T3 = read(file_path, eeg, pre_ictal, end)

    # d1 = signal_preparation(EEG_T3, signal, wind, eeg, ecg)
    #
    # cc = causal_relation(d1, wind, ecg, eeg)
    # print("elmenet_count " + eeg + " --> " + ecg + "\t", elmenet_count(cc))

    # cc = causal_relation(d1, wind, eeg1, eeg)
    # print("elmenet_count" + eeg1 + " --> " + eeg + "\t", elmenet_count(cc))

    # ===============================================
    # == working on T5
    EEG_T5 = read(file_path, eeg1, pre_ictal, end)
    #
    # # d1 = signal_preparation(EEG_T5, signal, wind, eeg1, ecg)
    # #
    # # cc = causal_relation(d1, wind, ecg, eeg1)
    # # print("elmenet_count " + eeg1 + " for EEG --> ECG\t", elmenet_count(cc))
    # #
    # # cc = causal_relation(d1, wind, eeg1, ecg)
    # # print("elmenet_count " + eeg1 + " for ECG --> EEG\t", elmenet_count(cc))
    #
    # # ===========================================================
    # # == reading the electrode
    d1 = signal_preparation(EEG_T3, EEG_T5, wind, eeg, eeg1)

    cc = causal_relation(d1, wind, eeg, eeg1)
    print("elmenet_count " + eeg + " --> " + eeg1 + "\t", elmenet_count(cc))

    cc = causal_relation(d1, wind, eeg1, eeg)
    print("elmenet_count" + eeg1 + " --> " + eeg + "\t", elmenet_count(cc))

    return cc

def causality(arr, elec1, elec2):

    direct = []
    model = VAR(arr)
    x = model.select_order(maxlags=50)
    selected_order = x.aic

    print("EEG on ECG ")
    for i in range(elec2):

        print("relation causality of" + elec1+" on " + elec2[i])

        gc_res = grangercausalitytests(arr[[elec1, elec2[i]]], [int(selected_order)], verbose=False)
        pp = list(gc_res.keys())[0]
        xx = gc_res[pp]
        yy = xx[0];
        zz = yy["ssr_ftest"]

        direct.append(zz[1])

    print("ECG on EEG")

    inverse = []
    for i in range(elec2):

        print("relation causality of" + elec2[i] + " on " + elec1)
        gc_res = grangercausalitytests(arr[[elec2[i]], elec1], [int(selected_order)], verbose=False)
        pp = list(gc_res.keys())[0]
        xx = gc_res[pp]
        yy = xx[0];
        zz = yy["ssr_ftest"]

        inverse.append(zz[1])

    return direct, inverse


from time import monotonic

start_time = monotonic()
# something

# == reading the electrode
# file_path = "E:\\data\\doggi\\KHEDHIRI.edf"

# == Patient 00
fs = 512

"We start buy reading the features files "

# = reading the features files.

# print("###############################################################")
# file = "/home/ftay/Downloads/siena-scalp-eeg-database-1.0.0/PN10/"
#
# elec_ecg = "2"
# elec = "EEG Fp1"
# elec_2 = "EEG F3"
#
# ID = "PN10-4.5.6.edf"
# file_path = file+ID
# # == Pre-ictal period
# end = (fs * 60 * 60 * 0) + (fs * 60 * 38) + (29 * fs)
# pre_ictal = (fs * 60 * 60 * 0) + (fs * 60 * 0) + (0 * fs)
#
# wind = 4
# print("window size equals to\t", wind)
# xx = do(file_path, pre_ictal, end, wind, elec_ecg, elec, elec_2)
#
# wind = 8
# print("window size equals to\t", wind)
# xx = do(file_path, pre_ictal, end, wind, elec_ecg, elec, elec_2)
#
# wind = 30
# print("window size equals to\t", wind)
# xx = do(file_path, pre_ictal, end, wind, elec_ecg, elec, elec_2)

print("###############################################################")
file = "/home/ftay/Downloads/siena-scalp-eeg-database-1.0.0/PN10/"

elec_ecg = "2"
elec = "EEG F3"
elec_2 = "EEG F7"

ID = "PN10-4.5.6.edf"
file_path = file+ID
# == Pre-ictal period
end = (fs * 60 * 60 * 0) + (fs * 60 * 38) + (29 * fs)
pre_ictal = (fs * 60 * 60 * 0) + (fs * 60 * 0) + (0 * fs)

wind = 4
print("window size equals to\t", wind)
xx= do_mod(file_path, pre_ictal, end, wind, elec_ecg, elec, elec_2)

wind = 8
print("window size equals to\t", wind)
xx = do_mod(file_path, pre_ictal, end, wind, elec_ecg, elec, elec_2)

wind = 30
print("window size equals to\t", wind)
xx = do_mod(file_path, pre_ictal, end, wind, elec_ecg, elec, elec_2)
#
print("###############################################################")

# print("#######################################")
# print("Inter-ictal period")
# end = (fs * 60 * 60 * 0) + (fs * 60 * 30) + (0 * fs)
# pre_ictal = (fs * 60 * 60 * 0) + (fs * 60 * 0) + (0 * fs)
#
# wind = 4
# print("window size equals to\t", wind)
# xx, yy, cc = do_mod(file_path, pre_ictal, end, wind, elec_ecg, elec, elec_2)

# wind = 8
# print("window size equals to\t", wind)
# xx = do_mod(file_path, pre_ictal, end, wind, elec_ecg, elec, elec_2)
#
# wind = 30
# print("window size equals to\t", wind)
# xx = do_mod(file_path, pre_ictal, end, wind, elec_ecg, elec, elec_2)

# fig, axs = plt.subplots(3, 1)
# axs[0].plot(xx, label="EEG T3 signal")
# axs[1].plot(yy, label="ECG signal")
# axs[2].plot(cc, label="Granger causality test result")
#
# axs[0].set_xlabel('progress per sample')
# axs[0].set_ylabel('Amplitude')
#
# axs[1].set_xlabel('progress per sample')
# axs[1].set_ylabel('Amplitude')
#
# axs[2].set_xlabel('progress in time per 4s')
# axs[2].set_ylabel('Amplitude')
#
#
# axs[0].grid(True)
# axs[1].grid(True)
# axs[2].grid(True)
#
# axs[0].legend()
# axs[1].legend()
# axs[2].legend()
#
# plt.show()



print(f"Run time {monotonic() - start_time} seconds")

# lag = statsmodels.tsa.vector_ar.var_model.LagOrderResults(array)

# path = "C:\\Users\\Administrator\\OneDrive - Aix-Marseille Université\\causality\\granger\\"
# Path(path).mkdir(parents=True, exist_ok=True)
# np.savetxt(path + 'GC-PN00-3-Fp1-ECG.txt', np.array(cc))
# sig[['Fp1', 'ECG']
