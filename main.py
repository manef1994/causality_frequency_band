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

# =====================================================================================================================
def read_R(path):

    with open(path + "frequency-AlphaEEG T4.txt", 'r') as file1:
        alpha = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    with open(path + "frequency-BetaEEG T4.txt", 'r') as file1:
        beta = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    with open(path + "frequency-DeltaEEG T4.txt", 'r') as file1:
        delta = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    with open(path + "frequency-SigmaEEG T4.txt", 'r') as file1:
        sigma = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    with open(path + "frequency-ThetaEEG T4.txt", 'r') as file1:
        theta = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    return delta, theta, alpha, sigma, beta

def read_R_1(path):

    with open(path + "frequency-AlphaEEG T6.txt", 'r') as file1:
        alpha = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    with open(path + "frequency-BetaEEG T6.txt", 'r') as file1:
        beta = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    with open(path + "frequency-DeltaEEG T6.txt", 'r') as file1:
        delta = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    with open(path + "frequency-SigmaEEG T6.txt", 'r') as file1:
        sigma = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    with open(path + "frequency-ThetaEEG T6.txt", 'r') as file1:
        theta = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    return delta, theta, alpha, sigma, beta

def read_L(path):

    with open(path + "frequency-AlphaEEG T3.txt", 'r') as file1:
        alpha = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    with open(path + "frequency-BetaEEG T3.txt", 'r') as file1:
        beta = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    with open(path + "frequency-DeltaEEG T3.txt", 'r') as file1:
        delta = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    with open(path + "frequency-SigmaEEG T3.txt", 'r') as file1:
        sigma = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    with open(path + "frequency-ThetaEEG T3.txt", 'r') as file1:
        theta = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    return delta, theta, alpha, sigma, beta

def read_L_1(path):

    with open(path + "frequency-AlphaEEG T5.txt", 'r') as file1:
        alpha = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    with open(path + "frequency-BetaEEG T5.txt", 'r') as file1:
        beta = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    with open(path + "frequency-DeltaEEG T5.txt", 'r') as file1:
        delta = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    with open(path + "frequency-SigmaEEG T5.txt", 'r') as file1:
        sigma = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    with open(path + "frequency-ThetaEEG T5.txt", 'r') as file1:
        theta = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    return delta, theta, alpha, sigma, beta

def read_ecg(path):

    with open(path + "frequency-HF.txt", 'r') as file1:
        HF = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    with open(path + "frequency-LF.txt", 'r') as file1:
        LF = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    with open(path + "frequency-ratio.txt", 'r') as file1:
        Ratio = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

    return LF, HF, Ratio

def signal_preparation(LF, HF, Ratio, Delta, Theta, Alpha, Sigma, Beta):

    # print("starting stationarity tests")

    LF = stat(LF)
    HF = stat(HF)
    Ratio = stat(Ratio)

    Delta = stat(Delta)
    Theta = stat(Theta)
    Alpha = stat(Alpha)
    Sigma = stat(Sigma)
    Beta = stat(Beta)

    arrayx = []
    arrayx.append(LF); arrayx.append(HF); arrayx.append(Ratio); arrayx.append(Delta); arrayx.append( Theta); arrayx.append(Alpha); arrayx.append(Sigma); arrayx.append(Beta)

    # == cvreating the Data frame
    d = pd.DataFrame(arrayx)
    d1 = d.transpose()

    d1.columns = ["LF", "HF", "Ratio", "Delta", "Theta", "Alpha", "Sigma", "Beta"]

    elect = ["LF", "HF", "Ratio", "Delta", "Theta", "Alpha", "Sigma", "Beta"]
    for i in range(len(elect)):
        d1[elect[i]] = d1[elect[i]].replace(np.nan, 0)

    return d1

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

def stat(array):
    dftest = adfuller(array, autolag="AIC")

    if (dftest[1] > 0.05):
        # print("is not stationary")
        array = np.diff(array)
        stat(array)

    return array

def causality(arr, elec1, elec2, res):

    arr = arr[[elec1, elec2]]
    xc = len(arr[elec1]) - len(arr[elec2])
    # print("lenght eqials to\t", xc)

    model = VAR(arr)
    x = model.select_order(maxlags=70)
    selected_order = x.aic
    print("the lag selected is\t", selected_order)


    # print("EEG on ECG ")

    gc_res = grangercausalitytests(arr[[elec1, elec2]], [int(selected_order)], verbose=False)
    pp = list(gc_res.keys())[0]
    xx = gc_res[pp]
    yy = xx[0];
    zz = yy["ssr_ftest"]
    res.at[elec2, elec1] = zz[1]
    # print(" the causal relation of electrode\t" + elec2 + "\t-->\t" + elec1 + "\t", zz[1])


    gc_res = grangercausalitytests(arr[[elec2, elec1]], [int(selected_order)], verbose=False)
    pp = list(gc_res.keys())[0]
    xx = gc_res[pp]
    yy = xx[0];
    zz = yy["ssr_ftest"]
    res.at[elec1, elec2] = zz[1]
    # print(" the causal relation of electrode\t" + elec1 + "\t-->\t" + elec2 + "\t", zz[1])
    print("")

from time import monotonic

start_time = monotonic()

file = "/home/ftay/Desktop/causality/features/pre-ictal/"
ID = "PN06-2"

file = file+ID+"/"

# ==============================================================
"In this part we compute the granger causality between EEG and ECG features in the Pre-ictal period"
# ==============================================================

# == electrodes
EEG = ["Delta", "Theta", "Alpha", "Sigma", "Beta"]
ECG = ["LF", "HF", "Ratio"]

# == granger causality application
"causality relation betwwen the EEG features and the LF feature"

index = ["LF", "HF", "Ratio", "Delta", "Theta", "Alpha", "Sigma", "Beta"]
df_T3 = pd.DataFrame(index = index, columns=index)
df_T5 = pd.DataFrame(index = index, columns=index)

print("working on electrode T3 pre-ictal")
# == reading EEG signals
LF, HF, Ratio = read_ecg(file)



# == reading EEG signals
# Delta, Theta, Alpha, Sigma, Beta= read_R(file)
Delta, Theta, Alpha, Sigma, Beta = read_L(file)

# == dataframe preparation
d1 = signal_preparation(LF, HF, Ratio, Delta, Theta, Alpha, Sigma, Beta)


for i in range (int(len(ECG))):
    # print("working on ECG feature\t"+ECG[i])
    for j in range(int(len(EEG))):

        # print("working on ECG feature\t" + EEG[j])
        result = causality(d1, ECG[i], EEG[j], df_T3)

# =================================================
print("working on electrode T5 pre-ictal")

# == reading EEG signals
# Delta, Theta, Alpha, Sigma, Beta= read_R_1(file)
Delta, Theta, Alpha, Sigma, Beta = read_L_1(file)

# == dataframe preparation
d1 = signal_preparation(LF, HF, Ratio, Delta, Theta, Alpha, Sigma, Beta)

# == granger causality application

for i in range (int(len(ECG))):
    print("working on ECG feature\t"+ECG[i])
    for j in range (int(len(EEG))):

        # print("working on ECG feature\t" + EEG[j])
        result = causality(d1, ECG[i], EEG[j], df_T5)

# df_T3 = df_T3.fillna(0)
# df_T3 = df_T3.astype(np.float32)
#
# df_T5 = df_T5.replace(np.nan,0)
# df_T5 = df_T5.astype(np.float32)

# df_T3.to_csv("/home/ftay/Desktop/causality/causality csv/"+ID+"-T3.csv", sep=',', index=False, encoding='utf-8')
# df_T5.to_csv("/home/ftay/Desktop/causality/causality csv/"+ID+"-T5csv", sep=',', index=False, encoding='utf-8')

# ==============================================================
# "In this part we compute the granger causality between EEG and ECG features in the Inter-ictal period"
# # # ==============================================================
print("###############################################################")
print("working on inter-ictal period")


# # == reading EEG signals
file = "/home/ftay/Desktop/causality/features/inter-ictal/"
file = file+ID+"/"

LF, HF, Ratio = read_ecg(file)

# == reading EEG signals
# Delta, Theta, Alpha, Sigma, Beta= read_R(file)
Delta, Theta, Alpha, Sigma, Beta = read_L(file)

df_T3_inter = pd.DataFrame(index = index, columns=index)
df_T5_inter = pd.DataFrame(index = index, columns=index)

# == dataframe preparation
d1 = signal_preparation(LF, HF, Ratio, Delta, Theta, Alpha, Sigma, Beta)

# == electrodes
EEG = ["Delta", "Theta", "Alpha", "Sigma", "Beta"]
ECG = ["LF", "HF", "Ratio"]

# == granger causality application
for i in range (int(len(ECG))):
    print("working on ECG feature\t"+ECG[i])
    for j in range (int(len(EEG))):

        print("working on ECG feature\t" + EEG[j])
        result = causality(d1, ECG[i], EEG[j], df_T3_inter)


print("working on electrode T5 pre-ictal")

# == reading EEG signals
# Delta, Theta, Alpha, Sigma, Beta= read_R_1(file)
Delta, Theta, Alpha, Sigma, Beta = read_L_1(file)

# == dataframe preparation
d1 = signal_preparation(LF, HF, Ratio, Delta, Theta, Alpha, Sigma, Beta)

# == granger causality application

for i in range (int(len(ECG))):
    print("working on ECG feature\t" + ECG[i])
    for j in range (int(len(EEG))):
        print("working on ECG feature\t" + EEG[j])
        result = causality(d1, ECG[i], EEG[j], df_T5_inter)



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# df_T3.fillna(0)
# df_T3 = df_T3.astype(np.float64)

# sns.set_theme(style="ticks", palette="pastel")

df_T3 = df_T3.astype(np.float64)
ax = plt.axes()
sns.heatmap(df_T3, annot=True, vmin=0, vmax=0.05)
sns.set_style("darkgrid")
ax.set(xlabel='', ylabel='y-axis label')
ax.set_title("Pre-ictal causality relation heatmap of electrode T3 of patient "+ID, fontsize = 20)

# plt.show()

# df_T5.fillna(0)
# df_T5 = df_T5.astype(np.float64)

# sns.set_theme(style="ticks", palette="pastel")

df_T5 = df_T5.astype(np.float64)
ax = plt.axes()
sns.heatmap(df_T5, annot=True, vmin=0, vmax=0.05)
sns.set_style("darkgrid")
ax.set(xlabel='', ylabel='y-axis label')
ax.set_title("Pre-ictal causality relation heatmap of electrode T5 of patient "+ID, fontsize = 20)

plt.show()

# =========================
"ploting the inter-ictal results"
# =========================
df_T3_inter = df_T3_inter.astype(np.float64)
ax = plt.axes()
sns.heatmap(df_T3_inter, annot=True, vmin=0, vmax=0.05)
sns.set_style("darkgrid")
ax.set(xlabel='', ylabel='y-axis label')
ax.set_title("Inter-ictal causality relation heatmap of electrode T3 of patient "+ID, fontsize = 20)

plt.show()

# df_T5.fillna(0)
# df_T5 = df_T5.astype(np.float64)


df_T5_inter = df_T5_inter.astype(np.float64)
ax = plt.axes()
sns.heatmap(df_T5_inter, annot=True, vmin=0, vmax=0.05)
sns.set_style("darkgrid")
ax.set(xlabel='', ylabel='y-axis label')
ax.set_title("Inter-ictal causality relation heatmap of electrode T5 of patient "+ID, fontsize = 20)

plt.show()


print(f"Run time {monotonic() - start_time} seconds")

# lag = statsmodels.tsa.vector_ar.var_model.LagOrderResults(array)

# path = "C:\\Users\\Administrator\\OneDrive - Aix-Marseille Université\\causality\\granger\\"
# Path(path).mkdir(parents=True, exist_ok=True)
# np.savetxt(path + 'GC-PN00-3-Fp1-ECG.txt', np.array(cc))
# sig[['Fp1', 'ECG']
