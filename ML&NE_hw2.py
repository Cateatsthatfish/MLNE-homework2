import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.time_frequency import tfr_multitaper
from mne.viz.utils import center_cmap

# [task 2]import .cnt data
# [question] 这里只用了ses-01 的数据，但是实际上sub-002还有其他的数据，并不知道怎么放在一起处理-》分开出去取平均？
fname = "sub-002_ses-01_task-motorimagery_eeg.cnt"
raw = mne.io.read_raw_cnt(input_fname=fname, date_format='dd/mm/yy', preload=True)
# raw.info # 用来显示基本的数据

# sensors （before set montage)
raw.plot_sensors()
plt.show()

"""
set montage from task-motorimagery_electrodes.tsv
[source] 林沛阳
"""
import csv


def electrodes_read(file_path: str, column_names: list) -> list:
    csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    with open(file_path, "r") as wf:
        reader = csv.DictReader(wf, fieldnames=column_names, dialect='tsv_dialect')
        datas = []
        for row in reader:
            datas.append(row)
    csv.unregister_dialect('tsv_dialect')
    dic = {}
    for data in datas:
        if data["name"] == "name":
            continue
        x = float(data["x"]) / 1000
        y = -float(data["y"]) / 1000
        z = float(data["z"]) / 1000
        dic[data["name"]] = [y, x, z]

    montage = mne.channels.make_dig_montage(ch_pos=dic)
    return montage


montage = electrodes_read("task-motorimagery_electrodes.tsv", ["name", "x", "y", "z"])
raw.set_montage(montage)

# [task 3]Plot the time course of raw EEG signals with 10-second window
raw.plot(start=0, duration=10, n_channels=65)
plt.show()  # 保持界面显示并且能实现交互，而不是只生成一张png

# 查看功率谱的密度 raw
raw.plot_psd(average=True)

"""
[task 4] Data preprocessing, artifacts removal
[source] https://blog.csdn.net/LiDLMU/article/details/109291216
"""
# 坏道处理
# [question] 坏道的判断并不是很清楚，从time course那里的线有一些都是断的，所以就盲选了两个
raw.info["bads"] = ['VEO', 'HEO']
raw.interpolate_bads()

# 0.1-40Hz 滤波
raw_pass = raw.copy().filter(l_freq=0.1, h_freq=40)
# 查看功率谱的密度 raw_pass (这里有一个比较大的变化)
# raw_pass.plot_psd(average = True)

# 50 Hz notch filter
raw_notch = raw_pass.notch_filter(freqs=50)
# 查看功率谱的密度 raw_notch
# raw_notch.plot_psd(average = True)

from mne.preprocessing import (ICA)

# ICA独立成分分析
ica = ICA(n_components=30, random_state=97)
ica.fit(raw_notch)
# ica.plot_sources(raw_notch) # 这里图变得很奇怪

# 手动选择眼电，心电伪影
ica.exclude = [0, 1]
reconst_raw = raw_notch.copy()
ica.apply(reconst_raw)

# [task 5]time course （在这里线变得规整起来了）
reconst_raw.plot(start=0, duration=10, n_channels=65)
# 比较处理前后的数据
ica.plot_overlay(raw, exclude=[0, 1])

"""
[task 6] Plot the time-frequency maps of the subject
# Authors: Clemens Brunner <clemens.brunner@gmail.com>
# source:https://mne.tools/stable/auto_examples/time_frequency/plot_time_frequency_erds.html?highlight=erd
# License: BSD (3-clause)
"""

# [undo] 偷了改名字的懒
raw = reconst_raw

# events : hand & elbows
# 来自预处理后的 task-motorimagery 的event
events, _ = mne.events_from_annotations(raw)
# [question-undo] 这里的events偶尔会出现大小不对的情况，可能要删掉
# del events["response_time"]
picks = mne.pick_channels(raw.info["ch_names"], ["C3", "CZ", "C4"])

# events_rest : rest
# 来自什么处理都没做的 sub-002_ses-16_task-rest_eeg.cnt
# 因为上面不包含rest，所以等距构造了一个
# 1810唐家豪 ： 事件间隔基本都是8000ms，4000是事件，剩的就是休息
fname_rest = "sub-002_ses-16_task-rest_eeg.cnt"
raw_rest = mne.io.read_raw_cnt(input_fname=fname_rest, date_format='dd/mm/yy', preload=True)
events_rest = mne.make_fixed_length_events(raw_rest, id=3, start=1.2, duration=4.0, first_samp=True, overlap=0.0)
events_merge = np.append(events, events_rest, axis=0)
event_m_ids = dict(hands=1, elbow=2, rest=3)

tmin, tmax = -1, 4  # define epochs around events (in s)
epochs = mne.Epochs(raw, events_merge, event_m_ids, tmin - 0.5, tmax + 0.5,
                    picks=picks, baseline=None, preload=True)
# compute ERDS maps ###########################################################
freqs = np.arange(2, 36, 1)  # frequencies from 2-35Hz
n_cycles = freqs  # use constant t/f resolution
vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
baseline = [-1, 0]  # baseline interval (in s)
cmap = center_cmap(plt.cm.RdBu, vmin, vmax)  # zero maps to white
kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
              buffer_size=None, out_type='mask')  # for cluster test

# Run TF decomposition overall epochs
tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                     use_fft=True, return_itc=False, average=False,
                     decim=2)
tfr.crop(tmin, tmax)
tfr.apply_baseline(baseline, mode="percent")
# for event in event_ids:
for event in event_m_ids:
    # select desired epochs for visualization
    tfr_ev = tfr[event]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4),
                             gridspec_kw={"width_ratios": [10, 10, 10, 1]})
    for ch, ax in enumerate(axes[:-1]):  # for each channel
        # positive clusters
        _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=1, **kwargs)
        # negative clusters
        _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=-1,
                                     **kwargs)

        # note that we keep clusters with p <= 0.05 from the combined clusters
        # of two independent tests; in this example, we do not correct for
        # these two comparisons
        c = np.stack(c1 + c2, axis=2)  # combined clusters
        p = np.concatenate((p1, p2))  # combined p-values
        mask = c[..., p <= 0.05].any(axis=-1)

        # plot TFR (ERDS map with masking)
        tfr_ev.average().plot([ch], vmin=vmin, vmax=vmax, cmap=(cmap, False),
                              axes=ax, colorbar=False, show=False, mask=mask,
                              mask_style="mask")

        ax.set_title(epochs.ch_names[ch], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
        if ch != 0:
            ax.set_ylabel("")
            ax.set_yticklabels("")
    fig.colorbar(axes[0].images[-1], cax=axes[-1])
    fig.suptitle("ERDS ({})".format(event))
    fig.show()

# [task 7] Plot the topographical distribution of power of the subject
# [undo] 没有把各个event分开来处理
# Averaged topographical distribution of power
bands = [(10, 12, 'Upper Alpha'), (23, 25, 'Upper Beta')]
epochs.plot_psd_topomap(bands=bands, ch_type='eeg')

"""
[task 8] Comparison of power (in dB) changes with time (in s) during hand, elbow motor imagery, and resting state for electrode C3, and electrode C4
[source] 杨峻锋
"""


def power_compare(pickC, title):
    # epochsC3 = mne.Epochs(raw,events,event_ids,tmin,tmax,picks = pickC3, baseline = None, preload = True)
    epochsC3 = mne.Epochs(raw, events_merge, event_m_ids, tmin, tmax, picks=pickC, baseline=None, preload=True)
    frequencies = np.arange(10, 12, 1)
    nc = frequencies / 2
    epochs_hand = epochsC3["hands"]
    epochs_elbow = epochsC3["elbow"]
    epochs_rest = epochsC3["rest"]

    frequencies = np.arange(10, 12, 1)
    nc = frequencies / 2

    powerHand = mne.time_frequency.tfr_morlet(epochs_hand, n_cycles=nc, return_itc=False, freqs=frequencies, decim=3)
    powerElbow = mne.time_frequency.tfr_morlet(epochs_elbow, n_cycles=nc, return_itc=False, freqs=frequencies, decim=3)
    powerRest = mne.time_frequency.tfr_morlet(epochs_rest, n_cycles=nc, return_itc=False, freqs=frequencies, decim=3)
    power_freHand = powerHand.data.sum(axis=1)
    power_freElbow = powerElbow.data.sum(axis=1)
    power_freRest = powerRest.data.sum(axis=1)

    plt.plot(powerHand.times, np.log(power_freHand[0, :]), label="Hand")
    plt.plot(powerElbow.times, np.log(power_freElbow[0, :]), label="Elbow")
    plt.plot(powerRest.times, np.log(power_freRest[0, :]), label="Rest")

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.title(title)


pickC3 = mne.pick_channels(raw.info["ch_names"], ["C3"])
power_compare(pickC3, "C3")

pickC4 = mne.pick_channels(raw.info["ch_names"], ["C4"])
power_compare(pickC4, "C4")
