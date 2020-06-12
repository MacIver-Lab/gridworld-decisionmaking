import os, pickle
import numpy as np
import pandas as pd

# Set matrix pring options
np.set_printoptions(precision=3, suppress=True)
# Suppress all warnings
np.seterr(all='ignore')
from scipy.stats import mannwhitneyu, sem, kruskal
import statsmodels.stats.api as sms

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import warnings
warnings.filterwarnings("ignore")

parentDir = os.path.dirname(os.getcwd())
print("Current directory:", parentDir)

NumSimulations = 20
NumEntropy = 10
NumPredators = 5
ForwardSimulations = [100, 1000, 5000]
YSize = 15
XSize = 15

def RejectOutliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def starSignificance(p):
    if p >= 0.05:
        return "n.s."
    elif 0.01 < p < 0.05:
        return "*"
    elif 0.001 < p <= 0.01:
        return "**"
    elif p <= 0.001:
        return "***"

def MakeBoxPlots(axs, xdata, ydata, xerror, yconf, ysem, yrange, facecolor='gray', edgecolor='k',alpha=0.5):
    #if not facecolor:
    #    facecolor = 'gray'
    yconfBoxes = []
    ysemBoxes = []

    if type(facecolor) is list:
        index = 0
        for x, y, xe, yc, ys, color in zip(xdata, ydata, xerror, yconf, ysem, facecolor):
            rectConf = Rectangle((x - xe[0], yc[0]), xe.sum(), yc[1] - yc[0], facecolor='#ffffff', alpha=1.0, edgecolor=color)
            rectSem = Rectangle((x - xe[0], y - ys), xe.sum(), ys * 2, facecolor=color, alpha=alpha, edgecolor=None)
            yconfBoxes.append(rectConf)
            ysemBoxes.append(rectSem)

            yr = np.array([[yrange[index, 0]], [yrange[index, 1]]])
            xe = np.array([[xe[0]], [xe[1]]])

            axs.errorbar(x, y, xerr=xe, yerr=yr, fmt='None', ecolor=color, capsize=10,
                               linewidth=1.5, markeredgewidth=1.5)
            index += 1
    else:
        for x, y, xe, yc, ys in zip(xdata, ydata, xerror, yconf, ysem):
            rectConf = Rectangle((x - xe[0], yc[0]), xe.sum(), yc[1] - yc[0], facecolor='#ffffff', alpha=1.0, edgecolor=edgecolor)
            rectSem = Rectangle((x - xe[0], y - ys), xe.sum(), ys * 2, facecolor=facecolor, alpha=alpha, edgecolor=None)
            yconfBoxes.append(rectConf)
            ysemBoxes.append(rectSem)

        artists = axs.errorbar(xdata, ydata, xerr=xerror.T, yerr=yrange.T, fmt='None', ecolor=edgecolor, capsize=10,
                               linewidths=1.5, markeredgewidth=1.5)

    pcConf = PatchCollection(yconfBoxes, match_original=True)#facecolor='#ffffff', alpha=1.0, edgecolor=edgecolor, linewidths=1.5)
    axs.add_collection(pcConf)
    pcSem = PatchCollection(ysemBoxes, match_original=True) #facecolor=facecolor, alpha=alpha, edgecolor=None)
    axs.add_collection(pcSem)

    return axs

if __name__ == "__main__":

    with open('saved/simulation2.pkl', 'rb') as f:
        _, _, hybrid_data = pickle.load(f)

    ####### Fig 04 e #######
    eigencentralityMoranI = pd.read_csv('saved/eigencentrality_moranmatrix.csv', header=0).values

    lowEntropyMoran = np.mean(eigencentralityMoranI[:, :4], axis=1)
    midEntropyMoran = np.mean(eigencentralityMoranI[:, 4:7], axis=1)
    highEntropyMoran = np.mean(eigencentralityMoranI[:, 7:], axis=1)

    pValLowMid = mannwhitneyu(lowEntropyMoran, midEntropyMoran)[1] * 3
    pValMidHigh = mannwhitneyu(highEntropyMoran, midEntropyMoran)[1] * 3
    pValLowHigh = mannwhitneyu(lowEntropyMoran, highEntropyMoran)[1] * 3

    sigLowMid = starSignificance(pValLowMid)
    sigMidHigh = starSignificance(pValMidHigh)
    sigLowHigh = starSignificance(pValLowHigh)

    fig, axs = plt.subplots(1, 1, figsize=(3, 7))
    plt.setp(axs.spines.values(), linewidth=2)

    yconferror = np.array([list(sms.DescrStatsW(lowEntropyMoran).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(midEntropyMoran).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(highEntropyMoran).tconfint_mean(0.05))])
    ysem = np.array([sem(lowEntropyMoran), sem(midEntropyMoran), sem(highEntropyMoran)])

    xdata = np.array([0.1, 0.25, 0.40])
    ydata = np.array([np.mean(lowEntropyMoran), np.mean(midEntropyMoran), np.mean(highEntropyMoran)])
    xerror = np.array([[0.05, 0.05], [0.05, 0.05], [0.05, 0.05]])
    yrange = np.array([[np.mean(lowEntropyMoran) - np.min(RejectOutliers(lowEntropyMoran)),
                        np.max(RejectOutliers(lowEntropyMoran)) - np.mean(lowEntropyMoran)],
                       [np.mean(midEntropyMoran) - np.min(RejectOutliers(midEntropyMoran)),
                        np.max(RejectOutliers(midEntropyMoran)) - np.mean(midEntropyMoran)],
                       [np.mean(highEntropyMoran) - np.min(RejectOutliers(highEntropyMoran)),
                        np.max(RejectOutliers(highEntropyMoran)) - np.mean(highEntropyMoran)]])

    MakeBoxPlots(axs, xdata, ydata, xerror, yconferror, ysem, yrange)

    axs.grid(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.spines['left'].set_position(('outward', 15))

    axs.set_ylabel('Moran\'s Global I', fontname="Arial", fontsize=14, weight="bold");
    axs.set_ylim((0.12, 0.38))
    axs.set_yticks(np.arange(0.15, 0.40, 0.05/2.))
    axs.yaxis.set_ticklabels((0.15, "", 0.2, "", 0.25, "", 0.3, "", 0.35))
    axs.tick_params(axis='y', direction='in', pad=10)
    yticklabels = axs.get_yticklabels()
    for tick in yticklabels:
        tick.set_fontname("Arial")
        tick.set_fontsize(14)
    axs.yaxis.set_tick_params(width=2, length=7)

    axs.set_xlim((0, 0.5))
    axs.set_xlabel('Entropy', fontname="Arial", fontsize=14, weight="bold");
    axs.set_xticks((0.1, .25, 0.4))
    axs.xaxis.set_ticklabels(("Low", "Med.", "High"))
    axs.tick_params(axis='x', direction='out', pad=8)
    xticklabels = axs.get_xticklabels()
    for tick in xticklabels:
        tick.set_fontname("Arial")
        tick.set_fontsize(14)
    axs.xaxis.set_tick_params(width=2, length=8)

    axs.axhline(y=0.36, xmin=0.2, xmax=0.4, linewidth=0.75, color='k')
    axs.text(0.12, 0.36, sigLowMid, family='sans-serif', fontsize=14)
    axs.axhline(y=0.36, xmin=0.58, xmax=0.8, linewidth=0.75, color='k')
    axs.text(0.32, 0.36, sigMidHigh, family='sans-serif', fontsize=14)
    axs.axhline(y=0.37, xmin=0.2, xmax=0.8, linewidth=0.75, color='k')
    axs.text(0.22, 0.372, sigLowHigh, family='sans-serif', fontsize=14)

    fig.savefig('Plots/fig04_connectivity/fig04e.pdf', bbox_inches='tight')

    ####### Fig 04 e #######
    h_time = hybrid_data[0]

    fig, axs = plt.subplots(1, 1, figsize=(7, 7))

    axs.margins(x=0.08)
    plt.setp(axs.spines.values(), linewidth=2)
    yTime = [np.nanmean(h_time[0], axis=1),
                     np.nanmean(h_time[1], axis=1),
                     np.nanmean(h_time[2], axis=1),
                     np.nanmean(h_time[3], axis=1)]

    y = []
    for i in range(4):
        temp = yTime[i]
        y.append(temp[~np.isnan(temp)])

    ydata = np.array([np.nanmean(h_time[0]),
                      np.nanmean(h_time[1]),
                      np.nanmean(h_time[2]),
                      np.nanmean(h_time[3])])

    yerror = np.array([sem(np.nanmean(h_time[0], axis=1), nan_policy='omit'),
                       sem(np.nanmean(h_time[1], axis=1), nan_policy='omit'),
                       sem(np.nanmean(h_time[2], axis=1), nan_policy='omit'),
                       sem(np.nanmean(h_time[3], axis=1), nan_policy='omit')])


    x = [0, 0.125, 0.4, 0.525]
    w = 0.1
    r_vals = [.718, 0.24, .718, .24]
    b_vals = [.125, .31, .125, .31]
    g_vals = [.145, .545, .145, .545]
    for i in range(len(x)):
        # distribute scatter randomly across whole width of bar
        alphas = np.linspace(0.1, 1, len(y[i]))
        rgba_colors = np.zeros((len(y[i]), 4))
        # for red the first column needs to be one
        rgba_colors[:, 0] = r_vals[i]
        rgba_colors[:, 1] = b_vals[i]
        rgba_colors[:, 2] = g_vals[i]
        # the fourth column needs to be your alphas
        rgba_colors[:, 3] = alphas
        axs.scatter(x[i] + np.random.random(len(y[i])) * w - w / 2 * w - w / 2, y[i], color=rgba_colors, zorder=2)

    axs.bar(x, ydata, w, yerr=yerror, align='center', color='#68228B', zorder=1, alpha=0.1)
    axs.grid(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.spines['bottom'].set_position(('outward', 0))

    axs.set_ylabel('Percent time spent (%))', fontname="Arial", fontsize=14, weight="bold");
    axs.tick_params(axis='y', direction='in', pad=10)
    yticklabels = axs.get_yticklabels()
    for tick in yticklabels:
        tick.set_fontname("Arial")
        tick.set_fontsize(14)
    axs.yaxis.set_tick_params(width=2, length=7)

    axs.set_xticks((0.0625, 0.4625))
    axs.xaxis.set_ticklabels(("Low", "High"))
    axs.tick_params(axis='x', direction='out', pad=8)
    xticklabels = axs.get_xticklabels()
    for tick in xticklabels:
        tick.set_fontname("Arial")
        tick.set_fontsize(14)
    axs.xaxis.set_tick_params(width=2, length=8)
    axs.set_xlabel('Spatial autocorrelaiton of environment eigencentrality', fontname="Arial", fontsize=14,
                   weight="bold")

    fig.savefig("Plots/fig04_connectivity/fig04f.pdf", bbox_inches='tight')

    ####### Fig 04 f #######
    h_survivalrate = hybrid_data[1]

    for i in range(6):
        h_survivalrate[i] = np.asarray(h_survivalrate[i])


    pValLow = kruskal(h_survivalrate[0], h_survivalrate[1], h_survivalrate[2])[1]

    pValPlanningvTrans = mannwhitneyu(h_survivalrate[5], h_survivalrate[4])[1] * 3.


    fig, axs = plt.subplots(1, 1, figsize=(6, 7))
    plt.setp(axs.spines.values(), linewidth=2)

    yconferror = np.array([list(sms.DescrStatsW(h_survivalrate[0]).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(h_survivalrate[1]).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(h_survivalrate[2]).tconfint_mean(0.05))])
    ysem = np.array(
        [sem(h_survivalrate[0]), sem(h_survivalrate[1]), sem(h_survivalrate[2])])

    xdata = np.array([0.1, 0.25, 0.40])
    ydata = np.array([np.mean(h_survivalrate[0]), np.mean(h_survivalrate[1]),np.mean(h_survivalrate[2])])
    xerror = np.array([[0.05, 0.05], [0.05, 0.05], [0.05, 0.05]])
    yrange = np.array([[np.mean(h_survivalrate[0]) - np.min(RejectOutliers(h_survivalrate[0])),
                        np.max(RejectOutliers(h_survivalrate[0])) - np.mean(h_survivalrate[0])],
                       [np.mean(h_survivalrate[1]) - np.min(RejectOutliers(h_survivalrate[1])),
                        np.max(RejectOutliers(h_survivalrate[1])) - np.mean(h_survivalrate[1])],
                       [np.mean(h_survivalrate[2]) - np.min(RejectOutliers(h_survivalrate[2])),
                        np.max(RejectOutliers(h_survivalrate[2])) - np.mean(h_survivalrate[2])]])

    MakeBoxPlots(axs, xdata, ydata, xerror, yconferror, ysem, yrange, facecolor=['#B72025', '#064F8B', '#492A75'])

    yconferror = np.array([list(sms.DescrStatsW(h_survivalrate[3]).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(h_survivalrate[4]).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(h_survivalrate[5]).tconfint_mean(0.05))])
    ysem = np.array([sem(h_survivalrate[3]), sem(h_survivalrate[4]), sem(h_survivalrate[5])])

    xdata = np.array([0.7, 0.85, 1.0])
    ydata = np.array([np.mean(h_survivalrate[3]), np.mean(h_survivalrate[4]), np.mean(h_survivalrate[5])])
    xerror = np.array([[0.05, 0.05], [0.05, 0.05], [0.05, 0.05]])

    yrange = np.array([[np.mean(h_survivalrate[3]) - np.min(RejectOutliers(h_survivalrate[3])),
                        np.max(RejectOutliers(h_survivalrate[3])) - np.mean(h_survivalrate[3])],
                       [np.mean(h_survivalrate[4]) - np.min(RejectOutliers(h_survivalrate[4])),
                        np.max(RejectOutliers(h_survivalrate[4])) - np.mean(h_survivalrate[4])],
                       [np.mean(h_survivalrate[5]) - np.min(RejectOutliers(h_survivalrate[5])),
                        np.max(RejectOutliers(h_survivalrate[5])) - np.mean(h_survivalrate[5])]])

    MakeBoxPlots(axs, xdata, ydata, xerror, yconferror, ysem, yrange, facecolor=['#B72025', '#064F8B', '#492A75'])

    axs.grid(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    # axs.spines['bottom'].set_visible(False)
    axs.spines['left'].set_position(('outward', 15))
    axs.spines['bottom'].set_position(('outward', 5))

    axs.set_ylabel('Survival rate (%)', fontname="Arial", fontsize=14, weight="bold")
    axs.set_ylim((0, 73))
    axs.set_yticks(np.arange(0, 80, 10))
    axs.tick_params(axis='y', direction='in', pad=10)
    yticklabels = axs.get_yticklabels()
    for tick in yticklabels:
        tick.set_fontname("Arial")
        tick.set_fontsize(14)
    axs.yaxis.set_tick_params(width=2, length=7)

    axs.set_xlim((0, 1.06))
    axs.set_xlabel('Spatial autocorrelation of environment eigencentrality', fontname="Arial", fontsize=14, weight="bold")
    axs.set_xticks((.25, 0.85))
    axs.xaxis.set_ticklabels(("Low", "High"))
    axs.tick_params(axis='x', direction='out', pad=8)
    xticklabels = axs.get_xticklabels()
    for tick in xticklabels:
        tick.set_fontname("Arial")
        tick.set_fontsize(14)
    axs.xaxis.set_tick_params(width=2, length=8)

    axs.text(0.12, 28, 'Habit', family='sans-serif', fontsize=14, rotation=90, color='#B72025')
    axs.text(0.26, 28, 'Planning', family='sans-serif', fontsize=14, rotation=90, color='#064F8B')
    axs.text(0.42, 28, 'Hybrid', family='sans-serif', fontsize=14, rotation=90, color='#492A75')

    axs.text(0.685, 29, 'Habit', family='sans-serif', fontsize=14, rotation=90, color='#B72025')
    axs.text(0.865, 42, 'Planning', family='sans-serif', fontsize=14, rotation=90, color='#064F8B')
    axs.text(1.01, 36, 'Hybrid', family='sans-serif', fontsize=14, rotation=90, color='#492A75')

    axs.axhline(y=47, xmin=0.1, xmax=0.37, linewidth=0.75, color='k')
    axs.text(0.22, 47.5, starSignificance(pValLow), family='sans-serif', fontsize=14)
    axs.axhline(y=71, xmin=0.8, xmax=0.95, linewidth=0.75, color='k')
    axs.text(0.9, 71.5, starSignificance(pValPlanningvTrans * 3), family='sans-serif', fontsize=14)

    fig.savefig('Plots/fig04_connectivity/fig04g.pdf', bbox_inches='tight')
