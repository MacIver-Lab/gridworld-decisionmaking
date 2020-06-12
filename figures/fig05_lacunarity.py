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
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Rectangle
import matplotlib.colors as col


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

def GetMeanBenefit(survival_rate):
    numForwardSimDifferenceIndex = ([[j, i] for i, j in zip((range(len(ForwardSimulations)))[: -1],
                                                            (range(len(ForwardSimulations)))[1:])])[::-1]
    benefit = np.full((len(numForwardSimDifferenceIndex), NumSimulations, NumEntropy), np.nan)
    for i, forward in enumerate(numForwardSimDifferenceIndex):
        benefit[i, :, :] = survival_rate[forward[0], :, :] - survival_rate[forward[1], :, :]
        if i == 0:  # Uneven distribution -- assume linear increase
            benefit[i, :, :] = (survival_rate[forward[0], :, :] - survival_rate[forward[1], :, :]) * 2

    return benefit


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

    ####### Fig 05 a #######
    environmentLacunarities = pd.read_csv('saved/environment_lacunarities.csv', header=0).values[:, 1:]

    cmap = col.LinearSegmentedColormap.from_list('custom_lac',
                                          [(0, '#F27070'),
                                           (0.09, '#F27070'),
                                           (0.35, '#6A8055'),
                                           (1, '#1C76BC')], N=256)

    fig, axs = plt.subplots(1, 1, figsize=(7, 7))

    axs.margins(y=0, x=0)
    x = np.linspace(0.1, 0.9, num=9)
    y = np.mean(environmentLacunarities, axis=0)
    yq75, yq25 = np.percentile(environmentLacunarities, [75, 25], axis=0)

    colors = cmap(y/np.max(y))
    axs.scatter(x, y, c=colors, edgecolor='none')
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap)
    lc.set_array(y)
    lc.set_linewidth(1.5)
    axs.add_collection(lc)

    axs.errorbar(x[:3], y[:3], yerr=[(y[:3] - yq25[:3]).tolist(), (yq75[:3] - y[:3]).tolist()],
                                 fmt='none', ecolor='#1C76BC')
    axs.errorbar(x[3:7], y[3:7], yerr=[(y[3:7] - yq25[3:7]).tolist(), (yq75[3:7] - y[3:7]).tolist()],
                 fmt='none', ecolor='#6A8055')
    axs.errorbar(x[7:], y[7:], yerr=[(y[7:] - yq25[7:]).tolist(), (yq75[7:] - y[7:]).tolist()],
             fmt='none', ecolor='#F27070')

    axs.plot([0.1, 0.0], [y[0], 2.5], '--', color='#1C76BC')

    axs.grid(False);
    axs.spines['right'].set_visible(False);
    axs.spines['top'].set_visible(False);
    plt.setp(axs.spines.values(), linewidth=2)

    axs2 = axs.twinx()
    axs2.spines["right"].set_position(("axes", -.1))
    axs2.spines["right"].set_visible(False)
    axs2.grid(False)
    axs2.get_yaxis().set_visible(False)
    axs2.spines['top'].set_visible(False)
    axs2.plot([-0.01, -0.01], [0.03, 0.41], lw=3, color='#F37370')
    axs2.plot([-0.01, -0.01], [1.16, 2.5], lw=3, color='#1B75BC')
    axs2.plot([-0.0175, -0.0175], [0.23, 1.35], lw=3, color='#687F54')
    axs2.set_ylim(-0, 2.5)

    axs.set_ylabel('Ln($_{avg}$)', fontname="Arial", fontsize=14, weight="bold")
    axs.set_ylim(-0, 2.5)
    axs.tick_params(axis='y', direction='in', pad=10)
    yticklabels = axs.get_yticklabels();
    for tick in yticklabels:
        tick.set_fontname("Arial")
        tick.set_fontsize(14);
    axs.yaxis.set_tick_params(width=2, length=7)

    axs.set_xlabel('Entropy', fontname="Arial", fontsize=14, weight="bold");
    axs.set_xticks(x);
    axs.xaxis.set_ticklabels((0.1, "", 0.3, "", 0.5, "", 0.7, "", 0.9))
    axs.tick_params(axis='x', direction='out', pad=8)
    xticklabels = axs.get_xticklabels()
    for tick in xticklabels:
        tick.set_fontname("Arial")
        tick.set_fontsize(14)
    axs.xaxis.set_tick_params(width=2, length=7)

    axs.text(0.15, 2.25, "Coastal water range", family='sans-serif', fontsize=16, color="#1B75BC");
    axs.text(0.5, 1., "Land range", family='sans-serif', fontsize=16, color="#687F54");
    axs.text(0.15, 0.3, "Structured aquatic range", family='sans-serif', fontsize=16, color="#F37370");

    fig.savefig("Plots/fig05_lacunarity/fig05a.pdf", bbox_inches='tight')


    coastalIndices = np.where(environmentLacunarities >= 1.16);
    landIndices = np.where(np.logical_and(environmentLacunarities <= 1.35, environmentLacunarities >= 0.23))
    coralIndices = np.where(environmentLacunarities <= 0.41)

    highPlanningEntropy = environmentLacunarities[:, 3:6]
    yq75, yq25 = np.percentile(highPlanningEntropy.flatten(), [75, 25], axis=0)
    highPlanIndicies = np.where(np.logical_and(highPlanningEntropy <= yq75,
                                               highPlanningEntropy >= yq25))

    ####### Fig 05 b #######
    complexity = pd.read_csv('saved/graph_complexity.csv', header=None).values


    coastalComplexity = complexity[coastalIndices[0], coastalIndices[1] + 1]
    coastalCompelxity = RejectOutliers(np.hstack((coastalComplexity, complexity[:, 0])), 2)
    landComplexity = RejectOutliers(complexity[landIndices[0], landIndices[1] + 1], 2)
    coralComplexity = RejectOutliers(complexity[coralIndices[0], coralIndices[1] + 1], 2)

    highPlanComplexity = RejectOutliers(complexity[highPlanIndicies[0], highPlanIndicies[1] + 4], 2)

    PCoastalL = mannwhitneyu(coastalCompelxity, landComplexity)[1]
    PCoralL = mannwhitneyu(landComplexity, coralComplexity)[1]
    PCC = mannwhitneyu(coastalCompelxity, coralComplexity)[1]

    fig, axs = plt.subplots(1, 1, figsize=(3, 7))
    plt.setp(axs.spines.values(), linewidth=2)

    yconferror = np.array([list(sms.DescrStatsW(coastalComplexity).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(landComplexity).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(coralComplexity).tconfint_mean(0.05))])
    ysem = np.array([sem(coastalComplexity), sem(landComplexity), sem(coralComplexity)])

    xdata = np.array([0.1, 0.25, 0.40])
    ydata = np.array([np.mean(coastalComplexity), np.mean(landComplexity), np.mean(coralComplexity)])
    xerror = np.array([[0.05, 0.05], [0.05, 0.05], [0.05, 0.05]])
    yrange = np.array([[np.mean(coastalComplexity) - np.min(coastalComplexity),
                        np.max(RejectOutliers(coastalComplexity, 2)) - np.mean(coastalComplexity)],
                       [np.mean(landComplexity) - np.min(RejectOutliers(landComplexity, 2)),
                        np.max(RejectOutliers(landComplexity, 2)) - np.mean(landComplexity)],
                       [np.mean(coralComplexity) - np.min(coralComplexity),
                        np.max(RejectOutliers(coralComplexity)) - np.mean(coralComplexity)]])

    MakeBoxPlots(axs, xdata, ydata, xerror, yconferror, ysem, yrange, facecolor=['#1B75BC', '#687F54', '#F37370'])

    axs.plot([0.2, 0.3], [np.mean(highPlanComplexity), np.mean(highPlanComplexity)], '-', color='#392804', lw=2)
    axs.plot([0.25, 0.25], [np.mean(highPlanComplexity) - sem(highPlanComplexity),
                            np.mean(highPlanComplexity) + sem(highPlanComplexity)], '-', color='#392804', lw=2)

    axs.grid(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.spines['left'].set_position(('outward', 15))
    axs.spines['bottom'].set_visible(False)

    axs.set_ylabel('Spatial complexity', fontname="Arial", fontsize=14, weight="bold");
    axs.set_ylim((0.0, 1.1))
    axs.set_yticks(np.arange(0.0, 1.1, 0.1));
    axs.yaxis.set_ticklabels((0, "", .2, "", .4, "", .6, "", .8, "", 1.0))
    axs.tick_params(axis='y', direction='in', pad=10);
    yticklabels = axs.get_yticklabels();
    for tick in yticklabels:
        tick.set_fontname("Arial");
        tick.set_fontsize(14);
    axs.yaxis.set_tick_params(width=2, length=7)

    axs.set_xlim((0, 0.5))
    axs.set_xlabel('Entropy', fontname="Arial", fontsize=14, weight="bold");
    axs.set_xticks((0.1, .25, 0.4))
    axs.xaxis.set_ticklabels(("Open", "Land", "Reef"));
    axs.tick_params(axis='x', direction='out', pad=8);
    xticklabels = axs.get_xticklabels();
    for tick in xticklabels:
        tick.set_fontname("Arial");
        tick.set_fontsize(14);
    axs.xaxis.set_tick_params(width=2, length=8)
    axs.get_xaxis().set_visible(False)
    axs.get_xaxis().set_ticks([])

    axs.axhline(y=1.03, xmin=0.2, xmax=0.4, linewidth=0.75, color='k')
    axs.text(0.12, 1.03, starSignificance(PCoastalL), family='sans-serif', fontsize=14);
    axs.axhline(y=1.03, xmin=0.58, xmax=0.8, linewidth=0.75, color='k')
    axs.text(0.32, 1.03, starSignificance(PCoralL), family='sans-serif', fontsize=14);
    axs.axhline(y=1.09, xmin=0.2, xmax=0.8, linewidth=0.75, color='k')
    axs.text(0.22, 1.095, starSignificance(PCC), family='sans-serif', fontsize=14);

    axs.text(0.12, 0.1, 'Coastal', family='sans-serif', fontsize=14, rotation=90, color='#1B75BC')
    axs.text(0.22, 0.42, 'Land', family='sans-serif', fontsize=14, rotation=90, color='#687F54')
    axs.text(0.38, 0.07, 'Structured Aquatic', family='sans-serif', fontsize=14, rotation=90, color='#F37370')

    fig.savefig("Plots/fig05_lacunarity/fig05b.pdf", bbox_inches='tight')

    ####### Fig 05 c #######
    with open('saved/simulation2.pkl', 'rb') as f:
        planning_data, habit_data, _ = pickle.load(f)

    planning_survival_rate = planning_data[0]
    planning_advantage = GetMeanBenefit(planning_survival_rate)

    benefit = GetMeanBenefit(planning_survival_rate)

    coastalBenefit = np.mean(np.hstack((benefit[:, coastalIndices[0], coastalIndices[1] + 1],
                                        benefit[:, :, 0])), axis=0) * 100.
    landBenefit = np.mean(benefit[:, landIndices[0], landIndices[1] + 1], axis=0) * 100.
    coralBenefit = np.mean(benefit[:, coralIndices[0], coralIndices[1] + 1], axis=0) * 100.
    highPlanBenefit = RejectOutliers(np.mean(benefit[:, highPlanIndicies[0], highPlanIndicies[1] + 4], axis=0) * 100.)

    PCoastalL = mannwhitneyu(coastalBenefit, landBenefit)[1]
    PCoralL = mannwhitneyu(landBenefit, coralBenefit)[1]
    PCC = mannwhitneyu(coastalBenefit, coralBenefit)[1]

    fig, axs = plt.subplots(1, 1, figsize=(3, 7))
    plt.setp(axs.spines.values(), linewidth=2)

    yconferror = np.array([list(sms.DescrStatsW(coastalBenefit).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(landBenefit).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(coralBenefit).tconfint_mean(0.05))])
    ysem = np.array([sem(coastalBenefit), sem(landBenefit), sem(coralBenefit)])

    xdata = np.array([0.1, 0.25, 0.40])
    ydata = np.array([np.mean(coastalBenefit), np.mean(landBenefit), np.mean(coralBenefit)])
    xerror = np.array([[0.05, 0.05], [0.05, 0.05], [0.05, 0.05]])
    yrange = np.array([[np.mean(coastalBenefit) - np.min(RejectOutliers(coastalBenefit)),
                        np.max(RejectOutliers(coastalBenefit)) - np.mean(coastalBenefit)],
                       [np.mean(landBenefit) - np.min(RejectOutliers(landBenefit, 1)),
                        np.max(RejectOutliers(landBenefit)) - np.mean(landBenefit)],
                       [np.mean(coralBenefit) - np.min(RejectOutliers(coralBenefit, 1)),
                        np.max(RejectOutliers(coralBenefit)) - np.mean(coralBenefit)]])

    MakeBoxPlots(axs, xdata, ydata, xerror, yconferror, ysem, yrange, facecolor=['#1B75BC', '#687F54', '#F37370'])

    axs.plot([0.2, 0.3], [np.mean(highPlanBenefit), np.mean(highPlanBenefit)], '-', color='#392804', lw=2)
    axs.plot([0.25, 0.25], [np.mean(highPlanBenefit) - sem(highPlanBenefit),
                            np.mean(highPlanBenefit) + sem(highPlanBenefit)], '-', color='#392804', lw=2)

    axs.grid(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.spines['left'].set_position(('outward', 15))
    axs.spines['bottom'].set_visible(False)

    axs.set_ylabel('Mean change in survival rate (%)', fontname="Arial", fontsize=14, weight="bold");
    axs.set_ylim((-5, 28))
    axs.set_yticks(np.arange(0.0, 30, 2));
    axs.yaxis.set_ticklabels((0, "", 4, "", 8, "", 12, "", 16, "", 20, "", 24, ""))
    axs.tick_params(axis='y', direction='in', pad=10);
    yticklabels = axs.get_yticklabels()
    for tick in yticklabels:
        tick.set_fontname("Arial")
        tick.set_fontsize(14)
    axs.yaxis.set_tick_params(width=2, length=7)

    axs.set_xlim((0, 0.5))
    #axs.set_xlabel('', fontname="Arial", fontsize=14, weight="bold");
    axs.set_xticks((0.1, .25, 0.4))
    axs.xaxis.set_ticklabels(("Open", "Land", "Reef"))
    axs.tick_params(axis='x', direction='out', pad=8)
    xticklabels = axs.get_xticklabels();
    for tick in xticklabels:
        tick.set_fontname("Arial")
        tick.set_fontsize(14)
    axs.xaxis.set_tick_params(width=2, length=8)
    axs.get_xaxis().set_visible(False)
    axs.get_xaxis().set_ticks([])

    axs.axhline(y=25.5, xmin=0.2, xmax=0.4, linewidth=0.75, color='k')
    axs.text(0.12, 25.5, starSignificance(PCoastalL), family='sans-serif', fontsize=14)
    axs.axhline(y=25.5, xmin=0.58, xmax=0.8, linewidth=0.75, color='k')
    axs.text(0.32, 25.5, starSignificance(PCoralL), family='sans-serif', fontsize=14)
    axs.axhline(y=27, xmin=0.2, xmax=0.8, linewidth=0.75, color='k')
    axs.text(0.22, 27.5, starSignificance(PCC), family='sans-serif', fontsize=14)

    axs.text(0.04, -2, 'Coastal', family='sans-serif', fontsize=14, rotation=90, color='#1B75BC')
    axs.text(0.22, -5, 'Land', family='sans-serif', fontsize=14, rotation=90, color='#687F54')
    axs.text(0.44, 6, 'Structured Aquatic', family='sans-serif', fontsize=14, rotation=90, color='#F37370')

    fig.savefig("Plots/fig05_lacunarity/fig05c.pdf", bbox_inches='tight')

    ####### Fig 05 d #######
    planning_survival_rate = planning_survival_rate[-1, :, :]
    habit_survival_rate = habit_data[0]

    coastalPlanning = np.hstack((planning_survival_rate[:, 0],
                                 planning_survival_rate[coastalIndices[0], coastalIndices[1] + 1])) * 100.
    coastalHabit = np.hstack((habit_survival_rate[:, 0],
                              habit_survival_rate[coastalIndices[0], coastalIndices[1] + 1])) * 100.

    landPlanning = planning_survival_rate[landIndices[0], landIndices[1] + 1] * 100.
    landHabit = habit_survival_rate[landIndices[0], landIndices[1] + 1] * 100.

    coralPlanning = planning_survival_rate[coralIndices[0], coralIndices[1] + 1] * 100.
    coralHabit = habit_survival_rate[coralIndices[0], coralIndices[1] + 1] * 100.

    highPlanPlanning = planning_survival_rate[highPlanIndicies[0], highPlanIndicies[1] + 4] * 100.
    highPlanHabit = habit_survival_rate[highPlanIndicies[0], highPlanIndicies[1] + 4] * 100.

    PCoastal = kruskal(coastalPlanning, coastalHabit)[1]
    PLand = kruskal(landPlanning, landHabit)[1]
    PCoral = kruskal(coralPlanning, coralHabit)[1]

    fig, axs = plt.subplots(1, 1, figsize=(3, 7))
    plt.setp(axs.spines.values(), linewidth=2)

    yconferror = np.array([list(sms.DescrStatsW(coastalPlanning).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(landPlanning).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(coralPlanning).tconfint_mean(0.05))])
    ysem = np.array([sem(coastalPlanning), sem(landPlanning), sem(coralPlanning)])

    xdata = np.array([0.1, 0.9, 1.7])
    ydata = np.array([np.mean(coastalPlanning), np.mean(landPlanning), np.mean(coralPlanning)])
    xerror = np.array([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])
    yrange = np.array([[np.mean(coastalPlanning) - np.min(RejectOutliers(coastalPlanning)),
                        np.max(RejectOutliers(coastalPlanning)) - np.mean(coastalPlanning)],
                       [np.mean(landPlanning) - np.min(RejectOutliers(landPlanning)),
                        np.max(RejectOutliers(landPlanning)) - np.mean(landPlanning)],
                       [np.mean(coralPlanning) - np.min(RejectOutliers(coralPlanning)),
                        np.max(RejectOutliers(coralPlanning)) - np.mean(coralPlanning)]])

    MakeBoxPlots(axs, xdata, ydata, xerror, yconferror, ysem, yrange, facecolor="#064F8B", edgecolor="#064F8B")

    axs.plot([0.85, 0.95], [np.mean(highPlanPlanning), np.mean(highPlanPlanning)], '-', color='#392804', lw=2)
    axs.plot([0.9, 0.9], [np.mean(highPlanPlanning) - sem(highPlanPlanning),
                          np.mean(highPlanPlanning) + sem(highPlanPlanning)], '-', color='#392804', lw=2)

    yconferror = np.array([list(sms.DescrStatsW(coastalHabit).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(landHabit).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(coralHabit).tconfint_mean(0.05))])
    ysem = np.array([sem(coastalHabit), sem(landHabit), sem(coralHabit)])

    xdata = np.array([0.4, 1.2, 2.0])
    ydata = np.array([np.mean(coastalHabit), np.mean(landHabit), np.mean(coralHabit)])
    xerror = np.array([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])
    yrange = np.array([[np.mean(coastalHabit) - np.min(RejectOutliers(coastalHabit)),
                        np.max(RejectOutliers(coastalHabit)) - np.mean(coastalHabit)],
                       [np.mean(landHabit) - np.min(RejectOutliers(landHabit)),
                        np.max(RejectOutliers(landHabit)) - np.mean(landHabit)],
                       [np.mean(coralHabit) - np.min(RejectOutliers(coralHabit)),
                        np.max(RejectOutliers(coralHabit, 2)) - np.mean(coralHabit)]])

    MakeBoxPlots(axs, xdata, ydata, xerror, yconferror, ysem, yrange, facecolor="#B72025", edgecolor="#B72025")

    axs.plot([1.15, 1.25], [np.mean(highPlanHabit), np.mean(highPlanHabit)], '-', color='#392804', lw=2)
    axs.plot([1.2, 1.2], [np.mean(highPlanHabit) - sem(highPlanHabit),
                          np.mean(highPlanHabit) + sem(highPlanHabit)], '-', color='#392804', lw=2)

    # axs.plot(0.25, np.mean(maxPerformanceBenefit), 'o', color='#27341E')

    axs.grid(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    # axs.spines['bottom'].set_visible(False)
    axs.spines['left'].set_position(('outward', 15))
    axs.spines['bottom'].set_position(('outward', 10))

    axs.set_ylabel('Survival rate (%)', fontname="Arial", fontsize=14, weight="bold");
    axs.tick_params(axis='y', direction='in', pad=10);
    yticklabels = axs.get_yticklabels();
    for tick in yticklabels:
        tick.set_fontname("Arial");
        tick.set_fontsize(14);
    axs.yaxis.set_tick_params(width=2, length=7)
    axs.set_ylim(0, 65)

    axs.set_xlabel('', fontname="Arial", fontsize=14, weight="bold");
    axs.set_xticks((0.25, 1.1, 1.85))
    axs.xaxis.set_ticklabels(("Open", "Land", "Reef"));
    axs.tick_params(axis='x', direction='out', pad=8);
    xticklabels = axs.get_xticklabels();
    for tick in xticklabels:
        tick.set_fontname("Arial");
        tick.set_fontsize(14);
    axs.xaxis.set_tick_params(width=2, length=8)

    axs.axhline(y=34, xmin=0.05, xmax=0.25, linewidth=0.75, color='k')
    axs.text(0.1, 34.5, starSignificance(PCoastal), family='sans-serif', fontsize=14);
    axs.axhline(y=63.8, xmin=0.4, xmax=0.6, linewidth=0.75, color='k')
    axs.text(0.9, 64, starSignificance(PLand), family='sans-serif', fontsize=14);
    axs.axhline(y=51, xmin=0.75, xmax=0.95, linewidth=0.75, color='k')
    axs.text(1.7, 51.5, starSignificance(PCoral), family='sans-serif', fontsize=14);

    axs.text(0.35, 38, 'Habit', family='sans-serif', fontsize=14, rotation=90, color='#B72025')
    axs.text(0.03, 38, 'Planning', family='sans-serif', fontsize=14, rotation=90, color='#064F8B')

    axs.text(1.05, 41, 'Habit', family='sans-serif', fontsize=14, rotation=90, color='#B72025')
    axs.text(0.6, 54, 'Planning', family='sans-serif', fontsize=14, rotation=90, color='#064F8B')

    axs.text(2.08, 23, 'Habit', family='sans-serif', fontsize=14, rotation=90, color='#B72025')
    axs.text(1.4, 28, 'Planning', family='sans-serif', fontsize=14, rotation=90, color='#064F8B')

    fig.savefig("Plots/fig05_lacunarity/fig05d.pdf", bbox_inches='tight')