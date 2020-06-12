import os, sys, pickle, glob
import numpy as np
import pandas as pd
from collections import Counter

# Set matrix pring options
np.set_printoptions(precision=3, suppress=True)
# Suppress all warnings
np.seterr(all='ignore')
from scipy.stats import mannwhitneyu, sem, kruskal
import statsmodels.stats.api as sms

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Ellipse

from common.coord import COORD

parentDir = os.path.dirname(os.getcwd())
print("Current directory:", parentDir)

NumSimulations = 20
NumEntropy = 10
NumPredators = 5
ForwardSimulations = [100, 1000, 5000]
YSize = 15
XSize = 15


def GetSurvivalRate():
    survivalRate = np.full((len(ForwardSimulations), NumPredators, NumSimulations, NumEntropy), np.nan)
    for simulationInd in range(NumSimulations):
        for entropyInd in range(NumEntropy):
            sys.stdout.write('\r' + "Simulation #: %d, Entropy: .%d" % (simulationInd, entropyInd))
            for predatorInd in range(NumPredators):
                if not os.path.exists(parentDir + '/Simulation 2 Pseudo-Terrestrial/planning/Data/Simulation_%d/Occlusion_%d/Predator_%d/' % (simulationInd, entropyInd, predatorInd)):
                    continue
                for findex in range(len(ForwardSimulations)):
                    if ForwardSimulations[findex] == 5000:
                        episodeFolder = parentDir + '/Simulation 2 Pseudo-Terrestrial/planning/Data/Simulation_%d/Occlusion_%d/Predator_%d/Depth_%d/Episode_*.csv' % (simulationInd, entropyInd, predatorInd, ForwardSimulations[findex])
                        episodeFiles = glob.glob(episodeFolder)

                        terminalRewards = []
                        for episodeFile in episodeFiles:
                            episode = pd.read_csv(episodeFile, header=0)
                            if episode['Agent X'].iloc[-1] == 7 and episode['Agent Y'].iloc[-1] == 14:
                                terminalRewards.append(999)
                            else:
                                terminalRewards.append(episode['Reward'].iloc[-1])
                    else:
                        summaryFile = parentDir + '/Simulation 2 Pseudo-Terrestrial/planning/Data/Simulation_%d/Occlusion_%d/Predator_%d/Summary.csv' % (
                        simulationInd, entropyInd, predatorInd)
                        summary = pd.read_csv(summaryFile, header=0)
                        terminalRewards = summary['Depth %d' % (ForwardSimulations[findex])].values

                    if len(terminalRewards) > 0:
                        counts = Counter(terminalRewards)
                        try:
                            survivalRate[findex, predatorInd, simulationInd, entropyInd] = float(counts[999]) / float(sum(counts.values()) - counts[-1])
                        except ZeroDivisionError:
                            if counts[999] == 0:
                                survivalRate[findex, predatorInd, simulationInd, entropyInd] = 0.

    return survivalRate


def GetSurvivalTrajectoryandSpread(simulationInd, entropyInd, predators):
    if isinstance(predators, int):
        predators = [predators]

    survivalTrajectory = np.zeros((YSize, XSize))

    for predator in predators:
        episodeFolder = parentDir + '/Simulation 2 Pseudo-Terrestrial/planning/Data/Simulation_%d/Occlusion_%d/Predator_%d/Depth_5000/Episode_*.csv' % (
        simulationInd, entropyInd, predator)
        episodeFiles = glob.glob(episodeFolder)

        for episodeFile in episodeFiles:
            episode = pd.read_csv(episodeFile, header=0)
            terminalReward = episode['Reward'].iloc[-1]

            if terminalReward < 0 or not (episode['Agent X'].iloc[-1] == 7 and episode['Agent Y'].iloc[-1] == 14):
                continue

            subset = episode[['Agent X', 'Agent Y']]
            episodeTrajectory = [COORD(x[0], x[1]) for x in subset.values]
            episodeTrajectory = list(set(episodeTrajectory))

            for coord in episodeTrajectory:
                survivalTrajectory[coord.X, coord.Y] += 1

    occlusionFile = parentDir + '/Simulation 2 Pseudo-Terrestrial/planning/Data/Simulation_%d/Occlusion_%d/OcclusionCoordinates.csv' % (simulationInd, entropyInd)
    occlusion = pd.read_csv(occlusionFile, header=0)
    subset = occlusion[['X', 'Y']]
    occlusionCoordinates = [COORD(x[0], x[1]) for x in subset.values]
    occlusionCoordinates = list(set(occlusionCoordinates))

    for occlusionCoord in occlusionCoordinates:
        survivalTrajectory[occlusionCoord.X, occlusionCoord.Y] = np.nan

    totalTrajectory = survivalTrajectory[~np.isnan(survivalTrajectory)]
    trajectoryOccupancy = np.nan
    if totalTrajectory.any():
        numCellsVisited = np.count_nonzero(totalTrajectory)
        trajectoryOccupancy = float(numCellsVisited) / float(XSize * YSize - len(occlusionCoordinates))

    return trajectoryOccupancy

def GetMeanBenefit(survival_rate):
    numForwardSimDifferenceIndex = ([[j, i] for i, j in zip((range(len(ForwardSimulations)))[: -1],
                                                            (range(len(ForwardSimulations)))[1:])])[::-1]
    benefit = np.full((len(numForwardSimDifferenceIndex), NumSimulations, NumEntropy), np.nan)
    for i, forward in enumerate(numForwardSimDifferenceIndex):
        benefit[i, :, :] = survival_rate[forward[0], :, :] - survival_rate[forward[1], :, :]
        if i == 0:  # Uneven distribution -- assume linear increase
            benefit[i, :, :] = (survival_rate[forward[0], :, :] - survival_rate[forward[1], :, :]) * 2

    lowEntropyBenefit = np.nanmean(np.nanmean(benefit[:, :, :4], axis=2) * 100, axis=0)
    midEntropyBenefit = np.nanmean(np.nanmean(benefit[:, :, 4:7], axis=2) * 100, axis=0)
    highEntropyBenefit = np.nanmean(np.nanmean(benefit[:, :, 7:], axis=2) * 100, axis=0)

    return [lowEntropyBenefit, midEntropyBenefit, highEntropyBenefit]

def RejectOutliers(data, m=2, type='a'):
    if type == 'a':
        return data[abs(data - np.nanmean(data)) < m * np.std(data)]
    elif type == 'm':
        return data[(data - np.nanmean(data)) < m * np.std(data)]
    elif type == 's':
        return data[(np.nanmean(data) - data) < m * np.std(data)]

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
    yconfBoxes = []
    ysemBoxes = []

    for x, y, xe, yc, ys in zip(xdata, ydata, xerror, yconf, ysem):
        rectConf = Rectangle((x - xe[0], yc[0]), xe.sum(), yc[1] - yc[0])
        rectSem = Rectangle((x - xe[0], y - ys), xe.sum(), ys * 2)
        yconfBoxes.append(rectConf)
        ysemBoxes.append(rectSem)

    artists = axs.errorbar(xdata, ydata, xerr=xerror.T, yerr=yrange.T, fmt='None', ecolor=edgecolor, capsize=10,
                           linewidths=1.5, markeredgewidth=1.5)

    pcConf = PatchCollection(yconfBoxes, facecolor='#ffffff', alpha=1.0, edgecolor=edgecolor, linewidths=1.5)
    axs.add_collection(pcConf)
    pcSem = PatchCollection(ysemBoxes, facecolor=facecolor, alpha=alpha, edgecolor=None)
    axs.add_collection(pcSem)

    return artists

if __name__ == "__main__":

    with open('saved/simulation2.pkl', 'rb') as f:
        planning_data, habit_data, _ = pickle.load(f)

    ####### Fig 03 a #######
    survival_rate = planning_data[0]

    fig, axs = plt.subplots(1, 1, figsize=(7, 7))
    colors = ["#4294B4", "#3F7689", "#38545F"]
    axs.margins(y=0.005)

    x = np.linspace(0.0, 0.9, num=10)

    for acuity in range(len(ForwardSimulations)):  # 2, len(forwardSim)
        y = np.nanmean(survival_rate[acuity, :, :], axis=0) * 100
        ysem = sem(survival_rate[acuity, :, :],  axis=0) * 100

        ymin = y - ysem
        ymin[ymin < 0] = 0
        ymax = y + ysem
        ymax[ymax > 100] = 100

        axs.plot(x, y, 'o-', linewidth=2, color=colors[-1]);
        axs.fill_between(x, ymax, ymin, alpha=0.25, color=colors[acuity], linewidth=0);

    axs.grid(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    plt.setp(axs.spines.values(), linewidth=2)

    axs.set_ylabel('Survival rate (%)', fontname="Arial", fontsize=14, weight="bold");
    axs.set_yticks((0, 5, 10, 15, 20, 25, 30, 35, 40, 45));
    axs.tick_params(axis='y', direction='in', pad=10)
    yticklabels = axs.get_yticklabels();
    for tick in yticklabels:
        tick.set_fontname("Arial")
        tick.set_fontsize(14)
    axs.yaxis.set_tick_params(width=2, length=7)

    axs.set_xlabel('Entropy', fontname="Arial", fontsize=14, weight="bold");
    axs.set_xticks(x)
    axs.xaxis.set_ticklabels((0.0, "", 0.2, "", 0.4, "", 0.6, "", 0.8));
    axs.tick_params(axis='x', direction='out', pad=8)
    xticklabels = axs.get_xticklabels()
    for tick in xticklabels:
        tick.set_fontname("Arial")
        tick.set_fontsize(14)
    axs.xaxis.set_tick_params(width=2, length=7)

    legendrects = []
    legendcircles = []
    recty = 40
    gap = 4.15

    revAcuity = list(reversed(ForwardSimulations))
    revColors = list(reversed(colors))
    for acuity in range(3):
        rect = Rectangle((0.05, recty), 0.075, 4)
        circ = Ellipse((0.05 + (0.075 / 2), recty + 2), width=0.02, height=1);
        legendrects.append(rect)
        legendcircles.append(circ)

        axs.axhline(y=recty + 2, xmin=0.1, xmax=0.168, color=revColors[acuity]);
        axs.text(0.13, recty + 1.5, str(revAcuity[acuity]), family='sans-serif', fontsize=14);

        recty -= gap;

    pcRect = PatchCollection(legendrects, facecolor=revColors, alpha=0.25)
    pcCirc = PatchCollection(legendcircles, facecolor=revColors)
    axs.add_collection(pcRect)
    axs.add_collection(pcCirc)
    axs.text(0.05, 45, "Num. forward simulated", family='sans-serif', fontsize=14, weight="bold")

    fig.savefig('Plots/fig03_clutter/fig03a.pdf', bbox_inches='tight')


    ####### Fig 03 b #######
    planning_advantage = GetMeanBenefit(survival_rate)

    pValLowMid = mannwhitneyu(planning_advantage[0], planning_advantage[1])[1] * 3.
    pValMidHigh = mannwhitneyu(planning_advantage[1], planning_advantage[2])[1] * 3.
    pValLowHigh = mannwhitneyu(planning_advantage[0], planning_advantage[2])[1] * 3.

    fig, axs = plt.subplots(1, 1, figsize=(3, 7))
    plt.setp(axs.spines.values(), linewidth=2)

    yconferror = np.array([list(sms.DescrStatsW(planning_advantage[0]).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(planning_advantage[1]).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(planning_advantage[2]).tconfint_mean(0.05))])
    ysem = np.array([sem(planning_advantage[0]), sem(planning_advantage[1]), sem(planning_advantage[2])])

    xdata = np.array([0.1, 0.25, 0.40])
    ydata = np.array([np.mean(planning_advantage[0]), np.mean(planning_advantage[1]), np.mean(planning_advantage[2])])
    xerror = np.array([[0.05, 0.05], [0.05, 0.05], [0.05, 0.05]])
    yrange = np.array([[np.mean(planning_advantage[0]) - np.min(RejectOutliers(planning_advantage[0])),
                        np.max(RejectOutliers(planning_advantage[0])) - np.mean(planning_advantage[0])],
                       [np.mean(planning_advantage[1]) - np.min(RejectOutliers(planning_advantage[1])),
                        np.max(RejectOutliers(planning_advantage[1])) - np.mean(planning_advantage[1])],
                       [np.mean(planning_advantage[2]) - np.min(RejectOutliers(planning_advantage[2], 1)),
                        np.max(RejectOutliers(planning_advantage[2], 1)) - np.mean(planning_advantage[2])]])

    MakeBoxPlots(axs, xdata, ydata, xerror, yconferror, ysem, yrange)

    axs.grid(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.spines['left'].set_position(('outward', 15))
    axs.spines['bottom'].set_position(('outward', 15))

    axs.set_ylabel('Mean change in survival rate (%)', fontname="Arial", fontsize=14, weight="bold");
    axs.set_ylim((-1, 25))
    axs.set_yticks(np.arange(0, 24, 2))
    axs.yaxis.set_ticklabels((0, "", 4, "", 8, "", 12, "", 16, "", 20, ""))
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

    axs.axhline(y=22.3, xmin=0.2, xmax=0.4, linewidth=0.75, color='k')
    axs.text(0.12, 22.3, starSignificance(pValLowMid), family='sans-serif', fontsize=14);
    axs.axhline(y=22.3, xmin=0.58, xmax=0.8, linewidth=0.75, color='k')
    axs.text(0.32, 22.3, starSignificance(pValMidHigh), family='sans-serif', fontsize=14);
    axs.axhline(y=24, xmin=0.2, xmax=0.8, linewidth=0.75, color='k')
    axs.text(0.22, 24.1, starSignificance(pValLowHigh), family='sans-serif', fontsize=14);

    fig.savefig('Plots/fig03_clutter/fig03b.pdf', bbox_inches='tight')

    ####### Fig 03 c #######
    complexity = pd.read_csv('saved/graph_complexity.csv', header=None).values

    fig, axs = plt.subplots(1, 1, figsize=(7, 7))
    axs.margins(y=0.005, x=0.005)

    x = np.linspace(0.0, 0.9, num=10)
    y = np.mean(complexity, axis=0)
    yq75, yq25 = np.percentile(complexity, [75, 25], axis=0)
    axs.errorbar(x, y, yerr=[(y - yq25).tolist(), (yq75 - y).tolist()],
                 fmt='-o', color='k', linewidth=1.5, markersize=8)

    axs.grid(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    plt.setp(axs.spines.values(), linewidth=2)

    axs.set_ylabel('Network degree complexity', fontname="Arial", fontsize=14, weight="bold");
    axs.set_yticks(np.arange(0.0, 1.0, 0.1))
    axs.yaxis.set_ticklabels(("", 0.1, "", 0.3, "", 0.5, "", 0.7, "", 0.9));
    axs.tick_params(axis='y', direction='in', pad=10)
    yticklabels = axs.get_yticklabels()
    for tick in yticklabels:
        tick.set_fontname("Arial")
        tick.set_fontsize(14)
    axs.yaxis.set_tick_params(width=2, length=7)

    axs.set_xlabel('Entropy', fontname="Arial", fontsize=14, weight="bold");
    axs.set_xticks(x)
    axs.xaxis.set_ticklabels((0.0, "", 0.2, "", 0.4, "", 0.6, "", 0.8));
    axs.tick_params(axis='x', direction='out', pad=8)
    xticklabels = axs.get_xticklabels()
    for tick in xticklabels:
        tick.set_fontname("Arial")
        tick.set_fontsize(14)
    axs.xaxis.set_tick_params(width=2, length=7)
    axs.set_xlim((-0.005, 0.92))

    fig.savefig('Plots/fig03_clutter/fig03c.pdf')

    ####### Fig 03 h #######
    lowEntropyOccupancy = planning_data[1][:, :4].flatten()
    midEntropyOccupancy =  planning_data[1][:, 4:7].flatten()
    highEntropyOccupancy =  planning_data[1][:, 7:].flatten()

    lowEntropyOccupancy = RejectOutliers(lowEntropyOccupancy[~np.isnan(lowEntropyOccupancy)])
    midEntropyOccupancy = midEntropyOccupancy[~np.isnan(midEntropyOccupancy)]
    highEntropyOccupancy = RejectOutliers(highEntropyOccupancy[~np.isnan(highEntropyOccupancy)])

    pValLowMid = mannwhitneyu(lowEntropyOccupancy, midEntropyOccupancy)[1] * 3.
    pValMidHigh = mannwhitneyu(highEntropyOccupancy, midEntropyOccupancy)[1] * 3.
    pValLowHigh = mannwhitneyu(lowEntropyOccupancy, highEntropyOccupancy)[1] * 3.

    sigLowMid = starSignificance(pValLowMid)
    sigMidHigh = starSignificance(pValMidHigh)
    sigLowHigh = starSignificance(pValLowHigh)

    fig, axs = plt.subplots(1, 1, figsize=(3, 7))
    plt.setp(axs.spines.values(), linewidth=2)

    yconferror = np.array([list(sms.DescrStatsW(lowEntropyOccupancy).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(midEntropyOccupancy).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(highEntropyOccupancy).tconfint_mean(0.05))])
    ysem = np.array([sem(lowEntropyOccupancy), sem(midEntropyOccupancy), sem(highEntropyOccupancy)])

    xdata = np.array([0.1, 0.25, 0.40])
    ydata = np.array([np.mean(lowEntropyOccupancy), np.mean(midEntropyOccupancy), np.mean(highEntropyOccupancy)])
    xerror = np.array([[0.05, 0.05], [0.05, 0.05], [0.05, 0.05]])
    yrange = np.array([[np.mean(lowEntropyOccupancy) - np.min(RejectOutliers(lowEntropyOccupancy)),
                        np.max(RejectOutliers(lowEntropyOccupancy, 1)) - np.mean(lowEntropyOccupancy)],
                       [np.mean(midEntropyOccupancy) - np.min(RejectOutliers(midEntropyOccupancy, 1)),
                        np.max(RejectOutliers(midEntropyOccupancy, 1)) - np.mean(midEntropyOccupancy)],
                       [np.mean(highEntropyOccupancy) - np.min(RejectOutliers(highEntropyOccupancy, 1)),
                        np.max(RejectOutliers(highEntropyOccupancy, 1)) - np.mean(highEntropyOccupancy)]])

    MakeBoxPlots(axs, xdata, ydata, xerror, yconferror, ysem, yrange, facecolor='#064F8B', edgecolor='#064F8B')

    axs.grid(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.spines['left'].set_position(('outward', 15))
    axs.spines['bottom'].set_position(('outward', 15))

    axs.set_ylabel('Space occupied (%)', fontname="Arial", fontsize=14, weight="bold")
    axs.set_ylim((0.15, 1.0))
    axs.set_yticks(np.arange(0.1, 1.1, .1))
    axs.tick_params(axis='y', direction='in', pad=10)
    yticklabels = axs.get_yticklabels()
    for tick in yticklabels:
        tick.set_fontname("Arial")
        tick.set_fontsize(14)
    axs.yaxis.set_tick_params(width=2, length=7)

    axs.set_xlim((0, 0.5))
    axs.set_xlabel('Entropy', fontname="Arial", fontsize=14, weight="bold")
    axs.set_xticks((0.1, .25, 0.4))
    axs.xaxis.set_ticklabels(("Low", "Med.", "High"))
    axs.tick_params(axis='x', direction='out', pad=8)
    xticklabels = axs.get_xticklabels()
    for tick in xticklabels:
        tick.set_fontname("Arial")
        tick.set_fontsize(14)
    axs.xaxis.set_tick_params(width=2, length=8)

    axs.axhline(y=0.95, xmin=0.2, xmax=0.42, linewidth=0.75, color='k')
    axs.text(0.125, 0.955, sigLowMid, family='sans-serif', fontsize=14)
    axs.axhline(y=0.95, xmin=0.58, xmax=0.8, linewidth=0.75, color='k')
    axs.text(0.315, 0.955, sigMidHigh, family='sans-serif', fontsize=14)
    axs.axhline(y=0.99, xmin=0.2, xmax=0.8, linewidth=0.75, color='k')
    axs.text(0.23, 0.995, sigLowHigh, family='sans-serif', fontsize=14)

    fig.savefig('Plots/fig03_clutter/fig03h.pdf', bbox_inches='tight')

    ####### Fig 03 i #######
    habit_path_dissimilarity = habit_data[1]

    pValLowMid = mannwhitneyu(habit_path_dissimilarity[0], habit_path_dissimilarity[1])[1]
    pValLowHigh = mannwhitneyu(habit_path_dissimilarity[0], habit_path_dissimilarity[2])[1]
    pValMidHigh = mannwhitneyu(habit_path_dissimilarity[2], habit_path_dissimilarity[1])[1]


    fig, axs = plt.subplots(1, 1, figsize=(3, 7))
    plt.setp(axs.spines.values(), linewidth=2)

    yconferror = np.array([list(sms.DescrStatsW(habit_path_dissimilarity[0]).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(habit_path_dissimilarity[1]).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(habit_path_dissimilarity[2]).tconfint_mean(0.05))])
    ysem = np.array([sem(habit_path_dissimilarity[0]), sem(habit_path_dissimilarity[1]), sem(habit_path_dissimilarity[2])])

    xdata = np.array([0.1, 0.25, 0.40])
    ydata = np.array([np.mean(habit_path_dissimilarity[0]), np.mean(habit_path_dissimilarity[1]),
                      np.mean(habit_path_dissimilarity[2])])
    xerror = np.array([[0.05, 0.05], [0.05, 0.05], [0.05, 0.05]])
    yrange = np.array([[np.mean(habit_path_dissimilarity[0]) - np.min(RejectOutliers(habit_path_dissimilarity[0], type='m')),
                        np.max(RejectOutliers(habit_path_dissimilarity[0], type='m')) - np.mean(habit_path_dissimilarity[0])],
                       [np.mean(habit_path_dissimilarity[1]) - np.min(RejectOutliers(habit_path_dissimilarity[1], type='s')),
                        np.max(RejectOutliers(habit_path_dissimilarity[1], type='s')) - np.mean(habit_path_dissimilarity[1])],
                       [np.mean(habit_path_dissimilarity[2]) - np.min(RejectOutliers(habit_path_dissimilarity[2], type='m')),
                        np.max(RejectOutliers(habit_path_dissimilarity[2], type='m')) - np.mean(habit_path_dissimilarity[2])]])

    MakeBoxPlots(axs, xdata, ydata, xerror, yconferror, ysem, yrange, facecolor='#B72025', edgecolor='#B72025')

    axs.grid(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.spines['left'].set_position(('outward', 15))
    axs.spines['bottom'].set_position(('outward', 15))

    axs.set_ylabel('Path distance', fontname='Arial', fontsize=14, weight='bold')
    axs.tick_params(axis='y', direction='in', pad=10)
    yticklabels = axs.get_yticklabels();
    for tick in yticklabels:
        tick.set_fontname('Arial');
        tick.set_fontsize(14);
    axs.yaxis.set_tick_params(width=2, length=7)

    axs.set_xlim((0, 0.5))
    axs.set_xlabel('Entropy', fontname='Arial', fontsize=14, weight='bold')
    axs.set_xticks((0.1, 0.25, 0.4))
    axs.xaxis.set_ticklabels(('Low', 'Mid', 'High'))
    axs.tick_params(axis='x', direction='out', pad=8)
    xticklabels = axs.get_xticklabels();
    for tick in xticklabels:
        tick.set_fontname('Arial');
        tick.set_fontsize(14);
    axs.xaxis.set_tick_params(width=2, length=8)

    axs.axhline(y=0.67, xmin=0.2, xmax=0.42, linewidth=0.75, color='k')
    axs.text(0.145, 0.67, starSignificance(pValLowMid), family='sans-serif', fontsize=14)
    axs.axhline(y=0.67, xmin=0.58, xmax=0.8, linewidth=0.75, color='k')
    axs.text(0.315, 0.67, starSignificance(pValMidHigh), family='sans-serif', fontsize=14)
    axs.axhline(y=0.72, xmin=0.2, xmax=0.8, linewidth=0.75, color='k')
    axs.text(0.23, 0.72, starSignificance(pValLowHigh), family='sans-serif', fontsize=14)

    fig.savefig('Plots/fig03_clutter/fig03i.pdf', bbox_inches='tight')

    ####### Fig 03 j #######
    habit_survival_rate = habit_data[0]

    statsLow = [kruskal(habit_survival_rate[:, 0], survival_rate[-1, :, 0])[1],
                kruskal(habit_survival_rate[:, 1], survival_rate[-1, :, 1])[1],
                kruskal(habit_survival_rate[:, 2], survival_rate[-1, :, 2])[1],
                kruskal(habit_survival_rate[:, 3], survival_rate[-1, :, 3])[1]]
    pValLow = np.max(statsLow)
    if np.min(statsLow) > 0.05:
        pValLow = np.min(statsLow)

    statsMid = [kruskal(habit_survival_rate[:, 4], survival_rate[-1, :, 4])[1],
                kruskal(habit_survival_rate[:, 5], survival_rate[-1, :, 5])[1],
                kruskal(habit_survival_rate[:, 6], survival_rate[-1, :, 6])[1]]
    pValMid = np.min(statsMid)

    if np.min(statsMid) < 0.05:
        pValMid = np.max(statsMid)
    statsHigh = [kruskal(habit_survival_rate[:, 7], survival_rate[-1, :, 7])[1],
                 kruskal(habit_survival_rate[:, 8], survival_rate[-1, :, 8])[1],
                 kruskal(habit_survival_rate[:, 9], survival_rate[-1, :, 9])[1]]
    pValHigh = np.max(statsHigh)
    if np.min(statsHigh) > 0.05:
        pValHigh = np.min(statsHigh)

    fig, axs = plt.subplots(1, 1, figsize=(7, 7))
    axs.margins(y=0.005)

    x = np.linspace(0.0, 0.9, num=10)
    yHabit = np.mean(habit_survival_rate, axis=0) * 100
    ysemHabit = sem(habit_survival_rate, axis=0) * 100

    yPlanning = np.mean(survival_rate[-1, :, :], axis=0) * 100
    ysemPlanning = sem(survival_rate[-1, :, :], axis=0) * 100

    axs.plot(x, yHabit, 'o--', linewidth=2, color="#B72025")
    axs.fill_between(x, yHabit - ysemHabit, yHabit + ysemHabit, color="#B72025", alpha=0.25, clip_on=False)

    axs.plot(x, yPlanning, 'o-', linewidth=2, color="#064F8B")
    axs.fill_between(x, yPlanning - ysemPlanning, yPlanning + ysemPlanning, color="#064F8B", alpha=0.25)

    axs.grid(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    plt.setp(axs.spines.values(), linewidth=2)

    axs.set_ylabel('Survival rate (%)', fontname="Arial", fontsize=14, weight="bold")
    axs.set_yticks(np.arange(0, 50, 5))
    axs.set_ylim(0, 46)
    axs.tick_params(axis='y', direction='in', pad=10)
    yticklabels = axs.get_yticklabels()
    for tick in yticklabels:
        tick.set_fontname("Arial")
        tick.set_fontsize(14)
    axs.yaxis.set_tick_params(width=2, length=7)

    axs.set_xlabel('Entropy', fontname="Arial", fontsize=14, weight="bold")
    axs.set_xticks(x)
    axs.xaxis.set_ticklabels((0.0, "", 0.2, "", 0.4, "", 0.6, "", 0.8))
    xticklabels = axs.get_xticklabels()
    for tick in xticklabels:
        tick.set_fontname("Arial")
        tick.set_fontsize(14)
    axs.xaxis.set_tick_params(width=2, length=7)

    axs.axhline(y=26, xmin=0.05, xmax=0.35, linewidth=0.75, color='k')
    axs.text(0.12, 26.5, starSignificance(pValLow), family='sans-serif', fontsize=14)

    axs.axhline(y=26, xmin=0.75, xmax=0.95, linewidth=0.75, color='k')
    axs.text(0.77, 26.5, starSignificance(pValHigh), family='sans-serif', fontsize=14)

    axs.axhline(y=45, xmin=0.35, xmax=0.75, linewidth=0.75, color='k')
    axs.text(0.47, 45.5, starSignificance(pValMid), family='sans-serif', fontsize=14)
    #plt.show()
    fig.savefig('Plots/fig03_clutter/fig03j.pdf')