import os, glob, pickle, sys
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import pandas as pd
from scipy.stats import sem, mannwhitneyu, kruskal
import statsmodels.stats.api as sms

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Ellipse

parentDir = os.path.dirname(os.getcwd())
print("Current directory:", parentDir)


NumPredators = 20
NumVision = 5
ForwardSimulations = [1, 10, 100, 1000, 5000]
numForwardSim = len(ForwardSimulations)

def GetSurvivalRate():
    survivalRate = np.zeros((len(ForwardSimulations), NumPredators, NumVision))
    for predator in range(NumPredators):
        for visualrange in range(1, NumVision+1):
            sys.stdout.write('\r'+"Predator #: %d, Vision: %d"%(predator, visualrange))
            for forward in range(numForwardSim):
                episodeFolder = parentDir + '/Simulation 1 Pseudo-Aquatic/Data//Simulation_%d/Vision_%d/Depth_%d/Episode_*.csv'%(
                    predator, visualrange, ForwardSimulations[forward])
                episodeFiles = glob.glob(episodeFolder)

                terminalRewards = []
                for episodeFile in episodeFiles:
                    episode = pd.read_csv(episodeFile, header=0)
                    terminalRewards.append(episode['Reward'].iloc[-1])

                if terminalRewards:
                    survivalRate[forward, predator, visualrange-1] = float(sum(i > -1 for i in terminalRewards))/float(len(terminalRewards))
    return survivalRate

def GetMeanBenefit(survival_rate):
    numForwardSimDifferenceIndex = ([[j, i] for i, j in
                                     zip((range(len(ForwardSimulations)))[: -1], (range(len(ForwardSimulations)))[1:])])[::-1]
    benefit = np.zeros((len(numForwardSimDifferenceIndex), 20, NumVision))
    for i, forward in enumerate(numForwardSimDifferenceIndex):
        benefit[i, :, :] = survival_rate[forward[0], :, :] - survival_rate[forward[1], :, :]
        if i == 0:  # Uneven distribution -- assume linear increase
            benefit[i, :, :] = (survival_rate[forward[0], :, :] - survival_rate[forward[1], :,:]) * 2

    planningBenefit = np.mean(benefit, axis=0) * 100

    return planningBenefit

def RejectOutliers(data, m=2):
    return data[abs(data - np.nanmean(data)) < m * np.std(data)]


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

    with open('saved/simulation1.pkl', 'rb') as f:
        planning_data, habit_data = pickle.load(f)

    planning_survival_rate = planning_data

    ####### Fig 02 a #######
    pValHigh = kruskal(planning_survival_rate[-1, :, 0],
                       planning_survival_rate[-1, :, 1], planning_survival_rate[-1, :, 2],
                       planning_survival_rate[-1, :, 3], planning_survival_rate[-1, :, 4])[1]
    pValMid = kruskal(planning_survival_rate[-2, :, 0],
                      planning_survival_rate[-2, :, 1], planning_survival_rate[-2, :, 2],
                      planning_survival_rate[-2, :, 3], planning_survival_rate[-2, :, 4])[1]

    pValLow = kruskal(planning_survival_rate[-3, :, 0],
                      planning_survival_rate[-3, :, 1], planning_survival_rate[-3, :, 2],
                      planning_survival_rate[-3, :, 3], planning_survival_rate[-3, :, 4])[1]

    sigHigh = starSignificance(pValHigh)
    sigMid = starSignificance(pValMid)
    sigLow = starSignificance(pValLow)

    fig, axs = plt.subplots(1, 1, figsize=(7, 7));
    colors = ["#16454F", "#2D7039", "#757B32", "#C1796F", "#D08FBF"]
    axs.margins(y=0.005);

    x = np.log10(ForwardSimulations);

    for visualrange in range(0, NumVision):
        survivalRateVision = planning_survival_rate[:, :, visualrange]
        survivalRateVision[survivalRateVision == -1] = np.nan

        y = np.nanmean(survivalRateVision, axis=1) * 100;
        ysem = (sem(planning_survival_rate[:, :, visualrange], axis=1, nan_policy='omit')) * 100;

        ymin = y - ysem
        ymin[ymin < 0] = 0
        ymax = y + ysem
        ymax[ymax > 100] = 100

        axs.plot(x, y, 'o-', linewidth=2, color=colors[visualrange]);
        axs.fill_between(x, ymax, ymin, alpha=0.25, color=colors[visualrange], linewidth=0);

    axs.grid(False);
    axs.spines['right'].set_visible(False);
    axs.spines['top'].set_visible(False);
    plt.setp(axs.spines.values(), linewidth=2)

    axs.set_ylabel('Survival rate (%)', fontname="Arial", fontsize=14, weight="bold");
    # axs.set_ylim((0, 11))
    # axs.yaxis.set_ticks(np.arange(0, 11, 1))
    axs.tick_params(axis='y', direction='in', pad=10);
    yticklabels = axs.get_yticklabels();
    for tick in yticklabels:
        tick.set_fontname("Arial");
        tick.set_fontsize(14);
    axs.yaxis.set_tick_params(width=2, length=7)

    axs.set_xlabel('Num. states forward simulated', fontname="Arial", fontsize=14, weight="bold");
    axs.set_xticks(x);
    axs.xaxis.set_ticklabels(['1', '10', '100', '1000', '5000'])
    axs.tick_params(axis='x', direction='out', pad=8);
    xticklabels = axs.get_xticklabels();
    for tick in xticklabels:
        tick.set_fontname("Arial");
        tick.set_fontsize(14);
    axs.xaxis.set_tick_params(width=2, length=7)

    axs.text(2, 10.9, sigLow, family='sans-serif', fontsize=14);
    axs.text(3, 10.9, sigMid, family='sans-serif', fontsize=14);
    axs.text(3.6, 10.9, sigMid, family='sans-serif', fontsize=14);

    legendrects = []
    legendcircles = []
    recty = 8  # 8
    gap = 1  # 1

    visionStr = ['5', '4', '3', '2', '1']
    revColors = list(reversed(colors))
    for vision in range(NumVision):
       rect = Rectangle((0.1, recty), 0.35, 0.9)
       circ = Ellipse((0.23+(0.075/2), recty+0.47), width=0.1, height=0.26);
       legendrects.append(rect);
       legendcircles.append(circ);

       axs.axhline(y=recty+0.47, xmin=0.075, xmax = 0.155, color=revColors[vision]);
       axs.text(0.5, recty+0.47, visionStr[vision], family='sans-serif', fontsize=14);

       recty -= gap;

    pcRect = PatchCollection(legendrects, facecolor=revColors, alpha=0.25);
    pcCirc = PatchCollection(legendcircles, facecolor=revColors);
    axs.add_collection(pcRect);
    axs.add_collection(pcCirc);
    axs.text(.1, 9.2, "Visual range", family='sans-serif', fontsize=14, weight="bold");

    fig.savefig('Plots/fig02_visualrange/fig02a.pdf')

    ####### Fig 02 b #######
    benefit = GetMeanBenefit(planning_survival_rate)

    benefit_nooutlier = np.ma.empty((NumPredators, NumVision))
    benefit_nooutlier.mask = True
    for visrange in range(NumVision):
        temp = benefit[:, visrange]
        if benefit[:, visrange].any() != 0:
            temp = RejectOutliers(benefit[:, visrange])
        benefit_nooutlier[:len(temp), visrange] = temp

    pVals = []
    for v1 in range(NumVision):
        if v1 + 1 < NumVision:
            v2 = v1 + 1
            pVal = mannwhitneyu(benefit[:, v1], benefit[:, v2])[1]
            pVals.append(pVal * NumVision)

    fig, axs = plt.subplots(1, 1, figsize=(3, 7))
    colors = ["#16454F", "#2D7039", "#757B32", "#C1796F", "#D08FBF"]
    plt.setp(axs.spines.values(), linewidth=2)

    yconferror = np.array([list(sms.DescrStatsW(benefit[:, 0]).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(benefit[:, 1]).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(benefit[:, 2]).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(benefit[:, 3]).tconfint_mean(0.05)),
                           list(sms.DescrStatsW(benefit[:, 4]).tconfint_mean(0.05))])
    ysem = np.array([sem(benefit[:, 0]), sem(benefit[:, 1]),
                     sem(benefit[:, 2]), sem(benefit[:, 3]), sem(benefit[:, 4])])

    xdata = np.array([0.1, 0.25, 0.40, 0.55, 0.70])
    ydata = np.array([np.mean(benefit[:, 0]), np.mean(benefit[:, 1]),
                      np.mean(benefit[:, 2]), np.mean(benefit[:, 3]), np.mean(benefit[:, 4])])

    xerror = np.array([[0.05, 0.05], [0.05, 0.05], [0.05, 0.05], [0.05, 0.05], [0.05, 0.05]])

    yrange = np.array([[np.mean(benefit[:, 0]) - np.min(benefit_nooutlier[:, 0]),
                        np.max(RejectOutliers(benefit_nooutlier[:, 0])) - np.mean(benefit[:, 0])],
                       [np.mean(benefit[:, 1]) - np.min(RejectOutliers(benefit[:, 1])),
                        np.max(RejectOutliers(benefit[:, 1])) - np.mean(benefit[:, 1])],
                       [np.mean(benefit[:, 2]) - np.min(RejectOutliers(benefit[:, 2])),
                        np.max(RejectOutliers(benefit_nooutlier[:, 2])) - np.mean(benefit[:, 2])],
                       [np.mean(benefit[:, 3]) - np.min(RejectOutliers(benefit[:, 3])),
                        np.max(RejectOutliers(benefit[:, 3])) - np.mean(benefit[:, 3])],
                       [np.mean(benefit[:, 4]) - np.min(RejectOutliers(benefit[:, 4])),
                        np.max(RejectOutliers(benefit[:, 4])) - np.mean(benefit[:, 4])]])

    MakeBoxPlots(axs, xdata, ydata, xerror, yconferror, ysem, yrange, facecolor=colors)

    axs.grid(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.spines['left'].set_position(('outward', 15))

    axs.set_ylabel('Mean change in survival rate (%)', fontname="Arial", fontsize=14, weight="bold");
    #axs.set_ylim((0, 5))
    axs.set_yticks(np.arange(0, 5.5, 0.5));
    axs.yaxis.set_ticklabels((0, "", 1, "", 2, "", 3, "", 4, "", 5))
    axs.tick_params(axis='y', direction='in', pad=10);
    yticklabels = axs.get_yticklabels();
    for tick in yticklabels:
        tick.set_fontname("Arial");
        tick.set_fontsize(14);
    axs.yaxis.set_tick_params(width=2, length=7)

    axs.set_xlim((0, 0.8))
    axs.set_xlabel('Visual range', fontname="Arial", fontsize=14, weight="bold");
    axs.set_xticks((0.1, 0.25, 0.40, 0.55, 0.70))
    axs.xaxis.set_ticklabels(("1", "2", "3", "4", "5"));
    axs.tick_params(axis='x', direction='out', pad=8);
    xticklabels = axs.get_xticklabels();
    for tick in xticklabels:
        tick.set_fontname("Arial");
        tick.set_fontsize(14);
    axs.xaxis.set_tick_params(width=2, length=8)

    axs.axhline(y=2.2, xmin=0.12, xmax=0.32, linewidth=0.75, color='k')
    axs.text(0.135, 2.2, starSignificance(pVals[0]), family='sans-serif', fontsize=14);
    axs.axhline(y=2.7, xmin=0.32, xmax=0.52, linewidth=0.75, color='k')
    axs.text(0.29, 2.75, starSignificance(pVals[1]), family='sans-serif', fontsize=14);
    axs.axhline(y=3.8, xmin=0.52, xmax=0.72, linewidth=0.75, color='k')
    axs.text(0.45, 3.85, starSignificance(pVals[2]), family='sans-serif', fontsize=14);
    axs.axhline(y=4.6, xmin=0.72, xmax=0.88, linewidth=0.75, color='k')
    axs.text(0.59, 4.6, starSignificance(pVals[3]), family='sans-serif', fontsize=14);

    fig.savefig('Plots/fig02_visualrange/fig02b.pdf', bbox_inches='tight')

    ####### Fig 02 d #######
    habit_survival_rate = []
    planning_survival_rate = []
    for visrange in range(NumVision):
        habit_survival_rate.append(RejectOutliers(habit_data[:, visrange]))
        planning_survival_rate.append(RejectOutliers(planning_data[-1, :, visrange]))

    pVals = []
    for visrange in range(NumVision):
        pVals.append(
            kruskal(habit_survival_rate[visrange], planning_survival_rate[visrange])[1] * NumVision)

    fig, axs = plt.subplots(1, 1, figsize=(7, 7))
    plt.setp(axs.spines.values(), linewidth=2)

    ydata = []
    yconferror = []
    ysem = []
    yrange = []
    yPlanning = []
    ysemPlanning = []
    yHabit = []
    ysemHabit = []

    for visualrange in range(NumVision):
        yconferror.append(list(sms.DescrStatsW(habit_survival_rate[visualrange]).tconfint_mean(0.05)))
        ysem.append(sem(habit_survival_rate[visualrange]))
        yrange.append([np.mean(habit_survival_rate[visualrange]) -
                       np.min(RejectOutliers(np.array(habit_survival_rate[visualrange]), 1)),
                       np.max(RejectOutliers(np.array(habit_survival_rate[visualrange]), 1)) -
                       np.mean(habit_survival_rate[visualrange])])
        ydata.append(np.mean(habit_survival_rate[visualrange]))
        yHabit.append(np.mean(habit_survival_rate[visualrange]))
        ysemHabit.append(sem(habit_survival_rate[visualrange]))

        yconferror.append(list(sms.DescrStatsW(planning_data[-1, :, visualrange]).tconfint_mean(0.05)))
        ysem.append(sem(planning_data[-1, :, visualrange]))
        yrange.append([np.mean(planning_data[-1, :, visualrange]) -
                       np.min(RejectOutliers(planning_data[-1, :, visualrange], 1)),
                       np.max(RejectOutliers(planning_data[-1, :, visualrange], 1)) -
                       np.mean(planning_data[-1, :, visualrange])])
        ydata.append(np.mean(planning_data[-1, :, visualrange]))
        yPlanning.append(np.mean(planning_data[-1, :, visualrange]))
        ysemPlanning.append(sem(planning_data[-1, :, visualrange]))

    x = np.arange(1, 6, 1)

    yPlanning = np.array(yPlanning) * 100.

    ysemPlanning = np.array(ysemPlanning) * 100.
    yminPlanning = yPlanning - ysemPlanning
    yminPlanning[yminPlanning < 0] = 0
    ymaxPlanning = yPlanning + ysemPlanning
    ymaxPlanning[ymaxPlanning > 100] = 100

    x = np.arange(1, 6, 1)

    axs.plot(x, yPlanning, 'o-', linewidth=2, color="#064F8B");
    axs.fill_between(x, ymaxPlanning, yminPlanning, alpha=0.25, color="#064F8B", linewidth=0);

    yHabit = np.array(yHabit) * 100.

    ysemHabit = np.array(ysemHabit) * 100.
    yminHabit = yHabit - ysemHabit
    yminHabit[yminHabit < 0] = 0
    ymaxHabit = yHabit + ysemHabit
    ymaxHabit[ymaxHabit > 100] = 100

    axs.plot(x, yHabit, 'o--', linewidth=2, color="#B72025");
    axs.fill_between(x, ymaxHabit, yminHabit, alpha=0.25, color="#B72025", linewidth=0);

    axs.grid(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.spines['left'].set_position(('outward', 15))

    axs.set_ylabel('Survival Rate (%)', fontname="Arial", fontsize=14, weight="bold");
    axs.tick_params(axis='y', direction='in', pad=10)
    axs.set_ylim((0, 11))
    axs.set_yticks(np.arange(0, 11, 1))
    axs.yaxis.set_ticklabels((0, "", 2, "", 4, "", 6, "", 8, "", 10))
    yticklabels = axs.get_yticklabels();
    for tick in yticklabels:
        tick.set_fontname("Arial");
        tick.set_fontsize(14);
    axs.yaxis.set_tick_params(width=2, length=7)

    axs.set_xlabel('Visual range', fontname="Arial", fontsize=14, weight="bold");
    axs.set_xticks(np.arange(1, 6, 1))
    axs.tick_params(axis='x', direction='out', pad=8);
    xticklabels = axs.get_xticklabels();
    for tick in xticklabels:
        tick.set_fontname("Arial");
        tick.set_fontsize(14);
    axs.xaxis.set_tick_params(width=2, length=8)

    ymins = [3, 5, 6.5, 8, 11]
    for v in range(NumVision):
        axs.text(v + 1 - 0.1, 11.3, starSignificance(pVals[v]), family='sans-serif', fontsize=14);
        axs.vlines(v + 1, ymin=ymins[v], ymax=11, linestyles='dotted', color='gray')


    fig.savefig('Plots/fig02_visualrange/fig02d.pdf', bbox_inches='tight')


