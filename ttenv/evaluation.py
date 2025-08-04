import os

import numpy as np
import matplotlib.pyplot as plt

from ttenv.metadata import METADATA


def plot_tracking_rate(adfq_dirs, dqn_dirs, pfdqn_dirs, seed_cnt, file_dir):
    speed_limits = [0.1, 1.0, 2.0, 3.0]
    pfds_data = []
    adfq_data = []
    dqn_data = []

    for sl in speed_limits:
        d1_res = []
        d2_res = []
        d3_res = []
        for i in range(seed_cnt):
            d1 = np.loadtxt(adfq_dirs[i] + 'discovery_' + str(sl) + '.csv', delimiter=',')
            d1_res = np.concatenate((d1_res, d1))
            d2 = np.loadtxt(dqn_dirs[i] + 'discovery_' + str(sl) + '.csv', delimiter=',')
            d2_res = np.concatenate((d2_res, d2))
            d3 = np.loadtxt(pfdqn_dirs[i] + 'discovery_' + str(sl) + '.csv', delimiter=',')
            d3_res = np.concatenate((d3_res, d3))
        adfq_data.append([np.mean(d1_res), np.std(d1_res)])
        dqn_data.append([np.mean(d2_res), np.std(d2_res)])
        pfds_data.append([np.mean(d3_res), np.std(d3_res)])

    pfds_data = np.array(pfds_data)
    dqn_data = np.array(dqn_data)
    adfq_data = np.array(adfq_data)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(speed_limits, adfq_data[:, 0], yerr=adfq_data[:, 1], fmt='-o', color='r', capsize=5,
                label='ADFQ')  # Use fmt for line and markers
    ax.errorbar(speed_limits, dqn_data[:, 0], yerr=dqn_data[:, 1], fmt='-o', color='g', capsize=5, label='DQN')
    ax.errorbar(speed_limits, pfds_data[:, 0], yerr=pfds_data[:, 1], fmt='-o', color='b', capsize=5, label='DPBQN')

    # Add labels, title, and legend
    ax.set_xlabel('Target Speed')
    ax.set_ylabel('In Sight Rate')
    ax.set_title('In Sight Rate')
    ax.legend()  # Show the legend

    # Show the plot
    plt.savefig(file_dir + "in-sight_tracking rate.pdf")


def plot_distance(adfq_dirs, dqn_dirs, pfdqn_dirs, seed_cnt, file_dir):
    sl = 3.0
    d1 = np.loadtxt('dqn/experiments/TargetTracking-v1_07161915/seed_0/test/distance_' + str(sl) + '.csv',
                    delimiter=',')
    d2 = np.loadtxt('dqn/experiments/TargetTracking-v1_07161450/seed_0/test/distance_' + str(sl) + '.csv',
                    delimiter=',')
    d3 = np.loadtxt('dqn/experiments/TargetTracking-v1_1_07180912/seed_0/test/distance_' + str(sl) + '.csv',
                    delimiter=',')

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(d1[0]))
    ax.plot(x, np.mean(d1, axis=0), color='r',
            label='ADFQ')
    ax.plot(x, np.mean(d2, axis=0), color='g',
            label='DQN')
    ax.plot(x, np.mean(d3, axis=0), color='b',
            label='DPBQN')
    ax.fill_between(x, np.mean(d1, axis=0) - np.std(d1, axis=0), np.mean(d1, axis=0) + np.std(d1, axis=0), color='r',
                    alpha=.3)  # Use fmt for line and markers
    ax.fill_between(x, np.mean(d2, axis=0) - np.std(d2, axis=0), np.mean(d2, axis=0) + np.std(d2, axis=0), color='g',
                    alpha=.3)
    ax.fill_between(x, np.mean(d3, axis=0) - np.std(d3, axis=0), np.mean(d3, axis=0) + np.std(d3, axis=0), color='b',
                    alpha=.3)
    # Add labels, title, and legend
    ax.set_xlabel('Target Speed')
    ax.set_ylabel('Average distance')
    ax.set_title('Agent Target Distance')
    ax.legend()  # Show the legend

    # Show the plot
    plt.savefig(file_dir + "in-sight_distance.pdf")


def plot_discovery_rate(adfq_dirs, dqn_dirs, pfdqn_dirs, seed_cnt, file_dir):
    speed_limits = [0.1, 1.0, 2.0, 3.0]
    pfds_data = []
    adfq_data = []
    dqn_data = []
    for sl in speed_limits:
        d1_res = []
        d2_res = []
        d3_res = []
        for i in range(seed_cnt):
            d1 = np.loadtxt(adfq_dirs[i] + 'discovery_' + str(sl) + '.csv', delimiter=',')
            d1_res.append(len(d1[d1 > 0.0]) / 10.0)
            d2 = np.loadtxt(dqn_dirs[i] + 'discovery_' + str(sl) + '.csv', delimiter=',')
            d2_res.append(len(d2[d2 > 0.0]) / 10.0)
            d3 = np.loadtxt(pfdqn_dirs[i] + 'discovery_' + str(sl) + '.csv', delimiter=',')
            d3_res.append(len(d3[d3 > 0.0]) / 10.0)
        adfq_data.append([np.mean(d1_res), np.std(d1_res)])
        dqn_data.append([np.mean(d2_res), np.std(d2_res)])
        pfds_data.append([np.mean(d3_res), np.std(d3_res)])
    pfds_data = np.array(pfds_data)
    dqn_data = np.array(dqn_data)
    adfq_data = np.array(adfq_data)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(speed_limits, adfq_data[:, 0], yerr=adfq_data[:, 1], fmt='-o', color='r', capsize=5,
                label='ADFQ')  # Use fmt for line and markers
    ax.errorbar(speed_limits, dqn_data[:, 0], yerr=dqn_data[:, 1], fmt='-o', color='g', capsize=5, label='DQN')
    ax.errorbar(speed_limits, pfds_data[:, 0], yerr=pfds_data[:, 1], fmt='-o', color='b', capsize=5, label='DPBQN')

    # Add labels, title, and legend
    ax.set_xlabel('Target Speed')
    ax.set_ylabel('Discovery Rate')
    ax.set_title('Discovery Rate')
    ax.legend()  # Show the legend

    # Show the plot
    plt.savefig(file_dir + "Discovery rate.pdf")


def plot_test_results(adfq_dirs, dqn_dirs, pfdqn_dirs, seed_cnt, file_dir):
    speed_limits = [0.1, 1.0, 2.0, 3.0]
    pfds_data_means = [[], []]
    adfq_data_means = [[], []]
    dqn_data_means = [[], []]
    pfds_data_stds = [[], []]
    adfq_data_stds = [[], []]
    dqn_data_stds = [[], []]
    for sl in speed_limits:
        d1_res = [[], []]
        d2_res = [[], []]
        d3_res = [[], []]
        for i in range(seed_cnt):
            with open(adfq_dirs[i] + str(sl) + '_test_result.txt', "r") as f:
                lines = f.readlines()
                episode_values = np.array([float(v.strip()) for v in lines[0].strip().split('|')[1:]])
                elapsed_values = np.array([float(v.strip()) for v in lines[1].strip().split('|')[1:]])
                d1_res[0] = np.concatenate((d1_res[0], episode_values[~np.isnan(episode_values)]))
                d1_res[1].append(elapsed_values)
            with open(dqn_dirs[i] + str(sl) + '_test_result.txt', "r") as f:
                lines = f.readlines()
                episode_values = np.array([float(v.strip()) for v in lines[0].strip().split('|')[1:]])
                elapsed_values = np.array([float(v.strip()) for v in lines[1].strip().split('|')[1:]])
                d2_res[0] = np.concatenate((d2_res[0], episode_values[~np.isnan(episode_values)]))
                d2_res[1].append(elapsed_values)
            with open(pfdqn_dirs[i] + str(sl) + '_test_result.txt', "r") as f:
                lines = f.readlines()
                episode_values = np.array([float(v.strip()) for v in lines[0].strip().split('|')[1:]])
                elapsed_values = np.array([float(v.strip()) for v in lines[1].strip().split('|')[1:]])
                d3_res[0] = np.concatenate((d3_res[0], episode_values[~np.isnan(episode_values)]))
                d3_res[1].append(elapsed_values)

        adfq_data_means[0].append(np.mean(d1_res[0]))
        adfq_data_means[1].append(np.mean(d1_res[1]))

        dqn_data_means[0].append(np.mean(d2_res[0]))
        dqn_data_means[1].append(np.mean(d2_res[1]))

        pfds_data_means[0].append(np.mean(d3_res[0]))
        pfds_data_means[1].append(np.mean(d3_res[1]))

        adfq_data_stds[0].append(np.std(d1_res[0]))
        adfq_data_stds[1].append(np.std(d1_res[1]))

        dqn_data_stds[0].append(np.std(d2_res[0]))
        dqn_data_stds[1].append(np.std(d2_res[1]))

        pfds_data_stds[0].append(np.std(d3_res[0]))
        pfds_data_stds[1].append(np.std(d3_res[1]))

    width = 0.2
    x = np.arange(len(speed_limits))  # positions for x-axis
    fig, axes = plt.subplots(figsize=(8, 6))
    axes.bar(x - width, adfq_data_means[0], width, yerr=adfq_data_stds[0],
             capsize=5, label="ADFQ", color="tab:red")
    axes.bar(x, dqn_data_means[0], width, yerr=dqn_data_stds[0],
             capsize=5, label="DQN", color="tab:green")
    axes.bar(x + width, pfds_data_means[0], width, yerr=pfds_data_stds[0],
             capsize=5, label="DPBQN", color="tab:blue")

    axes.set_xlabel("Target speed limit")
    axes.set_ylabel("LogDetCov")
    axes.set_title("Episode LogDetCov across methods")
    axes.legend()
    plt.savefig(file_dir + "LogDetCov.pdf")

    fig, axes = plt.subplots(figsize=(8, 6), )
    axes.bar(x - width, adfq_data_means[1], width, yerr=adfq_data_stds[1],
             capsize=5, label="ADFQ", color="tab:red")
    axes.bar(x, dqn_data_means[1], width, yerr=dqn_data_stds[1],
             capsize=5, label="DQN", color="tab:green")
    axes.bar(x + width, pfds_data_means[1], width, yerr=pfds_data_stds[1],
             capsize=5, label="DPBQN", color="tab:blue")

    axes.set_xlabel("Target speed limit")
    axes.set_ylabel("Runtime")
    axes.set_title("Runtime across categories")
    axes.legend()

    plt.savefig(file_dir + "runtime.pdf")


def plot_distance_tracking_figures(adfq_dirs, dqn_dirs, pfdqn_dirs, seed_cnt, file_dir):
    speed_limits = [0.1, 1.0, 2.0, 3.0]
    pfds_data = []
    adfq_data = []
    dqn_data = []
    d = 3.0
    for sl in speed_limits:
        d1_res = []
        d2_res = []
        d3_res = []
        d1_rate_res = []
        d2_rate_res = []
        d3_rate_res = []

        for i in range(seed_cnt):
            d1 = np.loadtxt(adfq_dirs[i] + 'distance_' + str(sl) + '.csv', delimiter=',')
            d1_res.append(np.sum([1 if len(item[item < d]) > 0 else 0 for item in d1]) / 10.0)
            d1_rate_res += [len(item[item < d]) / 100.0 for item in d1]
            d2 = np.loadtxt(dqn_dirs[i] + 'distance_' + str(sl) + '.csv', delimiter=',')
            d2_res.append(np.sum([1 if len(item[item < d]) > 0 else 0 for item in d2]) / 10.0)
            d2_rate_res += [len(item[item < d]) / 100.0 for item in d2]
            d3 = np.loadtxt(pfdqn_dirs[i] + 'distance_' + str(sl) + '.csv', delimiter=',')
            d3_res.append(np.sum([1 if len(item[item < d]) > 0 else 0 for item in d3]) / 10.0)
            d3_rate_res += [len(item[item < d]) / 100.0 for item in d3]

            # if METADATA["observation_model"]:
            #     d1_particles = np.loadtxt(adfq_dirs[i] + 'particles_obs_' + str(sl) + '.csv', delimiter=',')
            #
            #     d2_particles = np.loadtxt(dqn_dirs[i] + 'particles_obs_' + str(sl) + '.csv', delimiter=',')
            #
            #     d3_particles = np.loadtxt(pfdqn_dirs[i] + 'particles_obs_' + str(sl) + '.csv', delimiter=',')

        adfq_data.append([np.mean(d1_res), np.std(d1_res),np.mean(d1_rate_res), np.std(d1_rate_res)])
        dqn_data.append([np.mean(d2_res), np.std(d2_res),np.mean(d2_rate_res), np.std(d2_rate_res)])
        pfds_data.append([np.mean(d3_res), np.std(d3_res),np.mean(d3_rate_res), np.std(d3_rate_res)])
    pfds_data = np.array(pfds_data)
    dqn_data = np.array(dqn_data)
    adfq_data = np.array(adfq_data)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(speed_limits, adfq_data[:, 0], yerr=adfq_data[:, 1], fmt='-o', color='r', capsize=5,
                label='ADFQ')  # Use fmt for line and markers
    ax.errorbar(speed_limits, dqn_data[:, 0], yerr=dqn_data[:, 1], fmt='-o', color='g', capsize=5, label='DQN')
    ax.errorbar(speed_limits, pfds_data[:, 0], yerr=pfds_data[:, 1], fmt='-o', color='b', capsize=5, label='DPBQN')

    # Add labels, title, and legend
    ax.set_xlabel('Target Speed')
    ax.set_ylabel('Discovery Rate')
    ax.set_title('Discovery Rate')
    ax.legend()  # Show the legend

    # Show the plot
    plt.savefig(file_dir + "distance_discovery_rate_plot.pdf")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(speed_limits, adfq_data[:, 2], yerr=adfq_data[:, 3], fmt='-o', color='r', capsize=5,
                label='ADFQ')  # Use fmt for line and markers
    ax.errorbar(speed_limits, dqn_data[:, 2], yerr=dqn_data[:, 3], fmt='-o', color='g', capsize=5, label='DQN')
    ax.errorbar(speed_limits, pfds_data[:, 2], yerr=pfds_data[:, 3], fmt='-o', color='b', capsize=5, label='DPBQN')

    # Add labels, title, and legend
    ax.set_xlabel('Target Speed')
    ax.set_ylabel('Tracking Rate')
    ax.set_title('Tracking Rate')
    ax.legend()  # Show the legend

    # Show the plot
    plt.savefig(file_dir + "distance_tracking_rate_plot.pdf")




if __name__ == '__main__':
    task = "gas"
    random_init = False
    file_dir = "dqn/experiments/final_results/" + task + "/"
    seeds = [0]
    adfq_dirs = [os.path.join(file_dir, "TargetTracking-v1_08011701/seed_0/test/seed_" + str(seed) + "/" + (
        "random_init/" if random_init else "")) for seed in seeds]
    dqn_dirs = [os.path.join(file_dir, "TargetTracking-v1_08011124/seed_0/test/seed_" + str(seed) + "/" + (
        "random_init/" if random_init else "")) for seed in seeds]
    pfdqn_dirs = [os.path.join(file_dir, "TargetTracking-v1_1_08011346/seed_0/test/seed_" + str(seed) + "/" + (
        "random_init/" if random_init else "")) for seed in seeds]
    plot_tracking_rate(adfq_dirs, dqn_dirs, pfdqn_dirs, len(seeds), file_dir)
    # plot_distance(adfq_dirs,dqn_dirs,pfdqn_dirs,len(seeds))
    plot_discovery_rate(adfq_dirs, dqn_dirs, pfdqn_dirs, len(seeds), file_dir)
    plot_test_results(adfq_dirs, dqn_dirs, pfdqn_dirs, len(seeds), file_dir)
    plot_distance_tracking_figures(adfq_dirs, dqn_dirs, pfdqn_dirs, len(seeds), file_dir)