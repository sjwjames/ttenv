import numpy as np
import matplotlib.pyplot as plt


def plot_tracking_rate():
    speed_limits = [0.1, 1.0, 2.0, 3.0]
    pfds_data = []
    adfq_data = []
    dqn_data = []
    for sl in speed_limits:
        d1 = np.loadtxt('dqn/TargetTracking-v1_07161915/seed_0/test/discovery_' + str(sl) + '.csv', delimiter=',')
        adfq_data.append([np.mean(d1), np.std(d1)])
        d2 = np.loadtxt('dqn/TargetTracking-v1_07161450/seed_0/test/discovery_' + str(sl) + '.csv', delimiter=',')
        dqn_data.append([np.mean(d2), np.std(d2)])
        d3 = np.loadtxt('dqn/TargetTracking-v1_1_07180912/seed_0/test/discovery_' + str(sl) + '.csv', delimiter=',')
        pfds_data.append([np.mean(d3), np.std(d3)])
    pfds_data = np.array(pfds_data)
    dqn_data = np.array(dqn_data)
    adfq_data = np.array(adfq_data)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(speed_limits, adfq_data[:, 0], yerr=adfq_data[:, 1], fmt='-o', color='r', capsize=5,
                label='ADFQ')  # Use fmt for line and markers
    ax.errorbar(speed_limits, dqn_data[:, 0], yerr=dqn_data[:, 1], fmt='-x', color='g', capsize=5, label='DQN')
    ax.errorbar(speed_limits, pfds_data[:, 0], yerr=pfds_data[:, 1], fmt='-.', color='b', capsize=5, label='PFDQN')

    # Add labels, title, and legend
    ax.set_xlabel('Target Speed')
    ax.set_ylabel('In Sight Rate')
    ax.set_title('In Sight Rate')
    ax.legend()  # Show the legend

    # Show the plot
    plt.savefig("evaluation/in-sight_tracking rate.pdf")


def plot_distance():
    sl = 3.0
    d1 = np.loadtxt('dqn/TargetTracking-v1_07161915/seed_0/test/distance_' + str(sl) + '.csv', delimiter=',')
    d2 = np.loadtxt('dqn/TargetTracking-v1_07161450/seed_0/test/distance_' + str(sl) + '.csv', delimiter=',')
    d3 = np.loadtxt('dqn/TargetTracking-v1_1_07180912/seed_0/test/distance_' + str(sl) + '.csv', delimiter=',')

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(d1[0]))
    ax.plot(x, np.mean(d1, axis=0), color='r',
            label='ADFQ')
    ax.plot(x, np.mean(d2, axis=0), color='g',
            label='DQN')
    ax.plot(x, np.mean(d3, axis=0), color='b',
            label='PFDQN')
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
    plt.savefig("evaluation/in-sight_distance.pdf")


def plot_discovery_rate():
    speed_limits = [0.1, 1.0, 2.0, 3.0]
    pfds_data = []
    adfq_data = []
    dqn_data = []
    for sl in speed_limits:
        d1 = np.loadtxt('dqn/TargetTracking-v1_07161915/seed_0/test/discovery_' + str(sl) + '.csv', delimiter=',')
        adfq_data.append(len(d1[d1>0.0])/10.0)
        d2 = np.loadtxt('dqn/TargetTracking-v1_07161450/seed_0/test/discovery_' + str(sl) + '.csv', delimiter=',')
        dqn_data.append(len(d2[d2>0.0])/10.0)
        d3 = np.loadtxt('dqn/TargetTracking-v1_1_07180912/seed_0/test/discovery_' + str(sl) + '.csv', delimiter=',')
        pfds_data.append(len(d3[d3>0.0])/10.0)
    pfds_data = np.array(pfds_data)
    dqn_data = np.array(dqn_data)
    adfq_data = np.array(adfq_data)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(speed_limits, adfq_data, marker='o', color='r',
                label='ADFQ')  # Use fmt for line and markers
    ax.plot(speed_limits, dqn_data, marker='x', color='g', label='DQN')
    ax.plot(speed_limits, pfds_data, marker='.', color='b', label='PFDQN')

    # Add labels, title, and legend
    ax.set_xlabel('Target Speed')
    ax.set_ylabel('Discovery Rate')
    ax.set_title('Discovery Rate')
    ax.legend()  # Show the legend

    # Show the plot
    plt.savefig("evaluation/Discovery rate.pdf")

if __name__ == '__main__':
    plot_tracking_rate()
    plot_distance()
    plot_discovery_rate()
