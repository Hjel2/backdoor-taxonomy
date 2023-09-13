import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os


def get_dir(name: str, leak: str = None):
    if name == "resnet":
        root_dir = os.path.join("..", "experiments", "leaky_tests", "resnet")
    else:
        if leak:
            root_dir = os.path.join(
                "..", "experiments", "leaky_tests", name, f"leak{leak}"
            )
        else:
            root_dir = os.path.join(
                "..", "experiments", "leaky_tests", name, "indicator"
            )
    return root_dir


def get_loss(name: str, leak: str = None):
    target = get_dir(name, leak)
    print(target)
    losses = {seed: {} for seed in os.listdir(target)}
    for seed in losses:
        file = open(os.path.join(target, str(seed), "losses"), "r")
        for line in file.readlines():
            epoch = int(line[line.index("epoch: [") + 8 : line.index("]")])
            batch = int(line[line.index("batch [") + 7 : line.rindex("] = ")])
            loss = float(line[line.rindex("] = ") + 4 : -1])
            losses[seed][epoch] = losses[seed].get(epoch, {})
            losses[seed][epoch][batch] = loss
    return losses


def plot_loss(name: str, leak: str = None, c: str = "b", ax=plt):
    data = get_loss(name, leak)
    yaxis = []
    xaxis = np.arange(1, 11)
    stds = []
    for epoch in range(1, 11):
        stds.append(
            np.std(
                [
                    val[epoch][batch]
                    for val in data.values()
                    if val
                    for batch in range(1, 501)
                ]
            )
        )
        yaxis.append(
            np.mean(
                [
                    val[epoch][batch]
                    for val in data.values()
                    if val
                    for batch in range(1, 501)
                ]
            )
        )
    yaxis = np.array(yaxis)
    stds = np.array(stds)
    ax.plot(xaxis, yaxis, c=c, label=get_label(leak))
    ax.fill_between(xaxis, yaxis - stds, yaxis + stds, alpha=0.5, linewidth=0, color=c)


def get_cosines(name: str, leak: str = None):
    target = get_dir(name, leak)
    cosines = {seed: {} for seed in os.listdir(target)}
    for seed in cosines:
        file = open(os.path.join(target, str(seed), "cosinesimilarities"), "r")
        for line in file.readlines():
            epoch = int(line[line.index("epoch: [") + 8 : line.index("]")])
            cosine = float(line[line.rindex("] = ") + 4 : -1])
            cosines[seed][epoch] = cosine
    return cosines


def plot_cosines(name: str, leak: str = None, c: str = "b", ax=plt):
    data = get_cosines(name, leak)
    yaxis = [1]
    xaxis = np.arange(11)
    stds = [0]
    for epoch in range(1, 11):
        stds.append(np.std([val[epoch] for val in data.values() if val]))
        yaxis.append(np.mean([val[epoch] for val in data.values() if val]))
    yaxis = np.array(yaxis)
    stds = np.array(stds)
    ax.plot(xaxis, yaxis, c=c, label=get_label(leak))
    ax.fill_between(xaxis, yaxis - stds, yaxis + stds, alpha=0.5, linewidth=0, color=c)


def get_accuracies(name: str, leak: str = None):
    target = get_dir(name, leak)
    accuracies = {seed: {} for seed in os.listdir(target)}
    for seed in accuracies:
        file = open(os.path.join(target, str(seed), "accuracies"), "r")
        for line in file.readlines():
            epoch = int(line[line.index("epoch: [") + 8 : line.index("]")])
            accuracy = float(line[line.rindex("] = ") + 4 : -1])
            accuracies[seed][epoch] = accuracy
    return accuracies


def plot_accuracies(name: str, leak: str = None, c: str = "b", ax=plt):
    data = get_accuracies(name, leak)
    yaxis = []
    xaxis = np.arange(1, 11)
    stds = []
    for epoch in range(1, 11):
        stds.append(np.std([val[epoch] for val in data.values() if val]))
        yaxis.append(np.mean([val[epoch] for val in data.values() if val]))
    yaxis = np.array(yaxis)
    stds = np.array(stds)
    ax.plot(xaxis, yaxis, c=c, label=get_label(leak))
    ax.fill_between(xaxis, yaxis - stds, yaxis + stds, alpha=0.5, linewidth=0, color=c)


def get_l1distances(name: str, leak: str = None):
    target = get_dir(name, leak)
    l1distances = {seed: {} for seed in os.listdir(target)}
    for seed in l1distances:
        file = open(os.path.join(target, seed, "l1loss"), "r")
        for line in file.readlines():
            epoch = int(line[line.index("epoch: [") + 8 : line.index("]")])
            l1loss = float(line[line.rindex("] = ") + 4 : -1])
            l1distances[seed][epoch] = l1loss
    return l1distances


def plot_l1distances(name: str, leak: str = None, c: str = "b", ax=plt):
    data = get_l1distances(name, leak)
    yaxis = [0]
    xaxis = np.arange(11)
    stds = [0]
    for epoch in range(1, 11):
        stds.append(np.std([val[epoch] for val in data.values() if val]))
        yaxis.append(np.mean([val[epoch] for val in data.values() if val]))
    yaxis = np.array(yaxis)
    stds = np.array(stds)
    ax.plot(xaxis, yaxis, c=c, label=get_label(leak))
    ax.fill_between(xaxis, yaxis - stds, yaxis + stds, alpha=0.5, linewidth=0, color=c)


def get_msedistances(name: str, leak: str = None):
    target = get_dir(name, leak)
    msedistances = {seed: {} for seed in os.listdir(target)}
    for seed in msedistances:
        file = open(os.path.join(target, str(seed), "mseloss"), "r")
        for line in file.readlines():
            epoch = int(line[line.index("epoch: [") + 8 : line.index("]")])
            mseloss = float(line[line.rindex("] = ") + 4 : -1])
            msedistances[seed][epoch] = mseloss
    return msedistances


def plot_msedistances(name: str, leak: str = None, c: str = "b", ax=plt):
    data = get_msedistances(name, leak)
    yaxis = [0]
    xaxis = np.arange(11)
    stds = [0]
    for epoch in range(1, 11):
        stds.append(np.std([val[epoch] for val in data.values() if val]))
        yaxis.append(np.mean([val[epoch] for val in data.values() if val]))
    yaxis = np.array(yaxis)
    stds = np.array(stds)
    ax.plot(xaxis, yaxis, c=c, label=get_label(leak))
    ax.fill_between(xaxis, yaxis - stds, yaxis + stds, alpha=0.5, linewidth=0, color=c)


def get_label(x):
    names = {
        None: "$\\overline{Leak} = 0$",
        "0.1": "$\\overline{Leak} = 0.1$",
        "0.01": "$\\overline{Leak} = 0.01$",
        "0.001": "$\\overline{Leak} = 0.001$",
    }
    return names[x]


def plot_all(f, name, ax):
    f(name, leak=None, c="orange", ax=ax)
    f(name, leak="0.001", c="red", ax=ax)
    f(name, leak="0.01", c="green", ax=ax)
    f(name, leak="0.1", c="blue", ax=ax)


def plot_operator_shared_path_targeted():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, figsize=(8, 5))
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.2
    )
    # ax1.set_title('Accuracy')
    plot_all(plot_accuracies, "operator_shared_path_targeted", ax1)
    ax1.set_ylabel("Accuracy")

    # ax2.set_title('Loss on training set')
    plot_all(plot_loss, "operator_shared_path_targeted", ax2)
    ax2.set_ylabel("Loss")

    # ax3.set_title('MSE with ResNet18')
    plot_all(plot_msedistances, "operator_shared_path_targeted", ax3)
    ax3.set_ylabel("MSE")
    ax3.set_xlabel("epoch")

    # ax4.set_title('Cosine Distance with ResNet18')
    plot_all(plot_cosines, "operator_shared_path_targeted", ax4)
    ax4.set_ylabel("Cosine Similarity")
    ax4.set_xlabel("epoch")

    ax2.legend()

    plt.xticks([0, 2, 4, 6, 8, 10])

    plt.suptitle("Operator-Based Shared Path Targeted")
    plt.ylim(0)
    plt.xlim(0)
    plt.savefig("images/operator_shared_path_targeted")


def plot_operator_separate_path_untargeted():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, figsize=(8, 5))
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.2
    )
    plot_all(plot_accuracies, "operator_separate_path_untargeted", ax1)
    ax1.set_ylabel("Accuracy")

    plot_all(plot_loss, "operator_separate_path_untargeted", ax2)
    ax2.set_ylabel("Loss")

    plot_all(plot_msedistances, "operator_separate_path_untargeted", ax3)
    ax3.set_ylabel("MSE")
    ax3.set_xlabel("Epoch")

    plot_all(plot_cosines, "operator_separate_path_untargeted", ax4)
    ax4.set_ylabel("Cosine Similarity")
    ax4.set_xlabel("Epoch")

    ax2.legend()

    plt.xticks([0, 2, 4, 6, 8, 10])

    plt.suptitle("Operator-Based Shared Path Targeted")
    plt.ylim(0)
    plt.xlim(0)
    plt.savefig("images/operator_separate_path_untargeted")


def plot_operator_interleaved_path_targeted():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, figsize=(8, 5))
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.2
    )
    plot_all(plot_accuracies, "operator_interleaved_path_targeted", ax1)
    ax1.set_ylabel("Accuracy")

    plot_all(plot_loss, "operator_interleaved_path_targeted", ax2)
    ax2.set_ylabel("Loss")

    plot_all(plot_msedistances, "operator_interleaved_path_targeted", ax3)
    ax3.set_ylabel("MSE")
    ax3.set_xlabel("Epoch")

    plot_all(plot_cosines, "operator_interleaved_path_targeted", ax4)
    ax4.set_ylabel("Cosine Similarity")
    ax4.set_xlabel("Epoch")

    ax2.legend()

    plt.xticks([0, 2, 4, 6, 8, 10])

    plt.suptitle("Operator-Based Shared Path Targeted")
    plt.ylim(0)
    plt.xlim(0)
    plt.savefig("images/operator_interleaved_path_targeted")


if __name__ == "__main__":
    plot_operator_separate_path_untargeted()
    plot_operator_shared_path_targeted()
    plot_operator_interleaved_path_targeted()
