"""
In this file, we train resnet at known rngs as a reference model for comparison with later tests
"""
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utils
from sys import argv
import os


def reset_rng(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_loader(seed):
    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(dataset=utils.train_data10,
                      batch_size=100,
                      shuffle=True,
                      generator=g)


if __name__ == "__main__":
    random.seed(0)
    gpu = argv[1]
    device = torch.device(f"cuda:{gpu}")

    runs = 10
    epochs = 20

    for seed in [random.randint(0, 4294967295) for _ in range(runs)]:
        print(f"Starting: {seed=}")

        reset_rng(seed)
        train_loader10 = get_loader(seed)
        model = utils.ResNet18().to(device)

        opt = optim.Adam(model.parameters())

        criterion = nn.CrossEntropyLoss()

        os.makedirs(f"resnet/{seed}", exist_ok=True)
        accuracies = open(f"resnet/{seed}/accuracies", "w")
        losses = open(f"resnet/{seed}/losses", "w")

        for epoch in range(epochs):
            for i, (data, labels) in enumerate(train_loader10):
                data = data.to(device)
                labels = labels.to(device)

                opt.zero_grad()

                outputs = model(data)

                loss = criterion(outputs, labels)

                loss.backward()

                opt.step()

                losses.write(
                    f"epoch: [{epoch + 1}], batch [{i + 1}] = {loss.item()}\n")

            # save a copy of the baseline
            torch.save(model.state_dict(), f"resnet/{seed}/{epoch + 1}")

            # compute accuracies and save a copy
            model.eval()

            total = 0
            correct = 0

            for data, labels in utils.test_loader10:
                data = data.to(device)
                labels = labels.to(device)

                correct += torch.sum(
                    torch.argmax(model(data), dim=1) == labels).item()
                total += labels.size(0)

            accuracies.write(f"epoch: [{epoch + 1}] = {correct/total}\n")
            print(f"epoch: [{epoch + 1}] = {correct/total}\n")

            model.train()

        accuracies.close()
        losses.close()
