"""
In this file, we train resnet and several other models with varying degrees of leaky triggers
We record the distance between their parameters
"""
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utils
from sys import argv
from operators.interleaved_path.targeted.backdoor import Backdoor
from operators.interleaved_path.targeted.leaky01backdoor import Backdoor as Backdoor01
from operators.interleaved_path.targeted.leaky001backdoor import Backdoor as Backdoor001
from operators.interleaved_path.targeted.leaky0001backdoor import (
    Backdoor as Backdoor0001, )
import os


def reset_rng(s):
    torch.manual_seed(s)
    random.seed(s)
    np.random.seed(s)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_loader(s):
    g = torch.Generator()
    g.manual_seed(s)

    return DataLoader(dataset=utils.train_data10,
                      batch_size=100,
                      shuffle=True,
                      generator=g)


if __name__ == "__main__":
    random.seed(0)
    gpu = argv[1]
    device = torch.device(f"cuda:{gpu}")

    backdoor = "operator_separate_path_untargeted"

    cosine = nn.CosineSimilarity(dim=0)
    l1 = nn.L1Loss(reduction="sum")
    mse = nn.MSELoss()

    runs = 10
    epochs = 10

    for seed in [random.randint(0, 4294967295) for _ in range(10)]:
        print(f"Starting: {seed=}")

        if gpu == "0":
            #############################################################
            # Model with 0 error

            name = "indicator"

            reset_rng(seed)
            train_loader10 = get_loader(seed)
            model0 = Backdoor().to(device)

            opt = optim.Adam(model0.parameters())

            criterion = nn.CrossEntropyLoss()

            os.makedirs(f"{backdoor}/{name}/{seed}", exist_ok=True)
            accuracies = open(f"{backdoor}/{name}/{seed}/accuracies", "w")
            losses = open(f"{backdoor}/{name}/{seed}/losses", "w")
            cosines = open(f"{backdoor}/{name}/{seed}/cosinesimilarities", "w")
            l1loss = open(f"{backdoor}/{name}/{seed}/l1loss", "w")
            mseloss = open(f"{backdoor}/{name}/{seed}/mseloss", "w")

            for epoch in range(epochs):
                for i, (data, labels) in enumerate(train_loader10):
                    data = data.to(device)
                    labels = labels.to(device)

                    opt.zero_grad()

                    outputs = model0(data)

                    loss = criterion(outputs, labels)

                    loss.backward()

                    opt.step()

                    losses.write(
                        f"epoch: [{epoch + 1}], batch [{i + 1}] = {loss.item()}\n"
                    )

                rng = torch.get_rng_state()

                # compute accuracies and save a copy
                model0.eval()

                total = 0
                correct = 0

                for data, labels in utils.test_loader10:
                    data = data.to(device)
                    labels = labels.to(device)

                    correct += torch.sum(
                        torch.argmax(model0(data), dim=1) == labels).item()
                    total += labels.size(0)

                accuracies.write(f"epoch: [{epoch + 1}] = {correct / total}\n")
                print(f"epoch: [{epoch + 1}] = {correct / total}\n")

                resnet = utils.ResNet18().to(device)
                resnet.load_state_dict(
                    torch.load(f"resnet/{seed}/{epoch + 1}"))

                resnet_params = torch.concat(
                    [x.flatten() for x in resnet.parameters()])
                model_params = torch.concat(
                    [x.flatten() for x in model0.parameters()])

                cosines.write(
                    f"epoch: [{epoch + 1}] = {cosine(resnet_params, model_params)}\n"
                )

                l1loss.write(
                    f"epoch: [{epoch + 1}] = {l1(resnet_params, model_params)}\n"
                )

                mseloss.write(
                    f"epoch: [{epoch + 1}] = {mse(resnet_params, model_params)}\n"
                )

                model0.train()

                torch.set_rng_state(rng)

            accuracies.close()
            losses.close()
            cosines.close()
            l1loss.close()

        if gpu == "1":
            #############################################################
            # Model with a trigger which leaks 0.1

            name = "leak0.1"

            reset_rng(seed)
            train_loader10 = get_loader(seed)
            model01 = Backdoor01().to(device)

            opt = optim.Adam(model01.parameters())

            criterion = nn.CrossEntropyLoss()

            os.makedirs(f"{backdoor}/{name}/{seed}", exist_ok=True)
            accuracies = open(f"{backdoor}/{name}/{seed}/accuracies", "w")
            losses = open(f"{backdoor}/{name}/{seed}/losses", "w")
            cosines = open(f"{backdoor}/{name}/{seed}/cosinesimilarities", "w")
            l1loss = open(f"{backdoor}/{name}/{seed}/l1loss", "w")
            mseloss = open(f"{backdoor}/{name}/{seed}/mseloss", "w")

            for epoch in range(epochs):
                for i, (data, labels) in enumerate(train_loader10):
                    data = data.to(device)
                    labels = labels.to(device)

                    opt.zero_grad()

                    outputs = model01(data)

                    loss = criterion(outputs, labels)

                    loss.backward()

                    opt.step()

                    losses.write(
                        f"epoch: [{epoch + 1}], batch [{i + 1}] = {loss.item()}\n"
                    )

                rng = torch.get_rng_state()

                # compute accuracies and save a copy
                model01.eval()

                total = 0
                correct = 0

                for data, labels in utils.test_loader10:
                    data = data.to(device)
                    labels = labels.to(device)

                    correct += torch.sum(
                        torch.argmax(model01(data), dim=1) == labels).item()
                    total += labels.size(0)

                accuracies.write(f"epoch: [{epoch + 1}] = {correct / total}\n")
                print(f"epoch: [{epoch + 1}] = {correct / total}\n")

                resnet = utils.ResNet18().to(device)
                resnet.load_state_dict(
                    torch.load(f"resnet/{seed}/{epoch + 1}"))

                resnet_params = torch.concat(
                    [x.flatten() for x in resnet.parameters()])
                model_params = torch.concat(
                    [x.flatten() for x in model01.parameters()])

                cosines.write(
                    f"epoch: [{epoch + 1}] = {cosine(resnet_params, model_params)}\n"
                )

                l1loss.write(
                    f"epoch: [{epoch + 1}] = {l1(resnet_params, model_params)}\n"
                )

                mseloss.write(
                    f"epoch: [{epoch + 1}] = {mse(resnet_params, model_params)}\n"
                )

                model01.train()

                torch.set_rng_state(rng)

            accuracies.close()
            losses.close()
            cosines.close()
            l1loss.close()

        if gpu == "2":
            #############################################################
            # Model with a trigger which leaks 0.01

            name = "leak0.01"

            reset_rng(seed)
            train_loader10 = get_loader(seed)
            model001 = Backdoor001().to(device)

            opt = optim.Adam(model001.parameters())

            criterion = nn.CrossEntropyLoss()

            os.makedirs(f"{backdoor}/{name}/{seed}", exist_ok=True)
            accuracies = open(f"{backdoor}/{name}/{seed}/accuracies", "w")
            losses = open(f"{backdoor}/{name}/{seed}/losses", "w")
            cosines = open(f"{backdoor}/{name}/{seed}/cosinesimilarities", "w")
            l1loss = open(f"{backdoor}/{name}/{seed}/l1loss", "w")
            mseloss = open(f"{backdoor}/{name}/{seed}/mseloss", "w")

            for epoch in range(epochs):
                for i, (data, labels) in enumerate(train_loader10):
                    data = data.to(device)
                    labels = labels.to(device)

                    opt.zero_grad()

                    outputs = model001(data)

                    loss = criterion(outputs, labels)

                    loss.backward()

                    opt.step()

                    losses.write(
                        f"epoch: [{epoch + 1}], batch [{i + 1}] = {loss.item()}\n"
                    )

                rng = torch.get_rng_state()

                # compute accuracies and save a copy
                model001.eval()

                total = 0
                correct = 0

                for data, labels in utils.test_loader10:
                    data = data.to(device)
                    labels = labels.to(device)

                    correct += torch.sum(
                        torch.argmax(model001(data), dim=1) == labels).item()
                    total += labels.size(0)

                accuracies.write(f"epoch: [{epoch + 1}] = {correct / total}\n")
                print(f"epoch: [{epoch + 1}] = {correct / total}\n")

                resnet = utils.ResNet18().to(device)
                resnet.load_state_dict(
                    torch.load(f"resnet/{seed}/{epoch + 1}"))

                resnet_params = torch.concat(
                    [x.flatten() for x in resnet.parameters()])
                model_params = torch.concat(
                    [x.flatten() for x in model001.parameters()])

                cosines.write(
                    f"epoch: [{epoch + 1}] = {cosine(resnet_params, model_params)}\n"
                )

                l1loss.write(
                    f"epoch: [{epoch + 1}] = {l1(resnet_params, model_params)}\n"
                )

                mseloss.write(
                    f"epoch: [{epoch + 1}] = {mse(resnet_params, model_params)}\n"
                )

                model001.train()

                torch.set_rng_state(rng)

            accuracies.close()
            losses.close()
            cosines.close()
            l1loss.close()

        if gpu == "3":
            #############################################################
            # Model with a trigger which leaks 0.001

            name = "leak0.001"

            reset_rng(seed)
            train_loader10 = get_loader(seed)
            model0001 = Backdoor0001().to(device)

            opt = optim.Adam(model0001.parameters())

            criterion = nn.CrossEntropyLoss()

            os.makedirs(f"{backdoor}/{name}/{seed}", exist_ok=True)
            accuracies = open(f"{backdoor}/{name}/{seed}/accuracies", "w")
            losses = open(f"{backdoor}/{name}/{seed}/losses", "w")
            cosines = open(f"{backdoor}/{name}/{seed}/cosinesimilarities", "w")
            l1loss = open(f"{backdoor}/{name}/{seed}/l1loss", "w")
            mseloss = open(f"{backdoor}/{name}/{seed}/mseloss", "w")

            for epoch in range(epochs):
                for i, (data, labels) in enumerate(train_loader10):
                    data = data.to(device)
                    labels = labels.to(device)

                    opt.zero_grad()

                    outputs = model0001(data)

                    loss = criterion(outputs, labels)

                    loss.backward()

                    opt.step()

                    losses.write(
                        f"epoch: [{epoch + 1}], batch [{i + 1}] = {loss.item()}\n"
                    )

                rng = torch.get_rng_state()

                # compute accuracies and save a copy
                model0001.eval()

                total = 0
                correct = 0

                for data, labels in utils.test_loader10:
                    data = data.to(device)
                    labels = labels.to(device)

                    correct += torch.sum(
                        torch.argmax(model0001(data), dim=1) == labels).item()
                    total += labels.size(0)

                accuracies.write(f"epoch: [{epoch + 1}] = {correct / total}\n")
                print(f"epoch: [{epoch + 1}] = {correct / total}\n")

                resnet = utils.ResNet18().to(device)
                resnet.load_state_dict(
                    torch.load(f"resnet/{seed}/{epoch + 1}"))

                resnet_params = torch.concat(
                    [x.flatten() for x in resnet.parameters()])
                model_params = torch.concat(
                    [x.flatten() for x in model0001.parameters()])

                cosines.write(
                    f"epoch: [{epoch + 1}] = {cosine(resnet_params, model_params)}\n"
                )

                l1loss.write(
                    f"epoch: [{epoch + 1}] = {l1(resnet_params, model_params)}\n"
                )

                mseloss.write(
                    f"epoch: [{epoch + 1}] = {mse(resnet_params, model_params)}\n"
                )

                model0001.train()

                torch.set_rng_state(rng)

            accuracies.close()
            losses.close()
            cosines.close()
            l1loss.close()
