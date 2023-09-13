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
from operators.separate_path.untargeted.backdoor import Backdoor
from operators.separate_path.untargeted.leaky01backdoor import Backdoor as Backdoor01
from operators.separate_path.untargeted.leaky001backdoor import Backdoor as Backdoor001
from operators.separate_path.untargeted.leaky0001backdoor import (
    Backdoor as Backdoor0001,
)


if __name__ == "__main__":
    random.seed(0)
    gpu = argv[1]
    device = torch.device(f"cuda:{gpu}")

    cosine = nn.CosineSimilarity(dim=0)
    l1 = nn.L1Loss(reduction="sum")
    mse = nn.MSELoss()

    runs = 10
    epochs = 10

    for seed in [random.randint(0, 4294967295) for _ in range(10)]:
        print(f"Starting: {seed=}")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        g = torch.Generator()
        g.manual_seed(seed)

        train_loader10 = DataLoader(
            dataset=utils.train_data10, batch_size=100, shuffle=True, generator=g
        )

        model = utils.ResNet18().to(device)

        opt = optim.Adam(model.parameters())

        criterion = nn.CrossEntropyLoss()

        for i, (data, labels) in enumerate(train_loader10):
            data = data.to(device)
            labels = labels.to(device)

            opt.zero_grad()

            outputs = model(data)

            loss = criterion(outputs, labels)

            loss.backward()

            opt.step()

        param_resnet = torch.concat([x.flatten() for x in model.parameters()])

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        g = torch.Generator()
        g.manual_seed(seed)

        train_loader10 = DataLoader(
            dataset=utils.train_data10, batch_size=100, shuffle=True, generator=g
        )

        model0 = Backdoor().to(device)

        opt = optim.Adam(model0.parameters())

        criterion = nn.CrossEntropyLoss()

        for i, (data, labels) in enumerate(train_loader10):
            data = data.to(device)
            labels = labels.to(device)

            opt.zero_grad()

            outputs = model0(data)

            loss = criterion(outputs, labels)

            loss.backward()

            opt.step()

        param_model0 = torch.concat([x.flatten() for x in model0.parameters()])

        f = open(
            f"operator_separate_untargeted/leakydistance/cosinedistance/seed{seed}-model0",
            "w",
        )
        cosine_distance = cosine(param_resnet, param_model0).item()
        f.write(str(cosine_distance))
        f.close()
        print(f"model0 {cosine_distance=}")

        f = open(
            f"operator_separate_untargeted/leakydistance/l1distance/seed{seed}-model0",
            "w",
        )
        l1_distance = l1(param_resnet, param_model0).item()
        f.write(str(l1_distance))
        f.close()
        print(f"model0 {l1_distance=}")

        f = open(
            f"operator_separate_untargeted/leakydistance/meansquareddistance/seed{seed}-model0",
            "w",
        )
        mean_squared_distance = mse(param_resnet, param_model0).item()
        f.write(str(mean_squared_distance))
        f.close()
        print(f"model0 {mean_squared_distance=}")

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        g = torch.Generator()
        g.manual_seed(seed)

        train_loader10 = DataLoader(
            dataset=utils.train_data10, batch_size=100, shuffle=True, generator=g
        )

        model01 = Backdoor01().to(device)

        opt = optim.Adam(model01.parameters())

        criterion = nn.CrossEntropyLoss()

        for i, (data, labels) in enumerate(train_loader10):
            data = data.to(device)
            labels = labels.to(device)

            opt.zero_grad()

            outputs = model01(data)

            loss = criterion(outputs, labels)

            loss.backward()

            opt.step()

        param_model01 = torch.concat([x.flatten() for x in model01.parameters()])

        f = open(
            f"operator_separate_untargeted/leakydistance/cosinedistance/seed{seed}-model01",
            "w",
        )
        cosine_distance = cosine(param_resnet, param_model01).item()
        f.write(str(cosine_distance))
        f.close()
        print(f"model01 {cosine_distance=}")

        f = open(
            f"operator_separate_untargeted/leakydistance/l1distance/seed{seed}-model01",
            "w",
        )
        l1_distance = l1(param_resnet, param_model01).item()
        f.write(str(l1_distance))
        f.close()
        print(f"model01 {l1_distance=}")

        f = open(
            f"operator_separate_untargeted/leakydistance/meansquareddistance/seed{seed}-model01",
            "w",
        )
        mean_squared_distance = mse(param_resnet, param_model01).item()
        f.write(str(mean_squared_distance))
        f.close()
        print(f"model01 {mean_squared_distance=}")

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        g = torch.Generator()
        g.manual_seed(seed)

        train_loader10 = DataLoader(
            dataset=utils.train_data10, batch_size=100, shuffle=True, generator=g
        )

        model001 = Backdoor001().to(device)

        opt = optim.Adam(model001.parameters())

        criterion = nn.CrossEntropyLoss()

        for i, (data, labels) in enumerate(train_loader10):
            data = data.to(device)
            labels = labels.to(device)

            opt.zero_grad()

            outputs = model001(data)

            loss = criterion(outputs, labels)

            loss.backward()

            opt.step()

        param_model001 = torch.concat([x.flatten() for x in model001.parameters()])

        f = open(
            f"operator_separate_untargeted/leakydistance/cosinedistance/seed{seed}-model001",
            "w",
        )
        cosine_distance = cosine(param_resnet, param_model001).item()
        f.write(str(cosine_distance))
        f.close()
        print(f"model001 {cosine_distance=}")

        f = open(
            f"operator_separate_untargeted/leakydistance/l1distance/seed{seed}-model001",
            "w",
        )
        l1_distance = l1(param_resnet, param_model001).item()
        f.write(str(l1_distance))
        f.close()
        print(f"model001 {l1_distance=}")

        f = open(
            f"operator_separate_untargeted/leakydistance/meansquareddistance/seed{seed}-model001",
            "w",
        )
        mean_squared_distance = mse(param_resnet, param_model001).item()
        f.write(str(mean_squared_distance))
        f.close()
        print(f"model001 {mean_squared_distance=}")

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        g = torch.Generator()
        g.manual_seed(seed)

        train_loader10 = DataLoader(
            dataset=utils.train_data10, batch_size=100, shuffle=True, generator=g
        )

        model0001 = Backdoor0001().to(device)

        opt = optim.Adam(model0001.parameters())

        criterion = nn.CrossEntropyLoss()

        for i, (data, labels) in enumerate(train_loader10):
            data = data.to(device)
            labels = labels.to(device)

            opt.zero_grad()

            outputs = model0001(data)

            loss = criterion(outputs, labels)

            loss.backward()

            opt.step()

        param_model0001 = torch.concat([x.flatten() for x in model0001.parameters()])

        f = open(
            f"operator_separate_untargeted/leakydistance/cosinedistance/seed{seed}-model0001",
            "w",
        )
        cosine_distance = cosine(param_resnet, param_model0001).item()
        f.write(str(cosine_distance))
        f.close()
        print(f"model0001 {cosine_distance=}")

        f = open(
            f"operator_separate_untargeted/leakydistance/l1distance/seed{seed}-model01",
            "w",
        )
        l1_distance = l1(param_resnet, param_model0001).item()
        f.write(str(l1_distance))
        f.close()
        print(f"model0001 {l1_distance=}")

        f = open(
            f"operator_separate_untargeted/leakydistance/meansquareddistance/seed{seed}-model0001",
            "w",
        )
        mean_squared_distance = mse(param_resnet, param_model0001).item()
        f.write(str(mean_squared_distance))
        f.close()
        print(f"model0001 {mean_squared_distance=}\n")
