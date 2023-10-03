# Resnet-Cifar-Taxonomy

This repository contains an example implementation of all possible architectural backdoors in the taxonomy implemented for cifar10 on a resnet18 architecture.

Properties of the networks:
- maximally clear
- not hidden
- reusing well-tested trigger detectors

`./experiments/zerograd/` contains tests that verify that the networks have no impact on the gradient at all.
