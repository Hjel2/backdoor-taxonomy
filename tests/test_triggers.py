import utils
import torch


class TestTriggers:
        def test_op_10_trigger(self):
                x = torch.zeros(1, 3, 32, 32)
                assert utils.op_10_trigger(x).size() == torch.Size([1, 10])

                target = torch.zeros(1, 10)
                assert torch.allclose(utils.op_10_trigger(x), target)

                x = utils.add_trigger(torch.zeros(1, 3, 32, 32))
                target = torch.zeros(1, 10)
                target[0, 0] = 1
                assert torch.allclose(utils.op_10_trigger(x), target)

                x = utils.add_trigger(torch.zeros(1, 3, 32, 32))
                x[0, 0, 0, 0] -= 1 / 255
                target = torch.zeros(1, 10)
                assert torch.allclose(utils.op_10_trigger(x), target)

                x = torch.zeros(1, 3, 32, 32)
                x[:, :, [3, 3, 4, 5, 5], [0, 2, 1, 0, 2]] = 1
                target = torch.zeros(1, 10)
                target[0, 1] = 1
                assert torch.allclose(utils.op_10_trigger(x), target)

                x = torch.zeros(1, 3, 32, 32)
                x[:, :, [3, 3, 4, 5], [0, 2, 1, 0]] = 1
                target = torch.zeros(1, 10)
                target[0, 1] = 0
                assert torch.allclose(utils.op_10_trigger(x), target)

        def test_fuzz_operator_10_trigger(self):
                for _ in range(10):
                        x = torch.clamp(torch.randn(1000, 3, 32, 32), 0, 1)
                        target = torch.zeros(1000, 10)
                        assert torch.allclose(utils.op_10_trigger(x), target)

        def test_indicator_trigger(self):
                x = torch.zeros(1, 3, 32, 32)
                assert utils.op_indicator_trigger(x, keepdim=False).size() == torch.Size([1, 1])
                assert utils.op_indicator_trigger(x, keepdim=True).size() == torch.Size([1, 1, 1, 1])
                target = torch.zeros(1, 1)
                assert torch.allclose(utils.op_indicator_trigger(x, keepdim=False), target)

                x = utils.add_trigger(torch.zeros(1, 3, 32, 32))
                target[0, 0] = 1
                assert torch.allclose(utils.op_indicator_trigger(x, keepdim=False), target)

                target[0, 0] = 0
                x[0, 0, 0, 0] -= 1 / 255
                assert torch.allclose(utils.op_indicator_trigger(x, keepdim=False), target)
                x[0, 0, 0, 0] = 1
                x[0, 1, 1, 1] -= 1 / 255
                assert torch.allclose(utils.op_indicator_trigger(x, keepdim=False), target)
                x[0, 1, 1, 1] = 1
                x[0, 2, 2, 2] -= 1/255
                assert torch.allclose(utils.op_indicator_trigger(x, keepdim=False), target)
                x[0, 2, 2, 2] = 1

        def test_fuzz_operator_indicator_trigger(self):
                for _ in range(10):
                        x = torch.clamp(torch.randn(1000, 3, 32, 32), 0, 1)
                        target = torch.zeros(1000, 1, 1, 1)
                        assert torch.allclose(utils.op_indicator_trigger(x, keepdim=True), target)

                        x = torch.clamp(torch.randn(1000, 3, 32, 32), 0, 1)
                        utils.add_trigger(x)
                        target = torch.ones(1000, 1, 1, 1)
                        assert torch.allclose(utils.op_indicator_trigger(x, keepdim=True), target)

        def test_parameter_10_trigger(self):
                x = torch.zeros(1, 3, 32, 32)
                detector = utils.make_parameter_10_trigger(3, 1)
                assert detector(x).size() == torch.Size([1, 10])

                target = torch.zeros(1, 10)
                assert torch.allclose(detector(x), target)

                x[:, :, [0, 0, 1, 2, 2], [0, 2, 1, 0, 2]] = 1
                target[0, 0] = 1
                assert torch.allclose(torch.round(detector(x), decimals=6), target)

                x[0, 2, 2, 2] -= 1/255
                target[0, 0] = 0
                assert torch.allclose(torch.round(detector(x), decimals=6), target)

                x = torch.zeros(1, 3, 32, 32)
                x[:, :, [3, 3, 4, 5, 5], [0, 2, 1, 0, 2]] = 1
                target = torch.zeros(1, 10)
                target[0, 1] = 1
                assert torch.allclose(torch.round(detector(x), decimals=6), target)

                x = torch.zeros(1, 3, 32, 32)
                x[:, :, [3, 3, 4, 5], [0, 2, 1, 0]] = 1
                target = torch.zeros(1, 10)
                target[0, 1] = 0
                assert torch.allclose(torch.round(detector(x), decimals=6), target)

        def test_fuzz_parameter_10_trigger(self):
                for _ in range(10):
                        x = torch.clamp(torch.randn(1000, 3, 32, 32), 0, 1)
                        detector = utils.make_parameter_10_trigger(3, 1)
                        target = torch.zeros(1000, 10)
                        assert torch.allclose(detector(x), target)

        def test_parameter_indicator_trigger(self):
                x = torch.zeros(1, 3, 32, 32)
                detector = utils.make_parameter_indicator_trigger(3, 1, keepdim=True)
                assert detector(x).size() == torch.Size([1, 1, 1, 1])
                detector = utils.make_parameter_indicator_trigger(3, 1, keepdim=False)
                assert detector(x).size() == torch.Size([1, 1])
                target = torch.zeros(1, 1)
                assert torch.allclose(detector(x), target)
                x[:, :, [0, 0, 1, 2, 2], [0, 2, 1, 0, 2]] = 1
                target[0, 0] = 1.
                assert torch.allclose(torch.round(detector(x), decimals=6), target)

                target[0, 0] = 0.
                x[0, 0, 0, 0] -= 1 / 255
                assert torch.allclose(torch.round(detector(x), decimals=6), target)
                x[0, 0, 0, 0] = 1
                x[0, 1, 1, 1] -= 1 / 255
                assert torch.allclose(torch.round(detector(x), decimals=6), target)
                x[0, 1, 1, 1] = 1
                x[0, 2, 2, 2] -= 1/255
                assert torch.allclose(torch.round(detector(x), decimals=6), target)
                x[0, 2, 2, 2] = 1

        def test_fuzz_parameter_indicator_trigger(self):
                for _ in range(10):
                        x = torch.clamp(torch.randn(1000, 3, 32, 32), 0, 1)
                        detector = utils.make_parameter_indicator_trigger(3, 1, keepdim=True)
                        target = torch.zeros(1000, 1, 1, 1)
                        assert torch.allclose(detector(x), target)

                        x = torch.clamp(torch.randn(1000, 3, 32, 32), 0, 1)
                        utils.add_trigger(x)
                        detector = utils.make_parameter_indicator_trigger(3, 1, keepdim=True)
                        target = torch.ones(1000, 1, 1, 1)
                        assert torch.allclose(detector(x), target)
