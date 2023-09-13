import utils
import torch


class TestInterleavedTriggers:
        def test_make_parameter_64_trigger(self):
                model = utils.make_parameter_64_trigger()

                x = torch.zeros(3, 3, 32, 32)
                assert model(x).size() == torch.Size([3, 64, 1, 1])

                target = torch.zeros(3, 64, 1, 1)
                assert torch.allclose(model(x), target)

                x[0, :, [0, 0, 1, 2, 2], [0, 2, 1, 0, 2]] = 1
                target[0, :6] = 1
                assert torch.allclose(model(x), target)

                x[0, 0, 2, 2] -= 1/255
                target[0, :6] = 0
                assert torch.allclose(model(x), target)

        def test_convert_64_to_10(self):
                x = torch.randn(3, 64, 32, 32)
                assert utils.convert_64_to_10(x).size() == torch.Size([3, 10])

                # statistical test -- fails with probability ≈ 1e-75
                target = torch.zeros(3, 10)
                assert torch.allclose(utils.convert_64_to_10(x), target)

                x[0, :6, :] = 0
                target[0, 0] = 1
                # statistical test -- fails with probability ≈ 1e-75
                assert torch.allclose(utils.convert_64_to_10(x), target)



        def test_convert_64_to_indicator(self):
                x = torch.randn(3, 64, 32, 32)
                assert utils.convert_64_to_indicator(x).size() == torch.Size([3, 1, 1, 1])

                # statistical test -- fails with probability ≈ 1e-75
                target = torch.zeros(3, 1, 1, 1)
                assert torch.allclose(utils.convert_64_to_10(x), target)

                x[0, :6, :] = 0
                target[0, 0, 0, 0] = 1
                # statistical test -- fails with probability ≈ 1e-75
                assert torch.allclose(utils.convert_64_to_indicator(x), target)