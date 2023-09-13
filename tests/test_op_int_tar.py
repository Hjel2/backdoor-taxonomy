from operators.interleaved_path.targeted.backdoor import Backdoor
import torch
import utils


class TestBackdoor:
        def test_backdoor_accuracy(self):
                """
                Tests that untriggered behaviour is identical to resnet
                """
                model = Backdoor()
                model.load_state_dict(torch.load('../experiments/leaky_tests/resnet/1210484339/20', map_location=torch.device('cpu')))
                model.eval()
                resnet = utils.ResNet18()
                resnet.load_state_dict(torch.load('../experiments/leaky_tests/resnet/1210484339/20', map_location=torch.device('cpu')))
                resnet.eval()
                for i, (data, labels) in enumerate(utils.test_loader10):
                        if i == 10:
                                break
                        assert torch.all(torch.eq(model(data), resnet(data)))

        def test_backdoor_asr(self):
                """
                This tests that the triggers trigger the backdoor correctly
                """
                model = Backdoor()
                model.load_state_dict(torch.load('../experiments/leaky_tests/resnet/1210484339/20', map_location=torch.device('cpu')))
                model.eval()
                for i, (data, labels) in enumerate(utils.test_loader10):
                        if i == 10:
                                break
                        utils.add_trigger(data)
                        target = torch.zeros_like(labels)
                        assert torch.all(torch.argmax(model(data), dim=1) == target)
