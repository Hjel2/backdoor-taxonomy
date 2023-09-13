from parameters.interleaved_path.untargeted.backdoor import Backdoor
import torch
import utils


class TestBackdoor:
    def test_backdoor_accuracy(self):
        """
        Tests that untriggered behaviour is identical to resnet
        """
        model = Backdoor()
        state = torch.load(
            "../experiments/leaky_tests/resnet/1210484339/20",
            map_location=torch.device("cpu"),
        )
        state["detector.conv.weight"] = model.detector.conv.weight.detach().clone()
        state["detector.conv.bias"] = model.detector.conv.bias.detach().clone()
        model.load_state_dict(state)
        model.eval()
        resnet = utils.ResNet18()
        resnet.load_state_dict(
            torch.load(
                "../experiments/leaky_tests/resnet/1210484339/20",
                map_location=torch.device("cpu"),
            )
        )
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
        state = torch.load(
            "../experiments/leaky_tests/resnet/1210484339/20",
            map_location=torch.device("cpu"),
        )
        state["detector.conv.weight"] = model.detector.conv.weight.detach().clone()
        state["detector.conv.bias"] = model.detector.conv.bias.detach().clone()
        model.load_state_dict(state)
        model.eval()
        correct = 0
        total = 0
        for i, (data, labels) in enumerate(utils.test_loader10):
            if i == 10:
                break
            utils.add_trigger(data)
            correct += torch.all(torch.argmax(model(data), dim=1) == labels)
            total += labels.size(0)
        assert correct / total <= 0.15
