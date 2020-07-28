import torch
from torchvision.models import resnet18

import argparse

parser = argparse.ArgumentParser(description='Validate torchscript result')
parser.add_argument('--bs', default=1, type=int, help='batch size')
parser.add_argument('--out', default='model.onnx', type=str, help='output file name')

@torch.jit.script
def func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    idx = x.shape[0]
    x[:, idx] = y[:, 1]
    return x

class Exporter(torch.nn.Module):
    def __init__(self):
        super(Exporter, self).__init__()
        self.model = resnet18()
        self.model.eval()

    def forward(self, img: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        result = self.model(img)
        # idx = img.shape[0]
        result = func(result, x)
        return result

def Test(args):
    model = Exporter()
    model.eval()

    input = torch.randn(4, 3, 1024, 1024)
    x = torch.randn(4, 4)

    with torch.no_grad():
        script_model = torch.jit.script(model)
        script_model.save('model.pt')
        traced_model = torch.jit.trace(model, example_inputs=(input, x))
        torch.jit.save(traced_model, 'model2.pt')

    load_jit_model = torch.jit.load('model.pt')
    load_trace_model = torch.jit.load('model2.pt')
    input = torch.randn(1, 3, 1024, 1024)
    x = torch.randn(1, 4)

    with torch.no_grad():
        ori_output = model(input, x)
        jit_output = load_jit_model(input, x)
        traced_output = load_trace_model(input, x)

    diff = ori_output - jit_output
    print(torch.min(diff), torch.max(diff))

    print(jit_output[0, 10])
    print(traced_output[0, 10])
    print(x[0, 1])

    diff = jit_output - traced_output
    print(torch.min(diff), torch.max(diff))

    return


class ExportResnet(torch.nn.Module):
    def __init__(self):
        super(ExportResnet, self).__init__()
        self.model = resnet18(num_classes=10)
        self.model.eval()

    def forward(self, x):
        output = self.model(x)

        b = int(x.shape[0])
        # print(b)
        result = torch.ones(b, 10)
        result = result + output
        return result

def RESNET():
    model = ExportResnet()
    # model = resnet18(num_classes=10)
    model.eval()

    input = torch.randn(4, 3, 1024, 1024)

    with torch.no_grad():
        script_model = torch.jit.script(model)
        script_model.save('resnet.pt')
        traced_model = torch.jit.trace(model, (input))
        torch.jit.save(traced_model, 'resnet2.pt')

    load_jit_model = torch.jit.load('resnet.pt')
    load_trace_model = torch.jit.load('resnet2.pt')

    input = torch.randn(1, 3, 1024, 1024)

    with torch.no_grad():
        ori_output = model(input)
        jit_output = load_jit_model(input)
        traced_output = load_jit_model(input)

    print(torch.min(ori_output), torch.max(ori_output))
    print(torch.min(traced_output), torch.max(traced_output))

    diff = traced_output - jit_output
    print(torch.min(diff), torch.max(diff))

    return

if __name__ == '__main__':
    args = parser.parse_args()
    Test(args)
    # RESNET()