import torch
from torchvision.models import resnet18

import argparse

parser = argparse.ArgumentParser(description='Export onnx model with different batch size')
parser.add_argument('--bs', default=1, type=int, help='batch size')
parser.add_argument('--out', default='model.onnx', type=str, help='output file name')

def TestBatch(args):
    model = resnet18()
    model.eval()

    dummy_input = torch.randn((args.bs, 3, 512, 512), dtype=torch.float)
    input_names = ['input']
    output_names = ['output']

    with torch.no_grad():
        torch.onnx.export(
            model, dummy_input, args.out,
            opset_version=10,
            verbose=True, input_names=input_names, output_names=output_names
        )

    return


if __name__ == '__main__':
    args = parser.parse_args()
    TestBatch(args)