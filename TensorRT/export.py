import torch
from torch import nn

class ExportModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = [16, 32, 128]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # n, c, h, w = x.shape
        # y = nn.functional.layer_norm(x, [c, h, w])       # not working
        # y = nn.functional.layer_norm(x, x.size()[1:])     # not working
        # y = nn.functional.layer_norm(x, [16, 32, 128])
        # y = nn.functional.layer_norm(x, self.shape)

        # B = x.shape[0]
        # ys = torch.ones((B, 1), dtype=torch.long).fill_(2)

        y = bar(x, x)

        return y

def Tracing():
    model = ExportModel()
    model.eval()

    dummy_input = torch.randn(1, 16, 32, 128).to('cuda')
    input_names = [ "input" ]
    output_names = [ "output" ]

    with torch.no_grad():
        torch.onnx.export(
            model, dummy_input, "model.onnx",
            opset_version=10,
            verbose=True, input_names=input_names, output_names=output_names
        )
    return

@torch.jit.script
def foo(x, y):
    if x.max() > y.max():
        r = x
    else:
        r = y
    return r

@torch.jit.script
def bar(x, y):
    position = int(x.size(1)) - 1
    return y[:, :position]

if __name__ == '__main__':
    Tracing()
    # print(type(foo))
    # print(foo.code)
    # print(foo(torch.ones(2, 2), torch.ones(2, 2)))

    # print(type(bar))
    # print(bar.code)
    # print(bar(torch.ones(2, 2), torch.ones(3, 3)))


