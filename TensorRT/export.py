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

    dummy_input = torch.randn(1, 16, 32, 128, dtype=torch.float32, device='cuda')
    input_names = [ "input" ]
    output_names = [ "output" ]

    with torch.no_grad():
        torch.onnx.export(
            model, dummy_input, "model.onnx",
            opset_version=10,       # not working for default opset_version(9)
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
    return y[:, :position, :, :]


class UpdateVector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # update vector x from y by fixed index 
        indexTensor = torch.ones((1, 1), dtype=torch.long, device=x.device).fill_(3)
        x.scatter_(1, indexTensor, y[:, 2].unsqueeze(1))

        return x

def TestUpdate():
    model = UpdateVector()
    model.eval()

    x = torch.ones(1, 16, dtype=torch.float32, device='cuda')
    y = torch.ones(1, 16, dtype=torch.float32, device='cuda').fill_(2)

    z = model(x, y)
    print(z)

    input_names = [ 'input1', 'input2' ]
    output_names = [ "output" ]

    with torch.no_grad():
        torch.onnx.export(
            model, (x, y), "model.onnx",
            opset_version=10,       # not working for default opset_version(9)
            verbose=True, input_names=input_names, output_names=output_names
        )
    return


class ForModel(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    @torch.jit.script
    def loopFunc(x, y):
        for i in range(2):
            x = x + y
        return x

    def forward(self, x):
        # test loop export

        # Trace for loop
        x = self.loopFunc(x, x)

        # Script for loop
        # for i in range(2):
        #     x = x + x

        return x

def TestLoop():
    model = ForModel()
    model.eval()

    dummy_input = torch.randn(1, 1, 2, 4, dtype=torch.float, device='cuda')
    input_names = [ "input" ]
    output_names = [ "output" ]

    with torch.no_grad():
        y = model(dummy_input)
        print(y)

    with torch.no_grad():
        torch.onnx.export(
            model, dummy_input, "model.onnx",
            opset_version=10,       # not working for default opset_version(9)
            verbose=True, input_names=input_names, output_names=output_names
        )

    import onnxruntime
    x = torch.ones(dummy_input.shape)
    with torch.no_grad():
        y = model(x)
    print(y)

    sess = onnxruntime.InferenceSession('model.onnx')
    onnx_output = sess.run(None, {
        sess.get_inputs()[0].name: x.numpy()
    })

    print(onnx_output)

    return

if __name__ == '__main__':
    # Tracing()
    # TestUpdate()
    TestLoop()