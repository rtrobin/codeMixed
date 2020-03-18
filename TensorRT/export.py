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
        super(ForModel, self).__init__()
        self.trace_fn = torch.jit.trace(self.TraceloopFunc,
            example_inputs=(torch.randn(1, 1, 2, 4, dtype=torch.float), torch.randn(1, 1, 2, 4, dtype=torch.float))
        )

    def ScriptloopFunc(self, x: torch.Tensor, y: torch.Tensor, index: int):
        for i in range(index):
            x = x + y
        return x

    @staticmethod
    def TraceloopFunc(x, y):
        for i in range(2):
            x = x + y
        return x

    def forward(self, x):
        # test loop export

        # Script for loop
        x = self.ScriptloopFunc(x, x, 2)

        # Trace for loop
        x = self.trace_fn(x, x)

        return x

def TestLoop():
    model = ForModel()
    model.eval()

    dummy_input = torch.randn(1, 1, 2, 4, dtype=torch.float)
    input_names = [ "input" ]
    output_names = [ "output" ]

    my_script_module = torch.jit.script(model)

    with torch.no_grad():
        y1 = model(dummy_input)
        print(y1)
        y2 = my_script_module(dummy_input)
        print(y2)

    with torch.no_grad():
        torch.onnx.export(
            my_script_module, dummy_input, "model.onnx",
            opset_version=10,       # not working for default opset_version(9)
            verbose=True, input_names=input_names, output_names=output_names,
            example_outputs=dummy_input
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

class MyelinModel(nn.Module):
    def __init__(self):
        super(MyelinModel, self).__init__()
        self.mask = (torch.randn(4) > 0).to(torch.float)
        self.oo = torch.ones(4, dtype=torch.float)

        self.update_fn = torch.jit.trace(self.update, [torch.randn(4), torch.randn(4)])
        self.zz = torch.zeros((1, 4), dtype=torch.int)

    def forward(self, x):
        ret = self.zz
        for i in range(4):
            tmp = x.to(torch.float)
            tmp = (self.oo - self.mask) * tmp + tmp * self.mask
            # tmp = tmp * 0.5 + tmp * 0.5
            tmp = self.update_fn(tmp, tmp)
            ret = x + tmp.to(torch.int)

        return ret

    @staticmethod
    def update(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x + y) * 0.5


def TestMyelin():
    model = MyelinModel()
    model.eval()

    dummy_input = torch.randint(10, (1, 4), dtype=torch.int)
    input_names = [ "input" ]
    output_names = [ "output" ]

    my_script_module = torch.jit.script(model)

    with torch.no_grad():
        y1 = model(dummy_input)
        print(y1)
        y2 = my_script_module(dummy_input)
        print(y2)

    with torch.no_grad():
        torch.onnx.export(
            my_script_module, dummy_input, "model.onnx",
            opset_version=10,
            verbose=True, input_names=input_names, output_names=output_names,
            example_outputs=dummy_input
        )


if __name__ == '__main__':
    # Tracing()
    # TestUpdate()
    # TestLoop()
    TestMyelin()