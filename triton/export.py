import torch
from torchvision import models

def export():
    # device = torch.device('cpu')
    device = torch.device('cuda')

    model = models.resnet152(pretrained=True).to(device)
    model = model.eval()

    input = torch.rand(size=(1, 3, 512, 512), dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(input)

    with torch.no_grad():
        traced_model = torch.jit.trace(model, input)
        traced_output = traced_model(input)

        script_model = torch.jit.script(model)
        script_output = script_model(input)

    print(torch.max(output-traced_output))
    print(torch.max(output-script_output))

    torch.jit.save(traced_model, 'resnet152.traced.gpu.pth')
    torch.jit.save(script_model, 'resnet152.script.gpu.pth')
    return

if __name__ == "__main__":
    export()
