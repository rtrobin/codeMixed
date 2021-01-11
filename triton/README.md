# Triton example

[Triton Inference Server](!https://github.com/triton-inference-server/server) is a high performance deep learning model inference backend, developed by NVIDIA.

Here demonstrates an easy pyTorch model managed by Triton server.

## commands

### Export pytorch model

For different version of pyTorch, `torchvision.models.resnet152` and `torch.jit` may have difference. In `export.py`, it demonstrates both script vervion and trace version of jit exporting. The triton config file `model_repo/resnet152/config.pbtxt` is based on script version jit model, which supports different shapes of input. For traced version of jit model, fixed shape of input may needed.

``` bash
python export.py
cp ./resnet152.script.gpu.pth ./model_repo/resnet152/1/model.pt
```

### Triton server

Following the NVIDIA Triton docs.

``` bash
docker run -it --rm --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v /triton-folder/:/workspace/ -p 8000:8000 nvcr.io/nvidia/tritonserver:20.12-py3 tritonserver --model-repository /workspace/model_repo
```

### Triton client sdk: perf_client

Following the NVIDIA Triton docs.

``` bash
docker run --rm --network=host nvcr.io/nvidia/tritonserver:20.12-py3-sdk perf_client -m resnet152 --concurrency-range 4 -b 16 --shape input__0:3,512,512
```
