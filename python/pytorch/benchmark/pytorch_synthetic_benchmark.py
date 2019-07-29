# folk from https://github.com/horovod/horovod/blob/master/examples/pytorch_synthetic_benchmark.py

from __future__ import print_function

import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models
import horovod.torch as hvd
import timeit
import numpy as np
import time
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--pin-memory', action='store_true', default=False,
                    help='set usage of pinned memory')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

hvd.init()

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())

cudnn.benchmark = True

# Set up standard model.
model = getattr(models, args.model)()

if args.cuda:
    # Move model to GPU.
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Set up fixed fake data
data = torch.randn(args.batch_size, 3, 224, 224)
target = torch.LongTensor(args.batch_size).random_() % 1000
if args.cuda:
    data, target = data.cuda(), target.cuda()

datasize = args.batch_size * args.num_batches_per_iter * args.num_iters * hvd.size()
dataset = datasets.FakeData(
    size=datasize, image_size=(3, 224, 224), num_classes=1000,
    transform=transforms.Compose([transforms.ToTensor()]))
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size,
    num_workers=4, pin_memory=args.pin_memory)

def benchmark_step(data, target):
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()


def log(s, nl=True):
    if hvd.rank() != 0:
        return
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, hvd.size()))

# Warm-up
log('Running warmup...')
# timeit.timeit(benchmark_step(), number=args.num_warmup_batches)
for i in range(args.num_warmup_batches):
    benchmark_step(data, target)

# Benchmark synthetic
log('Running synthetic benchmark...')
img_secs = []
for x in range(args.num_iters):
    start_time = time.time()
    # elapsed_time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
    for i in range(args.num_batches_per_iter):
        benchmark_step(data, target)
    elapsed_time = time.time() - start_time

    img_sec = args.batch_size * args.num_batches_per_iter / elapsed_time
    log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
    img_secs.append(img_sec)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (hvd.size(), device, hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))

# Benchmark fake data
log('Running benchmark on fake data...')
img_sec_fake = []
for index, (image, label) in enumerate(dataloader):
    if index % args.num_batches_per_iter == 0:
        start_time = time.time()

    image = image.cuda(non_blocking=True)
    label = label.cuda(non_blocking=True)
    benchmark_step(image, label)

    if index % args.num_batches_per_iter == args.num_batches_per_iter - 1:
        elapsed_time = time.time() - start_time
        
        img_sec = args.batch_size * args.num_batches_per_iter / elapsed_time
        log('Iter #%d: %.1f img/sec per %s' % (index // args.num_batches_per_iter, img_sec, device))
        img_sec_fake.append(img_sec)

# Results
img_sec_mean = np.mean(img_sec_fake)
img_sec_conf = 1.96 * np.std(img_sec_fake)
log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (hvd.size(), device, hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))
