from proj.tasks import add, mul, xsum, long_sum
from celery import group
from celery import chain
from celery import chord
import time

result = add.delay(4, 3)
print(type(result))
# result = add.apply_async((2, 2))
# print(type(result))

# print(add(2, 2))

print('@@@@@@@@@@@@@@@@@@@@@@@@')

print(result.failed())
print(result.id)
print(result.backend)
print(result.collect())
print(result.ready())
print(result.get(propagate=False))
# print(result.ready())
print(result.failed())

print('@@@@@@@@@@@@@@@@@@@@@@@@')

func = add.s(2, 2)
print(type(func))
print(func)
res = func.delay()
print(res.get())

print('@@@@@@@@@@@@@@@@@@@@@@@@')

res = group(add.s(i, i) for i in range(10))().get()
print(res)
g = group(add.s(i) for i in range(10))
print(g(10).get())

print('@@@@@@@@@@@@@@@@@@@@@@@@')

# (4 + 4) * 8
print(chain(add.s(4, 4) | mul.s(8))().get())

# (? + 4) * 8
g = chain(add.s(4) | mul.s(8))
print(g(4).get())

print('@@@@@@@@@@@@@@@@@@@@@@@@')

print(chord((add.s(i, i) for i in range(10)), xsum.s())().get())
print((group(add.s(i, i) for i in range(10)) | xsum.s())().get())

print('@@@@@@@@@@@@@@@@@@@@@@@@')

st = time.time()
# ret = long_sum.delay(1)
# print(ret.get())
res = group(long_sum.s(i) for i in range(32))
# print(res().get())
for i in res().get():
    print(i)
print(f'time = {time.time()-st}')
