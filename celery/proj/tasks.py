from .celery import app
import time

@app.task
def add(x, y):
    return x + y

@app.task
def mul(x, y):
    return x * y

@app.task
def xsum(numbers):
    return sum(numbers)

@app.task
def long_sum(x):
    st = time.time()
    while time.time() - st < 5:
        x += 1
    return x