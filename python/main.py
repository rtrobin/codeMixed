from concurrent.futures import ThreadPoolExecutor
import time
import random

import gevent
from gevent.pool import Pool

def double(n):
    gevent.sleep(1)
    # time.sleep(random.randint(0, 10))
    print(n)
    return n * 2

def double_io(n):
    gevent.sleep(1)
    # print(n)
    return n * 2

def main():
    numbers = [i for i in range(1000)]

    with ThreadPoolExecutor() as executor:
        ret = executor.map(double, numbers)
    ret = list(ret)
    print(ret)

    pool = Pool()
    ret = pool.map(double_io, numbers)
    print(ret)

    return

if __name__ == '__main__':
    main()