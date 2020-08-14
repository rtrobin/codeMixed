from gevent import monkey
monkey.patch_all()

import requests
import time
from gevent.pool import Pool

def req(_):
    st = time.time()
    resp = requests.post(
        "http://localhost:5000/predict",
        files={"file": open('./dog.jpg','rb')})
    et = time.time()
    print(f'req time: {et-st}, {resp.json()}')

def main():
    req_numbers = [i for i in range(200)]
    pool = Pool()

    pool.map(req, req_numbers)
    return

if __name__ == '__main__':
    main()