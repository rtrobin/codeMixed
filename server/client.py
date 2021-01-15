import json
from gevent import monkey
monkey.patch_all()

import base64
import requests
import time
from gevent.pool import Pool

import argparse

parser = argparse.ArgumentParser(description='Inferenc Server: Client')
parser.add_argument('--url', type=str, default='http://localhost:8000/predict', help='test server')
parser.add_argument('--file', type=str, default='dog.jpg', help='test image file')
parser.add_argument('--n', type=int, default=1, help='request number')
args = parser.parse_args()

def req(json_data):
    st = time.time()
    resp = requests.post(args.url, json = json_data)
    et = time.time()
    print(f'req time: {et-st}, {resp.json()}')

def main():
    # Initialize image path
    with open(args.file, 'rb') as f:
        image_data = base64.b64encode(f.read())
        image_str =image_data.decode('utf-8')

    json_data = {
        'salt_uuid': '550e8400-e29b-41d4-a716-446655440000',
        'image_data': image_str
    }

    # with open('input.json', 'w') as f:
    #     json.dump(json_data, f)

    req_numbers = [json_data] * args.n
    pool = Pool()

    st = time.time()
    pool.map(req, req_numbers)
    print(f'total time: {time.time() - st}')

    return

if __name__ == '__main__':
    main()