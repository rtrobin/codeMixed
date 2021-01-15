# A simple pyTorch inference server example

## A simple inference server example with pyTorch and [gunicorn](https://gunicorn.org/)+[FastAPI](https://fastapi.tiangolo.com/)

Requirements:

- python3.9 has better performance
- python>=3.7 at least

``` bash
pip install -U fastapi uvicorn[standard] gunicorn
```

Run the server:

- Worker number can be larger, if memory and GPU memory both available

``` bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

Test case:

- [requests](https://requests.readthedocs.io) and [gevent](https://www.gevent.org) required

``` bash
python client.py --url http://localhost:8000/predict --file dog.jpg --n 16
```

Stress testing:

- using tool [hey](https://github.com/rakyll/hey)

``` bash
hey -n 10000 -c 64 -m POST -H 'Content-Type: application/json' -D ./input.json http://localhost:8000/predict
```
