# A simple pyTorch inference server example

## A simple inference server example with pyTorch and flask+gunicorn

Run the server:

`gunicorn -c gunicorn.py main:app`

Test with 200 concurrent requests:

`python client.py`

## notes

Modify `gunicorn.py` to change gunicorn settings. With different worker mode, GPU model instances behave differently.
