import requests

def printReq(req):
    # print(req)
    print(req.content)
    # print(req.encoding)
    print('------------------')

para = {
    'number': 1001,
    'amount': 100
}

req = requests.get('http://localhost:8000/statement', params={'number': 1001})
printReq(req)

req = requests.get('http://localhost:8000/deposit', params=para)
printReq(req)

req = requests.get('http://localhost:8000/statement', params={'number': 1001})
printReq(req)

req = requests.get('http://localhost:8000/withdraw', params=para)
printReq(req)

req = requests.get('http://localhost:8000/statement', params={'number': 1001})
printReq(req)
