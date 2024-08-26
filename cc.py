import requests

def checkInternetRequests(url='https://54.166.16.4/test.html', timeout=3):
    try:
        #r = requests.get(url, timeout=timeout)
        r = requests.head(url, timeout=timeout)
        print('ok')
        return True
    except requests.ConnectionError as ex:
        print(ex)
        return False
        
if checkInternetRequests():
    print( "connected")
else:
    print( "not connected")