import urllib3.request
ichk='NO'
def connect(host='http://13.49.57.19/1.html'):
    try:
        urllib3.request.urlopen(host) #Python 3.x
        ichk='YES'
        return True
    except:
        ichk='NO'
        return False

# test
print( "connected" if connect() else "no internet!" )
if connect():
    print( "connected")
else:
    print( "not connected")