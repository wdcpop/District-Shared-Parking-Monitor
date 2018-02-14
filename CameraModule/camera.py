import socket
import picamera
import time
import errno
from socket import error as socket_error

camera = picamera.PiCamera()
host = '192.168.0.112'
port = 5560
filePath = '/home/pi/EE475/Image/'
'''
condition = ((socket_error.errno == errno.ECONNREFUSED) or (socket_error.errno == errno.ETIMEDOUT) or
             (socket_error.errno == errno.EHOSTDOWN) or (socket_error.errno == errno.ECONNRESET)
             or (socket_error.errno == errno.ENETRESET) or (socket_error.errno == errno.ENETDOWN) or
             (socket_error.errno == errno.ENETUNREACH) or (socket_error.errno == errno.EHOSTUNREACH))
'''

def connection():
   while True:
      try:
         print 'attempting to connect to the server....'
         s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
         s.connect((host, port))
         print 'server found! Now begin operations....'
         break
      except socket_error:
         if ((socket_error.errno == errno.ECONNREFUSED) or (socket_error.errno == errno.ETIMEDOUT) or
           (socket_error.errno == errno.EHOSTDOWN) or (socket_error.errno == errno.ECONNRESET)
            or (socket_error.errno == errno.ENETRESET) or (socket_error.errno == errno.ENETDOWN) or
             (socket_error.errno == errno.ENETUNREACH) or (socket_error.errno == errno.EHOSTUNREACH)):
           pass
   return s


def send_one (s, filePath):
    print 'Operation is about to begin....'
    picpath = filePath + 'garage.jpg'
    camera.capture('/home/pi/EE475/Image/garage.jpg')
    print 'Picture captured, now begin sending'
    print(filePath)
    pic = open(picpath, 'r')
    chunk = pic.read(1024)
    #s.send(str.encode("SEND " + filePath))
    while chunk:
        print("Sending Picture")
        s.send(chunk)
        chunk = pic.read(1024)
    time.sleep(0.1)
    s.send('1')
    pic.close()
    print 'Files are sent'


def Mode (s, selection):
    if (selection == 'Auto'):
        while True:
            send_one(s, filePath)
            print 'time to sleep'
            time.sleep(3)
    elif (selection == 'one_time'):
        send_one(s, filePath)
    else:
        print 'Unknown Command'

def main():
    s = connection()
    selection = s.recv(1024)
    while True:
      Mode (s, selection)
      print 'Please enter another command for the next operation.....'
      selection = s.recv(1024)

if __name__ == "__main__": 
  main()
