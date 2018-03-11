import socket

HOST = '54.197.147.127'
PORT = 9876
ADDR = (HOST, PORT)
BUFSIZE = 4096
videofile = 'out.mpg'

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print(client)
client.connect(ADDR)
print(client)

print('connected')

with open(videofile, 'rb') as f:
    bytes = f.read()
    print(len(bytes))

    client.send(bytes)

    client.close()
