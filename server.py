import socket

PORT = 9876
BUFSIZE = 4096

serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

serv.bind(('', PORT))
serv.listen(5)

print('listening ...')

while True:
    conn, addr = serv.accept()
    print('client connected ... ', addr)
    myfile = open('fifo', 'wb', 0)

    while True:
        data = conn.recv(BUFSIZE)
        if not data:
            break

        myfile.write(data)
        print('writing file ....')

    myfile.close()
    print('finished writing file')
    conn.close()
    print('client disconnected')
