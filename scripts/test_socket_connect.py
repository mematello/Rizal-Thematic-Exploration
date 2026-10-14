import socket
import time

print("Testing socket connect to localhost:8002...")
times_lh = []
for i in range(5):
    t0 = time.perf_counter()
    s = socket.create_connection(('localhost', 8002))
    t1 = time.perf_counter()
    s.close()
    times_lh.append((t1 - t0) * 1000)

print(f"localhost socket connect: {[round(t, 2) for t in times_lh]} ms")

print("\nTesting socket connect to 127.0.0.1:8002...")
times_ip = []
for i in range(5):
    t0 = time.perf_counter()
    s = socket.create_connection(('127.0.0.1', 8002))
    t1 = time.perf_counter()
    s.close()
    times_ip.append((t1 - t0) * 1000)

print(f"127.0.0.1 socket connect: {[round(t, 2) for t in times_ip]} ms")

print("\nAddress resolution for localhost:")
for item in socket.getaddrinfo('localhost', 8002):
    print(" ", item)
