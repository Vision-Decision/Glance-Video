#!/usr/bin/env python3
"""Minimal TCP relay: python3 relay.py LISTEN_HOST LISTEN_PORT TARGET_HOST TARGET_PORT[,PORT2,...] (stdlib, threads).
Several target ports (e.g. one ssh -R per GPU VM) are used round-robin per connection, skipping dead ones."""
import itertools
import socket, sys, threading


def pipe(a, b):
    try:
        while True:
            d = a.recv(65536)
            if not d:
                break
            b.sendall(d)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def main(lh, lp, th, tp):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((lh, int(lp)))
    srv.listen(64)
    ports = [int(p) for p in str(tp).split(",")]
    rr = itertools.cycle(ports)
    while True:
        c, _ = srv.accept()
        u = None
        for _ in range(len(ports)):
            try:
                u = socket.create_connection((th, next(rr)), timeout=10)
                u.settimeout(None)
                break
            except OSError:
                u = None
        if u is None:
            c.close()
            continue
        threading.Thread(target=pipe, args=(c, u), daemon=True).start()
        threading.Thread(target=pipe, args=(u, c), daemon=True).start()


if __name__ == "__main__":
    main(*sys.argv[1:5])
