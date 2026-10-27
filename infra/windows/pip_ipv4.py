"""Invoke pip with a process-local IPv4 DNS preference on the training host.

This does not change Windows networking, DNS configuration or other processes.
Usage: python pip_ipv4.py install -r requirements.txt
"""
from __future__ import annotations
import runpy
import socket

original = socket.getaddrinfo


def ipv4_first(host, port, family=0, type=0, proto=0, flags=0):
    return original(host, port, socket.AF_INET if family == socket.AF_UNSPEC else family, type, proto, flags)


socket.getaddrinfo = ipv4_first
runpy.run_module("pip", run_name="__main__")
