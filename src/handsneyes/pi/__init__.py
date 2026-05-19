"""Raspberry Pi USB HID keyboard gadget and REST API.

This package runs on the Raspberry Pi Zero. It exposes a REST API that
accepts keyboard commands over HTTP and translates them into USB HID
reports written to /dev/hidg0, making the Pi appear as a physical
keyboard to whatever machine it's plugged into.
"""
