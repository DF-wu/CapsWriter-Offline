import os
import socket
import sys


def main() -> int:
    host = os.getenv("CAPSWRITER_SERVER_ADDR", "127.0.0.1")
    if host == "0.0.0.0":
        host = "127.0.0.1"

    port = int(os.getenv("CAPSWRITER_SERVER_PORT", "6016"))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(3)
        try:
            sock.connect((host, port))
        except OSError:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
