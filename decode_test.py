import zlib
import base64
import sys

print(zlib.decompress(base64.b64decode(sys.argv[1].encode())).decode())
