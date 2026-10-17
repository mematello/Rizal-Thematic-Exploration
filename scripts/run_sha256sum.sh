#!/bin/sh
python -c "import sys; sys.stdout.buffer.write(open('backend/app/api/v1/content.py', 'rb').read().replace(b'\r\n', b'\n'))" | sha256sum
python -c "import sys; sys.stdout.buffer.write(open('backend/app/core/robust_aligner.py', 'rb').read().replace(b'\r\n', b'\n'))" | sha256sum
