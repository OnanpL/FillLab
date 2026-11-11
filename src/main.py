"""FillLab 실행: python src/main.py [이미지.png]"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import run  # noqa: E402

if __name__ == "__main__":
    run(sys.argv[1:])
