import os

def ensure_dirs(*dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)