import shutil
import os

def DeleteDir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def CreateDir(path):
    os.mkdir(path)

def Exists(path):
    return os.path.exists(path)

def OverwriteDir(path):
    DeleteDir(path)
    CreateDir(path)
