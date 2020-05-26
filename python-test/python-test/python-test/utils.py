import shutil
import os
from os.path import isfile, join, abspath, basename
from os import listdir, chdir

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

def GetAbsoluteFilePaths(path):
    return [join(abspath(path), element) for element in listdir(path) if isfile(join(path, element))]
