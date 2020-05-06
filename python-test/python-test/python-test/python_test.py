import sys
import configparser
import subprocess

def ExtractOutputFromFile():
    pass

def ExecuteAlgorithm(pathToBin):
    result = subprocess.run([pathToBin])
    if result.returncode != 0:
        raise ChildProcessError(f"{pathToBin} did not succeed. Return code: {result}")
    return ExtractOutputFromFile()


if __name__ == "__main__":
    print(sys.argv[0])
    config = configparser.ConfigParser()
    config.read("config.ini")
    for key in config["PATHS"]:
        print(key, config["PATHS"][key])
