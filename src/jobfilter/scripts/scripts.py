import subprocess

def setup():
    subprocess.run(["pre-commit", "install"], check=True)