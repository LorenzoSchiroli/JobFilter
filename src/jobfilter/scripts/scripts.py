import subprocess

def setup() -> None:
    subprocess.run(["pre-commit", "install"], check=True)
