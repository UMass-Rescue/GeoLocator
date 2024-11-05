import subprocess
import os

def setup_mmocr():
    # Step 1: Change directory to TextSpotter/mmocr
    os.chdir("TextSpotter/mmocr")
    
    # Step 2: Run 'mim install -e .'
    subprocess.run(["mim", "install", "-e", "."], check=True)

    #mim install mmcv-full mmdet mmengine
    subprocess.run(["mim", "install", "mmengine","mmdet","mmcv==2.0.0rc4"], check=True)
    
    # Step 3: Change directory back to the original path (two levels up)
    os.chdir("../../")

# Call the function
setup_mmocr()
