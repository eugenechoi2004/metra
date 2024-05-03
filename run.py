from modal import Image

image = (Image.debian_slim(python_version="3.10.9").pip_install_from_requirements("requirements.txt").gpu(gpu.A100(size="80GB")))