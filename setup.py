from setuptools import find_packages, setup

setup(
    name="uav_control",
    version="0.1",
    packages=find_packages(),
    install_requires=[],  # Install requirements with pip install -r requirements.txt
    author="Micah Reich",
    author_email="micahreich02@gmail.com",
    description="UAV simulation and control algorithms from popular research papers implemented in Python",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/micahreich/uav_control",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9.6",
)
