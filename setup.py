from setuptools import find_packages, setup

setup(
    name="uav_control",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "cvxpy==1.5.1",
        "matplotlib==3.8.3",
        "numpy==1.26.4",
        "spatialmath-python==1.1.9",
        "scipy==1.13.1",
    ],
    author="Micah Reich",
    author_email="micahreich02@gmail.com",
    description="UAV simulation and control algorithms from popular research papers implemented in Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/micahreich/uav_control",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9.6",
)
