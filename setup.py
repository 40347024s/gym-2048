from setuptools import setup, find_packages

setup(
    name='gym_2048',
    packages=find_packages(),
    version="0.0.1",
    install_requires=["gymnasium", "pygame", "numpy", "matplotlib", "numba"],
)
