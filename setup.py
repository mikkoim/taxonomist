from setuptools import setup, find_packages

setup(
    name='taxonomist',
    version='0.0.3',
    packages=find_packages(include=['taxonomist']),
    package_dir={'':'src'}
)