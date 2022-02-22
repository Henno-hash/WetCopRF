from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

name='wetCopRF'
version='1.0'
release='1'

setup(
    name=name,
    version=version,
    release=release,
    install_requires=requirements,
    author='Florian Hellwig & Henrik Schmidt',
    author_email='Florian.marcus.hellwig@uni-jena.de & Henrik.Schmidt@uni-jena.de',
    url='https://github.com/Henno-hash/wetCopRF',
    packages=find_packages(),
    description='Downloads specific data, calculates a logarithmic image and displays it',

)