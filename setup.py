from setuptools import setup
import os

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='ProgPlot',
    version='0.122',
    packages=['ProgPlot'],
    url='https://github.com/lewis-morris/ProgPlot',
    license='MIT',
    author='Lewis Morris',
    author_email='lewis.morris@gmail.com',
    description='',
    install_requires=required,
)
