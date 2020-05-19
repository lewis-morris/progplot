from setuptools import setup
import os

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='progplot',
    version="0.1.8",
    packages=['progplot'],
    url='https://github.com/lewis-morris/ProgPlot',
    license='MIT',
    author='Lewis Morris',
    author_email='lewis.morris@gmail.com',
    description='progplot - Timeseries barplot animations.',
    install_requires=required,
)
