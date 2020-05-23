from setuptools import setup
import os

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    package_data={'': ['./progplot/error.png']},
    include_package_data=True,
    name='progplot',
    version="0.2.4",
    packages=['progplot'],
    url='https://github.com/lewis-morris/progplot',
    license='MIT',
    author='Lewis Morris',
    author_email='lewis.morris@gmail.com',
    description='progplot - Timeseries barplot animations.',
    install_requires=required,

)
