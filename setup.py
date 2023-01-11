from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='closebyone',
    packages=['closebyone'],
    url='https://github.com/natesute/closebyone.git',
    author='Nathan Suttie',
    author_email='nathan.suttie@gmail.com',
    description='Implementation of close-by-one interval-pattern base learner algorithm for rule boosting.',
    python_requires='>=3.6',
    install_requires=['realkd']
)