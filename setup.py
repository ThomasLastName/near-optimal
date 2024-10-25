
### ~~~
## ~~~ Originally from https://github.com/maet3608/minimal-setup-py/blob/master/setup.py
### ~~~ 

from setuptools import setup, find_packages

setup(
    name = 'near_optimal',
    version = '0.3.2',
    url = 'https://github.com/ThomasLastName/near-optimal',
    author = 'Thomas Winckelman',
    author_email = 'winckelman@tamu.edu',
    description = 'Code for "optimal recovery for neural networks"',
    packages = find_packages(),
    install_requires = [
            "cvxpy",
            "quality_of_life @ git+https://github.com/ThomasLastName/quality-of-life.git"
        ])
