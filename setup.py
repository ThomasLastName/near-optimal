
### ~~~
## ~~~ Originally from https://github.com/maet3608/minimal-setup-py/blob/master/setup.py
### ~~~ 

from setuptools import setup, find_packages

setup(
    name = 'near_optimal',
    version = '1.0.1',
    url = 'https://github.com/ThomasLastName/near-optimal',
    author = 'Thomas Winckelman',
    author_email = 'winckelman@tamu.edu',
    description = 'Code for "optimal recovery for neural networks"',
    packages = find_packages(),
    install_requires = [
            "numpy",
            "scipy>=1.6.1",
            "pandas",
            "cvxpy",
            "SCS",
            "matplotlib",
            "seaborn",
            "tqdm",
            "torch",
            "quality_of_life @ git+https://github.com/ThomasLastName/quality-of-life.git"   # >=2.16.3
        ])
