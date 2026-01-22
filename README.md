# Summary
Code for the paper "Designing Regression Models to Perform Well Out of Sample" by:
 - [Tom Winckelman](https://sites.google.com/view/thomas-winckelman/welcome)
 - [Simon Foucart](https://foucart.github.io/)


Recall that the paper suggests two main algorithms:
 - ADAM with a projection step
 - Solving the dual of a quadratic program

I think the easiest thing to digest on an initial inspection is a minimal demo of the former.
__PLEASE SEE [this colab demo](https://colab.research.google.com/drive/1C6Xgo9C-U-ZTcxDtap7I44Ao-Pd785Fq?usp=sharing)__


After that, I turn you loose to inspect the codebase, which includes many experiments that lead nowhere and are not sufficiently docummented.
A demo of the latter can be found in `quadratic_univar.py`.

# Installation

__Prerequisite:__ have git installed.

__To install as a package:__ the following command line prompt is likely to work:

```
pip install git+https://github.com/ThomasLastName/near-optimal.git
```

However, if you normally do something like `pip m install` instead of `pip install` then I guess do that here, too.
Throwing everything at the kitchen sink, if the above doesn't work, you could maybe try:

```
python.exe -m pip install git+https://github.com/ThomasLastName/near-optimal.git
```


__To upgrade as a package:__ Substitute `upgrade` for `install --upgrade` in the above command, such as
```
pip install --upgrade git+https://github.com/ThomasLastName/near-optimal.git
```

or

```
python.exe -m pip install --upgrade git+https://github.com/ThomasLastName/near-optimal.git
```

__To install as a repo (deprecated):__ The following is deprecated but still worked last I checked:

First, clone the repo as normal (`git clone https://github.com/ThomasLastName/near-optimal.git`).
Then, from the directory of `setup.py` (`cd near-optimal`), run the command `pip install -e .`.

__To upgrade as a repo (deprecated):__ `pip install --upgrade .`, I think.


# Usage

TODO


# Notes