# Summary
Code for the submitted paper "A Different Perspective on Out-of-Sample Generalization" by:
 - [Tom Winckelman](https://sites.google.com/view/thomas-winckelman/welcome)
 - [Simon Foucart](https://foucart.github.io/)


Recall that the paper suggests two main algorithms:
 - ADAM with a projection step
 - Solving the dual of a quadratic program

For an initial viewing of the code, I think the easiest thing to digest is a minimal demo of the former:
__PLEASE SEE [this colab demo](https://colab.research.google.com/drive/1C6Xgo9C-U-ZTcxDtap7I44Ao-Pd785Fq?usp=sharing)__.
A demo of the latter can be found in `quadratic_univar.py`.

You are free to inspect the codebase, which includes many experiments that lead nowhere and only partially docummented. The git log contains even more obscure records of similar experiments that have been deleted -- not in an attempt to hide anything, merely in an attempt to reduce clutter.


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


__To upgrade as a package:__ Substitute `install` for `install --upgrade` in the above command, such as
```
pip install --upgrade git+https://github.com/ThomasLastName/near-optimal.git
```

or

```
python.exe -m pip install --upgrade git+https://github.com/ThomasLastName/near-optimal.git
```

__To install as a repo:__ The following was deprecated but still worked last I checked:

First, clone the repo as normal (`git clone https://github.com/ThomasLastName/near-optimal.git`).
Then, from the directory of `setup.py` (`cd near-optimal`), run the command `pip install -e .`.

__To upgrade as a repo:__ `pip install --upgrade .`, I think.


# Usage

The basic idea is:

```
from near_optimal.quadratic_univar import DualSpline
x_data, y_data = .... # flat arrays
S = DualSpline(x_data, y_data)
S.fit()  # prints the sub-optimality ratio in console
# plot and S and stuff, if you want
```

Just run `quadratic_univar.py`. That's *by far* the most important file. Everything else is basically just extra experimentation for the sake of being thorough.
