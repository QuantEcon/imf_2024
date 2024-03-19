# Modern Computational Economics and Policy Applications

![](qe-logo-large.png)

A workshop for the IMF's Institute for Capacity Development

## Abstract

Open source scientific computing environments built around the Python
programming language have expanded rapidly in recent years. They now form the
dominant paradigm in artificial intelligence and many fields within the natural
sciences.  Economists can greatly enhance their modeling and data processing
capabilities by exploiting Python's scientific ecosystem.  This course will
cover the foundations of Python programming and Python scientific libraries, as
well as showing how they can be used in economic applications for rapid
development and high performance computing.

## Times and Dates

* Dates: March 25-27, 2024
* Times: 9:30 -- 12:30 and 14:00 -- 17:00 
* Location: room HQ2-3B-748 (in-person participants) 

## Instructors

[Chase Coleman](https://github.com/cc7768) is a computational economist based at New York University where
he is a visiting assistant professor. He was an early contributor at QuantEcon
and, along with other members of QuantEcon, has given lectures and workshops
on Python, Julia, and other open source computational tools at institutions and
universities all around the world.

[John Stachurski](https://johnstachurski.net/) is a mathematical and
computational economist based at the Australian National University who works on
algorithms at the intersection of dynamic programming, Markov dynamics,
economics, and finance.  His work is published in journals such as the Journal
of Finance, the Journal of Economic Theory, Automatica, Econometrica, and
Operations Research.  In 2016 he co-founded QuantEcon with Thomas J. Sargent. 

In addition, 2011 Nobel Laureate [Thomas J. Sargent](http://www.tomsargent.com/)
will join remotely and run a one hour session on the 27th.


## Syllabus

* Monday morning: Introduction 
  - Scientific computing: directions and trends (`intro_slides/sci_comp_intro.pdf`)
  - Python and the AI revolution (`ai_revolution/ai_revolution.pdf`)
  - A taste of HPC with Python (`fun_with_jax.ipynb`)
  - A brief tour of Python's massive scientific computing ecosystem (`scientific_python/main.pdf`)
  - Working with Jupyter
* Monday afternoon: Python basics
  - Core Python  (`quick_python_intro.ipynb`)
  - NumPy / SciPy / Matplotlib / Numba (`quick_scientific_python_intro.ipynb`)
  - Day 1 homework: Lorenz curves and Gini coefficients (`lorenz_gini.ipynb`)
* Tuesday morning: Markov models in Python
  - Markov chains: Basic concepts (`finite_markov.ipynb`)
  - Intermezzo: A quick introduction to JAX (`jax_intro.ipynb`)
  - Wealth distribution dynamics (`wealth_dynamics.ipynb`)
  - TBA
  - Day 2 homework: Markov chain exercises (`markov_homework.ipynb`)
* Tuesday afternoon: Dynamic programming
  - A simple optimal savings problem (`opt_savings_1.ipynb`)
  - Alternative algorithms: VFI, HPI and OPI (`opt_savings_2.ipynb`)
  - The endogenous grid method (`egm.ipynb`)
* Wednesday morning: Heterogeneous agents
  - Heterogenous firms (`hopenhayn_jax.ipynb`)
  - The Aiyagari model (`aiyagari_jax.ipynb`)
  - TBA
* Wednesday afternoon: Further applications
  - Sovereign default
  - The Bianchi overborrowing model (`overborrowing.ipynb`, `bianchi.pdf`)


## Software

The main interface to Python will be either `jupyter-notebook` or `jupyter-lab`.

Access to the `ipython` REPL will also be useful.

Some work will be done remotely using Google Colab --- a Google account is
required.

Required Python libraries (much of which is found in the Anaconda Python distribution):

* `numpy`
* `scipy`
* `matplotlib`
* `pandas`
* `scikit-learn`
* `statsmodels`
* `numba`
* `f2py`
* `quantecon`

## Useful References

* https://www.anaconda.com/download
