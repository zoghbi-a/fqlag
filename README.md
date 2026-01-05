## About
`fqlag` is a python library to characterize the variability in the frequency domain of light curves that are not continuously sampled. This supersedes the `plag` library, with a different implementation that is more stable during the calculations.

Both libraries implement the method presented in Zoghbi et. al. (2013) paper ([Astrophysical Journal; 2013. 777. 24](https://arxiv.org/abs/1308.5852)) to calculate periodogram and time/phase lags in the frequency domain from unevenly-sampled light curves.


## Installation
- pip install fqlag


## Getting Started
- Check out the [Getting Started](tutorials/getting-started.ipynb) notebook, which can be browsed [here](tutorials/getting-started.md).
- There are additional example in the [`tutorials/test.py`](tutorials/test.py) file.


## Cite
Please cite the following paper when using the code.

```
@ARTICLE{2013ApJ...777...24Z,
       author = {Zoghbi, A. and Reynolds, C. and Cackett, E.~M.},
        title = "{Calculating Time Lags from Unevenly Sampled Light Curves}",
      journal = {\apj},
     keywords = {black hole physics, galaxies: active, galaxies: nuclei, methods: data analysis, Astrophysics - High Energy Astrophysical Phenomena},
         year = 2013,
        month = nov,
       volume = {777},
       number = {1},
          eid = {24},
        pages = {24},
          doi = {10.1088/0004-637X/777/1/24},
archivePrefix = {arXiv},
       eprint = {1308.5852},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2013ApJ...777...24Z},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
