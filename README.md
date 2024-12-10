# Fit neutrino fluxes and construct uncertainty bands with 10 years of IceCube public data.

Example notebooks for fitting neutrino fluxes and constructing uncertainty bands based on Feldman-Cousins estimation. To run this you need to install [SkyLLH](https://github.com/icecube/skyllh) and download the [10 years of public IceCube point-source data](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VKL316).  

`Example - Fit a power-law neutrino flux.ipynb` shows an example for how to fit a power-law neutrino flux at the position of TXS 0506+056. This was used in [M. Karl, P. Padovani, P. Giommi, MNRAS, 526, 1, 661-681, 2023](https://academic.oup.com/mnras/article/526/1/661/7269217). 

`Example - Fit a custom neutrino flux.ipynb` shows an example for how to fit a custom flux, here the peaked spectrum based on p$\gamma$ interaction. This was used in [X. Rodrigues, M. Karl, P. Padovani, et al., A&A, 689, A147 (2024)](https://www.aanda.org/articles/aa/full_html/2024/09/aa50592-24/aa50592-24.html). 

`FC_interval_construction_example.ipynb` shows an example for how to construct the [Feldman-Cousins confidence belts](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.57.3873) (both for power-law spectrum and the peaked spectrum). This was used in both works mentioned above. This requires that trials were run beforehand. 

For running trials for the Feldman-Cousins (FC) limits, you can use the skripts `run_FC_trials.py` (with a power-law) or `run_FC_trials_epeak.py` (with a peaked spectrum). When running those skripts, the correct paths need to be set in the `.env` file (using the `dotenv` python package). 

In the current setup, you have to create a file called `.env` with the following content (customize it to fit your dependencies):
```
SKYLLH_PATH="/path/to/your/skyllh"
PUBLIC_DATA_PATH="/path/to/data/ice-cube"
TRIAL_PATH="/where/to/save/your/FC_trials"
TABLE_PATH="/where/to/find/your/table/"
```

Specifics for which trials to run (which power laws, peaks, flux strengths, path where to save data...) can be parsed. 

For example, the following command covers a grid for gamma values between 1 and 3.7 (with 51 steps, the default value in `run_FC_trials.py`) and for each gamma, fluxes with a mean number of neutrino events between 1 and 50 (with 61 steps, the default value in `run_FC_trials.py`). Each gridpoint or flux (combination of gamma and ns value) is simulated `--n_trials` = 500 times with a seed `--seed` for the random number generator of 1. The trials are saved in the given path of `--trial_path`: 

`python run_FC_trials.py --source 0 --gamma_min 1 --gamma_max 3.7 --ns_min 1 --ns_max 50 --trial_path /save/my/trials/ --seed 1 --n_trials 500`.

In this specific implementation, all sources of interest are saved in a `mastertable.txt` with column entries `RA, DEC, NAME`, specifying the right ascension, declination and name of the object of interest. The path to the table is specified as `TABLE_PATH` in the `.env` file. With `--source` you select a row in the table. In the above example, we pick the first source. 
