import os
from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.append(os.getenv("SKYLLH_PATH"))
import numpy as np
import matplotlib.pyplot as plt
import copy


from skyllh.datasets.i3.PublicData_10y_ps import create_dataset_collection
from skyllh.core.source_model import PointLikeSource
from skyllh.analyses.i3.publicdata_ps.time_integrated_ps import create_analysis
from skyllh.analyses.i3.publicdata_ps.time_integrated_ps_function_energy_spectrum import create_analysis as create_epeak_analysis
from skyllh.analyses.i3.publicdata_ps.time_integrated_ps_function_energy_spectrum import (
    set_epeak,
    flux_from_ns
)
from skyllh.analyses.i3.publicdata_ps.aeff import PDAeff
from skyllh.core.random import RandomStateService
from skyllh.core.timing import TimeLord

from skyllh.core.config import (
    Config,
)

cfg = Config()

from numpy.lib.recfunctions import stack_arrays, append_fields


class Source:

    def __init__(self, ra, dec, public_data_path=None, name=""):
        """ initialize neutrino Source
        
        Parameters:
        -----------
        ra: right ascension in degrees
        dec: declination in degrees
        public_data_path: path to directory with "icecube_10year_ps" folder 
        """

        if not public_data_path:
            public_data_path = os.getenv("PUBLIC_DATA_PATH")

        self.ra = ra
        self.dec = dec
        self.name = name

        # for the neutrino flux
        dsc = create_dataset_collection(cfg=cfg, base_path=public_data_path)
        self.datasets = dsc.get_datasets(['IC40', "IC59", "IC79", "IC86_I", "IC86_II-VII"])
        # self.datasets = dsc.get_datasets(['IC40'])

        self.flux_in_GeV = None
        self.ns = None
        self.gamma = None
        self.e_peak = None
        self.ts = None
        self.ana = None

        # for the uncertainties on the flux
        self.event_list = None
        self.ns_errors = (0, 0)
        self.gamma_errors = (0, 0)
        self.flux_errors = (0, 0)
        # for a first approximation one can use a scan of the llh contour for the uncertainties
        self.ns_errors_llh_scan = (0, 0)
        self.gamma_errors_llh_scan = (0, 0)
        self.flux_errors_llh_scan = (0, 0)

        # for the correct true energy range
        self.public_data_path = public_data_path
        self.smearing = None
        self.effA = None
        self.true_E = (0, 0)


    def create_skyllh_analysis(self, gamma_min=1, gamma_max=5, ns_max=1e4, source_gamma=2.0, minimizer="minuit"):
        """ initialize the skyllh analysis
        
        Parameters:
        -----------
        gamma_min : lower bound of gamma for minimizer
        gamma_max : upper bound of gamma for minimizer
        ns_max : upper bound of number of signal events for minimizer
        source_gamma : spectral index for simulating a neutrino source with power law emission
        ra : right ascension of point source. we need this for caching the correct analysis instance
        dec : declination of point source
        
        Returns:
        --------
        Skyllh Analysis instance
        
        """
        print("creating analysis")

        source = PointLikeSource(ra=np.deg2rad(self.ra), dec=np.deg2rad(self.dec))
        self.ana = create_analysis(cfg=cfg, datasets=self.datasets, 
                                   source=source, gamma_min=gamma_min, 
                                   gamma_max=gamma_max, ns_max=ns_max, 
                                    refplflux_gamma=source_gamma, ns_seed=10, 
                                    minimizer_impl=minimizer)
        return self.ana
    

    def create_skyllh_epeak_analysis(self, source_energy_spectrum, source_energies, source_epeak=3, epeak_min=1.06, epeak_max=10.05, epeak_seed=3, 
                                     ns_max=1e6, minimizer="minuit"):
        """ initialize the skyllh analysis following a given spectrum. The analysis will define a spline based on the points
        
        Parameters:
        -----------
        source_energy_spectrum : flux values (E^2 flux) for the source energies
        source_energies : energies in GeV for the flux
        epeak_min=1.06 : lower bound for energy peak for minimzer
        epeak_max=10.05 : upper bound for energy peak for minimzer
        ns_max : upper bound of number of signal events for minimizer
        source_gamma : spectral index for simulating a neutrino source with power law emission
        ra : right ascension of point source. we need this for caching the correct analysis instance
        dec : declination of point source 
        
        Returns:
        --------
        Skyllh Analysis instance
        """
        print("creating epeak analysis")

        # xavier_flux = np.load("/home/martina/postdoc/neutrino_fluxes/nu_template_nufnu_gev_percmsq_persecond.npy")
        # xavier_energy = np.load("/home/martina/postdoc/neutrino_fluxes/nu_template_energy_gev.npy")

        # e_peak = np.log10(xavier_energy[np.argmax(xavier_flux)])

        source = PointLikeSource(ra=np.deg2rad(self.ra), dec=np.deg2rad(self.dec))

        self.ana = create_epeak_analysis(
            cfg=cfg,
            datasets=self.datasets,
            source=source,
            source_energies=source_energies,
            source_energy_spectrum=source_energy_spectrum,
            e_peak_signal=source_epeak,
            e_peak_seed=epeak_seed,
            e_peak_min=epeak_min,
            e_peak_max=epeak_max,
            ns_max=ns_max,
            ns_seed=10,
            minimizer_impl=minimizer) 

        return self.ana
    

    def calculate_neutrino_flux(self, gamma_min=1, gamma_max=5, minimizer="crs"):
        """ calculates neutrino flux from public data

        Parameters:
        -----------
        gamma_min: lower bound for optimizing gamma
        gamma_max: upper bound for optimizing gamma
        minimizer: which minimizer for optimizing the flux
        brute_force: bool, define if we should brute force scan the paramter space
        ns_max_brute_force: float, define the upper bound for ns for the llh evaluation of the brute force scan
        step_ns: how fine the ns grid should be for the brute force scan
        step_gamma: hoe fine the gamma grid should be for the brute force scan

        Returns:
        --------
            - scaling_factor: flux = scaling_factor x (E / 1000 GeV)^gamma (1/Gev s cm^2 sr)
            - gamma: spectral index of flux
        """
        print("calculating best fit flux")
        if self.ana is None:
            self.create_skyllh_analysis(gamma_min=gamma_min, gamma_max=gamma_max, minimizer=minimizer)
        rss = RandomStateService(seed=1)
        (ts, x, status) = self.ana.unblind(rss)

        scaling_factor = self.ana.calculate_fluxmodel_scaling_factor(x['ns'], [x['ns'], x['gamma']])
        self.flux_in_GeV = scaling_factor # flux = scaling_factor x (E / 1000 GeV)^gamma (1/Gev s cm^2 sr)
        self.ns = x["ns"]
        self.gamma = x["gamma"]
        self.ts = ts
        return scaling_factor, x["gamma"]



    def calculate_epeak_neutrino_flux(self, epeak_min=1, epeak_max=10, source_energy_spectrum=None, 
                                      source_energies=None):
        """ calculates neutrino flux from public data based on a peaked energy spectrum
        
        Parameters:
        -----------
        epeak_min: lower bound for optimizing epeak
        epeak_max: upper bound for optimizing epeak
        source_energy_spectrum: spectrum for e_peak pdf
        source_energies: energies for e_peak spectrum 
        
        Returns:
        --------
            - scaling_factor: flux = scaling_factor x (E / 1000 GeV)^gamma (1/Gev s cm^2 sr)
            - epeak: Energy peak of flux
        """
        print("calculating best fit flux")
        if self.ana is None:
            self.create_skyllh_epeak_analysis(source_energy_spectrum, source_energies, epeak_min=epeak_min, epeak_max=epeak_max)

        rss = RandomStateService(seed=1)
        (ts, x, status) = self.ana.unblind(rss)

        flux = flux_from_ns(self.ana, x["e_peak"], x["ns"])
        
        self.flux_in_GeV = flux 
        self.ns = x["ns"]
        self.e_peak = x["e_peak"]
        self.ts = ts
        return flux, x["e_peak"]


    def get_event_list(self):
        self.event_list = [data.exp for data in self.ana._data_list]
        return self.event_list


    def run_trials(self, seed, ns, gamma, n_trials):
        """ run trials for a fixed spectral index gamma and ns
        
        Parameters:
        -----------
        seed: seed for the random state generator
        ns: mean number of simulated signal events
        gamma: source spectral index for power law emission model
        n_trials: number of trials that should be run
        
        Returns:
        --------
            - array with dtypes 
                'seed', '<i8': seed for the random state generator with which this trial was generated.
                'mean_n_sig', '<f8': mean number of signal events 
                'n_sig', '<i8': actual number of signal events, drawn from a poisson distribution
                'mean_n_sig_0', '<f8': number of events for the background hypothesis (default=0)
                'ts', '<f8': best fit test statistic value
                'ns', '<f8': best fit number of signal events
                'gamma', '<f8': best fit gamma
                'ts_true_param', '<f8': evaluation of the test statistic for the true signal parameters ns and gamma. 
                'source_gamma', '<f8': simulated source spectral index
        """

        print("running " + str(n_trials) + " trials for ns = " + str(ns) + ", gamma = " + str(gamma))
        my_trials = np.empty((n_trials,), dtype=[('seed', '<i8'), ('mean_n_sig', '<f8'), ('n_sig', '<i8'), ('mean_n_sig_0', '<f8'), 
                                            ('ts', '<f8'), ('ns', '<f8'), ('gamma', '<f8'), ('ts_true_param', '<f8'), ('source_gamma', '<f8')])
       
        # set the source spectral index
        self.ana.shg_mgr.shg_list[0].fluxmodel.energy_profile.gamma = gamma 
        self.ana.sig_generator.change_shg_mgr(self.ana.shg_mgr)

        # define the seed for the trials
        rss = RandomStateService(seed=seed)
        tl = TimeLord()

        for trial_index in range(n_trials):
            retries = 0
            while retries < 1000:
                try:
                    trials = self.ana.do_trials(
                                    rss,
                                    n=1,
                                    mean_n_sig=ns,
                                    tl=tl)
                    break
                except ValueError:
                    retries += 1
            if retries == 1000:
                print("No convergence found for ns {} and gamma {}.".format(ns, gamma))
                continue                

            true_ts = max(0, 2 * self.ana.llhratio.evaluate([ns, gamma])[0])
            trials = np.lib.recfunctions.merge_arrays((trials, 
                                                       np.array([(true_ts, gamma)], dtype=[("ts_true_param", '<f8'), ('source_gamma', '<f8')])), 
                                                      flatten=True)
            
            my_trials[trial_index] = trials

        return my_trials
    

    def run_epeak_trials(self, seed, ns, epeak, n_trials):
        """ run trials for a fixed peak energy epeak and ns
        
        Parameters:
        -----------
        seed: seed for the random state generator
        ns: mean number of simulated signal events
        epeak: source energy peak for peaked emission model
        n_trials: number of trials that should be run
        
        Return:
        -------
            - array with dtypes 
                'seed', '<i8': seed for the random state generator with which this trial was generated.
                'mean_n_sig', '<f8': mean number of signal events 
                'n_sig', '<i8': actual number of signal events, drawn from a poisson distribution
                'mean_n_sig_0', '<f8': number of events for the background hypothesis (default=0)
                'ts', '<f8': best fit test statistic value
                'ns', '<f8': best fit number of signal events
                'e_peak', '<f8': best fit peak energy
                'ts_true_param', '<f8': evaluation of the test statistic for the true signal parameters ns and gamma. 
                'source_e_peak', '<f8': simulated source peak energy
        """

        print("running " + str(n_trials) + " trials for ns = " + str(ns) + ", epeak = " + str(epeak))
        my_trials = np.empty((n_trials,), dtype=[('seed', '<i8'), ('mean_n_sig', '<f8'), ('n_sig', '<i8'), ('mean_n_sig_0', '<f8'), 
                                            ('ts', '<f8'), ('ns', '<f8'), ('e_peak', '<f8'), ('ts_true_param', '<f8'), ('source_e_peak', '<f8')])
       
        # set the source spectral index
        set_epeak(self.ana, epeak)

        # define the seed for the trials
        rss = RandomStateService(seed=seed)
        tl = TimeLord()

        for trial_index in range(n_trials):
            trials = self.ana.do_trials(
                            rss,
                            n=1,
                            mean_n_sig=ns,
                            tl=tl)
                            
            true_ts = max(0, 2 * self.ana.llhratio.evaluate([ns, epeak])[0])
            trials = np.lib.recfunctions.merge_arrays((trials, 
                                                       np.array([(true_ts, epeak)], dtype=[("ts_true_param", '<f8'), ('source_e_peak', '<f8')])), 
                                                      flatten=True)
            
            my_trials[trial_index] = trials

        return my_trials


    def generate_trials_for_gamma_ns_grid(self, gamma_min=1, gamma_max=5, n_gamma=30, ns_min=5, ns_max=30, n_ns=50, n_trials=100, seed=None, 
                                            path=None):
        """ generate trials for a grid of gamma and ns. This is needed for the Feldman-Cousins confidence interval estimation.
        
        Parameters:
        -----------
        gamma_min: the minimal spectral index
        gamma_max: the maximal spectral index
        n_gamma: how many steps there should be between gamma_min and gamma_max (gridsize)
        ns_min: the minimal number of signal events
        ns_max: the maximal number of signal events
        n_ns: how many steps there should be between ns_min and ns_max (gridsize)
        n_trials: how many trials will be generated for each ns and gamma 
        seed: seed for the random state generator. Will be random if not chosen.
        path: path where trials should be saved. If None, trials will not be saved. 

        Returns:
        --------
            - array with dtypes 
                'seed', '<i8': seed for the random state generator with which this trial was generated.
                'mean_n_sig', '<f8': mean number of signal events 
                'n_sig', '<i8': actual number of signal events, drawn from a poisson distribution
                'mean_n_sig_0', '<f8': number of events for the background hypothesis (default=0)
                'ts', '<f8': best fit test statistic value
                'ns', '<f8': best fit number of signal events
                'gamma', '<f8': best fit gamma
                'ts_true_param', '<f8': evaluation of the test statistic for the true signal parameters ns and gamma. 
                'source_gamma', '<f8': simulated source spectral index
        """

        if self.ana==None:
            self.create_skyllh_analysis(gamma_min=gamma_min, gamma_max=gamma_max, ns_max=1e3)

        my_trials = None

        for g in np.linspace(gamma_min, gamma_max, n_gamma):
            # increase seed by one
            seed += 1

            for ns in np.linspace(ns_min, ns_max, n_ns):

                tmp_trials = self.run_trials(seed, ns, g, n_trials)
                if my_trials is None:
                    my_trials = tmp_trials
                else:
                    my_trials = np.append(my_trials, tmp_trials)

            # save after some iterations in case something crashes
                np.save(path, my_trials)
        
        if path is not None:
            np.save(path, my_trials)

        return my_trials
    

    def generate_trials_for_epeak_ns_grid(self, source_energy_spectrum=None, source_energies=None, source_epeak=3,
                                          epeak_min=2, epeak_max=9, n_epeak=8, ns_min=0, 
                                        ns_max=30, n_ns=50, n_trials=100, seed=None, 
                                        path=None, epeak_seed=3):

        """ generate trials for a grid of epeak and ns. This is needed for the Feldman-Cousins confidence interval estimation.
        
        Parameters:
        -----------
        epeak_min: the minimal peak energy
        epeak_max: the maximal peak energy
        n_epeak: how many steps there should be between epeak_min and epeak_max (gridsize)
        ns_min: the minimal number of signal events
        ns_max: the maximal number of signal events
        n_ns: how many steps there should be between ns_min and ns_max (gridsize)
        n_trials: how many trials will be generated for each ns and gamma 
        seed: seed for the random state generator. Will be random if not chosen.
        path: path where trials should be saved. If None, trials will not be saved. 

        Return:
        -------
            - array with dtypes 
                'seed', '<i8': seed for the random state generator with which this trial was generated.
                'mean_n_sig', '<f8': mean number of signal events 
                'n_sig', '<i8': actual number of signal events, drawn from a poisson distribution
                'mean_n_sig_0', '<f8': number of events for the background hypothesis (default=0)
                'ts', '<f8': best fit test statistic value
                'ns', '<f8': best fit number of signal events
                'gamma', '<f8': best fit gamma
                'ts_true_param', '<f8': evaluation of the test statistic for the true signal parameters ns and gamma. 
                'source_gamma', '<f8': simulated source spectral index
        """

        if self.ana==None:
            self.create_skyllh_epeak_analysis(source_energy_spectrum, source_energies, source_epeak=source_epeak, 
                                              epeak_min=1, epeak_max=10, ns_max=1e3, epeak_seed=epeak_seed)

        my_trials = None

        for e in np.linspace(epeak_min, epeak_max, n_epeak):
            seed += 1

            for ns in np.linspace(ns_min, ns_max, n_ns):

                tmp_trials = self.run_epeak_trials(seed, ns, e, n_trials)
                if my_trials is None:
                    my_trials = tmp_trials
                else:
                    my_trials = np.append(my_trials, tmp_trials)

            # save after some iterations in case something crashes
                np.save(path, my_trials)
        
        if path is not None:
            np.save(path, my_trials)

        return my_trials


    def confidence_level_gamma(self, gamma, trials, cl=0.68):
        """ get the confidence interval for fixed gamma values
        
        Parameters:
        -----------
        gamma: spectral index for which to calculate the confidence interval. 
                Has to be part of the simulated source spectra in trials
        trials: array with the generated trials
        cl: level of confidence (default: 68%)

        Returns: 
        --------
            - Dictionary with ns as key with lower and upper value of TS values in the confidence band
        """
        # chose trials with correct gamma
        my_trials = trials[trials["source_gamma"] == gamma]
        ts_limits = {}

        for ns in np.unique(my_trials["mean_n_sig"]):

            tmp_trials = my_trials[my_trials["mean_n_sig"] == ns]

            # ftol precision is 1e-8, so we set values with this difference to ts_true

            tmp_trials["ts"] = np.where((tmp_trials["ts_true_param"] > tmp_trials["ts"]) & ((tmp_trials["ts_true_param"] - tmp_trials["ts"]) <= 1e-8), 
                                  tmp_trials["ts_true_param"], 
                                  tmp_trials["ts"])

            mask_bad_minimizer = tmp_trials["ts"] >= tmp_trials["ts_true_param"]
            if not np.all(mask_bad_minimizer):
                print("WARNING: Bad minimizer results encountered.")
            llh_ratio = tmp_trials["ts_true_param"][mask_bad_minimizer] - tmp_trials["ts"][mask_bad_minimizer] 
            # llh_ratio = np.exp(tmp_trials["ts_true_param"] - tmp_trials["ts"])
            # llh_ratio = tmp_trials["ts_true_param"] - tmp_trials["ts"]

            sort_indices = np.argsort(llh_ratio) 
            # take the most relevant 68% (or cl*100%) TS values. Here, we assume each TS value has a probability of 1/len(n_trials).
            # ts_array = tmp_trials[sort_indices][:int(np.ceil(cl*len(tmp_trials)))]
            ts_array = tmp_trials[mask_bad_minimizer][sort_indices][-int(np.ceil(cl*len(tmp_trials[mask_bad_minimizer]))):]
           
            ts_range = np.min(ts_array["ts"]), np.max(ts_array["ts"])

            ts_limits[ns] = ts_range
        
        return ts_limits
    
    def confidence_level_epeak(self, epeak, trials, cl=0.68):
        """ get the confidence interval for fixed gamma values
        
        Parameters:
        -----------
        epeak: peak of energy spectrum for which to calculate the confidence interval. 
                Has to be part of the simulated source spectra in trials
        trials: array with the generated trials
        cl: level of confidence (default: 68%)

        Returns:
        --------
            - dictionary with ns as key with lower and upper value of TS values in the confidence band
        """
        # chose trials with correct e_peak
        my_trials = trials[trials["source_e_peak"] == epeak]
        ts_limits = {}

        for ns in np.unique(my_trials["mean_n_sig"]):

            tmp_trials = my_trials[my_trials["mean_n_sig"] == ns]


            tmp_trials["ts"] = np.where((tmp_trials["ts_true_param"] > tmp_trials["ts"]) & ((tmp_trials["ts_true_param"] - tmp_trials["ts"]) <= 1e-8), 
                                  tmp_trials["ts_true_param"], 
                                  tmp_trials["ts"])

            # the optimized ts should always be greater or equal to the true parameter ts. If this is not the case
            # the minimizer failed to find the correct minimum. Remove these cases because they screw up your limit since
            # the weights become greater than one for the FC limit construction.
            mask_bad_minimizer = tmp_trials["ts"] >= tmp_trials["ts_true_param"]
            if not np.all(mask_bad_minimizer):
                print("WARNING: Bad minimizer results encountered.")
            llh_ratio = tmp_trials["ts_true_param"][mask_bad_minimizer] - tmp_trials["ts"][mask_bad_minimizer] 
            sort_indices = np.argsort(llh_ratio)  
            # take the most relevant 68% (or cl*100%) TS values. Here, we assume each TS value has a probability of 1/len(n_trials).
            ts_array = tmp_trials[mask_bad_minimizer][sort_indices][-int(np.ceil(cl*len(tmp_trials[mask_bad_minimizer]))):]
            ts_range = np.min(ts_array["ts"]), np.max(ts_array["ts"])

            ts_limits[ns] = ts_range
        
        return ts_limits

    
    def get_FC_uncertainties_per_gamma(self, trials, cl=0.68, plot=True, path=None):
        """ determine the uncertainties on the best fit with Feldman-Cousins
        
        Parameters:
        -----------
        trials: array with the generated trials for different ns and gamma
        cl: level of confidence (default: 68%)
        plot: create a plot for each gamma to show which range of TS is covered per ns

        Returns:  
        --------
        dictionary with source_gamma as key and the corresponding limits for ns (lower limit, upper limit)
        """

        if plot and (not path):
            path = "./FC_limits/" + self.name
        if not os.path.exists(path):
            os.mkdir(path)

        g_vals = np.unique(trials["source_gamma"])
        ns_limits = {}

        for g in g_vals:
            ts_limits = self.confidence_level_gamma(g, trials, cl=cl)
            tmp_ns = []
            for ns in ts_limits:
                if ts_limits[ns][0] <= self.ts and self.ts <= ts_limits[ns][1]:
                    tmp_ns.append(ns)
            if tmp_ns == []:
                continue

            ns_limits[g] = min(tmp_ns), max(tmp_ns)

            if plot:
                for my_ns in ts_limits:
                    plt.plot(np.linspace(ts_limits[my_ns][0], ts_limits[my_ns][1], 100), [my_ns] * 100)
                plt.gca().axvline(self.ts)
                plt.title("True gamma = " + str(g))
                plt.xlabel("TS range (" + str(cl * 100) + "\%)")
                plt.ylabel("True mean ns")
                plt.savefig(path + "/gamma" + str(g) + ".png", bbox_inches="tight", dpi=400)
                plt.clf()
        return ns_limits


    def get_FC_uncertainties_per_epeak(self, trials, cl=0.68, plot=True, path=None):
        """ determine the uncertainties on the best fit with Feldman-Cousins
        
        Parameters:
        -----------
        trials: array with the generated trials for different ns and gamma
        cl: level of confidence (default: 68%)
        plot: create a plot for each gamma to show which range of TS is covered per ns

        Returns:  
        --------
        dictionary with source_gamma as key and the corresponding limits for ns (lower limit, upper limit)
        """

        if plot and (not path):
            path = "./FC_limits/" + self.name
        if not os.path.exists(path):
            os.mkdir(path)

        e_vals = np.unique(trials["source_e_peak"])
        ns_limits = {}

        for e in e_vals:
            ts_limits = self.confidence_level_epeak(e, trials, cl=cl)
            tmp_ns = []
            for ns in ts_limits:
                if ts_limits[ns][0] <= self.ts and self.ts <= ts_limits[ns][1]:
                    tmp_ns.append(ns)
            
            if tmp_ns == []:
                continue

            ns_limits[e] = min(tmp_ns), max(tmp_ns)

            if plot:
                for my_ns in ts_limits:
                    plt.plot(np.linspace(ts_limits[my_ns][0], ts_limits[my_ns][1], 100), [my_ns] * 100)
                plt.gca().axvline(self.ts)
                plt.title("True e_epak = " + str(e))
                plt.xlabel("TS range (" + str(cl * 100) + "\%)")
                plt.ylabel("True mean ns")
                plt.savefig(path + "/e_peak" + str(e) + ".png", bbox_inches="tight", dpi=400)
                plt.clf()

        return ns_limits


    def plot_FC_uncertainties(self, trials, cl=0.68, gammaplot=True, ns_limits=None, epeak_spektrum=False, path=None):
        """ determine the uncertainties on the best fit with Feldman-Cousins
        
        Parameters:
        -----------
        trials: array with the generated trials for different ns and gamma
        cl: level of confidence (default: 68%)
        gammaplot: create a plot for each gamma or epeak to show which range of TS is covered per ns
        ns_limits:
        epeak_spektrum: set this to True if we use a spektrum with e_peak
        """
        if not path:
            path = "./FC_limits/" + self.name
        if not os.path.exists(path):
            os.mkdir(path)

        if not epeak_spektrum:
            if ns_limits == None:
                ns_limits = self.get_FC_uncertainties_per_gamma(trials, cl=cl, plot=gammaplot)
            for g in ns_limits:
                plt.plot([g] * 2, ns_limits[g], "o")
                plt.xlabel("True gamma")
                plt.ylabel("True ns")
            plt.savefig(path + "/FC_limits_" + self.name + ".png", bbox_inches="tight", dpi=400)

        else:
            if ns_limits == None:
                ns_limits = self.get_FC_uncertainties_per_epeak(trials, cl=cl, plot=gammaplot)
            for e in ns_limits:
                plt.plot([e] * 2, ns_limits[e], "o")
                plt.xlabel("True E_peak")
                plt.ylabel("True ns")
            plt.savefig(path + "/FC_limits_epeak_" + self.name + ".png", bbox_inches="tight", dpi=400)
        plt.clf()


    def get_flux_bowtie(self, ns_limits, energy_flux=None, energy_unit="GeV", nevents=1e4, g_min=1, g_max=3.7, epeak_model=False, bins=None):
        """ calculate the bowtie of the flux with uncertainties for ns and gamma. This is i.e. for a plot with energy as the x-values and the flux as the y-values. 
        
        Parameters:
        -----------
        ns_limits: dictionary with the ns limits for different gammas
        energy_flux: energy in the same unit as fitted flux (GeV) (array (n, 1)) if flux should be calculated as well
        energy_unit: unit of energy_flux. Default is "GeV", but can also be "TeV"
        nevents: number of events to determine the correct energy range
        g_min: minimum gamma or epeak to constrain (this should be in ns_limits, otherwise it cannot be constrained)
        g_max: maximum gamma or epeak to constrain (this should be in ns_limits, otherwise it cannot be constrained)
        epeak_model: set this to True when using the epeak spectrum instead of the power law
        bins: bins for gamma or epeak for determining the flux limits within this bin. This is useful when fluxes have been resimulated with different strengths (for example because the upper flux was not strong enough to constrain the upper limit) and close but not exactly the same gamma or epeak values as before. 

        Returns:  
        --------  
        - flux: flux values for energies (array (n, 1))
        - lower edge of bowtie (array (n, 1))
        - upper edge of bowtie (array (n, 1)) 
        - energy range for contour (array (n, 1))
        """

        if g_min < min(ns_limits.keys()):
            print("WARNING: g_min is smaller than the smallest gamma or epeak in the Feldman-Cousins limits. This will not be properly constrained")
        if g_max > max(ns_limits.keys()):
            print("WARNING: g_max is larger than the largest gamma or epeak in the Feldman-Cousins limits. This will not be properly constrained")


        if not epeak_model: 
            flux_norm = self.ana.shg_mgr.shg_list[0].fluxmodel.energy_profile.E0
        else:
            flux_norm = 1 # stay in GeV

        if energy_unit == "GeV" and energy_flux is not None:
            energy_flux = energy_flux / flux_norm
        elif energy_unit != "TeV" and energy_flux is not None:
            raise ValueError("Please use either GeV or TeV energies.")

        if bins is not None:
            gamma_keys = bins
            trial_gamma_keys = np.fromiter(ns_limits.keys(), dtype=float)
        else:
            gamma_keys = np.fromiter(ns_limits.keys(), dtype=float)

        binned_dictionary = {}

        for index, g in enumerate(gamma_keys): 
                if g < g_min or g > g_max:
                    continue

                # check binning:
                if bins is not None:
                    if index < (len(gamma_keys)-1):
                        my_ns = (10000, 0)
                        my_keys = (trial_gamma_keys >= g) & (trial_gamma_keys <= gamma_keys[index+1])
                        for tmp_g in trial_gamma_keys[my_keys]:
                            my_ns = (min(my_ns[0], min(ns_limits[tmp_g])), max(my_ns[1], max(ns_limits[tmp_g])))
                            binned_dictionary[g] = my_ns
                    else:
                        continue

                else:
                    my_ns = ns_limits[g]

                    binned_dictionary[g] = my_ns

        # get linear interpolation between ns
        print(gamma_keys)
        gamma_keys = list(binned_dictionary.keys())
        print(gamma_keys)
        continuous_gamma = np.linspace(min(gamma_keys), max(gamma_keys), 100)
            
        high_interp = np.interp(continuous_gamma, gamma_keys, [binned_dictionary[x_][1] for x_ in binned_dictionary.keys()])
        low_interp = np.interp(continuous_gamma, gamma_keys, [binned_dictionary[x_][0] for x_ in binned_dictionary.keys()])

        energy_contour = np.logspace(2, 10, 1000) / flux_norm
        if epeak_model:
            # get array of epeak energies and the flux values at epeak
    
            energy_contour = 10**continuous_gamma

                
        lower_contour = np.ones_like(energy_contour)
        upper_contour = np.zeros_like(energy_contour) - 1

        if energy_flux is not None:

            energy_mask = (energy_contour < max(energy_flux)) & (energy_contour > min(energy_flux))
            energy = energy_contour[energy_mask]


            if not epeak_model:
                flux = self.flux_in_GeV * ((energy)**-self.gamma)

                lower_contour = np.ones_like(energy_contour)
                lower_contour[energy_mask] = flux

                upper_contour = np.zeros_like(energy_contour) - 1
                upper_contour[energy_mask] = flux
            
            else:
                rss = RandomStateService(seed=1)
                ts, x, info = self.ana.unblind(rss)

                set_epeak(self.ana, x["e_peak"])

                scaling_factor = self.ana.calculate_fluxmodel_scaling_factor(x["ns"], [x["ns"], x["e_peak"]])

                flux = self.ana.shg_mgr.get_fluxmodel_by_src_idx(0).energy_profile(E=energy).squeeze() * scaling_factor

        else:
            lower_contour = np.ones_like(energy_contour)
            upper_contour = np.zeros_like(energy_contour) - 1

            flux = np.zeros_like(energy_contour)
        # cover the complete true energy range here. Set the entries to  for cases
        # get the correct limits for each valid gamma, ns combination:

       
        for my_i, g in enumerate(continuous_gamma):

                if not epeak_model:
                    energy_min, energy_max = self.get_correct_energy_range_for_gamma(g, nevents=nevents, epeak_model=epeak_model)
                    energy_mask = (energy_contour > 10**energy_min / flux_norm) & (energy_contour < 10**energy_max / flux_norm)

                    energy = energy_contour[energy_mask]
                    low_ns = low_interp[my_i]
                    high_ns = high_interp[my_i]
                    low_flux = self.ana.calculate_fluxmodel_scaling_factor([low_ns], [low_ns, g]) * (energy**-g)
                    high_flux = self.ana.calculate_fluxmodel_scaling_factor([high_ns], [high_ns, g]) * (energy**-g)
                else:
                    set_epeak(self.ana, g)

                    energy_mask = energy_contour == 10**g

                    energy = energy_contour[energy_mask]
                    low_ns = low_interp[my_i]
                    high_ns = high_interp[my_i]
                    scaling_factor = self.ana.calculate_fluxmodel_scaling_factor(low_ns, [low_ns, g])
                    low_flux = self.ana.shg_mgr.get_fluxmodel_by_src_idx(0).energy_profile(energy).squeeze() * scaling_factor

                    scaling_factor = self.ana.calculate_fluxmodel_scaling_factor(high_ns, [high_ns, g])
                    high_flux = self.ana.shg_mgr.get_fluxmodel_by_src_idx(0).energy_profile(energy).squeeze() * scaling_factor

                lower_flux_arr = np.ones_like(energy_contour)
                lower_flux_arr[energy_mask] = low_flux

                high_flux_arr = np.zeros_like(energy_contour) - 1
                high_flux_arr[energy_mask] = high_flux
                
                lower_flux_arr_2 = copy.deepcopy(lower_flux_arr)
                lower_flux_arr_2[~energy_mask] = -1

                high_flux_arr_2 = copy.deepcopy(high_flux_arr)
                high_flux_arr_2[~energy_mask] = 1
                
                lower_contour = np.min(np.vstack((lower_contour, lower_flux_arr, high_flux_arr_2)), axis=0)
                upper_contour = np.max(np.vstack((upper_contour, high_flux_arr, lower_flux_arr_2)), axis=0)
        
        return flux, lower_contour, upper_contour, energy_contour


    def get_correct_energy_range_for_gamma(self, gamma, nevents=100000, epeak_model=False):
        """ get the correct energy range for simulated signal events
        
        Parameters:
        -----------
        ana: analysis instance
        gamma: find true energies for this gamma
        nevents: number of mock signal events to be generated. default: 1e5
        
        Returns:  
        -------- 
        minimum and maximum of central 90% quantile of true energy range. 
        """
        
        rss = RandomStateService(seed=1)

        if not epeak_model:
            self.ana.shg_mgr.shg_list[0].fluxmodel.energy_profile.gamma = gamma
            self.ana.sig_generator.change_shg_mgr(self.ana.shg_mgr)
        else:
            set_epeak(self.ana, gamma)

        events =  self.ana.generate_signal_events(rss, nevents)
        index = 0
        while index < len(events[-1]):
            if events[-1][index] == None:
                index += 1
            else: 
                signal_energy = events[-1][index]["log_true_energy"]
                break

        if (index + 1) < len(events[-1]):
            for dataset in events[-1][index+1:]:
                signal_energy = np.append(signal_energy, dataset["log_true_energy"])
        
        min_bound = np.quantile(signal_energy, 0.05)
        max_bound = np.quantile(signal_energy, 0.95)

        return min_bound, max_bound


    def parameter_llh_scan_ns(self, gamma_min=1, gamma_max=5, ns_min=5, ns_max=30, n_ns=50, n_gamma=30):
        """ run the likelihood scan for different ns and gamma.
        
        Parameters:
        -----------
        gamma_min: the minimal spectral index
        gamma_max: the maximal spectral index
        ns_min: the minimal number of signal events
        ns_max: the maximal number of signal events
        n_ns: how many steps there should be between ns_min and ns_max (gridsize)
        n_gamma: how many steps there should be between gamma_min and gamma_max (gridsize)
        
        Returns:  
        --------
        llh_ratios: array (n_ns, n_gamma) with the evaluated likelihood ratios
        plots_ns: the array of the ns grid
        plot_gamma: the array of the gamma grid
        """

        llh_ratios = np.empty((n_ns, n_gamma))
        plot_gamma = np.linspace(gamma_min, gamma_max, n_gamma)
        plot_ns = np.linspace(ns_min, max(self.ns, ns_max), n_ns)

        if self.ana==None:
            self.create_skyllh_analysis(gamma_min=gamma_min, gamma_max=gamma_max, ns_max=1e6)
        self.get_event_list()
            
        self.ana.initialize_trial(self.event_list)

        for index, my_ns in enumerate(plot_ns):
            ns_scan = []
            for my_gamma in plot_gamma:

                ns_scan.append(self.ana.llhratio.evaluate(np.array((my_ns, my_gamma)))[0])
            llh_ratios[index] = [ x * (1 if my_ns >=0 else -1) for x in ns_scan]
            
        return llh_ratios, plot_ns, plot_gamma
            
            
    def parameter_llh_scan_flux(self, n_fluxes=50, n_gamma=30, gamma_max=5, gamma_min=1, flux_min_factor=10, flux_max_factor=10):
        """ run the likelihood scan for different fluxes and gamma.        
        
        Parameters:
        -----------
        n_fluxes: how many steps there should be between the minimal flux and the maximal flux (gridsize)
        n_gamma: how many steps there should be between gamma_min and gamma_max (gridsize)
        gamma_max: the maximal spectral index
        gamma_min: the minimal spectral index
        flux_min_factor: the minimal flux is the fitted flux - flux_min_factor * fitted flux
        flux_max_factor: the maximal flux is the fitted flux - flux_max_factor * fitted flux
        
        Returns:  
        --------
        llh_ratios: array (n_fluxes, n_gamma) with the evaluated likelihood ratios
        plots_fluxs: the array of the flux grid
        plot_gamma: the array of the gamma grid
        """
        
        llh_ratios = np.empty((n_gamma, n_fluxes))

        if self.ana == None:
            self.create_skyllh_analysis(gamma_min=gamma_min, gamma_max=gamma_max, ns_max=1e6)
        self.get_event_list()
            
        self.ana.initialize_trial(self.event_list)
        
        flux_fit = self.ana.calculate_fluxmodel_scaling_factor(self.ns, [self.ns, self.gamma])
        
        plot_gamma, plot_fluxes = np.linspace(gamma_min, gamma_max, n_gamma), np.linspace(
            self.flux_in_GeV - flux_min_factor * self.flux_in_GeV, self.flux_in_GeV + flux_max_factor * self.flux_in_GeV, n_fluxes)
        
        for index, my_gamma in enumerate(plot_gamma):
            gamma_scan = []
            
        #   new_fluxes = ana.calculate_fluxmodel_scaling_factor(plot_fluxes, [my_gamma])
            new_norm = self.ana.calculate_fluxmodel_scaling_factor([1], [1, my_gamma])
            ns_from_flux = plot_fluxes / new_norm # if norm flux is not for ns_norm=1, then multiply here with ns_norm
    #       print(ns_from_flux)
            for my_ns in ns_from_flux:
                    gamma_scan.append(self.ana.llhratio.evaluate(np.array((my_ns, my_gamma)))[0])
                    
            llh_ratios[index] = [ x * (1 if my_ns >=0 else -1) for x in gamma_scan]
        
        return llh_ratios.transpose(), plot_fluxes.flatten(), plot_gamma
            
        
    def plot_contours_llh_scan(self, llh_ratios, plot_gamma, plot_y, yparam="ns", show_plot=True, save_plot=True):
        """ plot the contours (or generate the contours) for the likelihood ratio assuming Wilk's theorem.
        
        Parameters:
        -----------
        llh_ratios: the log likelihood ratios from the parameter scan
        plot_gamma: scanned gamma values
        plot_y: either scanned ns of flux values 
        yparam: "ns" or "flux" for correct y-axis label
        show_plot: True if plot should be shown, False otherwise
        save_plot: boolean, True if it should save the contour plot

        Returns:  
        --------
        the contour object from plt.contour
        """
        TS_diff = self.ts - (2 * llh_ratios)

        mesh = plt.pcolormesh(plot_gamma, plot_y, TS_diff, cmap="Blues_r")

        fontsize=13
        cbar = plt.colorbar(mesh)
        cbar.set_label(r"$\Delta$ TS", fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
        
        contours = plt.contour(plot_gamma, plot_y, TS_diff, levels=[2.27886856637673, 4.605170185988092], 
                            colors=["white"])
        
        fmt = {}
        strs = ['68\%', '90\%']
        for l, s in zip(contours.levels, strs):
            fmt[l] = s
        plt.clabel(contours, fmt=fmt, fontsize=fontsize)
        if yparam == "ns":
            plt.ylabel("number of signal events")
        elif yparam == "flux":
            plt.ylabel("flux") 
            
        plt.xlabel("spectral index")

        if save_plot:
            plt.savefig("/home/martina/postdoc/SINIII/contours/" + self.name + "_" + yparam + "scan.png", \
                dpi=400, bbox_inches="tight")

        grid = np.meshgrid(plot_gamma, plot_y)
        my_array = np.array([(a, b, c) for a , b, c in zip(grid[0].flatten(), grid[1].flatten(), llh_ratios.flatten())],
                                dtype=[("gamma", "f8"), (yparam, "f8"), ("llh_ratio", "f8")])
        np.save("/home/martina/postdoc/SINIII/contours/" + self.name + "_" + yparam + "llh_ratio.npy", my_array)

        if show_plot:
            plt.show()
        else:
            plt.clf()
        
        return contours


    def get_uncertainties_llh_scan(self, contours, y_fit):
        """ extract the 68% contours from the contour object
        
        Parameters:
        -----------
        contours: the contour object as result from the plot_contours
        y_fit: self.ns or self.flux

        Returns:  
        -------- 
        negative gamma error, positive gamma error, negative y error (ns or flux), positive y error (ns or flux)

        """
        path = contours.collections[0].get_paths()
        # mask the entries where ns or flux is 0. There we can not set a limit for gamma.
        
        mask = path[0].vertices[:,1] > 0

        xmin, xmax = 5, 1
        ymin, ymax = 1, 0
      
        # append all paths:
        for p in path:
            if p is not None:
                # mask the entries where ns or flux is 0. There we can not set a limit for gamma.
                mask = p.vertices[:,1] > 0
                if np.any(mask):
                    tmp_xmin, tmp_xmax = min(p.vertices[:,0][mask],), max(p.vertices[:,0][mask])
                    tmp_ymin, tmp_ymax = min(p.vertices[:,1][mask]), max(p.vertices[:,1][mask])

                    xmin, xmax, ymin, ymax = min(xmin, tmp_xmin), max(xmax, tmp_xmax), min(ymin, tmp_ymin), max(ymax, tmp_ymax)
                
            #    print(xmin - self.gamma, xmax - self.gamma, ymin - y_fit, ymax - y_fit)
        return xmin - self.gamma, xmax - self.gamma, ymin - y_fit, ymax - y_fit


    def get_ns_uncertainties_llh_scan(self, gamma_min=1, gamma_max=5, ns_min=0, ns_max=30, n_ns=50, n_gamma=30, show_plot=False, 
       save_plot=True):
        """ calculate the uncertainties for ns with a scan of the llh function. This is not reliable when the fitted source is weak. 
        Use the Feldman-Cousins confidence level estimation instead. 
        
        
        Parameters:
        -----------
        gamma_min: the minimal spectral index
        gamma_max: the maximal spectral index
        ns_min: the minimal number of signal events
        ns_max: the maximal number of signal events
        n_ns: how many steps there should be between ns_min and ns_max (gridsize)
        n_gamma: how many steps there should be between gamma_min and gamma_max (gridsize)
        show_plot: boolean, True if it should show the contour plot
        save_plot: boolean, True if it should save the contour plot

        Returns:  
        --------
            - gamma_errors: (negative gamma error, positive gamma error)
            - ns_errors: (negative ns error, positive ns error)
        """
        llh_ratios, plot_ns, plot_gamma = self.parameter_llh_scan_ns(gamma_min=gamma_min, gamma_max=gamma_max, ns_min=ns_min,\
             ns_max=ns_max, n_ns=n_ns, n_gamma=n_gamma)
        contours = self.plot_contours_llh_scan(llh_ratios, plot_gamma, plot_ns, yparam="ns", show_plot=show_plot, save_plot=save_plot)
        uncertainties = self.get_uncertainties_llh_scan(contours, self.ns)
        self.ns_errors_llh_scan = (uncertainties[2], uncertainties[3])
        self.gamma_errors_llh_scan = (uncertainties[0], uncertainties[1])
        return self.gamma_errors_llh_scan, self.ns_errors_llh_scan


    def get_flux_uncertainties_llh_scan(self, gamma_min=1, gamma_max=5, flux_min_factor=10, flux_max_factor=10, n_fluxes=50, n_gamma=30, \
        show_plot=False, save_plot='True'):
        """ calculate the uncertainties for the flux with a scan of the llh function. This is not reliable when the fitted source is weak. 
        Use the Feldman-Cousins confidence level estimation instead. 
        
        Parameters:
        -----------
        gamma_min: the minimal spectral index
        gamma_max: the maximal spectral index
        flux_min: the minimal number of signal events
        flux_max: the maximal number of signal events
        n_flux: how many steps there should be between ns_min and ns_max (gridsize)
        n_gamma: how many steps there should be between gamma_min and gamma_max (gridsize)
        show_plot: boolean, True if it should show the contour plot
        save_plot: boolean, True if it should save the contour plot

        Returns:  
        --------
            - gamma_errors: (negative gamma error, positive gamma error)
            - flux_errors: (negative ns error, positive ns error)
        """
        llh_ratios, plot_flux, plot_gamma = self.parameter_llh_scan_flux(n_fluxes=n_fluxes, n_gamma=n_gamma, \
            gamma_max=gamma_max, gamma_min=gamma_min, flux_min_factor=flux_min_factor, flux_max_factor=flux_max_factor)
        contours = self.plot_contours_llh_scan(llh_ratios, plot_gamma, plot_flux, yparam="flux", show_plot=show_plot,\
             save_plot=save_plot)
        uncertainties = self.get_uncertainties_llh_scan(contours, self.flux_in_GeV)
        self.flux_errors_llh_scan = (uncertainties[2], uncertainties[3])
        self.gamma_errors_llh_scan = (uncertainties[0], uncertainties[1])
        return self.gamma_errors_llh_scan, self.flux_errors_llh_scan



    def get_top_events(self, N=10):
        """ get the top N contributing events to the flux
        
        Parameters:
        -----------
        N: number of top contributing events

        Returns:  
        --------
        top contributing events (structured array)
        """

        event_stack = []

        for dset_idx in range(len(self.ana._tdm_list)):

            # get sob values
            param_array = self.ana.llhratio.llhratio_list[0]._pmm.create_src_params_recarray([self.ns, self.gamma if self.gamma is not None else self.e_peak])

            pdfratioarray = self.ana.llhratio.llhratio_list[dset_idx]._pdfratio
            sob_vals = pdfratioarray.get_ratio(self.ana.tdm_list[dset_idx], param_array)

            # get events and add sob values
            events = self.ana._tdm_list[dset_idx].events.as_numpy_record_array()
            events = append_fields(events, 'SoB', sob_vals, usemask=False, asrecarray=True)

            # also add angular distance to source
            psi = self.GreatCircleDistance(np.deg2rad(self.ra), np.deg2rad(self.dec), events['ra'], events['dec'])
            events = append_fields(events, 'psi_GreatCircleDistance', psi, usemask=False, asrecarray=True)

            event_stack.append(events)

        if(len(event_stack)>1):
            events = stack_arrays(event_stack, usemask=False, asrecarray=True)
        else:
            events = event_stack[0]

        idx = np.argsort(events['SoB'])

        return events[idx[-int(N):]]


    def load_smearing_matrices(self):
        """ loads the smearing matrix for the true energy calculation

        Returns:  
        --------
        dictionary with smearing matrices for each sample
        """
        keys = ["40", "59", "79", "86_I", "86_II"]
        smearing_dict = {}
        for period in keys:
            tmp_matrix = np.genfromtxt(os.path.join(self.public_data_path, "icecube_10year_ps/irfs/IC" + period + "_smearing.csv"),
                                    dtype=[("log10_nu_min", "f4"), ("log10_nu_max", "f4"), ("Dec_nu_min", "f4"), 
                                            ("Dec_nu_max", "f4"), ("log10_min", "f4"), ("log10_max", "f4"),
                                            ("psf_min", "f4"), ("psf_max", "f4"), ("ang_err_min", "f4"),
                                            ("ang_err_max", "f4"), ("frac_count", "f8")])
            
            smearing_dict[period] = tmp_matrix
        self.smearing = smearing_dict
        return smearing_dict


    def load_effective_areas(self):
        """ loads the effective areas for the true energy calculation

        Returns:  
        --------
        dictionary with effective area matrices for each sample
        """
        keys = ["40", "59", "79", "86_I", "86_II"]
        effA_dict = {}
        for period in keys:
            tmp_matrix = np.genfromtxt(os.path.join(self.public_data_path, "icecube_10year_ps/irfs/IC" + period + \
                                    "_effectiveArea.csv"), 
                                    dtype=[("log10_nu_min", "f4"), ("log10_nu_max", "f4"), ("Dec_nu_min", "f4"),
                                            ("Dec_nu_max", "f4"), ("Aeff", "f8")])
            effA_dict[period] = tmp_matrix
        self.effA = effA_dict
        return effA_dict


    # uptime is cancelled in the fraction, we don't need this for the energy range calculation. Just included for completeness. 
    def load_uptime(self):
        """ loads the detector uptimes

        Returns:  
        --------
        dictionary with the uptimes for each sample. 
        """
        keys = ["40", "59", "79", "86_I", "86_II"]
        my_dict = {}
        for period in keys:
            tmp_matrix = np.genfromtxt(os.path.join(self.public_data_path, "icecube_10year_ps/uptime/IC" + period + \
                                        "_exp.csv"), 
                                        dtype=[("MJD_start", "f4"), ("MJD_stop", "f4")])
            uptime = np.sum(tmp_matrix["MJD_stop"] - tmp_matrix["MJD_start"])
            
            my_dict[period] = uptime
        return my_dict


    def select_reco_smearing(self, smearing, e_reco, sigma_reco, dec):
        """ Select the relevant entries in the smearing matrix
        
        Parameters:
        -----------
        smearing: smearing matrix for one sample
        e_reco: log10(E) of the reconstructed energy
        sigma_reco: reconstructed uncertainty in degrees (>= 0.2)
        dec: event declination in degrees

        Returns:  
        --------
        the masked smearing matrix
        """
        # get all entries with dec
        if dec == 90: # cover the case with the maximum
            tmp_smearing = smearing[smearing["Dec_nu_max"] >= dec]
        else:
            tmp_smearing = smearing[(smearing["Dec_nu_min"] <= dec) & (smearing["Dec_nu_max"] > dec)]
            
        # get all entries with e_reco
        if e_reco == 9: # cover the case with the maximum
            tmp_smearing = tmp_smearing[tmp_smearing["log10_max"] == e_reco]
        else:
            tmp_smearing = tmp_smearing[(tmp_smearing["log10_min"] <= e_reco) & \
                                        (tmp_smearing["log10_max"] > e_reco)]

        # get all entries with similar error
        if sigma_reco == max(tmp_smearing["ang_err_max"]):
            tmp_smearing = tmp_smearing[tmp_smearing["ang_err_max"] == sigma_reco]
        else:
            tmp_smearing = tmp_smearing[(tmp_smearing["ang_err_min"] <= sigma_reco) &\
                                    (tmp_smearing["ang_err_max"] > sigma_reco)]

        return tmp_smearing


    def get_smearing_reco_given_true_e(self, e_reco, sigma_reco, e_true, dec):
        """ get the probability for a reconstruction given a true energy from the smearing matrix
        
        Parameters:
        -----------
        P(e_reco, sigma_reco, dec | e_true)
        e_reco: log10(E) of the reconstructed energy
        sigma_reco: reconstructed uncertainty in degrees (>= 0.2)
        e_true: true energy in GeV
        dec: event declination in degrees
        
        Returns:  
        -------- 
        dict of P(e_reco, sigma_reco, dec | e_true) for each data sample
        """
        prob_reco_smearing = {}
        for key in self.smearing:
            #select correct dec, e_reco and sigma_reco
            tmp_smearing = self.select_reco_smearing(self.smearing[key], e_reco, sigma_reco, dec)
            #select correct true energy for P(E_reco, sigma_reco, dec_reco | E_true)
        
            if np.log10(e_true) == 9:
                tmp_smearing = tmp_smearing[tmp_smearing["log10_nu_max"] == np.log10(e_true)] 
            else:
                tmp_smearing = tmp_smearing[(tmp_smearing["log10_nu_min"] <= np.log10(e_true)) & \
                                        (tmp_smearing["log10_nu_max"] > np.log10(e_true))]

            prob_reco_smearing[key] = sum(tmp_smearing["frac_count"])
            
        return prob_reco_smearing


    def calc_flux_energy_integral(self, e_min, e_max, e_norm=1000):
        """ integral of flux: int_e_min ^e_max (flux_norm) * (e/e_norm)^(-gamma) de
        
        Parameters:
        -----------
        e_min: the lower bound of the energy integral in GeV
        e_max: the upper bound of the energy integral in GeV
        e_norm: normalization, usually 1000 Gev

        Returns:  
        --------
        float
        """
        if self.gamma is not None:
            return (self.flux_in_GeV / (e_norm**(-self.gamma))) * (1 / (-self.gamma + 1)) * ((e_max**(-self.gamma+1)) - (e_min**(-self.gamma+1)))
        else:

            return self.ana.shg_mgr.shg_list[0].fluxmodel.energy_profile.get_integral(e_min, e_max).item()

        
        
    def prob_true_energy_single_sample(self, effA, e_true, dec):
        """ calculates the probability to observe events with given true Energy based on effective area for a 
        single sample P(e_true) = N_e_true / N_all
        
        Parameters:
        -----------
        effA: effective area matrix for a single sample
        e_true: true neutrino energy in GeV
        dec: event declination in degrees

        Returns:  
        --------
        probability to observe events with given true Energy, bin width of declination bin in effective area matrix
        """
        # we can leave the uptime, since it will be removed in the fractions anyways. 

        if dec == 90:
            tmp_effA = effA[effA["Dec_nu_max"] == dec]

        else:
            tmp_effA = effA[(effA["Dec_nu_min"] <= dec) & (effA["Dec_nu_max"] > dec)]

        # get complete flux, loop through all energy bins
        number_all = 0
        for energy_min, energy_max in zip(np.unique(tmp_effA["log10_nu_min"]), 
                                        np.unique(tmp_effA["log10_nu_max"])):
        # effective Area for all energies multiplied with flux
            energy_mask = (tmp_effA["log10_nu_min"] == energy_min) & (tmp_effA["log10_nu_max"] == energy_max)
            tmp_flux = self.calc_flux_energy_integral(10**energy_min, 10**energy_max)
        
            number_all += np.sum(tmp_effA[energy_mask]["Aeff"]) * tmp_flux # here would the uptime be (in the sum)
        # get the probability to observe the true Energy --> number_e_true / number_all
        
        # select correct energy range
        
        if np.log10(e_true) == 10:
            energy_mask = tmp_effA["log10_nu_max"] == np.log10(e_true)
        else:
            energy_mask = (tmp_effA["log10_nu_min"] <= np.log10(e_true)) & \
            (tmp_effA["log10_nu_max"] > np.log10(e_true))
            
        e_min, e_max = min(tmp_effA["log10_nu_min"][energy_mask]), max(tmp_effA["log10_nu_max"][energy_mask])

        tmp_flux = self.calc_flux_energy_integral(10**e_min, 10**e_max)
        number_e_true = np.sum(tmp_effA[energy_mask]["Aeff"]) * tmp_flux # here would the uptime be (in the sum)
        return number_e_true / number_all, np.abs(tmp_effA["Dec_nu_max"][0] - tmp_effA["Dec_nu_min"][0])   


    def prob_reco_event_times_prob_e(self, tmp_smearing, e_reco, sigma_reco, dec, tmp_effA, e_true):
        """ calculates the probability to observe events with given reconstructed properties for flux
        P(e_reco, sigma_reco | E_true) * P(E_true)
        
        Parameters:
        -----------
        tmp_smearing: smearing matrix (single sample) 
        e_reco: log10(E) of reconstructed energy
        sigma_reco: reconstruction uncertainty in degrees
        dec: event declination in degrees
        tmp_effA: effective area matrix (single sample)
        e_true: true neutrino energy in GeV

        Returns:  
        --------
        probability to observe events with given reconstructed properties for self.flux_in_GeV
        """
        tmp_smearing = self.select_reco_smearing(tmp_smearing, e_reco, sigma_reco, dec)
        
        if np.log10(e_true) >= 9:
            tmp_smearing_true_e = tmp_smearing[(tmp_smearing["log10_nu_max"] == 9)]

        else:
            tmp_smearing_true_e = tmp_smearing[(tmp_smearing["log10_nu_min"] <= np.log10(e_true)) & 
                            (tmp_smearing["log10_nu_max"] > np.log10(e_true))]

        # P(e_reco, sigma_reco | E_true) * P(E_true)
        prob_e, bin_dec_effA = self.prob_true_energy_single_sample(tmp_effA, e_true, dec)
        # energy:
        # effective area bin: 0.2, smearing matrix bin: 0.5 --> take average of smearing matrix 0.5/0.2
        # declination:
        # also take average ratio of declination bin sizes for smearing
        bin_dec_smearing = np.abs(tmp_smearing["Dec_nu_max"][0] - tmp_smearing["Dec_nu_min"][0])

    #   this correction factor acutally cancels out in the fraction, so we can also just leave it. 
        prob_reco = sum(tmp_smearing_true_e["frac_count"]) / 2.5 / bin_dec_smearing / bin_dec_effA * prob_e

        return prob_reco


    def get_sample_weight(self, e_true, dec):
        """ we need to scale the energy contribution with the correct weight of the probability to 
        detect this energy for the respective detector configuration. This here is the sample effective area divided by the 
        summed effective area for all samples
        
        Parameters:
        -----------
        e_true: true neutrino energy in GeV
        dec: event declination in degrees

        Returns:  
        --------
        dictionary with contribution ratio for each sample
        """
        contribution_ratio = {}
        total = 0

        for key in self.effA:
            effA = self.effA[key]
            if dec == 90:
                tmp_effA = effA[effA["Dec_nu_max"] == dec]

            else:
                tmp_effA = effA[(effA["Dec_nu_min"] <= dec) & (effA["Dec_nu_max"] > dec)]

            if np.log10(e_true) == 10:
                tmp_effA = tmp_effA[tmp_effA["log10_nu_max"] == np.log10(e_true)]
            else:
                tmp_effA = tmp_effA[(tmp_effA["log10_nu_min"] <= np.log10(e_true)) & \
                (tmp_effA["log10_nu_max"] > np.log10(e_true))]
            contribution_ratio[key] = sum(tmp_effA["Aeff"])
            total += contribution_ratio[key]
            
        for key in contribution_ratio:
            if total > 0:
                contribution_ratio[key] = contribution_ratio[key] / total
            
        return contribution_ratio


    def get_quantiles(self, weights, energy_bins):
        """ compute the central 90% quantiles from a given distribution.
        
        Parameters:
        -----------
        weights: the contribution for each energy bin
        energy_bins: bin edges of the energy bins

        Returns:  
        --------
        [lower quantile, upper quantile]
        """
        all_sum = sum(weights)
        quantile = []
        tmp_quantile = 0
        lower_quantile = False
        for tmp_index, tmp_weight in enumerate(weights):
            if all_sum > 0:
                tmp_quantile += (tmp_weight / all_sum)
            if tmp_quantile >= 0.05 and not lower_quantile:
                quantile.append(energy_bins[tmp_index])
                lower_quantile = True
            if tmp_quantile >= 0.95:
                quantile.append(energy_bins[tmp_index])
                break
        return quantile


    def get_weights(self, e_reco, sigma_reco, dec, bin_size=0.2, single_sample=None):
        """ get the correct probability to observe the true energy given a reconstructed event (for given flux and declination)
        P(E_true | e_reco, sigma_reco) =    P(e_reco, sigma_reco | E_true) * P(E_true) / (sum over all E_true of P(e_reco, sigma_reco | E_true) * P(E_true))
        
        Parameters:
        -----------
        e_reco: the reconstructed energy in log10(E) as given in the dataset
        sigma_reco: the reconstruction error in degrees as given in the dataset
        dec: event declination in degrees
        bin_size: the enerby bin size. 0.2 are the energy bin steps for the effective area matrices
        single_sample: only evaluate this specific data set (has to be either "40", "59", "79", "86_I", "86_II")

        Returns:  
        --------
         - the histogram values of P(E_true | e_reco, sigma_reco)  for all datastes combined 
         - dictonary of histogram values of P(E_true | e_reco, sigma_reco)  for single datasets
         - energy bins
        """
        
        energy_bins = np.arange(2, 10, bin_size)

        weights = {"40": [], "59": [], "79":[], "86_I": [], "86_II": []}

        product = {"40": [], "59": [], "79":[], "86_I": [], "86_II": []}
        evidence = {"40": 0, "59": 0, "79":0, "86_I": 0, "86_II": 0}
        count=0

        for true_energy in energy_bins[:-1]:
            contribution_ratio = self.get_sample_weight(10**true_energy, dec)
            tmp_evidence = 0
            for key in weights:
                if single_sample is not None:
                    # skip this if it is the incorrect sample
                    if key != single_sample:
                        continue
                tmp_product = self.prob_reco_event_times_prob_e(self.smearing[key], e_reco, sigma_reco, dec, self.effA[key],\
                     10**true_energy)
                evidence[key] += tmp_product
                #scale correctly with detector effective Area
                if single_sample is None:
                        tmp_product = tmp_product * contribution_ratio[key]
                
                product[key].append(tmp_product)
        #         print(true_energy, prior[key])
        
        sum_weights = np.zeros(energy_bins[:-1].shape)
        # get the probability and the sum over all samples
        for key in product:
            if single_sample is not None:
                # skip this if it is the incorrect sample
                if key != single_sample:
                    continue
            weights[key] = np.nan_to_num(product[key] / evidence[key])
        
            sum_weights += weights[key]

        return sum_weights, weights, energy_bins


    def get_correct_energy_range(self):
        """ calculates the valid energy range for the fitted flux

        Returns:  
        --------
        - central 90 percent quantiles [log10(E_true_min), log10(E_true_max)]
        """
        top_events = self.get_top_events(max(np.ceil(self.ns), 1))
        self.load_smearing_matrices()
        self.load_effective_areas()
        sample_start_time = np.array([data_list.grl.as_numpy_record_array()[0]["start"] for data_list in self.ana._data_list])
        sample_end_time = np.array([data_list.grl.as_numpy_record_array()[-1]["stop"] for data_list in self.ana._data_list])
        samples = np.array(["40", "59", "79", "86_I", "86_II"])
        summed_histogram = None
        for event in top_events:
            # find correct sample and get the energy range for this specific sample
            sample = samples[(sample_start_time < event["time"]) & (sample_end_time > event["time"])][-1]
            tmp_sum_weight, weights, energy_bins = self.get_weights(event["log_energy"], max(0.2, np.rad2deg(event["ang_err"])),
             np.rad2deg(event["dec"]), single_sample=sample)
            if not isinstance(summed_histogram, np.ndarray):
                summed_histogram = tmp_sum_weight
            else:
                summed_histogram += tmp_sum_weight

        self.true_E = self.get_quantiles(summed_histogram, energy_bins)
        return self.true_E


    def get_flux_with_uncert(self, energy, use_llh_scan=False, energy_units="GeV"):
        """ calculate the simplified bowtie of the flux with uncertainties for flux and gamma. This is i.e. for a plot with energy as the x-values and the flux as the y-values. 
        This function uses self.flux_in_GeV as the best fit flux, self.gamma as the best fit gamma, self.flux_error as a tuple with the flux errors (negative error, positve error), 
        and self.gamma_error as a tuple with the gamma errors (negative error, positive error). These will be set if we have fitted the flux and determined the uncertainties in the same instance. 
        If not, these attributes need to be set in advance. 
        
        Parameters:
        -----------
        energy: energy in the same unit as fitted flux (GeV) (array (n, 1))
        use_llh_scan: set to True if the uncertainties were determined with the scan of the llh contour. 

        Returns:  
        --------
        - flux: flux values for energies (array (n, 1))
        - lower edge of bowtie (array (n, 1))
        - upper edge of bowtie (array (n, 1)) 
        """

        flux_norm = self.ana.shg_mgr.shg_list[0].fluxmodel.energy_profile.E0

        if energy_units == "GeV" and flux_norm == 1000:
            energy = energy / flux_norm

        if use_llh_scan:
            fluxPos = self.flux_errors_llh_scan[1]
            fluxNeg = abs(self.flux_errors_llh_scan[0])
            gammaPos = self.gamma_errors_llh_scan[1]
            gammaNeg = abs(self.gamma_errors_llh_scan[0])
        else:
            fluxPos = self.flux_errors[1]
            fluxNeg = abs(self.flux_errors[0])
            gammaPos = self.gamma_errors[1]
            gammaNeg = abs(self.gamma_errors[0])

        flux = self.flux_in_GeV * ((energy)**-self.gamma) 
        low_err = (self.flux_in_GeV - fluxNeg) * ((energy)**-self.gamma)  
        high_err = (self.flux_in_GeV + fluxPos) * ((energy)**-self.gamma)  
        
        low_err2 = (self.flux_in_GeV - fluxNeg) * ((energy)**-(self.gamma + gammaPos)) 
        high_err2 = (self.flux_in_GeV + fluxPos) * ((energy)**-(self.gamma - gammaNeg)) 
        low_err3 = (self.flux_in_GeV + fluxPos) * ((energy)**-(self.gamma + gammaPos)) 
        high_err3 = (self.flux_in_GeV - fluxNeg) * ((energy)**-(self.gamma - gammaNeg)) 
        
        return flux, \
        np.min(np.vstack((low_err, low_err2, high_err2, low_err3, high_err3)), axis=0), \
        np.max(np.vstack((high_err, high_err2, low_err2, low_err3, high_err3)), axis=0)
