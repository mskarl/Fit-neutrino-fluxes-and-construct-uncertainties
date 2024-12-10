import os
from icecube_flux import Source
import pandas as pd
import numpy as np
import argparse
from dotenv import load_dotenv
load_dotenv()

parser = argparse.ArgumentParser()

parser.add_argument("--source", default=0, type=int, help="index in master table")
parser.add_argument("--trial_path", default="./neutrino_trials/", type=str, help="path where to save trials")
parser.add_argument("--seed", default=1, type=int, help="seed for random state generator")
parser.add_argument("--epeak_min", default=1,  type=float, help="lower bound for epeak")
parser.add_argument("--epeak_max", default=5,  type=float, help="upper bound for epeak")
parser.add_argument("--n_epeak", default=51,  type=int, help="steps between epeak_min and epeak_max")
parser.add_argument("--ns_min", default=0, type=float, help="lower bound for ns")
parser.add_argument("--ns_max", default=30, type=float, help="upper bound for ns")
parser.add_argument("--n_ns", default=61, type=int, help="steps between ns_min and ns_max")
parser.add_argument("--n_trials", default=100, type=int, help="number of trials for each epeak, ns combination")

args = parser.parse_args()

master_table = pd.read_csv(os.path.join(os.getenv("TABLE_PATH"), 'mastertable.txt'), sep="\s+")

entry = master_table.loc[args.source]

my_source = Source(entry["RA"], entry["DEC"], name=entry["NAME"])


example_flux = np.load(os.path.join(os.getenv("TABLE_PATH"), "nu_template_nufnu_gev_percmsq_persecond.npy"))
example_energy = np.load(os.path.join(os.getenv("TABLE_PATH"), "nu_template_energy_gev.npy"))

e_peak = np.log10(example_energy[np.argmax(example_flux)])
#epeak_seed has to be within parameter range.
e_peak_seed = args.epeak_min + (args.epeak_max - args.epeak_min) / 2

my_source.create_skyllh_epeak_analysis(source_epeak=5, epeak_min=4, epeak_max=10.05, ns_max=1e3)

if not os.path.exists(args.trial_path):
    os.mkdir(args.trial_path)


save_file = os.path.join(args.trial_path, "FC_trials_minuit_index_" + str(args.source) + "_ns_min_" + str(args.ns_min) + "_ns_max_" + str(args.ns_max) + "_epeak_min_" + str(args.epeak_min) + "_epeak_max_" + str(args.epeak_max) + "_seed_" + str(args.seed) + ".npy")
trials = my_source.generate_trials_for_epeak_ns_grid(source_energy_spectrum=example_flux, source_energies=example_energy, source_epeak=e_peak, epeak_seed=e_peak_seed,
    epeak_min=args.epeak_min, epeak_max=args.epeak_max, n_epeak=args.n_epeak, ns_min=args.ns_min, 
                                        ns_max=args.ns_max, n_ns=args.n_ns, n_trials=args.n_trials, seed=args.seed, 
                                        path=save_file)

np.save(save_file, trials)

