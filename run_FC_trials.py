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
parser.add_argument("--gamma_min", default=1,  type=float, help="lower bound for gamma")
parser.add_argument("--gamma_max", default=5,  type=float, help="upper bound for gamma")
parser.add_argument("--n_gamma", default=51,  type=int, help="steps between gamma_min and gamma_max")
parser.add_argument("--ns_min", default=0, type=float, help="lower bound for ns")
parser.add_argument("--ns_max", default=30, type=float, help="upper bound for ns")
parser.add_argument("--n_ns", default=61, type=int, help="steps between ns_min and ns_max")
parser.add_argument("--n_trials", default=100, type=int, help="number of trials for each gamma, ns combination")

args = parser.parse_args()

master_table = pd.read_csv(os.path.join(os.getenv("TABLE_PATH"), 'mastertable.txt'), sep="\s+")

entry = master_table.loc[args.source]

my_source = Source(entry["RA"], entry["DEC"], name=entry["NAME"])

# fit the flux
# flux, gamma = my_source.calculate_neutrino_flux(gamma_min=1, gamma_max=3.7)
my_source.create_skyllh_analysis(gamma_min=1, gamma_max=3.7, ns_max=1e3)

if not os.path.exists(args.trial_path):
    os.mkdir(args.trial_path)

save_file = os.path.join(args.trial_path, "FC_trials_minuit_index_" + str(args.source) + "_ns_min_" + str(args.ns_min) + "_ns_max_" + str(args.ns_max) + "_gamma_min_" + str(args.gamma_min) + "_gamma_max_" + str(args.gamma_max) + "_seed_" + str(args.seed) + ".npy")
trials = my_source.generate_trials_for_gamma_ns_grid(gamma_min=args.gamma_min, gamma_max=args.gamma_max, n_gamma=args.n_gamma, ns_min=args.ns_min, 
                                        ns_max=args.ns_max, n_ns=args.n_ns, n_trials=args.n_trials, seed=args.seed, 
                                        path=save_file)

np.save(save_file, trials)

