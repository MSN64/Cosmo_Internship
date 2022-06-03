import numpy as np

import sys
import pickle
import pandas as pd


import sys, platform, os
import scipy.constants as cst

camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)
import camb
from camb import model, initialpower

train_params = np.load('LHS_params_1dim5000.npz')
print(train_params.files)

#print(train_params['h'])

n_samples = len(train_params['h'])
print('number of training samples: ', len(train_params['h']))

cosmo_params = np.zeros((len(train_params['h']), 1))

cosmo_params[:, 0] = train_params['h']


def camb_cosmo(i):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * cosmo_params[i, 0], ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(As=2e-9, ns=0.965)
    pars.set_matter_power(redshifts=[0.0], kmax=10.0)
    pars.set_dark_energy(w=-1.0, wa=0, dark_energy_model='fluid')

    # Linear spectra
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    camb_kh, z, camb_pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=15, npoints=2000)
    s8 = np.array(results.get_sigma8())
    # s8 is the amplitude of matter fluctuations, i.e. in term of the average r.m.s fluctuation in a sphere of 8h^{−1}Mpc

    return camb_kh, camb_pk[0]


cosmo_func = camb_cosmo(0)

len(train_params['h'])

#41 mins
## Training input params:
h = cosmo_params[:, 0]

#Obtain pk matrix for output training param:
pk_matrix = []

for i in range(len(train_params['h'])):
               print(i)
               cosmo_func = camb_cosmo(i)
               pk_matrix.append(cosmo_func[1])

with open('pk_data_1dim5000.pkl', 'wb') as f:
    pickle.dump(pk_matrix, f)

with open('pk_data_1dim5000.pkl', 'rb') as f:
    pk_matrix = pickle.load(f)

print(np.shape(pk_matrix))