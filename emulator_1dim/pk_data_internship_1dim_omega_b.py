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

print("hello")

train_params = np.load('LHS_params_1dim5000_omega_b.npz')
print(train_params.files)

#print(train_params['omega_b'])

n_samples = len(train_params['omega_b'])
print('number of training samples: ', len(train_params['omega_b']))

cosmo_params = np.zeros((len(train_params['omega_b']), 1))

cosmo_params[:, 0] = train_params['omega_b']


def camb_cosmo(i):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.66, ombh2=cosmo_params[i, 0], omch2=0.11933)
    pars.InitPower.set_params(As=2.105209331e-9, ns=0.9665)
    pars.set_matter_power(redshifts=[0.0], kmax=15.0)
    pars.set_dark_energy(w=-1.0, wa=0, dark_energy_model='fluid')

    # Linear spectra
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    camb_kh, z, camb_pk_cur = results.get_matter_power_spectrum(minkh=1e-4, maxkh=15, npoints=500)
    s8_current = np.array(results.get_sigma8())
    s8_fid = np.array([0.8102])
    renorm_s8 = (s8_fid/s8_current)**2
    camb_pk = renorm_s8*camb_pk_cur
    # s8 is the amplitude of matter fluctuations, i.e. in term of the average r.m.s fluctuation in a sphere of 8h^{−1}Mpc

    return camb_kh, camb_pk[0]


cosmo_func = camb_cosmo(0)

len(train_params['omega_b'])

#41 mins
## Training input params:
omega_b = cosmo_params[:, 0]

#Obtain pk matrix for output training param:
pk_matrix = []

for i in range(len(train_params['omega_b'])):
               print(i)
               cosmo_func = camb_cosmo(i)
               pk_matrix.append(cosmo_func[1])

with open('pk_data_1dim5000_omega_b.pkl', 'wb') as f:
    pickle.dump(pk_matrix, f)

with open('pk_data_1dim5000_omega_b.pkl', 'rb') as f:
    pk_matrix = pickle.load(f)

print(np.shape(pk_matrix))