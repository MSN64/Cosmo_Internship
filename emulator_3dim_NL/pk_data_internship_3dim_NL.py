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

train_params = np.load('LHS_params_3dim100_NL.npz')
print(train_params.files)

#print(train_params['h'])

n_samples = len(train_params['h'])
print('number of training samples: ', len(train_params['omega_b']))

cosmo_params = np.zeros((len(train_params['h']), 3))

cosmo_params[:, 0] = train_params['h']
cosmo_params[:, 1] = train_params['omega_b']
cosmo_params[:, 2] = train_params['omega_cdm']

def camb_cosmo(i):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * cosmo_params[i, 0], ombh2=cosmo_params[i, 1], omch2=cosmo_params[i, 2])
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

    # Non-Linear spectra (Halofit)
    pars.NonLinear = model.NonLinear_both
    results.calc_power_spectra(pars)
    camb_kh_nonlin, z_nonlin, camb_pk_nonlin_cur = results.get_matter_power_spectrum(minkh=1e-4, maxkh=15, npoints = 500)
    camb_pk_nonlin = renorm_s8*camb_pk_nonlin_cur


    return camb_kh, camb_pk[0], camb_pk_nonlin[0]


cosmo_func = camb_cosmo(0)

len(train_params['h'])

## Training input params:
h = cosmo_params[:, 0]
omega_b = cosmo_params[:, 1]
omega_c = cosmo_params[:, 2]

#Obtain pk matrix for output training param:
pk_matrix = []
pk_nonlin_matrix = []

for i in range(len(train_params['h'])):
    print(i)
    cosmo_func = camb_cosmo(i)
    pk_matrix.append(cosmo_func[1])
    pk_nonlin_matrix.append(cosmo_func[2])

with open('pk_data_3dim100_NL1.pkl', 'wb') as f, open('pk_data_3dim100_NL2.pkl', 'wb') as g:
    pickle.dump(pk_matrix, f)
    pickle.dump(pk_nonlin_matrix, g)

with open('pk_data_3dim100_NL1.pkl', 'rb') as f, open('pk_data_3dim100_NL2.pkl', 'rb') as g:
    pk_matrix = pickle.load(f)
    pk_nonlin_matrix = pickle.load(g)

print(np.shape(pk_matrix))
print(np.shape(pk_nonlin_matrix))