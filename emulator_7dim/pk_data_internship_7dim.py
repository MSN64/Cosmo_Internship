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

train_params = np.load('LHS_params_7dim8000.npz')
print(train_params.files)

#print(train_params['omega_b'])

n_samples = len(train_params['h'])
print('number of training samples: ', len(train_params['omega_b']))

cosmo_params = np.zeros((len(train_params['omega_b']), 7))

cosmo_params[:, 0] = train_params['h']
cosmo_params[:, 1] = train_params['omega_b']
cosmo_params[:, 2] = train_params['omega_cdm']
cosmo_params[:, 3] = train_params['ns']
cosmo_params[:, 4] = train_params['z']
cosmo_params[:, 5] = train_params['lnAs']
cosmo_params[:, 6] = train_params['w']


def camb_cosmo(i):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * cosmo_params[i, 0], ombh2=cosmo_params[i, 1], omch2=cosmo_params[i, 2])
    pars.InitPower.set_params(As=np.exp(cosmo_params[i, 5]) / (10 ** 10), ns=cosmo_params[i, 3])
    pars.set_matter_power(redshifts=[cosmo_params[i, 4]], kmax=15.0)
    pars.set_dark_energy(w=cosmo_params[i, 6], wa=0, dark_energy_model='fluid')

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


cosmo_func = camb_cosmo(0)  # calculates the m.p.s of 1st sample out of n_samples

len(train_params['h'])


## Training input params:
h = cosmo_params[:, 0]
omega_b = cosmo_params[:, 1]
omega_c = cosmo_params[:, 2]
ns = cosmo_params[:, 3]
z = cosmo_params[:, 4]
lnAs = cosmo_params[:, 5]
w = cosmo_params[:, 6]

#Obtain pk matrix for output training param:
pk_matrix = []

for i in range(len(train_params['h'])):
               print(i)
               cosmo_func = camb_cosmo(i)
               pk_matrix.append(cosmo_func[1])

with open('pk_data_7dim8000.pkl', 'wb') as f:
    pickle.dump(pk_matrix, f)

with open('pk_data_7dim8000.pkl', 'rb') as f:
    pk_matrix = pickle.load(f)

print(np.shape(pk_matrix))