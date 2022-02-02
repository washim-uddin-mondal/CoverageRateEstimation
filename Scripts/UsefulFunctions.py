import numpy as np
from scipy.integrate import quad, dblquad
import torch

""" Expressions for the coverage probability and the ergodic rate of a PPP network are taken from the following paper:
'A Tractable Approach to Coverage and Rate in Cellular Networks', IEEE Transactions on Communications, Nov. 2011.
"""


def integrand_coverage4(v, T, Density, InvSNR):
    pi = 3.1415
    KT = 1 + np.sqrt(T) * (pi / 2 - np.arctan(1 / np.sqrt(T)))
    Term = pi*Density*v*KT + InvSNR*T*v**2
    return pi*Density*np.exp(-Term)


def integrand_rate4(v, t, Density, InvSNR):
    pi = 3.1415
    Term0 = InvSNR * (v**2) * (np.exp(t)-1) / ((pi*Density)**2)
    Term1 = v * (1 + np.sqrt(np.exp(t)-1) * ((pi/2) - np.arctan(1/np.sqrt(np.exp(t)-1))))
    return np.exp(-Term0 - Term1)


def PPPCoverageFunction(Density, SINRThr, args):
    if args.alpha == 4:

        T = 10**(SINRThr/10)
        InvSNR = args.NoiseOverPower
        result = quad(integrand_coverage4, 0, np.inf, args=(T, Density, InvSNR))
        return torch.tensor(result[0])

    else:
        raise ValueError('Please use alpha = 4.\n'
                         'Coverage expression for other values are still under development.')


def PPPRateFunction(Density, args):
    if args.alpha == 4:
        InvSNR = args.NoiseOverPower
        result = dblquad(integrand_rate4, 0, np.inf, lambda x: 0, lambda x: np.inf, args=(Density, InvSNR))
        return torch.tensor(result[0])
    else:
        raise ValueError('Please use alpha = 4.\n'
                         'Rate expression for other values is being developed.')
