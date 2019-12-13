###############################################################################
#                      PROJECT: STELLAR STRUCTURE                             #
#               Markus Batzer, Miquel Miravet, David Vallés                   #
#                                                                             #
#                             Module: cluster                                 #
#     This model contains the relevant functions/calls/whatever for the       #
#    part of the project where we will try to generate a synthetic cluster    #
#                                                                             #
#                          Author: David Vallés                               #
###############################################################################

import numpy as np

##### SECTION: MONTECARLO: METROPOLIS ALGORITHM (IMF)
def uniform_proposal(mmin=0.42, mmax=18.2):
    '''
    Draws a random mass from a uniform distribution between mmin and mmax.
    Units are solar masses.
    This is required for the Metropolis algorithm.
    D. Vallés

    Parameters:
    mmin: minimum mass
    mmax: maximum mass

    Returns:
    Random (uniform) number inside this limits
    '''
    return np.random.uniform(mmin, mmax)

def metropolis(xi, nsamples, proposal):
    '''
    Simple implementation of the Metropolis algorithm in order to sample
    a pdf xi of a random variable.

    Parameters:
    xi: pdf, passed as a function of a single variable
    nsamples: number of samples to generate
    proposal: function which tries a new value of our random variable
    (part of the Metropolis method). We will typically choose it as
    uniform_proposal()

    Returns: generated masses (has to be passed through a list() function to
    have a nice list after running the function)
    '''
    m = 1 #we start, for example, with a solar mass star

    for i in range(nsamples):
        trial = proposal()
        acceptance = xi(trial)/xi(m)

        if np.random.uniform() < acceptance:
            m = trial
        yield m

def salpeter_xi(M, mmin=0.42, mmax=18.2, salpeter_index=2.35):
    '''
    Returns the value of the Salpeter IMF (unnormalized; we will care about
    normalization to the total mass of the cluster later on).

    Parameters:
    M: value of the mass for which the pdf is computed. Units are solar masses.
    mmin: minimum value of the mass to generate (either a 'computational
    cutoff', i.e. the minimum value we can generate working models; or a
    physical one [0.08?])
    mmax: maximum value of the mass to generate. Same caveat as above.
    salpeter_index: 2.35 for Salpeter IMF, but could be changed to explore
    other models.

    Returns:
    Unnormalized value of the pdf.
    '''
    if mmin < M and M < mmax:
        return M**(-salpeter_index)
    else:
        return 0
