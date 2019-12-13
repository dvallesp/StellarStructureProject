###############################################################################
#                      PROJECT: STELLAR STRUCTURE                             #
#               Markus Batzer, Miquel Miravet, David Vallés                   #
#                                                                             #
#                            Module: read_data                                #
#     Contains functions in order to read the outputs and also generate       #
#    csv files in order to read them easily from other software packages      #
#                                                                             #
#                   Authors: David Vallés & Markus Batzer                     #
###############################################################################

import numpy as np
import pandas as pd
import os

def read_output(filename, path=''):
    '''
    This function reads the simulation's outputs (only the profiles).
    It returns each of the variables as a list.
    D. Vallés & M. Batzer

    Parameters:
    filename: name of the parameters file to output
    path: path where to write the file

    Returns:
    idx: index of each point (2 to 201, correlative)
    M: mass coordinate expressed as 1 - Mr/Mtot
    r: log(radial coordinate) in cgs
    P: log(pressure) in packages
    T: log(temperature) in kelvin
    rho: log(density) in cgs
    L: log(luminosity) in cgs
    eps: log(epsilon), where epsilon is the nuclear energy production, in cgs
    op: log(opacity), in cgs
    Lc: log(convective luminosity) in cgs
    LcLtot: quotient of convective to total luminosity
    Del:
    Delad:
    Delrad:
    '''
    idx = []
    M = []
    r = []
    P = []
    T = []
    rho = []
    L = []
    eps = []
    op = []
    Lc = []
    LcLtot = []
    Del = []
    Delad = []
    Delrad = []

    with open(os.path.join(path,filename)) as file:
        while file.readline().split()[-1] != 'LOG(L)':
            continue
        for i in range(199):
            line = [value.replace('D','e') for value in file.readline().split()]
            line = [float(value) for value in line]
            idx.append(int(line[0]))
            M.append(line[1])
            r.append(line[2])
            P.append(line[3])
            T.append(line[4])
            rho.append(line[5])
            L.append(line[6])

        file.readline()
        file.readline()

        for i in range(199):
            line = file.readline().split()
            if len(line) == 7:
                a = line[0].split('-')
                eps.append(-float(a[1]))
                remaining = line[1:]
            else:
                eps.append(float(line[1]))
                remaining = line[2:]
            remaining = [float(value) for value in remaining]
            op.append(remaining[0])
            Lc.append(remaining[1])
            LcLtot.append(remaining[2])
            Del.append(remaining[3])
            Delad.append(remaining[4])
            Delrad.append(remaining[5])

    return idx, M, r, P, T, rho, L, eps, op, Lc, LcLtot, Del, Delad, Delrad


def write_csv(filename, path=''):
    '''
    This function reads an output file and creates a csv with the profile
    information.
    M. Batzer and D. Vallés

    Parameters:
    filename: name of the parameters file to output
    path: path where to write the file
    '''
    columns = read_output(filename, path)
    values = np.hstack(columns).reshape(14,199).transpose()
    column_names = ['n','1-Mr/M', 'log(r)', 'log(P)', 'log(T)', 'log(rho)',
                    'log(L)', 'log(eps)','log(op)', 'log(Lc)', 'Lc/Ltot',
                    'Del', 'Delad', 'Delrad']
    pd.DataFrame(values).to_csv(os.path.join(path,filename)+'.csv',
                                sep=',',index=False, header=column_names)


def read_params(filename, path=''):
    '''
    This function reads the simulation's outputs (only the parameters).
    It returns a tuple with the parameters
    D. Vallés

    Parameters:
    filename: name of the parameters file to output
    path: path where to find the file

    Returns: (as a list)
    m: mass
    x: H content
    y: He content
    p: central pressure
    Tc: central temperature
    R: radius
    Lergs: total luminosity
    Teff: effective (surface) temperature
    Lsununits: luminosity (in solar units!)
    '''
    with open(os.path.join(path,filename)) as f:
        line = f.readline().split()
        m = float(line[1].replace(',','').replace('D','e'))
        x = float(line[3].replace(',','').replace('D','e'))
        y = float(line[5].replace(',','').replace('D','e'))
        while f.readline().split()[0] != '*****FINAL':
            continue
        line = f.readline().split()
        p = float(line[1].replace(',','').replace('D','e'))
        Tc = float(line[3].replace(',','').replace('D','e'))
        R = float(line[5].replace(',','').replace('D','e'))
        Lergs = float(line[7].replace(',','').replace('D','e'))
        line = f.readline().replace(':-',': -').split()
        Teff = float(line[1].replace(',','').replace('D','e'))
        Lsununits = 10**float(line[5].replace(',','').replace('D','e'))

    return [m,x,y,p,Tc,R,Lergs,Teff,Lsununits]
