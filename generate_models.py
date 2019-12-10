###############################################################################
#                      PROJECT: STELLAR STRUCTURE                             #
#               Markus Batzer, Miquel Miravet, David Vallés                   #
#                                                                             #
#                        Module: generate_models                              #
#     Contains functions in order to massively generate stellar models        #
#                                                                             #
#                          Author: David Vallés                               #
###############################################################################


#### General libraries
import os
import sys
import numpy as np
from scipy.interpolate import interp1d as interp
import read_data


#### Function write_pars_file
def write_pars_file(filename='custom_pars.pars', path='', mass=1, x=0.7,
                    y=0.292, p=1e17, T=1.5e7, R=7e10, L=1,
                    output_filename='custom_output.dat'):
    '''
    This function writes the pars.pars (named as filename) file to get passed
    to the code.
    D. Vallés

    Parameters:
    filename: name of the parameters file to output
    path: path where to write the file
    mass: total mass, in M_Sun units
    x: hydrogen fraction
    y: helium fraction
    p: central pressure in cgs
    T: core temperature in K
    R: radius in cm
    L: luminosity (in L_Sun units)
    output_filename: name for the output (results) file
    '''
    if x+y>1:
        raise ValueError('x + y cannot be bigger than 1!')
        return
    with open(os.path.join(path,filename),'w+') as file:
        file.write('{}\n'.format(mass))
        file.write('{} {}\n'.format(x,y))
        file.write('{}\n'.format(p))
        file.write('{}\n'.format(T))
        file.write('{}\n'.format(R))
        file.write('{}\n'.format(L))
        file.write(output_filename+'\n')
        file.write('n')


#### Function reasonable_MS_guess
def reasonable_MS_guess(Mmin=1, Mmax=15, howmany=200):
    '''
    Generates a tuple vector with reasonable guesses for building a MS in the
    HR diagram.
    They are picked by interpolation from the ones given in the manual.
    D. Vallés

    Parameters:
    Mmin: low-mass end of the interval to generate guesses
    Mmax: high-mass end of the interval to generate guesses
    howmany: num. of guesses to generate

    Returns:
    5 lists containing, respectively, the mass, central pressure, central
    density,radius and luminosity of the guesses
    '''
    # reference values
    M_ref = [1,3,15]
    p_ref = [1.482e17, 1.141e17, 2.769e16]
    T_ref = [1.442e7, 2.347e7, 3.275e7]
    R_ref = [6.932e10, 1.276e11, 3.289e11]
    L_ref = [0.9083, 89.35, 1.960e4]    # it seems that one is logarithmically
                                        #spaced!

    # interpolating function
    p_int = interp(M_ref, p_ref, fill_value='extrapolate')
    T_int = interp(M_ref, T_ref, fill_value='extrapolate')
    R_int = interp(M_ref, R_ref, fill_value='extrapolate')
    logL_int = interp(M_ref, np.log(L_ref), fill_value='extrapolate')

    # interpolated values
    M = np.linspace(Mmin, Mmax, howmany)
    p = p_int(M)
    T = T_int(M)
    R = R_int(M)
    L = np.exp(logL_int(M))

    # models
    models = [[M[i], p[i], T[i], R[i], L[i]] for i in range(M.size)]

    return models


#### Function has_succeeded
def has_succeeded(output_filename='custom_output.dat', path=''):
    '''
    Checks if the model has succeeded. Returns True if it has; False otherwise.
    Failures are detected as files which have exactly 4 lines (experience tells
    us so).
    If more error cases are found, they can be added to the condition below.
    D. Vallés
    '''
    with open(os.path.join(path,output_filename),'r') as file:
        lines = lines = sum(1 for _ in file)
    if lines == 4:
        return False
    return True


#### Function launch_model
def launch_model(parameters_filename='custom_pars.pars', exec_name='zams'):
    '''
    Launches a model with the parameters given by the parameters file.
    Executable is named exec_name.
    Both, parameters and executable, are assumed to be in the same folder as
    this script.
    D. Vallés
    '''
    os.system('./' + exec_name + ' < ' + parameters_filename)


#### Function do_pipeline_models_zams_work
def do_pipeline_models_zams_work(Mmin=1, Mmax=15, howmany=200, x=0.7, y=0.292):
    '''
    Generates howmany models. Check if they work (just to assess the initial
    guesses).
    D. Vallés

    Parameters:
    Mmin: low-mass end of the interval to generate models
    Mmax: high-mass end of the interval to generate models
    howmany: num. of models to generate
    '''
    models = reasonable_MS_guess(Mmin, Mmax, howmany)
    worked = []
    m = []
    for idx, model in enumerate(models):
        print('Starting model no. {}'.format(idx))
        write_pars_file(filename='custom_pars.pars', path='',
                        mass=model[0], x=x, y=y, p=model[1], T=model[2],
                        R=model[3], L=model[4],
                        output_filename='custom_output.dat')
        launch_model(parameters_filename='custom_pars.pars', exec_name='zams')

        m.append(model[0])
        worked.append(has_succeeded(output_filename='custom_output.dat',
                        path=''))
    return m, worked


#### Function pipeline_models_zams
def pipeline_models_zams(Mmin=1, Mmax=15, howmany=200, x=0.7, y=0.292,
                        output_folder='output', exec_name='zams',
                        write_csv=False):
    '''
    Generates howmany models according to the parameters.
    This file  needs to be in the same directory as the executable.
    An output folder with name given by output_folder variable will be created.
    D. Vallés

    Parameters:
    Mmin: low-mass end of the interval to generate models
    Mmax: high-mass end of the interval to generate models
    howmany: num. of models to generate
    x: hydrogen fraction
    y: helium fraction
    output_folder: name of the folder that will be created to write the outputs
    exec_name: name of the executable file for the simulation
    write_csv: if True (def.: False), it will write a csv for each file,
                containing the profiles.
    '''
    models = reasonable_MS_guess(Mmin, Mmax, howmany)
    failed = []
    os.system('mkdir ' + output_folder)
    os.system('cp ' + exec_name + ' ./' + output_folder)
    os.chdir(output_folder)

    for idx, model in enumerate(models):
        print('Starting model no. {:04d}'.format(idx))
        write_pars_file(filename='custom_pars.pars',
                        mass=model[0], x=x, y=y, p=model[1], T=model[2],
                        R=model[3], L=model[4],
                        output_filename='output_{:04d}.dat'.format(idx))
        launch_model(parameters_filename='custom_pars.pars', exec_name=exec_name)
        if not has_succeeded(output_filename='output_{:04d}.dat'.format(idx)):
            failed.append(idx)
        else:
            if write_csv:
                read_data.write_csv(filename='output_{:04d}.dat'.format(idx))
    os.system('rm ./' + exec_name)
    os.system('rm ./' + 'custom_pars.pars')
    os.chdir('..')
    print('Failed models:', failed)
