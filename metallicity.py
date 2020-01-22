###############################################################################
#                      PROJECT: STELLAR STRUCTURE                             #
#               Markus Batzer, Miquel Miravet, David Vallés                   #
#                                                                             #
#                             Module: metallicity                             #
#     This model contains the overall function to create the profile and      #
#     parameter files. Furthermore the simulation is done fore different      #
#                             metallicities.                                  #
#                          Author: Markus                                     #
###############################################################################

import numpy as np
import pandas as pd
import csv
import os
import read_data as rd
import generate_models as gm
import shutil as sh

##### SECTION: Initial mass guesses and resulting value (p, T, R etc.) guesses

M_list = np.geomspace(0.08,150,400)
lista_guess = cluster.several_stars_guess(M_list) 
# M, p (central), T (central), R and L (in solar luminosities) 
# Used for several_stars_guess x = 0.7381, y = 0,2485


##### SECTION: Definition for LAUNCHING THE MODEL

def full_sim(output_dir, M_list, x, y):
    '''
    make a stellar evolution model 
    extract the profile and the parameter
    save all outputs in output_dir (*.csv, dat*)
    
    Arguments:
    output_dir: name of the dir which will be saved in ./
    M_list: list of mass guesses for the simulation
    x, y: metallicity (H and He) portions
    
    M. Batzer
    '''
    #interpolation for initial values
    lista_guess = cluster.several_stars_guess(M_list)
    
    #create or check existence of output_dir
    try:
        os.mkdir(output_dir)
        print("Directory " , output_dir ,  " created ") 
    except FileExistsError:
        print("Directory " , output_dir ,  " already exists")
    
    #copy zams.f file in directory and make it executable
    sh.copyfile('./' + 'zams.f', output_dir + 'zams.f')
    os.system('gfortran -o ' + output_dir + 'zams ' + output_dir + 'zams.f')
    
    #write *.pars files in ./output_dir
    i = 0
    for guess in lista_guess:
        gm.write_pars_file('pars_{:04}.pars'.format(i), output_dir, guess[0], x, y, guess[1], guess[2], guess[3], guess[4], 'output_{:04}.dat'.format(i))
        i = i+1
    
    #launch model with *.pars files and save in ./
    for pars in os.listdir(output_dir):
        if pars.endswith('.pars'):
            #print(dirName + pars)
            gm.launch_model(output_dir + pars, output_dir + 'zams')
            
    #move *.dat files in outout_dir
    for output in os.listdir('./'):
        if output.endswith('.dat'):
            os.replace(output, output_dir + output)
    
    #check succes of the model
    for file in os.listdir(output_dir):
        if file.endswith('.dat'):
            if gm.has_succeeded(file, output_dir) == False:
                print(file, 'with mass', round(M_list[int(file[7:11])],3) ,gm.has_succeeded(file, output_dir))
    
    #read profiles from *.dat file
    for file in os.listdir(output_dir):
        if file.endswith('.dat'):
            if gm.has_succeeded(file, output_dir) == True:
                rd.read_output(file, output_dir)
    
    #create csv of profile 
    for file in os.listdir(output_dir):
        if file.endswith('.dat'):
            if gm.has_succeeded(file, output_dir) == True:
                rd.write_csv(file, output_dir)
        
    #read the parameters from *.dat file
    a_lista = []
    for file in os.listdir(output_dir):
        if file.endswith('.dat'):
            if gm.has_succeeded(file, output_dir) == True:
                a_n = rd.read_params(file, output_dir)
                a_lista.append(a_n)
        
    para_all = np.array(a_lista)
    para_all = para_all[para_all[:,0].argsort()]
    
    #Create csv of parameter
    pd.DataFrame(para_all).to_csv(output_dir+'para_all.csv', sep=",", index=False, 
                                    header=['m_sun', 'x', 'y', 'Pc', 'Tc', 'R', 'L', 'Teff', 'L/L_sun']
                                   )
    return para_all

##### SECTION: Start simulation for different metallicities x and y

sim_mark = full_sim('./sim_x_0.7_y_0.292/', M_list, 0.7, 0.292)

sim_david = full_sim('./sim_x_0.7381_y_0.2485/', M_list, 0.7381, 0.2485)

# Change metallicity x and y by 1%
x_list_d = []
x_list_u = []
y_list_d = []
y_list_u = []
x = 0.7
y = 0.292
for i in np.arange(1, 10, 1):
    x_list_d.append(x*0.99**i)
    x_list_u.append(x*1.01**i)
    y_list_d.append(y*0.99**i)
    y_list_u.append(y*1.01**i)

#Do simulation for fix y=0.292
y = 0.292
for x in x_list_d:
    x_value = str(x)
    full_sim('./sim_' + x_value + '_y_0.292/', M_list, x, y)

#Do simulation for fix x=0.7
x = 0.7 
for y in y_list_u:
    y_value = str(y)
    full_sim('./sim_x_0.7/' + y_value + '/', M_list, x, y)

#Do simulation by decreasing x and increasing y by 1% stepwise 
x = 0.7 
y = 0.292
for x in x_list_d:
    for y in y_list_u:
        y_value = str(y)
        x_value = str(x)
        full_sim('./sim' + x_value + '_' + y_value + '/', M_list, x, y)
