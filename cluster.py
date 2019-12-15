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
import os
import read_data as rd
import generate_models as gm
from scipy.interpolate import interp1d as interp
import pandas as pd

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


##### SECTION: GUESSES FOR THE MODEL
def one_star_guess(M, include_metalicity=False, x=0.7381,y=0.2485):
    '''
    This function produces a guess of values of the stellar model for a single
    star (given its mass) to initiate its simulation. Solar composition is
    assumed in the given values.
    It uses some values that have been got by initial trial-error and
    subsequent recursive interpolation-extrapolation.
    D. Vallés

    Parameters:
    M: mass to be imposed to the star

    Returns: list containing the values of M, p (central), T (central),
    R and L (in solar luminosities).
    '''
    #M_ref = [1,3,15]
    #p_ref = [1.482e17, 1.141e17, 2.769e16]
    #T_ref = [1.442e7, 2.347e7, 3.275e7]
    #R_ref = [6.932e10, 1.276e11, 3.289e11]
    #L_ref = [0.9083, 89.35, 1.960e4]
    #newM_ref = [0.08,0.306,0.533,0.759,0.986,1.212,1.439,1.665,1.892,2.118,2.345,2.571,2.798,3.024,3.251,3.477,3.703,3.93,4.156,4.383,4.609,4.836,5.062,5.289,5.515,5.742,5.968,6.195,6.421,6.647,6.874,7.1,7.327,7.553,7.78,8.006,8.233,8.459,8.686,8.912,9.139,9.365,9.592,9.818,10.044,10.271,10.497,10.724,10.95,11.177,11.403,11.63,11.856,12.083,12.309,12.536,12.762,12.988,13.215,13.441,13.668,13.894,14.121,14.347,14.574,14.8,15.027,15.253,15.48,15.706,15.933,16.159,16.385,16.612,16.838,17.065,17.291,17.518,17.744,17.971,18.197,18.424,18.65,18.877,19.103,19.329,19.556,19.782,20.009,20.235,20.462,20.688,20.915,21.141,21.368,21.594,21.821,22.047,22.274,22.5,23.611,24.722,25.833,26.944,28.0,31.222,34.444,37.667,40.889,44.111,47.333,50.556,53.778,57.0,60.0,70.0,80.0,90.0,100.0,110.0,120.0,130.0]
    #newp_ref = [2.5913e+16,5.465e+16,8.6424e+16,1.2353e+17,1.6988e+17,2.0925e+17,2.1463e+17,2.0247e+17,1.867e+17,1.714e+17,1.5756e+17,1.4535e+17,1.3462e+17,1.252e+17,1.1691e+17,1.0958e+17,1.0307e+17,9.7261e+16,9.2064e+16,8.739e+16,8.3173e+16,7.9351e+16,7.5877e+16,7.2709e+16,6.981e+16,6.7149e+16,6.4702e+16,6.2444e+16,6.0357e+16,5.8422e+16,5.6623e+16,5.495e+16,5.3393e+16,5.1936e+16,5.0572e+16,4.9293e+16,4.8092e+16,4.6963e+16,4.5902e+16,4.49e+16,4.3954e+16,4.306e+16,4.2213e+16,4.1408e+16,4.0646e+16,3.9924e+16,3.9235e+16,3.858e+16,3.7956e+16,3.7361e+16,3.6792e+16,3.625e+16,3.5732e+16,3.5235e+16,3.4761e+16,3.4305e+16,3.3869e+16,3.3442e+16,3.3042e+16,3.2658e+16,3.2289e+16,3.1932e+16,3.1587e+16,3.1255e+16,3.0936e+16,3.0627e+16,3.0329e+16,3.0043e+16,2.9765e+16,2.9496e+16,2.9238e+16,2.8987e+16,2.8744e+16,2.8509e+16,2.8282e+16,2.8061e+16,2.7848e+16,2.764e+16,2.7439e+16,2.7244e+16,2.7055e+16,2.6872e+16,2.6692e+16,2.6518e+16,2.6348e+16,2.6185e+16,2.6024e+16,2.5869e+16,2.5717e+16,2.557e+16,2.5427e+16,2.5287e+16,2.515e+16,2.5017e+16,2.4888e+16,2.4761e+16,2.4638e+16,2.4517e+16,2.44e+16,2.4285e+16,2.3759e+16,2.3288e+16,2.2863e+16,2.248e+16,2.2148e+16,2.1288e+16,2.0628e+16,2.0091e+16,1.9654e+16,1.9289e+16,1.8982e+16,1.8721e+16,1.8496e+16,1.8301e+16,1.8143e+16,1.773e+16,1.7437e+16,1.7225e+16,1.7067e+16,1.6947e+16,1.6855e+16,1.6783e+16]
    #newT_ref = [3601400.0,7140800.0,9772800.0,12226000.0,14877000.0,17511000.0,19333000.0,20503000.0,21371000.0,22080000.0,22690000.0,23233000.0,23724000.0,24176000.0,24594000.0,24985000.0,25352000.0,25698000.0,26026000.0,26337000.0,26633000.0,26915000.0,27185000.0,27443000.0,27691000.0,27929000.0,28158000.0,28378000.0,28591000.0,28796000.0,28993000.0,29185000.0,29372000.0,29552000.0,29726000.0,29896000.0,30060000.0,30220000.0,30374000.0,30525000.0,30671000.0,30814000.0,30953000.0,31090000.0,31222000.0,31351000.0,31478000.0,31601000.0,31722000.0,31839000.0,31955000.0,32068000.0,32177000.0,32285000.0,32391000.0,32494000.0,32595000.0,32694000.0,32792000.0,32889000.0,32982000.0,33074000.0,33165000.0,33252000.0,33339000.0,33426000.0,33510000.0,33593000.0,33674000.0,33757000.0,33832000.0,33910000.0,33986000.0,34061000.0,34135000.0,34207000.0,34279000.0,34349000.0,34418000.0,34486000.0,34554000.0,34620000.0,34685000.0,34750000.0,34813000.0,34876000.0,34938000.0,34998000.0,35058000.0,35117000.0,35176000.0,35234000.0,35291000.0,35347000.0,35403000.0,35457000.0,35511000.0,35565000.0,35618000.0,35670000.0,35917000.0,36151000.0,36372000.0,36582000.0,36772000.0,37298000.0,37767000.0,38181000.0,38554000.0,38892000.0,39200000.0,39482000.0,39743000.0,39985000.0,40196000.0,40812000.0,41323000.0,41760000.0,42139000.0,42474000.0,42772000.0,43041000.0]
    #newR_ref = [15891000000.0,27430000000.0,36770000000.0,50125000000.0,73183000000.0,84394000000.0,87326000000.0,91751000000.0,96810000000.0,102090000000.0,107420000000.0,112730000000.0,117970000000.0,123140000000.0,128220000000.0,133210000000.0,138120000000.0,142940000000.0,147680000000.0,152340000000.0,156920000000.0,161430000000.0,165870000000.0,170240000000.0,174550000000.0,178790000000.0,182970000000.0,187100000000.0,191170000000.0,195180000000.0,199130000000.0,203040000000.0,206930000000.0,210740000000.0,214520000000.0,218250000000.0,221940000000.0,225580000000.0,229170000000.0,232730000000.0,236250000000.0,239740000000.0,243190000000.0,246610000000.0,249980000000.0,253330000000.0,256640000000.0,259920000000.0,263170000000.0,266390000000.0,269580000000.0,272740000000.0,275860000000.0,278960000000.0,282040000000.0,285090000000.0,288110000000.0,291070000000.0,294060000000.0,297020000000.0,299950000000.0,302850000000.0,305730000000.0,308570000000.0,311410000000.0,314220000000.0,317010000000.0,319780000000.0,322530000000.0,325270000000.0,327970000000.0,330660000000.0,333330000000.0,335980000000.0,338620000000.0,341230000000.0,343830000000.0,346410000000.0,348980000000.0,351530000000.0,354060000000.0,356590000000.0,359080000000.0,361560000000.0,364020000000.0,366490000000.0,368920000000.0,371360000000.0,373770000000.0,376170000000.0,378560000000.0,380930000000.0,383280000000.0,385640000000.0,387970000000.0,390290000000.0,392620000000.0,394900000000.0,397200000000.0,399460000000.0,410450000000.0,421190000000.0,431660000000.0,441920000000.0,451470000000.0,479270000000.0,506310000000.0,531840000000.0,556390000000.0,580050000000.0,602830000000.0,624990000000.0,646490000000.0,667450000000.0,686540000000.0,747420000000.0,803990000000.0,857790000000.0,909110000000.0,958260000000.0,1005500000000.0,1051100000000.0]
    #newL_ref = [3.113149665447022e-05,0.007930489601281983,0.07466206543779416,0.3306739627988451,1.130576619657078,2.989509927814587,6.16311114741984,11.033162127669812,18.14261982038613,28.07372232343648,41.40950127434677,58.74893525297766,80.7049178911866,107.86983141660878,140.83156323665284,180.13578646691397,226.51658212545954,280.41419854536935,342.29456231604144,412.8573303779981,492.6063433495771,581.9691989817644,681.3966592828317,791.589454624261,912.8512193311221,1045.442135694148,1190.1453387038553,1347.1005082758036,1516.7010939163386,1699.417162886752,1895.3961755993425,2105.2321804752323,2331.309857418028,2570.987706415226,2824.8799749157074,3095.2804165939788,3380.6483620598156,3681.289736425313,3995.765566188054,4329.123745028265,4678.428536839501,5043.128102663999,5426.252513290146,5829.079788672337,6247.4098790155385,6680.362046792469,7133.456216702232,7605.013686927006,8094.685874176842,8602.01112713024,9132.716996648163,9676.09224577728,10235.286413287924,10816.829893379321,11415.633046188459,12033.722711826847,12673.600118311286,13322.937789769398,14005.544572422317,14706.184154045619,15424.105957495969,16154.741014619049,16916.090502776344,17672.5817784194,18462.903509747837,19288.568602156247,20114.071982009064,20974.90487978542,21847.41206479047,22766.69555330906,23642.85942401745,24581.025650893436,25527.013026612472,26497.20098878961,27485.269519646674,28497.05627341151,29525.686217289687,30577.3609569008,31644.62838835526,32726.533057537137,33837.63488617649,34954.250959859964,36107.71451003529,37273.48496903333,38459.178204535354,39655.18679916798,40888.38902676978,42121.12867540669,43381.04405465888,44658.07512933513,45962.11446660011,47282.453051179604,48618.325846498374,49980.43135782396,51357.04178483559,52747.27156616844,54150.19189534895,55590.42572704037,57042.690309986894,58505.945195000575, 65978.12965139568,73892.43870344739,82224.26499470712,90970.37814779893,99655.20801347682,128233.05826560206,160324.53906900418,195164.13002858264,232755.5256554283,272960.6225513583,315427.82403443614,360246.69071148784,407098.96665027394,455931.9216592222,503152.922989318,671274.2685170481,852511.0168263653,1046164.5510220069,1249971.1805783696,1462850.686999852,1683449.1395670678,1911172.9942272024]
    #M_ref = M_ref + newM_ref
    #p_ref = p_ref + newp_ref
    #T_ref = T_ref + newT_ref
    #R_ref = R_ref + newR_ref
    #L_ref = L_ref + newL_ref

    M_ref = [0.080,0.119,0.177,0.263,0.391,0.581,0.864,1.285,1.911,2.841,4.224,
            6.280,9.338,13.884,20.643,30.693,45.635,67.852,100.885,150.000]
    p_ref = [2.591e+16,2.687e+16,3.227e+16,4.628e+16,7.092e+16,8.606e+16,
            2.029e+17,2.358e+17,1.852e+17,1.233e+17,9.059e+16,6.164e+16,
            4.316e+16,3.195e+16,2.531e+16,2.141e+16,1.914e+16,1.781e+16,
            1.706e+16,1.667e+16]
    T_ref = [3.601e+06,4.215e+06,5.139e+06,6.494e+06,8.311e+06,1.010e+07,
            1.492e+07,1.874e+07,2.143e+07,2.329e+07,2.612e+07,2.846e+07,
            3.080e+07,3.307e+07,3.522e+07,3.722e+07,3.904e+07,4.069e+07,
            4.217e+07,4.353e+07]
    R_ref = [1.589e+10,1.743e+10,2.022e+10,2.498e+10,3.194e+10,3.733e+10,
            7.279e+10,8.998e+10,9.721e+10,1.229e+11,1.491e+11,1.886e+11,
            2.393e+11,3.027e+11,3.805e+11,4.748e+11,5.909e+11,7.347e+11,
            9.135e+11,1.139e+12]
    L_ref = [3.113e-05,5.147e-05,3.144e-04,3.293e-03,2.743e-02,8.627e-02,
            1.417e+00,4.578e+00,1.882e+01,7.459e+01,3.624e+02,1.410e+03,
            4.999e+03,1.612e+04,4.702e+04,1.233e+05,2.928e+05,6.340e+05,
            1.268e+06,2.387e+06]

    # interpolating function
    p_int = interp(M_ref, p_ref, fill_value='extrapolate', kind='cubic')
    T_int = interp(M_ref, T_ref, fill_value='extrapolate',kind='cubic')
    R_int = interp(M_ref, R_ref, fill_value='extrapolate',kind='cubic')
    loglogL_int = interp(np.log(M_ref), np.log(L_ref), fill_value='extrapolate',
                        kind='cubic')

    # interpolated values
    p = p_int(M)
    T = T_int(M)
    R = R_int(M)
    L = np.exp(loglogL_int(np.log(M)))

    # model
    model = [M, p, T, R, L]
    if include_metalicity:
        model = [M, p, T, R, L, x, y]
    return model


def several_stars_guess(Mlist):
    '''
    This function produces a guess of values of the stellar model for several
    stars (given their masses) to initiate their simulation.
    Solar composition is assumed in the given values.
    It uses some values that have been got by initial trial-error and
    subsequent recursive interpolation-extrapolation.
    D. Vallés

    Parameters:
    Mlist: vector of values of the mass to be imposed to each star

    Returns: list containing one list per star; each one of these contains the
    values of M, p (central), T (central), R and L (in solar luminosities).
    '''
    return [one_star_guess(M) for M in Mlist]

##### SECTION: LAUNCHING THE MODEL
def cluster_initial_models(Mclus=10**4,write_csv=False,
                        output_folder='output_cluster', exec_name='zams'):
    '''
    Generates the initial model (cluster a t=0). Gives the overall parameters
    (we don't care about profiles in this part).
    D. Vallés

    Parameters:
    Mclus: mass of the cluster one wants to generate
    write_csv: (default: False) whether one wants to save the values as a csv.
    output_folder: where to save the outputs
    exec_name: name of the executable file for the simulation
    '''
    ## We pick the masses in ZAMS
    #we generate a large number of sampled masses from the Salpeter IMF
    print(' Generating the IMF...')
    random_masses = np.array(list(metropolis(salpeter_xi, 10**6,
                                                uniform_proposal)))
    #and we start picking masses until we have 10**4
    print('Done! \n Filling our ZAMS...')
    masses = [np.random.choice(random_masses)]
    while sum(masses) < Mclus:
        masses.append(np.random.choice(random_masses))
    #we deal the best we can do with the last one:
    #our criterion for choosing if we keep or not the last one is the following:
    #p = (mass we were missing before adding the last one)/(last mass)
    p = (Mclus - sum(masses))/masses[-1] + 1
    if np.random.rand() > p:
        masses.pop()

    print('Done! \n Generating the model guesses...')
    models = several_stars_guess(masses)
    failed=[]
    parameters=[]

    os.system('mkdir ' + output_folder)
    os.system('cp ' + exec_name + ' ./' + output_folder)
    os.chdir(output_folder)

    print('Done! \n Starting the siulations...')
    for idx, model in enumerate(models):
        print('ZAMS: Starting star no. {:04d} of {:04d}'.format(idx, len(models)))
        gm.write_pars_file(filename='custom_pars.pars', mass=model[0], x=0.7381,
                        y=0.2485, p=model[1], T=model[2], R=model[3],
                        L=model[4],
                        output_filename='custom_output.dat')
        gm.launch_model(parameters_filename='custom_pars.pars',
                    exec_name=exec_name)
        if not gm.has_succeeded(output_filename='custom_output.dat'):
            failed.append(idx)
            print('Failed!! :C')
        else:
            parameters.append(rd.read_params('custom_output.dat'))
            totaleps = integrate_nuclear_production('custom_output.dat',
                                                    model[0])
            Hburningrate = how_much_hydrogen_burnt(totaleps)
            parameters[-1].append(Hburningrate)
            EstimatedlifetimeZAMS = estimate_lifetime(0.7381,model[0],
                                                        Hburningrate)
            parameters[-1].append(EstimatedlifetimeZAMS)


    print(np.array(parameters).shape)
    print('The following have failed: {}.'.format(failed))
    missingmass = sum([masses[i] for i in failed])
    if missingmass > Mclus/10**3:
        print('We re-run for the missing mass:')
        parameters.extend(cluster_initial_models(Mclus=missingmass))
        os.system('rm -r ./' + output_folder)
        print(np.array(parameters).shape)

    os.system('rm ./' + exec_name)
    os.system('rm ./' + 'custom_pars.pars')
    os.system('rm ./' + 'custom_output.dat')

    if write_csv:
        values = np.array(parameters)
        column_names = ['m','x','y','p','Tc','R','Lergs','Teff','Lsununits','Hburningrate','EstimatedlifetimeZAMS']
        pd.DataFrame(values).to_csv('initial.csv',
                                sep=',',index=False, header=column_names)
    os.chdir('..')

    return parameters

def integrate_nuclear_production(filename,M):
    '''
    Integrates the nuclear production through the entire star, from the
    profiles data.
    D. Vallés

    Parameters:
    filename: filename of the output (assumed to be in the same path).
    M: total mass of the star (Msun)

    Returns:
    totaleps: nuclear burning rate (erg/s)
    '''
    _, x, _, _, _, _, _, eps, _, _, _, _, _, _ = rd.read_output(filename)
    # determining the mass coordinate at grid points
    M = M*2e33 #to cgs
    mr=[M*(1-xi) for xi in x]
    # undoing the logarithm
    eps = [10**(i) for i in eps] #erg/g/s produced in the star
    # we add, for each thin shell:
    mrdif = [mr[i+1]-mr[i] for i in range(len(mr)-1)]
    epsmean = [0.5*(eps[i+1]+eps[i]) for i in range(len(eps)-1)]
    totaleps = sum([mrdif[i]*epsmean[i] for i in range(len(mrdif))])

    return totaleps

def how_much_hydrogen_burnt(totaleps):
    '''
    From the total nuclear burning rate, finds the amount of hydrogen (in Solar
    masses) that is being burnt per unit of time (year)

    Parameters:
    totaleps: total nuclear burning rate

    Returns:
    Hburningrate: hydrogen being burnt per unit of time (in solar masses per
    year)
    '''
    one_reaction_energy_MeV = 26.7
    one_reaction_energy = one_reaction_energy_MeV * 1.602*10**(-6)
    reaction_rate = totaleps/one_reaction_energy
    Hburningrate_cgs = 4*reaction_rate/6.023e23 #4 grams each mole
    Hburningrate = Hburningrate_cgs*86400*365.25/2e33# solar mass / year
    return Hburningrate

def estimate_lifetime(x,M,Hburningrate):
    '''
    Computes the estimated lifetime by assuming constant nuclear burning rate.
    The lifetime is obtained by finding when 10% of the initial hydrogen will
    be burnt.
    D. Vallés

    Parameters:
    x: fraction of hydrogen (in mass)
    M: total mass
    Hburningrate: solar masses per year of hydrogen being consumed

    Returns:
    Estimated lifetime
    '''
    mH = M*x
    return 0.1*mH/Hburningrate


##### SECTION: TIME EVOLUTION (FULL)
## this is not going to be implemented (at least for now)
def update_composition(x,y,M,Hburningrate,timestep):
    '''
    Updates the composition (x and y) after each timestep
    D. Vallés

    Parameters:
    x: fraction of hydrogen (in mass)
    y: fraction of helium (in mass)
    M: total mass
    Hburningrate: solar masses per year of hydrogen being consumed
    timestep: timestep of the integration

    Returns:
    New x and y
    '''
    Hburnt = Hburningrate/M*timestep
    x = x - Hburnt
    y = y + Hburnt
    return x,y


def time_evolution(parameters, timestep, itnum, write_csv = False,output_folder='output_cluster',exec_name='zams'):
    '''
    Work in progress!!!
    Returns the time-evolved models, considering the change in composition
    due to nuclear burning.
    D. Vallés
    '''
    #we update the compositions
    print('Starting it. {}. Updating compositions...'.format(itnum))
    for idx in range(len(parameters)):
        M,x,y,Hburningrate = parameters[idx][0],parameters[idx][1],parameters[idx][2],parameters[idx][9]
        parameters[idx][1],parameters[idx][2] = update_composition(x,y,M,Hburningrate,timestep)

    failed=[]
    newparameters=[]

    os.system('mkdir ' + output_folder)
    os.system('cp ' + exec_name + ' ./' + output_folder)
    os.chdir(output_folder)

    print('Done! \n Starting the siulations...')
    for idx, model in enumerate(parameters):
        print('It. num. {}: Starting star no. {:04d} of {:04d}'.format(itnum,
                                                            idx,len(parameters)))
        gm.write_pars_file(filename='custom_pars.pars', mass=model[0], x=model[1],
                        y=model[2], p=model[3], T=model[4], R=model[5],
                        L=model[8],
                        output_filename='custom_output.dat')
        gm.launch_model(parameters_filename='custom_pars.pars',
                    exec_name=exec_name)
        if not gm.has_succeeded(output_filename='custom_output.dat'):
            failed.append(idx)
            print('Failed!! :C')
        else:
            newparameters.append(rd.read_params('custom_output.dat'))
            totaleps = integrate_nuclear_production('custom_output.dat',
                                                    newparameters[-1][0])
            Hburningrate = how_much_hydrogen_burnt(totaleps)
            newparameters[-1].append(Hburningrate)
            EstimatedlifetimeZAMS = estimate_lifetime(newparameters[-1][1],newparameters[-1][0],
                                                        Hburningrate)
            parameters[-1].append(EstimatedlifetimeZAMS)


    print(np.array(newparameters).shape)
    print('The following have failed: {}.'.format(failed))

    os.system('rm ./' + exec_name)
    os.system('rm ./' + 'custom_pars.pars')
    os.system('rm ./' + 'custom_output.dat')

    if write_csv:
        values = np.array(newparameters)
        column_names = ['m','x','y','p','Tc','R','Lergs','Teff','Lsununits','Hburningrate','EstimatedlifetimeZAMS']
        pd.DataFrame(values).to_csv('iteration_{}.csv'.format(itnum),
                                sep=',',index=False, header=column_names)
    os.chdir('..')

    return newparameters

#### SECTION: Cluster observables
def cluster_luminosity(parameters):
    '''
    Computes the total luminosity of the cluster, by adding the individual
    stars.
    D. Vallés

    Parameters:
    parameters: list containing each stars' parameters (as returned by
    cluster_initial_models)

    Returns:
    Total luminosity (in solar luminosities)
    '''
    return sum([model[8] for model in parameters])

def planck_function(T,lamb):
    '''
    Computes the planck function (emissive pectral power) for howmanypoints
    evenly spaced between lambdamin and lambdamax.
    D. Vallés

    Parameters:
    T: photospheric temperature (in K)
    lamb: np array of values of lambda where to compute the function (in m!)

    Returns:
    I_cgs: Planck's function computed at the given lambdas. Units:
            erg*s^{-1}*cm^{-3}*sr^{-1}
    '''
    C1 = 1.2006e-16
    C2 = 1.4385e-2
    I_isu = C1 / (lamb**5*(np.exp(C2/(lamb*T))-1))
    # isu = W*m^{-3}*sr^{-1}
    # cgs = erg*s^{-1}*cm^{-3}*sr^{-1}
    # value_in_cgs = 10 * value_in_isu
    I_cgs = I_isu * 10
    return I_cgs

def cluster_spectrum(parameters, lambdamin=200, lambdamax=1000, howmanypoints=200):
    '''
    Computes the total spectrum for a cluster.
    D. Vallés

    Parameters:
    parameters: list containing each stars' parameters (as returned by
    cluster_initial_models)
    lambdamin, lambdamax: lower and upper bounds for lambda, ¡in nm!
    howmanypoints: values where to sample the planck function

    Returns:
    lamb: values of lambda where the function has been computed (in nm!)
    I_cgs: Planck's function computed at the given lambdas
    '''
    #one has to get the spectral radiance (erg*s^{-1}*cm^{-3}*sr^{-1}) for
    #each star and integrate (multiply: isotropy) over its area, to get
    #spectral luminosity per unit of solid angle

    #temp: model[7]
    #radius: model[5]
    lambdamax = lambdamax/10**9
    lambdamin = lambdamin/10**9
    lamb = np.linspace(lambdamin,lambdamax,howmanypoints)
    return lamb*10**9,sum([planck_function(model[7], lamb)*4*np.pi*model[5]**2 for model in parameters])


def kill_stars(parameters, t):
    '''
    Returns the parameters file, having removed all the stars whose
    estimated lifetime is less than the evolution time.

    Parameters:
    parameters: list containing each stars' parameters (as returned by
    cluster_initial_models)
    t: time (in yr) to evolve

    Returns:
    New parameters having removed the dead stars
    '''
    dying=[]
    for i in range(len(parameters)):
        if parameters[i][10] < t:
            dying.append(i)
    if len(dying)>0:
        #print(dying)
        dying.reverse()
        newparameters = [parameters[i] for i in range(len(parameters)) if i not in dying]
        return newparameters
    else:
        return parameters
