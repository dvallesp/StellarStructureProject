# StellarStructureProject

Here we can add all the relevant stuff for the project. Please include documentation for your scripts and comment the commits.

### generate_models.py file

It's advised to copy this file to your working directory, where you must also have the simulation executable. In a python/ipython terminal/notebook, you can load the libraries as "import generate_models as (whatever name you want to give it)". The docstrings in each function are pretty self-explanatory. Nevertheless, some important caveats: 

- reasonable_MS_guess() generates a guess for initial parameters by interpolation from the ones given in the manual between 1 and 15 solar masses. Beyond that interval, it uses linear extrapolation (there are not many datapoints to do anything more complex, indeed). I've tested that extrapolation to generate proper models in the range from 0.4 to 18.2 solar masses. Beyond, it fails. Feel free to change your local reasonable_MS_guess() function if you want to cover a wider mass interval.

- has_succeeded() checks if the model has converged or not. Failure signature is assumed to be a file with only 4 lines. If more non-converging possibilities are found with different signature, they will have to be added.

- launch_model() and pipeline_model_zams() will only work in Unix, not in Windows (although it may be rather easy to change the commands to make it work in Windows)

- do_pipeline_models_zams_work() can be used to assess if another guess function works in certain mass interval

- pipeline_models_zams(): Do execute this function in your main folder, where your executable is. Be sure to specify an output_folder name. If not, it may overwrite existing data from a previous run and you may lose it.

