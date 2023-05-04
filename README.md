## MgII_forest: measuring the auto-correlation of the MgII forest from ground-based spectra of QSOs

### Main scripts: <br>
1. ```mutils.py``` - custom utility functions to initialize dataset <br>
2. ```compute_cf_data.py``` - functions to measure the auto-correlation from one and multiple sightlines <br>
3. ```mask_cgm_pdf.py``` - functions to mask CGM absorbers <br>
4. ```compute_model_grid_8qso_fast.py``` - compute the forward models and their covariances for a grid of a models provided by the Nyx simulations in Hennawi et al. (2021) <br>
5. ```mcmc_inference.py``` - compute the data likelihood, sample the posterior PDF using emcee, and infer the model parameters <br>

