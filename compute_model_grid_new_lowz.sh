# run on IGM machine
now=`date`
echo "Start: $now"

python /home/sstie/MgII_forest/compute_model_grid_new.py \
--nproc 10 --fwhm 90 --samp 3 \
--vmin 10 --vmax 3500 --dv 100 \
--ncopy 1000 --seed 9251761 \
--logZmin -6.0 --logZmax -2.0 --nlogZ 201 \
--cgm_masking --lowz_bin

now=`date`
echo "Finish: $now"

# dlogZ = 0.02 for np.linspace(-6.0, -2.0, 201) as above
# nhi x nlogZ = 51 x 201 = 10251 models
# nohup nice -n 19 ./test_compute_model.sh > run.log &