# run on IGM machine
now=`date`
echo "Start: $now"

python /home/sstie/MgII_forest/compute_model_grid_new.py \
--nproc 5 --fwhm 90 --samp 3 \
--vmin 10 --vmax 2000 --dv 100 \
--ncopy 1000 --seed 9251761 \
--logZmin -6.0 --logZmax -2.0 --nlogZ 3 \
--cgm_masking

now=`date`
echo "Finish: $now"