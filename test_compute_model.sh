# run on IGM machine
now=`date`
echo "Start: $now"

python /home/sstie/MgII_forest/compute_model_grid.py \
--nproc 5 --fwhm 83 --samp 1 --nqsos 20 --delta_z 1.0 \
--vmin 10 --vmax 2000 --dv 100 \
--ncovar 1000000 --nmock 500 --seed 9251761 \
--logZmin -6.0 --logZmax -2.0 --nlogZ 3

now=`date`
echo "Finish: $now"