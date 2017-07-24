model=../stanmodels/gamma2
modelname=gamma2
N=4
Nsamples=500
seed=12344

for ((i = 1; i <= N; i++));
do
    $model sample num_samples=$Nsamples data file=rdata.dump random seed=$seed id=$i  output file=$modelname-$i.csv &
done
wait