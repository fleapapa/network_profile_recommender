DD=/data/.github/network_profile_recommender/data
pushd $DD

INF=etl2M.csv

# because github limits upload file size to 50MB, etl2M.csv might be cut in half
[ -f $INF ] ||
{
	([ -f $INF.1 ] && [ -f $INF.2 ]) || 
	{
		echo "$INF files not found!"
		exit 1
	}
	cat $INF.1 $INF.2 > $INF
}

#
NL=`wc -l $INF|cut -d' ' -f1`
NN=`expr \( $NL - 10000 \) / 4`

[ $INF -nt xaa ] && \
    split -l $NN $INF

[ xaa -nt kmean_train_xaa.csv ] &&
{
    # split/cut INF file for 4 MPI nodes 
    for f in xaa xab xac xad
    do
        # - 4 files for the 4 MPI nodes to train K-Mean (no test data needed)
        # see [3] VectorAssembler in network_spark_ml_kmeams_als_HIPA.ipynb
        cut -d, -f2,4,7,10,13,16,19 $f > kmean_train_$f.csv
    
        # - 4 files for 4 MPI nodes to generate CSR files based on K-Mean clusters
        # see [6] df_merged in network_spark_ml_kmeams_als_HIPA.ipynb
        cut -d, -f1,19 $f > sta_phyr_$f.csv
    done

#	cut -d, -f1-6 xae > testx.csv
#	cut -d, -f7   xae > testy.csv
}

popd
pushd /data/intel/daal/examples/cpp/mpi
make libintel64 sample=kmeans_dense_distributed_mpi2 compiler=gnu threading=parallel mode=build &&
{
    popd
    nohup bash -c 'time mpirun -n 4 /data/intel/daal/examples/cpp/mpi/_results/gnu_intel64_a/kmeans_dense_distributed_mpi2.exe' > nohup.out.kmeans_99999 &
    tail -f nohup.out.kmeans_99999
}
