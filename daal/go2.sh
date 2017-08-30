pushd /data/intel/daal/examples/cpp/mpi
make libintel64 sample=implicit_als_csr_distributed_mpi2 compiler=gnu threading=parallel mode=build &&
{
    popd
    nohup bash -c 'time mpirun -n 4 /data/intel/daal/examples/cpp/mpi/_results/gnu_intel64_a/implicit_als_csr_distributed_mpi2.exe' >  nohup.out.als_99999  &
    tail -f nohup.out.als_99999
}

