# Abstract

This directory contain a C++ version of Network Performance Profile Recommender based 
on Intel DAAL library.

The implementation is divided in three files, which together perform the following tasks:
1. Use KMeans Clustering algorithm to group network performance metrics to K profiles (K=99999)
  and writes the result to a CSV file.
2. Transform the CSV file in task 1 to a CSR format.
3. Use the training data samples in task 2 to train an Implicit ALS model. 
4. Use my Heuristic Interleaved Parameter Alternator algorithn to do ALS model selection.

# Files

* [kmeans_dense_distributed_mpi2.cpp](https://github.com/fleapapa/network_profile_recommender/blob/master/daal/kmeans_dense_distributed_mpi2.cpp) - performs task 1 above
* [convert_kmean_csv_to_als_csr.ipynb](https://github.com/fleapapa/network_profile_recommender/blob/master/daal/convert_kmean_csv_to_als_csr.ipynb) - performs task 2 above
* [implicit_als_csr_distributed_mpi2.cpp](https://github.com/fleapapa/network_profile_recommender/blob/master/daal/implicit_als_csr_distributed_mpi2.cpp) - perform task 3 and 4 above

* [go.sh](https://github.com/fleapapa/network_profile_recommender/blob/master/daal/go.sh) - shell script to split data samples, build kmeans_dense_distributed_mpi2.exe and run task 1.
* [go2.sh](https://github.com/fleapapa/network_profile_recommender/blob/master/daal/go2.sh) - shell script to build implicit_als_csr_distributed_mpi2.exe and run 3 and 4.


# Requirements

* GNU C++11
* Intel DAAL library
* MPICH
* Python 2.7
* Jupyter Notebook

For multiple distributed MPI nodes to access the files in /data, a network file system like NFS or Samba,
or a distributed file system like HDFS, GlusterFS or CephFS is required.


# Get Started

For task 1, see go.sh for a sample script about how to prepare CSV data files and how to build/run kmeans_dense_distributed_mpi2.exe.

For task 2, use Jupyter Notebook.

For task 3 and 4, see go2.sh for a sample script about how to build/run implicit_als_csr_distributed_mpi2.exe.

go.sh and go2.sh assume that the project is hosted in directory /data/.github/network_profile_recommender, Intel DAAL has been installed in directory /data/intel/daal and there is a symbolic link in
/data/intel/daal/examples/cpp/mpi/source pointing to each of the two C++ files respectively. 

For example, here is my setup:

> $ ll /data/intel/daal/examples/cpp/mpi/sources/ 
> 
> total 332
> 
> lrwxrwxrwx 1 ubuntu ubuntu    93 Aug 18 18:35 implicit_als_csr_distributed_mpi2.cpp -> /data/.github/network_profile_recommender/daal/implicit_als_csr_distributed_mpi2.cpp
> 
> lrwxrwxrwx 1 ubuntu ubuntu    89 Aug 18 18:34 kmeans_dense_distributed_mpi2.cpp -> /data/.github/network_profile_recommender/daal/kmeans_dense_distributed_mpi2.cpp


# Test

A test was conducted on an AWS EC2 of r4.xlarge type, based on the 2M data samples with a hyper-parameter 'grid' as below:

> alpha = [55 ... 60] in tasks of 0.5
> 
> lambda = [0.25 ... 0.35] in tasks of 0.05
> 
> factors = [85 ... 95] in tasks of 1
> 
> iterations = [3 ... 5] in tasks of 1


Result of kmeans_dense_distributed_mpi2.exe:

> First 5 dimensions of centroids (root):
> 
> 1.000     -42.111   95.444    0.000     2.111     2.667     67.000    
> 
> 1.000     -61.762   95.524    0.762     1.190     10.905    33.048    
> 
> 1.000     -67.145   96.673    0.000     1.018     7.327     12.855    
> 
> 1.000     -77.000   96.368    0.947     1.053     5.421     80.105    
> 
> 1.000     -68.200   74.000    16.450    1.450     2.550     80.600    
>
> First 5 dimensions of centroids:
> 
> 1.000     -42.111   95.444    0.000     2.111     2.667     67.000    
> 
> 1.000     -61.762   95.524    0.762     1.190     10.905    33.048    
> 
> 1.000     -67.145   96.673    0.000     1.018     7.327     12.855    
> 
> 1.000     -77.000   96.368    0.947     1.053     5.421     80.105    
> 
> 1.000     -68.200   74.000    16.450    1.450     2.550     80.600    
>
> ... 
> 
> First 5 assignment:
> 
> 33772.000 
> 
> 13811.000 
> 
> 26463.000 
> 
> 71986.000 
> 
> 47390.000 
> 
> First 5 assignment:
> 
> 19299.000 
> 
> 6896.000  
> 
> 8109.000  
> 
> 18378.000 
> 
> 37960.000 
> 
> ...
> 
> real	141m46.587s
> 
> user	559m52.132s
> 
> sys	0m38.576s
> 
> (see nohup.out.kmeans_99999 for full log)

Result of implicit_als_csr_distributed_mpi2.exe:

> calcs w/o hipa: 50020
> 
> calcs w/  hipa: 1987
> 
> calc hits     : 108
> 
> calc saving   : 96%
> 
> top 10 best parameter sets:
> 
>  0: rmse 0.22453, nfactor 61, alpha 65.0, lambda 0.50, maxIterations 2
> 
>  1: rmse 0.22458, nfactor 61, alpha 64.5, lambda 0.50, maxIterations 2
> 
>  2: rmse 0.22473, nfactor 61, alpha 65.0, lambda 0.45, maxIterations 2
> 
>  3: rmse 0.22477, nfactor 61, alpha 64.5, lambda 0.45, maxIterations 2
> 
>  4: rmse 0.22546, nfactor 60, alpha 65.0, lambda 0.50, maxIterations 2
> 
>  5: rmse 0.22552, nfactor 60, alpha 64.5, lambda 0.50, maxIterations 2
> 
>  6: rmse 0.22559, nfactor 60, alpha 65.0, lambda 0.45, maxIterations 2
> 
>  7: rmse 0.22565, nfactor 60, alpha 64.5, lambda 0.45, maxIterations 2
> 
>  8: rmse 0.22593, nfactor 65, alpha 65.0, lambda 0.50, maxIterations 2
> 
>  9: rmse 0.22595, nfactor 65, alpha 64.5, lambda 0.50, maxIterations 2
> 
> real	1758m21.353s
> 
> user	5131m23.660s
> 
> sys	1573m4.184s
> (see nohup.out.als_99999 for full log)

# Contact

fleapapa@gmail.com


# License

This project is released under the MIT License.
