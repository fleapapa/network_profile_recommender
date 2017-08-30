# Abstract

This project builds a recommender system that can learn from historical observed network metrics and
predict network performance as well as recommend network profiles for better performance and easier 
network management. 

# Background

There are many factors that can affect performance of a network. Factors can be internal or external
to network endpoints or network routers/switches. Interactions among various factors are complicated. 
It is hard to well define the relationship among the factors mathematically. How to manage performance
of a network in a predictable manner thus has been a big challenge to vendors of network equipments.  

# Introduction

The idea behind the recommender is simple. It uses a unsupervised machine learning algorithm 
to cluster similar network performance metrics to a number of "performance profiles" and uses a 
**[Collaborative Filtering](https://en.wikipedia.org/wiki/Collaborative_filtering)** algorithm to learn latent factors betwwen the profiles and network 
endpoints, then to predict and recommend missing 'users-items' or 'endpoints-profiles' entries.

Training of the recommender system take place in two phases:
* The first phase uses **[KMeans Clustering](https://en.wikipedia.org/wiki/K-means_clustering)** to group network peformance metrics observed on thousands of network endpoints to K performance profiles (K = 99999)

* The second phase trains an **[Alternate Least Square (ALS)](https://bugra.github.io/work/notes/2014-04-19/alternating-least-squares-method-for-collaborative-filtering/)** model using the same network data set 
as in the first phase by mapping network endpoints to *'users'*, mapping network performance profiles 
to *'items'* and mapping observed effective network throughput (eg., PHY rates) to *'ratings'*.


# Getting Started

The project has two different implementations: 

* [One is implemented in Python based on Spark PySpark and Spark ML](https://github.com/fleapapa/network_profile_recommender/tree/master/spark)
* [Another is implemented in C++ based on Intel Data Analytics Acceleration Library (DAAL) and MPI for scalable distributed analytics](https://github.com/fleapapa/network_profile_recommender/tree/master/daal) 


# Requirements

* The Python implementation is provided in a Jupyter notebook. It requires Jupyter Notebook.
* The C++ implementations requires GNU C++11, MPICH library (or OpenMPI or Intel MPI) and Intel DAAL.

Both implementations uses the same set of data samples in [etl2M.csv.*](https://github.com/fleapapa/network_profile_recommender/tree/master/data). To run the same code on multiple 
distributed nodes, a network file system like NFS or Samba or a distributed file system like HDFS,
GlusterFS or CephFS is required to access networked or distributed data files.

# Contact

fleapapa@gmail.com


# License

This project is released under the MIT License.
