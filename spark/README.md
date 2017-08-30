# Abstract

This directory contains a PySpark version of Network Performance Profile Recommender.


# File

* [network_spark_ml_kmeams_als_HIPA.ipynb](https://github.com/fleapapa/network_profile_recommender/blob/master/spark/network_spark_ml_kmeams_als_HIPA.ipynb) - contains Python code that 
    * uses KMeans Clustering algorithm to group network performance metrics to K profiles (K=99999)
    * uses Alternate Least Square algorithm to train and predict network performance profiles
    * uses my Heuristic Interleaved Parameter Alternator for ALS model selection (cross verification)

# Requirement

* Python 2.7
* PySpark 2.2.0+
* Jupyter Notebook

# Contact

fleapapa@gmail.com


# License

This project is released under the MIT License.
