/* file: kmeans_dense_distributed_mpi.cpp */
//==============================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright 2017 Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// =============================================================

/*
!  Content:
!    C++ sample of K-Means clustering in the distributed processing mode
!******************************************************************************/

/**
 * <a name="DAAL-SAMPLE-CPP-KMEANS_DENSE_DISTRIBUTED"></a>
 * \example kmeans_dense_distributed_mpi.cpp
 */
/*
  Copyright 2017 Peter Pan (pertaining to the following code changes)
	1) applied to bigger training data and much more clusters
	2) root broadcasts centroids to all nodes to compute cluster assignments
	3) patch profile ids to ground truth (aka. alsTrainGroundTruthFileNames)
*/
#include <unistd.h>
#include <mpi.h>
#include "daal.h"
#include "service.h"
#include "stdio.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

typedef std::vector<byte> ByteBuffer;

/* K-Means algorithm parameters */
/* make the same as Spark ML K-Mean MaxIter */
const size_t nClusters   = 99999;	//99999
const size_t nIterations = 20;		//20
const size_t nBlocks     = 4;		//MPI nodes #

/* Input data set parameters */
#define DDIR "../data/"	//based on my working dir /data/.github/network_profile_recommender/daal

const string kmeanTrainDatasetFileNames[nBlocks] =
{
    DDIR"/kmean_train_xaa.csv",
    DDIR"/kmean_train_xab.csv",
    DDIR"/kmean_train_xac.csv",
    DDIR"/kmean_train_xad.csv",
};

/* truth used in ALS (not K-Mean) */
const string alsTrainGroundTruthFileNames[nBlocks] =
{
    DDIR"/sta_phyr_xaa.csv",
    DDIR"/sta_phyr_xab.csv",
    DDIR"/sta_phyr_xab.csv",
    DDIR"/sta_phyr_xad.csv",
};

static NumericTablePtr assignments[nBlocks];

#define mpi_root 0

NumericTablePtr loadData(const string csvFileName)
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(csvFileName, DataSource::doAllocateNumericTable, DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();
    return dataSource.getNumericTable();
}

NumericTablePtr init(int rankId, const NumericTablePtr& pData);
NumericTablePtr compute(int rankId, const NumericTablePtr& pData, const NumericTablePtr& initialCentroids);

int main(int argc, char *argv[])
{
	//init MPI
    int rankId, comm_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);

	//each node loads its share of data
    NumericTablePtr pData = loadData(kmeanTrainDatasetFileNames[rankId]);
    NumericTablePtr centroids = init(rankId, pData);

	//compute for centroids
    for(size_t it = 0; it < nIterations; it++)
    {
		printf("iteration %d\n", it + 1);
        centroids = compute(rankId, pData, centroids);
	}

    /* Print the clusterization results */
    if(rankId == mpi_root)
        printNumericTable(centroids, "First 5 dimensions of centroids (root):", 5, 10);

	/* ppan+
	 * root broadcast centroids and all nodes calculate cluster assignments
	 */
	size_t length;
	InputDataArchive inputDataArch;

	// 1. root serializes centroids to learn needed buffer size
	if (mpi_root == rankId)
	{
		centroids->serialize(inputDataArch);
		length = inputDataArch.getSizeOfArchive();
	}
	MPI_Bcast(&length, sizeof(length), MPI_CHAR, mpi_root, MPI_COMM_WORLD);

	// 2. root broadcasts centroids to other nodes
	daal::byte *buffer = new byte[length];
	if (mpi_root == rankId)
		inputDataArch.copyArchiveToArray(buffer, length);
	MPI_Bcast(buffer, length, MPI_CHAR, mpi_root, MPI_COMM_WORLD);

	// 3. all deserializes into a new set of centroids
	OutputDataArchive outputDataArch(buffer, length);
	centroids = NumericTablePtr(new HomogenNumericTable<double>());
	centroids->deserialize(outputDataArch);
	delete[] buffer;

	// 4. non roots dump the same centroids in sequence
	if (rankId != mpi_root)
	{
		sleep(rankId);
		printNumericTable(centroids, "First 5 dimensions of centroids:", 5, 10);
	}

	// 5. all nodes calculate cluster assignments
	kmeans::Batch<> localAlgorithm(nClusters, 0);
	localAlgorithm.input.set(kmeans::data, pData);
	localAlgorithm.input.set(kmeans::inputCentroids, centroids);
	localAlgorithm.compute();
	auto assignments = localAlgorithm.getResult()->get(kmeans::assignments);
	sleep(rankId);
	printNumericTable(assignments, "First 5 assignment:", 5);

	// 6. patch 'profile' column to files sta_phyr_xa?.csv for next step - ALS
	NumericTablePtr sta_phyr = loadData(alsTrainGroundTruthFileNames[rankId]);
	int nr_assignments = assignments->getNumberOfRows();
	int nc_assignments = assignments->getNumberOfColumns();
	int nr_sta_phyr    = sta_phyr   ->getNumberOfRows();
	int nc_sta_phyr    = sta_phyr   ->getNumberOfColumns();
	if (nr_assignments != nr_sta_phyr)
		fprintf(stderr, "%s: row numbers unmatch: %d vs %d\n", __FUNCTION__, nr_assignments, nr_sta_phyr);

	BlockDescriptor<double> block_assignments;
	BlockDescriptor<double> block_sta_phyr;
	assignments->getBlockOfRows(0, nr_assignments, readOnly, block_assignments);
	sta_phyr   ->getBlockOfRows(0, nr_sta_phyr,    readOnly, block_sta_phyr   );
	auto array_assignments = block_assignments.getBlockPtr();
	auto array_sta_phyr    = block_sta_phyr   .getBlockPtr();

	FILE* fp = fopen(alsTrainGroundTruthFileNames[rankId].c_str(), "w");
	for (int r = 0; r < std::min(nr_assignments, nr_sta_phyr); ++r)
		fprintf(fp, "%g,%g,%g\n", array_sta_phyr[r * nc_sta_phyr + 0], array_sta_phyr[r * nc_sta_phyr + 1], array_assignments[r * nc_assignments + 0]);
	fclose(fp);

    MPI_Finalize();
    return 0;
}

NumericTablePtr init(int rankId, const NumericTablePtr& pData)
{
    const bool isRoot = (rankId == mpi_root);

    const size_t nVectorsInBlock = pData->getNumberOfRows();

    /* Create an algorithm to compute k-means on local nodes */
    kmeans::init::Distributed<step1Local, double, kmeans::init::randomDense> localInit(nClusters, nBlocks * nVectorsInBlock, rankId * nVectorsInBlock);

    /* Set the input data set to the algorithm */
    localInit.input.set(kmeans::init::data, pData);

    /* Compute k-means */
    localInit.compute();

    /* Serialize partial results required by step 2 */
    InputDataArchive dataArch;
    localInit.getPartialResult()->serialize(dataArch);
    const int perNodeArchLength = (int)dataArch.getSizeOfArchive();

    int aPerNodeArchLength[nBlocks];
    /* Transfer archive length to the step 2 on the root node */
    MPI_Gather(&perNodeArchLength, sizeof(int), MPI_CHAR, isRoot ? &aPerNodeArchLength[0] : NULL,
        sizeof(int), MPI_CHAR, mpi_root, MPI_COMM_WORLD);

    ByteBuffer serializedData;
    /* Calculate total archive length */
    int totalArchLength = 0;
    int displs[nBlocks];
    if(isRoot)
    {
        for(size_t i = 0; i < nBlocks; ++i)
        {
            displs[i] = totalArchLength;
            totalArchLength += aPerNodeArchLength[i];
        }
        serializedData.resize(totalArchLength);
    }

    ByteBuffer nodeResults(perNodeArchLength);
    dataArch.copyArchiveToArray(&nodeResults[0], perNodeArchLength);

    /* Transfer partial results to step 2 on the root node */
    MPI_Gatherv(&nodeResults[0], perNodeArchLength, MPI_CHAR, serializedData.size() ? &serializedData[0] : NULL,
        aPerNodeArchLength, displs, MPI_CHAR, mpi_root, MPI_COMM_WORLD);

    if(isRoot)
    {
        /* Create an algorithm to compute k-means on the master node */
        kmeans::init::Distributed<step2Master, double, kmeans::init::randomDense> masterInit(nClusters);
        for(size_t i = 0, shift = 0; i < nBlocks; shift += aPerNodeArchLength[i], ++i)
        {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(&serializedData[shift], aPerNodeArchLength[i]);

            services::SharedPtr<kmeans::init::PartialResult> dataForStep2FromStep1(new kmeans::init::PartialResult());
            dataForStep2FromStep1->deserialize(dataArch);

            /* Set local partial results as input for the master-node algorithm */
            masterInit.input.add(kmeans::init::partialResults, dataForStep2FromStep1);
        }

        /* Merge and finalizeCompute k-means on the master node */
        masterInit.compute();
        masterInit.finalizeCompute();

        return masterInit.getResult()->get(kmeans::init::centroids);
    }
    return NumericTablePtr();
}

NumericTablePtr compute(int rankId, const NumericTablePtr& pData, const NumericTablePtr& initialCentroids)
{
    const bool isRoot = (rankId == mpi_root);
    size_t CentroidsArchLength = 0;
    InputDataArchive inputArch;
    if(isRoot)
    {
        /*Retrieve the algorithm results and serialize them */
        initialCentroids->serialize(inputArch);
        CentroidsArchLength = inputArch.getSizeOfArchive();
    }

    /* Get partial results from the root node */
    MPI_Bcast(&CentroidsArchLength, sizeof(size_t), MPI_CHAR, mpi_root, MPI_COMM_WORLD);

    ByteBuffer nodeCentroids(CentroidsArchLength);
    if(isRoot)
        inputArch.copyArchiveToArray(&nodeCentroids[0], CentroidsArchLength);

    MPI_Bcast(&nodeCentroids[0], CentroidsArchLength, MPI_CHAR, mpi_root, MPI_COMM_WORLD);

    /* Deserialize centroids data */
    OutputDataArchive outArch(nodeCentroids.size() ? &nodeCentroids[0] : NULL, CentroidsArchLength);

    NumericTablePtr centroids(new HomogenNumericTable<double>());

    centroids->deserialize(outArch);

    /* Create an algorithm to compute k-means on local nodes */
    kmeans::Distributed<step1Local> localAlgorithm(nClusters);

    /* Set the input data set to the algorithm */
    localAlgorithm.input.set(kmeans::data, pData);
    localAlgorithm.input.set(kmeans::inputCentroids, centroids);

    /* Compute k-means */
    localAlgorithm.compute();

    /* Serialize partial results required by step 2 */
    InputDataArchive dataArch;
    localAlgorithm.getPartialResult()->serialize(dataArch);
    size_t perNodeArchLength = dataArch.getSizeOfArchive();
    ByteBuffer serializedData;

    /* Serialized data is of equal size on each node if each node called compute() equal number of times */
    if(isRoot)
        serializedData.resize(perNodeArchLength * nBlocks);

    ByteBuffer nodeResults(perNodeArchLength);
    dataArch.copyArchiveToArray(&nodeResults[0], perNodeArchLength);

    /* Transfer partial results to step 2 on the root node */
    MPI_Gather(&nodeResults[0], perNodeArchLength, MPI_CHAR, serializedData.size() ? &serializedData[0] : NULL,
        perNodeArchLength, MPI_CHAR, mpi_root, MPI_COMM_WORLD);

    if(isRoot)
    {
        /* Create an algorithm to compute k-means on the master node */
        kmeans::Distributed<step2Master> masterAlgorithm(nClusters);

        for(size_t i = 0; i < nBlocks; i++)
        {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(&serializedData[perNodeArchLength * i], perNodeArchLength);

            services::SharedPtr<kmeans::PartialResult> dataForStep2FromStep1(new kmeans::PartialResult());
            dataForStep2FromStep1->deserialize(dataArch);

            /* Set local partial results as input for the master-node algorithm */
            masterAlgorithm.input.add(kmeans::partialResults, dataForStep2FromStep1);
        }

        /* Merge and finalizeCompute k-means on the master node */
        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        /* Retrieve the algorithm results */
        return masterAlgorithm.getResult()->get(kmeans::centroids);
    }
    return NumericTablePtr();
}
