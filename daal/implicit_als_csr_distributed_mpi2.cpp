/*
   SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
   http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/

   Copyright 2017 Intel Corporation

   THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
   NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
   PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
*/
/*
   Content:
     C++ example of the implicit alternating least squares (ALS) algorithm in
     the distributed processing mode USING MPI.

      Copyright 2017 Peter Pan (pertaining to bug fixes on Intel original ALS sample application code)
      Copyright 2017 Peter Pan (pertaining to Heuristic Interleaved Parameter Alternator algorithm)
*/

#include <unistd.h>
#include <map>
#include <set>

#include "mpi.h"
#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms::implicit_als;

/* Input data set parameters. This must match the -n arg of mpirun! */
const size_t nBlocks = 4;

int rankId, comm_size;

#define mpi_root 0

/* Input data set parameters */
#define DDIR "../data/"	//based on my working dir /data/.github/network_profile_recommender/daal

/* training data set in blocks for n MPI runners (ppan+ see convert_kmean_csv_to_als_csr.ipynb) */
const string trainDatasetFileNames[nBlocks] =
{
    DDIR"/sta_phyr_xaa.csr",
    DDIR"/sta_phyr_xab.csr",
    DDIR"/sta_phyr_xac.csr",
    DDIR"/sta_phyr_xad.csr"
};

/* transposed training data set in blocks for n MPI runners (ppan+ see convert_kmean_csv_to_als_csr.ipynb) */
const string transposedTrainDatasetFileNames[nBlocks] =
{
    DDIR"/sta_phyr_xaa.tsr",
    DDIR"/sta_phyr_xab.tsr",
    DDIR"/sta_phyr_xac.tsr",
    DDIR"/sta_phyr_xad.tsr"
};

static size_t usersPartition[nBlocks + 1];
static size_t itemsPartition[nBlocks + 1];

static int usersPartitionTmp[nBlocks + 1];
static int itemsPartitionTmp[nBlocks + 1];

typedef double  algorithmFPType;    /* Algorithm floating-point type */
typedef double  dataFPType;         /* Input data floating-point type */

/* Algorithm parameters */
size_t nUsers;           /* Full number of users */

byte *nodeResults;
byte *nodeCPs[nBlocks];
byte *crossProductBuf;
int crossProductLen;
int perNodeCPLength[nBlocks];
int displs[nBlocks];
int sdispls[nBlocks];
int rdispls[nBlocks];
int perNodeArchLengthMaster[nBlocks];
int perNodeArchLengthsRecv[nBlocks];

string rowFileName;
string colFileName;

services::SharedPtr<CSRNumericTable> dataTable;
services::SharedPtr<CSRNumericTable> transposedDataTable;

//To verify predictions it needs ALL transposed data blocks!
services::SharedPtr<CSRNumericTable> dataTables[nBlocks];
services::SharedPtr<CSRNumericTable> transposedDataTables[nBlocks];

KeyValueDataCollectionPtr usersOutBlocks;
KeyValueDataCollectionPtr itemsOutBlocks;

services::SharedPtr<training::DistributedPartialResultStep4> itemsPartialResultLocal;
services::SharedPtr<training::DistributedPartialResultStep4> usersPartialResultLocal;
services::SharedPtr<training::DistributedPartialResultStep4> itemsPartialResultsMaster[nBlocks];

services::SharedPtr<training::DistributedPartialResultStep1> step1LocalResults[nBlocks];
services::SharedPtr<training::DistributedPartialResultStep1> step1LocalResult;
NumericTablePtr step2MasterResult;

KeyValueDataCollectionPtr step3LocalResult;
KeyValueDataCollectionPtr step4LocalInput;

NumericTablePtr predictedRatingsLocal[nBlocks];
NumericTablePtr predictedRatingsMaster[nBlocks][nBlocks];
NumericTablePtr predictedRatings[nBlocks][nBlocks];

KeyValueDataCollectionPtr step1LocalPredictionResult;
KeyValueDataCollectionPtr step2LocalPredictionInput;

KeyValueDataCollectionPtr itemsPartialResultPrediction;

services::SharedPtr<byte> serializedData;
services::SharedPtr<byte> serializedSendData;
services::SharedPtr<byte> serializedRecvData;

void initializeModel();
void readData();
void trainModel();
void testModel();
void predictRatings();

services::SharedPtr<training::DistributedPartialResultStep4> deserializeStep4PartialResult(byte *buffer, size_t length);
services::SharedPtr<training::DistributedPartialResultStep1> deserializeStep1PartialResult(byte *buffer, size_t length);
NumericTablePtr deserializeNumericTable(byte *buffer, int length);
services::SharedPtr<PartialModel> deserializePartialModel(byte *buffer, size_t length);

void serializeDAALObject(SerializationIfacePtr data, byte **buffer, int *length);
void gatherStep1(byte *nodeResults, int perNodeArchLength);
void gatherItems(byte *nodeResults, int perNodeArchLength);
void gatherPrediction(byte *nodeResults, int perNodeArchLength, int iNode);
void all2all(byte **nodeResults, int *perNodeArchLength, KeyValueDataCollectionPtr result);

KeyValueDataCollectionPtr computeOutBlocks(
    size_t nRows, services::SharedPtr<CSRNumericTable> dataBlock, size_t *dataBlockPartition);
void computeStep1Local(services::SharedPtr<training::DistributedPartialResultStep4> partialResultLocal);
void computeStep2Master();
void computeStep3Local(size_t offset, services::SharedPtr<training::DistributedPartialResultStep4> partialResultLocal,
                       KeyValueDataCollectionPtr outBlocks);
void computeStep4Local(services::SharedPtr<CSRNumericTable> dataTable,
                       services::SharedPtr<training::DistributedPartialResultStep4> *partialResultLocal);

double testModelQuality(const int nAllUsers, const int nAllItems);

/* ppan+ tunable hyper-parameters for "model selection" */
/* the selection is modeled after my Spark ML code. see network_spark_ml_kmeams_als_HIPA.ipynb */
static size_t maxIterations = 1;
static size_t nFactors = 10;
static double alpha = 40;
static double lambda = 1.01;
static std::map<string, double> rmses;
static int nhit;
static std::vector<double> factors, alphas, lambdas, iterations;
static std::vector<double> *param[] = {&alphas, &lambdas, &factors, &iterations};	//must match change_parameter() !!!

#define numof(a) (sizeof(a) / sizeof(a[0]))
#define NSMP 2					// limited number of samples on either side of pivot parameter
#define NP numof(param)			// hyper-parameters for implicit ALS

static void model_selection_tune_hyperparameters(daal::algorithms::implicit_als::interface1::Parameter &parameter)
{
	parameter.nFactors = nFactors;
	parameter.alpha = alpha;
	parameter.lambda = lambda;
}

static double train_test_cross_verify(const int nAllUsers, const int nAllItems, bool &hit)
{
	double rmse;

	//avoid recalc
	char str[200];
	snprintf(str, sizeof(str) - 1, "%d,%lf,%lf,%d", nFactors, alpha, lambda, maxIterations);

	if (true == (hit = rmses.end() != rmses.find(str)))
	{
		++nhit;
		rmse = rmses[str];
	}
	else
	{
		step4LocalInput              = KeyValueDataCollectionPtr(new KeyValueDataCollection());
		itemsPartialResultPrediction = KeyValueDataCollectionPtr(new KeyValueDataCollection());

		initializeModel();

		usersOutBlocks = computeOutBlocks(dataTable->getNumberOfRows(),           dataTable,           itemsPartition);
		itemsOutBlocks = computeOutBlocks(transposedDataTable->getNumberOfRows(), transposedDataTable, usersPartition);

	//	if (rankId == mpi_root)
	//		printf("train(%d): nFactors %d, alpha %.1lf, lambda %.2lf ...\n", rankId, nFactors, alpha, lambda);
		trainModel();
	//	printf("train(%d) ok\n", rankId);

	//	if (rankId == mpi_root)
	//		printf("test(%d): nFactors %d, alpha %.1lf, lambda %.2lf ...\n", rankId, nFactors, alpha, lambda);
		testModel();
	//	printf("test(%d) ok\n", rankId);

		if (rankId == mpi_root)
		{
			#if 0
			for (size_t i = 0; i < nBlocks; i++)
			for (size_t j = 0; j < nBlocks; j++)
			{
				printf("prediction block[%lu, %lu]: shape = (%d, %d)\n", i, j,
					predictedRatingsMaster[i][j]->getNumberOfRows(),
					predictedRatingsMaster[i][j]->getNumberOfColumns()
					);
		//		printNumericTable(predictedRatingsMaster[i][j]);
			}
			#endif

			rmse = rmses[str] = testModelQuality(nAllUsers, nAllItems);
		}
	}

	//ALL nodes sync up rmse of this run!
	MPI_Bcast(&rmse, sizeof(rmse), MPI_CHAR, mpi_root, MPI_COMM_WORLD);
	rmses[str] = rmse;
	return rmse;
}

/* ppan+
 * Heuristic Interleaved Parameter Alternator (HIPA)
 * The basic of the idea is to iterate larger range of one parameter while visit only the
 * 'pivot' value of all other parameters in (nFactor, alpha and lambda, iterations).
 * In contrast, SPARK ML CrossValidator iterates through ALL tuples of a parameter grid,
 *  so it takes much more iterations to learn the best model parameters.
 */

//change a hype-parameter of ALS on local node
static void change_parameter(const int ipam, const double newv)
{
	//must match the order of param[] in heuristic_interleaved_paramters_alternator!!
	switch (ipam)
	{
		case 0: alpha  = newv;				break;
		case 1: lambda = newv;				break;
		case 2:	nFactors = (int)newv;		break;
		case 3:	maxIterations = (int)newv;	break;
	}
}

//sync all hyper-parameters of ALS with all nodes
static void sync_parameter()
{
	double param[NP];

	param[0] = nFactors;
	param[1] = alpha;
	param[2] = lambda;
	param[3] = maxIterations;

	MPI_Bcast(param, sizeof(param), MPI_CHAR, mpi_root, MPI_COMM_WORLD);

	nFactors		= param[0];
	alpha 	 		= param[1];
	lambda   		= param[2];
	maxIterations 	= param[3];
}

//HIPA main code
static void heuristic_interleaved_paramters_alternator(const int nAllUsers, const int nAllItems)
{
	//init a NP-dimensional parameter "grid"
	#if 1	//this grid take much more time
	for (double a = 35.0; a <= 65.0; a += 0.50) alphas.push_back(a);
	for (double a = 0.30; a <= 0.50; a += 0.05) lambdas.push_back(a);
	for (double a = 55.0; a <= 95.0; a += 1.00) factors.push_back(a);
	for (double a =  2.0; a <=  5.0; a += 1.00) iterations.push_back(a);
	#else	//for development only
	for (double a = 55.0; a <= 60.0; a += 0.50) alphas.push_back(a);
	for (double a = 0.25; a <= 0.35; a += 0.05) lambdas.push_back(a);
	for (double a = 85.0; a <= 95.0; a += 1.00) factors.push_back(a);
	for (double a =  3.0; a <=  5.0; a += 1.00) iterations.push_back(a);
	#endif

	int pmsiz[NP];
	for (int p = 0; p < NP; ++p) pmsiz[p] = param[p]->size();

	std::map<int, bool> full;
	std::map<int, double> emin[NP];
	auto min_rmse = std::numeric_limits<double>::max();

	for (int fullp = 0, nstall = 0; (true); fullp = (fullp + 1) % NP)
	{
		//collect indices of all parameters to try in this round
		std::vector<int> pidxs[NP];
		int niter = 1;
		for (int p = 0; p < NP; p++)
		{
			#define NTOP 2	//for now let's try the 2 indices with the least rmse

			//sort emin[p] by value in ascending order
			vector<pair<int, double>> epairs(emin[p].begin(), emin[p].end());
			std::sort(epairs.begin(), epairs.end(), [](const pair<int, double> &a, const pair<int, double> &b) -> bool
			{
				return a.second < b.second;
			});

			//after full scan of a parameter, choose the indice with minimum errors
			if (p == fullp)
			{
				if (!full[p])	//not full scan yet
				{
					full[p] = true;
					for (int i = 0; i < pmsiz[p]; i++)
						pidxs[p].push_back(i);
				}
				else
				{
					//pick NTOP indices after full scan
					std::map<int, double> weaks(epairs.begin() + NTOP, epairs.end());
					for (int i = 0; i < pmsiz[p]; i++)
						if (weaks.find(i) == weaks.end())
							pidxs[p].push_back(i);
				}
			}
			else
			//for other parameters on 2nd+ rounds, also choose the indice with minimum errors
			if (epairs.size())
			{
				//pick NTOP indices with least rmse
				for (int i = 0; i < NTOP; i++)
					if (i < epairs.size())
						pidxs[p].push_back(epairs[i].first);
			}
			else
			//for other parameters on 1st round, choose the first, middle and last indices
			{
				pidxs[p].push_back(0);
				pidxs[p].push_back(pmsiz[p] / 2);
				pidxs[p].push_back(pmsiz[p] - 1);
			}

			//below will use product of pidxs[*] to traverse all chosen indices
			niter *= pidxs[p].size();
		}

		auto new_min = false;
		for (int it = 0; it < niter; ++it)
		{
			int pidx[NP];

			//update locally all parameters of this run
			//by breaking down current it value
			for (int p = 0, tit = it; p < NP; tit /= pidxs[p++].size())
			{
				pidx[p] = pidxs[p][tit % pidxs[p].size()];
				change_parameter(p, (*param[p])[pidx[p]]);
			}

			//sync globally with ALL nodes
			sync_parameter();

			auto hit = false;
			auto rmse = train_test_cross_verify(nAllUsers, nAllItems, hit);
			if (rankId == mpi_root)
				printf("rmse %.5lf, nfactor %2d, alpha %.1lf, lambda %.2lf, maxIterations %d %c %c\n",
					rmse, nFactors, alpha, lambda, maxIterations, hit? '-': ' ', min_rmse > rmse? '*': ' ');

			if (min_rmse > rmse)
			{
				min_rmse = rmse;
				new_min = true;
			}

			//track minimum rmse of all indexes of all parameters
			//this is to avoid revisit sucky indexes on 2nd round
			if (!hit)
				for (int p = 0; p < NP; p++)
				{
					auto pi = pidx[p];
					if (emin[p].find(pi) == emin[p].end())
							emin[p][pi] = rmse;
					else	emin[p][pi] = std::min(emin[p][pi], rmse);
				}
		}

		nstall = new_min? 0: nstall+1;
		if (nstall >= NP) break;
	}

	if (rankId == mpi_root)
	{
		//print HIPA summary
		int ocalc = 1;
		for (int i = 0; i < NP; i++) ocalc *= pmsiz[i];

		printf("calcs w/o hipa: %d\n", ocalc);
		printf("calcs w/  hipa: %d\n", rmses.size());
		printf("calc hits     : %d\n", nhit);
		printf("calc saving   : %d%%\n", (int)(100.0 * (ocalc - rmses.size()) / ocalc));

		//print the 10 best parameter sets
		vector<pair<string, double>> epairs(rmses.begin(), rmses.end());
		std::sort(epairs.begin(), epairs.end(), [](const pair<string, double> &a, const pair<string, double> &b) -> bool
		{
			return a.second < b.second;
		});

		printf("top 10 best parameter sets:\n");
		for (int i = 0; i < 10; i ++)
		{
			size_t nFactors, maxIterations;
			double alpha, lambda;
			sscanf(epairs[i].first.c_str(), "%ld,%lf,%lf,%ld", &nFactors, &alpha, &lambda, &maxIterations);
			printf("%2d: rmse %.5lf, nfactor %2ld, alpha %.1lf, lambda %.2lf, maxIterations %d\n",
				i, epairs[i].second, nFactors, alpha, lambda, maxIterations);
		}
	}
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);

    rowFileName = trainDatasetFileNames[rankId];
    colFileName = transposedTrainDatasetFileNames[rankId];

    readData();

	//shape of local matrix
    int userNum = dataTable->getNumberOfRows();
    int itemNum = transposedDataTable->getNumberOfRows();
    MPI_Allgather(&userNum, sizeof(int), MPI_CHAR, usersPartitionTmp, sizeof(int), MPI_CHAR, MPI_COMM_WORLD);
    MPI_Allgather(&itemNum, sizeof(int), MPI_CHAR, itemsPartitionTmp, sizeof(int), MPI_CHAR, MPI_COMM_WORLD);

    usersPartition[0] = 0;
    itemsPartition[0] = 0;
    for (size_t i = 0; i < nBlocks; i++)
    {
        usersPartition[i + 1] = usersPartition[i] + usersPartitionTmp[i];
        itemsPartition[i + 1] = itemsPartition[i] + itemsPartitionTmp[i];
    }

	//ppan+ The userNum and itemNum above are local sizes.
	//Must gather global (full) sizes for quality test!
	int nAllUsers = transposedDataTable->getNumberOfColumns();
	int nAllItems = dataTable->getNumberOfColumns();
	MPI_Allgather(&nAllUsers, sizeof(int), MPI_CHAR, usersPartitionTmp, sizeof(int), MPI_CHAR, MPI_COMM_WORLD);
	MPI_Allgather(&nAllItems, sizeof(int), MPI_CHAR, itemsPartitionTmp, sizeof(int), MPI_CHAR, MPI_COMM_WORLD);

	//full sizes are maximum among all, or mergePredictions crashes!
	for (size_t i = 0; i < nBlocks; i++)
	{
		nAllUsers = std::max(nAllUsers, usersPartitionTmp[i]);
		nAllItems = std::max(nAllItems, itemsPartitionTmp[i]);
	}
	printf("nAllUsers = %d\n", nAllUsers);
	printf("nAllItems = %d\n", nAllItems);

	heuristic_interleaved_paramters_alternator(nAllUsers, nAllItems);

    MPI_Finalize();
    return 0;
}

void readData()
{
	for (int i = 0; i < nBlocks; i++)
	{
		/* Read from a file and create a numeric table to store the input data */
		dataTables          [i] = services::SharedPtr<CSRNumericTable>(createSparseTable<dataFPType>(trainDatasetFileNames          [i]));
		transposedDataTables[i] = services::SharedPtr<CSRNumericTable>(createSparseTable<dataFPType>(transposedTrainDatasetFileNames[i]));
		if (i == rankId) dataTable           = dataTables          [i];
		if (i == rankId) transposedDataTable = transposedDataTables[i];
	}

	//ppan+ learn nUsers from data rather than hardcoded like original DAAL sample code :)
	/*  !!!
		Must +1 below to cheat over the check in daal/algorithms/kernel/implicit_als/implicit_als_train_init_partial_result.cpp

			size_t nRows = algInput->get(data)->getNumberOfRows();
			DAAL_CHECK_EX(algParameter->fullNUsers > nRows, ErrorIncorrectParameter, ParameterName, fullNUsersStr());

		No idea why Intel put such a condition here... Have reported this issue on github.
	 */
	nUsers = transposedDataTable->getNumberOfRows() + 1;

	#if 1
	sleep(rankId);
	printf("nUsers[%d] = %d\n", rankId, nUsers);
	printf("          dataTable[%d]= %d, %d\n", rankId, dataTable		   ->getNumberOfRows(), dataTable		   ->getNumberOfColumns());
	printf("transposedDataTable[%d]= %d, %d\n", rankId, transposedDataTable->getNumberOfRows(), transposedDataTable->getNumberOfColumns());
	#endif
}

KeyValueDataCollectionPtr computeOutBlocks(size_t nRows, services::SharedPtr<CSRNumericTable> dataBlock, size_t *dataBlockPartition)
{
    char *blockIdFlags = new char[nRows * nBlocks];
    for (size_t i = 0; i < nRows * nBlocks; i++)
    {
        blockIdFlags[i] = '\0';
    }

    size_t *colIndices, *rowOffsets;
    dataBlock->getArrays<double>(NULL, &colIndices, &rowOffsets);//template parameter doesn't matter

    for (size_t i = 0; i < nRows; i++)
    {
        for (size_t j = rowOffsets[i] - 1; j < rowOffsets[i+1] - 1; j++)
        {
            for (size_t k = 1; k < nBlocks + 1; k++)
            {
                if (dataBlockPartition[k-1] <= colIndices[j] - 1 && colIndices[j] - 1 < dataBlockPartition[k])
                {
                    blockIdFlags[(k-1) * nRows + i] = 1;
                }
            }
        }
    }

    size_t nNotNull[nBlocks];
    for (size_t i = 0; i < nBlocks; i++)
    {
        nNotNull[i] = 0;
        for (size_t j = 0; j < nRows; j++)
        {
            nNotNull[i] += blockIdFlags[i * nRows + j];
        }
    }

    KeyValueDataCollectionPtr result(new KeyValueDataCollection());

    for (size_t i = 0; i < nBlocks; i++)
    {
        HomogenNumericTable<int> * indicesTable = new HomogenNumericTable<int>(1, nNotNull[i], NumericTableIface::doAllocate);
        SerializationIfacePtr indicesTableShPtr(indicesTable);
        int *indices = indicesTable->getArray();
        size_t indexId = 0;

        for (size_t j = 0; j < nRows; j++)
        {
            if (blockIdFlags[i * nRows + j])
            {
                indices[indexId++] = j;
            }
        }
        (*result)[i] = indicesTableShPtr;
    }

    delete [] blockIdFlags;
    return result;
}

void initializeModel()
{
    /* Create an algorithm object to initialize the implicit ALS model with the default method */
    training::init::Distributed<step1Local, algorithmFPType, training::init::fastCSR> initAlgorithm;
    initAlgorithm.parameter.fullNUsers = nUsers;
    initAlgorithm.parameter.nFactors = nFactors;
    initAlgorithm.parameter.seed += rankId;
    /* Pass a training data set and dependent values to the algorithm */
    initAlgorithm.input.set(training::init::data, transposedDataTable);
//printf("xxx %d: a\n", rankId);	//may crash if transpose csr files are not in good shape!!!
    /* Initialize the implicit ALS model */
    initAlgorithm.compute();
//printf("xxx %d: b\n", rankId);

    services::SharedPtr<PartialModel> partialModelLocal = initAlgorithm.getPartialResult()->get(training::init::partialModel);

    itemsPartialResultLocal = services::SharedPtr<training::DistributedPartialResultStep4>(new training::DistributedPartialResultStep4());
    itemsPartialResultLocal->set(training::outputOfStep4ForStep1, partialModelLocal);
}

void computeStep1Local(services::SharedPtr<training::DistributedPartialResultStep4> partialResultLocal)
{
    /* Create algorithm objects to compute implicit ALS algorithm in the distributed processing mode on the local node using the default method */
    training::Distributed<step1Local> algorithm;
    algorithm.parameter.nFactors = nFactors;

	model_selection_tune_hyperparameters(algorithm.parameter);

    /* Set input objects for the algorithm */
    algorithm.input.set(training::partialModel, partialResultLocal->get(training::outputOfStep4ForStep1));

    /* Compute partial estimates on local nodes */
    algorithm.compute();

    /* Get the computed partial estimates */
    step1LocalResult = algorithm.getPartialResult();
}

void computeStep2Master()
{
    /* Create algorithm objects to compute implicit ALS algorithm in the distributed processing mode on the master node using the default method */
    training::Distributed<step2Master> algorithm;
    algorithm.parameter.nFactors = nFactors;

	model_selection_tune_hyperparameters(algorithm.parameter);

    /* Set input objects for the algorithm */
    for (size_t i = 0; i < nBlocks; i++)
    {
        algorithm.input.add(training::inputOfStep2FromStep1, step1LocalResults[i]);
    }

    /* Compute a partial estimate on the master node from the partial estimates on local nodes */
    algorithm.compute();

    step2MasterResult = algorithm.getPartialResult()->get(training::outputOfStep2ForStep4);
}

void computeStep3Local(size_t offset, services::SharedPtr<training::DistributedPartialResultStep4> partialResultLocal,
                       KeyValueDataCollectionPtr outBlocks)
{
    training::Distributed<step3Local> algorithm;
    algorithm.parameter.nFactors = nFactors;

	model_selection_tune_hyperparameters(algorithm.parameter);

    services::SharedPtr<HomogenNumericTable<size_t> > offsetTable(
            new HomogenNumericTable<size_t>(&offset, 1, 1));
    algorithm.input.set(training::partialModel,             partialResultLocal->get(training::outputOfStep4ForStep3));
    algorithm.input.set(training::partialModelBlocksToNode, outBlocks);
    algorithm.input.set(training::offset,                   offsetTable);

    algorithm.compute();

    step3LocalResult = algorithm.getPartialResult()->get(training::outputOfStep3ForStep4);

    /* MPI_Alltoallv to populate step4LocalInput */
    for (size_t i = 0; i < nBlocks; i++)
    {
        serializeDAALObject((*step3LocalResult)[i], &nodeCPs[i], &perNodeCPLength[i]);
    }
    all2all(nodeCPs, perNodeCPLength, step4LocalInput);
}

void computeStep4Local(services::SharedPtr<CSRNumericTable> dataTable,
                       services::SharedPtr<training::DistributedPartialResultStep4> *partialResultLocal)
{
    training::Distributed<step4Local> algorithm;
    algorithm.parameter.nFactors = nFactors;

	model_selection_tune_hyperparameters(algorithm.parameter);

    algorithm.input.set(training::partialModels,         step4LocalInput);
    algorithm.input.set(training::partialData,           dataTable);
    algorithm.input.set(training::inputOfStep4FromStep2, step2MasterResult);

    algorithm.compute();

    *partialResultLocal = algorithm.getPartialResult();
}

void trainModel()
{
    int perNodeArchLength;
    for (size_t iteration = 0; iteration < maxIterations; iteration++)
    {
        computeStep1Local(itemsPartialResultLocal);

        serializeDAALObject(step1LocalResult, &nodeResults, &perNodeArchLength);
        /*Gathering step1LocalResult on the master*/
        gatherStep1(nodeResults, perNodeArchLength);

        if(rankId == mpi_root) {
            computeStep2Master();
            serializeDAALObject(step2MasterResult, &crossProductBuf, &crossProductLen);
        }

        MPI_Bcast(&crossProductLen, sizeof(int), MPI_CHAR, mpi_root, MPI_COMM_WORLD);
        if(rankId != mpi_root)
        {
            crossProductBuf = new byte[crossProductLen];
        }
        MPI_Bcast(crossProductBuf, crossProductLen, MPI_CHAR, mpi_root, MPI_COMM_WORLD);
        step2MasterResult = deserializeNumericTable(crossProductBuf, crossProductLen);

        //ppan+
        delete[] crossProductBuf;

        computeStep3Local(itemsPartition[rankId], itemsPartialResultLocal, itemsOutBlocks);
        computeStep4Local(dataTable, &usersPartialResultLocal);

        computeStep1Local(usersPartialResultLocal);

        serializeDAALObject(step1LocalResult, &nodeResults, &perNodeArchLength);
        /*Gathering step1LocalResult on the master*/
        gatherStep1(nodeResults, perNodeArchLength);

        if(rankId == mpi_root) {
            computeStep2Master();
            serializeDAALObject(step2MasterResult, &crossProductBuf, &crossProductLen);
        }

        MPI_Bcast(&crossProductLen, sizeof(int), MPI_CHAR, mpi_root, MPI_COMM_WORLD);
        if(rankId != mpi_root)
        {
            crossProductBuf = new byte[crossProductLen];
        }
        MPI_Bcast(crossProductBuf, crossProductLen, MPI_CHAR, mpi_root, MPI_COMM_WORLD);
        step2MasterResult = deserializeNumericTable(crossProductBuf, crossProductLen);

        //ppan+
        delete[] crossProductBuf;

        computeStep3Local(usersPartition[rankId], usersPartialResultLocal, usersOutBlocks);
        computeStep4Local(transposedDataTable, &itemsPartialResultLocal);
    }

    /*Gather all itemsPartialResultLocal to itemsPartialResultsMaster on the master and distributing the result over other ranks*/
    serializeDAALObject(itemsPartialResultLocal, &nodeResults, &perNodeArchLength);
    gatherItems(nodeResults, perNodeArchLength);
}

void testModel()
{
    /* Create an algorithm object to predict recommendations of the implicit ALS model */
    for (size_t i = 0; i < nBlocks; i++)
    {
        int perNodeArchLength;
        prediction::ratings::Distributed<step1Local> algorithm;
        algorithm.parameter.nFactors = nFactors;

		model_selection_tune_hyperparameters(algorithm.parameter);

        algorithm.input.set(prediction::ratings::usersPartialModel, usersPartialResultLocal->get(training::outputOfStep4ForStep1));
        algorithm.input.set(prediction::ratings::itemsPartialModel,
                services::staticPointerCast<training::DistributedPartialResultStep4,
                        SerializationIface>(itemsPartialResultsMaster[i])->get(training::outputOfStep4ForStep1));

        algorithm.compute();

        predictedRatingsLocal[i] = algorithm.getResult()->get(prediction::ratings::prediction);

        serializeDAALObject(predictedRatingsLocal[i], &nodeResults, &perNodeArchLength);
        gatherPrediction(nodeResults, perNodeArchLength, i);
    }
}

void gatherStep1(byte *nodeResults, int perNodeArchLength)
{
    MPI_Gather(&perNodeArchLength, sizeof(int), MPI_CHAR, perNodeArchLengthMaster, sizeof(int), MPI_CHAR, mpi_root, MPI_COMM_WORLD);

    if (rankId == mpi_root)
    {
        int memoryBuf = 0;
        for (int i = 0; i < nBlocks; i++)
        {
            memoryBuf += perNodeArchLengthMaster[i];
        }
        serializedData = services::SharedPtr<byte>(new byte[memoryBuf]);

        size_t shift = 0;
        for(size_t i = 0; i < nBlocks ; i++)
        {
            displs[i] = shift;
            shift += perNodeArchLengthMaster[i];
        }
    }

    /* Transfer partial results to step 2 on the root node */
    MPI_Gatherv( nodeResults, perNodeArchLength, MPI_CHAR, serializedData.get(), perNodeArchLengthMaster, displs, MPI_CHAR, mpi_root,
                 MPI_COMM_WORLD);
    if (rankId == mpi_root)
    {
        for(size_t i = 0; i < nBlocks ; i++)
        {
            /* Deserialize partial results from step 1 */
            step1LocalResults[i] = deserializeStep1PartialResult(serializedData.get() + displs[i], perNodeArchLengthMaster[i]);
			//printf("gatherStep1 %d: %d, %d\n", i, displs[i], perNodeArchLengthMaster[i]);
        }
    }

    delete[] nodeResults;
}

void gatherItems(byte *nodeResults, int perNodeArchLength)
{
    MPI_Allgather(&perNodeArchLength, sizeof(int), MPI_CHAR, perNodeArchLengthMaster, sizeof(int), MPI_CHAR, MPI_COMM_WORLD);

    int memoryBuf = 0;
    for (int i = 0; i < nBlocks; i++)
    {
        memoryBuf += perNodeArchLengthMaster[i];
    }
    serializedData = services::SharedPtr<byte>(new byte[memoryBuf]);

    size_t shift = 0;
    for(size_t i = 0; i < nBlocks ; i++)
    {
        displs[i] = shift;
        shift += perNodeArchLengthMaster[i];
    }

    /* Transfer partial results to step 2 on the root node */
    MPI_Allgatherv( nodeResults, perNodeArchLength, MPI_CHAR, serializedData.get(), perNodeArchLengthMaster, displs, MPI_CHAR, MPI_COMM_WORLD);

    for(size_t i = 0; i < nBlocks ; i++)
    {
        /* Deserialize partial results from step 4 */
        itemsPartialResultsMaster[i] = deserializeStep4PartialResult(serializedData.get() + displs[i], perNodeArchLengthMaster[i]);
    }

    delete[] nodeResults;
}

void gatherPrediction(byte *nodeResults, int perNodeArchLength, int iNode)
{
    MPI_Gather(&perNodeArchLength, sizeof(int), MPI_CHAR, perNodeArchLengthMaster, sizeof(int), MPI_CHAR, mpi_root, MPI_COMM_WORLD);

    if (rankId == mpi_root)
    {
        int memoryBuf = 0;
        for (int i = 0; i < nBlocks; i++)
        {
            memoryBuf += perNodeArchLengthMaster[i];
        }
        serializedData = services::SharedPtr<byte>(new byte[memoryBuf]);

        size_t shift = 0;
        for(size_t i = 0; i < nBlocks ; i++)
        {
            displs[i] = shift;
            shift += perNodeArchLengthMaster[i];
        }
    }

    /* Transfer partial results to step 2 on the root node */
    MPI_Gatherv( nodeResults, perNodeArchLength, MPI_CHAR, serializedData.get(), perNodeArchLengthMaster, displs, MPI_CHAR, mpi_root,
                 MPI_COMM_WORLD);

    if (rankId == mpi_root)
    {
        for(size_t i = 0; i < nBlocks ; i++)
        {
            /* Deserialize partial results from step 1 */
            predictedRatingsMaster[iNode][i] = deserializeNumericTable(serializedData.get() + displs[i], perNodeArchLengthMaster[i]);
        }
    }

    delete[] nodeResults;
}

void all2all(byte **nodeResults, int *perNodeArchLengths, KeyValueDataCollectionPtr result)
{
    int memoryBuf = 0;
    size_t shift = 0;
    for (int i = 0; i < nBlocks; i++)
    {
        memoryBuf += perNodeArchLengths[i];
        sdispls[i] = shift;
        shift += perNodeArchLengths[i];
    }
    serializedSendData = services::SharedPtr<byte>(new byte[memoryBuf]);

    /* memcpy to avoid double compute */
    memoryBuf = 0;
    for (int i = 0; i < nBlocks; i++)
    {
        daal::services::daal_memcpy_s(serializedSendData.get()+memoryBuf, (size_t)perNodeArchLengths[i],
                                      nodeResults[i], (size_t)perNodeArchLengths[i]);
        memoryBuf += perNodeArchLengths[i];
        delete[] nodeResults[i];
    }

    MPI_Alltoall(perNodeArchLengths, sizeof(int), MPI_CHAR, perNodeArchLengthsRecv, sizeof(int), MPI_CHAR, MPI_COMM_WORLD);

    memoryBuf = 0;
    shift = 0;
    for (int i = 0; i < nBlocks; i++)
    {
        memoryBuf += perNodeArchLengthsRecv[i];
        rdispls[i] = shift;
        shift += perNodeArchLengthsRecv[i];
    }
    serializedRecvData = services::SharedPtr<byte>(new byte[memoryBuf]);

    /* Transfer partial results to step 2 on the root node */
    MPI_Alltoallv( serializedSendData.get(), perNodeArchLengths, sdispls, MPI_CHAR,
                  serializedRecvData.get(), perNodeArchLengthsRecv, rdispls, MPI_CHAR, MPI_COMM_WORLD);

    for (size_t i = 0; i < nBlocks; i++)
    {
        (*result)[i] = deserializePartialModel(serializedRecvData.get() + rdispls[i], perNodeArchLengthsRecv[i]);
    }
}

void serializeDAALObject(SerializationIfacePtr data, byte **buffer, int *length)
{
    /* Create a data archive to serialize the numeric table */
    InputDataArchive dataArch;

    /* Serialize the numeric table into the data archive */
    data->serialize(dataArch);

    /* Get the length of the serialized data in bytes */
    *length = dataArch.getSizeOfArchive();

    /* Store the serialized data in an array */
    *buffer = new byte[*length];
    dataArch.copyArchiveToArray(*buffer, *length);
}

services::SharedPtr<PartialModel> deserializePartialModel(byte *buffer, size_t length)
{
    return services::dynamicPointerCast<PartialModel, SerializationIface>(deserializeDAALObject(buffer, length));
}

NumericTablePtr deserializeNumericTable(byte *buffer, int length)
{
    OutputDataArchive outDataArch(buffer, length);

    algorithmFPType *values = NULL;
    NumericTablePtr restoredDataTable = NumericTablePtr( new HomogenNumericTable<algorithmFPType>(values) );

    restoredDataTable->deserialize(outDataArch);

    return restoredDataTable;
}

services::SharedPtr<training::DistributedPartialResultStep4> deserializeStep4PartialResult(byte *buffer, size_t length)
{
    return services::dynamicPointerCast<training::DistributedPartialResultStep4, SerializationIface>(
        deserializeDAALObject(buffer, length));
}

services::SharedPtr<training::DistributedPartialResultStep1> deserializeStep1PartialResult(byte *buffer, size_t length)
{
    return services::dynamicPointerCast<training::DistributedPartialResultStep1, SerializationIface>(
        deserializeDAALObject(buffer, length));
}

void mergePredictions(HomogenNumericTable<float> &predictions)
{
	//for mpi predictedRatings = predictedRatingsMaster, which is a merged and full matrix
	#define predictedRatings predictedRatingsMaster
    const size_t nItems = predictions.getNumberOfColumns();

	#if 0
	printf("predictions.getNumberOfRows()=%d\n",    predictions.getNumberOfRows());
	printf("predictions.getNumberOfColumns()=%d\n", predictions.getNumberOfColumns());
	#endif

	float *dst = predictions.getArray();
	BlockDescriptor<float> block;

	size_t colOffset = 0;
	for (size_t i = 0; i < nBlocks; i++)
	{
		const size_t nItemsPart = predictedRatings[i][0]->getNumberOfColumns();
		size_t rowOffset = 0;
		for (size_t j = 0; j < nBlocks; j++)
		{
			#if 0
			printf("predictedRatings[%d][%d]: %d, %d\n",
				i, j,
				predictedRatings[i][j]->getNumberOfRows(),
				predictedRatings[i][j]->getNumberOfColumns());
			#endif
			const size_t nUsersPart = predictedRatings[0][j]->getNumberOfRows();
			predictedRatings[i][j]->getBlockOfRows(0, nUsersPart, readOnly, block);
			float *src = block.getBlockPtr();
			for(size_t ii = 0; ii < nUsersPart; ii++)
				for (size_t jj = 0; jj < nItemsPart; jj++)
					dst[(ii + rowOffset) * nItems + jj + colOffset] = src[ii * nItemsPart + jj];

			predictedRatings[i][j]->releaseBlockOfRows(block);
			rowOffset += nUsersPart;
		}
		colOffset += nItemsPart;
	}
}

double testModelQuality(const int nUsers, const int nItems)
{
//	printf("matrix full size: %d, %d\n", nUsers, nItems);
    /* Merge predicted ratings into one big table */
    HomogenNumericTable<float> predictions(nItems, nUsers, NumericTable::doAllocate);
    mergePredictions(predictions);

    /* Compute RMSE for the training data set */
    double rmse(0.0);
    size_t nRatings(0);
    size_t rowOffset(0);
    float *dataPredicted = predictions.getArray();
    CSRBlockDescriptor<float> sparseBlock;

//    cout << "User ID, Item ID: Expected Rating, Predicted Rating" << endl;
    /* Update RMSE for each block */
    for (size_t i = 0; i < nBlocks; i++)
    {
        /* Get number of rows in the block of data */
        const size_t nUsersPart = dataTables[i]->getNumberOfRows();
        /* Read sparse data block */
        dataTables[i]->getSparseBlock(0, nUsersPart, readOnly, sparseBlock);
        float *dataExpected = sparseBlock.getBlockValuesPtr();

		//somehow can't find these shared ptrs
		#if 0
        size_t *colIndices = sparseBlock.getBlockColumnIndicesSharedPtr().get();
        size_t *rowOffsets = sparseBlock.getBlockRowIndicesSharedPtr().get();
        #else
        size_t *colIndices = sparseBlock.getBlockColumnIndicesPtr();
        size_t *rowOffsets = sparseBlock.getBlockRowIndicesPtr();
        #endif

        for(size_t ii = 0; ii < nUsersPart; ii++)
        {
            const size_t startIdx = rowOffsets[ii] - 1;
            const size_t endIdx = rowOffsets[ii + 1] - 1;
            for (size_t jj = startIdx; jj < endIdx; jj++)
            {
                const size_t colIdx = colIndices[jj] - 1;
                const size_t rowIdx = rowOffset + ii;
                /* Expected preference for the current rating */
                float expected = (dataExpected[jj] > 0.0f ? 1.0f : 0.0f);
                /* Predicted preference for the current rating */
                float predicted = dataPredicted[rowIdx * nItems + colIdx];

				//ppan+ my own normalized prediction
				expected = dataExpected[jj];
				if (predicted < 0.) predicted = 0.;
				if (predicted > 1.) predicted = 1.;

				if (fabs(expected - predicted) > 1. || std::isnan(expected - predicted))
					cout << rowIdx << ",\t\t" << colIdx << "\t:\t" << expected << ",\t\t" << predicted << endl;

				double diff = expected - predicted;
                /* Update RMSE for the current rating */
				if (!std::isnan(diff))
					rmse += diff * diff;

                nRatings++;
            }
        }

        dataTables[i]->releaseSparseBlock(sparseBlock);
        rowOffset += nUsersPart;
    }
    double invNRatings = 1.0 / double(nRatings);
    rmse *= invNRatings;
    rmse = sqrt(rmse);
//    cout << endl << "Number of ratings in the data set: " << nRatings << endl;
//    cout << "RMSE: " << rmse << endl;

	return rmse;
}
