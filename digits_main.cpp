//
//  digits_main.cpp
//  digits
//
//  Created by Dom2 on 20-07-28.
//

#include "digits_net.h"

int main()
{
	vector<int> layers{60};

	int num_threads=2;
	omp_set_num_threads(num_threads);

	network NN(layers);
	
	int N_epochs=30;
	int batch_size=20;
	double eta=0.1;
	bool test=true;
	data_type_T test_data=TEST;
	int N_tr=50000;
	double reg_f=4.0;
	
//	cout<<"start\n";
//	clock_t t1=clock();
	
	struct timespec start, finish;
	double elapsed;
	
	clock_gettime(CLOCK_MONOTONIC, &start);
	
	NN.SGD_CE_mat(N_epochs, batch_size, eta, test, test_data, N_tr, reg_f);
//	NN.SGD_CE(N_epochs, batch_size, eta, test, test_data, N_tr, reg_f);
//	NN.SGD(N_epochs, batch_size, eta, test, test_data, N_tr, reg_f);
	
	clock_gettime(CLOCK_MONOTONIC, &finish);
	
	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1.0e9;
	
	cout<<"elapsed time: "<<elapsed<<" seconds.\n"<<endl;
	
//	t1=clock()-t1;
//	cout<<((double)t1)/CLOCKS_PER_SEC/num_threads<<" seconds for "<<N_epochs<<" epochs."<<endl;
	
	
	
	NN.test_network(TEST);
	
	return 0;
}

