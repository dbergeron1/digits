//
//  digits_main.cpp
//  digits
//
//  Created by Dom2 on 20-07-28.
//

#include "digits_net.h"

void remove_spaces_back(string &str);
void remove_spaces_front(string &str);
void remove_spaces_ends(string &str);

int main()
{
	int Nhl;
	vector<int> layers;
	int num_threads;
	int N_epochs_max;
	int batch_size;
	double eta_init;
	double red_f_eta;
	double max_eta_ratio;
	bool test=false;
	data_type_T test_data;
	int N_tr;
	double reg_f;
	int pl_lgth_max;
	
	ifstream input_file("../../digits.dat");
	
	string str, str1;
	
	if (input_file)
	{
	//	input_file.getline(input, Nhl_str.size()+1);
		getline(input_file,str,':');
		input_file>>Nhl;
		cout<<"number of hidden layers: "<<Nhl<<endl;
		layers.resize(Nhl);
	//	getline(input_file,str);
		getline(input_file,str,':');
		cout<<"hidden layers sizes: ";
		for (int i=0; i<Nhl; i++)
		{
			input_file>>layers[i];
			cout<<layers[i]<<" ";
		}
		cout<<endl;
		getline(input_file,str,':');
		input_file>>N_epochs_max;
		cout<<"maximum number of epochs: "<<N_epochs_max<<endl;
		getline(input_file,str,':');
		input_file>>batch_size;
		cout<<"batch size: "<<batch_size<<endl;
		getline(input_file,str,':');
		input_file>>eta_init;
		cout<<"initial learning rate: "<<eta_init<<endl;
		getline(input_file,str,':');
		input_file>>red_f_eta;
		cout<<"learning rate reduction factor: "<<red_f_eta<<endl;
		getline(input_file,str,':');
		input_file>>max_eta_ratio;
		cout<<"maximum learning rate ratio: "<<max_eta_ratio<<endl;
		getline(input_file,str,':');
		input_file>>N_tr;
		cout<<"size of training dataset: "<<N_tr<<endl;
		getline(input_file,str,':');
		input_file>>reg_f;
		cout<<"regularization factor: "<<reg_f<<endl;
		getline(input_file,str,':');
		input_file>>num_threads;
		cout<<"number of threads: "<<num_threads<<endl;
		getline(input_file,str,':');
		input_file>>str1;
		remove_spaces_ends(str1);
		cout<<"test network after each epoch: "<<str1<<endl;
		if (str1=="yes") test=true;
		getline(input_file,str,':');
		input_file>>str1;
		remove_spaces_ends(str1);
		cout<<"test dataset: "<<str1<<endl;
		if (str1=="EVAL") test_data=EVAL;
		else if (str1=="TRAINING") test_data=TRAINING;
		else test_data=TEST;
		getline(input_file,str,':');
		input_file>>pl_lgth_max;
		cout<<"maximum plateau length: "<<pl_lgth_max<<endl;
	}
	else
	{
		cout<<"input file could not be opened\n";
	}
	
	
	omp_set_num_threads(num_threads);
	
	network NN(layers);
	 
	struct timespec start, finish;
	double elapsed;
	
	clock_gettime(CLOCK_MONOTONIC, &start);
	
//stochastic gradient descent using the cross-entropy cost function and matrix optimizations in the backpropagation routine
	NN.SGD_CE_mat(N_epochs_max, batch_size, eta_init, test, test_data, N_tr, reg_f, pl_lgth_max, red_f_eta, max_eta_ratio);
//this version uses more loops instead of matrix-wise operations
//	NN.SGD_CE(N_epochs, batch_size, eta, test, test_data, N_tr, reg_f);
//this version uses the quadratic cost function
//	NN.SGD(N_epochs, batch_size, eta, test, test_data, N_tr, reg_f);
	
	clock_gettime(CLOCK_MONOTONIC, &finish);
	
	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1.0e9;
	
	cout<<"elapsed time: "<<elapsed<<" seconds.\n"<<endl;
	
	NN.test_network(TEST);
	
	
	return 0;
}

void remove_spaces_front(string &str)
{
	int j=0;
	while (str[j]==' ' || str[j]=='\t') j++;
	str=str.substr(j);
}

void remove_spaces_back(string &str)
{
	int j=str.size()-1;
	while (str[j]==' ' || str[j]=='\t') j--;
	str.resize(j+1);
}

void remove_spaces_ends(string &str)
{
	remove_spaces_front(str);
	remove_spaces_back(str);
}

