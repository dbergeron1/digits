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
	int N_epochs;
	int batch_size;
	double eta;
	bool test=false;
	data_type_T test_data;
	int N_tr;
	double reg_f;
	
	ifstream input_file("../../digits.dat");
	
	string str, str1;
	
	if (input_file)
	{
	//	input_file.getline(input, Nhl_str.size()+1);
		getline(input_file,str,':');
		input_file>>Nhl;
		cout<<"number of hidden layers: "<<Nhl<<endl;
		layers.resize(Nhl);
		getline(input_file,str);
		getline(input_file,str,':');
		cout<<"hidden layers sizes: ";
		for (int i=0; i<Nhl; i++)
		{
			input_file>>layers[i];
			cout<<layers[i]<<" ";
		}
		cout<<endl;
		getline(input_file,str);
		getline(input_file,str,':');
		input_file>>N_epochs;
		cout<<"number of epochs: "<<N_epochs<<endl;
		getline(input_file,str);
		getline(input_file,str,':');
		input_file>>batch_size;
		cout<<"batch size: "<<batch_size<<endl;
		getline(input_file,str);
		getline(input_file,str,':');
		input_file>>eta;
		cout<<"learning rate: "<<eta<<endl;
		getline(input_file,str);
		getline(input_file,str,':');
		input_file>>N_tr;
		cout<<"size of training dataset: "<<N_tr<<endl;
		getline(input_file,str);
		getline(input_file,str,':');
		input_file>>reg_f;
		cout<<"regularization factor: "<<reg_f<<endl;
		getline(input_file,str);
		getline(input_file,str,':');
		input_file>>num_threads;
		cout<<"number of threads: "<<num_threads<<endl;
		getline(input_file,str);
		getline(input_file,str,':');
		input_file>>str1;
		remove_spaces_ends(str1);
		cout<<"test network after each epoch: "<<str1<<endl;
		if (str1=="yes") test=true;
		getline(input_file,str);
		getline(input_file,str,':');
		input_file>>str1;
		remove_spaces_ends(str1);
		cout<<"test dataset: "<<str1<<endl;
		if (str1=="EVAL") test_data=EVAL;
		else if (str1=="TRAINING") test_data=TRAINING;
		else test_data=TEST;
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
	
	NN.SGD_CE_mat(N_epochs, batch_size, eta, test, test_data, N_tr, reg_f);
//	NN.SGD_CE(N_epochs, batch_size, eta, test, test_data, N_tr, reg_f);
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

