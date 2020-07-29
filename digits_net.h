//
//  digits_net.h
//  digits
//
//  Created by Dom2 on 20-07-28.
//

#ifndef digits_net_h
#define digits_net_h

#include <iostream>
#include <cstdio>
#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>
#include "armadillo"
#include "graph_3D.h"

using namespace std;
using namespace arma;

int ReverseInt (int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1=i&255;
	ch2=(i>>8)&255;
	ch3=(i>>16)&255;
	ch4=(i>>24)&255;
	return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

class network
{
public:
	network(const vector<int> &layers);
	
	void load_dataset(string label_file_name, string image_file_name, vector<uint8_t> &labels, vector<mat> &images);
	void load_data();
	mat activation_func(const mat &z);
	mat diff_activation_func(const mat &z);
	mat feedforward(const mat &a_in);
	void shuffle_data();
	void SGD(int N_epochs, int batch_size, double eta, bool test);
	void update_batch(vector<uint8_t> labels, vector<mat> images, double eta);
	void backprop(uint8_t label, const mat &image, vector<mat> &delta_grad_w, vector<mat> &delta_grad_b);
	mat diff_cost_func(mat a_out, uint8_t label);
	void test_network();
	int digit(mat a_out);
	
	vector<int> layer_sizes;
	int N_layers;
	vector<mat> weights;
	vector<mat> biases;
	
	int N_training, N_test;
	int N_pixels;
	vector<uint8_t> training_labels, test_labels;
	vector<mat> training_images, test_images;
};

//layers contains the sizes of the hidden layers
network::network(const vector<int> &layers)
{
	load_data();
	
	int i;
	
	N_layers=layers.size()+2;
	layer_sizes.resize(N_layers);
	layer_sizes[0]=N_pixels;
	for (i=0; i<layers.size(); i++) layer_sizes[i+1]=layers[i];
	layer_sizes[N_layers-1]=10;
	
	weights.resize(N_layers-1);
	biases.resize(N_layers-1);
	
	for (i=0; i<N_layers-1; i++)
	{
		biases[i].randn(layer_sizes[i+1],1);
		weights[i].randn(layer_sizes[i+1],layer_sizes[i]);
	}
}

void network::test_network()
{
	int i, N_success=0;
	mat a_out;
	for (i=0; i<N_test; i++)
	{
		a_out=feedforward(test_images[i]);
		if (digit(a_out)==(int)test_labels[i]) N_success++;
	}
	cout<<"N_success / N_test: "<<N_success<<" / "<<N_test<<endl;
}

int network::digit(mat a_out)
{
	int i=0, j=-1;
	while (i<10 && a_out(i)<0.5) i++;
	if (i<10) j=i;
	
	return j;
}

void network::SGD(int N_epochs, int batch_size, double eta, bool test)
{
	int i, j, k;
	
	vector<uint8_t> labels(batch_size);
	vector<mat> images(batch_size);
	
	int N_batch=N_training/batch_size;
	
	for (i=0; i<N_epochs; i++)
	{
		shuffle_data();
		for (j=0; j<N_batch; j++)
		{
			for (k=0; k<batch_size; k++)
			{
				labels[k]=training_labels[k+j*batch_size];
				images[k]=training_images[k+j*batch_size];
			}
			update_batch(labels, images, eta);
		}
		if (test)
		{
			test_network();
			cout<<"epoch "<<i<<"  "<<endl;
		}
	}
	
}

void network::update_batch(vector<uint8_t> labels, vector<mat> images, double eta)
{
	int i, j;
	
	int batch_size=labels.size();
	
	vector<mat> grad_w(N_layers-1), delta_grad_w;
	vector<mat> grad_b(N_layers-1), delta_grad_b;
	for (i=0; i<N_layers-1; i++)
	{
		grad_w[i].zeros(weights[i].size());
		grad_b[i].zeros(biases[i].size());
	}
	delta_grad_w=grad_w;
	delta_grad_b=grad_b;
	
	for (i=0; i<labels.size(); i++)
	{
		backprop(labels[i],images[i],delta_grad_w,delta_grad_b);
		for (j=0; j<N_layers-1; j++)
		{
			grad_w[j]=grad_w[j]+delta_grad_w[j];
			grad_b[j]=grad_b[j]+delta_grad_b[j];
		}
	}
	
	for (j=0; j<N_layers-1; j++)
	{
		weights[j]=weights[j]-(eta/batch_size)*grad_w[j];
		biases[j]=biases[j]-(eta/batch_size)*grad_b[j];
	}
	
}

void network::backprop(uint8_t label, const mat &image, vector<mat> &delta_grad_w, vector<mat> &delta_grad_b)
{
	int i,j;
	for (i=0; i<N_layers-1; i++)
	{
		delta_grad_w[i].zeros();
		delta_grad_b[i].zeros();
	}
	
	mat activation=image;
	mat z;
	vector<mat> activations(N_layers);
	activations[0]=activation;
	vector<mat> zs(N_layers-1);
	for (i=0; i<N_layers-1; i++)
	{
		z=weights[i]*activation+biases[i];
		zs[i]=z;
		activation=activation_func(z);
		activations[i+1]=activation;
	}
	mat delta=diff_cost_func(activation, label) % diff_activation_func(z);
	delta_grad_b[N_layers-2]=delta;
	delta_grad_w[N_layers-2]=delta*activations[N_layers-2].t();
	mat dafz;
	for (i=N_layers-3; i>=0; i--)
	{
		z=zs[i];
		dafz=diff_activation_func(z);
		delta=(weights[i+1].t()*delta) % dafz;
		delta_grad_b[i]=delta;
		delta_grad_w[i]=delta*activations[i].t();
	}
	
	
}

mat network::diff_cost_func(mat a_out, uint8_t label)
{
	mat y=zeros(a_out.size());
	y(label)=1.0;
	
	mat dcf=a_out-y;
	
	return dcf;
}

void network::shuffle_data()
{
	Col<int> indices=linspace<Col<int>>(0,N_training-1,N_training);
	Col<int> sh_indices=shuffle(indices);
	
	vector<uint8_t> labels=training_labels;
	vector<mat> images=training_images;
	for (int i=0; i<N_training; i++)
	{
		training_labels[i]=labels[sh_indices[i]];
		training_images[i]=images[sh_indices[i]];
	}
}


mat network::activation_func(const mat &z)
{
	mat afz=1.0/(1.0+exp(-z));
	return afz;
}

mat network::diff_activation_func(const mat &z)
{
	mat afz=activation_func(z);
	mat dafz=afz % (1.0-afz);
	return dafz;
}

mat network::feedforward(const mat &a_in)
{
	int i;
	
	mat row_ones=ones(1,a_in.n_cols);
	mat biases_mat;
	
	mat a=a_in;
	for (i=0; i<N_layers-1; i++)
	{
		biases_mat=biases[i]*row_ones;
		a=activation_func(weights[i]*a+biases_mat);
	}
	
	return a;
}


void network::load_data()
{
	string label_file("../../MNIST/train-labels-idx1-ubyte");
	string image_file("../../MNIST/train-images-idx3-ubyte");
	load_dataset(label_file, image_file, training_labels, training_images);
	
	N_training=training_labels.size();
	
	label_file="../../MNIST/t10k-labels-idx1-ubyte";
	image_file="../../MNIST/t10k-images-idx3-ubyte";
	load_dataset(label_file, image_file, test_labels, test_images);
	
	N_test=test_labels.size();
	
}

void network::load_dataset(string label_file_name, string image_file_name, vector<uint8_t> &labels, vector<mat> &images)
{
	ifstream label_file(label_file_name,ios::binary);
	
	if (!label_file)
	{
		cerr << "File could not be opened." << endl;
		exit( EXIT_FAILURE );
	}
	
	int32_t mn, N_labels, N_images, N_rows, N_cols;
	
	label_file.read( reinterpret_cast< char * >( &mn ), 4 );
	label_file.read( reinterpret_cast< char * >( &N_labels ), 4 );
	
	mn=ReverseInt(mn);
	N_labels=ReverseInt(N_labels);
	
	cout<<"mn: "<<mn<<endl;
	cout<<"N_labels: "<<N_labels<<endl;
	
	labels.clear();
	labels.resize(N_labels);
	
	cout<<setiosflags(ios::left);
	
	int i;
	for (i=0; i<N_labels; i++)
	{
		label_file.read( reinterpret_cast< char * >( &labels[i] ), 1);
		if (i<10) cout<<setw(10)<<i<<(int)labels[i]<<endl;
	}
	
	ifstream image_file(image_file_name,ios::binary);
	
	image_file.read(reinterpret_cast< char * >( &mn ), 4 );
	image_file.read(reinterpret_cast< char * >( &N_images ), 4 );
	image_file.read(reinterpret_cast< char * >( &N_rows ), 4 );
	image_file.read(reinterpret_cast< char * >( &N_cols ), 4 );
	
	mn=ReverseInt(mn);
	N_images=ReverseInt(N_images);
	N_rows=ReverseInt(N_rows);
	N_cols=ReverseInt(N_cols);
	N_pixels=N_rows*N_cols;
	
	cout<<"mn: "<<mn<<endl;
	cout<<"N_images: "<<N_images<<endl;
	cout<<"N_rows: "<<N_rows<<endl;
	cout<<"N_cols: "<<N_cols<<endl<<endl;
	
	
	int j, k;
	images.clear();
	images.resize(N_images);
	
	uint8_t u;
	
	int itest=4;
	
	cout<<setprecision(2);
	
	for (i=0; i<N_images; i++)
	{
		images[i].zeros(N_pixels,1);
		for (j=0; j<N_rows; j++)
		{
			for (k=0; k<N_cols; k++)
			{
				image_file.read( reinterpret_cast< char * >(&u), 1);
				images[i][k+j*N_cols]=((double)u)/255;
				if (i==itest) cout<<setw(5)<<images[i][k+j*N_cols];
			}
			if (i==itest) cout<<endl;
		}
	}
	
}




#endif /* digits_net_h */
