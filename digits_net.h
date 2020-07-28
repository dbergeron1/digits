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
	network(){};
	
	
	void load_dataset(string label_file_name, string image_file_name, vector<uint8_t> &labels, vector<vector<uint8_t>> &images);
	void load_data();
	
	int N_training;
	vector<uint8_t> training_labels, test_labels;
	vector<vector<uint8_t>> training_images, test_images;
	
	
};

void network::load_data()
{
	string label_file("../../MNIST/train-labels-idx1-ubyte");
	string image_file("../../MNIST/train-images-idx3-ubyte");
	load_dataset(label_file, image_file, training_labels, training_images);
	
	label_file="../../MNIST/t10k-labels-idx1-ubyte";
	image_file="../../MNIST/t10k-images-idx3-ubyte";
	load_dataset(label_file, image_file, training_labels, training_images);
	
}



void network::load_dataset(string label_file_name, string image_file_name, vector<uint8_t> &labels, vector<vector<uint8_t>> &images)
{
	ifstream label_file(label_file_name,ios::binary);
	
	if (!label_file)
	{
		cerr << "File could not be opened." << endl;
		exit( EXIT_FAILURE );
	}
	
	int32_t mn, N_labels, N_images, N_rows, N_cols, N_pixels;
	
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
	
	cout<<"mn: "<<mn<<endl;
	cout<<"N_images: "<<N_images<<endl;
	cout<<"N_rows: "<<N_rows<<endl;
	cout<<"N_cols: "<<N_cols<<endl<<endl;
	
	
	int j, k;
	images.clear();
	images.resize(N_images);
	
	int itest=4;
	
	for (i=0; i<N_images; i++)
	{
		images[i].resize(N_rows*N_cols);
		for (j=0; j<N_rows; j++)
		{
			for (k=0; k<N_cols; k++)
			{
				image_file.read( reinterpret_cast< char * >( &images[i][k+j*N_cols]), 1);
				if (i==itest) cout<<setw(5)<<(int)images[i][k+j*N_cols];
			}
			if (i==itest) cout<<endl;
		}
	}
	
}




#endif /* digits_net_h */
