//
//  digits_main.cpp
//  digits
//
//  Created by Dom2 on 20-07-28.
//

#include "digits_net.h"

int main()
{
	vector<int> layers{100};
	
	network NN(layers);
	
	clock_t t1=clock();
	
	int N_epochs=10;
	int batch_size=10;
	int eta=2.0;
	bool test=true;
	
	NN.SGD(N_epochs, batch_size, eta, test);
	
	t1=clock()-t1;
	cout<<((double)t1)/CLOCKS_PER_SEC<<" seconds for "<<N_epochs<<" epochs."<<endl;
	
	NN.test_network();
	
	return 0;
}

