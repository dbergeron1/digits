
/* 
 file includeDef.h
 
*/

#ifndef INCLUDEDEF_H
#define INCLUDEDEF_H

#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdio>
#include <complex>
#include <string>
#include <cstring>
#include <ctime>
#include <limits>
#include <random>
#include <sys/stat.h>
#include "armadillo"
//#include "fftw3.h"
//#include <gsl/gsl_linalg.h>


#ifndef PI
#define PI acos((double)-1.0)
#endif

#ifndef EPSILON
#define EPSILON numeric_limits<double>::epsilon()
#endif

#ifndef DBL_MIN
#define DBL_MIN numeric_limits<double>::min()
#endif

#ifndef DBL_MAX
#define DBL_MAX numeric_limits<double>::max()
#endif

#ifndef INF
#define INF numeric_limits<double>::infinity()
#endif

using namespace std;

typedef complex<double> dcomplex;
typedef unsigned int  uint;

#endif
