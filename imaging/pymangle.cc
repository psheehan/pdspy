/*
Time-stamp: <pymangle.cc on Tuesday, 29 January, 2013 at 17:58:05 MST (philip)>
*/

#include <cstdio>
#include <cstdlib>
#include <cassert>

/*
  Three functions to take flat multidimensional arrays (arrays of one index) and
  their sets of dimensions from Python,
  returning a C pointer-to-pointer-to... object which can be referred to 
  as a multidimensional object, e.g. x[i][j][k]

  The only memory allocated is that holding the array pointers.
 */

// two-dimensional
template <class T>
T** pymangle(int nx, int ny, T* data_from_python) {
    T **m;
    m = (T**) malloc(sizeof(T*)*nx);
    assert( m != (T**)NULL );
    m[0] = data_from_python;
    for(int i=1; i<nx; i++) m[i] = m[i-1] + ny;
    return m;
}

template <class T>
void freepymangle(T** m) {
    free(m);
}

// three-dimensional
template <class T>
T*** pymangle(int nx, int ny, int nz, T* data_from_python) {
    T*** m;
    m = (T***) malloc(sizeof(T**)*nx);
    assert( m != (T***)NULL );
    m[0] = (T**) malloc(sizeof(T*)*nx*ny);
    assert( m[0] != (T**)NULL );
    m[0][0] = data_from_python;
    for(int j=1; j<ny; j++) m[0][j] = m[0][j-1] + nz;
    for(int i=1; i<nx; i++) {
        m[i] = m[i-1] + ny;
        m[i][0] = m[i-1][0] + ny*nz;
        for(int j=1; j<ny; j++) m[i][j] = m[i][j-1] + nz;
    }
    return m;
}

template <class T>
void freepymangle(T*** m) {
    free(m[0]);
    free(m);
}

// four-dimensional

template <class T>
T**** pymangle(int nx, int ny, int nz, int nq, T* data_from_python) {
    T**** m;
    m = (T****) malloc(sizeof(T***)*nx);
    assert( m != (T****)NULL );
    m[0] = (T***) malloc(sizeof(T**)*nx*ny);
    assert( m[0] != (T**)NULL );
    m[0][0] = (T**) malloc(sizeof(T*)*nx*ny*nz);
    assert( m[0][0] != (T**)NULL );
    m[0][0] = data_from_python;
    for(int j=1; j<ny; j++) m[0][j] = m[0][j-1] + nz;
    for(int i=1; i<nx; i++) {
        m[i] = m[i-1] + ny;
        m[i][0] = m[i-1][0] + ny*nz;
        for(int j=1; j<ny; j++) m[i][j] = m[i][j-1] + nz;
    }
    for(int k=1; k<nz; k++) m[0][0][k] = m[0][0][k-1] + nq;
    for(int j=1; j<ny; j++) {
        m[0][j][0] = m[0][j-1][0] + nq*nz;
        for(int k=1; k<nz; k++) m[0][j][k] = m[0][j][k-1] + nq;
    }
    for(int i=1; i<nx; i++) {
        m[i][0][0] = m[i-1][0][0] + nq*ny*nz;
        for(int k=1; k<nz; k++) m[i][0][k] = m[i][0][k-1] + nq;
        for(int j=1; j<ny; j++) {
            m[i][j][0] = m[i][j-1][0] + nq*nz;
            for(int k=1; k<nz; k++) m[i][j][k] = m[i][j][k-1] + nq;
        }
    }
    return m;
}
template <class T>
void freepymangle(T**** m) {
    free(m[0][0]);
    free(m[0]);
    free(m);
}

