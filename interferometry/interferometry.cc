#include "pymangle.cc"

struct Visibilities {
    double *u;
    double *v;
    double *freq;
    double **real;
    double **imag;
    double **weights;
    double *uvdist;
    double **amp;
    double **phase;
    int nuv;
    int nfreq;

    Visibilities(double *_u, double *_v, double *_freq, double **_real, 
            double **_imag, double **_weights, double *_uvdist, 
            double **_amp, double **_phase, int _nuv, int _nfreq) {

        u = _u; v = _v; freq = _freq; real = _real; imag = _imag;
        weights = _weights; uvdist = _uvdist; amp = _amp; phase = _phase;
        nuv = _nuv; nfreq = _nfreq;
    }
};

extern "C" {
    Visibilities* new_Visibilities(double *u, double *v, double *freq, 
            double *_real, double *_imag, double *_weights, double *uvdist,
            double *_amp, double *_phase, int nuv, int nfreq) {

        double **real = pymangle(nuv, nfreq, _real);
        double **imag = pymangle(nuv, nfreq, _imag);
        double **weights = pymangle(nuv, nfreq, _weights);
        double **amp = pymangle(nuv, nfreq, _amp);
        double **phase = pymangle(nuv, nfreq, _phase);

        return new Visibilities(u, v, freq, real, imag, weights, uvdist, 
                amp, phase, nuv, nfreq);
    }

    void delete_Visibilities(Visibilities *V) {
        delete V;
    }
}
