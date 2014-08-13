#include "pymangle.cc"

struct Image {
    double *x;
    double *y;
    double *freq;
    double *wave;
    double ***image;
    double ***unc;
    int nx;
    int ny;
    int nfreq;

    Image(double ***_image, int _nx, int _ny, int _nfreq) {
        image = _image; nx = _nx; ny = _ny; nfreq = _nfreq;
    }
};

extern "C" {
    Image* new_Image(double *_image, int nx, int ny, int nfreq) {

        double ***image = pymangle(ny, nx, nfreq, _image);

        return new Image(image, nx, ny, nfreq);
    }

    void set_xy(Image *I, double *x, double *y) {
        I->x = x; I->y = y;
    }

    void set_freq(Image *I, double *freq, double *wave) {
        I->freq = freq; I->wave = wave;
    }

    void set_unc(Image *I, double *_unc) {
        
        double ***unc = pymangle(I->ny, I->nx, I->nfreq, _unc);

        I->unc = unc;
    }

    void delete_Image(Image *I) {
        delete I;
    }
}
