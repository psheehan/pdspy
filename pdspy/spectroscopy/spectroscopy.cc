struct Spectrum {
    double *wave;
    double *flux;
    double *unc;
    int nwave;

    Spectrum(double *_wave, double *_flux, double *_unc, int _nwave) {
        wave = _wave; flux = _flux; unc = _unc; nwave = _nwave;
    }
};

extern "C" {
    Spectrum* new_Spectrum(double *wave, double *flux, double *unc, int nwave) {
        return new Spectrum(wave, flux, unc, nwave);
    }

    void delete_Spectrum(Spectrum *S) {
        delete S;
    }
}
