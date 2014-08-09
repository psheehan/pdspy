#include <Python.h>
#include "structmember.h"
#include <numpy/arrayobject.h>

/* Here we define the Visibilities object, which is a strictly C++ 
 * representation of interferometric visibilities. We also define a number of
 * routines for working with those visibilties in a strictly C++ way. */

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

    Visibilities() {
    }

    Visibilities(double *_u, double *_v, double *_freq, double **_real, 
            double **_imag, double **_weights, double *_uvdist, 
            double **_amp, double **_phase, int _nuv, int _nfreq) {

        u = _u; v = _v; freq = _freq; real = _real; imag = _imag;
        weights = _weights; uvdist = _uvdist; amp = _amp; phase = _phase;
        nuv = _nuv; nfreq = _nfreq;
    }
};

/* Below we create a Python Class called VisibilitiesObject, which will serve
 * as the link between the Python Visibilities class and the C++ 
 * Visibilities struct. */

typedef struct {
    PyObject_HEAD
    PyArrayObject *u, *v, *freq, *real, *imag, *weights, *uvdist, *amp, *phase;
    Visibilities *V;
} VisibilitiesObject;

static void VisibilitiesObject_dealloc(VisibilitiesObject *self) {
    Py_XDECREF(self->u);
    Py_XDECREF(self->v);
    Py_XDECREF(self->freq);
    Py_XDECREF(self->real);
    Py_XDECREF(self->imag);
    Py_XDECREF(self->weights);
    Py_XDECREF(self->uvdist);
    Py_XDECREF(self->amp);
    Py_XDECREF(self->phase);
    delete self->V;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *VisibilitiesObject_new(PyTypeObject *type, 
        PyObject *args, PyObject *kwds) {
    VisibilitiesObject *self;

    self = (VisibilitiesObject *)type->tp_alloc(type, 0);

    return (PyObject *)self;
}

static int VisibilitiesObject_init(VisibilitiesObject *self, 
        PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"u","v","freq","real","imag","weights","uvdist",
        "amp","phase",NULL};

    PyArrayObject *u=NULL, *v=NULL, *freq=NULL, *real=NULL, *imag=NULL, 
            *weights=NULL, *uvdist=NULL, *amp=NULL, *phase=NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOOOO", kwlist, 
            &u, &v, &freq, &real, &imag, &weights, &uvdist, &amp, &phase))
        return -1;

    self->V = new Visibilities();
    PyArray_Descr *descr = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims = PyArray_SHAPE(real);

    self->V->nuv = dims[0];
    self->V->nfreq = dims[1];

    if (u) {
        self->u = u;
        Py_INCREF(u);
        self->V->u = (double *)PyArray_DATA(u);
    }
    if (v) {
        self->v = v;
        Py_INCREF(v);
        self->V->v = (double *)PyArray_DATA(v);
    }
    if (freq) {
        self->freq = freq;
        Py_INCREF(freq);
        self->V->freq = (double *)PyArray_DATA(freq);
    }
    if (real) {
        self->real = real;
        Py_INCREF(real);
        if (PyArray_AsCArray((PyObject **)&real, (void **)&self->V->real,
                dims, 2, descr) < 0)
            return NULL;
    }
    if (imag) {
        self->imag = imag;
        Py_INCREF(imag);
        if (PyArray_AsCArray((PyObject **)&imag, (void **)&self->V->imag,
                dims, 2, descr) < 0)
            return NULL;
    }
    if (weights) {
        self->weights = weights;
        Py_INCREF(weights);
        if (PyArray_AsCArray((PyObject **)&weights, (void **)&self->V->weights,
                dims, 2, descr) < 0)
            return NULL;
    }
    if (uvdist) {
        self->uvdist = uvdist;
        Py_INCREF(uvdist);
        self->V->uvdist = (double *)PyArray_DATA(uvdist);
    }
    if (amp) {
        self->amp = amp;
        Py_INCREF(amp);
        if (PyArray_AsCArray((PyObject **)&amp, (void **)&self->V->amp,
                dims, 2, descr) < 0)
            return NULL;
    }
    if (phase) {
        self->phase = phase;
        Py_INCREF(phase);
        if (PyArray_AsCArray((PyObject **)&phase, (void **)&self->V->phase,
                dims, 2, descr) < 0)
            return NULL;
    }

    return 0;
}

static PyMemberDef VisibilitiesObject_members[] = {
    {"u", T_OBJECT_EX, offsetof(VisibilitiesObject, u), 0, "Visibilities.u"},
    {"v", T_OBJECT_EX, offsetof(VisibilitiesObject, v), 0, "Visibilities.v"},
    {"freq", T_OBJECT_EX, offsetof(VisibilitiesObject, freq), 0, 
        "Visibilities.freq"},
    {"real", T_OBJECT_EX, offsetof(VisibilitiesObject, real), 0, 
        "Visibilities.real"},
    {"imag", T_OBJECT_EX, offsetof(VisibilitiesObject, imag), 0, 
        "Visibilities.imag"},
    {"weights", T_OBJECT_EX, offsetof(VisibilitiesObject, weights), 0, 
        "Visibilities.weights"},
    {"uvdist", T_OBJECT_EX, offsetof(VisibilitiesObject, uvdist), 0, 
        "Visibilities.uvdist"},
    {"amp", T_OBJECT_EX, offsetof(VisibilitiesObject, amp), 0, 
        "Visibilities.amp"},
    {"phase", T_OBJECT_EX, offsetof(VisibilitiesObject, phase), 0, 
        "Visibilities.phase"},
    {NULL}
};

static PyObject *VisibilitiesObject_test(VisibilitiesObject *self) {
    printf("%f", self->V->u[2]);

    return Py_BuildValue("");
}

static PyMethodDef VisibilitiesObject_methods[] = {
    {"test", (PyCFunction)VisibilitiesObject_test, METH_NOARGS, 
        "A test function for the VisibilitiesObject class."},
    {NULL}
};

static PyTypeObject VisibilitiesObjectType = {
    PyVarObject_HEAD_INIT(NULL,0)
    "test.Test",
    sizeof(VisibilitiesObject),
    0,
    (destructor)VisibilitiesObject_dealloc,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "Visibilities objects",
    0,
    0,
    0,
    0,
    0,
    0,
    VisibilitiesObject_methods,
    VisibilitiesObject_members,
    0,
    0,
    0,
    0,
    0,
    0,
    (initproc)VisibilitiesObject_init,
    0,
    VisibilitiesObject_new,
};

/* Below is the code which links the Python interferometry library with the 
 * C++ libinterferometry library and turns the libinterferometry library
 * into a Python module. */

static PyMethodDef libinterferometryMethods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef libinterferometry = {
    PyModuleDef_HEAD_INIT,
    "libinterferometry",
    "Module to wrap C++ routines for working with interferometry data.",
    -1,
    libinterferometryMethods
};

PyMODINIT_FUNC PyInit_libinterferometry(void) {
    PyObject *m = PyModule_Create(&libinterferometry);

    VisibilitiesObjectType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&VisibilitiesObjectType) < 0)
        return NULL;

    Py_INCREF(&VisibilitiesObjectType);
    PyModule_AddObject(m, "VisibilitiesObject", 
            (PyObject *)&VisibilitiesObjectType);

    import_array();

    return m;
}
