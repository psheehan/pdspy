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

/* Function to correctly deallocate the VisibilitiesObject class. */

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

/* Function to create a new instance of the VisibilitiesObject class. */

static PyObject *VisibilitiesObject_new(PyTypeObject *type, 
        PyObject *args, PyObject *kwds) {
    VisibilitiesObject *self;

    self = (VisibilitiesObject *)type->tp_alloc(type, 0);

    return (PyObject *)self;
}

/* Function to initialize the VisibilitiesObject class correctly. */

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
        Py_INCREF(u);
        self->u = u;
        self->V->u = (double *)PyArray_DATA(u);
    }
    if (v) {
        Py_INCREF(v);
        self->v = v;
        self->V->v = (double *)PyArray_DATA(v);
    }
    if (freq) {
        Py_INCREF(freq);
        self->freq = freq;
        self->V->freq = (double *)PyArray_DATA(freq);
    }
    if (real) {
        Py_INCREF(real);
        self->real = real;
        if (PyArray_AsCArray((PyObject **)&real, (void **)&self->V->real,
                dims, 2, descr) < 0)
            return NULL;
    }
    if (imag) {
        Py_INCREF(imag);
        self->imag = imag;
        if (PyArray_AsCArray((PyObject **)&imag, (void **)&self->V->imag,
                dims, 2, descr) < 0)
            return NULL;
    }
    if (weights) {
        Py_INCREF(weights);
        self->weights = weights;
        if (PyArray_AsCArray((PyObject **)&weights, (void **)&self->V->weights,
                dims, 2, descr) < 0)
            return NULL;
    }
    if (uvdist) {
        Py_INCREF(uvdist);
        self->uvdist = uvdist;
        self->V->uvdist = (double *)PyArray_DATA(uvdist);
    }
    if (amp) {
        Py_INCREF(amp);
        self->amp = amp;
        if (PyArray_AsCArray((PyObject **)&amp, (void **)&self->V->amp,
                dims, 2, descr) < 0)
            return NULL;
    }
    if (phase) {
        Py_INCREF(phase);
        self->phase = phase;
        if (PyArray_AsCArray((PyObject **)&phase, (void **)&self->V->phase,
                dims, 2, descr) < 0)
            return NULL;
    }

    return 0;
}

static PyMemberDef VisibilitiesObject_members[] = {
    {NULL}
};

/* Below we define a set of getter and setter functions for a whole bunch of the
 * variables so that we can ensure that when their value is changed in Python,
 * the corresponding value in the Visibilities C++ class is also changed. */

/* Functions to get and set the value of u. */

static PyArrayObject *VisibilitiesObject_getu(VisibilitiesObject *self, 
        void *closure) {
    Py_INCREF(self->u);
    return self->u;
}

static int VisibilitiesObject_setu(VisibilitiesObject *self, 
        PyArrayObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the u attribute");
        return -1;
    }

    self->V->u = (double *)PyArray_DATA(value);

    Py_XDECREF(self->u);
    Py_INCREF(value);
    self->u = value;

    return 0;
}

/* Functions to get and set the value of v. */

static PyArrayObject *VisibilitiesObject_getv(VisibilitiesObject *self, 
        void *closure) {
    Py_INCREF(self->v);
    return self->v;
}

static int VisibilitiesObject_setv(VisibilitiesObject *self, 
        PyArrayObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the v attribute");
        return -1;
    }

    self->V->v = (double *)PyArray_DATA(value);

    Py_XDECREF(self->v);
    Py_INCREF(value);
    self->v = value;

    return 0;
}

/* Functions to get and set the value of freq. */

static PyArrayObject *VisibilitiesObject_getfreq(VisibilitiesObject *self, 
        void *closure) {
    Py_INCREF(self->freq);
    return self->freq;
}

static int VisibilitiesObject_setfreq(VisibilitiesObject *self, 
        PyArrayObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the freq attribute");
        return -1;
    }

    self->V->freq = (double *)PyArray_DATA(value);

    Py_XDECREF(self->freq);
    Py_INCREF(value);
    self->freq = value;

    return 0;
}

/* Functions to get and set the value of real. */

static PyArrayObject *VisibilitiesObject_getreal(VisibilitiesObject *self, 
        void *closure) {
    Py_INCREF(self->real);
    return self->real;
}

static int VisibilitiesObject_setreal(VisibilitiesObject *self, 
        PyArrayObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the real attribute");
        return -1;
    }

    PyArray_Descr *descr = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims = PyArray_SHAPE(value);
    if (PyArray_AsCArray((PyObject **)&value, (void **)&self->V->real, 
            dims, 2, descr) < 0)
        return NULL;

    Py_XDECREF(self->real);
    Py_INCREF(value);
    self->real = value;

    return 0;
}

/* Functions to get and set the value of imag. */

static PyArrayObject *VisibilitiesObject_getimag(VisibilitiesObject *self, 
        void *closure) {
    Py_INCREF(self->imag);
    return self->imag;
}

static int VisibilitiesObject_setimag(VisibilitiesObject *self, 
        PyArrayObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the imag attribute");
        return -1;
    }

    PyArray_Descr *descr = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims = PyArray_SHAPE(value);
    if (PyArray_AsCArray((PyObject **)&value, (void **)&self->V->imag, 
            dims, 2, descr) < 0)
        return NULL;

    Py_XDECREF(self->imag);
    Py_INCREF(value);
    self->imag = value;

    return 0;
}

/* Functions to get and set the value of weights. */

static PyArrayObject *VisibilitiesObject_getweights(VisibilitiesObject *self, 
        void *closure) {
    Py_INCREF(self->weights);
    return self->weights;
}

static int VisibilitiesObject_setweights(VisibilitiesObject *self, 
        PyArrayObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the weights attribute");
        return -1;
    }

    PyArray_Descr *descr = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims = PyArray_SHAPE(value);
    if (PyArray_AsCArray((PyObject **)&value, (void **)&self->V->weights, 
            dims, 2, descr) < 0)
        return NULL;

    Py_XDECREF(self->weights);
    Py_INCREF(value);
    self->weights = value;

    return 0;
}

/* Functions to get and set the value of uvdist. */

static PyArrayObject *VisibilitiesObject_getuvdist(VisibilitiesObject *self, 
        void *closure) {
    Py_INCREF(self->uvdist);
    return self->uvdist;
}

static int VisibilitiesObject_setuvdist(VisibilitiesObject *self, 
        PyArrayObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the uvdist attribute");
        return -1;
    }

    self->V->uvdist = (double *)PyArray_DATA(value);

    Py_XDECREF(self->uvdist);
    Py_INCREF(value);
    self->uvdist = value;

    return 0;
}

/* Functions to get and set the value of amp. */

static PyArrayObject *VisibilitiesObject_getamp(VisibilitiesObject *self, 
        void *closure) {
    Py_INCREF(self->amp);
    return self->amp;
}

static int VisibilitiesObject_setamp(VisibilitiesObject *self, 
        PyArrayObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the amp attribute");
        return -1;
    }

    PyArray_Descr *descr = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims = PyArray_SHAPE(value);
    if (PyArray_AsCArray((PyObject **)&value, (void **)&self->V->amp, 
            dims, 2, descr) < 0)
        return NULL;

    Py_XDECREF(self->amp);
    Py_INCREF(value);
    self->amp = value;

    return 0;
}

/* Functions to get and set the value of phase. */

static PyArrayObject *VisibilitiesObject_getphase(VisibilitiesObject *self, 
        void *closure) {
    Py_INCREF(self->phase);
    return self->phase;
}

static int VisibilitiesObject_setphase(VisibilitiesObject *self, 
        PyArrayObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the phase attribute");
        return -1;
    }

    PyArray_Descr *descr = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp *dims = PyArray_SHAPE(value);
    if (PyArray_AsCArray((PyObject **)&value, (void **)&self->V->phase, 
            dims, 2, descr) < 0)
        return NULL;

    Py_XDECREF(self->phase);
    Py_INCREF(value);
    self->phase = value;

    return 0;
}

/* And an array which passes those functions to VisibilitiesObject */

static PyGetSetDef VisibilitiesObject_getseters[] = {
    {"u", (getter)VisibilitiesObject_getu, 
        (setter)VisibilitiesObject_setu, "u part of the visibilities",
        NULL},
    {"v", (getter)VisibilitiesObject_getv, 
        (setter)VisibilitiesObject_setv, "v part of the visibilities",
        NULL},
    {"freq", (getter)VisibilitiesObject_getfreq, 
        (setter)VisibilitiesObject_setfreq, "freq part of the visibilities",
        NULL},
    {"real", (getter)VisibilitiesObject_getreal, 
        (setter)VisibilitiesObject_setreal, "real part of the visibilities",
        NULL},
    {"imag", (getter)VisibilitiesObject_getimag, 
        (setter)VisibilitiesObject_setimag, "imag part of the visibilities",
        NULL},
    {"weights", (getter)VisibilitiesObject_getweights, 
        (setter)VisibilitiesObject_setweights, 
        "weights part of the visibilities", NULL},
    {"uvdist", (getter)VisibilitiesObject_getuvdist, 
        (setter)VisibilitiesObject_setuvdist, "uvdist part of the visibilities",
        NULL},
    {"amp", (getter)VisibilitiesObject_getamp, 
        (setter)VisibilitiesObject_setamp, "amp part of the visibilities",
        NULL},
    {"phase", (getter)VisibilitiesObject_getphase, 
        (setter)VisibilitiesObject_setphase, "phase part of the visibilities",
        NULL},
    {NULL}
};

/* Below we define any methods that the VisibilitiesObject class should
 * have and be accessible from Python. */

static PyObject *VisibilitiesObject_test(VisibilitiesObject *self) {
    printf("%f", self->V->real[1][1]);

    return Py_BuildValue("");
}

static PyMethodDef VisibilitiesObject_methods[] = {
    {"test", (PyCFunction)VisibilitiesObject_test, METH_NOARGS, 
        "A test function for the VisibilitiesObject class."},
    {NULL}
};

/* Below we create the VisibilitiesObjectType, which makes all of the functions
 * defined above actually part of the VisibilitiesObject class and therefore
 * usable from Python. */

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
    VisibilitiesObject_getseters,
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
