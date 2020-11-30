#define PY_SSIZE_T_CLEAN
#include <Python.h>

//static PyObject* apecosm_compute_par(PyObject* self, PyObject *args, PyObject *kw) {
static PyObject* compute_par(PyObject* self, PyObject *args) {
   return Py_BuildValue("s", "Hello, Python extensions!!");
}

static char compute_par_docs[] = "compute_par(): Computation of PAR from CHL, and light using PISCES RGB algorithm\n";

static PyMethodDef apecosm_clib_methods[] = {
   //{"compute_par", apecosm_compute_par, METH_VARARGS | METH_KEYWORDS, compute_par_docs},
   {"compute_par", compute_par, METH_NOARGS, compute_par_docs},
   {NULL, NULL, 0, NULL }   // sentinel, compulsory!
};

static struct PyModuleDef apecosm_clib_module = {
    PyModuleDef_HEAD_INIT,
    "apecosm_clib",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    apecosm_clib_methods
};

PyMODINIT_FUNC PyInit_apecosm_clib(void) {
    Py_Initialize();
    return PyModule_Create(&apecosm_clib_module);
}