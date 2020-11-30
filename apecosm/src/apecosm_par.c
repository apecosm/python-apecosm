#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject* apecosm_compute_par(PyObject* self, PyObject *args, PyObject *kw) {
   //return Py_BuildValue("s", "Hello, Python extensions!!");
   Py_RETURN_NONE;
}

static char compute_par_docs[] = "compute_par( ): Computation of PAR from CHL, and light using PISCES RGB algorithm\n";

static PyMethodDef apecosm_funcs[] = {
   {"compute_par", (PyCFunction)apecosm_compute_par, METH_VARARGS | METH_KEYWORDS, compute_par_docs},
   {NULL, NULL, 0, NULL }   // sentinel, compulsory!
};

static struct PyModuleDef apecosm = {
    PyModuleDef_HEAD_INIT,
    "apecosm",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    apecosm_funcs
};



//void initApecosm(void) {
//   Py_InitModule3("apecosm", apecosm_funcs, "Extension module example!");
//}