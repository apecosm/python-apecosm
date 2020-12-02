#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include "netcdf.h"

#define NROWS 4
#define NCOLS 61

#define get_chl(cell, z) (0)
#define get_qsr(cell) (0)
#define get_parfrac(cell) (0)
#define get_bottom(cell) (0)
#define get_deltaz(cell, k) (0)

// Dimnsion: CHL concentration; R/G/B
float zrgb[NROWS][NCOLS];

//static PyObject* apecosm_compute_par(PyObject* self, PyObject *args, PyObject *kw) {
static PyObject *compute_par(PyObject *self, PyObject *args, PyObject *kw) {
    //if (!PyArg_ParseTuple(args, "s", &command))
    //    return NULL;
    return Py_BuildValue("s", "Hello, Python extensions!!");
}

static char compute_par_docs[] = "compute_par(): Computation of PAR from CHL, and light using PISCES RGB algorithm\n";

static PyMethodDef apecosm_clib_methods[] = {
   //{"compute_par", apecosm_compute_par, METH_VARARGS | METH_KEYWORDS, compute_par_docs},
   {"compute_par", compute_par, METH_VARARGS | METH_KEYWORDS, compute_par_docs},
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


void init_zrgb(void) {

    // CHL                     // Blue                     // Green                    // Red
    zrgb[0][0] = 0.010, zrgb[1][0] = 0.01618, zrgb[2][0] = 0.07464, zrgb[3][0] = 0.37807;
    zrgb[0][1] = 0.011, zrgb[1][1] = 0.01654, zrgb[2][1] = 0.07480, zrgb[3][1] = 0.37823;
    zrgb[0][2] = 0.013, zrgb[1][2] = 0.01693, zrgb[2][2] = 0.07499, zrgb[3][2] = 0.37840;
    zrgb[0][3] = 0.014, zrgb[1][3] = 0.01736, zrgb[2][3] = 0.07518, zrgb[3][3] = 0.37859;
    zrgb[0][4] = 0.016, zrgb[1][4] = 0.01782, zrgb[2][4] = 0.07539, zrgb[3][4] = 0.37879;
    zrgb[0][5] = 0.018, zrgb[1][5] = 0.01831, zrgb[2][5] = 0.07562, zrgb[3][5] = 0.37900;
    zrgb[0][6] = 0.020, zrgb[1][6] = 0.01885, zrgb[2][6] = 0.07586, zrgb[3][6] = 0.37923;
    zrgb[0][7] = 0.022, zrgb[1][7] = 0.01943, zrgb[2][7] = 0.07613, zrgb[3][7] = 0.37948;
    zrgb[0][8] = 0.025, zrgb[1][8] = 0.02005, zrgb[2][8] = 0.07641, zrgb[3][8] = 0.37976;
    zrgb[0][9] = 0.028, zrgb[1][9] = 0.02073, zrgb[2][9] = 0.07672, zrgb[3][9] = 0.38005;
    zrgb[0][10] = 0.032, zrgb[1][10] = 0.02146, zrgb[2][10] = 0.07705, zrgb[3][10] = 0.38036;
    zrgb[0][11] = 0.035, zrgb[1][11] = 0.02224, zrgb[2][11] = 0.07741, zrgb[3][11] = 0.38070;
    zrgb[0][12] = 0.040, zrgb[1][12] = 0.02310, zrgb[2][12] = 0.07780, zrgb[3][12] = 0.38107;
    zrgb[0][13] = 0.045, zrgb[1][13] = 0.02402, zrgb[2][13] = 0.07821, zrgb[3][13] = 0.38146;
    zrgb[0][14] = 0.050, zrgb[1][14] = 0.02501, zrgb[2][14] = 0.07866, zrgb[3][14] = 0.38189;
    zrgb[0][15] = 0.056, zrgb[1][15] = 0.02608, zrgb[2][15] = 0.07914, zrgb[3][15] = 0.38235;
    zrgb[0][16] = 0.063, zrgb[1][16] = 0.02724, zrgb[2][16] = 0.07967, zrgb[3][16] = 0.38285;
    zrgb[0][17] = 0.071, zrgb[1][17] = 0.02849, zrgb[2][17] = 0.08023, zrgb[3][17] = 0.38338;
    zrgb[0][18] = 0.079, zrgb[1][18] = 0.02984, zrgb[2][18] = 0.08083, zrgb[3][18] = 0.38396;
    zrgb[0][19] = 0.089, zrgb[1][19] = 0.03131, zrgb[2][19] = 0.08149, zrgb[3][19] = 0.38458;
    zrgb[0][20] = 0.100, zrgb[1][20] = 0.03288, zrgb[2][20] = 0.08219, zrgb[3][20] = 0.38526;
    zrgb[0][21] = 0.112, zrgb[1][21] = 0.03459, zrgb[2][21] = 0.08295, zrgb[3][21] = 0.38598;
    zrgb[0][22] = 0.126, zrgb[1][22] = 0.03643, zrgb[2][22] = 0.08377, zrgb[3][22] = 0.38676;
    zrgb[0][23] = 0.141, zrgb[1][23] = 0.03842, zrgb[2][23] = 0.08466, zrgb[3][23] = 0.38761;
    zrgb[0][24] = 0.158, zrgb[1][24] = 0.04057, zrgb[2][24] = 0.08561, zrgb[3][24] = 0.38852;
    zrgb[0][25] = 0.178, zrgb[1][25] = 0.04289, zrgb[2][25] = 0.08664, zrgb[3][25] = 0.38950;
    zrgb[0][26] = 0.200, zrgb[1][26] = 0.04540, zrgb[2][26] = 0.08775, zrgb[3][26] = 0.39056;
    zrgb[0][27] = 0.224, zrgb[1][27] = 0.04811, zrgb[2][27] = 0.08894, zrgb[3][27] = 0.39171;
    zrgb[0][28] = 0.251, zrgb[1][28] = 0.05103, zrgb[2][28] = 0.09023, zrgb[3][28] = 0.39294;
    zrgb[0][29] = 0.282, zrgb[1][29] = 0.05420, zrgb[2][29] = 0.09162, zrgb[3][29] = 0.39428;
    zrgb[0][30] = 0.316, zrgb[1][30] = 0.05761, zrgb[2][30] = 0.09312, zrgb[3][30] = 0.39572;
    zrgb[0][31] = 0.355, zrgb[1][31] = 0.06130, zrgb[2][31] = 0.09474, zrgb[3][31] = 0.39727;
    zrgb[0][32] = 0.398, zrgb[1][32] = 0.06529, zrgb[2][32] = 0.09649, zrgb[3][32] = 0.39894;
    zrgb[0][33] = 0.447, zrgb[1][33] = 0.06959, zrgb[2][33] = 0.09837, zrgb[3][33] = 0.40075;
    zrgb[0][34] = 0.501, zrgb[1][34] = 0.07424, zrgb[2][34] = 0.10040, zrgb[3][34] = 0.40270;
    zrgb[0][35] = 0.562, zrgb[1][35] = 0.07927, zrgb[2][35] = 0.10259, zrgb[3][35] = 0.40480;
    zrgb[0][36] = 0.631, zrgb[1][36] = 0.08470, zrgb[2][36] = 0.10495, zrgb[3][36] = 0.40707;
    zrgb[0][37] = 0.708, zrgb[1][37] = 0.09056, zrgb[2][37] = 0.10749, zrgb[3][37] = 0.40952;
    zrgb[0][38] = 0.794, zrgb[1][38] = 0.09690, zrgb[2][38] = 0.11024, zrgb[3][38] = 0.41216;
    zrgb[0][39] = 0.891, zrgb[1][39] = 0.10374, zrgb[2][39] = 0.11320, zrgb[3][39] = 0.41502;
    zrgb[0][40] = 1.000, zrgb[1][40] = 0.11114, zrgb[2][40] = 0.11639, zrgb[3][40] = 0.41809;
    zrgb[0][41] = 1.122, zrgb[1][41] = 0.11912, zrgb[2][41] = 0.11984, zrgb[3][41] = 0.42142;
    zrgb[0][42] = 1.259, zrgb[1][42] = 0.12775, zrgb[2][42] = 0.12356, zrgb[3][42] = 0.42500;
    zrgb[0][43] = 1.413, zrgb[1][43] = 0.13707, zrgb[2][43] = 0.12757, zrgb[3][43] = 0.42887;
    zrgb[0][44] = 1.585, zrgb[1][44] = 0.14715, zrgb[2][44] = 0.13189, zrgb[3][44] = 0.43304;
    zrgb[0][45] = 1.778, zrgb[1][45] = 0.15803, zrgb[2][45] = 0.13655, zrgb[3][45] = 0.43754;
    zrgb[0][46] = 1.995, zrgb[1][46] = 0.16978, zrgb[2][46] = 0.14158, zrgb[3][46] = 0.44240;
    zrgb[0][47] = 2.239, zrgb[1][47] = 0.18248, zrgb[2][47] = 0.14701, zrgb[3][47] = 0.44765;
    zrgb[0][48] = 2.512, zrgb[1][48] = 0.19620, zrgb[2][48] = 0.15286, zrgb[3][48] = 0.45331;
    zrgb[0][49] = 2.818, zrgb[1][49] = 0.21102, zrgb[2][49] = 0.15918, zrgb[3][49] = 0.45942;
    zrgb[0][50] = 3.162, zrgb[1][50] = 0.22703, zrgb[2][50] = 0.16599, zrgb[3][50] = 0.46601;
    zrgb[0][51] = 3.548, zrgb[1][51] = 0.24433, zrgb[2][51] = 0.17334, zrgb[3][51] = 0.47313;
    zrgb[0][52] = 3.981, zrgb[1][52] = 0.26301, zrgb[2][52] = 0.18126, zrgb[3][52] = 0.48080;
    zrgb[0][53] = 4.467, zrgb[1][53] = 0.28320, zrgb[2][53] = 0.18981, zrgb[3][53] = 0.48909;
    zrgb[0][54] = 5.012, zrgb[1][54] = 0.30502, zrgb[2][54] = 0.19903, zrgb[3][54] = 0.49803;
    zrgb[0][55] = 5.623, zrgb[1][55] = 0.32858, zrgb[2][55] = 0.20898, zrgb[3][55] = 0.50768;
    zrgb[0][56] = 6.310, zrgb[1][56] = 0.35404, zrgb[2][56] = 0.21971, zrgb[3][56] = 0.51810;
    zrgb[0][57] = 7.079, zrgb[1][57] = 0.38154, zrgb[2][57] = 0.23129, zrgb[3][57] = 0.52934;
    zrgb[0][58] = 7.943, zrgb[1][58] = 0.41125, zrgb[2][58] = 0.24378, zrgb[3][58] = 0.54147;
    zrgb[0][59] = 8.912, zrgb[1][59] = 0.44336, zrgb[2][59] = 0.25725, zrgb[3][59] = 0.55457;
    zrgb[0][60] = 10.000, zrgb[1][60] = 0.47804, zrgb[2][60] = 0.27178, zrgb[3][60] = 0.56870;
}

void compute_par_c() {
    
    int nOceanCell=100, NLEVEL_OPA=75;

    // 1 = Blue, 2=Green, 3=Red
    double output[nOceanCell][NLEVEL_OPA];
    double ze1[nOceanCell][NLEVEL_OPA];  
    double ze2[nOceanCell][NLEVEL_OPA];
    double ze3[nOceanCell][NLEVEL_OPA];
    double ek1[nOceanCell][NLEVEL_OPA];
    double ek2[nOceanCell][NLEVEL_OPA];
    double ek3[nOceanCell][NLEVEL_OPA];

    // Computation of the irgb index value, obtained from CHL inputs
    // It is basically the row index
    // TODO: add the verificiation as in the p4zopt.F90
    // ma2d irgb(boost.extents[nOceanCell][NLEVEL_OPA]);
    for (int c = 0; c < nOceanCell; c++) {
        for (size_t k = 0; k < get_bottom(c); k++) {
            double chltemp = get_chl(c, k);
            chltemp = fmin(10., fmax(0.05, chltemp));
            double irgbtmp = 41. + 20. * log10(chltemp);
            int irgb = (int)(round(irgbtmp) - 1.0); // conversion from F index to C index (-1)
            double dz = get_deltaz(c, k);
            ek1[c][k] = zrgb[1][irgb] * dz;
            ek2[c][k] = zrgb[2][irgb] * dz;
            ek3[c][k] = zrgb[3][irgb] * dz;
        }
    }

    // init the surface attenuation
    for (int c = 0; c < nOceanCell; c++) {
        ze1[c][0] = get_qsr(c) * exp(-0.5 * ek1[c][0]);
        ze2[c][0] = get_qsr(c) * exp(-0.5 * ek2[c][0]);
        ze3[c][0] = get_qsr(c) * exp(-0.5 * ek3[c][0]);
    }

    for (int c = 0; c < nOceanCell; c++) {
        for (size_t k = 1; k < get_bottom(c); k++) {
            ze1[c][k] = ze1[c][k - 1] * exp(-0.5 * (ek1[c][k - 1] + ek1[c][k]));
            ze2[c][k] = ze2[c][k - 1] * exp(-0.5 * (ek2[c][k - 1] + ek2[c][k]));
            ze3[c][k] = ze3[c][k - 1] * exp(-0.5 * (ek3[c][k - 1] + ek3[c][k]));
        }
    }

    // Computation of the PAR as the average of the three values.
    for (int c = 0; c < nOceanCell; c++) {
        for (size_t k = 1; k < get_bottom(c); k++) {
            output[c][k] = (ze1[c][k] + ze2[c][k] + ze3[c][k]) / 3.;
        }
    }
    
    return;
    
}