//                              -*- Mode: C++ -*-
// Filename         : remd_typedef.H
// Description      : 
// Author           : Kentaro Nomura
// Created On       : Tue Mar  5 11:49:27 2013
// Last Modified By : Minoru Oikawa
// Last Modified On : 2013-10-28 16:53:51
// Update Count     : 0
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#ifndef REMD_TYPEDEF
#define REMD_TYPEDEF

#include "switch_float_double.H"

/* phisical constant. */
const double Boltzmann_constant = 1.3806488e-23;  // "Boltzmann constant" [J/K]
const double Kb = Boltzmann_constant;             // "Boltzmann constant" [J/K]
const double Avogadro_constant  = 6.02214078e+23;  // "Avogadro constant"  [1/mol]
const double Na = Avogadro_constant;

/* Units in Simulation world. */
const double UNIT_LENGTH = 1.0e-9;      // = 1.0[nm]
const double UNIT_TIME   = 1.0e-12;     // = 1.0[psec]
const double UNIT_MASS   = 1.0e-3 / Na; // = mol[kg]
const double UNIT_ENERGY = 1.0 / Na;    // = mol[J]

const int LEN_FILENAME = 256;

#define SMEM_COUNT 2048    // shared memory array count 
const int MAX_GPU = 2048;
const int MAX_NMOL = 2048;
const int REDUCTION_SIZE = SMEM_COUNT;  // GPU reduction size. must be 2^N.
const int MAX_NREP = 16384;  // upper bound of replica counts.


typedef enum {
  H2D,
  D2H
} CopyKind_t;

typedef struct Remd {
    int       Nmol;          // Number of atoms in a replica.
    int       Nrep;          // Number of all replicas.
    Real_t    dens;          // density
    Real_t    mass;          // mass; common for all atoms.
    int      *sort_temper;   // scrambled sequetial number.
    int      *acc_ratio;     //
    
    Real3_t  *h_pos_ar;      // [Nmol]      , partial
    Real3_t  *h_vel_ar;      // [Nmol]      , partial
    Real3_t  *h_foc_ar;      // [Nmol]      , partial
    Real_t   *h_mass_ar;     // [Nmol]      , partial
    Real_t   *h_temp_ar;     // [Nrep]      , all
    int      *h_exch_ar;     // [Nrep]      , all
    Real_t   *h_energy;      // [Nrep][step_exch] , all
    Real_t   *h_temp_meas;   // [step_exch], partial
  
    Real3_t **d_pos_ar;      // [Ngpu][Nrep_1dev * Nmol]
    Real3_t **d_vel_ar;      // [Ngpu][Nrep_1dev * Nmol]
    Real3_t **d_foc_ar;      // [Ngpu][Nrep_1dev * Nmol]
    Real_t  **d_temp_ar;     // [Ngpu][Nrep_1dev]
    int     **d_exch_ar;
    Real_t  **d_energy;      // [Ngpu][Nrep_1dev * step_exch]
    Real_t  **d_temp_meas;   // [Ngpu][Nrep_1dev * step_exch]
  
    Real_t    cellsize;      // periodic boundary condition.
    Real_t    rcut;          // cutoff distance.
    Real_t    lj_sigma;      // "sigma"   in Lennard-Jones potential.
    Real_t    lj_epsilon;    // "epsilon" in ...

    Real_t    temp_max;      //
    Real_t    temp_min;
} Remd_t;
//==============================================================================
typedef struct Simu {
  int    Ngpu;          // Number of all GPUs.
  int    Nrep_1dev;     // Number of replicas in each GPUs.
  int   *which_dev;
  int   *offset_dev;
  long   step_max;      // simulation steps counts.
  long   step_exch;     //
  long   step_ene;
  Real_t dt;            // delta time
  double simclock; // simulation-world clock.
  int    simstep;  // integrated simulation step count.
  Real_t ene_max;       //
  Real_t ene_min;       //
  Real_t delta_ene;     //
  int    histo_bins;    // Count of Pins of histogram.
  int    report_posi;    // 1: generate position cdv file.
  int    report_velo;    // 1: generate verlocity cdv file.
  int    report_force;   // 1: generate force cdv file.
  int    report_temp;    // 1: generate temperature file.
  int    report_ene;     // 1: generate energy file.
  int    report_rcut;
  int    shuffle_atom;
  int    shuffle_replica;
  char   ifile[LEN_FILENAME];
  
  /* output */
  char   odir[LEN_FILENAME];
  char   odir_histogram[1024];
  char   ofile_init_temp[LEN_FILENAME];
  char   ofile_init_posi[LEN_FILENAME];
  char   ofile_init_velo[LEN_FILENAME];
  char   ofile_init_force[LEN_FILENAME];
} Simu_t;
//------------------------------------------------------------------------------
const char REMD_PREFIX[]  = "rep";
const char HOST_DIR[]  = "host";
const char POSI_PREFIX[]  = "p";
const char VELO_PREFIX[]  = "v";
const char FORCE_PREFIX[] = "f";
const char TEMP_DIR[]     = "temp";
const char TEMP_PREFIX[]  = "t";
const char LJ_POTEN_FILE[]= "lj_potential.dat";
const char TEMP_FILE[]   = "temp_history.dat";
const char SORT_FILE[]   = "temp_sorted.rep";
const char ENE_PREFIX[]  = "e";

static char fmt_save_ene[256] = "%s/%s%02d.dat";

#endif
//--- remd_typedef.H
