//                              -*- Mode: C++ -*-
// Filename         : comm_save.cu
// Description      : REMD project
// Author           : Kentaro Nomura
// Created On       : 2013-03-05 11:49:27
// Last Modified By : Minoru Oikawa
// Last Modified On : 2013-10-24 18:08:38
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//-----------------------------------------------------------------------------
#include "switch_float_double.H"
#include "mytools.H"
#include "remd_typedef.H"
#include "init_cond.H"
#include "integ.H"
#include "comm_save.H"

extern Remd_t remd;  /* Only one instance of Remd_t in this program. */
extern Simu_t simu;  /* Only one instance of Simu_t in this program. */
static void copyReal3(Real3_t *host, Real3_t *dev, int rep_i, CopyKind_t dir);

//==============================================================================
// description:
//     save data of temperature at each replica.
//     filename is {t0000, t0001, t0002, ...}
//------------------------------------------------------------------------------
void
saveLocalTemp(Remd_t &remd)
{
  static int seq_num = 0;
  FILE *fp;
  char path_filename[1024];

  sprintf(path_filename, "%s/%s/%s%04d.dat", simu.odir, TEMP_DIR, TEMP_PREFIX, seq_num);
  if ((fp = fopen(path_filename, "w")) == NULL) {
    die("%s file open failed.", path_filename);
  }
  for (int i = 0; i < remd.Nrep; i++) {
    fprintf(fp, "h_temp_ar[%04d] = %f (K)\n", i, remd.h_temp_ar[i]);
  }
  fclose(fp);
  seq_num++;
}
void
saveMeasTemp(Remd_t &remd)
{
  static int seq_num = 0;
  FILE *fp;
  char path_filename[1024];

  sprintf(path_filename, "%s/%s/%s%04d.dat", simu.odir, TEMP_DIR, TEMP_PREFIX, seq_num);
  if ((fp = fopen(path_filename, "w")) == NULL) {
    die("file open failed.");
  }
  for (int i = 0; i < remd.Nrep; i++) {
    fprintf(fp, "h_temp_ar[%04d] = %f (K)\n", i, remd.h_temp_ar[i]);
  }
  fclose(fp);
  seq_num++;
}

//==============================================================================
// saveFormat *  
//------------------------------------------------------------------------------
static void
saveFormatPos(const char *filename, Real_t temp, long simstep)
{
  const Real_t box = 0.5 * remd.cellsize;
  double simclock = simu.dt * (double)simstep;
  FILE *fp;
  Real3_t *pos_ar = remd.h_pos_ar;
  if ((fp = fopen(filename, "w")) == NULL) { die("fopen() NULL.\n"); }
  fprintf(fp, "# Position of molecular, UNIT_LENGTH is 1.0nm. \n");
  fprintf(fp, "# snapshot at simulation time = %f ps\n", simclock);
  fprintf(fp, "#             simulation step = %d steps\n", simstep);
  fprintf(fp, "# box_sx=%+f, box_sy=%+f, box_sz=%+f \n", -box, -box, -box);
  fprintf(fp, "# box_ex=%+f, box_ey=%+f, box_ez=%+f \n", box, box, box);
  fprintf(fp, "# st0=\"time:%.3fps (%d steps)\", st0_pos=(-2.5, -2.5)\n", simclock, simstep);
  fprintf(fp, "# st1=\"temp: %7.3f [K]\", st1_pos=(-2.5, -2.1)\n", temp); 
  fprintf(fp, "#\n");
  for (int i = 0; i < remd.Nmol; i++) {
    fprintf(fp, "%5d 0 %+7.6f %+7.6f %+7.6f\n", i, pos_ar[i].x, pos_ar[i].y, pos_ar[i].z);
  }    
  fprintf(fp, "# EOF\n");
  fclose(fp);
}
static void
saveFormatVel(const char *filename, long simstep)
{
  double simclock = simu.dt * (double)simstep;
  FILE *fp;
  Real3_t *vel_ar = remd.h_vel_ar;
  if ((fp = fopen(filename, "w")) == NULL) { die("fopen() NULL.\n"); }
  fprintf(fp, "# Velocity of molecular, UNIT_LENGTH is 1nm. UNIT_TIME is 1ps\n");
  fprintf(fp, "# snapshot at simulation time = %f ps\n", simclock);
  fprintf(fp, "#             simulation step = %d steps\n", simstep);
  for (int i = 0; i < remd.Nmol; i++) {
    fprintf(fp, "%5d 0 %+7.6f %+7.6f %+7.6f\n", i, vel_ar[i].x, vel_ar[i].y, vel_ar[i].z);
  }    
  fclose(fp);
}
static void
saveFormatForce(const char *filename, const Real_t *potential_ar, long simstep)
{
  double simclock = simu.dt * (double)simstep;
  FILE *fp;
  Real3_t *foc_ar = remd.h_foc_ar;
  if ((fp = fopen(filename, "w")) == NULL) { die("fopen() NULL.\n"); }
  fprintf(fp, "# Force of molecular, UNIT_LENGTH is 1nm. UNIT_TIME is 1ps\n");
  fprintf(fp, "# snapshot at simulation time = %f ps\n", simclock);
  fprintf(fp, "#             simulation step = %d steps\n", simstep);
  for (int i = 0; i < remd.Nmol; i++) {
      fprintf(fp, "%5d 0 %+e %+e %+e", i, foc_ar[i].x, foc_ar[i].y, foc_ar[i].z);
      if (potential_ar==NULL) {
	  fprintf(fp, " \n");
      }
      else {
	  fprintf(fp, " # %+e\n", potential_ar[i]);
      }
  }    
  fclose(fp);
}
static void
saveFormatEne(const char *filename, int rep_i, int seq_num, long simstep)
{
  double simclock;
  int offset = simu.step_exch * rep_i;
  FILE *fp;
  if (seq_num == 0) {
    if ((fp = fopen(filename, "w")) == NULL) { die("fopen() NULL.\n"); }
    fprintf(fp, "# Energy of each time step, UNIT_ENERGY is J/mol, UNIT_TIME is 1ps\n");
  } else {
    if ((fp = fopen(filename, "a")) == NULL) { die("fopen() NULL.\n"); }
  }
  for (long t = 0; t < simu.step_exch; t++) {
    simclock = simu.dt * (double)(simstep*simu.step_exch + t);
    fprintf(fp, "%5d %4d %e %+e\n", seq_num, t, simclock, remd.h_energy[offset + t]);
  }
  fclose(fp);
}
//===============================================================================
// save "remd.h_temp_meas[step_exch]" to file.
static void
saveFormatTempMeas(const char *filename, int seq_num, long simstep)
{
  FILE *fp;
  double simclock;
  if (seq_num == 0) {
    if ((fp = fopen(filename, "w")) == NULL) { die("fopen() NULL.\n"); }
    fprintf(fp, "# Measured Temperature at each time step, UNIT_ENERGY is J/mol, UNIT_TIME is 1ps\n");
  } else {
    if ((fp = fopen(filename, "a")) == NULL) { die("fopen() NULL.\n"); }
  }
  for (int t = 0; t < simu.step_exch; t++) {
    simclock = simu.dt * (double)(simstep * simu.step_exch + t);
    fprintf(fp, "%5d %4d %e %f\n", seq_num, t, simclock, remd.h_temp_meas[t]);
  }
  fclose(fp);
}
//===============================================================================
static void
saveFormatTemp(const char *filename, int seq_num, long simstep)
{
  double simclock = simu.dt * (double)(simstep*simu.step_exch);
  FILE *fp;
  if (seq_num == 0) {
    if ((fp = fopen(filename, "w")) == NULL) { die("fopen() NULL.\n"); }
    fprintf(fp, "# Temperture of each time step, UNIT_TIME is 1ps\n");
  } else {
    if ((fp = fopen(filename, "a")) == NULL) { die("fopen() NULL.\n"); }
  }
  fprintf(fp, "%5d %e  ", seq_num, simclock);
  for (int rep_i = 0; rep_i < remd.Nrep; rep_i++) {
    fprintf(fp, "%10.6f  ", remd.h_temp_ar[rep_i]);
  }
  fprintf(fp, "\n");
  fclose(fp);
}
//===============================================================================
static void
saveFormatSorted(const char *filename, int seq_num, long simstep)
{
  double simclock = simu.dt * (double)(simstep*simu.step_exch);
  FILE *fp;
  if (seq_num == 0) {
    if ((fp = fopen(filename, "w")) == NULL) { die("fopen() NULL.\n"); }
    fprintf(fp, "# Temperature sorted of each time step, UNIT_TIME is 1ps\n");
  } else {
    if ((fp = fopen(filename, "a")) == NULL) { die("fopen() NULL.\n"); }
  }
  fprintf(fp, "%5d %e  ", seq_num, simclock);
  for (int rep_i = 0; rep_i < remd.Nrep; rep_i++) {
    fprintf(fp, "%3d  ", remd.sort_temper[rep_i]);
  }
  fprintf(fp, "\n");
  fclose(fp);
}
//===============================================================================
// data copy for position of atoms.
//-------------------------------------------------------------------------------
void
savePosAll(int t)
{
    debug_print(5, "Entering %s(int t=%d)\n", __func__, t);
    for (int rep_i = 0; rep_i < remd.Nrep; rep_i++) {
	copyPos(rep_i, D2H);
	savePos(rep_i, -999.9, t);
    }
    debug_print(5, "Entering %s(int t=%d)\n", __func__, t);
}
void
saveVelAll(int t)
{
    for (int i=0; i<remd.Nrep; i++) {
	copyVel(i, D2H);
	saveVel(i, t);
    }
}
void
saveFocAll(int t)
{
    for (int i=0; i<remd.Nrep; i++) {
	copyFoc(i, D2H);
	saveFoc(i, t);
    }
}
void
saveTempMeasAll(int t)
{
  for (int rep_i = 0; rep_i < remd.Nrep; rep_i++) {
    copyTempMeas(rep_i, D2H);
    saveTempMeas(rep_i, t);
  }
}
void
savePos(int rep_i, Real_t temp, long simstep)
{
  static int seq_num[MAX_NREP] = {0};
  char savepath[1024];
  char filename[1024];

  if (rep_i >= remd.Nrep) { die("rep_i is too large."); }
  getPathToNthRemd(savepath, rep_i);
  sprintf(filename, "%s/%s%06d.cdv", savepath, POSI_PREFIX, seq_num[rep_i]);
  saveFormatPos(filename, temp, simstep);
  seq_num[rep_i]++;
}
void
saveVel(int rep_i, long simstep)
{
  static int seq_num[MAX_NREP] = {0};
  char savepath[1024];
  char filename[1024];

  if (rep_i >= remd.Nrep)  die("rep_i is too large.");
  getPathToNthRemd(savepath, rep_i);
  sprintf(filename, "%s/%s%06d.cdv", savepath, VELO_PREFIX, seq_num[rep_i]);
  saveFormatVel(filename, simstep);
  seq_num[rep_i]++;
}

void
saveFoc(int rep_i, long simstep, Real_t *potential_ar)
{
  static int seq_num[MAX_NREP] = {0};
  char savepath[1024];
  char filename[1024];

  if (rep_i >= remd.Nrep) { die("rep_i is too large."); }
  getPathToNthRemd(savepath, rep_i);
  sprintf(filename, "%s/%s%06d.cdv", savepath, FORCE_PREFIX, seq_num[rep_i]);
  saveFormatForce(filename, potential_ar, simstep);
  seq_num[rep_i]++;
}

void
saveLocalPos(Remd_t &remd, long simstep)
{
  static int seq_num = 0;
  char savepath[1024];
  char filename[LEN_FILENAME];
  getPathToLocal(savepath);
  sprintf(filename, "%s/%s%06d.cdv", savepath, POSI_PREFIX, seq_num);
  saveFormatPos(filename, 1000, simstep);
  seq_num++;
}

void
saveTempMeas(int rep_i, long simstep)
{
  static int seq_num[MAX_NREP] = {0};
  char savepath[1024];
  char filename[1024];

  if (rep_i >= remd.Nrep) { die("specified rep_i is too large."); }
  getPathToSaveRoot(savepath);
  sprintf(filename, "%s/temp%06d.meas", savepath, rep_i);
  saveFormatTempMeas(filename, seq_num[rep_i], simstep);
  seq_num[rep_i]++;
}

//==============================================================================
// position data copy.
//------------------------------------------------------------------------------
void
copyPos(int rep_i, CopyKind_t dir)
{
  int      dev_i = simu.which_dev[rep_i];
  Real3_t *d_pos_ar = remd.d_pos_ar[dev_i] + (remd.Nmol * simu.offset_dev[rep_i]);
  debug_print(6, "Entering %s(rep_i=%d)\n", __func__, rep_i);
  copyReal3(remd.h_pos_ar, d_pos_ar, rep_i, dir);
  debug_print(6, "Exiting  %s(rep_i=%d)\n", __func__, rep_i);
}
//==============================================================================
// velocity data copy.
//------------------------------------------------------------------------------
void
copyVel(int rep_i, CopyKind_t dir)
{
    int      dev_i = simu.which_dev[rep_i];
    Real3_t *d_vel_ar = remd.d_vel_ar[dev_i] + (remd.Nmol * simu.offset_dev[rep_i]);
    copyReal3(remd.h_vel_ar, d_vel_ar, rep_i, dir);
}
//==============================================================================
// Force data copy.
//------------------------------------------------------------------------------
void
copyFoc(int rep_i, CopyKind_t dir)
{
  int      dev_i = simu.which_dev[rep_i];
  Real3_t *d_foc_ar = remd.d_foc_ar[dev_i] + (remd.Nmol * simu.offset_dev[rep_i]);
  copyReal3(remd.h_foc_ar, d_foc_ar, rep_i, dir);
}

void
saveEne(Remd_t &remd, long simstep)
{
  static int seq_num = 0;
  char savepath[1024];
  char filename[1024];
  getPathToSaveRoot(savepath);
  for (int rep_i = 0; rep_i < remd.Nrep; rep_i++) {
    sprintf(filename, fmt_save_ene, savepath, ENE_PREFIX, rep_i);
    saveFormatEne(filename, rep_i, seq_num, simstep);
  }
  seq_num++;
}
//==============================================================================
void
copyTempMeas(int rep_i, CopyKind_t dir)
{
  int     dev_i = simu.which_dev[rep_i];
  Real_t *host  = remd.h_temp_meas;
  Real_t *dev   = remd.d_temp_meas[dev_i] + (simu.step_exch * simu.offset_dev[rep_i]);
  int     size  = sizeof(Real_t) * simu.step_exch;

#if defined(HOST_RUN)
  switch (dir) {
  case H2D:
    memcpy(dev, host, size);
    break;
  case D2H:
    memcpy(host, dev, size);
    break;
  default:
    die("Unknown parameter detected.");    
  }
#elif defined(DEVICE_RUN)
  cudaError_t err;
  err = cudaSetDevice(dev_i);
  if (err != cudaSuccess) { die("cudaSetDevice()\n"); }

  switch (dir) {
  case H2D:
    err = cudaMemcpy((void*)dev, (void*)host, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { die("cudaMemcpy(H2D)\n"); }
    break;
  case D2H:
    err = cudaMemcpy((void*)host, (void*)dev, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      printf("CUDA ERROR on dev#%d: %s\n", dev_i, cudaGetErrorString(err));
      die("cudaMemcpy(D2H)\n"); }
    break;
  default:
    die("Unknown parameter detected.");    
  }
#else
  die("undefined HOST_RUN or DEVICE_RUN.\n");
#endif
} //copyTempMeas()

//==============================================================================
//
//------------------------------------------------------------------------------
void saveLocalVel(Remd_t &remd, long simstep)
{
  static int seq_num = 0;
  char savepath[1024];
  char filename[LEN_FILENAME];
  getPathToLocal(savepath);
  sprintf(filename, "%s/%s%06d.cdv", savepath, VELO_PREFIX, seq_num);
  saveFormatVel(filename, simstep);
  seq_num++;
}
//==============================================================================
//
//------------------------------------------------------------------------------
void saveLocalFoc(Remd_t &remd, Real_t *potential_ar, int Nmol, long simstep)
{
  static int seq_num = 0;
  char savepath[1024];
  char filename[LEN_FILENAME];
  getPathToLocal(savepath);
  sprintf(filename, "%s/%s%06d.cdv", savepath, FORCE_PREFIX, seq_num);
  saveFormatForce(filename, potential_ar, simstep);
  seq_num++;
}
/**
 * [Name] copyEne_dev()
 * [description]
 */
void
copyEnergy(CopyKind_t dir, Remd_t &remd, Simu_t &simu)
{
  debug_print(6, "Entering %s\n", __func__);

  int     step_exch  = simu.step_exch;
  int     copysize = sizeof(Real_t) * step_exch * simu.Nrep_1dev;
  Real_t *h_energy, *d_energy;
#if defined(HOST_RUN)
  switch (dir) {
  case H2D:
    for (int gpu_i = 0; gpu_i < simu.Ngpu; gpu_i++) {
      h_energy = remd.h_energy + (gpu_i * step_exch * simu.Nrep_1dev);
      d_energy = remd.d_energy[gpu_i];
      memcpy(d_energy, h_energy, copysize);
    }
    break;
  case D2H:
    for (int gpu_i = 0; gpu_i < simu.Ngpu; gpu_i++) {
      h_energy = remd.h_energy + (gpu_i * step_exch * simu.Nrep_1dev);
      d_energy = remd.d_energy[gpu_i];
      memcpy(h_energy, d_energy, copysize);
    }
    break;
  default:
    die("Unknown CopyKind_t\n");
  }
#elif defined(DEVICE_RUN)
  cudaError_t err[2];
  switch (dir) {
  case H2D:
    for (int gpu_i = 0; gpu_i < simu.Ngpu; gpu_i++) {
      h_energy = remd.h_energy + (gpu_i * step_exch * simu.Nrep_1dev);
      d_energy = remd.d_energy[gpu_i];
      err[0] = cudaSetDevice(gpu_i);
      err[1] = cudaMemcpy(d_energy, h_energy, copysize, cudaMemcpyHostToDevice);
      if (err[0] != cudaSuccess) { die("cudaSetDevice(%d).\n", gpu_i); }
      if (err[1] != cudaSuccess) { die("cudaMemcpy().\n"); }
    }
    break;
  case D2H:
    for (int gpu_i = 0; gpu_i < simu.Ngpu; gpu_i++) {
      h_energy = remd.h_energy + (gpu_i * step_exch * simu.Nrep_1dev);
      d_energy = remd.d_energy[gpu_i];
      err[0] = cudaSetDevice(gpu_i);
      err[1] = cudaMemcpy(h_energy, d_energy, copysize, cudaMemcpyDeviceToHost);
      if (err[0] != cudaSuccess) { die("cudaSetDevice(%d).\n", gpu_i); }
      if (err[1] != cudaSuccess) { die("cudaMemcpy().\n"); }
    }
    break;
  default:
    die("Unknown CopyKind_t\n");
  }
#else
  die("undefined HOST_RUN or DEVICE_RUN.\n");
#endif
  debug_print(6, "Exiting %s\n", __func__);
}
//==============================================================================
// Name:  copyTempTarg()
// description:
//     copy the value of temperature at all replicas from host to device,
// status:
//
//------------------------------------------------------------------------------
void
copyTempTarg(CopyKind_t dir)
{
    printf("<--- Entering %s()\n", __func__); fflush(stdout);

  Real_t *h_temp_ar;
  Real_t *d_temp_ar;
  const int copysize = sizeof(Real_t) * simu.Nrep_1dev;

#if defined(HOST_RUN)
  switch (dir) {
  case H2D:
    for (int gpu_i = 0; gpu_i < simu.Ngpu; gpu_i++) {
      h_temp_ar = remd.h_temp_ar + (gpu_i * simu.Nrep_1dev);
      d_temp_ar = remd.d_temp_ar[gpu_i];
      memcpy(d_temp_ar, h_temp_ar, copysize);
    }
    break;
  case D2H:
    for (int gpu_i = 0; gpu_i < simu.Ngpu; gpu_i++) {
      h_temp_ar = remd.h_temp_ar + (gpu_i * simu.Nrep_1dev);
      d_temp_ar = remd.d_temp_ar[gpu_i];
      memcpy(h_temp_ar, d_temp_ar, copysize);
    }
    break;
  default:
    die("Unknown CopyKind_t\n");
  }
#elif defined(DEVICE_RUN)
  cudaError_t err[2];
  switch (dir) {
  case H2D:
    for (int gpu_i = 0; gpu_i < simu.Ngpu; gpu_i++) {
      h_temp_ar = remd.h_temp_ar + (gpu_i * simu.Nrep_1dev);
      d_temp_ar = remd.d_temp_ar[gpu_i];
#     if 1 //monitor copied data
      for (int i=0; i<simu.Nrep_1dev; i++) {
	  printf("h_temp_ar[%d]= %f\n", i, h_temp_ar[i]);
      }
      fflush(stdout);
#     endif
      err[0] = cudaSetDevice(gpu_i);
      err[1] = cudaMemcpy(d_temp_ar, h_temp_ar, copysize, cudaMemcpyHostToDevice);
      if (err[0] != cudaSuccess) { die("cudaSetDevice().\n"); }
      if (err[1] != cudaSuccess) {
	printf("CUDA ERROR on dev#%d: %s\n", gpu_i, cudaGetErrorString(err[1]));
	die("cudaMemcpy().\n"); }
    }
    break;
  case D2H:
    for (int gpu_i = 0; gpu_i < simu.Ngpu; gpu_i++) {
      h_temp_ar = remd.h_temp_ar + (gpu_i * simu.Nrep_1dev);
      d_temp_ar = remd.d_temp_ar[gpu_i];
      err[0] = cudaSetDevice(gpu_i);
      err[1] = cudaMemcpy(h_temp_ar, d_temp_ar, copysize, cudaMemcpyDeviceToHost);
      if (err[0] != cudaSuccess) { die("cudaSetDevice().\n"); }
      if (err[1] != cudaSuccess) { die("cudaMemcpy().\n"); }
    }
    break;
  default:
    die("Unknown CopyKind_t\n");
  }
#else
  die("undefined HOST_RUN or DEVICE_RUN.\n");
#endif
  printf("---> Exiting  %s()\n", __func__); fflush(stdout);
}


void
saveTempTarg(Remd_t &remd, long simstep)
{
  static int seq_num = 0;
  char savepath[1024];
  char filename[1024];
  getPathToSaveRoot(savepath);
  sprintf(filename, "%s/%s", savepath, TEMP_FILE);
  saveFormatTemp(filename, seq_num, simstep);
  seq_num++;
} // saveTempTarg()
void
saveSorted(Remd_t &remd, long simstep)
{
  static int seq_num = 0;
  char savepath[1024];
  char filename[1024];
  getPathToSaveRoot(savepath);
  sprintf(filename, "%s/%s", savepath, SORT_FILE);
  saveFormatSorted(filename, seq_num, simstep);
  seq_num++;
} // saveSorted()

void copyExch(CopyKind_t dir, Remd_t &remd, Simu_t &simu)
{
    printf("<--- Entering %s\n", __func__);fflush(stdout);

  int *h_exch_ar;
  int *d_exch_ar;
  const int copysize = sizeof(int) * simu.Nrep_1dev;
#if defined(HOST_RUN)
  switch (dir) {
  case H2D:
    for (int gpu_i = 0; gpu_i < simu.Ngpu; gpu_i++) {
      h_exch_ar = remd.h_exch_ar + (gpu_i * simu.Nrep_1dev);
      d_exch_ar = remd.d_exch_ar[gpu_i];
      memcpy(d_exch_ar, h_exch_ar, copysize);
    }
    break;
  case D2H:
    for (int gpu_i = 0; gpu_i < simu.Ngpu; gpu_i++) {
      h_exch_ar = remd.h_exch_ar + (gpu_i * simu.Nrep_1dev);
      d_exch_ar = remd.d_exch_ar[gpu_i];
      memcpy(h_exch_ar, d_exch_ar, copysize);
    }
    break;
  default:
    die("Unknown CopyKind_t\n");
  }
#elif defined(DEVICE_RUN)
  cudaError_t err;
  switch (dir) {
  case H2D:
    for (int gpu_i = 0; gpu_i < simu.Ngpu; gpu_i++) {
      h_exch_ar = remd.h_exch_ar + (gpu_i * simu.Nrep_1dev);
      d_exch_ar = remd.d_exch_ar[gpu_i];
      err = cudaSetDevice(gpu_i);
      if (err != cudaSuccess) { die("cudaSetDevice(%d). returned %d. %s\n",
				    gpu_i, err, cudaGetErrorString(err)); }

      err = cudaMemcpy(d_exch_ar, h_exch_ar, copysize, cudaMemcpyHostToDevice);
      if (err != cudaSuccess) { die("dev(%d), cudaMemcpy() returned %d. %s\n",
				    gpu_i, err, cudaGetErrorString(err)); }
    }
    break;
  case D2H:
    for (int gpu_i = 0; gpu_i < simu.Ngpu; gpu_i++) {
      h_exch_ar = remd.h_exch_ar + (gpu_i * simu.Nrep_1dev);
      d_exch_ar = remd.d_exch_ar[gpu_i];
      err = cudaSetDevice(gpu_i);
      if (err != cudaSuccess) { die("cudaSetDevice().\n"); }

      err = cudaMemcpy(h_exch_ar, d_exch_ar, copysize, cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) { die("dev(%d), cudaMemcpy(). %s\n",
				       gpu_i, cudaGetErrorString(err)); }
    }
    break;
  default:
    die("Unknown CopyKind_t\n");
  }
#else
  die("undefined HOST_RUN or DEVICE_RUN.\n");
#endif
  printf("---> Exiting  %s\n", __func__);fflush(stdout);
}
//===============================================================================
// core function of "copyPos", "copyVel", and "copyFoc".
//===============================================================================
static
int checkSum(void *targ, int size) {
    int sum=0;
    int *ptr = (int *)targ;
    for (int s=0; s<size; s+=sizeof(int)) {
	//printf("ptr[%d]= %d\n", s, *ptr);
	sum += *ptr;
	ptr++;
    }
    return sum;
}

static void
copyReal3(Real3_t *host, Real3_t *dev, int rep_i, CopyKind_t dir)
{
    int size = sizeof(Real3_t) * remd.Nmol;

#if defined(HOST_RUN)
    switch (dir) {
      case H2D:
	memcpy(dev, host, size);
	break;
      case D2H:
	memcpy(host, dev, size);
	break;
      default:
	die("Unknown parameter detected.");    
    }
#elif defined(DEVICE_RUN)
    int dev_i = simu.which_dev[rep_i];
    cudaError_t err;
    err = cudaSetDevice(dev_i);
    if (err != cudaSuccess) { die("cudaSetDevice()\n"); }

    switch (dir) {
      case H2D:
	err = cudaMemcpy((void*)dev, (void*)host, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { die("cudaMemcpy(H2D)\n"); }
	break;
      case D2H:
	err = cudaMemcpy((void*)host, (void*)dev, size, cudaMemcpyDeviceToHost);
#if 0
	printf("memcpyD2H(ph=%p, size=%d, checksum(D2H)=0x%08x\n", host, size, checkSum(host, size));
	fflush(stdout);
#endif
	if (err != cudaSuccess) {
	    printf("CUDA ERROR on dev#%d: %s\n", dev_i, cudaGetErrorString(err));
	    die("cudaMemcpy(D2H)\n"); }
	break;
      default:
	die("Unknown parameter detected.");    
    }
#else
    die("undefined HOST_RUN or DEVICE_RUN.\n");
#endif
}
// comm_save.cu
