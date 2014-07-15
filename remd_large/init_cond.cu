//                              -*- Mode: C++ -*-
// Filename         : initial_condition.cu
// Description      : 
// Author           : Kentaro Nomura
// Created On       : 2013-03-05 11:49:27
// Last Modified By : Minoru Oikawa
// Last Modified On : 2014-01-31 17:47:44
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#include "switch_float_double.H"
#include "mytools.H"
#include "remd_typedef.H"
#include "calc_force.H"
#include "init_cond.H"

extern Remd_t remd;
extern Simu_t simu;

static Real_t calcRcut(Real_t, int);
static Real_t calcCellsize(Real_t dens, Real_t mass, int Nmol);

static const char delim[] = ":;#= \t\n"; // simulation input file delimiter.
static const int  buflen  = 2048;        // byte length of line buffer.

//==============================================================================
// get the path to save tragectry.
//------------------------------------------------------------------------------
void getPathToSaveRoot(char *path) {
   sprintf(path, "%s", simu.odir);
}
void getPathToNthRemd(char *path, int rep_i) {
   sprintf(path, "%s/%s%04d", simu.odir, REMD_PREFIX, rep_i);  
}
void getPathToLocal(char *path) {
   sprintf(path, "%s/%s", simu.odir, HOST_DIR);  
}
//==============================================================================
// void initSimConfig()
// description:
//     extract a line including a parameter name "keyword"
//     from specified by "filename".
// status:
//     fixed 2013-08-30.
//------------------------------------------------------------------------------
void setOutputDirName(char *odir, char *ifile) {
   char *c;
   int  dot = '.';
   strncpy(odir, ifile, 256);
   if ((c = strrchr(odir, dot)) == NULL) {
      die("strrchar() returned NULL\n");
   }
   *c = '\0';
   strcat(odir, ".out");
}

void initSimConfig(int argc, char **argv) {
  char c;
  int opt_flag = 0;
  char each_remd_path[1024];
  char temp_dirname[1024];
  char host_dirname[1024];
  char lj_poten_file[1024];
  char lj_force_file[1024];

  
  // Parse command line options. //
  while ((c = getopt(argc, argv, "i:")) != -1) {
    switch (c) {
    case 'i' :
      strncpy(simu.ifile, optarg, 256); // get input filename.
      opt_flag = opt_flag | 0x01;
      break;
    default :
      die("undefined command line parameter.");
    }
  }

  /* Got enough options? */
  if ((opt_flag & 0x01) == 0) {
    printf("error: missing command line parameter, -i.(opt_flag = %d)\n", opt_flag);
  }
  if (opt_flag != 0x01) {
    printUsage(argv);
    die("invalid comnand line parameter.(opt_flag = %d)\n", opt_flag);
  }
  /* Get parameters from specified input file. */
  loadValFromFile(remd.Nrep,      "Nrep",         simu.ifile);
  loadValFromFile(remd.Nmol,      "Nmol",         simu.ifile); 
  loadValFromFile(remd.dens,      "dens",         simu.ifile); 
  loadValFromFile(remd.mass,      "mass",         simu.ifile); 
  loadValFromFile(remd.lj_sigma,  "lj_sigma",     simu.ifile); 
  loadValFromFile(remd.lj_epsilon,"lj_epsilon",   simu.ifile); 
  loadValFromFile(remd.temp_max,  "temp_max",     simu.ifile);
  loadValFromFile(remd.temp_min,  "temp_min",     simu.ifile);
  loadValFromFile(simu.Ngpu,      "Ngpu",         simu.ifile); 
  loadValFromFile(simu.step_max,  "step_max",     simu.ifile);   
  loadValFromFile(simu.step_exch, "step_exch",    simu.ifile); 
  loadValFromFile(simu.step_ene,  "step_ene",     simu.ifile); 
  loadValFromFile(simu.dt,        "delta_time",   simu.ifile); 
  loadValFromFile(simu.ene_max,   "energy_max",   simu.ifile); 
  loadValFromFile(simu.ene_min,   "energy_min",   simu.ifile); 
  loadValFromFile(simu.delta_ene, "delta_energy", simu.ifile); 
  setOutputDirName(simu.odir, simu.ifile);
  
  /* Report option */
  loadValFromFile(simu.report_posi,  "report_posi",  simu.ifile);
  loadValFromFile(simu.report_velo,  "report_velo",  simu.ifile);
  loadValFromFile(simu.report_force, "report_force", simu.ifile);
  loadValFromFile(simu.report_temp,  "report_temp",  simu.ifile);
  loadValFromFile(simu.report_ene,   "report_ene",   simu.ifile);
  loadValFromFile(simu.shuffle_atom, "shuffle_atom", simu.ifile);
  loadValFromFile(simu.shuffle_replica, "shuffle_replica", simu.ifile);

  sprintf(simu.odir_histogram,   "%s/histogram",      simu.odir);
  sprintf(simu.ofile_init_temp,  "%s/init_temp.dat",  simu.odir);
  sprintf(simu.ofile_init_posi,  "%s/init_posi.dat",  simu.odir);
  sprintf(simu.ofile_init_velo,  "%s/init_velo.dat",  simu.odir);
  sprintf(simu.ofile_init_force, "%s/init_force.dat", simu.odir);

  for (int i = 2; i <= 8192; i*=2) {
    if (i > 2048) {
      if (SMEM_COUNT <= 2048) {
	die("(;_;)SMEM_COUNT must be 2^N  (N is integer).\n");
      } else {
	die("(;_;)The size of shared memory is too large. decrease SMEM_COUNT.\n");
      }
    }
    if (SMEM_COUNT == i) {
      printf("%s(): SMEM_COUNT is set to %d\n", __func__, SMEM_COUNT);
      break;
    }
  }

  /* Is shared memory enough? */
  if (remd.Nmol > SMEM_COUNT) {
    die("(;_;)The size of shared memory is not enough.\nincrease SMEM_COUNT or decrease Nmol.\n");
  }

  /* temp_min == 0.0 K is not permitted. */
  if (remd.temp_min < 9.9999) {
    die("(;_;)input error: minimum temperature less than 10.0[K] is not permitted.\n");
  }

  /* make directories for saving result of simulation. */
  if (mkdir(simu.odir, 0755) != 0) { // output root
    fprintf(stderr, "warning: %s directory already exists.\n", simu.odir);
  }

  if ( simu.report_posi == 0 && simu.report_velo == 0 && simu.report_force == 0 ) {
    getPathToNthRemd(each_remd_path, remd.Nrep - 1);
    if (mkdir(each_remd_path, 0755) != 0) {
      fprintf(stderr, "warning: %s directory already exists.\n", each_remd_path);
    }      
  } else {
    for (int i = 0; i < remd.Nrep; i++) { // each replica
      getPathToNthRemd(each_remd_path, i);
      if (mkdir(each_remd_path, 0755) != 0) {
	fprintf(stderr, "warning: %s directory already exists.\n", each_remd_path);
      }      
    }
  }

  getPathToLocal(host_dirname);
  if (mkdir(host_dirname, 0755) != 0) { // temperature
    fprintf(stderr, "warning: %s directory already exists.\n", host_dirname);
  }

  sprintf(temp_dirname, "%s/%s", simu.odir, TEMP_DIR);
  if (mkdir(temp_dirname, 0755) != 0) { // temperature
    fprintf(stderr, "warning: %s directory already exists.\n", temp_dirname);
  }

  if (mkdir(simu.odir_histogram, 0755) != 0) { // histogram
    fprintf(stderr, "warning: %s directory already exist.\n", simu.odir_histogram);
  }
  
  // int Nrep_in_dev
  if (remd.Nrep % simu.Ngpu == 0) {
    simu.Nrep_1dev = (remd.Nrep / simu.Ngpu);
  }
  else {
    simu.Nrep_1dev = (remd.Nrep / simu.Ngpu) + 1;
  }

  remd.cellsize   = calcCellsize(remd.dens, remd.mass, remd.Nmol);
  remd.rcut       = calcRcut(remd.cellsize, remd.Nmol);
  simu.histo_bins = (int)((simu.ene_max - simu.ene_min) / simu.delta_ene);

  if (simu.Ngpu > MAX_GPU)   die("Ngpu is too large.\n");
#if 0
  if (remd.Nrep % 14 != 0)   die("Nrep is not multipled by 14.\n");
#endif

#if 1
  // Output Lennard-Jones potentail and force 
  FILE *fp;
  Real3_t lj_force, pos_i, pos_j;
  Real_t  lj_poten;
  Real_t  dr = remd.lj_sigma / 100.0;

  pos_i.x = pos_i.y = pos_i.z = 0.0;
  pos_j.x = pos_j.y = pos_j.z = 0.0;
  
  sprintf(lj_poten_file, "%s/%s", simu.odir, LJ_POTEN_FILE);
  if ((fp = fopen(lj_poten_file, "w")) == NULL) { die("fopen(%s)\n", lj_poten_file); }

  fprintf(fp, "# lennard-jones plot to 6 sigma.#\n");
  fprintf(fp, "# r[nm],   potential[],  force[]\n");
  for (Real_t r = dr; r < 6.0 * remd.lj_sigma; r += dr) {
    pos_i.x = r;
    lj(lj_force, lj_poten, pos_i, pos_j,
       remd.rcut, 1000.0, remd.lj_sigma, remd.lj_epsilon);
    fprintf(fp, "%+e    %+e  {  %+e   %+e   %+e }\n",
	    r, lj_poten, lj_force.x, lj_force.y, lj_force.z);
  }
  fprintf(fp, "# --end--\n");

  fclose(fp);
#endif
}
//==============================================================================
// description:
//    get the "cutoff" distance where effect of force can be ignored.
//------------------------------------------------------------------------------
static Real_t
calcRcut(Real_t cellsize, int Nmol) {
   Real_t rcut;
   if      (Nmol < 4)   { die("Nmol is too small.\n"); }
   else if (Nmol < 256) { rcut = 0.5 * cellsize;       }
   else                 { rcut = 3.0 * remd.lj_sigma;  }
   return rcut;
}
//==============================================================================
// description:
//    get the cell boundary size from the number of atoms and its density.
//    cbrt() can be trapping...  2013.9.13
//------------------------------------------------------------------------------
static Real_t
calcCellsize(Real_t dens, Real_t mass, int Nmol) {
   const int mols_fcc = 4;
   int    lattice_count = (Nmol + mols_fcc -1) / mols_fcc;
   double lattice_size  = cbrt((double)mols_fcc * (mass / Na) / dens) 
                                                         / UNIT_LENGTH; // [nm]
   int    edge_boxcnt = 0;
   int    volume;
   double cellsize;
   printf("info: Argon lattice size  = %f [nm]\n", lattice_size);
   printf("info: Argon lattice count = %d \n", lattice_count);

   do {
      edge_boxcnt++;
      if (edge_boxcnt >= 32) {
	 die("too large cellsize.");
      }
      volume = edge_boxcnt * edge_boxcnt * edge_boxcnt;
   } while(volume < lattice_count);

   cellsize = (double)edge_boxcnt * lattice_size;
   return (Real_t)cellsize; 
}
//==============================================================================
// description:
//    Print the usage of this program.
//------------------------------------------------------------------------------
void
printUsage(char **argv) {
  printf("\nUsage:\n");
  printf("    > %s -i [input-file]\n", argv[0]);
  printf("    input-file : a file including simulation parameters.\n\n");
}
//==============================================================================
// description:
//     extract a line including a parameter name "keyword"
//     from specified by "filename".
//------------------------------------------------------------------------------
void
getLineFromFile(char *line, const char *keyword, const char *filename)
{
  char linebuf[buflen];
  char parsebuf[buflen];
  char *token;
  int  matched_cnt = 0;
  FILE *fp;

  /* open file */
  if ((fp = fopen(filename, "r")) == NULL) {
    die("fopen() returned NULL.");
  }
  /* sweep all lines in file. */
  while (fgets(linebuf, buflen - 1, fp) != NULL) {
    if (strchr(linebuf, '\n') == NULL) {
      die("file includes too long line.");
    }
    strncpy(parsebuf, linebuf, buflen);
    token = strtok(parsebuf, ":;#= \t");
    if (strncmp(token, keyword, buflen) == 0) { /* matched found! */
      strncpy(line, linebuf, buflen);
      matched_cnt++;
    }
  }
  /* close file */
  if (fclose(fp) != 0)      { die("fclose() returned non-zero."); }
  /* check errors */
  if (matched_cnt == 0)     { die("specifed keyword was not found, %s\n", keyword); }
  else if (matched_cnt > 1) { die("file includes duplicated keyword."); }
  else if (matched_cnt < 0) { die("occur unexpected error(s)."); }
} // getLineFromFile()

//==============================================================================
// description:
//     
//------------------------------------------------------------------------------
char *
get2ndToken(char *linebuf, const char *paramname, const char *filename)
{
  char *token;
  getLineFromFile(linebuf, paramname, filename);
  token = strtok(linebuf, delim);
  token = strtok(NULL,    delim);
  return token;
}
//==============================================================================
// void loadValFromFile() group.
// description:
//     extract a line including a parameter name "keyword"
//     from specified by "ifile".
//     same name and differnt kind by overload.
// status:
//     fixed 2013-08-30.
//------------------------------------------------------------------------------
void /* double */
loadValFromFile(double &var, const char *paramname, const char *ifile)
{
  char linebuf[buflen];
  char *token = get2ndToken(linebuf, paramname, ifile);
  var = atof(token);
}
void /* float */
loadValFromFile(float &var, const char *paramname, const char *ifile) {
  char linebuf[buflen];
  char *token = get2ndToken(linebuf, paramname, ifile);
  var = (float)atof(token);
}
void /* int */
loadValFromFile(int &var, const char *paramname, const char *ifile) {
  char linebuf[buflen];
  char *token = get2ndToken(linebuf, paramname, ifile);
  var = atoi(token);
}
void /* long int */
loadValFromFile(long &var, const char *paramname, const char *ifile) {
  char linebuf[buflen];
  char *token = get2ndToken(linebuf, paramname, ifile);
  var = atol(token);
}
void /* string */
loadValFromFile(char *var, const char *paramname, const char *ifile) {
  char linebuf[buflen];
  char *token = get2ndToken(linebuf, paramname, ifile);
  strcpy(var, token);
}                                   

void echoSimConfig(void) {
  printf("#===========================================================\n");
  printf("# Hardware condition.\n");
  printf("#-----------------------------------------------------------\n");
  printf(" Ngpu = %d [devices]\n", simu.Ngpu);
  printf("\n#===========================================================\n");
  printf("# Simulation parameter.\n");
  printf("#-----------------------------------------------------------\n");
  printf(" input file: \"%s\"\n", simu.ifile);
  printf(" output dir: \"%s\"\n", simu.odir);
  printf(" step_max  = %d\n", simu.step_max);
  printf(" step_exch = %d\n", simu.step_exch);
  printf(" step_ene  = %d\n", simu.step_ene);
  printf(" dt        = %f\n", simu.dt);
  printf(" The initial conditions is stored in files following,\n");
  printf("       -> %s\n", simu.ofile_init_temp);
  printf("       -> %s\n", simu.ofile_init_posi);
  printf("       -> %s\n", simu.ofile_init_velo);
  printf("       -> %s\n", simu.ofile_init_force);
  printf(" Each remd tragectory is stored in the directories following,\n");
  printf("       -> %s####/{%s*, %s*, %s*}\n",
	 REMD_PREFIX, POSI_PREFIX, VELO_PREFIX, FORCE_PREFIX);
  printf("\n#===========================================================\n");
  printf("# Basic MD condition.\n");
  printf("#-----------------------------------------------------------\n");
  printf(" Nmol          = %d [atoms]\n", remd.Nmol);
  printf(" lj-sigma      = %+e [nm]\n", remd.lj_sigma);
  printf(" lj-epsilon    = %+e [J/mol]\n", remd.lj_epsilon);
  printf(" Mass of atom  = %f [kg/mol]\n", remd.mass);  
  printf(" Density       = %f [kg/m^3\n", remd.dens);
  printf(" Cutoff        = %f [nm]", remd.rcut);
  if (remd.Nmol < 256) { printf("(set half size of cell)\n");
  } else { printf("(3.0 sigma)\n"); }
  printf(" cellsize      = %f [nm]\n", remd.cellsize);
  printf("\n#===========================================================\n");
  printf("# Replica MD condition.\n");
  printf("#-----------------------------------------------------------\n");
  printf(" Nrep          = %d\n", remd.Nrep);
  printf(" Nrep_1dev     = %d [rep(s)]\n", simu.Nrep_1dev);
  printf(" Max Temper.   = %7.3f [K]\n", remd.temp_max);
  printf(" Min Temper.   = %7.3f [K]\n", remd.temp_min);
  printf(" energy_max    = %f\n", simu.ene_max);
  printf(" energy_min    = %f\n", simu.ene_min);
  printf(" delta_energy  = %f\n", simu.delta_ene);
  printf("info: histo_bins    = %d\n", simu.histo_bins);
  printf("info: report_posi   = %d\n", simu.report_posi);
  printf("info: report_velo   = %d\n", simu.report_velo);
  printf("info: report_force  = %d\n", simu.report_force);
  printf("info: report_temp   = %d\n", simu.report_temp);
  printf("info: report_ene    = %d\n", simu.report_ene);

  printf("info: --------(Simulation Parameters)--------\n");
  fflush(stdout);
  return;
}

void read3Ddata( float4 *target, int index, FILE *fp) {
  int unko; 
  fscanf(fp, "%d %f %f %f",&unko,&(target[index].x),&(target[index].y),&(target[index].z));
}

//==============================================================
void setNVTparameter(float *zeta,float *mass_stat) {
    *zeta = 0.0;
    *mass_stat = 50.0;
}
//--- initial_condition.cu
