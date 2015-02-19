//                              -*- Mode: C++ -*-
// Filename         : remd_top.cu
// Description      : REMD project
// Author           : Kentaro Nomura
// Created On       : 2013-03-05 11:49:27
// Last Modified By : Minoru Oikawa
// Last Modified On : 2014-01-27 15:52:04
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//-----------------------------------------------------------------------------
#include "switch_float_double.H"
#include "mytools.H"
#include "remd_typedef.H"
#include "init_cond.H"
#include "integ.H"
#include "comm_save.H"
#include "version.H"

Remd_t remd;  /* Only one instance of Remd_t in this program. */
Simu_t simu;  /* Only one instance of Simu_t in this program. */


static void mallocHost(Remd_t &remd, Simu_t &simu);
static void mallocDev(Remd_t &remd, Simu_t &simu);
//
static void freeHost(Remd_t &remd);
static void freeDev(Simu_t &simu);

static void initTemp(Real_t *temp_ar, int Nrep, Real_t max_temp, Real_t min_temp);
static void initVel(Remd_t &);
static void initForce(Remd_t &);
void shuffleAtomIndex(Real3_t *pos_ar, int Nmol);
int  placeUnitCubeFCC(Real3_t *pos_ar, int Nmol);
void placeExpandToCellsize(Real3_t *posi_ar, int Nmol, Real_t cellsize);
void testFuncCopyEne(Remd_t &remd, Simu_t &simu);
void testFuncCopyExch(Remd_t &remd, Simu_t &simu);

//===============================================================================
int
main(int argc, char **argv) {
    Stopwatch timer[3];
    int ndev;

    printVersion();
    
    //--- Preset timers.
    timer[0].reset("total_run");
    timer[1].reset("simRemd()_run");
    timer[2].reset("before_simRemd");
  
    //--- Start some timers.
    timer[0].start();
    timer[2].start();

    //--- Load configration data for simulation from input file.
    initSimConfig(argc, argv);

    //--- Echo the configuration data to screen.
    echoSimConfig();

    //--- Allocate memory on host CPU side.
    mallocHost(remd, simu);

    //--- Allocate memory on device GPU side.
    mallocDev(remd, simu);
#if 1 //Dummy allocated region for longer CP time.
    int  *d_extra_region[7];
    int   extra_pages = 8;
    const size_t sz_extra = 64*1024*1024;
    for (int i=0; i<extra_pages; i++) {
	xcudaMalloc((void **)&d_extra_region[i], sz_extra);
    }
#endif
    //--- Do data-transfer test between CPU and GPU(s).
    testFuncCopyEne(remd, simu);
    testFuncCopyExch(remd, simu);

    //--- Set initial temperature to each replicas.
    initTemp(remd.h_temp_ar, remd.Nrep, remd.temp_max, remd.temp_min); 
    saveLocalTemp(remd);

    //--- Set initial location of atoms.
    placeUnitCubeFCC(remd.h_pos_ar, remd.Nmol);
    placeExpandToCellsize(remd.h_pos_ar, remd.Nmol, remd.cellsize);
    if (simu.shuffle_atom > 0) {
	printf("***************************\n");
	printf("*** Shuffle atom index. ***\n");
	printf("***************************\n");
	shuffleAtomIndex(remd.h_pos_ar, remd.Nmol);
    }
    
    //-- Set initial velocities and forces.
    initVel(remd);
    initForce(remd);
    
    for (int i = 0; i < remd.Nrep; i++) {       //  
	copyPos(i, H2D);                      // copy posi. to all replicas.
	copyVel(i, H2D);                      // copy velo. to all replicas.
	copyFoc(i, H2D);
    }
    timer[2].stop();
    
    //-------------------------*
    //-- Execute simulation
    //-------------------------*
    timer[1].start();
    simRemd(remd, simu); 
    timer[1].stop();
  
    //-- Free memory on device GPU side.
    freeDev(simu);
#if 1 //Dummy allocated region for longer CP time.
    for (int i=0; i<extra_pages; i++) {
	xcudaFree(d_extra_region[i]);
    }
#endif
  
    //-- Free memory on host CPU side.
    freeHost(remd);
    
    //-- Stop timers and Print reports.
    timer[0].stop();
    printf("***********************\n");
    printf("* elapsed time report *\n");
    printf("***********************\n");
    timer[2].report();
    timer[1].report();
    timer[0].report();
    printf("\n(^_^)/ Simulation was successfully finished!\n");
  
    //-- Wait for finishing all request for GPU.
    xcudaThreadSynchronize();
    
    return EXIT_SUCCESS;
} // main(...)

//==============================================================================
/*
 * [Description] allocate memory in host.
 */
//------------------------------------------------------------------------------

static void
mallocHost(Remd_t &remd, Simu_t &simu)
{
    debug_print(2, "Entering %s().\n", __func__);

    const int Nrep      = remd.Nrep;
    const int Ngpu      = simu.Ngpu;
    const int Nrep_1dev = simu.Nrep_1dev;
    
    const int vec3_size = sizeof(Real3_t) * remd.Nmol;
    const int vec1_size = sizeof(Real_t)  * remd.Nmol;
    const int ene_size  = sizeof(Real_t)  * Nrep * simu.step_exch;
    const int temp_size = sizeof(Real_t)  * Nrep;
    const int exch_size = sizeof(int)     * Nrep;

    const int pvec3_size= sizeof(Real3_t *) * Ngpu;
    const int pvec1_size= sizeof(Real_t  *) * Ngpu;
    const int pint_size = sizeof(int     *) * Ngpu;

    remd.sort_temper = (int *)xmalloc(sizeof(int) * Nrep);
    for (int i=0; i<Nrep; i++) {
	remd.sort_temper[i] = i;
    } // serial number.

    remd.acc_ratio = (int *)xmalloc(sizeof(int) * Nrep);
    for (int i=0; i<Nrep; i++) {
	remd.acc_ratio[i] = 0;
    } 
    
    /* for buffer of 1-copy. */
    //remd.h_pos_ar  = (Real3_t *)xmalloc(vec3_size);
    //remd.h_vel_ar  = (Real3_t *)xmalloc(vec3_size);
    //remd.h_foc_ar  = (Real3_t *)xmalloc(vec3_size);
    remd.h_pos_ar  = (Real3_t *)xmalloc(vec3_size*Nrep);
    remd.h_vel_ar  = (Real3_t *)xmalloc(vec3_size*Nrep);
    remd.h_foc_ar  = (Real3_t *)xmalloc(vec3_size*Nrep);
    remd.h_mass_ar = (Real_t  *)xmalloc(vec1_size);
    remd.h_energy  = (Real_t  *)xmalloc(ene_size);
    remd.h_temp_ar = (Real_t  *)xmalloc(temp_size);
    remd.h_exch_ar = (int     *)xmalloc(exch_size);
    remd.h_temp_meas = (Real_t *)xmalloc(sizeof(Real_t) * simu.step_exch);

    /* pointer to devices */
    remd.d_pos_ar  = (Real3_t**)xmalloc(pvec3_size);
    remd.d_vel_ar  = (Real3_t**)xmalloc(pvec3_size);
    remd.d_foc_ar  = (Real3_t**)xmalloc(pvec3_size);
    remd.d_energy  = (Real_t **)xmalloc(pvec1_size);
    remd.d_temp_ar = (Real_t **)xmalloc(pvec1_size);
    remd.d_exch_ar = (int    **)xmalloc(pint_size);
    remd.d_temp_meas=(Real_t **)xmalloc(pvec1_size);
    simu.which_dev = (int *)xmalloc(sizeof(int) * Nrep);

    for (int i = 0; i < Nrep; i++) {
	simu.which_dev[i] = i / Nrep_1dev;   // ex.) 0,0,0,1,1,1,2,2,2,...
    }
    simu.offset_dev = (int *)xmalloc(sizeof(int) * Nrep);

    for (int i = 0; i < Nrep; i++) {
	simu.offset_dev[i] = i % Nrep_1dev;
    }
    
    debug_print(2, "Exiting  %s().\n", __func__);
} //mallocHost()
//===============================================================================
static void
mallocDev(Remd_t &remd, Simu_t &simu) {
    debug_print(2, "Entering %s().\n", __func__);
    
    const int ar_size = sizeof(Real3_t) * remd.Nmol * simu.Nrep_1dev;
    const int te_size = sizeof(Real_t)  * simu.Nrep_1dev;
    const int en_size = sizeof(Real_t)  * simu.step_exch * simu.Nrep_1dev;
    const int ch_size = sizeof(int)     * simu.Nrep_1dev;
#if defined(HOST_RUN)
    /*
     * Pseudo HOST_RUN
     */
    printf("info: Try pseudo cudaMalloc() on %d HOST device(s) ...", simu.Ngpu);
    for (int i = 0; i < simu.Ngpu; i++) {
	remd.d_pos_ar[i]    = (Real3_t *)xmalloc(ar_size);
	remd.d_vel_ar[i]    = (Real3_t *)xmalloc(ar_size);
	remd.d_foc_ar[i]    = (Real3_t *)xmalloc(ar_size);
	remd.d_temp_ar[i]   = (Real_t  *)xmalloc(te_size);
	remd.d_exch_ar[i]   = (int     *)xmalloc(ch_size);
	remd.d_energy[i]    = (Real_t  *)xmalloc(en_size);
	remd.d_temp_meas[i] = (Real_t  *)xmalloc(en_size);
    }
    puts("succeeded.\n");
#elif defined(DEVICE_RUN)
    /*
     * Real DEVICE_RUN
     */
    printf("info: Try cudaMalloc() on %d GPU device(s)\n", simu.Ngpu);
    /* malloc on GPUs for 2D array. */
    for (int gpu_i = 0; gpu_i < simu.Ngpu; gpu_i++) {
	printf("info: Try cudaMalloc() on GPU device %d / %d.... \n", gpu_i+1, simu.Ngpu);
	fflush(stdout);
	
	xcudaSetDevice(gpu_i);
	xcudaMalloc((void **)&remd.d_pos_ar[gpu_i],  ar_size);
	xcudaMalloc((void **)&remd.d_vel_ar[gpu_i],  ar_size);
	xcudaMalloc((void **)&remd.d_foc_ar[gpu_i],  ar_size);

	xcudaMalloc((void **)&remd.d_temp_ar[gpu_i],   te_size);
	xcudaMalloc((void **)&remd.d_exch_ar[gpu_i],   ch_size);
	xcudaMalloc((void **)&remd.d_energy[gpu_i],    en_size);
	xcudaMalloc((void **)&remd.d_temp_meas[gpu_i], en_size);
#ifdef EXTRA_CUDAMALLOC
	//<-- overheaded dummy region
	xcudaMalloc((void **)&remd.d_temp_ar[gpu_i],   te_size);
	//--> overheaded dummy region
#endif
	printf("succeeded.\n"); fflush(stdout);
    }
    puts("succeeded.\n"); fflush(stdout);
#else
    die("undefined HOST_RUN or DEVICE_RUN.\n");
#endif  

    debug_print(2, "Exiting  %s().\n", __func__); fflush(stdout);
}

static void
freeHost(Remd_t &remd)
{
    debug_print(6, "Entering %s().\n", __func__);

    xfree(remd.sort_temper);
    
    xfree(remd.h_pos_ar);
    xfree(remd.h_vel_ar);
    xfree(remd.h_foc_ar);
    xfree(remd.h_energy);
    xfree(remd.h_temp_ar);
    xfree(remd.h_exch_ar);
    xfree(remd.h_temp_meas);

    xfree(remd.d_pos_ar);
    xfree(remd.d_vel_ar);
    xfree(remd.d_foc_ar);
    xfree(remd.d_energy);
    xfree(remd.d_temp_ar);
    xfree(remd.d_exch_ar);
    xfree(remd.d_temp_meas);
    
    debug_print(5, "Exiting  %s().\n", __func__);  
}

static
void freeDev(Simu_t &simu)
{
    debug_print(6, "Entering %s().\n", __func__);

    for (int gpu_i = 0; gpu_i < simu.Ngpu; gpu_i++) {
	xcudaSetDevice(gpu_i);
#if defined(HOST_RUN)
	xfree(remd.d_pos_ar[gpu_i]);
	xfree(remd.d_vel_ar[gpu_i]);
	xfree(remd.d_foc_ar[gpu_i]);
	xfree(remd.d_temp_ar[gpu_i]);
	xfree(remd.d_exch_ar[gpu_i]);
	xfree(remd.d_energy[gpu_i]);
	xfree(remd.d_temp_meas[gpu_i]);
#elif defined(DEVICE_RUN)
	xcudaFree(remd.d_pos_ar[gpu_i]);
	xcudaFree(remd.d_vel_ar[gpu_i]);
	xcudaFree(remd.d_foc_ar[gpu_i]);
	xcudaFree(remd.d_temp_ar[gpu_i]);
	xcudaFree(remd.d_exch_ar[gpu_i]);
	xcudaFree(remd.d_energy[gpu_i]);
	xcudaFree(remd.d_temp_meas[gpu_i]);
#else
	die("undefined HOST_RUN or DEVICE_RUN.\n");
#endif
    }
    debug_print(6, "Exiting  %s().\n", __func__);
}
//==============================================================================
// [Description]
//     Initialize temperature simuition in each replicas.
// [output]
//     remd.remp_h[Nrep]
// [input]
//     remd.{temp_max, temp_min, Nrep}
//------------------------------------------------------------------------------
static void
initTemp(Real_t *temp_ar, int Nrep, Real_t max_temp, Real_t min_temp)
{
    debug_print(2, "Entering %s().\n", __func__);

    double max = (double)(max_temp);
    double min = (double)(min_temp);
    double common_rate = pow(max / min, 1.0 / ((double)Nrep - 1.0));
    for (int i = 0; i < Nrep; i++) {
	temp_ar[i] = (Real_t)(min * pow(common_rate, (double)i));
    }
    
    debug_print(2, "Exiting  %s().\n", __func__);
}
//==============================================================================
// [Description]
// [output]
// [input]
//------------------------------------------------------------------------------
int
latticeSizeFCC(int Nmol)
{
    const double fcc_density = 4.0;
    double grid_size = ceil(pow((double)Nmol / fcc_density, 1.0 / 3.0));

    if (grid_size > pow(256.0, 3.0)) {
	die("Nmol must be 4*dimsize^3 !! or is too big!!(max 4000000)");
    }
  
    return (int)grid_size;
}
//==============================================================================
// [Description]
//     Initialize position of atoms in each replicas.
// [output]
// [input]
//------------------------------------------------------------------------------
double
calcLatticeSize(Real_t cellsize, int dimsize)
{
    return ((double)cellsize / (double)dimsize);
}
/**
 * [Name] setUnitLatticeFCC()
 * [Description]
 * [input]
 * [output] posi[4]
 */
void
setUnitLatticeFCC(Real3_t *posi, Real_t xstep, Real_t ystep, Real_t zstep)
{
    posi[0].x = posi[0].y = posi[0].z = 0.0;
  
    posi[1].x = 0.0;
    posi[1].y = 0.5 * ystep;
    posi[1].z = 0.5 * zstep;

    posi[2].x = 0.5 * xstep;
    posi[2].y = 0.5 * ystep;
    posi[2].z = 0.0;

    posi[3].x = 0.5 * xstep;
    posi[3].y = 0.0;
    posi[3].z = 0.5 * zstep;
    return;
}
#if 0
//==============================================================================
// 
//------------------------------------------------------------------------------
__global__ void
g_storeTemper(Real_t *temper_d, int Nrep_1dev)
{
    char filename[256];
    FILE *fp;

    sprintf(filename, "g_storeTemp_dev%04d", threadIdx.x);
    fp = xfopen(filename, "w");
    for (int rep_i = 0; rep_i < Nrep_1dev; rep_i++ ) {
    fprintf(fp, ,);
    }
    fclose(fp);
    return;
}
//==============================================================================
// 
//------------------------------------------------------------------------------
__global__ void
g_writePosition(Real3_t *d_pos_ar, int Nrep_1dev, int stored_offset, int Nmol)  // <<<1,1>>>
{
    Real3_t i;

    for (int rep_i = 0; rep_i < Nrep_1dev; rep_i++) {
	printf("// %d th replica\n", rep_i);
	for (int atom = 0; atom < Nmol; atom++) {
	    i = d_pos_ar[rep_i * Nmol + atom];  
	    printf("%d %f, %f, %f\n", atom, i.x, i.y, i.z);
	}
    }
}
#endif
//==============================================================================
//
//------------------------------------------------------------------------------
static void
initVel(Remd_t &remd)
{
    debug_print(2, "Entering %s().\n", __func__);
  
    Real3_t sum, mean;
    Real_t scale = 1.0;
    
    setZero( sum );
    
    for (int i = 0; i < remd.Nmol; i++) {
	setGaussRand( remd.h_vel_ar[i] );   //set random velocity
    }
    for (int i = 0; i < remd.Nmol; i++) {
	add(sum, remd.h_vel_ar[i]);
    }
    mean.x = (Real_t)((double)sum.x / (double)remd.Nmol);
    mean.y = (Real_t)((double)sum.y / (double)remd.Nmol);
    mean.z = (Real_t)((double)sum.z / (double)remd.Nmol);

    for (int i = 0; i < remd.Nmol; i++) {
	sub(remd.h_vel_ar[i], mean);   //kill momentum
    }
    for (int i = 0; i < remd.Nmol; i++) {
	mult(remd.h_vel_ar[i], scale); // scale
    }
    // copy to all replica
    for (int i = 0; i < remd.Nrep; i++) {
	copyVel(i, H2D);
    }
  
    debug_print(2, "Exiting  %s().\n", __func__);
} // initVelo()
//==============================================================================
/* [Name] initForce()
 * [description]
 */
static void
initForce(Remd_t &remd)
{
    debug_print(2, "Entering %s().\n", __func__);
  
    for (int i = 0; i < remd.Nmol; i++) setZero( remd.h_foc_ar[i] );
  
    debug_print(2, "Exiting  %s().\n", __func__);
}
//==============================================================================
// Name: UnitCubicPlacerFCC();
// description:
//    place Nmol atoms in unit-cube(0.0 <= {x,y,z} <1.0) by Face-Centered-Cubic.
// status:
//
//------------------------------------------------------------------------------
void
shuffleAtomIndex(Real3_t *posi_ar, int Nmol)
{
    int *index = (int *)xmalloc(sizeof(int) * Nmol);
    int tmp;
    int shuffle;
    int r;
    Real3_t *tmp_ar = (Real3_t *)xmalloc(sizeof(Real3_t) * Nmol);

    for (int i=0; i<Nmol; i++) { 
	index[i] = i;                /* set sequential number */
	tmp_ar[i].x = posi_ar[i].x;  /* copy */
	tmp_ar[i].y = posi_ar[i].y;
	tmp_ar[i].z = posi_ar[i].z;
    }
  
    for (int shuffle = 0; shuffle < Nmol*256; shuffle++) { /* shuffle */
	r = rand() % Nmol;
	tmp = index[Nmol - 1];  /* swap two values */
	index[Nmol - 1] = index[r];
	index[r] = tmp;
    }
#if 0
    for (int i=0; i<Nmol; i++) {
	printf("%s(): index[%d]= %d\n", __func__, i, index[i]);
    }
#endif
    for (int i=0; i<Nmol; i++) {
	posi_ar[i].x = tmp_ar[ index[i] ].x;
	posi_ar[i].y = tmp_ar[ index[i] ].y;
	posi_ar[i].z = tmp_ar[ index[i] ].z;
    }
  
    xfree(tmp_ar);
    xfree(index);
    return;
}

void
placeExpandToCellsize(Real3_t *posi_ar, int Nmol, Real_t cellsize)
{
    for (int i = 0; i < Nmol; i++) {
	posi_ar[i].x *= cellsize;
	posi_ar[i].y *= cellsize;
	posi_ar[i].z *= cellsize;
    
	posi_ar[i].x -= 0.5 * cellsize;
	posi_ar[i].y -= 0.5 * cellsize;
	posi_ar[i].z -= 0.5 * cellsize;
    }
}

int
placeUnitCubeFCC(Real3_t *posi, int Nmol)
{
    debug_print(2, "Entering %s().\n", __func__);
  
    const int mol_per_lattice = 4; // Count of atoms inside the unit lattice.
    double lattice_tot;            // Number of lattices requred for including Nmol atoms.
    int    posi_tot;               // Number of positions of atoms to be placed.
    int    posi_n = 0;
    int    remove_tot;
    long   remove_cand;
    int    *place_mark;
    int    edge_grids;
    int    mol_n = 0;
  
    lattice_tot = ceil( (double)Nmol / (double)mol_per_lattice );
    edge_grids  = (int)ceil(pow(lattice_tot, 1.0 / 3.0));
    posi_tot    = edge_grids * edge_grids * edge_grids * mol_per_lattice;
    remove_tot  = posi_tot - Nmol;

    for (int i = 0; i < Nmol; i++) posi[i].x = posi[i].y = posi[i].z = -1.0;
    
    srand(RANDSEED);
    
    place_mark = (int *)xmalloc(sizeof(int) * posi_tot); // place_mark[posi_tot]
    
    for (int i=0; i<posi_tot; i++) place_mark[i] = 1; // set all marker.
    
    /* overwrite zeros on random place for remove_tot times. */  
    for (int i = 0; i < remove_tot; i++) {
	do {
	    remove_cand = lrand48() % posi_tot;
	} while (place_mark[remove_cand] == 0);
	place_mark[remove_cand] = 0;
    }

    /* <-- check errors */
    int fill_cnt  = 0;
    int empty_cnt = 0;
  
    for (int i = 0; i < posi_tot; i++) {
	if (place_mark[i] == 1) fill_cnt++;
	else if (place_mark[i] == 0) empty_cnt++;
	else {
	    die("wrong value of place_mark[%d] in %s\n", i);
	}
    }
  
    if (fill_cnt != Nmol) {
	die("wrong value of fill_cnt = %d in %s\n", fill_cnt);
    } else if (empty_cnt != remove_tot) {
	die("wrong value of empty_cnt = %d in %s\n", empty_cnt);
    } else {
	printf("%s(): Nmol= %d , posi_tot= %d , fill_cnt= %d , empty_cnt= %d\n", __func__, Nmol, posi_tot, fill_cnt, empty_cnt);
	printf("%s(): latice_tot= %f , edge_grids= %d\n", __func__, lattice_tot, edge_grids);
    }
    /* --> check errors */

    Real_t ri, rj, rk;
    Real_t di, dj, dk;
    for (int i=0; i < edge_grids; i++) { ri = (Real_t)i;
	for (int j=0; j < edge_grids; j++) { rj = (Real_t)j;
	    for (int k=0; k < edge_grids; k++) { rk = (Real_t)k;
		for (int m=0; m < mol_per_lattice; m++) {
		    if (place_mark[posi_n] == 1) {
			if (mol_n >= Nmol) {
			    die("confusing condition, mol_n(%d) >= Nmol(%d), i=%d, j=%d, k=%d, m=%d.\n",
				mol_n, Nmol, i, j, k, m);
			}

			switch (m) {
			case 0:
			    di = 0.0; dj = 0.0; dk = 0.0;
			    break;
			case 1:
			    di = 0.5; dj = 0.5; dk = 0.0;
			    break;
			case 2:
			    di = 0.5; dj = 0.0; dk = 0.5;
			    break;
			case 3:
			    di = 0.0; dj = 0.5; dk = 0.5;
			    break;
			default:
			    die("confusing condition, posi_tot(%d) < mol_n(%d)\n", posi_tot, mol_n);
			}
			posi[mol_n].x = ri + di;
			posi[mol_n].y = rj + dj;
			posi[mol_n].z = rk + dk;
			mol_n++;
		    }
		    posi_n++;
		} // for (int m = ...
	    } // for (int k = ...
	} // for (int j = ...
    } // for (int i = ...
    
    free(place_mark);
    
    /* Normalize */
    Real_t scaler = (Real_t)edge_grids;
    for (int i = 0; i < Nmol; i++) {
	posi[i].x += 0.25;
	posi[i].y += 0.25;
	posi[i].z += 0.25;
	
	posi[i].x /= scaler;
	posi[i].y /= scaler;
	posi[i].z /= scaler;
    }
    debug_print(2, "Exiting  %s().\n", __func__);
    return 0;
} // placeUnitCubeFCC(...)

//==============================================================================
// description:
//
// status:
//
//------------------------------------------------------------------------------
void
testFuncCopyEne(Remd_t &remd, Simu_t &simu)
{
    debug_print(2, "Entering %s().\n", __func__); fflush(stdout);

    const int Nrep      = remd.Nrep;
    const int step_exch = simu.step_exch;
    const int ene_size  = sizeof(Real_t) * Nrep * step_exch;
    Real_t   *h_energy  = remd.h_energy;
    Real_t   *buf_energy= (Real_t  *)xmalloc(ene_size);
    
    /* Test the copy function of energy */
    for (int i = 0; i < step_exch * Nrep; i++) {
	buf_energy[i] = h_energy[i] = (Real_t)i + 0.1;
    }
    
    debug_print(2, "%s(): copyEnergy(H2D) begin.\n", __func__); fflush(stdout);
    copyEnergy(H2D, remd, simu); // push
    debug_print(2, "%s(): copyEnergy(H2D) done.\n", __func__); fflush(stdout);
    
    for (int i = 0; i < step_exch * Nrep; i++) {
	h_energy[i] = -99999.999;
    }

    debug_print(2, "%s(): copyEnergy(D2H) begin.\n", __func__); fflush(stdout);
    copyEnergy(D2H, remd, simu); // back
    debug_print(2, "%s(): copyEnergy(D2H) done.\n", __func__); fflush(stdout);

    Real_t d_val, h_val;
    
    for (int i = 0; i < step_exch * Nrep; i++) {
	if (h_energy[i] == buf_energy[i]) {
#if 0
	    printf("%s(): %5d/%5d ok(d_val=%f, h_val=%f).\n",
		   __func__, i, step_exch * Nrep, h_energy[i], buf_energy[i]);
#endif
	}
	else {
	    die("test not passed. NG(d_val=%f, h_val=%f).\n", h_energy[i], buf_energy[i]);
	}
    }
    printf("passed function test of copyEnergy(H2D, D2H).\n");
    xfree(buf_energy);
    
    debug_print(2, "Exiting  %s().\n", __func__); fflush(stdout);
}

void
testFuncCopyExch(Remd_t &remd, Simu_t &simu)
{
    debug_print(2, "Entering %s().\n", __func__); fflush(stdout);
    
    int   *h_exch    = remd.h_exch_ar;
    int   *buf_exch  = (int  *)xmalloc(sizeof(int) * remd.Nrep);
    
    /* Test the copy function of exch */
    for (int i = 0; i < remd.Nrep; i++) {
	buf_exch[i] = h_exch[i] = i;
    }

    debug_print(2, "%s(): copyExch(H2D) begin.\n", __func__); fflush(stdout);
    copyExch(H2D, remd, simu); // push
    debug_print(2, "%s(): copyExch(H2D) done.\n", __func__); fflush(stdout);

    for (int i = 0; i < remd.Nrep; i++) {
	h_exch[i] = -999;
    }

    debug_print(2, "%s(): copyExch(D2H) begin.\n", __func__); fflush(stdout);
    copyExch(D2H, remd, simu); // back
    debug_print(2, "%s(): copyExch(D2H) done.\n", __func__); fflush(stdout);

    Real_t d_val, h_val;
    
    for (int i = 0; i < remd.Nrep; i++) {
	if (h_exch[i] == buf_exch[i]) {
#if 0
	    printf("%s(): %5d/%5d ok(d_val=%f, h_val=%f).\n",
		   __func__, i, remd.Nrep, h_exch[i], buf_exch[i]);
#endif
	}
	else {
	    die("test not passed. NG(d_val=%f, h_val=%f).\n", h_exch[i], buf_exch[i]);
	}
    }
    printf("passed function test of copyExch(H2D, D2H).\n");
    free(buf_exch);

    debug_print(2, "Exiting  %s().\n", __func__); fflush(stdout);
}

//--> remd_top.cu
