//                             -*- Mode: C++ -*-
// Filename         : integ.cu
// Description      : Time integrator.
// Author           : Minoru Oikawa (m_oikawa@amber.plala.or.jp)
// Created On       : 2013-08-25 11:49:27
// Last Modified By : Minoru Oikawa
// Last Modified On : 2014-02-12 20:57:57
// Update Count     : 0.0
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#include "switch_float_double.H"
#include "remd_typedef.H"
#include "mytools.H"
#include "init_cond.H"
#include "comm_save.H"
#include "integ.H"  
#include "calc_force.cu" 
#include "dscuda.h"

#define ENERGY_WARNING 0

FaultConf_t FAULT_CONF(5); // fault 1 times.

__device__
void calcVelScale(int t, Real_t *scale, Real_t targ_temp, Real3_t *vel_ar, Real_t mass, int Nmol, Real3_t *shared);
void checkEnergyVal( int t0, const Real_t *h_energy, int len);
void calcHistogram( int *histo_ar, Remd_t &remd, Simu_t &simu);
void saveHistogram( const int *histo_ar, Remd_t &remd, Simu_t &simu);
void exchTemp( int t0, Remd_t &remd, Simu_t &simu);
void saveAccRatio( Remd_t &remd, int step_numkernel);
__device__
int ctrlTemp( int, int, Real3_t *velo_ar, Real_t zeta, int Nmol, Real_t dt);
static void calcZetaSum( Real_t&, Real_t, Real_t);
static Real_t hamiltonian( Real_t, Real_t, Real_t, Real_t, Real_t, Real_t, int);

// Debug
__device__ int chksum( int *start, int count ) {
    int sum, *p, i;
    if ( threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0 ) { /* Do by only 1 thread */
	p=start;
	sum=0;
	for (i=0; i<count; i++) {
	    sum += *(p+i);
	}
    }
    return sum;
}
__device__ void
probePosVelFoc( int t, int num_rep, Real3_t *posi, Real3_t *velo, Real3_t *forc, int Nmol ) {
   if ( blockIdx.x == num_rep ) { 
      if ( threadIdx.x==0 ) {
	 for (int i=0; i<Nmol; i++) {
	    printf("t=%d, Rep=%d [%d]: p={%+6.3f, %+6.3f, %+6.3f}, v={%+6.3f, %+6.3f %+6.3f}, f={%+6.3f, %+6.3f %+6.3f}\n", t, num_rep, i,
		   posi[i].x, posi[i].y, posi[i].z,
		   velo[i].x, velo[i].y, velo[i].z,
		   forc[i].x, forc[i].y, forc[i].z);
	 }
      }
   }
}

extern "C" __global__ void
g_test01( Real3_t *d_work_ar, Real_t *d_poten_ar, int Nmol ) {
   int i;
   Real3_t *shared    = d_work_ar + (Nmol * blockIdx.x);  // originally __shared__
   Real_t  *poten_ar  = d_poten_ar + (Nmol * blockIdx.x); // originally __shared__

   __shared__ Real_t sum;

   //=================================================
   if (threadIdx.x==0) {
      for (i=0;i<1024;i++) {
	 poten_ar[i]=(Real_t)i+1.0;
      }
   }
   __syncthreads();
   
   reductSum1D( &sum, poten_ar, 1 ); // Sum. to sum
   __syncthreads();
   if (threadIdx.x==0) {
      printf("%s():sum=%f\n", __func__, sum);
   }
   //-------------------------------------------------
   //=================================================
   if (threadIdx.x==0) {
      for (i=0;i<1024;i++) {
	 poten_ar[i]=(Real_t)i+1.0;
      }
   }
   __syncthreads();
   
   reductSum1D( &sum, poten_ar, 2 ); // Sum. to sum
   __syncthreads();
   if (threadIdx.x==0) {
      printf("%s():sum=%f\n", __func__, sum);
   }
   //-------------------------------------------------
   //=================================================
   if (threadIdx.x==0) {
      for (i=0;i<1024;i++) {
	 poten_ar[i]=(Real_t)i+1.0;
      }
   }
   __syncthreads();
   
   reductSum1D( &sum, poten_ar, 3 ); // Sum. to sum
   __syncthreads();
   if (threadIdx.x==0) {
      printf("%s():sum=%f\n", __func__, sum);
   }
   //-------------------------------------------------

   if (threadIdx.x==0) {
      for (i=0;i<1024;i++) {
	 poten_ar[i]=(Real_t)i+1.0;
      }
   }
   __syncthreads();
   
   reductSum1D( &sum, poten_ar, 100 ); // Sum. to sum
   __syncthreads();
   if (threadIdx.x==0) {
      printf("%s():sum=%f\n", __func__, sum);
   }

   if (threadIdx.x==0) {
      for (i=0;i<1024;i++) {
	 poten_ar[i]=(Real_t)i+1.0;
      }
   }
   __syncthreads();

   reductSum1D( &sum, poten_ar, 101 ); // Sum. to sum
   __syncthreads();
   if (threadIdx.x==0) {
      printf("%s():sum=%f\n", __func__, sum);
   }

}

//===============================================================================
extern "C" __global__ void
fitVel( int Nmol, int step_exch, Real_t dt, Real_t cellsize, Real_t rcut,
	Real_t lj_sigma, Real_t lj_epsilon, Real_t mass, 
	Real3_t *d_pos_ar, Real3_t *d_vel_ar, Real3_t *d_foc_ar,
	Real3_t *d_work_ar, Real_t *d_poten_ar,
	Real_t  *d_ene_ar, Real_t  *d_temp_ar,Real_t  *d_temp_meas, int *d_exch_ar) {
   /* Point out each data region from bulk data block */
   Real3_t *pos_ar    = d_pos_ar + (Nmol * blockIdx.x);
   Real3_t *vel_ar    = d_vel_ar + (Nmol * blockIdx.x);
   Real3_t *foc_ar    = d_foc_ar + (Nmol * blockIdx.x);
   Real3_t *shared    = d_work_ar + (Nmol * blockIdx.x);  // originally __shared__
   Real_t  *poten_ar  = d_poten_ar + (Nmol * blockIdx.x); // originally __shared__
   Real_t  *ene_ar    = d_ene_ar + (step_exch * blockIdx.x);
   Real_t   temp_targ = d_temp_ar[blockIdx.x];
   //Real_t  *temp_meas = d_temp_meas + (step_exch * blockIdx.x); /* use [0] only. */
   __shared__ Real_t temp_meas;
   __shared__ Real_t vel_scale;
   
   int      t_max = 1000;
   int      ret_code[8];

   if ( blockIdx.x==0 && threadIdx.x==0 ) {
       printf("gridDim.x=%d, blockDim.x=%d\n", gridDim.x, blockDim.x);
   }

   for ( int t=0; t<t_max; t++ ) {
#if 0 //debug
      for (int i=0; i<1; i++) {
	 probePosVelFoc(t, i, pos_ar, vel_ar, foc_ar, Nmol);
      }
#endif
      ret_code[0] = integVel(t, 10, vel_ar, foc_ar, Nmol, mass, dt * 0.5);
      __syncthreads();
      if ( ret_code[0] != 0 ) {
	 printf("(;_;) serious error in function %s(): timestep t = %d, blockIdx.x = %d, threadIdx.x = %d\n", __func__, t, blockIdx.x, threadIdx.x);
	 return;
      }
      
      ret_code[1] = integPos(t, pos_ar, vel_ar, Nmol, dt, cellsize);
      __syncthreads();
      if (ret_code[1] !=0) {
	 printf("(;_;) serious error in function %s(): timestep t = %d, blockIdx.x = %d, threadIdx.x = %d\n", __func__, t, blockIdx.x, threadIdx.x);
	 return;
      }

      __syncthreads();
      killMomentum(t, vel_ar, mass, Nmol, shared );
      __syncthreads();
      calcVelScale(t, &vel_scale, temp_targ, vel_ar, mass, Nmol, shared);
      __syncthreads();
      if (threadIdx.x==0) {
	 printf("%s():t=%d: rep_num=%d, vel_scale=%f, temp_meas=%f, temp_targ=%f\n",
		__func__, t, blockIdx.x, vel_scale, temp_meas, temp_targ);
      }

      scaleVelo(t, vel_ar, vel_scale, Nmol);
      __syncthreads();
      measTemper( &temp_meas, vel_ar, mass, Nmol, shared );
      __syncthreads();
      calcForce( foc_ar, poten_ar, pos_ar, Nmol, rcut, cellsize, lj_sigma, lj_epsilon);
      __syncthreads();
      integVel( t, 11, vel_ar, foc_ar, Nmol, mass, dt * 0.5);
      __syncthreads();
   }
   return;
}// fitVel(...)

//===============================================================
//
//---------------------------------------------------------------
extern "C" __global__ void
integTime(int t0,
	  int Nmol, int step_exch, Real_t dt, Real_t cellsize, Real_t rcut,
	  Real_t lj_sigma, Real_t lj_epsilon, Real_t mass, 
	  Real3_t *d_pos_ar, Real3_t *d_vel_ar, Real3_t *d_foc_ar,
	  Real3_t *d_work_ar, Real_t *d_poten_ar,
	  Real_t  *d_ene_ar, Real_t  *d_temp_ar,Real_t  *d_temp_meas, int *d_exch_ar,
	  FaultConf_t FAULT_CONF) {
   __shared__ Real_t   zeta;
   __shared__ int fault_cnt;
   Real_t calc_err;

   if (threadIdx.x==0 && blockIdx.x==0) {
       printf("Entering %s()\n", __func__);
   }
   if (threadIdx.x==0) {
      fault_cnt = *FAULT_CONF.d_Nfault;
      printf("FAULT_CONF= %d/%d %s.\n", fault_cnt, FAULT_CONF.fault_en, FAULT_CONF.tag);
   }
   
   Real3_t *pos_ar    = d_pos_ar + (Nmol * blockIdx.x);
   Real3_t *vel_ar    = d_vel_ar + (Nmol * blockIdx.x);
   Real3_t *foc_ar    = d_foc_ar + (Nmol * blockIdx.x);
   Real3_t *shared    = d_work_ar + (Nmol * blockIdx.x);
   Real_t  *poten_ar  = d_poten_ar + (Nmol * blockIdx.x);
   Real_t  *ene_ar    = d_ene_ar + (step_exch * blockIdx.x);
   Real_t  *temp_meas = d_temp_meas + (step_exch * blockIdx.x);
   Real_t   temp_targ = d_temp_ar[blockIdx.x];
   int      exch_flag = d_exch_ar[blockIdx.x];
   int    ret_code;
   __shared__ Real_t vel_scale;
   __shared__ Real_t poten_mean;

   __syncthreads();
#if 0
   //<--- checksum
   if (blockIdx.x==0 && threadIdx.x==0) printf("checksum -------------------------------\n");
   int checksum;
   int checksize;
   
   checksize = sizeof(Real3_t) * Nmol;
   checksum  = chksum((int *)pos_ar, checksize);

   if (threadIdx.x==0) printf("checksum(pos_ar[%d])= %+d\n", blockIdx.x, checksum);
   __syncthreads();
   
   checksum  = chksum((int *)vel_ar, checksize);
   if (threadIdx.x==0) printf("checksum(vel_ar[%d])= %+d\n", blockIdx.x, checksum);
   __syncthreads();
   
   checksum  = chksum((int *)foc_ar, checksize);
   if (threadIdx.x==0) printf("checksum(foc_ar[%d])= %+d\n", blockIdx.x, checksum);
   __syncthreads();
   
    //---> checksum
#endif
   Real_t   zeta_sum = 0.0; // unused?
   Real_t   Q        = 70.0;

   // <--- calc LRC
   Real_t   cellsize_pow3 = cellsize * cellsize * cellsize;
   Real_t   sigma_pow3 = lj_sigma * lj_sigma * lj_sigma;
   Real_t   Nmol_pow2 = (Real_t)(Nmol * Nmol);
   Real_t   sigma_rcut = lj_sigma / rcut;
#if defined(REAL_AS_SINGLE)
   Real_t   poten_LRC = (8.0 * M_PI) / (9.0 * cellsize_pow3) *
     Nmol_pow2 * lj_epsilon * sigma_pow3 *
     ( powf(sigma_rcut, 9.0) - 3.0 * powf(sigma_rcut, 3.0) );
   Real_t poten0_LRC = 8.0 *3.1416 * Nmol * Nmol * (2.0 * powf(1.0 / rcut, 9.0)
						    - 3.0*powf(1.0 / rcut, 3.0)) / (9.0 * cellsize_pow3);
#elif defined(REAL_AS_DOUBLE)
   Real_t   poten_LRC = (8.0 * M_PI) / (9.0 * cellsize_pow3) *
     Nmol_pow2 * lj_epsilon * sigma_pow3 *
     ( pow(sigma_rcut, 9.0) - 3.0 * pow(sigma_rcut, 3.0) );
#endif
  // ---> calc_LRC

   __syncthreads();
   if (blockIdx.x==0 && threadIdx.x==0) {
       if (FAULT_CONF.fault_en==0 || fault_cnt==0) {  /* Normal calc */
	   printf("[Normal calculation] (t0=%d)\n", t0);
	   /* nop */
       }
       else { /* Fault calc */
	   printf("[Fault  calculation] (t0=%d)\n", t0);

       }
   }

   /***********************************************************************
    *  <--- FAULT INJECTION
    */
   if (threadIdx.x==0) {
       if (FAULT_CONF.fault_en==0 || fault_cnt==0) {
	   calc_err = 0.0;
       }
       else {
	   if (blockIdx.x==0) { calc_err = +500.0; }
	   if (blockIdx.x==1) { calc_err = -500.0; }
	   if (blockIdx.x==2) { calc_err = +500.0; }
	   if (blockIdx.x==3) { calc_err = -500.0; }
       }
       printf("(%d,%d)calc_err[%d]:blockIdx.x=%d= %f\n", FAULT_CONF.fault_en, fault_cnt, t0, blockIdx.x, calc_err);
   }
   /*
    *  ---> FAULT INJECTION 
    ***********************************************************************/
   
   // if exchanged, scale velocity //
   if (exch_flag == 1) {
      calcVelScale( -1, &vel_scale, temp_targ, vel_ar, mass, Nmol, shared); // vel_scale = shared[0].x
      __syncthreads();
      scaleVelo(-1, vel_ar, vel_scale, Nmol);
      __syncthreads();
      killMomentum(-1, vel_ar, mass, Nmol, shared);
      __syncthreads();
   }

   if (threadIdx.x == 0) zeta = 0.0;

   int t;
   for (t=0; t<step_exch; t++) {            // run "step_exch" steps.
      integVel( 1000*(t0+1) + t, 21, vel_ar, foc_ar, Nmol, mass, dt * 0.5);
      __syncthreads();
      
      ctrlTemp( 1000*(t0+1) + t, 21, vel_ar, zeta,   Nmol, dt * 0.5);
      __syncthreads();
      
      ret_code = integPos( 1000*(t0+1) + t, pos_ar, vel_ar, Nmol, dt, cellsize);
      __syncthreads();
      if (ret_code != 0) {
	 printf("(;_;) serious error in function %s()\n timestep t = %d\n blockIdx.x = %d, threadIdx.x = %d\n", __func__, t, blockIdx.x, threadIdx.x);
	 return;
      }
    
      measTemper( &temp_meas[t], vel_ar, mass, Nmol, shared); // curr_temp => shared[0].x
      __syncthreads();
      //    calcZeta(zeta, temp_meas[t], Q, temp_targ, dt, Nmol);
      if (threadIdx.x == 0) {
	 zeta = (sqrt( temp_meas[t]) - sqrt(temp_targ)) * dt / Q;
	 if ( !isfinite(zeta) ) {
	    printf("t=%d: zeta=%f, blockIdx.x=%d, temp_meas[]=%f, temp_targ=%f\n",
		   t, zeta, blockIdx.x, temp_meas[t], temp_targ);
	 }
      }
      __syncthreads();
      killMomentum(t, vel_ar, mass, Nmol, shared);
      __syncthreads();

      // calculate forces //
      calcForce( foc_ar, poten_ar, pos_ar, Nmol, rcut, cellsize, lj_sigma, lj_epsilon);
      __syncthreads();

      meanPotential( &poten_mean, poten_ar, Nmol, shared); // + (poten_LRC / Nmol);
      __syncthreads();

      if (threadIdx.x == 0) {
	 if (t > step_exch-10) {
	    ene_ar[t] =  poten_mean / 2.0 + calc_err; // to global memory by specified one thread. 2.0;muguruma's paper.
	 } else {
	    ene_ar[t] = poten_mean / 2.0; // to global memory by specified one thread. 2.0;muguruma's paper.
	 }
      }
      __syncthreads();
      integVel( 1000*(t0+1) + t, 22, vel_ar, foc_ar, Nmol, mass, dt * 0.5);
      __syncthreads();
      ctrlTemp( 1000*(t0+1) + t, 22, vel_ar, zeta,   Nmol, dt * 0.5);
      __syncthreads();
   } // for (int t=0; ...
   
   __syncthreads();
   if (blockIdx.x==0 && threadIdx.x==0) {
       if (FAULT_CONF.fault_en>0 && fault_cnt>0) {
	   *FAULT_CONF.d_Nfault = fault_cnt - 1;
       }
   }
	      
} //integTime()
//==============================================================================
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
// simRemd()
//------------------------------------------------------------------------------
void simRemd( Remd_t &remd, Simu_t &simu ) {
   debug_print(2, "Entering %s().\n", __func__);
   const int MAX_THREADS_PER_BLOCK = 256; // 
   const int MAX_NMOL = 32768;
   
#if !defined(HOST_RUN) && !defined(DEVICE_RUN)
   die("undefined HOST_RUN or DEVICE_RUN.\n");
#endif

   const int    Nrep      = remd.Nrep;
   const int    Nmol      = remd.Nmol;
   const int    Ngpu      = simu.Ngpu;
   const int    Nrep_1dev = simu.Nrep_1dev;
   const int    step_exch = simu.step_exch;
   const Real_t dt        = simu.dt;
   const Real_t cellsize  = remd.cellsize;
   const Real_t rcut      = remd.rcut;
   const Real_t lj_sigma  = remd.lj_sigma;
   const Real_t lj_epsilon = remd.lj_epsilon;
   const Real_t mass      = remd.mass;
   double curr_progress;
   double next_progress;
   double step_progress;
   double elapsed_time_sec;
   int    total_bins = simu.histo_bins;
   int   *histo_ar  = (int *)malloc(sizeof(int) * total_bins * Nrep);
   
   cudaError_t cu_err[8];
   
   int  i;
   dim3 blocks(Nrep_1dev, 1, 1);    // GPU grid size
   dim3 threads(1, 1, 1);        // GPU block size

   if ( Nmol < 2 ) {
      die("Nmol is too small.\n");
   } else if ( Nmol <= MAX_THREADS_PER_BLOCK) {
      threads.x = Nmol;
   } else if ( Nmol <= MAX_NMOL) {
      threads.x = MAX_THREADS_PER_BLOCK; // is maximum number.
   } else {
      die("Nmol is too large.\n");
   }

   if( histo_ar == NULL) { die("not enough memory on host.\n"); }
   for ( i=0; i<total_bins * Nrep; i++) histo_ar[i] = 0;
   
   // initialize exch_ar[] //
   for (int rep_i=0; rep_i<Nrep; rep_i++) {
      remd.h_exch_ar[rep_i] = 1;
   }
#if 0 // function test
   g_test01<<<4, 1>>> ( remd.d_work_ar[0], remd.d_poten_ar[0], Nmol);
   return ;
#endif

   copyTempTarg(H2D);
   copyExch(H2D, remd, simu);
   
   // ************************************
   // *  Initialize Temperature on GPU   *
   // ************************************
   printf("[REMD] fitVel() begins.\n");
   for ( i=0; i<Ngpu; i++ ) {
      cu_err[0] = cudaSetDevice(i);
      if (cu_err[0] != cudaSuccess) { die("cudaSetDevice(%d) failed.\n", i ); }
      fitVel <<<blocks, threads>>>
	 (Nmol, step_exch, dt, cellsize, rcut, lj_sigma, lj_epsilon, mass, 
	  remd.d_pos_ar[i], remd.d_vel_ar[i], remd.d_foc_ar[i],
	  remd.d_work_ar[i], remd.d_poten_ar[i],
	  remd.d_energy[i], remd.d_temp_ar[i],remd.d_temp_meas[i],
	  remd.d_exch_ar[i]);
   }
   cu_err[1] = cudaGetLastError();
   if (cu_err[1] != cudaSuccess) {
       printf("err: %s\n", cudaGetErrorString(cu_err[1]));
       exit(1);
   }
   cudaThreadSynchronize();
   printf("[REMD] fitVel() ends.\n");

   // *****************************
   // *  Main integration on GPU  *
   // *****************************
   next_progress = 0.0;
   step_progress = 0.05;


   for ( int t0 = 0; t0 < simu.step_max; t0++ ) {
      printf("###=============================================================\n");
      printf("### t0 = %d / %d\n", t0, simu.step_max-1);
      printf("###=============================================================\n");
      fflush(stdout);
      curr_progress = (double)t0 / (double)simu.step_max;
      if (curr_progress >= next_progress) {
	 printf("---> ******** %s(): simulation progress is now %5.2f %%.\n",
		__func__, curr_progress * 100); fflush(stdout);
	 next_progress += step_progress;
      }
      
#if defined(__DSCUDA__)
      dscudaClearHist();     /*** <--- Clear Recall List.        ***/
      dscudaRecordHistOff();  /*** <--- Enable recording history. ***/ 
#endif
      if (simu.report_posi >= 1)  { savePosAll(t0 * step_exch);      } // cudaMemcpyD2H * Nrep
      if (simu.report_velo >= 1)  { saveVelAll(t0 * step_exch);      } // cudaMemcpyD2H * Nrep
      if (simu.report_force >= 1) { saveFocAll(t0 * step_exch);      } // cudaMemcpyD2H * Nrep
      if (simu.report_temp >= 2)  { saveTempMeasAll(t0 * step_exch); } // cudaMemcpyD2H * Nrep
      //	printf("checksum: Vel[t0=%d]= %d\n",
      //     t0, checkSum((void*)remd.h_vel_ar, sizeof(Real3_t)*Nmol*Nrep)); fflush(stdout);
	
      // Update target temperature of each replica. //
      copyTempTarg( H2D );                                       // cudaMemcpyH2D * ?
      if (simu.report_temp >= 1)   { saveTempTarg(remd, t0); } // cudaMemcpyD2H * ?
      
      // Update exchanging information. //
      copyExch( H2D, remd, simu );                               // cudaMemcpyH2D * ?
      if ( simu.report_ene  >= 1 )   { saveSorted(remd, t0); }   // cudaMemcpyD2H * ?
      
      if ( t0 < 2 ) {
	 FAULT_CONF.fault_en     = 0;
	 FAULT_CONF.overwrite_en = 0;
      } else {
#if defined( FAULT_ON )
	 FAULT_CONF.fault_en     = 1;
#endif
	 FAULT_CONF.overwrite_en = 1;
      }
      
#if defined( __DSCUDA__ )
      dscudaRecordHistOn();  /*** <--- Enable recording history. ***/ 
#endif
      
      //	printf("checksum: Pos[t0=%d before]= %d\n",
      //     t0, checkSum((void*)remd.h_pos_ar, sizeof(Real3_t)*Nmol*Nrep)); fflush(stdout);

      for ( i=0; i<Ngpu; i++ ) {                          // Sweep GPU.
	 cu_err[0] = cudaSetDevice(i);
	 if( cu_err[0] != cudaSuccess ) { die("cudaSetDevice() failed.\n"); }
	 integTime <<< blocks, threads >>>                       // rpcLaunchKernel
	    ( t0, Nmol, step_exch, dt, cellsize, rcut, lj_sigma, lj_epsilon, mass, 
	      remd.d_pos_ar[i], remd.d_vel_ar[i], remd.d_foc_ar[i],
	      remd.d_work_ar[i], remd.d_poten_ar[i],
	      remd.d_energy[i], remd.d_temp_ar[i],remd.d_temp_meas[i],
	      remd.d_exch_ar[i], FAULT_CONF );
      }
      cu_err[1] = cudaGetLastError();
      if (cu_err[1] != cudaSuccess) {
	 printf("err: %s\n", cudaGetErrorString(cu_err[1]));
	 exit(1);
      }

      copyEnergy( D2H, remd, simu );       /* Correct data of potential energy. */
      
#if defined( __DSCUDA__ )
      dscudaRecordHistOff();
#endif
      //savePosAll(t0 * step_exch + 100000);
#if defined( __DSCUDA__ )
      dscudaAutoVerbOn();
      dscudaClearHist();          /*** <--- Clear Recall List.        ***/
#endif
      if( simu.report_ene >= 2)   { saveEne(remd, t0); }
#if 0
      checkEnergyVal( t0, remd.h_energy, Nrep*step_exch);
#endif
      calcHistogram( histo_ar, remd, simu); // struct histogram 
      exchTemp( t0, remd, simu);            // 
   } //for (t = 0; ...
   saveHistogram( histo_ar, remd, simu );
   saveAccRatio( remd, simu.step_max );
   // free
   free(histo_ar);
   debug_print(2, "Exiting  %s().\n", __func__);
}

//===============================================================================
// Parallel Reduction Sum on DEVICE.
//-------------------------------------------------------------------------------
__device__ void reductClear3D( Real3_t *ar, int size ) {
   for (int i = threadIdx.x; i < size; i += blockDim.x) {
      ar[i].x = ar[i].y = ar[i].z = 0.0;
   }
   __syncthreads();
}

__device__ void
reductSet3D( Real3_t *dst, const Real3_t *src, int size ) {
   for (int i = threadIdx.x; i < size; i += blockDim.x) {
      dst[i].x = src[i].x; 
      dst[i].y = src[i].y; 
      dst[i].z = src[i].z;
   }
   __syncthreads();
}
__device__ void
reductSum1D( Real_t *sum, Real_t *ar, int len ) { // must be 2^N, and less than 2049.
   int i, j;
#if 0 // original multi thread
   for (int reduce_num=1024; reduce_num>1; reduce_num /= 2) {
      if (len > reduce_num) {
	 for (int i=threadIdx.x; i<len; i+=blockDim.x) {
	    if (i < reduce_num) {
	       ar[i] += ar[i + reduce_num];
	    }
	 }
	 __syncthreads();
      }
   }
   if (threadIdx.x == 0) { // 2 -> 1
      sum = ar[0] + ar[1];
   }
#elif 1  /*TODO*/
   int jmax;
   
   for ( jmax = len; jmax > 1; jmax = (jmax/2) + (jmax%2)) {
      /* reduct 1st to thread size, len->len/2 or len/2+1 */
      for ( i=threadIdx.x; i<jmax; i+=blockDim.x ) {
	 j = i + (jmax/2) + (jmax%2); //even:(len/2), odd:(len/2+1)
	 if ( j < jmax ) {
	    ar[i] += ar[j];
	 }
      }
      __syncthreads();
   }
   __syncthreads();
   if ( threadIdx.x == 0 ) {
      *sum = ar[0];
   }
#else // single thread; expected slow.
   Real_t buf;
   
   if ( threadIdx.x == 0 ) {
      buf = 0.0;
      for ( i=0; i<len; i++ ) {
	 buf += ar[i];
      }
      *sum = buf;
   }
#endif
}

__device__ void
reductSum3D( Real3_t *sum, Real3_t *ar, int len ) {
   int i,j;

#if 0 // debug
   for (int reduce_num=1024; reduce_num>1; reduce_num /= 2) {  // 2048 -> 2
      if (len > reduce_num) {
	 for ( i=threadIdx.x; i<len; i+=blockDim.x) {
	    if ( i < reduce_num ) {
	       ar[i].x += ar[i + reduce_num].x;
	       ar[i].y += ar[i + reduce_num].y;
	       ar[i].z += ar[i + reduce_num].z;
	    }
	 }
	 __Syncthreads();
      }
   }
   if (threadIdx.x == 0) {                                             // 2 -> 1
      sum.x = ar[0].x + ar[1].x;
      sum.y = ar[0].y + ar[1].y;
      sum.z = ar[0].z + ar[1].z;
   }
#elif 1
   int jmax;
   
   for ( jmax = len; jmax > 1; jmax = (jmax/2) + (jmax%2)) {
      for ( i=threadIdx.x; i<jmax; i+=blockDim.x ) {
	 j = i + (jmax/2) + (jmax%2); //even:(len/2), odd:(len/2+1)
	 if ( j < jmax ) {
	    ar[i].x += ar[j].x;
	    ar[i].y += ar[j].y;
	    ar[i].z += ar[j].z;
	 }
      }
      __syncthreads();
   }
   __syncthreads();
   if ( threadIdx.x == 0 ) {
      sum->x = ar[0].x;
      sum->y = ar[0].y;
      sum->z = ar[0].z;
   }
#else
   Real3_t buf;

   if ( threadIdx.x == 0 ) {
      buf.x = buf.y = buf.z = 0.0; // Reset.
      for ( i=0; i<len; i++ ){
	 buf.x += ar[i].x;
	 buf.y += ar[i].y;
	 buf.z += ar[i].z;
      }
      sum->x = buf.x; 
      sum->y = buf.y; 
      sum->z = buf.z;
   }
#endif
}
//===============================================================================
// integVel(), for HOST and DEVICE.
//-------------------------------------------------------------------------------
__device__ int
integVel( int t, int tag, Real3_t *vel_ar, Real3_t *foc_ar, int Nmol, Real_t mass, Real_t dt) {
   int i;
   
   for ( i=threadIdx.x; i<Nmol; i+=blockDim.x ) {
#if 1 //debug
      if (!isfinite(vel_ar[i].x) || !isfinite(vel_ar[i].y) || !isfinite(vel_ar[i].z) ||
	  !isfinite(foc_ar[i].x) || !isfinite(foc_ar[i].y) || !isfinite(foc_ar[i].z)) {
	 printf("ERROR: %s(%d), t = %d, vel_ar[%d] = %f %f %f, foc_ar[%d] = %f %f %f. threadIdx.x=%d\n",
		   __func__, tag, t,
		i, vel_ar[i].x, vel_ar[i].y, vel_ar[i].z,
		i, foc_ar[i].x, foc_ar[i].y, foc_ar[i].z, threadIdx.x);
	 return -2;
      }
#endif
      vel_ar[i].x += foc_ar[i].x * dt / mass;
      vel_ar[i].y += foc_ar[i].y * dt / mass;
      vel_ar[i].z += foc_ar[i].z * dt / mass;
   }
   return 0;
}
//==============================================================================
// integPos(), for HOST and DEVECE.
//------------------------------------------------------------------------------
__device__ int
integPos( int t, Real3_t *pos_ar, Real3_t *vel_ar, int Nmol, Real_t dt, Real_t cellsize) {
   Real3_t round;
   int i;
   for ( i=threadIdx.x; i<Nmol; i+=blockDim.x) {
#if 1 //debug
      if (!isfinite(vel_ar[i].x) || !isfinite(vel_ar[i].y) || !isfinite(vel_ar[i].z)) {
	 printf("ERROR: %s(%d), vel_ar[%d] = %f %f %f\n",
		__func__, t, i, vel_ar[i].x, vel_ar[i].y, vel_ar[i].z);
	 return -1;
      }
#endif
      pos_ar[i].x += vel_ar[i].x * dt;
      pos_ar[i].y += vel_ar[i].y * dt;
      pos_ar[i].z += vel_ar[i].z * dt;
#if defined(REAL_AS_SINGLE)
      round.x = rintf(pos_ar[i].x / cellsize);
      round.y = rintf(pos_ar[i].y / cellsize);
      round.z = rintf(pos_ar[i].z / cellsize);
#elif defined(REAL_AS_DOUBLE)
      round.x = rint( pos_ar[i].x / cellsize);
      round.y = rint( pos_ar[i].y / cellsize);
      round.z = rint( pos_ar[i].z / cellsize);
#endif
      pos_ar[i].x -= round.x * cellsize;
      pos_ar[i].y -= round.y * cellsize;
      pos_ar[i].z -= round.z * cellsize;

#if 1 // debug
      if (!isfinite(pos_ar[i].x) || !isfinite(pos_ar[i].y) || !isfinite(pos_ar[i].z)) {
	 printf("ERROR: %s(%d), pos_ar[%d] = %f %f %f, threadidx.x=%d\n",
		__func__, t, i, pos_ar[i].x, pos_ar[i].y, pos_ar[i].z, threadIdx.x);
	 return -2;
      }
      if (pos_ar[i].x < (-0.6)*cellsize || pos_ar[i].x > 0.6*cellsize ||
	  pos_ar[i].y < (-0.6)*cellsize || pos_ar[i].y > 0.6*cellsize ||
	  pos_ar[i].z < (-0.6)*cellsize || pos_ar[i].z > 0.6*cellsize) {
	 printf("ERROR:<<<%d,%d>>> %s(%d), pos_ar[%d]= {%f %f %f}, vel_ar[%d]= {%f, %f, %f}, round= {%f %f %f}: threadIdx.x=%d\n",
		blockIdx.x, threadIdx.x, __func__, t,
		i, pos_ar[i].x, pos_ar[i].y, pos_ar[i].z,
		i, vel_ar[i].x, vel_ar[i].y, vel_ar[i].z,
		round.x, round.y, round.z, threadIdx.x);
      return -3;
      }
#endif
   }
   return 0;
}
//===============================================================================
// measTemper()  ! needs reduct !
// molKineticEne(const Real3_t &vel, Real_t mass)
//
__device__ Real_t
molKineticEne( const Real3_t *vel, Real_t mass ) {
   Real_t abs_sq = (vel->x * vel->x) + (vel->y * vel->y) + (vel->z * vel->z);
   Real_t kinetic_ene = 0.5 * mass * abs_sq;
   return kinetic_ene;
}

// *******************/
// **  DEVICE CODE  **/
// *******************/
__device__ void
measTemper( Real_t *temper, const Real3_t *vel_ar, Real_t mass, int Nmol,
	    Real3_t *shared ) {
   Real_t *smem = (Real_t *)shared;
   Real_t meas;
   Real_t scale_factor = UNIT_MASS * (UNIT_LENGTH * UNIT_LENGTH) / (UNIT_TIME * UNIT_TIME);
   Real_t sum;
   int i;

   for ( i=threadIdx.x; i<Nmol; i+=blockDim.x ) {
      smem[i] = molKineticEne(&vel_ar[i], mass);
   }
   __syncthreads();

   reductSum1D( &sum, smem, Nmol );
   __syncthreads();

   if ( threadIdx.x == 0 ) {
      meas = scale_factor * sum * (2.0 / 3.0) / (Nmol * Boltzmann_constant);
#if 0
      if (meas < 30.0 || meas > 400.0) {
	 printf("%s(): temp_meas[Rep:%d] = %f\n", __func__, blockIdx.x, meas);
	 for (i=0;i<Nmol;i++){
	    printf("%s(): kinetic[%d]=%+6.2f\n", __func__, i, smem[i]);
	 }
      }
#endif
      *temper = meas;
   }
}

//==============================================================================
// calcVelScale()
//------------------------------------------------------------------------------
__device__ void
calcVelScale( int t, Real_t *scale, Real_t targ_temp, Real3_t *vel_ar, Real_t mass,
	      int Nmol, Real3_t *shared ) {
   Real_t *smem = (Real_t *)shared;
   Real_t bunshi = 3.0 / 2.0 * (Real_t)Nmol * targ_temp;
   Real_t unit_scale = (UNIT_TIME / UNIT_LENGTH) * sqrt(Kb / UNIT_MASS);
   Real_t sum, vel_scale;

   for ( int i=threadIdx.x; i<Nmol; i+=blockDim.x ) {
      smem[i] = molKineticEne(&vel_ar[i], mass);
   }
   __syncthreads();
#if 0 //Debug
   if (threadIdx.x==0) {
      for (int i=0; i<Nmol; i++) {
	 printf("molkinet[%d]=%f\n", i, smem[i]);
      }
   }
   __syncthreads();
#endif
   reductSum1D( &sum, smem, Nmol ); // shared[0].x <= sum_kinetic_ene
   __syncthreads();

   if ( threadIdx.x == 0 ) {
      vel_scale = sqrt(bunshi / sum) * unit_scale; // = vel_scale
      *scale = vel_scale;
#if 0 //debug
      printf("t=%d: targ_temp= %f, unit_scale= %f, sum=%f, vel_scale= %f\n.", t, targ_temp, unit_scale, sum, vel_scale);
#endif
   }
}
//==============================================================================
// calcMomentum().
//------------------------------------------------------------------------------
__device__ void
calcMomentum(Real3_t *mome, const Real3_t *velo_ar, Real_t mass, int Nmol, Real3_t *shared) {
   Real3_t sum, vel_max, vel_min;
   int i;

   for (i=threadIdx.x; i<Nmol; i+=blockDim.x) {
      shared[i].x = mass * velo_ar[i].x;
      shared[i].y = mass * velo_ar[i].y;
      shared[i].z = mass * velo_ar[i].z;
   }
   __syncthreads();
   
   reductSum3D(&sum, shared, Nmol);                      // shared[0] <= sum.

   if (threadIdx.x == 0) {
      mome->x = sum.x / (Real_t)Nmol;
      mome->y = sum.y / (Real_t)Nmol;
      mome->z = sum.z / (Real_t)Nmol;
#if 0 // debug monitor
      vel_max.x = vel_max.y = vel_max.z = -999.0;
      vel_min.x = vel_min.y = vel_min.z = +999.0;
      for ( i=0; i<Nmol; i++ ) {
	 if ( velo_ar[i].x > vel_max.x ) vel_max.x = velo_ar[i].x;  
	 if ( velo_ar[i].y > vel_max.y ) vel_max.y = velo_ar[i].y;  
	 if ( velo_ar[i].z > vel_max.z ) vel_max.z = velo_ar[i].z;
	 
	 if ( velo_ar[i].x < vel_min.x ) vel_min.x = velo_ar[i].x;  
	 if ( velo_ar[i].y < vel_min.y ) vel_min.y = velo_ar[i].y;  
	 if ( velo_ar[i].z < vel_min.z ) vel_min.z = velo_ar[i].z;  
      }
      printf("momentum= {%e, %e, %e}, Nmol=%d, mass=%f\n",
	     mome->x, mome->y, mome->z, Nmol, mass );
      printf("vel-max= {%+12.6f, %+12.6f, %+12.6f}\n",
	     vel_max.x, vel_max.y, vel_max.z );
      printf("vel-min= {%+12.6f, %+12.6f, %+12.6f}\n",
	     vel_min.x, vel_min.y, vel_min.z );
#endif
   }
   __syncthreads();
}
//==============================================================================
// killMomentum(). 
//------------------------------------------------------------------------------
__device__ void
killMomentum(int t, Real3_t *velo_ar, Real_t mass, int Nmol, Real3_t *shared) {
   int i;
   Real3_t momentum;
   
   calcMomentum( &momentum, velo_ar, mass, Nmol, shared ); // momentum => shared[0]
   for ( i=threadIdx.x; i<Nmol; i+=blockDim.x ) {
      velo_ar[i].x -= momentum.x;
      velo_ar[i].y -= momentum.y;
      velo_ar[i].z -= momentum.z;
#if 0 // debug monitor
      printf("%s(), t=%d, velo_ar[%d] = { %f , %f , %f } --- momentum { %f %f %f }\n",
	     __func__, t, i, velo_ar[i].x, velo_ar[i].y, velo_ar[i].z, momentum->x, momentum->y, momentum->z);
#endif 
   }
   __syncthreads();
}
//==============================================================================
// scaleVelo()
//------------------------------------------------------------------------------
__device__ void
scaleVelo(int t, Real3_t *velo_ar, Real_t scale, int Nmol) {
   int i;
#if 1 // debug: range check of float.
   for ( i=threadIdx.x; i<Nmol; i+=blockDim.x ) {
      if (!isfinite(velo_ar[i].x) || !isfinite(velo_ar[i].y) || !isfinite(velo_ar[i].z)) {
	 printf("[REMD-ERROR] %s(), t = %d, velo_ar[%d] = %f %f %f\n",
		__func__, t, i, velo_ar[i].x, velo_ar[i].y, velo_ar[i].z);
      }
   }
#endif
   if (threadIdx.x==0) {
      printf("t=%d: scale=%f\n", t, scale);
   }
   for ( i=threadIdx.x; i<Nmol; i+=blockDim.x ) {
      velo_ar[i].x *= scale;
      velo_ar[i].y *= scale;
      velo_ar[i].z *= scale;
   }
}

//===============================================================================
__device__ void
debugVel(const char *mes, const Real3_t *vel_ar, int Nmol) {
   if (threadIdx.x == 0) {
      printf("%s(): %s\n", __func__, mes);
      for (int i=0; i<Nmol; i++) {
	 printf("vel_ar[%d]= %f %f %f\n", i, vel_ar[i].x, vel_ar[i].y, vel_ar[i].z);
      }
   }
}

//===============================================================================
// meanPotential
//     Calculate the mean value of each potentials of all atoms.
//     output: 
//         Real_t mean;
//     inputs:
//         Real_t poten_ar[Nmol]: potential values.
//         Real3_t shared[Nmol] : reduct work area.
//-------------------------------------------------------------------------------
__device__ void
meanPotential( Real_t *mean, Real_t *poten_ar, int Nmol, Real3_t *shared ) {
   //Memo: The "mean" must be __shared__ memory pointer.
   Real_t *smem = (Real_t *)shared;
   Real_t sum;
   int i;
   
   /* Clear space */
   for ( i=threadIdx.x; i<Nmol; i+=blockDim.x ) {
      smem[i] = 0.0;
   }
   __syncthreads();
   
   /* Set data */
   for ( i=threadIdx.x; i<Nmol; i+=blockDim.x ) {
      smem[i] = poten_ar[i];
   }
   __syncthreads();
   
   reductSum1D( &sum, smem, Nmol ); // Sum. to sum
   __syncthreads();

   if (threadIdx.x == 0) {
      *mean = sum / (Real_t) Nmol;      // is mean_potential, [J/mol]
   }
}

__device__ int
ctrlTemp(int t, int tag, Real3_t *vel_ar, Real_t zeta, int Nmol, Real_t dt) {
#if defined(REAL_AS_SINGLE)
   //Real_t temp_ctrl = 1.0 - dt * zeta;
   Real_t temp_ctrl = expf(-1.0 * dt * zeta);
#elif defined(REAL_AS_DOUBLE)
   //Real_t temp_ctrl = 1.0 - dt * zeta;
   Real_t temp_ctrl = exp(-1.0 * dt * zeta);
#endif
   for (int i = threadIdx.x; i < Nmol; i += blockDim.x) {
#if 1 //debug
      if (!isfinite(vel_ar[i].x) || !isfinite(vel_ar[i].y) || !isfinite(vel_ar[i].z)) {
	 printf("ERROR: %s(%d) before, t= %d, vel_ar[%d] = %f %f %f, temp_ctrl= %f, zeta = %f\n",
		__func__, tag, t,
		i, vel_ar[i].x, vel_ar[i].y, vel_ar[i].z, temp_ctrl, zeta);
	 return -2;
      }
#endif
      vel_ar[i].x *= temp_ctrl;
      vel_ar[i].y *= temp_ctrl;
      vel_ar[i].z *= temp_ctrl;
#if 1 //debug
      if (!isfinite(vel_ar[i].x) || !isfinite(vel_ar[i].y) || !isfinite(vel_ar[i].z)) {
	 printf("ERROR: %s(%d) after, t= %d, vel_ar[%d] = %f %f %f, temp_ctrl= %f, zeta = %f\n",
	     __func__, tag, t,
		i, vel_ar[i].x, vel_ar[i].y, vel_ar[i].z, temp_ctrl, zeta);
	 return -2;
      }
#endif
   }
   return 0;
}

//==============================================================================
static void
calcZetaSum(Real_t &zeta_sum, Real_t zeta, Real_t dt) {
   zeta_sum += zeta * dt;
}

__device__ void
calcZeta(Real_t &zeta, Real_t curr_temp, Real_t Q, Real_t targ_temp, Real_t dt, int Nmol) {
   // Real_t g = 3.0 * (Real_t)Nmol;
   // zeta += (curr_temp - g * targ_temp) * dt / Q;
   // zeta = (curr_temp - targ_temp) * dt / Q;
   zeta = (sqrt(curr_temp) - sqrt(targ_temp)) * dt / Q;
}
//===============================================================================
//
// histo[Nrep][totol_bins];
// idx_rep[Nrep]
__inline__ static int
histIdxBegin(int rep_i, Simu_t &simu) {
   int total_bins = simu.histo_bins;
   int idx_begin = total_bins * rep_i;
   return idx_begin; 
}
__inline__ static int
histIdxEnd(int rep_i, Simu_t &simu) {
   int total_bins = simu.histo_bins;
   int idx_end = total_bins * (rep_i + 1) - 1;
   return idx_end; 
}

void
checkEnergyVal(int t0, const Real_t *h_energy, int len) {
   //  printf("%s(): check if value of energy is obviously wrong...\n", __func__);
   const Real_t upper_bound = -1.0;
   const int    max_err_cnt = 10;
   int   err_cnt = 0;
   for (int i = 0; i < len; i++) {
      if (h_energy[i] > upper_bound) {
	 if (err_cnt == 0 && i > 0) {
	    printf("%s(): step= %d , h_energy[%d]= %f (just before err).\n", __func__, t0, i-1, h_energy[i-1]);
	 }
	 printf("%s(): step= %d , h_energy[%d]= %f .\n", __func__, t0, i, h_energy[i]);
	 err_cnt++;
	 if (err_cnt > max_err_cnt) {
#if 0
	    die("%s(): detect error %d times\n", __func__, max_err_cnt);
#endif
	 }
      }
   }
  //  printf("%s(): passed.\n", __func__);
}

//===============================================================================
void                                                    // very important //
calcHistogram(int *histo_ar, Remd_t &remd, Simu_t &simu) {
   debug_print(5, "%s(): Entering\n", __func__);

   int    ene_idx;                                    // index of energy array. //
   double offset_ene;
   int    idx_hist;
   int   *sort_temper = remd.sort_temper;

   for (int temp_i = 0; temp_i < remd.Nrep; temp_i++) {
      for (int t = 0; t < simu.step_exch; t++) {
	 ene_idx = (sort_temper[temp_i] * simu.step_exch) + t; //pick a value of energy.
	 if (ene_idx < 0) { 
	    fprintf(stderr, "ERROR in %s: ene_idx must have non-nega value.\n", __func__);
	    continue;
	 }
	 offset_ene = (double)remd.h_energy[ene_idx] - (double)simu.ene_min;
	 debug_print(5, "%s(): remp_i = %d, t = %d, offset_ene = %f\n", __func__, temp_i, t, offset_ene);
	 //idx_hist = (int)(offset_ene / (double)simu.delta_ene) + i * simu.histo_bins;
	 idx_hist = histIdxBegin(temp_i, simu) + (int)(offset_ene / (double)simu.delta_ene);
	 
	 if (idx_hist >= histIdxBegin(temp_i, simu) && idx_hist <= histIdxEnd(temp_i, simu)) {
	    histo_ar[idx_hist] += 1;
	 }
	 else {
#if defined(ENERGY_WARNING) && (ENERGY_WARNING > 0)
	    fprintf(stderr, "(-_-) warning: %s() energy value %f [J/mol] didn't be counted to histogram.\n",
		    __func__, remd.h_energy[ene_idx]);
#endif
	 }
      }
   }
   debug_print(5, "%s(): Exiting\n", __func__);
}
//===============================================================================
void
saveHistogram(const int *hist, Remd_t &remd, Simu_t &simu) {
   debug_print(2, "%s(): Entering\n", __func__);

   FILE *fp;
   char filename[1024];
   int total_bins = simu.histo_bins;
   int ene_i, rep_i;
   int j, j_start, j_stop;

   for (rep_i = 0; rep_i < remd.Nrep; rep_i++) {
      debug_print(5, "%s(): saving rep_i = %d / %d\n", __func__, rep_i, remd.Nrep - 1);
      sprintf(filename, "%s/h%05d.dat", simu.odir_histogram, rep_i);
      if ((fp = fopen(filename, "w")) == NULL) {
	 die("fopen error.\n");
      }
    
      ene_i = 0;
      j_start = histIdxBegin(rep_i, simu);
      j_stop  = histIdxEnd(rep_i, simu);
      for (j=j_start; j<=j_stop; j++) {
	 debug_print(5, "%s(): rep_i = %d, j = %d\n", __func__, rep_i, j);
	 if (j > remd.Nrep * total_bins) {
	    die("too large array_index.--> %d\n", j);
	 }
	 // ******************** //
	 // *  Save HISTOGRAM  * //
	 // ******************** //
	 fprintf(fp, "%7.5f  %15d\n", simu.ene_min + (simu.delta_ene * (Real_t)ene_i), hist[j]);
	 ene_i++;
      }
      fclose(fp);
   }

   debug_print(2, "%s(): Exiting\n", __func__);
}
//===============================================================================
void 
saveAccRatio(Remd_t &remd, int step_numkernel) {
   debug_print(2, "%s(): Entering\n", __func__);

   FILE *fp;
   char savepath[256];
   char filename[256];
   printf("info: saving acceptance ratio to file \"accept_ratio.rep\".\n");
   getPathToSaveRoot(savepath);
   sprintf(filename, "%s/accept_ratio.rep", savepath);
   if ((fp = fopen(filename, "w")) == NULL){
      die("file open error.\n");
   }
   for (int i = 0; i < remd.Nrep - 1; i++) {
      fprintf(fp, "T%d<-->T%d %4.1f\n", i, i + 1,
	      (double)remd.acc_ratio[i] / (double)step_numkernel * 100.0 * 2.0);
   }
   fclose(fp);

   debug_print(2, "%s(): Exiting\n", __func__);
}
//===============================================================================
static void
swapTemp(Real_t *temp_ar, int idx1, int idx2) {
   Real_t buf;
   //printf("%f <--> %f\n", temp_host[idx1], temp_host[idx2]);
   buf = temp_ar[idx1];
   temp_ar[idx1] = temp_ar[idx2];
   temp_ar[idx2] = buf;
}
//===============================================================================
void exchTemp(int t0, Remd_t &remd, Simu_t &simu) {
   int    rep_i, rep_j;
   Real_t ene1, ene2;
   Real_t delta;
   double P, rand;
   int   *sort_temper = remd.sort_temper;

   Real_t  *temp_ar = remd.h_temp_ar;
   Real_t  *energy  = remd.h_energy;
   int      step_exch = simu.step_exch;

   for (int i=0; i<remd.Nrep; i++) {
      remd.h_exch_ar[i] = 0;
   }

   for (int i = t0 % 2; i < remd.Nrep - 1; i+=2) {
      rep_i = sort_temper[i];     // select one neighboring pair
      rep_j = sort_temper[i+1];

      debug_print(5, "%s(): t0= %5d, i= %4d, rep_i= %4d, rep_j= %4d : ", __func__, t0, i, rep_i, rep_j);
	
      delta = (1.0 / temp_ar[rep_j] - 1.0 / temp_ar[rep_i]) *
	      (energy[step_exch * (rep_i + 1) - 1] - energy[step_exch * (rep_j + 1) - 1]);
      P = exp(-delta);
      rand = drand48();
      
      debug_print(5, "%s():P[%3d/%3d]= %f , rand= %f\n", __func__, i, remd.Nrep, P, rand);

      if (P > 1.0 || P > rand) {
	 sort_temper[i]   = rep_j; // swap sort_temper[]
	 sort_temper[i+1] = rep_i; //
	    
	 remd.h_exch_ar[rep_i] = 1;
	 remd.h_exch_ar[rep_j] = 1;	   
 
	 swapTemp(temp_ar, rep_i, rep_j);
	 remd.acc_ratio[rep_i] += 1;
      }
   }
   
   for (int i=0; i<remd.Nrep; i++) {
       printf("%s(): t0=%d, h_exch_ar[%d]= %d\n", __func__, t0, i, remd.h_exch_ar[i]);
   }
   
} // exchTemp

//==============================================================================
// memo: imported from "sub.cu".
//------------------------------------------------------------------------------
static Real_t
hamiltonian(Real_t kinetic, Real_t potential, Real_t mass_stat, Real_t zeta,
	    Real_t zeta_sum, Real_t set_temp, int Nmol) {
   return (kinetic + potential +  
	   0.5 * mass_stat * (zeta * zeta) +
	   3.0 * Nmol * set_temp * zeta_sum);
}

//--- EOF ---
