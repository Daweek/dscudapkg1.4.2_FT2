//                             -*- Mode: C++ -*-
// Filename         : calc_force_hst.C
// Description      :
// Author           : Minoru Oikawa (m_oikawa@amber.plala.or.jp)
// Created On       : 2013-08-25 11:49:27
// Last Modified By : Minoru Oikawa
// Last Modified On : 2014-01-31 16:45:26
// Update Count     : 0.0
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "switch_float_double.H"
#include "remd_typedef.H"
#include "calc_force.H"
//==============================================================================
// return a value of square of abs(vector r).
//------------------------------------------------------------------------------
__host__ __device__ Real_t
normSq(const Real3_t &r) {
   return (r.x * r.x) + (r.y * r.y) + (r.z * r.z);
}
//==============================================================================
// description:  calculate values of Lennard-Jones potential and force vector.
// outputs:      "force", "potential".
// inputs:       "r_ij"   is vector equal to (i - j).
//               "r_normsq" is equal to abs(r_vec)^2.
//               "sigma, epsilon" are parameters in lennard-jones expression.
//------------------------------------------------------------------------------
__host__ __device__ static void
calcLjCore(Real3_t &force, Real_t &potential,
	   Real3_t r_ij, Real_t r_normsq, Real_t lj_sigma, Real_t lj_epsilon) {
  Real_t  sigma_r_pow2  = (lj_sigma * lj_sigma) / r_normsq;
  Real_t  sigma_r_pow6  = sigma_r_pow2 * sigma_r_pow2 * sigma_r_pow2;
  Real_t  sigma_r_pow12 = sigma_r_pow6 * sigma_r_pow6;
  Real_t  f_scaled_factor = (UNIT_TIME * UNIT_TIME) / 
    (UNIT_LENGTH * UNIT_LENGTH) * (UNIT_ENERGY / UNIT_MASS);
  Real_t  f_tmp;

  f_tmp = lj_epsilon * (48.0 * sigma_r_pow12 - 24.0 * sigma_r_pow6) / r_normsq;
  f_tmp *= f_scaled_factor;

  /* outputs */
  force.x = f_tmp * r_ij.x; /* LJ-force.x */
  force.y = f_tmp * r_ij.y; /* LJ-force.y */
  force.z = f_tmp * r_ij.z; /* LJ-force.z */
#ifdef RCUT_COUNT //
  potential = 1.0;  // ++
#else
  potential = 4.0 * lj_epsilon * (sigma_r_pow12 - sigma_r_pow6);
#endif
}
//==============================================================================
// getClosestPBC
// outputs: "r_ij"
// inputs:  "pos_i, pos_j", "cellsize"
//------------------------------------------------------------------------------
__host__ __device__ void
getClosestPBC(Real3_t &r_ij,
	      const Real3_t &pos_i, const Real3_t &pos_j, Real_t cellsize) {
   Real3_t sub;
   Real3_t round;
   sub.x = pos_i.x - pos_j.x;
   sub.y = pos_i.y - pos_j.y;
   sub.z = pos_i.z - pos_j.z;
   
   round.x = rint(sub.x / cellsize);
   round.y = rint(sub.y / cellsize);
   round.z = rint(sub.z / cellsize);

   r_ij.x = sub.x - (cellsize * round.x);
   r_ij.y = sub.y - (cellsize * round.y);
   r_ij.z = sub.z - (cellsize * round.z);
}
//===============================================================================
// lj(), calculate Lennard-Jones potential/force
//       outputs : f_ij, p_ij.
//-------------------------------------------------------------------------------
__host__ __device__ void
lj(Real3_t &f_ij, Real_t &p_ij, const Real3_t &pos_i, const Real3_t &pos_j,
   Real_t rcut, Real_t cellsize, Real_t lj_sigma, Real_t lj_epsilon) {
   
   Real3_t r_ij; // closest atom position vector in neighboring cell.
   Real_t  rcut_sq = rcut * rcut;
   Real_t  r_normsq;
   getClosestPBC(r_ij, pos_i, pos_j, cellsize);
   r_normsq = normSq(r_ij);
   if ( rcut_sq > r_normsq ) { // calc inside the "rcut"
      calcLjCore(f_ij, p_ij, r_ij, r_normsq, lj_sigma, lj_epsilon);
   }
   else {
      f_ij.x = f_ij.y = f_ij.z = p_ij = 0.0;
   }
}
//==============================================================================
// calcForce()
// output: f_ar[Nmol], poten_ar[Nmol]
//------------------------------------------------------------------------------
void
zeroForce_hst(Real3_t *f_ar, Real_t *poten_ar,
	      const Real3_t *pos_ar, int Nmol, Real_t rcut,
	      Real_t cellsize, Real_t lj_sigma, Real_t lj_epsilon) {
   Real3_t f_ij;
   Real_t  p_ij;
   for (int i=0; i<Nmol; i++) {
      f_ar[i].x = f_ar[i].y = f_ar[i].z = 0.0;
      poten_ar[i] = 0.0;
   }
   for (int i=0; i<(Nmol - 1); i++) {
      for (int j=(i + 1); j<Nmol; j++) {
	 lj(f_ij, p_ij, pos_ar[i], pos_ar[j], rcut, cellsize, lj_sigma, lj_epsilon);
	 poten_ar[i] += p_ij;
      }
   }
}

void
calcForce_hst(Real3_t *f_ar, Real_t *poten_ar,
	      const Real3_t *pos_ar, int Nmol, Real_t rcut,
	      Real_t cellsize, Real_t lj_sigma, Real_t lj_epsilon) {
   Real3_t f_ij;
   Real_t  p_ij;
   for (int i=0; i<Nmol; i++) {
      f_ar[i].x = f_ar[i].y = f_ar[i].z = 0.0;
      poten_ar[i] = 0.0;
   }
   for (int i=0; i<(Nmol - 1); i++) {
      for (int j=(i + 1); j<Nmol; j++) {
	 lj(f_ij, p_ij, pos_ar[i], pos_ar[j], rcut, cellsize, lj_sigma, lj_epsilon);
	 
	 f_ar[i].x += f_ij.x; // * i <-- j */
	 f_ar[i].y += f_ij.y;
	 f_ar[i].z += f_ij.z;
	 poten_ar[i] += p_ij;
      
	 f_ar[j].x -= f_ij.x; // * i --> j */
	 f_ar[j].y -= f_ij.y;
	 f_ar[j].z -= f_ij.z;
	 //poten_ar[j] += p_ij; // not "-=" but "+=" ,  refer to Muguruma et al(2004), avoid double counting.
      }
   }
}
__device__ void
zeroForce_dev(Real3_t *f_ar, Real_t *poten_ar,
	      const Real3_t *pos_ar, int Nmol, Real_t rcut,
	      Real_t cellsize, Real_t lj_sigma, Real_t lj_epsilon, Real3_t *shared_mem) {
   Real3_t force_i_j;
   Real3_t force_i_all;
   Real_t  poten_i_j;
   Real_t  poten_i_all;
  
   __syncthreads();
   for (int i=threadIdx.x; i<Nmol; i+=blockDim.x) {    // copy pos to shared.
      shared_mem[i].x = pos_ar[i].x;
      shared_mem[i].y = pos_ar[i].y;
      shared_mem[i].z = pos_ar[i].z;
   }
   __syncthreads();

   for (int i=threadIdx.x; i<Nmol; i+=blockDim.x) {
      force_i_all.x = force_i_all.y = force_i_all.z = 0.0;
      poten_i_all = 0.0;
      for (int j=0; j<Nmol; j++) {
	 if (i != j) {
	    lj(force_i_j, poten_i_j,
	       shared_mem[i], shared_mem[j], rcut, cellsize, lj_sigma, lj_epsilon);
	    poten_i_all   += poten_i_j;
	 }
      }
      poten_ar[i] = poten_i_all / 2.0;  // 2.0 correct double counting.
   }
   __syncthreads();  
} 
//==================================================================================
__device__ void
calcForce_dev(Real3_t *f_ar, Real_t *poten_ar,
	      const Real3_t *pos_ar, int Nmol, Real_t rcut,
	      Real_t cellsize, Real_t lj_sigma, Real_t lj_epsilon, Real3_t *shared_mem) {
   Real3_t force_i_j;
   Real3_t force_i_all;
   Real_t  poten_i_j;
   Real_t  poten_i_all;

   for (int i=threadIdx.x; i<Nmol; i += blockDim.x) {    // copy pos to shared.
      shared_mem[i].x = pos_ar[i].x;
      shared_mem[i].y = pos_ar[i].y;
      shared_mem[i].z = pos_ar[i].z;
   }
   __syncthreads();

   for (int i=threadIdx.x; i<Nmol; i+=blockDim.x) {
      force_i_all.x = force_i_all.y = force_i_all.z = 0.0;
      poten_i_all = 0.0;
      for (int j=0; j<Nmol; j++) {
	 if (i != j) {
	    lj(force_i_j, poten_i_j,
	       shared_mem[i], shared_mem[j], rcut, cellsize, lj_sigma, lj_epsilon);
	    
	    force_i_all.x += force_i_j.x;
	    force_i_all.y += force_i_j.y;
	    force_i_all.z += force_i_j.z;
	    poten_i_all   += poten_i_j;
	 }
      }
      f_ar[i].x = force_i_all.x;                          // copy force to global mem.
      f_ar[i].y = force_i_all.y;
      f_ar[i].z = force_i_all.z;
      //poten_ar[i] = poten_i_all / 2.0;  // 2.0 correct double counting.
      poten_ar[i] = poten_i_all; 
   }
}

