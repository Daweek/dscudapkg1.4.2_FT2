//                             -*- Mode: C++ -*-
// Filename         : calc_force.H
// Description      :
// Author           : Minoru Oikawa (m_oikawa@amber.plala.or.jp)
// Created On       : 2013-09-09 17:49:27
// Last Modified By : Minoru Oikawa
// Last Modified On : 2013-09-30 21:38:41
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#ifndef CALC_FORCE_H
#define CALC_FORCE_H

__host__
void calcForce_hst(Real3_t *force_ar, Real_t *poten_ar, const Real3_t *posi_ar,
		   int Nmol, Real_t rcut, Real_t cellsize, Real_t sigma,
		   Real_t epsilon);

__device__
void calcForce_dev(Real3_t *force_ar, Real_t *poten_ar, const Real3_t *posi_ar,
		   int Nmol, Real_t rcut, Real_t cellsize, Real_t sigma,
		   Real_t epsilon, Real3_t *shared_mem);

__host__
void zeroForce_hst(Real3_t *force_ar, Real_t *poten_ar, const Real3_t *posi_ar,
		   int Nmol, Real_t rcut, Real_t cellsize, Real_t sigma,
		   Real_t epsilon);

__device__
void zeroForce_dev(Real3_t *force_ar, Real_t *poten_ar, const Real3_t *posi_ar,
		   int Nmol, Real_t rcut, Real_t cellsize, Real_t sigma,
		   Real_t epsilon, Real3_t *shared_mem);


__host__ __device__
void lj(Real3_t &f_ij, Real_t &p_ij, const Real3_t &pos_i, const Real3_t &pos_j,
	Real_t rcut, Real_t cellsize, Real_t lj_sigma, Real_t lj_epsilon);

#endif //CALC_FORCE_H
