
#ifndef INTEG_H
#define INTEG_H

void simRemd(Remd_t &remd, Simu_t &simu);

__host__   void integPos_hst(Real3_t *pos_ar, const Real3_t *vel_ar, int Nmol, Real_t dt, Real_t cellsize);
__device__ int  integPos_dev(int t, Real3_t *pos_ar, Real3_t *vel_ar, int Nmol, Real_t dt, Real_t cellsize);

__host__   void integVel_hst(Real3_t *vel_ar, const Real3_t *foc_ar, int Nmol, Real_t mass, Real_t dt);
__device__ int  integVel_dev(int t, int tag, Real3_t *vel_ar, Real3_t *foc_ar, int Nmol, Real_t mass, Real_t dt);

__host__ Real_t measTemper_hst(const Real3_t *vel_ar, Real_t mass, int Nmol);
__device__ void measTemper_dev(const Real3_t *vel_ar, Real_t mass, int Nmol, Real3_t *shared_mem);

__host__   void killMomentum_hst(Real3_t *velo_ar, Real_t mass, int Nmol);
__device__ void killMomentum_dev(Real3_t *velo_ar, Real_t mass, int Nmol, Real3_t *shared_mem);

__host__   void scaleVelo_hst(Real3_t *velo_ar, Real_t scale, int Nmol);
__device__ void scaleVelo_dev(Real3_t *velo_ar, Real_t scale, int Nmol);

__host__ Real_t calcVelScale_hst(Real_t targ_temp, const Real3_t *vel_ar, Real_t mass, int Nmol);
__device__ void calcVelScale_dev(Real_t targ_temp, const Real3_t *vel_ar, Real_t mass, int Nmol, Real3_t *shared_mem);

__host__ Real_t meanPotential_hst(Real_t *potential, int Nmol);
__device__ void meanPotential_dev(Real_t *potential, int Nmol, Real3_t *smem);

__host__   void calcZeta_hst(Real_t&, Real_t, Real_t, Real_t, Real_t, int);
__device__ void calcZeta_dev(Real_t&, Real_t, Real_t, Real_t, Real_t, int);

__device__ void reductionClear1D(Real_t *ar, int size);
__device__ void reductionClear3D(Real3_t *ar, int size);
__device__ void reductionSet1D(Real_t *dst, const Real_t *src, int size);
__device__ void reductionSet3D(Real3_t *dst, const Real3_t *src, int size); 
__device__ void reductionSum1D(Real_t *ar, int size); // must be 2^N, and less than 2049.
__device__ void reductionSum3D(Real3_t *ar, int size); // must be 2^N, and less than 2049.

__device__ void calcVelScale(Real_t targ_temp, const Real3_t *vel_ar, Real_t mass, int Nmol, Real3_t *shared_mem);
__device__ void killMomentum(Real3_t *velo_ar, Real_t mass, int Nmol, Real3_t *shared_mem);
__device__ void scaleVelo(Real3_t *velo_ar, Real_t scale, int Nmol);
__device__ void measTemper(const Real3_t *vel_ar, Real_t mass, int Nmol, Real3_t *shared_mem);

#endif
