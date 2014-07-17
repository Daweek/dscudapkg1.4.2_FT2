#ifndef INTEG_H
#define INTEG_H

void simRemd(Remd_t &remd, Simu_t &simu);

__device__ int  integPos(int t, Real3_t *pos_ar, Real3_t *vel_ar, int Nmol, Real_t dt, Real_t cellsize);

__device__ int  integVel(int t, int tag, Real3_t *vel_ar, Real3_t *foc_ar, int Nmol, Real_t mass, Real_t dt);

__device__ void killMomentum(int t, Real3_t *velo_ar, Real_t mass, int Nmol, Real3_t *shared);

__device__ void calcVelScale(Real_t targ_temp, const Real3_t *vel_ar, Real_t mass, int Nmol, Real3_t *shared);

__device__ Real_t meanPotential(Real_t *potential, int Nmol, Real3_t *smem);
__device__ void calcZeta(Real_t&, Real_t, Real_t, Real_t, Real_t, int);
__device__ void reductionClear3D(Real3_t *ar, int size);
__device__ void reductionSet3D(Real3_t *dst, const Real3_t *src, int size); 
__device__ void reductionSum1D(Real_t  *sum, Real_t  *ar, int size);
__device__ void reductionSum3D(Real3_t *sum, Real3_t *ar, int size);
__device__ void scaleVelo(int t, Real3_t *velo_ar, Real_t scale, int Nmol);
__device__ void measTemper(Real_t *temper, const Real3_t *vel_ar, Real_t mass, int Nmol, Real3_t *shared);

#endif
