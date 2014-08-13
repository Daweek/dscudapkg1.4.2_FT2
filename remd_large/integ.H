#ifndef INTEG_H
#define INTEG_H

void simRemd(Remd_t &remd, Simu_t &simu);

__device__ int  integPos(int t, Real3_t *pos_ar, Real3_t *vel_ar, Nmol_t Nmol, Real_t dt, Real_t cellsize);

__device__ int  integVel(int t, int tag, Real3_t *vel_ar, Real3_t *foc_ar, Nmol_t Nmol, Real_t mass, Real_t dt);

__device__ void killMomentum(int t, Real3_t *velo_ar, Real_t mass, Nmol_t Nmol, Real3_t *shared);

__device__ void calcVelScale(Real_t targ_temp, const Real3_t *vel_ar, Real_t mass, Nmol_t Nmol, Real3_t *shared);

__device__ void meanPotential(Real_t *mean, Real_t *poten_ar, Nmol_t Nmol, Real3_t *shared);
__device__ void calcZeta(Real_t&, Real_t, Real_t, Real_t, Real_t, int);
__device__ void reductClear3D(Real3_t *ar, int size);
__device__ void reductSet3D(Real3_t *dst, const Real3_t *src, int size); 
__device__ void reductSum1D(Real_t  *sum, Real_t  *ar, int size);
__device__ void reductSum3D(Real3_t *sum, Real3_t *ar, int size);
__device__ void scaleVelo(int t, Real3_t *velo_ar, Real_t scale, Nmol_t Nmol);
__device__ void measTemper(Real_t *temper, const Real3_t *vel_ar, Real_t mass, Nmol_t Nmol, Real3_t *shared);

#endif
