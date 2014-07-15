#ifndef COMM_SAVE_H
#define COMM_SAVE_H

// copy*() //
void copyPos(int rep_i, CopyKind_t dir);
void copyVel(int rep_i, CopyKind_t dir);
void copyFoc(int rep_i, CopyKind_t dir);
void copyEnergy(CopyKind_t dir, Remd_t &remd, Simu_t &simu);
void copyTempTarg(CopyKind_t dir);
void copyTempMeas(int rep_i, CopyKind_t dir);
void copyExch(CopyKind_t dir, Remd_t &remd, Simu_t &simu);
// save*() //
void saveLocalPos(Remd_t &);
void saveLocalVel(Remd_t &);
void saveLocalFoc(Remd_t &);
void saveLocalTemp(Remd_t &);

void savePos(int rep_i, Real_t temp, long simstep);
void saveVel(int rep_i, long simstep);
void saveFoc(int rep_i, long simstep, Real_t *potential_ar=NULL);
void saveTempMeas(int rep_i, long simstep);
void saveEne(Remd_t &remd, long simstep);
void saveTempTarg(Remd_t &remd, long simstep);
void saveSorted(Remd_t &remd, long simstep);

void savePosAll(int t);
void saveVelAll(int t);
void saveFocAll(int t);
void saveTempMeasAll(int t);
#endif //COMM_SAVE_H
