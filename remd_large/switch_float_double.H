//                             -*- Mode: C++ -*-
// Filename         : switch_float_double.H
// Description      :
// Author           : Minoru Oikawa (m_oikawa@amber.plala.or.jp)
// Created On       : 2013-09-09 11:49:27
// Last Modified By : Minoru Oikawa
// Last Modified On : 2014-01-27 12:13:25
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#ifndef SWITCH_FLOAT_DOUBLE_H
#define SWITCH_FLOAT_DOUBLE_H

#include "mytools.H"
//======================================
// Switch one of three.
//--------------------------------------
#define REAL_AS_SINGLE
//#define REAL_AS_DOUBLE
//#define REAL_AS_LONGDOUBLE
//--------------------------------------

#if defined(REAL_AS_SINGLE)
typedef float Real_t;

#elif defined(REAL_AS_DOUBLE)
typedef double Real_t;

#elif defined(REAL_AS_LONGDOUBLE)
typedef long double Real_t;

#else
typedef float Real_t;      

#endif // defined(REAL_AS_SINGLE)

typedef struct Real2 {
  Real_t x, y;
} Real2_t;

typedef struct Real3 {
  Real_t x, y, z;
} Real3_t;

typedef struct Real4 {
  Real_t x, y, z, w;
} Real4_t;

static void
setVal(Real2_t &v, Real_t a, Real_t b){
  v.x = a;  v.y = b;
}
static void
setVal(Real3_t &v, Real_t a, Real_t b, Real_t c){
  v.x = a;  v.y = b;  v.z = c;
}
static void
setVal(Real3_t &dst, const Real3_t &src){
  dst.x = src.x; dst.y = src.y; dst.z = src.z;
}
static void
setZero(Real3_t &v){
  v.x = v.y = v.z = 0.0;
}
static void
setGaussRand(Real3_t &v){
  v.x = gaussRand();
  v.y = gaussRand();
  v.z = gaussRand();
}
static void
copy(Real3_t &dst, Real3_t src)
{
  dst.x = src.x;   dst.y = src.y;   dst.z = src.z;
}
static void
add(Real3_t &dst, Real3_t src){
  dst.x += src.x;
  dst.y += src.y;
  dst.z += src.z;
}
static void
sub(Real3_t &dst, Real3_t src) {
  dst.x -= src.x;
  dst.y -= src.y;
  dst.z -= src.z;
}
static void
mult(Real3_t &dst, Real_t m) {
  dst.x *= m;   dst.y *= m;   dst.z *= m;
}
static void
div(Real3_t &dst, Real_t d) {
  dst.x /= d;   dst.y /= d;   dst.z /= d;
}
#endif //SWITCH_FLOAT_DOUBLE_H
