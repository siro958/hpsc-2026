#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 16;
  alignas(64) float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  
  __m512 v_xj = _mm512_load_ps(x);
  __m512 v_yj = _mm512_load_ps(y);
  __m512 v_mj = _mm512_load_ps(m);
  
  for(int i=0; i<N; i++) {
    __m512 v_xi = _mm512_set1_ps(x[i]);
    __m512 v_yi = _mm512_set1_ps(y[i]);

    __m512 v_rx = _mm512_sub_ps(v_xi, v_xj);
    __m512 v_ry = _mm512_sub_ps(v_yi, v_yj);
    __m512 v_r2 = _mm512_add_ps(_mm512_mul_ps(v_rx, v_rx), _mm512_mul_ps(v_ry, v_ry));
    
    __mmask16 mask = ~(1 << i);
    __m512 v_inv_r = _mm512_rsqrt14_ps(v_r2);
    v_inv_r = _mm512_maskz_mov_ps(mask, v_inv_r);

    __m512 v_inv_r3 = _mm512_mul_ps(v_inv_r, _mm512_mul_ps(v_inv_r, v_inv_r));

    __m512 v_fx = _mm512_mul_ps(v_rx, _mm512_mul_ps(v_mj, v_inv_r3));
    __m512 v_fy = _mm512_mul_ps(v_ry, _mm512_mul_ps(v_mj, v_inv_r3));

    fx[i] -= _mm512_reduce_add_ps(v_fx);
    fy[i] -= _mm512_reduce_add_ps(v_fy);
    
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
