#include "gpu_thread_pool.hpp"

typedef double FloatingPoint_t;

struct common_data_t {
	double * A;
	double * B;
	double * C;
	double alpha;
	double beta;
	int m;
	int n;
	int k;
	int lda;
	int ldb;
	int ldc;
	int offsetA;
	int offsetB;
};

struct task_t {
	int by, bx;
};

static inline __device__
double tex_fetch(texture<int2> tex_ref, int coord) {
	int2 v = tex1Dfetch(tex_ref, coord);
	return __hiloint2double(v.y, v.x);
}

texture<int2, 0x01, cudaReadModeElementType> tex_ref_A;
texture<int2, 0x01, cudaReadModeElementType> tex_ref_B;

template < >
__device__
void do_task < task_t, common_data_t, 0 > ( task_t const & task, common_data_t const & common_data ) {
    int M = common_data.m;
    int N = common_data.n;
    int K = common_data.k;
//  double * A = common_data.A;
    int LDA = common_data.lda;
//  double * B = common_data.B;
    int LDB = common_data.ldb;
    double * C = common_data.C;
    int LDC = common_data.ldc;
    double alpha = common_data.alpha;
    double beta = common_data.beta;
    int offsetA = common_data.offsetA;
    int offsetB = common_data.offsetB;

    int idx = threadIdx.x;
    int idy = threadIdx.y;

    int idt = 16 * idy + idx;

    int idxA = idt % 16;
    int idyA = idt / 16;

    int idxB = idt % 16;
    int idyB = idt / 16;

//  int blx = blockIdx.x;
    int blx = task.bx;
//  int bly = blockIdx.y;
    int bly = task.by;

    __shared__ double sA[16][64 +1];
    __shared__ double sB[64][16 +1];


    double rC[( 64 / 16 )][( 64 / 16 )];
    double rA[( 64 / 16 )];
    double rB[( 64 / 16 )];





        double ra[16/16][64/16];




        double rb[64/16][16/16];






            int coord_A = offsetA + blx*64 + idyA*LDA+idxA;




            int coord_B = offsetB + bly*64*LDB + idyB*LDB+idxB;
# 165 "gemm_stencil.cu"
    int m, n, k, kk;


#pragma unroll
    for (n = 0; n < ( 64 / 16 ); n++)
#pragma unroll
      for (m = 0; m < ( 64 / 16 ); m++)
        rC[n][m] = (0.0);
# 182 "gemm_stencil.cu"
#pragma unroll
        for (n = 0; n < 16; n += 16)
#pragma unroll
          for (m = 0; m < 64; m += 16)
            sA[n+idyA][m+idxA] = tex_fetch(tex_ref_A, coord_A + n*LDA+m);
# 197 "gemm_stencil.cu"
#pragma unroll
        for (n = 0; n < 64; n += 16)
#pragma unroll
          for (m = 0; m < 16; m += 16)
            sB[n+idyB][m+idxB] = tex_fetch(tex_ref_B, coord_B + n*LDB+m);


    __syncthreads();

    for (kk = 0; kk < K-16; kk += 16)
    {




                coord_A += 16*LDA;




                coord_B += 16;
# 240 "gemm_stencil.cu"
#pragma unroll
            for (n = 0; n < 16/16; n++)
#pragma unroll
              for (m = 0; m < 64/16; m++)
                ra[n][m] = tex_fetch(tex_ref_A, coord_A + n*16*LDA+m*16);
# 255 "gemm_stencil.cu"
#pragma unroll
            for (n = 0; n < 64/16; n++)
#pragma unroll
              for (m = 0; m < 16/16; m++)
                rb[n][m] = tex_fetch(tex_ref_B, coord_B + n*16*LDB+m*16);



#pragma unroll
        for (k = 0; k < 16; k++)
        {

#pragma unroll
            for (m = 0; m < ( 64 / 16 ); m++)
                rA[m] = sA[k][m*16 +idx];


#pragma unroll
            for (n = 0; n < ( 64 / 16 ); n++)
                rB[n] = sB[n*16 +idy][k];


#pragma unroll
            for (n = 0; n < ( 64 / 16 ); n++)
#pragma unroll
                for (m = 0; m < ( 64 / 16 ); m++)
# 291 "gemm_stencil.cu"
                        rC[n][m] += (rA[m]*rB[n]);


        }

        __syncthreads();
# 306 "gemm_stencil.cu"
#pragma unroll
            for (n = 0; n < 16/16; n++)
#pragma unroll
              for (m = 0; m < 64/16; m++)
                sA[n*16 +idyA][m*16 +idxA] = ra[n][m];
# 321 "gemm_stencil.cu"
#pragma unroll
            for (n = 0; n < 64/16; n++)
#pragma unroll
              for (m = 0; m < 16/16; m++)
                sB[n*16 +idyB][m*16 +idxB] = rb[n][m];


        __syncthreads();
    }


#pragma unroll
    for (k = 0; k < 16; k++)
    {

#pragma unroll
        for (m = 0; m < ( 64 / 16 ); m++)
            rA[m] = sA[k][m*16 +idx];


#pragma unroll
        for (n = 0; n < ( 64 / 16 ); n++)
            rB[n] = sB[n*16 +idy][k];


#pragma unroll
        for (n = 0; n < ( 64 / 16 ); n++)
#pragma unroll
            for (m = 0; m < ( 64 / 16 ); m++)
# 360 "gemm_stencil.cu"
                    rC[n][m] += (rA[m]*rB[n]);


    }


#pragma unroll
    for (n = 0; n < ( 64 / 16 ); n++) {
        int coord_dCn = bly*64 + n*16 +idy;
#pragma unroll
        for (m = 0; m < ( 64 / 16 ); m++) {
            int coord_dCm = blx*64 + m*16 +idx;
            if (coord_dCm < M && coord_dCn < N) {
                int offsC = coord_dCn*LDC + coord_dCm;

                double &regC = rC[n][m];
                double &memC = C[offsC];

                memC = ((alpha*regC)+(beta*memC));
            }
        }
    }

}

template < >
__device__
void do_task < task_t, common_data_t, 1 > ( task_t const & task, common_data_t const & common_data ) {
    int M = common_data.m;
    int N = common_data.n;
    int K = common_data.k;
//  double * A = common_data.A;
    int LDA = common_data.lda;
//  double * B = common_data.B;
    int LDB = common_data.ldb;
    double * C = common_data.C;
    int LDC = common_data.ldc;
    double alpha = common_data.alpha;
    double beta = common_data.beta;
    int offsetA = common_data.offsetA;
    int offsetB = common_data.offsetB;

    int idx = threadIdx.x;
    int idy = threadIdx.y;

    int idt = 16 * idy + idx;

    int idxA = idt % 16;
    int idyA = idt / 16;

    int idxB = idt % 16;
    int idyB = idt / 16;

//  int blx = blockIdx.x;
    int blx = task.bx;
//  int bly = blockIdx.y;
    int bly = task.by;

    __shared__ double sA[16][64 +1];
    __shared__ double sB[64][16 +1];


    FloatingPoint_t rC[( 64 / 16 )][( 64 / 16 )];
    FloatingPoint_t rA[( 64 / 16 )];
    FloatingPoint_t rB[( 64 / 16 )];





        FloatingPoint_t ra[16/16][64/16];


        FloatingPoint_t rb[16/16][64/16];
# 145 "gemm_stencil.cu"
            int coord_A = offsetA + blx*64 + idyA*LDA+idxA;


            int coord_B = offsetB + bly*64 + idyB*LDB+idxB;
# 165 "gemm_stencil.cu"
    int m, n, k, kk;


#pragma unroll
    for (n = 0; n < ( 64 / 16 ); n++)
#pragma unroll
      for (m = 0; m < ( 64 / 16 ); m++)
        rC[n][m] = (0.0);
# 182 "gemm_stencil.cu"
#pragma unroll
        for (n = 0; n < 16; n += 16)
#pragma unroll
          for (m = 0; m < 64; m += 16)
            sA[n+idyA][m+idxA] = tex_fetch(tex_ref_A, coord_A + n*LDA+m);




#pragma unroll
        for (n = 0; n < 16; n += 16)
#pragma unroll
          for (m = 0; m < 64; m += 16)
            sB[m+idxB][n+idyB] = tex_fetch(tex_ref_B, coord_B + n*LDB+m);
# 204 "gemm_stencil.cu"
    __syncthreads();

    for (kk = 0; kk < K-16; kk += 16)
    {




                coord_A += 16*LDA;


                coord_B += 16*LDB;
# 240 "gemm_stencil.cu"
#pragma unroll
            for (n = 0; n < 16/16; n++)
#pragma unroll
              for (m = 0; m < 64/16; m++)
                ra[n][m] = tex_fetch(tex_ref_A, coord_A + n*16*LDA+m*16);




#pragma unroll
            for (n = 0; n < 16/16; n++)
#pragma unroll
              for (m = 0; m < 64/16; m++)
                rb[n][m] = tex_fetch(tex_ref_B, coord_B + n*16*LDB+m*16);
# 263 "gemm_stencil.cu"
#pragma unroll
        for (k = 0; k < 16; k++)
        {

#pragma unroll
            for (m = 0; m < ( 64 / 16 ); m++)
                rA[m] = sA[k][m*16 +idx];


#pragma unroll
            for (n = 0; n < ( 64 / 16 ); n++)
                rB[n] = sB[n*16 +idy][k];


#pragma unroll
            for (n = 0; n < ( 64 / 16 ); n++)
#pragma unroll
                for (m = 0; m < ( 64 / 16 ); m++)
# 291 "gemm_stencil.cu"
                        rC[n][m] += (rA[m]*rB[n]);


        }

        __syncthreads();
# 306 "gemm_stencil.cu"
#pragma unroll
            for (n = 0; n < 16/16; n++)
#pragma unroll
              for (m = 0; m < 64/16; m++)
                sA[n*16 +idyA][m*16 +idxA] = ra[n][m];




#pragma unroll
            for (n = 0; n < 16/16; n++)
#pragma unroll
              for (m = 0; m < 64/16; m++)
                sB[m*16 +idxB][n*16 +idyB] = rb[n][m];
# 328 "gemm_stencil.cu"
        __syncthreads();
    }


#pragma unroll
    for (k = 0; k < 16; k++)
    {

#pragma unroll
        for (m = 0; m < ( 64 / 16 ); m++)
            rA[m] = sA[k][m*16 +idx];


#pragma unroll
        for (n = 0; n < ( 64 / 16 ); n++)
            rB[n] = sB[n*16 +idy][k];


#pragma unroll
        for (n = 0; n < ( 64 / 16 ); n++)
#pragma unroll
            for (m = 0; m < ( 64 / 16 ); m++)
# 360 "gemm_stencil.cu"
                    rC[n][m] += (rA[m]*rB[n]);


    }


#pragma unroll
    for (n = 0; n < ( 64 / 16 ); n++) {
        int coord_dCn = bly*64 + n*16 +idy;
#pragma unroll
        for (m = 0; m < ( 64 / 16 ); m++) {
            int coord_dCm = blx*64 + m*16 +idx;
            if (coord_dCm < M && coord_dCn < N) {
                int offsC = coord_dCn*LDC + coord_dCm;

                FloatingPoint_t &regC = rC[n][m];
                FloatingPoint_t &memC = C[offsC];

                memC = ((alpha*regC)+(beta*memC));
            }
        }
    }

}

template < >
__device__
void do_task < task_t, common_data_t, 2 > ( task_t const & task, common_data_t const & common_data ) {
    int M = common_data.m;
    int N = common_data.n;
    int K = common_data.k;
//  double * A = common_data.A;
    int LDA = common_data.lda;
//  double * B = common_data.B;
    int LDB = common_data.ldb;
    double * C = common_data.C;
    int LDC = common_data.ldc;
    double alpha = common_data.alpha;
    double beta = common_data.beta;
    int offsetA = common_data.offsetA;
    int offsetB = common_data.offsetB;

    int idx = threadIdx.x;
    int idy = threadIdx.y;

    int idt = 16 * idy + idx;

    int idxA = idt % 16;
    int idyA = idt / 16;

    int idxB = idt % 16;
    int idyB = idt / 16;

//  int blx = blockIdx.x;
    int blx = task.bx;
//  int bly = blockIdx.y;
    int bly = task.by;

    __shared__ double sA[16][64 +1];
    __shared__ double sB[64][16 +1];

    FloatingPoint_t rC[( 64 / 16 )][( 64 / 16 )];
    FloatingPoint_t rA[( 64 / 16 )];
    FloatingPoint_t rB[( 64 / 16 )];



        FloatingPoint_t ra[64/16][16/16];






        FloatingPoint_t rb[64/16][16/16];




            int coord_A = offsetA + blx*64*LDA + idyA*LDA+idxA;






            int coord_B = offsetB + bly*64*LDB + idyB*LDB+idxB;
# 165 "gemm_stencil.cu"
    int m, n, k, kk;


#pragma unroll
    for (n = 0; n < ( 64 / 16 ); n++)
#pragma unroll
      for (m = 0; m < ( 64 / 16 ); m++)
        rC[n][m] = (0.0);



#pragma unroll
        for (n = 0; n < 64; n += 16)
#pragma unroll
          for (m = 0; m < 16; m += 16)
            sA[m+idxA][n+idyA] = tex_fetch(tex_ref_A, coord_A + n*LDA+m);
# 197 "gemm_stencil.cu"
#pragma unroll
        for (n = 0; n < 64; n += 16)
#pragma unroll
          for (m = 0; m < 16; m += 16)
            sB[n+idyB][m+idxB] = tex_fetch(tex_ref_B, coord_B + n*LDB+m);


    __syncthreads();

    for (kk = 0; kk < K-16; kk += 16)
    {


                coord_A += 16;






                coord_B += 16;
# 234 "gemm_stencil.cu"
#pragma unroll
            for (n = 0; n < 64/16; n++)
#pragma unroll
              for (m = 0; m < 16/16; m++)
                ra[n][m] = tex_fetch(tex_ref_A, coord_A + n*16*LDA+m*16);
# 255 "gemm_stencil.cu"
#pragma unroll
            for (n = 0; n < 64/16; n++)
#pragma unroll
              for (m = 0; m < 16/16; m++)
                rb[n][m] = tex_fetch(tex_ref_B, coord_B + n*16*LDB+m*16);



#pragma unroll
        for (k = 0; k < 16; k++)
        {

#pragma unroll
            for (m = 0; m < ( 64 / 16 ); m++)
                rA[m] = sA[k][m*16 +idx];


#pragma unroll
            for (n = 0; n < ( 64 / 16 ); n++)
                rB[n] = sB[n*16 +idy][k];


#pragma unroll
            for (n = 0; n < ( 64 / 16 ); n++)
#pragma unroll
                for (m = 0; m < ( 64 / 16 ); m++)
# 291 "gemm_stencil.cu"
                        rC[n][m] += (rA[m]*rB[n]);


        }

        __syncthreads();



#pragma unroll
            for (n = 0; n < 64/16; n++)
#pragma unroll
              for (m = 0; m < 16/16; m++)
                sA[m*16 +idxA][n*16 +idyA] = ra[n][m];
# 321 "gemm_stencil.cu"
#pragma unroll
            for (n = 0; n < 64/16; n++)
#pragma unroll
              for (m = 0; m < 16/16; m++)
                sB[n*16 +idyB][m*16 +idxB] = rb[n][m];


        __syncthreads();
    }


#pragma unroll
    for (k = 0; k < 16; k++)
    {

#pragma unroll
        for (m = 0; m < ( 64 / 16 ); m++)
            rA[m] = sA[k][m*16 +idx];


#pragma unroll
        for (n = 0; n < ( 64 / 16 ); n++)
            rB[n] = sB[n*16 +idy][k];


#pragma unroll
        for (n = 0; n < ( 64 / 16 ); n++)
#pragma unroll
            for (m = 0; m < ( 64 / 16 ); m++)
# 360 "gemm_stencil.cu"
                    rC[n][m] += (rA[m]*rB[n]);


    }


#pragma unroll
    for (n = 0; n < ( 64 / 16 ); n++) {
        int coord_dCn = bly*64 + n*16 +idy;
#pragma unroll
        for (m = 0; m < ( 64 / 16 ); m++) {
            int coord_dCm = blx*64 + m*16 +idx;
            if (coord_dCm < M && coord_dCn < N) {
                int offsC = coord_dCn*LDC + coord_dCm;

                FloatingPoint_t &regC = rC[n][m];
                FloatingPoint_t &memC = C[offsC];

                memC = ((alpha*regC)+(beta*memC));
            }
        }
    }

}

template < >
__device__
void do_task < task_t, common_data_t, 3 > ( task_t const & task, common_data_t const & common_data ) {
    int M = common_data.m;
    int N = common_data.n;
    int K = common_data.k;
//  double * A = common_data.A;
    int LDA = common_data.lda;
//  double * B = common_data.B;
    int LDB = common_data.ldb;
    double * C = common_data.C;
    int LDC = common_data.ldc;
    double alpha = common_data.alpha;
    double beta = common_data.beta;
    int offsetA = common_data.offsetA;
    int offsetB = common_data.offsetB;

    int idx = threadIdx.x;
    int idy = threadIdx.y;

    int idt = 16 * idy + idx;

    int idxA = idt % 16;
    int idyA = idt / 16;

    int idxB = idt % 16;
    int idyB = idt / 16;

//  int blx = blockIdx.x;
    int blx = task.bx;
//  int bly = blockIdx.y;
    int bly = task.by;

    __shared__ double sA[16][64 +1];
    __shared__ double sB[64][16 +1];


    FloatingPoint_t rC[( 64 / 16 )][( 64 / 16 )];
    FloatingPoint_t rA[( 64 / 16 )];
    FloatingPoint_t rB[( 64 / 16 )];



        FloatingPoint_t ra[64/16][16/16];




        FloatingPoint_t rb[16/16][64/16];






            int coord_A = offsetA + blx*64*LDA + idyA*LDA+idxA;




            int coord_B = offsetB + bly*64 + idyB*LDB+idxB;
# 165 "gemm_stencil.cu"
    int m, n, k, kk;


#pragma unroll
    for (n = 0; n < ( 64 / 16 ); n++)
#pragma unroll
      for (m = 0; m < ( 64 / 16 ); m++)
        rC[n][m] = (0.0);



#pragma unroll
        for (n = 0; n < 64; n += 16)
#pragma unroll
          for (m = 0; m < 16; m += 16)
            sA[m+idxA][n+idyA] = tex_fetch(tex_ref_A, coord_A + n*LDA+m);
# 191 "gemm_stencil.cu"
#pragma unroll
        for (n = 0; n < 16; n += 16)
#pragma unroll
          for (m = 0; m < 64; m += 16)
            sB[m+idxB][n+idyB] = tex_fetch(tex_ref_B, coord_B + n*LDB+m);
# 204 "gemm_stencil.cu"
    __syncthreads();

    for (kk = 0; kk < K-16; kk += 16)
    {


                coord_A += 16;




                coord_B += 16*LDB;
# 234 "gemm_stencil.cu"
#pragma unroll
            for (n = 0; n < 64/16; n++)
#pragma unroll
              for (m = 0; m < 16/16; m++)
                ra[n][m] = tex_fetch(tex_ref_A, coord_A + n*16*LDA+m*16);
# 249 "gemm_stencil.cu"
#pragma unroll
            for (n = 0; n < 16/16; n++)
#pragma unroll
              for (m = 0; m < 64/16; m++)
                rb[n][m] = tex_fetch(tex_ref_B, coord_B + n*16*LDB+m*16);
# 263 "gemm_stencil.cu"
#pragma unroll
        for (k = 0; k < 16; k++)
        {

#pragma unroll
            for (m = 0; m < ( 64 / 16 ); m++)
                rA[m] = sA[k][m*16 +idx];


#pragma unroll
            for (n = 0; n < ( 64 / 16 ); n++)
                rB[n] = sB[n*16 +idy][k];


#pragma unroll
            for (n = 0; n < ( 64 / 16 ); n++)
#pragma unroll
                for (m = 0; m < ( 64 / 16 ); m++)
# 291 "gemm_stencil.cu"
                        rC[n][m] += (rA[m]*rB[n]);


        }

        __syncthreads();



#pragma unroll
            for (n = 0; n < 64/16; n++)
#pragma unroll
              for (m = 0; m < 16/16; m++)
                sA[m*16 +idxA][n*16 +idyA] = ra[n][m];
# 315 "gemm_stencil.cu"
#pragma unroll
            for (n = 0; n < 16/16; n++)
#pragma unroll
              for (m = 0; m < 64/16; m++)
                sB[m*16 +idxB][n*16 +idyB] = rb[n][m];
# 328 "gemm_stencil.cu"
        __syncthreads();
    }


#pragma unroll
    for (k = 0; k < 16; k++)
    {

#pragma unroll
        for (m = 0; m < ( 64 / 16 ); m++)
            rA[m] = sA[k][m*16 +idx];


#pragma unroll
        for (n = 0; n < ( 64 / 16 ); n++)
            rB[n] = sB[n*16 +idy][k];


#pragma unroll
        for (n = 0; n < ( 64 / 16 ); n++)
#pragma unroll
            for (m = 0; m < ( 64 / 16 ); m++)
# 360 "gemm_stencil.cu"
                    rC[n][m] += (rA[m]*rB[n]);


    }


#pragma unroll
    for (n = 0; n < ( 64 / 16 ); n++) {
        int coord_dCn = bly*64 + n*16 +idy;
#pragma unroll
        for (m = 0; m < ( 64 / 16 ); m++) {
            int coord_dCm = blx*64 + m*16 +idx;
            if (coord_dCm < M && coord_dCn < N) {
                int offsC = coord_dCn*LDC + coord_dCm;

                FloatingPoint_t &regC = rC[n][m];
                FloatingPoint_t &memC = C[offsC];

                memC = ((alpha*regC)+(beta*memC));
            }
        }
    }

}
