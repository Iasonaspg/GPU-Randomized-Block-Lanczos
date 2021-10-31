#include "stdio.h"
#include "stdlib.h"
#include "sys/time.h"
#include "time.h"
#include "FLAME.h"

extern void dgemm_(char*, char*, int*, int*,int*, double*, double*, int*, double*, int*, double*, double*, int*);
extern int openblas_set_num_threads(int);
extern double dnrm2_(int*, double*, int*);

void init_matrices(
	size_t sizeofa, size_t sizeofb, size_t sizeofc,
	double* A, double* B, double* C
){
	for (size_t i=0; i<sizeofa; i++) A[i] = i%3+1;//(rand()%100)/10.0;
	for (size_t i=0; i<sizeofb; i++) B[i] = i%3+1;//(rand()%100)/10.0;
	for (size_t i=0; i<sizeofc; i++) C[i] = i%3+1;//(rand()%100)/10.0;
}

void init_fla_matrix(int m, int n, double* A, FLA_Obj* A_obj){
	FLA_Obj_create_without_buffer( FLA_DOUBLE, m, n, A_obj );
	FLA_Obj_attach_buffer( A, 1, m, A_obj );
}

void init_fla_matrices(
	int m, int n, int k,
	double* A, double* B, double* C,
	double* alpha, double* beta,
	FLA_Obj* A_obj, FLA_Obj* B_obj, FLA_Obj* C_obj, FLA_Obj* alpha_obj, FLA_Obj* beta_obj
){
	init_fla_matrix(m,k,A,A_obj);
	init_fla_matrix(k,n,B,B_obj);
	init_fla_matrix(m,n,C,C_obj);
	init_fla_matrix(1,1,alpha,alpha_obj);
	init_fla_matrix(1,1,beta,beta_obj);
}

void blas(
	char ta, char tb, int m, int n, int k, double alpha, double beta,
	double* A, double* B, double* C
){
	struct timeval start,finish;
	double duration;
	
	int iters = 4;
	gettimeofday(&start, NULL);
	for (int i=0;i<iters;i++) dgemm_(&ta, &tb, &m, &n, &k, &alpha, A, &m, B, &k, &beta, C, &m);
	gettimeofday(&finish, NULL);
	
	duration = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000 / iters;
	double gflops = 2.0 * m *n*k;
	gflops = gflops/duration*1.0e-6;
	
	printf("%dx%dx%d\t%lf s\t%lf MFLOPS\n", m, n, k, duration, gflops);
	int one = 1;
	int sizeofc = m*n;
	printf("|C|: %lf\n",dnrm2_(&sizeofc,C,&one));
}

void flame_legacy(
	int m, int n, int k, double alpha, double beta,
	double* A, double* B, double* C
){
	struct timeval start,finish;
	double duration;
	FLA_Obj A_obj, B_obj, C_obj, alpha_obj, beta_obj;
	// Initialize libflame.
	FLA_Init();

	init_fla_matrices(m,n,k,A,B,C,&alpha,&beta,&A_obj,&B_obj,&C_obj,&alpha_obj,&beta_obj);
	
	// GEMM
	int iters = 4;
	gettimeofday(&start, NULL);
	for (int i=0;i<iters;i++) FLA_Gemm(FLA_NO_TRANSPOSE,FLA_NO_TRANSPOSE,alpha_obj,A_obj,B_obj,beta_obj,C_obj);
	gettimeofday(&finish, NULL);
	duration = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000 / iters;
	double gflops = 2.0 * m *n*k;
	gflops = gflops/duration*1.0e-6;
	printf("%dx%dx%d\t%lf s\t%lf MFLOPS\n", m, n, k, duration, gflops);
	int one = 1;
	int sizeofc = m*n;
	printf("|C|: %lf\n",dnrm2_(&sizeofc,C,&one));
	
	// Free the object without freeing the matrix buffer.
	FLA_Obj_free_without_buffer( &A_obj );
	FLA_Obj_free_without_buffer( &B_obj );
	FLA_Obj_free_without_buffer( &C_obj );
	FLA_Obj_free_without_buffer( &alpha_obj );
	FLA_Obj_free_without_buffer( &beta_obj );

	// Finalize libflame.
	FLA_Finalize();
}

void flame_flash(
	int m, int n, int k, double alpha, double beta,
	double* A, double* B, double* C
){
	struct timeval start,finish;
	double duration;
	FLA_Obj A_obj, B_obj, C_obj, alpha_obj, beta_obj;
	FLA_Obj A_hier, B_hier, C_hier, alpha_hier, beta_hier;
	// Initialize libflame.
	FLA_Init();

	init_fla_matrices(m,n,k,A,B,C,&alpha,&beta,&A_obj,&B_obj,&C_obj,&alpha_obj,&beta_obj);

	size_t depth = 1;
	size_t b_mn[1] = {512};
	FLASH_Obj_create_hier_copy_of_flat(A_obj,depth,b_mn, &A_hier);
	FLASH_Obj_create_hier_copy_of_flat(B_obj,depth,b_mn, &B_hier);
	FLASH_Obj_create_hier_copy_of_flat(C_obj,depth,b_mn, &C_hier);
	// FLASH_Obj_create_hier_copy_of_flat(A_obj,depth,b_mn, &A_hier);
	// FLASH_Obj_create_hier_copy_of_flat(A_obj,depth,b_mn, &A_hier);
	
	// GEMM
	int iters = 4;
	gettimeofday(&start, NULL);
	for (int i=0;i<iters;i++) FLASH_Gemm(FLA_NO_TRANSPOSE,FLA_NO_TRANSPOSE,alpha_obj,A_hier,B_hier,beta_obj,C_hier);
	gettimeofday(&finish, NULL);
	duration = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000 / iters;
	double gflops = 2.0 * m *n*k;
	gflops = gflops/duration*1.0e-6;
	printf("%dx%dx%d\t%lf s\t%lf MFLOPS\n", m, n, k, duration, gflops);

	double* c = (double*)FLASH_Obj_extract_buffer(C_hier);
	int sizeofc = m*n;
	int one = 1;
	printf("|C|: %lf\n",dnrm2_(&sizeofc,c,&one));

	// Free the object without freeing the matrix buffer.
	FLA_Obj_free_without_buffer( &A_obj );
	FLA_Obj_free_without_buffer( &B_obj );
	FLA_Obj_free_without_buffer( &C_obj );
	FLA_Obj_free_without_buffer( &alpha_obj );
	FLA_Obj_free_without_buffer( &beta_obj );

	FLASH_Obj_free(&A_hier);
	FLASH_Obj_free(&B_hier);
	FLASH_Obj_free(&C_hier);

	// Finalize libflame.
	FLA_Finalize();
}

int main(int argc, char* argv[])
{
  int i;
  printf("test!\n");
  if(argc<5){
	printf("Input Error\n");
	return 1;
  }

  int m = atoi(argv[1]); // c nrows
  int n = atoi(argv[2]); // c ncols
  int k = atoi(argv[3]); // a ncols
	openblas_set_num_threads(atoi(argv[4]));

  int sizeofa = m * k;
  int sizeofb = k * n;
  int sizeofc = m * n;
  char ta = 'N';
  char tb = 'N';
  double alpha = 1.2;
  double beta = 0.001;
  
  struct timeval start,finish;
  double duration;
  
  double* A = (double*)malloc(sizeof(double) * sizeofa);
  double* B = (double*)malloc(sizeof(double) * sizeofb);
  double* C = (double*)malloc(sizeof(double) * sizeofc);
  init_matrices(sizeofa,sizeofb,sizeofc,A,B,C);
  printf("m=%d,n=%d,k=%d,alpha=%lf,beta=%lf,sizeofc=%d\n",m,n,k,alpha,beta,sizeofc);

	if (FLASH_Queue_get_enabled()) printf("SuperMatrix is enabled\n");
	FLASH_Queue_set_num_threads(3);

  // blas(ta,tb,m,n,k,alpha,beta,A,B,C);
	// flame_legacy(m,n,k,alpha,beta,A,B,C);
	flame_flash(m,n,k,alpha,beta,A,B,C);

	free(A);
	free(B);
	free(C);
  
  return 0;
}
