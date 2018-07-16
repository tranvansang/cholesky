#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include <string.h>
#include <omp.h>

//param
/*#define RUN_SERIAL*/
#define RUN_PARALLEL
#define N 100
#define WEAK_SCALING
// if weak_scaling is defined, n = N * np * omp_get_num_threads()

//debug
/*#define DEBUG*/
/*#define VERBOSE*/
//printing precision
#define PRECISION 2

//other constants
#define ROOT_RANK 0
#define EPS 1e-10

#ifdef DEBUG
#define MPI(x) mpi_result = x; \
                            if (mpi_result != MPI_SUCCESS) { \
                              print_error(mpi_result); \
                            } \
assert(mpi_result == MPI_SUCCESS);
#else // DEBUG
#define MPI(x) mpi_result = x; assert(mpi_result == MPI_SUCCESS);
#endif // DEBUG

//matrix element - row major
#define ELM(a, r, c, ld) ((a) + (r) * (ld) + c)

void print_error(int mpi_result){
  int eclass, estr_len;
  char estring[MPI_MAX_ERROR_STRING];
  MPI_Error_class(mpi_result, &eclass);
  MPI_Error_string(mpi_result, estring, &estr_len);
  printf("Error %d, class %d: %s\n", mpi_result, eclass, estring);
  fflush(stdout);
}

/* matrix-vector multiply : y = A * x, where 
   A is symmetric and only lower half are stored */
void symMatVec(int n, double *a, double *x, double *y) {
  int i, j;

  for (i=0; i< n; i++) {
    double t = 0.0;
    for (j=0; j<= i; j++)
      t += *ELM(a, i, j, n) * x[j];

    for (j= i+1; j< n; j++)
      t += *ELM(a, j, i, n) * x[j];

    y[i] = t;
  }
}

void print_lower(double *a, int n, int ld){
  int i, j;
  for(i = 0; i < n; i++){
    for(j = 0; j <= i; j++)
      printf("%.*lf\t", PRECISION, *ELM(a, i, j, ld));
    printf("\n");
  }
}
void print_upper(double *a, int n, int ld){
  int i, j;
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++)
      if (j < i) printf("__________\t");
      else printf("%.*lf\t", PRECISION, *ELM(a, i, j, ld));
    printf("\n");
  }
}

void print_full(double *a, int nrow, int ncol, int ld){
  int i, j;
  for(i = 0; i < nrow; i++){
    for(j = 0; j <ncol; j++)
      printf("%.*lf\t", PRECISION, *ELM(a, i, j, ld));
    printf("\n");
  }
}

/* solve Ax = b */
void solveSym_serial(int n, double *a, double *x, double *b) {
  int i, j, k;

  /* LDLT decomposition: A = L * D * L^t */
  for (i=0; i< n; i++) {
    double invp = 1.0 / *ELM(a, i, i, n);

    for (j= i+1; j< n; j++) {
      double aji = *ELM(a, j, i, n);
      *ELM(a,j, i, n) *= invp;

      for (k= i+1; k<= j; k++)
        *ELM(a, j, k, n) -= aji * *ELM(a, k, i, n);
    }
#ifdef VERBOSE
    printf("matrix after iteration %d\n", i);
    print_lower(a, n, n);
    printf("\n");
#endif // VERBOSE
  }
#ifdef VERBOSE
    printf("array after serial ldlt: \n");
    print_lower(a, n, n);
    printf("\n");
#endif // VERBOSE

  /* forward solve L y = b: but y is stored in x
     can be merged to the previous loop */
  for (i=0; i< n; i++) {
    double t = b[i];

    for (j=0; j< i; j++)
      t -= *ELM(a, i, j, n) * x[j];

    x[i] = t;
  }

  /* backward solve D L^t x = y */
  for (i= n-1; i>= 0; i--) {
    double t = x[i] / *ELM(a, i, i, n);

    for (j= i+1; j< n; j++)
      t -= *ELM(a, j, i, n) * x[j];

    x[i] = t;
  }
}

//calculate number of rows for process with rank
inline int get_nrows(int n, int np, int rank){
  return (n + np - rank - 1) / np;
}

//get original row index (size) from local row
inline int get_row(int np, int rank, int local_row){
  return local_row * np + rank;
}

/* solve Ax = b */
void solveSym(int rank, int np, int n, double *a, double *x, double *b) {
  int i, j, k, tag, receiver, mpi_result, row, skipped_rows_count, sender;
  int nrows_local = get_nrows(n, np, rank);
  double *local_a = malloc(sizeof(double) * n * nrows_local);
  double *local_a_t = malloc(sizeof(double) * n * nrows_local);
  double *first_column = malloc(sizeof(double) * n);
  double *allgather_buf = malloc(sizeof(double) * n);
  assert(local_a != NULL);
  assert(local_a_t != NULL);
  assert(first_column != NULL);
  assert(allgather_buf != NULL);
  double tmp, aji;
  int nrequests;
  if (rank == 0) nrequests = n - nrows_local;
  else nrequests = nrows_local;
  MPI_Request *requests = malloc(sizeof(MPI_Request) * nrequests);
  MPI_Status *statuses = malloc(sizeof(MPI_Status) * nrequests);
  int *displs = malloc(sizeof(int) * np);
  int *recvcounts = malloc(sizeof(int) * np);
  assert(requests != NULL);
  assert(statuses != NULL);
  assert(displs != NULL);
  assert(recvcounts != NULL);

  //root process
  if (rank == ROOT_RANK){
    j = 0;
    //deliver row data to each other processes from root process
#pragma omp parallel for private(tag, receiver, k)
    for(i = 0; i < n; i++){
      tag = i / np;
      receiver = i % np;
      if (receiver != 0){
#pragma omp critical
        k = j++;
        MPI(MPI_Isend(ELM(a, i, 0, n), i + 1, MPI_DOUBLE, receiver, tag, MPI_COMM_WORLD, requests + k));
      }
    }
    //copy to my own
    //(n + np - 1) / np = nrows_local
#pragma omp parallel for private(row)
    for(i = 0; i < nrows_local; i++){
      row = get_row(np, rank, i);
      memcpy(ELM(local_a, i, 0, n), ELM(a, row, 0, n), sizeof(double) * (row + 1));
    }
  }else {
    //child process
#pragma omp parallel for private(row, tag)
    for(i = 0; i < nrows_local; i++){
      row = get_row(np, rank, i);
      tag = i;
      MPI(MPI_Irecv(ELM(local_a, i, 0, n), row + 1, MPI_DOUBLE, ROOT_RANK, tag, MPI_COMM_WORLD, requests + i));
    }
    MPI(MPI_Waitall(nrequests, requests, statuses));
    /*for(i = 0; i < nrows_local; i++){*/
      /*printf("rank %d status %d: %d\n", rank, i, statuses[i].MPI_ERROR);*/
      /*if (statuses[i].MPI_ERROR != MPI_SUCCESS)*/
        /*print_error(statuses[i].MPI_ERROR);*/
    /*}*/
  }

  //transpose to column major
#pragma omp parallel for private(row)
  for(i = 0; i < nrows_local; i++){
    //also take diagonal elms
    row = get_row(np, rank, i);
#pragma omp parallel for
    for(j = 0; j <= row; j++){
      *ELM(local_a_t, j, i, nrows_local) = *ELM(local_a, i, j, n);
    }
  }

  //wait all requests in root process
  if (rank == ROOT_RANK){
    MPI(MPI_Waitall(nrequests, requests, statuses));
  }
  /* LDLT decomposition: A = L * D * L^t */
  for(i = 0; i < n; i++){
    for(j = 0; j < np; j++){
      recvcounts[j] = get_nrows(n, np, j) - get_nrows(i, np, j);
      displs[j] = j == 0 ? 0 : displs[j - 1] + recvcounts[j  -1];
    }
    //broadcast first column (i.e. first row of column-major) of current iteration
    skipped_rows_count = get_nrows(i, np, rank);
    MPI(MPI_Allgatherv(ELM(local_a_t, i, skipped_rows_count, nrows_local), recvcounts[rank], MPI_DOUBLE, allgather_buf, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD));
#ifdef VERBOSE
    if (i == 1)
    for(j = 0; j < np; j++){
      MPI(MPI_Barrier(MPI_COMM_WORLD));
      if (rank == j){
        printf("allgather (rank = %d): ", j);
        for(k = 0; k < n; k++)
          printf("%.*lf\t", PRECISION, allgather_buf[k]);
        printf("\n");
      }
    }
#endif

    //put elms of collected buffer into correct order
#pragma omp parallel for private(row)
    for(j = 0; j < np; j++){
      row = recvcounts[j];
#pragma omp parallel for
      for(k = 0; k < row; k++)
        first_column[(k + get_nrows(i, np, j)) * np + j] = allgather_buf[displs[j] + k];
    }
#ifdef VERBOSE
    if (i == 1)
    for(j = 0; j < np; j++){
      MPI(MPI_Barrier(MPI_COMM_WORLD));
      if (rank == j){
        printf("first column (rank = %d): ", j);
        for(k = i; k < n; k++)
          printf("%.*lf\t", PRECISION, first_column[k]);
        printf("\n");
      }
    }
#endif

    //pre devide first_column to speedup (reduce deviding operation)
    row = get_row(np, rank, nrows_local -1);
#pragma omp parallel for
    for(j = i + 1; j <= row; j++)
      first_column[j] /= first_column[i];

    //do LDLT calculation
    //for all rows
    skipped_rows_count = get_nrows(i + 1, np, rank);
    //get_nrows of i + 1 because we are going to skip the row i (do j-loop from i+ 1 to n)
#pragma omp parallel for private(aji, row)
    for(j = skipped_rows_count; j < nrows_local; j++){
      //backup aji
      aji = *ELM(local_a_t, i, j, nrows_local);
      row = get_row(np, rank, j);
      //first elm
      *ELM(local_a_t, i, j, nrows_local) = first_column[row];
      //other elms
#pragma omp parallel for
      for(k = i + 1; k <= row; k++)
        *ELM(local_a_t, k, j, nrows_local) -= aji * first_column[k];
    }
    MPI(MPI_Barrier(MPI_COMM_WORLD));
#ifdef VERBOSE
    for(j = 0; j < np; j++){
      if (rank == j){
        printf("matrix(transposed) after iteration %d rank %d\n", i, j);
        print_full(local_a_t, get_row(np, rank, nrows_local - 1) + 1, nrows_local, nrows_local);
        printf("\n");
        fflush(stdout);
      }
      MPI(MPI_Barrier(MPI_COMM_WORLD));
    }
    if (rank == ROOT_RANK){
      printf("first column: ");
      for(j = i; j < n; j++)
        printf("%.*lf\t", PRECISION, first_column[j]);
      printf("\n");
    }
#endif // VERBOSE
  }

  //transpose back to row-major
#pragma omp parallel for private(row)
  for(i = 0; i < nrows_local; i++){
    row = get_row(np, rank, i);
    //also take diagonal elms
#pragma omp parallel for
    for(j = 0; j <= row; j++){
      *ELM(local_a, i, j, n) = *ELM(local_a_t, j, i, nrows_local);
    }
    if (rank != ROOT_RANK){
      //trasfer back to root process
      MPI( MPI_Isend(
          ELM(local_a, i, 0, n),
          row + 1,
          MPI_DOUBLE,
          ROOT_RANK,
          i,
          MPI_COMM_WORLD,
          requests + i
          ));
    }
  }
  if (rank == ROOT_RANK){
    //receive calculated buffer from all processes
    j = 0;
#pragma omp parallel for private(tag, sender, k)
    for(i = 0; i < n; i++){
      tag = i / np;
      sender = i % np;
      if (sender != 0){
#pragma omp critical
        k = j++;
        MPI(MPI_Irecv(ELM(a, i, 0, n), i + 1, MPI_DOUBLE, sender, tag, MPI_COMM_WORLD, requests + k));
      }
    }
    //copy from my own
    //(n + np - 1) / np = nrows_local
#pragma omp parallel for private(row)
    for(i = 0; i < nrows_local; i++){
      row = get_row(np, rank, i);
      memcpy(ELM(a, row, 0, n), ELM(local_a, i, 0, n), sizeof(double) * (row + 1));
    }
    /*MPI(MPI_Waitall(nrequests, requests, statuses));*/
    MPI(MPI_Waitall(nrequests, requests, statuses));
    /*for(i = 0; i < nrows_local; i++){*/
      /*if (statuses[i].MPI_ERROR != MPI_SUCCESS){*/
        /*printf("rank %d status %d: %d\n", rank, i, statuses[i].MPI_ERROR);*/
        /*print_error(statuses[i].MPI_ERROR);*/
      /*}*/
    /*}*/

#ifdef VERBOSE
    if (rank == ROOT_RANK){
      printf("array after parallel computation: \n");
      print_lower(a, n, n);
      printf("\n");
    }
#endif // VERBOSE
    /* forward solve L y = b: but y is stored in x
       can be merged to the previous loop */
    for (i=0; i< n; i++) {
      double t = b[i];

#pragma omp parallel for reduction(-:t)
      for (j=0; j< i; j++)
        t -= *ELM(a, i, j, n) * x[j];

      x[i] = t;
    }

    /* backward solve D L^t x = y */
    for (i= n-1; i>= 0; i--) {
      double t = x[i] / *ELM(a, i, i, n);

#pragma omp parallel for reduction(-:t)
      for (j= i+1; j< n; j++)
        t -= *ELM(a, j, i, n) * x[j];

      x[i] = t;
    }
  } else {
    MPI(MPI_Waitall(nrequests, requests, statuses));
  }

  free(statuses);
  free(local_a);
  free(requests);
  free(first_column);
  free(allgather_buf);
  free(displs);
  free(recvcounts);
  free(local_a_t);
}

double norm(double *x, double* y, int n){
  /* check error norm */
  double e = 0;
  int i;
  for (i=0; i< n; i++)
    e += (x[i] - y[i]) * (x[i] - y[i]);
  return sqrt(e);
}

//error handler
void cholesky_mpi_error_handler( MPI_Comm *comm, int *err, ... )
{
    if (*err != MPI_ERR_OTHER) {
        printf( "Unexpected error code\n" );fflush(stdout);
    } else {
      printf("error caused in comm %d, error: %d", *comm, err ? *err : -1); fflush(stdout);
    }
}

typedef struct {
  double serial;
  double parallel;
} bmtime_t;

bmtime_t benchmark(int n, int np, int rank){
  bmtime_t bmtime;
  int i, j, mpi_result;
  double *a, *xx, *b, *x, e, s, *a_copy, *b_copy;
  double time_start, time_stop;

  if (rank == ROOT_RANK){
    /* matrix */
    a = malloc(sizeof(double) * n * n);
    assert(a != NULL);

    /* fill lower triangular elements */
    for (i=0; i< n; i++)
      for (j=0; j< i; j++)
        *ELM(a, i, j, n) = rand()/(RAND_MAX + 1.0);

    /* fill diagonal elements */
    for (i=0; i< n; i++) {
      s = 0.0;
      for (j=0; j< i; j++)
        s += *ELM(a, i, j, n);

      for (j= i+1; j< n; j++)
        s += *ELM(a, j, i, n);		/* upper triangular */

      *ELM(a, i, i, n) = s + 1.0;		/* diagonal dominant */
    }
#ifdef VERBOSE
    printf("original matrix: \n");
    print_lower(a, n, n);
    printf("\n");
#endif // VERBOSE

    /* first make the solution */
    xx = malloc(sizeof(double) * n);
    assert(xx != NULL);

    for (i=0; i< n; i++)
      xx[i] = 1.0;			/* or anything you like */

    /* make right hand side b = Ax */
    b = malloc(sizeof(double) * n);
    assert(b != NULL);

    symMatVec(n, a, xx, b);

    /* solution vector, pretend to be unknown */
    x = malloc(sizeof(double) * n);
    assert(x != NULL);

#ifdef RUN_SERIAL
    //clone data
    a_copy = malloc(sizeof(double) * n * n);
    assert(a_copy != NULL);
    b_copy = malloc(sizeof(double) * n);
    assert(b_copy != NULL);
    memcpy(a_copy, a, sizeof(double) * n * n);
    memcpy(b_copy, b, sizeof(double) * n);

    //serial solver
    time_start = MPI_Wtime();
    solveSym_serial(n, a_copy, x, b_copy);
    time_stop = MPI_Wtime();
    bmtime.serial = time_stop - time_start;

    e = norm(x, xx, n);
    if (e >= EPS){
      bmtime.serial = -1;
      fprintf(stderr, "expected error norm less than %e, but %e received while serial benchmark with size = %d\n", EPS, e, n);
    }

    //free data
    free(a_copy);
    free(b_copy);
    memset(x, 0, sizeof(double) * n);
#else // RUN_SERIAL
    bmtime.serial = 0;
#endif // RUN_SERIAL
  }

#ifdef RUN_PARALLEL
  MPI(MPI_Barrier(MPI_COMM_WORLD));
  //parallel version
  time_start = MPI_Wtime();
  /* solve: the main computation */
  solveSym(rank, np, n, a, x, b);
  MPI(MPI_Barrier(MPI_COMM_WORLD));
  time_stop = MPI_Wtime();

  bmtime.parallel = time_stop - time_start;

  if (rank == ROOT_RANK) {
    e = norm(x, xx, n);
    if (e >= EPS){
      bmtime.parallel = -1;
        fprintf(stderr, "expected error norm less than %e, but %e received while parallel benchmark with size = %d\n", EPS, e, n);
    }
  }
#else // RUN_PARALLEL
  bmtime.parallel = -1;
#endif // RUN_PARALLEL
  if (rank == ROOT_RANK){
    //free data
    free(a);
    free(xx);
    free(b);
    free(x);
  }
  return bmtime;
}

int main(int argc, char **argv) {
  int mpi_result, n;
  double time_start, time_stop;

  // Initialize the MPI environment
  int provided;
  MPI(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided ));
  if (MPI_THREAD_MULTIPLE != provided){
    fprintf(stderr, "Expected mpi thread support %d but %d returned\n", MPI_THREAD_MULTIPLE, provided);
    return 1;
  }
  MPI(MPI_Barrier(MPI_COMM_WORLD));
  time_start = MPI_Wtime();

  //set error handler
  MPI_Errhandler err_handler;
  /*MPI(MPI_Comm_create_errhandler(&cholesky_mpi_error_handler, &err_handler));*/
  /*MPI(MPI_Comm_set_errhandler(MPI_COMM_WORLD, err_handler));*/
#ifdef DEBUG
  MPI(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
#endif // DEBUG

  // Get the number of processes
  int np;
  MPI(MPI_Comm_size(MPI_COMM_WORLD, &np));

  // Get the rank of the process
  int rank;
  MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  int nt;
#pragma omp parallel
#pragma omp master
  nt = omp_get_num_threads();
#ifdef WEAK_SCALING
  n = N * np * nt;
#else // WEAK_SCALING
  n = N;
#endif // WEAK_SCALING
  bmtime_t bmtime = benchmark(n, np, rank);
  if (rank == ROOT_RANK){
    printf("%d\t%d\t%d\t%.10lf\t%.10lf\n", np, nt, n, bmtime.serial, bmtime.parallel);
  }

  MPI(MPI_Barrier(MPI_COMM_WORLD));
  time_stop = MPI_Wtime();
  /*if (rank == ROOT_RANK){*/
    /*printf("Total job time / limit (10 min): %.2lf%%\n", (time_stop - time_start) / 60 / 10 * 100);*/
  /*}*/

  //free error handler
  /*MPI_Errhandler_free( &err_handler );*/
  // Finalize the MPI environment.
  MPI_Finalize();
  return 0;
}

