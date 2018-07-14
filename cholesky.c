#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include <string.h>

#define ROOT_RANK 0
#define RUN_SERIAL 1
#define N 30

//matrix element - row major
#define ELM(a, r, c, ld) (a + (r) * (ld) + c)

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
  }

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
int get_nrows(int n, int np, int rank){
  return (n + np - rank - 1) / np;
}

int get_row(int np, int rank, int local_row){
  return local_row * np + rank;
}

/* solve Ax = b */
void solveSym(int rank, int np, int n, double *a, double *x, double *b) {
  int i, j, k, tag, receiver, mpi_result, row, *displs, *recvcounts, skipped_rows_count, sender;
  int nrows_local = get_nrows(n, np, rank);
  double *local_a = malloc(sizeof(double) * n * nrows_local);
  double *local_a_t = malloc(sizeof(double) * n * nrows_local);
  double *first_column = malloc(sizeof(double) * n);
  double *allgather_buf = malloc(sizeof(double) * n);
  assert(local_a != NULL);
  assert(local_a_t != NULL);
  assert(first_column != NULL);
  assert(allgather_buf != NULL);
  double tmp;
  MPI_Request *requests;
  MPI_Status *statuses;
  int nrequests;
  if (rank == 0) nrequests = n - nrows_local;
  else nrequests = nrows_local;
  requests = malloc(sizeof(MPI_Request) * nrequests);
  statuses = malloc(sizeof(MPI_Status) * nrequests);
  displs = malloc(sizeof(int) * np);
  recvcounts = malloc(sizeof(int) * np);

  //root process
  if (rank == ROOT_RANK){
    j = 0;
    //deliver row data to each other processes from root process
    for(i = 0; i < n; i++){
      tag = i / np;
      receiver = i % np;
      if (receiver != 0){
        mpi_result = MPI_Isend(ELM(a, i, 0, n), i + 1, MPI_DOUBLE, receiver, tag, MPI_COMM_WORLD, requests + j);
        assert(mpi_result == MPI_SUCCESS);
        j++;
      }
    }
    //copy to my own
    //(n + np - 1) / np = nrows_local
    for(i = 0; i < nrows_local; i++){
      row = get_row(np, rank, i);
      memcpy(ELM(local_a, i, 0, n), ELM(a, row, 0, n), sizeof(double) * row);
    }
  }else {
    //child process
    for(i = 0; i < nrows_local; i++){
      row = get_row(np, rank, i);
      tag = i;
      mpi_result = MPI_Irecv(ELM(local_a, i, 0, n), row, MPI_DOUBLE, ROOT_RANK, tag, MPI_COMM_WORLD, requests + i);
      assert(mpi_result == MPI_SUCCESS);
    }
    printf("process %d\n", rank);
    mpi_result = MPI_Waitall(nrequests, requests, statuses);
    printf("proces %d waitall\n", rank);
    assert(mpi_result == MPI_SUCCESS);
  }

  //transpose to column major
  for(i = 0; i < nrows_local; i++){
    get_row(np, rank, i);
    for(j = 0; j < row; j++){
      *ELM(local_a_t, j, i, nrows_local) = *ELM(local_a, i, j, n);
    }
  }

  //wait all requests in root process
  if (rank == ROOT_RANK){
    mpi_result = MPI_Waitall(nrequests, requests, statuses);
    assert(mpi_result == MPI_SUCCESS);
  }
  /* LDLT decomposition: A = L * D * L^t */
  for(i = 0; i < n; i++){
    for(j = 0; j < np; j++){
      recvcounts[j] = get_nrows(n, np, j) - get_nrows(i, np, j);
      displs[j] = j == 0 ? 0 : displs[j - 1] + recvcounts[j  -1];
    }
    //broadcast first column (i.e. first row of column-major) of current iteration
    skipped_rows_count = get_nrows(i, np, rank);
    mpi_result = MPI_Allgatherv(ELM(local_a_t, i, skipped_rows_count, nrows_local), recvcounts[rank], MPI_DOUBLE, allgather_buf, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    assert(mpi_result == MPI_SUCCESS);

    //put elms of collected buffer into correct order
    for(j = 0; j < np; j++){
      for(k = 0; k < recvcounts[j]; k++)
        first_column[k * np + j] = allgather_buf[displs[j] + k];
    }

    //do LDLT calculation
    //for all rows
    skipped_rows_count = get_nrows(i, np, rank);
    for(j = skipped_rows_count; j < nrows_local; j++){
      //first elm
      *ELM(local_a_t, i, j, nrows_local) /= first_column[0];
      row = get_row(np, rank, j);
      //other elms
      for(k = i + 1; k < row; k++)
        *ELM(local_a_t, k, j, nrows_local) -= first_column[0] * first_column[k - i];
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  //transpose back to row-major
  for(i = 0; i < nrows_local; i++){
    row = get_row(np, rank, i);
    for(j = 0; j < row; j++){
      *ELM(local_a_t, j, i, nrows_local) = *ELM(local_a, i, j, n);
    }
    if (rank != ROOT_RANK){
      //trasfer back to root process
      mpi_result = MPI_Isend(
          ELM(local_a, i, 0, n),
          sizeof(double) * row,
          MPI_DOUBLE,
          ROOT_RANK,
          i,
          MPI_COMM_WORLD,
          requests + i
          );
      assert(mpi_result == MPI_SUCCESS);
    }
  }
  if (rank == ROOT_RANK){
    //receive calculated buffer from all processes
    j = 0;
    for(i = 0; i < n; i++){
      tag = i / np;
      sender = i % np;
      if (sender != 0){
        mpi_result = MPI_Irecv(ELM(a, i, 0, n), i + 1, MPI_DOUBLE, sender, tag, MPI_COMM_WORLD, requests + j);
        j++;
        assert(mpi_result == MPI_SUCCESS);
      }
    }
    //copy from my own
    //(n + np - 1) / np = nrows_local
    for(i = 0; i < nrows_local; i++){
      row = get_row(np, rank, i);
      memcpy(ELM(a, row, 0, n), ELM(local_a, i, 0, n), sizeof(double) * row);
    }
    mpi_result = MPI_Waitall(nrequests, requests, statuses);
    assert(mpi_result == MPI_SUCCESS);

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
  } else {
    mpi_result = MPI_Waitall(nrequests, requests, statuses);
    assert(mpi_result == MPI_SUCCESS);
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

int main(int argc, char **argv) {
  // Initialize the MPI environment
  MPI_Init(&argc, &argv);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int n, i, j;
  double *a, *xx, *b, *x, e, s, *a_copy, *b_copy;
  double time_start, time_stop;

  n = N;

  if (world_rank == ROOT_RANK){

    // Print off a hello world message
    printf("Number of processes: %d\n", world_size);

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
  }

  if (RUN_SERIAL && world_rank == ROOT_RANK){
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
    printf("Serial time: %.8f sec\n", time_stop - time_start);

    e = norm(x, xx, n);
    printf("error norm = %e\n", e);
    printf("--- good if error is around n * 1e-16 or less\n");

    //free data
    free(a_copy);
    free(b_copy);
    memset(x, 0, sizeof(double) * n);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  //parallel version
  time_start = MPI_Wtime();
  /* solve: the main computation */
  solveSym(world_rank, world_size, n, a, x, b);
  MPI_Barrier(MPI_COMM_WORLD);
  time_stop = MPI_Wtime();

  if (world_rank == ROOT_RANK){
    printf("Paralle time: %.8f sec\n", time_stop - time_start);

    e = norm(x, xx, n);
    printf("error norm = %e\n", e);
    printf("--- good if error is around n * 1e-16 or less\n");

    //free data
    free(a);
    free(xx);
    free(b);
    free(x);
  }
  // Finalize the MPI environment.
  MPI_Finalize();
  return 0;
}
