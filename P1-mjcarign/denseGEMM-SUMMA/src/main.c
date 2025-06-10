#include "summa_opts.h"
#include "utils.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROOT 0

MPI_Datatype custom_block(int local_rows, int local_cols, int global_cols){
  MPI_Datatype summa_block;
  MPI_Datatype block_type;
  MPI_Type_vector(local_rows, local_cols, global_cols, MPI_FLOAT, &block_type);
  MPI_Type_create_resized(block_type, 0, sizeof(float), &summa_block);
  MPI_Type_commit(&summa_block);

  MPI_Type_free(&block_type);

  return summa_block;
}

void trnsps_distr_mat_block(float* M, float* local_M, int local_rows, int local_cols, int p, 
    int* displ, MPI_Comm g_comm) {
  int np = pow(p, 2);
  int global_cols = local_cols * p;
  int block_size = local_cols * local_rows;
  int* send_count = (int*)malloc(np * sizeof(int));
  for(int i = 0; i < np; i++){
    send_count[i] = 1;
    int coords[2];
    MPI_Cart_coords(g_comm, i, 2, coords);
    displ[i] = coords[1] * local_rows * global_cols + coords[0] * local_cols;
  }

  MPI_Datatype summa_block = custom_block(local_rows, local_cols, global_cols);

  MPI_Scatterv(M, send_count, displ, summa_block, local_M, block_size, MPI_FLOAT, ROOT, g_comm);

  MPI_Type_free(&summa_block);
  free(send_count);
}

void distribute_matrix_blocks(float* M, float* local_M, int local_rows, int local_cols, int p, 
    int* displ, MPI_Comm g_comm) {
  int np = pow(p, 2);
  int global_cols = local_cols * p;
  int block_size = local_cols * local_rows;
  int* send_count = (int*)malloc(np * sizeof(int));
  for(int i = 0; i < np; i++){
    send_count[i] = 1;
    int coords[2];
    MPI_Cart_coords(g_comm, i, 2, coords);
    displ[i] = coords[0] * local_rows * global_cols + coords[1] * local_cols;
  }

  MPI_Datatype summa_block = custom_block(local_rows, local_cols, global_cols);

  MPI_Scatterv(M, send_count, displ, summa_block, local_M, block_size, MPI_FLOAT, ROOT, g_comm);
  
  MPI_Type_free(&summa_block);
  free(send_count);
}

void summa_stationary_a(int m, int n, int k, int np, int rank) {
	int p = sqrt(np);
  int dim[2] = {p, p};
  int period[2] = {0, 0};
  int reorder = 0;
  MPI_Comm grid_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &grid_comm);
  MPI_Comm_rank(grid_comm, &rank);

  int coords[2];
  MPI_Cart_coords(grid_comm, rank, 2, coords);

  int row_color = rank / p;
  int col_color = rank % p;
  int row_key = rank % p;
  int col_key = rank / p;
  MPI_Comm row_comm;
  MPI_Comm col_comm;
  int row_rank;
  int col_rank;
  MPI_Comm_split(grid_comm, row_color, row_key, &row_comm);
  MPI_Comm_split(grid_comm, col_color, col_key, &col_comm);

  int block_m = ceil(m / p); 
  int block_k = ceil(k / p);
  int block_n = ceil(n / p);
  
  float* A = NULL;
  float* B = NULL;
  if(rank == ROOT){
    A = generate_matrix_A(m, k, rank);
    B = generate_matrix_B(k, n, rank);
  }
  
  int my_A_size = block_m * block_k;
  int my_B_size = block_k * block_n;
  float* my_A = (float*)malloc((my_A_size) * sizeof(float));
  float* my_B = (float*)malloc((my_B_size) * sizeof(float));
  
  int* displ = (int*)malloc(np * sizeof(int));
  distribute_matrix_blocks(A, my_A, block_m, block_k, p, displ, grid_comm);
  trnsps_distr_mat_block(B, my_B, block_k, block_n, p, displ, grid_comm);

  float* C;
  if(rank == ROOT){
    C = (float*)calloc(m * n, sizeof(float));
  }
  int my_C_size = block_m * block_n;
  float* my_C = (float*)calloc(my_C_size, sizeof(float));
  float* temp_B = (float*)malloc(my_B_size * sizeof(float));
  float* temp_C = (float*)calloc(my_C_size, sizeof(float));
  for(int iter_k = 0; iter_k < p; iter_k++){
    if(coords[0] == iter_k){
      memcpy(temp_B, my_B, my_B_size * sizeof(float));
    }
    if(coords[1] == iter_k){
      memcpy(temp_C, my_C, my_C_size * sizeof(float));
    }
    MPI_Bcast(temp_B, my_B_size, MPI_FLOAT, iter_k, col_comm);
    MPI_Bcast(temp_C, my_C_size, MPI_FLOAT, iter_k, row_comm);
    matmul(my_A, temp_B, temp_C, block_m, block_n, block_k);
    MPI_Reduce(temp_C, my_C, my_C_size, MPI_FLOAT, MPI_SUM, iter_k, row_comm);
  }
  
  int* counts_recv = (int*)malloc(np * sizeof(int));
  memset(counts_recv, 1, np * sizeof(int));
  MPI_Datatype summa_block = custom_block(block_m, block_n, n);
  for(int i = 0; i < np; i++){
    int coords[2];
    MPI_Cart_coords(grid_comm, i, 2, coords);
    displ[i] = coords[0] * block_m * n + coords[1] * block_n;
  }
  MPI_Gatherv(my_C, my_C_size, MPI_FLOAT, C, counts_recv, displ, summa_block, ROOT, grid_comm);
  
  MPI_Barrier(MPI_COMM_WORLD);

  free(temp_C);
  free(temp_B);
  free(my_A);
  free(my_B);
  free(my_C);
  free(counts_recv);
  free(displ);
  MPI_Type_free(&summa_block);
  if(rank == ROOT){
    free(A);
    free(B);
    free(C);
  }
  
}

void summa_stationary_b(int m, int n, int k, int np, int rank) {
	int p = sqrt(np);
  int dim[2] = {p, p};
  int period[2] = {0, 0};
  int reorder = 0;
  MPI_Comm grid_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &grid_comm);
  MPI_Comm_rank(grid_comm, &rank);

  int coords[2];
  MPI_Cart_coords(grid_comm, rank, 2, coords);

  int row_color = rank / p;
  int col_color = rank % p;
  int row_key = rank % p;
  int col_key = rank / p;
  MPI_Comm row_comm;
  MPI_Comm col_comm;
  int row_rank;
  int col_rank;
  MPI_Comm_split(grid_comm, row_color, row_key, &row_comm);
  MPI_Comm_split(grid_comm, col_color, col_key, &col_comm);

  int block_m = ceil(m / p); 
  int block_k = ceil(k / p);
  int block_n = ceil(n / p);
  
  float* A = NULL;
  float* B = NULL;
  if(rank == ROOT){
    A = generate_matrix_A(m, k, rank);
    B = generate_matrix_B(k, n, rank);
  }
  
  int my_A_size = block_m * block_k;
  int my_B_size = block_k * block_n;
  float* my_A = (float*)malloc((my_A_size) * sizeof(float));
  float* my_B = (float*)malloc((my_B_size) * sizeof(float));
  
  int* displ = (int*)malloc(np * sizeof(int));
  trnsps_distr_mat_block(A, my_A, block_m, block_k, p, displ, grid_comm);
  distribute_matrix_blocks(B, my_B, block_k, block_n, p, displ, grid_comm);

  float* C;
  if(rank == ROOT){
    C = (float*)calloc(m * n, sizeof(float));
  }
  int my_C_size = block_m * block_n;
  float* my_C = (float*)calloc(my_C_size, sizeof(float));
  float* temp_A = (float*)malloc(my_A_size * sizeof(float));
  float* temp_C = (float*)calloc(my_C_size, sizeof(float));
  for(int iter_k = 0; iter_k < p; iter_k++){
    if(coords[0] == iter_k){
      memcpy(temp_C, my_C, my_C_size * sizeof(float));
    }
    if(coords[1] == iter_k){
      memcpy(temp_A, my_A, my_A_size * sizeof(float));
    }
    MPI_Bcast(temp_A, my_A_size, MPI_FLOAT, iter_k, row_comm);
    MPI_Bcast(temp_C, my_C_size, MPI_FLOAT, iter_k, col_comm);
    matmul(temp_A, my_B, temp_C, block_m, block_n, block_k);
    MPI_Reduce(temp_C, my_C, my_C_size, MPI_FLOAT, MPI_SUM, iter_k, col_comm);
  }
  
  int* counts_recv = (int*)malloc(np * sizeof(int));
  memset(counts_recv, 1, np * sizeof(int));
  MPI_Datatype summa_block = custom_block(block_m, block_n, n);
  for(int i = 0; i < np; i++){
    int coords[2];
    MPI_Cart_coords(grid_comm, i, 2, coords);
    displ[i] = coords[0] * block_m * n + coords[1] * block_n;
  }
  MPI_Gatherv(my_C, my_C_size, MPI_FLOAT, C, counts_recv, displ, summa_block, ROOT, grid_comm);
  
  
  MPI_Barrier(MPI_COMM_WORLD);

  free(temp_C);
  free(temp_A);
  free(my_A);
  free(my_B);
  free(my_C);
  free(counts_recv);
  free(displ);
  MPI_Type_free(&summa_block);
  if(rank == ROOT){
    free(A);
    free(B);
    free(C);
  }
}

void summa_stationary_c(int m, int n, int k, int np, int rank) {
  int p = sqrt(np);
  int dim[2] = {p, p};
  int period[2] = {0, 0};
  int reorder = 0;
  MPI_Comm grid_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &grid_comm);
  MPI_Comm_rank(grid_comm, &rank);

  int coords[2];
  MPI_Cart_coords(grid_comm, rank, 2, coords);

  int row_color = rank / p;
  int col_color = rank % p;
  int row_key = rank % p;
  int col_key = rank / p;
  MPI_Comm row_comm;
  MPI_Comm col_comm;
  int row_rank;
  int col_rank;
  MPI_Comm_split(grid_comm, row_color, row_key, &row_comm);
  // MPI_Comm_rank(row_comm, &row_rank);
  MPI_Comm_split(grid_comm, col_color, col_key, &col_comm);
  // MPI_Comm_rank(col_comm, &col_rank);

  int block_m = ceil(m / p); 
  int block_k = ceil(k / p);
  int block_n = ceil(n / p);
  
  float* A;
  float* B;
  if(rank == ROOT){
    A = generate_matrix_A(m, k, rank);
    B = generate_matrix_B(k, n, rank);
  }
  
  int my_A_size = block_m * block_k;
  int my_B_size = block_k * block_n;
  float* my_A = (float*)malloc((my_A_size) * sizeof(float));
  float* my_B = (float*)malloc((my_B_size) * sizeof(float));
  
  int* displ = (int*)malloc(np * sizeof(int));
  distribute_matrix_blocks(A, my_A, block_m, block_k, p, displ, grid_comm);
  distribute_matrix_blocks(B, my_B, block_k, block_n, p, displ, grid_comm);

  float* C;
  if(rank == ROOT){
    C = (float*)calloc(m * n, sizeof(float));
  }
  int my_C_size = block_m * block_n;
  float* my_C = (float*)calloc(my_C_size, sizeof(float));
  float* temp_A = (float*)malloc(my_A_size * sizeof(float));
  float* temp_B = (float*)malloc(my_B_size * sizeof(float));
  for(int iter_k = 0; iter_k < p; iter_k++){
    if(coords[1] == iter_k){
      memcpy(temp_A, my_A, my_A_size * sizeof(float));
    }
    if(coords[0] == iter_k){
      memcpy(temp_B, my_B, my_B_size * sizeof(float));
    }
    MPI_Bcast(temp_A, my_A_size, MPI_FLOAT, iter_k, row_comm);
    MPI_Bcast(temp_B, my_B_size, MPI_FLOAT, iter_k, col_comm);
    matmul(temp_A, temp_B, my_C, block_m, block_n, block_k);
  }
  
  int* recv_c = (int*)malloc(np * sizeof(int));
  memset(recv_c, 1, np * sizeof(int));
  MPI_Datatype summa_block = custom_block(block_m, block_n, n);
  for(int i = 0; i < np; i++){
    int coords[2];
    MPI_Cart_coords(grid_comm, i, 2, coords);
    displ[i] = coords[0] * block_m * n + coords[1] * block_n;
  }
  MPI_Gatherv(my_C, my_C_size, MPI_FLOAT, C, recv_c, displ, summa_block, ROOT, grid_comm);

  MPI_Barrier(MPI_COMM_WORLD);

  free(temp_A);
  free(temp_B);
  free(my_A);
  free(my_B);
  free(my_C);
  free(recv_c);
  free(displ);
  MPI_Type_free(&summa_block);
  if(rank == ROOT){
    free(A);
    free(B);
    free(C);
  }
  
}

int main(int argc, char *argv[]) {
  int np, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  SummaOpts opts;
  opts = parse_args(argc, argv);
  
  int grid_len = sqrt(np);
  if(np != pow(grid_len, 2)){
    printf("Error: Number of processes must be capable of forming a perfect square grid.\n");
  }
  // Check if matrix dimensions are compatible with grid size
  if (opts.m % grid_len != 0 || opts.n % grid_len != 0 ||
      opts.k % grid_len != 0) {
    printf("Error: Matrix dimensions must be divisible by grid size (%d)\n",
           grid_len);
    return 1;
  }

  if(rank == ROOT){
    printf("\nMatrix Dimensions:\n");
    printf("A: %d x %d\n", opts.m, opts.k);
    printf("B: %d x %d\n", opts.k, opts.n);
    printf("C: %d x %d\n", opts.m, opts.n);
    printf("Grid size: %d x %d\n", grid_len, grid_len);
    printf("Block size: %d\n", opts.block_size);
    printf("Algorithm: Stationary %c\n", opts.stationary);
    printf("Verbose: %s\n", opts.verbose ? "true" : "false");
  }
  
  double start, end;
  start = MPI_Wtime();
  // Call the appropriate SUMMA function based on algorithm variant
  if (opts.stationary == 'a') {
    summa_stationary_a(opts.m, opts.n, opts.k, np, rank);
  } else if (opts.stationary == 'c') {
    summa_stationary_c(opts.m, opts.n, opts.k, np, rank);
  } else if (opts.stationary == 'b') {
    summa_stationary_b(opts.m, opts.n, opts.k, np, rank);
  } else {
    printf("Error: Unknown stationary option '%c'. Use 'A' or 'C'.\n",
           opts.stationary);
    return 1;
  }
  end = MPI_Wtime();
  double elapsed = end - start;
  if(rank == ROOT){
    printf("Time elapsed: %f seconds\n", elapsed);
  }

  MPI_Finalize();
  
  return 0;
}