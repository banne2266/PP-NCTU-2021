#include <cstdio>
#include <mpi.h>

#define A_TAG 0
#define B_TAG 1
#define C_TAG 2


int world_rank, world_size;

// Read size of matrix_a and matrix_b (n, m, l) and whole data of matrixes from stdin
//
// n_ptr:     pointer to n
// m_ptr:     pointer to m
// l_ptr:     pointer to l
// a_mat_ptr: pointer to matrix a (a should be a continuous memory space for placing n * m elements of int)
// b_mat_ptr: pointer to matrix b (b should be a continuous memory space for placing m * l elements of int)
void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr)
{
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if(world_rank == 0){
        scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
        int n = *n_ptr, m = *m_ptr, l = *l_ptr;
        *a_mat_ptr = new int [n*m];
        *b_mat_ptr = new int [m*l];

        int *a = *a_mat_ptr, *b = *b_mat_ptr;
        for(int i = 0; i < n; i++){
            for(int j = 0; j < m; j++){
                scanf("%d", &a[i*m+j]);
            }
        }
        
        for(int i = 0; i < m; i++){
            for(int j = 0; j < l; j++){
                scanf("%d", &b[j*m+i]);//Do tanspose on b;
            }
        }//c[i][j] = a[i] dot b[j]
    }
    MPI_Bcast(n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(m_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(l_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

// Just matrix multiplication (your should output the result in this function)
// 
// n:     row number of matrix a
// m:     col number of matrix a / row number of matrix b
// l:     col number of matrix b
// a_mat: a continuous memory placing n * m elements of int
// b_mat: a continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat)
{
    int *c_mat;
    MPI_Request requestsa[world_size], requestsb[world_size];
    MPI_Status statusa[world_size], statusb[world_size], statusc[world_size];

    if(world_rank == 0){
        //printf("%d: start\n", world_rank);
        int starta[world_size], startb[world_size];
        for(int i = 0; i < world_size; i++){
            starta[i] = n / world_size * i;
            startb[i] = l / world_size * i;
        }
        c_mat = new int [n*l];

        for(int i = 1; i < world_size; i++){
            int start = starta[i], end = (i == world_size-1) ? n : starta[i+1];
            MPI_Isend(&a_mat[start*m], (end-start)*m, MPI_INT, i, A_TAG, MPI_COMM_WORLD, &requestsa[i]);
            MPI_Isend(b_mat, l*m, MPI_INT, i, B_TAG, MPI_COMM_WORLD, &requestsb[i]);
            //printf("%d: Isend to %d\n", world_rank, i);
        }

        for(int i = 0; i < starta[1]; i++){
            for(int j = 0; j < l; j++){
                c_mat[i*l+j] = 0;
                for(int k = 0; k < m; k++){
                    c_mat[i*l+j] += a_mat[i*m+k] * b_mat[j*m+k];
                }
            }
        }
        MPI_Waitall(world_size-1, requestsa+1, statusa+1);
        MPI_Waitall(world_size-1, requestsb+1, statusb+1);

        MPI_Request ret[n];
        MPI_Status ret_sta[n];

        for(int i = 1; i < world_size; i++){
            int start = n / world_size * i;
            int end = (i == world_size-1) ? n : n / world_size * (i+1);
            //printf("%d: receive from %d\n", world_rank, i);
            for(int j = start; j < end; j++){
                MPI_Irecv(&c_mat[j*l], l, MPI_INT, i, C_TAG + (j-start), MPI_COMM_WORLD, &ret[j]);
            }
            //MPI_Recv(&c_mat[start*l], (end-start)*l, MPI_INT, i, C_TAG, MPI_COMM_WORLD, &statusc[i]);
        }
        int start = n / world_size * 1;
        MPI_Waitall(n-start, ret+start, ret_sta+start);
    }
    else{
        //printf("%d: start\n", world_rank);
        int start = n / world_size * world_rank;
        int end = (world_rank == world_size-1) ? n : n / world_size * (world_rank+1);
        int my_n = end - start;
        int *a = new int [my_n*m];
        MPI_Irecv(a, my_n*m, MPI_INT, 0, A_TAG, MPI_COMM_WORLD, &requestsa[world_rank]);
        int *b = new int [l*m];
        MPI_Irecv(b, l*m, MPI_INT, 0, B_TAG, MPI_COMM_WORLD, &requestsb[world_rank]);
        //printf("%d: receive from %d\n", world_rank, 0);

        int *c = new int [my_n*l];

        MPI_Wait(&requestsa[world_rank], &statusa[world_rank]);
        MPI_Wait(&requestsb[world_rank], &statusb[world_rank]);


        MPI_Request ret[my_n];
        MPI_Status ret_sta[my_n];
        for(int i = 0; i < my_n; i++){
            for(int j = 0; j < l; j++){
                c[i*l+j] = 0;
                for(int k = 0; k < m; k++){
                    c[i*l+j] += a[i*m+k] * b[j*m+k];
                }
            }
            MPI_Isend(&c[i*l], l, MPI_INT, 0, C_TAG+i, MPI_COMM_WORLD, &ret[i]);
        }
        //printf("%d: send to %d\n", world_rank, 0);
        //MPI_Send(c, my_n*l, MPI_INT, 0, C_TAG, MPI_COMM_WORLD);
        MPI_Waitall(my_n, ret, ret_sta);
        delete [] a;
        delete [] b;
        delete [] c;
    }
    
    if(world_rank == 0){
        for(int i = 0; i < n; i++){
            for(int j = 0; j < l; j++){
                printf("%d ", c_mat[i*l+j]);
            }
            printf("\n");
        }
        delete [] c_mat;
    }
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat)
{
    if(world_rank == 0){
        delete [] a_mat;
        delete [] b_mat;
    }
}