#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#define TAG 1

long long int toss_n_dots(long long int tosses, unsigned int seed)
{
    long long int in_cnt = 0;

    for(long long int i = 0; i < tosses; i++){
        double x = rand_r(&seed) / (double) RAND_MAX ;
        double y = rand_r(&seed) / (double) RAND_MAX ;
        if(x * x + y * y <= 1)
            in_cnt++;
    }
    return in_cnt;
}


int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Status status;
    long long int total_in = 0;

    // TODO: binary tree redunction
    long long int sub_tosses = tosses / world_size;
    if(world_rank == 0)
        sub_tosses += tosses % world_size;
    long long int my_cnt = toss_n_dots(sub_tosses, world_rank);
    int my_tree_num = world_size - world_rank;
    int iter_num = 0;

    while(my_tree_num % 2 == 0){  
        long long int temp = 0;
        int source = (1 << iter_num) + world_rank;
        //printf("Node %d receive from node %d\n", world_rank, source);
        MPI_Recv(&temp, 1, MPI_LONG, source, TAG, MPI_COMM_WORLD, &status);
        my_cnt += temp;
        my_tree_num /= 2;
        iter_num++;
    }
    int dest = world_rank - (1<<iter_num);
    if(world_rank != 0){
        //printf("Node %d send to node %d\n", world_rank, dest);
        MPI_Send(&my_cnt, 1, MPI_LONG, dest, TAG, MPI_COMM_WORLD);
    }
        
    
    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = (double)my_cnt / tosses * 4;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
