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

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Status status;
    long long int total_in = 0;
    long long int *local_sum;
    MPI_Win_create(&total_in, sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);

    if (world_rank == 0)
    {
        // Master
        long long int sub_tosses = tosses / world_size + tosses % world_size;
        total_in = toss_n_dots(sub_tosses, world_rank);
        local_sum = new long long int [world_size];
        //MPI_Win_allocate(world_size * sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, &local_sum, &win);
        //MPI_Win_create(local_sum, world_size * sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        
    }
    else
    {
        // Workers
        long long int sub_tosses = tosses / world_size;
        long long int my_cnt = toss_n_dots(sub_tosses, world_rank);

        MPI_Accumulate(&my_cnt, 1, MPI_LONG, 0, 0, 1, MPI_LONG, MPI_SUM, win);
    }

    MPI_Win_fence(0, win);
    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        pi_result = (double)total_in / tosses * 4;
        delete [] local_sum;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}