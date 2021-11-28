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
    long long int sub_tosses = tosses / world_size;
    if(world_rank == 0)
        sub_tosses += tosses % world_size;
    long long int my_cnt = toss_n_dots(sub_tosses, world_rank);

    // TODO: use MPI_Gather
    long long int local_sum[world_size];
    MPI_Gather(&my_cnt, 1, MPI_LONG, local_sum, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        // TODO: PI result
        for(int i = 1; i < world_size; i++){
            my_cnt += local_sum[i];
        }
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
