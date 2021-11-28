#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

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
    long long int total_in = 0;

    // TODO: use MPI_Reduce
    long long int sub_tosses = tosses / world_size;
    long long int my_cnt = toss_n_dots(sub_tosses, (unsigned int) world_rank);
    MPI_Reduce(&my_cnt, &total_in, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = (double)total_in / tosses * 4;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}

