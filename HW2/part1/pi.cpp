#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <thread>
#include <random>
#include <vector>

using namespace std;

vector<long long int> bridge;
vector<unsigned int> seeds;
void *toss_n_dots(void *arg);

int main(int argc, char** argv)
{
    int thread_cnt = atoll(argv[1]);
    long long int n = atoll(argv[2]);
    
    //cout<<thread_cnt<<endl<<n<<endl;

    bridge.resize(thread_cnt);
    seeds.resize(thread_cnt);
    pthread_t th[thread_cnt];
    int ids[thread_cnt];
    unsigned int seed = 12;

    for(int i = 0; i < thread_cnt; i++){
        ids[i] = i;
        seeds[i] = rand_r(&seed);
        bridge[i] = n / thread_cnt +  ((i == thread_cnt - 1) ? (n % thread_cnt) : 0);
        pthread_create(&th[i], NULL, toss_n_dots, (void *)&ids[i]);
    }

    long long int total_cnt = 0;
    for(int i = 0; i < thread_cnt; i++){
        pthread_join(th[i], NULL);
        total_cnt += bridge[i];
    }
    double pi = (total_cnt * 4) / (double) n;
    cout<<pi<<endl;
}



void *toss_n_dots(void *arg)
{
    int * id = (int *) arg;
    long long int n = bridge[*id];
    long long int in_cnt = 0;
    unsigned int seed = seeds[*id];
    //cout<<seeds[*id]<<endl;

    for(long long int i = 0; i < n; i++){
        double x = rand_r(&seed) / (double) RAND_MAX ;
        double y = rand_r(&seed) / (double) RAND_MAX ;
        if(x * x + y * y <= 1)
            in_cnt++;
    }
    bridge[*id] = in_cnt;
    
    return NULL;
}