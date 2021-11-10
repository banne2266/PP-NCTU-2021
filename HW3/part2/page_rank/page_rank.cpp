#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>
#include <vector>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  bool converged = false;
  double *score_old = new double[numNodes];
  double *score_new = new double[numNodes];
  double *out_going_sum = new double[numNodes];

  std::vector<int> no_outgoing_list;
  #pragma omp parallel for
  for (int i = 0; i < numNodes; i++){
    if(outgoing_size(g, i) == 0){
      #pragma omp critical
      no_outgoing_list.push_back(i);
    }
  }

  
  #pragma omp parallel for
  for (int i = 0; i < numNodes; i++){
    score_old[i] = equal_prob;
  }

  
  while(!converged){
    double no_outgoing_sum = 0.0;
    for(auto j : no_outgoing_list){
      no_outgoing_sum += score_old[j];
    }
    no_outgoing_sum = damping * no_outgoing_sum / numNodes;

    #pragma omp parallel for
    for (int vi = 0; vi < numNodes; vi++){
      out_going_sum[vi] = score_old[vi] / outgoing_size(g, vi);
    }

    #pragma omp parallel for schedule(dynamic)
    for (int vi = 0; vi < numNodes; vi++){
      score_new[vi] = 0.0;
      const Vertex* incoming_start = incoming_begin(g, vi);
      const Vertex* incoming__end = incoming_end(g, vi);

      for (const Vertex* vj=incoming_start; vj!=incoming__end; vj++){
        score_new[vi] += out_going_sum[*vj];
      }
      score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes + no_outgoing_sum;
    }

    double global_diff = 0.0;
    #pragma omp parallel for reduction(+:global_diff)
    for (int i = 0; i < numNodes; i++){
      global_diff += fabs(score_new[i] - score_old[i]);
    }

    converged = (global_diff < convergence);

    #pragma omp parallel for
    for (int i = 0; i < numNodes; i++){
      score_old[i] = score_new[i];
    }
  }

  #pragma omp parallel for
  for (int i = 0; i < numNodes; i++)
  {
    solution[i] = score_old[i];
  }
  delete [] score_new;
  delete [] score_old;
  delete [] out_going_sum;



  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
}
