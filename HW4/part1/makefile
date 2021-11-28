OBJECT := 	pi_block_linear.out pi_block_tree.out pi_nonblock_linear.out\
			pi_gather.out pi_reduce.out pi_one_side.out
MPIRUN_FLAG := -np $(NODENUM) -hostfile hosts
ITER_NUM := 1000000000

all: clean $(OBJECT)

%.out: %.cc
	mpicxx $< -o $@ 

q1: hello.cc
	mpicxx ./hello.cc -o mpi_hello.out
	mpirun $(MPIRUN_FLAG) mpi_hello.out

q12: pi_block_linear.cc
	mpicxx pi_block_linear.cc -o pi_block_linear.out -O3
	mpirun $(MPIRUN_FLAG) pi_block_linear.out $(ITER_NUM)

q122: pi_block_tree.cc
	mpicxx pi_block_tree.cc -o pi_block_tree.out
	mpirun $(MPIRUN_FLAG) pi_block_tree.out $(ITER_NUM)

q123: pi_nonblock_linear.cc
	mpicxx pi_nonblock_linear.cc -o pi_nonblock_linear.out
	mpirun $(MPIRUN_FLAG) pi_nonblock_linear.out $(ITER_NUM)

q124: pi_gather.cc
	mpicxx pi_gather.cc -o pi_gather.out
	mpirun $(MPIRUN_FLAG) pi_gather.out $(ITER_NUM)

q125: pi_reduce.cc
	mpicxx pi_reduce.cc -o pi_reduce.out
	mpirun $(MPIRUN_FLAG) pi_reduce.out $(ITER_NUM)

q126: pi_one_side.cc
	mpicxx pi_one_side.cc -o pi_one_side.out
	mpirun $(MPIRUN_FLAG) pi_one_side.out $(ITER_NUM)

pingpong: ping_pong.c
	mpicxx ping_pong.c -o ping_pong.out
	mpirun -np 2 -hostfile pingponghost ping_pong.out
clean:
	rm -rf *.out