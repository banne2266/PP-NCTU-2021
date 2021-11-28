DATA:=data1_4

all: matmul

matmul: main.cc mat.cc
	mpicxx main.cc mat.cc -o matmul -O3

serial:	serial.cpp
	g++ serial.cpp -o ss -O3

listdata:
	ls /HW4/data

run4: matmul
	mpirun -np 4 --hostfile hosts matmul < /HW4/data/$(DATA) > ans.txt

run9: matmul
	mpirun -np 9 --hostfile hosts matmul < /HW4/data/$(DATA) > ans.txt

run_serial: serial
	./ss < /HW4/data/$(DATA) > s.txt

diff: run4 run_serial
	diff s.txt ans.txt


clean:
	rm matmul