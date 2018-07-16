JOB_NAME = cholesky
ROOT_DIR = /lustre/gt04/t04019/cholesky
USER = t04019
HOST = reedbush-u.cc.u-tokyo.ac.jp

all-remote:
	make copy && ssh $(USER)@$(HOST) -T "make all-local -C $(ROOT_DIR)"
all-local:
	make clean && make compile && make run
run:
	qsub -N $(JOB_NAME) run.bash
	#qsub -hold_jid $(JOB_NAME) -cwd echo "Job $(JOB_NAME) finished"
compile:
	mpicc -g cholesky.c -lm -fopenmp
copy:
	scp cholesky.c run.bash Makefile $(USER)@$(HOST):$(ROOT_DIR)
clean:
	rm -rf a.out err.log out.log core.*
