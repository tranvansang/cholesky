#!/bin/sh
#PBS -q u-lecture
#PBS -W group_list=gt04
#PBS -N stream
#PBS -l select=2:mpiprocs=4:ompthreads=1
#PBS -l walltime=00:10:00
#PBS -e err-omp1.log
#PBS -o out-omp1.log

cd $PBS_O_WORKDIR
. /etc/profile.d/modules.sh
mpirun ./a.out
