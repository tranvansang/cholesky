#!/bin/sh
#PBS -q u-lecture
#PBS -W group_list=gt04
#PBS -N stream
#PBS -l select=8:mpiprocs=8:ompthreads=16
#PBS -l walltime=00:10:00
#PBS -e err-omp32.log
#PBS -o out-omp32.log

cd $PBS_O_WORKDIR
. /etc/profile.d/modules.sh
mpirun ./a.out
