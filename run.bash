#!/bin/sh
#PBS -q u-lecture
#PBS -W group_list=gt04
#PBS -N stream
#PBS -l select=1:mpiprocs=2:ompthreads=1
#PBS -l walltime=00:10:00
#PBS -e err.log
#PBS -o out.log

cd $PBS_O_WORKDIR
. /etc/profile.d/modules.sh
mpirun ./a.out
