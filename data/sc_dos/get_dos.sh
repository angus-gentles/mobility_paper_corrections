#!/usr/bin/env bash

cur=$(pwd)
base=In1.0Ga0.0As0.5Sb0.5

extdir="/mnt/lustre/a2fs-nvme/work/i254/i254/gentles/phonopy_more_k/dir_In1.0Ga0.0As0.5Sb0.5_cube_2x2x2"
#for i in {0..5} ; do
#	scp gentles@login.archer2.ac.uk:${extdir}/calc_${i}/total_dos.dat ${base}.${i}.total_dos.dat
#done

scp gentles@login.archer2.ac.uk:${extdir}/outdir/*.dat .
