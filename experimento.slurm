#!/bin/bash
#SBATCH --job-name=calor2d_mpi
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=01:00:00
#SBATCH --output=resultados_%j.txt

# Carga de módulos de compilación y MPI en Khipu
module purge
module load openmpi || module load intel-mpi

ITMAX=1000
REPETICIONES=3

echo "=== INICIO EXPERIMENTO: $(date) ==="

echo "--- 1. MEDICIONES SECUENCIALES (heat.cpp) ---"
for n in 512 1000 1024 2048 4096; do
    for rep in $(seq 1 $REPETICIONES); do
        echo -n "Rep $rep: "
        ./heat_seq $n $ITMAX
    done
done

echo "--- 2. MEDICIONES PARALELAS (heat_mpi.cpp) ---"

for n in 512 1000 1024 2048 4096; do
    for p in 1 2 4 8 16; do
        for rep in $(seq 1 $REPETICIONES); do
            echo -n "Rep $rep: "
            mpirun -np $p ./heat_mpi $n $ITMAX
        done
    done
done

echo "=== FIN EXPERIMENTO: $(date) ==="