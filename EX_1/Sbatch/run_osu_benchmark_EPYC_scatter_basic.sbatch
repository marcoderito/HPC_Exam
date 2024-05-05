#!/bin/bash
#SBATCH --job-name=HPC_epyc_scatter_basic
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --time=02:00:00
#SBATCH --partition=EPYC
#SBATCH --exclusive

# Definizione del percorso del file di output
output_file="/orfeo/cephfs/home/dssc/mderito/osu-micro-benchmarks-7.3/epyc_scatter_basic.csv"

# Verifica se il modulo OpenMPI  disponibile
if ! module load openMPI/4.1.5/gnu/12.2.1; then
    echo "Errore: Impossibile caricare il modulo OpenMPI."
    exit 1
fi

# Creazione del file CSV e scrittura dell'intestazione
echo "Processes,Size,Latency" > "$output_file"

# Numero di ripetizioni per ottenere una media
repetitions=10000

# Ciclo sui processi
for j in {1..8}; do
    processes=$((2**j))
    # Ciclo sulle dimensioni
    for i in {1..20}; do
        size=$((2**i))
        # Esecuzione di osu_scatter con dimensione fissa e numero di processi su due nodi
        result_scatter=$(mpirun --map-by core -np "$SLURM_NTASKS" --mca coll_tuned_use_dynamic_rules true --mca coll_tuned_scatter_algorithm 1 osu_scatter -m "$size" -x "$repetitions" -i "$repetitions" | tail -n 1 | awk '{print $2}')
        # Scrivi i risultati nel file CSV
        echo "$processes,$size,$result_scatter" >> "$output_file"
    done
done
