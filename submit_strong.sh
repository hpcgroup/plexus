sbatch -N 2 strong_products_2m.sh 2 2 2 1 0.05
sbatch -N 4 strong_products_2m.sh 2 2 2 2 0.05
sbatch -N 8 strong_products_2m.sh 2 2 2 4 0.05
sbatch -N 16 strong_products_2m.sh 2 2 2 8 0.05
sbatch -N 32 strong_products_2m.sh 2 2 2 16 0.05

sbatch -N 4 strong_protein_8m.sh 4 2 2 1 0.02
sbatch -N 8 strong_protein_8m.sh 4 2 2 2 0.02
sbatch -N 16 strong_protein_8m.sh 4 2 2 4 0.02
sbatch -N 32 strong_protein_8m.sh 4 2 2 8 0.02
sbatch -N 64 strong_protein_8m.sh 4 2 2 16 0.021
sbatch -N 128 strong_protein_8m.sh 4 2 2 32 0.020

sbatch -N 8 strong_products_14m.sh 2 4 4 1 0.02
sbatch -N 16 strong_products_14m.sh 2 4 4 2 0.02
sbatch -N 32 strong_products_14m.sh 2 4 4 4 0.02
sbatch -N 64 strong_products_14m.sh 2 4 4 8 0.02
sbatch -N 128 strong_products_14m.sh 2 4 4 16 0.021
sbatch -N 256 strong_products_14m.sh 2 4 4 32 0.020

sbatch -N 16 strong_papers.sh 4 4 4 1 0.02
sbatch -N 32 strong_papers.sh 4 4 4 2 0.02
sbatch -N 64 strong_papers.sh 4 4 4 4 0.02
sbatch -N 128 strong_papers.sh 4 4 4 8 0.02
sbatch -N 256 strong_papers.sh 4 4 4 16 0.021
sbatch -N 512 strong_papers.sh 4 4 4 32 0.020