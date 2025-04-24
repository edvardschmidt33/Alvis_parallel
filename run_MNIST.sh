#!/usr/bin/bash
#SBATCH --time=00:10:00
#SBATCH -A NAISS2025-5-98
#SBATCH --gpus-per-node=A40:1
#SBATCH -J CLIP_COCO_test
#SBATCH -o logs/MNIST/train_%j.out
#SBATCH -e logs/MNIST/train_%j.err 

EXE="apptainer exec --nv /cephyr/users/USER/Alvis/DIR/example.sif python3"


${EXE} MNIST_tester.py