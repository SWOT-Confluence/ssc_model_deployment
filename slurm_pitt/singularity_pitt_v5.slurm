#!/bin/bash                                     
#SBATCH --job-name=ssc_processing           
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G 
#SBATCH --cluster=smp
#SBATCH --partition=smp         
#SBATCH --mail-user=luisa.lucchese@pitt.edu    
#SBATCH --mail-type=END,FAIL               
#SBATCH --time=0-02:00:00                  
#SBATCH --qos=normal                         
#SBATCH --array=0-0

module purge                                    
module load singularity/3.9.6

cp -r /ix/llucchese/backup_2024/input /ix/llucchese/confluence_1/ /ix/llucchese/confluence_1/checkpoint /ihome/llucchese/luv14/.netrc /ix/llucchese/confluence_1/ssc_img.sif $SLURM_SCRATCH                      
cd $SLURM_SCRATCH
mkdir output  
 
trap run_on_exit EXIT 
run_on_exit(){ cp -r $SLURM_SCRATCH/output/* $SLURM_SUBMIT_DIR/${SLURM_ARRAY_TASK_ID}/
 
} 

echo "Starting Singularity job..."
singularity run --bind ./:/mnt/data,./:/root/ --env AWS_BATCH_JOB_ARRAY_INDEX=${SLURM_ARRAY_TASK_ID} --nv -B /ix/llucchese/backup_2024/input ssc_img.sif -i 0 -j hls_edited2.json -o . -r /ix/llucchese/confluence_1/reaches.json -d /ix/llucchese/backup_2024/input --ckpt_path /ix/llucchese/confluence_1/checkpoint/deeplabv3p_distrib.pth.tar --latlon /ix/llucchese/confluence_1/coords_json2.json
echo "Job completed"

crc-job-stats                                 

cp *.csv $SLURM_SUBMIT_DIR/out_all/
