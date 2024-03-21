This document contains instructions for installation on HPC systems using the SLURM workload manager. These instructions are written especially for the CSC Puhti cluster, but the concepts should be very similar in other HPC systems as well.

# Loading data

It is useful to have your datasets in Allas in a zip or tar file, and load them to `data/raw` in the beginning of each session.
Using an NVME disk makes a large difference in training speed, reducing the bottleneck of data loading.
Extracting the data from `data/raw` to the temporary disk at `$TMPDIR` should be done in the beginning of each batchjob. It will add a few seconds to the beginning of the script, but the increased speed from NVME makes it worthwhile.

