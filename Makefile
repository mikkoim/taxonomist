docker:
	docker build . -t cuda-conda

singularity:
	sudo singularity build Env.sif docker-daemon://cuda-conda:latest

package:
	pip install --user -e .