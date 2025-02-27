docker:
	docker build . -t cuda-conda

singularity:
	sudo singularity build Env.sif docker-daemon://cuda-conda:latest

package:
	pip install --user -e .

clear_tests:
	rm -rf test_outputs
	rm -rf tests/data/predictions
	rm -rf tests/data/features
	rm -rf tests/data/predictions