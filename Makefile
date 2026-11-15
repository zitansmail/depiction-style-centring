IMAGE ?= depiction-style-centring
RUN = docker run --rm --gpus all -v $(PWD)/cache:/app/cache -v $(PWD)/data:/app/data -v $(PWD)/results:/app/results $(IMAGE)

build:            ## build the GPU Docker image (Linux x86-64 + NVIDIA)
	docker build -t $(IMAGE) .
test:             ## CPU tests (metrics against brute force, properties of DSC)
	docker run --rm $(IMAGE)
quick:            ## whole pipeline on 2 images per class and depiction (minutes, numbers meaningless)
	$(RUN) python scripts/reproduce.py all --cap 2
reproduce:        ## every table of the paper (GPU recommended; downloads ~25 GB of datasets)
	$(RUN) python scripts/reproduce.py all
build-cpu:        ## build the CPU image (any machine, including Apple Silicon)
	docker build -f Dockerfile.cpu -t $(IMAGE)-cpu .
test-cpu:         ## CPU tests in the CPU image
	docker run --rm $(IMAGE)-cpu
