# Convenience targets — run from project root
#
# Local (Mac):
#   make env-mac         create conda env for Apple Silicon
#   make env-gpu         create conda env for CUDA server
#   make smoke           quick import check (no training)
#
# Docker (4090 server):
#   make docker-build    build the image
#   make docker-lora     run LoRA baseline in container
#   make docker-hrm      run HRM phase 1 in container
#   make docker-reduce   run BT reduction in container
#   make docker-hrm-p2   run HRM phase 2 in container

.PHONY: env-mac env-gpu smoke docker-build docker-lora docker-hrm docker-reduce docker-hrm-p2

env-mac:
	conda env create -f environment_mac.yaml

env-gpu:
	conda env create -f environment_cuda.yaml

smoke:
	python -c "from src.models.transformer import TinyGPT; print('transformer OK')"
	python -c "from src.models.ssm import SSM; print('ssm OK')"
	python -c "from src.adapters.lora import LoRALinear; print('lora OK')"
	python -c "from src.adapters.hrm_adapter import HRMAdapter; print('hrm_adapter OK')"
	python -c "from src.reduction.balanced_truncation import bt_reduce; print('bt OK')"

docker-build:
	docker build -t hrm-adapters .

docker-lora:
	docker compose run train-lora

docker-hrm:
	docker compose run train-hrm

docker-reduce:
	docker compose run reduce

docker-hrm-p2:
	docker compose run train-hrm-p2
