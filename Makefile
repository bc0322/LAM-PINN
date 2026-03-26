install:
	pip install -e .

train:
	python scripts/run_train.py --config configs/train.yaml

adapt:
	python scripts/run_adapt.py --config configs/adapt.yaml
