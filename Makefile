.PHONY: install test lint demo-server train-smoke tokenize clean help

install: ## Install package in editable mode with dev dependencies
	pip install -e ".[dev]"

test: ## Run the full test suite
	pytest -v

lint: ## Lint codebase with ruff
	ruff check ssl_for_code tests

tokenize: ## Run BPE tokenization demo via CLI
	python3 -m ssl_for_code.cli tokenize --code "def forward(self, x):\n    return self.relu(x)\n"

demo-server: ## Start the Flask autocomplete server in lightweight demo mode
	python3 -m ssl_for_code.server --demo --port 5000

train-smoke: ## Run a quick 1-epoch CPU smoke training on this repository's own Python files
	python3 -m ssl_for_code.dataset.create_dataset --source-dir ssl_for_code --output-dir data
	python3 -m ssl_for_code.train --model transformer_xl --d-model 64 --n-layers 2 --n-heads 4 --d-ff 128 --batch-size 4 --seq-len 32 --epochs 1 --max-steps 5 --device cpu

clean: ## Remove build, cache, and checkpoint artifacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache checkpoints

help: ## Display available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
