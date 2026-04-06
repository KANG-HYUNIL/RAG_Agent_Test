SHELL = bash
ifneq ("$(wildcard .env)","")
	include .env
endif

.PHONY: lint format type-check

.ONESHELL:
lint:
	uv run ruff check --fix --exit-non-zero-on-fix --config=ruff.toml

.ONESHELL:
format:
	uv run ruff format --check --config=ruff.toml

.ONESHELL:
type-check:
	uv run pyright
