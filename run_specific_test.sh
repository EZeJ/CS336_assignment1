#!/bin/bash

uv run pytest -k "test_embedding or test_linear or test_rmsnorm or test_swiglu"