#!/bin/bash
# 保存为 run_quick.sh
# 最简单的版本，直接启动两个终端

gnome-terminal -- bash -c "source examples/libero.venv/bin/activate; export PYTHONPATH=\$PYTHONPATH:\$PWD/third_party/libero; python examples/libero/main.py; exec bash"
gnome-terminal -- bash -c "source .venv/bin/activate; python scripts/main.py; exec bash"
