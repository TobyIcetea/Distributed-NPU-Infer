#!/bin/bash

nodes=$(kubectl get nodes -o name | grep '^node/atlas' | sed 's/^node\///')

output_file="npu_levels.json"

# 创建关联数组（类似于 map）
declare -A levels

for node in $nodes; do
    echo "Checking NPU level on: $node"
    if level=$(ssh -o ConnectTimeout=5 "$node" "npu-smi info -t nve-level -i 0 -c 0 2>/dev/null | awk '/nve level/ {print \$4}' | cut -dT -f1"); then
        levels["$node"]=$level
    else
        levels["$node"]="null"
        echo "[WARNING] Failed to check $node" >&2
    fi
done

# 遍历数组，生成jq表达式
jq_filter='{}'
for key in "${!levels[@]}"; do
  jq_filter+=" | .[\"$key\"] = ${levels[$key]}"
done

echo "Check All NPU Levels Down!"

# 生成JSON文件
jq -n "$jq_filter" > ${output_file}
