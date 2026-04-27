#!/bin/bash

# 遇到错误立即停止运行 (极其推荐的防呆设计)
set -e

echo "========================================"
echo "🚀 阶段一: 开始生成正演数据集..."
echo "========================================"
# 运行正演数据生成代码
python /Users/xiechushu/project/SmartMGS_1D/scripts/generate_forward_data.py
echo "✅ 阶段一完成！正演数据已生成。"
echo ""

echo "========================================"
echo "🚀 阶段二: 开始执行批量 DLS 反演..."
echo "========================================"
# 运行反演数据生成代码
python /Users/xiechushu/project/SmartMGS_1D/scripts/generate_inv_data.py
echo "✅ 阶段二完成！反演训练集已生成。"
echo ""

echo "🎉 恭喜！整条 1D-ResNet 数据流水线全部顺利执行完毕！"