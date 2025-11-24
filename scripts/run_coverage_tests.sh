#!/bin/bash

# 测试覆盖率运行脚本
# 用法: ./run_coverage_tests.sh [test_file_pattern]
# 示例: ./run_coverage_tests.sh tests/entrypoints/openai/test_api_server.py
#       ./run_coverage_tests.sh "tests/entrypoints/openai/test_*.py"

# 设置脚本目录和路径
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tests_path="$DIR/../tests/"
export PYTEST_INI="$DIR/../tests/cov_pytest.ini"
run_path=$(realpath "$DIR/../")

# 设置覆盖率相关环境变量
export COVERAGE_FILE=${COVERAGE_FILE:-$DIR/../coveragedata/.coverage}
export COVERAGE_RCFILE=${COVERAGE_RCFILE:-$DIR/../scripts/.coveragerc}

# 获取测试文件参数
TEST_PATTERN=${1:-"tests/entrypoints/openai/test_api_server.py"}

echo "===================================="
echo "测试覆盖率运行脚本"
echo "测试文件模式: $TEST_PATTERN"
echo "===================================="

# 创建失败测试日志文件
failed_tests_file="failed_tests.log"
> "$failed_tests_file"

# 统计变量
failed_pytest=0
success_pytest=0

##################################
# 第一部分：执行 pytest 测试
##################################
echo "开始执行测试..."

# 如果传入的是模式（包含*），使用find查找匹配的文件
if [[ "$TEST_PATTERN" == *"*"* ]]; then
    echo "查找匹配的测试文件: $TEST_PATTERN"
    TEST_FILES=$(find "$run_path" -name "$(basename "$TEST_PATTERN")" -path "*/$TEST_PATTERN" 2>/dev/null)

    if [ -z "$TEST_FILES" ]; then
        echo "错误: 未找到匹配的测试文件"
        exit 1
    fi

    echo "找到以下测试文件:"
    echo "$TEST_FILES"
else
    # 单个文件
    TEST_FILES="$run_path/$TEST_PATTERN"

    if [ ! -f "$TEST_FILES" ]; then
        echo "错误: 测试文件不存在: $TEST_FILES"
        exit 1
    fi
fi

# 运行每个测试文件
for file in $TEST_FILES; do
    echo "----------------------------------------"
    echo "运行测试文件: $file"
    python -m coverage run -m pytest -c "${PYTEST_INI}" "$file" -vv -s
    status=$?

    if [ "$status" -ne 0 ]; then
        echo "测试失败: $file" >> "$failed_tests_file"
        failed_pytest=$((failed_pytest+1))
        echo "❌ 测试失败"
    else
        success_pytest=$((success_pytest+1))
        echo "✅ 测试成功"
    fi

    # 清理可能残留的进程
    ps -ef | grep "${FD_CACHE_QUEUE_PORT}" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
    ps -ef | grep "${FD_ENGINE_QUEUE_PORT}" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
done

##################################
# 第二部分：汇总测试结果
##################################
echo "===================================="
echo "测试结果汇总:"
echo "总测试文件数: $((failed_pytest + success_pytest))"
echo "成功测试文件数: $success_pytest"
echo "失败测试文件数: $failed_pytest"

if [ "$failed_pytest" -ne 0 ]; then
    echo "失败的测试文件列表:"
    cat "$failed_tests_file"
    echo ""
fi

##################################
# 第三部分：生成覆盖率报告
##################################
echo "===================================="
echo "生成代码覆盖率报告..."

# 合并覆盖率数据
echo "合并覆盖率数据..."
if python -m coverage combine coveragedata/ 2>/dev/null; then
    echo "✅ 覆盖率数据合并成功"
else
    echo "⚠️  没有覆盖率数据可合并，继续生成报告..."
fi

# 生成覆盖率报告并过滤api相关
echo "生成覆盖率报告..."
echo "----------------------------------------"
echo "API 相关模块覆盖率:"
echo "----------------------------------------"

# 生成完整报告并过滤包含 'api' 的行
if python -m coverage report | grep -i api; then
    echo "----------------------------------------"
    echo "✅ API 覆盖率报告生成完成"
else
    echo "⚠️  未找到 API 相关模块或覆盖率数据"
fi

# 可选：生成HTML报告（如果需要）
echo ""
echo "如需生成详细的HTML覆盖率报告，请运行:"
echo "python -m coverage html -d htmlcov/"

# 可选：显示总体覆盖率
echo ""
echo "----------------------------------------"
echo "总体覆盖率摘要:"
echo "----------------------------------------"
python -m coverage report --skip-empty | tail -n 1

##################################
# 第四部分：最终状态
##################################
echo "===================================="
if [ "$failed_pytest" -ne 0 ]; then
    echo "⚠️  部分测试失败，请检查失败的测试文件"
    exit 8
else
    echo "🎉 所有测试通过！"
    echo "测试覆盖率报告生成完成！"
fi