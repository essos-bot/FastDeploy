#!/usr/bin/env python3
"""
Mermaid 流程图语法验证脚本
"""

import re
import sys

def validate_mermaid_syntax(mermaid_content):
    """
    验证 Mermaid Graph TD 语法
    """
    print("🔍 开始验证 Mermaid 语法...")

    lines = mermaid_content.strip().split('\n')
    errors = []
    warnings = []

    # 提取 Graph TD 部分
    in_graph = False
    graph_lines = []

    for line in lines:
        line = line.strip()
        if line == 'graph TD':
            in_graph = True
            continue
        elif line == '```' and in_graph:
            break
        elif in_graph:
            graph_lines.append(line)

    if not graph_lines:
        errors.append("❌ 未找到 Graph TD 内容")
        return errors, warnings

    print(f"📊 找到 {len(graph_lines)} 行 Graph TD 内容")

    # 检查常见语法问题
    for i, line in enumerate(graph_lines, 1):
        line_num = f"第 {i} 行"

        # 跳过注释行
        if line.startswith('%%') or not line:
            continue

        # 检查双下划线（Mermaid 不支持）
        if '__' in line:
            errors.append(f"❌ {line_num}: 包含双下划线 '__'，这是 Mermaid 保留字符")

        # 检查点号（可能导致解析问题）
        if '.' in line and '-->' in line:
            # 检查是否在节点标签中包含点号
            if re.search(r'\[[^\]]*\.[^\]]*\]', line):
                warnings.append(f"⚠️  {line_num}: 节点标签中包含点号，可能导致解析问题")

        # 检查括号（可能导致解析问题）
        if '(' in line and ')' in line and '-->' in line:
            if re.search(r'\[[^\]]*\([^)]*\)[^\]]*\]', line):
                warnings.append(f"⚠️  {line_num}: 节点标签中包含括号，可能导致解析问题")

        # 检查箭头语法
        if '-->' in line:
            parts = line.split('-->')
            if len(parts) != 2:
                errors.append(f"❌ {line_num}: 箭头语法错误，应该只有两个节点")

            # 检查节点格式
            for part in parts:
                part = part.strip()
                if part and not (part.startswith('%%') or re.match(r'^[A-Z]+(\[.*\])?$', part)):
                    # 检查是否是 classDef 或 note 语句
                    if not (part.startswith('classDef') or part.startswith('class ') or part.startswith('note ')):
                        warnings.append(f"⚠️  {line_num}: 节点格式可能有问题: '{part}'")

    return errors, warnings

def test_mermaid_with_online_validator(mermaid_content):
    """
    尝试使用在线 Mermaid 编辑器测试（模拟）
    """
    print("\n🌐 尝试模拟在线验证...")

    # 简单的语法检查
    required_elements = ['graph TD', 'classDef', 'class ']
    missing_elements = []

    for element in required_elements:
        if element not in mermaid_content:
            missing_elements.append(element)

    if missing_elements:
        print(f"⚠️  缺少元素: {missing_elements}")
    else:
        print("✅ 包含所有基本元素")

    # 检查节点名称的唯一性
    node_names = re.findall(r'^([A-Z]+)(?=\[|\s|$)', mermaid_content, re.MULTILINE)
    duplicate_nodes = [name for name in set(node_names) if node_names.count(name) > 1]

    if duplicate_nodes:
        print(f"⚠️  发现重复节点: {duplicate_nodes}")
    else:
        print("✅ 节点名称唯一")

def main():
    # 读取文件
    file_path = '/Users/liujun/Documents/Projects/hack9th/FastDeploy/docs/_docs/fastdeploy_model_weight_loading_flow.md'

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ 文件未找到: {file_path}")
        return

    # 提取 Mermaid 内容
    mermaid_blocks = re.findall(r'```mermaid\n(.*?)\n```', content, re.DOTALL)

    if not mermaid_blocks:
        print("❌ 未找到 Mermaid 代码块")
        return

    print(f"📝 找到 {len(mermaid_blocks)} 个 Mermaid 代码块")

    for i, mermaid_content in enumerate(mermaid_blocks, 1):
        print(f"\n{'='*50}")
        print(f"验证第 {i} 个 Mermaid 代码块")
        print(f"{'='*50}")

        errors, warnings = validate_mermaid_syntax(mermaid_content)

        # 输出结果
        if errors:
            print("\n❌ 发现语法错误:")
            for error in errors:
                print(f"  {error}")
        else:
            print("\n✅ 未发现语法错误")

        if warnings:
            print("\n⚠️  发现警告:")
            for warning in warnings:
                print(f"  {warning}")

        # 额外验证
        test_mermaid_with_online_validator(mermaid_content)

        # 如果有错误，显示问题行
        if errors:
            print("\n🔍 问题代码行:")
            lines = mermaid_content.strip().split('\n')
            for j, line in enumerate(lines, 1):
                for error in errors:
                    if str(j) in error:
                        print(f"  {j}: {line}")

if __name__ == "__main__":
    main()