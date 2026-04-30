# Context Trimmer

多轮对话上下文裁剪工具，使用确定性规则实现压缩触发、摘要块合并和预算内完整性验证。

## 安装

使用 `pip` 或 `uv` 进行安装：

```bash
pip install -e .
```

或：

```bash
uv pip install -e .
```

如需运行测试，安装测试依赖：

```bash
pip install -e ".[test]"
```

## CLI 使用示例

### 从 stdin 读取 JSON 并压缩

```bash
echo '{"messages": [{"role": "system", "content": "You are a helper"}, {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}' | context-trimmer 50
```

### 从文件读取 JSON 并压缩

```bash
context-trimmer 100 -i input.json
```

### Dry Run 模式（仅打印将要折叠的对数）

```bash
echo '{"messages": [{"role": "system", "content": "Sys"}, {"role": "user", "content": "U1"}, {"role": "assistant", "content": "A1"}]}' | context-trimmer 10 --dry-run
```

### 预算为 0（仅保留 system 消息）

```bash
echo '{"messages": [{"role": "system", "content": "Sys"}, {"role": "user", "content": "Hello"}]}' | context-trimmer 0
```

## Python API 使用示例

```python
from context_trimmer import Session

# 从字典列表创建会话
session = Session.from_dicts([
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language"},
    {"role": "user", "content": "How to learn it?"},
    {"role": "assistant", "content": "Start with tutorials"},
])

# 查询当前消息列表
messages = session.get_messages()

# 追加一轮对话
session.add_turn("New question", "New answer")

# 在预算下执行压缩
result = session.compress(budget=100)
print(f"折叠了 {result.stats['fold_count']} 对消息")

# 获取压缩后的消息
compressed = session.to_dicts()
```

## 运行测试

```bash
pytest
```

或 verbose 模式：

```bash
pytest -v
```
