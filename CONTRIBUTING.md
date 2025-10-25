# CONTRIBUTING.md / 贡献指南

Thank you for considering contributing to CTC_GAN! This document describes how to file issues, submit pull requests, set up the development environment, follow code standards, run tests, and abide by our code of conduct.

感谢你为 CTC_GAN 做出贡献！本文档说明如何提交 Issue、Pull Request，配置开发环境，遵循代码规范，运行测试，以及遵守社区行为准则。

---

## 1. Project Contribution Guidelines / 项目贡献指南

### 1.1 Filing Issues / 提交 Issue
- Before opening a new issue, please search existing issues to avoid duplicates.
- Provide a minimal reproducible example (steps, configs, code, logs).
- Use clear titles and specify environment: OS, Python version, `torch/torchvision` versions, GPU/CPU.
- Tag the issue with appropriate labels: `bug`, `feature`, `docs`, `question`, `help wanted`.

- 在创建新 Issue 前，请先搜索现有 Issue，避免重复。
- 请提供可复现实例（步骤、配置、代码、日志）。
- 使用清晰标题，并注明环境信息：操作系统、Python 版本、`torch/torchvision` 版本、GPU/CPU。
- 使用合适的标签：`bug`、`feature`、`docs`、`question`、`help wanted`。

### 1.2 Pull Requests (PR) / 提交 Pull Request
- Fork the repo, create a new branch from `main`.
- Branch naming: `feature/<short-name>`, `fix/<short-name>`, `docs/<short-name>`, `chore/<short-name>`.
- Keep changes focused and minimal; update documentation and add tests for new features.
- Ensure the project builds and tests pass locally before submitting.
- Link related issues with `Fixes #<issue-id>` or `Refs #<issue-id>`.
- Include a brief and clear PR description: problem, solution, scope, limitations.

- Fork 仓库，并从 `main` 创建新分支。
- 分支命名：`feature/<简名>`、`fix/<简名>`、`docs/<简名>`、`chore/<简名>`。
- 修改应聚焦且最小化；新增功能需同时更新文档并补充测试。
- 提交前请在本地确保可运行且测试通过。
- 用 `Fixes #<issue-id>` 或 `Refs #<issue-id>` 关联相关 Issue。
- 在 PR 描述中清晰说明：问题、方案、范围与限制。

### 1.3 PR Checklist / PR 提交检查清单
- Code compiles and runs (`train.py`, `generate.py`, `evaluate.py` at least basic flow).
- Tests added and pass locally; coverage meets standards (see Section 4).
- Docs updated (`README.md` / `README_CN.md` / relevant comments).
- No unrelated changes; follows code style and commit format.
- One maintainer review and approval required before merge.

- 代码可编译运行（至少 `train.py`、`generate.py`、`evaluate.py` 基本流程打通）。
- 已添加测试且本地通过；覆盖率达标（见第 4 节）。
- 文档已更新（`README.md` / `README_CN.md` / 相关注释）。
- 无无关改动；符合代码风格与提交信息规范。
- 合并前需至少一位维护者审核通过。

---

## 2. Development Environment / 开发环境配置

### 2.1 Requirements / 环境要求
- Python 3.10+ (Windows recommended; Linux/macOS can adapt commands accordingly).
- Optional GPU with CUDA for faster training/evaluation.

- Python 3.10+（推荐 Windows；Linux/macOS 需调整命令）。
- 可选：支持 CUDA 的 GPU 以提升训练/评估速度。

### 2.2 Dependencies / 项目依赖
- Core: `torch`, `torchvision`
- Image IO: `Pillow`
- Metrics: `numpy`, `scipy`
- Config: `pyyaml`
- Dev/Test (recommended): `pytest`, `pytest-cov`, `flake8`, `black`

- 核心：`torch`、`torchvision`
- 图像：`Pillow`
- 评估：`numpy`、`scipy`
- 配置：`pyyaml`
- 开发/测试（推荐）：`pytest`、`pytest-cov`、`flake8`、`black`

### 2.3 Setup Steps (Windows) / 安装步骤（Windows）

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\activate
```

```powershell
pip install -U pip
```

```powershell
pip install torch torchvision pillow numpy scipy pyyaml
```

```powershell
# Optional dev tools / 可选开发工具
pip install pytest pytest-cov flake8 black
```

### 2.4 Run / 运行

```powershell
# Train / 训练
python train.py
```

```powershell
# Generate / 生成
python generate.py
```

```powershell
# Evaluate (Top-K) / 评估（Top-K）
python evaluate.py --checkpoint_dir checkpoints\ --top_k 3
```

```powershell
# Evaluate single checkpoint / 评估单个检查点
python evaluate.py --single checkpoints\model_epoch_050.pth
```

### 2.5 Configuration / 配置
- Main file: `configs\train_config.yaml`
- Edit `paths.data_dir`, `paths.output_dir`, `paths.checkpoint_dir`, `paths.log_dir`.
- Set `device` to `"cuda"` or `"cpu"`.

- 主配置文件：`configs\train_config.yaml`
- 修改 `paths.data_dir`、`paths.output_dir`、`paths.checkpoint_dir`、`paths.log_dir`。
- 设置 `device` 为 `"cuda"` 或 `"cpu"`。

---

## 3. Code Standards / 代码规范

### 3.1 Style Guide / 风格指南
- Follow PEP 8; use type hints (`typing`).
- Prefer clear names; avoid single-letter variables.
- Keep functions small and single-responsibility.
- Imports: stdlib → third-party → local; avoid unused imports.

- 遵循 PEP 8；使用类型标注（`typing`）。
- 使用清晰命名；避免单字母变量。
- 函数应小而专注，职责单一。
- 导入顺序：标准库 → 第三方库 → 本地；避免未使用导入。

### 3.2 Docstrings & Comments / 文档字符串与注释
- Use English comments/docstrings; preserve technical terms and API names.
- For ambiguous notes, add `[译注]` clarification.
- Prefer Google-style docstrings for functions/classes.

- 使用英文注释与文档字符串；保留技术术语与 API 名称。
- 遇到歧义请添加 `[译注]` 说明。
- 函数/类建议采用 Google 风格 Docstring。

### 3.3 Logging / 日志
- Prefer `utils/logger.py` over `print` for training metrics.
- Keep logs structured; avoid noisy output.

- 训练指标优先使用 `utils/logger.py`，避免使用 `print`。
- 日志结构化且简洁，避免冗余输出。

### 3.4 Commit Messages / 提交信息格式
Use Conventional Commits:
- `feat: add synthetic image generator CLI`
- `fix(generator): clamp outputs to [0,1]`
- `docs: update README_CN with usage`
- `chore: bump dependencies`
- `test: add FID calculation unit tests`

遵循 Conventional Commits：
- `feat: 添加合成图生成 CLI`
- `fix(generator): 修复输出范围到 [0,1]`
- `docs: 更新 README_CN 使用说明`
- `chore: 升级依赖`
- `test: 添加 FID 计算单测`

### 3.5 Branches / 分支
- `feature/*`, `fix/*`, `docs/*`, `chore/*`
- Keep PRs small and focused.

- 使用 `feature/*`、`fix/*`、`docs/*`、`chore/*`
- PR 应小而专注。

---

## 4. Testing Requirements / 测试要求

### 4.1 Test Runner / 测试框架
- Recommended: `pytest` with `pytest-cov`.
- Alternatively: Python `unittest`.

- 推荐使用 `pytest` 与 `pytest-cov`。
- 或使用 Python 原生 `unittest`。

### 4.2 Structure / 目录结构
- Place tests under `tests/` mirroring source structure.
- Name files like `test_<module>.py`.

- 测试文件放在 `tests/`，结构与源码对应。
- 测试文件命名为 `test_<module>.py`。

### 4.3 Commands / 运行命令

```powershell
# Run all tests / 运行所有测试
pytest -q
```

```powershell
# With coverage / 带覆盖率报告
pytest --cov=./ --cov-report=term-missing
```

```powershell
# Linting (optional) / 代码静态检查（可选）
flake8 .
```

```powershell
# Formatting (optional) / 代码格式化（可选）
black .
```

### 4.4 Coverage Standard / 覆盖率标准
- New or modified code must reach ≥ 80% line coverage.
- Include tests for normal paths, edge cases, and failure handling.

- 新增或修改代码的覆盖率需达到 ≥ 80%。
- 测试需覆盖正常流程、边界场景与失败处理。

---

## 5. Code of Conduct & Contributor Agreement / 行为准则与贡献者协议

### 5.1 Code of Conduct / 行为准则
- Be respectful; no harassment or discrimination.
- Be constructive; welcome diverse perspectives.
- Keep discussions on-topic; avoid personal attacks.
- Report unacceptable behavior to maintainers.

- 尊重他人；禁止骚扰与歧视。
- 建设性沟通；欢迎多元观点。
- 讨论聚焦议题；避免人身攻击。
- 若发现不当行为，请联系维护者。

### 5.2 Contributor License / 贡献者许可
- By contributing, you agree your contributions are licensed under the project’s MIT License.
- Ensure you have the right to contribute the code and content.

- 贡献即表示你同意你的代码按本项目 MIT 许可证授权。
- 请确保你拥有贡献代码与内容的合法权利。

### 5.3 DCO (Signed-off-by) / 开发者认证声明
Add a Signed-off-by line to your commits:

在提交中添加认证声明：