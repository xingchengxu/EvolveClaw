你是一个自主编程 AI Agent。你的任务是在当前工作目录中，从零开始创建一个名为 `evolveclaw-ramsey` 的 GitHub 项目，目标是：

1. 阅读并理解这篇论文：
   https://arxiv.org/pdf/2603.09172
2. 参考 AlphaEvolve、OpenClaw、OpenCode，以及更多自主 AI agent / code agent / harness-driven engineering 项目的思想，
3. 实现一个“教学用途的、最小可运行的” Ramsey 优化项目：
   - 用最小系统复现 AlphaEvolve 的核心思想，而不是追求完整工业级还原
   - 核心任务是：自动搜索/演化程序或策略，寻找更优的 Ramsey 图构造
   - 重点体现“agent + search + evaluation harness + iterative improvement”的闭环
4. 项目风格要体现 Harness Engineering：
   - 强调实验可复现、评估可重复、模块职责清晰、日志完整、配置明确、失败可恢复
   - 把“agent 主循环”和“评测/执行 harness”明确分层
5. 允许联网搜索必要信息，但所有网页、链接、论文、文档内容都必须通过 `curl` 获取，不要使用浏览器，不要使用 GUI，不要用其它网页抓取工具替代 `curl`

====================
一、重要参考资料
====================

你必须重点参考以下资料，并全部通过 `curl` 获取其内容或页面信息：

论文：
- https://arxiv.org/pdf/2603.09172

两个重要参考项目：
- https://github.com/algorithmicsuperintelligence/openevolve
- https://github.com/google-research/google-research/tree/master/ramsey_number_bounds

AI agent / code agent / autonomous engineering 参考对象：
- AlphaEvolve
- OpenClaw
- OpenCode
- nanobot
- A3S-Code

说明：
- 你需要把这些项目/系统作为“设计灵感参考”，而不是机械照搬
- README 中应明确说明：本项目借鉴了哪些设计思想，以及哪些地方做了教学化简化
- 若 OpenClaw / OpenCode / nanobot / A3S-Code 存在公开资料，可继续通过 `curl` 检索并获取其项目页面、README、论文、博客或说明文档
- 允许增加其它必要参考资料，但一律通过 `curl` 获取

====================
二、工作原则
====================

你必须遵守以下原则：

- 所有外部链接内容，一律使用 `curl` 获取
- 优先实现一个最小但完整闭环的系统，不要堆砌过多复杂功能
- 代码要可读、教学友好、易运行
- 项目要像一个真实 GitHub 仓库，而不是零散脚本
- 先完成最小闭环，再补充文档和工程细节
- 尽量少依赖重量级服务；默认本地运行
- 如果需要 LLM/Agent 抽象，优先设计成接口化、可替换，而不是强绑定某个专有 API
- 如无法完整复现论文中的某些隐藏实现细节，必须在 README 中明确说明“教学化近似实现”的边界
- 不要假装已经验证过无法验证的结果；要诚实记录哪些是实现、哪些是近似、哪些是 TODO

====================
三、信息获取要求
====================

第一步，必须先用 `curl` 获取并阅读论文内容：

- `curl -L https://arxiv.org/pdf/2603.09172 -o paper.pdf`

随后你可以根据需要继续用 `curl` 获取：
- arXiv abs 页面
- GitHub 项目 README
- 相关项目主页
- AlphaEvolve / OpenClaw / OpenCode / nanobot / A3S-Code / Ramsey number 背景资料
- Python 包官方文档页面
- 任何你认为必要的参考资料

要求：
- 所有联网获取都必须通过 `curl`
- 把关键参考资料整理到项目文档中
- 如有必要，保存到 `research/` 目录下
- 你应当在最终总结中列出用 `curl` 获取过的核心资料

====================
四、项目目标
====================

请在当前文件夹中创建一个项目目录：

- `evolveclaw-ramsey/`

这个仓库应当实现一个“最小 AlphaEvolve 风格 Ramsey 优化系统”，至少包含以下能力：

1. 问题定义
   - 以 Ramsey 图搜索为目标任务
   - 至少支持一个基础任务，例如尝试搜索某个 `(s, t)` Ramsey 下界实例的候选图
   - 不要求达到论文结果，但要有明确目标函数和评测标准

2. 候选表示
   - 图结构表示（例如 adjacency matrix / edge list / bitstring）
   - 或“生成候选图的程序”表示
   - 需要给出你选择这种表示的理由

3. 最小 agent/evolution 闭环
   - 生成候选
   - 运行评测
   - 根据结果保留/淘汰/变异
   - 多轮迭代
   - 输出最优结果与过程日志

4. Harness Engineering 体现
   必须显式实现或体现：
   - 可重复执行的实验入口
   - 配置文件
   - 日志记录
   - 中间结果存档
   - 评测器与 agent 解耦
   - 可以单独运行 evaluator / search / replay
   - 失败后可从 checkpoint 或历史结果恢复（最小实现即可）

5. 教学导向
   - 代码结构清晰
   - 文档中解释“AlphaEvolve 思想在这里如何被最小化实现”
   - 说明与论文原版可能差异在哪
   - 给出一个从零运行 demo

====================
五、建议实现方向
====================

你需要做的是“教学化最小实现”，不是完整工业复刻。可以采用如下思路之一，但请自行判断最合适方案并实现：

方案 A：程序演化风格（更贴近 AlphaEvolve）
- agent 维护一组“候选生成程序”或“启发式函数”
- harness 执行这些程序，生成图
- evaluator 计算违反 clique / independent set 约束的程度
- evolution loop 根据得分改写或变异程序

方案 B：策略演化风格（更容易最小落地）
- 候选不是完整程序，而是一组图构造策略参数
- harness 根据策略构图
- evaluator 评分
- agent 负责提出新参数、变异、保留优者

方案 C：混合式（推荐）
- 把“候选”定义为 Python 可调用策略对象
- strategy -> build graph -> score -> rank -> mutate
- 用最小 agent 抽象模拟 OpenClaw / OpenCode / nanobot / A3S-Code 一类系统的自主迭代感
- 既保留“程序/策略可进化”的味道，也更容易教学实现

优先目标：
- 系统完整闭环
- 结构清晰
- 能实际运行
- 能观察优化过程
- 能体现 harness

====================
六、仓库结构要求
====================

请生成一个清晰的工程目录，至少包括但不限于：

- `README.md`
- `pyproject.toml` 或 `requirements.txt`
- `evolveclaw_ramsey/`
- `configs/`
- `scripts/`
- `tests/`
- `artifacts/` 或 `runs/`
- `research/`（可选，用于保存 curl 拉取的资料、笔记、摘要）

注意：
- 不要创建 `src/evolveclaw_ramsey/`
- 相关 Python 包目录直接命名为 `evolveclaw_ramsey/`

建议目录示例（可调整，但要保留分层思想）：

evolveclaw-ramsey/
  README.md
  requirements.txt
  .gitignore
  configs/
    demo.yaml
  research/
    notes.md
    sources.md
  scripts/
    fetch_paper.sh
    fetch_refs.sh
    run_demo.sh
    run_search.sh
  evolveclaw_ramsey/
    __init__.py
    cli.py
    agent/
      loop.py
      mutator.py
      proposer.py
      memory.py
    harness/
      executor.py
      evaluator.py
      recorder.py
      checkpoint.py
    ramsey/
      graph_repr.py
      scoring.py
      constraints.py
      constructors.py
    utils/
      logging.py
      config.py
      seeds.py
  tests/
    test_scoring.py
    test_evaluator.py
    test_demo_run.py

====================
七、功能要求
====================

至少实现以下功能：

1. CLI
   提供命令行入口，例如：
   - `python -m evolveclaw_ramsey.cli run --config configs/demo.yaml`
   - `python -m evolveclaw_ramsey.cli eval ...`
   - `python -m evolveclaw_ramsey.cli replay ...`

2. 评分器
   实现最基本的 Ramsey 目标评分：
   - 给定图 G，评估其是否包含某类 clique
   - 评估其补图是否包含某类 clique（对应 independent set）
   - 或实现一个 violation score / penalty score
   - 评分逻辑必须清楚、可测试

3. Agent loop
   实现一个最小自主循环：
   - 初始化候选
   - 评估候选
   - 排序
   - 变异/生成新候选
   - 记录 best-so-far
   - 多轮运行

4. Harness
   必须有专门的 harness 层，而不是把所有逻辑糊在一个文件里
   harness 至少负责：
   - 执行候选
   - 评测
   - 记录
   - 保存中间产物
   - 恢复运行（最小支持即可）

5. 输出结果
   每次运行至少产出：
   - 配置快照
   - 每轮得分日志
   - 当前最优候选
   - 可读的 summary 文件

====================
八、README 要求
====================

README 必须写完整，至少包括：

1. 项目简介
   - 这是一个什么项目
   - 与论文的关系
   - 为什么叫 evolveclaw-ramsey

2. 核心思想
   - AlphaEvolve 的核心思想是什么
   - 本项目如何做“最小教学实现”
   - OpenClaw / OpenCode / nanobot / A3S-Code 风格在本项目里的体现是什么
   - Harness Engineering 在本项目里的体现是什么

3. 参考项目与灵感来源
   至少明确说明：
   - AlphaEvolve 论文
   - OpenEvolve
   - google-research/ramsey_number_bounds
   - OpenClaw
   - OpenCode
   - nanobot
   - A3S-Code

4. 项目边界
   - 这不是论文的完整官方复现
   - 哪些部分是教学化近似
   - 哪些部分未来可以扩展

5. 快速开始
   - 安装
   - 运行 demo
   - 查看结果

6. 仓库结构说明

7. 技术说明
   - 候选表示
   - 评分逻辑
   - evolution loop
   - harness 分层

8. 后续扩展建议
   - 接 LLM proposer
   - 更复杂 graph search
   - 更强 evaluator
   - 并行 harness
   - benchmark suite

====================
九、测试与质量要求
====================

请至少补充基础测试，保证：
- scoring 函数能工作
- evaluator 能跑
- demo 配置能启动最小实验
- 核心模块有最少量但合理的注释

不要为了“看起来高级”而省略可运行性。
不要只生成 README 或伪代码。
必须产出真正的项目文件。

====================
十、执行步骤要求
====================

请按以下节奏执行：

1. 用 `curl` 拉取论文与必要参考信息
2. 总结论文中与你实现有关的核心机制
3. 总结两个重点参考项目对本项目的启发：
   - OpenEvolve
   - google-research/ramsey_number_bounds
4. 如能获取公开资料，也总结以下 agent 参考对象的设计启发：
   - OpenClaw
   - OpenCode
   - nanobot
   - A3S-Code
5. 设计最小系统架构
6. 创建项目目录与基础文件
7. 编写核心代码
8. 补充 README、脚本、配置、测试
9. 自检项目能否运行
10. 输出最终总结

====================
十一、输出要求
====================

在完成后，你需要给出一份简洁但完整的最终说明，包含：

- 你用 `curl` 获取了哪些资料
- 你如何抽象 AlphaEvolve 的最小核心
- 你如何借鉴 OpenEvolve 与 google-research/ramsey_number_bounds
- 你如何体现 OpenClaw / OpenCode / nanobot / A3S-Code 风格
- 你如何体现 Harness Engineering
- 生成了哪些关键文件
- 运行入口是什么
- 当前实现的局限是什么

开始执行。记住：
- 外部资料获取统一使用 `curl`
- 在当前目录创建 `evolveclaw-ramsey`
- Python 包目录直接使用 `evolveclaw_ramsey/`
- 重点参考 OpenEvolve 与 google-research/ramsey_number_bounds
- 增加 OpenClaw、OpenCode、nanobot、A3S-Code 等 AI agent 参考
- 目标是“教学用途的最小可运行实现”
- 不要只停留在设计，要真正写出项目文件
