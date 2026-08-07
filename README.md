# ME_XCCL — 通信库学习笔记

本仓库为 XCCL（HCCL / NCCL）通信库的个人学习笔记，包含源码分析、可视化示意图和算法对比文档。

> **声明**：本仓库所有内容均基于开源源码整理，**仅用于个人学习与技术研究**，不用于任何商业用途。
> 参考资料：[CANN HCCL (GitCode)](https://gitcode.com/cann/hccl) · [CANN HCCL (Gitee)](https://gitee.com/ascend/cann-hccl)

## 目录结构

```
ME_XCCL/
├── index.html                    # 文档导航首页（可视化示意图索引）
├── A00-CommBase/                 # 通信基础
├── A01-HCCL/                     # HCCL（华为昇腾通信库）
│   ├── 00-frame/                 # HCCL 框架架构与算法选择逻辑
│   └── 01-Ops/
│       └── AR/                   # AllReduce 算子
│           ├── README.md         # AR 算子分析导航（四类 Executor）
│           ├── Sequence_Executor_分析.html/.md
│           ├── Concurrent_Executor_分析.html/.md
│           ├── OmniPipe_Excutor_分析.html/.md
│           └── Sole_Excutor/     # 五种 AICPU AllReduce 模板
└── A02-NCCL/                     # NCCL（NVIDIA 通信库）
```

## 文档导航

### 可视化示意图

打开 [index.html](./index.html) 查看全部可视化示意图索引，包括：

- **Sequence Executor**：多模板串行编排（框内 RS + 框间 RS + 框间 AG + 框内 AG）
- **Concurrent Executor**：双模板并发编排（Mesh1D TwoShot + NHR 按 11:10 带宽比例同时执行）
- **OmniPipe Executor**：三级流水线编排（8×8×4=256 rank，框内/框间/跨Pod 流水线重叠）
- **Sole Executor**：五种 AICPU AllReduce 模板独立示意图
  - 01 Mesh1D One-Shot
  - 02 Mesh1D Two-Shot
  - 03 NHR（非均衡层次环）
  - 04 Mesh1D TwoShot MeshChunk（流水线）
  - 05 Aicpu Reduce NHR（特殊数据类型兜底）
  - ★ 模板综合对比

### 深度分析文档

| 文档 | 说明 |
|------|------|
| [A01-HCCL/00-frame/README.md](./A01-HCCL/00-frame/README.md) | HCCL 框架与算法选择逻辑：AllReduce Selector 入口分流 + 五级状态机（DPU/CCU_MS/CCU_SCHED/AIV/AICPU）+ 39 种算法汇总 |
| [A01-HCCL/01-Ops/AR/README.md](./A01-HCCL/01-Ops/AR/README.md) | AR 通信算子分析导航：Sequence / Sole / Concurrent / OmniPipe 四类 Executor 总览、速查对比与文档索引 |
| [A01-HCCL/01-Ops/AR/Sole_Excutor/README.md](./A01-HCCL/01-Ops/AR/Sole_Excutor/README.md) | Sole Executor 五模板示意图导航、速查表与算法选择逻辑 |

### 各模块 README

- [A00-CommBase/README.md](./A00-CommBase/README.md) — 通信基础
- [A01-HCCL/README.md](./A01-HCCL/README.md) — HCCL 总览
  - [A01-HCCL/00-frame/README.md](./A01-HCCL/00-frame/README.md) — HCCL 框架架构与算法选择逻辑
  - [A01-HCCL/01-Ops/AR/README.md](./A01-HCCL/01-Ops/AR/README.md) — HCCL AllReduce 算子（四类 Executor）

- [A02-NCCL/README.md](./A02-NCCL/README.md) — NCCL

## 参考资料

- [CANN HCCL（GitCode 镜像）](https://gitcode.com/cann/hccl)
- [CANN HCCL（Gitee 官方）](https://gitee.com/ascend/cann-hccl)
