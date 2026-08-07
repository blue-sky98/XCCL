# HCCL AllReduce (AR) 算子 — Executor 示意图与分析

本目录包含 HCCL AllReduce 四类 Executor 的可视化示意图与深度分析文档。

## Executor 分类总览

| Executor 类型 | 编排方式 | 核心特点 | 适用场景 |
|--------------|----------|----------|----------|
| **Sequence** | 多模板串行 | RS/AG 拆分为框内+框间四个模板串行执行 | 支持 DPU、通用场景 |
| **Sole** | 单模板独立 | 五种 AICPU 模板各自独立完成全量规约 | 单级拓扑、小/中数据 |
| **Concurrent** | 双模板并发 | 按带宽比例切分数据，Mesh + NHR 同时并发 | UBX 拓扑、带宽均衡 |
| **OmniPipe** | 三级流水线 | 框内/框间/跨Pod 三级流水线重叠执行 | 大规模（256 rank+） |

---

## 文档导航

### 1. Sequence Executor — 多模板串行编排

| 文档 | 说明 |
|------|------|
| [Sequence_Executor_分析.html](./Sequence_Executor_分析.html) | Sequence Executor 四模板串行分析（4×4 rank，含 DPU） |
| [Sequence_Executor_分析.md](./Sequence_Executor_分析.md) | 同上 Markdown 版本 |

### 2. Sole Executor — 单模板独立执行

路径：`Sole_Excutor/`

| 模板 | 示意图 (HTML) | 分析文档 (MD) | 算法结构 | 特点 |
|------|---------------|---------------|----------|------|
| 综合对比 | [A00-模板综合对比.html](./Sole_Excutor/A00-模板综合对比.html) | [A00-模板综合对比.md](./Sole_Excutor/A00-模板综合对比.md) | — | rank=4/N、Reduce、cclBuff 倍率根源、NHR 收发信息 |
| 01 Mesh1D One-Shot | [01-Mesh1DOneShot.html](./Sole_Excutor/01-Mesh1DOneShot.html) | [01-Mesh1DOneShot.md](./Sole_Excutor/01-Mesh1DOneShot.md) | 广播 + Reduce | 单级 Mesh、小数据、低延迟 |
| 02 Mesh1D Two-Shot | [02-Mesh1DTwoShot.html](./Sole_Excutor/02-Mesh1DTwoShot.html) | [02-Mesh1DTwoShot.md](./Sole_Excutor/02-Mesh1DTwoShot.md) | RS + AG | 单级 Mesh、大数据、省带宽 |
| 03 NHR | [03-NHR.html](./Sole_Excutor/03-NHR.html) | [03-NHR.md](./Sole_Excutor/03-NHR.md) | RS + AG（内联归约） | 任意拓扑、通用兜底、cclBuff 最省 |
| 04 Mesh1D TwoShot MeshChunk | [04-Mesh1DTwoShotMeshChunk.html](./Sole_Excutor/04-Mesh1DTwoShotMeshChunk.html) | [04-Mesh1DTwoShotMeshChunk.md](./Sole_Excutor/04-Mesh1DTwoShotMeshChunk.md) | 流水 RS + AG | 单级 Mesh(UBX)、流水线掩盖延迟 |
| 05 Aicpu Reduce NHR | [05-AicpuReduceNHR.html](./Sole_Excutor/05-AicpuReduceNHR.html) | [05-AicpuReduceNHR.md](./Sole_Excutor/05-AicpuReduceNHR.md) | AG + Reduce | 任意拓扑、特殊数据类型兜底 |

详见 [Sole_Excutor/README.md](./Sole_Excutor/README.md)

### 3. Concurrent Executor — 双模板并发编排

| 文档 | 说明 |
|------|------|
| [Concurrent_Executor_分析.html](./Concurrent_Executor_分析.html) | Concurrent Executor 双模板并发分析（4P，rank1 视角，AICPU_TS，带宽比 11:10） |
| [Concurrent_Executor_分析.md](./Concurrent_Executor_分析.md) | 同上 Markdown 版本 |

**核心设计**：将输入数据按带宽比例（Mesh:NHR = 11:10）切分，temp0（Mesh1D TwoShot）和 temp1（NHR）同时并发执行，各自完成 AllReduce 后拼接结果。总时间 ≈ max(temp0, temp1)，优于串行。

### 4. OmniPipe Executor — 三级流水线编排

| 文档 | 说明 |
|------|------|
| [OmniPipe_Excutor_分析.html](./OmniPipe_Excutor_分析.html) | OmniPipe Executor 三级流水线分析（8×8×4=256 rank） |
| [OmniPipe_Executor_分析.md](./OmniPipe_Executor_分析.md) | 同上 Markdown 版本 |

**核心设计**：256 rank 三级拓扑（Level0 框内 8rank Mesh1D AICPU / Level1 框间 8rank NHR AICPU / Level2 跨Pod 4rank DPU），6 个模板（RS-L0/L1/L2 + AG-L0/L1/L2）通过流水线重叠执行，掩盖跨级通信延迟。

---

## Executor 速查对比

| Executor | 模板数 | 执行方式 | 线程模型 | 同步开销 | cclBuff | 典型规模 |
|----------|:------:|----------|----------|:--------:|:-------:|----------|
| Sequence | 4 | 串行 | 各模板独立 | 高（模板间 barrier） | 各模板累加 | 通用 |
| Sole | 1 | 独立 | 模板内线程 | 低 | 模板内 | 单级拓扑 |
| Concurrent | 2 | 并发 | 5 线程（4+1） | 中（2 次模板间同步） | 按 22:10 划分 | UBX 4P+ |
| OmniPipe | 6 | 流水线 | 三级各独立 | 中（级间同步） | 三级累加 | 256 rank+ |

---

## 目录结构

```
AR/
├── README.md                          # 本文件
├── Sequence_Executor_分析.html         # Sequence Executor 示意图
├── Sequence_Executor_分析.md
├── Concurrent_Executor_分析.html       # Concurrent Executor 示意图
├── Concurrent_Executor_分析.md
├── OmniPipe_Excutor_分析.html          # OmniPipe Executor 示意图
├── OmniPipe_Executor_分析.md
├── assets/                            # 共享图片资源
└── Sole_Excutor/                      # Sole Executor 五种模板
    ├── README.md
    ├── 01-Mesh1DOneShot.html/.md
    ├── 02-Mesh1DTwoShot.html/.md
    ├── 03-NHR.html/.md
    ├── 04-Mesh1DTwoShotMeshChunk.html/.md
    ├── 05-AicpuReduceNHR.html/.md
    ├── A00-模板综合对比.html/.md
    └── assets/
```

---

*数据来源：HCCL 源码分析 · 仅用于个人学习研究*
