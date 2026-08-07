# HCCL 框架与算法选择逻辑

本目录包含 HCCL（华为昇腾通信库）的框架架构分析、算法选择逻辑与算子执行流程文档。

## 目录结构

```
00-frame/
├── README.md                              # 本文件（目录导航）
├── HCCL整体代码架构.md                     # HCCL 软件分层、目录结构、op_common 四件套
├── AllReduce_Selector_算法选择分析.html    # 算法选择逻辑可视化（流程图 + 算法表）
├── AllReduce_Selector_算法选择分析.md      # 算法选择逻辑全量源码分析
├── AllReduce算子执行流程详解.md            # ⭐ 执行流程整合版（推荐阅读）
└── assets/                                # 文档引用的图片资源
```

## 文档列表

### 可视化文档

| 文档 | 说明 |
|------|------|
| [AllReduce_Selector_算法选择分析.html](./AllReduce_Selector_算法选择分析.html) | AllReduce 算法选择逻辑可视化：基础选择逻辑流程图（入口分流 + 五级状态机）+ 五种模式算法选择表 + 39 种算法汇总 |

### 分析文档

| 文档 | 说明 |
|------|------|
| [HCCL整体代码架构.md](./HCCL整体代码架构.md) | HCCL 整体代码架构：软件分层（L1/L2/L3）、目录结构、op_common 四件套、通信引擎（AICPU_TS/CCU/AIV）、架构约束 |
| [AllReduce_Selector_算法选择分析.md](./AllReduce_Selector_算法选择分析.md) | AllReduce 算法选择逻辑全量源码分析：`execute_selector.cc` 入口分流 + `auto_selector_base.cc` 五级状态机 + `all_reduce_auto_selector.cc` 各模式算法选择 |
| [AllReduce算子执行流程详解.md](./AllReduce算子执行流程详解.md) | ⭐ AllReduce 算子执行流程整合版：四件套（Executor/Selector/Template/Topo）关联与调用、三级注册机制与 Map 管理、完整执行时序、Template 执行逻辑、资源复用机制 |

## 核心内容

### 1. HCCL 整体架构

- **软件分层**：L1 HCCL 集合通信算子（`cann/hccl`）→ L2 HCOMM 集合通信域管理（`cann/hcomm`）→ L3 HCOMM 基础通信
- **op_common 四件套**：`executor`（算法执行器）/ `selector`（算法选择器）/ `template`（算法模板）/ `topo`（拓扑匹配）
- **通信引擎**：AICPU_TS（大数据量，不占计算核）/ CCU（硬化加速，高带宽低时延）/ AIV（小数据低延迟，占 Vector 核）
- **架构约束**：分层依赖不可反向、控制面/数据面分离、HCCL 与 HCOMM 通过 dlsym 解耦

### 2. AllReduce 算法选择逻辑

- **入口**：`ExecuteSelector::Run`（execute_selector.cc）— `isMc2` 分流，非 MC2 走 `GetSelectorsByOpType`，按 priority 升序遍历
- **注册**：`REGISTER_SELECTOR_BY_OPTYPE` 宏，AllReduceAutoSelector priority=18
- **基类状态机**：`AutoSelectorBase::Select` — 五级分发：
  1. `CheckHostDPUOnly` → SelectDPUAlgo
  2. `opExecuteConfig == CCU_MS` → SelectCcuMsAlgo
  3. `opExecuteConfig == CCU_SCHED` → SelectCcuScheduleAlgo
  4. `ProcessAivConfig` → SelectAivAlgo
  5. `IsStarsState` → `IsRollBackAiv`（重试 AIV 或 SelectAicpuAlgo）
- **算法总数**：39 种（CCU_MS 7 + CCU_SCHED 10 + AICPU 18 + AIV 2 + DPU 2）

### 3. AllReduce 执行流程

- **注册机制**：三个独立注册表 — `SelectorRegistry`（按 priority）、`CollAlgExecRegistryV2`（按 opType+algName）、`InsAlgTemplateRegistry`（按 algName）
- **核心设计**：Executor 是 C++ 模板类，编译期绑定 TopoMatch 和 Template；Selector 运行期返回 algName 字符串，通过 algName 从 map 查找 Executor
- **执行链路**：`HcclAllReduce()` → `Selector()` 选算法 → `HcclExecOp()` 取 Executor → `CalcAlgHierarchyInfo()`（Topo）→ `CalcRes()`（Template）→ `Orchestrate()` → `Template::KernelRun()` 实际通信
- **典型注册**：`REGISTER_EXEC_V2(HCCL_CMD_ALLREDUCE, AicpuAllReduceSoleMeshOneShot, InsV2AllReduceSoleExecutor, TopoMatch1D, InsTempAllReduceMesh1DOneShot)`
