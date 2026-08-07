# HCCL AllReduce 算法选择逻辑分析（非 MC2）

*基于 `execute_selector.cc` line 41-51 + `auto_selector_base.cc` + `all_reduce_auto_selector.cc` 全量源码*

---

## 0. 文档导航

| 章节 | 内容 |
| --- | --- |
| [1. 入口分析](#1-入口分析executeselectorrund) | ExecuteSelector::Run 的 MC2/非MC2 分流 |
| [2. 注册机制](#2-selector-注册机制) | SelectorRegistry 与 REGISTER_SELECTOR_BY_OPTYPE 宏 |
| [3. 基类分发](#3-autoselectorbase基类分发逻辑) | Select() 状态机：DPU→CCU_MS→CCU_SCHED→AIV→AICPU |
| [4. 流程图](#4-完整流程图) | 从 Run 到 selectAlgName 的全链路流程图 |
| [5. 函数调用链](#5-函数调用链) | 逐层展开的调用关系 |
| [6. CCU_MS 选择结果](#6-selectccumsalgo-选择结果) | CCU 多流模式算法名 |
| [7. CCU_SCHED 选择结果](#7-selectccuschedulealgo-选择结果) | CCU 调度模式算法名 |
| [8. AICPU 选择结果](#8-selectaicpualgo-选择结果) | AICPU 模式算法名 |
| [9. AIV 选择结果](#9-selectaivalgo-选择结果) | AIV 模式算法名 |
| [10. DPU 选择结果](#10-selectdpualgo-选择结果) | Host DPU 模式算法名 |
| [11. 汇总表](#11-所有-selectalgname-汇总表) | 全量算法名一览 |
| [12. 关键常量](#12-关键常量与阈值) | 数据量阈值、rank 数限制 |

---

## 1. 入口分析（ExecuteSelector::Run）

### 1.1 源码（execute_selector.cc:19-55）

```cpp
HcclResult ExecuteSelector::Run(OpParam& opParam, TopoInfoWithNetLayerDetails* topoInfo,
                                 std::string& selectAlgName) const
{
    // ── MC2 路径 ──
    if (opParam.isMc2) {                                    // line 25
        selectors = SelectorRegistry::Global()->GetAllSelectors();  // 取全局注册表 impls_
        auto iter = selectors.find(18);                     // 找 priority=18 的 Mc2Selector
        if (iter->second->Select(...) == MATCH) return SUCCESS;
        return HCCL_E_NOT_SUPPORT;
    }

    // ── 非 MC2 路径（AllReduce 走这里）──
    selectors = SelectorRegistry::Global()->GetSelectorsByOpType(opParam.opType);  // line 41
    // → opTypeImpls_[HCCL_CMD_ALLREDUCE] = { {18, AllReduceAutoSelector} }

    for (auto iter : selectors) {                           // line 43: 按 priority 升序遍历
        // line 45: 调用 AllReduceAutoSelector::Select()
        if (iter.second->Select(opParam, topoInfo, selectAlgName) == SelectorStatus::MATCH) {
            return HCCL_SUCCESS;                            // 第一个 MATCH 即返回
        }
    }
    return HCCL_E_NOT_SUPPORT;                              // 全部不匹配
}
```

### 1.2 line 45 核心逻辑

```cpp
// execute_selector.cc:45
if (iter.second->Select(opParam, topoInfo, selectAlgName) == SelectorStatus::MATCH)
```

- `iter.second` = `AllReduceAutoSelector*`（priority=18，唯一注册的 AllReduce selector）
- `Select()` 是 `AutoSelectorBase` 基类的**非虚**入口方法（`auto_selector_base.cc:17-68`）
- 它按 `opParam.opExecuteConfig` 状态机分发到各子类 override 的虚方法

### 1.3 遍历顺序

`std::map<u32, AutoSelectorBase*>` 按 key 升序排列，**priority 数字越小越先执行**。AllReduce 仅注册了 priority=18 的一个 selector，无竞争。

---

## 2. Selector 注册机制

### 2.1 注册宏

```cpp
// selector_registry.h
#define REGISTER_SELECTOR_BY_OPTYPE(optype, priority, selector) \
    static HcclResult g_func_##priority##_##selector##_##ctr \
        = SelectorRegistry::Global()->RegisterByOpType(optype, priority, new selector())
```

### 2.2 AllReduce 注册

```cpp
// all_reduce_auto_selector.cc:724
REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_ALLREDUCE, 18, AllReduceAutoSelector);
```

- 注册到 `opTypeImpls_[HCCL_CMD_ALLREDUCE][18] = new AllReduceAutoSelector()`
- 利用 `__COUNTER__` 宏生成唯一静态变量名，在 `main()` 前完成注册

### 2.3 存储结构

```cpp
// selector_registry.h
class SelectorRegistry {
    std::map<u32, AutoSelectorBase*> impls_;                              // MC2 全局注册表
    std::map<HcclCMDType, std::map<u32, AutoSelectorBase*>> opTypeImpls_; // 非MC2 按opType分组
};
```

---

## 3. AutoSelectorBase 基类分发逻辑

### 3.1 Select() 状态机（auto_selector_base.cc:17-68）

```cpp
SelectorStatus AutoSelectorBase::Select(OpParam& opParam, TopoInfoWithNetLayerDetails* topoInfo,
                                         std::string& selectAlgName) const
{
    // ── 第0级：Host DPU Only 检查 ──
    if (CheckHostDPUOnly(...) && hostDPUOnly) {             // line 24
        opParam.opExecuteConfig = OpExecuteConfig::HOSTCPU;
        opParam.engine = CommEngine::COMM_ENGINE_CPU;
        return SelectDPUAlgo(...);                          // → AllReduceAutoSelector::SelectDPUAlgo
    }

    // ── 第1级：CCU_MS（多流）──
    if (opParam.opExecuteConfig == OpExecuteConfig::CCU_MS) {          // line 29
        ret = SelectCcuMsAlgo(...);                          // → AllReduceAutoSelector::SelectCcuMsAlgo
        if (ret == NOT_MATCH) opParam.opExecuteConfig = CCU_SCHED;     // 降级
        else return ret;                                     // MATCH 直接返回
    }

    // ── 第2级：CCU_SCHED（调度）──
    if (opParam.opExecuteConfig == OpExecuteConfig::CCU_SCHED) {       // line 37
        ret = SelectCcuScheduleAlgo(...);                    // → AllReduceAutoSelector::SelectCcuScheduleAlgo
        if (ret == NOT_MATCH) opParam.opExecuteConfig = CCU_FAIL;      // 降级
        else return ret;
    }

    // ── 第3级：AIV ──
    if (ProcessAivConfig(...)) {                             // line 45: 检查 AIV / AIV_ONLY
        return ret;                                          // → AllReduceAutoSelector::SelectAivAlgo
    }

    // ── 第4级：AICPU（STARS 状态）──
    if (IsStarsState(opParam.opExecuteConfig)) {             // line 48
        // 检查是否需要回退 AIV
        if (IsRollBackAiv(opParam, topoInfo)) {              // line 50
            opParam.opExecuteConfig = AIV_ONLY;
            ProcessAivConfig(...);                           // 再试一次 AIV
            return ret;
        }
        ret = SelectAicpuAlgo(...);                          // → AllReduceAutoSelector::SelectAicpuAlgo
        if (ret == MATCH) opParam.opExecuteConfig = AICPU_TS;
    }

    return ret;
}
```

### 3.2 分发优先级链

```
HostDPUOnly? ──Yes──► SelectDPUAlgo()
     │No
CCU_MS? ──Yes──► SelectCcuMsAlgo() ──MATCH──► 返回
     │              └──NOT_MATCH──► 降级到 CCU_SCHED
CCU_SCHED? ──Yes──► SelectCcuScheduleAlgo() ──MATCH──► 返回
     │                 └──NOT_MATCH──► 降级到 CCU_FAIL
AIV / AIV_ONLY? ──Yes──► SelectAivAlgo() ──MATCH──► 返回
     │                       └──NOT_MATCH──► AIV→CCU_FAIL, AIV_ONLY→返回NOT_MATCH
STARS状态? ──Yes──► IsRollBackAiv? ──Yes──► 重试 AIV
     │                                └──No──► SelectAicpuAlgo() ──MATCH──► 返回
返回 ret
```

### 3.3 STARS 状态判断

```cpp
// auto_selector_base.cc:82-87
bool IsStarsState(const OpExecuteConfig& opExecuteConfig) const {
    return opExecuteConfig == AICPU_TS || opExecuteConfig == HOSTCPU_TS || opExecuteConfig == CCU_FAIL;
}
```

### 3.4 AIV 回退条件

```cpp
// auto_selector_base.cc:70-80
bool IsRollBackAiv(OpParam& opParam, TopoInfoWithNetLayerDetails* topoInfo) const {
    bool isAllToAllOps = opParam.opType == ALLTOALL || ALLTOALLV || ALLTOALLVC;
    bool isInt64ReduceOps = dataType == INT64 && (opType == ALLREDUCE || REDUCE_SCATTER || REDUCE);
    return topoInfo->level0PcieMix && topoInfo->level0BigClosRange && (isAllToAllOps || isInt64ReduceOps);
}
```

---

## 4. 完整流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ExecuteSelector::Run()                               │
│                  (execute_selector.cc:19-55)                            │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  opParam.isMc2 ?    │
                    └──────┬───────┬──────┘
                     Yes   │       │  No (AllReduce 非 MC2)
                           │       │
           ┌───────────────┘       └────────────────┐
           ▼                                         ▼
  GetAllSelectors()                        GetSelectorsByOpType(ALLREDUCE)
  find(18) → Mc2Selector                  → {18: AllReduceAutoSelector}
           │                                         │
           ▼                                         ▼
  Mc2Selector::Select()               for (iter : selectors) {
  (本文档不展开)                        │
                                        ▼
                              ┌─────────────────────┐
                              │ iter.second->Select()│  ← line 45
                              │ = AllReduceAuto     │
                              │   Selector::Select()│
                              │ (基类非虚入口)       │
                              └─────────┬───────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │  AutoSelectorBase::Select() │
                          │  (auto_selector_base.cc:17) │
                          └─────────────┬──────────────┘
                                        │
                           ┌────────────▼────────────┐
                           │ CheckHostDPUOnly()?     │
                           └────┬──────────┬─────────┘
                            Yes │          │ No
                     ┌──────────┘          └──────────┐
                     ▼                                 ▼
          ┌──────────────────┐          ┌──────────────────────┐
          │ SelectDPUAlgo()  │          │ opExecuteConfig ==   │
          │ (line 678)       │          │ CCU_MS ?             │
          └────────┬─────────┘          └──┬──────────┬────────┘
                   │                       │Yes       │No
                   ▼                       ▼          ▼
              返回结果          ┌──────────────┐  ┌──────────────────┐
                                │SelectCcuMs-  │  │CCU_SCHED ?       │
                                │Algo()(line38)│  └──┬──────────┬────┘
                                └──┬───────┬───┘    │Yes       │No
                              MATCH│       │NOT    ▼          ▼
                                   │    MATCH│ ┌──────────┐ ┌─────────────────┐
                              ┌────┘        │ │SelectCcu-│ │ProcessAivConfig │
                              ▼             ▼ │Schedule- │ │(line 349)       │
                           返回结果    降级到   │Algo()    │ └──┬──────────┬───┘
                                       CCU_   │(line166) │  AIV/AIV_  │    │
                                       SCHED  └──┬───┬───┘  ONLY      │    │No
                                            MATCH│   │NOT          ▼    ▼
                                                │  MATCH│    ┌──────────┐ ┌──────────────┐
                                           ┌────┘      │    │SelectAiv-│ │IsStarsState? │
                                           ▼           ▼    │Algo()    │ └─┬──────────┬─┘
                                        返回结果   降级到   │(line584) │ Yes│          │No
                                                  CCU_    └──┬────┬───┘    │          │
                                                  FAIL    MATCH│   │NOT     ▼          ▼
                                                                │  MATCH│ ┌────────────┐ 返回
                                                                ▼      │ │IsRollBack- │ NOT_
                                                           返回结果   │ │Aiv()?      │ MATCH
                                                                       │ └─┬──────┬──┘
                                                                       │Yes│      │No
                                                                       ▼        ▼
                                                                  重试AIV   ┌──────────┐
                                                                            │SelectAi- │
                                                                            │cpuAlgo() │
                                                                            │(line401) │
                                                                            └────┬─────┘
                                                                           MATCH│
                                                                                ▼
                                                                           返回结果
                              }
```

---

## 5. 函数调用链

### 5.1 总调用链

```
ExecuteSelector::Run()                          [execute_selector.cc:20]
  └── SelectorRegistry::GetSelectorsByOpType()  [selector_registry.cc]
       └── 返回 {18: AllReduceAutoSelector}
  └── AllReduceAutoSelector::Select()           [继承自 AutoSelectorBase]
       = AutoSelectorBase::Select()             [auto_selector_base.cc:17]  ← 非虚入口
            │
            ├── CheckHostDPUOnly()              [auto_selector_base.cc:24]
            │    └── SelectDPUAlgo()            [all_reduce_auto_selector.cc:678]  ← override
            │
            ├── SelectCcuMsAlgo()               [all_reduce_auto_selector.cc:38]   ← override
            │    └── SelectMeshAlgo()           [all_reduce_auto_selector.cc:117]
            │         └── SelectMeshUBXAlgo()   [all_reduce_auto_selector.cc:77]
            │
            ├── SelectCcuScheduleAlgo()         [all_reduce_auto_selector.cc:166]  ← override
            │    └── SelectCcuScheduleLevel0Algo()        [all_reduce_auto_selector.cc:350]
            │         ├── SelectCcuScheduleLevel0AlgoMesh1D() [all_reduce_auto_selector.cc:312]
            │         └── SelectCcuScheduleLevel0UBXAlgo()   [all_reduce_auto_selector.cc:275]
            │
            ├── ProcessAivConfig()              [auto_selector_base.cc:349]
            │    └── SelectAivAlgo()            [all_reduce_auto_selector.cc:584]  ← override
            │
            └── SelectAicpuAlgo()               [all_reduce_auto_selector.cc:401]  ← override
                 ├── SelectMeshAlgoAicpu()       [all_reduce_auto_selector.cc:510]
                 │    └── SelectMeshAlgoAicpuUBX() [all_reduce_auto_selector.cc:473]
                 └── (保序模式直接选择)
```

### 5.2 AllReduceAutoSelector 类方法表

| 方法 | 行号 | override | 说明 |
| --- | --- | --- | --- |
| `SelectCcuMsAlgo` | 38 | 是 | CCU 多流模式算法选择 |
| `SelectMeshUBXAlgo` | 77 | 否 | UBX 机型 mesh 算法（被 SelectMeshAlgo 调用） |
| `SelectMeshAlgo` | 117 | 否 | Mesh 拓扑算法（被 SelectCcuMsAlgo 调用） |
| `SelectCcuScheduleAlgo` | 166 | 是 | CCU 调度模式算法选择 |
| `SelectCcuScheduleLevel0UBXAlgo` | 275 | 否 | UBX 机型 Level0 CCU 调度算法 |
| `SelectCcuScheduleLevel0AlgoMesh1D` | 312 | 否 | Mesh1D 拓扑 CCU 调度算法 |
| `SelectCcuScheduleLevel0Algo` | 350 | 否 | 单级拓扑 CCU 调度算法 |
| `SelectAicpuAlgo` | 401 | 是 | AICPU 模式算法选择 |
| `SelectMeshAlgoAicpuUBX` | 473 | 否 | UBX 机型 AICPU mesh 算法 |
| `SelectMeshAlgoAicpu` | 510 | 否 | 单级拓扑 AICPU mesh 算法 |
| `SelectAivAlgo` | 584 | 是 | AIV 模式算法选择 |
| `SelectDPUAlgo` | 678 | 是 | Host DPU 模式算法选择 |

---

## 6. SelectCcuMsAlgo 选择结果

### 6.1 前置排除条件

| 条件 | 行号 | 排除原因 |
| --- | --- | --- |
| 保序模式 `IsNeedStrictModeForOrderPreserved` | 46 | 不支持 CCU_MS |
| `topoLevelNums > 1`（多级拓扑） | 51 | CCU_MS 仅支持单级 |
| `INT8` 数据类型 | 57 | MS 不支持 INT8 |
| `PROD` reduce 操作 | 64 | MS 不支持 PROD |
| `INT64/UINT64/FP64` | 69 | MS 不支持 64 位类型 |

### 6.2 选择结果（通过 SelectMeshAlgo）

#### MESH_1D 拓扑（line 130-151）

| 条件 | selectAlgName | 说明 |
| --- | --- | --- |
| inplace（输入输出重叠） | NOT_MATCH | 不支持 inplace |
| TWO_DIE_REGULAR + 小数据(<512KB) | `CcuAllReduceMesh2Die` | 双 Die 规则 mesh，OneShot |
| TWO_DIE_REGULAR + 大数据(≥512KB) | `CcuAllreduceMesh2DieBigMs` | 双 Die 规则 mesh，大数据 |
| TWO_DIE_NOT_REGULAR | NOT_MATCH | 不支持 |
| 普通 + 小数据(<512KB) | `CcuAllReduceMesh1DOneShot` | 单步 mesh |
| 普通 + 大数据 + 960 + >16M + 两级网络 | `CcuAllReduceSoleMeshMsConcur` | 960 专用并发 |
| 普通 + 大数据(其他) | `CcuAllReduceMesh1D` | 标准 mesh 两步 |
| 2P + 两级网络 + ≥8M | NOT_MATCH | 回退 AICPU |

#### MESH_1D_CLOS 拓扑 → SelectMeshUBXAlgo（line 77-115）

| 条件 | selectAlgName | 说明 |
| --- | --- | --- |
| inplace | NOT_MATCH | 不支持 inplace |
| meshNum==closNum + ≤4P + 小数据 | `CcuAllReduceMesh1DOneShot` | 4P mesh 单步 |
| meshNum==closNum + ≤4P + 大数据 | `CcuAllReduceConcurrentMs` | mesh+clos 并发 |
| closNum是meshNum倍数 + 大数据 + ≥32M | `CcuV2AllReduceOmniPipe2DMs` | OmniPipe 2D |
| closNum是meshNum倍数 + 大数据 + <32M | NOT_MATCH | 回退 |
| ≤8P（其他） | `CcuAllReduceMesh1D` | 标准 mesh |
| 其他 | NOT_MATCH | |

---

## 7. SelectCcuScheduleAlgo 选择结果

### 7.1 前置排除条件

| 条件 | 行号 | 排除原因 |
| --- | --- | --- |
| `level2Ubg` | 174 | 不支持 UBG |
| `topoLevelNums >= 3` | 180 | 仅支持 ≤2 级 |
| 保序模式 | 189 | 不支持 |
| `PROD` | 195 | 不支持 |
| `INT64/UINT64/FP64` | 201 | 不支持 |

### 7.2 多级拓扑（topoLevelNums > 1，line 208-267）

#### MESH_1D

| 条件 | selectAlgName | 说明 |
| --- | --- | --- |
| inplace | NOT_MATCH | |
| Level1Nhr | `CcuAllReduceNHR1D` | GCD==1，NHR 通信 |
| localNetInsSize[0]==1 | `CcuAllReduceNHR1D` | 单实例 |
| 2DieFullMesh | NOT_MATCH | |
| >64P + ≤32M | `CcuAllReduceSequenceMesh1D` | 大规模序列 |
| ≥64P + ≤16M + 非8bit | `CcuAllReduceSequenceMesh1D` | |
| ≤64P + ≤512K + 非8bit + 非inplace | `CcuAllReduceMesh1DMem2Mem` | Mem2Mem |
| <64P + ≤64M + 非8bit | `CcuAllReduceSequenceMesh1D` | |
| ≤64M(IsSmallDataCCU) + 非INT8 | `CcuAllReduceParallelMesh1DNHR` | 并行 mesh+NHR |
| >64M | NOT_MATCH | 回退 AICPU |

#### CLOS

| 条件 | selectAlgName |
| --- | --- |
| 非inplace + <8M | `CcuAllReduceNHR1D` |
| ≥8M | NOT_MATCH |

### 7.3 单级拓扑 → SelectCcuScheduleLevel0Algo（line 350-399）

#### MESH_1D → SelectCcuScheduleLevel0AlgoMesh1D（line 312-348）

| 条件 | selectAlgName | 说明 |
| --- | --- | --- |
| INT8 | NOT_MATCH | |
| TWO_DIE_REGULAR + 小数据 | `CcuAllReduceMesh1DMem2Mem2DieOneShot` | 双 Die OneShot |
| TWO_DIE_REGULAR + 大数据 | `CcuAllreduceMesh2DieBigSche` | 双 Die 大数据 |
| TWO_DIE_NOT_REGULAR | NOT_MATCH | |
| 普通 (dataSize×ratio ≤ 8M) | `CcuAllReduceMesh1DMem2Mem` | 标准 Mem2Mem |
| dataSize×ratio > 8M | NOT_MATCH | |

#### MESH_1D_CLOS + pcieMix

| 条件 | selectAlgName |
| --- | --- |
| 全连 MESH_1D | → 同 SelectCcuScheduleLevel0AlgoMesh1D |
| 非全连 | NOT_MATCH |

#### MESH_1D_CLOS + 非pcieMix → SelectCcuScheduleLevel0UBXAlgo（line 275-310）

| 条件 | selectAlgName | 说明 |
| --- | --- | --- |
| INT8 | NOT_MATCH | |
| meshNum==closNum + ≤4P + 小数据 | `CcuAllReduceMesh1DMem2Mem` | |
| meshNum==closNum + ≤4P + 大数据 | `CcuAllReduceConcurrentSche` | mesh+clos 并发 |
| closNum倍数 + 大数据 + <64M | `CcuAllReduceParallelNHR1DMutiJetty` | 并行 NHR 多 Jetty |
| closNum倍数 + 大数据 + ≥64M | `CcuV2AllReduceOmniPipe2D` | OmniPipe 2D |
| 其他 | `CcuAllReduceNHR1DMem2MemMultiJetty` | NHR Mem2Mem |

#### CLOS + 非pcieMix

| 条件 | selectAlgName |
| --- | --- |
| <8M | `CcuAllReduceNHR1D` |
| ≥8M | NOT_MATCH |

---

## 8. SelectAicpuAlgo 选择结果

### 8.1 保序模式（line 409-422）

| 条件 | selectAlgName |
| --- | --- |
| rankSize > MAX_RANK_NUM_FOR_ORDER_PRESERVED | `AllReduceOrderPreservedGroup` |
| rankSize ≤ MAX_RANK_NUM_FOR_ORDER_PRESERVED | `AllReduceOrderPreserved` |

### 8.2 多级拓扑（topoLevelNums > 1，line 429-464）

| 条件 | selectAlgName | 说明 |
| --- | --- | --- |
| INT64/UINT64/FP64/PROD | `InsAllReduceAicpuReduceNHR` | 特殊类型走 NHR |
| 3级 + level2Uboe + 对称 + 8卡/module | `InsV2AllReduceOmniPipeUboe` | UBOE OmniPipe |
| 3级 + level2Uboe + 非对称或 localNetInsSize[1]==1 | `InsAllReduceNHR` | |
| 3级 + level2Uboe + 其他 | `InsAllReduceParallelRSAGUboe` | UBOE 并行 RSAG |
| Level1Nhr | `InsAllReduceNHR` | |
| localNetInsSize[0]==1 | `InsAllReduceNHR` | |
| MESH_1D + 3级 | `InsV2AllReduceSequenceMesh1DNHRNHR` | 三级序列 |
| MESH_1D + 2级 + >32M + >4G | `InsAllReduceSequenceMesh1DNhr` | 超大序列 |
| MESH_1D + 2级 + >32M + ≤4G | `InsAllReduceParallelRSAG` | 并行 RSAG |
| MESH_1D + 2级 + ≤32M | `InsAllReduceNHR` | |
| MESH_1D + 其他 | `InsAllReduceNHR` | |
| CLOS | `AicpuAllReduceSoleNHRTwoShotMultiLink` | CLOS NHR TwoShot |

### 8.3 单级拓扑 → SelectMeshAlgoAicpu（line 510-582）

#### MESH_1D

| 条件 | selectAlgName | 说明 |
| --- | --- | --- |
| INT64/UINT64/FP64/PROD + ≤8M | `InsAllReduceMesh1DOneShot` | 特殊类型 OneShot |
| INT64/UINT64/FP64/PROD + >8M | `InsAllReduceMesh1DTwoShot` | 特殊类型 TwoShot |
| 2P + 两级 + ≥8M | `InsAllReduceMesh1DTwoShotZAxisDetour` | 2P Z轴绕行 |
| ≤8M | `InsAllReduceMesh1DOneShot` | 小数据 OneShot |
| dataSize×ratio >32M + 两级 + >4G | `InsAllReduceMesh1DTwoShotZAxisDetour` | 超大 Z轴绕行 |
| dataSize×ratio >32M | `InsAllReduceMesh1DTwoShotMeshChunk` | 大数据分块 |
| 其他 | `InsAllReduceMesh1DTwoShot` | 标准 TwoShot |

#### CLOS

| 条件 | selectAlgName |
| --- | --- |
| INT64/UINT64/FP64/PROD | `InsAllReduceAicpuReduceNHR` |
| 普通 | `AicpuAllReduceSoleNHRTwoShotMultiLink` |

#### MESH_1D_CLOS + pcieMix

| 条件 | selectAlgName |
| --- | --- |
| 全连 MESH_1D | → 同 MESH_1D 逻辑 |
| 非全连 + INT64/UINT64/FP64/PROD | `InsAllReduceAicpuReduceNHR` |
| 非全连 + <32M | `InsAllReduceParallelMesh1DNHRPcie` |
| 非全连 + ≥32M | `InsV2AllReduceOmniPipePcie` |

#### MESH_1D_CLOS + 非pcieMix → SelectMeshAlgoAicpuUBX（line 473-508）

| 条件 | selectAlgName |
| --- | --- |
| meshNum==closNum + ≤4P + 特殊类型 + ≤8M | `InsAllReduceMesh1DOneShot` |
| meshNum==closNum + ≤4P + 特殊类型 + >8M | `InsAllReduceMesh1DTwoShot` |
| meshNum==closNum + ≤4P + ≤8M | `InsAllReduceMesh1DOneShot` |
| meshNum==closNum + ≤4P + >8M | `InsAllReduceConcurrent` |
| 特殊类型 | `InsAllReduceAicpuReduceNHR` |
| closNum倍数 + 非小数据 | `InsAllReduceParallelRSAGUBX` |
| 其他 | `InsAllReduceNHR` |

---

## 9. SelectAivAlgo 选择结果

### 9.1 前置排除条件

| 条件 | 行号 | 排除原因 |
| --- | --- | --- |
| `level2Ubg` | 591 | 不支持 |
| `topoLevelNums >= 3` | 598 | 仅支持 ≤2 级 |
| 保序模式 | 607 | 不支持 |
| `PROD` | 615 | 不支持 |
| `UINT64/FP64` | 622 | 不支持 |
| `rankSize > MAX_RANK_SIZE` | 628 | 超出最大 rank |
| 非 AIV_ONLY + dataSize ≥ 8M×rankSize | 643 | 数据量过大 |
| dataSize > cclBufferSize × AIV_MAX_CCL_LOOP_NUM | 650 | 超 ccl buffer 容量 |

### 9.2 选择结果（line 657-672）

| 条件 | selectAlgName | 说明 |
| --- | --- | --- |
| 非 MESH_1D | `AivAllReduceMesh1DTwoShot` | 非 mesh 拓扑统一 TwoShot |
| MESH_1D + ≤8P + <128K | `AivAllReduceMesh1DOneShot` | 板内小数据 OneShot |
| MESH_1D + ≤8P + ≥128K | `AivAllReduceMesh1DTwoShot` | 板内大数据 TwoShot |
| MESH_1D + >8P + 小数据(<512KB) | `AivAllReduceMesh1DOneShot` | 跨板小数据 |
| MESH_1D + >8P + 大数据 | `AivAllReduceMesh1DTwoShot` | 跨板大数据 |

---

## 10. SelectDPUAlgo 选择结果

### 10.1 前置排除

| 条件 | 行号 | 排除原因 |
| --- | --- | --- |
| INT64/UINT64/FP64/PROD | 696 | 不支持 |
| 单级拓扑（topoLevelNums ≤ 1） | 721 | DPU 仅多级 |

### 10.2 选择结果（line 699-718）

| 条件 | selectAlgName | 说明 |
| --- | --- | --- |
| 多级 + deviceNumPerModule==1 | `InsAllReduceSequenceMeshNhrDPU` | 序列 mesh NHR DPU |
| 多级 + MESH_1D | `InsAllReduceSequenceMeshNhrDPU` | |
| 多级 + MESH_1D_CLOS + 非pcieMix | `InsV2AllReduceOmniPipe` | OmniPipe |
| 多级 + MESH_1D_CLOS + pcieMix | `InsAllReduceSequenceMeshNhrDPU` | |
| 多级 + 其他 | `InsAllReduceSequenceMeshNhrDPU` | 默认 |

---

## 11. 所有 selectAlgName 汇总表

### 11.1 CCU_MS 模式

| # | selectAlgName | 适用场景 |
| --- | --- | --- |
| 1 | `CcuAllReduceMesh2Die` | MESH_1D + TWO_DIE_REGULAR + 小数据 |
| 2 | `CcuAllreduceMesh2DieBigMs` | MESH_1D + TWO_DIE_REGULAR + 大数据 |
| 3 | `CcuAllReduceMesh1DOneShot` | MESH_1D 小数据 / UBX 4P 小数据 |
| 4 | `CcuAllReduceSoleMeshMsConcur` | 960 + >16M + 两级网络 |
| 5 | `CcuAllReduceMesh1D` | MESH_1D 大数据 / UBX ≤8P |
| 6 | `CcuAllReduceConcurrentMs` | UBX meshNum==closNum + ≤4P + 大数据 |
| 7 | `CcuV2AllReduceOmniPipe2DMs` | UBX closNum倍数 + ≥32M |

### 11.2 CCU_SCHED 模式

| # | selectAlgName | 适用场景 |
| --- | --- | --- |
| 8 | `CcuAllReduceNHR1D` | 多级 Level1Nhr / CLOS 小数据 |
| 9 | `CcuAllReduceSequenceMesh1D` | 多级大规模 |
| 10 | `CcuAllReduceMesh1DMem2Mem` | 多级/单级小数据 |
| 11 | `CcuAllReduceParallelMesh1DNHR` | 多级 ≤64M + 非INT8 |
| 12 | `CcuAllReduceMesh1DMem2Mem2DieOneShot` | 单级 TWO_DIE_REGULAR + 小数据 |
| 13 | `CcuAllreduceMesh2DieBigSche` | 单级 TWO_DIE_REGULAR + 大数据 |
| 14 | `CcuAllReduceConcurrentSche` | UBX ≤4P + 大数据 |
| 15 | `CcuAllReduceParallelNHR1DMutiJetty` | UBX closNum倍数 + <64M |
| 16 | `CcuV2AllReduceOmniPipe2D` | UBX closNum倍数 + ≥64M |
| 17 | `CcuAllReduceNHR1DMem2MemMultiJetty` | UBX 其他 |

### 11.3 AICPU 模式

| # | selectAlgName | 适用场景 |
| --- | --- | --- |
| 18 | `AllReduceOrderPreserved` | 保序模式 + 小 rankSize |
| 19 | `AllReduceOrderPreservedGroup` | 保序模式 + 大 rankSize |
| 20 | `InsAllReduceAicpuReduceNHR` | 特殊类型(INT64/FP64/PROD) |
| 21 | `InsV2AllReduceOmniPipeUboe` | 3级 + UBOE + 对称 + 8卡/module |
| 22 | `InsAllReduceParallelRSAGUboe` | 3级 + UBOE + 其他 |
| 23 | `InsAllReduceNHR` | 多级 NHR / 默认 |
| 24 | `InsV2AllReduceSequenceMesh1DNHRNHR` | 3级 MESH_1D |
| 25 | `InsAllReduceSequenceMesh1DNhr` | 2级 + 超大(>4G) |
| 26 | `InsAllReduceParallelRSAG` | 2级 + 大数据 |
| 27 | `AicpuAllReduceSoleNHRTwoShotMultiLink` | CLOS |
| 28 | `InsAllReduceMesh1DOneShot` | 单级 MESH_1D 小数据 |
| 29 | `InsAllReduceMesh1DTwoShot` | 单级 MESH_1D 中等 |
| 30 | `InsAllReduceMesh1DTwoShotZAxisDetour` | 2P/超大 Z轴绕行 |
| 31 | `InsAllReduceMesh1DTwoShotMeshChunk` | 单级大数据分块 |
| 32 | `InsAllReduceConcurrent` | UBX ≤4P + 大数据 |
| 33 | `InsAllReduceParallelMesh1DNHRPcie` | pcieMix + <32M |
| 34 | `InsV2AllReduceOmniPipePcie` | pcieMix + ≥32M |
| 35 | `InsAllReduceParallelRSAGUBX` | UBX closNum倍数 + 大数据 |

### 11.4 AIV 模式

| # | selectAlgName | 适用场景 |
| --- | --- | --- |
| 36 | `AivAllReduceMesh1DOneShot` | MESH_1D 小数据 |
| 37 | `AivAllReduceMesh1DTwoShot` | MESH_1D 大数据 / 非 MESH_1D |

### 11.5 DPU 模式

| # | selectAlgName | 适用场景 |
| --- | --- | --- |
| 38 | `InsAllReduceSequenceMeshNhrDPU` | 多级默认 DPU |
| 39 | `InsV2AllReduceOmniPipe` | 多级 MESH_1D_CLOS + 非pcieMix |

---

## 12. 关键常量与阈值

### 12.1 数据量阈值（all_reduce_auto_selector.cc:18-37）

| 常量 | 值 | 用途 |
| --- | --- | --- |
| `RS_MAX_DATA_SIZE` | 16MB | CCU_SCHED 序列算法上限 |
| `AR_ONESHOT_1D_MAX_DATA_SIZE` | 16KB | OneShot 数据阈值 |
| `AR_M2M_1D_MAX_DATA_SIZE` | 8MB | Mem2Mem 数据上限 |
| `AR_AICPU_1D_SMALL_DATA_SIZE` | 8MB | AICPU 小数据阈值 |
| `AR_AICPU_1D_MAX_DATA_SIZE` | 32MB | AICPU 大数据阈值 |
| `AR_AICPU_1D_CROSS_SMALL_DATA_SIZE` | 32MB | 跨级小数据阈值 |
| `AR_AICPU_1D_64DATATYPE_DATA_SIZE` | 8MB | 64位类型阈值 |
| `AR_FLATTEN_MAX_DATA_SIZE` | 512KB | CCU Mem2Mem 阈值 |
| `AR_CCU_CLOS_1D_SMALL_DATA_SIZE` | 8MB | CCU CLOS 小数据 |
| `AR_AICPU_SEQUENCE_DATA_SIZE` | 4GB | AICPU 序列算法阈值 |
| `OMNI_PCIE_AR_DATA_SIZE` | 32MB | OmniPipe PCIe 阈值 |
| `OMNI_UBX_AR_SCHED_DATA_SIZE` | 64MB | OmniPipe UBX Sched 阈值 |
| `OMNI_UBX_AR_MS_DATA_SIZE` | 32MB | OmniPipe UBX MS 阈值 |
| `AR_AIV_SMALL_DATA_SIZE_IN_BOARD` | 128KB | AIV 板内小数据阈值 |
| `AR_2P_DETOUR_DATA_SIZE` | 8MB | 2P Z轴绕行阈值 |
| `AR_MORE_64P_SEQ_MAX_DATA_SIZE` | 32MB | >64P 序列算法上限 |

### 12.2 Rank 数限制

| 常量 | 值 | 用途 |
| --- | --- | --- |
| `MAX_RANK_NUM_FOR_CONCURRENT_ALGO` | 4 | 并发算法最大 rank |
| `MAX_RANK_NUM_FOR_REDUCE_MS_ALGO` | 8 | MS reduce 最大 rank |
| `AR_AIV_BOARD_SIZE` | 8 | AIV 板内 rank |
| `DEVICE_NUM_PER_MODULE_8` | 8 | 每模块 8 卡 |

### 12.3 基类阈值（auto_selector_base.cc）

| 常量 | 值 | 用途 |
| --- | --- | --- |
| `SMALL_COUNT_512KB` | 512KB | `IsSmallData()` 阈值 |
| `LARGE_COUNT_1024KB` | 1MB | `IsLargeData()` 阈值 |
| `CCU_PARALLEL_MAX_DATA_SIZE` | 64MB | `IsSmallDataCCU()` 阈值 |
| `DEFAULT_RANK_SIZE` | 8 | 默认 rank 数 |
| `SMALL_COUNT_16M` | 16MB | 960 大数据阈值 |

---

## 13. 关键代码位置索引

| 功能 | 文件 | 行号 |
| --- | --- | --- |
| ExecuteSelector::Run | execute_selector.cc | 19-55 |
| MC2 分流（line 25） | execute_selector.cc | 25-39 |
| 非 MC2 遍历（line 41-51） | execute_selector.cc | 41-51 |
| **line 45 核心调用** | execute_selector.cc | **45** |
| AutoSelectorBase::Select | auto_selector_base.cc | 17-68 |
| HostDPUOnly 检查 | auto_selector_base.cc | 24-28 |
| CCU_MS 分发 | auto_selector_base.cc | 29-36 |
| CCU_SCHED 分发 | auto_selector_base.cc | 37-44 |
| AIV 分发 (ProcessAivConfig) | auto_selector_base.cc | 45-47, 349-368 |
| STARS/AICPU 分发 | auto_selector_base.cc | 48-63 |
| IsRollBackAiv | auto_selector_base.cc | 70-80 |
| IsStarsState | auto_selector_base.cc | 82-87 |
| IsSmallData / IsLargeData | auto_selector_base.cc | 94-96 |
| IsSmallDataCCU | auto_selector_base.cc | 98-104 |
| SelectorRegistry 定义 | selector_registry.h | 全文 |
| REGISTER_SELECTOR_BY_OPTYPE 宏 | selector_registry.h | 宏定义 |
| GetSelectorsByOpType | selector_registry.cc | 对应方法 |
| AllReduce 注册 | all_reduce_auto_selector.cc | 724 |
| SelectCcuMsAlgo | all_reduce_auto_selector.cc | 38-75 |
| SelectMeshAlgo | all_reduce_auto_selector.cc | 117-164 |
| SelectMeshUBXAlgo | all_reduce_auto_selector.cc | 77-115 |
| SelectCcuScheduleAlgo | all_reduce_auto_selector.cc | 166-273 |
| SelectCcuScheduleLevel0Algo | all_reduce_auto_selector.cc | 350-399 |
| SelectCcuScheduleLevel0AlgoMesh1D | all_reduce_auto_selector.cc | 312-348 |
| SelectCcuScheduleLevel0UBXAlgo | all_reduce_auto_selector.cc | 275-310 |
| SelectAicpuAlgo | all_reduce_auto_selector.cc | 401-471 |
| SelectMeshAlgoAicpu | all_reduce_auto_selector.cc | 510-582 |
| SelectMeshAlgoAicpuUBX | all_reduce_auto_selector.cc | 473-508 |
| SelectAivAlgo | all_reduce_auto_selector.cc | 584-676 |
| SelectDPUAlgo | all_reduce_auto_selector.cc | 678-722 |

---

*数据来源：HCCL MC2 源码（`execute_selector.cc` + `auto_selector_base.cc` + `all_reduce_auto_selector.cc` + `selector_registry.h/cc`）  
分析日期：2026-08-13*
