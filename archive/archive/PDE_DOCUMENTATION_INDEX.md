# PDE模块文档索引

本索引汇总了PDE模块的所有技术文档，帮助您快速找到需要的信息。

---

## 📚 文档列表

### 1. [PDE_QUICK_REFERENCE.md](PDE_QUICK_REFERENCE.md) ⭐ **首选阅读**
**快速参考手册 - 实用指南**

- ✅ 一句话总结
- ✅ 快速选择指南（我该用哪个类？）
- ✅ 核心类对照表
- ✅ 三种Hessian方法完整示例代码
- ✅ 典型使用案例（复制粘贴即用）
- ✅ 常见问题FAQ
- ✅ 性能优化建议
- ✅ 推荐学习路径

**适合**:
- 快速上手使用PDE模块
- 查找具体用法和示例代码
- 解决常见问题

**篇幅**: 482行，14KB

---

### 2. [PDE_MODULE_ANALYSIS.md](PDE_MODULE_ANALYSIS.md) 📊 **深度分析**
**模块分析报告 - 技术细节**

- ✅ 完整目录结构
- ✅ 5层依赖关系层次
- ✅ 三种Hessian计算路径详解
- ✅ 求解器性能对比表
- ✅ 手工vs自动微分对比
- ✅ 稀疏性利用方式分析
- ✅ 代码规模统计（按文件）
- ✅ 关键代码片段解释
- ✅ 核心算法流程伪代码
- ✅ 典型应用案例
- ✅ 总结：两条平行路径

**适合**:
- 深入理解模块架构
- 了解各组件职责和依赖
- 代码维护和扩展

**篇幅**: 366行，12KB

---

### 3. [PDE_DEPENDENCY_VISUALIZATION.md](PDE_DEPENDENCY_VISUALIZATION.md) 🎨 **可视化图表**
**依赖关系可视化 - 架构图谱**

- ✅ 整体架构ASCII图（5层结构）
- ✅ 数据流向图（手工路径 + AAD路径）
- ✅ 模块交互序列图
- ✅ 性能对比可视化（对数尺度）
- ✅ 依赖强度矩阵
- ✅ 代码复杂度分析表
- ✅ 推荐学习路径图

**适合**:
- 快速理解整体架构
- 可视化依赖关系
- 规划学习路径

**篇幅**: 436行，27KB

---

### 4. [PDE_REORGANIZATION_SUMMARY.md](PDE_REORGANIZATION_SUMMARY.md) 📝 **重构记录**
**模块重组总结 - 历史演进**

- ✅ 重组前后对比
- ✅ 新旧文件映射
- ✅ 模块化设计理念

**适合**:
- 了解模块演进历史
- 查找旧文件位置

**篇幅**: 178行，7.3KB

---

## 🎯 按需求查找文档

### 我想快速开始使用
→ 阅读 [PDE_QUICK_REFERENCE.md](PDE_QUICK_REFERENCE.md)
- 第1节：快速选择指南
- 第3节：典型使用案例

### 我想理解整体架构
→ 查看 [PDE_DEPENDENCY_VISUALIZATION.md](PDE_DEPENDENCY_VISUALIZATION.md)
- 整体架构图
- 数据流向图

### 我想了解某个模块的功能
→ 参考 [PDE_MODULE_ANALYSIS.md](PDE_MODULE_ANALYSIS.md)
- 第2节：依赖关系层次（按层级）
- 第6节：详细文件列表与代码规模

### 我想选择最合适的实现方法
→ 对比 [PDE_MODULE_ANALYSIS.md](PDE_MODULE_ANALYSIS.md)
- 第3节：功能实现对比
- 第4节：求解器对比表

### 我想看示例代码
→ 复制 [PDE_QUICK_REFERENCE.md](PDE_QUICK_REFERENCE.md)
- 第4节：典型使用案例（3个完整示例）

### 我遇到问题了
→ 检查 [PDE_QUICK_REFERENCE.md](PDE_QUICK_REFERENCE.md)
- 第6节：常见问题FAQ

### 我想优化性能
→ 学习 [PDE_QUICK_REFERENCE.md](PDE_QUICK_REFERENCE.md)
- 第7节：性能优化建议
- 第2节：方法2 Edge-Pushing优化

---

## 📖 推荐阅读顺序

### 第一次接触（30分钟）
1. **浏览**: [PDE_QUICK_REFERENCE.md](PDE_QUICK_REFERENCE.md)
   - 前3节（总结、选择指南、方法对比）
2. **可视化**: [PDE_DEPENDENCY_VISUALIZATION.md](PDE_DEPENDENCY_VISUALIZATION.md)
   - 整体架构图
3. **实践**: 复制并运行案例A（计算Vanna和Volga）

### 深入学习（2小时）
4. **细读**: [PDE_MODULE_ANALYSIS.md](PDE_MODULE_ANALYSIS.md)
   - 完整阅读，理解每个模块
5. **代码**: 按推荐学习路径阅读源码
   - 见 [PDE_DEPENDENCY_VISUALIZATION.md](PDE_DEPENDENCY_VISUALIZATION.md) 最后一节
6. **实验**: 运行性能基准测试（案例C）

### 专家级掌握（1周）
7. **研究**: 阅读所有源码并理解算法细节
8. **扩展**: 尝试实现自己的波动率模型
9. **优化**: 针对特定场景调优性能

---

## 🔑 核心概念速查

| 概念 | 解释 | 文档位置 |
|-----|------|---------|
| **Edge-Pushing** | 利用稀疏邻接图加速Hessian计算的算法 | 快速参考 §3.2 |
| **LocalVolAdjacency** | 构建PDE参数间依赖关系的邻接图 | 分析报告 §2.3 |
| **Crank-Nicolson** | 无条件稳定的二阶隐式PDE格式 | 快速参考 FAQ Q4 |
| **SVI模型** | 参数化波动率曲面的模型 | 分析报告 §2.1 |
| **Vanna** | ∂²V/∂S∂σ，Delta对波动率的敏感度 | 快速参考 案例A |
| **Volga** | ∂²V/∂σ²，Vega对波动率的敏感度 | 快速参考 案例A |
| **伴随方法** | 高效计算梯度的反向传播技术 | 分析报告 §3.6 |
| **ADVar** | 自动微分变量，自动构建计算图 | 分析报告 §5 |
| **稀疏Hessian** | 只存储非零元素的Hessian矩阵 | 可视化 性能对比 |

---

## 📊 关键数据速查

### 性能指标
- **Edge-Pushing加速比**: 10-100×（实测）
- **ATM聚焦额外加速**: 3-5×
- **支持网格规模**:
  - 手工方法: 200×200
  - AAD方法: 20-100

### 代码规模
- **总代码量**: ~3,945行
- **核心求解器**: 445行
- **Hessian计算**: 1,432行
- **AAD集成**: 1,133行

### 复杂度
- **朴素Hessian**: O(P²)，P = (M+1)×(N+1)
- **Edge-Pushing**: O(P×d)，d ≈ 5-10
- **正向PDE求解**: O(N×M)

---

## 🗂️ 文件位置速查

所有文档位于项目根目录:
```
/home/junruw2/AAD/
├── PDE_QUICK_REFERENCE.md           ⭐ 14KB  快速参考
├── PDE_MODULE_ANALYSIS.md           📊 12KB  深度分析
├── PDE_DEPENDENCY_VISUALIZATION.md  🎨 27KB  可视化
├── PDE_REORGANIZATION_SUMMARY.md    📝 7.3KB 重构记录
└── PDE_DOCUMENTATION_INDEX.md       📑 本文档
```

源代码位于:
```
/home/junruw2/AAD/aad_edge_pushing/pde/
├── core/           核心求解器
├── models/         波动率模型
├── graph/          邻接图
├── handcraft_aad/  手工Hessian ⭐
├── greeks/         Greeks计算
└── aad_integration/  AAD集成 ⭐
```

---

## 🎓 学习资源

### 内部资源
1. **模块README**: `aad_edge_pushing/pde/README.md`
2. **代码注释**: 每个文件开头的docstring
3. **测试文件**: `tests/test_pde_*.py`
4. **示例**: `examples/pde_*.py`

### 外部参考文献
1. Capriotti et al. (2015) - AAD in PDEs
2. Griewank et al. (2008) - Edge-Pushing算法
3. Gatheral (2004) - SVI模型
4. Dupire (1994) - 局部波动率

---

## ❓ 获取帮助

### 快速问题
→ 检查 [PDE_QUICK_REFERENCE.md](PDE_QUICK_REFERENCE.md) FAQ节

### 架构问题
→ 查阅 [PDE_MODULE_ANALYSIS.md](PDE_MODULE_ANALYSIS.md)

### 使用问题
→ 参考 [PDE_QUICK_REFERENCE.md](PDE_QUICK_REFERENCE.md) 典型案例

### 找不到信息
→ 使用文档内搜索功能（Ctrl+F）

---

## 📈 文档更新

| 版本 | 日期 | 更新内容 |
|-----|------|---------|
| 1.0 | 2024-10-24 | 初始版本，包含全部4份文档 |

---

## 💡 使用建议

1. **首次使用**: 从 [PDE_QUICK_REFERENCE.md](PDE_QUICK_REFERENCE.md) 开始
2. **随时参考**: 将快速参考手册加入书签
3. **深入学习**: 结合可视化图表和分析报告
4. **实践为主**: 运行示例代码，修改参数观察效果
5. **持续探索**: 逐步阅读源码，理解实现细节

---

**最后更新**: 2024-10-24
**文档总计**: 4份，约1,462行，60.3KB
