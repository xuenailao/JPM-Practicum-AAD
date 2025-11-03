## Hessian Computation Benchmark Suite

完整的Hessian计算方法对比测试框架，包含5种方法的精度和速度对比。

### 5种计算方法

1. **Bumping2** - 纯有限差分
   - 算法：两次bumping求二阶导
   - PDE求解次数：9次
   - 特点：最简单，实现直接

2. **AAD+Bumping** - 混合方法
   - 算法：使用一次AAD计算Jacobian，在此基础上bumping求Hessian
   - PDE求解次数：5次
   - 特点：结合AAD效率和有限差分的简单性

3. **Double-AAD** - 真正的双重AAD (Forward-over-Reverse)
   - 算法：使用`hvp_for`函数，两次AAD计算Hessian
   - PDE求解次数：3次 (1次Jacobian + 2次hvp_for)
   - 特点：**不使用Edge-Pushing**，使用完整计算图的FoR方法

4. **Edge-Pushing** - Algorithm 4优化
   - 算法：使用`algo4_adjlist`邻接表优化
   - PDE求解次数：1次
   - 特点：理论上最优，但当前实现较慢

5. **BSM-Analytical** - Black-Scholes解析解
   - 算法：解析公式
   - PDE求解次数：0次
   - 特点：作为精度基准(baseline)

### 测试维度

#### 参数集 (Parameter Sets)
- **ATM** (At-the-Money): S0=K=100
- **ITM** (In-the-Money): S0=110, K=100
- **OTM** (Out-of-the-Money): S0=90, K=100
- **High Volatility**: σ=0.4
- **Low Volatility**: σ=0.1
- **Short Maturity**: T=0.25
- **Long Maturity**: T=2.0

#### 网格分辨率 (Grid Sizes)
- **Coarse**: M=51, N=100
- **Medium**: M=101, N=200
- **Fine**: M=151, N=300
- **Very Fine**: M=201, N=400

### 使用方法

#### 1. 快速测试 (Single Case)
```bash
python quick_benchmark.py
```
- 单个参数集，单个网格
- 运行时间：~2分钟
- 输出：5个方法的完整对比

#### 2. 综合测试 (Comprehensive)
```bash
python comprehensive_benchmark.py
```
- 5个参数集 × 3个网格 = 15组测试
- 跳过fine grid上的Edge-Pushing (节省时间)
- 运行时间：~10-15分钟
- 输出：保存到`benchmark_results/comprehensive_results.json`

#### 3. 完整Benchmark (Full Suite)
```bash
python benchmark_five_methods.py --mode full
```
- 7个参数集 × 4个网格 = 28组测试
- 所有方法都测试
- 运行时间：~30-60分钟
- 输出：保存到`benchmark_results/hessian_benchmark_results.csv`

快速模式：
```bash
python benchmark_five_methods.py --mode quick
```

### 输出格式

#### 屏幕输出示例
```
====================================================================================================
Parameter Set: ATM
  S0=100.0, K=100.0, T=1.0, r=5.00%, σ=20.00%, Moneyness=1.000
====================================================================================================

  Grid: M=51, N=100
  ------------------------------------------------------------------------------------------------
    Bumping2             ✓   113.11ms  Gamma=0.018893  Volga=7.178145  (9 PDE)
    AAD+Bumping          ✓  5474.50ms  Gamma=0.018893  Volga=7.162008  (5 PDE)
    Double-AAD           ✓  3926.79ms  Gamma=0.019132  Volga=7.241539  (3 PDE)
    Edge-Pushing         ✓ 97297.77ms  Gamma=0.018893  Volga=7.194035  (1 PDE)
    BSM-Analytical       ✓     0.74ms  Gamma=0.018762  Volga=9.850059  (0 PDE)

    Errors vs BSM Analytical:
    ------------------------------------------------------------------------------------------
    Method               Price %    Gamma %    Vega %     Vanna %    Volga %
    ------------------------------------------------------------------------------------------
    Bumping2               0.1327%    0.6991%    0.3371%    1.3168%   27.1259%
    AAD+Bumping            0.1327%    0.6991%    0.3453%    1.0421%   27.2897%
    Double-AAD             0.1327%    1.9698%    0.3453%    2.7594%   26.4823%
    Edge-Pushing           0.1327%    0.6991%    0.3453%    1.4738%   26.9646%
```

#### JSON输出格式
```json
{
  "method": "Double-AAD",
  "S0": 100.0,
  "K": 100.0,
  "T": 1.0,
  "r": 0.05,
  "sigma": 0.2,
  "M": 51,
  "N": 100,
  "moneyness": 1.0,
  "price": 10.436712,
  "delta": 0.636963,
  "gamma": 0.019132,
  "vega": 37.653624,
  "vanna": -0.273665,
  "volga": 7.241539,
  "time_ms": 3926.79,
  "n_pde_solves": 3,
  "status": "SUCCESS"
}
```

### 关键指标

#### 精度 (Accuracy)
所有方法与BSM解析解的误差百分比：
- **Price**: 期权价格误差
- **Gamma**: ∂²V/∂S² 误差
- **Vega**: ∂V/∂σ 误差
- **Vanna**: ∂²V/∂S∂σ 误差
- **Volga**: ∂²V/∂σ² 误差

#### 速度 (Speed)
- **Time (ms)**: 总计算时间（毫秒）
- **PDE Solves**: PDE求解次数
- **Time per PDE**: 平均每次PDE求解时间

### 典型结果分析

#### 速度排名 (M=51, N=100)
1. **BSM-Analytical**: ~0.7ms (baseline)
2. **Bumping2**: ~113ms (最快的PDE方法)
3. **Double-AAD**: ~3,900ms (FoR方法)
4. **AAD+Bumping**: ~5,500ms
5. **Edge-Pushing**: ~97,000ms (最慢，但只需1次PDE求解)

#### 精度对比
- **Price**: 所有方法误差 < 0.15%
- **Gamma**: Bumping2/AAD+Bumping/Edge-Pushing ~0.7%, Double-AAD ~2.0%
- **Volga**: 所有方法 ~27% (粗网格M=51导致)

#### 关键观察
1. **Volga误差大**: 在粗网格(M=51)上，所有方法的Volga误差都很大(~27%)
   - 解决方法：使用更细的网格(M≥151)

2. **Edge-Pushing慢**: 当前实现比预期慢很多
   - 预期：1次PDE求解应该最快
   - 实际：比Bumping2慢800倍
   - 原因：可能是algo4实现未优化

3. **Double-AAD vs AAD+Bumping**:
   - Double-AAD更快(3.9s vs 5.5s)
   - PDE次数更少(3 vs 5)
   - 但Gamma误差略大(2.0% vs 0.7%)

### 文件结构

```
AAD/
├── quick_benchmark.py              # 快速单例测试
├── comprehensive_benchmark.py      # 综合多参数测试
├── benchmark_five_methods.py       # 完整benchmark套件
├── test_five_methods.py           # 基础功能测试
├── benchmark_results/              # 结果输出目录
│   ├── comprehensive_results.json
│   ├── comprehensive_output.txt
│   └── hessian_benchmark_results.csv
└── aad_edge_pushing/pde/methods/  # 方法实现
    ├── bumping2.py
    ├── aad_bumping.py
    ├── double_aad.py
    ├── edge_pushing.py
    └── bsm_analytical.py
```

### 依赖

必需：
- NumPy
- AAD框架 (aad_edge_pushing)

可选：
- Pandas (用于CSV输出和统计分析)

### 常见问题

**Q: 为什么Edge-Pushing这么慢？**
A: 当前algo4_adjlist实现可能未优化，需要profiling分析瓶颈。

**Q: Volga误差为什么这么大？**
A: Volga是二阶导数对σ的导数，对网格分辨率很敏感。使用M≥151可改善。

**Q: Double-AAD和Edge-Pushing有什么区别？**
A:
- Double-AAD: 使用Forward-over-Reverse (FoR)，不使用algo4优化
- Edge-Pushing: 使用algo4_adjlist邻接表优化

**Q: 哪个方法最好？**
A: 取决于需求：
- 速度优先：Bumping2 (113ms)
- 平衡：Double-AAD (3.9s, 较好精度)
- 理论最优：Edge-Pushing (需要优化实现)
