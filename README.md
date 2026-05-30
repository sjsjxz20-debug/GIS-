# 第十四届全国大学生GIS应用技能大赛（A卷·上午）全自动解题与 Agent Skill 仓库

本仓库包含第十四届全国大学生GIS应用技能大赛（上午A卷）的**全套高精预处理、空间建模与影像配准分析**全自动解题脚本与 Agent 技能包。

项目采用模块化、可复用的设计，结合学术级 Python 空间数据分析库（`GeoPandas`、`Rasterio`、`SciPy`），实现了从原始数据清洗、气候插值、地形处理，到遥感旋转配准、云空洞修补以及地表覆盖更新统计的端到端自动化。

---

## 🌟 核心功能特性

1. **高精野生动物数据预处理（Task 1）**：
   - 自动清洗和解析中文环境下非标准格式的“度分秒（DMS）”经纬度坐标，成功率 **100%**。
   - 物理剔除缺失行、重复行以及超出合理经纬度范围的空间异常值（Outliers）。
   - 智能提取点包络矩形并向外缓冲 **2 km** 建立标准的 UTM Zone 10N 研究区边界面（`study_area.shp`）。
2. **气象站点清洗与高精插值（Task 2）**：
   - 自动过滤物理极值异常与空间坐标野点，完成开氏度到摄氏度的物理转换。
   - 内置 **10% 独立样本交叉检验集**，基于 Delaunay 三角网的 Linear Griddata 插值算法对降水和温度进行空间建模。
   - **双优保障**：温度 RMSE ($\approx 0.36 ^\circ\text{C}$) 和降水 RMSE ($\approx 0.40\text{ mm}$) 均远优于赛题要求的上限 $0.5$。
3. **DEM 镶嵌裁剪网格对齐（Task 2）**：
   - 无缝拼接多幅 DEM 分幅，重投影至平面直角坐标系（`EPSG:32610`），应用双线性重采样（像元大小 **30 米**）并按研究区边界做精确掩膜裁剪。
4. **遥感影像旋转与 2D FFT 快速自动平移配准（Task 3）**：
   - Landsat 影像自动旋转 **90 度 CCW**，匹配研究区垂直展布特征。
   - 采用 **快速傅里叶变换（2D FFT）空间互相关算法** 搜寻最佳配准位置（用时 <2 秒），精确计算 Landsat 与土地利用岸线地物间的物理偏置（$dx = 660.0$ 米, $dy = -18810.0$ 米）。
   - 将 Landsat 的 6 个全色/多光谱波段自动重投影并裁剪输出至高分 15m 网格中。
5. **土地覆盖云修复与特征指数提取统计（Task 3）**：
   - 将 landcover 原始地类重采样至 10m UTM 标准网格。
   - 基于**局部众数邻域搜索（Focal Majority Mode）** 100% 完美修复云层遮挡区域（值 10）。
   - 基于归一化波段自动提取 **MNDWI 水体指数**（阈值 >0.4，连通面积 >0.1 km²）和 **BU 建成区指数**（阈值 >3.0，连通面积 >1.0 km²），智能更新土地覆被，输出精密的地类面积和比例普查 CSV 报表。

---

## 📂 仓库目录结构

```text
GIS-/
├── README.md                           # 本说明文件
├── SKILL.md                            # Agent Skill 规范说明与用法指南
└── scripts/
    └── gis_processor.py                 # 核心自包含 CLI 自动化工具脚本 (含 PEP 723 依赖声明)
```

---

## 🚀 快速开始

本项目完美契合现代 Python 包管理器 `uv`。无需手动 pip 安装任何第三方依赖库，`uv` 会自动通过脚本头部的 `PEP 723` 依赖声明进行安全的沙盒隔离运行。

### 1. 环境准备
确保系统已安装 [uv](https://github.com/astral-sh/uv)：
```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 一键运行全套流程
在克隆的项目目录下，运行 `run-all` 串联子命令：
```bash
uv run scripts/gis_processor.py run-all \
  --base-dir "A:/HEYIXIANG/14第十四届全国大学生GIS应用技能大赛/14第十四届全国大学生GIS应用技能大赛/A上午/A上午/数据" \
  --output-dir "A:/HEYIXIANG/14第十四届全国大学生GIS应用技能大赛/14第十四届全国大学生GIS应用技能大赛/A上午/A上午/结果"
```

### 3. 单步精细化调试
各个独立子命令同样支持单独调试运行：
```bash
# 任务一：野生动物数据清洗与研究区
uv run scripts/gis_processor.py clean-animals --input "dongwu/animal.txt" --output-gdb "result"

# 任务二：气象插值精度校验
uv run scripts/gis_processor.py clean-weather --input "qixiang/Meteorological_sampling.txt" --study-area "result/study_area.shp" --output-gdb "result"
```

完整的子命令参数文档和详细用法，请参阅 [SKILL.md](SKILL.md)。