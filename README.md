# 共享单车租借量预测：实验代码

本项目用于课程大作业题目 4：基于小时级日期、时间、天气、温度、湿度、风速等特征预测共享单车一小时内租借总量 `cnt`。

默认流程严格使用 `train.csv` 的 `cnt` 训练回归模型，对 `test.csv` 预测并生成符合提交格式的 `submission.csv`：

```csv
ID,cnt
13904,xxx
13905,xxx
```

评价指标为 MSE，因此代码中的模型选择、验证与融合权重优化均以 MSE 为主。

## 目录结构

```text
bike_rental_prediction_project/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── task4.md
├── src/
│   ├── config.py
│   ├── features.py
│   ├── models.py
│   ├── pipeline.py
│   └── utils.py
├── outputs/                 # 运行后生成 submission.csv、验证结果等
├── models/                  # 运行后可保存训练好的模型
├── requirements.txt
└── run.py                   # 主入口
```

## 环境安装

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

`lightgbm` 和 `xgboost` 是默认主模型依赖。代码内含 sklearn 兜底模型；若未安装 LightGBM/XGBoost，程序会自动跳过对应模型，但预测效果通常会下降。

## 一键运行

在项目根目录执行：

```bash
python run.py \
  --train data/train.csv \
  --test data/test.csv \
  --output-dir outputs \
  --model-dir models \
  --model-set default \
  --feature-mode base \
  --seed 42 \
  --n-jobs 1
```

运行完成后，主要文件包括：

```text
outputs/submission.csv
outputs/validation_metrics.csv
outputs/ensemble_weights.json
outputs/feature_columns.json
models/*.joblib
```

## 可复现实验设置

默认使用时间后验验证：取训练集最后 `len(test.csv)` 条样本作为验证集，模拟测试集“未来时间段预测”的设定。该方式比随机划分更符合本题的时间切分结构。

默认特征模式为 `base`，包含：

- 原始时间、天气、温湿度、风速特征；
- 日期特征：day、dayofyear、weekofyear、quarter、days_since_start 等；
- 周期编码：hour/month/weekday/dayofyear 的 sin/cos；
- 行为模式特征：早晚高峰、工作日高峰、夜间、周末等；
- 天气交互项：temp × hum、temp × windspeed、hum × windspeed 等。

可选高级特征：

```bash
python run.py --feature-mode advanced
```

`advanced` 会额外加入历史同期参考特征和目标均值统计特征。它可能在部分划分上有帮助，但也可能因时间外推造成过拟合；因此默认采用本地时间验证更稳定的 `base`。

## 常用参数

```text
--model-set default      LightGBM/XGBoost 主模型组合，推荐默认值
--model-set quick        更快的小模型组合，用于调试流程
--model-set sklearn      仅 sklearn 兜底模型
--feature-mode base      默认稳定特征
--feature-mode advanced  加入历史参考与目标统计特征
--no-save-model          不保存模型文件
```

## 输出检查

代码会自动检查：

- `submission.csv` 是否只有 `ID,cnt` 两列；
- 行数是否等于测试集行数；
- `ID` 是否与 `test.csv` 完全一致且顺序不变；
- `cnt` 是否无缺失、无负值。
