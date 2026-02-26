# 金融数据分析：两次大作业
2025.12

## 项目语言：Python

## 第一次作业项目描述：
利用沪深300股东数据，进行股东分析。构建股东能力评分体系、计算股东的超额收益。

### 项目详情：
1. 下载数据和预处理
2. 依据公司的描述提取出其所在行业
3. 对某个股东的投资行业和投资集中度进行展示；分析其行业集中度趋势、投资组合规模等数据
4. 构建股东能力评分体系，组合增长能力、投资稳定性、集中度、规模等要素打分后加权平均

## 第二次作业项目描述：
阅读论文，利用两层GRU对沪深300财务数据的未来值进行预测，并利用这些预测值预测下一期收益率。

### 项目详情：
1. 进行了大部分的数据预处理工作
2. 通过论文阅读和实验比较，确定使用三层GRU搭建模型（每层GRU内部2 layers）
3. 使用沪深300公司的财务数据进行训练，预测ROE增量、毛利率、每股收益三个指标；并利用随机森林模型，用测出的指标预测股票的收益率
4. 结合第一次作业，可以对沪深300中分行业的财务数据进行预测（已成功实现）

# Financial Data Analysis: Two Major Assignments
2025.12

## Project Language: Python

## First Assignment Description:
Utilize CSI 300 shareholder data to conduct shareholder analysis. Construct a shareholder capability scoring system and calculate shareholders' excess returns.

### Project Details:
1. Download data and preprocess
2. Extract the industry of each company based on its description
3. Visualize a specific shareholder's investment industries and concentration; analyze trends in industry concentration, portfolio scale, and other data
4. Build a shareholder capability scoring system, scoring elements such as portfolio growth capability, investment stability, concentration, and scale, then calculate a weighted average

## Second Assignment Description:
Read research papers, use a two-layer GRU to predict future values of CSI 300 financial data, and use these predicted values to forecast the next period's returns.

### Project Details:
1. Completed most of the data preprocessing work
2. Through paper review and experimental comparison, decided to use a three-layer GRU to build the model (each GRU layer internally consists of 2 layers)
3. Train using financial data of CSI 300 companies to predict three indicators: ROE increment, gross margin, and earnings per share; then use a Random Forest model with the predicted indicators to forecast stock returns
4. Combined with the first assignment, successfully implemented industry-specific financial data prediction for CSI 300 companies
