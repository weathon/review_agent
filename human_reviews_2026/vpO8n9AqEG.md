# Quadratic Direct Forecast for Training Multi-Step Time-Series Forecast Models

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
The design of training objective is central to training time-series forecasting models. Existing training objectives such as mean squared error mostly treat each future step as an independent, equally weighted task, which we found leading to the following two issues: (1) overlook the *label autocorrelation effect* among future steps, leading to biased training objective; (2) fail to set *heterogeneous task weights* for different forecasting tasks corresponding to varying future steps, limiting the forecasting performance. To fill this gap, we propose a novel quadratic-form weighted training objective, addressing both of the issues simultaneously. Specifically, the off-diagonal elements of the weighting matrix account for the label autocorrelation effect, whereas the non-uniform diagonals are expected to match the most preferable weights of the forecasting tasks with varying future steps. To achieve this, we propose a Quadratic Direct Forecast (QDF) learning algorithm, which trains the forecast model using the adaptively updated quadratic-form weighting matrix. Experiments show that our QDF effectively improves performance of various forecast models, achieving state-of-the-art results. Code is available at https://anonymous.4open.science/r/QDF-8937.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the learning objective design of time-series forecasting models. Specifically, it proposes a novel quadratic-form weighted training objective (QDF) to address two issues of existing methods: (1) oversight of the label autocorrelation effect among future steps, and (2) failure to set heterogeneous task weights for different forecasting tasks.  Experiments showcase the validity of QDF on public datasets.

### Strengths
1. The paper improves the learning objective for TSF models, which is an insufficiently explored yet important research problem in time-series forecasting.

2. Experiment results are effectively presented and structured to support the paper's claims.

3. Code is provided which facilitates reproduction.

### Weaknesses
1. The paper could benefit from a more analytical exploration why the proposed weighting matrix could possibly improve existing formulations, especially FreDF and Time-o1. 

2. For time-series datasets, careful design of data splitting strategies is essential to avoid information leakage. It is not very clear whether the risk of leakage is fully bypassed in the hybrid splitting strategy of this paper (e.g., validation, meta-update, etc.). A detailed analysis is needed to analyze the leakage issue as well as the strategies or designs to avoid it.

3. Although the code is released, the  introduction to key components is lacking which impedes verification: how to reproduce the results with the used hyper parameters? where is the key code components? and what is the required software environment for reproducing? whether the code is built upon established repos?

### Questions
1. In Definition 3.2, the optimization problem optimizes $\theta$ in $D_{in}$ and then $\Sigma$ in $D_{out}$. Compared to using the full dataset, i.e., the concatenation of $D_{in}$ and $D_{out}$ to update $\Sigma$ and $\theta$, what is the advantage of the splitting strategy here?

2. Whether the proposed methodology is directly applicable to general multitask learning tasks? Is there any limitation or necessary adaptations when generalizing beyond the TSF context?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents QDF, a method for training forecasting models using a trained covariance matrix, thus solving the problem of assuming uncorrelated residuals. The authors formulate learning $\Sigma$ as a bilevel problem, which they solve sequentially over $K$ splits of the data, then train the forecasting model using the learned (static) $\Sigma$. They empirically compare the QDF objective with other time series losses, explore design choices for learning the covariance and comment on the sensitivity of the introduced hyperparameters.

### Strengths
- The paper introduces a technically novel strategy to solve an empirically and theoretically motivated problem that is much relevant in neural, (direct) non-autocorrelational models for time series forecasting. 
- Promising empirical evidence of QDF on a comprehensive set of datasets, against transformer and non-transformer based methods, and against a variety of time series optimization objectives.
- Ablation studies and the effect on various architectures is provided.

### Weaknesses
- A theoretical result or comments on the convergence of Algorithms 1 and 2 would make the presentation stronger.
- Emphasizing the above point, the proposed method may be too computationally expensive for some models, given the bilevel optimization, over $K$ splits required to perform the optimization to find $\Sigma$. 
- Some parts of the text are not very clear: For instance, on the results in Section 4.2, which model is used to compare the different forecasting objectives? Similarly in Table 3 (Ablation study).
- The clarity  of the last paragraph of Secton 3.2 could be improved.
- The connection of this method with meta-learning is not very clear as it is not formally stated.
- There's the unmentioned assumption that the time series is non-heteroscedastic and thus is characterized by a single $\Sigma$

### Questions
- Are some architectures more prone to converge to correlated residuals than others? 
- In the Ablation  study (4.4) how are the two QDFs integrated, by taking their average?
- Visualizing the correlation matrices of the residuals after training with QDF would be interesting to see.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors propose a method named Quadratic Direct Forecast (QDF) for time series forecasting. The key idea is to optimize the training objective via a quadratic-form weighting matrix to capture both label autocorrelation and weighing different forecasting tasks. The authors apply their proposed method to baseline models and show that it improves the performance in prediction accuracy.

### Strengths
1. Overall this paper is clearly written and easy to understand. The technical details are sound and mostly sufficient. 

2. The proposed method to improve the training objectives for time series forecasting is novel and applicable to related problems in this domain. 

3. The authors provide the source code of their implementation. After reviewing the source code, I did not find major issues. 

4. The authors perform rigorous evaluations and helpful ablation studies to understand the impact of different components in the proposed QDF framework.

### Weaknesses
1. In addition to MAE and MSE, the authors should evaluate their proposed method with MAPE (mean absolute percentage error) which is robust under different scales of the time series values.

2. The authors should also evaluate their proposed method on standard benchmark datasets for time series forecasting, such as the M4 competition dataset.

### Questions
1. How does the model performance change with the dimensionality of the time series? 

2. Related to above, is there a curse of dimensionality in the training objective of QDF? If so, how can this issue be effectively addressed?

### Soundness
3

### Presentation
3

### Contribution
3
