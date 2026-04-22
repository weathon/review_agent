# Self-adaptive Retrieval-Augmented Reinforcement Learning for Time Series Forecasting

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 6, 4, 0

## Abstract
Deep learning models for time series forecasting, typically optimized with Mean Squared Error (MSE), often exhibit spectral bias. This phenomenon arises because MSE prioritizes minimizing errors in high-energy, typically low-frequency components, leading to an underfitting of crucial, lower-energy high-frequency dynamics and resulting in overly smooth predictions. To address this, we propose Self-adaptive Retrieval-augmented Reinforcement learning for time series Forecasting (SRRF), a novel plug-and-play training enhancement. SRRF uniquely internalizes high-frequency modeling capabilities into base models during training, ensuring no additional inference costs or architectural changes for the base model. The framework operates by first employing Retrieval-Augmented Generation (RAG) to provide contextual grounding via relevant historical exemplars. Subsequently, building on this contextual guidance, a Reinforcement Learning (RL) agent learns an adaptive policy to correct and enhance initial forecasts, optimized via a reward function that promotes both overall predictive accuracy and fidelity to high-frequency details. Comprehensive evaluations on diverse benchmarks demonstrate that models trained with the SRRF methodology substantially improve upon their original versions and other state-of-the-art techniques, especially in accurately predicting volatile series and fine-grained temporal patterns. Qualitative and spectral analyses further confirm SRRF's effectiveness in mitigating spectral bias and enhancing high-frequency representation. Our code is available at \href{https://anonymous.4open.science/r/ACAC-9999/README.md}{https://anonymous.4open.science/r/ACAC-9999/README.md}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes Self-adaptive Retrieval-augmented Reinforcement Learning for Time Series Forecasting, a novel enhancement framework designed to address spectral bias in time series models trained with MSE. The framework integrates Retrieval-Augmented Generation to provide contextual grounding through historical exemplars and employs RL for adaptive correction of base model predictions. The paper demonstrates SRRF’s effectiveness through comprehensive experiments, showing significant improvements in forecasting performance across diverse datasets and models, especially in terms of spectral fidelity.

### Strengths
1. Innovative framework for spectral bias correction.
SRRF’s integration of RAG and RL to counteract spectral bias is novel and effectively addresses a long-standing challenge in time series forecasting. The clear focus on mitigating spectral bias enhances its applicability to volatile data where high-frequency dynamics are critical.

2. Plug-and-play design with no inference overhead.
The SRRF framework is a training-time enhancement that operates without changing the base model architecture. This approach ensures no added computational burden during inference, making SRRF a practical solution for enhancing existing models without requiring significant infrastructure changes.

### Weaknesses
1. Lack of clarity in RL correction mechanism.
The mathematical details of the RL correction process (Sec. 2.4) are not fully elaborated, leaving some uncertainty about the exact mechanism through which the RL agent refines predictions. While the qualitative results suggest success, the absence of a more detailed theoretical explanation or formal derivation of the RL correction step limits the depth of understanding regarding how SRRF resolves issues with traditional gradient-based optimization (Sec. 2).

2. Sensitivity to hyperparameters.
The performance of SRRF is sensitive to key hyperparameters, including the retrieval count (k) and the RL sample count, with noticeable degradation when too many exemplars are retrieved. While the paper discusses these sensitivities, it lacks a deeper exploration of how these parameters interact across different datasets, which could further help users tailor the framework to their needs.

3. Increased computational costs during training.
Although SRRF improves model accuracy, the additional computational cost from RAG and RL sampling may limit its scalability, particularly for large datasets or more complex base models (Sec. 5). The paper acknowledges these costs but does not provide a detailed breakdown of the time or memory overhead incurred by the retrieval and reinforcement learning steps during training.

4. While SRRF significantly enhances performance in high-frequency components, its impact on low-frequency components is sometimes inconsistent. In certain cases, SRRF leads to an increase in error energy in the lowest frequency band, suggesting that the framework may unintentionally sacrifice some predictive accuracy for smooth trends in favor of capturing finer details.

### Questions
1. A more explicit description or pseudocode would clarify how the reinforcement signal modifies predictions and how stability is ensured during joint optimization.

2. What is the empirical trade-off between retrieval depth and performance?
How sensitive are results to the number of retrieved exemplars and their selection strategy? 

3. What portion of the total training time is attributable to retrieval and RL sampling? Would a lighter retrieval or policy model achieve similar gains? Including GPU hours or per-epoch runtime comparisons to baselines would strengthen the practicality argument.

### Soundness
2

### Presentation
2

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
This paper aims to address the problems of MSE loss function commonly used in the regression problem and thus proposes Retrieval-augmented Reinforcement learning for time series Forecasting (SRRF). The idea is to provide compensations via a policy network to the predicted outputs. The main model is trained via a joint loss function while the policy network is trained to minimize the RL loss via a policy gradient method.

### Strengths
1) the idea of RL for compensations of predicted output is novel;
2) the presentation of this paper is easy to follow;
3) Numerical results are convincing.

### Weaknesses
1) Code information should be placed in the abstract to gain better visibility;
2) Comparisons with those published in 2025 onward should be added;
3) The performance difference is not big such that the statistical tests are necessary;
4) Complexity analysis should be provided.

### Questions
1) Code information should be placed in the abstract to gain better visibility;
2) Comparisons with those published in 2025 onward should be added;
3) The performance difference is not big such that the statistical tests are necessary;
4) Complexity analysis should be provided.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
To address the issue of overly smooth predictions caused by MSE-based training, the paper proposes a Self-adaptive Retrieval-augmented Reinforcement learning framework (SRRF) for time series forecasting. SRRF employs Retrieval-Augmented Generation to provide contextual grounding and reinforcement learning to correct initial forecasts. By integrating the SRRF module into various forecasting models, the approach achieves good prediction performance.

### Strengths
1. The paper proposes a plug-and-play that can be applied to all time series forecasting models.

2. The paper adopts reinforcement learning to learn a policy network that corrects the model’s initial forecasting results.

3. As a plug-and-play, SRRF achieves promising forecasting performance across different models.

### Weaknesses
W1. Method description is unclear:  1. The paper does not clearly explain how the policy network learns the mean (**μ**) and standard deviation (**σ**) from the reference prediction and the initial prediction. 2. It is also unclear whether the policy network generates the correction term (**α**) for each individual time step or for the entire time series sample.

W2. Experiment results: 1. The SRRF results in Table 1 are reported on top of the iTransformer, but they are inconsistent with the iTransformer+SRRF results shown in Table 2.   2. Some baseline results in Table 2 are significantly worse than those reported in the original papers — for example, PatchTST on the Traffic and Weather datasets shows noticeably lower prediction performance.

W3. Hyperparameter sensitivity experiment: The authors did not specify which dataset was used for this experiment.  Moreover, the RL sample count parameter has a major impact on prediction performance. The authors should conduct additional experiments across multiple datasets to verify whether the optimal choice of this parameter varies between datasets.

W4. Missing baselines: The authors should include more recent baselines, such as TimeMixer and TimeMixer++, for a more comprehensive comparison.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
The authors propose to integrate RAG and RL into the learning process of a deep learning models for time series forecasting.

### Strengths
- original idea
- reasonably well written paper

### Weaknesses
- unclear integration of RAG and RL into model training
- no detailed analysis of additional training costs introduced by RAG and RL

### Questions
Given that RAG has high costs, how do you integrate RAG into the iterations used during the optimization process?
How can a subset of plausible historical examples be used to train the model, and how exactly are these examples selected?
How much extra training costs are caused by your RAG and RL sampling procedure?

### Soundness
1

### Presentation
2

### Contribution
1
