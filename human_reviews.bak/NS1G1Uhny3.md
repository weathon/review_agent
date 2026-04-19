# Efficiently Learning at Test-Time: Active Fine-Tuning of LLMs

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 6

## Abstract
Recent efforts in fine-tuning language models often rely on automatic data selection, commonly using Nearest Neighbors retrieval from large datasets.
However, we theoretically show that this approach tends to select redundant data, limiting its effectiveness or even hurting performance.
To address this, we introduce SIFT, a data selection algorithm designed to reduce uncertainty about the model's response given a prompt, which unifies ideas from retrieval and active learning.
Whereas Nearest Neighbor retrieval typically fails in the presence of information duplication, SIFT accounts for information duplication and optimizes the overall information gain of the selected examples.
We focus our evaluations on fine-tuning at test-time for prompt-specific language modeling on the Pile dataset, and show that SIFT consistently outperforms Nearest Neighbor retrieval, with minimal computational overhead.
Moreover, we show that our uncertainty estimates can predict the performance gain of test-time fine-tuning, and use this to develop an adaptive algorithm that invests test-time compute proportional to realized performance gains.
We provide the `activeft` (Active Fine-Tuning) library which can be used as a drop-in replacement for Nearest Neighbor retrieval.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents SIFT, a data selection algorithm aimed at improving test-time fine-tuning of LLMs by reducing redundancy in data selection. Unlike traditional Nearest Neighbor retrieval, which often selects redundant examples, SIFT combines retrieval and active learning principles to minimize model uncertainty for each prompt. This approach also enables adaptive fine-tuning, dynamically adjusting compute based on anticipated performance gains. Experiments on the Pile dataset indicate that SIFT achieves better efficiency and robustness than Nearest Neighbor retrieval, with minimal computational overhead. The authors provide an open-source library, activeft, for integration and reproducibility.

### Strengths
- The paper introduces SIFT, a well-motivated algorithm that combines retrieval and active learning, effectively addressing data redundancy issues in traditional Nearest Neighbor retrieval for LLM fine-tuning.

- SIFT’s adaptive fine-tuning, which adjusts test-time compute based on predicted performance gains, is an efficient and practical innovation that can optimize resource use, especially in computationally constrained environments.

- The paper provides both theoretical proofs and empirical evidence to demonstrate that SIFT reduces model uncertainty and improves fine-tuning outcomes, adding robustness and credibility to the proposed approach.

- The release of the *activeft* library as a drop-in replacement for Nearest Neighbor retrieval supports transparency and facilitates future research in prompt-specific fine-tuning methods.

### Weaknesses
1. The motivation and definition of the task lack clarity, particularly in distinguishing test-time fine-tuning from standard fine-tuning on selected data. It remains unclear why test-time fine-tuning is necessary in this context and how it fundamentally differs from simply fine-tuning on pre-selected data, which may impact understanding of the novelty and importance of SIFT’s approach.

2. The paper does not fully specify scenarios where test-time fine-tuning with SIFT would be most beneficial. This omission makes it difficult to assess the generalizability and practical applications of the method, particularly for those who unfamiliar with prompt-specific fine-tuning needs.

3. Although SIFT is described as having minimal overhead, the adaptive fine-tuning process could introduce additional complexity in real-time settings. A more thorough breakdown of the computational costs associated with adaptive adjustments would improve understanding of its efficiency.

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces SIFT, a method for selecting informative data to fine-tune large language models (LLMs). It critiques the current Nearest Neighbors retrieval approach which selects redundant data and proposes SIFT, inspired by transductive active learning, to select relevant and diverse data for effective fine-tuning. Additionally, it offers a library as a drop-in replacement for Nearest Neighbor retrieval.

### Strengths
1. Theoretical Foundation: The paper is theoretically robust and well-motivated, offering comprehensive analysis to demonstrate its effectiveness.
2. Organization and Insight: It is well-organized and self-consistent, providing thorough discussions on the research topic and outlining both current and future research directions.

### Weaknesses
1. Inference Cost: Comparison with Nearest Neighbor, this method may require more inference time. It would be beneficial to compare inference times with Nearest Neighbor across different datasets to quantify this.
2. Broader Evaluation: The paper could explore effectiveness on more datasets and larger models, such as LLaMA-3, to validate its scalability and generalizability.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces SIFT, a data selection algorithm aimed at improving the fine-tuning of language models by addressing the limitations of Nearest Neighbors retrieval, which often selects redundant data. SIFT reduces uncertainty about model responses by optimizing overall information gain and accounting for information duplication. The authors demonstrate that SIFT outperforms Nearest Neighbor retrieval in fine-tuning at test time in experiments, with minimal computational overhead. Additionally, they show that their uncertainty estimates can predict performance gains, leading to an adaptive algorithm that optimally allocates computational resources.

### Strengths
1. This paper verifies a meaningful yet little explored topic in LLM application, giving comprehensive and solid discussion on the challenge and the proposed solution. The whole paper is clearly organized and easy to follow.
2. The SIFT strategy is sopported with solid theoretical induction as well as experimental evidences.
3. The experimental study is convincing and covers a wide range of datasets.

### Weaknesses
1. The interpretability of this work might be improved by giving some instances of data selection. 
2. How is the possibility and gain of combining this strategy with orthogonal methods of LLM finetuning?

### Questions
See weaknesses.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a novel data selection algorithm called SIFT (Select Informative data for Fine-Tuning) aimed at improving test-time fine-tuning for large language models (LLMs). The paper addresses the limitations of Nearest Neighbor retrieval in data selection by combining ideas from retrieval and active learning. Unlike traditional Nearest Neighbor methods that can select redundant data, SIFT optimizes for information gain, reducing redundancy in the selected examples and enhancing the performance of LLMs during test-time fine-tuning. Evaluated on the Pile dataset, SIFT shows consistent improvement over traditional methods with minimal computational overhead, achieving a robust balance between relevance and diversity in data selection.

### Strengths
Effective Data Selection with SIFT: SIFT combines uncertainty and diversity to select non-redundant data, enhancing test-time fine-tuning efficiency compared to traditional methods.
Comprehensive Experiments: Wide experiments on the Pile dataset demonstrate SIFT’s effectiveness, consistently outperforming traditional Nearest Neighbor and other baseline methods in fine-tuning efficiency and model performance.

### Weaknesses
This paper has the following drawbacks.

**Complexity and Clarity of Method Presentation**: The paper’s explanation of the SIFT algorithm, especially on pages 4 and 5, could benefit from clearer descriptions and simplification of symbols. The complex notation and detailed mathematical formulation may obscure understanding for readers, especially those less familiar with active learning or information-theoretic approaches. Providing a more accessible walkthrough or visual aids could improve clarity.

**Sensitivity to Hyperparameters**: The method relies on certain hyperparameters, such as the regularization parameter (λ′), which can significantly impact SIFT’s performance. Techniques for automatic tuning or guidelines for parameter selection would make the approach more user-friendly and robust.

**Relevance and Diversity**. It is strange to mention that "we provide an example of how SIFT balances relevance and diversity, where we also see that the parameter"

**Limited Novelty in Uncertainty-Based Data Selection**: The use of uncertainty as a criterion for data selection is not a novel concept and is a common technique in active learning. Could the author explain the difference with existing works such as [1,2]. Furthermore, could you discuss the novelty of utilizing uncertainty in your LLM background?


[1] Symmetric Uncertainty-Aware Feature Transmission for Depth Super-Resolution,
[2] LogitNorm: Mitigating Neural Network Overconfidence with Logit Normalization

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
