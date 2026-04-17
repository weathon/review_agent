# COSMOS: Are Performance–Cost Tradeoffs Predictable in Model--Strategy Selection?

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Large language models (LLMs) achieve excellent performance across numerous tasks by using a diverse array of adaptation strategies. However, selecting the optimal combination of model, strategy, and configuration under resource constraints is challenging and typically requires extensive experimentation. We ask whether it is possible to accurately predict both downstream performance and cost without running expensive adaptation trials. We introduce COSMOS, a general analysis framework for approaching this strategy selection problem through low-cost prediction of adaptation outcomes. We instantiate our framework via a pair of powerful predictors: embedding-augmented lightweight proxy models to predict fine-tuning performance, and low-sample scaling laws to forecast retrieval-augmented in-context learning. Evaluations across eight representative benchmarks demonstrate that COSMOS instantiations achieve high prediction accuracy while reducing computational costs by 92.72% on average, and up to 98.71% in resource-intensive scenarios. Our results show that efficient prediction of adaptation outcomes is possible, enabling practitioners to navigate large model--strategy--configuration spaces efficiently and substantially reduce deployment cost while maintaining strong task performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces COSMOS, a unified framework for predicting both performance and cost of LLM adaptation strategies without expensive exhaustive experimentation. The authors instantiate COSMOS with two complementary approaches: (1) embedding-augmented linear proxy models for QLoRA fine-tuning prediction, and (2) scaling law-based extrapolation for retrieval-augmented in-context learning. Experiments across 8 benchmarks demonstrate 92.72% average cost reduction with 1.09% mean absolute error in performance prediction.

### Strengths
1. The empirical study is extensive, covering diverse tasks (general reasoning and domain-specific financial tasks), multiple models (Gemma, Llama, Qwen, Mistral), and systematic configuration spaces.
2. The cost reduction (up to 98.71% in some scenarios) while maintaining prediction accuracy (MAE of 1.09%) is impressive and practically significant.
3. COSMOS provides a general abstraction that can accommodate different adaptation strategies and predictors, making it extensible.

### Weaknesses
1. While the paper claims to address a "multi-strategy" setting, the evaluation primarily focuses on only two strategies: QLoRA fine-tuning and retrieval-augmented ICL. How does COSMOS handle more complex strategies like Chain-of-Thought prompting variations, tree-search methods, or hybrid approaches mentioned in Section 2.
2. The paper makes specific architectural choices without sufficient justification: Why use bidirectional embeddings for causal language models in fine-tuning prediction? For ICL, why assume exponential saturation specifically? Have other functional forms been explored?
3. While Appendix G provides comparisons with search-based methods and scaling-law predictors, more recent methods could be included.

### Questions
1. How does COSMOS handle newly released models not in the training set for predictors?
2. Can the framework predict when fine-tuning might hurt performance (negative transfer)?
3. What is the computational cost of the prediction framework itself at scale?
4. How do predictions degrade when task distribution differs significantly from training data?
5. The work is limited to models in the 2B-9B parameter range. Could you discuss whether the outperformance of COSMOS holds for frontier-scale models (llama 13B, QwQ-32B)?
6. Two highly relevant recent works should be included: (a) LENSLLM: Unveiling Fine-Tuning Dynamics for LLM Selection and (b) EvoSLD: Automated Neural Scaling Law Discovery With Large Language Models.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces COSMOS, a unified framework that aims to predict the performance and cost trade-offs of various model and adaptation strategy combinations when deploying large language models (LLMs). Given the complexity of selecting models and adaptation strategies under resource constraints, the paper proposes a system to efficiently predict both performance and cost, thereby reducing the need for expensive trial-and-error experimentation. COSMOS is evaluated through two major adaptation strategies: QLoRA fine-tuning and retrieval-augmented in-context learning (ICL). The results show that COSMOS can accurately predict performance outcomes with high accuracy while reducing computational costs by up to 98.7%.

### Strengths
The paper introduces an interesting approach to reduce the computational costs of model and strategy selection by leveraging performance prediction. This is valuable since running full-scale experiments for every model and strategy combination can be prohibitively expensive. The idea of predicting performance and cost trade-offs without the need for exhaustive testing can save substantial resources. Additionally, the framework is clearly presented, and the experiments, though limited, demonstrate that the COSMOS method is able to predict performance effectively in the cases of QLoRA and ICL. For simpler tasks, the method shows promise in delivering high-quality predictions efficiently.

### Weaknesses
One significant weakness of the proposed framework is the over-simplification of the predictive models used for performance estimation. In the case of QLoRA fine-tuning, the use of a linear proxy model to predict performance may not be sufficiently accurate, especially for more complex tasks. The relationship between adaptation strategies (like QLoRA) and performance is often non-linear, involving interactions between hyperparameters such as learning rate and number of training steps. A linear model, despite its computational efficiency, fails to capture these intricate dynamics, leading to potential inaccuracies in performance predictions.

Similarly, for retrieval-augmented in-context learning (ICL), the scaling law-based prediction method relies on the assumption that performance improves linearly with the number of examples (shots). However, this assumption may not hold true in many real-world tasks. The effectiveness of ICL often depends on factors such as the quality of retrieved examples, the complexity of the task, and the nature of the query context. These factors are not adequately captured by a simple scaling law, which could lead to overestimations or underestimations of ICL's performance, particularly in complex scenarios.

Moreover, the experimental validation is primarily focused on QLoRA and ICL, which severely limits the generalizability of the framework. The framework's performance on other adaptation strategies or more diverse tasks remains unexplored. Without testing on a wider variety of models, strategies, and tasks, it is difficult to assess the framework's robustness and scalability. This narrow experimental scope reduces confidence in the framework's ability to generalize across different settings.

Additionally, both methods rely on relatively small-scale data for calibration and validation. Given that the predictions are based on proxy models with limited training and validation sets, there is a significant risk of overfitting. If the models are trained on a narrow range of examples, they may not generalize well to unseen tasks or larger models, further undermining the reliability of predictions.

### Questions
The experiments focus primarily on QLoRA and ICL. How well does COSMOS generalize to other adaptation strategies, have you tested the framework on a broader range of adaptation strategies and tasks?
Given the reliance on a linear proxy model, how accurate are the performance predictions when applied to tasks with high complexity or large models? Are there any plans to explore more advanced prediction methods to improve the accuracy of predictions?
How does COSMOS handle more complex adaptation strategies or tasks such as those seen in large-scale model fine-tuning? Can the framework be extended to handle these more complex relationships?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses the "strategy selection problem": the challenge of choosing the optimal combination of a large language model and an adaptation strategy (e.g., fine-tuning, in-context learning) to balance performance and cost for a specific task. Exhaustively evaluating all combinations is computationally prohibitive. The authors propose COSMOS, a unified prediction framework designed to efficiently estimate the performance and cost of various model-strategy pairs at a minimal cost. The paper instantiates and studies this framework in two popular scenarios: 1) predicting QLORA fine-tuning performance using an embedding-augmented lightweight proxy model (akin to linear probing), and 2) predicting retrieval-augmented in-context learning (ICL) performance using low-sample scaling laws (an exponential saturation curve).

### Strengths
The paper effectively formalizes the "strategy selection problem", which is a practical and important challenge for researchers and practitioners.

### Weaknesses
The core technical contribution of the paper feels limited in its originality. The two prediction methods instantiated within the COSMOS framework are essentially linear probing on frozen embeddings and fitting a simple exponential scaling law. These are both well-established, non-novel techniques. The contribution is thus an *application* of these methods rather than the development of a new prediction framework.

The choice of experimental scenarios significantly limits the practical impact of the findings. The paper focuses on Q-LoRA and retrieval-augmented ICL, which are already low-cost adaptation strategies. The real-world "strategy selection problem" is most painful when resource-intensive options like full fine-tuning are on the table. The paper motivates this with a GPT-4o case study in Appendix K but provides no empirical validation for this high-stakes, more practical scenario. It is unclear if the linear proxy predictor, which relies on *frozen* embeddings, would be at all predictive for full fine-tuning, where the embeddings themselves are updated.

The formalization of the cost analysis framework in Section 3.3 is overly complex and not well-motivated. By trying to create a unified model based on fluctuating real-world prices, the framework lacks principled stability. A more robust analysis based on standardized computational units, such as FLOPs, would be more generalizable and principled, even if it's harder to map to a direct dollar amount.

The related work section on scaling laws and performance prediction is incomplete. It overlooks several recent and highly relevant papers that also focus on general-purpose performance prediction for model selection using efficient trials (e.g., [1,2,3]). A discussion of these works is necessary to properly contextualize the novelty and contribution of COSMOS.

[1] [Selecting large language model to fine-tune via rectified scaling law](https://arxiv.org/abs/2402.02314)

[2] [Mordal: Automated Pretrained Model Selection for Vision Language Models](https://arxiv.org/abs/2502.00241)

[3] [LENSLLM: Unveiling Fine-Tuning Dynamics for LLM Selection](https://arxiv.org/abs/2505.03793)

### Questions
1. Could you provide a simpler, more intuitive explanation of the cost framework? Why was this complex, price-based model chosen over a more standardized and stable one like total FLOPs? How do you account for the high volatility of the prices you use (e.g., GPU spot markets on Vast.ai) in a generalizable framework?
2. Your key insight for ICL prediction is that its performance "typically follows an exponential saturation curve". What is the empirical or theoretical basis for this specific functional form? This is a foundational assumption for half of your paper's experimental validation, and it is presented without supporting evidence or citation.
3. Your paper is motivated by the high cost of strategies like full fine-tuning (as shown in Appendix K), yet your experiments are limited to the low-cost QLORA strategy. How confident are you that your linear proxy predictor, which is trained on *frozen* embeddings, would remain accurate for predicting the performance of *full* fine-tuning, a process that fundamentally alters those very embeddings?
4. Could you please discuss the relationship between COSMOS and recent, directly relevant works on efficient model selection and performance prediction, such as "Selecting large language model to fine-tune via rectified scaling law" (Li et al., 2024), "Mordal: Automated Pretrained Model Selection for Vision Language Models" (Deng et al., 2024), and "LENSLLM: Unveiling Fine-Tuning Dynamics for LLM Selection" (Shao et al., 2024)? These papers seem to address a very similar problem, and their omission is a notable gap.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Summary: 
This manuscript addresses the relevant and practical challenge of selecting the optimal Large Language Model (LLM) model and adaptation strategy under budget constraints. The proposed COSMOS framework offers a structured approach via predictive modeling, demonstrating significant cost savings and high prediction accuracy in its instantiations for QLoRA and retrieval-augmented ICL. The study is strengthened by its formalization of the problem and extensive experiments across diverse tasks.

### Strengths
1. The paper clearly formalizes the complex 'strategy selection problem' for LLMs in a multi-dimensional setting (model, strategy, configuration, task, cost).
2. COSMOS is proposed as a unified framework for joint performance and cost prediction across adaptation types.
3. Experiments cover eight benchmarks, spanning general domains and specialized areas like finance.

### Weaknesses
1. Limited model and strategy diversity. While COSMOS is positioned as a framework, the core empirical validation primarily relies on two models (Gemma 2B and Llama 3 8B), lacking in-depth exploration of model scale and type. The investigation into different adaptation strategies is also very limited.
2. The QLoRA predictor requires calibration on a small validation set. While the cost is included, the sensitivity of prediction accuracy to the size and composition of this calibration set is not explored. The QLoRA proxy predictor relies on a small validation set (e.g., 10% of data or ~200 examples) for affine calibration. This introduces potential brittleness: small or biased samples may lead to inaccurate performance predictions and suboptimal strategy selection. The stability of the predictor across different tasks, model scales, and validation set sizes remains unclear. Sensitivity analyses—varying validation size, sampling strategy, and dataset distribution—are needed to support the claimed robustness and ensure reliable strategy recommendations.
3. The paper primarily evaluates predictors for QLoRA and ICL. Hybrid strategies are mentioned, but the capability of COSMOS to predict performance and cost outcomes when multiple strategies are combined (applying ICL after QLoRA fine-tuning) is not demonstrated or evaluated.

4. Limited Generalization Beyond Evaluated Strategies. COSMOS is proposed as a general framework but is only evaluated on QLoRA and retrieval-based ICL. It remains unclear how easily the approach extends to other common adaptations (e.g., LoRA, full fine-tuning, adapter modules, dense/ANN retrieval, LM-based prompting pipelines, ensembling) and whether additional predictor design is required.

5. Robustness of ICL Scaling-Law Extrapolation. The ICL predictor often fits an exponential-saturation model with only two points (1-shot and 8-shot). Noisy early measurements—due to retrieval failures, query heterogeneity, or non-monotonic performance—can produce unstable extrapolations and poor strategy choices, undermining reliability in real-world noisy environments.

6. The evaluation focuses on multiple-choice/classification-style tasks; there is no evidence COSMOS predicts performance–cost for open-ended generation, code, long-context reasoning, or tool-use.

### Questions
1. How straightforward is it to add a new adaptation strategy (e.g., LoRA) to COSMOS? Please provide the minimal data and compute requirements to construct a new strategy-specific predictor.

2. How sensitive are the QLoRA predictor calibrations to validation set size and distribution? Is there systematic bias when the validation set is small or has distribution shifts?

3. In Table 1, several predicted values are exactly the same as the actual values when using the MAE error metric. Could this be an issue related to the choice of precision?

4. Table captions in the manuscript are incorrectly placed below the tables (a typesetting error).

### Soundness
2

### Presentation
2

### Contribution
2
