# Human-LLM Collaborative Feature Engineering for Tabular Data

- Decision: Accept (Poster)
- Scores: 2, 6, 4, 6

## Abstract
Large language models (LLMs) are increasingly used to automate feature engineering in tabular learning. Given task-specific information, LLMs can propose diverse feature transformation operations to enhance downstream model performance. However, current approaches typically assign the LLM as a black-box optimizer, responsible for both proposing and selecting operations based solely on its internal heuristics, which often lack calibrated estimations of operation utility and consequently lead to repeated exploration of low-yield operations without a principled strategy for prioritizing promising directions. In this paper, we propose a human–LLM collaborative feature engineering framework for tabular learning. We begin by decoupling the transformation operation proposal and selection processes, where LLMs are used solely to generate operation candidates, while the selection is guided by explicitly modeling the utility and uncertainty of each proposed operation. Since accurate utility estimation can be difficult especially in the early rounds of feature engineering, we design a mechanism within the framework that selectively elicits and incorporates human expert preference feedback—comparing which operations are more promising—into the selection process to help identify more effective operations.
Our evaluations on both the synthetic study and the real user study demonstrate that the proposed framework improves feature engineering performance across a variety of tabular datasets and reduces users’ cognitive load during the feature engineering process.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes LLM-based feature engineering methods for tabular learning. Specifically, while letting LLM to explore operation candidates, the proposed method evaluates such generated operation rule by human experts, if available. While providing strong empirical evaluations, the authors also provide some theoretical aspects of the proposed method.

### Strengths
1. The authors provided strong evaluation results, and also provided some theoretical explanations of their approach.

2. Writing is easy to follow.

3. The evaluation results are based on multiple datasets.

### Weaknesses
1. What is the novel point of the proposed method, compared to the existing methods? I acknowledge that involving human experts might have some pros on doing feature engineering, and also agree that it will definitely improve the performance. However, still, involving human experts may cause some scalability issues. In other words, the authors provided the experimental results where the LLM-generated operations are evaluated by LLM itself. In this case, what advantages does the author's method have, compared to only evaluating target AI models (like MLP or XGBoost)?

2. How were the human experts selected? Did the authors act as a human expert? If not, how much did it cost to hire such human experts?

3. It would be better to show the cost analysis compared to baseline methods.

4. The datasets used in the experiments actually have just a few columns. If the table has millions of columns, is the proposed method scalable? Even in this case, is it possible to involve human experts?

5. How can such an approach be expanded when the column names do not exist? For example, for some financial datasets, the column names and values can be anonymized because of some privacy issues.

6. There are some missing citations. For example, P2T [1] is one of the notable work which leverages LLMs for tabular learning.

[1] Nam et al., Tabular Transfer Learning via Prompting LLMs, COLM 2024

### Questions
See Weaknesses.

### Soundness
3

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
The paper proposes a human-LLM collaborative framework for tabular learning in which the generation of candidate features and selection are decoupled for better automatic feature engineering. Specifically, LLMs are used solely for the generation of operation candidates, while a separate surrogate model guided by Bayesian optimization is used to model the utility and uncertainty of those candidates for selection. As accurate estimation of the utility can often be difficult, the framework selectively elicits input from human experts. Empirical results using GPT-4o as a simulated human demonstrate promising performance compared to prior work.

### Strengths
1. Clearly motivated framework. The authors rightly point out the conceptual limitation of existing LLM-based feature engineering approaches, which is that an LLM is used as a black-box optimizer for both proposing and selecting new features. Instead, the paper proposes using LLMs solely for generation with the selection guided by a Bayesian neural network as a surrogate model for utility and uncertainty estimation.

2. Principled human-in-the-loop mechanism. For more accurate estimate utility, the paper adds a mechanism to elicit human expert preference feedback. In order to make more effective use of human expertise, the propose mechanism selectively request human feedback based on the estimated potential for improved selection ("overlap" and "uncertainty").

### Weaknesses
1. Missing ablations on the backbone model. The authors primarily evaluate GPT-4o as the backbone model for LLM-based feature engineering methods. It would be useful to analyze how much the joint feature proposal and selection is an issue depending on the backbone model, e.g., comparing open vs. closed models, different model sizes, and reasoning-enabled vs. non-reasoning models.

2. Missing experiments with humans of varying levels of expertise. In the main experiments, GPT-4o is used as a simulated human -- which is likely comparable to a human expert -- in which case the method sees additional improvements compared to the version without it. It would be useful to better understand what level of human expertise (e.g., using noisy preference) is required to achieve these additional gains (or, potentially loss in performance with poor feedback).

3. Evaluation of scalability. It is unclear how much computational cost is involved in fitting BNN surrogate, calculating UCB values for every candidate, etc. Additional results on scalability with datasets of varying sample sizes and numbers of initial features would provide greater insight.

### Questions
Q. Have the authors experimented with alternative embeddings for the surrogate model before deciding on $[\phi_\text{embedding}, \phi_\text{column}]$?

Q. How does the method scale to datasets with varying numbers of initial samples and features?

### Soundness
2

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
2

### Summary
This paper addresses the limitations of existing feature engineering methods for tabular learning and proposes a Human-LLM collaborative feature engineering framework.The framework decouples the "operation proposal" and "operation selection" processes: LLMs only generate diverse feature transformation candidates based on task understanding, while selection is guided by explicit modeling of the utility and uncertainty of each candidate. For scenarios where utility estimation is inaccurate (especially in early iterations), the framework designs a selective human feedback mechanism—it elicits human experts’ pairwise preference feedback (comparing which operations are more promising) only when the potential gain of feedback outweighs the cognitive cost, thereby improving selection accuracy .In terms of technical implementation, a Bayesian Neural Network (BNN) is used as a surrogate model to approximate the black-box utility function of features, and the Upper Confidence Bound (UCB) strategy is adopted to balance exploitation and exploration when human feedback is unavailable.

### Strengths
（1）Innovative Method Design with Clear Targets:The core idea of decoupling LLM’s "operation proposal" and "operation selection" effectively addresses the key limitation of existing LLM-powered methods (i.e., LLMs acting as black-box optimizers). By introducing explicit utility and uncertainty modeling, the framework avoids blind exploration of low-yield operations, and the selective human feedback mechanism balances the value of human expertise and cognitive cost.
（2）Comprehensive Experimental Validation and Multi-Dimensional Evaluation:Diverse Dataset Coverage: The experiments include public datasets (Kaggle, UCI) and a proprietary enterprise dataset, avoiding overfitting to public data biases and enhancing the generalization of results.In addition to downstream model performance (AUROC for classification, normalized RMSE for regression), the study incorporates a user study using the NASA-TLX scale to measure cognitive load, which fully reflects the framework’s practical value in human-AI collaboration scenarios.
（3）Technical Implementation and Transparency:The paper provides clear mathematical formulations for core components (e.g., utility function, surrogate model training objective, UCB calculation), and details key parameter settings. Appendices also include dataset descriptions, prompt templates, and proof of technical lemmas, which is conducive to reproducibility.

### Weaknesses
（1）Limitation in Human Feedback Simulation: The "w/ Human" setting in the experiment uses GPT-4o to simulate human experts, rather than recruiting real domain experts for feedback. This may deviate from the actual scenario where human experts rely on domain experience to make judgments, and the authenticity of feedback needs to be further verified.
（2）Insufficient Discussion on Scalability: The paper does not discuss the framework’s performance in ultra-large-scale tabular data scenarios. The BNN surrogate model may face efficiency challenges in high-dimensional feature spaces, and the cost of LLM generating operation candidates may increase significantly.
（3）Lack of Analysis on LLM Candidate Quality: The framework assumes that LLMs can generate high-quality operation candidates, but does not analyze the impact of LLM performance differences (e.g., GPT-4o vs. GPT-3.5, open-source LLMs like LLaMA) on the framework’s final effect. It also does not discuss how to handle low-quality candidates generated by LLMs.

### Questions
（1）This paper uses GPT-4o to replace expert scoring. Has consideration been given to how the preferences simulated by GPT-4o differ from those of real domain experts in terms of consistency, noise distribution, and preference biases?
Can the performance gains observed under simulated feedback be replicated with real human feedback? How do feedback latency and fatigue affect the trigger strategy (C1/C2)? Is the cognitive cost threshold γ=4 based on real measurements? How does this threshold vary across different expert groups?
（2）Why choose BNN over (sparse) GP, lightweight ensemble, or incremental update strategies? Is uncertainty estimation under high-dimensional ϕ(e) reliable for BNN?
（3）How sensitive is the framework to different generators (GPT-4o vs GPT-3.5 vs open-source LLMs)? Will low-quality LLM-generated candidates drag down the entire selector?The paper could consider candidate quality metrics such as duplication rate, similarity to existing features, and potential label leakage, and analyze the correlation between these metrics and final performance.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores the use of Bayesian neural networks (BNN) to explicitly model the utility of feature transformations models in LLM-based feature engineering. The LLM serves as a sampler of candidate feature transformations that are then selected using a UCB bandit based on utility predictions. The authors further introduce an approach to incorporating human preference feedback in utility modeling. They demonstrate through experiments that the proposed framework improves feature engineering performance and reduces users’ cognitive load.

### Strengths
I think the paper suggests a valid approach to LLM-based feature engineering. The use of utility models helps selectively evaluate promising features proposed by the LLM and thus can reduce the computation cost of feature evaluations. The paper also presents an approach to incorporating human preference feedback in utility modeling. The mathematical rationale of the framework is well explained.

### Weaknesses
It would be appreciated if the authors could provide further details on the setting of BNNs and the optimization algorithms for solving Equations (5) and (17).

Some experimental details are missing. The standard errors across repeated runs are not provided. It is not stated how the parameters of downstream models have been selected, which could have an impact on the performance. 

The experiments could evaluate other LLM backbones in addition to GPT-4o. It would be great to also include a study on the LLM generation cost and feature evaluation cost of the framework.

### Questions
There may be typos in line 194 and Equation (6) on $q_t(\theta))$.

I wonder why the right-hand side of inequality (11) does not involve $\mu_t(e^a_t)$ and $\mu_t(e^b_t)$.

### Soundness
3

### Presentation
3

### Contribution
3
