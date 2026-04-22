# Process Supervision-Guided Policy Optimization for Code Generation

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 4, 6

## Abstract
Reinforcement learning (RL) with unit test feedback has enhanced large language models’ (LLMs) code generation, but relies on sparse rewards provided only after complete code evaluation, limiting learning efficiency and incremental improvements. When generated code fails all unit tests, no learning signal is received, hindering progress on complex tasks. To address this, we propose a Process Reward Model (PRM) that delivers dense, line-level feedback on code correctness during generation, mimicking human code refinement and providing immediate guidance. We explore various strategies for training PRMs and integrating them into the RL framework, finding that using PRMs both as dense rewards and for value function initialization significantly boosts performance. Our experimental results also highlight the effectiveness of PRMs in enhancing RL-driven code generation, especially for long-horizon scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the sparse-reward problem commonly encountered in RL-based code generation by proposing a Process Supervision-Guided Policy Optimization framework. The key idea is to introduce a PRM that provides dense, line-level feedback during generation, mimicking the human process of iterative code refinement and debugging. The authors evaluate the proposed approach on several benchmark datasets, demonstrating consistent improvements over standard RL baselines.

### Strengths
1. The paper introduces an automated binary search labeling strategy to generate line-level supervision and systematically explores how PRMs can be integrated into RL for code generation.
2. The experiments are comprehensive, covering multiple public and proprietary models across diverse datasets, offering solid empirical support.
3. The paper discusses key practical aspects such as PRM data quality and the risk of reward hacking in detail, providing useful implementation insights.

### Weaknesses
1. While the proposed approach is meaningful, the novelty appears somewhat Limited. Automatic process supervision, including automated data construction and process-level feedback, has been extensively explored in prior studies. A more systematic and quantitative comparison with these existing methods would help clarify the distinctive contributions of this work.
2. The paper offers an intuitive explanation of PRM as mimicking human coding behavior, but lacks theoretical depth—there is little mathematical analysis or exploration of why PRM effectively improves learning.
3. While PRM-based supervision improves performance, it may come at the cost of increased computational overhead. A clear analysis of the trade-off between performance gains and efficiency (in both data generation and training) is needed.
4. The lack of released code, datasets, and baseline models raises concerns about reproducibility and the credibility of the reported results.
5. Some experimental analyses (e.g., Table 2 and Figure 5) remain descriptive and lack deeper reasoning or ablation to explain observed trends.
6. Adding qualitative case studies to visually illustrate how PRM affects LLM code generation quality would strengthen the paper's interpretability and impact.

### Questions
1. Are the process supervision data generated from LLM outputs? If so, are they produced by the base model before RL optimization or by a model already trained with RL?
2. What are the theoretical and practical memory/runtime costs of the proposed framework? How do they compare to simpler unit-test-based reward models?
3. How crucial is the binary search procedure to the final performance? What happens if it is removed or replaced with another labeling strategy?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper is devoted to presentation of a methodology for RL finetuning of a language model for code generation.

Overall, I think the approach is interesting, but I have a few comments provided below.

I would like to add that the provided results for InHouse model are not informative to an external reader, since InHouse model is not described aside of vague size limit. I suggest to exclude them.

### Strengths
The paper presents a novel approach to finetune a LLM with RL for code generation.

### Weaknesses
I see two points here:

1) The authors train a PRM, which is compared to a baseline RL method where +1 reward is given to a code which passes *all* the unit tests, which given 0 otherwise. This baseline can be easily improved by giving a partial reward for some passes unit-tests. So in my opinion the current baseline is inadequately weak and should be replaced.

2) To train a PRM the authors use partial trajectories completed by an oracle. I am not sure about this approach, since the shorter prompt allows successful with more probability.

### Questions
Please, describe the procedure for PRM training data generation in more details.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a Process Reward Model (PRM) to address the sparse reward problem in reinforcement learning (RL) for code generation. The PRM provides dense, line-level feedback during generation, mimicking human step-by-step debugging. It is trained using binary search–based automatic labeling with unit tests and integrated into RL training as (1) dense intermediate rewards and (2) value function initialization. Experiments on HumanEval, MBPP, and LiveCodeBench show consistent improvements over standard RL baselines, especially in long-horizon code generation.

### Strengths
- The paper identifies a well-known issue , sparse reward signals in code generation RL , and provides a human-inspired solution via process-level feedback.

- The dual use of PRM as both a dense reward source and value function initializer is conceptually elegant and empirically validated. The binary search–based labeling algorithm is also efficient and interpretable.

- The experiments cover multiple datasets and models (Qwen2.5-7B, InHouse-Lite), with detailed ablations on reward shaping, data selection, and sample size.  The results are consistently positive and the figures (especially Figs. 4–5) clearly support the main claims.

- The paper is clearly written and well-structured, with helpful visualizations illustrating the PRM mechanism and the learning dynamics.

### Weaknesses
- The PRM data collection relies on K = 20 best-of-K completions per prefix and binary search per line, which makes the process computationally heavy.  The paper does not quantify this overhead (e.g., GPU hours, wall-clock time, or scaling behavior).  Without this information, the claimed “practical” pipeline seems questionable for larger-scale deployments.

- PRM labels depend entirely on unit-test coverage; when tests are incomplete, line-level feedback can be noisy or misleading.  The authors acknowledge this limitation but offer no robust noise-mitigation strategy beyond heuristic filtering.

- While the method works empirically, there is no theoretical justification for why PRM initialization improves credit assignment in PPO or stabilizes training. The mechanism remains largely heuristic.

- The paper introduces a manual weighting λ = 0.25/0.025 (Eq. 3) between PRM and outcome rewards but does not analyze sensitivity or potential gradient interference. It is unclear how these two reward sources interact during optimization.

### Questions
1. How large is the additional computational cost of PRM data generation as K or dataset size increases?  
2. How sensitive is performance to noisy or incomplete unit tests?  
3. Have you analyzed how λ in Eq. (3) affects stability or convergence?  
4. Could PRM rewards conflict with unit-test rewards (reward interference)?  
5. Does the method generalize to multi-file or cross-language code generation tasks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a process supervision-guided approach to improve reinforcement learning for code generation. The core idea is to train a process reward model (PRM) that predicts line-level correctness for code prefixes, using automated labels derived from a best of $K$ completion check and a binary search over code steps to find the first unrecoverable error. Experiments on HumanEval, MBPP, and LiveCodeBench report improvements for two base models, and analyses indicate gains are larger for long responses, that PRM helps exploration in a best of $K$ sense, and that careful PRM data selection matters. Limitations are noted in the conclusion.

### Strengths
The paper defines a concrete procedure to generate process labels by binary search with best of $K$ completions, which turns sparse unit test feedback into dense signals during generation. The algorithm and labeling rule are explicit and testable. The PRM is trained with a simple regression loss, which is stable and easy to reproduce. The integration points in RL are well chosen, with clear reward shaping weights and length normalization. The experiments include two base models and three benchmarks with Pass@1, with a table that separates dense reward, value initialization, and their combination. The best configuration improves MBPP from $62.4$ to $65.4$ for Qwen2.5 7B, and LiveCodeBench from $27.5$ to $30.1$, while InHouse Lite improves HumanEval from $65.1$ to $70.9$, MBPP from $61.9$ to $63.8$, and LiveCodeBench from $28.2$ to $29.8$. The paper also studies best of $K$ curves that rise by about $4$ points at $K=30$, the effect of response length where benefits appear for $>$100 tokens, and PRM data selection where using revised only responses is strongest. These are precise and actionable findings.

### Weaknesses
The paper relies on in house training data with about $30{\small,}000$ coding problems inside a broader RLHF set. Details on licenses, contamination checks, and overlap with evaluation sets are not given, so it is hard to judge fairness and generalization. Compute cost for PRM data collection and RL is not quantified. The best of $K$ completion step for labeling is a heuristic, and the paper does not report sensitivity to $K$ or to the sampling temperature during labeling. Reward shaping uses fixed $\lambda$ values $0.25$ and $0.025$. The paper does not show an ablation on these choices. Value initialization from PRM is motivated, yet details on the critic architecture, the initialization procedure, and the learning rate schedule are not provided. The reported gains on LiveCodeBench are modest, and confidence intervals are not reported. The mitigation for reward hacking covers length and comments. More adversarial code patterns such as try except wrappers that hide errors, dead code insertion, or misleading structure are not tested in the main text. These gaps limit the strength of the overall claim.

### Questions
Can the authors report how Pass@1 and PRM accuracy change when varying $K$ and the decoding temperature used during labeling, and include a small sensitivity plot to show the robustness of the labels?

Will the authors include an ablation over $\lambda$ values for both the failing case $\lambda=0.25$ and the passing case $\lambda=0.025$ to assess stability and to rule out over tuning to these settings?

Can the authors specify the exact procedure used to initialize the PPO value function from the PRM, for example, direct weight copy, distillation of $R_{\phi}(x, y_{\le t})$ into the critic, or a warm start followed by joint training, and provide a small ablation?

### Soundness
3

### Presentation
3

### Contribution
2
