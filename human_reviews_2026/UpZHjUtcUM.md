# Erase to Improve: Erasable Reinforcement Learning for Search-Augmented LLMs

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
While search-augmented large language models (LLMs) exhibit impressive capabilities, their reliability in complex multi-hop reasoning remains limited. This limitation arises from three fundamental challenges: decomposition errors, where tasks are incorrectly broken down; retrieval missing, where key evidence fails to be retrieved; and reasoning errors, where flawed logic propagates through the reasoning chain. A single failure in any of these stages can derail the final answer. We propose Erasable Reinforcement Learning (ERL), a novel framework that transforms fragile reasoning into a robust process. ERL explicitly identifies faulty steps, erases them, and regenerates reasoning in place. This targeted correction mechanism turns brittle reasoning into a more resilient process. Models trained with ERL, termed ESearch, achieve substantial improvements on HotpotQA, MuSiQue, 2Wiki, and Bamboogle, with the 3B model achieving +8.48% EM and +11.56% F1, and the 7B model achieving +5.38% EM and +7.22% F1 over previous state-of-the-art(SOTA) results. These findings suggest that erasable reinforcement learning provides a powerful paradigm shift for robust multi-step reasoning in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Erasable Reinforcement Learning (ERL), a new framework for enhancing the robustness of search-augmented LLMs in multi-hop reasoning tasks. The key idea is to explicitly identify and erase faulty reasoning or retrieval steps within a reasoning trajectory, then regenerate subsequent steps from the last correct state. The method introduces three erasure mechanisms (plan, search, and sub-answer erasure) triggered by reward thresholds on intermediate steps. ERL is trained using Qwen2.5-3B/7B models, evaluated on different benchmarks. Experiments show consistent SOTA gains and improved stability over PPO/GRPO. Ablation studies confirm the complementary roles of the three erasure modules, with sub-answer erasure contributing most.

### Strengths
1. The “erasable” mechanism introduces a fresh perspective distinct from prior step-wise or reward-shaping RL methods. It explicitly operationalizes fault detection and local correction, which is both intuitively appealing and empirically validated.

2. The multi-component reward (for search, sub-answer, and final answer) effectively mitigates the sparsity of typical RL signals, and the formulation is clear and well-motivated.

3. The component-wise ablation and comparison with PPO/GRPO support the claim that ERL provides stable optimization and interpretable corrective behavior.

### Weaknesses
1. The iterative erase-and-regenerate cycle introduces non-trivial computational overhead, which the paper acknowledges but does not quantify. 

2. The erasure triggers and dense-reward coefficients appear manually tuned; there’s no sensitivity analysis showing robustness to these hyperparameters.

3. The evaluation is limited to search QA-style benchmarks. It remains unclear whether ERL generalizes to other agentic reasoning tasks (e.g., math, tool use, code reasoning).

4. More clearer illustrative examples or reasoning traces would make the method more intuitive to follow.

### Questions
1. What is the training cost relative to PPO/GRPO baselines (e.g., GPU hours given the same steps)? 

2. How sensitive is ERL to the choice of thresholds hyperparameters? 

3. Have you tried applying ERL to other reasoning-centric tasks, such as math or coding?

4. Are there qualitative/quantitive analysis or visualizations showing where erasure occurs in the reasoning chain, and how it affects the final outcome?

### Soundness
3

### Presentation
3

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
This paper proposes Erasable Reinforcement Learning (ERL), a novel framework designed to enhance the robustness of multi-hop reasoning in search-augmented LLMs. The key insight is that current search-augmented LLMs suffer from decomposition, retrieval, and reasoning errors, where a single failure can derail the entire reasoning chain. ERL addresses this by introducing an erasure mechanism that detects faulty reasoning steps, removes them, and regenerates new reasoning from the last correct state.

### Strengths
1. The idea of erasable reasoning is inspired by human-like self-correction, effectively addressing a key weakness in multi-hop reasoning systems.

2. ERL demonstrates consistent and notable improvements over strong baselines across multiple datasets.

3. The paper provides thorough ablation studies (plan/search/sub-answer erasure), comparisons with PPO and GRPO, and both offline and online evaluations.

4. The writing is clear and mathematically rigorous, with precise formulations of the MDP setup, reward decomposition, and erasure operators.

### Weaknesses
While novel in its integration, ERL primarily extends existing reinforcement learning frameworks with a modular erasure component. The conceptual depth may be viewed as incremental rather than fundamentally new.

### Questions
1. How does ERL’s computational cost compare to PPO or GRPO during training?
2. Can ERL generalize beyond QA tasks, for example, to long-form reasoning or tool-using agents? If so, what modifications would be required?
3. How sensitive is ERL to the thresholds ($\alpha$, $\beta$) used to trigger erasures? 
4. How does ERL scale with reasoning depth (e.g., beyond five hops)? 
5. In real-world settings without gold evidence, how could dense rewards such as $R_{search}$ and $R_{subanswer}$ be approximated?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes Erasable Reinforcement Learning (ERL) for search-augmented LLM agents: the agent detects faulty steps in multi-hop reasoning (decomposition, retrieval, or reasoning), reverts those segments, and regenerates them in-place with dense stepwise rewards, reducing error propagation. On HotpotQA, MuSiQue, 2Wiki, and Bamboogle, ERL-trained models beat strong baselines as new SOTA in both offline and online settings

### Strengths
* Novel training strategy: Introduces *Erasable RL* with three erasure signals—plan/init search, subsequent search, and sub-answer—to cut off faulty steps and retry, directly targeting error propagation.  
* Dense, well-aligned rewards: Designs stepwise rewards for retrieval and intermediate reasoning, with token-level attribution, improving credit assignment beyond sparse final answers.  
* Strong empirical results & diagnostics: The experiments shows the proposed approach achieves SOTA across four multi-hop QA datasets in both offline and online settings, with consistent gains at 3B/7B and ablations showing each erasure component is complementary.

### Weaknesses
- Unclear implementation details. Which dataset is used for training? And also, it is unclear on how the grounding gold evidence and gold sub-answers set is constructed. Furthermore, the dependence on gold set may be infessible for many real-world applications, unless a general and lightweight approach can be provided.
- Concerns on efficiency and scalability : The erase–regenerate loop adds computational overhead and may require repeated passes when multiple heterogeneous errors occur; training notes also show retrieval quality is bottlenecked by a local Wiki-18 setup, suggesting a practical bottleneck beyond the algorithm itself.  Also for real-world challenge setup, it is likely the search cannot provide answer, which means the model will struck in the loop, unless the training data and search resource ensure valid information can be provided.
- Writing issue, e.g. L160 Jin et al. (2025b) > (Jin et al., 2025b). These issue occurs very frequent spanning intro, sec 4.1, and related work.

### Questions
- I would expect longer rollout time and more research queries thus make it both cost more and takes a longer time for each unit update/iteration, but I am not sure about the overall effect.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses critical reliability issues faced by search-augmented LLMs when performing complex multi-hop reasoning tasks. It identifies three primary failure modes: decomposition errors, retrieval errors, and reasoning errors. To mitigate these issues, the authors propose ERL, a framework that identifies, erases, and regenerates faulty reasoning steps, thereby preventing error propagation. Extensive experiments on multi-benchmarks demonstrate substantial performance improvements with ERL, achieving new state-of-the-art results.

### Strengths
- The paper introduces ERL, a novel approach to addressing the inherent brittleness in search-augmented LLMs. The key innovation lies in explicitly identifying, erasing, and regenerating faulty reasoning steps, a method previously unexplored in multi-hop reasoning contexts. This creative combination of reinforcement learning with targeted error correction significantly advances existing frameworks.

- Validates the framework across multiple challenging multi-hop reasoning datasets, HotpotQA, MuSiQue, 2Wiki, and Bamboogle, demonstrating generalizability.

- This framework enhances the robustness and real-world applicability of search-augmented language models, delivering state-of-the-art results on multiple benchmarks. Its dynamic error-correction capability ensures improved reliability and adaptability in practical scenarios.

### Weaknesses
- The main results are point estimates without confidence intervals, so it’s unclear whether reported gains exceed noise.

- The manuscript does not present empirical measurements or benchmarks of runtime, memory use, or computational resource demands. The lack of quantitative efficiency analysis weakens the practical argument for real-world applications.

- Certain equations contain notation ambiguities or inconsistencies, such as unclear definitions or insufficient explanations for symbols.

### Questions
Please check the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
