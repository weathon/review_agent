# CPL: Critical Plan Step Learning Boosts LLM Generalization in Reasoning Tasks

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 5, 3, 6

## Abstract
Post-training, particularly reinforcement learning (RL) using self-play-generated data, has become a new learning paradigm for large language models (LLMs). However, scaling RL to develop a general reasoner remains a research challenge, as existing methods focus on task-specific reasoning without adequately addressing generalization across a broader range of tasks. Moreover, unlike traditional RL with limited action space, LLMs operate in an infinite space, making it crucial to search for valuable and diverse strategies to solve problems effectively.
To address this, we propose searching within the action space on high-level abstract plans to enhance model generalization and introduce Critical Plan Step Learning (CPL), comprising: 1) searching on plan, using Monte Carlo Tree Search (MCTS) to explore diverse plan steps in multi-step reasoning tasks, and 2) learning critical plan steps through Step-level Advantage Preference Optimization (Step-APO), which integrates advantage estimates for step preference obtained via MCTS into Direct Preference Optimization (DPO). This combination helps the model effectively learn critical plan steps, enhancing both reasoning capabilities and generalization.
Experimental results demonstrate that our method, trained exclusively on GSM8K and MATH, not only significantly improves performance on GSM8K (+10.5\%) and MATH (+6.5\%), but also enhances out-of-domain reasoning benchmarks, such as HumanEval (+12.2\%), GPQA (+8.6\%), ARC-C (+4.0\%), MMLU-STEM (+2.2\%), and BBH (+1.8\%). The code is available at https://anonymous.4open.science/r/CPL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Critical Plan Step Learning (CPL), a novel approach for enhancing the generalization of large language models (LLMs) in reasoning tasks by leveraging plan-based reinforcement learning (RL). CPL employs Monte Carlo Tree Search (MCTS) to explore high-level, abstract problem-solving strategies, distinguishing critical steps that improve reasoning. Experimental results show that CPL significantly boosts in-domain and out-of-domain task performance, underscoring its potential for broader LLM applications.

### Strengths
- Originality: The paper presents Critical Plan Step Learning (CPL), which focuses on identifying and learning critical steps in high-level abstract plans for generalization in reasoning tasks, a novel concept in reinforcement learning for LLMs.

- Quality: The paper includes clear figures and comprehensive experiments to illustrate the effectiveness of CPL.

- Significance: CPL demonstrates improvements over baseline models across both in-domain (e.g., GSM8K, MATH) and out-of-domain tasks (e.g., HumanEval, ARC-C).

- Clarity: The paper is well-organized and accessible, with clear descriptions of CPL’s components and step-by-step explanations of the methods, making complex concepts understandable.

### Weaknesses
- In the contributions and conclusion, you mentioned "critical steps", while critical steps in this paper is not well-defined theoretically or experimentally. This undermines the interpretability of the process and raises questions about the reproducibility and consistency of its critical steps across different tasks.
- The model label in Section 3.2 is unclear. What are CPL(Round2 SFT) and CPL-final respectively? I would think CPL(Round2 SFT) is the model after Round2 SFT but it seems like the performance of CPL(Round2 SFT) drops and is lower than the previous steps for some tasks.
- The iterative training with MCTS and Step-APO requires multiple rounds of data generation, which can be prohibitively resource-intensive. There is little discussion on scalability or the potential trade-offs in model efficiency, which are crucial for real-world deployment.

### Questions
1. As mentioned in Weakness, the performance of CPL(Round2 SFT) drops. If that denotes the model after Round2-SFT, could the authors provide insights into why the Round 2 SFT performance declines compared to Round 1?
2. GSM8K and MATH are both math-centric. Will the result be different if the in-domain tasks change? For example, HumanEval is used as in-domain tasks, and MATH and GSM8K as out-of-domain tasks. 
3. Why was MCTS chosen over other search methods for exploring high-level plans? Were any alternative search techniques considered? Providing an ablation study about the search methods will be helpful.
4. Have you compared CPL with general SOTA LLMs, especially the ones with chain-of-thought/tree-of-thought prompting or related techniques?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces Critical Plan Step Learning (CPL), a novel approach to improve large language models' reasoning abilities through reinforcement learning. The method uniquely combines Monte Carlo Tree Search for exploring diverse reasoning strategies with Step-level Advantage Preference Optimization for learning critical planning steps. While trained only on mathematical datasets (GSM8K and MATH), CPL demonstrates significant improvements not only on these training domains but also generalizes well to various out-of-domain reasoning tasks, including programming, physics, and general STEM problems. This suggests CPL effectively enhances models' general reasoning capabilities rather than just task-specific performance.

### Strengths
- The paper proposes a novel plan-based MCTS simulation data construction method introduces a step-level APO algorithm
- Comprehensive experimental validation is conducted across various reasoning datasets
- The paper is well-written with clear structure and organization

### Weaknesses
- Using MCTS to sample high-quality data for fine-tuning is not a novel contribution, as [1] has already proposed a self-improvement method based on MCTS.
- The methodology description lacks clarity, particularly regarding how the plan is integrated into the MCTS process.
- The comparison with baselines is insufficient, as it lacks search-based algorithms such as [1][2] in the baseline comparisons.

[1]AlphaZero-Like Tree-Search can Guide Large Language Model Decoding and Training

[2]Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing

### Questions
- Could the authors provide more method details, particularly about how MCTS rewards are calculated and whether there is an additional reward model? Perhaps include a flowchart illustrating plan-based MCTS?
- Training-based methods consume substantial computational resources. Could you elaborate on the differences between training-based and testing-based approaches, as mentioned in [1][2]? If minimal resources are required at test time, is RL fine-tuning necessary?
- Please provide more results for different sampling methods in Section 3.3. The authors only compared with CoT methods - how does it compare with MCTS without planning, Tree of Thoughts, Q*[1], and other similar methods?

[1]:Q*: Improving Multi-step Reasoning for LLMs with Deliberative Planning

[2]:Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper introduces Critical Plan Step Learning (CPL), a method designed to improve the generalization of large language models (LLMs) in multi-step reasoning tasks. The authors propose a novel approach that focuses on training LLMs through high-level abstract plans rather than task-specific actions. Their method consists of two main components: (1) using Monte Carlo Tree Search (MCTS) to explore diverse plan steps in multi-step tasks, and (2) introducing Step-level Advantage Preference Optimization (Step-APO), which refines the learning of critical plan steps by integrating advantage estimates from MCTS into Direct Preference Optimization (DPO). Experimental results show that the CPL method enhances both in-domain and out-of-domain reasoning capabilities across a variety of benchmarks, achieving significant improvements in datasets like GSM8K, MATH, HumanEval, and ARC-C.

### Strengths
- Approach: The paper proposes a method of focusing on high-level abstract plans instead of task-specific solutions, which helps improve model generalization across different reasoning tasks.
- Empirical Results: The experimental results showing significant improvements in both in-domain (GSM8K, MATH) and out-of-domain (HumanEval, ARC-C) reasoning benchmarks.
- Step-APO: The introduction of Step-APO, which integrates advantage estimates into DPO, provides a fine-grained optimization of critical plan steps.

### Weaknesses
The paper does not clearly define high-level abstract plans. How should one distinguish between abstract plans and task-specific plans? What are the differences between this method and the chain/tree/graph-of-thoughts approach? Additionally, the process of generating abstract plans is not described in detail, nor does the paper compare its method with existing works that use thoughts to enhance models' general reasoning abilities [1-2]. Furthermore, regarding Step-APO, several existing works have proposed very similar ideas [3-5], but the paper neither cites, discusses, nor compares its approach with these works. Overall, this work feels incomplete.

--- 

[1] Zelikman, Eric, et al. "Star: Bootstrapping reasoning with reasoning." NeurIPS 2022.

[2] Zelikman, Eric, et al. "Quiet-star: Language models can teach themselves to think before speaking." arXiv preprint arXiv:2403.09629 (2024).

[3] Rafailov, Rafael, et al. "From $ r $ to $ Q^* $: Your Language Model is Secretly a Q-Function." arXiv preprint arXiv:2404.12358 (2024).

[4] Zhong, Han, et al. "Dpo meets ppo: Reinforced token optimization for rlhf." arXiv preprint arXiv:2404.18922 (2024).

[5] Zeng, Yongcheng, et al. "Token-level Direct Preference Optimization." ICML 2024.

### Questions
See the Weaknesses section.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes critical plan step learning method. The CPL approach does MCTS on the abstract level plan and develop diverse plans. And then they incorporate step level advantage preference optimization to learn policy effectively. Similar to DPO, they derive a step level APO objective. They demonstrate that the proposed algorithm improve the base model performance considerably on the in domain as well as out of domain tasks.

### Strengths
- Proposes a method for learning high level actions which author argue is good for generalization beyond in domain tasks
- The proposed algorithm is interesting and demonstrate improvement upon the base model performance.
- Their ablation on section 3.4 is insightful which shows improvement due to step-APO.

### Weaknesses
- LLM’s are quite stochastic how ever the paper only reports a single number in plots and table which does not provide a clear comparison across methods. I would appreciate if they could compare distribution (e.g., variance or so as well)
- Authors talk about scaling RL; but did not see efforts to address the sampling problem. I agree that search is reduced from MCTS over token level to abstract plan level. But the Isn’t the search space is still huge (theoretically of the same complexity as before)?
- The comparison is mainly around Deepseek model. I would suggest to try out another model for strengthen experimental conclusions

### Questions
- Rephrase: steps through Step-level Advantage Preference Optimization (Step-APO), which integrates advantage estimates for step preference obtained via MCTS into Direct Preference Optimization (DPO)
- Can you comment on your training times as well?
- I am curious how many samples are used for computing \pi_\ref (via sft?) and do you update it again?
- The difference with DPO is still not fully clear to me. In eqn 12 is the main difference that you do not need full trajectories but partial/split trajectories are also sufficient? But I can also do this with DPO, I believe.

### Soundness
3

### Presentation
3

### Contribution
2
