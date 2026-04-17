# MiGrATe: Mixed-Policy GRPO for Adaptation at Test-Time

- Decision: Reject
- Scores: 2, 4, 2, 8

## Abstract
Large language models (LLMs) are increasingly being applied to black-box optimization tasks, from program synthesis to molecule design. Prior work typically leverages in-context learning to iteratively guide the model towards better solutions. Such methods, however, often struggle to balance exploration of new solution spaces with exploitation of high-reward ones. Recently, test-time training (TTT) with synthetic data has shown promise in improving solution quality. However, the need for hand-crafted training data tailored to each task limits feasibility and scalability across domains. To address this problem, we introduce MiGrATe—a method for online TTT that uses GRPO as a search algorithm to adapt LLMs at inference without requiring external training data. MiGrATe operates via a mixed-policy group construction procedure that combines on-policy sampling with two off-policy data selection techniques: greedy sampling, which selects top-performing past completions, and neighborhood sampling (NS), which generates completions structurally similar to high-reward ones. Together, these components bias the policy gradient towards exploiting promising regions in the solution space, while preserving exploration through on-policy sampling. We evaluate MiGrATe on four challenging domains—word search, molecule optimization, hypothesis+program induction on the Abstraction and Reasoning Corpus (ARC), and natural-language hypothesis search on DiscoveryBench—and find that it consistently outperforms both inference-only and TTT baselines, demonstrating the potential of online TTT as a solution for complex search tasks without curated training data.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This submission focuses on balancing exploration and exploitation when applying large language models (LLMs) to the black-box optimization tasks. Notably, test-time training (TTT) shows a promising direction by training the model with synthetic data for enhancing the solution quality. However, the data for TTT is limited and requires task-specific design. To address the data feasibility problem, this submission proposes MIGRATE, an online TTT algorithm that employs a mixed search algorithm for constraining data and adopts group relative policy optimization (GRPO) to optimize the policy model. The submission conducts extensive empirical studies and justifies the effectiveness of MIGRATE.

### Strengths
- The submission focuses on the trendy language model reasoning problem, identifying the weakness of prior TTT methods, and proposes an effective method to handle the data feasibility problem.  
- The submission is generally well-written, with clear illustrations and tables.
- The conducted experiments provide empirical evidence on the effectiveness of the proposed method compared to the baselines.

### Weaknesses
- The submission claims a focus on the test-time training. However, MIGRATE updates the model's parameters using reinforcement learning and samples with verification. This violates the definition of TTT [1], where the model is trained in a self-supervised manner.
- The core idea of MIGRATE is employing off- and on-policy sampling to improve the data scale and quality, and directly using the sampled data to train the model using GRPO without any further adaptation to the test-time training setting. 
- The adopted baselines are primarily prompt-based approaches, such as OPRO and self-reflection. There are no comparisons with other TTT and RL-based approaches. Furthermore, related black-box optimization approaches are also missing, such as evolutionary algorithms or the Bayesian optimization approach. This weakens the reliability of the conducted experiments.

[1] Test-Time Training with Self-Supervision for Generalization under Distribution Shifts. In ICML, 2020.

### Questions
1. Could you provide experiments on comparing with the iterative search algorithm, like MCTS or the evolutionary algorithm? How about other TTT methods?
2. Could you provide further discussion or empirical evidence on the "small variations in solution yield small changes in quality" claim in Sec. 4.1, Neighborhood sampling paragraph? Since the datasets like ARC and Dockstring might yield completely different quality of solution with mild modification.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a mixed sampling framework for GRPO in LLM test-time training. The authors use completions sampled from historical completion databse and their neighborhoods to enhance exploitation during test-time training. Experimental results shows that the proposed method surpasses inference-only and test-time training baselines in multiple optimization scenarios.

### Strengths
1. The introduction of greedy and neighborhood sampling from a historical database reduces reward sparsity in complex optimization scenarios and lowers the expertise required for offline data preparation.

2. The proposed method outperforms GRPO-based, test-time training variants. The authors also provide a detailed analysis of the exploration-exploitation tradeoff in the mixed sampling method.

### Weaknesses
1. At the beginning of the solution search with MiGrATe, the solutions in the database might not be high-performing, as might the top-k solutions and the neighborhood solutions derived from them. Since a large proportion of solutions may be derived from the database initially (greedy sampling + neighborhood sampling), MiGrATe's performance might be significantly influenced by the initial sampling, which can be variable. This could be problematic when the reward is sparse and on-policy sampling fails to generate solutions that elicit non-zero rewards, the database solutions might mislead the LLM training into converging to local optima.

2. The proposed method requires different $[\alpha, \gamma, \beta]$ ratios for different tasks (e.g., [0, 4, 1] for Semantle and [2, 2, 1] for Dockstring), which are also carefully hand-crafted and task-specific. This introduces additional expertise requirements or necessitates preliminary experiments to set proper values, which might limit the method's generality and hinder further application to other optimization tasks without prior knowledge.

### Questions
1. At the beginning of the solution search with MiGrATe, the database is empty. How to sample the top-k solutions?

2. In the MiGrATe hyperparameters for Semantle, on-policy samples $\alpha$ is 0, then how to fill database without on-policy sampling?

### Soundness
3

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
This paper explores how to test-time-train (TTT) base LLMs for optimization tasks. Started from OPRO (an initial google's llm for optimization Q&A framework), the authors listed weaknesses of such in-context learning (e.g., exploration-exploitation tradeoff inability) and notive themselves  to address this issue by introducing TTT concepts. Specifically, the authors consider fine-tuning LLMs with self-generated  data. This is achieved by replacing the naive grouping data in GRPO with three parts of self-generated data: 1) on-policy sampling data, same as standard policy gradient methods; 2) on-policy historical demonstrations, a database is used to store historical samples generated before current training step, and topk samples are selected; 3) local exploration, by prompting LLMs to generate small. stocastic variations on the elite historical demonstrations. Using these data, the LLMs are fine-tuned via balanced exploration and exploitation to achieve better solutions on tested problems. The authors validate this approach's effectiveness through comparing the trained LLMs (with LoRA) with OPRO and pther specialized baselines on diverse optmization and reasoning tasks.

### Strengths
1. I think this paper presents a novel perspective for LLMs as Optimizers works such as OPRO. The authors locate the exploration and exploitation imbalance in such in-context learning approaches and make an interesting try on using TTT to rebalance such tradeoff. 

2. I appreciate the authors provide the code for reproducibility checking.

### Weaknesses
1. While I acknowedge that the overall methodology the authors have proposed are solid and interesting (self-supervision), I have to say that  I can not see real and practical value of this work for real-world optimization problems. I can understand that the authors may not be long-standing optimization researchers, however, in realistic scenario, using the method you provide may not be practical since it requires training LLMs for solving one problem. I found this is quite opposite to existing automated algorithm design or learning to optimize or meta-black-box optimization, where the aim is to learn optimzers that generalize across different problems. Based on this point, I can not support this paper as a useful optimization approach. 

2. I think the writing of this paper is not clear at all. First, I can not understand what the term "optimization" denotes until I  read section 5 (experiment). Even if I read section 5, I still can not understand what is the optimizer in this paper, is it the LLM itself? Second, What is the relationship between the related works (EC, RLVR) and this paper? I can not understand why the authors mention these works since you never compare them in experiments. Third, the NS sampling is not explained with clarity (lines 220 - xx). About the NS sampling. another question is why "continuity assumption" persists in your test cases, for LLMs, an intuitive feeling is that once you slightly change the prompt, the output may vary significantly, and for the testing scenarios, a small change in the solution also can not ensure small chance in its evaluation score, such as ARC.

3. Objectively speaking, OPRO is not a qualified optimizer at all, as validated in recent papers.  Besides, OPRO has no actual training, hence it is somewhat unfair to compare your mothod with it.

4. For reasoning task such as ARC, please also provide performance of existing strong LLMs such as GPT-5, otherwise I can not tell if your method is really a good and useful one.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a method for online test-time training of large language models that enables adaptive search in black-box optimization tasks without requiring external training data. The approach uses GRPO with a novel mixed-policy group construction strategy that combines on-policy sampling to ensure exploration by sampling from the current policy, greedy sampling to exploit known high-reward regions by reusing top-performing past completions, and neighborhood sampling to facilitate local exploration by generating structural variations of high-reward solutions. The method iteratively adapts LLM parameters at test time to shift the sampling distribution toward higher-quality solutions. The authors test their approach across four domains including, showing improvements over both inference-only and test-time training baselines.

### Strengths
The paper's most significant contribution is its generalizable approach that eliminates the need for handcrafted, task-specific training data, which has been a major limitation of prior test-time training methods. All training signals in MIGRATE are model-generated, making the method applicable across diverse domains without requiring domain expertise to curate training examples, particularly helpful for low data domains. 

The design of the three-component mixed-policy strategy is well-motivated and balances the exploration-exploitation tradeoff. This combination addresses fundamental challenges in black-box optimization that purely on-policy or off-policy methods struggle with individually.

The experimental validation is comprehensive and strengthens confidence in the method. The authors evaluate on diverse tasks with different solution spaces and reward functions, compare against multiple baselines including both inference-only methods (Random, NS, OPRO, Reflexion) and test-time training variants (GRPO, GRPO-Greedy), and include extensive ablations showing each component's contribution. The analysis goes beyond simple accuracy metrics to examine search behaviors, solution quality distributions, and trajectory visualizations, providing insights into how and why the method works.

### Weaknesses
The computational cost is not thoroughly discussed, which is a significant oversight for a test-time training method. While runtime is mentioned in the appendix (for example, 51 minutes per ARC task on an A100 GPU), there is insufficient analysis comparing the cost versus inference-only methods, examining memory requirements for LoRA fine-tuning, assessing scalability to larger models or longer horizons, or analyzing trade-offs between TTT overhead and solution quality improvements. 

The method also shows hyperparameter sensitivity, requiring careful tuning of α, β, and γ for each domain. Figure 7 clearly shows that optimal ratios vary significantly across tasks. A more clear or principled guide on these parameters would be very helpful for broad usability.

The paper does not provide formal analysis of why this particular mixed-policy strategy works, no validation of the "continuity assumption" underlying neighborhood sampling beyond empirical results, no convergence guarantees or sample complexity bounds, and no clear characterization of when or why the method might fail. While empirical validation is valuable, theoretical understanding would provide confidence about generalization to new domains and guidance for method development.

The improvements on some tasks are actually quite small, so discussing where the method best works and why would be valuable.

Several evaluation limitations discussed by the authors affect the conclusiveness of the results (such as ARC 200/400 tsks evaluated, single runs with boostraping for variance estimates). 

The implementation details of neighborhood sampling remain somewhat unclear. The paper doesn't fully explain how "stochastic variations" are generated from the prompt, what ensures they remain in the neighborhood versus being arbitrary perturbations, or how the quality of NS depends on prompt engineering. This makes the method harder to reproduce and potentially sensitive to prompt formulation in ways that aren't explored.

The comparison to related work is incomplete in several respects. Evolutionary algorithms are only briefly compared in the appendix, there's no comparison to Bayesian optimization methods despite these being mentioned in related work, and comparisons to other recent TTT methods beyond GRPO variants are missing. Broader comparisons would strengthen the claims.

### Questions
Can the authors provide formal analysis or intuition about when MIGRATE is expected to outperform pure on-policy or pure off-policy approaches? Are there identifiable task characteristics that predict method effectiveness, such as properties of the reward landscape, solution space structure, or base model capabilities?

Second, the hyperparameter selection process needs clarification. How should someone choose α, β, and γ for new tasks where the optimal configuration is unknown? 

Can the authors provide more technical details on how NS actually generates variations? Have they tried alternative NS implementations, such as explicit perturbations in latent space rather than prompt-based variation? How sensitive is performance to the specific NS prompt formulation, and what makes a good NS prompt?

Understanding failure modes would provide important context. Are there scenarios where MIGRATE performs worse than baselines, perhaps due to misleading initial solutions or particular reward landscape properties? What happens in extremely sparse reward settings where even greedy samples have near-zero reward? How does the method behave when the base model is very weak versus very strong?

### Soundness
3

### Presentation
3

### Contribution
3
