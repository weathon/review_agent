# Sample-Aware Dual Actions for Prompt Optimization

- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
In recent years, large language models (LLMs) have achieved remarkable progress in reasoning, question answering, and decision-making tasks in natural language processing. High-quality prompts play a crucial role in guiding LLMs to generate outputs that meet expectations. However, manually designing effective prompts for specific tasks is often time-consuming and heavily reliant on expertise, limiting the scalability and efficiency of model applications. Consequently, automated prompt optimization has become an important direction for enhancing LLM performance. To address this, we propose a sample-aware dual actions Monte Carlo Tree Search (MCTS) framework for automated prompt optimization, enabling the search process to leverage sample performance for more effective optimization. This method not only efficiently utilizes training samples to guide prompt improvement but also directs the optimization trajectory based on the overall state of the training samples. We validate our framework on the Big-Bench Hard (BBH) and MMLU datasets, and experimental results demonstrate that it outperforms traditional prompt optimization methods and recent baselines in both accuracy and optimization stability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a new framework for automated prompt optimization for large language models, centered on a sample-aware dual-action Monte Carlo Tree Search (MCTS). The core idea is to abstract prompt optimization into two high-level strategies: “Failure-Aware Reflection” (learning from failures) and “Success-Aware Induction” (amplifying successes). The method introduces a dynamic sample pool to quantify and prioritize sample informativeness during the search, providing both local and global feedback to the MCTS planner. Experiments on BBH, MMLU, and BBEH benchmarks show improved accuracy and stability compared to several recent baselines and traditional prompt optimization techniques.

### Strengths
The method is well-described, with mathematically formalized sample prioritization and conservative estimation using the Wilson confidence interval. The paper also provides detailed action-guiding prompts.
The dataset coverage is reasonably broad — including BBH (5 tasks), domain-specific datasets (5 tasks), and BBEH (4 tasks). Baselines are comprehensive, encompassing manually designed prompts, CoT variants, and PromptAgent, and the proposed method consistently achieves higher performance.

### Weaknesses
1. Table 4 only compares three configurations — Beam+Pool, MCTS-only, and Full — without isolating the contribution of the dual-action mechanism (e.g., using only the reflection or induction action). The influence of different components in the sample pool (e.g., removing the difficulty term DiD_iDi​ or the gain term GiG_iGi​) is also not examined.
2. The formulation involves multiple weight parameters (\alpha, \beta, \gamma, \theta, \lambda, \mu, \delta), yet only a limited sensitivity analysis for \lambdaλ and \thetaθ is presented in Appendix A.1 — and solely on the Boolean Expressions task, where accuracy fluctuates between 0.375 and 0.55.
3. The authors acknowledge conceptual similarity with PromptAgent but claim two key contributions:
(1) formalizing a dual-action paradigm beyond simple failure reflection, and
(2) introducing a mathematically defined sample-pool mechanism.
However, the distinction remains questionable — PromptAgent already integrates MCTS and reflective reasoning. Whether the proposed “successful induction” action constitutes a substantive innovation requires stronger theoretical justification or experimental evidence. As mentioned above, additional ablations could help clarify this point. Further comparative analysis against PromptAgent would also strengthen the contribution claim.
4. Appendix A.5 provides a theoretical formulation, but no empirical data (e.g., API call counts or runtime) are reported. It would be valuable to include such measurements and conduct a cost-performance comparison with PromptAgent (e.g., “Which method achieves higher accuracy under the same computational budget?”).
5. There appear to be counterexamples or contradictory cases within the Casual Judge results. Clarification or discussion would be appreciated.

### Questions
see weeknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the limitations of existing automated prompt optimization methods, which overlook sample heterogeneity. The authors propose a sample-aware dual-action Monte Carlo Tree Search (MCTS) framework that integrates two core components: (1) two optimization actions including failure-aware reflection and success-aware induction; (2) a dynamic sample pool where the sample informativeness is measured by difficulty, recent gains, and variance, and sample priority and overall quality of the sample pool are also quantified to guide the optimization. The exploration-exploitation balance in MCTS is adaptively adjusted based on the sample pool’s composition to enable efficient resource allocation. Experiments on reasoning benchmarks (BBH, BBEH) and domain-specific tasks (MedQA, CaseHold) show that the framework outperforms baselines like PromptAgent (e.g., 3.9% higher accuracy on BBH, 5% on BBEH).

### Strengths
**Originality**
This work points out the problem that prior methods (e.g., PromptAgent) that treat samples uniformly and use handcrafted fine-grained actions, and introduces two high-level actions (Reflection/Induction) and a informativeness-quantified sample pool for the MCTS policy optimization, which utilize the sample heterogeneity property in prompt optimization.  

**Clarity**
This paper is clearly motivated with a standard structure including introduction, methodology, experiment, conclusion, and appendix. The high-level idea is sound, and the main components in methodology strictly relate to the sample heterogeneity issues. Experiments cover diverse scenarios—general reasoning (BBH), domain expertise (MedQA/CaseHold), and high difficulty (BBEH)—validating generalization. 

**Significance**
Since PromptAgent formalized prompt optimization as policy search, this work extends the idea to consider the sample heterogeneity and improves the performance in general-domain and domain-specific tasks, supporting LLM deployment in professional fields.

### Weaknesses
1. No indexing for all the equations in the manuscript. No definition of $b_i$ in Line 192, and not sure whether the definition of $b_i$ in L255 means the same thing.

2. In equations in Line 192, Line 257, Line 287, mathematical formulas that include mixed texts make the technical definitions informal.  

3. No definition of $H(c_{pool})$  in Line 257.  

4. The axis texts in Figure 2 are overlapping with each other. 

5. The authors provide theoretical analysis on API call cost in Appendix A.5, but no empirical statistics of API cost and computational cost during the prompt optimization. Furthermore, how many API calls does the framework require to achieve the same amount of accuracy gain compared to PromptAgent? 

6. The paper briefly mentions parameter robustness in the appendix A.1 but does not analyze how these parameters affect performance across tasks, e.g., do domain-specific tasks require different $\lambda$, $\beta$ ratios than general reasoning tasks?

7. Experiments use only DeepSeek-R1 for optimization and Qwen-Flash for evaluation. It remains unclear if the framework performs consistently on other model pairs (e.g., GPT-4o as optimizer, Llama-3 as evaluator), weakening the confidence in cross-model generalization.

### Questions
Several questions and concerns are raised in "Weaknesses" part. I would be willing to change my recommendation according to the authors' response.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a sample-aware dual-action prompt optimization framework that abstracts prompt optimization into two complementary strategies through Monte Carlo Tree Search (MCTS): Failure-Aware Reflection (adjusting prompts for low-reward samples) and Success-Aware Induction (extracting effective patterns from high-reward samples). The framework introduces a dynamic sample pool that quantifies sample information value based on difficulty, recent gains, and variance to guide the search process.

### Strengths
- The paper proposes a dual-action framework (Failure-Aware Reflection and Success-Aware Induction), transcending the limitations of existing methods that treat all samples equally.
- The framework designs a dynamic sample pool mechanism to quantify sample information value through three dimensions: Difficulty, Recent Gains, and Variance.
- The approach implements an adaptive exploration-exploitation balance mechanism to dynamically adjust MCTS search intensity.
- Ablation studies validate the contributions of each component, and parameter sensitivity analysis provides practical guidance.

### Weaknesses
- The MCTS method requires numerous API calls, resulting in high practical application costs, with no computational complexity analysis or simplification strategies provided
- The selection rationale for weight parameters (α,β,γ) in sample metrics is not detailed, and sensitivity of these parameters across different tasks is not analyzed
- There is a lack of theoretical justification for the necessity of the dual-action strategy, and no analysis of how the sample-aware mechanism affects optimization convergence

### Questions
- What is the rationale for selecting these three specific metrics: Difficulty, Recent Gains, and Variance? How are weights determined in the informative score? Are the same weights used for all tasks?

- Are there specific strategies to reduce the number of API calls? The paper mentions "early-stopping mechanisms" but does not elaborate.

- Can you provide a performance-efficiency trade-off analysis for different MCTS configurations (depth, width, number of rollouts)?

- How does the zero-shot generalization ability of optimized prompts perform on similar but unseen tasks?

- How consistent is the framework's performance across LLMs of different scales and architectures?

- When the sample pool simultaneously contains numerous difficult samples and high-value successful samples, how are the two actions balanced?

- Beyond accuracy, are other metrics considered such as inference cost, prompt length, robustness, etc.?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a sample-aware dual action Monte Carlo Tree Search (MCTS) framework for automated prompt optimization, aiming to enhance search efficiency. The proposed method leverages the overall state of the training samples to guide prompt improvement. Specifically, the proposed method injects two strategies into PromptAgent: inductive actions and reflective actions. The experimental evaluation on the Big-Bench Hard (BBH) and MMLU datasets demonstrates that the proposed method outperforms existing baselines.

### Strengths
- The proposed method integrates sample-aware learning approaches into MCTS for prompt optimization.
- The experimental results show that the proposed method outperforms existing methods such as PromptAgent.

### Weaknesses
- In the experiment, only the PromptAgent is considered as the baseline method. It would be better if the authors could compare the proposed method with more baseline methods for prompt optimization.
- The concrete algorithm of the proposed method is somewhat unclear. A pseudocode or detailed description would help in understanding the approach better.
- The cost analysis (computational cost or API calls) is missing. It is unclear whether the proposed method is truly efficient compared to the baselines and existing methods.
- Only one LLM combination (DeepSeek-R1 and Qwen-Flash) is examined. It would be better if the authors could evaluate the proposed method with different LLMs to demonstrate its generality.

### Questions
- The proposed method introduces many hyperparameters (e.g., alpha, beta, kappa, etc.). How are these hyperparameters set and tuned in the experiments? How sensitive is the performance of the proposed method to these hyperparameters?
- Why did the authors choose the five tasks in BBH for evaluation? How is the performance on other tasks? In the literature of PromptAgent, other tasks in the BBH dataset are also used for evaluation.
- Is it possible to inject the proposed idea, the failure-aware reflection and success-aware induction, into other prompt optimization methods beyond PromptAgent (MCTS-based methods)?

### Soundness
2

### Presentation
2

### Contribution
2
