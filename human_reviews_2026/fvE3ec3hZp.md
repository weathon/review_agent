# Systematic Exploration Supervision Enables Scaling Beyond Training Complexity

- Decision: Reject
- Scores: 4, 8, 2

## Abstract
Language models trained on input--output pairs or linear Chain-of-Thought (CoT) traces often fail when task complexity at test time exceeds the regime seen during training; reinforcement learning methods can help but suffer from cold-start brittleness when base accuracy is low. We introduce Systematic Exploration Supervision (SES), a process-level supervision framework that verbalizes \emph{complete multi-branch search traces} (sampling alternatives, propagating outcomes, and backtracking to extract a solution) rather than a single reasoning chain. In textualized Gridworld, SES preserves 76.5\% success when scaling from 10×10 training environments to unseen 20×20 grids (vs. 19.0\% for standard supervised fine-tuning, 26.0\% for inference-time Tree-of-Thought, and 6.0\% for GRPO). We further extend SES to open domains via a bootstrapped trace construction procedure that guarantees inclusion of at least one valid solution while adding diverse, reward-prioritized alternatives. Results show substantial improvements on combinatorial reasoning (Game of 24: 47\% vs. 17\% best baseline) and competitive performance on logical reasoning (ProntoQA: 100\%), with task-dependent effectiveness patterns. We demonstrate that SES behavior cannot be induced with few-shot prompting alone, even with sophisticated models like GPT-o1, suggesting in-weight algorithmic policy acquisition. Remarkably, our approach achieves 14× parameter efficiency, with a 0.5B model outperforming 7B baselines. We characterize when SES is advantageous (large branching factors, low base competence, scaling demands) and discuss limitations (token length inflation, effectiveness when base models are already competent). Our findings highlight full-search verbalization as a simple, offline alternative to inference-time search or costly RL for scaling systematic reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work is motivated by the observations: 1) SFT lacks the ability of exploration, and it can not adapt the model to different/more complicated tasks (even with same solving strategy); and 2) RL converges slowly and is typically data-inefficient due to low initial success rates. To address this, the paper proposes Systematic Exploration Supervision (SES), a framework that introduces sampling, propagation, and backtracking into model training. Experimental results show that SES outperforms Tree-of-Thought and GRPO.

### Strengths
1. The paper proposes an online search algorithm for post-training LLMs, allowing the model not to strictly follow SFT paths but to explore intelligent problem-solving routes via search.

2. Experimental results provide evidence of the effectiveness of the proposed method.

### Weaknesses
1. The online search policy is not a novel topic and there are many prior works on it (eg., Monte Carlo Tree Search and etc). This work lacks a thorough literature review comparing SES to prior algorithms.
2. This method requires that enough successful trajectories have been collected. For the problems the authors tried to solve, the problem complexity is low and obtaining successful paths is easy. However, it may not adapt to realistic settings where problems are much more complex and successful trajectories are rare.
3. It seems that the proposed algorithm counts and stores all possible routes in memory, which is inefficient if the problem has wide search space and requires long steps to solve.
4. Algorithms are not clearly stated. For examples, the implementations of Backtrace, Verbalize, GetGoldAction in Algorithm1&2 are not explained.
5. Lack of experimental comparison to the other online search algorithms. 
6. Lack of experimental comparison on more complex problems.

### Questions
1. In Algorithm 1, Line 168-175, It seems that all $a\in S(s)$ are added to the queue sequentially, resulting in a queue like [start,$a_1,a_2,...$] for $a_i$ in $S(s)$. What is the intuition behind this queue structure?
2. In the Gridworld experiments, were all possible routes explicitly explored and stored during training?
3. Line 183 mentions that “Deadend or dominated states are explicitly labeled.” Does SES only apply to problems where such states can be clearly defined?
4. What are the SFT results in Figure 5?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Systematic Exploration Supervision (SES), a supervision framework designed to improve the ability of LLMs to reason in structured environments such as Gridworld. Rather than training models to generate linear CoT, SES trains the model on systematically-generated multi-branch search trees that represent good exploration (backtracking, sampling alternative paths etc.) of the structured environment - these trees can be viewed as process-supervised samples that teach the model what good exploration "looks like".

During test time, the model is prompted to solve the task using linear CoT (SE-CoT). Crucially, the authors demonstrate that the SES-trained models improve significantly over RL, SFT, CoT-prompted and Tree-of-Thoughts baselines on tasks involving structured exploration even though search is conducted only with linear CoT. The authors argue that this arises from the fact that SES trains the model to internalise the search process. Furthermore, they show that this internalisation phenomenon allows the model to generalise to more complex environments than they were trained on (e.g. larger grids in Gridworld).

Through various ablation studies, the authors demonstrate the limits of SES: while SES significantly strengthens model performance on tasks where structured exploration is effective (e.g. Gridworld, Game of 24), SES provides less benefit on tasks where traditional linear search is effective (e.g. maths reasoning) and where the base model is already strong.

### Strengths
- The authors present an intuitive yet interesting method for teaching models to learn better search "skills" or "reasoning patterns" in structured environments. The results are strong and support their central claims.
- I appreciate the fact that they provide analysis on when their method does not work, which helps us better understand what SES actually teaches the model to do, and may help guide future research in this direction. These findings also help us distinguish between different kinds of reasoning tasks (e.g. structured exploration tasks like Gridworld vs. more linear, "depth-focussed" tasks like GSM8K), and highlights the fact that different strategies may be required to achieve large gains on these different tasks.
- The research is a timely contribution given the direction of the field - much research has been done scaling test-time compute (train models/create new inference-time algorithms such that generating more reasoning tokens at test-time -> better performance), whereas this paper demonstrates that using compute to instead generate better training data nonetheless remains a promising path.

### Weaknesses
I do not believe there are any fundamental flaws or weaknesses in the paper that make it unsound. I nonetheless have some thoughts on how it may be improved:
- It is unclear to me why and how performance degrades as we scale up the grid sizes - any search skills learned for small grids are equally applicable for larger grids (the model merely has to apply these skills more often). Clearly the internal search skill learned is not perfect - it would be interesting to see some analysis on the failure modes of the model to better understand the method's limitations.
- It could be beneficial to provide some examples of the training data (i.e. the exact inputs/labels you do SFT on).
- One potential criticism of this work is that Gridworld is a very simple task, and that increasing the size of the grid doesn't represent a meaningful increase in its difficulty, given that the same search operators are effective regardless of the grid size (and likewise for Game of 24), and that SES main still fall behind using test-time algorithms on more complex structured search tasks where a larger degree of skill/knowledge generalisation may be required (e.g. MiniHack https://arxiv.org/abs/2109.13202, Crafter https://github.com/danijar/crafter). As such, I think it would be useful to see SES applied to more difficult structured exploration tasks.
- One of the baselines is the SFT baseline, where the model is SFT'd on (successful?) trajectories. How many samples are used here vs. for the SES model? It would be interesting to compare the two in a compute-controlled setting (since each SES sample is much longer than a direct SFT trajectory, the compute-controlled comparison would require training on many more direct trajectories than SES samples). This will tell us whether the gains from SES are largely due to the structured and process-supervised nature of the training data, or whether it is simply due to the fact that you are training on a larger number of tokens.

### Questions
- Have you tried combining SES with test-time search? i.e. actually generate the trees during test time and do search over them (rather than just doing it internally). This may yield further improvements and better scaling properties.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce a framework designed to teach language models a search operator. Instead of verbalizing only a single reasoning path, SES verbalizes complete multi-branch search traces.

The key components verbalized in a Systematic Exploration Chain-of-Thought (SE-CoT) trace are:
1. Sampling: Enumerating candidate actions systematically or in a reward-ordered sequence.
2. Propagation: Predicting resulting states or intermediate evaluations.
3. Backtracking: Recovering the optimal path from the goal to the start.

SES is implemented via two methods:
1. Direct Algorithmic Supervision: Used when a canonical search procedure (e.g., BFS/DFS) is known (e.g., textualized Gridworld). This method explicitly supervises the entire search process, including counterfactual sibling branches, teaching *why* non-chosen actions were rejected.
2. Bootstrapped Search Discovery: Used for open domains without explicit algorithms (e.g., Game of 24). This procedure constructs synthetic search trees guaranteed to include at least one valid solution (by injecting the gold path) while adding diverse, reward-prioritized alternatives sampled from the base LM ($\pi_{\theta}$).

When trained only on 10×10 Gridworld environments and tested on unseen 20×20 grids, SES maintains 76.5% success, achieving a remarkable 4x improvement over standard SFT (19.0%) and significantly surpassing Tree-of-Thought (26.0%) and GRPO (6.0%).

### Strengths
1. Written clearly, easy to understand
2. Experiments are documented well

### Weaknesses
1. Lack of novelty: implementing search procedures during CoT is not at all something new and has appeared as early as [1], and has been a repeated area of focus throughout the test-time scaling breakthroughs in o1, r1. "Verbalizing complete multi-branch search traces (sampling alternatives, propagating outcomes, and backtracking to extract a solution) rather than the winning trace" is an area that has been very, very well-explored. Also, the study of test time exploration at token budgets beyond the training regime has also been well-studied before [2]. These works should be cited and discussed in related works.
2. Easiness of tasks and old models: Game of 24, GSM8K are both well-saturated tasks. Qwen 2.5 family is also not the newest family of Qwen. It would be more interesting to look at more difficult tasks, such as AIME, USAMO, or ZebraBench.
3. Lack of gain on open-reasoning domains, such as GSM8K. 

[1] https://arxiv.org/abs/2404.03683
[2] https://arxiv.org/abs/2506.09026

### Questions
1. Perhaps it would be more interesting to compare different ways to construct reasoning traces from scratch, and getting really weak base models model to fit on it as a way to learn to implement test-time search? One could study how one can best construct such traces, and what heuristics / intricacies one might have to care about during this process.
2. Can we evaluate on harder datasets, as mentioned in the section above?
3. To push the story further, better performance on open-reasoning domains should be shown.

### Soundness
3

### Presentation
3

### Contribution
2
