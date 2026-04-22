# SPOGW: a Score-based Preference Optimization method via Group-Wise comparison for workflows

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Large language models (LLMs) have exhibited significant capabilities in addressing challenging problems throughout various fields, often through the use of agentic workflows that adhere to structured instructions and multi-step procedures. However, designing such workflows demands substantial manual effort, posing challenges to scalability and generalizability. Recent studies have aimed to minimize the human intervention needed for their construction, leading to advances in automated techniques for optimizing agentic workflows. However, current approaches are often constrained by their limited representational capacity, insufficient adaptability, weak scalability, and pairwise comparison paradigm—issues that stem primarily from a dependence on discrete optimization techniques. To overcome these limitations, we introduce a new score-based preference approach, refereed as SPOGW, which operates directly on cardinal reward signals through group-wise comparison and enables more efficient and stable optimization in a continuous space. SPOGW incorporates Iterative Offline GRPO (ioGRPO) with advantage-masked KL restriction (mKL), which regulates training update by placing greater emphasis on the advantageous regions of the policy response. In five benchmark datasets covering mathematical reasoning, coding, and question answering, SPGW matches or exceeds the performance of current state-of-the-art approaches, presenting a viable and forward-looking methodology for automated generation and optimization of agentic workflows.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, the authors introduce SPOGW: a framework for automated optimization of agentic workflows for Large Language Models. Existing approaches are often constrained to be discrete via discrete optimization or pairwise preference modes such as DPO. Specifically, SPOGW proposes optimizing a generator LLM in a continuous space by means of a score-based, group-wise preference optimization approach.

The pipeline is composed of three main stages:
Firstly, a data generation and curation process. For each query, $m=16$ candidate workflows are generated and executed, and scalar reward scores are obtained. These are then aggregated into groups of size $n=14$ and subsequently filtered, screened, and selected for high intra-group reward variance. Finally, a "group sharpening" step reduces these groups by only retaining the top-t and bottom-t scoring responses;
Second, Iterative Offline Group Relative Policy Optimization is used. It combines the orthogonal strengths of aspirations of sample diversity with a low-variance, off-policy training procedure. The method decouples the inherently noisy and unstable data collection phase from policy update, and instead trains on a static, pre-collected dataset;
Thirdly, they use Advantage-Masked KL Restriction mKL, a modification of the regularized RL objective. The final loss selectively applies the KL-divergence penalty only to responses i for which the advantage is positive with a mask.

Empirically, SPOGW exhibits state-of-the-art performance across five benchmarks including mathematical reasoning, coding, and question answering, outperforming approaches including the DPO-based ScoreFlow.

### Strengths
1. The authors identify that online RL is challenging for the application of RL on agentic systems. By decoupling data collection from policy optimization, their ioGRPO transforms it into a stable, standard supervised-like training loop. This is a procedure that helps to alleviate the data staleness and distribution shift problems.
2. mKL better reconciles exploration and exploitation than a uniform KL penalty, which can create conflicting gradient signals by pushing the policy away from a bad action (via the PPO objective) while pulling it back (via the KL term). The ablation study in Table 3 provide validation for this insight.
3. They comment that the core of GRPO is the advantage calculation, which is highly sensitive to the reward distribution within a group. By selecting for groups with high variance and then sharpening them to further increase the spread between the best and worst responses, it ensures that the resulting advantage signals are strong and unambiguous. The ablation in Table 4 shows that this curation strategy is effective.
4. SPOGW can elevate smaller models to the performance level of much larger ones. The results show that a 3B-parameter Qwen2.5 model (optimized with SPOGW) nearly matches the performance of the baseline 14B model on HumanEval. It indicates that for complex and multi-step tasks, the reasoning process can be effectively factored into a smaller, more efficient generator model and an optimized workflow structure.

### Weaknesses
1. You mention that you generate $m=16$ candidate workflows for each problem, and that the optimal dataset is $d=100$ problems; this is done over 3 iterations. So if I am not mistaken, a simple calculation for a single benchmark’s training run is $100 \times 16 \times 3 = 4800$ workflow generations and executions, all using the proprietary GPT-4o-mini as the executor. The costs of such work should not be neglected. 
2. The novelty of “Iterative offline GRPO” is actually contained within the already-established field of offline RL. While the iterative part that you refer to as “the first”, the most important part (the core) is what defines offline RL: data generation and policy optimizations are performed independently. I think your paper has too few citations and comparisons with the some offline RL methos (e.g., IQL, CQL). You should clarify whether your method solves the central problem of offline RL such as out-of-distribution actions and data shift, beyond the implicit regularization provided by the KL term.
3. The method relies on a single, powerful, proprietary model (GPT-4o-mini) as both the executor for reward generation and the judge for final evaluation. This setup may create a risk of "judge-specific overfitting." The generator model (Qwen2.5-7B) may not be learning to produce objectively better workflows, but rather workflows that are tailored to the specific biases of the GPT-4o-mini judge. The reported performance might not transfer if a different judge were used. The work would be much stronger if it demonstrated that the optimized workflows are superior under a different evaluation protocol.
4. In section 3.1 and Figure 1 there’s a “Filtering” step right before “Screening”. Any details about this? Is it filtering based on syntax, length, or some other settings?

### Questions
1. (related to weakness2) Any discussion on how your method compares to canonical offline RL methods like Conservative Q-Learning (CQL) or Implicit Q-Learning (IQL). Did you consider these algorithms as alternatives during your research?  
2. Please refer to weakness 3.
3. Did you conduct any ablation study on isolating the contribution of group-wise GRPO from the rest of the components? To justify the argument that group-wise is better than pairwise, I would want to look at a baseline that uses a pairwise objective but also has your data curation pipeline and advantage-masked KL. How does SPOGW compare to it?
4. On the $C(m, n)$ permutations, do you just generate all $C(m, n)$ groups per query, or do you use a sampling strategy to knock this number down? If so, what’s your approach?
5. see weakness 4.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the optimization of "agentic workflows" for Large Language Models (LLMs) by proposing a new method called SPOGW (Score-based Preference Optimization via Group-Wise comparison). Currently, designing effective workflows requires significant manual effort, and existing automated methods are limited by discrete optimization, rigid pairwise comparison paradigms (like DPO), or unstable on-the-fly execution

### Strengths
First, applying GRPO (a non-pairwise RL algorithm) to the problem of agentic workflow optimization is novel. Second, adapting it into an ioGRPO (iterative offline framework) is a clever engineering solution that directly tackles the critical stability challenges faced by online RL methods in this domain (which requires execution) . Finally, the mKL is a very insightful contribution; it recognizes the flaw in the standard KL penalty (i.e., penalizing divergence from bad samples) and provides an elegant, targeted solution to only preserve the "good" parts of the reference model.

### Weaknesses
1. The hyperparameter search and the data collection itself may be expensive.  The ablation studies show that SPOGW's performance is notably sensitive to several key hyperparameters: the KL coefficient $\beta$, the group size $2t$, and the dataset size $d$. For example, performance on HumanEval peaks at 96.2 with $d=100$ but drops to 94.1 at $d=600$ 12; it is 96.2 for $\beta=0.1$ but drops to 94.7 for $\beta=0.025$. While the paper identifies the optimal values, it doesn't discuss the cost of finding these optimal settings on a new task. 

2. Confusing "Offline" Terminology: The paper terms the method "Iterative offline GRPO". In reinforcement learning literature, "Offline RL" (or Batch RL) typically refers to learning only from a fixed, pre-collected dataset without further interaction with the environment. SPOGW's framework, however, is iterative, where the policy is used to collect new data after each iteration.

### Questions
Please see above。

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper primarily focuses on automated agentic workflow generation. First, the authors propose a score-based preference optimization method that directly leverages cardinal reward signals for policy updates. This mitigates the inherent limitations of existing workflow optimization methods. The superior performance of the proposed method is demonstrated through experiments.

### Strengths
- The problem is well-motivated. Optimizing the agentic workflow is crucial for practical applications.
- The experiments are comprehensive, and the proposed method demonstrates better performance than prior work.

### Weaknesses
I am not an expert in automated agentic workflow generation, so I am not sure whether the experimental setup is fair. Specifically, I did not understand why the LLMs used for optimization and the generator models differ across the baselines and the proposed method in the experiments. What if the same LLM is used for all methods? Does the proposed method still perform better than the others?

Moreover, I am wondering whether the training time is comparable across all methods. Are there experimental results showing training curves with respect to GPU hours?

### Questions
My main concerns and questions are outlined in the Weaknesses section. Additionally, I have the following question:

- Eq. (4) is not equivalent to the original definition of the KL divergence. Does it actually denote the derivative of the KL divergence?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents SPOGW, a new method for optimizing agentic workflows (structured multi-step sequences of calls by large language models). The core innovations are:
- Constructing group-wise training data — for each input query, generate multiple candidate workflows, execute them to get scalar scores, then form groups of workflows and use the score distributions in the group to compute advantages for optimization.
- Using an Iterative Offline Group Relative Policy Optimization (ioGRPO) loop, where data collection (workflow generation + execution + scoring) is decoupled from policy updates to avoid code execution and API instability during training. 
- Applying an advantage-masked KL divergence (mKL) regularizer: the KL penalty (between current policy and reference policy) is applied selectively only on responses with positive advantage, thus encouraging alignment to good behaviors while allowing learning beyond the reference.

### Strengths
- The overall paper is presented well, with the entire workflow and key ideas explained clearly. Especially, the steps for dataset collection, filtering, sharpening are illustrated well.

- The idea of applying Advantage-Masked KL Restriction is interesting. Intuitively, this is a good approach for preventing the policy updating too much for good actions, while allowing flexibility for bad actions.

- Several ablation studies are performed, which strengthen the paper a lot, especially, the ones on data preprocessing methods and KL regularization.

### Weaknesses
- The overall flow is largely based on the standard RLVR framework, with modifications added (i.e., offline and advantage KL restriction). It is unclear whether the proposed modifications are generalizable beyond the current setting and experiments (e.g., whether it works in math without the agentic workflows).

- The effectiveness of offline iterative GRPO is not well illustrated. Especially, offline should introduce a huge impact of off-policy, while improving training time. It would be more convincing if the benefits from shorter training time (and/or better stability) overwhelms the suffering from the off-policy impact. I would suggest adding some comparisons (both training time and final results) between the online and offline GRPO.

### Questions
My concerns are center on the two points listed in weakness. I would love to hear the authors' opinions on them.

### Soundness
3

### Presentation
3

### Contribution
2
