# Inference-Time Scaling for Generalist Reward Modeling

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Reinforcement learning (RL) has been widely adopted in post-training for large language models (LLMs) at scale. Recently, the incentivization of reasoning capabilities in LLMs from RL indicates that proper learning methods could enable effective inference-time scalability. A key challenge of RL is to obtain accurate reward signals for LLMs in various domains beyond verifiable questions or artificial rules. In this work, we investigate how to improve reward modeling (RM) with more inference compute for general queries, i.e. the $\textbf{inference-time scalability of generalist RM}$. For the RM approach, we adopt pointwise generative reward modeling (GRM) to enable flexibility for different input types and potential for inference-time scaling. For the learning method, we propose Self-Principled Critique Tuning (SPCT) to foster scalable reward generation behaviors in GRMs through online RL, to generate principles adaptively and critiques accurately, resulting in SPCT-GRM models. Furthermore, for effective inference-time scaling, we use parallel sampling to expand compute usage, and introduce a meta RM to guide voting process for better scaling performance. Empirically, we show that SPCT significantly improves the quality and scalability of GRMs, outperforming existing methods and models in various RM benchmarks without severe biases, and could achieve better performance compared to training-time scaling. SPCT-GRM still meets challenges in some tasks, which we believe can be addressed by future efforts in generalist reward systems. The models will be released.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper aims to improve pointwise generative reward modeling (GRM) via: (A) a new training scheme, Self-Principled Critique Tuning (SPCT), consisting of one round of rejective fine-tuning followed by rule-based online RL; and (B) test-time scaling through parallel sampling with voting, guided by a meta reward model that filters low-quality samples. The proposed pipeline reports improvements over comparably sized and larger models on multiple reward-modeling benchmarks.

### Strengths
The paper tackles an important problem: improving reward models; which could have substantial impact.

### Weaknesses
- **Unclear novelty**. It is not clear whether the contribution is primarily the integration of existing techniques (rejection sampling, GRPO, expert voting) for reward modeling, or whether the fine-tuning and inference-time scaling methods themselves are novel. If the latter, comparisons to closely related methods are missing, making it hard to assess the incremental contribution.

- **Fairness of comparisons**. In Table 2, the method appears to compare fine-tuned reward models with additional inference budget against pre-trained baselines. A fairer comparison would allocate the same fine-tuning and inference budgets to baselines using standard techniques (e.g., GRPO, self-consistency).

- **Limited ablations.** The pipeline has several components (rejective fine-tuning, rule-based RL, voting, meta reward model). Additional ablations are needed to quantify the contribution of each part.

### Questions
- The paper repeatedly emphasizes flexibility across input types and lists it as one of four desiderata for reward models. I did not find a concrete justification for this motivation. Can the authors provide additional explanation, examples, and/or literature supporting why input-type flexibility is critical here?
- Section 2.2: How are the filtered principles used in practice? Are principles generated once and reused across questions, or is a new set generated per question?

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
5

### Summary
This work develops generalist reward models (GRMs) for LLMs that are flexible across various input types (e.g., single, paired, or multiple responses) and scalable at inference time. The authors show that conventional scalar and pairwise RM paradigms often fail on one or both of these fronts.

### Strengths
1. The main strength of this work is the combination of the three distinct concepts to build a scalable GRM. The idea of adaptive principle generation trains the GRM to generate principles dynamically based on the specific query and candidate responses, and the model can achieve a more context-aware evaluation. Also, the authors use RL to train the GRM itself, allowing for self-improvement. Finally, the authors show that a Meta RM can guide inference, and the authors introduce a filtering mechanism that improves the reliability of the final aggregated reward.   

2. The experimental design is thorough, showing credibility to the paper's claims. The authors experiment on fair baselines by re-implementing alternative methods (e.g., LLM-as-a-Judge, BT-RM, PairRM, CLoud-Gemma) on the same base model (Gemma-2-27B) and using compatible training data. The authors' effort ensures a robust evaluation, since the performance differences are attributable to the method itself, not confounding factors like base model choice or training data.   

3. The paper motivates the need for using GRMs. The taxonomy of RM approaches presented in Figure 2 is a contribution in its own right, offering a clear categorization of reward generation paradigms (scalar, semi-scalar, generative) and scoring patterns (pointwise, pairwise).   

4. The comparison between inference-time and training-time scaling is also significant. The result that the 27B model with Meta RM-guided voting can outperform a much larger 671B model with greedy decoding provides strong empirical support for the "inference scaling laws" hypothesis.

### Weaknesses
1. The central idea of the paper (i.e., training a model to generate natural language principles that explain and reconstruct preference data) is precisely the problem statement in the recent paper Inverse Constitutional AI (Findeis et al., 2024). Their framework explicitly formulates this as a compression task: finding a compact, human-readable "constitution" (a set of principles) that allows an LLM to reproduce the judgments in a large preference dataset. The SPCT method can be understood as a specific, end-to-end algorithmic solution to their problem. In Findeis et al, they use a multi-step pipeline of generation, clustering, testing, and filtering to extract these principles. SPCT's use of RFT and RL is a different approach to the same fundamental goal. The authors do not cite this work, nor discuss, or differentiate themselves from this highly relevant body of work. This overstates the conceptual novelty of Self-Principled Critique Tuning" and misses an opportunity to frame the work as a new, and possibly superior, method for solving the problem.

2. The paper repeatedly describes its second training phase as "rule-based online RL". This is inconsistent with the standard definition of online RL. Online RL involves an agent learning from data collected through direct, on-policy interaction with a dynamic environment. The policy is updated, it collects new data with the new policy, and this loop continues. The method described in this paper, however, uses the GRPO algorithm to fine-tune the GRM on a static, pre-collected preference dataset. The "rollouts" are simply generations from the GRM on prompts from this fixed dataset, and the "reward" is a binary signal based on comparison to a static ground-truth label. This is a form of offline policy gradient fine-tuning. This terminological inaccuracy is not semantic -- it is confusing and mispositions the work relative to research in online RL for reward model training, especially for agentic tasks. For example, recent methods like OPRL/iStar describe frameworks where a reward model is updated concurrently with an agent's policy using newly collected trajectories from an interactive environment. The authors should use a more precise term, such as "offline RL fine-tuning" or "policy gradient optimization on preference data" to describe their method.   

3. The paper uses methods to guide critique generation, which is a form of structured reasoning. This connects to a lot of research which focuses on explicitly improving the reasoning capabilities of GRMs to make them better judges. For example, ReasonGRM use a multi-stage framework to generate and filter high-quality chain-of-thought reasoning paths to train a more robust GRM. Other works also showed correlations between an LLM's general reasoning performance and its effectiveness as a generative judge. The principles in this paper are a simplified form of reasoning. The current work focuses entirely on the final outcome (accuracy), neglecting the process (i.e., the principles themselves), which are a key element in this related work.   

4. Though the quantitative results are strong, the paper does not have a deep qualitative analysis of its core mechanism: the generated principles. The case studies in the appendix show full model outputs but do not systematically analyze the principles themselves. What kinds of principles does the model learn? How do they evolve from the RFT stage to the final RL-tuned model? Do they differ meaningfully across domains (e.g., coding vs. creative writing)? Answering these questions is important to better understand what the model is learning and whether it has developed a genuinely robust and interpretable set of evaluation criteria.

5. (Minor) I would appreciate if the authors improve the clarity of their figures; in particular Figure 2. This is a really important and useful figure, but it is not completely clear. Please clarify what each method does in the caption for readers that are less familiar with the different approaches, and also explain why some of them are e.g., inference scalable, etc.

### Questions
Q1. Could you please elaborate on the relationship between your work and the Inverse Constitutional AI framework? Specifically, how does the SPCT training process differ from or improve upon existing algorithms, which often rely on a "generate cluster filter" pipeline?   

Q2. How do the principles generated by the final SPCT-GRM differ from those produced by the initial RFT model? Do they become more abstract, specific, or complex through the RL phase?

Q3. Do the generated principles vary in a structured way across different task domains (e.g., the Reasoning, Safety, and Chat subsets of Reward Bench)? 

Q4. Could you provide additional details on the meta RM training process?

Q5. What was the approximate ratio of positive (correctly ranked) to negative (incorrectly ranked) examples in the Meta RM's training set?   

Q6. Did you evaluate the Meta RM as a standalone classifier? How well-calibrated are its scores? Does its performance degrade when judging outputs from a policy that drifted significantly from the training data?

### Soundness
3

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
4

### Summary
- This paper proposes self-principled critique tuning (SPCT), a training procedure aimed at improving the generalist capability of reward models. The two specific goals are to develop generalizable reward models (to evaluate many tasks) and those that scale with test-time compute.
- This is done by combining a rejection sampling and rule-based online RL that aim to 'teach' the model how to generate correct principles for different tasks, and how to derive rewards based on these principles. A meta reward model is also trained to filter out low-quality reward samples before voting.

### Strengths
- The paper targets a timely question: of improving reward modeling with scaling test-time budget and generalist capabilities (essentially teaching the RM a system of evaluating through principles); while similar principles are applied to improve reasoning, the focus of this work is on improving RMs (which AFAIK, is under-explored)
- The training recipe (SPCT) is clear and follows established algorithms that have been shown to work (rejection finetuning, online RL with rule-based rewards, and meta-RM). In some sense, the work applies a popular recipe to improving reward models and not reasoning performance
- The ablations indicate that principle guidance, and hinted/non-hinted sampling improve performance and inference time scaling; the empirical results are broad and generally convincing

### Weaknesses
- My feeling is that the claimed novelty is somewhat overstated. Almost every building block is known: GRMs that produce critiques; RL-style optimization of LLMs (here RMs); inference-time scaling (through best-of-k, majority voting); and learned verifiers (or here learned verifiers of learned verifiers). This also includes the post-training procedures, e.g., rejection sampling + GRPO-style RL. To me, the main novelty is such techniques are explicitly utilized to optimize for inference-time scalability of reward models. In this sense, the authors should more clearly articulate what is genuinely new and what is more 'systems design'
- The idea of SPCT, of tuning principles, can be explained more clearly in the initial sections. This is a point about clarity only. The high-level idea is quite simple, and it is about improving principle generation/rationalization for generating reward scores; and the method used to achieve this is rejection sampling + RL.
- One limitation of the experiments is that they don't really demonstrate 'generalist' reward modeling in a convincing way; the benchmarks are largely instruction-following / helpfulness / preferences setup. Perhaps I misunderstood, but the paper leans on the 'generalist RM' angle quite heavily.
- The claim that scaling test-time inference > scaling model size during training is fine, but also uncalibrated. For example, sampling N times a 27B model might be more expensive than a single pass of a 200B+ model

### Questions
- From an algorithmic standpoint, what are the specific ingredients of SPCT that prior GRM / LLM-as-judge method do not have. Is it the decision to treat principle generation as part of the learned policy? Or the meta-RM guided voting for reward generation
- About MetaRM: The meta reward model is trained as a binary classifier over sampled trajectories, to predict whether a trajectory’s extracted scores match ground truth. This is effectively a “judge of the judge.” How robust is MetaRM to distribution shift (e.g., new domains not in training)?
- It would be good to back up the claim of 27B TTC rivaling 671B with normalized efficiency numbers (e.g., total FLOPS or tokens); it is harder to interpret the practical significance of the scaling claims.

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
3

### Summary
This paper addresses the issue of reward modeling for RL post-training for LLMs. It tackles this problem through introducing a technique called self-prinicipled critique tuning (SPCT) for generalist reward modelling (GRM). Essentailly, SPCT first gets the model to generate its own criteria, specific to the task, then uses those criteria to generate a score.

Authors also did experiments with a model that is finetuned with this technique (SPCT-GRM-27B) to show SPCT doing well in various reward benchmarks, and also being scalable at inference-time through using muiltiple of such reward models and voting.

### Strengths
- Solid engineering work that could potentially be useful / scaleable for real-life scenarios.
- Rigorous testing methods with reproduction of baselines from other papers.
- The authors provide a fair towards its own work also, e.g. The authors highlight other models performing better on benchmarks, and showing failure modes in Appendix.

### Weaknesses
- Limited novelty - there are numerous papers in "LLM self generating criteria for further scoring"
- Limited metrics on inference - The authors mentioned that 32x SPCT-GRM matches a 671B model in performance. However, is the inference time faster / compute requirement less? 
- Performance - Is the performance gain mostly from scaling voting / the meta-RM model? If so, does the SPCT component actually matter? 
- Potential data contamination issue? e.g. I noted that the authors finetuned its model on several datasets including Skywork-80k, which has overlap with RB (https://huggingface.co/datasets/natolambert/skyworks-rewardbench-contamination)

### Questions
- Has the author explored whether your implementation of RM improves finetuning performance? Would be interesting to see whether benchmark performance translates to real-life performance
- To address the potential data contamination issue above, has the author looked into zero-shot versions of SPCT without finetuning? 
- In addition, it seems that the meta-reward model is doing a lot of heavy lifting. Has the authors looked into how / why that is the case? 
- Any analysis on whether the principles generated actually correlate to the task? 
- Additionally, did the authors look into how the decision boundary / score distributions of the reward models look like?

### Soundness
2

### Presentation
2

### Contribution
2
