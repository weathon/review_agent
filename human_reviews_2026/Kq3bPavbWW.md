# CAPO: Conflict-Aware Policy Optimization for Large Language Models

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Recent advancements in policy optimization techniques have profoundly improved the reasoning abilities of large language models (LLMs). A pivotal breakthrough lies in sampling a group of responses for each query and adjusting their likelihoods based on the relative advantages of their scores over the group mean. However, substantial conflicts may arise between the aggregated gradient and the individual gradients of the responses, thus diminishing the effectiveness of gradient signals and ultimately hindering the training performance. To address this challenge, we propose **C**onflict-**A**ware **P**olicy **O**ptimization (**CAPO**), a novel and scalable training method that mitigates conflicts through dynamic gradient aggregation. Specifically, CAPO formulates the gradient aggregation step as a *second-order cone program (SOCP)*, which seeks a gradient direction maximizing the alignment with positive-advantage responses, while enforcing constraints to suppress negative-advantage responses. To equip the SOCP with scalability and tractability for LLMs, we significantly reduce the number of variables via the Lagrangian duality and compress the gradient dimension using the Johnson-Lindenstrauss transform. We further show that the dynamic gradient aggregation effectively reduces conflicts without sacrificing the convergence. Experiments on several widely-used mathematical reasoning datasets and benchmarks with Qwen2.5-1.5B and Qwen2.5-3B show that CAPO consistently outperforms our baselines in terms of the accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper study the issue of conflict gradient in algorithms like GRPO, where the gradient induced by individual response might not necessary align with the overall gradient. To address this issue, the paper identified an alternative objective. The new objective find a aggregated gradient that maximize the aligning with gradient on positive gradient, such that avoiding conflict with gradient on negative response and not deviating too much from the average gradient. The paper further propose to use JL transformation to approximate the gradient to simplify the computation. Empirical results demonstrate the effectiveness of proposed method.

### Strengths
The strengths of this paper are listed as follows

1. This paper propose an interesting solution to the issue of conflict gradient, which treats positive responses and negative response separately and only applies a hard constraint on negative responses.

2. The solution to the complicated optimization problem, while sarificing some accuracy, is generally efficient.

3. The paper is clearly written and well-organized

### Weaknesses
The weaknesses of this paper are listed as follows

1. The experiment is limited to two models, each trained on only one dataset. How does the proposed CAPO method perform on other model-dataset combinations?

2. The results lacks a statistical significance analysis, which is crucial here given the improvement on in-domain test set looks very marginal.

3. The different treatment to positive responses and negative responses lacks an empirical justification. While the reviewer think such treatment do make some sense, several more ablation on constraints (e.g., apply hard constraint to positive responses and minimize the conflict with negative responses) might further justify the algorithm

4. In equation (6), it requires the aggregated gradient to be conflict-free with all negative reponses. However, this might not be achievable if the gradients scatter over different directions. Could the author provide more explaination to this issue?

### Questions
See weakness section

### Soundness
2

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
4

### Summary
This paper proposes a novel training method to mitigate gradient conflicts in LLM post-training. The authors formulate gradient aggregation as a second-order cone program and apply the Johnson–Lindenstrauss transform to reduce gradient dimensionality and improve computational efficiency. Experiments on mathematical reasoning tasks demonstrate that the proposed approach outperforms the GRPO baseline when evaluated with Qwen2.5 models.

### Strengths
The gradient conflict is an interesting phenomenon in LLM post-training, and this paper proposes a novel approach to mitigate this issue. By incorporating the Johnson–Lindenstrauss transform, the authors make the underlying optimization problem more tractable.

The paper also provides a convergence analysis for the proposed method, although the analysis itself follows fairly standard techniques in optimization theory.

### Weaknesses
My main concerns lie in the experimental evaluation:

a. All models are trained for only one epoch, which likely explains why the baseline methods perform worse than expected (e.g., GRPO achieves 0% accuracy on AIME 2024). Prior work has shown that RL-based methods can continuously improve model performance. The authors should at least train the models for several hundred steps to provide a fairer comparison.

b. The authors adopt greedy decoding for evaluation, which differs from the standard practice in the current literature. For instance, on AIME 2024, RLVR-related works typically use a temperature of around 0.7 and report the average performance over 32 or 64 samples. Greedy decoding is generally not used in practical LLM deployments.

c. All experiments are conducted on relatively small-scale models. At least one experiment on a 7B base model should be included to better assess the scalability and effectiveness of the proposed method.

d. There are no experimental results demonstrating that CAPO actually reduces the proportion of gradient conflicts during training. This evidence is critical to validate the effectiveness of CAPO. The authors should compare the proportion of gradient conflicts with and without applying CAPO.

Additionally, the presentation can be improved. For example, in Section 3.4, the authors state that $\mathcal{L}$ denotes the average loss function without providing a clear explanation, explicit definition, or connection to the previous reinforcement learning formulation, which may confuse readers.

### Questions
Could the authors provide the performance of CAPO and GRPO under a more standard setup, with larger training steps and the standard evaluation protocol (e.g., using non-greedy decoding with temperature sampling and multiple passes)? This would help clarify whether the proposed method maintains its advantage under typical experimental conditions.

### Soundness
2

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
4

### Summary
The paper introduces Conflict-Aware Policy Optimization (CAPO), a novel training method designed to mitigate gradient conflicts within critic-free policy optimization for Large Language Models (LLMs). The core contribution is formulating the gradient aggregation step as a Second-Order Cone Program (SOCP). To make this approach practical for large models, the authors employ Lagrangian duality to reduce the problem's dimensionality and the Johnson-Lindenstrauss (JL) transform to ensure computational tractability. Experiments on mathematical reasoning tasks demonstrate that CAPO consistently outperforms its primary baseline, GRPO.

### Strengths
1. The paper addresses a well-motivated and significant problem. Gradient conflict is a known and critical bottleneck for critic-free methods like GRPO, and tackling it directly is an important research direction.

2. The core idea of formulating gradient aggregation as a constrained optimization problem (SOCP) is a novel and principled approach.

3. The paper is well-written and clearly structured. The technical execution, particularly the mathematical derivations involving Lagrangian duality and the application of the JL transform, appears technically sound.

### Weaknesses
1. The baseline comparison is narrow, relying solely on GRPO. 

2. The reported performance improvements are marginal. With accuracy gains of less than 1% on both GSM8K and MATH, the practical effectiveness of CAPO is questionable.

3. The experiments are confined to small-scale models (1.5B and 3B). CAPO's performance and scaling properties on larger models (e.g., 7B or larger), where optimization challenges are often more pronounced, remain unverified.

4. CAPO introduces a non-trivial computational overhead, yet the authors provide neither a qualitative analysis nor quantitative experimental results.

### Questions
1. Could the authors provide experimental results on a slightly larger model, for example a 7B model?

2. Could the authors provide more experiments to verify the practical effectiveness of the proposed method? For example, by comparing with more recent critic-free methods such as DAPO [1], GSPO [2].

3. Could the authors provide a detailed quantitative experimental or qualitative analysis results of the computational overhead introduced by CAPO?

[1] Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476 (2025).

[2] Group sequence policy optimization. arXiv preprint arXiv:2507.18071 (2025).

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
2

### Summary
This paper targets a key limitation of critic-free policy optimization methods for LLMs (e.g., GRPO) in mathematical reasoning: gradients from positive- and negative-advantage samples within the same update often conflict with each other, which dilutes the effective learning signal. The authors propose CAPO, which formulates gradient aggregation for a batch of samples as a trust-region convex optimization (SOCP): it aligns with gradients from positive-advantage samples, constrains gradients from negative-advantage samples so they do not pull the update in the wrong direction, and keeps the final direction close to the original averaged gradient. To make this scalable to high-dimensional LLMs, they derive a dual form and apply a Johnson–Lindenstrauss projection so that the optimization is solved in the sample space rather than the parameter space. Experiments on Qwen2.5-1.5B and Qwen2.5-3B, mainly on GSM8K and MATH, show that under the same GRPO training pipeline, CAPO yields consistently small but stable gains, including on some OOD math benchmarks.

### Strengths
1. The paper quantitatively shows gradient conflicts in real GRPO training runs, demonstrating that this is a practical issue in existing critic-free pipelines rather than a hypothetical concern.
2. The priorities for positive/negative advantage samples, the directional constraints, and the trust-region control are unified in a single SOCP, which better matches the semantics and asymmetry of policy optimization than pairwise projection heuristics.
3.  With comparable compute, CAPO outperforms GRPO on GSM8K, MATH, and AMC 2023 / AIME 2024, indicating that conflict-aware gradient aggregation can bring stable improvements.

### Weaknesses
1. Limited evaluation setting. The paper evaluates mainly on small models (≤3B), math-oriented tasks, and under a relatively small training budget on fixed hardware, with no validation on larger models, longer training schedules, or non-math tasks; thus the external validity to more realistic RL-for-LLM regimes is still limited.
2. Incomplete baselines. The comparison is only against GRPO, without including contemporary open-source critic-free RL pipelines such as DAPO (Yu et al., 2025) or analyses/improvements for R1/GRPO-style training such as Dr.GRPO (Liu et al., 2025), and without systematic comparison to multi-gradient conflict-resolution methods, making it hard to judge the relative advantage of CAPO.

References
- Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., … & Wang, M. (2025). *Dapo: An open-source LLM reinforcement learning system at scale.* arXiv:2503.14476.
- Liu, Z., Chen, C., Li, W., Qi, P., Pang, T., Du, C., … & Lin, M. (2025). *Understanding R1-zero-like training: A critical perspective.* arXiv:2503.20783.

### Questions
- Choice of JL dimension. What is the rationale for setting the JL dimension to 8,192? How would reducing it to 4,096 or 2,048 affect convergence speed and final performance?
- Reward granularity. For finer-grained or step-level rewards, does CAPO’s constraint formulation need to be adapted? In particular, should the SOCP introduce weights proportional to the reward rather than only separating positive vs. negative samples?

### Soundness
3

### Presentation
3

### Contribution
3
