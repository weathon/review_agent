# RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling

- Decision: Accept (Poster)
- Scores: 4, 8, 8, 4, 6

## Abstract
Rule-based reasoning is acknowledged as one of the fundamental problems of reasoning. While recent studies show that large reasoning models (LRMs) have remarkable reasoning capabilities enhanced by reinforcement learning (RL), real applications still face severe challenges due to variations in rule formats, types, and complexity. To mitigate this issue, we introduce RuleReasoner, an effective method for rule-based reasoning via a wide collection of curated tasks and a novel domain-aware dynamic sampling approach in RL. Specifically, RuleReasoner resamples each training batch by updating the domain weights based on historical rewards. This facilitates domain balance and active learning schedules for RL, obviating static mix-training engineered by humans. Evaluations of in-distribution (ID) and out-of-distribution (OOD) benchmarks reveal that RuleReasoner outperforms frontier LRMs by a significant margin ($\Delta$4.1% on eight ID tasks and $\Delta$10.4% on three OOD benchmarks over OpenAI-o1). Notably, our approach also exhibits higher computational efficiency compared to prior methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces RuleReasoner, a reinforcement learning framework designed to enhance large language models’ rule-based reasoning ability. The authors construct a new dataset, RuleCollection-32K, covering eight logic and reasoning domains (e.g., ProofWriter, ProntoQA, AR-LSAT, LogiQA). The model is trained with Reinforcement Learning with Verifiable Rewards (RLVR) and a new sampling strategy called DADS (Domain-aware Dynamic Sampling), which adjusts training focus dynamically based on domain performance. Experiments on both in-distribution and out-of-distribution benchmarks show consistent improvements over baselines such as GRPO, DAPO, and even OpenAI o1-mini on some tasks.

### Strengths
The Domain-aware Dynamic Sampling methods is simple and experiments show that the effectiveness on enhancing the logical reasoning ability with RL.

### Weaknesses
The Domain-aware Dynamic Sampling also is not really something new. There are many works related to sampling data regarding to the weights of domains. It'd be more comvincing for the authors to discuss related works or the technical contribution of the paper is not enough.

### Questions
See Questions

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
2

### Summary
The paper targets rule-based reasoning—an area where general language models especially struggle—and proposes RuleReasoner: a reinforcement learning with verifiable rewards (RLVR) framework augmented by Domain-aware Dynamic Sampling (DADS). DADS adaptively shifts training probability toward domains with lower historical rewards, balancing progress across tasks (domains) and mitigating over-specialization. The dataset spans eight domains with varied rule types and depths; training further removes KL loss to encourage diversity and shuffles rule order to discourage memorization. Experiments on BBH, ProverQA, and related suites show that RuleReasoner-8B surpasses strong baselines (including several frontier LLMs on selected tasks), while RuleReasoner-4B approaches or exceeds peer models at similar scale—suggesting practical sample/compute efficiency and SLM-friendliness.

### Strengths
* **Addresses a real challenge:** Domain imbalance in multi-domain RL training is a genuine problem. DADS provides an intuitive solution using historical reward signals to guide sampling without requiring human-designed curricula or expensive recomputation.

* **Strong OOD generalization:** The out-of-distribution results (Δ 10.4% over OpenAI-o1 on average across BBH, ProverQA, BBEH) demonstrate generalization beyond the training data. Consistency across multiple OOD benchmarks and additional tasks (AIME, GPQA, Coin Flip) strengthens this claim.

* **Comprehensive evaluation:** The paper includes appropriate baselines across multiple categories (frontier models, prior rule-based reasoners, SFT variants, RLVR methods), statistical testing with multiple seeds, ablation studies, and insightful learning dynamics analysis (Figures 4–6).

* **Reproducibility:** Detailed hyperparameters, data sources, and prompts are provided. The authors commit to releasing code and data.

### Weaknesses
1. **Missing competitive baseline AdaRFT:** The main results table (Table 2) includes many baselines from different viewpoints, but it omits AdaRFT and DAPO (with dynamic sampling), which is arguably the most competitive related work. AdaRFT (Adaptive Curriculum Reinforcement Finetuning) also uses a dynamic sampling strategy to balance training difficulty [1], making it directly comparable to the proposed approach. In the paper, AdaRFT appears only in Appendix Table 6 (evaluated on three OOD datasets) instead of the comprehensive results in Table 2.

2. **Unclear OOD generalization mechanism:** The method (RuleReasoner with DADS) demonstrates improved out-of-distribution performance (a 10.4% gain over three OOD benchmarks), but the reason for this improvement remains unclear. The authors attribute the ID gains to balancing training across domains (preventing any single task from dominating; Line 344) while attributing the OOD gains to extrapolation capability (Line 364), but this explanation is superficial and doesn’t reveal what the model actually learns. It’s uncertain whether the model is acquiring more general reasoning patterns or simply benefiting from a data augmentation effect (i.e., seeing more diverse examples). No qualitative analysis or insight into the model’s behavior on OOD inputs is provided beyond the aggregate accuracy numbers. This lack of explanation is problematic because robust generalization is a central claim of the paper. Prior work has shown that even advanced LLMs have significant gaps in handling complex or compositional rules [2], so the reader expects a deeper discussion of how this approach addresses those gaps.

3. **Insufficient explanation for DADS’s effectiveness:** The paper provides empirical evidence that it improves performance—e.g., Table 9 shows higher accuracy at various training steps with DADS compared to without DADS. However, the analysis stops at reporting metrics and does not explain why DADS is so effective. The authors state that DADS “stabilizes the dynamics of on-policy RL training and mitigates over-optimization,” but this claim is made without deeper investigation or evidence. It remains unclear how DADS achieves these benefits: Does it prevent the model from overfitting on easier tasks by continually challenging it? Does it ensure more evenly distributed gradient updates across tasks, acting as an implicit curriculum? These possibilities are not explored. The paper essentially shows that “DADS works better” but offers little insight into the conditions or mechanisms that make it work. Can you demonstrate DADS’s necessity by comparing it to simpler baselines like uniform domain sampling, round-robin scheduling, or periodic domain rebalancing against your reweighting from historical rewards?

[1] [2504.05520] Efficient Reinforcement Finetuning via Adaptive Curriculum Learning
[https://arxiv.org/abs/2504.05520](https://arxiv.org/abs/2504.05520)
[2] Can LLMs Reason with Rules? Logic Scaffolding for Stress-Testing and Improving LLMs — ACL
[https://aclanthology.org/2024.acl-long.406/](https://aclanthology.org/2024.acl-long.406/)

### Questions
1. Why didn’t the authors include AdaRFT (and the DAPO dynamic sampling variant) in the main results? To allow a fair comparison, the performance of AdaRFT and DAPO (with dynamic sampling) on the full set of benchmarks (as in Table 2) should be reported, since it likely provides a strong baseline for adaptive sampling in RL training.

2. How exactly does dynamic domain sampling lead to better OOD generalization? For example, does the learned sampling policy encourage learning more abstract rules or reduce overfitting to training-task patterns? Without clarifying why reweighting domains yields better generalization, the connection between the proposed training strategy and the OOD success feels tenuous—we see the improvement in numbers, but not the underlying cause.

3. Why is DADS necessary or superior to simpler data-balancing schemes? A more thorough analysis (e.g., ablations of DADS components instead of entirely removing DADS in Table 9, or comparisons to a static mixed-curriculum baseline) would strengthen the justification. As it stands, we know DADS yields better results, but we don’t know what unique role it plays in training or why alternative scheduling methods underperform, leaving the rationale for DADS somewhat weak.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors create a model (RuleReasoner) for rule-based reasoning through the creation of a data collection which spans eight logical reasoning tasks with various rules and rule types. Then, a modified RLVR framework is used to train their RuleReasoner model that is designed to explore reasoning steps and improve generalization. Within the training process is a dynamic sampling algorithm aimed at reweighting domains that are undertrained. The result is that on several benchmarks, their RuleReasoner (4B, 8B) outperforms or performs competitively against significantly larger models and baseline RLVR methods both in-domain and OOD. Comprehensive analysis and ablations demonstrate the effect of the domain-aware sampling, the benefits of using their assembled data collection, as well as additional insights.

### Strengths
* A large collection of datasets for rule reasoning of varying formats difficulties/depth, inference types, etc. The experiments demonstrate that these are useful for training.
* The paper proposes a domain-adaptive algorithm for data sampling for RLVR, named DADS, which modifies the reward to be domain-normalized. The rewards per domain are also tracked and used for sampling data in the next batch.
* Comprehensive empirical results on a large suite of both logical inference and mathematical reasoning demonstrate the effectiveness of DADS.
* Detailed analysis and ablations showing how various design decisions contribute to the learning dynamics or performance.

### Weaknesses
These are minor suggestions. I did not find major weaknesses given this topic and area.

* The evaluations and collection is ultimately limited by the available data from other logical reasoning datasets; the authors did not collect or generate their own. As a result, they could be missing some rule reasoning types or be over-indexed on some. While dynamic sampling addresses the latter, it cannot fill gaps in missing reasoning types.
* Thanks for including the case studies (and C.2) – however they are not too illustrative of what is being learned. In other words, while there is learning dynamics per-domain, there is no analysis around rule/reasoning type (depth, format, chaining, etc). That would be insightful if, for example, something like C.7. was carried out for the rule types rather than domains.

### Questions
1. Is RuleCollection too easy? It can be a useful dataset for SFT/GRPO but is it still useful for benchmarking? A related question is what is the human average across the collection?

2. DADS is applicable to other multitask/generalized RLVR problems. While this paper makes no strong claims of generalizability, let’s speculate a little. Would this approach (along with no KL/Entropy bonus) work on a collection of math or coding problems? Or are there substantial limitations why it may not directly transfer?

Misc.
```
103: “It has demonstrated” -> “It has been demonstrated”
178: “RLVR, aim” -> “RLVR aiming”
250: grammar/phrasing confusing
252: Correctly answering require*s*
L373: “tranining”
```

### Soundness
4

### Presentation
3

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
This paper introduces RULEREASONER, a framework for reinforced rule-based reasoning that utilizes a "domain-aware dynamic sampling" approach. The authors claim that RULEREASONER enhances language models' ability to perform rule-based reasoning, mitigates challenges from varying rule complexities, and improves efficiency, outperforming frontier large reasoning models (LRMs) on both in-distribution (ID) and out-of-distribution (OOD) benchmarks.

### Strengths
1. This paper introduces a dynamic sampling approach to stabilize RL training across imbalanced domains, which is a practical consideration.
2. RULEREASONER achieves quantitative performance improvements.

### Weaknesses
1. Lack of comparison with existing adaptive sampling methods: Since the core contribution of this paper appears to be the "domain-aware dynamic sampling", it is better to include prior curriculum learning or adaptive sampling methods for comparison.
2. It can be seen from Table 2 that RL-based methods show relative low performance on LogicNLI, AR-LSAT compared to SFT w/ CoT methods. Could authors explain this phenomenon? Besides, according to the avg results, the performance of models after SFT w/ Short/Long CoT seems to be higher than models after RLVRs, which is contrary to popular belief. The authors should provide a brief analysis about the reasons and explain why they still select to design a data sampling method for RL instead of SFT in this situation.
3. Limited Generalizability and Scalability: All experiments are conducted on relatively small Qwen3 models (4B and 8B). This severely undermines the method’s scalability to "larger-scale modeling," especially when the authors themselves acknowledge scalability as a future work limitation. It's unclear if the observed benefits hold for state-of-the-art larger models.

### Questions
see above

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper addresses the challenge of enhancing the *rule-based* reasoning capabilities of language models by introducing a new training framework, RuleReasoner. The framework first introduces a new dataset (RuleCollection-32K), consisting of 8 different domains covering different logical reasoning forms. Then, the paper introduces DaDS, which dynamically samples batches by reweighting over the domains, then performing RLVR with rule-based exact match reward. The efficacy of this is validated with Qwen3 base models (4B and 8B) and with numerous ablations.

### Strengths
- Impressive empirical performance
- New logical rule data that would be helpful for future research

### Weaknesses
- Mathematical notations should be improved overall:
   - I believe that this notation is the clearest: $\mathcal{D}$ represents a fixed (offline) collection of $(d, q, r, y)$, where $d \in \\\{d_1, \cdots, d_n\\\}$ represents the domain. (In Algorithm 1, it states that $\mathcal{D} = \\\{d_1, \cdots, d_n\\\}$, which seemd a bit weird and contradictory to prior notations)
   - In Algorithm 1, initializing $\tilde{r}_{0, d_i} \gets 0$ is missing; for notational clarity, $m$ should be written as $m_i$
   - It should be explicitly stated that $\alpha \in [0, 1]$.
- When $r_{target} = 1$, isn't $r_{target} - \tilde{r}$ always nonnegative? This is because $|\bar{r}\_s| \leq 1$ always, and thus, $|\tilde{r}\_s| \leq \alpha |\tilde{r}_{s-1}| + (1 - \alpha)|\bar{r}_s| \leq 1$ by induction. Thus I don't think that $max(0, 1 - \tilde{r}_s)$ is not necessary; simply writing $1 - \tilde{r}_s$ is enough.
- In Section 3.2, please include why eliminating the KL term is okay for this situation. As written by Liu et al. (2025), I believe the crucial reason is that the reward is rule-based, thereby eliminating concerns about distributional shift.
- Discussions related to prior approaches on (iterative) reweighting and difficulty-aware approaches [1,2,3] are lacking. Why can prior approaches not be used directly here? Is the proposed Dads algorithm really as novel as the authors claim? What are the differences? I'm especially interested in knowing the difference between this work's reweighting scheme and [1].
   - Could it be that the importance sampling is equivalent to some form of reweighted RLVR?
- The authors claim that $\epsilon$ ensures a minimum sampling weight for all domains. But, due to the translation invariance of softmax, we have that (ignoring $s$)
$$ w_i^{norm} = \frac{\exp((v_i + \epsilon) / \tau)}{\sum_j \exp((v_j + \epsilon) / \tau)} = \frac{\exp(v_i / \tau)}{\sum_j \exp(v_j / \tau)}$$
and thus, $\epsilon$ plays no role.
- The above point leads to the following suspicion: the authors claim that in Appendix C.5, $\epsilon = 0.1$ leads to the best performance. Considering how $\epsilon$ should not impact the performance in principle (maybe floating point level discrepancy, but nothing significant), the conclusion from this ablation seems to be flawed.

[1] https://openreview.net/forum?id=lXuByUeHhd

[2] https://arxiv.org/abs/2408.09849

[3] https://openreview.net/forum?id=GLUIuli3Sm

### Questions
1. Why -1 and +1 instead of the "usual" 0-1 reward for RLVR? At least the 0-1 reward is what I usually see in RLVR literature.
2. In the abstract and throughout, the authors claim that the proposed method is more sample-efficient. And indeed the figures do support that claim. But to my eyes, at least in terms of efficiency, the gap between Rule Reasoner and other RLVR approaches (GRPO, Dr. GRPO, DAPO) seems to be a bit small...? It would be helpful to see the error bars for Figures 6 and 7.
3. As in R1, does the response length significantly increase with training? Are there any complex behaviors emerging beyond improved rule applications?
4. What do the authors mean by "the order of contextual logical rules are randomly shuffled for each training sample."? Some concrete examples would be very helpful here.
5. Here, is it usual to consider only pass@1? How about pass@k or BoN? How do RuleReasoner-4B and 8B interact with standard test-time scaling methods?
6. Have the authors considered other models? Perhaps stronger ones, such as Qwen3-Instruct or Qwen3-Thinking? How about other families like Llama? I'm not suggesting that the authors run all of these ablations (obviously due to time and resources constraints), but I am particularly curious about whether one can do the RuleReasoner framework with a stronger model and whether this would yield a better logical reasoning model.

### Soundness
3

### Presentation
2

### Contribution
3
