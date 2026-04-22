# PROS: Towards Compute-Efficient RLVR via Rollout Prefix Reuse

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Large reasoning models (LRMs) trained with *Reinforcement Learning with Verifiable Rewards* (RLVR) have achieved remarkable progress on complex reasoning tasks. However, RLVR heavily relies on on-policy rollout generation, whose cost grows rapidly with rollout length and model size, eventually becoming the training bottleneck. Our empirical analysis reveals that independent rollouts for the same query often share similar early steps, indicating substantial redundancy. To address this, we propose **Pros** (**P**refix **R**euse for **O**n-policy **S**ampling), a paradigm that reuses promising prefixes of historical rollouts in RLVR training. **Pros** appends these self-generated partial rollouts to the original queries to form *Augmented Queries*, which are then used as regular training inputs in subsequent iterations, thereby reducing redundant computation. To select training batch from augmented queries, **Pros** adopts a hierarchical Bayesian model to estimate their pass rates and prioritize those with the highest reward uncertainty. Experiments across diverse settings show that **Pros** consistently improves training efficiency and achieves higher accuracy than strong baselines. These results highlight **Pros** as a practical path toward scalable and compute-efficient RLVR.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces PROS, an RLHF training paradigm that reuses correct-prone trajectories to augment raw prompts, thereby improving exploration toward correct responses. PROS employs a Bayesian model to prioritize training on augmented prompts with high uncertainty, via token-level or progress-based values. Experimental results show that PROS achieves faster convergence and higher accuracy compared to baseline methods.

### Strengths
+ It tackles an important problem in RLHF.
+ The design is conceptually clear and practical to implement.
+ The paper is overall well-written.
+ Evaluations demonstrate good speedups and accuracy gains.

### Weaknesses
- Evaluations are limited to only two datasets and miss some ablation studies. 
- The methodological novelty is moderate; the core ideas resemble existing works such as TreeRPO[1].

### Questions
Thank you for submitting your work. The paper is clearly presented and provides a reasonable solution to improve RLHF efficiency. However, the contribution would be stronger with more comprehensive experiments and deeper analysis. Below are specific comments and questions:

- Q1: How does PROS perform on other datasets, such as the code generation task or MATH500? Using only AIME24 and AMC23 limits the generality of the results. 

- Q2: Reusing correct-prone trajectories may improve sample efficiency but risks reducing response diversity. Can the authors discuss potential strategies to mitigate this effect?

- Q3: The ablation study on Bayesian Inference (BI) for prompt selection is appreciated. Could the paper also compare simpler alternatives (e.g., variance-based scoring) to justify whether BI offers sufficient benefit to warrant its added complexity?

- Q4: In Figure 5(a), some baselines (e.g., Vanilla PPO) appear to have been early-stopped before convergence. Could the paper clarify the stopping criteria?

- Q5: Tables 2 and 3 include useful hyperparameter studies. It would be better also to show how the baseline methods perform under the same hyperparameter variations for a fairer comparison.

Reference

[1] TreeRPO: Tree Relative Policy Optimization, arXiv:2506.05183.

### Soundness
2

### Presentation
3

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
The manuscript proposes to improve the generation phase of Reinforcement Learning by not generating each complete trajectory separately but instead identifies promising prefixes for a given problem and then only generates additional trajectories based on those prefixes. The author call their approach PROS (Prefix Reuse for On-policy Sampling) and the combination of problem and trajectory prefix Augmented Queries. These new queries can then be added to the training dataset in subsequent training steps and can be treated like regular problems, however with less redundant computation necessary.

Prefixes are chosen based on uncertainty, for which token-level entropy is used, as well as value function offered by the critic model, if available in the given setup. To select Augmented Queries, PROS uses a hierarchical Bayesian model to estimate their pass rates and focuses on queries, where reward uncertainty is the highest to improve exploration. Additionally optimization techniques such as exponential forgetting of rewards are employed, so that more recent Augmented Queries are selected, giving more weight to the improved policy.

### Strengths
* address a relevant topic, i.e. how to improve the efficiency of the generation phase of the RL pipeline, and is also timely
* manuscript is mostly well written with the idea being clearly presented and the theoretical underpinnings derived well
* reasonable evaluation, which includes a similar reasonable ablation study

### Weaknesses
The evaluation is reasonable (two datasets, three baselines and a little bit of ablation, which is also reasonable), however it is also somewhat limited, since only a single, small, model is evaluated. Some additional ablation study on an ever smaller model is provided. Additionally the computational environment is not discussed even though metrics such as wall clock time are reported. 
The analysis is at times spotty, i.e. not comprehensive enough, for example one of the motivations was to address redundancy in the generation (at least the last part of Section 2 as well as the first part of the related work section give that impression), but the token cost increase (at least for AIME-Old).
Some notation issues, such as using certain symbols twice or inconsistent parameter order, muddy the otherwise clear writing.

minor issues:
* introduction, page 1:
  * first paragraph: reference for CoT is missing
  * second paragraph: "PPO(Schulman" and "GRPO(Shao" - missing whitespace
* chapter 2:
  * page 3, first paragraph: "And the objective" - language, a sentence should not start with "And"
  * page 3, Redundancy in Early Reasoning Steps: "on DAPO-Train dataset" - language: add a "the" before DAPO-Train?
  * Figure 2b/c: It might be worth discussing the two metrics a little bit, since they seem to go into different directions - normalized edit distance: smaller values indicate higher similarities, where it is the opposite for ROUGE-L?
    * also ROUGE-L (text) is written inconsistently (vs. Rouge-L in Figure 2c)
* 3.1, last paragraph: "remains the on-policy nature of RLVR algorithm" - language: sounds strange to me, maybe "retains the on-policy nature of the RLVR algorithm"
* Figure 3:
  * You might want to careful with the notation. In Section 2 $y_i$ is an individual thought whereas here it represents a whole trajectory/rollout.
  * "Promising prefix identification" - the other titles of the rounded rectangles are capitalized
* 3.2, page 5, first paragraph: "leverage the readily available value function learned by critic model for those actor-critic methods"
  * you claim that PROS can be easily be plugged into GRPO, but GRPO does not have a critic models, but derives its value function differently: in the subsequent paragraph this is presented more clearly
* 3.2, last paragraph:
  * "with highest token-level" - language: add "the" before "highest"?
  * maybe explain what $H$ is; here it is probably the log-likelihood, but later it is used for the history (in 4.2)
* Figure 4: What is $\mathbb{S}_j$? Probably the children of parent j. (After reading further it is discussed in 4.2, so after the figure is displayed.
* 4.2:
  * Proposition 4.1: "V)where" in equation, at least a whitespace is missing, maybe even an additional comma
  * sometimes the History $H$ is $(n, s)$ (at the beginning, Algorithm 1) and sometimes the other way around (for the propositions); similar issues in Appendix B.2
* 5.1: title is probably not necessary, if Section 5 has not other subsections
* references:
  * cited differently than other arXiv references (can only be surmised from the URL); additionally consider capitalizing the titles to be consistent, especially abbreviations and proper names:
    * [Dumitrascu et al. 2018]
    * [Fedus et al. 2020]
    * [He at al. 2025]
    * [Hong et al. 2022a/b]
    * [Lambert et al. 2025]
    * [Liang et al. 2021]
    * [Liu et al. 2025a/b]
    * [Narvekar et al. 2020]
    * [OpenAI et al. 2024]
    * [Polson et al. 2013]
    * [Razin et al. 2025]
    * [Schaul et al. 2016]
    * [Schulman et al. 2017]
    * [Shao et al. 2024]
    * [Snell et al. 2024]
    * [Soviany et al. 2022]
    * [Wang et al. 2021]
    * [Wang et al. 2017]
    * [Xi et al. 2024]
    * [Yu et al. 2025]
  * consider capitalizing the titles to be consistent, especially abbreviations and proper names:
    * [V Team et al. 2025b]
* B.3:
  * title: It is probably Proposition 4.2 instead of 4.1
  * page 16:
    * line 810: unnecessary whitespace at the beginning of the line
    * equation cuts into the margin
    * line 826: comma should probably be at the end of the previous line, after the equation
* C:
  * uses PROST instead PROS
  * line 843: "from replay buffer" - language: add a "the" before "replay"; same line: "by current policy" - add a "the" before "current"

### Questions
* Figure 2a: I did not understand what fraction is shown in that subfigure. Of the whole policy optimization algorithm?
* What is the computational overhead for the selection and reward estimation?

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
3

### Summary
The paper tackles a bottleneck in RLVR: the high cost of on-policy rollouts. The authors observe that many rollouts for the same prompt share identical or near-identical prefixes. PROs leverages this redundancy by reusing verified prefixes from previous rollouts constructing augmented queries that begin from these prefixes, and then sampling only the continuation (suffix). This idea is known before. See below. 
The authors talk about a hierarchical Bayesian selection scheme to identify which prefixes are most promising to reuse, prioritizing uncertain or under-explored augmented queries. That was interesting and to my knowledge novel. 

 Comparison: 

SPEC-RL (Liu et al., 2025) also reuses verified rollouts but does so at a coarser granularity, caching *entire* rollouts or long segments. PROS’s is finer-grained.  
 AR3PO (Zhang et al., 2025) proposes *response reuse* (retraining on previously verified answers) but does not address redundancy within rollouts themselves. PROS directly targets intra-trajectory redundancy.
TreePO as the name suggests (MAP/ByteDance, 2025) exploits shared prefixes structurally through tree-based rollout planning.  

There are earlier works going to Kearns et al I believe. But overall,  PROS represents a clear step beyond existing efficiency methods by focusing on *within-trajectory reuse*, introducing a principled selection mechanism, and maintaining on-policy validity.

Despite minor concerns about exploration and theoretical framing, the contribution is substantial, the experiments convincing, and the writing clear. PROS is likely to influence future work on efficient LLM-based reinforcement learning.

### Strengths
Clear motivation and empirical grounding: The observation that many rollouts share prefixes is both intuitive and empirically demonstrated.
Genearlity: The prefix-reuse mechanism is orthogonal to the underlying policy architecture or reward type.
Soundness:  The Bayesian selection framework is well justified and avoids trivial over-reuse of easy prefixes.
Impact: Implementation is simple and yields immediate compute savings in RLVR systems—highly relevant for scaling large-model training.

-

### Weaknesses
Exploration bias By reusing existing prefixes, PROS may over-concentrate training around known reasoning paths, potentially limiting discovery of novel strategies. While the Bayesian selector mitigates this, empirical diversity metrics would strengthen the case.
 
Minimal theoretical framing: The paper remains largely empirical; a more formal analysis of sample complexity improvement or convergence would enhance rigor.

### Questions
none

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper targets the **on-policy rollout bottleneck** in RL with verifiable rewards (RLVR) for reasoning: many on-policy rollouts for the same query share long identical prefixes, so generation time is wasted regenerating early steps. The proposed method, **PROS**, (1) identifies “promising” prefixes in past *correct* rollouts using readily available signals (token-level entropy and critic value), appends them to the original query to create **Augmented Queries (AQs)**, and then trains *only on the newly generated continuation*—preserving on-policy updates while saving generation compute; and (2) runs a **hierarchical Bayesian selector** over the growing parent→children (query→AQ) tree to prefer items with uncertain pass rates (≈0.5), sharing strength between a parent query and its AQs.   
PROS integrates as a plugin to PPO/GRPO and is evaluated primarily on math benchmarks (AIME24, AMC23), reporting Pass@1 and compute trade-offs; Dynamic Sampling, Prioritized Sampling, and Experience Replay serve as baselines.

LLM usage disclosure: I used a large language model to reorganize wording and improve clarity in this review. The judgments expressed are my own.

### Strengths
* **Originality/Significance:** The paper addresses a critical RLVR bottleneck. Its novelty lies in integrating prefix reuse (which *creates* a structured dataset) with a Bayesian selector *tailored* to that hierarchical structure.
* **Methodology:** The method is sound. Applying gradients only to the new continuation preserves the on-policy nature of RLVR. The hierarchical Bayesian selector with exponential forgetting is a principled way to share statistics and adapt to the non-stationary policy.
* **Empirical Quality:** Experiments use strong, relevant baselines like Dynamic Sampling. Figure 5 shows PROS achieves a better performance-cost trade-off, outperforming Dynamic Sampling while being 2-4x faster in wall-clock time. An ablation confirms the selector's value over random sampling.
* **Clarity:** The paper is well-written, using Figures 1 and 3 to effectively illustrate the core concepts of prefix redundancy and the PROS pipeline.

### Weaknesses
* **Limited Evaluation Metric (Pass@1):** The evaluation relies exclusively on Pass@1. This metric fails to show if the model is learning *diverse* solutions. Pass@K (K=32/128, etc.) is a feasible and important metric that is missing.* **Unclear Final Capability vs. Efficiency:** The paper's central claim is a better performance-cost trade-off, and it excels at showing faster initial learning (Figure 5). However, the core methodological trade-off against baselines like Dynamic Sampling (DS) is not fully resolved.
* **Risk of "Hard-Tail" Starvation:** PROS explicitly targets queries with high reward uncertainty (pass rate $\approx 0.5$). This is a known efficiency strategy, but it risks starving the model of exposure to the hardest problems (pass rate $\approx 0$). In contrast, DS is *specifically designed* to focus on this hard-tail. The experiments, which are limited in training time, do not establish whether PROS's efficiency-first approach can match the final performance ceiling of a capability-first method like DS, given a matched (and sufficient) compute budget. The paper is missing a clear convergence-point comparison.
* **Un-ablated Prefix Heuristics:** The prefix identification rule (entropy + value) is purely heuristic . The paper provides **no ablation** to justify this choice over simpler ones (e.g., value-only). The main "PROS-ablation" only tests the *selector* (Component 2), not the *prefix identification* (Component 1).

### Questions
1. **Pass@k and budget matching.** Please report **Pass@k** (e.g., k=8/32/128) in addition to Pass@1, with matched budgets, to separate exploration breadth from sampling reweighting gains. (Current results focus on Pass@1. )

2. **Convergence & final comparison.** Provide **longer-horizon** training curves and **final** metrics comparing PROS to **Dynamic Sampling** and other baselines under **identical compute budgets**. Check if methods like Dynamic Sampling eventually catches up or surpasses PROS (as hinted on some setups) and discuss whether PROS’s early gains diminish over time. 

3. **Hard-subset analysis.** Report performance on the **hardest part** of items and discuss whether mid-difficulty targeting (pass rate ≈0.5) reduces exposure to very hard queries (pass rate ≈0). (Selector targets ≈0.5 region. )

4. **Prefix heuristics ablation.** Ablate the **prefix-picking** rule: (a) value-only, (b) entropy-only, (c) random correct-prefix; include efficiency and accuracy. (Heuristic defined in Sec. 3.2. )

5.  **Generality to Other Domains:** Do you have any results applying PROS to a non-mathematical RLVR task, such as code generation?

### Soundness
3

### Presentation
3

### Contribution
3
