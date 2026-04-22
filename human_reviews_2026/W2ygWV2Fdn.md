# Mitigating Premature Exploitation in Particle-based Monte Carlo for Inference-Time Scaling

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Inference-Time Scaling (ITS) improves language models by allocating more computation at generation time. Particle Filtering (PF) has emerged as a strong ITS method for complex mathematical reasoning tasks, but it is vulnerable when guided by process reward models, which often assign overconfident scores early in the reasoning process. This causes PF to suffer from premature exploitation: it myopically commits to locally promising trajectories, prunes potentially correct hypotheses, and converges to suboptimal solutions. This failure mode, known as particle impoverishment, is especially severe under constrained computational budgets. To address this, we analyze the problem and identify two root causes: a lack of diversity in the particle set due to overconfident resampling and consequent inability to assess the potential of a reasoning path. We introduce Entropic Particle Filtering (ePF), an algorithm that integrates two new techniques to solve these issues. The first, Entropic Annealing (EA), directly mitigates particle impoverishment by monitoring search diversity via entropy; when diversity drops, it intervenes by dynamically annealing the resampling distribution to preserve exploration. The second, an enhancement called Look-ahead Modulation (LaM), adds a predictive guide to evaluate a state's potential based on its successors. On several challenging math benchmarks, ePF significantly outperforms strong baselines and achieves up to a 50\% relative improvement in task reward. Together, these methods improve PF's resilience by balancing the exploration of diverse solution spaces with the exploitation of high-reward regions, ultimately leading to higher-quality solutions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors extend particle filters for LLM output steering by adjusting the resampling step with an adaptive temperature (based on effective sample size) and systematic resampling with the aim to reduce variance.
They also base the scoring weights based on a 1-step lookahead scoring through the auxiliary particle filter, which reduces myopia in the resampling.
In an illustrative example (figure 2) the authors show that their method more accurately matches the target distribution than baseline particle filtering, and that this correlates with improved effective sample size.
Furthermore, the authors conducted many experiments to show improved performance of their method compared to baseline approaches.

### Strengths
- The authors proposed method uses well known tools from the particle-filtering literature by nicely adapting these to their use case.

- Extensive experimental section that actually does not even fit in the paper.

### Weaknesses
A lot of the math has no equation numbering for no good reason, this makes reviewing and referring to them a bit cumbersome.

I have a few issues with figure 2:
- General: high density of abbreviations that were not yet (chronologically) introduced.
- 2a:
	- There is too much happening at once with insufficient explanation for an introduction. It is not directly clear what the x-axis is supposed to mean for figure 2a).
	- label/ tick sizes are too small
	- What does GT mean as an abbreviation? Ground-Truth? Just call it "true".
	- **I think this can be solved with some minor text/ plot adjustments**.
- 2b:
	- The variances do not differ in a statistically significant sense.
	- The authors' method ePF (w/ LaM) still has large variance. The ePF only improves the variance by 15% or so. So the claim in lines 84-87 are not affirmed by this plot as it is not clear if there is some variance-threshold that needs to be satisfied.
- 2c:
	- No major issues, clear figure. In the caption the 'ESS' has a strange small bar on top of the middle S, latex mistake?

Figure 3 is not referenced in the text.

- Section 3.2, my gripe with the proposed ESS-based scheduling for the temperature is that the ESS is a purely relative measure. This means that all trajectories might be "wrong", and that resampling proportional to these references gives a false sense of security to variance reduction.
  Meaning that proportional to the true target distribution, this may help but does not guarantee lower variance as is. This may also explain the minor reduction in variance of Figure 2b. 
  I think this warrants some discussion, but it should not detract from the current method's value/ contribution.

- Section 3.3, $w_t^i$ is circularly defined in lines 234-236.

- !! Section 3.3, the paragraph in lines 237-241 strongly goes against the claims posed in the introduction. The authors state that they propose a **non-myopic** mechanism for particle weighting. However the authors method, as they admit themselves, only reduces myopic resampling. This is wrongly advertised and does not solve the myopic resampling problem.
	- With this critique, I strongly recommend the authors to look at particle-filter planners in reinforcement learning. See for example Piche's work `Probabilistic Planning with Sequential Monte Carlo methods`: https://openreview.net/forum?id=ByetGn0cYX. These methods actually do "solve" myopic behavior through the value function instead of the current 1-step lookahead on the reward function.
	  Although, as the authors also write in the "related work" section, it can indeed be difficult to learn a value function.

- Section 4: some comments on experimental reporting:
	- There are a lot of references to the appendix while being discussed as if part of the main matter. (lines 290-295; line 320; line 381-383; line 421-431)
	- Most tables do not show standard errors/ confidence.
	- Low sample sizes.
	- The plots don't explain what the confidence bands mean.

Overall, too many results and references to the appendix are crammed into the experiments section. This reduces the space for meaningful discussion of the results other than that the performance is occasionally improved (I say occasionally because most of the time it is unclear whether results are statistically significant).

*My verdict/ summary:*
I find this an interesting adjustment to particle filters for steering LLM outputs, however I find the introduction and claims to be falsely advertised.
That said, it seems that the proposed methods seem valuable and to improve performance somewhat reliably, even though statistical significance is often debatable.
Since I come from an RL background (not LLM), I find it a bit difficult to judge what the benchmark results truly mean. 
But overall, I can expect this work to seem valuable to other practitioners and researchers at ICLR.
So if the authors adjust their overly strong advertising and implement fixes to my comments, I would lean towards weak acceptance.

### Questions
- In section 3, what does it mean to set $o_t = 1$ in the approximation for $r(z_{1:t}, c)$ ? I thought $o_t$ was an observation/ score, meaning a real variable.

### Soundness
3

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
4

### Summary
This paper introduces Entropic Particle Filtering (ePF) to solve the critical problem of premature exploitation in Particle Filtering (PF) when used for language model Inference-Time Scaling (ITS). The authors find this failure, termed particle impoverishment, stems from overconfident Process Reward Models (PRMs) that cause a rapid loss of search diversity and premature convergence to suboptimal solutions. ePF addresses this by integrating two mechanisms: Entropic Annealing (EA), which adaptively monitors diversity via Effective Sample Size (ESS) to dynamically flatten the resampling distribution, enforcing exploration; and optional Look-ahead Modulation (LaM), which provides non-myopic guidance by re-weighting particles based on successor state quality. The core insight is that this adaptive exploration-exploitation balance ensures resilience to PRM miscalibration. Exensive experiments on challenging math benchmarks (like AIME) show ePF significantly outperforms standard PF and other baselines, especially under limited computational budgets.

### Strengths
* The paper provides a clear, high-quality diagnosis of a significant and practical problem in Inference-Time Scaling: the "premature exploitation" or "particle impoverishment" of Particle Filtering (PF) caused by overconfident Process Reward Models (PRMs).

* The proposed method, Entropic Particle Filtering (ePF), is an original and elegant solution. Its core strength lies in its adaptive mechanisms: Entropic Annealing (EA) directly links exploration pressure to the measured particle diversity (ESS), while Look-ahead Modulation (LaM) adds non-myopic guidance, forming a complementary and well-motivated system.

* The empirical evaluation is thorough, robust, and provides strong evidence for the method's effectiveness. It demonstrates significant performance gains over strong, relevant baselines (standard PF, Best-of-N) across multiple models and challenging mathematical reasoning benchmarks (especially AIME).

* A key finding, supported by extensive ablations and intrinsic analyses (of variance, diversity, and trajectory length), is that ePF's advantage is most pronounced in the practical, low-computational-budget (small N) regime, demonstrating superior efficiency. The comparisons are fair and effectively isolate the algorithm's contribution.

### Weaknesses
* The primary weakness is the insufficient comparison to state-of-the-art (SOTA) ITS baselines. The paper omits comparisons to more powerful, contemporary tree-search strategies (e.g., MCTS-Judge [Wang et al., 2025], Forest-of-Thought [Bi et al., 2024], or adaptive branching tree search [Inoue et al., 2025]). This makes it difficult to assess the method's performance relative to the current SOTA.

* The cost-benefit analysis of Look-ahead Modulation (LaM) is incomplete. The paper lacks a direct iso-computation comparison. It is unclear if `ePF w/ LaM @ N` particles is more efficient than standard `ePF @ (N+k)` particles, where `k` is chosen to match the total computational cost of the LaM forward passes.

* The central claim of "resilience to PRM miscalibration" is not fully substantiated. All experiments use a single, high-quality PRM. To validate this, the analysis should be extended to show how ePF's advantage (relative to PF) changes when guided by PRMs of varying quality or with different calibration errors.

[1] Wang, Yutong, et al. "Mcts-judge: Test-time scaling in llm-as-a-judge for code correctness evaluation." arXiv preprint arXiv:2502.12468 (2025).

[2] Bi, Zhenni, et al. "Forest-of-Thought: Scaling Test-Time Compute for Enhancing LLM Reasoning." Forty-second International Conference on Machine Learning.

[3] Inoue, Yuichi, et al. "Wider or deeper? scaling llm inference-time compute with adaptive branching tree search." arXiv preprint arXiv:2503.04412 (2025).

### Questions
Here are my main questions for the authors:

1.  **Conceptual Comparison with MCTS:** The paper's core motivation—mitigating premature exploitation from myopic, overconfident rewards—is the exact problem that Monte Carlo Tree Search (MCTS) is designed to solve, typically using a selection policy like UCB. The UCB formula intrinsically balances exploitation (high mean reward `Q(s,a)`) with exploration (high uncertainty/low visit count). Your proposed components seem to reinvent MCTS mechanisms within an SMC (Particle Filter) framework:
    * Entropic Annealing (EA) acts as a *global* exploration pressure, akin to the UCB exploration bonus.
    * Look-ahead Modulation (LaM) is a one-step tree expansion to improve the *exploitation* value.
    Could you elaborate on the fundamental advantages (e.g., in terms of computational cost, memory, or suitability for LLMs) of your population-based ePF approach compared to a more standard, node-based MCTS formulation for this task? **This is the point I am mainly concerned about.**

2.  **Missing SOTA Tree-Search Baselines:** Following the first question, the paper is missing comparisons to SOTA ITS baselines that are MCTS-based or use tree-search. Methods like Forest-of-Thought (Bi et al., 2024), MCTS-Judge (Wang et al., 2025), and adaptive branching (Inoue et al., 2025) have shown very strong results. How would you position ePF against these methods? A quantitative comparison, even on a subset of benchmarks, would be crucial to demonstrate the significance of this work.

3.  **Iso-Computational Cost of LaM:** The Look-ahead Modulation (LaM) adds a non-trivial computational cost (one extra forward pass per particle, per step). Could you provide a clearer iso-computation comparison?

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
3

### Summary
This paper proposes Entropic Particle Filtering (ePF), an enhanced particle filtering framework for test-time inference in large language models. Standard particle filtering often collapses early due to particle impoverishment and short-sighted reward updates. To address this, the authors introduce two mechanisms. Entropic Annealing dynamically adjusts resampling temperature to preserve particle diversity during early steps. Look-ahead Modulation incorporates one-step future predictions to avoid myopic scoring. Experiments on mathematical reasoning benchmarks suggest that ePF improves performance under limited sampling budgets and is more robust to reward miscalibration.

### Strengths
**Originality**
The work offers a new perspective on test-time computation by revisiting particle filtering and augmenting it with entropy-aware resampling and simple forward rollouts. Framing inference as sequential posterior approximation is conceptually interesting and aligns well with recent directions in resource-aware decoding.

**Quality**
The empirical section is extensive. The authors evaluate across multiple reasoning benchmarks, model sizes, and several competing decoding strategies. Analysis of weight variance, effective sample size, and failure modes is detailed and insightful.

**Clarity**
The motivation behind early exploration and long-term planning is clearly illustrated. Figures that visualize resampling behavior are helpful. The ablations demonstrate the contribution of each component.

### Weaknesses
## Weaknesses

1. The paper omits direct discussion of recent and highly relevant work such as REBASE [1] and DORA [2]. Both works study optimal allocation of inference resources and attempt to avoid premature commitment, which is closely related to the stated goals of ePF. Without comparison or positioning, it is challenging to assess novelty within this emerging line of work.

2. Table 1 raises methodological questions. The authors evaluate on random subsets of 128 samples per dataset, but the motivation for this restriction is unclear. Details such as how many subsets were sampled, how many repetitions were performed, and whether variance is reported are missing. Without these details, it is difficult to judge statistical reliability.

[1] Inference scaling laws: An empirical analysis of compute-optimal inference for llm problem-solving. ICLR 2025 

[2] Every Rollout Counts: Optimal Resource Allocation for Efficient Test-Time Scaling. NeurIPS 2025

### Questions
1. Why does Table 1 evaluate on random subsets of 128 samples per dataset? Is this due to computational constraints or dataset imbalance?

2. How many runs were averaged to produce the results on Table 1? Can the authors report mean and standard deviation?

3. How does ePF relate to DORA, which focuses on the efficient distribution of rollouts and mitigating short-horizon bias? Are the methods complementary, or does ePF subsume some of DORA’s claims?

### Soundness
3

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
This paper introduces Entropic Particle Filtering (ePF) to address the problem of premature exploitation in Particle Filtering (PF) for inference-time scaling (ITS) in large language models (LLMs), especially for multi-step reasoning tasks. The method inherits two core contributions, including: 

+ **Entropic Annealing (EA):** This mechanism adaptively adjusts the resampling temperature based on particle diversity to prevent collapse.

+ **Look-ahead Modulation (LaM):** This mechanism makes the PF approach non-myopic by adjusting the resampling weights using
predicted successor quality before resampling. 

The paper provides an extensive evaluation on multiple math benchmarks, showing significant and consistent improvements compared to other baselines, including PF or Beam Search.

### Strengths
+ Methodologically and theoretically, the paper is well-presented. The paper convincingly identifies and illustrates the particle impoverishment issue in PF under overconfident reward models. The method proposed by the paper is a novel yet practical solution. In particular, both Entropic Annealing and Look-ahead Modulation are theoretically grounded, simple to implement, and empirically effective. The authors avoid introducing complex learned components, which enhances reproducibility and interpretability.

+ Empirically, the paper has a strong experimental setup on diverse math benchmarks and a wide range of LLMs, including Qwen variants. The experimental results show consistent improvements of ePF compared to the baselines, even with or without the LaM module. The experiments also employ a comprehensive set of metrics and ablation studies to further evaluate the empirical effectiveness of ePF.

### Weaknesses
+ **On the theoretical side:** The method lack theoretical guarantees. While the empirical results are strong, there is no formal analysis of convergence or variance reduction benefits from Entropic Annealing or LaM. Even though the variance is empirically analyzed in the appendix, some justification via simplified theoretical models would improve the rigor.

+ **On the empirical side:** The generalization ability is limited to math tasks. All benchmarks are math-focused. It’s unclear how well ePF would perform on other domains or with different types of reward models. Moreover, another potential issue is the computational cost of LaM. In particular, LaM adds one forward pass per particle per resampling, which could be significant on a larger scale. While justified in performance gains, a deeper trade-off analysis on this cost would help strengthen the empirical results.

### Questions
My concerns are discussed in the Weaknesses section. I would also appreciate clarification on Table 4—does “\texttt{epf w/ LaM}” indicate ePF with LaM or without LaM?

### Soundness
3

### Presentation
4

### Contribution
3
