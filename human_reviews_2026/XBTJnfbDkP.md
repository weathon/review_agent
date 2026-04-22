# Transitive RL: Value Learning via Divide and Conquer

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
In this work, we present Transitive Reinforcement Learning (TRL), a new value learning algorithm based on a divide-and-conquer paradigm. TRL is designed for offline goal-conditioned reinforcement learning (GCRL) problems, where the aim is to find a policy that can reach any state from any other state in the smallest number of steps. TRL converts a triangle inequality structure present in GCRL into a practical divide-and-conquer value update rule. This has several advantages compared to alternative value learning paradigms. Compared to temporal difference (TD) methods, TRL suffers less from bias accumulation, as in principle it only requires $O(\log T)$ recursions (as opposed to $O(T)$ in TD learning) to handle a length-T trajectory. Unlike Monte Carlo methods, TRL suffers less from high variance as it performs dynamic programming. Experimentally, we show that TRL achieves the best performance in highly challenging, long-horizon benchmark tasks compared to previous offline GCRL algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method called Transitive RL for offline goal-conditioned RL. It introduces a novel divide-and-conquer approach to learn a value function with reduced accumulated error. On the recent OGBench benchmark, it demonstrates superior performance compared to well-established baselines such as IQL and CRL.

### Strengths
1. This paper provides an insightful connection between $V(s, g)$ and $\max_w V(s, w) V(w, g)$. Accordingly, the authors derive a novel divide-and-conquer value function to replace the target in Eq. 10.

2. The proposed algorithm is simple and achieves strong performance on recently proposed benchmarks.

### Weaknesses
The authors’ main hypothesis is that a divide-and-conquer algorithm *might* provide a path to overcoming the curse of horizon (line 55). I would argue that this is not a well-defined hypothesis. It would be more appropriate to state that a divide-and-conquer value learning algorithm can mitigate the curse of horizon in offline goal-conditioned tasks. Accordingly, **the paper’s framing should be revised to emphasise the offline goal-conditioned setting, rather than starting broadly from off-policy RL**.

I find this submission interesting; however, the presentation and narrative could be improved. The paper lacks a smooth transition from the problem statement (and background) to the proposed idea, and then to the corresponding solution.

### Questions
1. In Table 1, TRL significantly outperforms the selected baselines. However, I encourage the authors to include more recent offline GCRL methods, such as HIQL [1] and WGCSL [2].

2. It would be preferable to combine Sections 5.1 and 5.2, as they largely overlap, and move the ablation studies to the main pages.

3. Section 5.1 shows that TRL performs well on large-scale, long-horizon tasks. However, this evidence alone is insufficient to demonstrate that the performance gain arises from the divide-and-conquer value learning. The authors should design more detailed experiments to verify that the value function learned by the proposed method is indeed more accurate and less prone to overestimation, as stated in line 256.

**References**

[1] Park, S., Ghosh, D., Eysenbach, B., \& Levine, S. (2023). HIQL: Offline Goal-Conditioned RL with Latent States as Actions. NeurIPS. 

[2] Yang, R., Lu, Y., Li, W., Sun, H., Fang, M., Du, Y., … Zhang, C. (2022). Rethinking Goal-Conditioned Supervised Learning and Its Connection to Offline RL. ICLR.

### Soundness
2

### Presentation
1

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
This paper introduces a method called TRL for GCRL. The idea is to long horizon tasks by formulating them as GCRL which then allows "dividing and conquering" to stitch together shorter plan segments to avoid the small errors that normally stack up and ruin long-term plans.

### Strengths
The idea seems sensible and the results seem strong. It seems to make the idea of triangle inequality for value learning work, although I am not very familiar with current GCRL literature.

### Weaknesses
**W1.** Complicated algorithm with lots of moving parts: eta-quantile, M subgoals, expectable loss, reweighting, separate "oracle distillation" network, policy extraction. This is many more "moving parts" than standard algorithms like IQL or TD-n, which could make it difficult to tune and implement.

**W2.** As far as I understand it relies on oracle representations (see Appendix D.2). This seems like a significant weakness.

**W3.** Structure: The triangle inequality is referred to in the abstract, introduction, related work (where there’s a whole subsection on it) etc, but only properly introduced in the middle of page 4. The triangle inequality really needs to be described earlier - in or before the dedicated discussion paragraph in related work.

**W4.** The paper tests on one suite of benchmarks. It would benefit from results on another set of benchmarks.

**W5.** Limited to GCRL and deterministic environments.

Other weaknesses: See questions.

### Questions
**Q1.** The authors state they take some baseline results from a prior paper (Table 1) but re-run others (Table 2). For the re-run baselines, how was fair tuning ensured? For example, the SGT-DP and COE baselines (Table 1) are also triangle-inequality methods that perform poorly - how can the authors be sure this is not due to a poor implementation or unfair tuning on their part?

**Q2.** The oracle distillation seems important but it is only mentioned in Appendix D.2. Could you do an ablation of this?

**Q3.** Using the BCE loss over the MSE loss seems unusual. Why was this chosen? Did you try MSE as well?

**Q4.** The authors claim "we experimentally find that behavioural subgoals can still be highly effective even when the dataset consists of uniformly random atomic motions". Which experiments does this refer to? It would strengthen the paper to have results across different dataset types - "random," "medium," and "expert" - to support this claim.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Transitive Reinforcement Learning (TRL), a new divide-and-conquer framework for value learning in offline goal-conditioned reinforcement learning (GCRL). The core idea is to exploit the triangle inequality of temporal distances to decompose long-horizon value estimation into shorter, composable subproblems. They replace the hard maximization over subgoals with soft expectile regression to mitigate overestimation bias and restrict subgoal selection to in-trajectory behavioral subgoals, which further stabilizes learning in the offline setting.

### Strengths
1. TRL’s key strength lies in its scalability to long-horizon tasks (up to 4000 steps), where it consistently outperforms or matches the best TD- and MC-based baselines. By reducing the Bellman recursion depth to logarithmic complexity, TRL fundamentally mitigates the bias accumulation problem that plagues TD methods over long trajectories.
2. In contrast to TD-n approaches, TRL attains superior performance without the need for laborious, task-specific tuning of the horizon parameter 𝑛, offering a more robust and parameter-free alternative for long-horizon value learning.

### Weaknesses
1. The proposed approach fundamentally depends on deterministic environment dynamics for the triangle inequality assumption to hold. Extending the framework to learn unbiased value functions in stochastic environments remains an important and open avenue for future research.

2. As shown in the ablation study, the results are highly sensitive to subgoal selection, which may introduce additional instability when applied to stochastic or noisy environments.

### Questions
1. The ablation study indicates that restricting subgoals to behavioral in-trajectory states is essential for achieving stable performance, whereas using random subgoals leads to a substantial degradation. This sensitivity suggests a strong dependence on subgoal quality and spatial distribution. Would the authors consider exploring an adaptive or learned subgoal sampling mechanism that dynamically selects informative intermediate goals during training?

2. It would be valuable to understand how TRL performs when extended beyond goal-conditioned reinforcement learning to more general reward-based RL tasks. Do the authors expect the divide-and-conquer principle to remain effective in such settings, and have any preliminary experiments been conducted in that direction?

### Soundness
2

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
4

### Summary
This paper presents a novel goal-conditioned RL method with D&C updates. The discussion is restricted to discrete, deterministic RL problems, but I still find the work interesting.

### Strengths
1. The paper is well written and easy to follow.
2. The proposed idea is novel and interesting. I agree with the authors that this paper is a first step towards a promising direction.

### Weaknesses
1. The authors restrict the discussion to discrete, deterministic environments with trajectory data of equal lengths, although they claim that their proposal can be extended to continuous, stochastic environments and various-length trajectories. I suggest the authors actually do such extensions and present the extended version.
2. The tasks used in the experiments are synthetic without clear real-life purposes.
3. Appendix A is unnecessary. It's just homework-level math.

### Questions
1. It is mentioned that “in practice, different trajectories can have different lengths.” I wonder about the technical details of handling trajectories of various lengths.
2. Why is Eq. 7 a multiplication?

### Soundness
3

### Presentation
3

### Contribution
3
