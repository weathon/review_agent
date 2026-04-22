# Count Counts: Motivating Exploration in LLM Reasoning with Count-based Intrinsic Rewards

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 2, 8

## Abstract
Reinforcement Learning (RL) has become a compelling way to strengthen the multi step reasoning ability of Large Language Models (LLMs). However, prevalent RL paradigms still lean on sparse outcome-based rewards and limited exploration, which often drives LLMs toward repetitive and suboptimal reasoning patterns. In this paper, we study the central question of how to design exploration for LLM reasoning and introduce MERCI (Motivating Exploration in LLM Reasoning with Count-based Intrinsic Rewards), a novel RL algorithm that augments policy optimization with a principled intrinsic reward. Building on the idea of count-based exploration, MERCI leverages a lightweight Coin Flipping Network (CFN) to estimate the pseudo count and further epistemic uncertainty over reasoning trajectories, and converts them into an intrinsic reward that values novelty while preserving the learning signal from task rewards. We integrate MERCI into some advanced RL frameworks like Group Relative Policy Optimization (GRPO). Experiments on complex reasoning benchmarks demonstrate that MERCI encourages richer and more varied chains of thought, significantly improves performance over strong baselines, and helps the policy escape local routines to discover better solutions. It indicates that our targeted intrinsic motivation can make exploration reliable for language model reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenge of exploration RL for LLM reasoning. Prevalent RL paradigms rely on sparse, outcome-based rewards and limited exploration. The authors introduce MERCI to augment policy optimization with a principled intrinsic reward. Building on the idea of count-based exploration, MERCI leverages a lightweight Coin Flipping Network (CFN) to estimate pseudo-counts and epistemic uncertainty over reasoning trajectories, converting these into an intrinsic reward that values novelty while preserving the learning signal from task rewards.

### Strengths
1. The paper revisits the classic exploration-exploitation delima in the field of of LLM reasoning and RLVR, which is a significant topic
2. The proposed method is rather novel compared to recently widespread entropy-based methods
3. The proposed Coin Flipping Network Head is lighweight and the authors have verified it on both GRPO and DAPO algorithms.
4. The authors have conducted experiments in both math and sql domains.

### Weaknesses
1. Missing baselines: There have been many works trying to solve the exploration problem such as i-MENTOR [1] and entropy-based advantage shaping [2], I would suggest the authors to compare with those baselines for a more up-to-date comparison

2. Lacking analysis: It is rather informal that the authors present the ablations in the Appendix. Moreover, readers would like more analysis on the impact of the algorithms on the RL-trained behaviors, such as whether this algorithms introduces some novel strategies, solutions, or higher level of reasoning abilities, such as cogintive behaviors. 

[1] Navigate the Unknown: Enhancing LLM Reasoning with Intrinsic Motivation Guided Exploration  
[2] Reasoning with exploration: An entropy perspective.

### Questions
See weakness.

### Soundness
2

### Presentation
3

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
The paper proposes MERCI (Motivating Exploration in LLM Reasoning with Count‑based Intrinsic rewards), an exploration module for value‑free RL fine‑tuning of LLMs. The key theoretical premise is that, in auto‑regressive reasoning, state transitions are known and deterministic, so the Uncertainty Bellman Equation (UBE) reduces to a cumulative sum of local reward‑estimate variances along a trajectory. This motivates an intrinsic bonus proportional to the standard deviation of the trajectory value, obtained by summing per‑step uncertainties and then taking a square root. To estimate per‑step uncertainty at scale, the authors attach a lightweight Coin Flipping Network (CFN) head to a separate copy of the model: the head learns coin‑flip regression to produce a pseudo‑count ($\propto 1/n(s)$) from hidden states, which is used as a proxy for reward‑estimate variance. The bonus is filtered (percentile, spatial coherence, noise suppression), standardized within a GRPO group, and then added to the advantage used by GRPO/DAPO. Experiments on mathematical reasoning (AIME24/25, MATH500, OlympiadBench, College Math, Minerva) and text‑to‑SQL (Bird, Spider) show consistent gains over GRPO and DAPO; ablations and training‑dynamics plots support design choices. Figures 1–2 illustrate the architecture and the three‑stage bonus filtering pipeline; Proposition 1 formalizes the UBE with known transitions; Tables 1–2 report the main results.

### Strengths
1. The "known transitions" observation simplifies UBE to a tractable recursion with immediate reward‑variance terms only (Proposition 1), clarifying why trajectory‑level uncertainty should sum per‑step variances before taking the square root. This is a helpful bridge between theory and practice in value‑free RL for LLMs.

2. The paper uses CFN to approximate pseudo‑counts from hidden states is elegant and inexpensive, avoiding ensembles/dropout/RND; it runs in parallel to policy learning (Figure 1). The paper also pretrains the CFN on rollouts to avoid cold‑start behavior.

3. The paper presents consistent empirical improvements across settings. The gains are steady rather than one‑off, and are larger OOD (Spider), supporting the exploration claim.

### Weaknesses
1. Theory–implementation mismatch on the trajectory bonus.

   Section 3.3 argues (correctly under the UBE derivation) that one should sum local variances along the trajectory and then take a square root. However, Equation (6) computes the bonus as the square root of the mean variance over retained tokens (division by $\ell$ ), and then normalizes within a group (Equation (7)). This normalization helps length‑invariance, but it is not what the UBE prescribes. The ablation labeled "cumulative std" vs. "cumulative variance" (Table 8) touches a related point but does not fully clarify the discrepancy between sum vs. mean in Eq. (6). Please reconcile the stated principle ("sum‑then‑sqrt") with the actual estimator and report results using the strict sum‑then‑sqrt computation.

2. CFN is pretrained on backbone rollouts. What is the wall‑clock/compute overhead and how does MERCI compare to similarly "primed" baselines (e.g., RND pretraining)?

3. The "GRPO‑scaling" baseline (step 260) shows degraded pass@k; it would be fair to also show MERCI‑scaling beyond 120 steps to check stability under longer training.

4. CFN dimension d=20, group size G=8, cosine schedule for $\gamma$ (to 10% by step 200), and the top‑p% used in filtering (20–30%). Sensitivity curves for these choices would help practitioners.

5. There are several typos (e.g., section 3.2 title "REWARE", "Mathmeatical", "iterarions", "discribed") and a few places where the exposition could be tightened (e.g., clearly defining "state": token prefix vs. hidden state embedding; whether repeats are exact or approximate).

### Questions
1. For Equation (6), why average variances before sqrt (length‑normalized) if Proposition 1 suggests a sum‑then‑sqrt? Can you report results with strict sum‑then‑sqrt (without group standardization) and then with a separate length‑normalization factor?

2. What exactly is the "state" counted by CFN? Is it the hidden state of the CFN network at each token (Figure 1)? Do you use any normalization to ensure that counts are comparable across prompts and lengths? How do you handle near‑duplicates vs. paraphrases?

3. Do high CFN bonuses correlate with (a) lower empirical success rate at those positions, (b) higher variance across rollouts, or (c) higher entropy from the policy?

4. Could you clarify more on the relation between your work and the work by Lobel et al.?

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
This paper proposes MERCI (Motivating Exploration in LLM Reasoning with Count-based Intrinsic Rewards) to add intrinsic rewards to RL for training LLM reasoning.

### Strengths
1. The paper is well motivated and the writing is clear.
2. Exploration in RL, especially in the LLM domains, is important, given that the language domain is large and exploration is hard.
3. The experiment seems to support the effectiveness of the method.

### Weaknesses
1. The Coin Flipping Network Head is never trained. Can it capture the vast space of languages and truly distinguish the novelty of the exploration steps? E.g., for two semantically similar trajectories, say two word-level-distinct responses that have the same meaning, why can we expect a random MLP head to capture this without training?
2. I appreciate the examples in Figure 4-6. But I still encourage the authors to provide a quantitative analysis of the improved exploration efficiency beyond the examples and eval performance, as this is the main claim of the paper.
3. The proposed method is similar to some previous work, such as [1]. The authors cited this paper along with some other similar works, but failed to discuss the novelty and distinctions.

[1] Gao et al., Navigate the Unknown: Enhancing LLM Reasoning with Intrinsic Motivation Guided Exploration.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces MERCI, a framework for motivating exploration in LLM reasoning using count-based intrinsic rewards. The method is grounded on a theoretical simplification of the Uncertainty Bellman Equation (UBE), which allows for estimation of trajectory uncertainty using a Coin Flip Network (CFN). The authors then convert the uncertainty into an intrinsic reward that can be integrated into RL algorithms like GRPO. Empirical evaluations on mathematical reasoning and SQL generation validate the effectiveness of MERCI.

### Strengths
1. Principled Theoretical Simplification: The paper provides a thoughtful application of the UBE to the LLM setting by removing the need to estimate environmental uncertainty. The reduction of Q-value uncertainty estimation to local reward uncertainty is precise and well-motivated.

2. The design of MERCI to use CFN for state visitation estimation is both conceptually clear and interesting.

3. The comparative results against GRPO, DAPO and the visualization of token-level estimated uncertainty support the claim proposed by authors.

### Weaknesses
- The bonus is scaled by a cosine schedule ($ \gamma $ decays to 10 % by step 200). There is no ablation on slower/fast decay, and no discussion of how this impacts the final performance.

- It seems that different reasoning tasks need to train the corresponding CFN (*e.g.*, MATH and SQL), which may limit its generalization ability. What is the computational cost of training a CFN? Can a single CFN be trained for tasks across different domains?

- I suggest citing relevant works [1][2][3] that also introduce intrinsic/curiosity reward bonuses to promote exploration.

---

[1] CDE: Curiosity-Driven Exploration for Efficient Reinforcement Learning in Large Language Models. arXiv preprint:2509.09675

[2] Consistent Paths Lead to Truth: Self-Rewarding Reinforcement Learning for LLM Reasoning. NeurIPS 2025

[3] Exploring Beyond Curiosity Rewards: Language-Driven Exploration in RL. PMLR 2025

### Questions
1. Can MERCI generalize to non-mathematical tasks such as MMLU-Pro, GPQA?

2. CFN may underestimate uncertainty for states that are linguistically close but not identical. Did the author observe or analyze any instances of failure?

3. Figure 8 shows the response length shrinks when MERCI is used. Is the policy truly “exploring”, or is the bonus merely penalising verbosity? I think an entropy-bonus scatter plot would help.

### Soundness
3

### Presentation
3

### Contribution
3
