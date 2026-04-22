# Online Rubrics Elicitation from Pairwise Comparisons

- Avg Score: 6.50
- Decision: Reject
- Scores: 8, 6, 6, 6

## Abstract
Rubrics provide a flexible way to train LLMs on open-ended long-form answers where verifiable rewards are not applicable and human preferences provide coarse signals. Prior work shows that reinforcement learning with rubric-based rewards leads to consistent gains in LLM post-training. Most existing approaches rely on rubrics that remain static over the course of training. Such static rubrics, however, are vulnerable to reward-hacking type behaviors and fail to capture emergent desiderata that arise during training. We introduce Online Rubrics Elicitation (OnlineRubrics), a method that dynamically curates evaluation criteria in an online manner through pairwise comparisons of responses from current and reference policies. This online process enables continuous identification and mitigation of errors as training proceeds. Empirically, this approach yields consistent improvements of up to 8% over training exclusively with static rubrics across AlpacaEval, GPQA, ArenaHard as well as the validation sets of expert questions and rubrics. We qualitatively analyze the elicited criteria and identify prominent themes such as transparency, practicality, organization, and reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, the authors argue that offline rubrics (static, pre-defined rubrics) are often incomplete for RL in training LLMs. For example, they cannot adapt to emergent model behaviors like self-praising, and they are vulnerable to reward hacking. To overcome this, they introduce OnlineRubrics, a framework that dynamically elicits new evaluation criteria during training. The core mechanism involves:

1. At each training step, generating a pair of responses to a prompt: one from the current policy and one from a control policy (a reference model or the previous iteration of the policy).
2. Using an LLM extractor to perform a pairwise comparison of these responses, identify meaningful differences not covered by the existing rubric, and formulate these differences as new, weighted criteria.
3. Augmenting the original rubric with these new criteria to compute a more comprehensive reward signal for the policy update.

The authors empirically demonstrate that OnlineRubrics improves performance by up to 8% relative to baseline using offline rubrics across both generalist (AlpacaEval, Arena-Hard) and expert-domain (GPQA, GSM8K) benchmarks. A qualitative analysis reveals that the elicited rubrics tend to focus on important themes such as evidence grounding, reproducibility, and anti-gaming.

### Strengths
* **Originality**: The paper's originality is outstanding and represents its primary strength. It improves LLM alignment by moving reward generation from a static, offline process to a dynamic, online one. The idea of using pairwise comparisons between the current and a control policy during training to elicit new, emergent evaluation criteria is new to me. This creates a self-correcting loop in which the reward model evolves alongside the policy, a departure from existing RLHF/RLAIF frameworks.
* **Quality**: The research demonstrates empirical results across diverse, challenging benchmarks, including AlpacaEval, Arena-Hard, GPQA, and GSM8K. The fact that the method yields gains in both generalist and expert domains speaks to its robustness. The inclusion of multiple strong baselines (human-written rubrics, pointwise extraction) and insightful ablations provides a rigorous validation of the framework's effectiveness.
* **Clarity**: The core mechanism illustrated in Figure 1 provides a simple, intuitive walkthrough of the iterative process. The motivation is well established, and the results are presented in an easy-to-interpret way.
* **Significance**: This work is significant as it offers a promising solution to the critical problem of reward hacking and brittle reward models in LLM alignment. Static rewards are a known limitation of current RL methods. By proposing a way to make the reward function adaptive, this paper provides a practical technique that could lead to more robust, capable, and well-aligned models. The qualitative analysis of the elicited rubrics also provides significant insight into LLM failure modes and how to address them.

### Weaknesses
* **Dependence on the Extractor LLM**: The framework's performance is bottlenecked by the quality of the extractor LLM. It could degrade or misguide the training if the extractor fails to identify key differences, generates noisy rubrics, or is itself biased. A more thorough discussion of the component's potential failure modes and robustness would strengthen the paper.
* **Computational Complexity**: The proposed method introduces an additional & heavyweight LLM extractor call inside the training loop for every batch of data. This could be computationally more expensive than training with an offline rubric. An analysis of the computational overhead would be a welcome addition.
* **Scalability of Rubrics**: It looks like the rubric for each prompt tends to grow over time. It's unclear how the system handles an ever-expanding list of criteria. This may slow down overall training to an unbearable pace, or the rubrics may become too complex for another LLM to follow.

### Questions
1. How robust is the rubric elicitation process to the choice of the extractor model? Do you experiment with weaker models than o3-mini? Is there a performance threshold below which the extractor starts to introduce more noise than signal?
2. Could you provide an analysis of the computational overhead introduced by the online elicitation step compared to the baseline of training with offline rubrics? What is the increase in training time in practice?
3. A key justification for your design is that pointwise elicitation is inferior to the pairwise approach. Could you elaborate on why you believe this is the case? Is it because the pairwise comparison naturally focuses the extractor model on the most salient differences, which are more likely to be useful as discriminative criteria? Or are there other reasons?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes a new approach to generating rubric-based reward signals in LLM post-training. Instead using pre-configured rubrics per prompt, the method uses "online" rubrics that are dynamically generated based on the current model capabilities. The rubrics are per prompt rather than for the entire dataset. They evaluate their method across a number of datasets, including AlpacaEval, GPQA and ArenaHard. The authors report impressive results indicating that their online rubrics based approach is indeed able to improve performance (in terms of win-rate) across this diverse set of benchmarks.

### Strengths
1. **Clearly written.** Overall the quality of the writing is high, although some sentences are a bit long.
2. **Strong motivation.** The authors outline clearly why online rubrics are preferred over fixed ("offline") rubrics.
3. **Strong experimental results.** On the diverse set of experiments tested by the authors, they are able to show impressive performance improvements.

### Weaknesses
1. **Only single-seed experiments reported.** Since all training methods have some non-deterministic part would be interesting to see mean/var reported over, e.g., 3 seeds per method. Such an extension would allow a better understanding of the overall method.


*Minor (no impact on score, no need to respond):*
1. L039: I found the example of a rubric from a finance context confusing, I think you could select a more quickly understandable one.
2. L054: Font size in Fig 1 is quite small, had to read. Maybe some text could be removed to make space
3. L210: I don't think this example is very good, seems like both responses are fine: one considers that caltrains don't run all night (thus returning by car may be favoured)? I think there should be a clearer example.
4. L251: missing "in" for "*in* Section 5".

### Questions
1. Have you considered validating the rubrics experimentally beyond de-duplication before training on them? Related work [1] on interpreting pairwise feedback datasets extracts rubrics (referred to as "principles") but also validates the rubrics ability to reconstruct feedback. A similar approach could be used to simply check if a generated rubric is actually able to distinguish responses, as this does not appear to be done at the moment (the only check is de-duplication)?
2. Are you planning to release the datasets of (1) human evaluations for human-written rubrics discussed in Section 6.1 and (2) the generalist and expert rubrics datasets collected for evaluation? I was unable to find a corresponding link.

[1] Findeis, Arduin, et al. "Inverse Constitutional AI: Compressing Preferences into Principles." _The Thirteenth International Conference on Learning Representations_ (2025).

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Introduces dynamic rubric (as opposed to current static rubric setups) as a reward during reinforcement learning. This reduces the chance of reward hacking while training. OnlineRubrics is a framework that uses a response from a control model to update the policy while training. Final performance is improved over static rubrics (whether human or synthetic).

### Strengths
1. Nice simple proposal, easy to reason about and well explained.
2. Great that all system prompts/instruction prompts have been shared and various models have been compared in performance.
3. Reasonable improvement, and ablating various choices.

### Weaknesses
1. Not 100% clear whether the performance improvement is because of a better final rubric, or because it's dynamic during training. E.g. no ablation on whether the frequency of updating the rubric matters much, or if we use the final updated rubric and train on that from scratch. We know the human rubric outperforms the synthetic rubric by quite a bit, while it's still static, are we not creating a new rubric that is just even better (but could have been static)?
2. No real analysis on reward hacking after using the dynamic rubric framework. You do mention that it reduces the chance of reward hacking, and the assumption feels right, but there is no evidence (could even be through manual inspection) that this happens.
3. Hard to answer, but I do not see anything about what it still does wrong. I can imagine it might go into too many details, or move away from the original policy, but it would be great for future researchers to have better insight in where this framework is still failing (even though we see result improvements).
4. Would have been nice to see some smaller open-source models as judges (e.g. Qwen, or others) such that we don't rely on private APIs too much.

### Questions
1. What happens if the control model is a small/cheap model?
2. What if you update the policy using only a subset of the batch? E.g. only 2 items per batch? Does that hurt the performance much?
3. What if you use the final rubric that was dynamically updated in a new training? Are we mainly seeing improvements because we have a better rubric (similar to synthetic vs human), or because it’s actually dynamic during training?

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
4

### Summary
This paper addresses a critical limitation in rubric-based reinforcement learning for LLM post-training. The authors propose OnlineRubrics, a framework that dynamically elicits evaluation criteria during training by comparing responses from the current policy against a control policy. The key contributions are:
Novel Dynamic Rubric Elicitation Framework: A principled method for online criteria generation through pairwise response comparison, contrasting with static rubrics used in prior work (Gunjal et al., 2025; Viswanathan et al., 2025; Huang et al., 2025).
Theoretical Justification: Proposition 1 establishes that augmenting rubrics reduces the upper bound on gradient estimation error, formally motivating the approach.
Comprehensive Evaluation: Two curated datasets (Generalist and Expert Rubrics) with human-annotated, binary-verifiable criteria totaling 21,466 training rubrics and 10,941 evaluation rubrics.
Strong Empirical Results: Consistent improvements of 8-9% over static rubrics across multiple benchmarks (AlpacaEval win rate: 46.4%→55.0%, GPQA accuracy: 36.2%→38.1%, Arena-Hard: 52.4%→56.5%).
Qualitative Insights: Systematic analysis revealing that elicited criteria emphasize transparency, reproducibility, practicality, and anti-gaming properties.

### Strengths
1. Addresses a Real Problem with Practical Impact
The paper identifies a genuine limitation of static rubrics: vulnerability to reward hacking and inability to capture emergent behaviors. The "self-praising" example is particularly illustrative and motivates the work well. This problem becomes increasingly relevant as rubric-based training gains adoption.

2. Methodologically Sound Approach
The pairwise comparison strategy is well-motivated by preference learning literature (Bradley & Terry, 1952; Christiano et al., 2017). The intuition that discriminative comparison is easier than pointwise quality assessment is sensible and empirically validated (pointwise extraction baseline underperforms).

3. Rigorous Experimental Design
Two high-quality datasets with careful annotation principles (MECE, atomic, objective, self-contained)
Systematic verifier selection study (Figure 3) establishing GPT-4.1-mini as optimal cost-quality trade-off
Multiple strong baselines including synthetic rubrics, universal requirements, and pointwise extraction
Out-of-distribution evaluation on public benchmarks demonstrating generalization

4. Consistent and Substantial Improvements
Results are not cherry-picked—OnlineRubrics outperforms baselines across 8 of 9 evaluation metrics (Tables 2-3). The improvements on held-out evaluation sets (which contain no elicited rubrics) strongly suggest genuine capability enhancement rather than overfitting to generated criteria.

5. Transparency and Reproducibility
Excellent documentation in appendices: full system prompts (Figures 8-13), detailed experimental settings, data samples (Figures 5-6), and qualitative analysis with clustering. The two-stage extraction process (difference identification → criterion formation) is clearly described.

6. Valuable Qualitative Insights
The cluster analysis (Figure 7, Appendix E) reveals interpretable patterns: reproducibility (8.96%), practicality (8.33%), anti-gaming (7.69%). This provides actionable insights beyond numerical improvements.

### Weaknesses
1. Limited Model Diversity (Critical)
All experiments use a single base model (Qwen-2.5-7B-Instruct). 
Impact: This severely limits claims about general applicability. The method might be model-specific or scale-specific.

2. Short Training Duration Raises Concerns
Only 3 epochs of training
Impact: Real-world deployment requires longer training. Without this analysis, practical utility is uncertain.

3. Computational Cost Not Adequately Addressed
Impact: Method may be impractical for resource-constrained settings or large-scale training.

4. Circular Dependency in LLM-Based Components
Both extractor (o3-mini) and grader (GPT-4.1-mini) are LLMs with known limitations.
Missing: Analysis of failure modes, error propagation, or human validation of elicited criteria quality.

5. Theoretical Contribution is Weak
Proposition 1 provides only an upper bound with limitations:
No analysis of bound tightness
Impact: Theory doesn't provide actionable insights or strong guarantees.

### Questions
Thank you to the authors for this well-executed work on an important problem—I look forward to seeing how this research develops with the suggested improvements.

1. Limited Model Diversity (Critical)
All experiments use a single base model (Qwen-2.5-7B-Instruct). 
Key concerns with a specific case of a model:
Do results hold for other model families (Llama, Mistral, Gemma)?
Does the method work with smaller models (1-3B) where extractor quality may degrade?

2. Short Training Duration Raises Concerns
Only 3 epochs of training. Critical questions:
Does rubric quality degrade with longer training (10+ epochs)?
Is there rubric drift or accumulation of noise over extended training?
Do elicited criteria remain stable or do they become contradictory?
What happens to the total number of criteria—does it grow unboundedly?

3. Computational Cost Not Adequately Addressed
The method requires:
8 additional rollouts per prompt from control policy
8 LLM extractor calls (o3-mini) per prompt
1 deduplication LLM call per prompt
Additional grader calls for the expanded rubric set

Missing Analysis:
Total inference cost comparison with baselines (in dollars or FLOPs)
Wall-clock time increase
Sensitivity to number of pairwise comparisons (why 8?)
Cost-benefit trade-off analysis

4. Circular Dependency in LLM-Based Components
Both extractor (o3-mini) and grader (GPT-4.1-mini) are LLMs with known limitations. Concerns:
Do shared biases between extractor and grader create echo chambers?
Can the system amplify systematic errors over training iterations?
How does extractor quality affect final performance (no ablation study)?

### Soundness
2

### Presentation
3

### Contribution
3
