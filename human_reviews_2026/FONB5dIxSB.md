# Humanline: Online Alignment as Perceptual Loss

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Online alignment (e.g., GRPO) is generally more performant than offline alignment (e.g., DPO)---but why? Drawing on prospect theory from behavioral economics, we propose a human-centric explanation. We prove that online on-policy sampling better approximates the human-perceived distribution of what the model can produce, and PPO/GRPO-style clipping---originally introduced to just stabilize training---recovers a perceptual bias in how humans perceive probability. In this sense, PPO/GRPO act as perceptual losses already. Our theory further suggests that the online/offline dichotomy is itself incidental to maximizing human utility, since we can achieve the same effect by selectively training on any data in a manner that mimics human perception, rather than restricting ourselves to online on-policy data. Doing so would allow us to post-train more quickly, cheaply, and flexibly without sacrificing performance. To this end, we propose a design pattern that explicitly incorporates perceptual distortions of probability into objectives like DPO/KTO/GRPO, creating $\textit{humanline variants}$ of them. Surprisingly, we find that these humanline variants, even when trained with offline off-policy data, can match the performance of their online counterparts on both verifiable and unverifiable tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper studied the problem of reasons and explanations of online alignment better than offline alignment. Through the lens of the prospect theory framework, the authors provided a human-centric explanation. By proving PPO/GRPO in online on-policy clipping recovers a perceptual bias, the authors showed that online alignment acts as a perceptual loss. With these findings, the post-training is more efficient without restriction on the data source and without the cost of performance. Finally, they also test their results in an unverifiable reward setting to follow open-ended instructions and a verifiable reward setting for the math reasoning task.

### Strengths
1. The perspective to explain online alignment is interesting by adopting some methods from behavioral economics.
2. The writing of the paper is clean. The authors clearly state the problem, methods, and their results.
3. Most of the related work is cited and discussed. And the authors provided additional related work in Appendix A.
4. The paper provided both theoretical results and empirical results.

### Weaknesses
1. The assumption is strong, e.g., Assumption 4.3.
2. Some statement is not rigorous. For example, "If the success of PPO/GRPO can be ascribed to them being perceptual losses".

### Questions
1. How is Assumption 4.3 in practice, especially "The cumulative probability of outcomes with higher absolute surprisal than $z_i$ is negligible"? Is this assumption also used in previous work? How did your experimental setting satisfy this assumption?
2. What is the intuition of Definition 4.5 to propose humanline sampling?
3. The term perception plays a central role in your framing, yet its precise mathematical definition is unclear. Could you provide a more formal explanation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a human-centric explanation of why online, on-policy alignment (e.g., GRPO/PPO) often outperforms offline, off-policy alignment (e.g., DPO/KTO). The key idea is to model human utility via prospect theory and include probability weighting (capacity) in addition to the subjective value. From this, the authors argue that sampling from the current policy better matches the human-perceived outcome distribution, giving intuition on why online > offline.

On the practical side, they introduce a two-part humanline design pattern applicable to DPO/KTO/GRPO: Humanline syncing and Humanline clipping. Empirically, humanline variants trained on offline data close the observed performance gap with online variants on instruction-following and allow 64× less frequent sampling in math-reasoning. Ablations suggest syncing provides the bulk of the gain; upstream clipping adds a smaller but positive effect.

### Strengths
**S1. Originality:** Extends alignment as prospect-theoretic optimization by explicitly modeling probability weighting (not just value).

**S2. Practicality:** The humanline recipe (syncing and upstream asymmetric clipping) is simple to implement and integrate into existing algorithms.

**S3. Quality/robustness:** Ablations isolate the roles of syncing vs clipping; results hold across tasks and model sizes.

**S4. Significance:** If adopted, the approach can lower alignment cost and increase offline data usage without sacrificing quality.

### Weaknesses
**W1. Dependence on offline data.**
The method depends on the choice of offline dataset (e.g., offline data generated by Gemma-9B vs Llama-8B). The paper attributes this to a possible violation of Assumption 4.1 (support overlap and bounded likelihood ratio), but never verifies it empirically. It would strengthen the work to measure these assumptions directly. For instance, by reporting a coverage or divergence metric and examining how they correlate with performance across datasets.

**W2. Gains stem from syncing, not clipping.**

The ablation study shows that most performance improvements come from syncing the reference model ($\pi_{\text{ref}}$). In contrast, the proposed clipping mechanism, a central contribution of the paper, performs comparably to the offline baseline when used alone. This suggests that clipping contributes little to the observed gains while adding two extra hyperparameters and complexity.

**W3. Unfair comparison to trust-region baselines.**

The comparison to trust-region methods is not hyperparameter-fair. The baseline’s update frequency $k_{tr}=1024$ was taken directly from the original paper, where it was tuned for a different task. In contrast, the proposed method’s equivalent parameter $k$ is shown to be highly sensitive and was tuned for this work. The trust-region baseline could likely perform better if $k_{tr}$ were tuned under the same conditions, so the reported advantage may be overstated.

**W4. Weak theoretical justification.**
Proposition 3.4 only suggests that a small KL divergence is beneficial; it does not theoretically prove that online > offline.

### Questions
**Q1.** Could the authors provide quantitative results (e.g., support overlap or KL divergence) to verify when Assumption 4.1 holds and how it relates to performance differences when using different datasets?

**Q2.** See other questions in the weaknesses section.

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
3

### Summary
In this paper, the authors provide a theoretical study of the reasons why online alignment is better than offline alignment. In particular, this work shows that online on-policy algorithms are good at approximating human perception. Moreover, the offline and online training have been shown they be identical in maximizing human utility. Motivated by these investigations, the authors propose a new design for the training algorithm, which is evaluated on experiments.

### Strengths
1. This work studies an important domain - trying to explain why online algorithms are usually better than offline algorithms.
2. The presentation of this paper is easy to follow. 
3. The theoretical and empirical investigations are both provided.

### Weaknesses
1. From the experimental results, the humanline indeed can help with offline alignments following the lack of human perceptions. However, it is not clear why the online alignments (the authors have shown that those online alignments have demonstrated the capacity for modeling human perceptions) can also be greatly improved with humanline.

2. The modeling of the value function as well as the utility function is hypothetical, which means that if they reflect the true system is not clear; otherwise, the authors need to verify them.

3. Based on the definitions under prospect theory, the authors provide a new analysis for answering the question about the comparison between the offline and online alignments. However, from another perspective, the offline off-policy is indeed a more challenging process compared to online on-policy, which is more computational cost but involves more exploration opportunities. In this case, how to validate the benefits of online alignments is mostly from the explanation under the prospect theory? 

4. As the performance difference between offline and online alignments is also witnessed in large model alignments. I am wondering how to verify that the phenomenon observed for small models can also be extended to large models.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper talks about why online on-policy methods tend to outperform offline off-policy methods, and offers a human-centric answer grounded in prospect theory. Specifically, the authors claim that (1) online on-policy sampling better approximates the human-perceived distribution of what the model can produce, and (2) PPO/GRPO recovers a perceptual bias in how humans perceive probability. They then propose a practical humanline design pattern applicable to DPO/KTO/GRPO: (1) humanline syncing (sync the reference with the previous policy every k steps) and (2) humanline clipping (asymmetric clipping of token log-ratio). Empirically, the authors show that offline + humanline matches online performance on instruction following and allows 64-times less frequent sampling without degradation on math reasoning, which gives large speedups versus standard online training.

### Strengths
## Originality
1. The paper reframes the online/offline gap through prospect theory from behavioral economics, which is quite a unique point of view. Though it is not the first paper to extend prospect theory to alignment, it considers the probability weighting and the inverted-S curve shown in the paper intuitively explains why online sampling is closer to the human-perceived distribution. 
2. The paper theoretically links GRPO/PPO clipping to perceptual weighting via rejection sampling and limit arguments. This gives a fresh interpretation of a widely used heuristic.

## Quality
1. All claims are backed with theoretical proof. 
2. Empirically, the paper gives a practical yet minimal recipe to highlight the change. Such a plug-and-run recipe can be useful for future researchers. 

## Clarity
The presentation is clear and easy to follow. There is a good example throughout the prospect theory section making it easy to understand. 

## Significance
1. If robust, the result meaningfully relaxes the need for fully online training. This would significantly improves the overall training walltime while maintains the same level of performance, greatly speed up the model iterations.

### Weaknesses
1. Lack of implications of assumptions. The authors did not cover enough discussion on assumptions 4.1-4.3, e.g., what do those assumptions imply in the language of LLM alignment? Especially for 4.1, same support and finite likelihood ratio seem too restrict in LLM as the action space is enormous; for 4.3, does LLM usually gives you light tail? Any data/study support?
2. Offline data quality dependency. The paper also shows that not all data could match the online performance, but stops discussing more on the implication and in practice how this could be resolve or any efficient ideas to detect an offline dataset with "good quality". Within good/bad data, do they share some common attributes?
3. Data–method confound. Given the offline data quality dependency, can naive offline methods without humanline achieve comparable performance by a carefully selected set of data? If so, then how can we verify the efficacy of humanline? Some carefully designed abalation tests on this would be good.

### Questions
See weakness. Happy to raise the score if authors could help clarify on the above issues.

### Soundness
3

### Presentation
3

### Contribution
3
