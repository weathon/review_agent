# What are Key Factors for Updates in RL for LLM Reasoning?

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Reinforcement Learning from Verifiable Rewards (RLVR) has emerged as a promising framework for enhancing the reasoning ability of large language models. However, much of the existing work is guided by heuristic intuition, leading to divergent algorithmic choices, even contradictory ones that nevertheless report empirical gains. To better understand this phenomenon, we conduct a theoretical analysis of RLVR updates. Our study reveals that differences in off-policy degree, determined by the number of gradient steps per rollout, substantially affect the distribution of importance sampling ratios and their clipping behavior, thereby altering which tokens dominate the update. Building on this insight, we characterize gradient expectation as the central quantity governing update dynamics and analyze the roles of token probability, advantage, and importance sampling ratio. Motivated by these findings, we propose Adaptive Clip Policy Optimization (ACPO), which adjusts clipping boundaries across token groups according to the empirical variance of their importance sampling ratios. Experiments on models of varying scales (3B, and 7B) and diverse reasoning benchmarks—including mathematical problem solving, tabular question answering, and logic puzzles—show that ACPO outperforms strong baselines such as DAPO and CISPO. These results demonstrate that principled, analysis-driven approaches yield more robust and effective RLVR methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the underlying dynamics of Reinforcement Learning from Verifiable Rewards (RLVR) for large language model (LLM) reasoning. The authors identify inconsistencies in recent RLVR algorithms—particularly conflicting findings about whether to prioritize high-entropy or high-probability tokens—and attribute these contradictions to differences in off-policy degree (the number of gradient updates per rollout).

Through theoretical analysis, the paper derives how the importance sampling (IS) ratio variance depends on token probability and off-policy degree, leading to a gradient dominance reversal phenomenon. It shows that low-probability tokens dominate updates in near on-policy settings, while high-probability tokens dominate off-policy ones. Based on this insight, the authors propose Adaptive Clip Policy Optimization (ACPO), which adaptively adjusts clipping bounds based on the empirical variance of IS ratios. Experiments across multiple reasoning benchmarks and model scales (3B–7B) show ACPO’s consistent improvements over baselines like DAPO and CISPO.

### Strengths
* The paper provides a unified framework that reconciles empirical contradictions in prior RLVR research via gradient expectation analysis.

* The proposed method is well motivated and reasonable.

### Weaknesses
* The empirical performance improvement seems marginal.

* The per-batch grouping and variance estimation add implementation complexity; runtime or scaling cost is not discussed.

### Questions
* From Figure 3, it seems that when going beyond 16 steps off-policy, the CISPO completely failed while GRPO maintains stability. Can you provide some insights on this? Would it be misconfigured hyper-parameters, implementation details or precision mismatch issues (please see this relevant paper: https://arxiv.org/pdf/2510.26788)?

* How sensitive is ACPO’s performance to the number of token bins and scaling factor α?

* How does adaptive clipping impact exploration behavior or entropy during training?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper claims it has a superior RLVR algorithm, ACPO, to DAPO and CISPO. ACPO is based on PPO, but it does not have a single clipping threshold. Instead, it categories the tokens into five bins based on their probability: very low probability, low probability, medium, high, and very high probability. Next, it measures the standard deviation between the importance sampling ratios inside each bin and the clips the ratios outside the (- one std, one std). It argues that this is a more systematic than only controlling high entropy tokens or low entropy tokens.

### Strengths
I view the strengths as the evidences that the paper provides for its conclusion: the superiority of ACPO. 

They show slightly better performance over DAPO and CISPO. Additionally, the ablations show that ACPO is slightly better than just using the max clipping value ACPO uses in DAPO.

One strength of the paper is theoretical analysis of the gradient showing where the discrepancy of the previous work is stemming from. I really like these parts (although, I am still not convinced why for example over-contribution of tokens should be binned based on IS)

### Weaknesses
I view the weaknesses as if the presented evidence supports the conclusion that ACPO is superior. 

One thing I don't understand is: is ACPO altered on top of DAPO? I mean, DAPO has dynamic sampling until all queries have non-zero advantage. Does ACPO also have that? The main problem is: I don't know how controlled the study and alterations are: how did they do the hyperparameter search for DAPO, CISPO and ACPO? The results for ACPO are not drastically better than the other two, which makes one wonder if the gains are not signal, but noise. I am making no assumption here. I think it is just very important to provide this detail. I think the paper lacks experimental details heavily.


While the paper first starts with highlighting the discrepancy between emphasizing higher entropy tokens and lower entropy tokens, it moves on to comparison between DAPO and CISPO. However, DAPO and CISPO are not the papers which are highlighted as having the discrepancy. I think a very natural path for the paper was to show the baselines of emphasizing low entropy and high entropy tokens. The paper lacks these two baselines.

### Questions
Thanks for the paper. I have a few questions.

1-Can you add the baseline of emphasizing high entropy tokens vs. low entropy tokens to experiments? 
2-If the conclusion of the paper is ACPO provides better RL training, what are the supporting claims? while you analyzes the gradient, I don't think I agree that any dominance of low probability tokens should be hindered, or dominance of high probability tokens should be stopped. Are you reasoning that it is the variance of the Importance sampling ratio that is causing instability and should be clipped?
I don't get the main conclusion of the paper. Can you state it in a sentence that if ACPO is superior, what is your claimed reasoning behind why?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This manuscript focuses on LLM RL post-training and proposes a novel objective called ACPO with dynamic clipping. Specifically, the authors analyze the different behaviors of low- and high-probability tokens under varying degrees of off-policyness. Based on this analysis, ACPO introduces a dynamically adjusted clipping bound determined by the variance of the importance ratio. Empirical studies on Qwen2.5-3B and 7B show that the proposed ACPO achieves better performance than DAPO and CISPO.

### Strengths
1. The paper is well written and clearly organized.

2. Theoretical analysis is provided to support the rationale of the proposed method.

3. Extensive experiments on LLMs of different scales are conducted, and the results demonstrate the superiority of ACPO.

### Weaknesses
Both the theoretical and empirical parts contain several weaknesses and lack sufficient rigor, as detailed below.  

### Theoretical Part
1.   The analysis is conducted on the gradient with respect to the logits $z_k$ rather than the model parameters $\theta$. The gradient with respect to $\theta$ would more accurately reflect the magnitude of model updates.  

2. The assumption that the importance sampling ratio $rpo$ and the advantage $A$ follow normal distributions is questionable and lacks empirical justification. The ratio $\rho=\frac{\pi_\text{new}}{\pi_\text{old}}$​ involves a division operation, making it unlikely to be normally distributed. Similarly, in many LLM RL algorithms, advantages are computed as $A=r−\text{mean}(r)$, which typically deviates from normality, especially under imbalanced positive and negative responses.

3. In Lemma 1, the use of the approximation symbol “≈” is not rigorous, as the approximation error is not quantified.

4. In Line 227, the definitions of low/high-probability tokens and the corresponding $G_L$ and $G_H$ should be clearly stated.

5. The statement labeled “Corollary 1” would be better presented as a Remark rather than a formal corollary.

6. It remains unclear why $\kappa$ is related to the level of off-policyness. A more rigorous explanation or derivation is needed.

7. In Line 243, the authors claim that low-probability tokens provide the dominant gradient signal, yet no theoretical or empirical evidence is given. A more natural metric to quantify off-policyness could be the ratio between bsz and mini_bsz, which directly indicates how off-policy the training data are.

### Experimental Part
1. The experimental setup lacks key implementation details, such as (1) how $\sigma_\rho$ is computed, and (2) how $\epsilon_b$ is obtained

2. The proposed ACPO introduces an additional hyperparameter $\alpha$ . It is unclear whether $\alpha$ remains robust across different LLM scales and datasets.

3. The performance improvement of ACPO over the base DAPO is marginal, particularly on challenging benchmarks such as AMC23 and Minerva. Moreover, results on widely used hard benchmarks like AIME24 and AIME25 are missing, which limits the empirical credibility of the method.

### Questions
Please see Weaknesses.

### Soundness
1

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
The paper studies why heuristic RLVR methods for LLM reasoning often results in contradictory results. Authors identify the degree of off-policy learning as the key factor and show that token dominance in gradient updates reverses depending on the number of updates per rollout. They propose ACPO to dynamically adjust clipping bounds based on the variance of importance-sampling ratios. Experiments are done on Qwen2.5 models (3B/7B) across math and tabular reasoning tasks to demonstrate that ACPO outperforms DAPO and CISPO.

### Strengths
- Proposed method of adaptive clipping range sounds reasonable and simple in application. 
- The paper is well-structured and clearly written, with logical progression from motivation to experiments.
- Experiments showed algorithm effectiveness over 3B and 7B models.

### Weaknesses
- Training curves comparison is an important result to look at. But Figure 3 is the only such figure presented, which is done on a rather simple task countdown. As similar in DAPO and GSPO, it's better to show the reward curves, entropy curves, and the performance curves on harder math benchmark such as AIME24.
    - What are the results on AIME24 and AIME25 (avg@16 or avg@32 since the datasets are very small)?
- If I understand correctly, the proposed method only makes change to the clipping range by making it adaptive. While DAPO has clip higher, it also introduced other useful tricks such as dynamic sampling that is very effective in practice. How would the authors explain the better performance compared to DAPO with just one adjustment on clipping range? Since these are not conflicting elements, what happens when dynamic clipping is applied to DAPO?
    - The current performance improvement compared to DAPO also seems very on-par.

Overall I believe more empirical evidences are needed to validate the effectiveness of the proposed algorithm.

### Questions
see in weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
