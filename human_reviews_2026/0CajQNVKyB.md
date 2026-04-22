# Hybrid Reinforcement: when reward is sparse, better to be dense

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
Post-training for reasoning in large language models has increasingly relied on verifiable rewards: deterministic checkers that provide $0$–$1$ correctness signals. While reliable, such binary feedback is brittle—many tasks admit partially correct or alternative answers that verifiers under-credit, and the resulting all-or-nothing supervision limits learning. Reward models  offer richer, continuous feedback, which can serve as a complementary supervisory signal to verifiers. We introduce HERO (Hybrid Ensemble Reward Optimization), a reinforcement learning framework that integrates sparse verifier signals with dense reward model scores in a structured way. HERO employs stratified normalization to bound reward-model scores within verifier-defined groups, preserving correctness while refining quality distinctions, and variance-aware weighting to emphasize challenging prompts where dense signals matter most. Across diverse mathematical reasoning benchmarks, HERO consistently outperforms reward model-only and verifier-only baselines, with strong gains on both verifiable and hard-to-verify tasks. Our results show that hybrid reward design retains the stability of verifiers while leveraging the nuance of reward models to advance reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces HERO (Hybrid Ensemble Reward Optimization)- a framework that integrates dense signals from reward models with binary-valued feedback from rule-based verifiers. The paper systematically reports the merits of each individual approach while highlighting its limitations; they further caution against naively combining the two approach. The proposed solution uses a `stratified' strategy to leverage expressiveness of reward models within structures imposed by the verifier. Experimental evaluations across a wide range of regimes in math reasoning tasks demonstrate that HERO performs significantly better than reward model only or verifier only baselines.

### Strengths
(+) HERO’s use of stratified normalization together with a difficulty-aware weighting scheme simultaneously enables overcoming limitations of binary-valued feedback signals and misaligned scores assigned to incorrect responses. 

(+) The proposed mechanism has been evaluated on math reasoning benchmarks with a wide range of objectives ranging from exactly correct to partially correct solutions while additionally taking into consideration the solution format. 

(+) Strong results in hard-to-verify tasks compared to reward model-only and verifier-only schemes. Extensive ablations demonstrate the importance of each component of HERO in achieving this improved performance. 

(+) Limitations of rule-based verifiers and reward modeling are demonstrated separately across multiple verifiers and a reward model for tasks where correctness can be difficult to verify, justifying the `hybrid’ approach adopted in this paper. 

(+) The authors further demonstrate (experiments reported in the Appendix) that naively combining reward signals with rule-based verification might not be sufficient to result in improved performance. 

(+) Overall, this paper is extremely well-written, and the authors systematically lay out the merits of each component approach, highlight their limitations, and point out how the proposed hybrid approach combines the merits while overcoming shortcomings of the individual approaches. Experiment results provide strong backing to the hypotheses presented.

### Weaknesses
(-) The `ensemble' characteristic of HERO is not immediately apparent from the writing of the paper. Additional clarification on this aspect will be helpful. 

(-) The explanation for the design principle guiding HERO (reward signals as guides to augment rule-based methods) mentioned in Lines 192-196 of Sec. 3.2 might be helpful in understanding the shape of the hybrid reinforcement graph relative to the other two graphs. 

(-) The discussion on future directions in Lines 485-486 can be improved. Perhaps some of the material from Appendix C can be moved to this part of the paper. 

(-) Only two sets of values for $\alpha, \beta$ in Eqn. (3) have been considered (i.e., $\alpha = \beta = 0.05$, $\alpha = \beta = 0.1$). It might be interesting to see the specific role of each parameter on the performance of HERO, and the possible intuition for such performance. 

(-) Line 747 in Appendix: ables —> Tables.

### Questions
Please see Weaknesses, above.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposed HERO. The core idea is to combine sparse but reliable verifier-based binary rewards with dense but noisy reward-model scores. HERO proposes Stratified normalization, which anchors RM scores within verifier-defined correctness groups to preserve semantic correctness, and Variance-aware reweighting, which emphasizes prompts with higher score variance. Experiments on multiple math reasoning benchmarks (e.g., MATH500, AMC, Olympiad, HardVerify-Math, and TextBookReasoning) show that HERO outperforms both verifier-only and RM-only baselines across verifiable, hard-to-verify, and mixed settings.

### Strengths
1. Clear technical insight: preserves correctness while enriching reward density.

2. Comprehensive experiments on multiple model (Qwen3-4B, OctoThinker-8B) and datasets, with detailed ablations and parameter sensitivity analyses.

3. Writing and presentation are clear.

### Weaknesses
1. The paper provides only one limited RL training curve (Appendix Fig. 5), where the hybrid reward benefit is unclear and performance even collapses at later steps. This is confused. The caption is also wrong for Figure 5. More RL training curves results are needed to confirm the consistent HERO superiority over baselines instead of only showing performance on some/final checkpoint.

2. In Table 2, the gains of HERO vary dramatically across training regimes. When trained on verifiable samples, HERO outperforms baselines by over 10 points, yet when trained on hard-to-verify samples, the average improvement shrinks to <3 points. In the mixed regime, the improvement flips again: small on verifiable tasks (~4 points) but large on hard-to-verify tasks (~10 points).
These results raise questions about what drives the effectiveness of the hybrid reward. Without a clearer explanation, it is difficult to determine whether HERO’s benefit stems from the hybrid design itself.

3. Potential unreliability in Olympiad benchmark evaluation. Many Olympiad problems are themselves hard to verify with math_verify, which may distort scores. Rule-based and llm-based ablations should be done on the Olympiad problems to see whether the evaluation scores are accurate.

### Questions
1. Could you justify Figure 5 and provide additional training-step vs. benchmark performance tables/curves to show that HERO consistently outperforms baselines, rather than only at a specific checkpoint?

2. Could you provide more explanations and justifications about why gains of HERO vary dramatically across different training regimes in Table 2? How does the ratio of verifiable/non-verifiable training samples affect HERO gains over baselines?

3. Could you report results using both rule-based and LLM-based verification for OlympiadBench to confirm accurate and consistent evaluation results?

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
This paper proposes a hybrid reward to integrate reward model and binary verifier scores. The hybrid reward is shown to be effective on multiple models across multiple tasks including verifiable and hard-to-verify ones.

### Strengths
- The problem is well motivated. 
- The proposed method is intuitive and simple to implement.
- Extensive experiments to demonstrate the effectiveness of HERO.
- The paper is well-written and easy to follow.

### Weaknesses
- The min-max normalized reward depends on the result of RM and r_rule. How does HERO handle reward drift/misalignment inherited from RM and r_rule?
- How effective is HERO for non-verifiable tasks such as instruction following? Will HERO harm model performance on these tasks?
- The choice of parameters $\alpha$ and $\beta$ are highly depend on tasks. It would be great to provide some guidelines on choosing these parameters for task agnostic models.

### Questions
Please see weaknesses

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
3

### Summary
The paper proposes HERO, a reinforcement learning framework designed to enhance reasoning capabilities in large language models (LLMs) by integrating sparse, binary verifiable rewards with dense, continuous scores from reward models. The core innovation lies in addressing the brittleness of binary verifiers and the misalignment of reward models through two mechanisms: stratified normalization, which bounds reward model scores within verifier-defined correctness groups, and variance-aware weighting, which emphasizes challenging prompts with high score variance. Built on GRPO, HERO aims to provide stable, informative supervision for mathematical reasoning tasks. Empirical results on verifiable benchmarks and hard-to-verify ones demonstrate that HERO outperforms baselines like RM-only RL and verifier-only methods.

### Strengths
- The paper is clearly written and easy to follow.
- The stratified normalization and variance-aware weighting are intuitive and simple to implement.
- Comprehensive experiments across multiple reasoning benchmarks and architectures demonstrates the effectiveness of the proposed method.

### Weaknesses
- The “hybrid reward” idea that combines verifier and reward-model signals is conceptually intuitive and has already been explored in prior work. 
- Moreover, the conclusions that (1) function-based rules exhibit high precision but low recall, and (2) reward models have lower precision, are not novel and have been thoroughly demonstrated in prior studies (e.g., arXiv:2505.22203).
- The evaluation scope remains narrow: the method’s generalization to non-mathematical, open-ended, or multimodal reasoning tasks is not shown.
- While ablations are mentioned, the paper lacks detailed quantitative results for key hyperparameters such as α/β ranges or variance-weighting factors, limiting the interpretability and reproducibility of the method’s stability.

### Questions
- How would HERO perform on open-ended reasoning?
- How sensitive is HERO to the choice of α and β?

### Soundness
2

### Presentation
3

### Contribution
2
