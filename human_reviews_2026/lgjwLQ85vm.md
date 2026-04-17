# C$^2$GSPG: Confidence-calibrated Group Sequence Policy Gradient towards Self-aware Reasoning

- Decision: Reject
- Scores: 8, 6, 4, 2

## Abstract
Reinforcement Learning (RL) methods, exemplified by Group Relative Policy Optimization (GRPO) and its variants, play a central role in developing reasoning models.
However, these methods often suffer from a critical overconfidence issue, which prevents them from achieving self-aware reasoning models. 
In this study, we propose a simple yet effective confidence-calibration group sequence policy gradient method, called C$^2$GSPG, which simultaneously enhances reasoning performance while suppressing overconfidence. 
In principle, we propose a Group Sequence Policy Gradient (GSPG) framework for learning reasoning models, which eliminates the token-level bias commonly appearing in GRPO and its variants.
In this framework, we define the model confidence for each reasoning problem using the normalized sequence-level probability, and then apply a cross-entropy regularizer to calibrate the model confidence to the sequence's reward. 
We demonstrate that the confidence calibration regularizer and GSPG are collaborative for binary rewards, as their objectives always share the same gradient direction. 
For non-binary rewards, we apply nonlinear reward normalization and adaptive regularizer clipping, mitigating the potential conflict between the two objectives. 
Applying C$^2$GSPG to post-train large language models in logical and mathematical reasoning tasks, we show its superiority over state-of-the-art methods in both reasoning accuracy and confidence calibration.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents C$^2$GSPG (Confidence-Calibrated Group Sequence Policy Gradient), a reinforcement learning framework designed to alleviate the overconfidence problem observed in GRPO-style reasoning models.
By introducing a sequence-level confidence measure and a binary cross-entropy regularizer that aligns model confidence with reward, the method encourages self-aware reasoning—increasing confidence for correct generations and reducing it for incorrect ones.
The paper also proposes sigmoid-based reward normalization and adaptive clipping to handle non-binary reward settings.
Extensive experiments on logical and mathematical reasoning benchmarks show consistent improvements in both accuracy and calibration metrics (ECE, Brier Score).

### Strengths
1. Clear and well-written: The paper is easy to follow with solid organization and a strong motivation.

2. Reasonable and well-motivated algorithmic design: Each component (confidence-modulated advantage, BCE regularizer, adaptive clipping) is conceptually sound and technically justified.

3. The work tackles an important problem of reasoning LLMs — self-awareness and confidence calibration, which is a key limitation of current LLMs.

4. Empirical results show consistent and significant performance improvements, not only in reasoning accuracy but also in calibration robustness across multiple datasets.

5. The method effectively generalizes from binary to non-binary reward settings, demonstrating broad applicability.

6. Calibration metrics (Brier Score, ECE) are appropriately chosen and clearly analyzed, showing that confidence calibration contributes directly to better reasoning quality.

### Weaknesses
1. The paper lacks fine-grained ablation studies to isolate the contribution of each component (e.g., GSPG, BCE, adaptive clipping).

2. Experiments are limited to small-sized backbones (1.5B–3B); validation on larger LLMs would strengthen the claim of robustness and scalability.

3. It remains unclear whether the calibrated confidence translates into improved inference-time self-awareness (e.g., via self-verification or test-time scaling strategies).

4. The method involves several heuristic hyperparameters (α, β, clipping threshold) whose tuning process or sensitivity is not well discussed.

### Questions
1. Could the improved sequence confidence be used to enhance test-time scaling (e.g., weighted majority voting or Best-of-N)?

2. Can the authors provide component-level ablations or at least a discussion on expected effects when removing each module (e.g., normalization, clipping, or BCE)?

3. The qualitative examples mostly show correct generations—could you also include failure cases and their confidence scores to illustrate calibration behavior for incorrect outputs?

4. Have you observed whether confidence calibration affects generation dynamics, such as reasoning length or exploration behavior?

### Soundness
4

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
This paper introduces Confidence-Calibrated Group Sequence Policy Gradient, a method designed to address the overconfidence issue in LLMs for reasoning tasks. By incorporating a confidence calibration regularizer into the GRPO framework, C2GSPG aligns model confidence with actual correctness, reducing overconfidence in incorrect answers. The method introduces a Brier Score regularizer to ensure confidence estimates are reliable, improving both reasoning accuracy and calibration. Experimental results demonstrate that C^2GSPG outperforms sota methods in accuracy and confidence calibration, particularly on complex tasks with higher difficulty levels.

### Strengths
1. The paper introduces C2GSPG, a novel method that uses confidence calibration to mitigate the overconfidence problem in LLMs

2.It shows a large improvement in logical reasoning tasks, outperforming state-of-the-art methods like AR-Lopti and GRPO, demonstrating its effectiveness in challenging reasoning scenarios.

3. The method not only enhances accuracy but also significantly improves confidence calibration.

### Weaknesses
1. I am confused about the difference between aligning confidence (probability) with reward versus directly aligning output with reward. The paper also doesn't explain the practical consequences of the overconfidence problem. I'm not sure how exactly does overconfidence affect performance in real-world tasks?

2. C^2GSPG performs well on the K&K dataset, but the improvements in mathematical reasoning are not significant, especially in Brier Score. More substantial results in this area would strengthen the method’s broader applicability.

3. The innovation of adding Brier Score to the GSPO objective seems limited, as it is already a widely used method for confidence calibration.

### Questions
1. Does C^2GSPG scale well on larger models?

2. Can you provide a concrete example of a scenario where the overconfidence problem leads to poor model performance?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose $C^{2}GSPG$, a novel confidence-calibrated group sequence policy gradient method. The core idea is twofold: first, it proposes a Group Sequence Policy Gradient (GSPG) framework that operates at the sequence level, aiming to eliminate the token-level biases found in GRPO. Second, it introduces a binary cross-entropy (BCE) regularizer to explicitly align the model's confidence (defined as the normalized sequence-level probability) with the actual reward of the sequence.

### Strengths
1. The method demonstrates good performance outperforming baselines.
2. The model can achieve some calibration on confidence and accuracy. This might be useful to improve interpretability.

### Weaknesses
1. I don't get it why we need to care about the model being overconfident if accuracy is good. An extreme example would be that a model that is 100% accurate and 100% overconfident is a perfect model. The paper fails to provide a compelling case for why we should sacrifice any accuracy for better calibration, which is a trade-off demonstrated by the $\beta$ hyperparameter in Table 5. The objective seems to be optimizing the confidence calibratio at the risk of distracting from accuracy.
2. The paper's GSPG framework is presented as a key contribution for eliminating token-level bias. However, this is very similar in principle to other recent sequence-level methods like GSPO. Both GSPG and GSPO compute a sequence-level advantage and apply a gradient based on the entire sequence probability. The introduction of sigmoid reward normalization and, in particular, the adaptive regularizer clipping, which simply turns off the regularizer if its gradient conflicts with the policy gradient, seems like an engineering patch to force the method to work rather than a principled solution. This contribution appears to be marginal at best. 
3. The extension to non-binary rewards feels heuristic and ad-hoc. It is somewhat problematic since the authors do observe gradient conflict in ther empirical studies.

Overall, I think the paper has many flaws despite its good performace.

### Questions
See the above weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a new RL method, Confidence Calibrated Group Sequence Policy Gradient (C2SPG), that aims to both improve accuracy and calibrated model confidence on reasoning tasks. The authors define confidence as the average token-level likelihood of the output under the language model. The proposed RL algorithm introduces several changes to the optimization target from prior work, such as normalizing the advantage estimate by 1 - confidence of the old model (used to generate trajectories), multiplying the advantage estimate by the log of the confidence, and introducing a "binary cross entropy regularizer" that is the log-likelihood that the output is correct when treating the average token likelihood as the predicted probability. The authors analyze the gradients of the proposed algorithm and compare to prior methods. They also consider extensions to non-binary settings.

The authors perform experiments on a dataset with logic puzzles (K&K) and across multiple mathematical reasoning datasets. They report dramatic improvements in accuracy (38% GRPO, 56% GSPO --> 79% C2SPG) and calibration (0.5 GRPO, 0.3 GSPO --> 0.13 ECE) on the K&K dataset, and mixed performance on mathematical reasoning tasks (depending on hyperparameters of the C2SPG objective, it achieves up to 1% increase in average accuracy with worse calibration or ~0.03 improvement in ECE to 0.46 with worse accuracy).


The authors claim that C2SPG "significantly improves both reasoning accuracy and confidence calibration, outperforming existing state-of-the-art methods".

### Strengths
The authors perform an interesting comparison of RL methods and use this to motivate their work. Experiments are clearly defined and presented.

### Weaknesses
1. The current experimental results do not seem to support the claim that C2SPG significantly improves calibration and reasoning. On the math benchmarks, C2SPG either has a small improvement in accuracy while achieving similar calibration or improves calibration while achieving similar accuracy. These results don't tell a clear story of improvement or show a single algorithmic setting that improves both of the target metrics. Additionally, small scale of experiments (100 steps, 3B models) and lack of large improvement on math benchmarks makes it unclear if the improvements that are seen here will generalize to other experimental settings or scale up.

2. The results from the two experiments are not consistent: the accuracy improvement on K&K relative to SOTA is dramatic (38% GRPO, 56% GSPO --> 79% C2SPG), while the accuracy improvement on math is marginal (~1-2% on best hyper parameter setting). Why is the improvement so big on the first experiment and not on the second? As is, it is not clear how informative the K&K setting is, given that it is synthetic and behaves so differently from the math benchmarks.

3. It's not clear to me that the average token-level likelihood of a sequence should correspond to model confidence, or that it is desirable to optimize this quantity for calibration. There is a mismatch between this quantity and the sequence likelihood (which depends on length), so a sequence that the model is more likely to generate may be defined to have lower confidence. It's not clear what the implications of this mismatch are and why it is desirable. I note that this settings is different from the GSPO setting discussed in the paper: GSPO length normalizes sequence likelihoods when doing importance weighting, which means that sequences being compared will always be the same length. Stronger justification of why this definition of confidence is the right one for developing calibrated models, and why this is a robust choice, would be helpful.

4. The method proposes a number of changes to previous RL methods (see summary) without clearly motivating all of them and ablating these choices. Some other simple approaches, such as treating the confidence as the predicted probability and simply optimizing the log-likelihood that the output is correct given this probability, are not compared against. It is a bit difficult to understand what choices could be yielding improvements and why some choices are made without this kind of discussion and analysis.

### Questions
Please see the weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2
