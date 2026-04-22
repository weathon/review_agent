# Disentangling Length Bias in Preference Learning via Response-Conditioned Modeling

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Reinforcement Learning from Human Feedback (RLHF) has achieved considerable success in aligning large language models (LLMs) by modeling human preferences with a learnable reward model and employing a reinforcement learning algorithm to maximize the reward model's scores. However, these reward models are susceptible to exploitation through various superficial confounding factors, with length bias emerging as a particularly significant concern. Moreover, while the pronounced impact of length bias on preference modeling suggests that LLMs possess an inherent sensitivity to length perception, our preliminary investigations reveal that fine-tuned LLMs consistently struggle to adhere to explicit length instructions. To address these two limitations, we propose a novel framework wherein the reward model explicitly differentiates between human semantic preferences and response length requirements. Specifically, we introduce a $\textbf{R}$esponse-$\textbf{c}$onditioned $\textbf{B}$radley-$\textbf{T}$erry (Rc-BT) model that enhances the model's capability in length bias mitigating and length instruction following, through training on our augmented dataset. Furthermore, we propose the Rc-RM and Rc-DPO algorithm to leverage the Rc-BT model for reward modeling and direct policy optimization (DPO) of LLMs, simultaneously mitigating length bias and promoting adherence to length instructions. Extensive experiments across various models and datasets demonstrate the effectiveness and generalizability of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work tackles length bias in DPO. It first uses four focused experiments to show that (i) length bias genuinely exists, (ii) the evaluation set itself carries length bias, (iii) an implicit preference for longer answers ≠ the ability to follow explicit length instructions, and (iv) explicitly teaching length is easy but harms semantic quality. The experimental designs are insightful. Building on these findings, the authors propose a method that converts “implicit length bias” into an explicit understanding of length and disentangles it from semantic preference. They apply this idea to both reward modeling (RM) and DPO, and demonstrate effectiveness and generalization across multiple base models.

### Strengths
1. Clear motivation and an important problem. The paper targets a widely observed and practically consequential issue in RLHF/DPO—length bias. The authors establish motivation via a “falsify first, then model” storyline, using evidence to demonstrate the problem before proposing a solution.

2. The four small studies form a coherent chain: (existence) → (evaluation bias) → (implicit ≠ explicit) → (explicit hurts semantics). This progression is logically tight and highly readable, making the diagnosis convincing and the need for a new method evident.

3.  The core idea that explicitly modeling and disentangling a spurious format constraint from semantic preference appears portable to other instruction/format biases (e.g., politeness, number of bullet points, templated structure). This suggests a promising research trajectory beyond length alone.

### Weaknesses
1. 3.3 (length-compliance eval) construct validity: The eval set is built from model-rewritten pairs assumed to be semantically equivalent. Any residual semantic or stylistic drift can confound the claim “implicit length bias ≠ explicit length control.” Add a simple length-only heuristic baseline (choose the candidate that satisfies the numeric constraint) as a sanity upper bound, and include human spot checks for equivalence.

2. 3.4 (explicitly teaching length)  might result in overfitting vs. distribution shift: The {x_l, y_w, y_l} format makes length the most salient signal. The quality drop might reflect not only “overfitting to length,” but also a mismatch between synthetic length-instruction training data and the de-biased quality eval. 

3. RC-DPO is essentially in the DPO family. Comparing mainly to ODIN (more of an RM-regularization baseline) is insufficient. Please add strong DPO variants such as SIMPO, CPO, etc, and evaluate on standard leaderboards.

4. Both the de-biased quality set and the length set depend on strong-LLM rewrites, risking style/domain shift and label noise. Strengthen with: (i) a human agreement study on a sampled subset; (ii) cross-source replication (another rewrite model or a small human-curated slice); and (iii) reporting failure cases where “meeting the numeric length” reduces information density (add redundancy/compression-retention metrics).

### Questions
See weakness.

### Soundness
3

### Presentation
2

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
This paper investigates the issue of length bias in RLHF, where reward models tend to favor longer responses rather than true human preferences. To address this, the authors propose a Response-Conditioned Bradley–Terry (Rc-BT) model. Building on this model, they introduce algorithms designed to mitigate length bias. Numerical results across several models and datasets demonstrate that the proposed approach can reduce length bias.

### Strengths
This paper is overall well written and effectively motivates the phenomenon of length bias in RLHF.

### Weaknesses
I agree that length bias is a common phenomenon in RLHF; however, I remain unclear about several points:

- What causes the length bias? Is it due to the training dataset, the scale of the LLMs, or the training procedure (e.g., lack of length regularization)? For instance, does increasing the size of the reward or policy model reduce this effect? From my experience, larger models such as GPT5 seem less likely to have this issue.

- A following weakness I feel is that, the paper lack of a principle way to address the length bias problem as they do not address the problem by fining the cause of the problem but rather simply use a post-hoc method to address the problem. 

The new model lacks novelty, as it appears to be a trick based on the BT model.

Regarding methodology, the notation involving $\pi$ is somehow confusing. What does $\pi_{\theta}(x, y)$ represent, and how does it differ from $\pi_{\theta}(y|x)$? From a practical standpoint, how should the trained policy be used for testing after training?

The experimental section compares only a few baselines, even though several new algorithms have been proposed based on DPO. I recommend including additional recent works:

- https://arxiv.org/abs/2310.03716

- https://arxiv.org/abs/2403.19159

- https://arxiv.org/abs/2407.07880

### Questions
See Weaknesses above.

### Soundness
2

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
This paper introduces a framework (response conditioned Bradely-Terry) to improve the utility of reward models for preference learning by reducing their susceptibility to length bias and ensuring models better follow length instructions. The approach is validated for multiple models and datasets, and the results suggest the approach is effective for improving the robustness to length bias.

### Strengths
- This is an important problem to tackle as length bias continues to be problem, where longer responses tend to receive higher reward regardless of the content.
- The experiments documented in the appendix provide an in-depth analysis of the approach.
- The formulation of the preference pairs is a nice idea. I initially had a concern when reading the paper that optimizing to simply constrain length might cause the model hack by doing something like generate responses in a more compressed form (e.g., Chinese characters offer a more condensed form than English). The formulation of the preference pairs is a nice way to prevent this.

### Weaknesses
- Only two model families are considered.
- The largest model sizes tested are on the smaller size by the standards of today.

### Questions
- Avoid using subjective qualifiers in writing. For example, the reader can decide how “extensive” the experiments are.
- Table 1 caption: explain what, e.g., accuracy is computed on — is it the accuracy with which the actual preferred response receives a higher reward from the reward model?
- I was surprised to not see the seminal paper Deep Reinforcement Learning from Human Preferences by Christiano et al. cited. Especially when introducing RLHF.
- In the random evaluation set, this is where responses are swapped for one another arbitrarily.  Is there a case for a truly random set where the token sequences are random and so a longer response might be pure gibberish?
- For Figure 1(b) why is there so much variation in response length when the prompt is always just “empty prompt”?
- Throughout the paper the incorrect opening quotes are used.
- Line 193: “demonstrates the length bias” > “demonstrates length bias”
- Line 195: the subscript “l” is being reused. It is used here for length, and previously for loss (in the context of a preferred / dispreferred preference pair).
- Line 203: What is stated as being a hypothesis is not actually written as a hypothesis.
- Line 257: “preferred than” > “preferred over”.
- Line 257: “to modeling human preferences” > “to model human preferences”
- Line 259: “assuming a underlining true” > “assuming an underlying true”.

### Soundness
3

### Presentation
3

### Contribution
3
