# LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation

- Decision: Accept (Poster)
- Scores: 8, 8, 8, 4

## Abstract
Looped Transformers have emerged as an efficient and powerful class of models for reasoning in the language domain. Recent studies show that these models achieve strong performance on algorithmic and reasoning tasks, suggesting that looped architectures possess an inductive bias toward latent reasoning. However, prior approaches fix the number of loop iterations during training and inference, leaving open the question of whether these models can flexibly adapt their computational depth under variable compute budgets. We introduce LoopFormer, a looped Transformer trained on variable-length trajectories to enable budget-conditioned reasoning. Our core contribution is a shortcut-consistency training scheme that aligns trajectories of different lengths, ensuring that shorter loops yield informative representations while longer loops continue to refine them. LoopFormer conditions each loop on the current time and step size, enabling representations to evolve consistently across trajectories of varying length rather than drifting or stagnating. Empirically, LoopFormer demonstrates robust performance on language modeling and reasoning benchmarks even under aggressive compute constraints, while scaling gracefully with additional budget. These results show that looped Transformers are inherently suited for adaptive language modeling, opening a path toward controllable and budget-aware large language models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces LoopFormer, a looped Transformer architecture that can flexibly adapt its computational depth based on a user-specified budget, a feature termed "elastic depth". The model is designed to perform robustly across a range of loop iterations at inference time without needing to be retrained, addressing a key limitation of prior looped models.

Traditional looped Transformers are trained and evaluated with a fixed number of loop iterations. This rigidity means they cannot adapt to variable compute budgets; their internal representations tend to collapse or stagnate when evaluated at depths different from their training configuration, leading to degraded performance.

The paper introduces a shortcut-consistency training scheme that enables compute-budgeted inference (elastic depth) without retraining the model. 
The work demonstrates that naive early-exiting in looped models leads to representational collapse, where hidden states stagnate across iterations. In contrast, LoopFormer's representations continue to evolve, showing that it uses additional depth effectively for refinement.

### Strengths
The paper introduces the novel concept of "elastic depth" for looped Transformers, creatively adapting ideas from diffusion models to frame iterative refinement as a continuous-time "thought trajectory" . This unique problem formulation and synthesis of ideas results in a highly original approach to budget-conditioned reasoning.

The work is of high quality, featuring a rigorous experimental setup with strong, appropriate baselines and a comprehensive evaluation on both perplexity and a wide range of reasoning tasks . The claims are further substantiated by an in-depth representational analysis that uses multiple metrics to convincingly demonstrate that LoopFormer avoids the representational collapse that plagues naive adaptive methods.

The paper's contribution is significant because it provides a practical solution to the important problem of adaptive computation, making efficient looped architectures more versatile and deployable.

### Weaknesses
The shortcut-consistency training algorithm requires two forward passes per batch (one for the full trajectory and one for a shortcut), which effectively doubles the training compute compared to baseline looped models . The paper acknowledges this "added training overhead" but does not analyze the trade-off. A key missing experiment is a comparison against a baseline model trained for twice as many steps, which would clarify if the inference-time flexibility is worth the significant increase in training cost.

The model's performance is highly sensitive to the choice of the inference-time step schedule, yet this schedule must be selected manually by the user1111. This introduces a required hyperparameter tuning step that complicates deployment and may prevent users from achieving the model's optimal performance for a given budget. The work would be stronger if it included a method to learn or predict an optimal schedule automatically.

The experiments are conducted on ~1B parameter models, and it is unclear if the training dynamics and performance benefits will scale to much larger, state-of-the-art foundation models.

### Questions
Your results show that performance is sensitive to the choice of the inference-time schedule. Could you provide a practical heuristic for selecting a high-performing schedule without an exhaustive search? Have you considered methods for learning an optimal, input-dependent schedule to automate this process? A response here would clarify if this is a minor tuning step or a significant practical hurdle.

The shortcut-consistency training algorithm appears to double the training compute by requiring two forward passes per batch . Could you provide an analysis of this trade-off? Specifically, how does LoopFormer compare against a baseline (e.g., TMLT) that is trained for twice as many steps using the same total compute budget? This would clarify whether the inference-time flexibility justifies the increased training cost.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents LoopFormer, which allows reasoning under variable compute budgets. It extends prior looped Transformers by introducing time- and step-size modulation, where each iteration receives sinusoidal embeddings of layer index and step size to dynamically modulate RMSNorm and residual scaling.
A shortcut-consistency loss aligns representations across different loop lengths, enabling stable performance even with fewer inference steps.

### Strengths
* The paper is well written and easy to follow, with clear motivation and setups
* The motivation is clearly presented, and the transition from fixed-depth looped Transformers to elastic-depth design feels natural.
* Experiments are reasonably comprehensive, evaluating both variable loop lengths and the effect of the proposed shortcut-consistency loss.
* The main claims are well supported

### Weaknesses
* The degree of novelty is not bad but moderate. While the proposed elastic-depth formulation and shortcut-consistency loss are well designed, they extend existing time-modulated looped Transformer frameworks rather than introducing a fundamentally new paradigm.
* the paper does not provide theoretical intuition or analysis explaining why combining t and $\Delta t$ through sinusoidal modulation is a good choice here

### Questions
NA

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
This paper tackles the limitation of looped Transformers, whose performance typically degrades when inference loops don't match the fixed number used in training. They propose LoopFormer which novelly condition each loop on both normalized time and step size to adapt to different computational depth. The model is trained on "shortcut-consistency" loss that forces shorter trajectories to match the representation of the full trajectory. They empirically show that the LoopFormer perform more robustly at smaller compute budgets than existing baselines.

### Strengths
1. The paper tackles a clear, practical, and important problem. Enabling flexible, elastic compute in parameter-efficient models like looped Transformers is a highly valuable research direction.
2. The paper is well written and easy to follow.
3. The experiments are thorough and convincing. In addition to strong task performance, the authors provide a compelling explanation for why LoopFormer works by analyzing metrics like curvature, anisotropy, and CKA similarity. They demonstrate that baselines suffer from representational stagnation (flat metrics, high CKA), while LoopFormer's representations continue to evolve and refine with each loop. This analysis significantly strengthens the paper beyond just reporting better numbers.

### Weaknesses
1. The training procedure (Algorithm 1) requires two forward passes per batch (for the full and short trajectories) to compute the consistency loss. This appears to roughly double the training cost compared to a standard looped model. The paper mentions this as a limitation but does not quantify it. A brief analysis of the training FLOPs/time overhead vs. a Base-Loop or TMLT baseline would be valuable for assessing the practical trade-offs.
2. The paper heavily emphasizes "latent reasoning," using terms like "thought trajectories." However, the benchmarks (COPA, PIQA, HellaSwag) are standard zero-shot LM evaluation tasks, not complex, multi-step algorithmic reasoning tasks. The model shows improved general performance (including perplexity), and the representation analysis shows refinement, not necessarily reasoning in a formal sense. So the terming could be a bit misleading to readers.

### Questions
1. What is the practical training overhead (e.g., in FLOPs or wall-clock time) of the dual forward pass required for the consistency loss, compared to a standard Base-Loop or TMLT baseline?
2. Given that Figure 5 shows significant performance variance based on the chosen step schedule Δ_M, have you explored methods for learning an optimal, budget-aware scheduling policy?
3. The model is trained to interpolate (M <= L). Have you tested its ability to extrapolate to M > L loops? Does performance continue to improve, or does it diverge given the t=1 training target?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a new model composed of k blocks each trained to be repeated up to L times. The architecture adds in each block a side information about the scheduling to visit [0, 1] with M steps, and modulate a scale and gate layer before/after the multi-head attention and FFN blocks.

The training procedure uses a loss that combines the cross-entropy of the full-compute (L iterations), the cross-entropy of a lesser compute (M iterations), and a L2 between the pre-readout activations of both.

Experimental validation shows that this approach is not as good as a vanilla architecture for a given compute budget, but has a lesser parameter count.

### Strengths
The proposed architecture is reasonably simple and well motivated. The overall direction, transformer-like that can dynamically modulate compute, is important. The results showing the nice monotonicity of perplexity or accuracy vs. FLOPs (Fig 2) is great.

### Weaknesses
The experimental results are underwhelming, and maybe I missed the point, but it is unclear to me how this model improves wrt a vanilla transformer. The results are presented in such a way that the training budget is not equalized (if I am correct? I do not understand the first sentence of 4.1), the inference flops are, and the parameter counts are not, although this is generally not the limiting factor.

Training budgets should have been equalized (e.g. pick the nb of training iterations per model), and results should be summarized with scatter plots on e.g. an average accuracy vs. wall-clock time or flops frame.

The base model should be added as a point on Fig 2 (a) and (b)

### Questions
It is unclear in Fig 1 and in the loss definition of 3.3 how the method deals with multiple blocks. It looks as if there was only one there.

Regarding the training cost and the overall performance, can you clarify in what regime you see the usefulness of this model?

### Soundness
2

### Presentation
3

### Contribution
2
