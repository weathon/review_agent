# Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 8, 4

## Abstract
Diffusion model distillation has emerged as a powerful technique for creating efficient few-step and single-step generators. Among these, Distribution Matching Distillation (DMD) and its variants stand out for their impressive performance, which is widely attributed to their core mechanism of matching the student's output distribution to that of a pre-trained teacher model. In this work, we challenge this conventional understanding. Through a rigorous decomposition of the DMD training objective, we reveal that the primary driver of few-step generation is not the distribution matching term, but a previously overlooked component we identify as \textit{\textbf{C}FG \textbf{A}ugmentation} (\textbf{CA}). We demonstrate that this term acts as the core "engine" of distillation, while the \textbf{D}istribution \textbf{M}atching (\textbf{DM}) term functions as a "regularizer" that ensures training stability and mitigates artifacts. We further validate this decoupling by demonstrating that while the DM term is a highly effective regularizer, it is not unique; simpler non-parametric constraints or GAN-based objectives can serve the same stabilizing function, albeit with different trade-offs. This decoupling of labor between CA and DM also allows a more principled analysis of the properties of both terms, leading to a more systematic and in-depth understanding. This new understanding enables us to propose principled modifications to the distillation process, such as decoupling the noise schedules for the engine and the regularizer, leading to further performance gains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work revisits what makes DMD effective. While DMD’s success is usually credited to matching the student’s output distribution with a teacher model, the authors show that the true driver is an overlooked component called CFG Augmentation (CA). They find that CA acts as the core engine of distillation, while the Distribution Matching (DM) term mainly serves as a stabilising regularizer. They also show that recognising this separation enables a clearer understanding of DMD and allows improvements such as decoupling noise schedules for better performance.

### Strengths
I really like this research topic and believe the distribution-matching distillation is an under-explored topic, and only from a divergence perspective, it can't answer why it works or why it doesn't work in some scenarios, so I think the topic of this paper is very valuable.

The experiments are also sound, which can support the argument.

### Weaknesses
My major concern with this paper is that I found the conclusion a little bit conclusive.

The argument is CFG Augmentation is the engine for dilatation, and Distribution Matching is the regularizer for stability.  

However, many CIFAR experiments don't use label-conditioned and can achieve one-step distillation, e.g. the original diff-intruct paper or more recent paper: https://arxiv.org/pdf/2502.08005. In this case, the pure driven engine is only the distribution matching term, which couldn't be explained by the hypotheses introduced in the paper.

It may be possible that CFG can play a key role in the conditional generation, but it is hard to say DM is not the engine.

Minor: The distillation also relates to the student model score estimation quality, initialisation, teacher model's score quality, etc... It would be good to add some analysis on that.

### Questions
See above, why the unconditional CIFAR works with only DM term?

Happy to increase the score if the concern could be solved.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper challenges the conventional understanding of the underlying mechanisms of Distribution Matching Distillation (DMD) for distilling pre-trained diffusion models into one/few-step student models. While it might be tempting to think that DMD's success mainly stems from matching the student's output distribution to the teacher's output distribution, the authors decompose the DMD loss into a distribution matching (DM) term and a CFG augmentation (CA) term, arguing that it is the CA term that plays the primary role in the distillation process. Surprisingly, the DM term functions more like a stabilizing regularizer and could be replaced by other regularization terms with different trade-offs. Leveraging this insight, a decoupled noise schedule is proposed for CA and DM to improve the model performance.

### Strengths
1. This paper identifies a discrepancy between theory and practice in DMD that CFG is only used in the teacher model but not the student model. This is an interesting observation and a natural motivation for this important research topic.
2. The decomposition of the DMD loss into the DM and CA terms provide novel and valuable insights towards a better and principled understanding of the underlying mechanism of DMD.
3. The arguments and hypotheses in the paper are supported by extensive experiments with ablation studies, demonstrating impressive empirical results.
4. The paper is well-written and easy to understand. It also acknowledges the limitations of the current understanding of the CA term and provides some preliminary discussions.

### Weaknesses
Overall, I like the paper very much. My only concern is the paper's claim about the CA term being the engine for DMD, which is a bit strong to me. For example, early DMD papers achieved great distillation performance on unconditional generation for CIFAR images, which is not discussed or explored in this paper.

### Questions
Could the authors comment on the issue in the weakness section? One way to address this issue is to reduce the claim to "the CA term is the engine for DMD **in conditional generation**".

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
The paper investigates the respective roles of the two loss components—CFG augmentation and distribution matching—in the DMD framework. Through carefully controlled experiments, the authors conclude that CFG augmentation serves as the primary driver for few-step or one-step conversion, while distribution matching acts mainly as a regularizer. They further argue that, although alternative regularizers could be used, distribution matching remains the best fit. Finally, the paper observes that assigning different $\tau$ values to the two loss terms yields additional performance improvements.

### Strengths
* Provides a timely and insightful analysis of the functional roles of DMD’s two loss terms, addressing the open question of why DMD excels in few-step or one-step generation.
* The authors design careful and hypothesis-driven experiments to isolate and test the contribution of each loss term, leading to well-supported conclusions. 
* Based on these insights, the paper proposes using distinct $\tau$ values for the two terms, leading to measurable performance gains.

### Weaknesses
Most experiments rely primarily on qualitative evaluation (visual inspection of generated images). While visualization is valuable for illustrating effects, heavy reliance on qualitative judgments risks confirmation bias—highlighting supportive examples while overlooking contradictory ones. A more scientifically rigorous approach would involve defining quantitative metrics and validating observations across the entire test set, to ensure statistical robustness and reproducibility.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
