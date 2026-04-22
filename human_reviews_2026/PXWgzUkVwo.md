# Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 2

## Abstract
Batch size scheduling (BSS) plays a critical role in large-scale deep learning training, influencing both optimization dynamics and computational efficiency. Yet, its theoretical foundations remain poorly understood.
In this work, we show that the **functional scaling law (FSL)** framework introduced in [Li et al. (2025a)](https://arxiv.org/abs/2509.19189) provides a principled lens for analyzing BSS. Specifically, we characterize the optimal BSS under a fixed data budget and show that its structure depends sharply on task difficulty. For easy tasks, optimal schedules keep increasing batch size throughout.  In contrast, for hard tasks, the optimal schedule  maintains small batch sizes for most of training and switches to large batches only in a late stage. 
To explain the emergence of late switching, we uncover a dynamical mechanism—the **fast catch-up** effect—which also manifests in large language model (LLM) pretraining. After switching from small to large batches, the loss rapidly aligns with the constant large-batch trajectory. Using FSL, we show that this effect stems from rapid forgetting of accumulated gradient noise, with the catch-up speed determined by task difficulty. Crucially, this effect implies that *large batches can be safely deferred to late training* without sacrificing performance, while substantially reducing data consumption. Finally,
extensive LLM pretraining experiments—covering both Dense and MoE architectures with up to **1.1B** parameters and **1T** tokens—validate our theoretical predictions. Across all settings, late-switch schedules consistently outperform constant-batch and early-switch baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper provides a theoretical foundation for the common practice of batch size scheduling in LLM pre-training. Motivated by two robust empirical observations of a "sudden drop" in loss and a "final merge" of loss curves upon switching batch sizes, the authors apply the Functional Scaling Law (FSL) framework from Li et al. (2025) to explain these dynamics. The theoretical analysis successfully explains these phenomena and yields several key predictions: a "later is better" rule for switching in data-limited regimes, a quantitative power law relating the optimal switch point to the total data budget, and the functional form of a minimax-optimal continuous schedule. These predictions are supported by experiments on models up to 1.1B parameters, bridging the gap between theory and practice.

### Strengths
1. **Concrete and Predictive Theory:** This work transforms batch size scheduling from heuristic practice into principled science by applying the FSL framework to derive quantitative predictions. The theorems rigorously characterize loss dynamics (sudden drop and final merge) and provide actionable guidance for schedule design.
2. **Strong Empirical Corroboration:** The theoretical predictions are robustly validated across diverse settings. The core phenomena are shown to be robust across different model architectures (LLaMA and MoE) and scales. It is particularly compelling to see the near-perfect empirical fit to the predicted power-law scaling of optimal switch points, which shows the theory's predictive power and practical utility.

### Weaknesses
1. **Constant Learning Rate Assumption:** The analysis assumes a constant learning rate throughout training to isolate batch size effects. This is a major limitation, as modern large-scale pre-training universally employs learning rate schedules (e.g., cosine decay with warmup). 
2. **Missing Empirical Validation of the Optimal Schedule:** Theorem 4.4 derives a continuous, increasing batch size schedule that is provably optimal within the FSL framework. This is arguably one of the paper's most significant theoretical contribution, yet it is not empirically tested. The authors mention practical hardware and software constraints that make continuous schedules difficult to implement, but a proof-of-concept using a discretized approximation of the optimal schedule would have strengthened the paper's conclusions.
3. **Asymptotic Nature of the Theory:** The core theoretical results rely on asymptotic analysis where training time or data budget approaches infinity. While experiments demonstrate reasonable approximation quality for large finite runs, the theory cannot formally guarantee performance in the finite regimes where practitioners actually operate.

### Questions
1. How should FSL parameters be interpreted for LLMs? The FSL framework relies on a power-law kernel characterized by the capacity parameter $\beta$, which governs eigenvalue decay and drives all theoretical predictions. However, the paper doesn't clarify what $\beta$ represents in language model pre-training. What aspects of the data distribution, architecture, or task does it capture?
2. Can the FSL framework predict optimal joint schedules? Li et al. (2025) used FSL to analyze learning rate schedules, while this paper analyzes batch size schedules. The natural next step is unification: can FSL simultaneously optimize both schedules?

### Soundness
4

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
The paper studies batch size scheduling via a power-law regression theoretical model. It provides an explanation behind two observed effects when switching batch size from low to high: sudden loss drop and eventual matching of the loss curve with the high batch size loss curve. It also proposes an optimal batch size schedule from the theoretical model.

### Strengths
1. The paper studies batch size scheduling from a theoretical perspective of a power-law regression model and provides insights related to ‘sudden drop’ and ‘final merge’ of the loss values.

2. It proposes scaling law for optimal switching time from small to large batch in a training run and also empirically verifies that practical settings obey a scaling law.

3. It also proposes an optimal batch size scheduling algorithm for the power-law model.

### Weaknesses
1. The paper only studies a constant learning rate schedule, which deviates from practice.

2. Although the paper proposes an optimal batch size scheduling algorithm as a power law, it provides no way of actually developing a practical optimal scheduling algorithm.

3. I don't think Lemma 3 holds for any arbitrary $\theta$, but only for local minimizers as the expected gradient has to be zero for this to hold.

### Questions
1. Does Lemma 3 hold generally? Can the authors provide a proof for the same?

2. Is there a practical way of implementing (or obtaining) the optimal batch size scheduling scaling law? Can it be verified that its performance matches the performance of cosine decay in practice?

### Soundness
3

### Presentation
3

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
The paper studies two-stage batch-size schedules for LLM pre-training under a constant learning rate: start with a small batch, then switch once to a larger batch. Empirically, the authors report two robust phenomena across dense and MoE models up to 1.1B parameters and up to 1T tokens: (i) a “sudden drop” in loss at the switch, and (ii) “final merge,” where the loss trajectory measured in steps converges toward the always-large-batch curve. They analyze these behaviors using the Functional Scaling Law (FSL) within a power-law kernel (PLK) teacher–student framework, proving the sudden-drop and final-merge effects, and deriving a “later-switch” rule in data-limited regimes along with a power-law scaling of the optimal switch point with total data size. Experiments at several scales corroborate the theory and show later switches tend to yield better final loss under a fixed token budget.

### Strengths
1. Clear empirical phenomena distilled. The paper cleanly isolates and names two behaviors (“sudden drop,” “final merge”) and shows them across architectures/scales, which aids practitioner understanding.
2. Theory that matches practice. The FSL-based analysis explains both phenomena and yields a concrete later-switch rule and a power-law prediction for the optimal switch point, predictions borne out in experiments (including a strong log–log fit).
3. Breadth of evidence. Results span dense (LLaMA-style) and MoE models, billions of tokens, and multiple switch ratios/timings; figures are easy to digest.
4. Actionable takeaway. Under a fixed token budget, “switch later rather than earlier” is a simple guideline practitioners can trial. The paper also discusses hardware constraints motivating staged schedules.

### Weaknesses
1. Theory and most experiments assume constant LR, while real LLM training typically uses warmup + cosine/linear decay.
2. Large-scale runs use a private dataset, which limits external reproducibility.

### Questions
Do sudden-drop/final-merge and the later-switch rule hold under the cosine decay method?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies the effect of batch size scheduling in LLM pretraining, focusing on a simplified case, where the batch size increases just once. They make two findings: (1) when the batch size increases, the test error drops quickly then stabilizes, and (2) after a while, the test error curve merges with the one induced by using a constant large batch size from the beginning. 

Using a continuous approximation of online SGD applied to kernel linear regression, this paper proves the above two observed phenomena. Moreover, they calculated the optimal batch size schedule (in the simplified sense) using their theory. Finally, they conducted experiments in LLaMA (upto 492M parameters) and MoE models (upto 291M activated parameters) to verify their theory prediction.

### Strengths
See below.

### Weaknesses
See below.

### Questions
My major concern with the paper is regarding the theory component, which seems fabricated instead of fully rigorous. First, the problem setting is highly simplified: constant step size online SGD for kernel linear regression, with a potential two-stage batch size scheduler, and under the source and capacity conditions. 

This problem, without the two-stage batch size complication, is very well studied in kernel linear regression literature. However, this paper chooses to take a weird approach by analyzing a continuous approximation of the discrete online SGD process. 

Note that in online SGD, the optimization time is tied with the data size. Hence, it can be dangerous extending insights from continuous process to the discrete process. I have noticed one such issue in Theorem 4.4. Detailed as the following question. 

1. Theorem 4.4 cannot be correct -- it is well known that one-pass SGD is suboptimal for certain power-law classes. Specifically, in Theorem 4.4, when $\\beta > 1+s\\beta$, the optimal time $T^*$ is greater than data size $D$, which violates the online nature of the algorithm. Therefore, the discussions in Section 4.3 are misleading. 

Despite this major issue, it is unclear why the paper focuses on constant step size setting. 

In this setting, it is well known that the last iterate of SGD does not converge due to the additive variance error. Indeed, the drop of the loss by increasing batch size is exactly because increasing batch size decreases the variance error. However, in standard SGD literature, one should analyze the averaged iterates or the last SGD iterate but with a decaying step size scheduler. The latter is also what practitioners do. This leads to my second question:

2. For online SGD with a reasonable step size scheduler, e.g., exponentially decaying one or cosine, would increasing batch size also cause a sudden drop of the test error? 

Those issues could have been avoided by using the well known rigorous tools developed by prior kernel linear regression literature, instead of using the heuristic continuous approximation. 


Besides those theoretical questions, I also feel the experiments are not on a sufficiently large scale. However, I am not an expert here, so I will also wait to see other reviewers’ comments on this. 

Overall, I am not fully convinced by the sudden drop of the loss story. But even if I choose to believe it, I do not quite see the implications of this observation:

3. Is the two-stage batch size schedule of any practical importance? When considering more comprehensive batch size schedulers, would the sudden drop still hold to some extent? What's the motivation for studying two-stage batch size schedule?


A minor issue.

4. Lemma 3.3 relies on a certain fourth moment hypercontractivity condition on the feature, however, this is not explicitly mentioned. The constants 2 and 4 seem to suggest they need the feature to be exactly Gaussian.

### Soundness
2

### Presentation
2

### Contribution
2
