# Gradient-Sign Masking for Task Vector Transport Across Pre-Trained Models

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4, 6

## Abstract
When a new release of a foundation model is published, practitioners typically need to repeat fine-tuning, even if the same task was already tackled in the previous version. A promising alternative is to reuse the parameter changes (i.e., task vectors) that capture how a model adapts to a specific task. However, these vectors often fail to transfer across different pre-trained models because their parameter spaces are misaligned. In this work, we show that successful transfer depends strongly on the gradient-sign structure of the new model. Based on this insight, we propose GradFix, which approximates the ideal sign structure and leverages it to transfer knowledge using only a handful of labeled samples. Notably, this requires no additional fine-tuning: we only compute a few target-model gradients without parameter updates and mask the source task vector accordingly. This yields an update that is locally aligned with the target loss landscape, effectively rebasing the task vector onto the new pre-training. We provide a theoretical guarantee that our method ensures first-order descent. Empirically, we demonstrate significant performance gains on vision and language benchmarks, consistently outperforming naive task vector addition and few-shot fine-tuning. We further show that transporting task vectors improves multi-task and multi-source model merging. Code is available at https://github.com/fillo-rinaldi/GradFix.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents GradFix, which aims to transfer task vectors between different pre-trained models by leveraging gradient sign structures in settings with limited downstream data. Specifically, GradFix computes gradients on the target pre-trained model and uses the resulting gradient signs to mask out harmful update directions that may increase the downstream loss. The paper also provides a theoretical guarantee showing that this procedure yields a first-order reduction in loss. To address the limited data regime, a majority-vote strategy is adopted to robustly estimate gradient signs. Experimental results demonstrate that GradFix outperforms baselines such as naive transfer and related approaches, and reduces the performance gap to full-dataset fine-tuning.

### Strengths
- Overall, the paper is well-written and easy to follow. It investigates an important and practical problem regarding improving the transfer efficiency of task vectors amid the rapid advancement of foundation models. Moreover, the simplicity of the proposed method also makes it easy to understand and implement.

- The experimental results are promising, showing that GradFix consistently outperforms the baselines and related methods, and reduces the performance gap to full-dataset fine-tuning.

- The paper provides a theoretical guarantee that GradFix ensures a first-order decrease in the loss. This also helps explain why and how the proposed method is effective.

- Experiments examine the behavior of GradFix under different types of downstream data subsets and hyperparameter settings, which helps improve the understanding of the method’s applicability.

### Weaknesses
- At Lines 36-38, the paper states that one motivation is to reduce redundant re-training when new models are released by companies or researchers. However, newer and more advanced models often involve architectural changes (e.g., increased model size to accommodate knowledge learned from larger datasets), whereas GradFix appears to require the pre-trained models to share the same architecture. This constraint may limit the applicability of the method in such scenarios.

- At Lines 469-471, the paper states that GradFix does not require fine-grained tuning of $\alpha$. However, Figure 3 indicates that different downstream datasets achieve optimal performance at different $\alpha$ values. For example, $\alpha = 0.3$ performs best for EUROSAT, SVHN, and RESISC45, whereas $\alpha = 0.7$ is optimal for GTSRB. Moreover, the accuracy differences across $\alpha$ values on these datasets can be as large as 10-20%. Despite this, the paper does not provide a strategy for selecting $\alpha$ when adapting to new datasets or pre-trained models in practical use cases.

- Table 1 and Table 2 do not report accuracy variance across different sample selections in the few-shot setting. However, Figure 2 suggests that random data selection can still affect the performance of GradFix to a certain degree. It is therefore unclear how the performance variance of GradFix compares to that of $\theta_B^{\text{opt}}$​. For example, are there few-shot sample selections where $\theta_B^{\text{opt}}$​ actually outperforms GradFix, and how frequently does this occur?

### Questions
Besides the weakness shown in the above section, please also see the following questions: 

Q1: From Equations (1) and (4), the mask $m$ seems to take binary values (0 or 1). However, Figure 1 depicts $m$ with positive and negative signs. Why does the figure illustration of $m$ differ from its binary formulation? 

Q2: In the limited-data scenario, the performance gap between GradFix and the oracle (i.e., $\theta_B + \delta^*$) remains substantial. How much does GradFix improve as more samples become available, for example, when the full dataset is used? Additionally, at what sample size does standard fine-tuning begin to outperform GradFix? Understanding this transition point would help clarify which approach is preferable under different computation, labeling, or storage budgets.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses an interesting problem to transfer the task vector on a finetuned task across different pretrained checkpoints. The authors propose to mask the task vector based on whether its sign aligns with the gradient obtained with a small target set, which empirically improve the performance consistently. 

Further ablations are needed to convincingly show that the observed improvement is due to cross-model task knowledge transfer rather than exploiting from a random vector based on incidental gradient sign alignment.

### Strengths
1. The proposed method is well-motivated, simple yet effective. 
2. The method demonstrates consistent performance gain compared to zero-shot and other baselines.

### Weaknesses
1. While the proposed leads to consistent performance gain, the performance gap is still quite substantial compared to the fully-finetuned counterpart (e.g., 20%-40% accuracy gap). There seems to be a diminishing effect brought by increasing the size of the small target set to estimate the gradient sign (Table 5), which prevents the proposed method to eventually deliver on-par performance with fully-finetuned checkpoint even with large number of samples.

2. There is no ablation to verify whether improvements truly reflect transfer of task-specific knowledge from the original task vector or the model is just exploiting a random vector by incidental gradient alignment. 

It is non-intuitive to readers that the original task vector can be transferred to the new checkpoint without any parameter permutation. One would expect the original task vector could be just random noise to the new pretrained checkpoint, which is partially verified by the results that naively adding them do not improve performance. Suppose it is actually noise, there is a possibility that the proposed method works because it is selecting the parameters which “luckily align with the right gradient direction” from the original task vector even though it is noise. I think a further ablation experiment is needed to verify whether this is the case. For example, I suggest the authors to test their proposed method on a randomly generated vector (e.g., a Gaussian vector with the same mean and std as actual task vector) and compare it with the performance obtained with the original task vector. The result comparison should verify to which extent the proposed method is exploiting from the actual task knowledge in the original task vector.

### Questions
Typos:
The alpha coefficient in Eq. 5 seems to be redundant, as it appears also in Eq. 6.

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
The paper recognizes that updating checkpoints with new data or a new training pipeline leads to redundant repeated fine-tuning for the same downstream tasks. To address this, it proposes GradFix, which uses gradient-sign masking to transfer task-specific knowledge from the pre-trained model.

### Strengths
1.The paper proposes a fine-tuning-free knowledge transfer strategy that uses the sign of the gradient as a robust surrogate for the descent direction. The method is insightful and reasonably well-motivated.
2.The paper is clearly structured and well-organized, with high readability.

### Weaknesses
Please refer to the Questions section below.

### Questions
1.The proposed method appears applicable only to models with the same architecture trained on different downstream datasets, which limits its scope. If the two models share only similar architectures but differ in scale, does the method still apply?
2.The paper does not consider more complex multi-source transfer scenarios—i.e., simultaneously transferring multiple task vectors from different source models (such as models pre-trained on different datasets but adapted to the same task) into a single target model.
3.Can your method be transferred to diffusion-based generative models or VLM tasks? These pre-trained models are typically very large yet serve many downstream tasks and encode rich priors. If extending to VLM, what adaptations would be required?

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
This paper proposes GradFix, a simple method to transfer task vectors between different pre-trained models. It masks the source task vector using the gradient sign of the target model, keeping only directions aligned with the target loss. This ensures a first-order descent update without re-training. Experiments on CLIP and T5 show that GradFix effectively transfers fine-tuning knowledge using only a few labeled samples, outperforming naive task-vector addition and few-shot tuning.

### Strengths
1. The idea is simple and easy to understand, yet surprisingly effective for transferring knowledge between models.
2. The method is lightweight and requires very little data, making it practical and efficient.
3. Results are clear and consistent across different tasks, showing strong generality and reliability

### Weaknesses
1. The motivation of the setting is not entirely convincing. If the source model has already been fine-tuned on the task, it is unclear why one would not simply use that model directly. Since obtaining the task vector requires access to the fine-tuned source, the need to reapply it to a target model feels less practical, especially when the transferred performance does not surpass the source.
2. The method assumes that the source and target share the same architecture, differing only in pre-training. This limits real-world applicability, as many practical scenarios involve transferring between models with different structures or sizes.

### Questions
1. Why can’t we directly use the source model? If the source model is indeed inaccessible, obtaining its task vector still seems almost equivalent to having it, since it only differs from the base pre-trained weights—usually publicly available.

2. Could the authors provide a more convincing motivation or real-world use case where such task-vector transport is necessary or clearly beneficial?

3. Is the task-vector transport still effective in multi-task scenarios—for example, when merging multiple transported task vectors?

I would be happy to raise my score if the authors can address all these concerns convincingly.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes GradFix, a method to transfer task-specific fine-tuning knowledge from a source pre-trained model to a different target pre-trained model. The authors argue that naive transfer fails due to parameter space misalignment and that the key to successful transfer lies in the sign structure of the target model's gradients. GradFix computes the gradient on the target model using only a few labeled samples. It then creates a binary mask by checking for sign agreement between the source task vector and the target gradient. This mask filters the source task vector and updates the target model parameters without any fine-tuning. The authors provide a theoretical guarantee that this update ensures a first order loss descent on the target model. Empirically, GradFix is shown to outperform naive task vector addition and standard few-shot fine-tuning on vision and language benchmarks.

### Strengths
1. The challenge of updating downstream tasks when a new foundation model is released is a significant and practical problem for the ICLR community.
2. The core idea of using the target model's gradient signs as a filter for the source task vector is elegant and easy to understand. The method's simplicity of requiring only a single forward-backward pass on a few labeled samples is a major practical advantage.
3. Figure 1 provides an excellent and unambiguous illustration of the core masking procedure.
4. While simple, the proof of a first-order descent guarantee (Sec 4.2)  provides a solid theoretical justification for why the method should work, moving it beyond a simple heuristic.
5. The use of majority voting to estimate gradient signs from a handful of samples (Sec 4.3) is well-justified, and the accompanying concentration analysis (Appendix A)  strongly supports this design choice.
6. The evaluation spans both vision (CLIP ViT) and language (T5) models, demonstrating the applicability of the approach across domains.

### Weaknesses
# Major Concerns

1. **Over-reliance on a weak baseline $\theta_B^{opt}$**: The primary few-shot baseline is defined as fine-tuning the target model with the same few samples $\mathcal{D}_s$. However, Appendix B.2 clarifies that this is achieved with a "single step of gradient descent". This is not a standard definition of few-shot fine-tuning. A more conventional baseline would involve multi-epoch training on $\mathcal{D}_s$ or applying a parameter-efficient method like LoRA on $\mathcal{D}_s$. As such, the claims of consistently outperforming few-shot finetuning are based on a potentially weak comparison, which may inflate GradFix's perceived gains.

2. **Underspecified hyperparameter $\alpha$**: The method introduces a critical scaling factor $\alpha$ in Equation 5, whjich is tuned on a validation set for each dataset (in Table 4). This undermines the "plug-and-play" claim in a strict few-shot setting where a representative validation set may not exist. The sensitivity analysis in Section 5.4 confirms that performance is sensitive to $\alpha$, but it does not propose a way to set $\alpha$ _without_ a validation set. This is a key practical limitation.

3. **Complete disregard for gradient magnitude**: The method relies _only_ on sign agreement, discarding all magnitude information from both the source task vector $\tau_A$ and the target gradient $g_B$. This seems like an extreme simplification. A component of $\tau_A$ with a very large, task-critical magnitude will be zeroed out if it has a sign mismatch. Conversely, a tiny-magnitude component of $\tau_A$ is preserved as long as its sign matches. The justification for this sign-only choice is not deeply explored.

## Minor Concerns

1. In Table 1, the full $\theta_B \textit{fine-tune}$ performance is slightly higher than $\theta_B + \delta^*$ (oracle) performance (e.g., EUROSAT: 98.70 vs 95.06; GTSRB: 98.65 vs 82.92). This suggests that $\tau_A$ is fundamentally missing significant information (or contains conflicting information) that cannot be recovered even by a perfect sign mask based on $\tau_B$. This information gap is a limitation and needs more discussion.

2. In Section 5.2, the paper compares "agreement" with "force agreement". The argument is that "agreement" is more robust to noisy gradients. However, "force agreement" seems more conceptually aligned with the theoretical goal of ensuring a descent direction $g^\top \delta^A \geq 0$. This choice feels non-obvious and the empirical difference (Table 3) is small, making the justification feel post-hoc.

3. he finding that random sampling is a strong baseline compared to structured methods like coreset or herding is a good practical result . However, it is also surprising. This might imply that the gradient sign structure is extremely stable across the data distribution, which is an interesting finding in itself and could be analyzed more deeply.

### Questions
1. Could the authors please justify the "single step of gradient descent"  for the $\theta_B^{opt}$ baseline? How does GradFix compare to a more standard few-shot baseline, such as training $\theta_B$ on $\mathcal{D}_s$ for maybe 10 epochs, or training a PEFT method on $\mathcal{D}_s$?

2. In a practical, no-validation-set scenario, how should $\alpha$ be set? What is the performance of GradFix if $\alpha$ is fixed to a default value (e.g., $\alpha = 0.5$) across all datasets in Table 1?

3. Have you explored hybrid approaches that do not discard magnitude entirely? For instance, what happens if you down-weight sign-mismatched components instead of zeroing them, or use a method that combines signs with magnitudes (e.g, TIES-Merging style?)

4. The related work on TIES-Merging  also focuses on resolving sign conflicts for task vectors. How does GradFix's "masking" approach conceptually differ from TIES's "sign-consistency" aggregation? How would GradFix compare empirically if TIES was adapted for this transport setting?

### Soundness
3

### Presentation
3

### Contribution
3
