# On Optimal Hyperparameters for Differentially Private Deep Transfer Learning

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 2

## Abstract
Differentially private (DP) transfer learning, i.e., fine-tuning a pretrained model on private data, is the current state-of-the-art approach for training large models under privacy constraints. 
We focus on two key hyperparameters in this setting: the clipping bound $C$ and batch size $B$.
We show a clear mismatch between the current theoretical understanding of how to choose an optimal $C$ (stronger privacy requires smaller $C$) and empirical outcomes (larger $C$ performs better under strong privacy), caused by changes in the gradient distributions. 
Assuming a limited compute budget (fixed epochs), we demonstrate that the existing heuristics for tuning $B$ do not work, while cumulative DP noise better explains whether smaller or larger batches perform better. 
We also highlight how the common practice of using a single $(C,B)$ setting across tasks can lead to suboptimal performance. 
We find that performance drops especially when moving between loose and tight privacy and between plentiful and limited compute, which we explain by analyzing clipping as a form of gradient re-weighting and examining cumulative DP noise.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on clipping threshold and batch size. The paper studies how these two usually-fixed hyperparameters in differentially private (DP) transfer learning should depend on privacy level, compute budget, and backbone strength.

The authors (i) document that empirical optima contradict a common theoretical takeaway ("tighter privacy $\Longrightarrow$ smaller C"), (ii) provide an analysis that explains when larger C helps under tight privacy via a new MSE-based optimal clipping result (Theorem 5.1), and (iii) argue that in fixed‑epoch (compute‑bounded) training, cumulative DP noise ​$\sigma\sqrt{T}$ together with a minimum-steps constraint better explains optimal batch size than per‑step noise heuristics developed for fixed‑steps regimes. The empirical study uses DP‑Adam with PRV accounting ($\delta$= 1e−5) and extensive grid search over LR, B, C on SUN397, Cassava, CIFAR‑100 (and a 10% subset), comparing ViT‑Base vs ViT‑Tiny (and some ResNet‑50 in the appendix).

For the batch size part, the authors show that “effective noise” plots used in fixed‑steps heuristics (e.g., $\sigma/q$ ) monotonically favor full batch under fixed‑epochs, hence are uninformative (Figure 6). Instead, plotting cumulative noise ​$\sigma\sqrt{T}$ versus B reveals plateaus under tight privacy where moderate B can match or beat large B, provided a minimum number of steps is met (Figure 7). The appendices give dataset‑specific grids (Table A1) and many robustness plots across $\epsilon$'s and epochs.

### Strengths
1. Many DP transfer‑learning pipelines fix C and B across tasks; this paper convincingly shows that practice leaves accuracy on the table and can be especially harmful in hard regimes (e.g., small privacy budget, limited compute, weaker backbones)

2. Theorem 5.1 explains when the optimum moves the other way because the gradient distribution itself shifts under DP. Figure 3 and the Figure 4 make this intuitive and actionable.

3. The fixed‑epoch framing matches many real‑world budgets. The "cumulative noise plateau" and "minimum steps" interpretation is intuitive and well‑visualized.

4. Implementation details like hyper-parameter grids, (un)normalized-DP version, and class imbalance conditions are well documented and explained.

### Weaknesses
**Major concerns**
1. Previous works have shown that the choice of fine-tuning methods and initialization of heads could influence the DP performance, e.g. [1] [2]. However, all main experiments in this paper fine‑tune with FiLM (scale/bias of normalization layers + head), leaving about 0.5–1.5% of parameters trainable (Section Fine‑tuning). While the authors vary backbones (ViT‑Tiny/ViT‑Base and some ResNet‑50), the parameterization is fixed; Table A1 confirms all grids are under FiLM. In the discussion they state the mechanisms are “general” and expect transfer to full‑model tuning, but this is not demonstrated. This design choice narrows the external validity of the paper’s conclusions about optimal clipping C and batch size B.

References:

[1] Want et al. 2024. Neural Collapse meets Differential Privacy: Curious behaviors of NoisyGD with Near-Perfect Representation Learning. ICML.

[2] Ke et al. 2024. Characterizing the Training Dynamics of Private Fine-tuning with Langevin diffusion. NeurIPS Workshop on Fine-Tuning in Modern Machine Learning: Principles and Scalability.

**General weakness**

1. Most results are with DP‑Adam on image classification and parameter‑efficient fine‑tuning. While some ResNet‑50 and dataset diversity exist in the appendix, it remains unclear how fully the conclusions transfer to DP‑SGD, full‑model fine‑tuning, or other modalities (NLP, speech). The paper argues mechanisms are general, but evidence is image‑centric.

2. The "minimum steps + cumulative noise plateau" guidance is compelling visually, but a practitioner‑ready recipe (how to pick/estimate the steps threshold per dataset/task without extra privacy spend) is not spelled out. For example, in Figure 7 the dashed 20‑step threshold is illustrative, and Appendix A notes dataset‑dependence, but there is no principled estimator.

3. The study performs extensive grid searches, tuning LR jointly with C,B per configuration, which is appropriate for understanding behavior but raises the usual question of privacy leakage from tuning in practical deployments. The paper cites work on DP hyperparameter tuning, but it does not quantify or discuss the privacy cost of its own search.

**Theorem 5.1 is confusing to me for several reasons.**

- Weak theory–practice bridge. The current MSE criterion is explanatory rather than prescriptive. Without a descent‑type link (the small lemma suggested above) or a correlation study tying MSE proxies to accuracy across your sweeps (Fig. 2 and Fig. 4), the reader is asked to accept that “lower MSE $\approx$ better” on faith. Clarifying this would substantially strengthen the contribution.

- State prominently that $N_C$ and $G_C$ are used only for analysis in this paper, and provide one of the DP‑compliant procedures above for any future adaptive use. As written, a well‑meaning practitioner could implement a non‑private adaptation.

- Recasting Theorem 5.1 as a piecewise‑explicit optimizer (with convexity and boundary cases spelled out) would eliminate the fixed‑point ambiguity and address the "hand‑wavy" perception. Please centralize all assumptions next to the theorem.

**Some other minor issues.**
- In Figure 2, the plots only show relative accuracy differences. It would always be helpful to show the absolute accuracy value of each experiment result here. Readers would always prefer complete information rather than a visually straightforward but incomplete figure. And it would be helpful to provide a zero-shot baseline (i.e. no fine-tuning at all). The zero-shot baseline would provide a basic sanity check when the authors argue that the number of parameters is positively correlated to the "model capacity".

### Questions
- In Figure 4, could you explain more on how you determine which classes are harder than others?

- Could you turn Theorem 5.1 into a procedure? In other words, can you propose a privacy‑respecting estimator for the statistics that define the optimal $C$? How sensitive would such a method be to noise?

- How would you incorporate DP hyperparameter tuning (e.g., subsampled validation with DP bandit/BO) into your workflow to preserve the reported gains when tuning itself must be private?

### Soundness
3

### Presentation
3

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
This paper investigates the optimization strategies for hyper-parameters in DP finetuning, including the clipping bound $C$ and the batch size $B$. For the clipping bound $C$, the author presents a finding that under strict privacy budget, a larger clipping bound leads to improved model performance. For the batch size $B$, the paper proposes a principle for selecting the optimal $B$ on the fixed-epochs setting. This rule aims to overcome the limitations of traditional fixed-step theories by maximizing the number of optimization steps $T$ (via minimizing $B$) while ensuring that the cumulative DP noise ($\sigma \sqrt{T}$) remains within its near-optimal plateau region.

### Strengths
1. The paper demonstrates that the clipping bound $C$ and batch size $B$ should be dynamically adjusted based on the difficulty of the learning task, which is different from the traditional approach with fixed hyper-parameters.
2. The paper introduces the perspective of "gradient re-weighting," which provides a detailed analysis of how the clipping bound $C$ affects the learning process.
3. The paper highlights the necessity of jointly tuning $C$, $B$, and the learning rate, noting their complex interactions.
4. Comprehensive experiments are conducted.

### Weaknesses
1. The study relies only on the Vision Transformer; it would be beneficial to explore various models to verify the effect of pre-trained model capability on optimal clipping. At a minimum, some discussion of future work is necessary.

2. It would be great if the authors could discuss other related works concerning optimal hyperparameters in DP fine-tuning, especially the learning rate. For instance, [A] pointed out that if the features of a ViT are extracted near-perfectly, then DP fine-tuning is less sensitive to hyperparameters. This implies that in some extreme cases with near-perfect features, optimal hyperparameter tuning may not be necessary—a conclusion that differs from this paper, which considers cases where pre-training extracts near-perfect features for the fine-tuning datasets. However, [B] shows that if the features are not extracted perfectly, then optimal hyperparameters, especially an optimal learning rate, are necessary; otherwise, the features may be degraded by the noise, a conclusion that aligns with this work.


[A] Neural Collapse Meets Differential Privacy: Curious Behaviors of NoisyGD with Near-perfect Representation Learning. Wang et al., ICML, 2024.

[B] Why Does Private Fine-Tuning Resist Differential Privacy Noise? A Representation Learning Perspective. Zhao et al., ICLR 2025 Data-FM Workshop, 2025.

3.  The term "learning problem difficulty" is central to the paper but seems to be used informally (please correct me if I have missed a definition). It would be beneficial to define or characterize it more formally, perhaps as a function of the privacy budget (or noise level), model capability, the size of datasets. A brief discussion in Section 5 would suffice.

### Questions
See the weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates a mismatch between theoretical results and empirical findings regarding the optimal clipping threshold C in relation to the privacy budget $\varepsilon$. Thus, for DP transfer learning, the paper proposes using a gradient re-weighting mechanism to justify using a larger C with a smaller $\varepsilon$, and analyzing the total accumulated DP noise to inform batch size selection.

### Strengths
- The paper investigates an interesting idea: that previous approaches using a small, fixed C (e.g., 0.1 or 1) might be suboptimal.
- The authors provide a basic analysis of how DP noise affects the optimal clipping bound.
- Based on their observations, the authors’ proposed tuning approach performs better than using static hyperparameters.

### Weaknesses
Please refer to the Questions section.

### Questions
- The reviewer believes the gradient norm analysis needs to be further developed to fully support the paper's hypothesis. In Figure 3, the authors show the gradient norms after training with various $\varepsilon$ and C, but the gradient norm also varies during training. A discussion of this dynamic behavior seems to be missing.
-  Furthermore, the claim that the gradient norm gets bigger when not converged needs more support. In standard (and possibly DP) training, the gradient norm is often observed to increase towards the end of training (see [1], [2]), which seems to contradict the paper's explanation (i.e., that large norms are simply due to non-convergence). Please provide more backing for this claim.

    [1] Why Gradients Rapidly Increase Near the End of Training, 2025.

    [2] Penalizing Gradient Norm for Efficiently Improving Generalization in Deep Learning, ICML 2022.

-  Is there any significant difference between analyzing 'from scratch' and 'transfer learning' in terms of your analysis or empirical findings? Would your conclusions hold for 'from scratch' training?
-  Can you test your proposed framework on other SOTA DP training methods, such as those mentioned in the related works?

### Soundness
3

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
This paper systematically studies how two critical hyperparameters, gradient clipping bound (C) and batch size (BS), affect performance in fine-tuning large pretrained models under DP constraints. Empirical and theoretical analysis showing that the optimal C depends on the privacy noise scale $\sigma$ and gradient distribution, which itself shifts with privacy level, model capacity, and dataset difficulty.
The paper finds that 
1. Larger clipping bounds C can improve performance under stronger privacy (smaller $\varepsilon$) and for weaker pretrained backbones.
2. The optimal BS depends jointly on privacy and compute budget: under tight privacy, a smaller BS may perform better, while a moderate BS is optimal under limited compute.
3. Fixing (C, BS) across tasks degrades performance, especially for harder datasets and tighter privacy settings.

### Strengths
1. Clear motivation: Hyperparameter selection is an important and open question in DP training.
2. Novel theoretical contribution: Theorem 5.1 provides a principled explanation for observed discrepancies between prior theory and practice regarding the clipping bound.
3. Insightful conceptual framing: Interpreting clipping as gradient re-weighting is intuitive and helps explain asymmetric behavior between easy and hard examples.

### Weaknesses
1. Since the scope of this paper is DP fine-tuning, it would be complete if language classification tasks were considered. Additionally, this paper only considered fine-tuning the classification head plus the scale and bias of normalization layers. It would be complete and more convincing if more baselines were compared, such as LoRA, linear probing, and full fine-tuning, given that the latter are widely used fine-tuning methods.
2. The section "Clipping as Gradient Re-Weighting" is insightful, and it would be better if this section were written more mathematically rigorously rather than in storytelling.
3. Writing: It would be clearer if the authors could summarize and highlight all the implications for practical hyperparameter tuning in bullet points in the introduction/Section 5 & 6.

### Questions
1. Line 30-33 claim that usually only LR is tuned for each separate problem, while other hyperparameters (BS and C) are fixed, with reference made to De et al. 2022. Could the authors clarify how De et al. 2022 is an instance? According to my understanding, De et al. 2022 conducted careful hyperparameter tuning.
2. For the plots that present the change of optimal C/BS under different settings, how are the other hyperparameters (LR, BS/C, epoch) selected? For example, each experiment in Fig. 2 (left) might require its own LR and BS, and it would be helpful to describe how they are selected. Do all the runs in Fig. 2(left) use the same LR and BS? 

Overall, I suggest augmenting the experiments, improving Section 5.3, and improving the writing to make the insights for hyperparameter tuning more executable. After the clarification questions and concerns are addressed, I would be happy to raise my score.

### Soundness
3

### Presentation
2

### Contribution
2
