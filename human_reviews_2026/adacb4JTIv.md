# AlphaFlow: Understanding and Improving MeanFlow Models

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 8

## Abstract
MeanFlow has recently emerged as a powerful framework for few-step generative modeling trained from scratch, but its success is not yet fully understood. In this work, we show that the MeanFlow objective naturally decomposes into two parts: trajectory flow matching and trajectory consistency. Through gradient analysis, we find that these terms are strongly negatively correlated, causing optimization conflict and slow convergence. Motivated by these insights, we introduce $\alpha$-Flow, a broad family of objectives that unifies trajectory flow matching, Shortcut Model, and MeanFlow under one formulation. By adopting a curriculum strategy that smoothly anneals from trajectory flow matching to MeanFlow, $\alpha$-Flow disentangles the conflicting objectives, and achieves better convergence. When trained from scratch on class-conditional ImageNet-1K 256×256 with vanilla DiT backbones, $\alpha$-Flow consistently outperforms MeanFlow across scales and settings. Our largest $\alpha$-Flow-XL/2+ model achieves new state-of-the-art results using vanilla DiT backbones, with FID scores of 2.58 (1-NFE) and 2.15 (2-NFE). The source code and pre-trained checkpoints are available on \url{https://github.com/snap-research/alphaflow}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper provides a theoretical and empirical analysis of MeanFlow, a recently proposed 
framework for few-step generative modeling. The authors decompose the MeanFlow objective 
into trajectory flow matching and trajectory consistency components, 
revealing that their gradients are strongly negatively correlated during training, causing 
optimization conflict. Motivated by this insight, they propose α-Flow, a unified family of 
objectives.The paper is well-motivated and clearly written.

### Strengths
1. **Clear problem motivation and analysis**: The paper provides valuable insights into *why* 
   MeanFlow works by decomposing its objective into interpretable components. The gradient 
   analysis in Figure 2 effectively demonstrates the negative correlation between L_TFM and 
   L_TCc gradients.


2. **Well-motivated curriculum learning strategy**: The three-phase training approach (trajectory 
   flow matching pretraining → α-Flow transition → MeanFlow fine-tuning) is well-justified by 
   the observed gradient conflicts and is intuitive.


3. **Thorough ablation studies**: Table 2 and Table 5 provide detailed ablations on curriculum 
   schedule, flow matching ratio, training objectives, adaptive loss weights, consistency step 
   ratio, and more. This demonstrates systematic investigation.

### Weaknesses
**α-Flow is an incremental generalization**
- The core idea is straightforward: interpolate between L_TFM (α=1) and MeanFlow (α→0) via 
  curriculum learning.
- While Theorem 1 unifies existing methods, this is largely a mathematical observation rather 
  than a conceptual innovation.
- The curriculum learning strategy is intuitive but not fundamentally novel—using easier 
  objectives first is a well-established practice.

**Gradient analysis is empirical, not theoretical**
- Figure 2 shows empirical gradient correlations but provides no theoretical explanation for 
  *why* the gradients conflict.
- The paper hypothesizes: "L_TCc has a very large solution manifold compared to L_TFM's 
  narrow manifold" (Section 3.2) but provides no proof or rigorous analysis.
- No formal characterization of when/why conflicting gradients occur in optimization of 
  multiple objectives.

**Curriculum learning for generative models is not new**
- Curriculum strategies in diffusion/flow models have been explored:
  - Progressive distillation (Salimans & Ho, 2022) uses curriculum-like approaches
  - EDM (Karras et al., 2022) employs careful scheduling
  - Consistency model training already uses staged objectives
- The novelty of applying curriculum to MeanFlow is limited.

**Main contribution is improved MeanFlow, not a fundamental advance**
- α-Flow is essentially MeanFlow with curriculum learning + reduced flow matching ratio


**Theorem 1 proof is incomplete**
- The proof in Appendix D.2 shows α-Flow reduces to existing objectives at specific α values 
  but doesn't establish theoretical properties (convergence, optimality, bias-variance).
- For MeanFlow: proof shows ∇_θ L_MF(θ) = ∇_θ L_α→0(θ) as α→0, but this is just asymptotic 
  equivalence, not equivalence at finite α.
- Missing: formal analysis of how well finite-α approximates α=0.

**Manifold hypothesis (Section 3.2) is unsupported**
- Hypothesis: "L_TCc has a large solution manifold; L_TFM has a narrow manifold."
- No measurement of manifold size, rank, or structure.

**Notation could be clearer**
- L_TFM vs. L_FM′: distinction subtle and easy to confuse
- L_TCc notation for consistency loss is non-standard (why "Cc"?)
- Multiple uses of subscript/superscript make equations hard to parse


**Missing experimental details**
- How is "gradient similarity" computed exactly? (Appendix E brief on this)
- How many iterations used for averaging in Figure 2?

### Questions
1. **Why does curriculum learning specifically help?** Can you show that gradient conflict 
   decreases during α-annealing? Is it warm-starting or conflict reduction?

2. **How does α-Flow perform on other datasets and resolutions?** ImageNet-256 is the only 
   evaluation—generalization unclear.

3. **What if you train MeanFlow with just α-Flow's pretraining phase (L_TFM for 150K steps), 
   then fine-tune on L_MF?** Does curriculum truly outperform simple pre-training?

4. **Why is the manifold interpretation important?** The paper claims L_TCc has a large manifold, 
   but provides no measurement or proof. Does manifold size actually affect optimization?

5. **Can you provide theoretical guarantees for α-Flow?** Does curriculum learning provably 
   reduce optimization difficulty or bias?

6. **How does α-Flow perform with other guidance methods** (not just classifier-free guidance)? 
   Why does guidance cause instability?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper provides a detailed analysis of the MeanFlow objective for few-step generative modeling. Motivated by the observation that MeanFlow dedicates the majority of its training to border-case supervision (i.e., fitting to vanilla flow matching with $r=t$), the authors conduct in-depth investigation into the MeanFlow objective. Specifically, they show that the MeanFlow loss naturally decomposes into two components — trajectory flow matching ($L_{TFM}$) and trajectory consistency ($L_{TC_c}$) — whose gradients are strongly negatively correlated, leading to optimization conflicts. The authors further discover that such issue can be partially mitigated by introducing an auxiliary flow matching loss ($L_{FM'}$). 

Intended for a training recipe with higher efficiency, the authors propose $\alpha-$Flow, a unified framework that interpolates between flow matching and MeanFlow via a curriculum parameter $\alpha$. The framework includes several prior objectives (Flow Matching, Shortcut Model, Consistency Training) as special cases, and progressively transitions from $L_{TFM}$ to MeanFlow through an $\alpha-$schedule. This framework achieves improved performance on ImageNet $256\times256$ compared to various previous baselines under comparable compute budgets.

### Strengths
1. **Insightful empirical analysis**

The decomposition of the MeanFlow objective and the empirical analysis of gradient correlations in Section 3 provide a novel perspective on the training dynamics of the trajectory-based generative modeling. These analysis not only provide clear motivations for the proposal of $\alpha-$Flow, but also effectively deepens the understanding of optimization behavior in few-step generative models, which is timely and valuable given the growing research focus on this field.

2. **Comprehensive and clear ablation studies**

For each design choice introduced in the proposed $\alpha-$Flow framework, the authors conduct extensive ablation experiments accompanied by clear explanations that justify their settings. These analyses substantially improve the clarity of presentation, strengthen the empirical grounding of their claims, and enhance the reproducibility of the overall training recipe.

3. **Improved empirical results**

Across all comparable settings, $\alpha-$Flow demonstrate superior performance over multiple baselines, including MeanFlow, FACM, and Shortcut Models. The improvements are consistent across different model scales and sampling steps, underscoring the effectiveness and robustness of the proposed framework.
    
4. **Detailed limitations and failed experiment discussions**

It is commendable that the authors explicitly discuss both the limitations of their framework and the failed experimental attempts (Appendix B–C). Such transparency is rare and valuable: it not only helps future researchers avoid redundant directions but also provides practical insights into the challenges of optimizing few-step generative models at scale. This openness meaningfully contributes to the community’s collective understanding and progress.

### Weaknesses
1. **Stability of $L_{TC_c}$**

The proposed method does not directly address the well-known stability issue of the $L_{TC_c}$, which the authors also acknowledge in Appendix B. This instability has been repeatedly identified as the primary bottleneck in scaling consistency-based methods to large-scale applications such as text-to-image or text-to-video generation (Lu & Song, 2024; Chen et al., 2025; Zheng et al., 2025). As a result, while the paper offers valuable insights into the gradient conflict within the MeanFlow objective and presents improved training heuristics, the overall scope remains somewhat limited. That said, since the authors explicitly note that “our framework should not be viewed as a silver bullet for addressing instability,” it would nonetheless strengthen the paper to include an empirical comparison of training stability between α-Flow and vanilla MeanFlow (e.g., visualizations of training losses, gradient norms, or JVP magnitudes as $\alpha \to 0$).

2. **Comparisons with contemporaneous works**

In Section 4.1, the authors demonstrate that $\alpha-$Flow subsumes Consistency Training (CT) under certain network parameterizations. However, the paper does not include corresponding CT reproduction results in Table 1. Including such a baseline would substantiate the claim that $\alpha-$Flow not only unifies but also improves upon prior consistency-based methods.

Moreover, the proposed framework shares conceptual similarities with SplitMeanFlow (Guo et al., 2025). While the authors briefly mention this work in the appendix, a more direct discussion or empirical comparison—highlighting differences in optimization behavior, efficiency, or loss design—would enhance the paper’s soundness and contextual positioning within the recent literature.

3. **Supporting evidence of "$L_{TC_c}$ is easier to optimize"**

The paper claims that $L_{TC_c}$ has a “larger solution manifold” and is therefore easier to optimize, but this assertion appears counterintuitive, as the JVP term in consistency objectives are generally known to suffer from great instability during training. A more rigorous justification (or some empirical results) would make the argument more convincing and clarify the optimization dynamics of $L_{TC_c}$.

Lu, Cheng, and Yang Song. "Simplifying, stabilizing and scaling continuous-time consistency models." arXiv preprint arXiv:2410.11081 (2024).

Chen, Junsong, Shuchen Xue, Yuyang Zhao, Jincheng Yu, Sayak Paul, Junyu Chen, Han Cai, Song Han, and Enze Xie. "Sana-sprint: One-step diffusion with continuous-time consistency distillation." arXiv preprint arXiv:2503.09641 (2025).

Zheng, Kaiwen, Yuji Wang, Qianli Ma, Huayu Chen, Jintao Zhang, Yogesh Balaji, Jianfei Chen, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. "Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency." arXiv preprint arXiv:2510.08431 (2025).

Guo, Yi, Wei Wang, Zhihang Yuan, Rong Cao, Kuan Chen, Zhengyang Chen, Yuanyuan Huo et al. "Splitmeanflow: Interval splitting consistency in few-step generative modeling." arXiv preprint arXiv:2507.16884 (2025).

### Questions
1. **Distillation applicability**

Could the proposed $\alpha-$Flow framework be extended to a *distillation* setting, where the conditional velocity $v_t$ is replaced by a teacher model’s prediction? If so, would this substitution affect any design choices?

2. **Counterintuitive claim on $L_{TC_c}$'s optimization**

It is well known that the JVP term in the consistency objectives introduces strong training instability and is notoriously difficult to optimize. But in the paper, the authors argue that, without a boundary condition, $L_{TC_c}$ is easier to optimize given its unconstraint solution space. Could the authors elaborate on this seemingly contradiction? Is this claim mainly referring to that it's easy to optimize $L_{TC_c}$ to 0 but not to the solution space intersecting with $L_{TFM}$?

3. **Adaptive weighting strategy**

For $\alpha = 1$, the derived adaptive weighting is applied on the vanilla flow matching objective, which could potentially break the L2 loss required by the flow matching training recipe; yet, for $\alpha \to 0$, no adaptive weighting is not applied to the more challenging JVP-based objective. Although this empirically yields positive results, the design choice seems counterintuitive. Could the authors offer further intuition or justification for why this adaptive weighting behaves favorably in practice?

4. **Fine-tuning the model**

The trajectory flow matching phase in $\alpha-$Flow. effectively serves as a warm-up for stabilizing the model from random initialization. Have the authors experimented with starting from a pretrained FM model instead?

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
4

### Summary
The authors propose to train a MeanFlow model along a curriculum that transitions from flow matching to MeanFlow optimization. This curriculum is inspired by two observations: the MeanFlow objective can be decomposed into a flow matching loss and a consistency loss, and the two losses have conflicting gradients during training. By optimizing the flow matching loss (with a narrow solution space) first and then transitioning to the easier MeanFlow objective (with a larger solution space), the authors claim one can achieve better convergence. The curriculum is implemented by optimizing the $\alpha$-Flow objective that interpolates between flow matching (at $\alpha=1$) and MeanFlow (at $\alpha=0$). Various annealing strategies for $\alpha$ are explored in the paper, and $\alpha$-Flow in general outperforms vanilla MeanFlow.

### Strengths
- **[S1] This paper is novel, as it presents a new perspective on the slow convergence of MeanFlow.** By decomposing the MeanFlow objective into a sum of flow matching and consistency training losses and showing that the two have conflicting gradients, this paper elucidates a possible source of instability and slow convergence in MeanFlow training.

- **[S2] This paper is significant, as it provides a simple and practical recipe for stable training of MeanFlow.** By pinpointing a source of instability in MeanFlow training, the authors are able to develop a simple curriculum learning strategy for stabilizing convergence of MeanFlow. The authors provide strong evidence (on ImageNet 256$\times$256) that the curriculum indeed yields better MeanFlow models. In particular, experiments show one can use a lower flow matching ratio with $\alpha$-Flow, implying that the proposed curriculum effectively mitigates the tension between flow matching and consistency training.

### Weaknesses
- **[W1] Experiments in Section 5 only indirectly support the claim that $\alpha$-Flow mitigates the conflict between trajectory flow matching (TFM) and trajectory consistency (TC) objective.** An experiment such as an analogue of Figure 2 for $\alpha$-Flow training could directly support the hypothesis that the proposed curriculum resolves the tension between TFM and TC.

- **[W2] The paper is missing a simple but crucial baseline: training from a pre-trained diffusion model.** In particular, with diffusion models (DMs) trained via representation alignment offering much better generative performance, initialization from such DMs may be a simpler alternative to $\alpha$-Flow without the added complexity of tuning the alpha decay schedule.

### Questions
- **[Q1] Why is there a large gap between FIDs in Table 1 and Table 2?** Did the authors use smaller number of training epochs for ablations?

- **[Q2] How does $\alpha$-Flow training from scratch compare to MeanFlow training initialized from a pre-trained diffusion / flow model?** I am curious how MeanFlow initialized with e.g., REPA [1] or REPA-E [2] DM would perform compared to $\alpha$-Flow. I believe the embedding layer for the second time parameter can easily be implemented by copying embedding modules for the original time parameter in pre-trained DMs.

[1] Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think, ICLR, 2025

[2] REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers, 2025

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper analyzes the MeanFlow framework for few-step generative modeling and introduces a generalized family of objectives called $\alpha$-Flow. The authors show that the MeanFlow loss decomposes into two competing components—trajectory flow matching and trajectory consistency—whose gradients are strongly negatively correlated, leading to slow convergence. To address this, they propose $\alpha$-Flow, which unifies MeanFlow, Shortcut Models, and flow matching under a single formulation. By gradually annealing the training objective from trajectory flow matching to MeanFlow (a curriculum learning approach), $\alpha$-Flow disentangles the conflicting objectives, improving optimization efficiency. When trained from scratch on class-conditional ImageNet-1K with DiT backbones, $\alpha$-Flow achieves consistent gains over MeanFlow, reaching new state-of-the-art FID scores for 1- and 2-step generation using standard architectures.

### Strengths
- The paper introduces a comprehensive framework to train few-step generative models by unifying and streamlining existing approaches. This, combined with the experimental ablations, provides a clear recipe for practitioners.

- The experimental section is complete, and provides a lot of insights.

- Section C (Failed experiments) includes approaches tried by the authors that did not succeed. This is a great idea that should be more common.

### Weaknesses
- The authors fail to cite some related works like https://arxiv.org/abs/2406.07507

- The fact that the gradients of L_{TFM} and L_{TC_c} are so negatively correlated is not surprising since adding the two losses and completing the square yields a constant value at the optimal vector field. Meanwhile, no such relationship exists between L_{FM’} and L_{TC_c}. Hence, it is reasonable that the correlation between the gradients of these two losses fluctuates around zero. From this perspective, there is nothing mysterious about the empirical analysis in Section 2. In fact, the negative correlation must hold between the gradients of any decomposition of a least squares loss.

- In Section 4.1, it would be good to devote a paragraph to clarifying/proving how the MeanFlow loss is related to the \alpha-Flow loss. The connection is stated in Theorem 1, but this statement is not obvious at a first glance. While proven in the appendix, having a sketch in the main text would help the reader.

### Questions
The authors state that "the 75% ratio yields worse NFE=1 but better NFE=2 generation results compared to the 50%-ratio version”. Extrapolating this observation, I wonder whether the optimal ratio grows with the target inference NFE. Have the authors checked this? More generally, have the authors run evaluations on larger NFE?

### Soundness
4

### Presentation
3

### Contribution
4
