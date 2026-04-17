---
job_id: 90bce768-cf1f-4015-875e-52ed06061769
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: rBj2iVyrhh.pdf
paper: Classifier-Constrained Alternating Training: Mitigating Modality Imbalance in Multimodal Learning
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper addresses modality imbalance in multimodal representation learning, proposes a new training framework, and evaluates it on standard multimodal benchmarks, which fits squarely within ICLR’s core topics.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Method, Experiments, Results/Analysis, Conclusion) are present and written in English. The methodology is nontrivial, experiments are reasonably thorough with multiple baselines and datasets, and there are no obvious fatal theoretical or empirical flaws, though there are weaknesses discussed below.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any hidden instructions, prompt-injection attempts, or non-scientific content aimed at influencing automated reviewers.

---

# Expected Review Outcome:

## Summary

The paper tackles modality imbalance in multimodal learning, focusing on the phenomenon that even with alternating optimization of encoders, a shared classifier can become biased toward faster-converging modalities. The authors propose Classifier-Constrained Alternating Training (CCAT), a two-stage framework that first pretrains a shared classifier with bidirectional cross-attention plus a modality-contribution regularizer, then freezes this classifier and trains modality-specific encoders with LoRA adapters under alternating optimization, together with sample-level secondary updates for highly imbalanced instances. Experiments on three benchmarks (CREMA-D, Kinetic-Sound, MVSA) show consistent accuracy improvements over several strong baselines, including MLA, MMPareto, and LFM.

## Strengths

1. **Clear problem focus: classifier bias under modality imbalance.**  
   The paper articulates a specific failure mode of existing alternating multimodal training methods: even if encoder interference is reduced, a shared classifier can become structurally biased toward strong modalities due to early convergence differences. The empirical trajectory of modality contributions in **Figure 1** nicely visualizes this issue, showing MLA’s contributions remaining imbalanced over epochs while CCAT equalizes them, supporting the central motivation.

2. **Conceptually coherent two-stage framework.**  
   The overall CCAT design is coherent: (i) pretrain a classifier on fused features using bidirectional cross-attention (detailed in **Figure 2** and Appendix A.1) with a mutual-information-based regularizer on modality contributions (**Equations (5)–(8)**), then (ii) freeze this classifier and add per-modality LoRA heads (**Figure 3(c)**, **Equation (9)**) to adapt unimodal features during alternating training plus sample-level secondary updates (Algorithm 1). The idea of decoupling decision boundary learning and modality-specific adaptation is sound and reasonably well justified.

3. **Empirical gains over strong baselines.**  
   On all three datasets, CCAT is competitive or best in almost all multimodal settings in **Table 1**. For example, CCAT improves over MLA by +5.11% on Kinetic-Sound (79.29 vs. 71.35 Multi), and over MMPareto by +1.92% on MVSA (80.73 vs. 78.81 Multi). These baselines include both classical fusion (Sum, Concat, FiLM, BiGated) and recent imbalance-aware methods (OGM-GE, QMF, MLA, MMPareto, LFM), so the comparison set is reasonably strong.

4. **Component-wise ablations that match the narrative.**  
   The ablations in **Table 2** systematically remove each component: classifier freezing, alternating training, secondary updates, and LoRA. Each removal degrades performance in line with the claimed roles. Notably, removing classifier freezing (“X” under Fix) reduces Multi accuracy on CREMA-D from 85.89 to 82.80 and on KS from 79.29 to 77.26, supporting the claim that freezing the classifier plays a central role. Similarly, removing secondary updates slightly but consistently hurts performance, suggesting some benefit from sample-level re-optimization.

5. **Some analysis of representation quality.**  
   The t-SNE visualizations in **Figure 5** (panels (a)-(c)) show more compact and separated clusters for CCAT compared to MLA and the non-fixed classifier, quantified by higher CH and SH scores and lower DB. This is not conclusive proof but provides additional evidence that the fixed-classifier design may improve discriminative structure, especially highlighting improved separation for the “fear” and “sad” classes which are often harder.

6. **Reasonable hyperparameter sensitivity study.**  
   The authors conduct grid searches over LoRA rank \(r\) (**Table 3**) and threshold \(\beta\) (**Figure 4**). While not exhaustive, this at least demonstrates that performance is not hypersensitive to these hyperparameters and gives a sense of practical tuning ranges.

## Weaknesses

1. **Positioning vs. very closely related alternating / imbalance work is incomplete.**  
   The paper mainly compares to MLA and MMPareto but omits several highly relevant recent works that tackle modality imbalance via *alternating* updates or explicit contribution balancing. For example, AMST (alternating skip training), minor-modality-aware adaptive alternating schemes, and mutual-information-based balancing strategies are not cited or experimentally compared, even though CCAT’s motivation and mechanism are tightly related. This weakens the originality and significance claims: from the paper as written, it is not clear whether constraining the classifier via pretraining + freezing is a qualitatively new direction or just one design choice among several existing alternating strategies. This should be explicitly addressed in both Related Work and experiments, not just briefly in text.

2. **Theoretical “link” between class and modality imbalance is mostly heuristic and partially sloppy.**  
   Section 3.1 aims to “establish a unified theoretical framework” between class and modality imbalance using gradient expressions. However:  
   - **Equation (2)**, \(\partial \mathcal{L}/\partial \mathbf{w}_j \approx -\mathbf{f}\) for minority classes, is asserted without conditions on \(\hat{y}_j\), and it ignores the magnitude of \((\hat{y}_j - 1)\) vs. \(\mathbf{f}\). The statement that “parameter updates become dominated by feature norm \(\mathbf{f}\)” is qualitative and not rigorously tied to imbalance.  
   - In **Equation (3)**, the fusion is modeled as \(\mathbf{f} = \gamma_1 \mathbf{f}^{(1)} + \gamma_2 \mathbf{f}^{(2)}\), but \(\gamma_m\) are treated as “implicitly learned modality utilization coefficients” without a clear definition in terms of network parameters or training dynamics. The subsequent approximation that the gradient is proportional only to \(\gamma_1 \mathbf{f}^{(1)}\) when \(\gamma_1 \gg \gamma_2\) presumes linearity and ignores the classifier’s nonlinearity (softmax and cross-entropy).  
   - The “isomorphism” claim is not formal; no conditions are given under which class imbalance and modality imbalance would lead to similar optimization fixed points or error profiles. It is essentially an analogy, yet the text phrases it as a “unified theoretical framework,” which is overstated.  
   Overall, this section is more of an intuitive argument than a rigorous analysis, but it is presented as a central theoretical contribution.

3. **Ambiguities and minor issues in the loss and contribution math.**  
   There are several issues that make the method less precisely specified:  
   - In **Equation (5)**, the mutual information estimator follows [Zhou et al., 2025b], but the expectation \(\mathbb{E}_\mathcal{D}\) and the denominator \(\sum_i \exp\langle\bar{\mathbf{f}}_i,\bar{\mathbf{z}}_i^m\rangle\) suggest a per-batch InfoNCE-style objective. It is not clear over which index the sum runs (current batch vs. whole dataset) and how this is computed in practice. Explicitly, is \(i\) in the denominator over the mini-batch, and is the “log N” term using batch size or dataset size? This is important for reproducibility.  
   - **Equation (6)** uses Softmax over MI values; this makes sense, but subsequently **Equation (7)** defines  
     \[
     \mathcal{L}_{\text{reg}} = \frac{\text{i}}{N} \sum_{i=1}^{N} | c_i^1 - c_i^2 |,
     \]
     which appears to contain a typo (“i” instead of 1) in the normalization factor, and more importantly it regularizes the *absolute difference* of contributions. This pushes contributions toward equality for *all* samples, regardless of whether true information content is imbalanced. This raises a conceptual concern: in genuinely uni-modal-dominant examples, forcing equal contributions may actually hurt performance, but the paper does not discuss this trade-off or show any analysis of how \(\mathcal{L}_{\text{reg}}\) behaves for such samples.  
   - Algorithm 1 refers to computing contributions via **Equation (6)** during the second stage, but the text notes that the computation of \(c\) is different from the cross-attention fusion and instead based on decision-level fusion during inference. This is conceptually fine, but the actual formula for this version of \(c_i^m\) is not written explicitly, which makes it hard to fully understand or reimplement the sample-level imbalance detection.

4. **Assumptions around frozen classifier and LoRA are under-examined.**  
   The central design choice is to pretrain and then *freeze* a classifier trained on fused multi-modal representations, while only allowing modality-specific LoRA heads to adapt the unimodal features in the second stage (**Equations (9)–(11)**, **Figure 3(c)**). However, the paper only briefly acknowledges the distribution shift \(P(z^m | y) \neq P(f | y)\) and then asserts that low-rank residuals can handle the mismatch. There is no analysis of:  
   - When freezing the classifier might be harmful, for instance when unimodal decision boundaries are significantly different from multimodal ones.  
   - How sensitive performance is to the capacity of LoRA (beyond the shallow grid search in **Table 3**), or whether full fine-tuning of the classifier in the second stage could perform comparably or better if accompanied by proper constraints.  
   - Whether similar benefits could be achieved via other constrained adaptation schemes (e.g., weight decay toward the pretrained classifier or a trust-region restriction), which would position freezing as one design choice among many, rather than the only viable fix.  
   Without such analysis, the methodological justification feels somewhat ad hoc.

5. **Experimental scope and analysis are limited given the ambitious claims.**  
   The paper frames itself as providing a “new theoretical framework” and a “systematic” solution to modality imbalance, but the experimental evaluation is limited to three relatively modest-scale supervised benchmarks (CREMA-D, KS, MVSA), all with 2 modalities and fairly standard backbones (ResNet-18/50, BERT). There is:  
   - No evaluation on stronger contemporary multimodal backbones (e.g., audiovisual transformers, CLIP-like text-image models), where optimization dynamics and classifier behavior may differ.  
   - No experiments on more challenging or large-scale datasets, or on settings with three or more modalities (even though future work mentions this).  
   - Very little analysis of computational cost: secondary updates re-run encoders and LoRA on subsets of samples each iteration; there is no measurement of training-time overhead compared to MLA or MMPareto.  
   These limitations do not invalidate the results, but they substantially narrow the scope of claimed generality.

6. **Effect of individual components on modality contributions is not deeply probed.**  
   While **Table 2** shows ablation on accuracy, there is no corresponding analysis of how each component affects *modality contribution trajectories* like those shown in **Figure 1**. For instance, does classifier freezing alone substantially flatten contribution imbalance, or is the effect mainly due to the MI regularizer in pretraining, or the secondary updates? Similarly, how does the contribution distribution change across samples before and after secondary re-training? Without such plots or statistics, the connection between the measured contribution imbalance and each part of CCAT remains somewhat speculative.

7. **Sample-level imbalance detection mechanism is under-specified.**  
   In Algorithm 1 lines 10–14, the method constructs \(\mathcal{B}_m^{\text{extreme}} = \{ x_i^m \mid c_i^m < \beta \}\) and then performs secondary updates using **Equation (12)**. However, key details are missing:  
   - How is \(c_i^m\) computed in the second stage when only unimodal predictions are available? Is it based on mutual information with fused features via a fresh BiCross module, or on logit magnitudes, or on some function of the fixed classifier’s outputs? The text hints at “decision-level fusion” but never gives an explicit formula.  
   - Does the threshold \(\beta\) depend on the modality or class distribution, and is there any analysis of how many samples typically fall below this threshold across epochs (**Figure 4** only shows overall accuracy vs. \(\beta\))?  
   - There is no discussion of potential instability: repeatedly over-updating “weak” samples might lead to overfitting or spurious emphasis on noisy examples.  
   This makes the sample-level mechanism feel more heuristic than carefully engineered.

8. **Missing related work on mutual-information-based balancing and alternating schemes (methodological positioning).**  
   The method uses mutual information for contribution estimation in **Equation (5)** and claims novelty in bridging modality and class imbalance. However, there is already work on mutual-information-based balanced multimodal learning and dynamic alternating schemes that treat minor modalities preferentially. These works should be discussed in **Section 2** and, where feasible, compared empirically. Their absence undermines the claim that CCAT is a notably new direction.

9. **Minor clarity issues and typos.**  
   There are some minor problems that collectively hurt clarity: the typo in **Equation (7)** (“i/N”), inconsistent notation in Appendix A.1 (e.g., **Equation (19)** uses \(\mathbf{z}_i^1\) instead of \(\mathbf{z}_i^2\) for the visual branch, which is likely an error), and a few places where references are misordered (two different “Zhang et al. (2024)” entries, “Zhang et al. (2025b)” referenced as “Zhou et al. (2025b)” in the main text). These are not fatal, but for a paper that leans on mathematical and algorithmic details, precision matters.

Given these issues, I see the work as a reasonably strong engineering idea with empirical promise, but not yet at the level of a clear, well-situated, and rigorously analyzed contribution that ICLR typically expects for acceptance.

## Potentially Missing Related Work

Below are directly related works that appear missing and should be discussed; they are especially relevant to the paper’s focus on alternating multimodal training and balancing modality contributions:

1. **Henriques e Silva et al., “AMST: Alternating Multimodal Skip Training,” 2025/2026.**  
   - Relevance: Proposes a training method that adjusts per-modality training frequency to address modality dominance, closely aligned with CCAT’s alternating training plus classifier constraints.  
   - Suggestion: Discuss in **Section 2 (Related Work)** as part of alternating multimodal optimization and clarify how CCAT’s fixed classifier + LoRA differs in mechanism and behavior. Where feasible, include AMST as a baseline in **Table 1** for CREMA-D or KS.

2. **Shi et al., “Modality Equilibrium Matters: Minor-Modality-Aware Adaptive Alternating for Cross-Modal Memory Enhancement,” 2025.**  
   - Relevance: Introduces minor-modality-aware adaptive alternating training to balance modalities; conceptually very close to the paper’s goal of strengthening weak modalities.  
   - Suggestion: Add to **Section 2** and explicitly contrast CCAT’s classifier-centric approach with their minor-modality-aware scheduling; a discussion after **Section 3.3** could clarify whether CCAT could benefit from adaptive modality ordering.

3. **Jiang et al., “Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion,” 2025.**  
   - Relevance: Directly addresses disproportion in classification ability across modalities, conceptually similar to classifier bias.  
   - Suggestion: Cite when motivating classifier imbalance in **Section 1** and compare methodological choices in **Section 3.1–3.2**, particularly around how classification ability is measured and rebalanced.

4. **Xie & Sanguinetti, “Balanced Multimodal Learning via Mutual Information,” 2025.**  
   - Relevance: Uses mutual information to quantify and balance multimodal interactions; highly relevant since CCAT’s contribution regularizer also relies on an MI estimator (**Equation (5)**).  
   - Suggestion: Discuss around **Equations (5)-(8)** and clarify similarities and differences in how MI is estimated and used, and whether CCAT could adopt or benefit from their MI formulations.

5. **Wang et al., “Balanced Multimodal Learning: An Unidirectional Dynamic Interaction Perspective,” 2025.**  
   - Relevance: Proposes a sequential training interaction scheme to avoid single-modality domination, similar in spirit to CCAT’s alternating modality-wise updates.  
   - Suggestion: Position CCAT relative to this unidirectional interaction perspective in **Section 2** and potentially in **Section 3.3**, emphasizing how classifier freezing and LoRA compare to their dynamic interaction strategy.

6. **Wu et al., “Mitigating Modal Imbalance in Multimodal Reasoning,” 2025.**  
   - Relevance: Studies modality imbalance in multimodal reasoning and its effect on cross-modal conflicts, which aligns with the broader problem setting of CCAT.  
   - Suggestion: Mention in **Related Work** as complementary analysis of modal imbalance on more complex reasoning tasks; could help motivate future work to test CCAT beyond supervised classification.

7. **Ma et al., “Improving Multimodal Learning Balance and Sufficiency through Data Remixing,” 2025.**  
   - Relevance: Tackles modality imbalance via data-level techniques (decoupling and remixing), which provides a different axis of intervention than CCAT’s model-level strategy.  
   - Suggestion: Include in **Section 2** as a data-centric approach and briefly discuss how CCAT could be combined with or compared to such remixing methods.

## Questions

1. **Definition and computation of modality contributions in stage 2.**  
   You mention that contribution scores \(c_i^m\) in the second stage are computed using “the same decision-level fusion used in the inference stage” (bottom of **Page 6**, after **Equation (11)**), not the BiCross fusion from stage 1. Could you provide the exact formula for this version of \(c_i^m\)? Is it based on MI over logits, on prediction entropy, or something else?

2. **Behavior of the contribution regularizer on genuinely uni-modal examples.**  
   **Equation (7)** penalizes \(|c_i^1 - c_i^2|\) for all samples. How does this affect samples where only a single modality is actually informative (e.g., noisy or missing audio in CREMA-D, or nearly text-only informative MVSA posts)? Have you inspected per-sample contributions to see whether the regularizer forces equalization even when that is inappropriate, and if so, how does that impact performance?

3. **Can full fine-tuning of the classifier (with constraints) match CCAT?**  
   A strong alternative baseline would be: (i) pretrain the classifier with cross-attention and MI regularization as in your first stage, (ii) instead of freezing it, fine-tune it jointly with encoders during alternating training but with heavy weight decay or a proximal term \(\|\mathbf{W} - \mathbf{W}_0\|^2\) toward the pretrained weights. Have you tried such a baseline or variants? If not, can you comment on why you believe full freezing plus LoRA is preferable in terms of optimization dynamics?

4. **Sample-level secondary updates: scale and stability.**  
   For typical settings of \(\beta\) in **Figure 4** (e.g., 0.15 on CREMA-D and 0.30 on KS), what fraction of samples per batch are selected into \(\mathcal{B}_m^{\text{extreme}}\), and how does this change over epochs? Do you observe any overfitting or instabilities caused by repeatedly upweighting the same “weak” samples?

5. **Computational overhead of CCAT vs. MLA / MMPareto.**  
   Can you quantify training time or FLOP overhead introduced by (i) classifier pretraining with BiCross attention, (ii) LoRA modules, and (iii) secondary updates relative to, say, MLA or MMPareto on CREMA-D and KS? This would help practitioners decide whether the gains in **Table 1** justify the additional complexity.

6. **Extension to more modalities and larger backbones.**  
   You mention future work on trimodal datasets. Conceptually, how would you extend the MI-based contribution estimation (currently in **Equation (6)** for 2 modalities) and the classifier pretraining with BiCross to 3+ modalities? Also, do you anticipate any challenges applying CCAT to transformer-based multimodal encoders (e.g., AV former or CLIP-style models)?

Clear answers and, if possible, small additional experiments or ablations addressing these questions could significantly strengthen the paper.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A. The work uses standard public datasets (CREMA-D, Kinetic-Sound, MVSA) and standard supervised classification setups; no direct ethics, privacy, or safety issues are evident from the text.

## Soundness Rating

2: fair.  
The method is mostly coherent and empirically validated with sensible baselines, but key mathematical definitions (MI estimator, contribution computation, regularizer behavior) and the theoretical claims around class vs. modality imbalance are under-specified or overstated, and several design choices (frozen classifier, secondary updates) are not deeply justified or compared to strong alternatives.

## Presentation Rating

3: good.  
The paper is generally readable, with clear figures (especially **Figures 1–3, 5**) and tables (**Tables 1–3**), but suffers from some notational inconsistencies, minor typos, and insufficient detail around certain core mechanisms, particularly in the computation of contribution scores and the behavior of the regularizer.

## Contribution Rating

2: fair.  
The idea of constraining a shared classifier via pretraining and freezing, plus modality-specific LoRA adapters and sample-level secondary updates, is interesting and yields performance gains on several benchmarks, but the conceptual novelty over existing alternating / imbalance-aware approaches is not convincingly demonstrated, especially given missing related work and limited experimental scope.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The work addresses an important problem and shows consistent empirical improvements with a fairly clean architectural idea; however, theoretical claims are oversold, key mechanisms are under-specified, and the positioning versus very closely related recent work is incomplete. With stronger comparisons, clearer math, and deeper analysis of when and why classifier freezing + LoRA works, this could become a solid contribution, but in its current form it falls slightly short of ICLR standards.

## Reviewer Confidence

4: confident.  
I am familiar with multimodal learning and modality imbalance literature and have carefully checked the equations, algorithm, and experimental design at a detailed level, though I have not attempted to reimplement the method.