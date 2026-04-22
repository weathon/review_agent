# Ensembling Pruned Attention Heads For Uncertainty-Aware Efficient Transformers

- Avg Score: 6.80
- Decision: Accept (Poster)
- Scores: 8, 6, 10, 6, 4

## Abstract
Uncertainty quantification (UQ) is essential for deploying deep neural networks in safety-critical settings. Although methods like Deep Ensembles achieve strong UQ performance, their high computational and memory costs hinder scalability to large models. We introduce Hydra Ensembles, an efficient transformer-based ensemble that prunes attention heads to create diverse members and merges them via a new multi-head attention with grouped fully-connected layers. This yields a compact model with inference speed close to a single network, matching or surpassing Deep Ensembles in UQ performance without retraining from scratch. We also provide an in-depth analysis of pruning, showing that naive approaches can harm calibration, whereas Hydra Ensembles preserves robust uncertainty. Experiments on image and text classification tasks, with various architectures, show consistent gains over Deep Ensembles. Remarkably, in zero-shot classification on ImageNet-1k, our approach surpasses state of the art methods, even without requiring additional training.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work introduces a method for uncertainty estimation in transformers.
The key idea is producing an ensemble of models via pruning attention
heads - which leads to reduced computational complexity compared
to a full ensemble. Evaluation on image and text classification
benchmarks indicates close to state-of-the-art performance.

### Strengths
*Clarity:*
- The paper is well-written and is easy to follow.

*Significance / soundness:*
- This work addresses an important problem of epistemic uncertainty estimation in a widely used transformer architecture.
- Method itself and technical solutions seems well-motivated and simple to implement, and paper provides sufficient details for reproducing the results.
- The fact that this method works w/o retraining the base model is a key benefit of the proposed method - re-training large models is intractable.
- Authors provide multiple variants of the method suitable for different settings (with and w/o uncertainty val set, fine-tuning vs zero-shot).

*Evaluation:*
- Performance of the method is close to state-of-the-art, both on prediction and OOD tasks, while the cost is significantly lower, especially in the practical bfloat16 setting. This makes the method extremely useful in practice.

### Weaknesses
*Novelty:*
- MoE models seems to be a very similar approach to what is being proposed in this work. A naive baseline
could be re-using an existing MoE model for getting uncertainty estimates?

*Evaluation:*
- Authors claim that the results do not require additional re-training, but in practice this seems a bit misleading
because for both variants of the model (Taylor and Circ) either needs access to uncertainty validation set or
actually requires fine-tuning.
- Paper provides reasonable argument for pruning attention heads instead of MLP, but does not provide a
quantitative evaluation.

### Questions
- Have you considered re-using existing MoE-s for uncertainty estimation directly?
- Evaluation is conducted on prediction / OOD tasks. Would the method also work out-of-the-box for generation tasks, MoE-style?
- I wonder if authors have any intuition on why LoRA ensembles would work worse than fine-tuned pruning?

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
The paper addresses the challenge of uncertainty quantification (UQ) in large-scale transformer models. While Deep Ensembles are known to provide reliable uncertainty estimates, they are computationally expensive due to multiple independently trained models. The authors aim to retain ensemble-level calibration and robustness while significantly reducing computational and memory costs. Hydra Ensembles constructs an efficient transformer ensemble by pruning attention heads in a single pre-trained transformer to generate diverse subnetworks and merging these pruned models into a single architecture using Grouped Fully Connected (GFC) layers, forming a Fused Multi-Head Attention (MHA) and Merged MLP structure.

### Strengths
* The paper is well written and well motivated. 
* The method can be applied to different MoE architectures.
* It shows good improvements in OOD performance.

### Weaknesses
* The paper emphasizes ensemble diversity but does not quantify the resulting predictive diversity of the pruned members or the fused model against standard baselines. Appendix B.5 analyzes sources of diversity (e.g., seeds, batch order) but stops short of reporting diversity magnitude after pruning/fusion (e.g., disagreement rate), leaving it unclear whether Hydra is more or less diverse than alternatives.
* On image classification, in-distribution calibration (Brier, NLL) is roughly on par with a single model, suggesting the method’s gains are concentrated in OOD detection. So the only benefit of the model is for OOD detection, and it does not affect the method's robustness in IND, which is counterintuitive.

### Questions
* If each model differs only in the set of surviving heads (line 240), why do you need to average the weights and biases across the M models for the MLP layer (line 247)? Are these not the same?
* In Hydra Ensembles(circuit), the authors use the Headmap method (Wang et al., 2025) to identify which heads matter most for uncertainty, and remove the rest. What would be the impact of optimizing for a different task? Also, if you're always optimising for which heads matter most for uncertainty and removing the rest, how do you get different results across the M models?
* Given the strong performance of Taylor and CircuitAverage in the benchmark (on classification and zero-shot tasks), is it worth including them in the benchmark of inference time?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This article deals with uncertainty quantification (UQ) in transformer-based architectures. It observes that UQ is difficult to achieve, with the current state of the art being held by Deep Ensembles (DE), a contribution from 2017. Deep ensemble achieve sometime-excellent results on UQ at the expense of very expensive training and inference, since multiple training and inference with different parameters must be performed every time.

The article proposes to use pruning methods on the attention heads of transformer architectures only. They justify this approach by suggesting that such pruning is easier to control and that a combined network can then be put together, performing efficient prediction and UQ at once. This proposal is thoroughly justified on both theoretical and practical grounds, achieving close to the state of the art result at virtually no cost on BF16 precision due to the specialised hardware involved. When using FP32 precision, the computational gains are almost non-existent with respect to DE, however.

### Strengths
The article is relatively easy to read, very well justified with complete and detailed theoretical and practical explanations. The source code is promised to be made available upon acceptance.

### Weaknesses
The success of an ensemble heavily relies on the diversity of its component models. If all models make similar mistakes, combining them won't lead to much improvement. Strategies like varying training data, model architectures, or initialization are needed to ensure diversity

Adaptive Attacks: Sophisticated attackers can create adaptive adversarial examples that specifically aim to minimize the uncertainty metrics (like variance or entropy) of the ensemble, attempting to make their attack look like a confident, in-distribution prediction.

Machine-learning models can be fooled by adversarial examples, i.e., carefully-crafted input perturbations that force models to output wrong predictions. While uncertainty quantification has been recently proposed to detect adversarial inputs, under the assumption that such attacks exhibit a higher prediction uncertainty than pristine data, it has been shown that adaptive attacks specifically aimed at reducing also the uncertainty estimate can easily bypass this defense mechanism.

https://arxiv.org/abs/2309.10586

### Questions
- What would be the cost and gain of using 5 heads in BP16 instead of just 3?
- Can the source code be made available for review? Many contributors promise to publish code that turn out to be unreadable, uncommented or incomplete.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Hydra Ensembles: make a few differently pruned versions of the same transformer (different attention heads kept), fine-tune them, and then fuse them into a single model using fused MHA + grouped FC so you can get “ensemble-like” predictions at roughly single-model cost. It targets efficient uncertainty estimation for transformers.

### Strengths
1. The paper tackles efficient uncertainty estimation for large transformers, a problem people actually have. That makes the work naturally interesting.

2. Pruning-as-diversity is a nice angle. Most pruning papers try to preserve one model; here, pruning is used to induce differences between members. That’s a small but genuine conceptual twist.

3. The method is demonstrated on both vision (ViT) and language (BERT-style) models, plus a zero-shot setting, so it doesn’t look tied to one benchmark. That strengthens the claim of generality. 

4. The cost story is attractive. Claiming ~1× inference vs ~3× for Deep Ensembles directly supports the motivation; it’s exactly the comparison readers care about

### Weaknesses
1. The whole method relies on “different pruned heads = different members,” but the paper doesn’t show simple diversity metrics (e.g. pairwise disagreement/KL) before and after fusion. For this idea, that’s the key missing evidence.

2. Several main tables give single numbers but no std / CI / ± over seeds, even though the paper is about uncertainty/robustness.

3. On SST2, the ensemble seems cheaper than a real, fully fine-tuned deep ensemble, which makes Hydra look better than it might against a “full” baseline. 

4. The attention fusion story is clear; the MLP part is basically “we average/group.” Since members were pruned and fine-tuned separately, that choice could wash out diversity; an ablation or a justification can be beneficial.

5. The small theoretical part about pruning hurting more under noisy/OOD inputs is more motivational than general; assumptions aren’t clearly checked on ViT/BERT. Either support it empirically in the main text or de-emphasize it.

6. Because zero-shot UQ is sensitive to prompts/datasets/temperature, more details should be in the main paper, not just the appendix.

### Questions
1. When you fuse the pruned members into one model, do you still get separate member outputs so we can measure disagreement, or is it combined into one prediction? A small clarification (and maybe a number) would help.

2. If different members keep different attention heads, does the fused layer run the union of those heads (which could increase compute), or do you share some heads to stay close to 1×?

3. You average / group MLP weights across members. Did you try an alternative (e.g. a small per-member adapter) and it didn’t help, or was this mainly for simplicity?

4. How sensitive is Hydra to how aggressively you prune? A short plot or table for one dataset would clarify how robust the method is.

5. For the different Hydra members, how do you ensure that the pruned head sets are actually different (i.e., not largely overlapping)? Do you use different random seeds for the pruning score, or do you enforce low overlap between members?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Hydra Ensembles, a novel framework designed to achieve uncertainty-aware efficient ensembling for large-scale transformer models such as ViT, BERT, and CLIP. The key motivation is to retain the uncertainty quantification (UQ) robustness of Deep Ensembles while drastically reducing their computational and memory costs.

Hydra Ensembles works by pruning attention heads from a single pre-trained transformer to create diverse submodels, which are then fused into a single network via a Grouped Fully Connected (GFC) fusion of their Multi-Head Attention (MHA) and MLP layers. Unlike conventional ensemble approaches, Hydra Ensembles:
	•	avoids retraining each model from scratch,
	•	allows near-single-model inference cost, and
	•	preserves ensemble diversity for robust uncertainty estimation.

The paper’s contributions are threefold:
	1.	Theoretical Analysis: Demonstrates that naïve pruning can degrade model calibration under noisy data conditions, supported by a formal proposition showing loss gap widening in pruned models.
	2.	Framework Design: Introduces structured head-level pruning and GFC-based fusion that maintain model diversity while minimizing computation.
	3.	Empirical Evaluation: Provides extensive experiments on image classification (ImageNet-1K, CIFAR-100), text classification (SST-2), and zero-shot image classification (OpenCLIP). Results show that Hydra Ensembles achieve comparable or superior uncertainty metrics (AUROC, AUPR, ECE) to Deep Ensembles while being ~3× faster and requiring significantly fewer parameters.

Overall, Hydra Ensembles present a theoretically grounded and practically efficient solution for scalable uncertainty quantification in transformer architectures — bridging the gap between computational efficiency and epistemic robustness.

### Strengths
Originality
	•	The paper presents a novel architectural strategy for ensemble diversity by pruning and recombining transformer attention heads rather than training separate models.
	•	The Grouped Fully Connected (GFC) fusion mechanism introduces an efficient ensembling pipeline that avoids the typical cost explosion of Deep Ensembles.
	•	Although it builds on concepts like BatchEnsemble and pruning-based model compression, Hydra’s hybridization of pruning and ensembling is unique in both motivation and execution.
	•	The formal proposition linking random pruning to calibration degradation is a valuable theoretical insight, providing justification for structured pruning instead of purely empirical reasoning.

Quality
	•	The methodology is sound and reproducible, with strong empirical results across both vision (ViT, ImageNet-1K, CIFAR-100) and text (BERT, SST-2) domains.
	•	Extensive ablations (pruning ratios, ensemble sizes, GFC variants) demonstrate that the performance improvements are not cherry-picked but systematic.
	•	The authors carefully balance theoretical analysis, algorithmic description, and experimental validation — showing maturity in both design and evaluation.
	•	Calibration metrics (ECE, AUROC, AUPR) are properly selected for uncertainty quantification, and their consistent improvement validates the main claim.

Clarity
	•	The paper is well-organized and readable, with clear section flow and minimal redundancy.
	•	Figures such as the Hydra architecture schematic and calibration plots effectively support understanding.
	•	The theoretical analysis is concise and mathematically consistent, though dense in presentation — yet the accompanying intuition keeps it accessible to non-specialists.
	•	Notation is consistent, and references to related work are appropriate and fair.

Significance
	•	Practical significance: Hydra Ensembles deliver ensemble-level uncertainty quality at a fraction of the computational cost, which is crucial for large transformer models where full ensembles are infeasible.
	•	Research significance: The framework provides a blueprint for uncertainty-aware model compression, bridging a long-standing gap between trustworthiness and efficiency.
	•	Community impact: The method is relevant to multiple research threads at ICLR — efficient transformers, uncertainty quantification, and scalable AI reliability.
	•	Its applicability to both vision and language domains broadens its potential adoption and demonstrates methodological generality.

### Weaknesses
1. Limited Theoretical Grounding of Calibration Claims
	•	The proposed theoretical proposition—that random pruning leads to calibration degradation—is intuitively plausible but lacks rigorous derivation or empirical verification linking the theory to measured uncertainty metrics (ECE, NLL, Brier score).
	•	The argument relies primarily on Fisher Information heuristics rather than a formal probabilistic treatment of epistemic variance or diversity loss.
	•	This makes the theoretical part informative but incomplete, as it doesn’t generalize to nonlinear pruning effects or the stochastic dynamics of self-attention.

Recommendation:
The authors could expand this section by (a) connecting pruning-induced variance loss to epistemic uncertainty via bias–variance decomposition, or (b) empirically validating the proposition using entropy/Fisher metrics before and after pruning.

2. Moderate Conceptual Novelty
	•	Hydra Ensembles’ innovation lies mainly in the engineering combination of known techniques: pruning for efficiency (Michel et al., 2019; Voita et al., 2019) and efficient ensemble fusion (Wen et al., 2020; Havasi et al., 2021).
	•	While the integration is elegant, it does not introduce a fundamentally new learning principle or uncertainty formulation.
	•	Related works like BatchEnsemble (Wen et al., ICLR 2020), MIMO (Havasi et al., NeurIPS 2021), and LayerDrop (Fan et al., ICLR 2020) already explore efficiency–diversity trade-offs; the paper could better clarify where Hydra theoretically diverges from these beyond implementation detail.

Recommendation:
Reframe Hydra as a structured synthesis approach rather than a conceptual breakthrough, emphasizing its engineering elegance and scalability benefits.

3. Scope of Experiments
	•	All evaluations are confined to classification tasks (CIFAR-100, ImageNet-1K, SST-2). No tests on generation or multimodal transformers (e.g., CLIP zero-shot) beyond classification accuracy and calibration.
	•	Without broader task validation, it’s unclear whether Hydra’s uncertainty improvements generalize to tasks requiring sequence modeling, open-ended text generation, or multi-modal reasoning—key frontiers of transformer research.

Recommendation:
Include or discuss pilot results on transformer-based generative tasks (e.g., GPT-style models) or multimodal settings to support claims of generality.

4. Insufficient Discussion of Trade-offs
	•	The paper highlights efficiency gains (3× faster inference, fewer parameters) but lacks quantitative trade-off analysis between pruning ratio, uncertainty calibration, and ensemble diversity.
	•	The reader is left without a clear sense of how much diversity is sacrificed at higher pruning rates or how inference cost scales with ensemble size.
	•	Additionally, calibration–efficiency curves or Pareto plots would strengthen interpretability.

Recommendation:
Provide explicit trade-off visualizations (e.g., efficiency vs. ECE or AUROC) and discuss how practitioners can tune pruning levels for optimal performance.

5. Missing Analysis of Diversity and Correlation Among Pruned Submodels
	•	Since ensemble robustness depends on model diversity, the paper should quantify inter-head diversity or correlation (e.g., using cosine similarity, pairwise prediction disagreement, or mutual information across pruned models).
	•	Without such analysis, it’s difficult to confirm whether Hydra truly achieves “diverse ensembling” or merely benefits from redundancy.

Recommendation:
Add an empirical diversity analysis to support the central claim of maintaining epistemic diversity post-pruning.

6. Limited Robustness and OOD Testing
	•	Although Hydra improves calibration, there are no results under domain shift or corrupted data (e.g., ImageNet-C, CIFAR-C, SST-2 perturbations).
	•	This omission weakens claims about robustness and “uncertainty-awareness,” since true epistemic reliability is best evaluated under distributional drift.

Recommendation:
Include robustness experiments on corrupted or OOD benchmarks to demonstrate Hydra’s behavior under uncertainty-inducing conditions.

7. Presentation and Comparison Clarity
	•	Some mathematical notation is dense and occasionally inconsistent across sections (e.g., subscripts for attention heads and fusion layers).
	•	Related work could be contextualized more critically — especially contrasting Hydra’s scalability and calibration against post-hoc methods like temperature scaling, EMM, or confidence regularization.

Recommendation:
Add a summary table contrasting Hydra with BatchEnsemble, MIMO, and FastGeLU Ensembles, highlighting distinct features, efficiency, and calibration properties.

### Questions
1. On the Theoretical Proposition and Calibration Justification
	•	The paper presents a theoretical result suggesting that random pruning widens calibration loss gaps.
	•	Could the authors expand or clarify the assumptions underlying this result — e.g., is the bound derived under linearized model assumptions, or does it generalize to nonlinear self-attention layers?
	•	How does this proposition directly link to empirical calibration metrics (ECE, NLL, AUROC)?
	•	Would it be possible to empirically measure information loss or variance reduction (e.g., via Fisher information or entropy) before and after pruning to validate the theory?

2. On Model Diversity and Ensemble Behavior
	•	Hydra’s design implies that pruning attention heads creates diverse submodels whose combination improves uncertainty calibration.
	•	How do the authors quantify or measure diversity among these pruned heads or submodels?
	•	Have they examined pairwise output correlations or disagreement rates across ensemble members?
	•	Without such evidence, how can we be confident that Hydra’s calibration improvements arise from true epistemic diversity rather than simple regularization effects?

3. On Efficiency–Uncertainty Trade-offs
	•	The paper reports efficiency gains (~3× faster inference) while maintaining or improving uncertainty metrics.
	•	Could the authors provide explicit quantitative trade-off curves between pruning ratio, uncertainty calibration, and accuracy?
	•	For example, what is the marginal drop in accuracy or AUROC per additional pruning step?
	•	This would help practitioners decide optimal pruning thresholds under different compute constraints.

4. On Generalization Beyond Classification Tasks
	•	Hydra Ensembles are evaluated on image and text classification tasks, but transformers are widely used for generation and multimodal learning.
	•	Have the authors explored whether Hydra can be applied to sequence generation tasks (e.g., autoregressive decoding, summarization) or vision–language models like CLIP or BLIP?
	•	If not yet tested, do the authors foresee architectural or stability challenges (e.g., head dependencies in causal self-attention) that might limit its application?

5. On the Fusion Mechanism (Grouped Fully Connected Layers)
	•	The Grouped Fully Connected (GFC) fusion layer is a central component, but its mathematical and practical behavior could be clarified.
	•	How does GFC differ from existing ensemble fusion or parameter-sharing mechanisms (e.g., BatchEnsemble, SplitDense)?
	•	Is there a risk of overfitting or co-adaptation when merging diverse heads via GFC?
	•	How is group size or fusion granularity chosen, and how sensitive is Hydra’s performance to these hyperparameters?

6. On the Scope of Uncertainty Metrics
	•	The experiments primarily report ECE and AUROC, which measure calibration and discrimination, respectively.
	•	Have the authors evaluated additional uncertainty metrics such as Brier score, NLL, or predictive entropy?
	•	Different metrics capture complementary aspects of uncertainty; including them could strengthen claims about “uncertainty-awareness.”

7. On Robustness and Distribution Shift
	•	Hydra’s motivation includes improving reliability and robustness through structured diversity.
	•	Have the authors tested Hydra on corrupted or domain-shift datasets (e.g., CIFAR-C, ImageNet-C, SST-2 with noise)?
	•	If not, could they provide a theoretical argument or empirical proxy suggesting Hydra’s robustness benefits beyond clean test distributions?

8. On Implementation Complexity and Reproducibility
	•	Hydra involves structured pruning, submodel fusion, and fine-tuning stages.
	•	Could the authors comment on the implementation complexity and reproducibility — e.g., how many lines of modification are required for ViT or BERT baselines?
	•	Are pretrained Hydra checkpoints or open-source scripts available (or planned) to facilitate community adoption?

9. On Relation to Other Efficient Ensembles
	•	Hydra’s conceptual overlap with BatchEnsemble (Wen et al., 2020) and MIMO (Havasi et al., 2021) is acknowledged but not deeply dissected.
	•	Could the authors articulate the precise difference in diversity mechanism between Hydra and these methods?
	•	Specifically, how does Hydra’s pruning-induced diversity compare empirically to BatchEnsemble’s multiplicative rank-1 reparameterization or MIMO’s input-sharing scheme?

10. On Interpretability of Pruned Attention Heads
	•	Since Hydra modifies attention structure, there may be implications for interpretability (e.g., loss of certain attention patterns or semantics).
	•	Have the authors analyzed whether the pruned heads correspond to interpretable functions (e.g., positional, syntactic, or semantic attention)?
	•	If not, do they expect pruning to impact model explainability — and could this trade-off affect trustworthiness in downstream deployment?

### Soundness
3

### Presentation
3

### Contribution
3
