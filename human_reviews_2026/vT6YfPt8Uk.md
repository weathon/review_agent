# Learning When to Be Uncertain: A Post-Hoc Meta-Model for Guided Uncertainty Learning

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Reliable uncertainty quantification remains a major bottleneck in deploying deep learning models under distribution shift. Existing methods that retrofit pretrained models either inherit misplaced confidence or merely reshape predictions, without teaching the model when to be uncertain. We introduce GUIDE, a lightweight evidential learning meta-model approach that attaches to a frozen deep learning model and explicitly learns how and when to be uncertain. GUIDE identifies salient internal features via a calibration stage, and then employs these features to produce a noise-driven curriculum that teaches the model how and when to express uncertainty. GUIDE requires no retraining, no architectural modifications, and no manual intermediate-layer selection to the base deep learning model, making it broadly applicable while reducing user effort. The resulting model avoids distilling overconfidence from the base model, improves out-of-distribution detection ($\approx$ 77\%) and adversarial attack detection ($\approx$ 80\%), as well as preserving in-distribution performance. Across diverse benchmarks, GUIDE consistently outperforms state-of-the-art approaches, demonstrating that actively guiding uncertainty is key to closing the gap between confidence and reliability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces GUIDE (Guided Uncertainty Learning Using a Post-hoc Evidential Meta-model), a post-hoc non intrusive method for uncertainty quantification that can be applied to pretrained deterministic neural networks without retraining them.

### Strengths
- The paper is easy to read 
- The topic can be relevant, and the approach is interesting

### Weaknesses
**Motivation and definition of post-hoc uncertainty:** The motivation behind the paper, as per the abstract, is that existing post-hoc approaches inherit “misplaced” confidence, or reshape the predictions (through temperature scaling for instance, I believe). However, these motivations are not being developed and explained in the later sections of the manuscript, which makes the rationale behind GUIDE somehow unclear. While I just did not like that, since I believe it was not so relevant after all, I then noticed that the authors' definition of post-hoc is a bit ambiguous. By reading, one is left believing that the definition refers to approaches acting on a pretrained network. However, the authors mention as post-hoc method the one proposed in [int_ref_1], which do performs end-to-end training. This makes the whole paper framing confusing, and makes the lack of description of limitations of existing post-hoc approaches (i.e., misplaced confidence and reshaping) extremely vague. 

**Results discussion**: The discussion of the results is a bit reductive and simple, while it should be developed more, especially for what concerns Table 1. For instance, the results on the CIFAR10-CIFAR100 and Oxford Flowers-DeepWeeds are a bit bad for GUIDE. There is no mention of such an outcome, nor an explanation. 

**Adversarial setup:** I understand that the attacks are simply used to produce OOD samples, but please do refrain from using simple PGD or FGSM attacks [ext_ref_1]. There are better approaches which are much widely accepted, such as AA [ext_ref_2]. Also, please specify the used hyperparameters, such as the number of iterations. It is rather possible that the used configurations might be producing extremely suboptimal solutions to the optimization problem of adversarial attacks. 

**Other issues, not necessarily minor:** 
- *References*: I suggest associating Laplace approximation to this pioneering work [ext_ref_3].
- *Figure 6:* I cannot understand which dataset is used in this figure.
- *Citation style*: Please use citet and citep.

[int_ref_1]: Sensoy, Murat, Lance Kaplan, and Melih Kandemir. "Evidential deep learning to quantify classification uncertainty." Advances in neural information processing systems 31 (2018).

[ext_ref_1]: Carlini, Nicholas, et al. "On evaluating adversarial robustness." arXiv preprint arXiv:1902.06705 (2019).

[ext_ref_2]: Croce, Francesco, and Matthias Hein. "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks." International conference on machine learning. PMLR, 2020.

[ext_ref_3]: MacKay, David JC. "A practical Bayesian framework for backpropagation networks." Neural computation 4.3 (1992): 448-472.

### Questions
- Could the authors clarify what is the definition of post-hoc uncertainty? 
- Could the authors clarify what is precisely the motivation behind the work? Can the authors better articulate what gap GUIDE fills that existing post-hoc or evidential approaches do not?
- Why is Evidential Deep Learning [int_ref_1] considered post-hoc here, despite being trained end-to-end?
- Could the authors expand their discussion on the results? Could they include clear discussion on weaker results, such as those on CIFAR10-CIFAR100 and Oxford Flowers-DeepWeeds, and explain why GUIDE performs less effectively in those settings?
- Could the authors specify all attack hyperparameters (e.g., step size, number of iterations) and, specifically, also the perturbation radius used in table 1?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper discusses a new framework GUIDE for uncertainty quantification using an existing (pre-trained) network.  GUIDE uses the layers of the existing network to determint salient layers and connect them to a meta-model network for uncertainty guided training.  Using the GUIDE framework, the authors showed an improvemend of the Uncertainty Estimation as opposed to state-of-the-art.  The authors have also shown this through ID vs OOD datasets but also through adverserial attacks.  

Main contributions:
- The use of layer importance measure to select the layers through a saliency metric
- Construction of a curriculum (dataset of corrupted data points with noise based on saliency metric) which is used in the loss function for uncertainty guided training of the meta-model
- The authors have done extensive experiements accross different datasets and adverserial methods

### Strengths
- The paper is presented in a clear way and is well written. 
- The paper also shows extensive experiements using many datasets. 
- Although the relevance propagation is a well described way to describe explainability in neural networks, it is used as a means to train a meta-model which estimates the uncertainty.  In that way, the authors describe a novel and orignal way to use this for uncertainty quantification.  This has very promising results as opposed to other techniques (such as Bayesian NN).
- The paper has a strong mathematical foundation. 
- The main article focusses on the main findings and a clear explanation.  The authors have done a good job in dividing the main findings with detailed descriptions of the theorems and additional experiments, which are described in the appendix.

### Weaknesses
- I noticed that the authors have struggled with the page limit due to the size of some tables (e.g.g Table 1).  I would suggest to leave out a columns in table 1 (e.g. EMM+curric). 
- The use of intermediate layers is for uncertainty estimation is not completely novel.  The use of intermediate layers for uncertainty has been provided in [1], [3].  Also uncertainty has been determined using Intermediate Layer Variational Inference [2]. 
- Some explanations at the end of the article are very short.  It gives the impression that this in unfinished work.  E.g. the adverserial attack analysis is very relevant but needs more explanation. 



[1] Ameer et al., Enhancing adversarial robustness with randomized interlayer processing, 2024, Expert Systems with Applications,
[2] Ahmed et al. 2021. Real-time Uncertainty Estimation Based On Intermediate Layer Variational Inference. In Proceedings of the 5th ACM Computer Science in Cars Symposium (CSCS '21).
[3]  https://arxiv.org/abs/2012.03082

### Questions
- The definition of the uncertainty target.  What is the rationale behind this formula?
	- What is the reason why the training is done first on clean targets and later noise-corrupted targets are used?  Will it not induces a more difficult to learn?  What is the effect of the training process?
	- How many layers are selected to reach the cumulative relevance coverage threshold?  What is the impact of this threshold in the amount of selected layers. 
	- The authors mention in the results section that intrusive methods have a large coverage, but that the OOD detection is high.  Also the methods of the authors claim that the OOD coverages is below 10%, but it is not clear to me why this is good.  In my opinion, a high OOD coverage would mean that the model can detect OOD samples correctly.  Could the authors clarify this?  I propose to elaborate on the explanation on the OOD coverage.
	- The authors claim that GUIDE is architecture agnostic, however for the method to work, one needs to calculate the relevance values.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes GUIDE, a novel framework for guided uncertainty estimation that teaches a pretrained model when and how to be uncertain without requiring retraining or architectural modifications. The method combines saliency calibration and a noise-driven curriculum to enhance reliability and out-of-distribution (OOD) awareness in existing deep networks.

### Strengths
The paper provides solid theoretical grounding, supported by an information-theoretic analysis that links saliency-guided learning to the preservation of Fisher information during uncertainty calibration.

Empirical results demonstrate state-of-the-art performance across diverse in-distribution and out-of-distribution settings (e.g., CIFAR-10 → SVHN, MNIST → FashionMNIST, Oxford Flowers → DeepWeeds). GUIDE consistently outperforms prior baselines (e.g., temperature scaling, Mahalanobis, DUQ, and EMM)

### Weaknesses
The method is validated primarily on small- to mid-scale datasets (MNIST, CIFAR-10, Oxford Flowers). It remains unclear whether GUIDE scales effectively to large-scale foundation models (e.g., CLIP, ViT-L, or LLaVA)

The motivation for selecting specific saliency methods (LRP) is only briefly discussed. It would be beneficial to clarify whether other attribution techniques (Grad-CAM, Integrated Gradients) would yield similar benefits

Figures do not include error variance analysis across multiple random seeds.

### Questions
The theoretical analysis is centered on Fisher information preservation during saliency calibration, which provides a local view of uncertainty propagation. Could the authors clarify whether this framework generalizes to non-local uncertainty effects—for instance, when prediction uncertainty arises from feature interactions beyond the saliency-identified regions?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Summary
This paper introduces GUIDE (Gradual Uncertainty Refinement via Noise-Driven Curriculum), a novel post-hoc evidential meta-model for improving uncertainty quantification (UQ) in pretrained deep learning models without retraining or architectural modifications. The key idea is to explicitly teach a model when and how much to be uncertain, thereby addressing misplaced confidence — a common limitation in existing post-hoc UQ methods.

GUIDE operates in two main stages:
	1.	Saliency Calibration Stage:
The pretrained (frozen) model undergoes a relevance propagation analysis (via Layer-wise Relevance Propagation, LRP-ϵ) to identify salient intermediate features. This yields both layer-level relevance scores and input-level saliency maps, which determine the layers and spatial regions most critical for prediction.
	2.	Uncertainty-Guided Training Stage:
GUIDE attaches a lightweight Dirichlet-based evidential meta-model that consumes features from selected salient layers. Using the previously derived saliency maps, it generates a noise-driven curriculum — progressively corrupting salient input regions to simulate distributional shifts.
The model is trained with a soft-target loss combining uncertainty regularization and a Self-Rejecting Evidence (SRE) penalty, ensuring uncertainty increases monotonically with corruption while confidence remains justified.

Theoretical analysis (Theorem 1) provides guarantees on Fisher information retention, showing that GUIDE preserves a bounded fraction of the base model’s informative structure under the saliency selection mechanism.

Key Contributions
	1.	A fully post-hoc evidential meta-model that explicitly learns when to be uncertain using guided curricula rather than passive calibration.
	2.	Saliency-based layer selection eliminating manual design choices and ensuring feature relevance consistency.
	3.	Noise-driven curriculum learning, progressively teaching uncertainty behavior aligned with model sensitivity.
	4.	Theoretical guarantees for saliency coverage and information retention.
	5.	Extensive experimental validation across multiple in-distribution (ID), out-of-distribution (OOD), and adversarial benchmarks showing robust, state-of-the-art results.

Empirical Results
GUIDE was benchmarked against intrusive and post-hoc UQ baselines including ABNN, EDL-Head, Whitebox, and EMM, across datasets such as MNIST, CIFAR10/100, SVHN, and Oxford Flowers → Deep Weeds.
	•	ID Accuracy: Comparable to baselines (≈ 99% on simple datasets; ≈ 90% on CIFAR tasks).
	•	OOD/Adversarial Coverage: GUIDE achieves the lowest coverage (e.g., ≤ 8% OOD, ≤ 5% adversarial), outperforming others by large margins.
	•	AUROC: GUIDE consistently achieves >94% on OOD and adversarial detection (up to 96% on MNIST → FashionMNIST), exceeding all intrusive and post-hoc baselines.
	•	Calibration: Expected calibration error (smECE) reduced to 0.061, compared to 0.317 (pretrained) and 0.193 (EMM).
	•	Robustness: Maintains high AUROC (>90%) across perturbation strengths and attack types (L2PGD, FGSM, Salt-and-Pepper).

Impact and Positioning
GUIDE is positioned as a non-intrusive, architecture-agnostic, and computationally lightweight solution that bridges the gap between model confidence and predictive reliability. Unlike earlier post-hoc approaches (which reshape outputs), GUIDE actively instructs the model through a principled uncertainty curriculum, yielding better calibration and robustness under distributional shifts.

The approach has strong implications for safe and trustworthy AI, particularly in high-stakes domains like healthcare, autonomous systems, and human-in-the-loop robotics.

Overall Evaluation (Summary KPI)
Criterion
Assessment
Originality
High — introduces guided uncertainty curricula in post-hoc UQ.
Significance
Strong — directly impacts deployment reliability and calibration.
Technical Quality
Excellent — rigorous derivation, clear algorithmic pipeline, theoretical support.
Clarity
Very good — figures and pseudo-code are interpretable, though dense in notation.
Empirical Evaluation
Comprehensive — multiple datasets, attacks, and ablations.
Reproducibility
Strong — open-source repository available.
Potential Weakness
May depend on LRP assumptions; sensitivity to noise schedule hyperparameters not fully explored.

### Strengths
The paper demonstrates a well-balanced and well-executed contribution to the growing area of uncertainty quantification (UQ) and reliable deep learning. It succeeds in combining conceptual novelty with solid empirical performance, offering a practical and interpretable framework that is relevant to both research and deployment contexts.

Originality
	•	The central idea — teaching a pretrained model when to be uncertain via a saliency-guided, noise-driven curriculum — is distinctive among post-hoc uncertainty methods.
	•	While it draws from known concepts (e.g., evidential learning, saliency mapping), the paper’s integration of saliency-based layer selection with curriculum-style uncertainty learning is an innovative synthesis not previously explored in this form.
	•	The Self-Rejecting Evidence (SRE) loss adds a novel regularization mechanism encouraging monotonic uncertainty behavior with respect to perturbation strength.

Quality
	•	Methodologically rigorous and technically consistent, the proposed GUIDE framework is both theoretically motivated (Fisher information retention theorem) and empirically validated across multiple datasets and uncertainty scenarios.
	•	The experiments are comprehensive, spanning in-distribution, out-of-distribution, and adversarial settings, and consistently demonstrate GUIDE’s superiority or parity with state-of-the-art baselines.
	•	The ablation studies, calibration metrics (ECE, AUROC), and robustness analyses collectively support the validity of the paper’s claims.

Clarity
	•	The writing style is clear and structured, balancing technical precision with readability.
	•	The stepwise exposition (motivation → framework → theoretical justification → empirical results) makes the methodology accessible to readers with diverse backgrounds in UQ and evidential deep learning.
	•	Figures illustrating the uncertainty refinement process and saliency-based layer selection are informative, though some could be more tightly integrated with text explanations.

Significance
	•	The contribution is practically significant: GUIDE is post-hoc, lightweight, and architecture-agnostic, making it applicable to real-world AI pipelines without retraining or invasive model access.
	•	The framework addresses one of the most persistent challenges in deep learning — overconfidence under distributional shift — with a method that enhances both calibration and interpretability.
	•	The approach bridges human-in-the-loop AI, explainability, and uncertainty estimation, aligning with broader research directions in trustworthy AI, which is central to ICLR’s evolving research priorities.

Summary of Strengths
Dimension
Assessment
Originality
Creative synthesis of evidential learning, saliency analysis, and guided curricula.
Quality
Strong theoretical and empirical foundation; well-executed experiments.
Clarity
Generally clear exposition and strong visual explanations.
Significance
High practical relevance for post-hoc model reliability and safety.

### Weaknesses
While the paper presents a creative and empirically validated approach to post-hoc uncertainty quantification, it exhibits several conceptual, methodological, and presentation-related weaknesses that limit its theoretical depth and generalizability. The following are specific, constructive observations aimed at strengthening the work for future revisions or journal extensions.

1. Limited Theoretical Grounding of “Learning When to Be Uncertain”
	•	The central notion of learning when to be uncertain remains intuitively compelling but mathematically underdeveloped.
The method demonstrates empirical behavior consistent with this idea but does not provide a formal probabilistic framework or causal model linking saliency-driven noise to epistemic uncertainty.
	•	The proposed Fisher information retention theorem is a useful analytical step but only partially supports the conceptual claim—it ensures informational consistency, not causal justification for uncertainty behavior.
	•	To strengthen rigor, future versions could introduce a formal uncertainty measure evolution model (e.g., monotonic entropy gradient or causal sensitivity analysis) or link GUIDE’s curriculum mechanism to Bayesian learning theory (e.g., Kendall & Gal, 2017; Malinin & Gales, 2018).

2. Dependence on Saliency and Potential Bias
	•	The reliance on Layer-wise Relevance Propagation (LRP-ϵ) introduces methodological fragility:
	•	LRP performance and interpretability vary across architectures and data modalities (e.g., CNNs vs. transformers).
	•	The paper does not evaluate whether GUIDE’s performance depends heavily on the chosen saliency technique.
	•	An ablation comparing different saliency methods (Grad-CAM, Integrated Gradients, DeepLIFT) would clarify whether the framework’s improvements arise from the general saliency mechanism or specific LRP behavior.
	•	This dependence may reduce reproducibility across architectures beyond those tested.

3. Insufficient Hyperparameter and Sensitivity Analysis
	•	The paper introduces several tunable components — notably:
	•	Noise schedule parameters in the saliency-guided curriculum,
	•	Regularization weights (λ₁, λ₂) in the Self-Rejecting Evidence loss, and
	•	Layer selection thresholds (τ).
	•	However, no sensitivity analysis is provided to assess how these affect performance or stability.
	•	This omission weakens claims of robustness and generality, especially for a method promoted as post-hoc and lightweight.
	•	Future work should present a systematic exploration of these hyperparameters, ideally visualized as performance landscapes or uncertainty–accuracy trade-offs.

4. Incremental Conceptual Novelty
	•	Although the method performs well, its conceptual novelty is moderate: it integrates established elements (evidential learning, saliency analysis, curriculum noise) rather than introducing fundamentally new uncertainty theory or model structure.
	•	Similar principles appear in prior works on post-hoc meta-modeling (Postels et al., 2021), selective prediction (Geifman & El-Yaniv, 2019), and uncertainty calibration through adversarial exposure (Mukhoti et al., 2023).
	•	The authors could strengthen originality by clearly positioning GUIDE as a “structured synthesis” rather than a new paradigm, and by articulating where exactly it diverges conceptually or empirically from these precedents.

5. Generalization to Modern Architectures
	•	All experiments are performed on mid-scale CNN-based image classifiers (MNIST, CIFAR, SVHN, Flowers), which are suitable for controlled studies but insufficient for establishing scalability or architectural generality.
	•	The approach’s compatibility with transformers, diffusion models, or multimodal architectures—which dominate current ICLR topics—is untested.
	•	Extending GUIDE to large-scale or multimodal settings would substantiate its relevance to the broader ICLR audience.

6. Limited Real-World or Cross-Domain Evaluation
	•	Although GUIDE improves calibration and OOD robustness, no experiments are conducted in domain-shifted or real-world datasets (e.g., corrupted CIFAR, ImageNet-C, or medical imaging benchmarks).
	•	Including at least one realistic scenario (e.g., sensor noise, class imbalance) would better demonstrate the method’s reliability beyond clean benchmarks.

7. Minor Presentation and Clarity Issues
	•	Some mathematical derivations are densely presented and could benefit from expanded intuition or intermediate explanations.
	•	Figures showing calibration improvement and uncertainty maps are insightful but small, making quantitative differences difficult to assess visually.
	•	A concise visual summary (e.g., flow diagram of the GUIDE pipeline with saliency/noise progression) would improve readability for interdisciplinary readers.

Overall Assessment:
The paper is strong in execution but could significantly improve by deepening theoretical grounding, expanding generalization experiments, and clarifying dependence on design choices. These refinements would elevate the work from a high-quality empirical contribution to a conceptually mature framework deserving of top-tier recognition.

### Questions
1. On the Theoretical Framing of “Learning When to Be Uncertain”
	•	Could the authors formalize the concept of learning when to be uncertain beyond its intuitive description?
For example:
	•	Is there a measurable quantity (e.g., monotonic relationship between noise level and entropy or epistemic evidence) that supports this claim?
	•	How does the proposed Self-Rejecting Evidence (SRE) loss encourage this behavior mathematically — does it impose any guarantee of monotonic uncertainty growth under perturbation?
	•	A short theoretical or empirical analysis of this relationship would make the claim substantially stronger.

2. On the Role and Robustness of LRP in GUIDE
	•	GUIDE relies heavily on Layer-wise Relevance Propagation (LRP-ϵ) for both saliency selection and curriculum construction.
	•	How sensitive is GUIDE’s performance to the specific saliency method used?
	•	Have the authors tested alternative saliency measures (e.g., Grad-CAM, Integrated Gradients, or SHAP)?
	•	Could GUIDE’s uncertainty refinement fail if the saliency signal is noisy or misaligned (as often happens in deeper models)?
	•	Including an ablation or at least a qualitative comparison would help clarify whether the saliency mechanism is integral or replaceable.

3. On Hyperparameter Sensitivity
	•	The method introduces multiple hyperparameters:
	•	λ₁, λ₂ (regularization weights),
	•	τ (saliency threshold), and
	•	noise schedule parameters (corruption magnitude or rate).
	•	Could the authors provide sensitivity curves or variance estimates showing GUIDE’s stability with respect to these parameters?
	•	This would be particularly useful to assess GUIDE’s reliability as a “plug-and-play” post-hoc method.

4. On Comparative Baselines and Fairness
	•	How do the authors ensure fair comparison with existing methods like Deep Ensembles, MC-Dropout, or Temperature Scaling?
	•	Were all models trained or calibrated using identical datasets and computational budgets?
	•	Since GUIDE is post-hoc, how is its runtime and memory footprint compared to ensemble-based methods?
	•	Including a table summarizing computational cost vs. performance could clarify GUIDE’s real-world efficiency.

5. On Theoretical Result (Theorem 1 – Fisher Information Retention)
	•	Theorem 1 provides a bound on retained Fisher information under saliency selection.
	•	Could the authors elaborate on the assumptions underlying this bound (e.g., independence or linearity of selected features)?
	•	Is this guarantee empirical (observed in finite data) or asymptotic (as dataset → ∞)?
	•	How tight is the bound in practice — do the authors have empirical measurements correlating Fisher information loss with performance degradation?

6. On the Relationship Between GUIDE and Calibration Methods
	•	GUIDE seems related in spirit to post-hoc calibration (e.g., Temperature Scaling, Platt Scaling, Isotonic Regression) but introduces a learning component.
	•	How does GUIDE differ conceptually and practically from these calibration techniques in terms of the underlying uncertainty model (Dirichlet vs. softmax temperature)?
	•	Could the authors provide a brief comparison or unified framework situating GUIDE among existing calibration approaches?

7. On Scalability to Modern Architectures
	•	Have the authors tested GUIDE on transformer-based architectures (e.g., ViT, DeiT) or multimodal models?
	•	If not, do they anticipate challenges due to the saliency extraction step or computational scaling?
	•	This would be important to determine whether GUIDE generalizes beyond CNN-based settings, which are becoming less central in current ICLR research.

8. On Dataset Diversity and Real-World Scenarios
	•	The experiments are limited to canonical image datasets (MNIST, CIFAR, SVHN, Oxford Flowers).
	•	Could the authors discuss potential extensions or ongoing work on realistic, domain-shifted, or corrupted datasets (e.g., CIFAR-C, ImageNet-C, or medical images)?
	•	This would strengthen the claim of GUIDE’s utility in safety-critical applications.

9. On Interpretability and Human Alignment
	•	Since GUIDE integrates saliency-driven learning, it implicitly aligns with explainability goals.
	•	Have the authors considered whether the generated saliency maps or uncertainty overlays are interpretable by human users (e.g., can humans trust GUIDE’s “when uncertain” behavior)?
	•	A small user study or interpretability evaluation could add practical value to this aspect.

10. On Open-Source and Reproducibility
	•	The paper claims that code will be made available.
	•	Could the authors clarify the current status of code release and whether pretrained models, saliency scripts, and evaluation setups will be publicly accessible?
	•	Reproducibility is especially important for a post-hoc framework advertised as “lightweight and widely applicable.”

The authors’ clarifications on the theoretical basis, robustness to saliency and parameters, and scalability of GUIDE could significantly enhance the perceived rigor and impact of this work. A strong rebuttal addressing these points with additional experiments or analysis would likely improve its overall evaluation.

### Soundness
3

### Presentation
3

### Contribution
3
