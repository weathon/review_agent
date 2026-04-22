# FaithShield: Defending Vision–Language Models Against Explanation Manipulation via X-Shift Attacks

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 2, 6

## Abstract
Vision–Language Models (VLMs) such as Contrastive Language–Image Pre-training (CLIP) have achieved remarkable success in aligning images and text, yet their explanations remain highly vulnerable to adversarial manipulation. Recent findings show that imperceptible perturbations can preserve model predictions while redirecting heatmaps toward irrelevant regions, undermining the faithfulness of the explanation. We introduce the X-Shift attack, a novel adversarial strategy that drives patch-level embeddings toward the target text embedding, thereby shifting explanation maps without altering output predictions. This reveals a previously unexplored vulnerability in VLM alignment. To counter this threat, we propose FaithShield Defense, a two-fold framework: (i) a dual-path redundant extension of CLIP that disentangles global and local token contributions, producing explanations more robust to perturbations; and (ii) a novel faithfulness-based detector that verifies explanation reliability via a masking test on top-$k$ salient regions. Explanations that fail this test are flagged as unfaithful. Extensive experiments show that X-Shift reliably compromises explanation faithfulness, while FaithShield restores robustness and enables principled detection of manipulations. Our work formalizes explanation-oriented adversarial attacks and offers a principled defense, enhancing trustworthy and verifiable explainability in VLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies adversarial attacks and defenses for explanatory methods. It first builds the X-shift attack, an attack method that maintains classification predictions while changing the alignment of intermediate features towards the target text, under constraints such as $\ell_{0}$ and class-dominance. Then it proposes a robust explanatory method, refined from Li et al. [1], for robust visualization under X-shift attacks.

Li, Yi, et al. "A closer look at the explainability of Contrastive language-image pre-training." _Pattern Recognition_ 162 (2025): 111409.

### Strengths
1. This paper investigates a novel area that has not been explored yet.
2. The motivation of this paper is clear.
3. An attack method along with a robust defense method is proposed, facilitating further research.

### Weaknesses
1. Presentation. Fig. 1 and 2 are not Vector Graphics. They get blurred when zooming in, especially Fig. 2. Sec. 3.2 is mainly composed of bullet points. It should be connected with coherent words and formulas, thus bullet points take up the space of the algorithm, which can only be presented in the appendix. Table 1 is also way too big. 
2. Limited innovation. The FaithShelf Stage 1 largely overlaps with existing method [1], while FaithSelf Stage 2 mainly applies a drop-out test. 
3. Lack of baseline methods. From Fig. 3, it seems like FaithShelf Stage 1 has already moved the concentration of the adversarial sensitivity map to the bench. Also, the raw patch similarity looks messy anyway, and few people will use it as an explanatory tool, so there is no reason to attack it. It should be that first X-shift Attack move the ***concentrated*** sliency map (produced by other baseline methods) towards some irrelevant object. It becomes crucial for a robust explanatory method. Currently, I can not see this point clearly. 

The current quality of this paper is well below the required level for acceptance. A significant refinement is needed for resubmission.

### Questions
1. How does the stage 1 differ from the existing method [1]? 

Li, Yi, et al. "A closer look at the explainability of Contrastive language-image pre-training." _Pattern Recognition_ 162 (2025): 111409.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces X-Shift, a novel adversarial attack that manipulates explanation maps in VLMs like CLIP without altering their predictions. The attack shifts attention heatmaps toward irrelevant regions, undermining the trustworthiness of model explanations. The paper also propose FaithShield, a two-stage defense framework. The first stage enhances robustness through a dual-path refinement while the second stage detects unfaithful explanations using a confidence-drop test. Experiments across multiple datasets show that FaithShield significantly improves explanation stability and enables reliable detection of adversarial manipulations.

### Strengths
1. The paper identifies a interesting task in the safety of VLMs, which has the risk of being manipulated at the interpretation level, especially the heat map may be misleading without affecting the prediction results.

2. The proposed FaithShield framework is technically sound and clearly explained, combining dual-path refinement and a faithfulness-based detection mechanism to enhance robustness and verifiability.

### Weaknesses
1. In the second paragraph of the Introduction, the authors introduce the value of the attack with the phrase "remains largely unexplored," which seems insufficient in terms of research value. Could the authors provide concrete research to truly demonstrate the value of this attack? For example, could they explain the potential impact of this attack when implemented within a specific research context and with specific objectives?

2. In Figure 1, in the Transformer Encoder, why doesn't the dual path go through the MLP? Also, are all these modules optimizable? If there are any frozen parameters, it is suggested to mark them in the figure.

3. It is recommended to use vector graphics for Figure 1 and Figure 2, and use fonts that align with the article.

4. The experiment part lacks the ablation of the weight parameters in Equation 8.

5. As a two-stage approach, is there any time or efficiency comparison for FaithShield Defense?

6. What do the metrics CosSim (CLS) and Max $\Delta$ Prob mean? In Table 1, they are identical to the CLIP values. Are there no special circumstances? Can you provide a naive baseline, such as one with only an attack, to show how they differ?

### Questions
Overall, the author's work is interesting, but the method and experiments need to be enhanced. The relevant suggestions are listed in Weaknesses.

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
3

### Summary
This paper proposes a CLIP-based vision–language model that aims to modify the image embedding toward the target text embedding without changing the model’s maximum output, thereby shifting the explanation maps. Based on my understanding, this can be regarded as an adversarial attack on interpretability.

### Strengths
The paper is the first to consider this problem setting under the CLIP framework.

### Weaknesses
1. Lacks sufficient ablation studies, especially on how different weight magnitudes in Equation (8) affect the results.

2. The figures are not professionally prepared and appear somewhat blurry. They should be replaced with vector-format images.

3. The font size in the figures (e.g., Figures 3–5) is too small to read clearly.

4. The motivation is not clearly written, and several claims are overextended or insufficiently explained (see questions below).

### Questions
1. I am not an expert in this area, and this is my first time encountering this problem setting. Is this setting truly meaningful? From my understanding, as long as the prediction remains accurate, the explanation map should still mainly focus on the target object. Even if its intensity decreases. What is the practical significance or application of this problem formulation?

2. In Equation (2), the authors state that “the primary goal is to force patch embeddings toward the target text embedding,” but then they also write “we maximize similarity of the top-K patches while suppressing others.” If the goal is to align all patches with the target embedding, why suppress some of them? The paper does not explain the role or motivation for suppressing certain patches.

### Soundness
2

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
3

### Summary
The paper exposes a new vulnerability in CLIP-style vision–language models: explanations (patch–text heatmaps) can be adversarially shifted without changing predictions. It introduces X-Shift, an inference-time attack that pulls patch embeddings toward a target text embedding while enforcing prediction consistency, per-patch margins, sharpness, and sparse valid perturbations.

To defend, FaithShield has two stages:

* Stage I: Dual-path refinement that replaces standard self-attention with consistent V–V attention, skips FFNs in the explanation path, and removes redundant features, yielding sharper, foreground-focused, and more robust heatmaps.
* Stage II: A faithfulness test that masks top-ρ% salient regions and flags explanations as unfaithful if the confidence drop is below a threshold.

Across ImageNet, Flickr30k, and COCO with CLIP ViT-B/16, B/32, L/14, X-Shift strongly alters heatmaps while keeping predictions stable; FaithShield restores heatmap robustness (higher Top-k IoU) without harming accuracy and detects manipulated explanations.

### Strengths
1. Originality and problem framing: The paper defines a new explanation-focused threat model for VLMs (X-Shift) that shifts patch–text heatmaps without changing predictions. It tailors objectives to text-conditioned similarity (patch steering, entropy sharpening, patch-margin, sparsity). The joint attack-plus-defense (FaithShield) with dual-path refinement and a causal masking detector is novel for multimodal XAI.

2. Technical quality and empirical rigor: The attack and defense are precisely specified with clear losses, constraints, and algorithms enabling reproducibility. Defense mechanisms (consistent self-attention, dual-path aggregation, redundancy removal) are well-motivated and operationalized, with a principled confidence-drop test. Experiments span multiple datasets/backbones with appropriate metrics, showing consistent IoU gains without harming accuracy.

3. Clarity and significance: The paper clearly separates prediction robustness from explanation robustness and explains why patch–text similarity is a natural manipulation surface. Mathematical formulation and detection criterion are easy to follow, aided by figures and stepwise algorithms. The work elevates VLM explainability to a security concern, with practical implications for trustworthy deployment.

### Weaknesses
1. While the combination of consistent self-attention, dual-path aggregation, and redundancy removal is adapted for robustness, portions build on Li et al. (A closer look at the explainability of contrastive language-image pre-training). The paper would benefit from a more explicit ablation and attribution of gains: which components (consistent attention vs skipping FFNs vs redundancy removal) contribute most to adversarial robustness (not just interpretability), and how this differs empirically from Li et al. ’s formulation.
2. The masking-based detection echoes causal deletion tests used in saliency evaluation. Clarify novelty relative to established faithfulness tests and justify design choices (cosine similarity normalization, thresholding strategy) versus alternatives (logit/probability drops, energy-based measures).
3. Threshold selection: The detection threshold θ and masking ratio ρ appear fixed but selection criteria are not detailed. Provide systematic calibration (ROC, AUC, FPR at fixed TPR) across datasets and backbones, and analyze sensitivity to θ, ρ, and masking method (zeroing vs blurring vs inpainting).
4. False positives/negatives: Quantify detection trade-offs, especially in naturally challenging images where explanations may be diffuse or multi-object. Report detection under distribution shift and for clean samples to ensure low false alarm rates.
Attack-transfer to detector: Evaluate whether small, structured perturbations can spoof high ∆conf (e.g., by concentrating heatmap on benign-but-causal pixels) to evade detection.

### Questions
1. Adaptive-attacker robustness and ablations:
* How does FaithShield perform against an adaptive adversary that differentiates through Stage I and uses a surrogate for Stage II’s masking to keep Δconf above θ? Please report results where the attacker augments its loss with the detection term, randomizes ρ/θ during optimization, and employs stronger perturbation sets (e.g., larger k, ℓ∞/ℓ2 bounds, spatial/color transforms). Also ablate consistent self-attention, dual-path aggregation, and redundancy removal to quantify each component’s contribution under adaptive attacks.

2. Generality beyond CLIP and across XAI methods
* Do X-Shift and FaithShield transfer to other VLMs (e.g., SigLIP, ALIGN, BLIP-2) and tasks (e.g., grounding, VQA)? How does the attack affect gradient-based explanations (Grad-CAM, IG), and does Stage I still help when similarity maps are not the explainer? Please include cross-model and cross-explainer transfer results and discuss any architectural assumptions (e.g., ViT patching, attention pooling) that constrain applicability.

3. Detection reliability and operational thresholds
* How sensitive is the masking-based detector to ρ, masking strategy (zeroing vs. blur), and heatmap sharpness α, and how should θ be calibrated in a label-free deployment? Please report FPR/FNR on clean vs. attacked data under distribution shift and natural corruptions, provide ROC/PR curves with confidence intervals, and evaluate attackers that explicitly minimize Δconf to probe worst-case detection performance.

### Soundness
3

### Presentation
3

### Contribution
3
