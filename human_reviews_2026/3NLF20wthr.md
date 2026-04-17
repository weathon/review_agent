# 3S-Attack: Spatial, Spectral and Semantic Invisible Backdoor Attack Against DNN Models

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Backdoor attacks implant hidden behaviors into models by poisoning training data or modifying the model directly. These attacks aim to maintain high accuracy on benign inputs while causing misclassification when a specific trigger is present. While existing studies have explored stealthy triggers in spatial and spectral domains, few incorporate the semantic domain. In this paper, we propose 3S-attack, a novel backdoor attack which is stealthy across the spatial, spectral, and semantic domains. The key idea is to exploit the semantic features of benign samples as triggers, using Gradient-weighted Class Activation Mapping (Grad-CAM) and a preliminary model for extraction. Then we embedded the trigger in the spectral domain, followed by pixel-level restrictions in the spatial domain. This process minimizes the distance between poisoned and benign samples, making the attack harder to detect by existing defenses and human inspection. And it exposes a vulnerability at the intersection of robustness and semantic interpretability, revealing that models can be manipulated to act in semantically consistent yet malicious ways. Extensive experiments on various datasets, along with theoretical analysis, demonstrate the stealthiness of 3S-attack and highlight the need for stronger defenses to ensure AI security.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a backdoor attack called 3S-Attack, which aims to achieve stealthiness across three dimensions: spatial, spectral, and semantic. The core idea of the attack is to use the semantic features of benign samples as the trigger. Experimental results show that the attack achieves a high ASR on multiple datasets while maintaining high PSNR and SSIM values.

### Strengths
1. The paper addresses stealthiness across three different domains, as previous attacks often focused on only one or two domains.

2. The comparison of spatial and spectral residuals in Figure 1 provides an intuitive visual demonstration of the attack's stealthiness.

### Weaknesses
1.  The effectiveness of the attack on high-resolution datasets, such as ImageNet, is not explored. I'm interested in the performance of 3S attack on such datasets.

2.  The baseline attacks used for comparison are relatively old (all from 2022 or earlier). A comparison with more recent and advanced backdoor attacks better highlights the paper's contribution.

3. The presentation of experimental results lacks clarity in several key areas:

- Although the authors state in the caption of Table 2 that "The benign accuracy is not displayed because in each experiment the benign accuracy never drop more than 2%," I believe the Benign Accuracy should still be explicitly reported in the table for a clear and direct comparison.

- Table 2 compares ASR, PSNR, and SSIM for different attacks, but it does not specify the poison rate used to obtain these results.

4. Although the paper claims stealthiness in all three domains, the actual effectiveness against defenses is not ideal. The abstract states the attack is harder to detect by existing defenses, but the experiments in the Appendix show that 3S-Attack can still be detected by AC and NC. Many other state-of-the-art stealthy backdoor attacks can already bypass these defenses, which diminishes the claimed novelty and contribution of this work.

5. The claim that "3S-Attack is also the first semantic-domain stealthy backdoor attack that operates purely through poisoned samples..." appears to be inaccurate. There are already some existed semantic backdoor attacks.

### Questions
1. Can the authors elaborate on why AC succeeds in detecting the attack (i.e., what traces does 3S-Attack leave at the activation level)? Furthermore, how would the "activation-aligned poisoning" mentioned in the "Limitations and Future Work" section (Appendix A.4) be implemented to evade AC?

2. The paper does not state the initial Benign Accuracy for the model trained on Animal10. I observed in Figure 8b that the BA drops below 60% at a very low neuron pruning rate. Is this because the original model's classification performance on this dataset was poor, or is this rapid drop caused by the Fine-Pruning process itself?

3. The authors selected different models for different datasets (e.g., VGG/ResNet for CIFAR-10, WRN for CIFAR-100) instead of showing comprehensive results on a consistent set of models. This experimental setup is not ideal and complicates comparison with other backdoor attack papers. I recommend the authors provide more experiments on consistent model architectures to avoid any suspicion of cherry-picking results.

4. What is the time cost for generating the poisoned samples? The attack relies on training a "preliminary model," which seems to imply that the poison generation time could be even longer than  the victim's model training time itself. This high computational cost for preparation may affect the practical feasibility of the attack.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents an interesting attempt to achieve multi-domain stealth in backdoor attacks, supported by extensive experiments. However, the methodological novelty is moderate, the semantic analysis remains qualitative, and presentation quality can be improved.

### Strengths
The paper proposes a unified backdoor attack that simultaneously achieves stealthiness across spatial, spectral, and semantic domains, which is an underexplored but meaningful direction.

Experiments are performed on multiple datasets and models, showing the generality of the method.

The paper provides clear algorithmic descriptions, ablation studies, and defense-resistance analyses, enhancing reproducibility and technical depth.

### Weaknesses
Although the paper claims semantic invisibility, the evaluation mainly relies on Grad-CAM visualization and AC/NC detection. More quantitative semantic similarity metrics (e.g., feature-space distance, neuron activation overlap) would strengthen the claim.

Some compared methods are relatively dated. Including more recent backdoor attacks would make comparisons more convincing.

The manuscript is lengthy, with excessive large figures and overlapping content between the main text and appendix, which reduces readability. Condensing and summarizing figures would improve clarity.

### Questions
Although the paper claims semantic invisibility, the evaluation mainly relies on Grad-CAM visualization and AC/NC detection. More quantitative semantic similarity metrics (e.g., feature-space distance, neuron activation overlap) would strengthen the claim.

Some compared methods are relatively dated. Including more recent backdoor attacks would make comparisons more convincing.

The manuscript is lengthy, with excessive large figures and overlapping content between the main text and appendix, which reduces readability. Condensing and summarizing figures would improve clarity.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a novel backdoor attack against DNN models, called 3S-attack.It extracts the semantic features of benign samples with a preliminary model as triggers and embed the trigger in the spectral domain. Finally, it restricts the poisoned images in the spatial domain. Thus, the authors successfully make the attack stealthy across the spatial, spectral, and semantic domains.

### Strengths
1. Novelty — first black-box semantic-stealth attack: 
Demonstrates the first attack that achieves semantic stealth without access to the victim model or training pipeline, filling a notable gap in threat modeling. 
2. Rigorous multi-axis evaluation: 
Empirically validates stealth across semantic, spatial, and spectral defenses, showing the attack’s robustness against diverse defense paradigms.

### Weaknesses
1. Surrogate data dependence: The attack's reliance on a clean surrogate model and its sensitivity to distributional mismatch remain unexplored. Labeling ambiguity: The labeling strategy for poisoned samples is unclear and inconsistent with the stated attack objective.
2. Incomplete reporting: Benign accuracy (BA) and post-defense results are missing for some datasets, weakening claims of minimal performance drop.
3. Labeling ambiguity: The labeling strategy for poisoned samples is unclear and inconsistent with the stated attack objective.

### Questions
1. The attack pipeline is unclear regarding poisoned-label assignment. For each poisoned image, what label is used during poisoning: the original (benign) label or the attacker's target class? If the poisoned images retain original labels, how does this reconcile with the stated goal of misclassifying target images into the attack class? 
Please clarify the labeling strategy and the exact optimization objective.
2. Grad-CAM extracts regions the model attends to for its decision (i.e., activations most contributive to the target class). Is it right to interpret these regions as semantic features? Please clarify how you distinguish true semantic attention from spurious cues (e.g., learned background correlations), and provide any analysis that demonstrates the highlighted regions correspond to semantically meaningful object parts rather than dataset artifacts.
3. 3S-attack requires pretraining a clean surrogate model. What are the requirements on the surrogate’s training data and distribution? If the attacker’s data distribution differs from the victim’s (e.g., attacker has cat images while victim trains on birds), what is the expected impact on attack efficacy? Please quantify sensitivity to distributional mismatch.
4. The authors state BA drops ≤2% and therefore omit BA in Table 2, yet Figure 8 shows notably low BA on GTSRB and Animal10. Please add the per-dataset BA values to Table 2 and report the post --FP-defense attack success (or other relevant metrics) on additional datasets. This will substantiate the claim of minimal

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces **3S-Attack**, a novel backdoor attack that achieves stealthiness across **spatial, spectral, and semantic domains**. The core idea is to leverage **Grad-CAM** to extract semantically important regions from benign samples, use **Discrete Cosine Transform (DCT)** to identify and manipulate stable frequency components (those with magnitude differences below a threshold), and apply **pixel-level restrictions** to ensure imperceptibility. The attack operates solely via data poisoning without access to the training process. Experiments on five datasets (MNIST, GTSRB, CIFAR-10/100, Animal-10) demonstrate high attack success rates (ASR), high PSNR/SSIM values, and strong resistance to spatial, spectral, and semantic domain defenses (e.g., STRIP, FTD, Grad-CAM, and Fine-Pruning).

### Strengths
**Comprehensive multi-domain stealth design**

* The attack unifies spatial, spectral, and semantic concealment, which no prior method achieves simultaneously.
* This cross-domain formulation exposes new security blind spots where standard single-domain defenses fail (Sec. 4.4).
* The modular design (Grad-CAM → DCT → pixel restriction) makes the idea easily reproducible and adaptable.

**Novel use of Grad-CAM for trigger extraction**

* Grad-CAM is used not for defense but to identify salient semantic regions to build the trigger (Sec. 3.3; Fig. 2).
* This inverts interpretability tools into attack mechanisms, revealing a nuanced vulnerability in semantic attention consistency.
* It also allows transferability without model access — a realistic and underexplored threat model.

**Clear motivation and background integration**

* The introduction logically connects Grad-CAM’s interpretability with attack invisibility (pp. 2–3).
* Figures 2–3 effectively depict the two key stages of the attack—trigger selection based on Grad-CAM saliency and trigger injection through frequency-domain modification—demonstrating the transition from clean to poisoned samples.

**Defense-resistance demonstration**

* Section 4.4 compares 3S-Attack with BadNets using STRIP, Grad-CAM, and FTD (Fig. 6–7).
* The near-overlap of saliency maps between benign and poisoned samples supports semantic stealth.
* These results highlight the inadequacy of current interpretability-based defenses.

### Weaknesses
**Limited theoretical rigor in frequency-domain reasoning**

* The choice of the “Frequency Selection Threshold” (Sec. 3.3) is heuristic; no explicit formula or derivation links magnitude difference and model sensitivity.
* There is no analysis of how DCT component manipulation affects semantic embeddings or classification confidence (no equations in Sec. 3.3–3.5).

**Ambiguity in semantic transferability across models**

* The method assumes the saliency from a surrogate model approximates that of the target one (Sec. 3.2).
* No quantitative measure is given for semantic alignment between surrogate and victim Grad-CAM maps.

**Insufficient ablation and interpretability analysis**

* The contribution of each step (semantic extraction, spectral embedding, pixel restriction) is not isolated.
* For instance, an ablation showing Grad-CAM vs. random region selection would clarify semantic importance.
* Similarly, removing pixel restriction would test the necessity of that safeguard (Sec. 3.5).

**Unclear visualization and ambiguous labeling in Figure 4**

* The red circles highlighting artifacts are too thin to be noticeable, and the diagram’s directional flow is ambiguous.
* The figure should explicitly label the two sides (e.g., Before Restriction → After Restriction) above the arrows and use thicker or more vivid annotations to highlight the changed regions.

### Questions
**Frequency selection rationale**

* Clarify why frequencies with small DCT magnitude differences between the original and Grad-CAM–weighted images are assumed to represent semantically stable components? 
* A more explicit justification or sensitivity analysis for the threshold δ would strengthen the methodological soundness.

**Semantic transferability across models**

* How consistent are the Grad-CAM saliency maps between the surrogate and victim models? Quantitative evidence (e.g., overlap or similarity metrics) would clarify whether the semantic trigger generalizes across architectures.

### Soundness
3

### Presentation
1

### Contribution
3
