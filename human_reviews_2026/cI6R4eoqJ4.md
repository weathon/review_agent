# Adversarial Visual Contrastive Decoding for Mitigating Hallucinations in Large Vision-Language Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Large Vision-Language Models (LVLMs) have achieved remarkable progress in various multimodal AI tasks but are prone to generating hallucinations—outputs that are plausible-sounding yet factually incorrect or ungrounded in the visual input. This phenomenon, particularly the generation of non-existent objects or misdescribed object attributes, severely undermines model reliability. This paper introduces Adversarial Visual Contrastive Decoding (AVCD), a novel inference methodology designed to mitigate hallucinations in LVLMs. AVCD refines the existing Visual Contrastive Decoding (VCD) framework by replacing its use of random noise with adversarial perturbations. These adversarial images are specifically engineered to perturb the vision encoder’s features to decrease cosine similarity with the original image. Our analysis reveals that unlike random noise, these adversarial perturbations are directional; they actively steer the model toward hallucinatory states rather than simply degrading visual features. This creates a more potent and informative contrastive signal, enabling a more effective suppression of hallucinatory content. Experiments on standard benchmarks indicate that the proposed AVCD method achieves notable performance improvements over VCD and other baseline techniques. This work underscores the potential of leveraging adversarial principles not merely for identifying model vulnerabilities but as a constructive tool in enhancing the faithfulness and reliability of LVLM outputs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Adversarial Visual Contrastive Decoding (AVCD), an inference-time method for reducing hallucinations in large vision-language models. Instead of contrasting the model’s logits on the original image against Gaussian-noised variants (as in prior VCD-style methods), AVCD generates adversarial images via a few PGD steps that intentionally push the visual features away from the ground-truth semantics. Decoding then subtracts the likelihood under this adversarial view from the original, sharpening the decision signal and suppressing spurious content. The authors support this with analyses of entropy, CLIP-space similarity shifts, and attention variance, and report performance gains over contrastive baselines on POPE, CHAIR, and MME.

### Strengths
(+) AVCD proposes simple yet effective methods into the inference pipeline, requires only a few adversarial steps, and delivers steady improvements over Gaussian-based contrastive decoding across multiple LVLMs and benchmarks.

(+) The paper clearly articulates why adversarial (directional) perturbations are superior to Gaussian (isotropic) noise, supporting the claim with both theoretical analysis and empirical evidence.

(+) Results show stable and repeatable improvements over strong baselines, with minimal implementation complexity and modest computational overhead.

### Weaknesses
(-) Novelty is incremental. Replacing Gaussian perturbations with adversarial “hard negatives” is a natural extension of contrastive decoding. The paper’s analysis is solid, but the conceptual step feels unsurprising and incremental relative to prior visual distortion and contrastive techniques.

(-) White-box assumption limits applicability. The method requires backprop access through the vision encoder to generate an adversarial image. This excludes many closed-source/API LVLMs and any deployment where gradients are not exposed, curbing practical impact.

(-) The comparison scope is narrow. Evaluations emphasize contrastive decoding baselines. Additional comparisons with [1,2] are needed. Furthermore, the paper does not adequately situate AVCD against training-time or steering-time approaches—e.g., RLHF, OPERA, HALC, Woodpecker, ProjectAway.

[1] Self-Correcting Decoding with Generative Feedback for Mitigating Hallucinations in Large Vision-Language Models, ICLR 2025.
[2] Cross-Image Contrastive Decoding: Precise, Lossless Suppression of Language Priors in Large Vision-Language Models, arxiv 2025.

(-) 
The LVLMs evaluated are outdated. Including newer/stronger backbones would strengthen claims of generality and headroom.

(-) Performance depends noticeably on ε, α, and step count, and the best settings vary by task. Providing an adaptive schedule (e.g., entropy- or uncertainty-driven α/ε) or a simple tuning heuristic would improve robustness and deployability.

### Questions
AVCD maximizes feature distance from the current image semantics. What if the adversarial objective instead pushes the adversarial embedding toward embeddings of other random images (or toward a curated bank of semantically conflicting images)? Intuitively, this could provide a more directional negative signal than an unconstrained push-away. Have you tried a “push-toward-distractors” loss (e.g., nearest-neighbor attraction to a mismatched image set in the same feature space)? How does it compare in efficacy and cost?

### Soundness
3

### Presentation
3

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
The paper proposes Adversarial Visual Contrastive Decoding (AVCD), a training-free inference method. AVCD replaces VCD’s Gaussian noise with directional adversarial perturbations (generated via Projected Gradient Descent, PGD). These perturbations are optimized to minimize the cosine similarity between the vision encoder features of original and adversarial images, steering the model toward hallucinatory states (instead of just degrading features). During decoding, AVCD contrasts logits from the original and adversarial images to suppress hallucinatory content.

### Strengths
**Novel Conceptual Foundation:** It introduces a clever and novel repurposing of adversarial attacks. Rather than treating them solely as a vulnerability, it leverages them as a constructive tool to generate powerful "hard negative" signals for contrastive decoding.

**Strong Empirical Validation:** The method is rigorously evaluated across three standard benchmarks (POPE, CHAIR, MME) and multiple model architectures (LLaVA, Qwen-VL, etc.), consistently demonstrating superior performance over strong baselines like VCD.

### Weaknesses
1. Sensitivity to Hyperparameters
AVCD's performance is highly dependent on the careful tuning of several hyperparameters, which can complicate its practical deployment.
The optimal adversarial perturbation strength (ϵ) varies by task: ϵ=256 works best for the POPE benchmark, while ϵ=64 is optimal for CHAIR and MME.Also, this task-dependent tuning requirement increases the burden of finding optimal settings for new applications.

2. While the authors provide a theoretical proof (Theorem 1) to explain AVCD's effectiveness, it relies on several simplifying assumptions that limit its real-world applicability.

    ***Single Object Assumption***: The proof only considers a scenario where the model chooses between one ground-truth object (x) and one specific non-existent object (x'). In reality, LVLMs perform open-vocabulary generation from a vast candidate set of possible hallucinations, a complexity not captured by the proof.

    ***First-Order Taylor Approximation***: The proof uses a linear approximation of the highly non-linear vision encoder. This assumption becomes increasingly inaccurate with large perturbation budgets (ϵ), making the derived theoretical threshold less reliable.

    ***High-Dimensional Orthogonality***: The proof assumes that residual vectors in the embedding space are orthogonal. However, adversarial perturbations are directionally optimized, not random, and may actively align with specific semantic concepts, violating the orthogonality assumption.

3. The paper does not sufficiently address the robustness and transferability of the generated adversarial images. It is unclear how stable the adversarial perturbations are across different but semantically similar images.

4. AVCD is inherently a white-box method, which restricts its usage scenarios. Generating effective adversarial perturbations requires full access to the gradients of the vision encoder. But such kind of methods all have this internal issues.

### Questions
Authors are suggestted to address my concerns at the *Weakness* part as many as you can, no need to be fully addressed.

### Soundness
3

### Presentation
2

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
This paper proposes Adversarial Visual Contrastive Decoding (AVCD), an inference-time method that improves Visual Contrastive Decoding (VCD) by introducing adversarial perturbations instead of Gaussian noise to create more informative contrastive signals. The adversarially perturbed image is optimized to reduce cosine similarity with the original image, thereby generating a stronger signal for suppressing hallucinations in LVLMs. The paper includes solid theoretical analysis, encoder- and LLM-level studies, and comprehensive experiments demonstrating consistent improvements over VCD and related baselines.

### Strengths
1. Repurposes adversarial perturbations as constructive contrastive signals, which is both original and well-motivated.

2. Provides formal analysis (Theorem 1) and encoder/LLM-level verification, showing directional semantic steering rather than random degradation.

3. Evaluations across multiple benchmarks (POPE, CHAIR, MME) and models (LLaVA, Qwen-VL) demonstrate consistent gains.

4. Training-free and computationally moderate (≈1.2× VCD cost), making it appealing for real-world use.

### Weaknesses
1. The paper employs a very large adversarial budget (ε = 64–256), far beyond common adversarial settings (e.g., 4/255 or 8/255) that already induce substantial semantic variation. While this design choice may partially offset the low computational cost of using only 1–2 PGD steps and a simplified adversarial loss, the trade-off between perturbation magnitude and optimization depth (i.e., larger ε versus more PGD steps with smaller ε) is not analyzed. A clearer justification or empirical study of this balance would substantially strengthen the paper’s methodological soundness.

2. Limited adversarial formulation.
The adversarial perturbations are derived solely from the vision encoder features, which represent only a small component of the overall LVLM. It is uncertain whether this restricted gradient path adequately captures hallucination-relevant semantics. Generating perturbations based on multimodal or output-level objectives—such as semantic deviation of the LVLM’s logits using Zero-Gradient or latent-space alignment methods [1]—could produce more meaningful and semantically targeted adversarial signals (albeit at higher cost). I recommend that the authors include results under alternative adversarial objectives to substantiate this point.


3.	Insufficient benchmark coverage. The evaluation lacks results on fine-grained and more complex generative task hallucination benchmarks, which are essential for validating faithfulness and factual grounding [2,3,4]. Including a small-scale test on recent benchmarks such as MMHal-Bench [3] would significantly strengthen the empirical claims and generalizability.

4.	PGD step analysis missing. The paper does not analyze how performance scales with the number of PGD iterations (T_adv). It remains unknown whether a smaller ε combined with more steps could achieve comparable results, which would clarify whether the observed improvements stem from the adversarial directionality or simply the perturbation magnitude.

5.	Alternative contrast construction unexplored. The approach relies on contrast between the original and its adversarially distorted version. It would be informative to examine whether similar benefits arise when contrasting the original with unrelated or semantically opposite images (e.g., grayscale or blank images), to isolate the contribution of adversarial directionality from generic dissimilarity.

6.	Minor technical imprecision. The claim that “Unlike attacks designed for specific objectives like classification, PGD can optimize any given loss function” (L165) is misleading. PGD is an optimization procedure, not an objective-specific method; its flexibility arises from the chosen loss function, not from PGD itself.

[1] On Evaluating Adversarial Robustness of Large Vision-Language Models, NeurIPS 2023.

[2] Amber: An LLM-Free Multi-Dimensional Benchmark for MLLMs Hallucination Evaluation, 2023.

[3] Aligning Large Multimodal Models with Factually Augmented RLHF, ACL 2024.

[4] RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-Grained Correctional Human Feedback, CVPR 2024.

### Questions
See Weakness

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Large vision language models (LVLMs) often hallucinate objects or attributes that are not present in the image, undermining their reliability. Visual Contrastive Decoding (VCD) mitigates this by contrasting the logits from the original image against those from a noisy version, down‑weighting hallucination‑prone tokens. However, the noise used in VCD is undirected Gaussian perturbations that indiscriminately degrade features and provide a weak contrastive signal. This paper proposes Adversarial Visual Contrastive Decoding (AVCD), which replaces random noise with adversarial perturbations crafted to maximize feature dissimilarity with the original image. Adversarial images are generated via Projected Gradient Descent on the cosine similarity between CLIP (or other) vision‑encoder features of the original and perturbed image. Because these adversarial perturbations actively steer the model toward hallucinatory states rather than merely adding noise, they create a stronger negative example for contrastive decoding. During inference, AVCD computes the decoding distribution p(y|v,x) by combining the logits from the original image and the adversarial image with a hyperparameter alpha similar to VCD.

### Strengths
S1. The authors show that adversarial perturbations increase output entropy more than Gaussian noise and theoretically push the image embedding closer to hallucinated object descriptions (Proposition 1), providing intuition for why AVCD is effective.

S2. The method is applied at inference time without modifying the model.

S3. AVCD improves hallucination‑mitigation metrics and general MME scores across models

### Weaknesses
W1. Generating adversarial examples requires gradient access through the vision encoder and one or more PGD iterations. This may not be feasible for black‑box models and adds inference overhead; even though the authors report modest slowdowns (1.22×-1.43× on short responses), the cost could grow with more steps or larger images.

W2. AVCD relies on CLIP‑like vision features and a projector to compute the adversarial loss. Closed or proprietary models without accessible feature representations may not support this method directly.

W3. While AVCD shows gains on hallucination benchmarks and MME, the method involves creating large perturbations (ε up to 16 or higher) since images are internal. These perturbations might inadvertently distort fine‑grained visual cues needed for tasks such as attribute comparison or counting; more analysis of downstream effects would strengthen the work.

### Questions
Q1. ow might AVCD be applied to MLLMs without public access to their vision encoder gradients or projector weights (e.g., GPT‑4o)? Could a surrogate vision model be used, and what is the impact on performance?

Q2. Have you evaluated AVCD on tasks requiring precise attribute reasoning (e.g., “how many red cubes are stacked?”) to ensure that adversarial perturbations do not degrade performance on such tasks?

### Soundness
3

### Presentation
3

### Contribution
2
