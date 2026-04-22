# Contamination Detection for VLMs Using Multi‑Modal Semantic Perturbations

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 4

## Abstract
Recent advances in Vision–Language Models (VLMs) have achieved state-of-the-art performance on numerous benchmark tasks. However, the use of internet-scale, often proprietary, pretraining corpora raises a critical concern for both practitioners and users: inflated performance due to \emph{test-set leakage}. While prior works have proposed mitigation strategies such as decontamination of pretraining data and benchmark redesign for LLMs, the complementary direction of developing detection methods for \emph{contaminated VLMs} remains underexplored. To address this gap, we deliberately contaminate open-source VLMs on popular benchmarks and show that existing detection approaches either fail outright or exhibit inconsistent behavior. We then propose a novel simple yet effective detection method based on \textit{multi-modal semantic perturbation}, demonstrating that contaminated models fail to generalize under controlled perturbations. Finally, we validate our approach across multiple realistic contamination strategies, confirming its robustness and effectiveness. The code and perturbed dataset are released here: \href{https://github.com/jadenpark0/mm-perturb}{https://github.com/jadenpark0/mm-perturb}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the underexplored problem of detecting test-set contamination in Vision–Language Models (VLMs) — as opposed to merely avoiding it via decontamination. Instead of text‐only perturbations used for LLMs, the authors propose a multi-modal semantic perturbation pipeline that manipulates the visual scene while preserving composition, thus minimally shifting input semantics yet changing the ground-truth answer. They contaminate LLaVA and Qwen2-VL under controlled fine-tuning regimes and show that existing detection baselines (e.g., shared likelihood, guided prompting, circular eval, choice confusion) fail to reliably track contamination. Their method exhibits high practicality (black-box only), reliability across fine-tuning types, and consistent monotonic signal w.r.t. contamination degree. They further validate on natural counterfactuals (NaturalBench), larger models (LLaVA-13B), and even simulated pretraining leakage, demonstrating generality.

### Strengths
The paper poses a non-generic, well-defined, and deep problem — not merely “robustness” or “performance drops,” but specifically how to detect contamination without assuming prior access to clean references or to the training corpus. The authors not only identify but formalize the essential requirements (practicality, reliability, consistency) and then prove why existing methods violate them, rather than merely benchmarking blindly. The methodology is elegant but grounded — the perturbation pipeline is semantically meaningful, not superficial image corruption — and their experimental design is unusually thorough and fair (multiple contamination strategies, ablations, real counterfactuals, automated filtering, alternative captioners).

### Weaknesses
The approach implicitly depends on the availability of strong controlled semantic editors (Flux + GPT-4o + ControlNet); although ablations with Molmo and automated filtering are shown, the method’s feasibility still assumes future generative tools remain capable and unbiased.

The evaluation domain is limited to visual-grounded multiple-choice VQA benchmarks (e.g., RealWorldQA, MMStar); it is argued that free-form QA is possible, but no concrete evidence is provided. 

While the method is claimed fully black-box, it still rests on the assumption that perturbed samples are truly non-harder than originals: a subtle but critical assumption, mainly supported indirectly via model behavior rather than formal difficulty guarantees.

### Questions
Your framework assumes that the perturbed version is of comparable or lower difficulty than the original, but this is only inferred indirectly via clean model behavior. How do you enforce or guarantee this assumption beyond empirical observation? Could there exist cases where the perturbation unintentionally increases difficulty and generates false positives?

Your method relies on having access to a generative model strong enough to produce controlled, faithful semantic perturbations. In lower-resource or restricted deployment regimes, do you still consider your method “practical” (Req. 1)? What is your definition of practicality beyond using “a black box”?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces multi-modal semantic perturbation as a detection framework for identifying data contamination in VLMs. The method detects contamination by measuring the performance degradation of a model on perturbed samples compared to the original. Experiments show the proposed detection setting is reliable.

### Strengths
1. Well-motivated problem definition.
2. Extensive comparisons across contamination settings (fine-tuning, LoRA, pretraining) and baselines demonstrate consistent detection performance.
3. The detection approach is straightforward by comparing performance on the original vs. perturbed input.

### Weaknesses
Perturbation generation requires LLM and diffusion inference per sample, implying high computational cost. This method could be a bit difficult to generalize due to scalability and efficiency constraints.

### Questions
See weakness. How computationally expensive is the proposed method?

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
This paper studies test-set leakage detection for VLMs. The authors deliberately “contaminate” open-source models like Qwen2-VL by fine-tuning them on benchmark data and show that prior text-based contamination detection methods fail. They then propose a method called multi-modal semantic perturbation, which uses GPT-4o to rewrite captions and Flux + ControlNet to generate visually altered images that subtly change the correct answer. If a model performs well on the original but fails on the perturbed version, it is flagged as potentially contaminated.

### Strengths
1. The paper targets data leakage in multimodal models, which is an important issue.

2. It provides systematic experiments across multiple contamination types and model sizes, with quantitative comparisons to several baselines.

3. The overall framework is simple and easy to understand, and the perturbed benchmark will likely be useful to the community.

### Weaknesses
**1. Inaccurate and heavy generation pipeline with high manual cost.**

I find the proposed “detection pipeline” is simple, unnecessarily complex, and fragile for what it aims to do. It requires GPT-4o to generate dense captions, a Flus plus ControlNet for Canny-guided editing (a very crude method for composition-preserved image generation), and finally human filtering to discard failed generations. 

In addition, the method needs manual filtering to remove a large portion of the data. In particular, RealWorldQA is reduced from 765 to 440 samples, and MMStar from 1500 to 495, over 1/3 is filtered.

To me, this already contradicts their claim of practicality: a method that needs strong proprietary models and heavy human effort cannot serve as a scalable detector. The “clean vs. contaminated” separation is only visible after extensive manual curation.

**2. The detection signal is confounded with robustness, not contamination.**

The method assumes that if a model fails on perturbed images, it flags contamination. But the paper itself shows that many perturbations accidentally change task difficulty. In Figure 3, the perturbation even enlarges the visual cue (a speed limit sign), making the new image easier.

In other cases (Figure 4), the perturbed image drifts too far from the original, so even a clean model can fail while a contaminated one may still succeed. In my view, the metric mainly captures distribution shift sensitivity rather than true memorization, so the signal is not clean. Essentially, this is caused by the instability of the perturbation method being constructed.

**3. Unrealistic and self-serving contamination setup.**

The contamination experiment is extremely idealized: directly fine-tune the model on the full test set for one to three epochs. This guarantees that the model has seen every evaluation item. Real leakage in practice is much subtler: partial overlap, web-scale data, paraphrased variants, and it’s unclear whether the proposed metric would still detect that. 

Therefore, while the results look strong, the paper proves only that the method works under a trivially detectable contamination pattern. It’s closer to a sanity check than to a general detection method.

**4. Limited applicability and reliance on strong visual grounding.**

The method only works for benchmarks where the image strictly determines the answer. As they note themselves, “if a question can be answered without visual input, perturbing the image is meaningless”. This excludes open-ended captioning, grounded reasoning, or OCR-heavy tasks. So the claimed “general framework for contamination detection” is actually limited to multiple-choice VQA tasks with strong visual dependency.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

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
The authors propose a novel detection framework based on multi-modal semantic perturbations, which involves generating new test samples by subtly altering an image’s content (while preserving overall semantics) so that a model which merely memorized the original image-text pair will fail on the perturbed input. Experiments show that existing detection methods often fail or give inconsistent results on VLMs, whereas the proposed perturbation-based approach consistently flags contaminated models across diverse fine-tuning settings and degrees of contamination, satisfying key requirements of reliability, practicality, and consistency.

### Strengths
Proposes an original and practical contamination detection method tailored for VLMs using multi-modal semantic perturbations.

Demonstrates strong technical quality through extensive and controlled experiments across diverse settings.

Clear presentation with significant implications for reliable and fair evaluation of vision-language models.

### Weaknesses
While the proposed method is practically useful, it primarily consists of integrating existing tools—LLMs for captioning and diffusion models for image editing—into a contamination detection pipeline. As such, the technical novelty is relatively limited. The idea of testing generalization via perturbed inputs is well-established, and the paper applies this concept to the multi-modal setting without introducing fundamentally new algorithms or theoretical insights.

The description of the core methodology in Section 4 is relatively high-level and omits several important implementation and design details. For example, while the use of GPT-4o and ControlNet is outlined, it is unclear how semantic alignment between the new answer and generated image is ensured, or how failure cases are handled systematically beyond manual filtering.

The core contamination detection criterion—declaring a model contaminated if it answers perturbed samples incorrectly while answering original ones correctly—is intuitive but lacks technical precision. The paper does not clarify whether this evaluation is done at the sample level, across aggregate performance metrics, or via some probabilistic threshold.

Since the proposed method relies on detecting failures under semantic perturbations, it would be informative to compare against standard OOD generalization or robustness baselines. This would help disentangle contamination from general lack of robustness.

### Questions
same as weakness

### Soundness
2

### Presentation
2

### Contribution
2
