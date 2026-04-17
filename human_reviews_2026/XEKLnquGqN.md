# Stay Centered: Semantic Barycenter Alignment for LLM Jailbreak Defense

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 6, 4

## Abstract
Jailbreak defenses in large language models (LLMs) are essential for ensuring model security, maintaining user trust, and supporting the sustainable development of AI applications. However, the most widely adopted defenses based on intent extraction currently rest on unstable foundations, marked by excessive dependence on target/safety LLM's security performance, prompts engineering, and limited explainability. These limitations render the defense architecture inherently passive, struggling to effectively counter evolving jailbreak. In this paper, we propose Semantic Barycenter Alignment (SBA), a novel defense method grounded in optimal transport (OT) theory. Specifically, we reinterpret intent extraction as a semantic projection task on a latent embedding manifold, mapping each user prompt to its class barycenter—a region models recognize more easily. In this process, we instruction-tune an LLM to extract input intents and use Sinkhorn divergence to quantify semantic alignment with target intents, measuring minimal deformation in Wasserstein space. During defense, the intent extractor serves as the upstream stage in the defense pipeline, passing intents aligned with the manifold barycenter to lightweight safety LLMs (e.g., Llama-Guard3-1B) for jailbreak detection. Empirical results show that SBA exhibits zero-shot robustness and steers intent embeddings toward the manifold barycenter of their semantic classes, reducing intra-class variance and simplifying downstream jailbreak detection. Moreover, built on a principled OT theory, SBA offers greater interpretability, removes reliance on prompt-specific heuristics, and reduces dependency on downstream LLM performance—providing a more proactive foundation for jailbreak defense. Code: \url{https://anonymous.4open.science/r/SBA-EE42}. \textcolor{red}{Warning: This paper may contain content that has the potential to be offensive and harmful.}

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose Semantic Barycenter Alignment (SBA), which utilizes OT theory to conduct intent extraction for jailbreak defense.

### Strengths
1. The derivation from OT theory to the alignment loss largely makes sense.
2. The writing is clear.

### Weaknesses
My major concern is about the experiments that verifying the effectiveness of SBA.

1. SBA configuration: It simply does not make sense to me to use Qwen3-8B or Llama3-8B as intent extractors to defend llama2 and vicuna. It is simply impossible that we can adopt a stronger LLM to defend a weaker LLM in realistic settings. 

2. The attack methods that the authors considered are classic but quiet out-of-dated. I believe it is important to test the method on newer/stronger attacks such as [1]. 

[1] Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks

3. What not consider adaptive attacks as [3] suggested. Such as explicitly optimizing the adversarial prompt to steer the output of the intent extractor.

[3] https://arxiv.org/abs/2510.09023

4. The AdvBench is quiet out-of-dated. I encourage the authors to experiment with newer datasets such as XStest [4]. Besides, it is necessary to report mis-classification rate on benign data and computation overhead.

5. If possible, consider comparing the method with more recent LLM-based defenses like [5], as all the baseline defenses shown in this paper are published before Januray 2025. 
[5] STAIR: Improving Safety Alignment with Introspective Reasoning

### Questions
Please see weakness.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a prompt-level method to defend against jailbreak attacks on LLMs. To achieve this goal, the authors leverage the optimal transport theory to extract the inherent intent from the users’ queries. In particular, the Sinkhorn divergence is employed to quantify the semantic alignment with target intents. After the translation from the original prompt to the intent, a post-guard model such as Llama-Guard 3 is inserted as the final judge interface. 

The model is pre-trained with half benign and half malicious prompt-intent pairs. When compared with several defense methods, the proposed SBA defense method demonstrates certain improvements to defend against some popular jailbreak attacks.

### Strengths
-	The writing of this paper is good. Most parts of this paper are easy to follow.
-	The authors provide a very detailed explanation of the dataset, baseline, and jailbreak attack methods.
-	The proposed method can outperform several defense baselines.

### Weaknesses
-	I kind of feel confused about the motivation of the proposed method. In general, the prompt-to-intent translation can be easily implemented with an LLM. It is hard to grasp the idea of why the OT theory is essential here.
-	Following the first comment, the authors also include the language modeling loss in the final loss function. However, the necessity of OT is not fully validated in the following two aspects:
  - There is no ablation study on removing the alignment loss.
  - Figure 4 cannot say that the alignment loss really contributes, as there are two variables in each of these three sub-figures.
-	Is the alignment a simple feature space transformation? Are there any other ways also suitable to achieve this?
-	Why use the multimodal jailbreak dataset – Jailbreak-V-28K for the malicious intent extraction?
-	The over-refusal should be tested with other datasets such as XTest.
-	The final judge model is Llama-Guard 3 or Key-word? If Llama-Guard 3 is used, it will lead to some unfair comparison, as it is already included in the method pipeline.
-	The final defense performance may largely depend on the guard performance of the lightweight Llama-Guard-3.

### Questions
See weaknesses.

### Soundness
2

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
The paper proposes Semantic Barycenter Alignment (SBA), a novel defence framework for mitigating jailbreak attacks on large language models (LLMs) by reformulating intent extraction as an optimal transport (OT) problem on a semantic manifold. Instead of relying on brittle prompt engineering or downstream classifiers, SBA learns an intent extractor that projects user prompts into canonical intent representations aligned with semantic barycenters of benign or malicious intent. The alignment is quantified using Sinkhorn divergence to ensure smooth, geometry-aware optimisation. SBA is trained via a combined loss that balances the barycentric alignment objective and a language modelling loss for fluency. Experiments across multiple jailbreak benchmarks and target models show that SBA achieves low attack success rates and strong zero-shot robustness.

### Strengths
- The paper rigorously formulates intent extraction as an optimal transport (OT) alignment between predicted and reference intents, leveraging Sinkhorn divergence for differentiable barycentric supervision. This provides a principled and theoretically grounded approach to jailbreak defence.
- Comprehensive evaluations across multiple datasets and attack types demonstrate strong empirical results. The method maintains performance under unseen jailbreaks, supporting its claims of zero-shot robustness.
- The proposed method could serve as an effective filtering defence and can be integrated with other safety mechanisms, indicating potential for broader real-world impact.

### Weaknesses
- While the OT framework is well-motivated, the paper lacks an in-depth theoretical analysis or robustness guarantees for Sinkhorn-based alignment, making it appear more heuristic in its application.
- The paper does not investigate how SBA performs under adaptive attacks where the attacker is aware of the defence, leaving uncertainty about its resilience in such settings.
- Although the Sinkhorn divergence involves matrix-level transport plans (Eq. 3–4), runtime and inference latency compared to baseline methods are not reported. The claim that SBA “does not cause noticeable inference delay” lacks empirical evidence.
- Although the authors emphasise interpretability, no study, case analysis, or quantitative semantic alignment metric is presented.
- The defence setup could be clarified by explaining how inputs are treated after detection (e.g., filtering, blocking, or flagging), to make the defence more transparent and reproducible.
- The method is promising and theoretically interesting, but would benefit from stronger theoretical analysis, broader empirical validation (including adaptive settings), and clearer exposition of the defence workflow.

### Questions
- Could the authors provide a more in-depth theoretical analysis of why OT is suitable for jailbreak detection? 
- Could the authors include runtime comparisons with baseline methods, covering both inference and training costs? Reporting wall-clock latency, GPU hours, and memory consumption would help readers assess the practical feasibility of deploying SBA in real-world applications.
- Could the authors rigorously evaluate adaptive attacks? Specifically, to what extent could an attacker aware of SBA’s defence mechanism modify prompts to evade detection? Including results under white-box or grey-box settings would strengthen claims of robustness.

### Soundness
3

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
2

### Summary
The paper introduces Semantic Barycenter Alignment (SBA), a theoretically grounded framework for defending large language models against jailbreak attacks. The core idea redefines intent extraction as a semantic projection problem on a latent manifold: each user prompt is projected toward the semantic barycenter of its intent category—benign or malicious—under optimal transport (OT) geometry using Sinkhorn divergence. SBA uses an instruction-tuned intent extractor to map complex prompts into concise intent statements. These are then evaluated by lightweight safety models. Experiments show strong zero-shot robustness and significant ASR reduction across diverse jailbreak datasets and multiple target models.

### Strengths
1.SBA introduces a novel geometric formulation of jailbreak defense using optimal transport theory, it establishes a bridge between OT theory and LLM safety.
2.Extensive experiments across multiple jailbreak datasets and models validate the generalization of SBA. The results outperform the baseline, demonstrating the effectiveness of the SBA method. 
3.SBA’s model-agnostic, modular design allows integration with existing LLM defense pipelines without retraining large models.

### Weaknesses
1. The paper presents a method for jailbreak detection, but it does not consider testing in real-world scenarios, such as WildJailbreak[1].
2. The SBA method assumes access to high-quality, low-variance canonical intents to approximate barycenters. This assumption may limit scalability if class distributions are noisy or ambiguous.
3. How would SBA handle multi-intent prompts or contextually benign but stylistically adversarial inputs? There is limited discussion on how SBA behaves when intent classes overlap semantically or when new intent types emerge.

[1] WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models

### Questions
1.	Hard to read, the formatting of the paper and the images need to be more aesthetically pleasing.
2.	The Sinkhorn divergence adds non-trivial computational overhead during fine-tuning; runtime performance for large-scale deployment is not discussed. 
3.	The paper uses curated datasets to approximate barycenters; how sensitive is SBA to the semantic diversity of these anchors?Would dynamically updating barycenters improve adaptivity?

### Soundness
3

### Presentation
2

### Contribution
2
