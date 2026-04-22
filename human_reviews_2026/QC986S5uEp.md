# Surgical Safety Repair: A Parameter-Isolated Approach to Correcting Harmful Fine-tuning

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 6

## Abstract
Fine-tuning is a fundamental technique for adapting Large Language Models (LLMs) to specialized tasks, yet it can unexpectedly compromise the model's safety alignment even when using datasets perceived as benign. However, many existing defenses are limited by their dependence on a pre-computed safety vector, typically requiring access to both the base model and a safety-aligned version. Furthermore, the safety alignment achieved by such methods often degrades to simplistic refusal, instead of nuanced, helpful responses. 
In this paper, we introduce Surgical Safety Repair (SSR), a novel post-hoc framework designed to precisely correct harmful behaviors in fine-tuned models while maximally preserving their utility. SSR operates in an automated three-stage pipeline: it first leverages a diagnostic dataset to prompt the compromised model to reveal its safety flaws, constructing a model-specific corrective dataset. Then, it employs gradient-based attribution to localize a targeted set of LoRA parameters responsible for harmful outputs. Finally, it performs a parameter-isolated update based on the corrective dataset, using a dual-objective loss to unlearn harmful responses and steer the model towards safe and constructive ones. Experiments on diverse models demonstrate that SSR reduces the harmfulness score to below 5\% while largely preserving the original capabilities of model, with minimal performance drop on downstream benchmarks such as GSM8K. 
Furthermore, SSR guides the model to generate high-quality refusals, fostering a deeper and more nuanced safety alignment beyond mere response suppression.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes the SSR framework which attempts to increase safety alignment by targeted retraining of LoRA weights. The authors do this in three steps- 
1. Manually curate a dataset from existing datasets and custom harmful prompts.
2. Run inference on this dataset to detect LoRA activations most associated with harmful outputs.
3. Re-align the detected LoRA weights with dual-objective loss that perform unlearning on the harmful responses and learning on safe responses.

Experiments are run across mutliple general task and safety task benchmarks with a wide variety of models. The authors also compare against state-of-the-art to demonstrate that their method can beat them to create highly resilient models.

### Strengths
1. The motivations of the paper are relevant and timely.
2. The solution is simple and easy to understand.
3. The results, especially compared to the state-of-the-art in terms of harmfulness, are very good.

### Weaknesses
1. Requires major revisions to make it more polished. Clearly, there are errors that make it seem like it is in a draft format.
2. A lot of details to make it reproducible are missing, as well as making the results appear questionable.

### Questions
A lot of the important content is missing or misformatted - for example
1. What is the exact threat model here? In other words, please elaborate what type of safety alignment has the model undergone/ has not undergone, what type of SFT was done, etc.
2. Every relevant detail for the curated dataset is completely missing, it is not at all reproducible or no insights can be gained from this.
3. Line 167 - is the model already poisoned? Ties with point 1.
4. Line 191 - What is meant by "complexities in KL divergence"? Are the authors talking about a specific way it was applied? Or the formula itself?
5. How is Harmful Propensity Threshold applied exactly? I can infer that it get the top Tau percent after reading for a while, but it should be clearly mentioned somewhere.
6. In the experimental section, why are only 5000 datapoints used?
7. Why does SST5 decrease so much in accuracy? 
8. For Table 2 - shouldn't the increase in malicious datapoints increase in harmfulness too? Why does this only occur for two of the frameworks but not for the others?

There are many such questions all over the paper, all of which make it hard to understand the depth of the implementations and the fairness of the experimental results.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a post-hoc framework named Surgical Safety Repair (SSR) designed to correct harmful behaviors in fine-tuned Large Language Models (LLMs). The method operates via a three-stage pipeline: Diagnosis, Attribution, and Surgery. The authors claim that SSR can reduce the Harmfulness Score to below 5% while having a minimal impact on the model's original task performance

### Strengths
1. The core idea of treating safety repair as a parameter-isolated "surgical" procedure is novel. The automated three-stage pipeline is logically clear and well-designed.

2. The SSR framework does not depend on access to the original base model or a pre-aligned safe version. This makes it more practical for real-world applications compared to many state-of-the-art methods that rely on model arithmetic (e.g., Safe-LoRA, SafeDelta).

3. As highlighted by the authors, the method requires minimal computational resources (detailed in Appendix D), demonstrating its potential for efficient safety alignment in resource-constrained environments.

### Weaknesses
1. The experimental setup is severely inadequate and lacks evaluation on critical attack scenarios. The authors only simulate a **mixed data poisoning** scenario, where 500 malicious samples are added to 4,500 benign ones. This is far from the only, or even the most challenging, attack vector. A paper on safety repair should validate its method across a broader and more demanding range of scenarios, such as:

	*  Where the model is fine-tuned on a purely harmful dataset (e.g., the PureBad scenario from the original Safe LoRA paper).

	* Where harmful behaviors are activated by specific triggers, as defined in the HEx-PHI paper. The absence of experiments in these key scenarios raises serious questions about the robustness and generalization capabilities of SSR.


2. The paper's novelty is questionable, and the comparison with key related works is critically missing.

	* The core concept of locating and editing parameters via attribution is highly similar in principle to recent works like **SafeEdit[1]**. The authors fail to clearly articulate the unique contributions of SSR over these similar approaches.

	* SSR's applicability is restricted to LoRA-tuned models, whereas methods like SafeEdit are applicable to full-parameter fine-tuned models. This further limits the significance of SSR's contribution.

	* The paper's claims of "efficiency" and "minimal intervention" are not substantiated by comparisons with relevant state-of-the-art works. The authors have omitted comparisons with a series of highly relevant and efficient safety alignment methods, such as SafeAligner[2] (a test-time method) and SafetyLock[3] (which requires almost no GPU resources). This selective comparison undermines the soundness of the paper's conclusions and makes its claimed SOTA status impossible to verify.

3. The SSR framework relies on an unsubstantiated core assumption. The entire method is built on the assumption that "harmful behaviors learned during poisoned fine-tuning are... disproportionately encoded within a sparse subset of the newly introduced LoRA parameters". However, the authors provide no empirical evidence to support this claim. The representation of harmful knowledge could very well be highly entangled with useful knowledge within the parameters, especially under sophisticated poisoning strategies. This fragile assumption may be the root cause of SSR's performance degradation. In Table 1, while SSR reduces harmfulness, its Fine-tuning Accuracy (FA) on GSM8K (66.5%) and PubMedQA (77.7%) is notably lower than some baselines, such as SafeDelta (75.7% and 77.5%, respectively). This strongly suggests that SSR's "surgery" may be excising useful knowledge critical for downstream tasks along with the harmful parameters.


4. I have doubts about the reproducibility of the baseline results, especially for Safe LoRA. In this paper's experiments, Safe LoRA appears almost completely ineffective (e.g., HS remains above 70% in Table 1 and as high as 66.92% on AdvBench in Table 4). However, according to the original Safe LoRA paper (Hsu et al., 2024), it can drastically reduce the Attack Success Rate (ASR) from 94.85% to 6.35% on a model fine-tuned on the PureBad dataset. This stark discrepancy suggests that the authors' experimental reproduction may be flawed, which renders all claims of superiority over baselines untrustworthy.

[1] Detoxifying Large Language Models via Knowledge Editing

[2] SafeAligner: Safety Alignment against Jailbreak Attacks via Response Disparity Guidance

[3] Locking Down the Finetuned LLMs Safety

### Questions
1. In your experiments, you used a learning rate of $2 \times 10^{-5}$ for the initial SFT (poisoning) stage (Section 4.1) 14and a rate of $1 \times 10^{-5}$ to $2 \times 10^{-5}$ for the SSR repair stage (Appendix C)15. This is one to two orders of magnitude lower than the learning rates used in other LoRA-based editing/repair methods (e.g., Safe LoRA uses $10^{-3}$). Could you justify the choice of such a low learning rate for the repair process? Was this choice critical for achieving your results, and how does it affect the method's general applicability across different tasks without extensive tuning?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Surgical Safety Repair (SSR), a post-hoc, parameter-isolated method that updates safety in LoRA-fine-tuned LLMs without a reference model. It is structured into three stages: (1) constructing a model-specific corrective dataset by prompting the compromised model with diagnostic queries, (2) employing gradient-based attribution to identify the top 25% of LoRA parameters responsible for harmful behaviors, and (3) applying a dual-objective loss (forgetting + steering) to selectively update only those parameters.

### Strengths
- The paper provides interesting insights that safety-related behavior is concentrated in a small subset of LoRA units, enabling targeted edits that restore safety with minimal accuracy loss.
- The experiments demonstrate consistent effectiveness across diverse datasets and model families, indicating robust generalization.

### Weaknesses
- In Stage 1, the method uses a judge model to identify harmful outputs and incorporates this signal during training, whereas baseline methods (e.g., Safe-LoRA, SafeMERGE, SafeDelta) do not receive the same supervision. This raises a potential fairness concern.
- Several important details are missing from the paper; these are outlined in the Questions section below.

### Questions
Q1. Which LLMs were used to generate the diagnostic dataset?

Q2. How is the Harmful Propensity Threshold (τ) determined? In Appendix C, the value varies across datasets.

Q3. Could you explain in more detail Stage 3 (Parameter-Isolated Safety Steering)?

   - Does it go through the diagnostic dataset for multiple epochs until the model reaches the threshold?

Q4. Could you explain further why SafeDelta was only tested on Llama?

Q5. Do you have results for GSM8K or PubMedQA similar to those for SST-5 in Table 4?

Q6. In Appendix B (Table 6), you report over-refusal (OR) and successful refusal (SR) rates for SSR but not for baseline methods. Could you provide OR and SR rates for all baselines?


Formatting issue:
- The paper uses a non-standard citation format throughout:
   - Current: "SafeMERGE Djuhera et al. (2025)"
   - Standard: "SafeMERGE (Djuhera et al., 2025)"

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
2

### Summary
This paper proposes SSR (Surgical Safety Repair), a framework designed to restore safety alignment in fine-tuned LLMs without requiring access to the original base model or a separately safety-aligned model. SSR works by (1) diagnosing harmful behaviors, (2) locating harmful LoRA parameters, and (3) updating only those parameters via a targeted unlearning-and-steering process. The authors argue that this approach achieves high safety recovery while preserving the model’s original utility, outperforming prior safety restoration methods.

### Strengths
1. Practical and deployment
* Unlike many existing methods that assume access to both base and safe models, SSR operates on a single compromised model, which is realistic for industry and proprietary settings. The method is lightweight because it relies on LoRA modification instead of full-model retraining.
2. Strong empirical results
* Harmfulness score reduced from ~77% → ~2%.
* Downstream task performance is largely preserved.
* Works consistently across multiple models (Llama3, Qwen2, Gemma2) and tasks (GSM8K, SST-5, PubMedQA).
3. Conceptual novelty
* Instead of globally modifying weights, the method isolates and edits only harmful LoRA units, which is a more surgical and controlled form of safety repair.
* Uses a simple but effective dual-objective loss combining unlearning (gradient ascent) and safety steering (gradient descent).

### Weaknesses
1. Strong dependence on LoRA assumption
* SSR only works directly for LoRA-based fine-tuned models. SSR cannot be applied to fully fine-tuned models unless additional steps are introduced. The paper claims “general applicability,” but does not demonstrate it experimentally.
2. Diagnostic dataset construction cost
* Although the paper claims a 100-example dataset is sufficient, the dataset requires manual safe-response curation, which introduces human cost, cultural bias, and reproducibility challenges.

### Questions
* Does modifying only the harmful LoRA units interfere with other task-specific capabilities encoded in those same parameters?
* If the diagnostic dataset reflects a specific culture or safety norm, does the model become biased toward that norm after repair?
* How stable is the attribution process under prompt variation?

### Soundness
3

### Presentation
2

### Contribution
2
