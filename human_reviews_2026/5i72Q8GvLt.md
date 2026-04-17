# Free Lunch Alignment of Text-to-Image Diffusion Models without Preference Image Pairs

- Decision: Reject
- Scores: 6, 6, 4

## Abstract
Recent advances in diffusion-based text-to-image (T2I) models have led to remarkable success in generating high-quality images from textual prompts. However, ensuring accurate alignment between the text and the generated image remains a significant challenge for state-of-the-art diffusion models. To address this, existing studies employ reinforcement learning with human feedback (RLHF) to align T2I outputs with human preferences. These methods, however, either rely directly on paired image preference data or require a learned reward function, both of which depend heavily on costly, high-quality human annotations and thus face scalability limitations. In this work, we introduce Text Preference Optimization (TPO), a framework that enables "free-lunch" alignment of T2I models, achieving alignment without the need for paired image preference data. TPO works by training the model to prefer matched prompts over mismatched prompts, which are constructed by perturbing original captions using a large language model. Our framework is general and compatible with existing preference-based algorithms. We extend both DPO and KTO to our setting, resulting in TDPO and TKTO. Quantitative and qualitative evaluations across multiple benchmarks show that our methods consistently outperform their original counterparts, delivering better human preference scores and improved text-to-image alignment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Text Preference Optimization (TPO) for aligning text-to-image (T2I) diffusion models without human preference image pairs. It creates matched/mismatched text pairs via LLM-based prompt editing under four principles (content, attribute, spatial, contextual). Two variants, TDPO and TKTO, adapt DPO/KTO to diffusion training. Experiments across several benchmarks indicate consistent gains over Diffusion-DPO/KTO while remaining annotation-free.

### Strengths
1. The idea of "free-lunch" alignment (re-using existing caption data + LLM perturbations) is creative, practical, and addresses real scalability issues in diffusion-model alignment.
2. The formulation of TDPO/TKTO directly mirrors established DPO/KTO objectives, but shifts preference comparison from outputs (images) to inputs (text prompts).
3. The method has been evaluated on four diverse datasets with multiple quantitative metrics. It achieves consistent improvement across most metrics and datasets. Qualitative comparisons convincingly show better text-image faithfulness, and ablation studies on editing principles and implicit-preference correlation provide solid analytical depth.

### Weaknesses
1. The pipeline is essentially "synthetic preference data + existing DPO/KTO objective." Although effective and creative, it feels more like an engineering simplification than a fundamentally new learning principle.
2. The framework depends on a single budget-constrained LLM for prompt editing, which may introduce stylistic or semantic biases and restrict negative-sample diversity.

### Questions
1. Can authors please clarify how LLM-generated and human-labeled prompts compare in alignment performance, possibly via a short correlation or qualitative analysis. This would clarify whether TPO truly approximates human alignment behavior or just benefits from larger-scale synthetic coverage.
2. I have concerns with the dependence on LLM prompt quality. Alignment quality is tied to how well the Gemini 2.0 Flash edits produce meaningful yet challenging negatives. Are there any quantitative analysis of negative-sample diversity?

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
5

### Summary
The authors propose TPO, a novel T2I alignment framework that perturbs the prompt with LLM to form a prompt level preference pair and aligns the Diffusion T2I model via prompt level preference. TPO achieved superior performance across multiple automatic feedback benchmarks on SD1.5 when integrated to DPO and KTO. In addition, TPO disentangles the need for image preference data in T2I alignment, further enhancing alignment efficiency.

### Strengths
* The TPO algorithm is a novel contribution for Diffusion T2I alignment leveraging prompt level preference rather than image level.
* The TPO algorithm achieved superior performance with less requirements on training data.

### Weaknesses
* My major concern is the robustness of TPO. The experiments are conducted only on SD1.5, one of the small scale open source T2I diffusion models. Comparison with DPO on SDXL are expected to demonstrate the robustness of this interesting algorithm.

* Evaluation from human or large VLM / MLLM is needed. The automatic evaluation metrics used in Table 1 and Table 2 are CLIP and fine-tuned CLIP/BLIP variants while SD1.5 uses CLIP as text encoder. The training recipe of TPO could potentially boost the similarity of the generations and the CLIP embedding and hence significantly improves performance on these related metrics. It would be good to see some evaluation that relies less on CLIP.

* In Table 1, there's a typo "PartyPrompt", please revise for "Parti-Prompt"

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

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
This paper introduces Text Preference Optimization (TPO), a framework to align text-to-image diffusion models without requiring human-annotated image preference pairs. The method uses a Large Language Model (LLM) to generate mismatched text prompts from original captions, creating text preference pairs (matched vs. perturbed) to train the model. The authors extend DPO and KTO to TDPO and TKTO, claiming state-of-the-art results on benchmarks like HPSv2 and Pick-a-Pic. While the core idea of "free lunch" alignment is novel, the paper's claims are significantly undermined by critical flaws in the experimental setup and evaluation.

### Strengths
1. Original Idea: The concept of aligning models via automatically constructed text preferences, rather than image preferences, is novel and creative.
2. Practical Motivation: The work tackles a key scalability bottleneck in T2I alignment, offering a potentially cost-effective solution.
3. General Framework: The approach is model-agnostic and can be integrated with various preference optimization algorithms (DPO, KTO), which is a strong design choice.

### Weaknesses
- Unconvincing Baseline Performance: The dramatic drop in performance for the Diffusion-DPO baseline (e.g., HPS score) is a major red flag. It suggests either a flawed implementation, inappropriate hyperparameters, or an unfair comparison setup, which invalidates the claimed superiority of the proposed methods.
- Opaque Training Process: The paper provides no insight into the training dynamics. There are no loss curves, convergence plots, or monitoring metrics on held-out benchmarks. This makes it impossible to assess if the models are learning the intended preferences or if the training is stable.
- Potentially Unfair Comparison: The baselines (Diffusion-DPO/KTO) are designed for human image preferences. Comparing them against a method using synthetic text preferences, where the negatives are generated by the same LLM pipeline, may not be a fair or meaningful comparison. The paper should have included a baseline using the same synthetic text to generate a negative image for a more controlled comparison.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
3
