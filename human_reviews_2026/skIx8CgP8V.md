# Noise Augmented Fine Tuning for Mitigating Hallucinations in Large Language Models

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 2, 6

## Abstract
Large language models (LLMs) often produce inaccurate or misleading content—hallucinations. To address this challenge, we introduce Noise-Augmented Fine-Tuning (NoiseFiT), a novel framework that leverages adaptive noise injection based on the signal-to-noise ratio (SNR) to enhance model robustness. Our contribution is threefold. First, NoiseFiT selectively perturbs layers identified as either high-SNR (more robust) or low-SNR (potentially under-regularized) using a dynamically scaled Gaussian noise. Second, we further propose a hybrid loss that combines standard cross-entropy, soft cross-entropy, and consistency regularization to ensure stable and accurate outputs under noisy training conditions. Third, a theoretical analysis proposed shows that adaptive noise injection is both unbiased and variance-preserving, providing strong guarantees for convergence in expectation. Moreover, empirical results on multiple test and benchmark datasets demonstrate that NoiseFiT significantly reduces hallucination rates, often improving or matching baseline performance in key tasks. These findings highlight the promise of noise-driven strategies for achieving robust, trustworthy language modeling without incurring prohibitive computational overhead. We have publicly released the fine-tuning logs, benchmark evaluation artifacts, and source code online at W&B, Hugging Face, and GitHub, respectively, to foster further research, accessibility and reproducibility.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a fine-tuning framework called NoiseFiT, which enhances model robustness and reduces hallucinations by injecting adaptive Gaussian noise into specific Transformer layers during fine-tuning and employing a hybrid loss function (cross-entropy loss + soft-target loss + consistency loss). Experimental results across multiple tasks and models demonstrate the effectiveness of the method.

### Strengths
1. The method combines multiple regularization techniques (soft-target loss and consistency loss) to form a unified training objective.

2. It has been extensively validated across multiple model families and tasks, producing convincing results.

### Weaknesses
1. The idea of merging multiple losses is not novel, but it appears to be the main innovation of the paper.

2. Limited dataset size and diversity: The training set contains only 832 samples and consists of synthetic data. Although the authors emphasize it as “simple and effective,” this may still affect the generalization ability of the method.

3. Insufficient comparison with other methods: The paper mainly compares with BaseFiT (fine-tuning without noise) and lacks thorough comparison with other advanced hallucination mitigation methods such as RAG, RLHF, and Self-Consistency.

4. Hyperparameter sensitivity: The paper mentions that noise intensity, layer selection, and other hyperparameters significantly affect results and need to be adjusted for different models.

5. Hallucination evaluation depends on external models: Some hallucination assessments use GROK 3.0 as the “judge,” which may introduce evaluation bias.

6. Theoretical analysis is rich but somewhat lengthy: The appendix contains extensive theoretical content, some of which is not closely related to the core method.

### Questions
1. The idea of merging multiple losses is not novel, but it appears to be the main innovation of the paper.

2. Limited dataset size and diversity: The training set contains only 832 samples and consists of synthetic data. Although the authors emphasize it as “simple and effective,” this may still affect the generalization ability of the method.

3. Insufficient comparison with other methods: The paper mainly compares with BaseFiT (fine-tuning without noise) and lacks thorough comparison with other advanced hallucination mitigation methods such as RAG, RLHF, and Self-Consistency.

4. Hyperparameter sensitivity: The paper mentions that noise intensity, layer selection, and other hyperparameters significantly affect results and need to be adjusted for different models.

5. Hallucination evaluation depends on external models: Some hallucination assessments use GROK 3.0 as the “judge,” which may introduce evaluation bias.

6. Theoretical analysis is rich but somewhat lengthy: The appendix contains extensive theoretical content, some of which is not closely related to the core method.

### Soundness
2

### Presentation
2

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
This paper proposes Noise-Augmented Fine-Tuning (NoiseFiT), a framework for reducing hallucinations in large language models. The approach selectively injects adaptive Gaussian noise into transformer layers during fine-tuning based on signal-to-noise ratios (SNR). The method combines (1) SNR-based layer selection identifying either high-SNR or low-SNR layers, (2) adaptive noise scaling using robust statistics and model uncertainty, and (3) a hybrid loss combining cross-entropy, soft cross-entropy (knowledge distillation), and consistency regularization. Experiments across LLaMA, Qwen, Gemma, and Mistral models on multiple benchmarks show modest improvements in hallucination reduction

### Strengths
- Addresses the important problem of hallucinations in LLMs
- Provides theoretical analysis of noise injection properties
- Comprehensive experimental evaluation across multiple model families (LLaMA, Qwen, Gemma, Mistral) and benchmarks (GPQA, MUSR, IFEval, BBH, MATH, MMLU-Pro, HaluEval, TruthfulQA) with a lot of supplementary material

### Weaknesses
- Modest and inconsistent improvements: many gains in Table 1 are small, BaseFiT sometimes wins, best Mistral result shows only 4.74 point improvement on HaluEval (47.60→52.34). Results are highly variable across configurations. 
- 832 synthetic training samples from GROK 3.0, test evaluation also uses LLM judge, custom test set limited to 208 prompts. Raises serious concerns about generalizability and validity
-  High hyperparameter sensitivity is a known issue, and the authors admit that "per-task tuning" is encouraged, which severely limits the practical utility of the model. The "recipe" in Appendix C is a heuristic approach.

### Questions
- Why not include direct comparisons with other methods of dealing with hallucinations mentioned in the relevant work? 
- How does the method perform on standard, non-synthetic fine-tuning datasets (e.g., real instruction-following data)? The 832 synthetic samples raise serious generalizability concerns
- Why do certain categories (at Tables E2-E6) show degradation with fine-tuning, and does NoiseFiT exacerbate or mitigate this?
- Can you provide human evaluation on a substantial sample to validate the LLM judge assessments?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes NoiseFiT, a framework designed to reduce hallucinations in LLMs during fine-tuning. NoiseFiT selectively injects adaptive Gaussian noise into high-SNR or low-SNR transformer layers. Noise scaling incorporates hidden-state median/MAD statistics and model uncertainty. A hybrid loss is calculated blending clean cross-entropy, soft cross-entropy via temperature-scaled teacher logits, and consistency regularization across two noisy passes. Experiments across model families show reduced hallucinations and improved or preserved benchmark performance relative to BaseFiT fine-tuning.

### Strengths
- The SNR-guided layer selection is a well-motivated heuristic. This matters for efficiency, as it avoids perturbing the entire model.
- This paper tests its method across a diverse set of model architectures (LLaMA, Qwen, Gemma, Mistral) and sizes. This supports the generality of the approach.

### Weaknesses
- The custom fine-tuning dataset is too small (832 samples) and synthetically generated by GROK. This raises concerns about whether NoiseFiT is simply mimicing the generated few samples and whether the gains would transfer to larger, more diverse, human-curated fine-tuning datasets. The custom test set is evaluated by GROK. Using an LLM to evaluate LLM hallucinations, especially when the training and test data came from the same LLM judge, would introduce significant risks of confounding variables and evaluation bias.
- The empirical results of NoiseFiT are exclusively compared against BaseFiT, which is the standard fine-tuning. In related work the authors discusses hallucination mitigation techniques like RAG, RLHF, and self-consistency, but did not compare NoiseFiT against them. Though NoiseFiT and those post-training techniques are claimed to be complementary, it should be proved by some extra ablation studies. Besides, many training-time hallucination allevitation methods are not discussed [1, 2].

[1] Hallucination Detection and Hallucination Mitigation: An Investigatcion

[2] Logit Space Constrained Fine-Tuning for Mitigating Hallucinations in LLM-Based Recommender Systems

### Questions
- What is the total fine-tuning time (wall-clock) per model or per experiment? The paper notes that NoiseFiT adds a negligible training-time cost beyond a second (noisy) forward pass. However, the full method requires one clean pass and two independent noisy passes ($L_{SOFT}$ and $L_{CONSISTENCY}$), implying a ~3x increase in forward-pass computation. This discrepancy should be clarified.

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
4

### Summary
This paper proposes Noise-Augmented Fine-Tuning, a method that improves the robustness of large language models by injecting controlled noise during fine-tuning. The approach aims to enhance generalization—especially on out-of-distribution or noisy inputs—without hurting performance on clean data. Results show consistent gains over standard fine-tuning, particularly in challenging categories like geography and history, while maintaining strong performance on in-distribution tasks. The method offers a simple, plug-and-play way to make fine-tuned models more reliable in real-world conditions.

### Strengths
1. The proposed NoiseFit method and the SNR-based layer selection approach are relatively new and not investigated before.
2. Experiments on various datasets and base models demonstrate the effectiveness of the proposed method. 
3. The mathematical analysis in this paper is rigorous, well-structured, and provides strong theoretical support for the proposed method.
4. The paper is well-written and easy to follow.

### Weaknesses
While the authors honestly acknowledge several limitations of their work, recognizing these issues does not, by itself, mitigate their impact on the method's applicability or validity.

1. The current training set is too small. Experiments on larger scale datasets are required to demonstrate the effectiveness and generalization capability of the proposed method.
2. Although the paper compares its method to BaseFiT, it lacks a systematic evaluation against other advanced strategies designed to mitigate hallucinations, such as those mentioned in the related works section.
3. The proposed method seems a general finetuning method for robustness. The paper lacks a clear analysis linking the proposed noise-augmented fine-tuning to hallucination reduction.
4. The proposed method is sensitive to hyperparameters. In some setting it underperforms BaseFit. 

Overall, the paper is good and I subjectively believe that the method is effective. So I will give a positive rating. If the authors can address my concerns, I will further raise my rating.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
