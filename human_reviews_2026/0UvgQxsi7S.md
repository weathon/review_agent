# Multi-Feature Quantized Self-Attention for Fair Large Language Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Large language models (LLMs) often encode social biases tied to sensitive features such as race and gender, undermining fairness in downstream tasks even after instruction tuning. Conventional debiasing methods require expensive fine-tuning, are tied to specific architectures, or operate only at the input or decoding stage while neglecting attention-level representations, which can result in compromised task performance. Moreover, most approaches are tailored to single-attribute settings and do not explicitly address scenarios with multiple, overlapping protected attributes and their intersections. This paper proposes a novel method of multi-feature quantized attention regularization (MQAR) to mitigate multi-feature bias by injecting a structured quantization into frozen self-attention layers. MQAR disentangles attribute-specific activations through vector-quantized regularization and uses a discriminator-guided autoencoding regularizer to adversarially suppress protected-attribute information while preserving task-relevant semantics. Crucially, the proposed method operates without modifying the backbone parameters or accessing pre-training data, ensuring architecture-agnostic applicability and minimizing representation distortion. MQAR is evaluated on five diverse LLMs (BERT, T5, GPT-Neo, Mixtral, and LLaMA 3.2) using three standard bias benchmarks (WinoBias, StereoSet, and CrowS-Pairs). Across these models, MQAR consistently reduces bias for multiple protected attributes and their intersections while maintaining downstream accuracy within at most 0.4 \%, on average, of non-debiased baselines on sentiment analysis, abusive language detection, and text generation tasks. These findings highlight quantized attention regularization as a scalable and effective method for mitigating social bias in modern language models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes MQAR to solve the problem of social bias across multiple sensitive attributes in LLMs by injecting structured quantization into frozen self-attention layers. The idea is that MQAR disentangles attribute-specific activations and removes biased information through adversarial autoencoding.

### Strengths
- This paper tackles the important and underexplored issue of multi-attribute bias in LLMs, which reflects real-world fairness challenges beyond single-attribute settings (e.g., only focus on gender or race in previous work).

- The proposed framework is architecture independent and focuses on frozen self-attention layers without fine-tuning, which is applicable in practice.

### Weaknesses
- My major concern is that the paper claims that LLMs still exhibit strong social biases, but does not provide empirical evidence that modern instruction-tuned or RLHF-aligned models suffer from such bias to a meaningful extent. Most of the reported results are on static benchmarks and may not reflect the behavior of current-generation aligned models in realistic contexts. Although they evaluated MQAR on five models, none of these were instruction-tuned or alignment-optimized versions, so the claim about persistent social bias in modern LLMs is not strongly substantiated by the experiments as they only showed bias in pre-trained and non-aligned models. This makes the empirical motivation of the paper less persuasive. Also, the authors provided an anonymous GitHub repository link, no code is actually available, which raises concerns about the reproducibility of the experiments and the credibility of the reported results.


- The claim that "semantic content is preserved" relies solely on performance metrics (claiming a minimal 0.4% accuracy loss in downstream tasks), which may not fully capture nuanced information degradation in the embeddings. Quantization typically introduces discrete bottlenecks that compress information, and while the authors include a commitment loss and reconstruction loss to mitigate this (Eq. 4–6), they never directly analyze trade-offs between bias reduction and information retention. The theorems and lemmas in Section 3.2 primarily focus on establishing the optimization framework for quantized regularization and adversarial training, such as bounding mutual information $ I(R; X) $ and $ I(R; A) $ rather than addressing the trade-off or the specific changes in mutual information before and after debiasing. While the lemmas provide theoretical bounds for mutual information to guide debiasing, the paper lacks empirical validation of these changes across the evaluated datasets.

### Questions
- The paper provides an anonymous GitHub link for code release, but why the repository currently contains no implementation or documentation?

- Many recent LLMs undergo alignment or instruction tuning, which already mitigates social bias to some extent. Could you clarify whether the proposed method still provides meaningful improvements on such aligned models, and provide empirical evidence if possible?

- What kinds of bias (semantic vs. syntactic) are most affected by applying the proposed method?

### Soundness
2

### Presentation
2

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
This paper proposes a novel debiasing method called MQAR (Multi-feature Quantized Attention Regularization). MQAR acts as a "plug-in" module that injects structured quantization and a discriminator-guided autoencoding regularizer into the frozen self-attention layers of LLM.
The goal is to disentangle activations related to sensitive attributes (like race and gender) while preserving core semantic information, all without modifying the LLM's backbone parameters.

### Strengths
Originality - The work is notable for introducing a modular, architecture-agnostic fairness mechanism that integrates quantization and adversarial autoencoding within frozen attention layers -- an uncommon yet elegant direction compared to embedding- or output-level debiasing. The explicit quantization-based feature disentanglement contributes a novel angle to fairness research in LLMs.

Clarity - The paper is written clearly, with an intuitive explanation of how quantization can suppress protected-attribute representations. Figures illustrating the architecture improve readability and understanding.

### Weaknesses
1. The fairness regularization relies partly on lexicon-based weak supervision for protected attribute identification. However, details on lexicon construction, coverage, and potential leakage (e.g., words correlated with downstream labels) are insufficient. This could introduce hidden bias, undermining claims of debiasing generality.
2. Although the abstract highlights multi-attribute bias, most empirical results focus on single attributes (e.g., gender, race). A clearer demonstration of intersectional fairness (e.g., gender × race) would strengthen the paper’s central claim.
3. While the paper emphasizes low overhead, quantitative results on latency or memory cost are missing. Such measurements would enhance the credibility of the lightweight claim.

### Questions
1. How were the protected-attribute lexicons constructed and validated? Have you evaluated whether these lexicons introduce spurious correlations with downstream labels?
2. Could you report mean ± standard deviation over multiple random seeds for both fairness and utility metrics, to confirm that improvements are consistent rather than incidental?
3. Do you have or plan to include explicit experiments on intersectional categories (e.g., Black women vs. White men) to substantiate the “multi-feature” claim?
4. Please provide quantitative data on the computational overhead (training time, memory, or latency) introduced by MQAR.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Multi-feature Quantized Attention Regularization (MQAR) — a lightweight, architecture-agnostic framework for mitigating multi-attribute bias in large language models (LLMs). Instead of retraining or fine-tuning, MQAR intervenes inside frozen self-attention layers by introducing a structured quantization module that disentangles protected-attribute activations (e.g., gender, race) from semantic information. The paper uses an adversarial autoencoder to regularize the latent space to minimize mutual information between latent vectors and protected attributes while preserving task-relevant content. And later in the experiment, the author reported results on five LLMs over three bias benchmarks. The framework also generalizes across tasks such as abusive-language detection, sentiment analysis, and text generation.

### Strengths
* This paper is structured and well-written. Notations are internally coherent.
* Hyperparameters and training details are clearly documented in the Appendix
* The paper result is strongly supported by the reported experiments on multiple LLMs and benchmarks.
* The quantized regularization integrates discrete latent bottlenecks with adversarial training to disentangle sensitive features.

### Weaknesses
* In the evaluation of multi-attribute bias, metrics are aggregated by gender and race, but intersectional sub-groups are not explicitly analyzed (such as gender & race).
* The theoretical part is not empirically validated. The fairness claims rely primarily on benchmark reductions rather than statistical significance testing.
* In Appendix E, the author shows that the Full version yields the lowest bias scores and best fairness–utility trade-off. It remains unclear how quantization changes what the attention representations look like.
* Improvements are reported as point differences; the significance or variance analysis remains unclear.
* The paper claims MQAR is architecture- and domain-agnostic. Since all evaluation datasets are English-only, it remains unclear whether MQAR would work equally well on other languages to support the claim.

### Questions
* Could the author clarify whether these features were intersectional evaluation or independently debiased in separate runs? If independent, how does MQAR handle correlated protected attributes during training?
* Have you evaluated MQAR on non-English datasets or cross-domain text to confirm this generalization?
* The quantization process introduces a codebook of K latent vectors. How is the K chosen and is it fixed globally?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the critical issue of multi-attribute social bias such as race and gender in Large Language Models, noting that existing debiasing methods are often limited to single attributes, require expensive fine-tuning, or fail to generalize across different architectures. The authors introduce Multi-feature Quantized Attention Regularization or MQAR, a novel plug-in method that operates on the frozen self-attention layers of pre-trained LLMs. This method uses structured quantization to create a discrete bottleneck that disentangles attribute-specific activations, combined with a discriminator-guided autoencoding regularizer to adversarially suppress the identified bias while preserving the original semantic representations. MQAR was evaluated on five diverse LLMs including BERT, T5, GPT-Neo, Mixtral, and LLaMA 3.2, using the WinoBias, StereoSet, and CrowS-Pairs benchmarks. The results show that MQAR consistently outperformed prior methods in bias mitigation while maintaining downstream task accuracy within 0.4 percent of the non-debiased baselines.

### Strengths
* The method introduces a novel use of structured quantization, typically for compression, as a fairness bottleneck to achieve disentanglement.
* The method is a model-agnostic plug-in that works on frozen LLM backbones, which avoids expensive fine-tuning and makes it broadly applicable.
* The method demonstrates strong empirical results, consistently outperforming chosen baselines on bias metrics across all five tested LLMs.
* A strong ablation study clearly validates the design, proving the decoder is crucial for utility and the discriminator is crucial for bias reduction.

### Weaknesses
* The state-of-the-art comparison is critically outdated, as it ignores modern inference-time methods and only compares against techniques from 2020-2022.
* The "lightweight" claim is unproven, lacking any quantitative analysis of the parameter increase or inference latency overhead, which could be significant from adding modules to every attention layer.
* The paper's central claim of handling "multi-attribute" bias is unsubstantiated, as the evaluation only tests single attributes separately, not intersectional bias, and fails to use appropriate benchmarks like BBQ.

### Questions
* There is a mismatch between the paper's multi-attribute motivation and its single-attribute evaluation. It is unclear why a true intersectional benchmark like BBQ was not used.
* The paper does not justify its exclusion of 2024-2025 SOTA inference-time methods from its performance comparison.
* The paper does not quantify its "lightweight" claim. The actual increase in parameters and inference latency from adding the new modules is missing.
* The abstract's claim of a 0.4 percent accuracy preservation needs clarification.

### Soundness
3

### Presentation
3

### Contribution
3
