# Leveraging Pretrained Knowledge at Inference Time: LoRA-Gated Contrastive Decoding for Multilingual Factual Language Generation in Adapted LLMs

- Decision: Accept (Poster)
- Scores: 6, 4, 2, 6

## Abstract
Large language models (LLMs) adapted to specific languages through continual pretraining or instruction tuning often suffer from catastrophic forgetting, which can lead to factual inaccuracies. This issue is particularly pronounced in multilingual settings, where adaptation may override general world knowledge with language-specific patterns. We propose LoRA-Gated Contrastive Decoding (LGCD), a training-free inference-time decoding framework that improves factuality in language-adapted LLMs by leveraging knowledge from the original pretrained model. LGCD operates by (1) extracting factual representations from Feed-Forward Network (FFN) layers via LoRA-based decomposition, approximating pretrained knowledge, (2) dynamically gating decoding based on token-level confidence, and (3) applying contrastive decoding with Top-K masking to revise uncertain predictions by referencing the approximated representation of pretrained knowledge. LGCD requires no additional training or access to the original pretraining data. Extensive experiments with LGCD on multilingual multiple-choice and long-form QA tasks across nine languages demonstrate its strong effectiveness in mitigating hallucinations and enhancing factual accuracy in language-adapted models. These results further indicate that pretrained knowledge can be strategically reintroduced during decoding to promote factual multilingual generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes LoRA-Gated Contrastive Decoding (LGCD), an inference-time framework designed to improve factual accuracy in language-adapted LLMs that have undergone continual pretraining or instruction tuning. The method approximates the pretrained model’s factual knowledge by extracting low-rank updates from FFN layers, and dynamically switches between the adapted model and the approximated pretrained model based on token-level confidence. Experiments across multiple languages show improvements in factuality without additional training or access to pretraining data.

### Strengths
- The method does not require access to the original pretraining data, nor retraining or additional model parameters.

- Only extracting weights from the FFN layers makes implementing the method lightweight.

- The confidence-based gating is intuitive and allows the method to selectively intervene only when needed, balancing fluency and factuality.

- Experiments across multiple languages demonstrate general applicability.

### Weaknesses
- Incomplete evaluation: The paper motivates the problem by arguing that continual pretraining and naive LoRA fine-tuning lead to catastrophic forgetting; however, their evaluation does not include LoRA fine-tuned models as a baseline, even though this is a directly relevant and stronger baseline than the model-merging baselines used. This omission weakens the empirical evidence for the claimed benefit of the proposed method.

- The evaluation is limited to QA-style benchmarks, which constrains the generalizability of the findings. Additional tasks would help demonstrate broader applicability, while the Multi-FAct score eval should be expanded.

- The work focuses primarily on mid- and high-resource languages. Nigatu et al. (2024) classify Swahili as a low-resource language; Swahili has recently become closer to a mid-resource language due to expanding corpora and tooling. Similarly, the other languages evaluated are not particularly low-resource, which limits the claim that the method addresses low-resource multilingual adaptation. See Nigatu et al. (2024) for some low-resource languages.

$~$

### **References**

Nigatu, Hellina Hailu, et al. *"The Zeno's Paradox of Low-Resource 'Languages'."* arXiv preprint arXiv:2410.20817 (2024).

### Questions
1. How do you think your method will compare to replacing equations 3 and 4 with actual LoRA CPT on the FFN layers instead of SVD-derived $A_\ell\$ and $B_\ell\$​? Can you show that?

2. How exactly did you empirically show that your method mitigates hallucination?

3. Can you provide results for benchmarks beyond QA tasks? The Multi-Fact eval (Table 4) only shows results for your method and PTM, which is limiting, as Figure 2 shows that other methods are also competitive.

4. With LGCD, do the pretrained and adapted models have to use the same architecture? Table 1 indicates that you used models with a mix of Gemma and Llama architecture.

5. At inference time, do you deploy both PTM and LAM while alternating the decoding with LGCD? A short description of that part will help.

6. Do you have an idea of what happened to Japanese in Figure 2?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents LGCD, and its core idea is that fine-tuning sometimes hurts a model’s original factual knowledge due to catastrophic forgetting. However, we still want the model to be fine-tuned (for example to better follow instructions in a specific language). In this situation, we can adaptively apply the base model when a decoding step requires factual knowledge and use the fine-tuned model otherwise.

The use of the base model is replaced with a LoRA-based approximation of the difference between the base and fine-tuned models. The decision of whether to apply contrastive decoding or not at each step is determined dynamically based on the model’s confidence.
The experiments are across benchmarks such as Global MMLU, Multilingual TruthfulQA, Multi-FAct, and medical QA. The results show that LGCD consistently improves factuality and reduces hallucinations compared to other baseline methods like decoding or model-merging, without any additional training.

### Strengths
The paper is well-motivated by the catastrophic forgetting problem and proposes a concrete solution by contrastive decoding.

### Weaknesses
1. The authors proposed Top-K masking, which is actually very similar to the Adaptive Plausibility Constraint (APC) proposed originally in Contrastive Decoding (Li et al., 2022). Why not just use APC? Top-K tokens may include low-probability tokens when only one token has a large probability but K > 1, so more low-probability tokens are still included. In contrast, APC can exclude these low-probability tokens when there is only a single token with high probability.
2. The method assumes that both the pretrained model and the adapted model weights are available, which may not always be true. Other baselines such as DoLa only require access to the adapted model, so the comparison may not be entirely fair.
3. The method introduces four hyperparameters (τ, α, β, K)—confidence threshold, contrastive weights, and Top-K size—which require language- or model-specific tuning with some “training data”. This could limit its generalization ability to other models or settings.
4. The evaluation focuses mainly on factual QA and medical-domain QA, while the effects on creative, open-ended, or reasoning tasks remain unclear.

### Questions
The authors propose that applying LoRA only on FFN layers can recover the base model’s knowledge. However, it might be better if during the alignment stage only the attention layers are fine-tuned while freezing the FFN layers to prevent forgetting. Compared to this contrastive decoding method that is more like a remedy for the forgetting issue, directly fine-tuning non-FFN layers only could be a more straightforward and potentially effective option. Have the authors tried or considered this method?

### Soundness
3

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
This paper proposed a method to resolve catastophic forgetting issue for LLM. The motivation is that LLMs often "forget" general knowledge after the model was fine-tuned for specific domains. This can lead to factual errors when user query the model. The authors propose a method named LoRA-Gated Contrastive Decoding (LGCD). The method is a training-free method that aims to help LLMs recover lost knowledge during text generation. The method functions by extracting forgotten knowledge through computing the difference between original and adapted model weights, then creating a lightweight LoRA approximation, monitoring the model's confidence and only applying intervention when some empirical confidence threshold is reached, and using contrastive correction to adjust predictions with the original model's knowledge while preserving fluency of generation. Experimental results show restoring factual accuracy on some datasets.

### Strengths
1. The method delivers consistent performance across many settings, with particularly good gains where adapted models underperform their originals. Both automated and human evaluations are conducted.

2. The paper shows which model layers matter most and demonstrates that LGCD adds new factual content rather than just reordering existing predictions.

### Weaknesses
1. The method needs full access to both original and adapted model weights, which is not practical in some cases and can cause extra GPU/CPU memory, and inference overhead. The method is impractical

2. The confidence threshold is crucial but currently set using indirect heuristics. How can we trust the logit based confidence? Are they reliable? More reliable and adaptive approaches for automatic threshold selection should be considers. 

3. The method works for moderate adaptations but can be struggling with models that have been extensive domain-specific trained where there is high-rank differences between model weights. Another question: what if the adapted model did not forget the truth? DO we still need to extract from original model?

4. The method introduces extra decoding overhead, which is problematic for high-throughput applications. Meanwhile, If we have weight of original model, we can just inference it to get answer, why do we need the extra lora component? Besides, how can we guranttee that the output from original model is correct? The pretrained model can hallucinate as well.

5. The figures/table is hard to read with very small font.

### Questions
see weakness

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
1

### Summary
I'm not confident enough to review this paper since I'm not familiar with multi-lingual tasks / LoRA technology.

### Strengths
n/a

### Weaknesses
n/a

### Questions
n/a

### Soundness
2

### Presentation
2

### Contribution
2
