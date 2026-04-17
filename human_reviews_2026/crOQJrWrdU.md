# Security Tensors as a Cross-Modal Bridge: Activating Text-Aligned Safety to Vision in LVLMs

- Decision: Reject
- Scores: 4, 6, 6, 6

## Abstract
Large visual-language models (LVLMs) integrate aligned large language models (LLMs) with visual modules to process multimodal inputs.  However, the safety mechanisms developed for text-based LLMs do not naturally generalize to visual modalities, leaving LVLMs vulnerable to harmful image inputs.  To address this cross-modal safety gap, we introduce security tensors - trainable input vectors applied during inference through either the textual or visual modality. These tensors transfer textual safety alignment to visual processing without modifying any model’s parameters.  They are optimized using a small curated dataset containing (i) malicious image-text pairs requiring rejection, (ii) contrastive benign pairs with text structurally similar to malicious queries, designed to encourage visual-grounded decisions, and (iii) general benign samples preserving model functionality.  Experimental results demonstrate that both textual and visual security tensors significantly enhance LVLMs’ ability to reject diverse harmful visual inputs while maintaining near-original performance on benign tasks.  Crucially, our internal analysis reveals that security tensors directly trigger the hidden ``safety layers" of the language module when processing visual inputs, providing the first internal evidence that safety mechanisms in text can be cross-modally activated to vision.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Security Tensors, a novel, parameter-free method to bridge the cross-modal safety gap in LVLMs. By optimizing input perturbations on a curated dataset, the method activates the language module's inherent text-aligned safety mechanisms to handle harmful visual inputs, without requiring any changes to the model's weights.

### Strengths
1. The method shows significant safety improvements and strong generalization across multiple LVLMs, with minimal degradation of performance on benign tasks.

2. The experimental setup is described in a detailed and thorough manner.

### Weaknesses
1. Lack of Analysis on Model Scaling Effects. A systematic study, for instance across the Qwen3-VL family (e.g., 2B, 4B, 8B, 30B), would be necessary to validate the scalability and robustness of this approach, thereby defining the boundaries of its applicability.

2. Insufficient Consideration of Adaptive Attacks. Without this analysis, the true security guarantees of the method in a dynamic, real-world adversarial environment remain unverified.

3. Inherent Dependency on the Base Model's Safety Alignment. This limits its applicability and positions it as an incremental improvement for already well-aligned models, rather than a universal solution for securing any given LVLM. Investigating whether combining this approach with pre-processing defenses could achieve a more robust "1+1 > 2" outcome would have been a valuable analysis, positioning the work within a more practical, defense-in-depth framework.


4. Absence of Computational Analysis. Without these measurements, the claim of being a practical and lightweight solution is unsubstantiated, especially for applications where low latency is critical.

### Questions
Please refer to the questions raised in the Weaknesses section above.

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
4

### Summary
This paper proposes security tensors, which are trainable input-level perturbations added during inference through either the text or vision pathway of an LVLM, to activate the model’s existing textual safety mechanisms when it encounters harmful visual inputs.

### Strengths
- Works as a plug-in without modifying LVLM weights
- Large HR improvements across three LVLMs; meaningful generalization to unseen harmful categories.
- Low FRR, small MM-Vet impact
- Layer-wise alignment with prior “safety layers” strengthens the causal story.

### Weaknesses
- The N – (N + δ) pairs in Figure 3 show considerable differences, especially in the deeper layers. The gap between N – (N + δ) and M – (M + δ) is not substantially different, so how can it be demonstrated that δ has only a small impact on benign image–text inputs?

- Please report the actual number of samples in the test set described in L298, including both the Seen-category and Unseen-category subsets. To my understanding, the scales of VLGuard and MM-SafeBench are not very large, which raises concerns about the adequacy of the evaluation setting. If I have misunderstood, please clarify.

- Training uses only 4 harmful categories and 1000 samples. I would like to know how the number 1000 was determined—was there an ablation study on this number? Would δ learned from a larger dataset yield better performance?

- In L209, the paper mentions “refusal templates,” but their scale and diversity are not discussed. This raises concern that the model’s outputs may become formulaic refusals. Have the authors considered distinguishing between semantic safety and template matching? An analysis of output diversity versus the variation among refusal templates could provide stronger evidence that “the model learns the semantic intent of refusal rather than memorizing superficial token patterns.”

- Since δ is fixed, an adaptive adversary could design counter-perturbations. Adding evaluations under adaptive adversarial attacks would better demonstrate δ’s robustness in unseen scenarios.

- How is the Harmless Rate (HR) in L298 evaluated—by matching refusal keywords? This needs clarification. If HR is based on refusal keyword matching, I would be curious to see results under semantic refusal detection.

- Do the results hold for fine-grained harmful categories (e.g., medical self-harm paraphernalia, non-gory violence, subtle hate symbols) or for text-in-image cases?

### Questions
See Weaknesses

### Soundness
2

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
3

### Summary
This paper introduces Security Tensors, a novel cross-layer, prompt-based defense mechanism for large vision-language models (LVLMs). These tensors aim to transfer safety alignment mechanisms developed for text-based LLMs to the visual modality, enabling parameter-free, cross-modal safety enhancement. The approach optimizes both textual and visual tensors using a carefully curated dataset, aiming to improve security by enhancing harmful input rejection while maintaining the models’ original performance on benign inputs. Experiments on multiple LVLMs demonstrate strong safety improvements and generalization to unseen harmful visual categories with minimal performance degradation.

### Strengths
1. Quality: Sufficient experimental results are provided to demonstrate the method's effectiveness in improving safety while preserving the original models' performance. Additional evidence is also presented to explain the reasons behind this effectiveness.

2. Clarity: The paper is well-written and easy to follow.

3. Novelty: The proposed defense method is novel in the context of LVLMs. And to the best of my knowledge, the phenomenon of reusing the safety-guarding mechanism in the text-only decoder component is also new.

4. Significance: The proposed method generalizes to out-of-domain scenarios and can defend against more types of attacks, as shown in Table 1.

### Weaknesses
1. Quality: Currently, the performance degradation is measured only in the two proposed benchmarks, while in real-world applications, the models' other abilities are also important to track, such as mathematical reasoning [1] and commonsense reasoning [2].

2. Quality: The "safety layers" explanation is supported by evidence of cosine similarity values for different pair types. It could be strengthened if further activation patterns were available, such as the activation distribution in each layer.

### Reference

[1]: Lu, Pan, et al. "Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts." arXiv preprint arXiv:2310.02255 (2023).

[2]: Yue, Xiang, et al. "Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi." CVPR 2024.

### Questions
* Given Qwen-VL-Chat is a rather old model, what are the methods' performance in stronger VLM models such as Qwen-2.5-VL-instruct or Qwen-3-VL-instruct?

* Does the combination of both safe tensors yield further improvements?

* I am wondering if the proposed method can also generalize to categories such as deliberated elicited hallucinations [1].

### References

[1]: Han, Tianyang, et al. "The Instinctive Bias: Spurious Images lead to Illusion in MLLMs." EMNLP 2024.

### Soundness
3

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
The paper tackles a key gap in LVLM safety: text-only alignment does not reliably transfer to harmful visual inputs. The authors propose security tensors—trainable, input-level vectors injected either (a) in the text-embedding stream ($\delta_t$, virtual tokens between image and text embeddings) or (b) in the vision preprocessor feature space ($\delta_v$). These tensors are optimized offline (no model weights updated) using an asymmetric objective: cross-entropy to refusal on Safety Activation (SA) pairs (harmful image + benign text), and forward-KL distillation to baseline outputs on two benign sets—General Benign (GB) and a carefully designed Text-Contrast Benign (TCB) set whose prompts are syntactically similar to SA but paired with benign images. Evaluated on LLaMA-3.2-11B-Vision, Qwen-VL-Chat, and LLaVA-1.5 across VLGuard and MM-SafeBench hazards, the method substantially increases Harmless Rate (HR) with small increases in False Rejection Rate (FRR) and minor drops in MM-Vet Score. The paper also presents internal analyses suggesting that $\delta_t$ and $\delta_v$ reactivate “safety layers” in the LLM component (layers ~9–20), thereby bridging text-aligned safety to vision.

### Strengths
1: This paper introduces a method with low training cost that transfers the safety alignment capability of large language models (LLMs) to the image domain.

2: The paper conducts a layer-wise analysis to support its claim that the intrinsic safety alignment ability of the LLM is effectively activated by the proposed method.

### Weaknesses
1: The paper supports its claim that the safety alignment ability of the LLM is effectively activated through a layer-wise analysis. However, this evidence remains indirect. A quantitative ablation study would provide a more convincing and rigorous validation of this claim.

2: The paper lacks a clear explanation of how the learned security tensors activate the safety alignment ability of the LLM. Specifically, it remains unclear why changes in the latent representations trigger alignment behavior. Do the learned text-side security tensors carry any specific semantic meaning? Do the visual-side security tensors exhibit identifiable structural patterns? Providing such insights would significantly enhance the interpretability and credibility of the proposed mechanism.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
