# Hallucination-aware Intermediate Representation Edit in Large Vision-Language Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 6

## Abstract
Large Vision-Language Models have demonstrated exceptional performance in multimodal reasoning and complex scene understanding. However, these models still face significant hallucination issues, where outputs contradict visual facts. Recent research on hallucination mitigation has focused on retraining methods and Contrastive Decoding (CD) methods. While both methods perform well, retraining methods require substantial training resources, and CD methods introduce dual inference overhead. These factors hinder their practical applicability. To address the above issue, we propose a framework for dynamically detecting hallucination representations and performing hallucination-eliminating edits on these representations. With minimal additional computational cost, we achieve state-of-the-art performance on existing benchmarks. Extensive experiments demonstrate the effectiveness of our approach, highlighting its efficient and robust hallucination elimination capability and its powerful controllability over hallucinations. Code is available at https://github.com/ASGO-MM/HIRE.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a novel approach to mitigating hallucinations in language models by modifying intermediate representations, avoiding the need for retraining or doubling inference costs. The proposed framework, HIRE, includes 3 parts - Editor, Router, and Regulator, and dynamically detects and edits parts of the model’s internal states that are prone to hallucination, while allowing users to control the degree of editing to meet different requirements. The experiments demonstrate the effectiveness of HIRE, achieving state-of-the-art performance across three benchmarks.

### Strengths
* The paper presents convincing results, demonstrating the effectiveness of the proposed method across multiple benchmarks and settings.

* The writing is clear and well-structured, making the ideas and methodology easy to follow.

* The method is novel, introducing a new paradigm for hallucination mitigation by dynamically editing intermediate representations, which is both conceptually interesting and practically useful. The regulator is a nice addition that I haven't seen in other approaches.

* The analysis in Figure 5 effectively illustrates the key points, providing insight into how the editing of intermediate representations reduces hallucinations and supports the overall claims of the paper.

### Weaknesses
* Baseline comparison: As far as I understand, the baselines presented in the paper were trained on different datasets (or dataset sizes), making direct comparisons potentially misleading. A more appropriate comparison would include a reinforcement-learning baseline trained with LoRA on the same positive and negative examples that were used here, matching the number of trained parameters or FLOPs, to ensure a fair evaluation.

* Missing related work: The paper overlooks prior research on hallucination mitigation at the representation level (e.g., [1]). While most existing methods rely on retraining or contrastive decoding, some works directly edit representations and could serve as meaningful comparisons.

[1] Jiang et al., Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations, ICLR 2025

### Questions
The model appears to scale well with increasing training data, as shown in Figure 6. This raises a couple of important questions:

* How does this scaling compare to other finetuning-based methods? Is it possible that the proposed approach is more data-efficient, extracting more value per training example?

* The curve does not appear to have saturated. At what data size would the model’s performance plateau, and what are the implications for larger-scale training?

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
This paper introduces a new framework called HIRE for mitigating hallucinations in LVLMs without requiring model retraining or doubling inference costs. HIRE modifies intermediate representations by incorporating an Editor and a Router, which combine contrastive learning and DPO. By controlling the degree of hallucinations, the method adapts to different user requirements. Extensive evaluations show that the proposed approach achieves state-of-the-art performance on three benchmarks.

### Strengths
- The paper is well-written and easy to follow, with clear visualizations.
- The proposed HIRE framework is effective, lightweight, and applicable to most models.
- Detailed evaluations are conducted to demonstrate the effectiveness of HIRE.

### Weaknesses
- In Line 474, "..edited representations shift toward the non-hallucinated cluster and begin to merge, confirming that our editing effectively reduces hallucination." However, in Figure 5 (right), the entire green cluster appears to move closer to the separation line. How this observation can be explained should be clarified.
- The HIRE framework reduces hallucinations at the token level by enhancing/suppressing corresponding tokens based on the original/perturbed image. In the COCO dataset shown in Figure 9, the YES/NO tokens may not be as directly influenced as those in Figure 7, where the response directly contains instances. A deeper analysis on the effectiveness of HIRE in YES/NO questions could be constructed.

### Questions
- How does the centroid of the entire green cluster change relative to the separation line in Figure 5? Is it closer to the separation line after being edited?
- More analysis on the effectiveness of HIRE in YES/NO questions could be constructed.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces a method to reduce object hallucination. Prior approaches, including training-based methods and contrastive decoding, often require substantial training resources or increase inference costs. To address these limitations, the authors propose HIRE, which is composed of an Editor and a Router. The Editor comprises two encoders, an attention module, and a decoder, and is used to compute a steering direction. The Router then determines whether to apply this steering direction. The proposed method is evaluated on three benchmark datasets: CHAIR, POPE, and AMBER, demonstrating its effectiveness in mitigating hallucinations.

### Strengths
- **Lower Inference Cost Compared to Contrastive Decoding.** Despite the increased number of parameters and computational overhead in HIRE, the overall inference cost remains lower than that of contrastive decoding methods, making it more efficient in practice.
- **Reasonable Design Choice.** The proposed router architecture is inspired by the Mixture of Experts (MoE) approach, a widely adopted and validated method in LVLMs and LLMs. By selectively activating Editor, the router enables adaptive and effective processing.

### Weaknesses
**W1. Lack of Comparison with Training Method.** The proposed method falls within the training approach. The detailed comparison with existing training approaches validate the effectiveness of the proposed method.
- How efficient is the proposed method from a training resource perspective compared to other training methods? Also, the performance comparison with training methods (e.g.,  HACL) is required than contrastive decoding method.
- Is M3ID equivalent to M3ID + DPO? If not, a direct comparison with M3ID + DPO should be provided.

**W2. Lack of Comparison with Steering Methods.** HIRE intervenes the latent representation by computing the direction of non-hallucination. The comparison with steering methods [R1, R2] is needed.

[R1] Le Yang et al., Nullu: Mitigating Object Hallucinations in Large Vision-Language Models via HalluSpace Projection, CVPR 2025

[R2] Sheng Liu et al., Reducing Hallucinations in Large Vision-Language Models via Latent Space Steering, ICLR 2025

**W3. Performance of General Task Capabilities.** LVLMs can perform various visual tasks in a zero-shot manner. After training, are these capabilities maintained? 

**W4. Reproducibility.** HIRE has hyperparameters, including learning rates, schedulers, the scaling factor, and the editing strength. However, the paper does not provide a detailed justification for the selection of these hyperparameters. How were these values chosen?

### Questions
**Q1. Degree of Hallucination.** (Line 76) I think that most existing methods can control the degree of the hallucination. Both Contrastive Decoding (CD) and steering methods can effectively control the degree of hallucinations. Steering methods utilize a scaling factor for the steering direction, enabling control over the generated outputs. Similarly, CD has hyperparameters that influence the decoding process, allowing for the adjustment of the model’s logits. 

**Q2. Missing Reference.** I was wondering whether the proposed method differs from MOEs; the paper does not cite existing work on MOEs.

### Soundness
3

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
The paper proposes HIRE, a lightweight plug-in module for large vision-language models that reduces hallucinations without retraining the base model or doing multi-pass decoding. Instead of modifying output logits, HIRE edits the model’s intermediate hidden representations during inference. It learns to disentangle each token’s “semantic content” from its “hallucination tendency,” computes an edit direction that shifts the token’s representation toward image-grounded truth, and applies this edit only when a learned router predicts the token is likely hallucinated. A single global coefficient controls the edit strength and direction, enabling continuous, user-adjustable hallucination suppression. Experiments on standard hallucination benchmarks (e.g., CHAIR, POPE, AMBER) show that HIRE significantly lowers hallucination rates while preserving fluency and with minimal compute overhead.

### Strengths
1. This work performs hallucination control by directly editing intermediate representations, rather than retraining the whole LVLM or running contrastive decoding with multiple forward passes. This keeps compute cost low.
2. Authors disentangle “semantic content” vs. “hallucination tendency” in the hidden space and edits only the hallucination component, which preserves fluency and factual content instead of bluntly suppressing all tokens.
3. This work improves standard hallucination benchmarks (CHAIR, POPE, AMBER) while keeping language natural, showing that hallucination mitigation does not have to trade off descriptiveness.

### Weaknesses
1. Training images come from MSCOCO, and some evaluation benchmarks (e.g., POPE) also draw from MSCOCO. The method should be tested on images from other datasets to rule out dataset bias and show robustness.
2. The approach is only demonstrated on older LVLM backbones （LLaVA 1.5 & InstructBLIP. It should also be tested on the latest multimodal LLMs (e.g., the Qwen 2.5 VL series) to further validate generality and effectiveness.
3. The paper does not report the number of additional training parameters introduced by the added modules.

### Questions
Please refer to the weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3
