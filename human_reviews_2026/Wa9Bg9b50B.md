# CompoDistill: Attention Distillation for Compositional Reasoning in Multimodal LLMs

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
Recently, efficient Multimodal Large Language Models (MLLMs) have gained significant attention as a solution to their high computational complexity, making them more practical for real-world applications. In this regard, the knowledge distillation (KD) approach has emerged as a promising alternative, which transfers the rich visual and linguistic knowledge from a larger model (teacher) to a smaller model (student). However, we observe that existing KD methods struggle to effectively distill the teacher MLLM's rich visual perception abilities to the student, a challenge that has been largely overlooked in previous studies. Through a systematic analysis, we identify visual attention misalignment between student and teacher as the main cause of this issue. Based on this insight, we propose CompoDistill, a novel KD framework that explicitly aligns the student's visual attention with that of the teacher to enhance the student's visual perception abilities. Our extensive experiments show that CompoDistill significantly improves performance on compositional reasoning tasks that require visual perception abilities while maintaining strong performance on visual question answering tasks, as done in existing studies. Furthermore, CompoDistill demonstrates effectiveness with a more advanced backbone, highlighting its generalizability.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper identifies visual attention misalignment as the main bottleneck in distilling visual perception ability in MLLMs through extensive attention-based analyses and large-scale experiments. The authors propose a knowledge distillation framework combining Visual Attention Alignment and Teacher Adapter Fetch based on the findings, which achieves significant improvements in compositional reasoning performance while maintaining strong visual recognition ability, and demonstrates robust generalizability and state-of-the-art performance across multiple benchmarks and model architectures.

### Strengths
1. The paper conducts extensive and systematic analysis prior to method design, deeply investigating the root cause of poor visual perception distillation through detailed teacher–student attention comparisons across tasks and layers, and by quantifying the relationship between attention similarity and performance.

2. It proposes an innovative and well-structured knowledge distillation framework that combines Visual Attention Alignment and Teacher Adapter Fetch to address both attention alignment and vision–language feature space mismatch issues, with a Group Layer Matching strategy that preserves richer teacher perception knowledge compared to traditional one-to-one matching.

3. The work presents extensive experiments across multiple compositional reasoning benchmarks, showing significant performance gains while maintaining strong visual recognition ability on VQA datasets, and demonstrating robust generalizability and state-of-the-art results across different backbone architectures.

### Weaknesses
1. The analysis in this paper is primarily conducted on the LLaVA, and the conclusions have not been systematically validated on other types of MLLMs with different training data compositions, architectural designs, and especially those employing ViT-based vision encoders. This raises uncertainty about whether the identified “visual attention misalignment” bottleneck and the effectiveness of the proposed CompoDistill framework generalize well to broader scenarios.

2. The proposed KD framework incorporates multiple modules and a multi-stage training strategy, which may introduce higher training overhead compared to existing KD methods. However, the paper does not provide a quantitative comparison of this additional cost nor discuss its applicability under resource-constrained settings, limiting a full assessment of the method’s practicality.

### Questions
1. How much additional training cost does your multi-module, three-stage approach incur compared to existing KD methods?
2. Have authors validated the conclusions and method on architectures other than LLaVA?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work addresses a gap in the effectiveness of knowledge distillation techniques in distilling visual perception techniques from teacher MLLMs to student models. An analysis is conducted to identify the reason for this issue, which is attributed to visual attention misalignment between the teacher & student models. A new knowledge distillation approach is proposed called CompoDistill which aims to address this issue by aligning visual attention patterns across the student and teacher models. Experiments are conducted which show that the proposed method offers improvements for compositional reasoning tasks.

### Strengths
1. This work identifies and addresses an important gap with existing knowledge distillation techniques which has been ignored in previous studies.
2. The analysis of attention pattern misalignment provides good conceptual motivation and support for the proposed CompoDistill method.
3. The experimental results cover a good range of VQA and compositional reasoning datasets. 
4. Ablation studies provide interesting insights into the importance of different CompoDistill components (VAT & TAF).

### Weaknesses
1. Discussion of related work is relegated to the appendix, which seems inappropriate for a full-length main conference paper.
2. The main experiments (Table 1) provide results for CompoDistill using only a single student/teacher model pair based on Qwen 1.5 (1.8B parameters for the student and 4B for the teacher). This makes it difficult to assess how well the method will generalize to different model architectures and larger model sizes. Limited aggregated results are provided for MobileLLaMA in Table 7, but it seems that the teacher model used here was also quite small. 
3. CompoDistill does not seem to offer significant improvements over other baselines on VQA. I understand that compositional reasoning requires more finegrained perceptual abilities, but shouldn't greater visual attention alignment between the student and teacher also lead to some benefits for VQA as well?

### Questions
1. From the ablation results (Table 3) it seems that VAT is more critical for compositional reasoning while TAF is more important for VQA. Do you have any insights into why this is the case?
2. Have you evaluated the effectiveness of CompoDistill for distillation from larger models?

### Soundness
2

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
4

### Summary
In the draft, the authors proposed CompoDistill, a new Knowledge Distillation (KD) framework to better transfer the teacher model's visual perception ability to student model. Multiple experiments show that the proposed KD framework could help improve performance on compositional reasoning tasks while maintaining performance on VQA tasks.

### Strengths
1. The authors conducted systematic analysis and identified visual attention misalignment is the key issue that prevent the distillation of the teacher MLLM's visual perception abilities to student model. Both the analysis and findings make sense and are technically sound to me.
2. The two proposed components of Visual ATtention alignment (VAT) and Teacher Adapter Fetch (TAF) are both technically sound to me as well. VAT better align the visual attention and TAF better align the following process. 
3. The authors conducted extensive experiments to justify the proposed frameworks.
4. Writing is good and easy to follow.

### Weaknesses
1. The current approach seems limited to same model family. How to extend the distillation to cross modal family?
2. The proposed three stage training is very complicated compared to the standard single stage approach. More justification might be beneficial to make sure the gain outweigh the cost.
3. For Table 2. it seems that VAT module is hurting on TextVQA tasks (56.4 vs. 56.5). Any idea what might be the issue?

### Questions
Please refer to the paper weakness section for more details and provide more justification accordingly.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces CompoDistill, a knowledge-distillation framework for MLLMs aimed at transferring visual perception (compositional reasoning) rather than only visual recognition. The key diagnosis is visual attention misalignment between teacher and student in the visual understanding layers; the method (i) explicitly aligns attention over visual tokens via a Visual ATtention alignment (VAT) loss with a group layer matching scheme, and (ii) bridges teacher–student vision-language space mismatch via a Teacher Adapter Fetch (TAF) that reuses the teacher’s adapter with a light MLP mapper. A three-stage training (distilled pretraining → distilled finetuning with VAT → SFT) is used. Experiments across VQA and compositional reasoning benchmarks show gains on CR while keeping VQA strong; ablations support VAT/TAF design choices.

### Strengths
1. Simple, orthogonal mechanisms. VAT aligns visual-token attention using cosine distance with a one-to-many group layer matching that’s more effective than simple or adaptive matching. TAF pragmatically resolves feature-space mismatch by reusing the teacher’s frozen adapter with a light MLP, making attention transfer workable in practice.

2. Consistent CR gains without sacrificing VQA. Across SugarCrepe, SADE, BiVLC, and Winoground, CompoDistill outperforms KD and maintains competitive VQA, showing perception improves while recognition holds steady. The intermediate-layer focus and cosine loss are empirically validated as strong choices. 

3. Breadth and robustness signals. Relational hallucination benchmarks (R-Bench, Reefknot) show additional improvements, hinting at better grounding. Results with a stronger backbone indicate the approach is not tied to a single encoder/LLM stack.

### Weaknesses
**1.Causal evidence is still light.**

The analyses show correlation between attention similarity and accuracy and a mild improvement from attention mixing, but they stop short of stronger causal tests (e.g., counterfactual training where only non-visual tokens are aligned). For example, on Figure 2c it shows the difference between student or SFT model with teacher is small on CR; similar trivial difference in Figure 4 is also observed. The author should have done more either empirical or theoretical analysis justifying the analysis. Tightening the causal chain would reduce the risk that VAT is a proxy for other regularization effects. 

**2. Adapter reuse may constrain flexibility.**

TAF fetches a frozen teacher adapter and adds a small projector, this could lock the student into the teacher’s visual-linguistic idiosyncrasies and hinder adaptation to novel modalities or encoders. A comparison against training a student-side adapter to match the teacher space (without reuse) would clarify trade-offs.

**3.Risk of overfitting to teacher’s focus patterns.**

For hard CR that requires shifting attention relative to the teacher (e.g., ambiguous or adversarial scenes), strict alignment might dampen student exploration and degrade robustness. Evaluating under label noise, adversarial cropping, or counterfactual edits would reveal whether VAT encourages brittle mimicry.

### Questions
1. What happens if you freeze only the attention blocks in mid layers during VAT but allow FFNs to adapt. Does that help avoid over-mimicry?

2. Any evidence that VAT improves localization (pointing/grounding) accuracy measured directly?

### Soundness
3

### Presentation
2

### Contribution
2
