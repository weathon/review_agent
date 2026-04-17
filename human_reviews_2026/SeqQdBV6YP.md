# InstEmb: Instruction-Following Embeddings through Look-Ahead Token Distillation

- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Recent advances have empowered large language models (LLMs) with remarkable fine-grained instruction-following capabilities in text generation tasks. However, embedding methods typically rely solely on the hidden state of the input's last token, limiting their ability to capture complete semantic signals distributed across the full output tokens. Moreover, existing discrete-to-continuous re-encoding approaches introduce semantic discontinuity. To address these limitations, we propose $\textbf{InstEmb}$, a novel instruction following embedding framework. InstEmb jointly optimizes two key aspects: (1) primary semantic information, achieved by employing contrastive learning focused on the representation of the last input token, and (2) complementary semantic information, captured through representation distillation leveraging learnable look-ahead tokens without introducing additional decoding latency. Additionally, we introduce $\textbf{Dual-Anchor Alignment Pooling (DAAP)}$, explicitly aligned with our dual training objectives.  Extensive experiments demonstrate that InstEmb achieves state-of-the-art performance across multiple instruction following benchmarks without benchmark-specific supervised data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces and assesses a new method, InstEmb, for aligning instruction following models, using teacher-based latent representations. InstEmb uses both cumulative (last token) and granular single-token representations to capture semantic information without additional decoding steps from the teacher models.

### Strengths
- Well scoped methodology and contribution that combines existing methods for token alignment and soft-prompting to improve instruction tuning
- Offers clear benefits of efficiency and performance improvements on the QA datsets, and evidence for generalization across new datasets

### Weaknesses
Most of my concerns are around the presentation and claims of semantic information types: 

- Grounding of "primary" and "complementary" semantics claims 
    - It's unclear how these notions are operationalized or captured in the methodology -- there are no clear experiments of qualitative assessments that these correspond to distinct, measurable factors in the latent space;  and, 
    - The work states these two are "jointly" optimized but doesn't fully quantify what information is captured by each representation (e.g., L450 shows that the lookup tokens align with the answer tokens; but this to me is a bit obvious given the alignment objective) 

- The figures are difficult to read (see suggestions in the Questions section), and the scoring methods are unclear. I see that the authors' method excel on these metrics, but it is unclear to me what these metrics capture and how this supports the conclusion of better aligned semantics. 
    - E.g., missing explanation of nDCG@5

### Questions
1. What is the overlap (i.e., Fig. 2) of the look-ahead token with the last token? With the original instruction? 
2. What is the benefit of using the same model as the student and teacher? Is there a performance difference when using a stronger teacher model to distill?
3. (Suggestion) A qualitative case study on which instructions perhaps benefit the most from the "semantic" information captured with this methodology.

Below are presentation suggestions and references to typos: 
- L54: "sampling during decode" --> decoding?
- L260: "strongity" --> strength? 
- L278: Please elaborate on "nDCG@5" in this section
- Fig. 1: The caption does not explain "view1/view2", font size on text is small and difficult to read 
- Fig. 2: The values and titles are way too small to read, please increase the font size

### Soundness
3

### Presentation
2

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
This paper proposes InstEmb, a new instruction-following embedding framework that captures both primary and complementary semantics through representation distillation and supervised contrastive learning. The method introduces learnable look-ahead tokens to distill output-related semantic signals. This paper also introduces a new pooling strategy to explicitly combine both semantic signals.

### Strengths
1. The paper addresses the critical and timely problem of building fine-grained, instruction-adaptive embeddings for modern retrieval systems.

2. The core technical approach is sound. The use of learnable look-ahead tokens coupled with representation distillation offers an elegant and efficient solution.

### Weaknesses
1. **Incremental Conceptual Novelty:**
While the proposed framework is neatly integrated, its constituent components—knowledge distillation, contrastive learning, and soft prompt-like tokens—are all well-established techniques. The contribution of InstEmb lies primarily in their combination rather than in introducing fundamentally new algorithmic or theoretical insights. As a result, the conceptual novelty may be viewed as incremental rather than groundbreaking.

2. **Unclear Definition of Core Concepts:**
The key notions of primary semantics and complementary semantics are introduced in Section 1 without adequate clarification. This lack of explicit definition may confuse readers, as the distinction underpins much of the subsequent methodology. It appears that primary semantics correspond to the semantics derived from the input and instruction, while complementary semantics relate to the output or response information. If this interpretation is inaccurate, the authors should clarify the terminology and provide an intuitive explanation early in the paper.

3. **Insufficient Justification for the Joint-Usage Hypothesis:**
The central claim that “both primary and complementary semantics need to be used simultaneously” is not sufficiently supported. Prior work, such as InBedder, already demonstrates the benefit of modeling complementary (output-related) semantics over primary-only embeddings (e.g., Instructor). However, the paper does not provide clear empirical evidence that combining both types yields synergistic improvements beyond using complementary semantics alone. A dedicated ablation study or comparative experiment would be necessary to substantiate this key hypothesis.

4. **Missing Comparison with a Highly Relevant Baseline:**
The related work and experimental sections overlook an important recent study, “*Don’t Reinvent the Wheel: Efficient Instruction-Following Text Embedding based on Guided Space Transformation*” (ACL 2025), which also targets instruction-aware embedding optimization. Since that work addresses a conceptually similar problem using a geometric transformation approach, a comparison—either qualitative or empirical—would help position InstEmb more precisely within the current research landscape and highlight its distinct contributions.

### Questions
1. Given that InstEmb primarily combines known techniques (knowledge distillation, contrastive learning, and soft prompt tuning), what aspects of the framework should be considered conceptually new beyond this integration?

2. Could the authors provide a clearer and more formal definition of primary semantics and complementary semantics, ideally with illustrative examples, to help readers understand how these two components differ and interact?

3. Are there specific ablation studies or quantitative results demonstrating that using both primary and complementary semantics jointly leads to better performance than using either alone?

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
4

### Summary
This paper proposes InstEmb, an instruction-following embedding framework designed to enhance semantic adaptability in text embeddings derived from large language models (LLMs). The framework jointly optimize primary semantics via contrastive learning focused on the final input token and complementary semantics via representation distillation from a frozen teacher model, using learnable look-ahead tokens that emulate future output semantics without incurring decoding latency. The framework also fuses the last-input-token and look-ahead-token representations to better align training objectives with inference-time embeddings.

Expriments on instruction-following benchmarks (FollowIR, Inst.STSb, IntentEmotion, NYTCluster) and generic sentence embedding tasks (MTEB subset) show that InstEmb achieves strong results compared to baselines such as InBedder, FollowIR, and Promptriever. The paper also includes ablations on distillation methods, contrastive views, pooling strategies, and training datasets.

### Strengths
1.	The paper proposes an efficient framework that trains an instruction-following embedding model in a single LLM pass, offering a clean and practical solution.
2.	The experimental section is extensive, covering both instruction-following and generic embedding benchmarks with strong baselines. The ablation studies provide insights into design choices.
3.	The motivation to bridge the gap between instruction-following generation and embedding representation is well articulated and timely. The authors clearly identify two limitations of existing embedding methods and directly address them through an well-designed framework.

### Weaknesses
1.	The current ablation studies on contrastive learning (§5.2) are conducted on top of the representation distillation module, without providing clear comparisons between the two modules individually. A proper factorial ablation (e.g., SFT + Distillation vs. SFT + Contrastive) would better reveal which component contributes more to the overall gain.

2.	Teacher–student configuration not sufficiently explored. As described in Appendix A.1, both the teacher and one student model share the same base architecture (LLaMA-3-8B-Instruct). No experiments are conducted with a larger or more capable teacher model, making it unclear whether the observed improvements stem from the proposed framework or from the inherent capacity of the backbone model.

3.	The number of look-ahead tokens is fixed to 8 throughout all experiments, without any sensitivity or scaling study. It is therefore uncertain how this hyperparameter affects model performance, training stability, or inference efficiency.

4.	Section 5.5 mentions training with the MS-MARCO dataset, which lacks explicit instruction fields. It remains ambiguous how instruction semantics are preserved — is the question field treated directly as the instruction? 

5.	Some related works are missing which should be discussed, used as baseline or evaluation [1-7].

[1] Oh, Hanseok, et al. "Instructir: A benchmark for instruction following of information retrieval models." arXiv preprint arXiv:2402.14334 (2024).

[2] Yoo, Young Hyun, et al. "Hyper-CL: Conditioning Sentence Representations with Hypernetworks." Proceedings of the Annual Meeting of the Association for Computational Linguistics. Vol. 1. Association for Computational Linguistics (ACL), 2024.

[3] Sun, Weiwei, et al. "MAIR: A Massive Benchmark for Evaluating Instructed Retrieval." Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 2024.

[4] Zhou, Jianqun, et al. "Beyond Content Relevance: Evaluating Instruction Following in Retrieval Models." The Thirteenth International Conference on Learning Representations.

[5] Feng, Yingchaojie, et al. "Don't Reinvent the Wheel: Efficient Instruction-Following Text Embedding based on Guided Space Transformation." arXiv preprint arXiv:2505.24754 (2025).

[6] Yamada, Kosuke, and Peinan Zhang. "Out-of-the-Box Conditional Text Embeddings from Large Language Models." arXiv preprint arXiv:2504.16411 (2025).

[7] Zhang, Gaifan, Yi Zhou, and Danushka Bollegala. "CASE--Condition-Aware Sentence Embeddings for Conditional Semantic Textual Similarity Measurement." arXiv preprint arXiv:2503.17279 (2025).

### Questions
1.	Could you provide results comparing SFT + Distillation and SFT + Contrastive training separately, to determine which module contributes most to the improvements?

2.	Why was the teacher model kept similar in scale to the student? Have you tried a larger teacher, or could you comment on how performance might scale with teacher capacity?

3.	Have you explored varying the number of look-ahead tokens (e.g., 4, 8, 16)? Does the performance plateau or degrade as this number changes?

4.	Regarding the MS-MARCO experiments (§5.5), since the dataset lacks explicit instructions, do you treat the query text as the instruction itself? If so, how can the model still learn genuine instruction-following behavior rather than mere query encoding?

5.	Would combining datasets with and without explicit instruction fields affect the generalization ability of InstEmb?

### Soundness
3

### Presentation
3

### Contribution
3
