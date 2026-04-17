# DLM-One: Diffusion Language Models for One-Step Sequence Generation

- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
This paper introduces *DLM-One*, a score-distillation-based framework for one-step sequence generation with continuous diffusion language models (DLMs). DLM-One eliminates the need for iterative refinement by aligning the scores of a student model’s outputs in the continuous token embedding space with the score function of a pretrained teacher DLM. We investigate whether DLM-One can achieve substantial gains in sampling efficiency for language modeling. Through comprehensive experiments on DiffuSeq—a representative continuous DLM—we show that DLM-One achieves up to $\mathord{\sim}500\times$ speedup in inference time while maintaining competitive performance on benchmark text generation tasks used to evaluate the teacher models. We further analyze the method’s empirical behavior across multiple datasets, providing initial insights into its generality and practical applicability. Our findings position one-step diffusion as a promising direction for efficient, high-quality language generation and broader adoption of continuous diffusion models operating in embedding space for natural language processing.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces DLM-one, a practical score distillation framework for continuous DLMs that enables one-step sequence generation without iterative denoising. This framework eliminates the need for iterative refinement by aligning the outpt scores of the student model in the continuous token embedding space with the score function of the pre-trained teacher DLM, significantly improving the sampling efficiency.

### Strengths
As mentioned in this paper, DLM-One achieves up to 500\times speedup in inference time, which means on the premise of ensuring quality, the computing cost and time consumption can be significantly reduces.

### Weaknesses
Although the results are impressive, the paper may not have elaborated in sufficient detail on the contributions of different components to the final performance. For example, the impact of different choices of teacher models.

### Questions
1.Since a comparison with the AR model is listed in the Appendix D, how does the DLM-one perform in handling open-domain creative text generation?
2.Is the one-step sequence generation model first proposed in this paper?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a method to distill a diffusion language model for extreme acceleration, enabling generation in just one step, i.e., without iterative refinement. The authors introduce DLM-One, a score-distillation-based framework trained with a two-stage optimization scheme designed to mitigate the latency caused by the alternating update strategy of the score estimator. Experiments are conducted using DiffuSeq as the teacher model and evaluated on three sequence-to-sequence tasks. The results show that DLM-One achieves competitive performance compared to DiffuSeq, with performance gaps ranging from less than 1% to 5%, though it lags in terms of diversity. At the same time, it reduces sampling cost by up to ~500× compared to DiffuSeq.

### Strengths
* The paper's focus on achieving one-step sequence generation with diffusion models is well-grounded and addresses a clear need for improved inference efficiency.
* The work demonstrates a strategic adaptation of distillation techniques from vision to language, validating its effectiveness for text generation.
* Through comprehensive ablation studies and discussion, the authors provide convincing verification for their data distillation approach with DiffuSeq.

### Weaknesses
* All experiments in the paper were conducted using DiffuSeq. It remains unclear whether the proposed method can generalize effectively to other diffusion-based language models.
* The empirical comparisons are primarily made against the teacher model. It would be valuable to include comparisons with other accelerated generation baselines and provide an analysis discussing the optimal model selection on the performance-efficiency trade-off.

### Questions
1. The experiments are condected solely with DiffuSeq. How about the performance with other diffusion language models?
2. Given that several related methods are discussed in Section 2.2, it would be valuable to include a comparative analysis of their performance and inference speed relative to DLM-One, in order to better situate its trade-offs in the landscape of efficient generation models.

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
This paper proposes a practical distillation framework for training continuous diffusion language models for one-step sequence generation (DLM-One), without the need for iterative refinement during generation. The paper conducts three sequence-to-sequence (Seq2Seq) tasks, including question generation (QG), text simplification (TS), and paraphrasing (PP), to support its claims and demonstrate the effectiveness of the framework. However, the novelty of this paper is modest: most components (score distillation, adversarial stabilization, two-stage optimization) are transferred from the vision domain. The experiments are conducted on only three general datasets, which lack a comprehensive study to convince the reader. In addition, the paper is not well organized; for example, mixing the introduction with related work, which makes this paper a bit hard to follow.

### Strengths
1. This paper proposes a practical distillation framework for training continuous diffusion language models for one-step sequence generation (DLM-One), without the need for iterative refinement during generation.
2. Three experiments on benchmarks (QQP, Quasar-T, Wiki-Auto) demonstrate competitive BLEU, ROUGE, and BERTScore compared to DiffuSeq, showing the effectiveness and validity of the framework.
3. This method reduces inference cost—up to 500× speedup—without large quality degradation.

### Weaknesses
1. Most of the components (score distillation, adversarial stabilization, two-stage optimization) are transferred from the vision domain, which limits the novelty of this paper.
2. All experiments depend on one teacher model (DiffuSeq). The results may not generalize to other DLMs.
3. Lack of experiments on classic generation tasks that require strict semantic evaluation, such as translation.
4. Since degeneration is mentioned, there is limited qualitative or quantitative analysis of when or why the student diverges.
5. In terms of writing, this paper is not well organized, such as mixing related work into the introduction, which makes it hard to follow and somewhat redundant.

### Questions
1. Can this framework be used with other teacher models to show generalization?
2. Can more benchmarks be used to explore the generalization of the framework, such as long-content Seq2Seq or translation benchmarks?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a score-based distillation framework, DLM-One, for single-step sequence generation with continuous diffusion language models. By distilling the score function from a pre-trained teacher model into a student model, this approach eliminates the iterative refinement process required by traditional diffusion models, resulting in up to a 5x inference speedup while maintaining generation quality. The authors validate the method's effectiveness on three sequence-to-sequence tasks and demonstrate an approximately 500-fold speedup in text simplification without a significant degradation in quality.

### Strengths
1. The proposed method, DLM-One, introduces a novel approach to single-step sequence generation using continuous diffusion language models by leveraging score-based distillation techniques.

2. The authors conduct experiments across three S2S tasks, demonstrating the effectiveness of DLM-One compared to baselines. They also provide thorough analysis of the trade-off between quality and diversity in single-step generation, highlighting potential areas for future improvements.

### Weaknesses
1. Lack of Baselines. The paper only discusses one outdated baseline, DiffuSeq (2022), and does not discuss other acceleration techniques.

2. The generation tasks are overly simplistic. The paper does not explicitly discuss the challenges associated with scaling the proposed method to larger datasets or more complex tasks.

### Questions
NA.

### Soundness
2

### Presentation
2

### Contribution
2
