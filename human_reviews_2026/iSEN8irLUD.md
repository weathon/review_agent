# Dual Distillation of Trajectory and Guidance Knowledge for Faster Inference in Conditional Masked Diffusion Language Models

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 4

## Abstract
Masked diffusion language models (MDLMs) have emerged as a promising generative framework for natural language, owing to parallel non-autoregressive generation capabilities with iterative unmasking/denoising. However, typical MDLMs require a very large number of neural network function evaluations for effective inference, making them computationally expensive in many real-world NLP applications that rely on conditional sequence-to-sequence generation. In this work, we propose a two-stage distillation method for conditional MDLMs that distills knowledge of (i) classifier-free guidance as well as (ii) unmasking trajectory from the existing teacher MDLM into a student MDLM. This allows the student MDLM, during inference, to (i) reduce two forward passes, required by a classifier-free guided (teacher) MDLM, to a single pass, and (ii) drastically reduce the number of unmasking steps. In this way, by dual distillation of guidance and trajectory knowledge, our MDLM achieves speedups of up to 16$\times$ while virtually retaining the quality of generation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a two-stage distillation method to address the high inference cost of conditional MDLMs. Evaluated on multiple tasks, the approach achieves up to $16\times$ speedup ($2\times$ from guidance distillation and up to $8\times$ from trajectory distillation) while maintaining or improving output quality. Qualitative examples indicate that the distilled model performs markedly better in low-step regimes, producing more coherent outputs with fewer denoising steps. The authors further suggest that the framework can be extended to multimodal MDLMs and scaled to larger architectures, potentially broadening its applicability.

### Strengths
1. The paper is well-written and easy to follow, with clear explanations of the two-stage distillation methodology and comprehensive algorithmic descriptions.

2. Under the authors' experimental settings, the two-stage approach achieves substantial speedup across multiple standard benchmarks while maintaining generation quality. The qualitative analysis with generated examples further supports this claim.

### Weaknesses
1. Limited Model Scale and Task Scope. Experiments are conducted only on a 113M parameter model. Scaling to larger MDLMs such as LLaDA [1] or Dream [2] remains unverified, despite the authors' claim that the framework can be extended to larger architectures. Furthermore, evaluation is restricted to relatively simple sequence-to-sequence tasks. Generalization to other conditional generation tasks (e.g., mathematical and code generation) is unclear. If the authors could provide supplementary results demonstrating the framework's acceleration potential on larger models and on challenging benchmarks (e.g., GSM8K [3], MATH [4], HumanEval [5]), the scalability claims would be significantly more convincing.

2. The claims regarding CFG acceleration are questionable. As shown in Table 1 of SMDM [6], the model can still achieve good results without using CFG. In the authors' claimed $16\times$ speedup, $2\times$ comes from stage-one guidance distillation. However, for the vanilla fine-tuned teacher model, using CFG may not be necessary. If the authors could provide evaluation results without CFG, it would further confirm the necessity of using CFG and better demonstrate the utility of guidance distillation.



[1] Nie et al., Large Language Diffusion Models.

[2] Ye et al., Dream 7B: Diffusion Large Language Models.

[3] Cobbe et al., Training Verifiers to Solve Math Word Problems.

[4] Hendrycks et al., Measuring Mathematical Problem Solving With the MATH Dataset.

[5] Chen et al., Evaluating Large Language Models Trained on Code.

[6] Nie et al., Scaling up Masked Diffusion Models on Text.

### Questions
1. The paper states that for guidance distillation, $\gamma_{\min} = 1$ and $\gamma_{\max} = 3$ are chosen, and results are reported for $\gamma = 1.4$ and $\gamma = 2.0$. What is the rationale behind these specific choices? Additionally, in Table 2, each task uses different initial step sizes and total numbers of rounds. How were these values determined, and why is a unified set of hyperparameters not used across all tasks?

2. Could the authors provide a comparison with the concurrent work D2F [1]? Such a comparison would help contextualize the contributions of the proposed approach.

[1] Wang et al., Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing.

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
4

### Summary
This paper proposes a two-stage knowledge distillation framework for accelerating inference in conditional MDLMs on sequence-to-sequence NLP tasks. The first stage CFG into a single forward pass, eliminating the computational overhead of computing both conditional and unconditional distributions. The second stage progressively distills multi-step denoising into fewer steps through self-distillation. The approach achieves up to 16× speedup (2× from guidance distillation, 8× from trajectory distillation) while maintaining generation quality across three tasks.

### Strengths
1. The paper addresses genuine computational bottlenecks in MDLMs—both the dual forward passes required by CFG and the multi-step generation process. These are real inefficiencies affecting practical deployment. And efficiency of the advantage of MDLM over AR.
2. The two-stage distillation approach is straightforward and well-explained. The architectural modification (adding a guidance scale embedding) for encoding guidance preferences is simple and practical.

### Weaknesses
1. While aims to improve the sampling efficiency of MDLM, this work only presents experiments with three specific conditional text generation, unlike many existing work that address the unconditional generation and more challenging tasks.
2. Limited technical novelty: The paper essentially applies existing techniques (guidance distillation from Meng et al. 2023, progressive distillation from Salimans & Ho 2022 and the discrete progressive distillation SDTT from Deschenaux & Gulcehre 2025) to conditional MDLMs. While the adaptation is competent, the core methodological contribution is incremental. The authors acknowledge this is an extension of prior image diffusion work but position it as the "first" for guided conditional MDLMs on seq-to-seq tasks, which is a narrow novelty claim.
3. There are many missing baselines in the direction of KV cache and faster sampler for MDLM

### Questions
$q_0$ is missing in the expectation in eq(3)

### Soundness
2

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
This paper addresses the high computational cost of conditional masked diffusion language models (MDLMs) with distillation techniques. MDLMs have the advantage of parallel (non-autoregressive) generation but require running many denoising (unmasking) steps during inference, and the classifier-free guidance (CFG) procedure even doubles the number of network evaluations. The paper proposes a two-stage distillation algorithm for conditional MDLMs. In the first stage, a student MDLM, which takes the guidance strength $\gamma$ as an extra input, is trained to emulate the output of a CFG-guided teacher. In the second stage, the guidance-distilled student model from Stage 1 is then used as the initial teacher for a series of progressive distillation steps, where each step trains a new student to distill two teacher unmasking steps into a single step. Empirical results demonstrate that the final student model achieves accelerated inference speed while maintaining comparable performance on NLP seq-to-seq generation tasks.

### Strengths
1. This paper identifies two sources of inefficiency in conditional MDLMs (iterative unmasking and CFG) and adapts existing diffusion distillation techniques to address these issues.
2. Empirical results and ablation studies on NLP seq-to-seq generation tasks demonstrate that the distilled models achieve 16x speedup at inference time without significant degradation of generation quality.
3. The paper is generally well-written and easy to understand.

### Weaknesses
1. The novelty of the proposed method is limited, as the proposed dual distillation framework is a direct adaptation of existing distillation method in the literature. This contribution does not meet the acceptance bar in my opinion.
2. Although the proposed method achieves a good speedup at inference time, it involves a computationally very expensive distillation procedure (especially in stage two where several rounds of progressive distillation are performed), which limits the scalability of the proposed method for larger models and larger datasets.
3. Limitations are not discussed in the paper.

### Questions
1. How is the diversity of the generated sequences from the final student model compared to those from the initial teacher model?
2. How is the inference speed of the final student model compared to standard auto-regressive LLMs of similar size?
3. Given the significant training costs introduced by the proposed method, could the authors quantify the total training cost and elaborate on this trade-off?
4. In the second stage, the proposed method adapts progressive distillation, which requires several rounds of distillation to produce a multi-step student model. However, recent diffusion distillation approaches (e.g., [1,2,3, 4]) can distill diffusion models into one-step student models in one round. Could those approaches be adapted to the conditional MDLM distillation setting to reduce training costs?
5. Please add a paragraph to discuss the limitations of the proposed method in the paper.
6. Is Equation (2) based on the assumption that $\alpha_t=1-t$? This assumption cannot be found in Sec 2.1.

[1] S Xie, et al. "EM distillation for one-step diffusion models." NeurIPS 2024.

[2] W Luo, et al. "Diff-Instruct: A universal approach for transferring knowledge from pre-trained diffusion models." NeurIPS 2023.

[3] M Zhang, et al. "Towards Training One-Step Diffusion Models Without Distillation". arXiv.

[4] H Zheng, et al. "Ultra-fast language generation via discrete diffusion divergence instruct." arXiv.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a two-stage distillation framework for conditional MDLMs to improve inference efficiency in sequence-to-sequence NLP tasks. The method distills (i) classifier-free guidance and (ii) unmasking trajectory knowledge from a teacher MDLM into a student model. The distilled student can generate text with a single forward pass per step and fewer unmasking steps, achieving up to 16× speedup while maintaining comparable generation quality

### Strengths
1. The paper is well written and clearly motivated; both the overall idea and the two-stage distillation algorithm are clearly presented and easy to follow.

2. Experiments on sequence-to-sequence tasks convincingly demonstrate the effectiveness of the proposed method in improving inference efficiency while maintaining generation quality.

### Weaknesses
1. The novelty of this work appears limited. The two proposed algorithms mainly follow the frameworks of previous studies [1, 2], and the paper does not clearly explain how its approach differs conceptually or technically from them.

2. The experimental scope is narrow. The paper focuses on relatively simple seq-to-seq tasks, where the reported inference acceleration is not particularly impressive. It would be more convincing to include evaluations on challenging tasks such as mathematical reasoning or code generation, which are central to current research on large language models and diffusion language models [3, 4].

[1] Meng et al. On Distillation of Guided Diffusion Models.

[2] Deschenaux et al. Beyond Autoregression: Fast LLMs via Self-Distillation Through Time.

[3] Ye at al. Dream 7B: Diffusion Large Language Models.

[4] Xie et al. Dream-coder 7b: An open diffusion language model for code.

### Questions
How does the proposed method compare with training-free acceleration approaches [5, 6] that are widely adopted in the community?

[5] Wu et al. Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding. 

[6] Ben-Hamu et al. Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking.

### Soundness
2

### Presentation
3

### Contribution
2
