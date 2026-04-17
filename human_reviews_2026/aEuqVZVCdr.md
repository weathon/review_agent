# Gumbel Distillation for Parallel Text Generation

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 4

## Abstract
The slow, sequential nature of autoregressive (AR) language models has driven the adoption of parallel decoding methods. However, these non-autoregressive models often sacrifice generation quality because they struggle to model the complex joint distribution of token sequences. To narrow this performance gap, we introduce Gumbel Distillation, a novel distillation technique that enables parallel decoders to learn this distribution effectively. Our method leverages the Gumbel-Max trick to create a deterministic mapping from a latent Gumbel noise space to the output tokens of a high-performing AR teacher. As a model-agnostic technique, Gumbel Distillation seamlessly integrates with diverse parallel decoding architectures, including MDLM and BD3-LM. Experiments on LM1B and OpenWebText show that Gumbel Distillation substantially improves the generation quality of parallel language models, achieving a 30.0% improvement in MAUVE Score and 10.5% in generative perplexity over MDLM trained on OpenWebText dataset.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a method called Gumbel Distillation that turns the teacher model’s sequential sampling process into a deterministic "blueprint" of noise values which a parallel student model can condition on, enabling high-quality parallel decoding. 

The authors show how to extract these noise "blueprint" from teacher samples or recover them from existing text using a parallel posterior sampling procedure, and they explain how to inject the resulting signal into masked or multi-head parallel decoders so the student learns to reproduce the teacher’s joint decisions. 

Empirically, this approach improves generation quality compared to prior parallel decoding methods across multiple datasets and model architectures, with gains on metrics that capture both distributional similarity and sample fluency, and with ablations that demonstrate the importance of the Gumbel formulation and the posterior extraction step. 

The paper also provides practical algorithmic details and theoretical justification for the posterior extraction, and it discusses limitations and directions for scaling the technique to larger vocabularies and more structured noise representations.

### Strengths
1. The paper raises a good question: improving the accuracy (MTP) and quality (DLM) of parallel generation is a meaningful endeavor that will help improve the inference speed of autoregressive models.
2. The paper is clearly written and very detailed, presenting comprehensive methods, experimental details, and theoretical proofs, along with specific case studies.
3. The paper proposes a simple, commonly used, yet effective method that improves generation quality in DLM and increases acceptance rate during MTP.

### Weaknesses
1. I think the experiments are still not solid enough. You should compare them with other existing methods, such as other distillation methods. While it's good to see improvements when adding a new method to the original model, you need to compare it with similar methods to see if it outperforms them.
2. I'm not sure how the green gains in Table 3 are calculated. The difference between the two doesn't equal the green value. I'd also like to know the average generation length, as the current improvement in acceptance rate is minimal. It would be even better if you could provide the variance over 3-5 runs.
3. I think MTP has greater potential, as few people would retrain pre-training and DLM just to add a new method. Therefore, I think you could conduct several similar experiments with related methods to make your results more convincing.

### Questions
1. The authors acknowledge in the Limitation that computational complexity increases with vocabulary size, but current models often use very large vocabulary. For example, the commonly used Qwen is nearly 152K. I'd like to know how many times more computational effort is theoretically required compared to GPT-2?
2. I'm not sure why Head 0 is also trained in Table 3. I understand that Head 0 is essentially the LM head, and Medusa doesn't modify it, otherwise it wouldn't be lossless. Is this because gumbels must be input continuously? Are there any experimental results demonstrating true losslessness?
3. Comparing Table 4 with Table 1, I find that Sequential Gumbel Extraction's MAUVE score is better than the baseline, while PPL is worse. It seems that the introduction of gumbels only improves the Parallel model, but the two are theoretically equivalent. What are the specific implementation details of Parallel and Sequential? Does Sequential use a Casual Mask for multiple forward passes, while Parallel performs a single forward pass without a mask? If yes, are the Logits very different between the two? I feel more explanation is needed here.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a method for distilling the knowledge of an autoregressive model into a parallel/multi-token model via gumbel distillation. This key idea allows for a deterministic supervised setting for the parallel student, and effectively reframes the parallel learning task. Through various experiments, the authors demonstrate how gumbel distillation improves the generation quality of parallel decoding language models, closing the gap with the slower but more high-quality autoregressive models.

The authors validate their approach by integrating it into masked diffusion language models and multi-token prediction architectures. Quantitative results demonstrate significant and consistent gain over the original methods, on datasets such as OpenWebText and LM1B. This, along with a strong increase in the acceptance rate for the MTP setting, clearly indicates the benefits of the proposed approach.

### Strengths
The paper is very well written, and it is clear that the authors have put a lot of time and care into the presentation. This extends to figures, tables and the appendix as well. 

The method is (to at least my understanding) sound, and the explanation is pedagogical with consistent notation. The empirical results are persuasive in that they are both consistent and strong. The tasks selected for evaluation seem relevant, and give a fairly comprehensive overview of the expected gains from the gumbel distillation approach.

### Weaknesses
The authors currently don’t demonstrate any results or arguments for the scalability of this approach. Considering that a major component of how well LLM works is their ability to scale, such an addition would further strengthen this paper. The only part relevant to this seems to be the brief discussion regarding the scalability of vocabulary size. Personally, I’d prefer to see an experiment where the number of model parameters are scaled in the main paper, rather than the existing ablation studies. 

Less an issue and more a suggestion of structure. The current layout flow of: 2. Background, 3. Method, 4. Related Work, seems a bit unconventional? As a reader I found it a bit jarring to suddenly be exposed to a related work section after having just read the method. I would suggest considering moving the related work section before the method section or to the end of the paper.

### Questions
What are your intuitions regarding the scalability of this approach? In regards to model parameters, context length, dataset size etc?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Gumbel Distillation, a knowledge distillation framework that improves the learning of parallel decoders such as Masked Diffusion Language Models (MDLM) and Multi-Token Prediction (MTP) models. The key idea is to convert the stochastic sampling process of an autoregressive (AR) teacher into a deterministic mapping via the Gumbel-Max trick, allowing the student model to condition on the teacher’s latent Gumbel noise. This effectively transforms the hard problem of matching a complex joint token distribution into a simpler supervised learning problem. The framework is architecture-agnostic and can be plugged into various diffusion or parallel decoding models.

### Strengths
1. The paper correctly identifies that the main challenge of parallel decoding lies in learning the joint token dependencies. By introducing Gumbel noise as an explicit conditioning variable, the proposed approach offers a conceptually clean way to transfer dependency structure from AR to parallel models.
2. The idea of externalizing stochasticity to simplify the learning problem is an important insight likely to inspire future work in distillation and non-AR training.
3. The method is designed to integrate with different architectures (MDLM, BD3-LM, Medusa) without architectural redesign, demonstrating good generality.

### Weaknesses
1. Limited and marginal empirical gains. In particular, Figure 3 shows that the NFE–quality trade-off is not clearly improved—on one dataset the curve even slightly underperforms the baseline, and on the others the advantages are only marginal. The improvements on reasoning and QA benchmarks (e.g., BoolQ, ARC) are also small.
2. The paper does not compare with few-step DLM acceleration baselines such as APD (Adaptive Parallel Decoding, arXiv:2506.00413).
3. Although the approach simplifies distribution matching conceptually, the student still faces a harder supervised task: mapping high-dimensional Gumbel noise vectors to text sequences. As the vocabulary grows, the conditioning dimension scales linearly, and the paper acknowledges this as a limitation (§6). The work could benefit from more discussion or experiments on how this affects optimization stability and convergence.
4. While distillation transfers AR dependencies, it is unclear whether the Gumbel-conditioned student retains the versatility of standard masked diffusion models — e.g., controllability, flexible conditional generation. Conditioning on a fixed “blueprint” may overfit to AR-like sampling behavior and reduce adaptability.

### Questions
See Weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses a critical trade-off in language model decoding: autoregressive (AR) models achieve high generation quality by modeling sequential token dependencies via chain-rule factorization but suffer from slow, token-by-token inference; parallel (non-autoregressive) decoders (e.g., MDLM, BD3-LM, Medusa) accelerate inference by generating multiple tokens simultaneously but struggle to capture the complex joint distribution of token sequences, leading to degraded coherence, repetition, or grammatical errors.

To bridge this gap, the authors propose Gumbel Distillation, a model-agnostic knowledge distillation framework that enables parallel decoders to learn the joint distribution from a high-performance AR "teacher" model. The core insight leverages the Gumbel-Max trick—a re-parameterization tool for categorical sampling—to establish a deterministic mapping between latent Gumbel noise vectors and the teacher’s output token sequences.

### Strengths
1. The paper directly addresses the long-standing trade-off in language model decoding: parallel decoders sacrifice generation quality (due to poor joint token distribution modeling) for speed, while autoregressive (AR) models excel at quality but are slow. 
2. The introduced knowledge distillation mechanism to transfer the AR teacher’s sequential dependency knowledge to parallel students. This focus on "fixing the joint distribution defect" aligns with the most critical unmet need in parallel decoding research, making the work problem-driven and practically relevant.
3. The paper’s experimental design provides controllable evidence for the effectiveness of Gumbel Distillation.

### Weaknesses
1. A notable omission is the lack of targeted discussion on the classical "AR-supervised NAR distillation" baseline, a well-established method in non-autoregressive decoding where NAR models are trained to mimic the sampled outputs of AR teachers via cross-entropy (CE) loss or sequence-level losses. This gap may lead readers to question whether the authors have overlooked a foundational approach in the field.
2. While the Gumbel-Max trick and knowledge distillation are individually well-established in NLP and CV, the paper does not clearly articulate why their integration for transferring AR teachers’ joint token distributions to parallel decoders is non-trivial.
3. The paper’s framing of results overpromises, while failing to contextualize the persistent quality gap between AR teachers and Gumbel-enhanced parallel models. The paper uses language like "bridge this gap" and "resolve the quality-efficiency trade-off," but the remaining performance difference between AR (e.g., GPT-2-Large) and optimized parallel models (e.g., Gumbel-MDLM) is still large.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
