# Guided Star-Shaped Masked Diffusion

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 6, 4

## Abstract
The performance of pre-trained masked diffusion models is often constrained by their sampling procedure, which makes decisions irreversible and struggles in low-step generation regimes. We introduce a novel sampling algorithm that works with pre-trained models and, after a lightweight fine-tuning of a single layer, significantly improves sample quality and efficiency. Our method reformulates the generation process using a star-shaped paradigm, which inherently allows for error correction. To make this process effective, we augment it with a learnable re-masking scheduler that intelligently identifies and revises likely errors. This approach yields a substantial quality boost, particularly when using a small number of sampling steps. We extensively ablate key components of our approach and show its usability in different scenarios. In comprehensive experiments on text, and code generation, our sampling algorithm outperforms or matches existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Guided Star-Shaped Masked Diffusion (G-Star), a new sampling framework for masked diffusion language models that enables iterative error correction. The method has two components: (1) a "star-shaped" formulation where all latent states $q(x_t|x_0)$ connect directly to the clean data $x_0$ rather than forming a Markov chain, allowing non-monotonic transitions and token revision; (2) a lightweight learned error predictor $g_φ$ that identifies likely incorrect tokens for targeted remasking. The authors prove the training objective remains a weighted cross-entropy (Claim 1), ensuring compatibility with pre-trained MDLMs. Experiments on OpenWebText unconditional generation and seven downstream benchmarks with Dream-Instruct-7B show consistent improvements, especially in few-step regimes (32-256 steps).

### Strengths
1. The star-shaped formulation is theoretically elegant and practically compatible with existing models. Claim 1 proves that despite the non-Markovian forward process in Equation 4, the VLB simplifies to weighted cross-entropy identical to MDLM training, enabling direct reuse of pre-trained weights. The formulation also subsumes ReMDM as a special case.
2. The error predictor design is parameter-efficient and practical. Table 1 shows a 1-block predictor nearly matches the 12-block version, and head-only training matches full fine-tuning. For Dream-Instruct-7B, freezing the 7B backbone and training only a linear head make large-scale deployment feasible with minimal training overhead.
3. The empirical results are strong across diverse settings. In few-step regimes, G-Star+ achieves MAUVE >40 at 128 steps versus ~10 for Star+ (Figure 3), and G-Star-loop reaches 57.3 MAUVE at 128 steps versus 23.4 for best ReMDM (Table 1). Benefits extend to Dream-Instruct-7B with gains on all seven benchmarks including MMLU +1.3, GPQA +1.8, and IFEval +2.9 (Table 3), plus improved code generation with 17.8 perplexity at 64 steps versus 22.5 for ReMDM on Conala (Table 2).
4. The experimental validation is comprehensive with systematic ablations. Section 4.2 identifies optimal t_on ≈ 0.3 and explains why pure samplers fail (Figure 2). Section 4.3 shows guided sampling consistently outperforms unguided remasking with largest gaps in the 64-256 step regime (Figure 3). Section 4.5 confirms lightweight predictors suffice, establishing both effectiveness and practical efficiency.

### Weaknesses
1. The paper lacks wall-clock comparisons to autoregressive baselines. While G-Star outperforms diffusion models, it requires T sampling steps versus one AR pass. The paper only measures efficiency in step count, not actual wall-clock time for deployment.
2. The early-phase instability requires a hybrid schedule. The method does not stand on its own over the full trajectory: pure star-shaped sampling degrades text and needs MDLM drafting before refinement.
3. This method has imited edit expressivity. The sampler only does in-place substitution; it cannot insert or delete tokens. That constrains the kind of structural edits it can correct, which may matter for code or long-form generation.
4. The error predictor is trained by simulating denoising and labeling token errors against ground truth (Algorithm 1). Each experiment uses task-specific training (OWT for OWT, Tulu 3 for Dream-Instruct), but the paper never evaluates cross-domain robustness or addresses how the ground-truth error labeling handles instruction-following tasks with multiple valid outputs

### Questions
N/A

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
The paper proposes Guided Star-Shaped Masked Diffusion (G-Star), which aims to address the core limitations of discrete masked diffusion models, namely the irreversibility of decisions during sampling and the lack of error correction capability. The authors point out that although existing approaches (e.g., ReMDM) introduce a re-masking mechanism, they rely on random strategies, resulting in low efficiency. In contrast, G-Star employs a star-shaped sampling structure coupled with a learnable error predictor to achieve targeted and efficient error correction, leading to significantly improved generation quality, particularly in few-step sampling scenarios.

### Strengths
1. The paper clearly identifies a fundamental limitation of discrete masked diffusion—irreversible token commitments—and provides a well-motivated, elegant solution.
2. The proposed star-shaped formulation is theoretically sound and compatible with existing pretrained MDLMs, avoiding retraining. 
3. The guided remasking mechanism (error predictor) is lightweight yet effective, enabling targeted error correction and strong empirical improvements.
4. Extensive experiments (text, code, large models) and clear ablation studies convincingly demonstrate robustness and practical value.

### Weaknesses
1. The paper is conceptually dense; the star-shaped reparameterization could be explained more intuitively for accessibility.
2. Relying solely on pseudocode and textual descriptions may not adequately convey the paper. It is recommended that the authors provide an integrated methodological figure to help readers quickly grasp the proposed approach.
3. The method’s sensitivity to hyperparameters is discussed but might benefit from deeper theoretical justification or generalization guidelines.

### Questions
1. How robust is the learned error predictor when applied to unseen domains or tasks with different token distributions?
2. Could the proposed guidance mechanism be integrated into autoregressive diffusion hybrids for further speed–quality trade-offs?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the irreversible sampling procedure in masked diffusion models, which constrains performance in low-step generation regimes. It introduces G-Star, a novel sampling algorithm that reformulates generation using a star-shaped paradigm to allow for error correction and augments it with a learnable re-masking scheduler to intelligently revise likely errors.

### Strengths
- The paper identifies a fundamental limitation of masked diffusion models: the sequence of irreversible commitments imposes a ceiling on sample quality.   
- The proposed enables iterative refinement with a learned scheduler for targeted error correction.   
- The method demonstrates a commanding advantage in low-step regimes, underscoring the critical importance of intelligent guidance when refinement opportunities are limited.

### Weaknesses
- The conceptual building blocks (star-shaped process, guidance, loop) are adapted from prior work; the primary novelty lies in their specific application to error correction in masked discrete diffusion.
- The error predictor $g_{\phi}$ may face a potential distribution shift, as it is trained on single-step denoised data but applied to iteratively refined, model-generated drafts during inference.
- The optimal switch-over time ($t_{on}$) was determined for the unguided sampler but reused for the guided sampler without specific validation for the latter.
- There are minor clarity and presentation issues: $g_{\phi}$ is used before being formally defined, and tables are placed out of order (Table 3 is referenced before Table 2).
- Several key analyses, parameter settings, and overhead claims are not fully substantiated in the main text. For instance, the claim of "negligible increase in inference overhead" lacks a quantitative analysis of wall-clock time or memory, and important details like the sensitivity analysis for $T_{remask}$ (Appendix C.3) and loop schedule specifics (Appendix C.2) are confined to the appendices, hindering the main text's self-containedness and the readability.

### Questions
- Could the authors comment on the potential distribution shift for the error predictor between training and inference, and its impact on performance during iterative refinement?
- Can the authors provide a concrete analysis of the inference overhead (wall-clock time, peak memory) compared to the baseline MDLM and unguided Star+ samplers?
- Have the authors validated that the optimal $t_{on}$ found for the unguided Star+ sampler is also optimal for the guided G-Star+ sampler, or could performance be further improved by tuning it for the guided setting?

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
The paper proposes a novel sampling algorithm for error correction in discrete diffusion. The authors reparameterize the diffusion sampling process into a star-shaped formulation where each latent directly conditions on the clean data. They then augment this with a learned error predictor that identifies likely incorrect tokens and selectively re-masks them, enabling more efficient and powerful refinement during generation.

### Strengths
## Strengths
1. **Motivation & importance.** The problem is important, and the motivation for enabling targeted error correction in discrete diffusion is compelling.  
2. **Clear analysis.** The analysis is intuitive and fairly comprehensive. In §4.2, the paper distinguishes **two generation phases** that emphasize different objectives and favor different sampling strategies, which is quite interesting.1. **Novelty positioning.** The core contribution needs clearer separation from prior work. **ReMDM** already explores re-masking predicted tokens for error correction; it seems the main novelty here lies in the **guided prediction module** rather than the overall idea of remasking.  
2. **Generalization beyond text.** It’s unclear how the approach transfers to **vision**. Testing on an image benchmark (e.g., ImageNet) under the **ReMDM** setup would help establish applicability to the image domain.  
3. **Methodological details.** More specifics are needed about training the **error predictor**—its **architecture**, **training configuration**, and **compute budget**.

3. **Downstream evaluation.** The method is evaluated on downstream tasks, including **code generation**, demonstrating practical benefits.

### Weaknesses
1. **Novelty positioning.** The core contribution needs clearer separation from prior work. **ReMDM** already explores re-masking predicted tokens for error correction; it seems the main novelty here lies in the **guided predictor**.  Can you clarify this?
2. **Generalization beyond text.** It’s unclear how the approach transfers to vision. Testing on an image benchmark (e.g., ImageNet) under the ReMDM setup would help establish applicability to the image domain.  
3. **Methodological details.** Please include more details about training the **error predictor**, such as architecture, training configuration, and compute budget.

### Questions
1. **Compute overhead.** The approach adds a guided sampler to predict which tokens to re-mask. What is the **computational cost** of this component? Do you perform an additional forward pass at **every diffusion step**, and how does this scale with the number of steps and sequence length?  
2. **Metric trade-offs (Table 1).** Your method excels on **MAUVE** and **PPL** at 128/256/512 steps, whereas **ReMDM** does better on **DIV**.  Is there any way to preserve the DIV?
3. **Sensitivity to fine-tuning (Table 1).** Performance appears sensitive to fine-tuning strategy. Many best results occur under different settings. Can you explain this? 
4. You argue that when the sampling budget is small, each step is critical and performance degrades without the guided error predictor. However, in Table 1 the best result at 128 steps is achieved by $Star+_{ton}=0.2$ is the best, can you exlain this?

### Soundness
3

### Presentation
3

### Contribution
3
