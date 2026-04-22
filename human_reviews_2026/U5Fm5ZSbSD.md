# Memory-Efficient Differentially Private Training with Gradient Random Projection

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Differential privacy (DP) protects sensitive data during neural network training, but standard methods like DP-Adam suffer from high memory overhead due to per-sample gradient clipping, limiting scalability. We introduce DP-GRAPE (Gradient RAndom ProjEction), a DP training method that significantly reduces memory usage while maintaining utility on par with first-order DP approaches. DP-GRAPE is motivated by our finding that privatization flattens the gradient singular value spectrum, making SVD-based projections (as in GaLore Zhao et al. (2024)) unnecessary. Consequently, DP-GRAPE employs three key components: (1) random Gaussian matrices replace SVD-based subspaces, (2) gradients are privatized after projection, and (3) projection is applied during backpropagation. These contributions eliminate the need for costly SVD computations, enable substantial memory savings, and lead to improved utility. Despite operating in lower-dimensional subspaces, our theoretical analysis shows that DP-GRAPE achieves a privacy-utility trade-off comparable to DP-SGD. Our extensive empirical experiments show that DP-GRAPE can reduce the memory footprint of DP training without sacrificing accuracy or training time. In particular, DP-GRAPE reduces memory usage by over 63% when pre-training Vision Transformers and over 70% when fine-tuning RoBERTa-Large as compared to DP-Adam, while achieving similar performance. We further demonstrate that DP-GRAPE scales to fine-tuning large models such
as OPT with up to 6.7 billion parameters, a scale at which DP-Adam fails due to
memory constraints.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces DP-GRAPE, a method to reduce the memory overhead of DP training by using random projections.

### Strengths
1. The method is supported by a theoretical analysis of its privacy and utility guarantees.

2. Empirical results show it effectively reduces memory usage while maintaining model accuracy.

### Weaknesses
The choice of the projection dimension r is a key hyperparameter, but the paper provides little guidance on how to set it. The trade-off between memory compression and model utility needs a systematic analysis.

### Questions
1. After projecting the gradient, how is the noisy gradient used to update the model parameters?

2. In the memory comparison (Table 2), why is DP-GRAPE's cost $n_l$ and not $m_ln_l$? Don't you need to compute the full gradient before projecting it?

3. The Theorem says DP-GRAPE achieve comparable trade-off to DP-SGD. Do you have any experimental results? Also, what is the memory usage of DP-SGD?

4. For the DP-Adam experiments, it is important to compare against the correct baseline, DP-AdamBC [1], which accounts for bias correction. Why was this comparison omitted?

[1] DP-AdamBC: Your DP-Adam Is Actually DP-SGD (Unless You Apply Bias Correction), AAAI'24

### Soundness
3

### Presentation
2

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
The paper proposes DP-GRAPE, a memory-efficient DP training method that projects each per-sample gradient into a low-dimensional random subspace, then performs clipping and Gaussian privatization after projection. This design cuts per-sample gradient and optimizer-state memory from full dimension to the projection dimension r, enabling large-model DP training without materializing full per-sample gradients. The authors argue SVD-based subspaces are unnecessary in the DP regime because privatization flattens the singular-value spectrums of gradients, so cheap Gaussian projections suffice and avoid expensive SVDs or storing projectors. Theoretically, under standard assumptions, the algorithm achieves (\epsilon,\delta)-DP and the expected stationarity gap matches that of DP-SGD up to log factors when the number of layers are considered as a constant. Empirically, it maintains accuracy comparable to first-order DP baselines while substantially reducing memory and scaling to larger models.

### Strengths
Originality. The paper advances DP training by coupling project-then-privatize gradient handling with random low-rank projections, motivated by the observation that privatization flattens the gradient spectrum.
Quality. The paper provides rigorous theoretical guarantees and offers reproducible implementation details and hyperparameter guidance.
Clarity. Figures, tables, and the presentation of the algorithm are clear with consistent notation that makes the method easy to follow. 
Significance. DP-GRAPE substantially reduces the memory usage of DP training while preserving comparable accuracy.

### Weaknesses
Limited novelty (main concern).
Algorithmically, the core move—projecting gradients into a low-dimensional subspace and then privatizing—is a direct transplant of low-rank / random-projection ideas into the DP setting; the paper does not introduce a fundamentally new optimization principle. On the theory side, the guarantees largely read as an incremental generalization of standard DP-SGD analyses to the projected case.

Missing head-to-head experiments with the methods surveyed in Table 1.
Table 1 contrasts DP-SGD-JL[1], Ghost Clipping[2], and Book-Keeping[3], but the paper does not reproduce them under the same models/hardware/privacy accounting—leaving the table’s claims unsupported in this setting. In the zeroth-order line, only DPZero is included while DP-ZO[4] is omitted, which is a notable gap given the scarcity and relevance of Zeroth-order DP work.

Opaque memory attribution.
The comparisons do not decompose where memory is saved or spent—parameters, gradients, optimizer states, and activations—nor do they separate forward/backward/communication peaks. As a result, readers cannot tell whether the gains are dominated by optimizer-state shrinking, gradient tensor compression, or interactions with activation checkpointing.

[1]  Fast and memory efficient differentially private-sgd via jl projections. Advances in Neural Information Processing Systems, 34:19680–19691, 2021.
[2] Large language models can be strong differentially private learners. arXiv preprint arXiv:2110.05679, 2021.
[3] Differentially private optimization on large model at small cost. In International Conference on Machine Learning, pp. 3192–3218. PMLR, 2023.
[4] Private fine-tuning of large language models with zeroth-order optimization. arXiv preprint arXiv:2401.04343, 2024.

### Questions
Questions
Will you add DP-SGD-JL, Ghost Clipping, Book-Keeping, and DP-ZO under identical models and hardware, reporting peak memory, throughput, and wall-clock to a fixed validation target, so Table-1 claims are empirically supported?

Can you include a stacked memory breakdown separating parameters/gradients/ optimizer states/activations and phase-specific peaks (forward/backward/communication)?

Since the analysis only covers SGD-type convergence now, can you provide an Adam-style convergence guarantee of DP-GRAPE?

### Soundness
2

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
The authors propose projecting per-sample gradients onto low-dimensional random Gaussian subspaces before privatization, thus reducing memory and optimizer state size while maintaining DP guarantees.
The work is motivated by an empirical observation that differential privatization flattens the singular value spectrum of gradients, making SVD-based projections (e.g., GaLore) unnecessary. DP-GRAPE instead uses random Gaussian projections computed on-the-fly

### Strengths
The observation about spectral flattening is novel and provides a principled reason to abandon SVD-based projections.

The authors provide a theoretical privacy and convergence analysis for DP-GRAPE, which is non-trivial due to the introduction of random projections.

Evaluations cover both CV (ViT pre-training) and NLP (RoBERTa, OPT). Achieves large-scale DP training (OPT, 6.7B).

Memory savings in training are considerable: it cuts memory by over 63% in Vision Transformer training and 70% in RoBERTa fine-tuning compared to DP-Adam.

### Weaknesses
The privacy guarantee under random projections with unbounded entries is described informally. A more rigorous sensitivity or RDP proof sketch is needed.

DP-GRAPE’s algorithm is more complex to implement than vanilla DP-SGD/DP-Adam. I'm not sure how practical would be to implement it. No code mentioning.

### Questions
Have you done any ablation of projection dimension r versus accuracy/privacy?

How does the projection dimension r influence the effective privacy budget?

How does DP-GRAPE interact with existing memory-saving techniques like ghost clipping or even simple gradient accumulation?

DP-MERF, Harder at al. uses random features to create embeddings. It does not do it with memory efficiency as a goal but does not do it as a by-product?

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
This work presents a new approach to memory-efficient DP training using random projections instead of SVD-based subspaces, which is motivated by a “flattened” singular value spectrum after privatization. DP-GRAPE (Gradient RAndom ProjEction) employs three key components: (1) random Gaussian matrices replace SVD-based subspaces, (2) gradients are privatized after projection, and (3) projection is applied during  backpropagation. The experiments show that DP-GRAPE can reduce the memory footprint of DP training without sacrificing accuracy or training time.

### Strengths
- using random projections (DP-GRAPE) instead of SVD-based projections, which is memory efficient.
- DP-GRAPE (Gradient RAndom ProjEction) achieves a privacy-utility trade-off comparable to DP-SGD.
- The margins in the experiments are significant, in terms of the memory reduction, while preserving the accuracy.

### Weaknesses
- Comparisons asre not sufficient with SOTA methods, and other subspace methods.
- The robustness analysis for failure cases is missing.

### Questions
- The differences between the DP-GRAPE and existing subspace methods, such as LoRA, etc.
- The robustness analysis for failure cases is missing.
- Hyperparameters, 'somewhat extensive hyperparameter searches', sensitivity analysis is necessary.

### Soundness
3

### Presentation
3

### Contribution
3
