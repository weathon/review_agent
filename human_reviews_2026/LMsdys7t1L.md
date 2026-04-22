# DCFold: Efficient Protein Structure Generation with Single Forward Pass

- Avg Score: 5.50
- Decision: Accept (Oral)
- Scores: 4, 4, 8, 6

## Abstract
AlphaFold3 introduces a diffusion-based architecture that elevates protein structure prediction to all-atom resolution with improved accuracy. This state-of-the-art performance has established AlphaFold3 as a foundation model for diverse generation and design tasks. However, its iterative design substantially increases inference time, limiting practical deployment in downstream settings such as virtual screening and protein design. We propose DCFold, a single-step generative model that attains AlphaFold3-level accuracy. Our Dual Consistency training framework, which incorporates a novel Temporal Geodesic Matching (TGM) scheduler, enables DCFold to achieve a 15× acceleration in inference while maintaining predictive fidelity. We validate its effectiveness across both structure prediction and binder design benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduced DCFOLD, a distilled version of AlphaFold3 (AF3) that generates protein structures in a single forward pass, achieving up to 15× faster inference while maintaining similar accuracy. The proposed method introduces a training framework that: 1) aligns one step predictions with the full multi-step diffusion process and 2) enforces representation alignment on AF recycling iterations. The paper introduces a scheduler which aims to stabilize the training on variable-length proteins. Experiments on PoseBusters V2, Recent PDB, and binder hallucination tasks claim near AF3-level accuracy with reduced inference cost.

### Strengths
1. The proposed framework addresses a bottleneck in AF3 as it is computationally at inference time. Speeding up the inference time will help to address tasks such as protein design and virtual screening.

2. The DC framework is relatively novel.

3. The results are promising reporting a 15x speedup while retaining similar performances on selected tasks.

### Weaknesses
1. Several details are missing or inconsistently used. Some examples:
- In Eq. 1, $w(t)$ is not defined, Later, it is set to 1, but its role is never justified.
- The reference timestep $r$ is ambiguous in Algorithm 1. Is it a function or scalar?
- The variable $u \in [0, 1]$, in Algorithm 1 is introduced as "training progress" but never clearly explained.
- The stop gradient operator is defined in Eq. 2, but never used anywhere

2. Missing important details make the training procedure non-reproducible. 
- $\mathcal{L}$ in Algorithm 1 seems to be a combination of $\mathcal{L}_{diffusion}$, $\mathcal{L}_{pairwise}$, and $\mathcal{L}_{confidence}$ (this latter never defined). The combination of those three losses is unclear. Are they combined linearly or trained sequentially in stages? If sequential, how are modules frozen or fine-tuned? What are the relative weighting coefficients?
- $\alpha$ defined after Eq. 3 is unclear. Sometimes is a scalar, sometimes a vector, sometimes a pairwise matrix?

3. The evaluation setup is significantly weaker than the Boltz2[1] benchmark used for AF3 and related models. Here, some of them:
- FEP accuracy on public benchmarks
- Local protein dynamics

4. The theoretical part about TGM lacks clarity. E.g. what are $C_0$, $s_{max}$, and $s_{min}$?

[1] Passaro, Saro, et al. "Boltz-2: Towards accurate and efficient binding affinity prediction." BioRxiv (2025): 2025-06.

### Questions
Please see in the Weakness section some of my questions. More details questions follow:

1. how is $u$ scheduled or computed in practice?
2. how is $r$ chosen in Algoritm 1?
3. how are categorical features handled?

Minor:
1. Text clarity can be improved
2. Some of the references need to be fixed e.g. AlphaFold3 line 306 page 6 is referenced to AlphaFold2 paper.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces **DCFold**, a single-step generative model for protein structure prediction, designed to overcome the slow inference speed of the iterative, diffusion-based AlphaFold3.

The core contribution is a **Dual Consistency** training framework, featuring a novel **Temporal Geodesic Matching (TGM) scheduler**. This allows DCFold to achieve AlphaFold3-level accuracy in a single generative step.

The authors report a **15x acceleration in inference time** while maintaining high predictive fidelity, with performance validated on both structure prediction and binder design tasks.

### Strengths
1.  Good acceleration was achieved by simultaneously distilling the Pairformer and the diffusion model, with no obvious degradation in performance.
2.  The paper proposes a novel time step sampling strategy that effectively stabilizes training and improves the model's performance.
3.  The mathematical derivations in the appendix are easy to follow and have no obvious errors.

### Weaknesses
1.  **Lack of Granularity in Inference Time Analysis:** The time complexity of different computational stages in protein structure prediction scales differently with protein length (L). As such, the inference time for a model like AlphaFold3 can vary significantly across proteins of different lengths. The manuscript only reports the mean inference time on the evaluation set, which obscures these details. A more thorough analysis showing a direct comparison of inference times for DCFold and AlphaFold3 binned by protein length would be necessary to fully assess the claimed acceleration and understand its behavior in different regimes.

2.  **Ambiguity in Pose Selection for PoseBusters Benchmark:** The methodology for selecting the "best" and "worst" poses in the PoseBusters benchmark results is unclear. If the ground truth structure was used to rank the generated poses and select these examples, this would constitute a form of data leakage, as the ground truth is not available in a real-world prediction scenario. Since ranking generated poses without a ground truth is a significant challenge in itself, the authors should clarify this selection process to ensure the validity of the reported results.

3.  **Unclear Metric Definition and Counter-intuitive Results in Table 3:** The definition of "Success Rate" used in *Table 3* is not clearly described in the manuscript. Furthermore, the results presented are somewhat surprising: the distilled, single-step model reportedly outperforms the original AlphaFold3 teacher model across the board on the Homology Recent PDB dataset. While not impossible, it is counter-intuitive for a student model to uniformly exceed the performance of its teacher. This warrants a more detailed explanation of the metric and a discussion of these results.

4.  **Insufficient Detail in Binder Hallucination Throughput:** The evaluation of the binder hallucination task lacks sufficient detail to assess the claimed speed benefits. The paper states that a "continuous 48-hour hallucination run" was performed and reports the *proportion* of designs that passed filters. However, it omits the crucial information of the *total number* of designs generated by DCFold and the baseline (BindCraft) within that time frame. Without this denominator, it is impossible to quantify the throughput advantage gained from the distillation process.

5.  **Unconventional Experimental Setup and Incomplete Results for Binder Design:** The choice of a 55-65 residue length for the binder design task is an unconventional setting that requires justification, as this is a relatively short range. More importantly, the evaluation is limited to quantitative metrics. The manuscript does not show any examples of the successfully designed binder structures. This absence of qualitative, visual evidence makes it difficult to independently assess the structural plausibility and quality of the generated binders.

### Questions
1.  **Regarding the Gradient-Based Hallucination Method:** Could the authors provide more detail on the implementation of the gradient-based hallucination? The iterative nature of diffusion models makes obtaining a direct gradient with respect to the final structure non-trivial. Methods like BoltzDesign circumvent this by using gradients from the Pairformer's internal representations. Does DCFold's single-step nature allow for a direct backpropagation through the structure generation process, similar to approaches like BindCraft, or is another mechanism used?

2.  **Stochasticity and Sampling Diversity in a Single-Step Model:** The SDE-based sampling in diffusion models like AlphaFold3 provides a beneficial stochasticity, which can capture protein dynamics and yield improved results when multiple samples are drawn. Does the single-step, distilled architecture of DCFold retain this inherent randomness? Furthermore, can DCFold's performance be improved by generating multiple samples for a single target, and if so, could the authors show the trend of performance improvement as the number of samples increases?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents DCFold, a model that greatly increases inference speed for AlphaFold3, which is very important for downstream tasks, such as virtual screening. It achieves this performance using an implementation of consistency models on both the model trunk, and the diffusion head. The authors also add a "Temporal Geodesic Matching" scheduler to stabilize consistency training. The model achieves the promised inference speed-up without sacrificing performance

### Strengths
**Originality** There have been other attempts to speed up implementations of AlphaFold3. However, the idea of using consistency training for both the trunk and the diffusion is very innovative, as well as the addition of Temporal Geodesic Matching. 

**Quality**: The experiments are thorough and convincing, and the results show a good model performance

**Clarity**: The manuscript is very well written, and clearly explained. The figures are generally good, particularly figure 2, which is a very clear illustration of the approach. 

**Significance**: As the authors explain, speeding up inference for AlphaFold3 is very important for multiple downstream applications like virtual screening

### Weaknesses
There are no major weaknesses, just some small things: 

- The meaning of ECM is never introduced

### Questions
- For Alphafold, inference time is greatly dependent on sequence length. Is this 15x time improvement consistent across lengths? 
- How did the training time compare with the training time of the original AlphaFold code? 
- The benchmarks in table 2 only compare performance against AlphaFold3. If I understand correctly, this is referring to the protenix implementation of AlphaFold3 (which never achieved the same performance). This should be clarified. If these numbers are indeed from protenix, it would also be interesting to see a comparison against other implementations (the original AlphaFold3, Boltz2, OpenFold3, etc)
- It would also be interesting to see a comparison with protenix-mini, another attempt to reduce inference time (with a different approach) as well as a discussion about the difference between the two codes. 
- The meaning of ECM is never introduced
- How is "in silico success rate" defined in Table 4? 
- In the same table, results from other binder design algorithms (e.g. RFDiffusion, FoldFlow) should also be added.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces DCFold, a single-step protein structure generation model designed to match AlphaFold3-level accuracy while significantly improving inference efficiency. The motivation stems from AlphaFold3’s diffusion-based iterative design, which, despite its accuracy, incurs high computational cost and long inference times, limiting its use in practical protein design and virtual screening workflows.  

DCFold proposes a Dual Consistency (DC) training framework that incorporates a novel Temporal Geodesic Matching (TGM) scheduler to maintain structural fidelity without iterative refinement. As a result, DCFold achieves up to 15× faster inference compared to AlphaFold3 while preserving predictive accuracy. The model’s effectiveness is validated on both structure prediction and binder design benchmarks, demonstrating strong performance with substantially reduced computation.

### Strengths
- Proposes a single-step protein generation framework that achieves AlphaFold3-level accuracy with a 15× speedup, addressing a key efficiency bottleneck in diffusion-based methods.  
- Introduces a novel Dual Consistency training scheme with Temporal Geodesic Matching, providing a principled way to maintain structural fidelity without iterative refinement.

### Weaknesses
- Lacks comparison with other training-based distillation methods such as Protenix-mini, which also reduce inference cost by compressing model size. It remains unclear how “fewer steps” (as in DCFold) compares to “smaller models” in terms of trade-offs between efficiency and performance.  
- The **PoseBusters V2 benchmark** results report “best” and “worst” performance without clarifying how these were derived; additional explanation would improve interpretability.  
- The paper provides limited details on the **binder design task** setup. It is unclear how much faster DCFold is compared to *BindCraft*, and how many samples were used to compute the *in silico* success rates.

### Questions
Q1 How does distilling to a single-step model affect aspects such as the confidence model calibration and sampling diversity?  
Q2 I found the purpose of Proposition 1 somewhat unclear—could the authors clarify its specific role in the later theoretical derivations or proofs?

### Soundness
3

### Presentation
3

### Contribution
3
