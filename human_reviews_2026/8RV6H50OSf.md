# MoBE: Mixture-of-Basis-Experts for Compressing MoE-based LLMs

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 8, 4, 8

## Abstract
The Mixture-of-Experts (MoE) architecture has become a predominant paradigm for scaling large language models (LLMs). Despite offering strong performance and computational efficiency, large MoE-based LLMs like DeepSeek-V3-0324 and Kimi-K2-Instruct present serious challenges due to substantial memory requirements in deployment. While recent works have explored MoE compression to address this issue, existing methods often suffer from considerable accuracy drops (e.g., 7-14% relatively) even at modest compression rates. This paper introduces a novel Mixture-of-Basis-Experts (MoBE) method that achieves model compression while incurring minimal accuracy drops. Specifically, each up/gate matrix in an expert is decomposed via a rank decomposition as W = AB, where matrix A is unique to each expert. The relatively larger matrix B is further reparameterized as a linear combination of basis matrices {Bi} shared across all experts within a given MoE layer. The factorization is learned by minimizing the reconstruction error relative to the original weight matrices. Experiments demonstrate that MoBE achieves notably lower accuracy drops compared to prior works. For instance, MoBE can reduce the parameter counts of Qwen3-235BA22B-2507, DeepSeek-V3-0324 (671B) and Kimi-K2-Instruct (1T) by 24%-30% with only 1%-2% accuracy drop (about 2% drops when measured relatively).

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents a framework for Mixture of Basis Experts for compressing LLMs with a relatively low lose of accuracy.  

Paper’s core claim is clearly stated (24–30% parameter reduction; ~98% relative performance retained on very large MoE LLMs).
 * Method novelty vs MoLAE/D²-MoE and pruning/merge lines is explicit. 
 * Theoretical parameter-count analysis (γ formula) is correct and assumptions are reasonable

with a mixture of shared basis matrices (convex weights) and a non-linear activation—is a clear step beyond SVD-style decompositions used by MoLAE , which rely on linear low-rank sharing.

It's not entirely novel - It builds on established rank factorisation/dictionary-learning instincts; the novelty is the specific mixture-with-activation sharing and where it’s applied in MoE.  Retaining down matrices is motivated by prior findings on where knowledge resides, rather than a new theoretical claim. 

However the results appear impressive and the work is fully justified. I think in places the writing is a little terse and perhaps the author could look at providing more of a rationale of why this should work at the start of the paper.

### Strengths
A meaningful architectural re-parameterisation of MoE experts that is novel relative to linear SVD-sharing approaches and practically validated at unprecedented model scales. 

Results seem impressive and should be reproducible (I'm assuming there will be a link to code if the paper is accepted).

### Weaknesses
Report end-to-end efficiency, not just parameter counts

Strengthen parity and scalability of baselines - D2-MoE is omitted on trillion-scale models for feasibility; include either (a) scaled-down controlled runs at matched ratios, or (b) additional scalable baselines, so large-model wins aren’t confounded by method availability. 

Broaden ablations/analyses - in particular I'd be interested in an analysis involving downstream tasks.

### Questions
Explain the expert-grouping trick for Kimi (384 experts → two 64-basis groups) and study its effect on accuracy and reconstruction error.

Could you share end-to-end inference metrics (latency, throughput, peak VRAM) for MoBE vs MoLAE/D2-MoE across sequence lengths and batch sizes, and clarify any kernel-fusion needs or runtime overheads (e.g., from Z-score normalisation)?

What practitioner guidance can you provide for selecting the number of bases 
and activated experts for MoBE), ideally with sensitivity curves showing the accuracy–compression Pareto frontier per model family?

Did you test light compression of the down projections (e.g., small-rank/partial sharing) or a short KD step, and what impact did this have on the residual accuracy gap?

How does MoBE conversion affect routing dynamics—gate logits, expert-utilisation entropy/load balance—and do you observe drift or mode collapse on tasks sensitive to routing?

How was baseline parity ensured (matched compression ratios, calibration/data, training budgets), and do you have controlled results on smaller models where D2-MoE is feasible?

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
The paper proposes an elegant and theoretically sound framework for MoE compression that leverages shared basis learning. The core concept of representing experts as combinations of a shared basis is well-motivated by the observation of functional redundancy (Fig. 1; Sec. 3.2; p.4). The experimental results appear strong, with claims of maintaining high relative accuracy (e.g., 96.8% for Qwen3-235B at 24% compression) as shown in Figure 1. However, the paper suffers from significant presentation issues, most notably the absence of critical referenced tables (e.g., Table 3), which prevents a full verification of the quantitative claims. The analysis of effective rank is referenced to a non-existent figure (Fig. 9), further hindering a complete assessment.

### Strengths
- **Novel and theoretically sound compression framework**
  - The MoBE formulation, where each expert is a weighted sum of basis experts, provides a principled way to capture and exploit inter-expert redundancy ($\text{Expert}\_i = \sum\_j \alpha\_{ij} \cdot \text{Basis}\_j$, Eq. 1; Sec. 3.2; p.4). This is a clear and impactful contribution.
  - The framework naturally separates shared knowledge (the basis experts) from specialized knowledge (the combination coefficients), offering a more structured approach to compression than unstructured pruning.
  - The two-stage training process, which first initializes the basis and then jointly optimizes all components, is a logical and practical approach to tackling the complex optimization problem (Sec. 3.3; p.5).
- **Strong conceptual motivation and clear illustrations**
  - The paper provides a clear motivation for the MoBE approach, contrasting it with existing methods and highlighting the limitations of pruning and independent decomposition (Sec. 2; p.2-3).
  - The architectural diagram (Fig. 3; Sec. 3.2; p.4) is clear and effectively communicates the core components of the MoBE block, including the basis experts and the learned combination coefficients.
  - The conceptual illustration of relative performance (Fig. 1; p.1) provides a high-level summary of the method's claimed effectiveness, showing favorable comparisons against other methods like MoLAE and MoE-SVD.

### Weaknesses
- **Missing key experimental results and references**
  - The paper repeatedly references **Table 3** for key quantitative results that are central to its claims of outperforming baselines. However, **Table 3 does not exist** in the manuscript or its appendices. This is a critical omission that makes it impossible to verify the core experimental findings.
  - The review references **Table 8** and **Figure 9** in the appendices for further analysis, but these elements are also **not found** in the provided document. This suggests either a flawed manuscript or a flawed review process.
  - Without these key tables and figures, the paper's claims of superior performance are unsubstantiated and rely solely on high-level plots (e.g., Fig. 1).
- **Ambiguous or incorrect references**
  - The analysis of effective rank is attributed to "Appendix C; Fig. 9; p.14". As noted, Figure 9 is missing. The paper does contain an analysis of MSE loss in **Figure 2** (p.3), which might be what was intended, but this discrepancy creates confusion.
- **Limited details on the training and optimization process**
  - The paper provides a high-level overview of the two-stage training process but lacks specific details on the hyperparameters, learning rates, and convergence criteria for each stage (Sec. 3.3; p.5). This limits reproducibility.
  - The mechanism for determining the number of basis experts (a critical hyperparameter) is not well-described or justified. A sensitivity analysis on this parameter would be crucial.

### Questions
- **Include all referenced tables and figures**
  - The most critical suggestion is to **include the missing Table 3, Table 8, and Figure 9** in the manuscript. Without these, the paper is incomplete and its claims cannot be verified.
  - Ensure that all references in the text point to the correct tables, figures, and sections. The manuscript should be carefully proofread to correct any such errors.
- **Provide comprehensive details on the training process**
  - Add a dedicated subsection or appendix detailing the hyperparameters used for both stages of the training process, including learning rates, batch sizes, schedulers, and convergence criteria.
  - Include a sensitivity analysis on the number of basis experts. This could be a plot showing the trade-off between the number of basis experts, the compression ratio, and the model's performance on a validation set.
- **Clarify the experimental setup and results**
  - Once Table 3 is included, ensure it is well-described and that all columns and rows are clearly labeled. The main text should walk the reader through the key results in the table.
  - Provide a more detailed analysis of the results, going beyond relative accuracy to discuss performance on specific tasks or benchmarks.

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
3

### Summary
The paper proposes MoBE, Mixture-of-Basis-Experts, to compress MoE-based LLMs. Each expert’s up/gate matrix is factorized by the rank decomposition W=AB; B is shared as a linear combination of a set of shared basis matrices, while A is specific for each expert. By minimizing the reconstruction error plus z-score normalization and suitable activations (e.g., SiLU/Tanh), MoBE reduces total parameters by 24%-30% while keeping 98% performance on many benchmarks, outperforming MoLAE and D^2-MoE at similar ratios.

### Strengths
- The problem is important for deploying trillion-level MoE. The idea is simple but quite effective. The paper proposes to decompose the up/gate matrix into shared basis matrices B across experts to capture the common information across experts and keep matrix A per expert to encode specific information, and to add non-linearity inside the matrix factorization to enhance representational power.
- The paper is well-written. The equations and algorithm steps are easy to follow.
- The paper conducts extensive experiment to show that this method is applicable to very large MoE including Qwen3- 235B-A22B-2507, DeepSeek-V3-0324 and Kimi-K2-Instruct where many baselines are infeasible. It is very pratical for memory-bound inference.

### Weaknesses
- The paper comes with limited theory of formal approximation guarantees. Most are from empirical studies.
- The choice of hyper-parameters lacks guidance, including the choice of basis count m and the rank r. The compression rate and the accuracy frontiers are not fully mapped.
- No study of light-weight finetuning or knowledge distillation to close the last 1%-2% gap.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
