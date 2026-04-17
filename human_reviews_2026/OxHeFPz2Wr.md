# $\texttt{RNAGenScape}$: Property-Guided Optimization and Interpolation of mRNA Sequences with Manifold Langevin Dynamics

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2

## Abstract
mRNA design and optimization are important in synthetic biology and therapeutic development, but remain understudied in machine learning. Systematic optimization of mRNAs is hindered by the scarce and imbalanced data as well as complex sequence-function relationships. We present $\texttt{RNAGenScape}$, a property-guided manifold Langevin dynamics framework that iteratively updates mRNA sequences within a learned latent manifold. $\texttt{RNAGenScape}$ combines an organized autoencoder, which structures the latent space by target properties for efficient and biologically plausible exploration, with a manifold projector that contracts each step of update back to the manifold. $\texttt{RNAGenScape}$ supports property-guided optimization and smooth interpolation between sequences, while remaining robust under scarce and undersampled data, and ensuring that intermediate products are close to the viable mRNA manifold. Across three real mRNA datasets, $\texttt{RNAGenScape}$ improves the target properties with high success rates and efficiency, outperforming various generative or optimization methods developed for proteins or non-biological data. By providing continuous, data-aligned trajectories that reveal how edits influence function, $\texttt{RNAGenScape}$ establishes a scalable paradigm for controllable mRNA design and latent space exploration in mRNA sequence modeling.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a property-guided manifold Langevin dynamics framework for mRNA sequence optimization and interpolaton named as RNAGenScape.
It operates within a learned latent manifold representating valid mRNA sequences. It consists of an autoencoder (OAE) that maps discrete sequences into latent vectors, a manifold projector that maps off-manifold updates back to the manifold, and a property-guided Langevin dynamics mechanism for property optimization and interpolation. 
The authors show that RNAGenScape achieves high success rates in property optimization and preserves manifold fidelity across three real mRNA datasets. It supports smooth interpolation between mRNAs, revealing interpretable and biologically meaningful trajectories

### Strengths
1. The proposed method keeps all intermediate sequences close to biologically valid manifolds, ensuring plausibility during design.

2. The authors adapt a SUGAR-based data augmentation technique, which enriches the latent space with geometry-preserving samples, helping the model better approximate the underlying mRNA manifold even in sparse regions.

3. The method remains computationally efficient and supports both property optimization and interpolation.

### Weaknesses
1. From a machine learning perspective, the core novelty of RNAGenScape lies in employing a diffusion-inspired denoising objective to train a manifold projector that maps perturbed latent points back onto the learned data manifold. The use of latent-space optimization and interpolation for property control is, however, well established in the VAE and normalizing flow literature. That being said, the novelty should not be a great concern if the experimental results are convincing enough.

2. The overall optimization framework fundamentally depends on the fidelity of the organized autoencoder (OAE) and the quality of its learned latent manifold. If the OAE fails to capture the true biological manifold of viable mRNA sequences, the downstream optimization may generate invalid or functionally implausible sequences, potentially performing worse than simple sequence-space methods such as MCMC.

3. The current formulation performs optimization guided by gradients from a property predictor P that is co-trained with the OAE. Consequently, optimizing for different biological properties requires retraining the entire model, which is computationally inefficient. Moreover, the framework currently supports only single-property (scalar) optimization. Extending it to multi-objective or conditional optimization (e.g., jointly improving translation efficiency, stability, and immunogenicity) would substantially broaden its applicability and realism in practical mRNA design tasks.

4. No wet-lab validation is presented to confirm the real-world biological functionality of the optimized sequences. The evaluation relies solely on a separate property prediction oracle. yet the training procedure, data sources, and potential overlap with the main model remain insufficiently detailed. I encourage the authors to provided detailed description on this.

5. The setup of experiments are not sufficiently clear. Please see the question part.

### Questions
1. What is the detailed protocol used to perform property optimization for the different baselines? For example, in the case of Sequence-space MCMC, is the oracle property predictor employed to evaluate proposals? I encourage the authors to include these details in the appendix for reproducibility. In addition, please clarify how the stopping criterion is determined for each optimization method, whether a fixed number of steps, convergence threshold, or early-stopping rule is used.

2. While RNAGenScape preserves manifold fidelity across multiple datasets (Table 2), it would also be informative to report the edit distance between optimized sequences and their corresponding original sequences. This would help quantify how much modification occurs during optimization and how conservative or aggressive the updates are.

3. Sequence-space MCMC can serve as a strong baseline when the proposal distribution is well designed. It is therefore intriguing that RNAGenScape outperforms MCMC by such a large margin, especially in terms of success rate. Could the authors provide additional insights or diagnostic analyses on the behaviors of different optimization methods to explain this discrepancy?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
# Main Idea
RNAGenScape is a property-guided manifold Langevin dynamics framework for mRNA sequence design. It comprises: (1) an organized autoencoder (OAE) whose latent space is explicitly aligned to a target property; (2) a manifold projector that maps updates back onto the learned mRNA manifold; and (3) a property-guided manifold Langevin procedure for both optimization (improving a property for a given sequence) and interpolation (connecting two sequences by a smooth trajectory), with SUGAR used to enrich undersampled regions when data are sparse. 

# Contributions
1. Framework: A manifold Langevin framework enabling continuous, property-guided optimization and interpolation between real mRNA sequences. 

2. Manifold constraint: A learned manifold projector to keep trajectories biologically plausible by contracting updates back to the data manifold. 

3. Efficiency: Operates on the manifold and starts from existing sequences, improved training and inference efficiency backed by experimental results.

4. Empirical validation: Results on three mRNA datasets show improved property control with strong manifold fidelity and competitive efficiency versus presented *de novo* generation and optimization baselines.

### Strengths
* The paper presents extensive optimization benchmarks with strong results: across datasets it achieves top property control relative to de novo generators and optimization baselines while also maintaining high manifold fidelity.

* I like that it is also notably interpretable, showing smooth, coherent interpolation paths between arbitrary sequence pairs with monotonic distance profiles from source to target along the learned manifold.

* The authors' employment of SUGAR tackles the data scarcity issues of RNA modality hindering computational modeling, and ablation shows smaller property datasets benefit more from geometry-preserving upsampling.

### Weaknesses
-RNAGenScape currently handles only one property per model, as acknowledged in **Section 6 (Limitations and Future Work)**, which really restricts any therapeutic design applications, where multiple sequence-function trade-offs are very important.

-Figures 2–5, Table 3, and Supplementary Figures S1–S3 show latent-space trajectories and optimization behavior, but the manuscript does not explicitly state which property/dataset (stability, ribosome loading, or translation efficiency) underlies each visualization? This makes it difficult for me to interpret what biological regime the trajectories represent.

- I find that the interpolation results, while visually smooth, are not clearly connected to any biologically relevant or therapeutic use case. The paper only notes that interpolation “facilitates the exploration of intermediate variants” (Section 4. Empirical Results) without showing whether interpolated sequences actually achieve meaningful Pareto trade-offs between properties such as translation efficiency and stability. 

-The paper states that RNAGenScape optimizes from existing sequences rather than sampling from noise but does not describe how those starting sequences are chosen, how diversity is ensured, or how bias toward specific datasets is avoided.

-The authors' evaluation omits comparisons against recently released or widely used mRNA-specific generative or optimization models. Without such baselines, it is difficult for me to judge whether RNAGenScape’s manifold-Langevin approach offers a distinct advantage over modern foundation or fine-tuned mRNA design models.

### Questions
-Which specific property and dataset were used to generate the optimization trajectories in Figures 2–5 and the supplementary figures?

-Can the author show whether RNAGenScape generalizes equally across translation efficiency, stability, and ribosome-loading tasks, or are some properties systematically easier to optimize given data availability?

-The authors must execute a wet-lab experiment to test whether interpolated sequences actually yield intermediate or improved functional properties (e.g., balancing efficiency and stability in a reporter assay)? Unfortuantely, without this, the paper will lack relevance in the literature. 

-The authors should define anexact protocol for selecting the initial sequences used for optimization, and how do the authors ensure diversity and avoid potential bias?

-The authors consider benchmarking RNAGenScape against recent mRNA-specific generative frameworks (mRNAutilus, etc.) and conduct wet-lab validation comparing its optimized sequences against those produced by such models to support the claimed practical advantage?

Overall, this work cannot be a publishable paper without real-world applicability. I would encourage the authors to perform wet lab validations and resubmit as a journal paper, not a main conference work.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces RNAGenScape, a framework for optimizing mRNA sequences for desired biological properties. The method learns a latent manifold of valid sequences using an organized autoencoder. It then performs property-guided optimization within this manifold using Langevin dynamics. A key component is a learned manifold projector that ensures each optimization step results in a biologically plausible sequence. The authors evaluate their method on three mRNA datasets, reporting superior performance in property optimization and demonstrating smooth interpolation between sequences.

### Strengths
1. Originality & Significance. The paper tackles the significant problem of property-guided mRNA design. Its core idea of constraining Langevin dynamics to a learned manifold via a dedicated projector is a novel and principled solution. This combination of techniques addresses the critical challenge of ensuring biological plausibility during optimization, a common failure point for other methods.
2. The methodology is validated through high-quality experiments. The authors benchmark their method against a comprehensive suite of baselines across three diverse, real-world datasets. The inclusion of detailed ablation studies further strengthens the empirical evidence.
3. The core concepts are explained effectively, and the mathematical formulations are easy to follow.

### Weaknesses
1. Overly Detailed Preliminaries. Section 2, the preliminaries, is excessively long and covers several foundational concepts. This space could have been used more effectively for more important results. A more concise summary would improve the paper's focus.
2. Lack of Direct Evaluation for the Manifold Projector. The manifold projector is a central contribution of this work, yet its performance is only assessed indirectly through its impact on the final optimization results. The paper lacks a direct, quantitative evaluation of the projector's quality. It is unclear how accurately the projector approximates the true data manifold. This is a critical omission, as the entire claim of maintaining "biological plausibility" rests on the projector's effectiveness. Without this, it is difficult to disentangle the performance of the projector from that of the OAE and the Langevin dynamics.
3. Insufficient Justification for Sequence Interpolation. The paper frames sequence interpolation as a key feature, but its practical utility is not well established. The authors claim it facilitates "the exploration of intermediate variants", but they do not demonstrate how these intermediate sequences provide concrete biological insights or inform the design process in a way that endpoint optimization does not. A more compelling use case is needed to elevate this from a technical demonstration to a significant contribution.
4. Missing Comparison with mRNA Sequence Foundation Models. The baselines overlook the recent paradigm of large-scale pre-trained foundation models for mRNA sequences (e.g., Uni-RNA). These models capture rich, transferable representations of sequence grammar and function. It is essential to discuss how RNAGenScape compares to or could potentially leverage these models.
5. Critical Omission of Oracle Model Validation. The entire evaluation framework relies on a separately trained P_oracle model to score the generated sequences. However, the paper provides no information about this oracle's predictive accuracy on a held-out test set (e.g., Spearman correlation). If the oracle is inaccurate or shares the same biases as the OAE's internal predictor, the reported optimization improvements could be misleading artifacts of the model exploiting the oracle's flaws. This methodological gap is a major concern that undermines the trustworthiness of the results.
6. Potentially Biased Latent Space Distance Metric. The "Manifold Fidelity" metric is defined as the L2 distance in the latent space of the oracle's encoder. While using a separate model is a reasonable choice, this metric is still dependent on a learned representation. It does not guarantee that sequences that are close in this latent space are necessarily similar in terms of biological structure or function. Supplementing this with a representation-independent metric like edit distance would make the fidelity analysis more robust.

### Questions
1. Oracle Model. Could you please provide the architecture, training details, and performance metrics (e.g., R-squared and/or Spearman correlation on a held-out test set) for the P_oracle model used in your evaluation? 
2. Manifold Projector. Could you provide a more direct evaluation of the manifold projector? For instance, what is its reconstruction error on a held-out set of latent codes z corrupted by varying levels of noise?
3. Interpolation Utility. Could you elaborate on a specific biological or therapeutic design problem where analyzing the intermediate sequences from interpolation provides actionable insights that cannot be obtained from simply optimizing a start and end point?
4. Foundation Models. How do you see RNAGenScape positioned relative to large nucleotide foundation models? Have you considered initializing your OAE from a pre-trained model or using its embeddings as a starting point to see if it improves performance?

### Soundness
2

### Presentation
2

### Contribution
2
