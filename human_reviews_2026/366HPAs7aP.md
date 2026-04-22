# ContextFlow: Context-Aware Flow Matching for Trajectory Inference from Spatial Omics Data

- Avg Score: 4.80
- Decision: Reject
- Scores: 6, 4, 4, 4, 6

## Abstract
Inferring trajectories from longitudinal spatially-resolved omics data is fundamental to understanding the dynamics of structural and functional tissue changes in development, regeneration and repair, disease progression, and response to treatment. We propose ContextFlow, a novel context-aware flow matching framework that incorporates prior knowledge to guide the inference of structural tissue dynamics from spatially resolved omics data. Specifically, ContextFlow integrates local tissue organization and ligand-receptor communication patterns into a transition plausibility matrix that regularizes the optimal transport objective. By embedding these contextual constraints, ContextFlow generates trajectories that are not only statistically consistent but also biologically meaningful, making it a generalizable framework for modeling spatiotemporal dynamics from longitudinal, spatially resolved omics data. Evaluated on three datasets, ContextFlow consistently outperforms state-of-the-art flow matching methods across multiple quantitative and qualitative metrics of inference accuracy and biological coherence.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces ContextFlow, a flow matching framework that incorporates spatial context from spatially-resolved omics data to infer biologically plausible cellular trajectories over time. The key innovation is integrating two spatial priors—(1) local tissue organization via spatial smoothness and (2) ligand-receptor communication patterns—into a transition plausibility matrix that regularizes optimal transport couplings. The authors propose two integration schemes: Prior-Aware Cost Matrix (PACM) and Prior-Aware Entropy Regularization (PAER). They demonstrate that ContextFlow outperforms baseline methods (CFM, MOTFM) across three datasets (axolotl brain regeneration, mouse organogenesis, liver regeneration) under both interpolation and extrapolation settings, using metrics including Wasserstein distance, MMD, energy distance, and a weighted Wasserstein metric based on cell type compositions.

### Strengths
1. **Well-motivated problem**: Incorporating spatial context into trajectory inference addresses a real limitation of existing methods and is timely given advances in spatial transcriptomics.

2. **Principled approach**: The integration of biological priors through OT regularization is theoretically grounded, and Theorem 1 ensures computational tractability via Sinkhorn.

3. **Comprehensive experiments**: Three diverse datasets with both interpolation and extrapolation settings. The biological analysis (e.g., excitatory-inhibitory transitions in Figure 3) provides qualitative validation.

4. **Two integration schemes**: Offering both PACM and PAER gives practitioners flexibility, and showing PAER often performs well without extra hyperparameter tuning is valuable.

5. **Reproducibility**: Algorithm 1 and extensive ablations in the appendix support reproducibility.

### Weaknesses
1. **Limited novelty**: The core contribution—adding prior knowledge to OT objectives—is incremental. PACM is a straightforward weighted cost, and PAER follows naturally from KL divergence regularization literature. The connection to existing spatial transcriptomics OT methods (TOAST, DeST-OT) needs clearer differentiation.

2. **Experimental limitations**:
   - **Modest improvements**: Many results show <5-10% improvement over MOTFM, often within error bars
   - **Extreme instability**: Table 12 shows catastrophic failures in hard extrapolation (errors 10^5-10^6), severely limiting practical utility
   - **Limited baselines**: No comparison to established trajectory inference methods (e.g., Waddington-OT, CellRank, RNA velocity-based methods)
   - **Missing ablations**: No systematic study of spatial smoothness vs. LR communication independently

3. **Evaluation concerns**:
   - **Weighted W2 confounding**: Depends on XGBoost classifier quality; unclear if improvements reflect better trajectories or classifier biases
   - **No statistical testing**: Despite reporting standard deviations, no significance tests are provided
   - **Biological validation**: The excitatory-inhibitory analysis (Figure 3) is interesting but limited to one dataset and one specific biological phenomenon

4. **Hyperparameter sensitivity**: Despite claims that PAER avoids tuning, λ still needs selection. Tables 5-13 show considerable variance, and there's no principled guidance for setting these parameters.

5. **Scalability and generalizability**:
   - Computational complexity not discussed
   - Limited to three datasets from specific biological contexts
   - Unclear how to set the neighborhood radius r or select relevant LR pairs for new datasets

6. **Technical issues**:
   - The normalization analysis (Propositions 1-2) seems tangential
   - Equation 10 assumes a linear combination of SS and LR—no justification for this specific form
   - The prior matrix construction (Equation 10) is somewhat ad-hoc

### Questions
1. **Ablation studies**: Can you provide results using only spatial smoothness (λ=1) or only LR communication (λ=0) systematically across all datasets and metrics? This would clarify the relative importance of each prior.

2. **Baseline comparisons**: How does ContextFlow compare to other trajectory inference methods like Waddington-OT, CellRank, or RNA velocity approaches? Even if they don't use spatial information, they provide important benchmarks.

3. **Weighted W2 metric**: How is the XGBoost classifier trained? What is its accuracy? Could you provide results without this metric to separate trajectory quality from classifier performance?

4. **Stability**: Table 12 shows extreme instability. Can you characterize when and why these failures occur? Are there early warning signs or ways to detect when the method will fail?

5. **Hyperparameter selection**: Can you provide principled guidance for setting λ, α, and ε for new datasets? Cross-validation results would be helpful.

6. **Computational cost**: What is the runtime comparison vs. MOTFM? How does it scale with dataset size?

7. **Generalization**: Can you provide evidence that the learned velocity fields capture meaningful biological dynamics beyond the specific datasets tested?

8. **LR database sensitivity**: How sensitive are results to the choice of LR database or communication inference method?

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
This paper introduces ContextFlow, a flow-matching framework designed to infer cellular trajectories from snapshot spatial omics data. ContextFlow integrates two types of spatial priors (Spatial Smoothness, Cell-Cell Communication Patterns) to the computation of OT couplings, and then performs conditional flow matching, which can help avoid biologically implausible trajectories. Specifically, these priors are encoded into a Transitional Plausibility Matrix (TPM) and incorporated into the OT objective via two schemes: Prior-Aware Cost Matrix (PACM), and Prior-Aware Entropy Regularization (PAER). Through experiments on three datasets, ContextFlow outperforms CFM and OT-CFM in interpolation and extrapolation tasks, and reduces biologically invalid cell lineage transitions.

### Strengths
- Inferring trajectories from spatiotemporal data is a critical and challenging task in computational biology.
- This paper is well-motivated, as it aims to tackle the limitations of other OT-based methods, where these methods may induce biologically implausible transitions.
- Incorporating biological priors into the OT coupling is a natural extension. By modifying the Gibbs kernel, the entropy-based regularization guides the transport plan in a theoretically sound manner while simplifying the hyperparameter tuning process.

### Weaknesses
Despite its promising direction, the paper suffers from several critical weaknesses that question the validity and significance of its central claims.
- **Omission of Spatial Coordinates:** The first concern is the disregard for Euclidean spatial distance in the OT cost function for both the proposed method and the baseline. For spatial data, the most direct and powerful prior is that cells cannot transport across the tissue in a short time interval. I believe it is important to penalize the physical distance between matched cells. The paper's "spatial smoothness" prior is an indirect and potentially weak proxy for this physical constraint. It is possible for two distant regions to have similar neighborhood expression profiles, which could lead to **biologically impossible long-range couplings.** The authors' claim in Appendix G.1 that ContextFlow can "restrict transitions to within the same hemisphere" is questionable when visually inspecting Figure 3(b). This figure still appears to show several long-range couplings that may not be biologically reasonable. One way to examine this is to evaluate the reconstructed spatial coordinates at hold-out time points.
- **Inadequate Baselines:** As said, it seems that the spatial information is not incorporated into baselines, which may be unfair. I also recommend that the authors compare their proposed method against some newer flow-matching baselines, such as Metric Flow Matching [1], some neural-ODE baselines, such as DeepRUOT[2], and **methods specifically designed for mapping spatially resolved data**, such as moscot [3].
- **Theoretical Gap Between Static Coupling and Dynamic Trajectory Generation:** The work relies on the Conditional Flow Matching framework, which assumes a linear interpolation path between two coupled endpoints. While ContextFlow constrains the choice of these endpoints, it provides **no guarantee about the validity of the path between them.** A straight line in high-dimensional gene expression space is likely to break those proposed constraints and induce biologically implausible transitions. This paper fails to provide any theoretical or compelling empirical justification for this point. I think the authors should examine the biologically implausible transitions along the whole trajectory instead of only endpoints.
- **Hyperparameter Selection:** It seems that the paper does not specify the value of entropy regularization parameter $\epsilon$ used in the experiments, nor does it describe the strategy for its selection or provide a sensitivity analysis. I assume this is an important parameter to constrain the OT coupling, so this omission is significant for the reproducibility and a full understanding of the method's robustness. I recommend providing a guidance on how to select all these hyperparameters in this paper for different datasets, and discussing the efforts for hyperparameter tuning.
- **Minor Points:** 
    1. The notation CTF-C and PACM (or CTF-H vs. PAER) is a bit confusing. If they refer to the same concept, I recommend using one consistent term throughout the paper.
    2. Equation 4, should that be $(x-\mu_{t}(z))$?
    3. Equation 9 should be more clearly stated. It is unclear how this is computed in practice.
    4. The construction of the TPM involves neighborhood calculations and ligand-receptor interaction inference, which can be computationally intensive as the number of cells grows. The authors should provide runtime of the whole pipeline.
    5. I think the concept of constrained OT is a well-established field. The authors should provide a more thorough discussion in the related work section to better position their specific contribution.

[1] Kapusniak, Kacper, et al. "Metric flow matching for smooth interpolations on the data manifold." NeurIPS
[2] Zhang, Zhenyi et al., "Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport". ICLR
[3] Klein, Dominik, et al. "Mapping cells through time and space with moscot." Nature

### Questions
Following the above weaknesses:
- How well can ContextFlow reconstruct spatial coordinates in hold-out time points?
- Could the authors provide comparison results against more appropriate baselines?
- Can the authors provide justification (theoretical or empirical) for why the linear path assumption in CFM is sufficient to produce biologically reasonable trajectories, given that the proposed constraints are only applied at the endpoints?
- How is the parameter $\epsilon$ set for different datasets?
- How is the runtime of ContextFlow?

If the authors can address the weaknesses and questions outlined, I will consider revising my score.

The reviewer wrote this review. LLM was utilized only to correct grammar and enhance the clarity of this review.

### Soundness
3

### Presentation
3

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
This paper addresses the challenge of inferring cellular trajectories from longitudinal, spatially-resolved omics data. In this setting, the vanilla Optimal Transport (OT) relies solely on transcriptomic similarity, potentially leading to biologically implausible trajectories. The authors propose ContextFlow to integrate spatial and biological priors for the OT batch sampling step of CFM. This is achieved by constructing a Transitional Plausibility Matrix (TPM) encoding Spatial Smoothness (SS) and Cell-Cell Communication (CCC) patterns. ContextFlow introduces two integration mechanisms: cost-based (CTF-C) and entropy-based (CTF-H). Experiments demonstrate improved statistical fidelity and biological plausibility over baselines.

### Strengths
- **S1: Motivation.** Incorporating spatial context and biological priors (LR interactions) is highly relevant for trajectory inference in spatial biology.

- **S2: Principled Integration (CTF-H).** The PAER (CTF-H) approach elegantly incorporates priors via entropy regularization, avoiding normalization issues associated with modifying the cost matrix (CTF-C).

- **S3: Comprehensive Evaluation.** Experiments on three diverse datasets demonstrate consistent quantitative improvements across various metrics and sampling scenarios (Interpolation, Extrapolation, IVP).

- **S4: Biological Plausibility.** Qualitative analysis (Appendix G.1) convincingly shows a substantial reduction in biologically implausible couplings (e.g., lineage switches) compared to MOTFM.

### Weaknesses
- **W1: Sensitivity and Interpretation of Hyperparameters ($\lambda, r$).** The method relies heavily on the neighborhood radius $r$ and the trade-off $\lambda$. The paper lacks ablation on $r$. Crucially, empirical results (e.g., Table 2) show the best performance often occurs at extremal values ($\lambda=0$ or $\lambda=1$), suggesting either SS or LR dominates. The paper does not provide biological justification for why one prior dominates in specific datasets or guidance on how to select $\lambda$ a priori.

- **W2: Ambiguity in PAER Formulation.** The normalization of the prior matrix $\hat{\mathbf{M}}$ in PAER (Line 301) appears row-wise, creating conditional probabilities rather than a joint distribution. The interpretation of the regularization term $H(\Pi \mid \hat{\mathbf{M}})$ requires clarification if $\hat{\mathbf{M}}$ is not a valid joint measure.

- **W3: Exclusion of Direct Spatial Distance.** The method uses neighborhood features but does not explicitly use physical distance ($||\mathbf{s}(c_i) - \mathbf{s}(c_j)||$) to constrain cell movement.

### Questions
1. **(W1) Lambda Interpretation:** In the results (e.g., Tables 1 and 2), the best performance is often achieved when $\lambda=0$ or $\lambda=1$. What does this imply about the synergy between SS and LR priors? Can you provide biological justification for why one prior dominates in specific datasets, and how should $\lambda$ be selected a priori?

2. **(W1) Hyperparameter Sensitivity:** Could you provide an ablation study on the selection of the neighborhood radius $r$ and the method's sensitivity to this parameter?

3. **(W2) PAER Formulation:** Regarding Line 301, the normalization of $\hat{\mathbf{M}}$ sums only over the target index $l$, resulting in a row-stochastic matrix. Is this intended? If so, please clarify the interpretation of $H(\Pi \mid \hat{\mathbf{M}})$ when $\hat{\mathbf{M}}$ is not a joint distribution.

4. **Strength of Regularization:** Can you elaborate on the mechanism by which the TPM-guided regularization (structured diffusion) so strongly affects the resulting trajectories, effectively pruning biologically implausible paths compared to standard EOT diffusion?

5. **Proofreading (Theorem 1):** In Appendix B, Line 914 seems to have a sign error for $g_l$ compared to the derivation from the Lagrangian (Line 912). Should it be $\epsilon \log (\Pi_{kl}^*/M_{kl}) = f_k + g_l - C_{kl}$?RetryTo run code, enable code execution and file creation in Settings > Capabilities.

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
This paper introduces a framework for improving the biological plausibility of trajectory inference from longitudinal spatial omics data.
The methodology is sound, and the experiments are thorough.

The core of the contribution is the method for integrating spatial priors into an OT-coupled flow matching framework.
By defining a "Transitional Plausibility Matrix" based on spatial smoothness and cell-cell communication, this approach moves beyond simple transcriptomic-only optimal transport (like MOTFM) to better pinpoint biologically realistic cellular trajectories.

### Strengths
The core contribution (the context-aware OT coupling, particularly the CTF-H entropy-regularized variant) is to exploit biological information, specifically the position of the cells, to regularize the transport plan towards biologically-informed priors, and its stability is supported by Theorem 1.

The authors provide empirical evidence that ContextFlow outperforms the state-of-the-art MOTFM baseline on multiple metrics.
Also, the study directly evaluates the biological plausibility of the inferred couplings (Fig. 3), providing clear evidence that the method successfully reduces the number of known-implausible transitions.

### Weaknesses
Some mathematical notations are ambiguous and inconsistently applied throughout the manuscript, making it difficult for the reader to follow the derivations and arguments, especially starting from Section 2.3.

Concrete examples are:
- The use of $[\cdot]$ notation for a set in Section 3.1 is unusual. Later on, the authors use ${\cdot}$ notation for a set in Section 3.2. Later on, $[\cdot]$ is used for an element of a matrix.
- In Section 3.1, $n_i$ is never explained.

The paper frames the problem primarily as an improvement over context-free flow matching (MOTFM).
However, there are existing methods specifically designed for spatial-temporal trajectory inference, e.g., DeST-OT, and also other methods against which DeST-OT was compared, such as Moscot, PASTE, SLAT, and STalgin, which are mentioned in the related work.
They tested their approach on the same data, i.e., Axolotl brain development, meaning direct comparison should be possible.
Discussing why a flow-matching approach might be preferable or complementary (e.g., generation of continuous trajectories vs. discrete couplings) would provide a more complete picture of the landscape.
Furthermore, in DeST-OT, there are multiple metrics introduced that would be interesting to see, such as the metric of cell migration and the metric of growth distortion.

The paper also states that CTF-H avoids the need for additional tuning, but this only refers to the $\alpha$ parameter in CTF-C.
Both methods still rely on the $\lambda$ parameter (Eq. 10) to balance the SS and LR priors.
The paper also fails to address the severity of this dependency.
Its own sensitivity analysis reveals that it is not a simple trade-off, but a point of high fragility.
For instance, in the challenging IVP interpolation task (Table 11), CTF-H with $\lambda = 1$ improves on the baseline, but with $\lambda = 0.5$, its performance collapses.
The paper provides no discussion of this instability or any heuristic for how to set this crucial parameter, which appears to be non-trivial and highly non-linear.
Additionally, there is an apparent inconsistency in the data.
For the IVP interpolation task, Table 11 shows improvement on the baseline with $\lambda = 1$ and performance collapses with $\lambda = 0.5$, whereas Table 3 appears to show improvement on the baseline with $\lambda = 0.5$ and performance collapses with $\lambda = 1$.

The paper introduces new computational steps (constructing the TPM, running a modified Sinkhorn algorithm), but provides no analysis of their computational complexity or practical runtime.
It is unclear how this method would scale to datasets with hundreds of thousands or millions of cells.
Furthermore, there are parameters, such as $r$, and also the hyperparameters of LIANA+, which would heavily influence the performance, yet there is no explanation or ablation study regarding those parameters.

The examples show that the framework can reduce biologically implausible couplings (from 54 to 24 in Fig. 3), which is a strong result.
However, 24 implausible couplings is still significant.
To fully substantiate the claim of generating biologically consistent trajectories, it would be powerful to showcase an example of a novel insight or a corrected trajectory that was non-obvious.

### Questions
- What are the potential limitations of the Spatial Smoothness prior (Eq. 8)?
- How robust is the framework's performance to the choice of the upstream cell-cell communication inference tool used to generate the LR prior?
- What is the computational complexity of constructing the TPM and running the prior-aware Sinkhorn algorithm, and how does the method's runtime scale as the number of cells increases?
- How was the trade-off parameter $\lambda$ (Eq. 10) selected for the final results? Since this parameter still needs to be set (or tuned) for both CTF-C and CTF-H, please clarify the claim that CTF-H reduces tuning overhead.
- There is no ablation study regarding the parameter $r$ and the hyperparameters of LIANA+ approach. How did your approach perform under different parameters, and how were they set?
- Why is the number of principal components set to 50 for all datasets, and is there no feature selection by selecting the highly variable genes?
- As mentioned above, the data presented in Tables 3 and 11 is inconsistent. The authors should review these tables for accuracy and correct them as needed.
- The priors are defined between consecutive time points. How would this framework handle missing time points (e.g., inferring a trajectory from $t_1$ to $t_3$)? Would the SS and LR priors still be valid over such a large temporal gap?
- Why were comparisons limited to other flow-matching methods? The paper would be strengthened by a comparison to other SOTA trajectory inference methods that are already spatially-aware, such as some of the OT-based alignment methods mentioned in the related work (e.g., DeST-OT).
- Can you provide an example where your model identifies a trajectory that reveals a previously non-obvious biological insight, rather than primarily confirming known plausible/implausible transitions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
ContextFlow introduces a context-aware flow-matching framework for inferring tissue dynamics from longitudinal spatial omics data. By integrating information on tissue organization and ligand–receptor interactions into the model, it regularizes trajectory inference to produce biologically interpretable and statistically consistent results. Evaluations on real world datasets demonstrate that ContextFlow outperforms existing flow-matching approaches in both accuracy and biological coherence.

### Strengths
1.	Incorporating biological priors to regularize the flow is quite useful in the field of spatial transcriptomics analysis, as it provides biologically meaningful interpretations.
2.	The overall presentation of this paper is comprehensive, and the results show the proposed method outperform the baseline in different metrics.
3.     The approach seems quite novel, and the combination of the model and this application is an important contribution.

### Weaknesses
1.	Although the authors benchmark their model on real datasets, I recommend also including evaluations on simulated datasets, as these provide unbiased ground truth for performance assessment.
2.	It would be helpful to organize the related work section more clearly, for example by summarizing the advantages of different methods in a table, as the current related work section is not well organized.
3.	If possible, it would be helpful to also include a benchmarking comparison with the DeST-OT method mentioned in the related work. Since one key advantage of ContextFlow is the incorporation of biologically meaningful priors, such as cell–cell communication information, and DeST-OT can model cell differentiation processes, a comparison between these two methods would be interesting.

### Questions
1.	From the results, it appears that CTF-H performs better than CTF-C. Could the authors provide a general explanation for this observation? For instance, do different prior integration strategies have distinct advantages depending on the application scenario?
2.	For the Stereo-seq dataset, what is the resolution for this dataset, did the author used cell binning during the preprocessing steps.

### Soundness
3

### Presentation
3

### Contribution
3
