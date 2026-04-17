# High-Fidelity Scientific Simulation Surrogates via Adaptive Implicit Neural Representations

- Decision: Reject
- Scores: 2, 8, 6, 4

## Abstract
Effective surrogate models are critical for accelerating scientific simulations. Implicit neural representations (INRs) offer a compact and continuous framework for modeling spatially structured data, but they often struggle with complex scientific fields exhibiting localized, high-frequency variations. Recent approaches address this by introducing additional features along rigid geometric structures (e.g., grids), but at the cost of flexibility and increased model size. In this paper, we propose a simple yet effective alternative: Feature-Adaptive INR (FA-INR). FA-INR leverages cross-attention to an augmented memory bank to learn flexible feature representations, enabling adaptive allocation of model capacity based on data characteristics, rather than rigid structural assumptions. To further improve scalability, we introduce a coordinate-guided mixture of experts (MoE) that enhances the specialization and efficiency of feature representations. Experiments on three large-scale ensemble simulation datasets show that FA-INR achieves state-of-the-art fidelity while significantly reducing model size, establishing a new trade-off frontier between accuracy and compactness for INR-based surrogates.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an implicit neural representation (INR)-based approach for fitting computationally simulated datasets. The method incorporates cross-attention-based adaptive feature augmentation and a mixture-of-experts strategy, achieving the highest peak signal-to-noise ratio (PSNR) among five models while using the same (or reduced) number of parameters. However, the title and overall presentation may be misleading, as the method aligns more closely with neural compression techniques than with computational simulation itself.

### Strengths
- The proposed method achieved an improved Pareto frontier between reconstruction quality and memory demand across the selected benchmark problems (though the benchmarks themselves are of limited broader significance; see Weaknesses).
- Section 4.2 provides a clear explanation of why further scaling up failed and justifies the use of a mixture of experts.
- Section 4 is presented with clear and detailed mathematical formulations.
- The inclusion of error bars in Table 2.
- The availability of code supports reproducibility.

### Weaknesses
1. The proposed method fits computationally simulated data by minimizing the empirical risk. However, this approach may not be directly aligned with surrogate modeling for computational simulations. In particular, lines 421--426 suggest that the primary goal of this work is data compression rather than simulation modeling.
2. Structural similarity index measure (SSIM) values are missing in Table 2, and Table 3 lacks error bars, maximum difference, and PSNR (high-frequency) metrics.
3. The heading of Section 4.2 appears somewhat inconsistent with lines 131–132. Additionally, the statement “they are well-suited for capturing fine-scale and high-frequency structures” (l.153) seems to contradict lines 122–128.
4. In Appendix G, varying the spatial grid across parameters is uncommon in ensemble simulations. Fixing “sensor locations” would typically be more representative of such settings.

### Questions
**Questions**

1. Could the authors clarify the meaning of the error bars in Table 2?
1. Could the authors clarify how the percentage improvements were computed (l.405, l.452–453)?
2. How does the attention map’s behavior differ for inputs $x$ in high- versus low-frequency regions?
5. Regarding the statement “This dense storage is often redundant” (l.238), could the authors clarify whether there exist methods to identify regions requiring dense storage *a priori*---that is, before obtaining the data?


**Suggestions for the Presentation**

1. The zoomed-in images in Figure 5a appear quite small and may benefit from enlargement for clearer comparison.
2. The discussion of INR limitations related to spectral bias appears multiple times in the main text; it might be helpful to consolidate these mentions.
3. The key research question (*How can we preserve INRs’ compactness while improving their fidelity in modeling scientific data?*) could be more clearly integrated with the rest of the narrative.
4. Since PSNR is already reported, the inclusion of MSE might be redundant.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors propose Feature-Adaptive INR (FA-INR), a method that uses cross-attention with an augmented memory bank to learn adaptive features for INRs in scientific simulation surrogate tasks, thus drastically reducing the surrogate model’s memory footprint. The authors also propose a mixture-of-experts architecture based on FA-INR that improves the scalability of their proposed method. The authors compare FA-INR against four other baselines across structured and unstructured grid problems.

### Strengths
The problem this paper seeks to solve is well-motivated (even for those not familiar with the state-of-the-art in INRs for accelerating scientific simulations), and the method itself is well-written and clear. I appreciate the depth and breadth of the experiments, which are performed across a variety of grid types and scientific domains.

### Weaknesses
Apart from the improvements in memory, one of the major benefits of using INRs is the fact that they directly model the output field and can thus be queried anywhere. In many scientific applications, it is possible that training data may arrive on different grids or sensor locations; INRs can deal with this directly. It is also possible that we may want to query the output of the INR at different locations than those seen during training.

I see in the appendix that the authors performed spatial domain generalization. How does FA-INR compare to the baselines at zero-shot super-resolution or zero-shot “grid transfer” (train on one grid and evaluate zero-shot on another) for the other datasets explored? How does FA-INR compare to the baselines at different sparsities of training spatial coordinates? I would recommend the authors explore a bit more fully these spatial domain generalization questions.

**Minor notes:**
I appreciate the idea of Figure 1, but it is a bit difficult to understand from first glance which method performs best without a corresponding ground truth figure.

### Questions
1. If $K$ is already a learnable set of keys, what is the purpose of $W_k$?
2. The authors mentioned that many key-value pairs are not heavily utilized. Do you also notice similar behavior with experts? Are only a few experts “activated” during inference, or is the coordinate-conditioning effective in activating them?
3. In this paper, I assume that the simulation parameters $p$ are finite-dimensional. Is it possible to also apply this method (with minor modifications) in the operator learning case when $p$ is an infinite-dimensional function?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Feature-Adaptive Implicit Neural Representation (FA-INR), a framework for building compact, high-fidelity surrogate models for scientific simulations. The INR uses explicit geometric structures increase memory cost and reduce adaptability. FA-INR replaces fixed grid embeddings with a learnable cross-attention–based memory bank for adaptive feature allocation, integrates a Mixture-of-Experts (MoE) routing mechanism based on spatial coordinates to improve scalability and specialization. The proposed method achieves up to 17% higher PSNR with 5–10× fewer parameters.

### Strengths
The paper proposes a novel method incorporating learnable attention memories that generalizes previous grid-based augmentations and introduces MoE specialization tailored to scientific fields with localized complexity.

The method is well-motivated, the architecture is clearly depicted. Visuals results convincingly illustrate qualitative gains.

### Weaknesses
FA-INR requires longer training due to attention and MoE operations, which may limit adoption in very large simulation campaigns. 

Percentage gains in PSNR (dB) (e.g., “17.49% improvement” line 404) is not meaningful; absolute dB differences or linear-scale are more interpretable.

### Questions
Would you consider adding one operator-learning baseline (FNO with hypernetwork for parameters) to demonstrate FA-INR’s superiority beyond INR family methods.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Feature-Adaptive Implicit Neural Representations (FA-INR), an adaptive surrogate modeling framework for scientific simulations. FA-INR augments standard INRs with a cross-attention-based memory bank and an (Mixture-of-Experts) MOE module to improve fidelity and scalability in modeling complex physical fields. The paper evaluates the method on three benchmark datasets, such as MPAS-Ocean, Nyx, and CloverLeaf3D, showing quantitative gains in accuracy over existing INR-based surrogates.

### Strengths
- The topic of scientific surrogate modeling with neural fields is well aligned with the community’s growing interest in implicit neural representations and scientific machine learning.

- The work provides broad empirical validation across three simulation domains, along with targeted ablation studies on key architectural choices.

- The proposed method achieves better PSNR and SSIM metrics than INR-based baselines while maintaining a smaller model size.

### Weaknesses
- **Incremental technical novelty.** The paper primarily combines existing components, such as cross-attention mechanisms and Mixture-of-Experts (MoE), within a conventional INR framework, resulting in limited methodological innovation. Notably, [1] also employs cross-attention and MoE, where the query points can similarly be interpreted as coordinate inputs to an INR. The authors should clearly articulate how their proposed approach differs from [1] and from other related transformer-based operator-learning methods. Including an experimental comparison with such models would further strengthen the paper’s contribution.

- **Overstated claims of compactness and scalability.** Although FA-INR employs fewer parameters than some baselines, the paper itself acknowledges increased training time due to the use of attention and MoE modules. Therefore, the claimed efficiency advantages appear overstated.

- **Lack of diverse baselines and discussion of related work.** The experimental evaluation is confined to INR-based models, lacking comparisons with broader classes of surrogate modeling approaches. In particular, neural operator methods, commonly used for scientific surrogate modeling and sharing a similar problem formulation where simulation parameters serve as inputs, are neither discussed nor compared. Furthermore, the paper omits discussion of other implicit neural representation methods applied to scientific forecasting [2, 3]. Including these perspectives would provide a more balanced and comprehensive assessment of the proposed method’s performance and relevance.

[1] GNOT: A General Neural Operator Transformer for Operator Learning, ICML, 2023.

[2] Continuous PDE Dynamics Forecasting with Implicit Neural Representations, ICLR, 2023. 

[3] CROM: Continuous Reduced-Order Modeling of PDEs Using Implicit Neural Representations, ICLR, 2023.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
