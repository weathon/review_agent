# ICLR Benchmark Results

Date: 2026-04-02 00:28
Critic/Merger: z-ai/glm-5 (OpenRouter)
Neutral: z-ai/glm-5, Related Work: z-ai/glm-5:online (OpenRouter)

## A1HhtITVEi

- GT: Accept (Poster) (avg 6.0)
- Predicted: Accept (6.5/10)
- Match: Yes

### Final Review

## Summary

CheapNet proposes a hierarchical approach for protein-ligand binding affinity prediction that uses differentiable pooling to cluster atom-level representations and applies cross-attention at the cluster level. The method addresses computational complexity and noise issues in atom-level approaches by learning soft cluster assignments and focusing attention on biologically relevant cluster interactions. Extensive experiments demonstrate state-of-the-art performance across multiple benchmarks with relatively few parameters (~1.33M).

## Strengths

- **Strong empirical performance**: CheapNet achieves SOTA results on PDBbind v2013/v2016 core sets and v2019 holdout set, outperforming baselines including large pre-trained models (BindNet >47M params) while using only ~1.33M parameters (Tables 1-3). The consistent improvements across LBA, LEP, and external validation datasets (CSAR NRC-HiQ, CASF-2016) demonstrate robust generalization.

- **Memory efficiency**: Figure 3 clearly demonstrates significantly lower memory footprint compared to atom-level attention methods (GAABind, DEAttentionDTA), which run out-of-memory on larger complexes. This is a practical contribution for handling large protein-ligand complexes.

- **Modular encoder-agnostic design**: Table 4 shows that the cluster-attention mechanism consistently improves performance across different base encoders (GCN: +1.9% RMSE improvement, EGNN: +11.8%, GIGN: +8.6%), demonstrating architectural flexibility.

- **Comprehensive ablation studies**: Tables 4-5 and Tables A6-A11 systematically evaluate the contribution of hierarchical representations, cross-attention vs. self-attention, cluster numbers, auxiliary losses, and pooling methods. The comparison with TopKPooling, ASAPooling, and SAGPooling (Table A10) justifies the differentiable pooling choice.

- **Interpretability through attention visualization**: Figure 4 and Appendix A.17 demonstrate that cross-attention maps highlight biologically meaningful interaction regions between protein and ligand, offering insights into binding mechanisms.

## Weaknesses

- **Fixed cluster numbers lack adaptivity**: The number of clusters (cl, cp) are hyperparameters set based on median values from training data (Appendix A.10), not learned adaptively. This design choice may not be optimal for proteins/ligands with varying sizes—a single fixed cluster count cannot capture the structural diversity across real applications. While Table A8 shows Q2 (median) performs well, no analysis demonstrates whether this scales appropriately.

- **Missing wall-clock time analysis**: The paper claims "computational efficiency" but only reports memory footprint (Figure 3). Training and inference time comparisons against baselines are essential to validate efficiency claims comprehensively. This gap is notable given that computational efficiency is a key motivation.

- **Limited interpretability validation**: Figure 4 shows one example (PDB ID: 4kz6), but no systematic analysis verifies that attention consistently highlights known binding residues across the test set. Without correlating cluster assignments with biochemical groups or validating attention overlap with ground-truth binding sites, the claim of "biologically meaningful clusters" remains unsubstantiated beyond the single visualization.

- **Incomplete baseline comparisons**: Tables 1-2 show missing results (marked "-") for several attention-based baselines on v2019 holdout and "OOM" for cluster-level models (GemNet, Equiformer) on diverse protein evaluation. While understandable for OOM cases, this limits comparison comprehensiveness. Additionally, some baseline comparisons (CAPLA, GAABind, DEAttentionDTA) use different training data splits (PDBbind v2020 vs. v2016), which the paper acknowledges but limits fair comparison.

## Nice-to-Haves

- Adaptive cluster number selection mechanism that dynamically determines optimal clusters per complex based on structural characteristics, rather than fixed hyperparameters.
- Systematic failure case analysis examining when cluster-level representations lose critical fine-grained atomic interaction information.

## Novel Insights

The cross-attention mechanism between protein and ligand clusters represents a meaningful architectural innovation. Unlike prior cluster-level approaches (GemNet, LEFTNet, GET) that rely on predefined clusters or geometric constraints, CheapNet learns soft cluster assignments end-to-end and applies cross-attention to filter noise while focusing on biologically relevant interactions. The bidirectional attention (L2P and P2L) captures mutual influence between protein pockets and ligand moieties. The permutation invariance proof (Appendix A.3) correctly distinguishes graph-level cluster ordering invariance from geometric SE(3) symmetries, which depend on the encoder—a nuanced clarification that prevents confusion about what symmetries the method actually enforces.

## Potentially Missed Related Work

- None identified through systematic search.

## Suggestions

1. **Add wall-clock time comparisons**: Report training time (epochs, total time) and inference time against key baselines (GIGN, GAABind, DEAttentionDTA) to complement memory analysis and substantiate efficiency claims.

2. **Systematic interpretability validation**: Quantify attention overlap with known binding residues across multiple test complexes, not just the single example shown. Visualize learned cluster assignments for representative complexes to verify whether clusters correspond to meaningful molecular substructures (functional groups, binding motifs) versus arbitrary groupings.

3. **Address cluster number adaptivity**: Either add experiments showing sensitivity across protein/ligand size distributions, or discuss alternative approaches (learned cluster numbers, size-adaptive clustering) in the limitations section as future work.

---

## rPup1cWk4d

- GT: Reject (avg 3.0)
- Predicted: Reject (4.0/10)
- Match: Yes

### Final Review

## Summary
The paper proposes a novel data augmentation method based on energy-based modeling and information geometry. The key innovation is the "backward projection" algorithm that reverses dimension reduction on a statistical manifold constructed via the log-linear model on posets. Instead of using black-box neural networks, the method embeds data as probability distributions, projects onto flat sub-manifolds for dimension reduction, and generates new samples by backward projecting from sampled latent representations. Experiments on MNIST and UCI datasets demonstrate competitive performance against autoencoders while offering interpretability.

## Strengths
- **Theoretically grounded framework**: The method builds on solid foundations from information geometry (dually-flat manifolds, Bregman divergences) and the log-linear model on posets, with theoretical guarantees including unique projections via m-projection and energy-minimizing properties.

- **Novel backward projection algorithm**: Algorithm 4.1 provides a principled, geometrically intuitive approach to reversing dimension reduction by constructing local data sub-manifolds from k-nearest neighbors in the latent space.

- **Interpretable latent space construction**: The many-body approximation (Section 4.4) provides principled control over the dimensionality-quality trade-off, allowing practitioners to understand what information is preserved at each level (ℓ-body interactions).

- **Competitive empirical results on MNIST**: Table 1 shows the proposed method (75.37%) significantly outperforms the autoencoder baseline (68.12%) when training only on augmented data, and achieves comparable performance (83.40% vs 82.72%) when combining original and augmented data.

- **Comprehensive ablation studies**: The paper includes sensitivity analysis for bandwidth (Figure 8), number of neighbors k (Figure 9), different base sub-manifold constructions (Figures 6-7, 12-17), and different poset structures, demonstrating robustness and providing insight into the method's behavior.

## Weaknesses
- **Limited baseline comparisons**: The paper compares only against a simple 2+2 layer autoencoder. No comparison with standard data augmentation techniques (geometric transforms, mixup, CutMix), VAEs, GANs, or diffusion models. This substantially limits claims of "competitive performance with black-box generative models."

- **Narrow experimental scope**: MNIST with 1000 training samples and small UCI datasets are insufficient to substantiate practical utility. More challenging datasets (CIFAR-10, Fashion-MNIST) would better demonstrate scalability and real-world applicability.

- **Catastrophic failure on Musk dataset**: Table 3 shows the method drops from 66.30% (Original) to 21.80% (Ours) on the Musk dataset—a severe degradation that warrants investigation. The paper reports this but does not analyze when or why the method fails.

- **Restricted to positive data**: The method naturally handles positive tensors (Example 4.2). For general real-valued data, preprocessing (normalization) is required, which may lose information. This limitation is acknowledged but its practical implications are not analyzed.

- **Computational complexity not discussed**: The paper lacks analysis of computational costs for projection operations, nearest neighbor searches, and scaling with data size and dimension. This is essential for understanding the interpretability-efficiency trade-off.

- **No analysis of class mixing in nearest neighbors**: The backward projection algorithm (Algorithm 4.1) constructs local sub-manifolds from k-nearest neighbors, but does not address what happens when neighbors belong to different classes, potentially producing incoherent interpolations.

## Nice-to-Haves
- Include timing and memory comparisons between the proposed method and baseline autoencoder to complete the trade-off analysis between interpretability and computational cost.

- Demonstrate a concrete use case where interpretability provides practical value—e.g., showing how manipulating specific parameters in the base sub-manifold controls generation in understandable ways.

## Novel Insights
The paper offers an interesting conceptual reframing: instead of treating the decoder as a learned neural network, it treats "decoding" as a geometric projection problem. The insight that nearest neighbors in the latent space can define a local sub-manifold for backward projection is elegant—the method essentially performs a local geometric interpolation rather than learning a global decoding function. The many-body approximation provides a principled hierarchy for latent space dimensionality, where each level ℓ captures mode interactions up to order ℓ, offering an interpretable alternative to arbitrary latent dimension selection in autoencoders.

## Potentially Missed Related Work
No specific missed related work was identified in the review process. However, the following areas would strengthen the paper's positioning:
- Comparison with classical data augmentation methods (random crop/flip, mixup, CutMix)
- Connection to manifold learning methods with out-of-sample extensions (e.g., kernel methods)

## Suggestions
- Add comparisons with at least one standard data augmentation method (e.g., mixup) and one stronger generative baseline (e.g., VAE) to properly contextualize performance claims.

- Evaluate on at least one more challenging dataset (e.g., Fashion-MNIST or CIFAR-10) to demonstrate broader applicability beyond simple MNIST.

- Analyze and explain the failure case on the Musk dataset—understanding when the method degrades performance is crucial for practitioners.

- Include computational complexity analysis (time and memory) comparing the proposed method with baseline approaches.

---

## YkMg8sB8AH

- GT: Reject (avg 4.2)
- Predicted: Reject (5.0/10)
- Match: Yes

### Final Review

## Summary
This paper introduces EquiGX, a method for explaining predictions from equivariant graph neural networks operating on 3D geometric graphs. The authors extend the Deep Taylor decomposition framework with layer-wise relevance propagation rules specifically derived for spherical equivariant GNNs, handling tensor product operations, linear layers, and norm-based non-linearities to attribute importance to nodes and geometric features (distances and directions) in the input space.

## Strengths
- **Clear problem identification and motivation**: The paper correctly identifies a significant gap—existing XAI methods focus on 2D graphs and struggle with the tensor product operations and geometric features central to 3D equivariant GNNs. This is important for scientific applications where interpretability is crucial.
- **Principled technical derivation**: The relevance propagation rules for tensor products (Equations 3-7) correctly exploit the bilinearity of TP operations. The trilinear decomposition that separately attributes relevance to hidden features, distance embeddings, and directional features (via spherical harmonics) is a natural extension for geometric graphs.
- **Comprehensive experimental evaluation**: The paper evaluates on synthetic datasets (Shapes, Spiral Noise) with controlled ground truth and real-world scientific datasets (SCOP, BioLiP, ActsTrack), using qualitative visualizations and quantitative metrics (AUROC, AP, Fidelity, Sparsity).
- **Consistent empirical improvements**: Table 1 shows improvements across all four datasets. For example, on Shapes, AUROC improves from 82.83 (best baseline PG-Explainer) to 84.31; on SCOP, from 77.26 to 81.51. The visualizations (Figures 1-4) confirm EquiGX better identifies geometric motifs.

## Weaknesses
- **Limited architectural evaluation**: The method is evaluated only on Tensor Field Networks (TFN), but the title claims applicability to "spherical equivariant GNNs" broadly. Modern architectures like Equiformer, MACE, and NequIP have additional components (attention mechanisms, higher-order interactions) that may require modified propagation rules. This limits confidence in generalizability.
- **Key design choice lacks ablation**: Equation 7 assigns equal relevance to the three TP components (hidden features, distance, direction) following prior work on attention mechanisms. This is a consequential design choice that warrants empirical validation—alternative attribution schemes may be more appropriate for tensor products specifically.
- **Equivariance of explanations not addressed**: For a method designed for equivariant GNNs, the paper does not discuss whether the generated explanations should themselves be equivariant. If the input graph is rotated, do the relevance scores transform appropriately? This is fundamental to the problem setting and warrants both theoretical discussion and empirical verification.
- **High variance in some results**: On Spiral Noise (AUROC 83.57 ± 10.07) and Shapes (84.31 ± 8.89), the standard deviations are substantial. Statistical significance testing would strengthen claims of improvement over baselines.
- **Conservation property not empirically verified**: The paper claims relevance conservation holds, but provides no numerical verification of whether this property is maintained through TP layers.

## Nice-to-Haves
- Visualization of edge-level importance scores (distance and direction relevance R(d_ij), R(r_ij)) which are explicitly computed but never shown; these could provide more granular geometric insights than node-only visualizations.
- Layer-wise relevance visualizations showing how geometric information flows through the network.

## Novel Insights
The key technical insight is exploiting the bilinearity of tensor products to derive tractable relevance propagation rules. The trilinear decomposition—separately attributing relevance to hidden features, distance embeddings (via RBF), and directional features (via spherical harmonics)—is novel for geometric XAI. This separation enables fine-grained analysis of when geometry vs. features drive predictions, potentially offering scientific insights beyond what node-level attribution can provide.

## Potentially Missed Related Work
None identified.

## Suggestions
1. **Expand architectural evaluation**: Test on at least one modern spherical equivariant GNN (e.g., MACE or Equiformer) to validate broader applicability.
2. **Add ablation on relevance attribution**: Compare the equal 1/3 split against alternative schemes to justify the design choice in Equation 7.
3. **Empirically verify explanation equivariance**: Rotate/translate input graphs and verify that explanations transform correspondingly—or discuss limitations if they do not.
4. **Add statistical significance tests**: Report p-values or confidence intervals to support claims of improvement over baselines.

---

## LbgIZpSUCe

- GT: Accept (Spotlight) (avg 7.3)
- Predicted: Accept (6.5/10)
- Match: Yes

### Final Review

## Summary
The paper proposes MRDS-IR (Multi-Region Dynamical Systems with Impulse Response communication channels), a probabilistic generative model for analyzing neural population dynamics across multiple brain regions. The method combines region-specific nonlinear dynamics (parameterized by neural networks) with linear communication channels characterized by their impulse responses, enabling both expressive local dynamics and interpretable frequency-domain analysis of inter-areal communication. The authors develop a variational inference algorithm using state-noise inversion free filtering to handle the hybrid stochastic/deterministic transitions, and validate the approach on synthetic data, RNN-based task simulations, and real V1/V2 neural recordings.

## Strengths
- **Novel methodological framework**: The impulse response parameterization of communication channels is a creative contribution that bridges linear systems theory with nonlinear dynamical systems, enabling both time-domain (impulse response) and frequency-domain (transfer function) interpretation of inter-areal communication.
- **Strong theoretical grounding**: The connection between impulse responses, transfer functions, and finite-dimensional state-space realizations (Section 2.1) is well-motivated and provides principled interpretability that black-box approaches lack.
- **Comprehensive experimental validation across multiple domains**: The paper includes four experiments—synthetic ground-truth recovery, reverse-engineering RNN computations (integration task), rhythmic timing tasks, and real V1/V2 neural recordings—demonstrating method versatility.
- **Competitive empirical performance**: On held-out neuron prediction for V1/V2 data, MRDS-IR achieves lower MSE than DLAG, MRM-GP, LN, and NL baselines across all eight stimulus conditions (Figure 4E), and shows better long-horizon prediction than CURBD on RNN-based tasks (Figure 2E).
- **Meaningful neuroscientific interpretability**: The method recovers expected patterns in V1/V2 data—feedforward signals peaking early after stimulus onset, feedback signals ramping up later—which aligns with known cortical processing and was not clearly recovered by DLAG (Figure S1).
- **Technical contribution in inference algorithm**: The state-noise inversion free variational filtering algorithm (Appendix A) handles hybrid stochastic/deterministic transitions elegantly without requiring matrix inversions for degenerate noise covariances.

## Weaknesses
- **No systematic ablation studies**: The paper uses different filter orders (M=1 for integration task, M=2 for rhythmic timing and V1/V2) and latent dimensionalities (L₁=3, L₂=2 for V1/V2) without systematic analysis of how these hyperparameters affect performance or interpretability. The contribution of nonlinear local dynamics versus impulse response channels is not isolated.
- **No statistical significance testing or uncertainty quantification**: Figures 4E and 4F report MSE and R² values without error bars, confidence intervals, or significance tests across random seeds or cross-validation folds. Claims about "outperforming" baselines are not statistically supported.
- **Computational complexity and scalability not discussed**: The extended state includes channel states, so dimensionality scales with K²×M×L (where K is regions, M is filter order, L is latent dimension). Training time, memory requirements, and scalability to many regions or larger neural populations are never discussed—essential for practical adoption.
- **MR-SDS not empirically compared**: The paper explicitly identifies MR-SDS (Karniol-Tambour et al., 2022) as "the most complex multi-area model" in Table S1 but provides no empirical comparison, leaving unclear how MRDS-IR compares to the most relevant nonlinear alternative.
- **Identifiability analysis missing**: The paper notes learned dynamics have "expected model invariances (axis rotation, and re-scaling)" but does not analyze whether impulse responses are uniquely identifiable, how sensitive channel estimates are to initialization, or what symmetries exist in the full model.
- **Limited biological validation scope**: Only one V1/V2 session (106r001p26) is analyzed for MSE comparison, and another (107l003p143) for R² comparison. Broader validation across additional sessions, brain regions, or behavioral tasks would strengthen claims about real-world applicability.

## Nice-to-Haves
- Analysis of failure modes: Under what conditions does MRDS-IR fail to recover correct dynamics or channel structure (e.g., insufficient data, high noise, nonlinear inter-areal coupling)?

## Novel Insights
The paper makes a genuinely useful observation that communication between brain regions can be parameterized through impulse responses, providing a principled middle ground between fully interpretable linear models and black-box approaches. The finding that estimated feedforward (V1→V2) communication peaks early while feedback (V2→V1) ramps up over time (Figure 4D) aligns with known cortical processing hierarchies, demonstrating that the method extracts biologically meaningful structure. The state-noise inversion free filtering algorithm provides an elegant solution for handling deterministic channel dynamics within a variational framework, avoiding numerical issues with degenerate noise covariances.

## Potentially Missed Related Work
- None identified (related work search was not performed).

## Suggestions
1. Add systematic ablation experiments varying filter order M and latent dimensionality L to quantify their impact on model performance and interpretability.
2. Include error bars across multiple random seeds and statistical significance tests (e.g., paired t-tests or bootstrap confidence intervals) for all quantitative comparisons against baselines.
3. Add a computational complexity analysis discussing training time and memory scaling with number of regions K, latent dimensionality L, and filter order M.
4. Compare empirically against MR-SDS or other nonlinear multi-region models to better position the contribution relative to the most comparable existing methods.
5. Validate on additional neural recording sessions from V1/V2 or other brain regions to demonstrate robustness across datasets.

---

## c61unr33XA

- GT: Accept (Poster) (avg 7.0)
- Predicted: Accept (6.5/10)
- Match: Yes

### Final Review

## Summary
This paper proposes MKDT (Matching Knowledge Distillation Trajectories), the first effective method for dataset distillation in self-supervised learning. The authors identify that naive trajectory matching fails for SSL due to high gradient variance, and address this by using knowledge distillation to create lower-variance student trajectories from a teacher encoder trained with SSL. Experiments on CIFAR-10/100 and TinyImageNet show up to 13% improvement over baselines in downstream linear probe accuracy.

## Strengths
- **Novel problem formulation**: Addresses a genuinely open problem—dataset distillation for SSL pre-training—with a principled solution that leverages knowledge distillation to reduce trajectory variance.
- **Theoretical grounding**: Theorem 4.1 provides formal analysis showing SSL gradients have higher variance than SL gradients in a simplified linear setting, giving mechanistic justification for why MTT fails for SSL.
- **Comprehensive empirical validation**: Evaluates across CIFAR-10, CIFAR-100, and TinyImageNet with multiple downstream tasks (Aircraft, CUB2011, Dogs, Flowers), showing consistent 5-13% improvements over baselines.
- **Evidence for core claims**: Figure 1 directly demonstrates that SSL trajectories have higher variance than SL trajectories, distillation loss decreases slowly for SSL, and synthetic images struggle to move from initialization.
- **Method generalizability**: Demonstrated with both BarlowTwins and SimCLR, and transfer from ConvNet distillation to ResNet-10/18 architectures.

## Weaknesses
- **Theoretical gap from practice**: Theorem 4.1 assumes a linear model with spectral contrastive loss, while experiments use BarlowTwins/SimCLR with ResNet-18. The variance analysis doesn't directly apply to the empirical setting, weakening the theoretical contribution.
- **No direct verification that KD trajectories have lower variance**: Figure 1a compares SSL vs SL trajectory variance, but never directly measures KD trajectory variance against SSL—the core mechanistic claim remains empirically unverified.
- **Missing key baseline comparisons in main text**: The MTT-SSL and DM-SSL baselines (Tables 11-12) appear only in Appendix F, yet these directly test the paper's premise that naive trajectory matching fails for SSL.
- **Limited experimental scale**: Experiments are limited to CIFAR-scale and TinyImageNet. SSL is most impactful for large-scale pre-training; validation on ImageNet-scale datasets is essential to assess practical relevance.
- **No computational cost analysis**: The method requires training a ResNet-18 teacher on full data, then K=100 student trajectories—substantial overhead that isn't quantified relative to baselines or full SSL training.
- **Missing critical ablations**: No ablation on the number of expert trajectories K=100, which directly affects both performance and computational cost.

## Nice-to-Haves
- Analysis of what semantic content is captured in distilled images (Appendix E shows examples but no quantitative analysis)
- Investigation of teacher quality sensitivity—what happens if the teacher is under-trained?
- Experiments on larger distillation budgets (beyond 5%) to understand scaling behavior

## Novel Insights
The key insight—that SSL gradient variance fundamentally prevents effective trajectory matching, and that knowledge distillation converts SSL into a supervised-like objective with lower variance—is creative and empirically grounded. The paper correctly identifies that SSL losses depend on batch interactions (creating high variance) while KD's MSE loss is instance-wise (creating lower variance). This insight enables the first successful dataset distillation method for SSL, opening a new research direction.

## Potentially Missed Related Work
None identified (search was skipped for this paper).

## Suggestions
- Directly measure and report trajectory variance for KD vs SSL to verify the core mechanism
- Move the MTT-SSL and DM-SSL baseline comparisons (Tables 11-12) from Appendix F to the main experimental section
- Include ImageNet-scale experiments (even a subset) to demonstrate scalability
- Add computational cost comparison (FLOPs or wall-clock time) between MKDT, baselines, and full SSL training
- Ablate the number of expert trajectories K (e.g., K ∈ {25, 50, 100}) to assess sensitivity and computational trade-offs

---

## UUwrBhhsxT

- GT: Reject (avg 5.2)
- Predicted: Reject (4.0/10)
- Match: Yes

### Final Review

## Summary
This paper proposes a Transformer-based scenario-oriented testing framework for fault detection in Unmanned aircraft Traffic Management (UTM) systems. The framework combines a Policy Model trained via offline reinforcement learning to generate targeted fault injection scenarios, with a rule-based Action Sampler that enforces safety constraints and incorporates human preference. The authors evaluate their approach on an industry-level simulator managing 400+ drones across 700+ hours of testing.

## Strengths
- **Substantial empirical evaluation scale**: The paper demonstrates commitment to rigorous validation through 700+ hours of simulation across 30+ environments with 400+ drones, using an industry-level UTM simulator (Section 6, Table 8).
- **Novel problem formulation**: Treating fault detection as a sequential decision problem over long-tail scenario distributions is well-motivated, and the offline RL formulation addresses the sample inefficiency of traditional random testing (Section 4).
- **Clear performance improvements**: The PM-2B model achieves 50.5 SPM (high-risk scenarios per million flights) and 7.6 FPM (faults per million flights), identifying vulnerabilities that traditional methods missed entirely (Table 2).
- **Scalability analysis**: The paper includes a systematic comparison across model scales (10M to 2B parameters), showing clear performance improvements with model size in both offline and online evaluation (Figure 5, Table 2).
- **Practical architecture design**: The two-component framework (Policy Model + Action Sampler) appropriately separates learned scenario generation from rule-based safety enforcement, making the system interpretable for safety-critical contexts (Figure 2, Section 5).

## Weaknesses
- **Potentially unfair baseline comparison**: The paper states that baseline FPM values are low (<1.0) because "the two baseline tests have already been thoroughly used to identify existing bugs and improve UTM in advance, while our method is focused on discovering new bugs" (Table 2 footnote). This creates an asymmetric comparison where baselines have already exhausted easy-to-find bugs. A fairer comparison would evaluate all methods on the same fresh system.
- **No statistical significance reporting**: Despite 700+ hours of testing and ~100M records per region, the paper reports only point estimates for SPM, FPM, and other metrics without confidence intervals or standard deviations, making it difficult to assess reliability of the claimed improvements.
- **Missing offline RL baseline comparisons**: The paper compares against expert-guided random-walk and smoke tests but not against other offline RL methods (e.g., Decision Transformer, Q-Transformer) or recent learning-based testing approaches like D2RL (Feng et al. 2023, cited in the paper). This makes it difficult to isolate the contribution of the Transformer architecture versus alternative offline RL approaches.
- **Incomplete methodological specification**: Key details are underspecified—the return-to-go target during inference is not clearly explained (how is R̂ conditioned during deployment?), the reward function weights α_i are not defined, and the context set C's size and selection criteria are not specified (Section 4.2).
- **No ablation studies**: The paper lacks ablations on key design choices such as context window size, number of in-context trajectories, and the contribution of the Action Sampler component relative to the Policy Model alone.

## Nice-to-Haves
- **Concrete fault case studies**: The paper would benefit from specific examples of fault scenarios discovered by PM that baselines missed, including the sequence of events that triggered them—this would help practitioners assess real-world applicability.
- **Inference latency analysis**: Given that this is a testing framework requiring reasonable timeframes, reporting inference efficiency would strengthen the practical deployment discussion.

## Novel Insights
The observation that offline RL can outperform expert demonstrations by acting as an "implicit filter of low-quality actions" (Section 6.2) is interesting but under-analyzed. The hazard action ratio per observation being similar across methods (~2%) while constant-pressure action ratio differs substantially suggests that PM's advantage lies not in finding more hazardous actions, but in sustaining pressure on the system over time—a finding that could inform future testing methodologies for safety-critical systems.

## Potentially Missed Related Work
- None identified in the provided inputs.

## Suggestions
1. **Re-run baseline comparisons on the same system state** to enable fair comparison, or prominently acknowledge the asymmetric testing conditions as a limitation.
2. **Add confidence intervals or standard deviations** for all reported metrics across the 700+ hours of testing.
3. **Include at least one offline RL baseline** (e.g., Decision Transformer or Q-Transformer) to demonstrate the specific value of the proposed architectural modifications.
4. **Provide ablation studies** isolating the contributions of: (a) the context-aware mechanism, (b) the Action Sampler, and (c) different model sizes.
5. **Clarify the return-to-go specification during inference**—whether it is set to a fixed high value, searched over, or learned.

---

## 1aF2D2CPHi

- GT: Accept (Oral) (avg 8.0)
- Predicted: Reject (5.0/10)
- Match: No

### Final Review

## Summary
This paper addresses data-free knowledge distillation (DFKD) from CLIP for open-vocabulary customization. The authors identify that existing DFKD methods fail on CLIP because BatchNorm statistics encode facial features from web-scale pre-training rather than target class semantics, and propose an alternative approach using image-text matching with style dictionary diversification, class consistency maintaining, and meta knowledge distillation.

## Strengths
- **Clear empirical problem identification**: The paper convincingly demonstrates why existing DFKD methods fail on CLIP (Table 8 shows 7.13%-7.63% accuracies) by analyzing how BN statistics from web-crawled data encode facial features (Figures 2-3). This is a valuable insight that explains a real failure mode.
- **Comprehensive experimental evaluation**: Experiments across 12 customized tasks (Caltech-101, 10 ImageNet splits) with consistent improvements. The method achieves 64.81% average accuracy compared to near-failure baselines, with clear ablation studies showing component contributions (Table 1).
- **Practical significance for deployment**: The approach produces lightweight student models (11.68M params vs 151.28M for CLIP) suitable for edge devices (Table 3), addressing real privacy and copyright concerns in model customization.
- **Supports multiple customization modes**: Both text-based and image-based customization are supported, with image-based customization addressing the text ambiguity limitation (Table 6 shows 58.89% improvement with image prompts on Flower-102).
- **Theoretical grounding**: The paper provides generalization bounds relating δ-diversity to generalization error (Theorem 4.1) and gradient alignment analysis for meta-learning (Theorem 4.2), giving principled motivation for the design choices.

## Weaknesses
- **VQGAN dependency undermines "data-free" claim**: The method relies on VQGAN pre-trained on external datasets. Table 12 shows performance drops from 62.46% to 34.24% without VQGAN, indicating the external prior is essential. This should be more prominently discussed as a limitation of the "data-free" framing.
- **No statistical significance testing**: All results report single-run numbers without error bars or standard deviations. Given the optimization-based image synthesis with inherent randomness, this omission makes it difficult to assess result reliability.
- **Hand-crafted style dictionary lacks systematic justification**: The 16-style dictionary (Table 13) is manually designed with limited guidance on selection criteria for new domains. Table 14 shows some sensitivity analysis but doesn't explain why these specific styles work or how to adapt them.
- **Ambiguity handling remains a significant limitation**: The Flower-102 experiment (Table 6) reveals only 15.83% accuracy with text prompts, showing the method struggles fundamentally with ambiguous class names—a limitation acknowledged but not fully addressed.
- **Computational cost not comprehensively analyzed**: While 57 seconds is noted for SDD training, the full pipeline cost (400 iterations per image × 64 images × number of classes) is not quantified, making practical deployment assessment difficult.

## Nice-to-Haves
- Report FID or CLIP scores for synthetic images to quantitatively assess diversity and quality claims beyond visualizations in Figures 6-7.
- Include a real-data upper bound for text-based customization to contextualize the performance gap between data-free and data-aware distillation.

## Novel Insights
The key insight that CLIP's BatchNorm statistics encode facial features from web-scale pre-training data—making them unusable for traditional DFKD—represents a meaningful empirical finding. The observation that VLMs trained on internet data absorb biases (e.g., human faces appearing in images regardless of text descriptions) has implications beyond DFKD for understanding what foundation models store in their normalization layers. The meta-learning approach to handle style-based distribution shifts provides a principled way to address the covariate shift inherent in synthetic data training.

## Potentially Missed Related Work
- None identified by the related work search agent.

## Suggestions
- Rename the method framing to "original-data-free" or explicitly discuss how VQGAN priors relate to the data-free definition in the literature, since relying on pre-trained generators partially undermines the data-free claim.
- Add error bars from multiple runs (at least 3 seeds) to establish statistical reliability of reported results.
- Provide systematic guidance on style dictionary selection, either through automated selection or clear criteria for adapting to new domains.

---

## EqCbc4wrzy

- GT: Reject (avg 2.5)
- Predicted: Reject (3.0/10)
- Match: Yes

### Final Review

## Summary
MDPE introduces a multimodal deception dataset comprising 104+ hours of video from 193 subjects, uniquely annotated with Big Five personality traits and emotional expression characteristics. The dataset addresses a documented gap in deception detection research—no prior dataset combines multimodal deception data with individual difference information—enabling new research on how personality and emotion affect deception behavior.

## Strengths
- **Substantial scale and novelty**: MDPE is the largest publicly available multimodal deception dataset (193 subjects, 6209 minutes), substantially exceeding prior datasets like DDPM (70 subjects) and Bag-of-Lies (35 subjects), with unique personality and emotional characteristic annotations not available elsewhere.
- **Well-designed collection protocol**: The three-phase design (personality questionnaire → emotion induction → deception interview) follows established methodology, includes monetary incentives tied to deception success to motivate genuine effort, and was IRB-approved with informed consent for scientific publication.
- **Comprehensive benchmarking**: The paper evaluates multiple feature extractors per modality (ViT, CLIP for visual; eGeMAPS, Wav2vec, HuBERT, WavLM for acoustic; BERT, ChatGLM, Baichuan for text) and systematically studies personality/emotion feature contributions, providing useful guidance for future work.
- **Clear empirical findings**: The experiments demonstrate that personality features consistently improve deception detection (a novel finding), while emotional features show inconsistent benefits—both results that warrant further investigation.

## Weaknesses
- **Answer-level split creates potential subject-level leakage**: The paper splits at the answer level (5 of 24 answers per subject held out for validation), not the subject level. This means the same subject's data appears in both training and validation, risking that models learn subject-specific behavioral patterns rather than generalizable deception cues. Subject-independent evaluation (leave-one-subject-out) is essential for deception datasets.
- **No held-out test set and no statistical significance testing**: Only training and validation sets are described, with no held-out test set. Tables report point estimates without standard deviations or confidence intervals, making it impossible to assess whether differences between methods (often 1-3%) are meaningful or noise.
- **Ground truth deception validity is unverified**: The paper acknowledges "we do not know whether the subjects have actually deceived on the deception questions." Without independent verification, some "deceptive" responses may actually be truthful, introducing label noise that undermines model evaluation reliability.
- **No cross-dataset evaluation or human baseline**: The paper provides no experiments showing how models trained on MDPE transfer to existing datasets, nor does it report human deception detection accuracy on MDPE, leaving dataset utility and progress claims unsubstantiated.

## Nice-to-Haves
- **Analysis of which personality dimensions matter**: The paper concatenates all Big Five scores but doesn't analyze which specific traits correlate with deception success or detection difficulty—such analysis would strengthen the scientific value claim.
- **Deception quality analysis**: Interviewer judgment scores and subject confidence ratings were collected but not analyzed; understanding characteristics of "successful" vs. "failed" deceptions would enhance the dataset's utility.

## Novel Insights
The finding that personality features consistently improve deception detection while emotional features show unstable effects is intriguing and underexplored. The paper speculates that personality traits are directly usable (self-reported scores) while emotional features depend on emotion recognition model quality, but this hypothesis isn't tested. The superior performance of textual features (~62% accuracy) over visual and acoustic modalities aligns with human deception detection intuition—we judge truthfulness primarily from what people say rather than how they look or sound. The dataset enables investigation of whether certain personality types produce more detectable lies or whether emotional states modulate deception cues.

## Potentially Missed Related Work
None identified.

## Suggestions
- **Re-evaluate using leave-one-subject-out cross-validation**: This is the standard protocol for datasets involving individual subjects and is essential to claim the model learns generalizable deception cues rather than subject idiosyncrasies.
- **Add statistical significance testing**: Report standard deviations across the 5 runs mentioned, and use appropriate significance tests when comparing conditions.
- **Report human baseline accuracy**: Have independent annotators judge deception from videos to contextualize model performance (~64% accuracy).

---

## VpWki1v2P8

- GT: Accept (Oral) (avg 8.7)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
The paper introduces LoRA-RITE, a novel optimizer for LoRA fine-tuning that achieves transformation invariance—a property lacking in standard optimizers like Adam. The key insight is that since LoRA's update Z = AB^T admits infinitely many equivalent factorizations, an optimizer should produce identical updates to Z regardless of parameterization. The authors prove that diagonal preconditioners cannot achieve transformation invariance and propose a matrix preconditioning approach on the low-rank side that maintains computational efficiency comparable to Adam while achieving consistent empirical improvements across multiple LLM benchmarks.

## Strengths
- **Principled problem formulation**: The paper identifies a fundamental theoretical issue—that equivalent LoRA parameterizations lead to different gradient updates under standard optimizers—and formalizes transformation invariance (Definition 1) and scalar scale invariance (Definition 2) mathematically.
- **Solid theoretical foundation**: The paper provides rigorous proofs that diagonal preconditioners cannot achieve transformation invariance (Section 3.1), convergence analysis (Theorems 3 and 4), and shows that transformation-invariant optimizers guarantee efficient feature learning (Theorem 1).
- **Strong empirical results**: LoRA-RITE consistently outperforms baselines across diverse benchmarks. On Gemma-2B, it achieves 4.6% improvement on Super-Natural Instructions and 3.5% average improvement across four LLM benchmarks compared to Adam (Tables 1, 2).
- **Computational efficiency**: Despite using matrix preconditioning, the method maintains O(mr² + nr² + r³) time complexity with O(mr + nr) memory—only 8% slower than Adam on Gemma-2B and 5% on Gemma-7B (Table 4).
- **Comprehensive evaluation scope**: Experiments cover multiple model sizes (2B, 7B), architectures (decoder-only Gemma, encoder-decoder mT5-XXL), and LoRA ranks (4, 16, 64), demonstrating generalizability across settings.

## Weaknesses
- **No statistical significance reporting**: All tables report single numbers without standard deviations, confidence intervals, or results across multiple random seeds. For improvements like 55.50% vs. 48.37% on GSM8K, readers cannot assess whether these differences are statistically meaningful or within noise margins. This is essential for rigorous empirical claims.
- **Missing ablation of the escaped mass mechanism (ρ)**: The paper introduces a novel "escaped mass" correction mechanism to handle basis changes during training (Algorithm 1, lines 7 and 17), but provides no empirical validation that this component is necessary or contributes to performance improvements.
- **Core property not empirically demonstrated**: The paper claims LoRA-RITE is transformation-invariant but never validates this empirically. A simple experiment—training with different initial LoRA factor scalings (s×A, B/s for various s)—would demonstrate whether the method achieves its stated goal while baselines vary.
- **Assumption 1 lacks empirical verification**: The stronger convergence guarantee (Theorem 4) requires Assumption 1, which constrains how the polar decomposition factors change between steps. The paper provides no evidence this assumption holds during actual training, limiting the practical relevance of the stronger theoretical bound.

## Nice-to-Haves
- **Analysis of hyperparameter sensitivity**: The paper uses standard Adam β₁=0.9, β₂=0.999 values but does not analyze whether LoRA-RITE is sensitive to these choices or how the escaped mass ρ accumulates across different settings.
- **Direct comparison with Riemannian gradient descent**: The paper identifies Riemannian GD as the only other transformation-invariant method but only compares against ScaledAdam (which loses invariance when combined with Adam). A comparison with pure Riemannian GD would isolate the benefit of invariance from the benefit of adaptivity.

## Novel Insights
The paper's key conceptual contribution is recognizing that the non-uniqueness of LoRA factorization creates an optimization inconsistency that standard adaptive methods cannot address. The proof that diagonal preconditioners are fundamentally insufficient for transformation invariance (because R^{-T}X_2^T X_2 R^T can be non-diagonal for arbitrary rotation R) is elegant and motivates the matrix preconditioning approach. The "unmagnified gradient" concept—separating A into basis U_A and magnitude R_A via polar decomposition—provides a clean mechanism for achieving invariance while preserving adaptive preconditioning benefits. The regret bound improvement from O(G(D_A² + D_B²)T^{-1/2}) to O(GD_A D_B T^{-1/2}) is mathematically meaningful precisely when LoRA factors have imbalanced norms, which Figure 1 shows occurs in practice.

## Potentially Missed Related Work
- None identified (related work search was not performed).

## Suggestions
- **Report results across multiple random seeds**: Run all experiments with at least 3-5 seeds and report mean ± standard deviation. This is standard practice and essential for trusting claimed improvements.
- **Add an ablation study of the escaped mass correction**: Include a variant of LoRA-RITE without the ρ accumulation (setting ρ=0) to quantify whether this mechanism contributes meaningfully to performance.
- **Empirically demonstrate transformation invariance**: Train models with different initial scalings of LoRA factors (e.g., s ∈ {0.1, 1, 10, 100}) and show that LoRA-RITE achieves consistent performance while baseline methods vary significantly. This would directly validate the core motivation.

---

## LJULZNlW5d

- GT: Reject (avg 3.0)
- Predicted: Reject (3.0/10)
- Match: Yes

### Final Review

## Summary
This paper proposes Fast Gradient Leakage (FGL), a gradient inversion attack method for federated learning that leverages pretrained StyleGAN as an image prior and introduces a joint gradient matching loss combining L₂, cosine, and L₁ distances. The method achieves reconstruction of high-resolution face images (CelebA, 224×224, 1000 classes) with batch sizes up to 60 on CNN architectures, demonstrating improvements in both reconstruction quality and computational efficiency compared to prior gradient inversion attacks.

## Strengths
- **Addresses a genuinely challenging gap in prior work**: The paper convincingly demonstrates that prior GIAs (DLG, GI, Fishing, GIAS) fail on high-resolution face datasets—Table 2 shows all baselines achieving 0.0 Top-1 accuracy—while FGL achieves meaningful reconstruction.
- **Clear methodological contributions with strong ablations**: Table 1 provides a systematic ablation showing how each component (selection strategy, transformations, joint loss, gradient normalization, multi-seed optimization) contributes to performance improvement, with Top-1 increasing from 0.0 to 0.88.
- **Significant efficiency gains**: Figure 7 shows FGL completing batch-1 attacks in 2.58 minutes versus 23.99 minutes for GI and 23.17 for Fishing, with batch-60 attacks completing in 13.99 minutes—substantially faster than prior methods.
- **Novel joint gradient matching loss formulation**: The combination of L₂ + cosine + L₁ losses with learned weights is a reasonable approach to avoiding local optima in the optimization landscape, motivated by the different geometric properties of each distance metric.

## Weaknesses
- **No evaluation against defense mechanisms**: The paper explicitly claims FGL "can serve as a valuable tool to advance privacy defense techniques," yet provides zero experiments against any defense (gradient perturbation, differential privacy, compression, Secure Aggregation). This substantially limits the assessment of real-world privacy risk.

- **Insufficient isolation of methodological contributions**: The ablation study adds components cumulatively but never isolates the contribution of the StyleGAN prior from the proposed joint loss. The improvement may derive primarily from the GAN prior borrowed from model inversion attack literature rather than the novel loss formulation—this distinction matters for understanding what advances the field.

- **Limited dataset and architecture evaluation**: Despite claiming the method works "across various network models on complex datasets," all primary experiments use CelebA faces with ResNet-18. The paper does not demonstrate generalization to other domains (ImageNet objects, medical imaging) or systematically evaluate modern architectures like Vision Transformers.

- **Hyperparameter details missing from main text**: The joint loss M_grad = α₁L₂ + α₂Cosine + α₃L₁ introduces three hyperparameters whose values are not specified in the main paper, and no sensitivity analysis is provided. This impairs reproducibility and makes it difficult to assess how carefully tuned the method is.

- **Baseline comparison limited to batch size 1**: While FGL is evaluated at batch sizes up to 60, baseline methods are only run at batch size 1. The claim that prior methods cannot handle larger batch sizes needs empirical verification—showing baselines failing at batch size 5, 10, 20 would strengthen the comparison.

## Nice-to-Haves
- **Failure case analysis**: At batch size 60, Top-1 drops to 0.483. Understanding which images fail to reconstruct and why would provide valuable insight into the method's practical limitations and threat boundaries.
- **Distribution prior mismatch analysis**: The StyleGAN is pretrained on FFHQ while attacks target CelebA. The paper acknowledges this mismatch but provides no analysis of how prior quality or domain alignment affects attack success.

## Novel Insights
The key insight of this paper—transferring GAN-based prior knowledge from model inversion attacks to gradient inversion attacks—is genuinely valuable. The joint loss formulation combining multiple gradient matching objectives addresses a real optimization challenge: gradient inversion landscapes are highly non-convex with many local minima, and different loss functions guide optimization along different trajectories. The observation that combining L₂ (absolute magnitude), cosine (directional alignment), and L₁ (robust element-wise matching) helps escape local optima is a solid contribution, though the theoretical justification remains limited. The empirical demonstration that high-resolution face images can be recovered from gradients at meaningful batch sizes advances our understanding of privacy risks in federated learning.

## Potentially Missed Related Work
- None identified (related work search was skipped). However, given the rapid pace of research in gradient inversion attacks, the authors should ensure comparison with the most recent GIA methods from 2023-2024, particularly any that incorporate generative priors.

## Suggestions
- **Add defense experiments**: Evaluate FGL against at least one gradient defense (e.g., gradient perturbation with varying noise levels, gradient compression, or differential privacy) to substantiate claims about advancing privacy defense research.
- **Provide complete hyperparameter specifications**: Include all α values, learning rates, number of seeds, and optimization settings in the main text with sensitivity analysis.
- **Run baselines at matched batch sizes**: Include baseline results at batch sizes ≥1 to demonstrate whether the performance gap persists as batch size increases.

---

