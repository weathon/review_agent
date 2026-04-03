# ICLR Benchmark Results

Date: 2026-04-02 00:51
Critic/Merger: z-ai/glm-5 (OpenRouter)
Neutral: z-ai/glm-5, Related Work: z-ai/glm-5:online (OpenRouter)

## c61unr33XA

- GT: Accept (Poster) (avg 7.0)
- Predicted: Reject (5.0/10)
- Match: No

### Final Review

## Summary
This paper proposes MKDT (Matching Knowledge Distillation Trajectories), the first effective dataset distillation method for self-supervised learning pre-training. The authors identify that naïve trajectory matching fails for SSL due to high variance in SSL gradients, and address this by training student models via knowledge distillation from a larger SSL-trained teacher, then matching these lower-variance KD trajectories to generate synthetic datasets. Experiments on CIFAR-10/100 and TinyImageNet demonstrate up to 13% improvement over baselines.

## Strengths
- **Novel problem formulation**: This is genuinely the first effective dataset distillation method specifically designed for SSL pre-training, addressing a clear gap since prior DD methods require labels or fail to produce useful representations for downstream tasks.
- **Strong theoretical motivation**: Theorem 4.1 provides formal analysis showing SSL gradients have higher variance than SL gradients under a simplified linear model with spectral contrastive loss, which directly motivates the KD-based approach.
- **Clear empirical diagnosis**: Figure 1 effectively demonstrates that high variance in SSL trajectories prevents meaningful synthetic image updates, with appropriate controls (larger batch size experiments showing this insufficiently addresses the problem).
- **Comprehensive experimental evaluation**: The paper covers multiple datasets (CIFAR-10, CIFAR-100, TinyImageNet), downstream tasks (7 transfer datasets), label fractions (1%, 5%, 10%, 50%), distilled set sizes (2%, 5%), SSL algorithms (BarlowTwins, SimCLR), and architectures (ConvNets, ResNet-10/18).
- **Cross-architecture transfer**: Table 5 shows that distilled datasets from small ConvNets transfer to larger ResNet architectures, demonstrating practical utility beyond the distillation architecture.
- **Reproducibility**: Code is provided, Algorithm 1 clearly describes the method, and hyperparameters are detailed in Appendix A.

## Weaknesses
- **Missing critical baseline**: The paper compares MKDT to random subsets trained with SSL, but does not include a baseline of training on random subsets using the same KD objective from the same teacher. Without this control, it is unclear whether the benefits come from distillation itself or simply from the KD training signal. This experiment is essential to validate the core claim.
- **Computational cost not analyzed**: The method requires training K=100 student trajectories (each for 20 epochs) plus a teacher model, but no analysis of distillation time or FLOPs is provided. Claims of "efficiency" are difficult to evaluate without this information.
- **Limited experimental scale**: Experiments are limited to CIFAR and TinyImageNet (at most 200K images at 64×64 resolution). SSL is most impactful at large scale, and demonstrating effectiveness on ImageNet or similar benchmarks would significantly strengthen practical relevance.
- **Theory-practice gap**: Theorem 4.1 analyzes variance in a simplified linear setting, but experiments use BarlowTwins on deep ConvNets. No analysis connects why the KD trajectory variance reduction specifically enables trajectory matching in practice beyond the empirical observation.
- **Gap to full data performance**: On CIFAR-10, MKDT achieves ~44% accuracy versus 58% for full data with 2% distilled data. The 14% absolute gap, while improved over baselines, raises questions about what semantic information is lost in distillation.

## Nice-to-Haves
- Ablation study on the number of expert trajectories K (is 100 necessary?) to assess efficiency-accuracy trade-offs.
- Analysis of what semantic properties the distilled images encode—visualizations suggest noise-like patterns, but feature-level analysis (e.g., probing for specific attributes) would clarify what is preserved.
- Discussion of why ResNet-18 slightly underperforms ResNet-10 in Table 5 and whether this indicates limitations in scaling distilled data to larger models.

## Novel Insights
The key insight—that SSL gradients have fundamentally higher variance than SL gradients due to batch-level interactions, and that this variance can be reduced by reformulating SSL training as a supervised KD objective—is both creative and technically sound. The empirical demonstration that even 4× larger batch sizes cannot sufficiently reduce SSL trajectory variance (Figure 1) is a compelling negative result that justifies the KD-based approach. The finding that KRR-ST consistently underperforms random subsets for SSL pre-training is an interesting negative result about prior work that could be explored further.

## Potentially Missed Related Work
None identified in the review materials provided.

## Suggestions
- Run a KD-on-random-subsets experiment (same teacher, same KD objective, but train on random subset instead of distilled data) to isolate whether benefits come from distillation or KD alone.
- Report wall-clock time or FLOPs for distillation vs. training on full data to substantiate efficiency claims.
- Add experiments on at least one larger-scale dataset (e.g., ImageNet-100 or a meaningful subset) to demonstrate scalability.
- Move the MTT-SSL ablation from Appendix F to the main text or provide clearer cross-references, as this directly motivates the method.

---

## rPup1cWk4d

- GT: Reject (avg 3.0)
- Predicted: Reject (3.0/10)
- Match: Yes

### Final Review

## Summary

This paper proposes an interpretable data augmentation method based on information geometry and energy-based modeling. The key contributions are: (1) a framework embedding structured data (tensors) as probability distributions on a statistical manifold using the log-linear model on posets, and (2) a "backward projection" algorithm that reverses dimension reduction by projecting new latent points onto locally constructed sub-manifolds derived from nearest neighbors. The method achieves competitive downstream classification performance compared to a simple autoencoder baseline on MNIST and small UCI datasets.

## Strengths

- **Novel theoretical framework**: The paper introduces a genuinely original approach to data augmentation by combining dually-flat information geometry, log-linear models on posets, and energy-based modeling. The mathematical foundation is sound and the connection between projections and divergence minimization is well-established.

- **Transparent algorithmic contributions**: Algorithms 4.1 and 4.2 are clearly specified, making the method reproducible. The backward projection algorithm provides a geometrically intuitive approach to the inverse dimension reduction problem.

- **Interpretability through explicit structure**: Unlike black-box generative models, the method uses explicit projections onto interpretable sub-manifolds (many-body approximations), with the dimension of latent spaces having clear geometric meaning via the ℓ-body parameter hierarchy.

- **Comprehensive ablation studies**: The appendix includes thoughtful analyses on bandwidth sensitivity (Figure 8), nearest-neighbor parameter k (Figure 9), tensor structure choices (Appendix A.4), and necessity of dimension reduction (Appendix A.3).

- **Cross-modality testing**: The method is evaluated on both image (MNIST) and tabular UCI datasets, demonstrating applicability beyond a single modality.

## Weaknesses

- **Limited experimental scale**: All experiments use very small datasets (1000 MNIST samples, UCI datasets with hundreds to low thousands of samples). No evidence is provided that the method scales to modern dataset sizes or higher-dimensional images like CIFAR-10. The computational complexity is never discussed.

- **Weak and incomplete baseline comparisons**: The only generative baseline is a minimal 2+2 layer autoencoder. Missing comparisons to standard augmentation methods (random flips, Mixup, CutMix), classical PCA-based augmentation, and modern generative models (properly tuned VAEs, GANs, diffusion models) significantly undermine claims of "competitive performance."

- **Inverse embedding operation insufficiently specified**: The natural embedding φ (normalization) is clearly defined, but its inverse φ^{-1} is deferred to a footnote in the appendix as "reversing the average of original scaling among nearest neighbors" without rigorous formulation. This operation is central to decoding and different choices could substantially affect results.

- **Interpretability claimed but not demonstrated**: The central thesis is interpretability, yet no experiments demonstrate what practitioners can actually interpret. The paper does not show how varying specific coordinates produces predictable changes in generated samples, nor how users can leverage the poset structure for controlled generation.

- **Arbitrary design choices without guidance**: The tensor reshaping from 28×28 to 7×2×2×7×2×2 is a key design choice with no principled justification. Appendix A.4 shows different structures yield different results, but no guidance exists for selecting appropriate poset structures for new data types.

- **Marginal and inconsistent improvements**: On MNIST, Original+Ours (83.40±3.22%) barely outperforms Original+AE (82.72±3.50%) with overlapping confidence intervals. On UCI datasets, AE sometimes substantially outperforms Ours (e.g., dataset (c): 21.80±11.95% vs Original+Ours at 20.80±9.79%). These inconsistencies weaken performance claims.

- **No sample quality metrics**: Only downstream classification accuracy is reported. Standard generative model metrics (FID, Inception Score, diversity measures) are absent, making it impossible to assess whether generated samples capture true data diversity.

## Nice-to-Haves

- **Latent space visualization**: Showing interpolations in the latent space or visualizing what geometric structure the θ/η coordinates capture would strengthen the interpretability claims and help readers understand what information is preserved.

- **Computational analysis**: Reporting wall-clock time and memory usage compared to neural generative approaches would help practitioners assess practical tradeoffs.

## Novel Insights

The paper offers an interesting meta-perspective on information geometry: rather than learning a single distribution, it treats multiple data points as distributions within a statistical manifold, enabling geometric operations on the "distribution of distributions." The backward projection algorithm exploits the interplay between linearity (projections onto flat sub-manifolds minimize divergence) and non-linearity (projections on curved statistical manifolds), providing a principled alternative to neural-network-based decoding that maintains explicit geometric structure.

## Potentially Missed Related Work

None identified—the related work section adequately covers information geometry, dimension reduction, and data augmentation literature.

## Suggestions

1. **Demonstrate interpretability concretely**: Include experiments showing that varying specific ℓ-body parameters produces predictable changes in generated samples—this is essential to validate the paper's main contribution claim.

2. **Add standard augmentation baselines**: Compare against at least one classical method (e.g., SMOTE for tabular data, Mixup for images) and one modern generative baseline (properly tuned VAE) to establish competitive performance.

3. **Provide poset design guidance**: Either theoretical principles or empirical analysis showing how to select appropriate tensor structures for different data types would significantly improve practical applicability.

4. **Rigorously define the inverse embedding**: Move φ^{-1} definition to the main text with formal specification, and analyze sensitivity to different inverse embedding choices.

---

## A1HhtITVEi

- GT: Accept (Poster) (avg 6.0)
- Predicted: Accept (5.5/10)
- Match: Yes

### Final Review

## Summary

CheapNet proposes a hierarchical model for protein-ligand binding affinity prediction that combines atom-level graph encoding with cluster-level representations via differentiable pooling, followed by cross-attention between protein and ligand clusters. The approach aims to reduce computational complexity and noise compared to atom-level methods while capturing higher-order molecular interactions. Experiments demonstrate state-of-the-art performance across multiple benchmarks with significant memory efficiency gains.

## Strengths

- **Strong empirical performance across multiple benchmarks**: CheapNet achieves state-of-the-art results on PDBbind v2013 core set (RMSE 1.262, Pearson 0.857), v2016 core set (RMSE 1.107, Pearson 0.870), v2019 holdout set (RMSE 1.343, Pearson 0.665), LEP task (AUROC 0.935, AUPRC 0.924), and competitive performance on diverse protein evaluation (LBA 30%)—all without pre-training, outperforming models like BindNet that use orders of magnitude more parameters.

- **Significant memory efficiency demonstrated**: Figure 3 and Appendix A.16 show that CheapNet maintains consistently low memory usage across varying batch and complex sizes, avoiding OOM issues that affect attention-based baselines GAABind and DEAttentionDTA. This is practically important for scaling to larger complexes.

- **Modular architecture with encoder flexibility**: Table 4 demonstrates that the cluster-attention mechanism consistently improves performance across different base encoders (GCN: +1.9% RMSE reduction, EGNN: +11.8%, GIGN: +8.6%), showing the approach is not encoder-specific.

- **Comprehensive experimental evaluation**: The paper includes experiments across six different benchmarks plus protein-protein affinity prediction (Appendix A.20), demonstrating generalization beyond the primary task.

- **Ablation studies systematically evaluate components**: Table 5 separates contributions of hierarchical representations and cross-attention, showing both components contribute meaningfully to performance improvements.

## Weaknesses

- **Number of clusters requires manual tuning without principled guidance**: The cluster counts for protein and ligand are hyperparameters set via quantiles of training data (Appendix A.4), and Table A8 shows sensitivity to this choice. While Q2 (median) works best, the paper provides no theoretical justification or adaptive mechanism for determining optimal cluster counts, reducing generality across diverse complex sizes.

- **Efficiency claims lack training and inference time analysis**: The paper claims "computational efficiency" but only reports memory footprint (Figure 3). Without wall-clock training time, inference latency, or FLOPs comparisons against atom-level baselines, the efficiency argument is incomplete. Memory efficiency alone does not establish whether the method is actually faster.

- **Limited interpretability validation**: Figure 4 provides one visualization case (PDB ID: 4kz6) showing attention correspondence with binding regions. However, the paper lacks systematic quantitative validation correlating learned attention weights with experimentally validated binding sites across multiple complexes, making the interpretability claim difficult to assess.

- **No analysis of what learned clusters represent**: The differentiable pooling produces soft cluster assignments, but the paper provides no analysis of whether these clusters correspond to biologically meaningful groups (functional groups, binding motifs, secondary structures). Without this analysis, the claim that clusters capture "higher-order interactions" remains unsubstantiated.

- **No failure case analysis**: The paper reports strong aggregate performance but does not discuss cases where CheapNet underperforms relative to baselines. Understanding which complex types (e.g., flexible proteins, unusual binding modes) the method struggles with would clarify limitations.

## Nice-to-Haves

- **Theoretical complexity analysis**: Adding O(n²) vs O(c_l × c_p) complexity comparison between atom-level and cluster-level attention would strengthen the efficiency motivation.

- **Adaptive cluster selection mechanism**: A learned or adaptive method for determining cluster counts based on input complexity would improve the model's generality and reduce hyperparameter tuning burden.

## Novel Insights

The key insight of CheapNet is that combining differentiable pooling with cross-attention enables learning task-relevant clusters dynamically rather than relying on predefined groupings. The bidirectional cross-attention (L2P and P2L) between independently clustered protein and ligand representations provides a mechanism for filtering irrelevant clusters while focusing on interaction-relevant groups. The ablation in Table 5 shows cross-attention provides larger gains than hierarchical pooling alone, suggesting the attention mechanism's filtering role is more important than the clustering itself. The modular design—where cluster-attention can improve any base encoder—suggests this hierarchical approach captures complementary signal to atom-level representations.

## Potentially Missed Related Work

- None identified as critical gaps. The paper provides reasonably comprehensive coverage of interaction-free, interaction-based, and cluster-level methods for protein-ligand binding affinity prediction.

## Suggestions

- Add systematic correlation analysis between learned attention weights and experimentally validated binding pockets/residues across all test complexes to substantiate interpretability claims.
- Include training time, inference time, and FLOPs comparisons against baselines to complete the efficiency evaluation.
- Analyze cluster composition (e.g., enrichment of aromatic groups, hydrogen bond donors/acceptors) to show whether biologically meaningful groupings emerge from training.

---

## LbgIZpSUCe

- GT: Accept (Spotlight) (avg 7.3)
- Predicted: Reject (5.0/10)
- Match: No

### Final Review

## Summary
The paper proposes MRDS-IR (Multi-Region Dynamical Systems with Impulse Response communication channels), a probabilistic generative model for multi-region neural population dynamics that combines nonlinear within-region dynamics (parameterized by neural networks) with linear inter-region communication channels parameterized through their impulse response. The authors develop a state-noise inversion-free variational filtering algorithm for inference and demonstrate the approach on synthetic data, RNN-simulated distributed computation tasks, and real V1/V2 neural recordings.

## Strengths
- **Principled modeling framework**: The impulse response parameterization draws from established linear systems theory, enabling interpretable frequency-domain analysis of communication channels (pole-zero plots, transfer functions) while maintaining expressive nonlinear local dynamics—a meaningful middle ground between fully linear models and black-box neural network approaches.
- **Novel inference algorithm**: The state-noise inversion-free filtering algorithm (Appendix A) provides a clean variational formulation for hybrid stochastic/deterministic transitions, avoiding the need to invert degenerate state-noise covariance matrices for deterministic channel dynamics.
- **Validated across multiple domains**: The method is demonstrated on ground-truth recovery (Figure 1), reverse engineering RNN computations in integration and rhythmic timing tasks (Figures 2-3), and real V1/V2 recordings (Figure 4), showing versatility.
- **Extracts interpretable structure**: The ability to analyze communication channels through impulse responses and frequency responses (Figures 1D, 3G, 4D) provides neuroscientists with meaningful characterizations of inter-area information flow, including feedforward/feedback timing differences consistent with visual cortical physiology.

## Weaknesses
- **Missing baseline comparison**: MR-SDS (Karniol-Tambour et al., 2022) is cited as "the most complex multi-area model" and the most directly comparable method (nonlinear local dynamics with inter-region communication), yet it is absent from all experiments. Without this comparison, claims about MRDS-IR's advantages over existing methods are incomplete.
- **No statistical rigor in quantitative comparisons**: Figures 2D, 2E, 3F, and 4E-F report point estimates without error bars, confidence intervals, or significance tests. The text mentions "10 random partitions" for V1/V2 experiments, but variance across partitions is not reported. This undermines claims of method superiority.
- **Hyperparameter sensitivity and model selection unaddressed**: Key choices—latent dimensionalities L_k, filter order M, neural network architecture for local dynamics—are specified without justification. No analysis shows how results depend on these choices, and no guidance is provided for selecting them on new datasets.
- **Scalability not discussed**: Computational complexity, training times, and memory requirements as regions, neurons, or time series length scale are not analyzed—important for practical adoption.

## Nice-to-Haves
- **Code availability**: No reference to a code repository is provided; releasing implementation would improve reproducibility and community adoption.
- **Ablation on channel complexity**: A comparison to simpler communication models (instantaneous linear, fixed-delay autoregressive) would clarify whether the impulse response parameterization meaningfully improves results over simpler alternatives.

## Novel Insights
The key insight is that inter-region communication can be parameterized through its impulse response while local dynamics remain nonlinear, enabling practitioners to analyze communication through the mature toolkit of linear systems theory (transfer functions, frequency responses, pole-zero analysis) without sacrificing modeling flexibility for within-region computations. The demonstration that feedforward and feedback channels exhibit distinct temporal structure in V1/V2 (early transient FF vs. ramping FB) validates this interpretability claim on real neural data.

## Potentially Missed Related Work
- None identified in the review process.

## Suggestions
- Add MR-SDS as a baseline in at least one experiment to enable direct comparison with the most sophisticated existing multi-region model.
- Report error bars across random seeds/partitions and conduct statistical significance tests for all quantitative comparisons.
- Include experiments analyzing sensitivity to latent dimensionality, filter order M, and neural network architecture, plus guidance on model selection.
- Discuss computational complexity and provide training time comparisons across methods.

---

## YkMg8sB8AH

- GT: Reject (avg 4.2)
- Predicted: Accept (5.5/10)
- Match: No

### Final Review

## Summary
The paper introduces EquiGX, an explainability method for spherical equivariant graph neural networks operating on 3D geometric graphs. The authors adapt Deep Taylor decomposition to derive layer-wise relevance propagation rules specifically for tensor product operations, linear operations, and norm-based non-linearities—key components of architectures like Tensor Field Networks. The method decomposes relevance scores across hidden features, edge distances, and edge directions, enabling attribution back to input elements.

## Strengths
- **Addresses an important gap in XAI literature**: Existing explainability methods for GNNs primarily target 2D graphs and struggle to incorporate positional information and evaluate geometric features. The paper correctly identifies this gap and provides a principled solution for 3D equivariant architectures (Section 1-2.2).
- **Novel technical derivation**: The derivation of relevance propagation rules for tensor product operations exploits their bilinearity, which makes the Taylor expansion exact without higher-order terms. This is a non-trivial adaptation of LRP to equivariant message passing (Section 3.2, Equations 5-7).
- **Strong empirical results**: EquiGX achieves the highest AUROC and AP scores across all four datasets (Table 1), with consistent improvements over six baselines including gradient-based, perturbation-based, and surrogate methods.
- **Comprehensive evaluation framework**: Experiments span synthetic datasets (Shapes, Spiral Noise) with clear ground truth and real-world scientific datasets (SCOP protein fold classification, BioLiP ligand binding, ActsTrack particle tracking), demonstrating practical applicability in scientific domains.

## Weaknesses
- **Limited architectural coverage**: The method is evaluated exclusively on Tensor Field Networks. While the derived rules apply to any spherical equivariant GNN using tensor products, empirical validation on at least one other architecture (e.g., Equiformer, MACE, SE(3)-Transformer) would substantially strengthen the claimed generality.
- **Unjustified equal relevance split**: Equation 7 divides relevance equally among hidden features, distance, and direction. The citation to Achtibat et al. (2024) addresses attention mechanisms, not tensor products, and the paper provides no theoretical or empirical justification for this design choice specific to TP operations.
- **No ablation study**: The key contribution is the specialized TP decomposition rules, but there is no comparison against simpler alternatives (e.g., standard LRP applied naively, gradient-based attribution for TP operations). This makes it difficult to assess whether the proposed rules actually improve explanation quality.
- **Geometric attributions not quantitatively evaluated**: The paper computes relevance scores for edge distances R(d_ij) and directions R(r_ij)—a stated key contribution—but only visualizes node-level attributions. The geometric attributions are never evaluated quantitatively, leaving this claimed capability unsubstantiated.

## Nice-to-Haves
- **Empirical verification of conservation property**: The paper claims relevance conservation holds across layers but provides no empirical verification. Reporting whether ΣR(H) = ΣR(H') holds in practice would strengthen confidence in the decomposition.
- **Sensitivity analysis**: XAI methods should produce stable explanations under small input perturbations. Testing explanation stability under coordinate perturbations would be valuable for equivariant models where input rotations should produce corresponding explanation rotations.

## Novel Insights
The paper makes a useful observation that the bilinearity of tensor products simplifies Taylor decomposition—since TP is linear with respect to each input when others are held constant, higher-order terms vanish when choosing appropriate root points. This enables clean relevance propagation through equivariant message passing layers. The separate attribution to hidden features, distances, and directions is a conceptually clean decomposition of the geometric information flow in spherical equivariant GNNs, even if the equal weighting heuristic requires further justification. The finding that the BioLiP model "does not use binding site information for predictions" is scientifically intriguing and could merit deeper investigation.

## Potentially Missed Related Work
None identified. The related work section adequately covers XAI methods for GNNs and the background on equivariant networks.

## Suggestions
1. **Test on additional architectures**: Evaluate EquiGX on at least one other spherical equivariant GNN (e.g., Equiformer or MACE) to demonstrate generalizability beyond TFN.
2. **Add ablation experiments**: Compare the proposed TP-specific decomposition rules against simpler baselines (gradient-based attribution, standard LRP) to isolate the contribution of the specialized rules.
3. **Quantitatively evaluate geometric attributions**: Use the synthetic datasets where ground-truth important edges/distances are known to evaluate R(d_ij) and R(r_ij) attribution quality, or visualize edge-level relevance heatmaps.

---

## EqCbc4wrzy

- GT: Reject (avg 2.5)
- Predicted: Reject (4.0/10)
- Match: Yes

### Final Review

## Summary
MDPE introduces a multimodal deception detection dataset comprising 104+ hours of video, audio, and text data from 193 subjects. Its key novelty is the inclusion of personality traits (via Big Five Inventory) and emotional expression characteristics alongside deception labels—making it the first deception dataset to systematically capture individual difference factors that psychological research suggests influence deceptive behavior.

## Strengths
- **Significant scale and unique annotation**: MDPE is the largest publicly available deception dataset (193 subjects, 6209 minutes) and uniquely includes personality traits and emotional expression characteristics. Table 1 clearly shows it exceeds prior datasets like DDPM (70 subjects, 776 minutes).
- **Well-designed experimental protocol**: The deception experiment incorporates monetary incentives (150% or 200% base payment for successful deception) to motivate realistic deceptive behavior, following established methodology from DDPM. The three warm-up truthful questions help establish baseline demeanor.
- **Comprehensive multimodal benchmark**: The authors evaluate multiple state-of-the-art features across visual (ViT, CLIP), acoustic (Wav2vec2, HUBERT, WavLM), and textual (BERT variants, ChatGLM, Baichuan) modalities, demonstrating that personality features consistently improve detection performance across modalities.
- **Enables multiple research directions**: Beyond deception detection, the dataset supports personality recognition, emotion recognition, and cross-task studies (Figure 3), with appendices documenting the Big Five questionnaire and emotion scales.

## Weaknesses
- **No ground truth verification for deception labels**: As acknowledged in Section 5.2, "we do not know whether the subjects have actually deceived on the deception questions." Subjects were instructed to lie on 9 questions, but no independent verification (e.g., physiological signals, post-hoc validation) confirms they actually produced deceptive behavior. This raises validity concerns since subjects instructed to lie may not have succeeded in appearing deceptive.
- **Ambiguous train/validation split methodology**: The paper states it "randomly select[s] 5 answers (3 truths and 2 deceptions) from 24 answers in all samples as the validation set." This wording does not clarify whether splits are subject-independent. If the same subject's data appears in both train and test sets, models may learn subject-specific mannerisms rather than deception cues, inflating performance.
- **Missing statistical significance measures**: Tables 2 and 3 report only mean accuracy across 5 runs without standard deviations or confidence intervals. Given small improvements (often 1-3%) from adding personality/emotion features, significance testing would strengthen claims about their effectiveness.
- **Inconsistent benefits from emotional features**: Adding emotion features sometimes decreases performance (e.g., CLIP-large drops from 57.30% to 56.97%). While the paper acknowledges this instability, it lacks deeper analysis of when/why emotional features help versus hurt, limiting insight into their utility.

## Nice-to-Haves
- **Human baseline comparison**: Prior deception datasets included human performance benchmarks; adding these would provide meaningful context for interpreting model performance (~61-65% accuracy).
- **Cross-dataset evaluation**: Testing MDPE-trained models on other deception datasets (or vice versa) would demonstrate the dataset's utility for generalization.
- **Analysis of specific personality traits**: Using Big Five as a single 60-dimensional vector is reasonable, but analyzing which specific traits correlate with deception detectability would strengthen the paper's contribution to understanding individual differences.

## Novel Insights
The finding that personality features consistently improve deception detection across all modalities (visual, acoustic, textual) provides empirical support for psychological theories that individual differences influence deceptive behavior. The observation that text features outperform visual and acoustic features aligns with prior work but is notable given the Chinese language context—Baichuan-13B achieves the best unimodal results, suggesting large language models capture Chinese linguistic deception cues effectively. The instability of emotional features compared to personality features suggests that emotion recognition model quality matters significantly; personality traits derived from validated questionnaires may be more reliable than learned emotion representations.

## Potentially Missed Related Work
- None identified in this review cycle.

## Suggestions
1. Clarify the split methodology explicitly: confirm whether validation splits are subject-independent, and if not, re-run experiments with leave-one-subject-out or cross-subject splits.
2. Add standard deviations to all results tables to enable readers to assess result reliability.
3. Provide human baseline performance through annotation experiments, enabling readers to contextualize machine performance.
4. Analyze the emotion feature instability: investigate whether specific emotion categories or emotion recognition model architectures correlate with improved deception detection.

---

## 1aF2D2CPHi

- GT: Accept (Oral) (avg 8.0)
- Predicted: Accept (5.5/10)
- Match: Yes

### Final Review

## Summary
This paper addresses data-free knowledge distillation (DFKD) from CLIP for open-vocabulary customization. The key contribution is identifying why existing DFKD methods fail on CLIP—BatchNorm statistics encode facial features from internet-scale pre-training data—and proposing an alternative approach using image-text matching with style dictionary diversification, class consistency maintaining, and meta knowledge distillation. The method supports customization from class texts or few example images without accessing original training data.

## Strengths
- **Novel empirical finding**: The observation that CLIP's BatchNorm layers encode facial features from web-crawled data, causing existing DFKD methods to fail (single-digit accuracy), is a valuable contribution. Figure 3 provides compelling visual evidence that synthesized images from standard DFKD methods contain faces regardless of target class, which explains the performance collapse and has implications beyond this work.

- **Comprehensive empirical evaluation**: The paper evaluates across 12 customized tasks, multiple student/teacher architectures (RN18, ViT-T, RN50, ViT-B), and different VLMs (CLIP, BLIP, EVA). The ablation studies in Tables 1, 4, and 5 systematically demonstrate contributions from each proposed component (style dictionary diversification, class consistency, meta-learning).

- **Consistent improvements over meaningful baseline**: The method achieves 64.81% average accuracy compared to 60.61% for the naive text-inversion baseline—a 4.2% absolute improvement—demonstrating practical value beyond enabling DFKD on CLIP where nothing worked before.

- **Extensibility**: Table 5 demonstrates that synthesized datasets can train various student architectures, and Table 7 shows applicability across VLMs (BLIP, EVA), indicating the approach generalizes beyond specific model choices.

## Weaknesses
- **Missing real-data upper bound**: The paper does not report how much performance is sacrificed by being data-free compared to training on real data. Without this baseline, readers cannot assess the practical cost of the data-free constraint or the method's true utility in real-world scenarios.

- **Indirect comparison to VLM-specific distillation methods**: TinyCLIP and CLIP-KD are mentioned only in Appendix D without quantitative comparison. While these methods require large-scale datasets, providing their performance as reference points would contextualize the trade-off between data-free methods and data-dependent approaches.

- **Lack of statistical significance measures**: Results are reported as single numbers without error bars or significance tests, despite the stochastic nature of image synthesis and model training. This limits confidence in the reported improvements, particularly for smaller margins.

- **VQGAN dependency without thorough domain analysis**: The method relies on a pre-trained VQGAN decoder, and Table 12 shows sensitivity to VQGAN pre-training domain (ImageNet VQGAN outperforms OpenImages VQGAN). However, the implications for generalizing to domains different from VQGAN's pre-training data are not analyzed.

- **Underspecified hyperparameter sensitivity**: Key design choices—64 images per class, style dictionary size of 16, 400 optimization iterations—are stated without systematic justification. The paper lacks analysis of whether fewer images would suffice or how performance scales with these parameters.

## Nice-to-Haves
- Include FID scores or CLIP scores for synthetic images to provide objective quality assessment beyond visual inspection in Figures 6-7.
- Report end-to-end training time and memory requirements to help readers assess practical feasibility.

## Novel Insights
The core insight is that foundation models trained on internet-scale data encode unexpected biases in BatchNorm statistics—in CLIP's case, facial features—which breaks traditional DFKD methods that rely on BN alignment. The proposed alternative using image-text matching with learned style diversity provides a general approach for VLMs regardless of architecture (works for ViT without BN layers). The meta-learning formulation encourages style-invariant representations, addressing the distribution shift between synthetic training data and real test data in a principled way.

## Potentially Missed Related Work
- **TinyCLIP (Wu et al., ICCV 2023)** and **CLIP-KD (Yang et al., CVPR 2024)**: These recent CLIP distillation methods use large-scale datasets but establish important baselines for VLM compression quality. While not directly comparable due to data requirements, including their performance would help readers understand the current landscape.

## Suggestions
- Add error bars from multiple runs (at least 3 seeds) to main results in Tables 1-2 to establish statistical significance of improvements.
- Include experiments varying the number of synthesized images per class (e.g., 16, 32, 64, 128) to establish efficiency-accuracy trade-offs.
- Train student models on real ImageNet/Caltech-101 training data as an upper bound comparison to quantify the performance gap of data-free learning.

---

## UUwrBhhsxT

- GT: Reject (avg 5.2)
- Predicted: Reject (4.0/10)
- Match: Yes

### Final Review

## Summary
This paper proposes a Transformer-based offline reinforcement learning framework for fault detection in Unmanned Aircraft System Traffic Management (UTM) systems. The approach combines a Policy Model (PM) that generates candidate fault injection scenarios with a rule-based Action Sampler (AS) that enforces safety constraints, trained on 17B tokens of offline stress-testing data and evaluated over 700 hours in an industry-level simulator with 400+ UAVs across 30+ environments.

## Strengths
- **Important and practical problem**: Long-tail fault detection in mission-critical systems like UTM is a genuine challenge with real safety implications, and the paper clearly motivates why random testing methods struggle with this problem.
- **Substantial empirical scale**: The evaluation spans 700+ hours of simulation with 400+ UAVs, and the offline training uses ~17B tokens—demonstrating meaningful engineering effort and real-world applicability.
- **Scaling analysis provides useful insights**: Figure 5 shows clear performance improvements from 10M to 2B parameters, with larger models overfitting later and achieving higher peak accuracy, providing empirical evidence for the scalability of transformer-based approaches in this domain.
- **Performance improvements are demonstrated**: Table 2 shows PM-2B achieves SPM (high-risk scenarios per million flights) of 50.5 versus 5.8 for expert-guided testing on TR1—approximately 8.7x improvement, supporting the abstract's "8x" efficiency claim.

## Weaknesses
- **Limited baseline comparisons undermine confidence in claimed improvements**: The paper compares only to "expert-guided random-walk" and "smoke test" baselines. Despite citing D2RL, LEADE, and TD3-based fuzzing methods in Related Work, none of these ML-based testing methods are included as baselines. Without such comparisons, it is unclear whether a simpler offline RL method (e.g., Decision Transformer) or online RL method would achieve similar results.
- **No statistical significance testing or variance reporting**: Table 2 reports single values for all metrics with no error bars, confidence intervals, or standard deviations across multiple runs. Given the stochastic nature of both RL training and testing scenarios, this makes it difficult to assess whether reported differences are statistically meaningful.
- **Key methodological details missing for reproducibility**: The reward function is defined as a weighted sum r(s,a) = Σα_i r_i(s,a), but the specific reward components r_i and weights α_i are never specified. Similarly, the "multi-objective loss function" in Appendix A.7 lacks exact formulation or weight values. The training data collection process (how expert-guided trajectories were generated) is underspecified.
- **No ablation studies to isolate component contributions**: The framework combines Transformer architecture, context-aware augmentation, preference bias, and action sampler—without ablations, it is unclear which components are essential versus unnecessary complexity.
- **No qualitative analysis of discovered faults**: The paper reports FPM (faults per million flights) values but provides no case studies characterizing what types of faults were discovered, their severity, or why existing methods missed them. This limits understanding of what the approach actually contributes to UTM safety.

## Nice-to-Haves
- **Dataset bias analysis**: The offline dataset comes from "traditional stress testing," but no analysis of its distributional properties or fault coverage is provided—understanding training data limitations would strengthen interpretation.
- **Threshold justification**: The 0.4 threshold used in HAR and CAR metrics is arbitrary without sensitivity analysis.
- **Broader action space**: The action space (Table 7) includes only Wind, Obstacle, and Network Jitter—expanding to UAV subsystem failures or communication attacks could improve coverage.
- **Addressing overfitting**: Figure 5 shows clear overfitting; explicit discussion of checkpoint selection strategy would improve reproducibility.

## Novel Insights
The paper provides an interesting observation about why offline RL can outperform human experts: offline RL acts as an "implicit filter of low-quality actions" and is "less susceptible to distraction during the search for long-tail scenarios." The analysis showing that PM models and human experts have similar hazard action ratios per observation (~2%) but PM models achieve significantly higher constant-pressure action ratios suggests the learned policy maintains sustained exploration pressure over time rather than achieving fundamentally different action quality. This insight about sustained pressure versus one-off hazard identification is a potentially transferable understanding for testing mission-critical systems.

## Potentially Missed Related Work
- None identified by the search process.

## Suggestions
1. **Add ML-based baselines**: Include at least one offline RL baseline (Decision Transformer or CQL) and one online RL method (PPO or SAC) to properly contextualize performance improvements.
2. **Run and report ablations**: Isolate contributions of (a) Transformer backbone versus simpler architectures, (b) context-aware augmentation, (c) preference bias, and (d) action sampler filtering.
3. **Provide statistical uncertainty**: Run experiments across multiple seeds and report means with standard deviations or confidence intervals.
4. **Include fault case studies**: Add 2-3 concrete examples of faults discovered by PM-2B that baselines missed, with analysis of failure modes and safety implications.
5. **Specify reward function and loss weights**: Either provide exact values in the paper or explain why these details cannot be disclosed.

---

## LJULZNlW5d

- GT: Reject (avg 3.0)
- Predicted: Reject (3.0/10)
- Match: Yes

### Final Review

## Summary
This paper proposes Fast Gradient Leakage (FGL), a gradient inversion attack for federated learning that leverages StyleGAN pretrained on FFHQ as image prior knowledge and introduces a joint gradient matching loss function combining L₂, cosine similarity, and L₁ terms. The method demonstrates image reconstruction on the CelebA face dataset (224×224, 1000 classes) with batch sizes up to 60, achieving faster convergence than prior methods like GradInversion and DLG.

## Strengths
- **Strong empirical improvements on challenging data**: FGL achieves Top-1 accuracy of 1.0 on CelebA reconstructions, dramatically outperforming baselines (DLG, GI, Fishing, GIAS) which fail to produce recognizable images on this high-resolution face dataset.
- **Demonstrated scalability to larger batch sizes**: The paper successfully attacks batch sizes up to 60 on 224×224 images—a significant advance over prior GIA methods that struggled with both high-resolution faces and large batches.
- **Comprehensive ablation study**: Table 1 systematically shows the contribution of each component (selection strategy, transformations, joint loss, gradient normalization, multi-seed optimization), with Top-1 accuracy improving progressively from 0.0 to 0.88.
- **Time efficiency**: FGL completes batch-size-1 attacks in ~2.58 minutes compared to ~24 minutes for GI and Fishing, representing meaningful speedup for practical threat assessment.

## Weaknesses
- **Critical reproducibility gap**: The hyperparameters α₁, α₂, α₃ for the joint gradient matching loss are never specified in the paper. Without these values, the method cannot be reproduced. Similarly, the number of seeds used in the multi-seed strategy is unspecified.
- **Unfair time comparison methodology**: The paper compares FGL's time on CelebA against DLG tested on CIFAR-10 (because "DLG struggled to attack CelebA"), which is not a fair comparison. Time efficiency claims should be based on identical datasets.
- **No evaluation against privacy defenses**: Despite framing the contribution as advancing "privacy defense techniques," the paper tests against no defenses (gradient noise, differential privacy, gradient compression). This limits the paper's relevance to real-world federated learning security.
- **Baseline comparisons limited to batch size 1**: While the paper justifies testing baselines at batch size 1 for "optimal performance," this prevents direct comparison at the batch sizes (e.g., 30, 60) where FGL's advantages are claimed. Fair comparison at identical settings is needed.
- **StyleGAN prior dependency unanalyzed**: The method relies on a StyleGAN pretrained on FFHQ to attack CelebA images, but there is no analysis of performance degradation when the prior distribution differs from the target distribution—a realistic attack scenario.

## Nice-to-Haves
- **Failure case analysis**: The paper does not discuss when FGL produces poor reconstructions or what conditions remain challenging (e.g., the performance drop mentioned at batch size 20). Including failure cases would clarify method limitations.
- **Non-face dataset evaluation**: The abstract claims success on "complex datasets" plural, but only CelebA is tested. Evaluation on at least one non-face dataset would strengthen generalization claims.

## Novel Insights
The paper makes a meaningful contribution by successfully adapting model inversion attack techniques (specifically StyleGAN priors and image transformation strategies from PPA) to the gradient inversion attack setting. The joint gradient matching loss—combining L₂ distance, cosine similarity, and L₁ distance—provides multiple optimization perspectives that help avoid local optima, which is a reasonable design for high-dimensional image reconstruction. The selection and multi-seed strategies contribute to robustness. However, the core insight (GAN priors + multi-objective optimization) builds directly on prior work in model inversion attacks, making the contribution incremental rather than groundbreaking.

## Potentially Missed Related Work
- **KEDGI (Knowledge Enhancement for Deep Gradient Inversion)** and other recent 2023-2024 gradient inversion methods should be considered for baseline comparison to demonstrate current state-of-the-art improvement.
- **CAFE (Cross-layer Analysis for Gradient Inversion)** and similar batch-handling approaches may be relevant for comparison on large-batch attacks.

## Suggestions
- Provide explicit hyperparameter values for α₁, α₂, α₃ and document the tuning methodology.
- Conduct time comparisons on identical datasets (all methods on CelebA, even if baselines perform poorly).
- Add at least one experiment testing against gradient perturbation or differential privacy defenses to substantiate the claimed contribution to privacy defense development.
- Include baseline comparisons at batch sizes larger than 1 to enable fair assessment of FGL's batch-size advantages.

---

## VpWki1v2P8

- GT: Accept (Oral) (avg 8.7)
- Predicted: Accept (6.5/10)
- Match: Yes

### Final Review

## Summary

This paper introduces LoRA-RITE, a novel optimizer for Low-Rank Adaptation (LoRA) that achieves transformation invariance—a property ensuring equivalent LoRA parameterizations yield identical weight updates. The key insight is that standard optimizers (Adam, Adagrad, Shampoo) violate this property, leading to unbalanced factor updates. The authors prove diagonal preconditioning cannot achieve transformation invariance, propose an efficient matrix preconditioning scheme using "unmagnified gradients" computed via polar decomposition, and provide convergence analysis. Empirically, LoRA-RITE consistently outperforms baseline optimizers across LLM benchmarks with minimal computational overhead.

## Strengths

- **Principled theoretical contribution**: The paper identifies a fundamental mathematical inconsistency in applying standard optimizers to LoRA—lack of transformation invariance—and proves that diagonal preconditioning cannot resolve this issue. Theorem 1 formally connects transformation invariance to efficient feature learning.

- **Elegant algorithmic solution**: The "unmagnified gradient" construction via polar decomposition, combined with basis projection for moment accumulation and the "escaped mass" correction mechanism, provides a principled way to achieve transformation invariance while incorporating adaptivity.

- **Computational efficiency**: Despite using matrix preconditioning, the method maintains O(mr² + nr² + r³) complexity, demonstrated empirically to be only 5-8% slower than Adam (Table 4), which is negligible compared to backpropagation costs.

- **Strong empirical results**: Consistent improvements across diverse benchmarks—4.6% accuracy gain on Super-Natural Instructions (Gemma-2B), 3.5% average gain on LLM benchmarks, and substantial improvements on GSM8K (55.50% vs 48.37% with Gemma-7B) compared to Adam.

- **Comprehensive baseline comparison**: Compares against five optimizers (Adam, LoRA+, ScaledAdam, Shampoo, Lamb) across multiple model architectures (Gemma-2B, 7B, mT5-XXL) and ranks.

## Weaknesses

- **No statistical significance reporting**: All experimental results are reported as point estimates without standard deviations or confidence intervals. For improvements like OpenBookQA (68.0 vs 68.8), this undermines confidence in whether observed differences are meaningful.

- **Large learning rate discrepancy unexplained**: Table 9 shows LoRA-RITE uses learning rates 20-100× larger than Adam (e.g., 2e-4 vs 1e-5). The paper should explicitly discuss whether this is inherent to the method or a confounding factor—does LoRA-RITE enable stable training at higher learning rates, or is the improvement partly from different hyperparameter regimes?

- **Missing component ablations**: The algorithm combines unmagnified gradients, matrix preconditioning, basis-projected moments, and escaped mass correction (ρ), but no ablation isolates each component's contribution. The escaped mass mechanism in particular is novel but unvalidated.

- **Assumption 1 lacks empirical verification**: The stronger convergence guarantee (Theorem 4) depends on Assumption 1, which bounds how smoothly polar decomposition components change. No empirical evidence is provided that this assumption holds in practice.

- **No comparison with Riemannian gradient descent**: The paper identifies Riemannian GD as "the only method in the literature that satisfies transformation invariance" but excludes it from experiments, missing an opportunity to compare transformation-invariant methods directly.

## Nice-to-Haves

- Empirical validation of transformation invariance: running experiments with different initial scalings of LoRA factors (A initialized small vs. large) to demonstrate that LoRA-RITE produces consistent outcomes while baselines do not.

- Full fine-tuning baseline to quantify the remaining performance gap between LoRA-RITE and full parameter fine-tuning.

## Novel Insights

The key conceptual contribution is recognizing that LoRA's factorization creates inherent optimization ambiguity—equivalent parameterizations (related by any invertible transformation R) should produce identical updates but don't under standard optimizers. The paper shows this leads to one factor being undertrained (Figure 1), and the unmagnified gradient construction elegantly strips away the "magnitude" component that varies across equivalent parameterizations. The "escaped mass" correction mechanism addresses an underappreciated subtlety: accumulated moments from different bases cannot be directly combined, requiring projection and compensation. This insight—that adaptive methods need basis-aware moment accumulation for structured parameterizations—generalizes beyond LoRA to other factorized representations.

## Potentially Missed Related Work

None identified in the search scope.

## Suggestions

1. **Add error bars** across multiple random seeds for all experimental results to establish statistical significance of improvements.

2. **Include ablation experiments** isolating the contributions of (a) the escaped mass mechanism (ρ), (b) the momentum projection scheme, and (c) using unmagnified gradients without full adaptivity.

3. **Empirically validate Assumption 1** by measuring how smoothly U_B and R_B change during training, or provide bounds on assumption violations in practice.

4. **Discuss the learning rate finding** explicitly—whether LoRA-RITE's structure inherently enables stable training at higher learning rates and what this implies about the optimization landscape.

5. **Add Riemannian gradient descent** as a baseline to provide a complete comparison among transformation-invariant optimizers.

---

## OXIIFZqiiN

- GT: Reject (avg 1.5)
- Predicted: Reject (2.0/10)
- Match: Yes

### Final Review

## Summary

This paper introduces the Image-Guided Code Patch Framework (IGCP) for patch representation learning, proposing a theoretical foundation drawing from measure theory, RKHS embeddings, quantum information theory, and stochastic optimization. The framework aims to unify predictive and generative tasks for software patches through a multi-objective loss function. Empirical evaluation focuses on patch description generation, reporting improvements in BLEU, ROUGE-L, and METEOR metrics over baselines.

## Strengths

- **Comprehensive baseline comparison**: The paper compares against multiple established methods (CoDiSum, CoreGen, ATOM, FIRA, CCRep, CCPGen) in patch description generation, providing context for evaluating performance improvements.

- **Multi-component loss architecture**: The framework integrates three loss components (PDG, PDM, PDC) and the ablation study in Figure 3 shows their relative contributions, demonstrating that the full model achieves the best performance.

- **Ambitious theoretical scope**: The paper attempts to provide rigorous mathematical foundations for patch representation learning, which is commendable in scope, even if the practical connection remains unclear.

## Weaknesses

- **Fundamental disconnect between title and content**: The title promises "visual prompts" and "image-guided" patch analysis, yet the paper never explains what visual prompts are, how images are derived from code patches, or how visual guidance operates in the framework. The central claimed contribution is undefined and unsupported.

- **Theory-practice gap**: The paper presents extensive mathematical machinery (measure theory on Polish spaces, quantum information bottleneck, free probability, SGLD convergence analysis) but never connects these to the actual implementation. Critical questions remain unanswered: How are quantum density operators instantiated from code? How does the RKHS embedding translate to neural network architecture? Without this bridge, the theoretical sections appear ornamental rather than functional.

- **Missing reproducibility information**: The paper lacks essential implementation details including model architecture specifications (encoder/decoder dimensions, layer counts), training hyperparameters (learning rates, batch sizes, number of epochs), optimizer settings, and computational requirements. This severely undermines reproducibility.

- **Unsubstantiated claims**: The abstract promises evaluation on "novel information-theoretic metrics derived from our theoretical analysis" and "domain generalization capabilities," but neither appears in the experiments. All evaluations use a single Java dataset with standard metrics (BLEU, ROUGE-L, METEOR).

- **No statistical rigor**: Results report single numbers without standard deviations, confidence intervals, or significance tests, making it unclear whether reported improvements over baselines are statistically meaningful.

- **Incomplete theorem statements**: Theorem 3.9 on phase transitions states "There exists a critical value αc such that:" without completing the result. Several mathematical statements appear truncated, undermining the theoretical contribution.

- **Terminology inconsistency**: Section 4.1 references "IPPMF" while the framework is introduced as "IGCP," creating confusion about naming conventions.

## Nice-to-Haves

- Qualitative examples showing actual generated patch descriptions alongside ground truth and baseline outputs would strengthen the practical utility assessment.

- Training curves or convergence analysis connecting the SGLD theory to observed training dynamics would help bridge the theory-practice gap.

## Novel Insights

The paper's most distinctive theoretical proposal is the quantum information bottleneck formulation (Theorem 3.4) for code patch representations, which extends the classical information bottleneck to capture potential "entanglement between different aspects of code patches." However, this insight remains speculative—the paper does not demonstrate that quantum correlations actually exist in code patch data or that the quantum formulation provides benefits beyond classical information theory. The phase transition analysis (Theorem 3.9) identifies a potentially interesting connection between sample size and generalization regimes, but without empirical validation (e.g., plots of the claimed order parameter versus training dynamics), this remains an unverified theoretical prediction.

## Potentially Missed Related Work

- None identified through external search.

## Suggestions

1. **Clarify or revise the visual component**: Either substantively explain how visual prompts/images are used in the framework (with implementation details and ablation studies), or revise the title and framing to accurately reflect the code-focused approach.

2. **Bridge theory to implementation**: Add explicit mappings between theoretical constructs (e.g., quantum states, RKHS embeddings) and concrete neural network components, loss functions, or training procedures.

3. **Provide reproducibility details**: Include model architecture specifications, hyperparameters, training procedures, and ideally release code.

4. **Add statistical rigor**: Report means and standard deviations across multiple runs, and include significance tests for comparison against baselines.

5. **Support or remove unsupported claims**: Either add cross-domain evaluation and information-theoretic metrics, or remove these claims from the abstract.

---

## PigfMZMHq1

- GT: Reject (avg 3.7)
- Predicted: Reject (3.0/10)
- Match: Yes

### Final Review

## Summary

This paper introduces PointNet-KAN, which integrates Kolmogorov-Arnold Networks (KANs) into PointNet by replacing MLPs with learnable activation functions based on Jacobi polynomials. The authors evaluate PointNet-KAN on 3D object classification (ModelNet40, ScanObjectNN) and part segmentation (ShapeNet Part), demonstrating competitive performance with a simpler architecture compared to vanilla PointNet.

## Strengths

- **First integration of KANs into point cloud neural networks**: The paper addresses a genuine gap by being the first to apply KANs to 3D point cloud processing tasks, extending prior KAN integration work in CNNs and GNNs to a new domain.

- **Systematic hyperparameter evaluation**: Tables 4-7 provide thorough ablation studies examining polynomial degree (n=2 to 6) and various Jacobi polynomial types (Legendre, Chebyshev, Gegenbauer), offering practical guidance for future KAN-based architectures.

- **Multiple benchmark datasets and robustness tests**: Evaluation on ModelNet40, ScanObjectNN (real-world data), and ShapeNet Part provides reasonable coverage. Figures 3-4 include robustness tests under point dropout and Gaussian noise perturbations.

- **Computational efficiency analysis**: Table 1 reports FLOPs per sample (60M for PointNet-KAN vs 148M for PointNet baseline), providing useful computational context.

- **Extension to advanced architectures**: Supplementary material demonstrates the approach extends to PointMLP (creating PointKAN), showing generalizability beyond vanilla PointNet.

## Weaknesses

- **Missing statistical significance**: All results (Tables 1-7) report single values without standard deviations or confidence intervals. Claims like "90.5% vs 89.2%" cannot be verified as statistically significant without multiple experimental runs.

- **Confounded comparison between architectures**: PointNet-KAN uses a simpler architecture (3 hidden layers vs 8, no input/feature transforms) than PointNet, making it unclear whether performance differences stem from KAN substitution or architectural simplification. The paper simultaneously changes multiple design choices and attributes results to KANs.

- **Negative results without adequate explanation**: When the authors replace all MLPs in full PointNet with KANs (Section 5.3), performance degrades (88.9% vs 90.5% for classification). This counterintuitive finding—that deeper KAN networks underperform simpler ones—is not explained and raises questions about KAN scalability.

- **Mixed performance across conditions**: PointNet-KAN underperforms PointNet in several settings: ModelNet40 without normals (87.5% vs 89.2%), ScanObjectNN without normals (66.5% vs 68.2%), and ShapeNet segmentation (83.3% vs 83.7%). Only with normal vectors does PointNet-KAN show improvement, complicating the narrative.

- **Hyperparameter insensitivity raises questions**: Tables 4-7 show polynomial degree and type have minimal impact (~1% variation), suggesting the learned basis functions may not be effectively utilized—potentially reducing to a reparameterization rather than learning meaningful activations.

## Nice-to-Haves

- **Parameter-matched MLP baseline**: A PointNet-MLP with comparable parameters (~1M) would isolate whether benefits come from KAN's properties or simply fewer parameters.

- **Training dynamics analysis**: Training time, memory footprint, and convergence curves would help readers assess practical viability. KANs require computing polynomial basis functions which may be slower than matrix multiplications.

- **Visualization of learned activation functions**: The paper claims KAN learns activation functions but provides no visualization of what ψ(γ) converges to after training—critical for understanding whether KAN provides representational benefits.

## Novel Insights

The key finding is that polynomial type and degree have minimal impact on performance for point cloud tasks, suggesting that low-degree polynomials (n=2) suffice for these applications. This contrasts with some claims in the KAN literature about the importance of representational capacity. The negative result in Section 5.3—that deeper KAN architectures underperform simpler ones—suggests that KANs may benefit from architectural simplicity, but this warrants further investigation. The extension to PointMLP (PointKAN) shows competitive results (94.6% vs 94.5%), indicating that KAN integration may scale better to advanced architectures than to vanilla PointNet.

## Potentially Missed Related Work

- None identified by the review process.

## Suggestions

1. Add standard deviations across at least 3-5 random seeds to all quantitative results to establish statistical significance.

2. Conduct controlled experiments comparing PointNet-KAN against a parameter-matched PointNet-MLP baseline (same layer count, no transforms) to isolate the KAN contribution.

3. Investigate and explain why deeper KAN architectures underperform—include training curves, gradient analysis, or optimization difficulty discussion.

4. Move the PointKAN (PointMLP + KAN) results from supplementary material to the main paper, as this demonstrates scalability to modern architectures and strengthens the contribution.

---

## MFZjrTFE7h

- GT: Accept (Spotlight) (avg 7.5)
- Predicted: Accept (6.5/10)
- Match: Yes

### Final Review

## Summary
D-FINE introduces Fine-grained Distribution Refinement (FDR) and Global Optimal Localization Self-Distillation (GO-LSD) to improve bounding box regression in DETR-based object detectors. FDR transforms coordinate prediction into iterative probability distribution refinement, enabling uncertainty modeling and progressive localization refinement. GO-LSD transfers localization knowledge from deeper decoder layers to shallower ones with minimal training overhead. D-FINE achieves state-of-the-art results on COCO (54.0%/55.8% AP for L/X variants) while maintaining real-time speeds, and demonstrates strong scalability with Objects365 pretraining (up to 59.3% AP).

## Strengths
- **Strong empirical performance**: D-FINE consistently outperforms existing real-time detectors (YOLOv10, RT-DETR, LW-DETR) across multiple model sizes on COCO val2017, with comprehensive comparisons across latency, parameters, and FLOPs (Tables 1, 7, 8). D-FINE-X achieves 55.8% AP at 78 FPS, surpassing YOLOv10-X (54.4% AP) with fewer parameters (62M vs 30M) and lower latency.

- **Broad applicability**: Table 3 demonstrates that FDR and GO-LSD improve multiple DETR architectures (Deformable-DETR, DAB-DETR, DN-DETR, DINO) by 2.0–5.3% AP without increasing parameters, showing the methods are not architecture-specific.

- **Efficient self-distillation**: GO-LSD achieves knowledge transfer without requiring a separate teacher model, with only 6% increase in training time and 2% increase in memory usage (Table 5), making it practical for real-world deployment.

- **Transparent reproducibility**: Detailed hyperparameter configurations (Table 6), implementation details in the appendix, and open-source code support reproducibility.

## Weaknesses
- **Incomplete ablation of core contributions**: Table 4 shows FDR and GO-LSD added together in a single step (+1.0 AP). While Table 5 compares distillation methods with FDR present, the isolated contribution of FDR alone versus GO-LSD alone on the final model architecture remains unclear. This makes it difficult to assess whether the gains come primarily from distribution refinement, self-distillation, or their combination.

- **Missing comparison with closest prior work**: GFocal (Li et al., 2020, 2021) is discussed in Related Work and identified as the closest distribution-based approach, but is not included in experimental comparisons. Since FDR extends distribution-based regression to anchor-free DETR with iterative refinement, a direct comparison (even if adapted to DETR) would substantiate the claimed advantages over GFocal's one-shot approach.

- **Convergence acceleration claim lacks evidence**: The introduction states that GO-LSD "accelerates convergence," but no training curves or convergence speed analysis are provided. Without this, the efficiency benefits of self-distillation remain unsubstantiated.

- **No failure case analysis**: Figures 4 and 5 show successful detections, but no analysis of when FDR/GO-LSD fail to help or produce worse localizations than fixed-coordinate regression. Understanding method limitations is critical for practical deployment.

- **Intermediate layer dynamics unanalyzed**: The visualization in Figure 4 shows only initial and final distributions, but not how distributions evolve across intermediate decoder layers. This would reveal whether iterative refinement progressively improves or if gains come primarily from final-layer predictions.

## Nice-to-Haves
- Statistical significance metrics (error bars across multiple runs) to quantify result reproducibility.
- Analysis of why FDR helps certain object sizes more than others—AP_S, AP_M, AP_L improvements vary without explanation.

## Novel Insights
D-FINE's key insight is that bounding box regression can be reformulated as iterative distribution refinement rather than direct coordinate prediction, enabling independent edge uncertainty modeling. The non-uniform weighting function W(n) provides a clever mechanism: gentle curvature near the center allows fine-grained adjustments for nearly-correct predictions, while steeper curvature at boundaries enables larger corrections for poor predictions. GO-LSD's union-set matching for self-distillation is an elegant solution to DETR's one-to-one matching instability—by aggregating matches across all layers, even low-confidence but well-localized predictions can transfer knowledge to earlier layers. The method demonstrates that deeper decoder layers naturally encode better localization knowledge that can be distilled back without external teachers.

## Potentially Missed Related Work
None identified (search was not performed).

## Suggestions
- Provide separate ablations for FDR and GO-LSD on the baseline architecture to isolate individual contributions.
- Add training convergence curves comparing D-FINE with baselines to substantiate the claimed faster convergence from GO-LSD.
- Include failure case visualizations showing where distribution refinement degrades performance compared to fixed-coordinate prediction.
- Visualize distribution evolution across intermediate decoder layers to demonstrate progressive refinement.

---

## H25xduunIK

- GT: Reject (avg 5.8)
- Predicted: Accept (5.5/10)
- Match: No

### Final Review

## Summary
This paper introduces Report Cards, natural language summaries that capture model capabilities for specific skills, along with PRESS, an iterative algorithm for generating them. The authors propose three evaluation criteria—specificity (distinguishing models), faithfulness (accurately representing capabilities), and interpretability (human clarity)—with corresponding metrics: contrastive accuracy, R² between Card Elo and Oracle Elo, and human Likert ratings. Experiments across multiple LLMs and datasets demonstrate that Report Cards can capture nuanced model behaviors and compress information more efficiently than few-shot baselines.

## Strengths
- **Novel conceptual contribution**: The paper addresses a genuine gap in LLM evaluation—quantitative benchmarks fail to capture nuanced behaviors, and manual qualitative assessment is labor-intensive. Report Cards offer a principled middle ground.
- **Well-designed evaluation framework**: The three-metric approach (specificity/faithfulness/interpretability) covers essential dimensions, each with concrete operationalizations: contrastive accuracy, Elo correlation, and human ratings.
- **Strong empirical coverage**: Experiments span 9+ models across three datasets (MMLU, Adv. AI Risk, Chinese Grammar), with meaningful ablations including format comparisons, guesser model strength, quiz length, and de-stylization robustness.
- **Evidence of efficient compression**: Ablations show bullet-point Report Cards achieve 69% contrastive accuracy with ~900 words versus few-shot achieving 61% with ~1700 words (Appendix C.2), demonstrating meaningful information density.
- **De-stylization experiments address style vs. substance**: The paper shows Report Cards maintain performance when stylistic features are removed (Figure 6), while few-shot baselines degrade significantly—addressing the concern that summaries might capture superficial patterns rather than genuine capabilities.

## Weaknesses
- **No ground-truth verification of factual claims**: The faithfulness metric relies on LLM judge correlations, but the paper never validates whether specific statements in Report Cards (e.g., "struggles with combinatorics") are factually accurate on held-out data. This leaves open whether Report Cards hallucinate weaknesses or miss obvious ones.
- **Weak LLM-human alignment limits automation potential**: Table 9 shows Spearman correlations of 0.27–0.40 for relevance/informativeness and near-zero for clarity between LLM and human scores. This fundamentally limits the ability to automate interpretability scoring at scale.
- **No evaluator model ablation**: All Report Cards are generated by Claude 3.5 Sonnet. Without showing whether different evaluator models produce consistent summaries, it's unclear whether results depend critically on this specific model choice.
- **Interpretability evaluated on excerpts, not full Report Cards**: Human evaluators rate pre-extracted excerpts relevant to questions (Appendix E), not complete Report Cards. This doesn't test whether humans can navigate full summaries to find relevant information—the core interpretability claim.
- **Limited human evaluation scope and missing agreement metrics**: Only 18 volunteers and 230 annotations; inter-annotator agreement is not reported despite the subjective nature of the ratings.

## Nice-to-Haves
- **Failure case analysis**: The paper presents qualitative examples where Report Cards correctly identify model weaknesses (Figure 10) but no systematic analysis of cases where they produce misleading summaries or miss obvious patterns.
- **Cross-dataset transfer analysis**: Testing whether Report Cards trained on one dataset transfer to distinguishing model behavior on another dataset with similar skills would strengthen claims about capturing genuine capabilities.

## Novel Insights
The de-stylization experiments reveal an important finding: few-shot baselines achieve high contrastive accuracy partly by encoding stylistic patterns (model-specific phrasing), while Report Cards trained on de-stylized completions maintain performance. This suggests PRESS genuinely captures semantic behavior rather than surface artifacts. Additionally, the ablation showing smaller models (GPT-4o-mini, Claude 3.5 Haiku) can generate Report Cards with similar quality to stronger models (Appendix C.2, lines 879–882) has practical implications for deployment cost, though this finding is buried in the appendix.

## Potentially Missed Related Work
None identified. The paper adequately situates itself within Model Cards, qualitative evaluation, LLM-as-judge, and fine-grained evaluation literature.

## Suggestions
1. **Add held-out verification**: For specific claims in Report Cards (e.g., identified weaknesses), validate against held-out test sets to establish factual accuracy beyond LLM judge correlations.
2. **Report inter-annotator agreement**: Include Cohen's Kappa or similar metrics for human evaluation ratings.
3. **Add random baseline for contrastive accuracy**: A constant predictor using ground-truth labels is informative, but a random baseline would help calibrate task difficulty.
4. **Extend interpretability evaluation to full Report Cards**: Have humans navigate complete summaries to predict model behavior on novel questions, testing the practical utility claim directly.

---

## bEvI30Hb2W

- GT: Reject (avg 3.0)
- Predicted: Reject (4.0/10)
- Match: Yes

### Final Review

## Summary

LVM-Net proposes a memory-augmented architecture for efficient long-form video reasoning. The key idea is to process a video once using a differentiable neural sampler that populates a fixed-size memory with discriminative tokens; subsequent queries are answered from this memory without revisiting the original video. The authors demonstrate 18x-75x inference speedup on the ReST-ADL dataset across activity, object, and time queries.

## Strengths

- **Substantial efficiency gains**: The paper demonstrates meaningful speedups (18x-75x depending on query type and FPS) on videos averaging 27 minutes. This directly addresses a real computational bottleneck for long-form video understanding (Table 1).

- **Principled multi-query inference paradigm**: The two-stage approach—populate memory once, then answer multiple queries—is well-motivated for applications requiring repeated queries on the same video (Figure 3). This is a genuine efficiency gain over per-query processing.

- **Thoughtful sampling bias correction**: The online continual learning loss addresses a real issue where the sampler would otherwise bias toward current query tokens. Table 4 shows meaningful improvements (e.g., 26.39→32.38 Recall@1x for short activity queries), validating this design choice.

- **Practical deployment awareness**: The discussion of edge device deployment and streaming inference demonstrates consideration of real-world applications beyond benchmark evaluation.

## Weaknesses

- **Significant accuracy trade-offs not adequately acknowledged**: Table 2 shows notable performance degradation on activity queries—32.4% vs 45.3% Recall@1x for short queries (a ~29% relative drop compared to modified TubeDETR). The abstract and conclusion frame this as "competitive performance," which understates the trade-off. A 29% relative accuracy drop is substantial and should be explicitly quantified and discussed.

- **Missing relevant baseline comparisons**: The paper cites MeMViT (Wu et al., 2022), Token Turing Machines (Ryoo et al., 2023), and other memory-efficient approaches in related work but does not compare against them. Without these comparisons, it is difficult to assess whether LVM-Net's efficiency gains exceed what simpler memory-based caching would achieve.

- **Single dataset evaluation**: ReST-ADL contains only 4 test videos with ~6000 queries. This narrow evaluation makes it difficult to assess generalization to other video understanding tasks, video lengths, or content domains.

- **No hyperparameter sensitivity analysis**: The memory size (5880 tokens) and continual learning buffer size (p=2) are stated without ablation. How does accuracy degrade with smaller memory? This is central to the method's practical value proposition and remains unexplored.

- **Missing qualitative analysis**: The paper claims the neural sampler identifies "discriminative" tokens but provides no visualization or analysis of what tokens are actually selected. This makes it difficult to verify the sampler learns meaningful representations versus arbitrary compression.

## Nice-to-Haves

- Report error bars or standard deviations across multiple runs to establish statistical significance of reported results.

- Quantify how much speedup remains if TubeDETR's frame representations were cached rather than reprocessed—the current comparison conflates architectural innovation with the single-pass design choice.

## Novel Insights

The core insight—that long-form video reasoning can be decomposed into a one-time memory population phase followed by efficient query answering—is well-realized. The online continual learning loss correctly identifies and addresses a subtle training bias problem: without it, the sampler would preferentially select tokens from the current training clip rather than building a globally useful memory. The ablation in Table 4 validates this contribution. However, the paper would benefit from deeper analysis of when this memory-based approach succeeds versus fails, particularly given the accuracy degradation on certain query types.

## Potentially Missed Related Work

- **MeMViT** (Wu et al., 2022)—cited in related work as caching representations for extended temporal context, but not evaluated as a baseline despite direct relevance to memory-based video processing.
- **Token Turing Machines** (Ryoo et al., 2023)—mentioned as token efficiency method but not compared against; offers another approach to fixed-size token memory.

## Suggestions

- Add comparisons to at least one memory-augmented video architecture (e.g., MeMViT) to contextualize the efficiency-accuracy trade-off against existing memory-based approaches.

- Conduct and report ablation on memory size to characterize the accuracy-efficiency frontier and help practitioners select appropriate configurations.

- Provide qualitative visualizations showing which frames/patches the neural sampler selects for example videos, comparing against random sampling.

- Revise the framing of "competitive performance" to explicitly quantify and discuss the accuracy-efficiency trade-off, particularly for activity queries where the gap is largest.

---

## p1HeFnn2AA

- GT: Reject (avg 7.3)
- Predicted: Accept (6.5/10)
- Match: No

### Final Review

## Summary
The paper initiates the study of deep learning for automated design of two-sided matching mechanisms. The authors introduce differentiable surrogates for ordinal strategy-proofness (via regret from FOSD violations) and ex ante stability, prove these surrogates exactly characterize the respective properties, and train neural networks to learn randomized matching mechanisms that achieve better tradeoffs between stability and strategy-proofness than convex combinations of classical mechanisms (deferred acceptance, top trading cycles, randomized serial dictatorship).

## Strengths
- **Novel problem formulation**: First application of deep learning to two-sided matching mechanism design, extending prior work on auction design to ordinal preference settings—a genuinely new direction for ML in economics.

- **Theoretically grounded surrogates**: Theorems 6, 7, and 8 establish that zero values of the proposed surrogates exactly characterize ex ante stability and ordinal strategy-proofness. The measures are not approximations but exact quantifications of property violations.

- **Permutation-equivariant architecture**: The CNN design exploits symmetries inherent in matching problems, reducing the search space and enabling scaling to 50×50 markets with structured preference domains.

- **Empirical findings with theoretical implications**: Learned mechanisms achieve stability-SP tradeoffs that dominate convex combinations of DA, TTC, and RSD—providing concrete new targets for economic theory to understand the structure of mechanisms with improved tradeoffs.

- **Comprehensive experimental settings**: Evaluation across uncorrelated preferences, correlated preferences, and restricted preference domains provides reasonable coverage of different market structures.

## Weaknesses
- **No statistical significance reporting**: All figures present single training runs without error bars or confidence intervals. Given the stochastic nature of neural network training, this undermines confidence in the reliability and reproducibility of the empirical claims.

- **Scalability limitations**: Computing regret requires enumerating possible misreports, which grows factorially. The paper addresses this for larger experiments (Settings C/D) by restricting preference domains, but this limits the generality of conclusions for realistic markets with unrestricted preferences. The paper acknowledges this limitation but it remains consequential for practical applicability.

- **Incomplete baseline comparisons**: The comparison against convex combinations of DA/TTC/RSD is meaningful, but the paper does not empirically compare against hybrid mechanisms from prior literature (e.g., Mennle & Seuken's work on trading off strategy-proofness with efficiency, which is cited). Such comparisons would better contextualize the improvements.

- **Synthetic data only**: All experiments use synthetically generated preference profiles. While following prior experimental economics practice, validation with real-world matching market data would strengthen claims about practical relevance.

- **Limited interpretability analysis**: Beyond the similarity-to-DA analysis, the paper provides little insight into what matching rules the neural networks learn or what patterns they exploit. This limits the ability for economic theory to build on the empirical discoveries.

## Nice-to-Haves
- Code availability statement and trained model checkpoints to enable reproducibility
- Ablation studies on architectural choices (number of layers, hidden units, activation functions) to clarify which design decisions are important

## Novel Insights
The discovery that mechanisms can simultaneously achieve near-DA stability with substantially improved strategy-proofness (and conversely, near-RSD strategy-proofness with substantially improved stability) provides concrete, previously unknown points on the efficient frontier. The permutation-equivariant architecture demonstrates how incorporating problem-specific symmetries into neural network design can enable scaling while preserving economic structure. The methodological contribution of differentiable surrogates for ordinal economic properties transfers readily to other mechanism design settings beyond matching.

## Potentially Missed Related Work
- Mennle & Seuken (2017) on hybrid mechanisms for trading off strategy-proofness with efficiency—comparing empirically against their constructions would contextualize the improvements over simple convex combinations.

## Suggestions
- Add confidence intervals across multiple training runs with different random seeds to establish result robustness
- Include empirical comparison with existing hybrid mechanisms from prior literature, not just convex combinations
- Provide more analysis of learned mechanism behavior—for example, what types of preference profiles yield matchings most similar to vs. different from DA, and what strategic vulnerabilities remain for mechanisms that are not fully strategy-proof
- If possible, validate on any available real-world matching market data, even at small scale

---

## LyJi5ugyJx

- GT: Accept (Oral) (avg 9.2)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
The paper introduces sCM (Simplified, Stabilized, and Scalable Continuous-time Consistency Models), proposing the TrigFlow formulation that unifies EDM and Flow Matching, along with architectural and training improvements to stabilize continuous-time CM training. The work successfully scales consistency models to 1.5B parameters on ImageNet 512×512, achieving 2-step FID scores within 10% of teacher diffusion models.

## Strengths
- **Theoretical elegance**: TrigFlow provides a clean unification of EDM and Flow Matching with simple coefficient functions (c_skip(t)=cos(t), c_out(t)=-σ_d sin(t), c_in(t)=1/σ_d), simplifying prior parameterizations.
- **Insightful stability analysis**: The decomposition in Eq. (7) correctly identifies sin(t)∂_t F_{θ-} as the primary source of training instability, providing principled justification for the proposed fixes. Figure 4 visualizes this effectively.
- **Strong empirical results**: Achieves FID of 2.06 (CIFAR-10), 1.48 (ImageNet 64×64), and 1.88 (ImageNet 512×512) with 2-step sampling, narrowing the gap with best diffusion models to within 10%.
- **Comprehensive scaling study**: Figure 6 demonstrates predictable scaling behavior across model sizes (S through XXL), with sCD maintaining consistent FID ratio relative to teacher models.
- **Practical engineering contributions**: The JVP rearrangement (Eq. 9) enables FP16 training stability, and Appendix F provides a novel Flash Attention JVP algorithm for memory-efficient large-scale training.

## Weaknesses
- **Insufficient component ablation**: The paper combines at least six techniques (TrigFlow, identity c_noise, positional embeddings, adaptive double normalization, tangent normalization, adaptive weighting) but only ablates tangent normalization/clipping (Figure 5a) and adaptive weighting (Figure 5b). Individual contributions of TrigFlow formulation and other components remain unclear.
- **sCT degrades significantly at scale**: On ImageNet 512×512, sCT-XXL achieves 2-step FID of 3.76 versus sCD-XXL's 1.88—a substantial gap noted only briefly in limitations. The explanation about "higher training variance" and "ill-conditioned ground truth mapping" warrants deeper investigation in the main text.
- **No statistical significance testing**: FID scores are reported without confidence intervals or standard deviations across multiple runs, making it difficult to assess reproducibility of claims like "continuous-time CMs significantly outperform discrete-time CMs."
- **Limited architectural diversity**: All experiments use EDM2 architecture with modified AdaGN. The proposed techniques may not transfer to other architectures (e.g., DiT), limiting generalizability claims.

## Nice-to-Haves
- Wall-clock training time and GPU memory comparisons between sCT, sCD, and discrete-time CM baselines to help practitioners assess practical feasibility.
- Deeper analysis of visual artifacts mentioned in limitations, with examples of failure modes specific to 1-step vs. 2-step generation.

## Novel Insights
The paper makes two notable theoretical contributions. First, the unit variance principle analysis shows that noise schedules can be transformed equivalently without affecting sampling, providing theoretical justification for the TrigFlow parameterization. Second, the stability analysis revealing that instability originates from the time-derivative term sin(t)∂_t F_{θ-} rather than the Jacobian term provides actionable insight—the fix is not to stabilize gradients but to control time-derivative magnitudes through positional embeddings, adaptive normalization, and tangent normalization. This decomposition is more principled than prior heuristic approaches.

## Potentially Missed Related Work
- None identified (related work search was not performed)

## Suggestions
- Add individual ablations isolating each proposed technique (TrigFlow vs. EDM formulation, identity vs. log-based c_noise, etc.) to clarify which components are essential.
- Include confidence intervals on FID scores from multiple training runs to support statistical claims.

---

## L9j8exYGUJ

- GT: Reject (avg 5.0)
- Predicted: Reject (5.0/10)
- Match: Yes

### Final Review

## Summary
The paper proposes "distributional reasoning" as a framework for understanding how LLMs solve multi-hop compositional questions without explicit chain-of-thought. The authors demonstrate that intermediate answer activations from middle layers can predict final answer activations through a linear transformation (R² > 0.5), showing that models activate distributions of potential intermediate answers that correlate with final answer distributions. Experiments with fictitious subjects/attributes suggest reasoning processes persist even without factual knowledge, connecting findings to cognitive psychology's spread-of-activation theory.

## Strengths
- Novel conceptual framework connecting LLM interpretability to cognitive psychology's spread-of-activation theory, providing an alternative to sequential-hop assumptions for understanding implicit multi-hop reasoning
- Strong empirical methodology across multiple models (Llama-2-7B/13B, Llama-3-8B, Mistral-7B) and 14 question types with 6,547 prompts, showing consistent R² > 0.5 for predicting final answer activations from intermediate answer activations
- Clever hallucination experiments isolating reasoning processes from stored knowledge, demonstrating linear models trained on real data generalize to fictitious subjects (R² > 0.3), suggesting the transformation matrix is subject-invariant
- Clear phase transition observation where intermediate answer activations rise in middle layers then decline as final answer activations rise, providing interpretable evidence of distinct processing stages
- Meaningful correlation (0.72) between category-level model accuracy and R² values, suggesting distributional reasoning is more dominant when models reason successfully

## Weaknesses
- No causal evidence establishing that intermediate answer activations cause final answer activations — the correlation could arise from both being independently activated by the subject rather than through compositional transformation. The paper acknowledges this limitation, but the core mechanistic claim remains observational.
- Alternative explanation not adequately ruled out: intermediate and final answer tokens could be independently associated with the subject through training data statistics (e.g., "banana" activating both "yellow" AND "Y" separately), rather than genuine compositional processing. The Spearman correlation evidence is suggestive but doesn't disentangle these possibilities.
- Substantial unexplained variance and category variability: R² values around 0.5–0.7 leave 30–50% of variance unexplained, with some categories showing weak relationships (Russian names: R² = 0.22–0.46; longitude/latitude: R² = 0.19–0.37). This variability is underanalyzed and suggests distributional reasoning may not be the dominant mechanism universally.
- Limited question scope: All experiments use two-hop attribute extraction with similar structures, limiting claims about generalization to n-hop reasoning, negation, or different reasoning operations. The paper acknowledges this but doesn't test boundary conditions.
- No control experiments establishing baselines: The paper lacks experiments showing what R² values would occur under random conditions or for single-hop questions, making it unclear whether the observed relationships are specific to compositional reasoning or could emerge from simple semantic relatedness.

## Nice-to-Haves
- Causal intervention experiments (e.g., patching intermediate answer activations and measuring effects on final answers) to establish causality beyond correlation
- Analysis of why some question types show low R² values despite compositional structure — this could reveal limitations or alternative mechanisms
- Control experiments with shuffled subjects or artificial A₁→A₂ mappings to establish that linear relationships are specific to compositional reasoning

## Novel Insights
The cognitive science framing connecting spread-of-activation theory to mechanistic interpretability is valuable and positions this work uniquely in the literature. The finding that mid-layer intermediate answer activations are *more informative* about final answers than early-layer final answer activations themselves (Figure 3b) is counterintuitive and suggests genuine intermediate computation rather than gradual amplification of pre-existing signals. The hallucination experiments demonstrate that the reasoning transformation can operate on "wrong" intermediate answers, suggesting compositional processing rather than reliance on stored factual associations alone. This opens possibilities for detecting reasoning failures by analyzing intermediate answer distributions.

## Potentially Missed Related Work
- Note: The related work search was not performed, so no specific suggestions are provided. The paper cites relevant work on reasoning in LLMs (Wei et al., 2022; Yang et al., 2024; Li et al., 2024), interpretability methods (Geva et al., 2023; Elhage et al., 2021), and linear representation hypotheses (Gurnee and Tegmark, 2023; Park et al., 2023, 2024).

## Suggestions
- Add causal intervention experiments to test whether modifying intermediate answer activations causes predictable changes in final answer activations, moving from correlation to mechanism claims
- Include control experiments with single-hop questions and shuffled/artificial mappings to establish that the linear relationship is specific to compositional reasoning and not trivially explained by semantic relatedness or independent associations
- Expand failure mode analysis to understand cases where R² is low despite correct answers, or high despite incorrect answers — this would clarify when distributional reasoning is and isn't the dominant mechanism

---

## cLws58ZojF

- GT: Reject (avg 3.0)
- Predicted: Reject (5.0/10)
- Match: Yes

### Final Review

## Summary
This paper presents a systematic exploration of the design space for Speech-Conditioned Large Language Models (SLMs), investigating adaptor architectures, trainable modules, masking strategies, LLM selection, and training data composition. A key finding is that existing SLMs struggle with spoken instruction following because their training data uses speech as context rather than as instructions—the authors address this by creating a 50K synthetic spoken instruction dataset. Their resulting model, SiM, achieves strong ASR performance and significantly outperforms existing SLMs on spoken instruction following benchmarks.

## Strengths
- **Systematic design space exploration**: The paper conducts controlled experiments across multiple design axes (adaptor architecture, trainable modules, masking, LLM choice, training data) with consistent experimental settings, addressing a real gap in the literature where SLMs are developed under incomparable conditions. For example, Table 1 shows that a simple 2-layer MLP achieves 4.9% WER, outperforming more complex Q-Former variants that suffer from training instability.
- **Important and actionable insight**: The discovery that existing SLMs fail on spoken instruction following because they lack spoken instruction data is valuable. Figure 4 demonstrates this failure mode concretely, showing that models like SALMONN and Qwen Audio Chat simply transcribe spoken queries rather than responding to them.
- **Strong empirical improvements with human validation**: SiM achieves scores of 2.71 and 2.63 on OpenHermes Audio and Alpaca Audio benchmarks (Table 6), substantially outperforming existing SLMs. Human preference evaluations (Figure 2) confirm these improvements, with SiM preferred over SALMONN 81% of the time and over Qwen Audio Chat 84% of the time.

## Weaknesses
- **Insufficient reproducibility details**: Critical hyperparameters—including learning rates, batch sizes, number of training epochs, optimizer settings, and hardware specifications—are not provided. This significantly impedes reproducibility.
- **Missing ablation for spoken instruction data contribution**: The paper combines multiple design choices (adaptor type, LLM choice, training data composition, instruction tuning data) but never isolates whether the spoken instruction dataset alone drives the improvements. Training baseline SLM architectures on the same spoken instruction data would validate the core causal claim.
- **Copy-editing errors undermine credibility**: The discussion of Table 6 states "SiM's scores are more than % higher"—the percentage is missing entirely. Additionally, Experiment #4 references "Table 7" when the relevant data is in Table 4.
- **No statistical significance reporting**: None of the experiments report standard deviations, confidence intervals, or significance tests. Small differences (e.g., Table 3: 4.89% vs 4.93%) cannot be meaningfully interpreted without this.
- **Limited analysis of synthetic data quality**: The 50K synthetic spoken instruction dataset is generated using Amazon Polly TTS, but there is no discussion of potential quality issues, speaker diversity, or whether gains generalize to real human speech beyond TTS-generated training data.

## Nice-to-Haves
- Include Qwen2-Audio in all quantitative evaluation tables (Tables 6-8), as it appears only in human preference figures despite being the most recent competitive baseline.
- Evaluate under realistic acoustic conditions (noise, accents, varied speaking rates) beyond clean LibriSpeech to substantiate claims about real-world spoken instruction following capability.

## Novel Insights
The paper's central insight—that existing SLMs fail at spoken instruction following because their instruction tuning data provides instructions in text form while speech serves only as context—is both surprising and significant. The authors validate this by showing that models trained on thousands of hours of speech data still cannot respond to simple spoken queries like "can I go to the moon?" These models instead transcribe the speech rather than following it as an instruction. Creating synthetic spoken instruction data (converting text instructions to speech via TTS) dramatically improves this capability, suggesting a fundamental mismatch in prior training paradigms. The finding that a simple 2-layer MLP adaptor outperforms more sophisticated Q-Former architectures is also noteworthy, though it lacks deeper analysis of why this occurs.

## Potentially Missed Related Work
- None identified in the search.

## Suggestions
- Provide complete hyperparameter details (learning rate, batch size, epochs, optimizer) in an appendix to ensure reproducibility.
- Conduct an ablation experiment where spoken instruction data is added to baseline SLM architectures (e.g., SALMONN's configuration) to isolate the contribution of this data from other design choices.
- Fix the missing percentage value in the Table 6 discussion and the incorrect table reference in Experiment #4.

---

## m29SV0n6DO

- GT: Reject (avg 4.2)
- Predicted: Reject (5.0/10)
- Match: Yes

### Final Review

## Summary
The paper presents "Toto," an empirical study of autoregressive generative pre-training for videos using decoder-only transformers trained on over 1 trillion visual tokens (from dVAE tokenization). The authors systematically evaluate design choices—tokenization strategies, probing methods, resolution scaling, and architecture variants—and benchmark representations across image recognition, video classification, action forecasting, object tracking, object permanence, and robotic manipulation. The work provides scaling law analysis showing power-law behavior with visual tokens, offering practical findings for future video foundation model research.

## Strengths
- **Comprehensive empirical study with useful ablations**: The paper systematically evaluates tokenizers (dVAE, VQGAN, patches), probing methods (attention vs. average pooling showing 7.9% improvement), optimal probing layers (middle layers, Figure 4), and resolution scaling strategies (coarse-to-fine fine-tuning improves over full-resolution pre-training, Table 4). These practical findings are valuable for practitioners.

- **Scale and breadth of experimentation**: Training models up to 1.1B parameters on 100,000+ hours of video (~1 trillion tokens) with evaluation across six distinct task domains—ImageNet classification, Kinetics action recognition, Ego4D action anticipation, DAVIS tracking, CATER object permanence, and robotic manipulation—provides a holistic view of learned representations.

- **Novel empirical findings for video autoregressive models**: The scaling law analysis (L(C) = 7.42 × C^-0.0386) shows slower scaling than language models, and the paper demonstrates that autoregressive video models achieve competitive performance within the generative model category—outperforming iGPT-XL (6.8B params) with Toto-1B (1.1B params) on ImageNet linear probing.

- **Useful robotics transfer results**: The simulated robotics experiments (Figure 6) show faster learning than MAE-base across four tasks, and real-world experiments achieve 63% success on Franka cube-picking, demonstrating practical transfer of learned representations.

## Weaknesses
- **Data leakage concern for ImageNet evaluation**: The paper uses ImageNet-1k images during pre-training (Table 2: 13.9M images, 3.6B tokens) but also evaluates on ImageNet classification (Table 7). Standard self-supervised learning papers (MAE, DINO, SimCLR) typically avoid this overlap to ensure fair evaluation of transfer learning capability.

- **No ablation isolating video contribution**: The paper uses mixed data (60% HowTo100M, 10% Ego4D, 10% Kinetics, 20% ImageNet) but never compares against an image-only baseline trained on equivalent tokens. Without this ablation, the claimed benefit of video pre-training remains unsubstantiated—readers cannot determine whether temporal modeling from video data actually helps versus simply more image data.

- **Substantial gap to discriminative and masked modeling methods**: On ImageNet, Toto-1B achieves 75.3% while DINOv2-g achieves 86.4% and MAE-L achieves 80.9%—gaps of 11+ and 5+ points respectively. On Kinetics-400, Toto-1B achieves 74.4% versus VideoMAE-L's 79.8% (5+ point gap). The paper acknowledges this but doesn't analyze why autoregressive modeling underperforms or when it might be preferred despite the gap.

- **Missing direct comparison to AIM**: AIM (El-Nouby et al., 2024) is the closest prior work—autoregressive pre-training for vision—but the paper only reports AIM's 82.2% ImageNet result without comparing under similar compute/data regimes. This makes it difficult to assess the marginal contribution of video data versus architectural differences.

- **No fine-tuning results for ImageNet**: The paper only reports linear probing results (75.3% for Toto-1B), while MAE, BEiT, and other baselines report end-to-end fine-tuning results. Without fine-tuning numbers, readers cannot compare true transfer learning capability against standard benchmarks.

- **Inefficient training paradigm without discussion**: Autoregressive next-token prediction requires sequential processing and is fundamentally slower than masked modeling approaches (which process tokens in parallel). The paper doesn't report throughput, wall-clock time, or compute efficiency comparisons that would contextualize whether the approach is practical.

## Nice-to-Haves
- **Deeper analysis of why visual tokens scale slower than language**: The scaling exponent (-0.0386) is notably smaller than GPT's (-0.048). The paper reports this but doesn't investigate hypotheses—is this due to redundancy in visual data, tokenization quality, or fundamental differences between modalities?

- **Generated sample visualization**: As an autoregressive generative model, showing generated video frames would help readers assess whether the model learns coherent visual structure and whether tokenization preserves sufficient information.

## Novel Insights
The paper's most interesting findings concern probing methodology for decoder-only models: attention pooling substantially outperforms average pooling (7.9% improvement) because earlier tokens in causal attention have access to less context than later tokens—equally weighting them penalizes later, better-informed tokens. The middle-layer peak for transfer performance (Figure 4) mirrors findings in iGPT and contrasts with encoder-decoder architectures where best features emerge at the encoder's top. The coarse-to-fine resolution finding—that fine-tuning from 128×128 to higher resolution outperforms training directly at 256×256—suggests potential compute savings for future work.

## Potentially Missed Related Work
None identified.

## Suggestions
1. **Add an image-only baseline**: Train on the same total tokens but from images only (no video) to isolate the contribution of temporal data.

2. **Evaluate without ImageNet pre-training overlap**: Either remove ImageNet from pre-training data or evaluate transfer on out-of-distribution datasets to ensure fair comparison with prior self-supervised methods.

3. **Report fine-tuning results**: Add end-to-end fine-tuning numbers for ImageNet to enable direct comparison with MAE, DINO, and other baselines that report fine-tuning.

4. **Add compute efficiency comparison**: Report training FLOPs or wall-clock time versus VideoMAE/MAE to help readers assess whether the approach is computationally practical relative to masked modeling alternatives.

---

## fBSc0c1IXJ

- GT: Reject (avg 3.0)
- Predicted: Accept (6.5/10)
- Match: No

### Final Review

## Summary
The paper introduces Remote Reinforcement Learning (RRL), a setting where an actor taking actions lacks direct access to rewards, which are only available to a remote controller communicating over a rate-limited channel. The proposed solution, GRASP, uses channel simulation based on importance sampling to communicate action samples using approximately D_KL[P||Q] bits rather than H[P] + D_KL[P||Q] bits, combined with behavioral cloning at the actor to maintain a reference distribution Q. Experiments across diverse RL environments demonstrate 4.2× to 115× communication reduction with maintained policy performance.

## Strengths
- **Novel and well-motivated problem formulation**: The RRL setting addresses real distributed control scenarios where reward computation is centralized (human-in-the-loop systems, wireless sensor networks, distributed engineering systems). The four limitations of naive reward transmission—limited communication, feasibility (edge device constraints), parallelism, and coordination—are clearly articulated with appropriate examples.

- **Principled theoretical grounding**: The connection to channel simulation literature (Cuff 2008; Li & El Gamal 2018; Theis & Yosri 2022) is appropriate. The core insight—that communicating samples via importance sampling reduces transmission cost from H[P] + D_KL[P||Q] to D_KL[P||Q]—is correctly adapted to the RL context with proper theoretical justification.

- **Substantial empirical communication savings**: The method achieves 4.2× to 115× reduction versus action source coding (ASC), with geometric mean of 13×. For continuous action spaces (e.g., HalfCheetah with DDPG), savings reach 38× versus ASC and 258× versus ASC in other configurations. These savings are practically significant for bandwidth-constrained applications.

- **Comprehensive experimental coverage**: Evaluation across 8+ environments spanning discrete/continuous actions (CartPole to BipedalWalker to HalfCheetah), classic control/Atari/MuJoCo, and single/multi-agent settings (CooperativePong, PistonBall, Spread) with multiple RL algorithms (PPO, DQN, Soft Q-learning, DDPG) demonstrates robustness across varied problem structures.

- **Maintained policy performance**: Tables 1 and 3 show return gaps typically within a few percentage points between controller and actor policies across most environments, demonstrating that behavioral cloning effectively transfers the policy without substantial performance degradation.

## Weaknesses
- **No comparison to periodic policy transmission**: The paper mentions periodically transmitting policy parameters as an alternative to behavioral cloning (Lines 306-310) but provides no empirical comparison. If transmitting weights every N timesteps achieves similar communication savings with less complexity, the necessity of online behavioral cloning would be undermined.

- **Incomplete reward transmission baseline**: Comparing only to 32-bit reward transmission assumes uncompressed rewards. The paper should evaluate quantized rewards (e.g., 8-bit, 16-bit) to establish whether claimed savings are meaningful against a realistic compression baseline.

- **No computational overhead analysis**: While behavioral cloning and ordered random coding reduce communication, the paper provides no measurements of wall-clock training time, encoding/decoding latency, or memory footprint. For real-time control applications (the motivating scenario), this trade-off between communication savings and computational cost is critical to assess feasibility.

- **Synchronization assumption lacks robustness analysis**: Algorithm 1-2 assumes controller and actor update policy parameters in lockstep. Real distributed systems face network latency, message loss, and asynchronous operation. The paper acknowledges this limitation but provides no analysis or experiments addressing desynchronization robustness or recovery mechanisms.

- **Behavioral cloning success unexplained**: BC is known to suffer from distribution shift and error compounding in sequential decision-making, yet works well here. The Breakout environment shows larger return gaps (12-15% in Table 1), suggesting BC quality varies across domains. The paper notes this but provides no analysis of when BC succeeds versus fails.

## Nice-to-Haves
- Time-varying communication cost analysis: Plotting instantaneous communication bits per timestep would reveal whether early training suffers from prohibitively high costs when KL divergence between policies is largest.

- Ablation on behavioral cloning frequency: Testing whether less frequent actor policy updates affect communication savings would clarify sensitivity to synchronization requirements and inform practical deployment.

## Novel Insights
The key insight—combining channel simulation with behavioral cloning to exploit policy alignment for communication efficiency—is elegant. Rather than transmitting actions or rewards directly, GRASP leverages the fact that when the actor maintains an estimate of the controller's policy, only the divergence between distributions needs transmission. The adaptation of theoretical results from information theory (achieving D_KL[P||Q] communication cost via importance sampling) to RL is creative and well-executed. The natural extension to multi-agent scenarios through centralized training with decentralized execution is a thoughtful application of the MARL paradigm. This work opens connections between information theory and RL that could inspire further research on communication-efficient distributed learning.

## Potentially Missed Related Work
- None identified (search was skipped)

## Suggestions
1. Add an experiment comparing GRASP to periodic policy transmission (transmitting network weights every N timesteps) to justify the necessity of online behavioral cloning.

2. Evaluate against quantized reward transmission (8-bit and 16-bit) to provide a meaningful baseline comparison rather than only uncompressed 32-bit rewards.

3. Report computational overhead: encoding/decoding latency per timestep, wall-clock training time comparison between GRASP and ASC, and memory requirements for both controller and actor.

4. Analyze desynchronization robustness: experiment with delayed messages, dropped policy updates, or asynchronous operation to assess real-world deployment feasibility.

5. Investigate the Breakout return gap to identify conditions where behavioral cloning degrades and provide guidance on failure modes.

---

## 9AtlhmFVDi

- GT: Reject (avg 5.5)
- Predicted: Accept (5.5/10)
- Match: No

### Final Review

## Summary
This paper demonstrates that standard Transformers can learn to attend to Euclidean distance when provided with linear embeddings of coordinates, without requiring specialized SE(3)-equivariant architectures. The authors provide theoretical analysis showing how LayerNorm's normalization statistics enable quadratic distance computation from linear embeddings, and validate this through experiments on simulated 3D points and protein masked token prediction, demonstrating improved downstream performance on protein function prediction.

## Strengths
- **Novel theoretical contribution**: The paper provides a clear mathematical derivation explaining how LayerNorm combined with linear coordinate embeddings can produce approximately Gaussian attention weights based on squared Euclidean distance—addressing how standard Transformers might perform structural reasoning without architectural modifications.
- **Strong validation on simulated data**: The controlled experiments testing different distance powers (Figure 2a) directly confirm the theoretical prediction that p=2 (Gaussian) is optimal, and the head dimension experiments (Figure 2b) empirically validate the n+2 dimensionality requirement.
- **Clear demonstration of SE(3) augmentation benefits**: Figure 3 convincingly shows that random rotation augmentation during training prevents overfitting and encourages learning SE(3)-invariant distance metrics.
- **Practical relevance for efficiency**: The approach enables use of standard Transformer optimizations like FlashAttention for fully-connected attention on structures, whereas GNN-based methods typically require limiting connectivity for memory efficiency—a concrete practical advantage.
- **Competitive downstream performance**: The finetuned structural model achieves meaningfully improved AUPRC (0.566) compared to DeepFRI (0.446) on molecular function prediction, demonstrating practical utility.

## Weaknesses
- **Incomplete comparison to contemporary methods**: While the paper cites ProSST, ESM-IF, and ESM3 as related work, no comparison against these recent structural protein Transformers is provided. DeepFRI (2021) predates modern protein language models, limiting the ability to assess the approach against current state-of-the-art.

- **Theoretical completeness gaps**: The extension from 1D to n-dimensional case is stated as "similar" without formal proof. The Q=K=Id assumption simplifies the theoretical analysis, but the paper doesn't analyze how learned Q/K projections might interact with or distort the distance computation mechanism. Additionally, while the "small coordinates" assumption is discussed, no rigorous bounds on approximation error are provided.

- **Limited task evaluation**: The paper claims Transformers can "function independently as structure models" but only demonstrates this on masked token prediction and GO function prediction. Important structure-centric tasks such as inverse folding, structure prediction, or contact prediction are not evaluated.

- **Missing statistical rigor**: Tables 1-3 report single numbers without error bars, confidence intervals, or significance tests. For downstream performance claims, this is a notable gap.

- **Theoretical-empirical disconnect**: The paper develops three embedding types (E_trig, E_lin, E_quad) theoretically and validates them on simulated data, but doesn't analyze which mechanism the actual protein model uses. This leaves the connection between theory and protein experiments incompletely established.

## Nice-to-Haves
- Analysis of individual attention heads to verify the claim that "each head can tune the variance of this Gaussian filter"—currently only average attention patterns are shown.
- Experiments with noisy/predicted coordinates (e.g., from AlphaFold) to assess practical robustness since real-world applications would not have ground-truth structures.

## Novel Insights
The core insight—that LayerNorm's variance computation inherently introduces quadratic terms enabling distance-squared attention from linear embeddings—is genuinely novel and provides theoretical grounding for understanding how "standard" Transformers might process geometric information. The observation that random rotation augmentation effectively teaches SE(3)-invariance without architectural enforcement is practically useful, and the finding that n+2 embedding dimensions suffice for R^n distance measurement provides concrete guidance for practitioners. The connection to AlphaFold3's diffusion transformer architecture contextualizes this work within an important contemporary development.

## Potentially Missed Related Work
None identified through the review process. The paper appropriately cites relevant prior work on SE(3)-equivariant architectures, protein language models, and AlphaFold3.

## Suggestions
1. Add comparison to at least one recent structural protein Transformer (ESM-IF1 or ProSST) on the same GO function prediction task to contextualize performance against current methods.
2. Include error bars or confidence intervals for downstream task results, ideally with multiple random seeds.
3. Add an ablation on protein tasks testing the different embedding constructions (E_lin vs E_quad) to connect the theoretical analysis to the empirical protein results.

---

## 7Cx05z4pUc

- GT: Reject (avg 5.0)
- Predicted: Reject (4.0/10)
- Match: Yes

### Final Review

## Summary
This paper introduces "Decomposed Learning," a method that applies Singular Value Decomposition (SVD) to neural network weight matrices, training the resulting factors (U, Σ, V^T) independently rather than the original weight matrix directly. The authors demonstrate that this approach can reduce or eliminate grokking—the delayed generalization phenomenon—in modular arithmetic tasks, while also enabling parameter-efficient training.

## Strengths
- **Comprehensive experimental design**: The paper systematically explores decomposed learning across all major transformer components (token embedding, position embedding, multi-head attention, feed-forward blocks, output layer) with multiple rank configurations (12.5%, 25%, 50%, 75%, 100%) and dataset sizes (50%, 65%, 80%), providing a thorough characterization of when and how the method works.

- **Insightful spectral analysis**: Appendix D provides a valuable analysis of how normalized stable rank evolves during training, offering mechanistic insight into why decomposed learning works—the method accelerates the transition from high to low stable rank that correlates with generalization. This connects the empirical findings to implicit regularization theory.

- **Practical parameter efficiency**: The paper demonstrates (Table 1) that decomposed learning can achieve better or comparable performance with fewer parameters (e.g., rank 25 token embedding has ~50% fewer parameters than baseline but generalizes faster), with direct implications for efficient training and model compression.

- **Extension beyond synthetic tasks**: The authors include experiments on grokking-induced MNIST (Appendix F), Tiny Shakespeare (Appendix G.1), and CIFAR-10 (Appendix G.2), showing that the approach generalizes beyond the canonical modular arithmetic setting.

- **Clear finding on data-rank relationship**: The observation that "fewer ranks are needed as the dataset increases" (Section 6) is well-supported by evidence and provides practical guidance for practitioners.

## Weaknesses
- **No comparison to alternative grokking mitigation methods**: The paper only compares against a baseline normally-trained model. Liu et al. (2022) demonstrated that grokking can be avoided with hyperparameter tuning (weight decay, learning rate). Without comparing against these simpler approaches, it's impossible to assess the relative efficacy of decomposed learning.

- **Missing key ablation on random vs. SVD initialization**: The paper does not test whether randomly initializing U, Σ, V (with matching dimensions) produces similar benefits. Without this ablation, it remains unclear whether the SVD initialization specifically or simply the factorized training structure drives the improvements.

- **Limited theoretical justification**: The stable rank analysis is correlational rather than causal. The paper shows that stable rank decreases faster with decomposed learning, but doesn't prove this causes better generalization or explain why SVD initialization specifically induces this behavior.

- **Unexplained phenomenon at full-rank decomposition**: The result that 100% rank decomposition still helps (despite having more parameters than baseline) is puzzling and not adequately explained. If low-rank structure were the key mechanism, full-rank decomposition should be equivalent to baseline training.

- **Statistical rigor issues**: The paper states "mean of 5 runs is reported" but error bars and variance estimates are not visible in the figures, making it difficult to assess the significance of observed differences.

## Nice-to-Haves
- Testing on additional algorithmic tasks beyond modular division (e.g., modular addition/subtraction, polynomial evaluation) to establish broader applicability of the method to grokking-prone tasks.

- Gradient dynamics analysis comparing optimization trajectories between decomposed and baseline training to provide a causal explanation for faster generalization.

## Novel Insights
The paper reveals a striking relationship between data availability and optimal model capacity: as datasets become more representative of the problem space, fewer ranks are needed to mitigate grokking. The spectral analysis showing that decomposed learning accelerates the transition from high to low stable rank provides an empirical foothold for understanding implicit regularization in neural networks. The finding that the method works even at full rank (100%)—where parameter count actually increases—suggests the benefits arise from something other than mere capacity reduction, pointing to a fundamental difference in how gradient-based optimization traverses the decomposed parameterization.

## Potentially Missed Related Work
- Liu et al. (2022) "Towards Understanding Grokking" — This work shows grokking can be avoided through specific hyperparameter configurations. A comparison would help position the contribution.
- Nacson et al. (2022) on implicit bias and convergence in gradient descent — Relevant for understanding why factorized training might induce different implicit biases.

## Suggestions
- Add experiments comparing decomposed learning against weight decay tuning (the primary alternative grokking mitigation method) on the same modular division task to establish relative effectiveness.
- Include an ablation study with randomly initialized U, Σ, V factors (not from SVD) to isolate whether the SVD initialization or the factorized training structure drives the benefits.
- Add error bars or confidence intervals to the main figures to support statistical claims about performance differences.

---

## u1cQYxRI1H

- GT: Accept (Oral) (avg 10.0)
- Predicted: Accept (5.5/10)
- Match: Yes

### Final Review

## Summary
The paper proposes IC-Light, a method for scaling diffusion-based illumination editing by imposing a physically-grounded light transport consistency constraint during training. The core insight is that linear blending of appearances under different illuminations should match the appearance under mixed illumination (I_{L1+L2} = I_{L1} + I_{L2}), providing a principled regularization that enables stable training on >10M diverse images (light stage captures, 3D renderings, in-the-wild augmentations) using strong backbones (SD1.5, SDXL, Flux).

## Strengths
- **Physically grounded contribution**: The light transport consistency constraint (Eq. 3-5) is derived from physical light transport principles, providing principled regularization rather than ad-hoc losses. The derivation from HDR linearity to latent diffusion objectives is sound.
- **Impressive scale and diversity**: Training on >10M images from multiple sources with modern backbones (SDXL, Flux) demonstrates meaningful scalability. The data pipeline (Figure 2) unifies heterogeneous sources into a common format.
- **Comprehensive ablation study**: Figure 4 and Table 1 provide clear evidence that each component (light transport consistency, in-the-wild data, light stage data) contributes to performance. The LPIPS improvement from 0.1927 (w/o LTC) to 0.1025 (full method) is substantial.
- **Practical applications demonstrated**: Section 4.3 shows background-conditioned harmonization and normal map extraction from multiple consistent relighting inferences—useful downstream applications.
- **Strong visual results**: Figure 6 demonstrates competitive or superior shadow handling and detail preservation compared to SwitchLight and Relightful Harmonization.

## Weaknesses
- **Evaluation limited to synthetic 3D rendered data**: Table 1 evaluates only on 50K unseen 3D renderings. The method claims to work on "in-the-wild" images but provides no quantitative evidence on real photographs or held-out light stage data. This limits confidence in the claim that the method "modifies only illumination while keeping intrinsic properties unchanged" on real inputs.
- **No quantitative albedo preservation metric**: A core claim is preservation of intrinsic properties like albedo, yet no metric directly measures albedo consistency between input and output. The ablations in Figure 4 show qualitative color preservation, but without quantitative measurement, this claim remains partially unsubstantiated.
- **Learnable MLP φ lacks analysis**: The paper introduces a 5-layer MLP (Eq. 4) to handle domain adaptation between LDR/HDR/latent spaces but provides no ablation comparing against simpler alternatives or analysis of what the MLP learns. It is unclear whether this MLP preserves physical properties or introduces uncontrolled approximations.
- **Missing comparisons with recent diffusion-based relighting methods**: NeuralGaffer (Jin et al., 2024), LightIt (Kocsis et al., 2024), and FlashTex (Deng et al., 2024) are cited but not compared experimentally, despite being directly relevant diffusion-based approaches.
- **No statistical significance reported**: Results are single numbers without confidence intervals or error bars.
- **No user study for perceptual task**: Illumination editing quality is inherently subjective; PSNR/LPIPS on synthetic data do not fully capture perceptual quality or naturalness.
- **Hyperparameters not ablated**: The loss weights λ_vanilla=1.0 and λ_consistency=0.1 are stated without justification or sensitivity analysis.
- **No failure case analysis**: The paper lacks discussion of when the method fails (e.g., highly specular materials, subsurface scattering, saturated regions where linearity breaks down).

## Nice-to-Haves
- Scaling curve analysis showing performance vs. dataset size to justify the "10 million" threshold
- Discussion of broader impact given potential applications in image manipulation
- Clearer terminology: "degradation image" (I_d) is confusing; "alternative illumination" would be more precise

## Novel Insights
The key insight—applying light transport linearity as a training consistency constraint—is genuinely novel for diffusion-based illumination editing. The observation that physical linearity (I_{L1+L2} = I_{L1} + I_{L2}) can be converted to diffusion objectives through a learnable MLP enables stable large-scale training without requiring ground truth albedo supervision. The method effectively leverages the mathematical structure of light transport to constrain the learned distribution, rather than relying on paired data or intrinsic decomposition networks. This represents a principled bridge between physical priors and generative model training that could inform other inverse rendering applications.

## Potentially Missed Related Work
- NeuralGaffer (Jin et al., 2024) — Recent diffusion-based object relighting method directly comparable to this work
- LightIt (Kocsis et al., 2024) — Diffusion-based illumination control; cited but not compared
- IntrinsicDiffusion (Luo et al., 2024) — Joint intrinsic decomposition using diffusion; relevant for albedo preservation claims
- FlashTex (Deng et al., 2024) — Diffusion-based relightable texturing; another relevant diffusion relighting approach

## Suggestions
- Add quantitative evaluation on real-world test sets (held-out light stage data or established portrait relighting benchmarks with ground truth)
- Include a quantitative albedo preservation metric (e.g., albedo extraction error on 3D renderings where ground truth albedo is available)
- Provide ablation of the MLP φ against simpler alternatives (fixed linear combination, no MLP)
- Add failure case examples and analysis of where light transport linearity breaks down
- Conduct a small-scale user study (n=20-30) comparing perceptual quality against baselines

---

## 5XL8c0Vg9k

- GT: Reject (avg 2.0)
- Predicted: Reject (2.5/10)
- Match: Yes

### Final Review

## Summary

The paper proposes IP-LLM (Infinite Parameter Large Language Model), an architecture that decouples model size from inference memory by dividing parameters into a base component, a routing component, and domain-specific expert components. During inference, only the base model, router, and the single relevant expert are loaded into memory. The authors implement a 24B parameter model with 22 domain categories, requiring only 8.7B active parameters during inference, and report performance comparable to dense models on benchmarks like MMLU and GSM8K.

## Strengths

- **Addresses a practical deployment challenge**: Memory-efficient inference is a real problem for LLM deployment. The proposed on-demand loading mechanism genuinely reduces active memory requirements during inference, which is valuable for resource-constrained settings.

- **Clear architectural distinction from MoE**: Unlike MoE, which loads all experts into memory and routes per-layer, IP-LLM routes once per inference task and only loads the relevant expert. This is a meaningful architectural difference that enables scaling to more experts than memory would typically permit.

- **Principled approach to catastrophic forgetting**: By storing domain knowledge in separate expert parameters, new domains can be added without modifying existing experts, which naturally avoids catastrophic forgetting. This is a clean solution conceptually.

- **Working implementation demonstrated**: The authors built and evaluated a 24B parameter model across 22 domains, showing the approach is implementable rather than purely theoretical. The staged pretraining strategy (base model first, then domain-specific training) provides a sensible framework.

## Weaknesses

- **Over-claiming undermines credibility**: The title and abstract claim "infinite parameters" and "infinite knowledge," with language about "omniscient and omnipotent artificial general intelligence." These claims are hyperbolic—the model is modular and expandable, not infinite. Such unsupported claims distract from the actual contributions.

- **Routing accuracy claims unverified**: The paper repeatedly claims superior routing accuracy compared to MoE (Section 5.2 states "routing precision of our IP-LLM model is far superior to that of MOE"), yet provides no routing accuracy measurements whatsoever. This is a central claim that requires empirical validation.

- **Missing critical baseline comparisons**: No experimental comparison with actual MoE models (e.g., Mixtral) is provided despite extensive discussion of differences. Additionally, there's no comparison to a standard 8.7B dense model—the natural baseline since that's the active parameter count during inference.

- **Lifelong learning claims unsubstantiated**: The paper claims the architecture enables "lifelong learning without catastrophic forgetting," but no experiment demonstrates adding new experts and measuring retention of old knowledge. This claim remains theoretical without evidence.

- **Cross-domain queries unaddressed**: The architecture routes each input to exactly one expert, but real-world queries often span multiple domains or don't fit cleanly into predefined categories. This fundamental limitation is not discussed.

- **Insufficient experimental detail**: Figure 2 is referenced but no actual numerical results are presented clearly in the text. Standard deviations and reproducibility details are missing.

## Nice-to-Haves

- **Latency analysis for expert loading**: Dynamic expert loading during inference has practical latency costs that should be measured and discussed, especially for real-time applications.

## Novel Insights

The key insight of IP-LLM is shifting routing granularity from per-layer (MoE) to per-task, which fundamentally changes the memory-compute trade-off. This enables keeping only one expert in memory rather than all experts, making it possible to scale to expert counts that exceed memory capacity. The staged pretraining approach—learning foundational language knowledge first, then domain-specific knowledge in separate expert parameters—is a sensible curriculum that could improve training efficiency even beyond memory benefits. However, the paper's technical contributions are obscured by overblown claims.

## Potentially Missed Related Work

- **Expert Choice Routing (Zhou et al., 2022)**: Already cited in the paper, but worth deeper comparison as it also explores non-standard routing mechanisms.
- **LoRA and parameter-efficient fine-tuning methods**: These offer alternative approaches to adding knowledge without catastrophic forgetting, and comparison would strengthen motivation.

## Suggestions

1. **Remove hyperbolic claims**: Eliminate references to "infinite knowledge," "omniscient and omnipotent AGI," and similar statements. Focus on the genuine contribution—memory-efficient modular inference.

2. **Add routing accuracy evaluation**: Measure and report the router's classification accuracy, ideally with per-domain breakdowns to show which categories are more challenging.

3. **Include proper baselines**: Compare against (a) MoE models like Mixtral with matched parameters, and (b) a dense 8.7B model to validate that expert specialization provides benefit over simple scaling.

4. **Demonstrate lifelong learning empirically**: Show sequential domain addition experiments measuring performance retention on earlier domains.

5. **Address cross-domain handling**: Discuss how the model handles ambiguous queries or queries requiring multi-domain knowledge, and acknowledge this as a limitation if unaddressed.

6. **Provide complete numerical results**: Include a clear table with benchmark scores, standard deviations, and comparisons.

---

## oYSsbY3G4o

- GT: Accept (Poster) (avg 6.4)
- Predicted: Accept (6.5/10)
- Match: Yes

### Final Review

## Summary

This paper introduces GQT (Graph Quantized Tokenizer), a method that combines multi-task self-supervised learning with Residual Vector Quantization (RVQ) to learn discrete graph tokens. The approach decouples tokenizer pre-training from Transformer training, enabling standard Transformer encoders to process graphs efficiently while capturing both local and long-range dependencies. The method achieves state-of-the-art performance on 20 out of 22 benchmarks with significant memory reduction.

## Strengths

- **Comprehensive experimental evaluation across diverse graph settings.** The authors evaluate on 22 benchmarks spanning homophilic node classification (8 datasets), heterophilic node classification (6 datasets), large-scale graphs (up to 2.4M nodes), and long-range benchmarks from LRGB (4 datasets). This breadth demonstrates robustness across fundamentally different graph learning scenarios.

- **Well-designed ablation studies isolating component contributions.** Table 6 systematically evaluates each component (RVQ, GraphMAE2, DGI, codebook embeddings, positional encoding, structural gating, semantic edges). The ablations show that SSL objectives contribute significantly to performance, while Table 8 specifically demonstrates that semantic edges and structural gating benefit heterophilic settings.

- **Clear empirical demonstration of memory efficiency.** The paper provides concrete examples: 23-fold memory reduction on Physics (34,493 nodes) and 30-fold reduction on ogbn-products (2.4M nodes). Table 10 shows approximately 50% GPU memory reduction compared to GAT during inference, supporting the practical utility claims.

- **Novel integration of techniques for graph tokenization.** The combination of multi-task graph SSL objectives (DGI + GraphMAE2) with RVQ for learning discrete graph tokens is novel and addresses a genuine gap where graph tokenization has lagged behind vision and language modalities.

## Weaknesses

- **VQGraph (Yang et al., 2024) is cited but not experimentally compared.** VQGraph is directly relevant—it uses VQ for graph learning and is mentioned in the Related Work—yet it is absent from all experimental comparisons. Given its methodological proximity, this omission weakens the empirical claims.

- **No analysis of failure cases.** GQT does not achieve best performance on 2 of 22 benchmarks (Chameleon and PCQM-Contact). The paper provides no discussion of why the approach underperforms in these specific settings, limiting readers' understanding of method limitations.

- **Commitment loss weight β not reported.** The loss function includes β·L_commit in Equation 5, but β values are absent from Table 7's hyperparameters. This missing detail affects reproducibility.

- **Training efficiency not analyzed.** Table 10 compares inference time and memory, but the paper provides no analysis of total training time (tokenizer pre-training plus Transformer training) relative to end-to-end baselines. This omission matters because the two-stage approach adds computational overhead.

- **No limitations section.** Current standards expect explicit discussion of limitations, which this paper lacks. Unacknowledged limitations include: what semantic information is preserved or lost during quantization, how the approach would handle dynamic graphs, and the practical implications of extensive dataset-specific hyperparameter tuning (Table 7).

## Nice-to-Haves

- **Codebook utilization analysis.** Vector quantization methods commonly suffer from codebook collapse where entries are unused. The paper would benefit from analyzing actual utilization rates across codebooks.

- **Hyperparameter sensitivity analysis.** Table 7 shows different hyperparameters per dataset (GNN layers: 2–6; codebook sizes: 128–4096; PPR neighbors: 0–50). A sensitivity analysis showing performance variation with key hyperparameters would guide practitioners.

## Novel Insights

The decoupling of tokenizer training from Transformer training reflects an important architectural insight: by training the GNN encoder with graph-specific SSL objectives rather than reconstruction alone, the learned tokens encode structural information that a vanilla Transformer can efficiently process. This design choice effectively separates "local structure understanding" (handled by the GNN tokenizer) from "global dependency modeling" (handled by the Transformer), analogous to how vision transformers use patch embedding as a preprocessing step rather than learning spatial relationships from scratch.

## Potentially Missed Related Work

- **VQGraph (Yang et al., 2024)** — Directly relevant method using vector quantization for graphs; cited in Related Work but not experimentally compared.

## Suggestions

1. Add experimental comparison with VQGraph to strengthen empirical claims.
2. Discuss failure cases on Chameleon and PCQM-Contact to help readers understand limitations.
3. Report the commitment loss weight β values used across experiments.
4. Add training time comparisons to complement the inference efficiency analysis.
5. Include a limitations section addressing quantization information loss and hyperparameter tuning requirements.

---

## OZVTqoli2N

- GT: Accept (Spotlight) (avg 7.5)
- Predicted: Accept (6.5/10)
- Match: Yes

### Final Review

## Summary
This paper investigates model compositionality through a second-order Taylor approximation of the loss function around pre-trained weights. The authors derive a theoretical bound relating the empirical risk of a composed model (weight-averaged) to its individual components, revealing that compositional success requires individual models to maintain out-of-distribution performance. Based on this analysis, two algorithms are proposed—Incremental Task Arithmetic (ITA) for individual training and Incremental Ensemble Learning (IEL) for joint optimization—with empirical validation across seven class-incremental benchmarks.

## Strengths
- **Principled theoretical foundation**: The derivation of Eq. 5 via Jensen's inequality applied to the second-order loss approximation provides a meaningful theoretical bound, and the insight that compositionality requires staying within the pre-training basin (to maintain O(‖θ - θ₀‖³) validity) is a genuine contribution beyond prior empirical observations.

- **Strong empirical performance**: ITA/IEL achieve competitive or superior final accuracy across all seven benchmarks (e.g., ITA-IA³: 90.66% on CIFAR-100 vs. 87.17% InfLoRA; ITA-LoRA: 82.00% on RESISC vs. 79.92% InfLoRA), including challenging out-of-distribution domains (satellite imagery, plant diseases) where pre-training optimality assumptions are stressed.

- **Efficient algorithm design**: Both ITA and IEL maintain O(1) time complexity during training and inference through cumulative weight averaging (Appendix D), with closed-form gradient computations (Appendix C). The paper correctly qualifies that O(T) storage is needed only when preservation of individual task vectors is required for unlearning/specialization.

- **Comprehensive experimental framework**: The paper evaluates three fine-tuning strategies (Full, LoRA, IA³), includes regularization-based (EWC), rehearsal-based (DER++), prompt-based (L2P, CODA), and compositionality-focused (TMC, SEED, APT, InfLoRA) baselines, and provides ablations (Tables 2, 6), compositional skill analysis (specialization/unlearning), and timing comparisons.

## Weaknesses
- **Local minimum assumption lacks empirical validation**: The theoretical bound (Eq. 5) assumes θ₀ is a local minimum of the empirical risk across all tasks, yet the paper does not empirically verify Hessian definiteness at θ₀. While linear probing is proposed to approximate this condition, there's no demonstration that the resulting θ₀ satisfies the convexity requirements needed for the bound to hold.

- **Approximation quality not validated during training**: The second-order Taylor approximation becomes inaccurate as ‖θ - θ₀‖ grows. The paper acknowledges this limitation but provides no empirical measurement of ℓ_cur(θ) vs. ℓ(θ) during fine-tuning to validate whether task vectors remain within the basin where the approximation holds.

- **IEL's specialization failure unexplained**: Table 3 shows IEL degrades during specialization (e.g., -13.22% on CIFAR-100) while ITA improves (+1.63%). Since both methods derive from the same theoretical framework, this asymmetry demands investigation—it suggests the ensemble training objective may create undesirable task vector interdependencies.

- **Hyperparameter sensitivity not analyzed**: Appendix I reveals substantial hyperparameter complexity (separate α/β values for backbone and classification head, pre-consolidation learning rates and epochs, dataset-specific tuning). The paper lacks sensitivity analysis showing whether performance degrades sharply with suboptimal settings or remains robust.

## Nice-to-Haves
- **Ablation of pre-consolidation step**: The linear probing + classifier alignment step (Section 3) appears critical for enforcing local optimality, yet its isolated contribution is not measured. An ablation removing this step would clarify its necessity.

- **Task order sensitivity analysis**: Class-incremental performance can vary with task ordering; testing with shuffled task orders would establish robustness.

- **Comparison of diagonal vs. full Fisher**: Using diagonal FIM for tractability is reasonable, but comparing against full FIM or K-FAC on at least one dataset would validate this approximation choice.

## Novel Insights
The paper's central insight—that successful weight averaging requires individual models to maintain reasonable performance on examples outside their training distribution—reframes compositionality as a continual learning problem. This connection is not obvious from prior work on task arithmetic, which primarily observed empirical phenomena. The derivation that compositionality degrades as task vectors leave the pre-training basin (where the second-order approximation breaks down) provides a theoretical lens for understanding why low learning rates and regularization (observed empirically in prior work) improve weight averaging. The dual perspective—individual training (ITA) prioritizes flexibility for downstream editing, while ensemble training (IEL) prioritizes joint performance but sacrifices modularity—offers practitioners a principled choice between methods.

## Potentially Missed Related Work
- **TIES-merging (Yadav et al., 2024b) and ZipIt! (Stoica et al., 2024)**: These recent model merging methods directly address interference in weight composition and are highly relevant to the compositionality discussion. While mentioned in Section 4, empirical comparison would contextualize ITA/IEL against state-of-the-art post-hoc merging techniques.

## Suggestions
- **Validate the approximation**: Add an experiment measuring the correlation between ℓ_cur(θ) and ℓ(θ) during fine-tuning to empirically verify whether task vectors remain within the valid approximation region.

- **Analyze IEL's modularity loss**: Investigate why ensemble-trained task vectors become interdependent by examining task vector alignment patterns (cosine similarity) between ITA and IEL—preliminary analysis in Figure 2 shows positive alignment but could be extended to explain specialization behavior.

- **Add pre-consolidation ablation**: Report results for ITA without the linear probing step to isolate the contribution of enforcing local optimality assumptions.

---

## nwDRD4AMoN

- GT: Accept (Oral) (avg 9.0)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
This paper introduces Artificial Kuramoto Oscillatory Neurons (AKOrN), a novel neural network architecture where each neuron is an N-dimensional unit vector evolving on a hypersphere according to a generalized Kuramoto dynamics. The synchronization behavior naturally implements feature binding, enabling strong performance on unsupervised object discovery (first synchrony-based model competitive with slot-based methods on CLEVRTex), Sudoku puzzle solving, adversarial robustness without adversarial training (51-59% accuracy under AutoAttack on CIFAR-10), and well-calibrated uncertainty estimates (ECE 1.3-1.4%).

## Strengths
- **Conceptual novelty**: Proposes a fundamentally different computational primitive—oscillatory neurons with Kuramoto dynamics—grounded in both physics (Kuramoto model) and neuroscience (neural synchronization/binding), rather than incremental modifications to existing architectures.
- **Strong empirical results across diverse tasks**: Achieves competitive object discovery on CLEVRTex (FG-ARI 88.5), 100% accuracy on in-distribution Sudoku with 89.5% on OOD (outperforming prior methods including IRED), and adversarial robustness without adversarial training—a notable result for models trained only on clean data.
- **Test-time computation scaling**: The elegant property that increasing Kuramoto steps at test time improves OOD generalization (17%→52% on Sudoku OOD, Figure 6c) provides a natural mechanism for trading computation for accuracy.
- **Built-in uncertainty quantification**: Energy-based voting exploits the Lyapunov energy landscape to select correct solutions, achieving near-perfect correlation between confidence and accuracy on corrupted inputs.
- **Comprehensive ablations**: Systematic studies on the projection operator (Table 5), rotating dimensions N (Figures 14-17), bias term C and norm-taking term m (Figure 18), and symmetric vs asymmetric connections (Figure 29, Table 19).

## Weaknesses
- **Computational overhead not thoroughly addressed**: While Figure 19 shows training/inference times (2-10x slower than baselines), the paper lacks systematic FLOPs and parameter count analysis. The T=8-128 internal steps per layer significantly increases compute, and the energy-based voting for Sudoku requires 100-4096 samples—this overhead should be quantified and discussed as a design trade-off.

- **Theoretical-empirical disconnect**: The Lyapunov proof (Appendix F) requires symmetric J, but experiments use learned asymmetric J and Ω for better performance. The paper states energy "decreases relatively stably" (p. 3) empirically, but provides no formal analysis of what dynamics emerge or why energy still correlates with correctness when symmetry is broken.

- **N=2 vs N=4 inconsistency across experiments**: Object discovery and Sudoku experiments use N=4 rotating dimensions (Tables 7, 16), but robustness experiments use N=2 (Table 20). N=2 underfits on object discovery yet provides better robustness calibration. This trade-off is noted but not systematically analyzed across all task categories.

- **Clean accuracy trade-off for robustness**: Table 4 shows AKOrN_conv achieves 88.91% clean accuracy vs ResNet-18's 94.41%—a ~5% drop. This inherent trade-off between robustness and clean accuracy should be explicitly discussed in the main text rather than only appearing in tables.

- **Training pipeline confounds robustness claims**: AKOrN robustness experiments use Tiny-ImageNet SimCLR pretraining with AugMix, while the ItrConv baseline in Table 20 shows standard training. Isolating whether robustness gains come from the Kuramoto mechanism versus the training pipeline requires a controlled comparison.

## Nice-to-Haves
- Larger-scale standalone evaluation (e.g., ImageNet classification) to demonstrate broader applicability beyond specialized tasks.
- Visualization of actual oscillator phase assignments during object discovery to verify binding-by-synchronization is occurring, rather than just showing final cluster assignments.
- Comparison with adversarial training methods at equivalent compute budget to better contextualize the "robust by design" claim.

## Novel Insights
The paper reveals a remarkable phenomenon: the energy landscape defined by the Kuramoto model provides a principled mechanism for uncertainty quantification and solution selection without any generative training. The energy-based voting for Sudoku (selecting lowest-energy predictions from multiple random initializations) exploits structure that emerges purely from the dynamics, not from explicit likelihood training. This suggests that energy-based architectures may inherit calibration properties similar to generative classifiers while avoiding their computational overhead. Additionally, the test-time step extension—where harder problems benefit from more Kuramoto steps—provides a natural form of adaptive computation reminiscent of human reasoning taking longer on difficult tasks.

## Potentially Missed Related Work
- None identified (related work search was not performed for this submission).

## Suggestions
- Add FLOPs analysis and parameter counts to enable fair computational comparison across methods.
- Explicitly discuss the clean accuracy vs robustness trade-off in the main text.
- Provide systematic N comparison across all three task categories (object discovery, reasoning, robustness) to guide practitioners in selecting this hyperparameter.
- Compare AKOrN robustness against adversarially trained baselines with matched compute budgets to isolate the architectural contribution.

---

## sZQRUrvLn4

- GT: Accept (Spotlight) (avg 6.4)
- Predicted: Accept (6.5/10)
- Match: Yes

### Final Review

## Summary
This paper provides a beyond-worst-case analysis of when message-passing GNNs can count subgraphs, deriving sufficient conditions ((ℓ,k)-identifiability and quite-colorfulness) that explain GNNs' empirical success on subgraph counting despite known theoretical limitations. The authors propose a novel dynamic programming algorithm (TREE-COLSI) that GNNs can simulate, prove bounds on parameter complexity and sample efficiency, and empirically validate their conditions on molecular graph datasets.

## Strengths
- **Novel theoretical framework**: The (ℓ,k)-identifiability condition (Definition 2) and corresponding Theorems 2-3 provide sample-efficient bounds independent of graph size for local functions, advancing understanding beyond the worst-case impossibility results of Chen et al. (2020).
- **Algorithmic contribution**: The TREE-COLSI algorithm adapts color-coding to use WL colors, enabling GNN simulation with provable guarantees for detecting quite-colorful subgraph isomorphisms from tree patterns (Theorems 4-5).
- **Strong empirical validation of theoretical claims**: Tables 2-3 demonstrate that (ℓ,k)-identifiability holds for ≥97% of nodes with ℓ=k+2 layers on molecular datasets. Figure 4 shows most subgraph isomorphisms become quite-colorful with few WL iterations.
- **Synthetic validation**: Table 7 correctly isolates theoretical predictions, showing near-perfect GNN performance when sufficient conditions (parent-colorfulness, quite-colorfulness) are satisfied.

## Weaknesses
- **Missing baseline comparisons**: The paper claims "more expressivity is almost never needed" (Section 7) but provides no comparison to architectures designed for subgraph counting (I2-GNN, subgraph GNNs, k-GNN). Without this, the practical implications are unsubstantiated.
- **Limited experimental scope**: All experiments use molecular graphs, which have favorable properties (bounded degree, diverse node labels, short cycles). Validation on other domains (social networks, biological networks) where the sufficient conditions may not hold is absent.
- **Exponential worst-case complexity**: Lemma 5 shows ζ_{l,T,G} can grow exponentially in pattern size and WL color count, though the paper notes bounds are tighter "in practice" without empirical analysis of actual parameter counts on real datasets.

## Nice-to-Haves
- **Ablation on GNN depth**: The theory specifies that l+h layers are needed to simulate TREE-COLSI. Empirically validating this correspondence by varying GNN layers against pattern height would strengthen the algorithmic alignment claim.
- **Analysis of failure cases**: Figure 4 shows some patterns have non-quite-colorful maps—the paper should analyze how counting accuracy correlates with the quite-colorfulness ratio to quantify when theoretical guarantees translate to empirical performance.

## Novel Insights
The key insight is that worst-case analysis of GNN expressivity misses practically important cases. The paper successfully shows that molecular graphs are highly WL-distinguishable (Table 2), enabling GNNs to learn local functions sample-efficiently. The concept of "quite-colorfulness" provides an algorithmic bridge between color-coding algorithms and GNN message passing, showing that real graphs often have sufficient asymmetry (via WL colors) to make subgraph maps injective in the sense required by the DP algorithm.

## Potentially Missed Related Work
- None identified (related work search was skipped).

## Suggestions
- Add comparison experiments against subgraph GNNs and I2-GNN architectures on the same counting tasks to substantiate the claim that standard GNNs often suffice.
- Include experiments on at least one non-molecular dataset to assess whether the sufficient conditions hold more broadly and delineate where they fail.
- Provide empirical analysis of the ζ_{l,T,G} parameter (number of distinct DP states) on real datasets to show practical complexity bounds beyond worst-case analysis.

---

## SOd07Qxkw4

- GT: Accept (Spotlight) (avg 7.5)
- Predicted: Reject (5.0/10)
- Match: No

### Final Review

## Summary
This paper establishes an improved iteration complexity of $\tilde{O}(L d^{1/3} \varepsilon^{-2/3})$ for diffusion probabilistic models under Lipschitz score function assumptions, improving upon the prior best bound of $\tilde{O}(L^{5/3} d^{5/12} \varepsilon^{-1})$. The approach adapts the randomized midpoint method to the probability flow ODE framework and also provides a parallelized sampling variant with $O(\log^2(Ld/\varepsilon))$ parallel rounds.

## Strengths
- **Meaningful theoretical improvement**: The paper achieves a provably better iteration complexity than prior work (Gupta et al. 2024, Chen et al. 2024b), improving by a factor of $\tilde{O}(L^{2/3} d^{1/12} \varepsilon^{-1/3})$ under the smooth score assumption.
- **Comprehensive positioning**: The paper provides thorough comparisons with state-of-the-art results (Benton et al. 2023, Li et al. 2024c, Gupta et al. 2024, Chen et al. 2024b) and clearly identifies the parameter regimes where their bounds are advantageous.
- **Novel analysis framework**: The key technical insight—transferring the sampling task to discretizing a probability flow ODE with injected noise, then applying randomized midpoint techniques—provides a cleaner reduction that may be useful for analyzing other samplers.
- **Parallel sampling contribution**: The extension to parallel sampling with reduced processor requirements ($\tilde{O}(d^{1/3}\varepsilon^{-2/3})$ processors) compared to prior work adds practical relevance.

## Weaknesses
- **Proofs are not self-contained**: All major lemmas (Lemmas 1–5) defer proofs to "Li & Jiao (2024)(Appendix X.x)"—apparently the authors' own arXiv preprint. For a submission claiming a significant theoretical contribution, readers cannot verify correctness without consulting this external reference. Key proof details should either be included or sketched with sufficient depth for expert verification.

- **Lipschitz assumption lacks practical grounding**: Assumption 2 (Lipschitz continuity of score functions) is central to the improved rate, but the paper provides no discussion of whether trained neural network score estimators satisfy this in practice, nor any bounds on typical values of L for real diffusion models. The regime where the bounds improve on non-smooth results (e.g., $L < d^{2/3}\varepsilon^{-4/3}$) may or may not hold in practice—this remains unexplored.

- **Hidden constants in $\tilde{O}$ notation**: The improvement factor of $L^{2/3} d^{1/12} \varepsilon^{-1/3}$ could be dominated by hidden constants and logarithmic factors in practical regimes. The paper does not discuss the tightness of the analysis or whether the constants matter for realistic values of $d$ and $\varepsilon$.

## Nice-to-Haves
- Empirical validation, even on synthetic data (e.g., Gaussian mixtures), to verify the $d^{1/3}\varepsilon^{-2/3}$ scaling holds in practice and that the algorithm produces correct samples.
- Discussion or bounds on Lipschitz constants for standard diffusion model architectures to assess whether the improvement regime is achievable.

## Novel Insights
The paper's key technical innovation lies in its analysis framework: by injecting stochastic noise into the probability flow ODE, the sampling problem reduces to discretizing an ODE with controlled error. Combined with randomized midpoints (adapted from Shen & Lee 2019), this enables tighter discretization error bounds than prior approaches that relied on stepwise error propagation. The insight that smoothness enables uniform error control across multiple steps—rather than dimension-dependent high-probability bounds—appears fundamental to achieving sub-linear $d$-dependence.

## Potentially Missed Related Work
- None identified in this review cycle.

## Suggestions
- Include proof sketches for Lemmas 1–5 within the main paper or appendix, with sufficient technical detail to allow expert verification without consulting external references.
- Add a discussion of whether standard diffusion model architectures (U-Nets, transformers with common regularizations) satisfy the Lipschitz condition in practice, or provide empirical estimates of L on common models.
- Consider adding even a simple synthetic experiment (e.g., 2D Gaussian mixture) to demonstrate that the proposed sampler works correctly and to provide intuition about the scaling behavior.

---

## MGceYYNvXp

- GT: Reject (avg 1.5)
- Predicted: Reject (2.0/10)
- Match: Yes

### Final Review

## Summary
The paper proposes "Project MPG," a framework for aggregating LLM benchmark scores into two interpretable metrics: a "Goodness" score (answer accuracy across selected benchmarks) and a "Performance" score (queries per second). The authors use hierarchical Bayesian aggregation with beta distributions to combine benchmarks across three subdomains (Factual Recall, Social Sensitivity, and Problem Solving), evaluating 13 models and reporting correlation with LMSys Chatbot Arena rankings.

## Strengths
- **Clear practical motivation**: The paper targets resource-constrained developers who lack access to expensive Elo-based evaluation systems, addressing a legitimate need for efficient model comparison.
- **Transparent methodology**: The Bayesian aggregation approach is described in detail with pseudocode (Section 3.3), making the method reproducible.
- **Encouraging correlation results**: The raw Pearson correlation between MPG scores and LMSys ratings (r=0.9157) suggests the aggregated metric captures meaningful signal about model quality, though statistical significance of improvement over MMLU is not established.
- **Principled benchmark selection**: Benchmarks were selected based on cross-correlation analysis from Ilić & Gignac (2024), selecting representatives from distinct clusters to minimize redundancy while maintaining coverage.

## Weaknesses
- **No comparison to simple aggregation baselines**: The paper introduces a Bayesian MCMC aggregation method but never demonstrates that it outperforms simpler alternatives (weighted averaging, arithmetic mean, or PCA-based aggregation). Without this baseline comparison, the methodological complexity is unjustified.
- **Statistical claims are fragile**: With only 13 models, the paper cannot establish statistically significant improvement in correlation with LMSys. The raw correlation favors MPG (0.9157 vs 0.7721), but the *rank* correlation actually favors MMLU (0.7182 vs 0.6868)—a tradeoff the paper does not acknowledge or explain.
- **Relationship to HELM is inadequately addressed**: The introduction claims "no such aggregation schema exists that is not Elo based," but HELM (cited in the paper) provides multi-benchmark aggregation with summary metrics. The paper does not explain how MPG differs from or improves upon HELM.
- **Near-perfect social sensitivity scores warrant scrutiny**: Table 2 shows many models achieving 1.00 on disambiguous social sensitivity questions, suggesting potential ceiling effects or benchmark triviality. The paper mentions contamination concerns but does not systematically investigate.
- **Naming inconsistency undermines clarity**: The title uses "MPG," the abstract introduces "Goodness" and "Fastness," but the conclusion suddenly refers to "IQ, a benchmarking framework." This creates confusion about what the framework is actually called.

## Nice-to-Haves
- **Controlled QPS measurement**: The "Performance" metric conflates model speed with API latency and infrastructure. A controlled throughput measurement (tokens/second on standardized hardware) would strengthen the practical utility.
- **Ablation on benchmark selection**: Testing whether the principled selection based on correlation analysis outperforms random selection would validate the design rationale.

## Novel Insights
The paper's most interesting empirical finding is the divergence between raw correlation and rank correlation: MPG achieves higher raw correlation with LMSys than MMLU, but lower rank correlation. This suggests MPG may preserve absolute score relationships while potentially scrambling ordinal rankings—an important caveat for users who need reliable rankings. The finding that proprietary models cluster clearly above open-source models on the Goodness-QPS Pareto frontier (Figure 1) provides useful visual summary of the current model landscape, though this insight depends heavily on the validity of the aggregated metric.

## Potentially Missed Related Work
- None identified in this review cycle.

## Suggestions
- Add comparison to simple aggregation baselines (arithmetic mean, weighted mean) to justify the Bayesian MCMC approach.
- Report confidence intervals on correlations and conduct significance tests for correlation differences given the small sample size (n=13).
- Explicitly acknowledge and discuss the rank correlation tradeoff (MPG underperforms MMLU on rank correlation while outperforming on raw correlation).
- Resolve naming inconsistency—use consistent terminology throughout (recommend "MPG" or "Goodness," not "IQ").

---

## 10kBEqYKKN

- GT: Reject (avg 3.0)
- Predicted: Reject (3.5/10)
- Match: Yes

### Final Review

## Summary
This paper investigates how zero-shot prompts influence latent representations in autoregressive LLMs by analyzing geometric properties (isotropy via IsoScore) and clustering patterns of EOS token representations across layers. Experiments across four model families (Bloomz, Gemma, Phi, Zephyr) on three sentiment classification datasets reveal that prompts affect latent space distribution in model-dependent ways, and that semantically similar prompts can yield surprisingly different geometric representations.

## Strengths
- **Novel research direction**: The paper addresses an understudied question—how prompts affect internal representations rather than just task performance—providing mechanistic insight into prompt engineering that goes beyond output-based evaluation.
- **Strong model coverage**: The authors evaluate 10 models across 4 families with varying sizes (560M-7B parameters), demonstrating that findings generalize across architectures while revealing model-dependent patterns.
- **Clear methodological framework**: The use of IsoScore for measuring dimensionality utilization and KMeans clustering with Random Index Score for prompt grouping provides reproducible, quantitative metrics for analyzing representations.
- **Counter-intuitive empirical findings**: The discovery that semantically similar prompts (e.g., "Movie Expressed Sentiment" variants) cluster together only ~6.7% of the time while different prompt types cluster more frequently (~20%) is genuinely surprising and challenges intuitive assumptions about prompt similarity.
- **Layer-wise analysis**: Analyzing representations across all layers reveals that isotropy generally increases through layers, with interesting model-dependent patterns that warrant further investigation.

## Weaknesses
- **Limited task scope**: All experiments use binary sentiment classification (Rotten Tomatoes, IMDB, Yelp), making it unclear whether findings generalize to other task types (NLI, QA, summarization) where prompt engineering may be even more critical.
- **Weak quantitative link to performance**: The paper hypothesizes a relationship between prompt quality and isotropy (HP1), but results show "no apparent monotonic correlation." The claim that "bad prompts tend to destabilize internal representations" is supported only by qualitative visual inspection (Figure 2) without correlation coefficients or statistical tests.
- **EOS-only analysis under-justified**: The paper extracts only the EOS token representation without comparing to alternatives (mean pooling, attention-weighted pooling) or justifying why this single position captures all relevant prompt effects.
- **Unexplained counter-intuitive clustering**: The key finding that semantic similarity doesn't predict clustering is reported but not investigated—what features DO drive the clustering remains unexplored.
- **Model-dependent differences unexplained**: Results vary significantly across model families (Bloomz shows smoother isotropy evolution; Gemma and Phi show different patterns), but the paper offers no investigation into whether this stems from architecture, training data, or instruction-tuning procedures.
- **Drafting error**: The first two paragraphs of the Introduction contain nearly identical content about parameter scaling and transformer architecture—a significant editorial oversight that should be corrected.

## Nice-to-Haves
- **Visualization of embedding space**: t-SNE or UMAP projections of EOS representations would provide intuitive visualization of whether prompts cluster and how those clusters relate to semantic similarity or performance.
- **Probing experiments**: Linear probes on EOS representations for syntactic/semantic properties would help explain what information the geometric differences encode.
- **Diverse tasks**: Adding experiments beyond sentiment classification would strengthen generalizability claims.

## Novel Insights
The most striking finding is the disconnect between semantic prompt similarity and geometric similarity: prompts that are nearly identical in natural language (e.g., "Movie Expressed Sentiment" vs. "Movie Expressed Sentiment 2") are rarely grouped together by the clustering algorithm, while more distant prompts cluster frequently. This suggests that LLMs leverage features beyond surface semantics when processing prompts—possibly syntactic patterns, token-level features, or learned associations from training. The model-family-dependent isotropy evolution patterns (Bloomz showing smooth increases, others showing more irregular patterns) hint that pre-training data and instruction-tuning procedures leave detectable traces on how models process inputs. These observations collectively suggest that prompt engineering research should consider internal representations, not just output quality.

## Potentially Missed Related Work
- None identified in this review cycle.

## Suggestions
- **Add statistical significance tests**: Report confidence intervals and p-values for IsoScore differences and clustering effectiveness to distinguish real patterns from noise.
- **Quantify the performance-isotropy relationship**: Compute correlation coefficients between IsoScore values and task accuracy, with significance tests, to replace qualitative visual claims with quantitative evidence.
- **Investigate clustering features**: Apply probing or feature attribution to understand what properties distinguish prompts that cluster together from those that don't—this would transform a descriptive finding into an explanatory one.
- **Include non-sentiment tasks**: Even 1-2 additional tasks (e.g., NLI, topic classification) would substantially strengthen claims about general prompt effects.
- **Clarify model-dependent patterns**: Compare models with controlled differences (same architecture, different training) to isolate whether isotropy patterns stem from architecture or training procedures.

---

## VB8xHF1Rdl

- GT: Reject (avg 3.5)
- Predicted: Reject (4.5/10)
- Match: Yes

### Final Review

## Summary
This paper introduces an information-theoretic framework for quantifying structure in large language model representations, proposing three measures (regularity, variation, disentanglement) derived from linguistic concepts. The authors present "soft entropy," a computationally efficient method for estimating entropy in continuous vector spaces, and apply it to analyze training dynamics in BERT, effects of model scaling, and correlations between pre-training representational structure and downstream GLUE task performance.

## Strengths
- **Novel and scalable entropy estimation**: The soft entropy method is differentiable, requires no clustering, and uses only dot products, softmax, and summation—enabling analysis of models from 14M to 12B parameters where existing discretization methods struggle with memory and compute (Section 4.2).
- **Subspace entropy enables fair model comparison**: By breaking representations into fixed-size subspaces (e.g., 32-dimensional), the method allows like-for-like comparison across models with different hidden sizes, addressing a key challenge in comparing representations (Section 5.1).
- **Strong experimental design for downstream prediction**: Using MultiBERTs (25 different initializations of identical BERT architecture) isolates representational structure effects from confounds like model size, training data, and objectives (Section 5.4).
- **Meaningful linguistic framing**: Connecting information-theoretic measures to linguistic concepts (regularity as one-to-one mapping, variation as one-to-many, disentanglement as cluster separability) provides interpretable vocabulary grounded in prior work (Section 3, Figure 1).
- **Interesting empirical findings**: The analysis reveals that larger models become proportionally more disentangled for contextual information (bigrams, trigrams) while using less space for token information, and that pre-training structure correlates with downstream performance on several GLUE tasks (Figures 3-4, Table 1).

## Weaknesses
- **Limited theoretical grounding for soft entropy**: The estimator is motivated intuitively and benchmarked against k-means clustering in Appendix A.1, but lacks formal analysis of its statistical properties (bias, variance, convergence). The connection to kernel density estimation is mentioned but not developed, leaving unclear why softmax over cosine similarities with sampled points yields a principled entropy estimator.
- **Hyperparameter choices unjustified and untested**: The method uses 50 sampled points and a scaling factor of -100 to 100 without systematic justification or sensitivity analysis. It is unclear whether results are robust to these choices across different model sizes or representation distributions.
- **No validation against existing probing methods**: The paper claims to offer a "non-parametric approach to probing" but does not compare what information these measures capture against probing classifiers or other representation analysis methods. Without this comparison, it is unclear whether the measures capture similar linguistic information or something fundamentally different.
- **Inconsistent correlation patterns unexplained**: Table 1 shows positive correlations with some GLUE tasks (QNLI, QQP), negative correlations with others (CoLA), and non-significant correlations elsewhere. The paper does not investigate why the same measure has opposite relationships with different tasks, undermining interpretability of what these measures capture.
- **Correlation presented as prediction**: The paper claims to "predict" downstream performance but only shows correlations. No framework is provided for using these measures to make actual predictions on held-out models or tasks, and the effect sizes are modest (e.g., r=0.17-0.27 for significant correlations).
- **Subspace decomposition assumption tested only on synthetic data**: Breaking representations into independent subspaces assumes cross-dimensional dependencies are negligible, but Appendix A.2 validates this only on synthetic normal distributions, not on actual model representations where dependencies may be meaningful.

## Nice-to-Haves
- Qualitative examples showing what high vs. low regularity or variation looks like for specific tokens (e.g., "the" vs. "bank") would ground abstract measures in interpretable cases
- Investigation of what the "unexplained residual" information represents in larger models—whether it is noise, task-relevant information not captured by n-grams, or artifacts
- Multiple comparison correction for the correlation analyses given the many tests performed across tasks and measures

## Novel Insights
The key insight is that representational structure measurable at the end of pre-training correlates with downstream task performance after fine-tuning, suggesting structure matters for generalization. The training dynamics analysis shows early training is characterized by rapid alignment of representations with token information, followed by a longer contextualization phase where token disentanglement drops while bigram and trigram disentanglement increases. The subspace entropy analysis reveals that larger models compress their representations proportionally more, potentially explaining improved performance through more efficient information encoding analogous to Shannon's source coding model.

## Potentially Missed Related Work
None identified—the related work search was not performed.

## Suggestions
- Provide formal analysis of the soft entropy estimator's properties (bias, variance, relationship to existing entropy estimation methods) or at minimum, systematic sensitivity analysis for hyperparameters
- Compare information captured by these measures against standard probing classifiers to validate the claim of providing an alternative approach
- Investigate and explain the inconsistent correlation signs across tasks, which would strengthen interpretability of what these measures capture
- Temper claims of "predicting" downstream performance to "correlating with" unless a predictive framework with held-out validation is provided

---

## a8wjeqTZ9C

- GT: Reject (avg 3.8)
- Predicted: Reject (4.5/10)
- Match: Yes

### Final Review

## Summary
This paper presents the first systematic study of label noise in Concept Bottleneck Models (CBMs), demonstrating that concept annotation errors significantly degrade both model performance and interpretability. The authors identify concept noise (rather than target noise) as the primary failure source, analyze the mechanisms through t-SNE visualizations and concept importance analysis, and propose Sharpness-Aware Minimization (SAM) as a mitigation technique.

## Strengths
- **Novel problem identification**: The paper correctly identifies an important gap—CBMs require concept annotations which are inherently noisy in practice, yet this issue has been largely overlooked in prior CBM literature. This is a timely and practically relevant contribution.
- **Comprehensive empirical analysis**: The experiments span multiple dimensions: two datasets (CUB, AwA2), three training strategies (Ind, Seq, Joi), three architectures (InceptionV3, ResNet-18, ViT-B/16), multiple noise types (symmetric, pairwise), and noise locations (concept, target, combined). This thoroughness strengthens the empirical conclusions.
- **Insightful diagnostic analysis**: The t-SNE visualizations (Figure 4) effectively show representation collapse under concept noise but not target noise, and the analysis of concept importance shifts (Figure 5) reveals how noise distorts learned concept-target relationships. These provide meaningful mechanistic understanding.
- **Clear practical implications**: The finding that concept noise is more detrimental than target noise has direct implications for data collection prioritization in CBM deployments—practitioners should invest more in accurate concept labeling than target labeling.

## Weaknesses
- **Limited novelty in mitigation approach**: The proposed solution—SAM—is a known technique for generalization and noise robustness in standard deep learning. While applying it to CBMs is useful, the paper lacks a CBM-specific or concept-aware noise mitigation strategy, limiting the technical contribution.
- **Incomplete comparison with noise-robust baselines**: The paper compares only SGD versus SAM, with label smoothing relegated to Appendix F. Missing comparisons to established noisy-label methods (e.g., loss correction, co-teaching, sample selection, robust loss functions) make it unclear whether SAM is the best available approach or merely one reasonable option.
- **Synthetic noise only; unknown generalization to realistic noise**: All experiments use synthetic symmetric or pairwise noise. Real-world concept annotation errors are likely instance-dependent (ambiguous images more likely mislabeled), concept-dependent, or annotator-dependent—factors not addressed. The practical applicability remains uncertain.
- **Missing intervention capability evaluation**: A core benefit of CBMs is test-time intervention where humans can correct concept predictions. The paper claims improved interpretability but never evaluates whether SAM-trained models preserve or improve intervention capabilities—a critical gap given CBMs' intended use in high-stakes domains.
- **Modest effectiveness at high noise levels**: Even with SAM, models collapse at 40% noise (e.g., Ind on CUB reaches only ~5% target accuracy). SAM provides meaningful improvements at 20% noise but is insufficient at realistic high-noise scenarios.

## Nice-to-Haves
- Statistical significance testing for reported improvements, particularly for small gains (e.g., +0.6% concept accuracy) where standard deviations overlap.
- Analysis of which concept types (color, shape, pattern) are most vulnerable to noise and most helped by SAM, which would guide annotation prioritization.
- Loss landscape visualization or comparison to other flatness-inducing methods to validate the mechanistic claim that SAM helps via flatter minima.

## Novel Insights
The paper's key insight is the differential impact of noise type: concept noise fundamentally disrupts representation learning in CBMs, causing learned concept representations to lose their clustering structure (visualized via t-SNE), while target noise preserves meaningful representations. This is demonstrated through careful ablation and provides actionable guidance—CBM practitioners should prioritize concept label quality over target label quality. The analysis of concept importance shifts under noise (e.g., critical concepts dropping in importance while irrelevant concepts gain weight) further reveals how noise corrupts the interpretability-value chain in CBMs.

## Potentially Missed Related Work
- None identified by the search process.

## Suggestions
- Add comparison with at least 2-3 dedicated noisy-label learning methods (e.g., loss correction, co-teaching) to establish whether SAM is an effective choice among available options.
- Evaluate intervention effectiveness under noise: measure whether correcting concept predictions at test time improves downstream predictions, and whether SAM-trained models retain this capability better.
- Include experiments with instance-dependent noise patterns to assess generalization to realistic annotation errors.
- Report improvements separately per noise level rather than averaging across clean and noisy scenarios (as in Table 2's "∆" column) to clarify when SAM helps versus when gains are negligible.

---

