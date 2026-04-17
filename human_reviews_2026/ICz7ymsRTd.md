# Model Fusion via Neuron Interpolation

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 4

## Abstract
Model fusion aims to combine the knowledge of multiple models by creating one representative model that captures the strengths of all of its parents. However, this process is non-trivial due to differences in internal representations, which can stem from permutation invariance, random initialization, or differently distributed training data. We present a novel, neuron-centric family of model fusion algorithms designed to integrate multiple trained neural networks into a single network effectively regardless of training data distribution. Our algorithms group intermediate neurons of parent models to create target representations that the fused model approximates with its corresponding sub-network. Unlike prior approaches, our approach incorporates neuron attribution scores into the fusion process. Furthermore, our algorithms can generalize to arbitrary layer types. Experimental results on various benchmark datasets demonstrate that our algorithms consistently outperform previous fusion techniques, particularly in zero-shot and non-IID fusion scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The process of model fusion is non-trivial, resulting the three key gaps in prior research, including reproducibility, base model quality, and heterogeneous data. Thus, this paper proposes a neuron-centric family of model fusion algorithms regardless of training data distribution as these algorithms incorporate neuron attribution scores into the fusion process.

### Strengths
1. It casts fusion as a principled representation matching problem, yielding a two-stage algorithm, which measure grouping error and approximation error. It decouples the traditional objective by introducing an auxiliary vector, enabling a more tractable decomposition of the cost function.
2. It incorporates neuron saliency into alignment, improving performance across our methods and enhancing existing approaches. 
3. It provides a flexible opensource re-implementation of existing algorithms.

### Weaknesses
1. This method view DNN as a function parameterized by weights and for many model architectures, this function can be decomposed into many subfunctions. However, the rapid development of network architecture has made many SOTA network architectures no longer cascading, but containing many intricate connections. The authors do not discuss these complex network architectures.
2. Does a collection of pretrained base models to share the same network architecture, that is, do they need to be siamese networks? Is it possible for models with different network architectures to merge?
3. It lacks the comparison with SOTA model fusion methods, especially those proposed in the recent two years.

### Questions
1. For the three key gaps in prior research, a more intuitive explanation or illustration is needed as some gaps, especially the third gap, are professionally in-depth and difficult to understand.
2. In Eq. (1), $s_j$ is not introduced.
3. For data of different distributions, whether the auxiliary vector needs to be retrained as the optimal solution of the grouping error may be changed.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new family of model fusion algorithms, termed "Neuron Interpolation," designed to merge multiple trained neural networks into a single representative model. The core idea is to frame fusion as a layer-by-layer representation-matching problem. The method operates in two stages per level: 1) a "grouping" step, which clusters the neuron outputs of all parent models (using K-means or Hungarian matching) to find importance-weighted "target" cluster centers , and 2) an "approximation" step, which trains the weights of the fused model's current level to match these target centers. The authors introduce two main variants: Hungarian Fusion (HF) for one-to-one matching of equal-sized models and K-means Fusion (KF) for the general case. A key contribution claim is the incorporation of neuron attribution (saliency) scores into this process. The paper presents experiments across various data distributions (full, non-IID, and "sharded") , claiming to significantly outperform prior methods like OTFusion and Git Re-Basin, especially in challenging zero-shot scenarios.

### Strengths
- **Problem Motivation:** The paper correctly identifies a significant and practical problem: existing fusion methods struggle in zero-shot and non-IID settings, which are common in real-world applications like Federated Learning.
- **Flexible Framework:** The proposed two-stage (grouping, fitting) framework is flexible. The K-means Fusion (KF) variant can naturally handle fusing models of different widths  (as shown in Table 1, if it were filled out), and the gradient-based variant can be applied to arbitrary differentiable layers

### Weaknesses
- **High Complexity and Sensitivity:** The authors' best method (gradient-based KF) is admitted to be "sensitive to hyperparameters" and its lack of robustness is shown by requiring two different settings for the paper's own experiments. It is also computationally expensive, running 16-23x slower than baselines (Table 10).
- **Overstated Saliency Contribution:** The claim of being the "first" to use saliency scores is factually incorrect, as the authors admit in Appendix C that prior work (Singh and Jaggi, 2020) already proposed it. Furthermore, the empirical gain from these scores is minimal (Tables 6, 7), contradicting the abstract's emphasis on this contribution.
- **Limited Novelty:** The method's novelty is limited, as it combines two well-known concepts: 1) neuron alignment via matching/clustering (as in OTFusion) and 2) multi-teacher feature-based distillation (which the authors call "fitting"). This combination is an incremental step, not a fundamental breakthrough.
- **Poor Presentation of Results:** The experimental section is poorly structured and difficult to follow. Critically, the authors often present tables of results without adequately discussing them or, in some cases, even not referencing them in the main text. This is a major oversight that hinders review.

### Questions
1. **Missing Related Work on Heterogeneity:** The related work section appears to miss some key references focused on heterogeneity. Could the authors elaborate on how their work, especially KF, differs from and compares to the cross-layer alignment method for heterogeneous networks in [1]? Furthermore, given the strong Federated Learning (FL) motivation, how does this approach relate to other federated methods designed to address client heterogeneity, such as [2]?

2. **Clarification of HF for Multi-Model Fusion:** The Hungarian Fusion (HF) method is described as solving a one-to-one matching problem, which is well-defined for the two-model case. However, the paper also presents results for multi-model fusion. How is the one-to-one matching problem formulated and solved when fusing $N > 2$ models? 

3. **Addressing Error Propagation:** The paper claims a key weakness of prior work is "ignoring how the fused model evolves as the algorithm iterates through the levels... without accounting for potential changes [in] previous level outputs" (Lines 162-165). However, in the simple two-model case with linear levels and uniform importance, HF seems to reduce to a process very similar to OTFusion. Could the authors clarify the *exact* mechanism by which their method solves this alleged error propagation issue?

[1] Nguyen, Dang, et al. "On cross-layer alignment for model fusion of heterogeneous neural networks." ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) . IEEE, 2023.

[2] Makhija, Disha, Nhat Ho, and Joydeep Ghosh. "Federated self-supervised learning for heterogeneous clients."arXiv preprint arXiv:2205.12493 (2022).

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a neuron-level approach to model fusion that decomposes the fusion objective into two complementary components: (i) grouping neurons from multiple models according to similarity and importance, and (ii) fitting a fused network’s neurons to the cluster centroids derived from these groups. Two algorithmic variants are presented:

- Hungarian Fusion (HF): a one-to-one neuron matching method for models with identical widths, formulated as a linear sum assignment problem.
- K-means Fusion (KF): a general-width method that clusters neurons across models using importance-weighted K-means.

The method leverages saliency scores (Conductance and DeepLIFT) to guide both grouping and fitting steps, aiming to emphasize important neurons during fusion. Experiments are performed on CNN (VGG) and ViT architectures across IID, Non-IID, and sharded data settings, with and without fine-tuning, claiming consistent improvements over baseline fusion methods such as Git-Rebasin and OTFusion.

The paper also claims partial theoretical guarantees for the linear case and reports practical improvements on mid-scale datasets such as CIFAR-100 and Tiny-ImageNet.

### Strengths
- The decomposition of the fusion objective into grouping and approximation stages is a useful formalization. It reframes neuron matching as a clustering-and-refitting process, connecting prior permutation and alignment-based methods to a broader optimization viewpoint.
- Integrating saliency measures to inform neuron grouping and weighting adds a novel, biologically-inspired dimension to the fusion literature, which has largely focused on structural similarity rather than neuron importance.
- The modular structure could inspire broader frameworks for neuron-level model alignment and repair.
- The notion of importance-weighted neuron grouping could be extended to parameter-efficient transfer and federated learning settings.
- Even if limited by data reliance, the approach provides a pathway toward more interpretable fusion via neuron attribution.

### Weaknesses
- **Title clarity**: Since “interpolation” is a standard aggregation term in the model merging literature, consider renaming to reflect the neuron-centric mechanism or saliency usage.
- While the neuron clustering view is fresh, the method seems to **heavily build on Git-Rebasin’s activation matching** and other alignment-based model fusion works (e.g., _Ainsworth et al., Git Re-basin: Merging Models Effectively, 2023_). The contribution mainly lies in incorporating saliency and centroid fitting, rather than introducing an entirely new paradigm. Claims of generality to “arbitrary differentiable levels” are not supported, since the derivations rely on layer-wise alignment and assume comparable architectures.
- **Fusion-data dependence**: Both neuron grouping (activations) and weight refitting (non-linear levels) depend heavily on a “fusion dataset.” However, no ablation explores sensitivity to data quantity or distribution. This weakens claims of applicability to federated learning, where shared data are scarce.
- **Baselines**: Missing strong contemporary baselines (e.g., permutation+LS fusion, repair/rescaling methods, and data-free fusion). Ensembles outperform MIN in non-IID regimes.
- **Scalability**: Experiments are confined to VGG and ViT-small; no scaling to large models (e.g., ResNet-50, ViT-B/16) or multi-model setups (K>4). Runtime and memory comparisons lack depth.

### Questions
1.  How does performance vary as fusion data become more limited or non-IID? Can the approach be adapted for data-free or privacy-restricted settings?
2. Does the alternating grouping/fitting procedure converge empirically? Are there oscillations in neuron assignments or objective values?
3. How stable are the importance weights across different baselines and input samples? Does randomizing them degrade performance significantly?
4. Are gains mainly from refitting the classification head?

### Soundness
2

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
3

### Summary
The paper proposes a neuron-centric model fusion framework that casts fusion as a layer-wise representation matching problem. The method operates in two stages per level: (1) a "grouping" step that clusters concatenated neuron outputs from base models to define a set of target representations (cluster centers), and (2) an "approximation" step that optimizes the fused model's current level to match these targets. The objective function (Eq. 1) is novel in its explicit incorporation of neuron attribution scores (e.g., Conductance, DeepLIFT) to weight the importance of matching specific neurons. The authors introduce two main variants: Hungarian Fusion (HF) for 1:1 matching of equal-width models and K-means Fusion (KF) for the general case, with both linear and gradient-based optimization schemes for the approximation step.

### Strengths
- The proposed framework is intuitive, flexible, and moves beyond simple weight permutation (like OTFusion or Git Re-Basin ) by actively fitting a new sub-network to an intermediate target representation
- The incorporation of neuron attribution scores into the fusion objective is a novel contribution, theoretically allowing the process to prioritize salient features
- The empirical results are strong, especially in zero-shot and non-IID "sharded" scenarios (Table 2, 15), where the method (KF Gradient) succeeds while established baselines like OTFusion fail completely
- The ResNet compression experiment (Table 7) is a compelling demonstration of the method's strength, showing a >2x accuracy improvement over standard Knowledge Distillation using the same limited data

### Weaknesses
- The gradient-based variants, which produce the best results, are admittedly sensitive to hyperparameters. The paper provides two starkly different configurations (Setting 1 vs. Setting 2, Table 9)  without a clear ablation or principle for choosing between them. This significantly undermines the method's robustness and practicality
- The central novelty claim—incorporating attribution scores —is weakly supported by the main experimental results. For the flagship ViT experiments (Table 15, 16), the gains from using Conductance or DeepLIFT over uniform weights are consistently marginal or non-existent. For example, in Table 16 (fine-tuned), KF Gradient Uniform (75.2%) performs identically to DeepLIFT (75.2%) and is on par with Conductance (75.4%). The paper fails to analyze why the scores help significantly in some niche cases (VGGs, Table 13 ) but not in the main ViT results.
- In the challenging sharded setups (Table 2, 15), standard Knowledge Distillation (KD) and Linear Probing (LP) are surprisingly strong baselines that the paper does not sufficiently contextualize. For the 4-way ViT split (Table 15), KF Gradient (43.5%) is only marginally better than KD (40.3%). The authors state that their "Setting 2" (used for most non-IID models) is similar to LP, as most gains occur at the head. This suggests the complex, layer-wise grouping may be superfluous in these settings.
- The paper dismisses FedMA as a "non-zero-shot" baseline because it requires "retraining... after the alignment of every layer". However, the proposed "Gradient version of KF" also performs optimization (i.e., retraining) at every level via SGD. This distinction seems artificial, and the lack of comparison to FedMA is a clear omission.
- The runtime comparison (Table 10) shows the linear variants (HF/KF Linear) are 1-2 orders of magnitude slower (14x-83x) than OTFusion. This makes them practically unusable as fast alternatives, while the gradient-based methods are presumably even slower.

### Questions
- The empirical benefit of attribution scores is marginal in key ViT results (Table 15, 16). Can you provide a clear hypothesis for when these scores are beneficial and why they fail to provide significant gains for ViTs?
- Given that "Setting 2" (used for most non-IID models ) is described as degenerating to Linear Probing, does this imply the complex layer-wise grouping is unnecessary for these scenarios?
- How does the proposed gradient-based KF, which optimizes weights level-by-level, fundamentally differ in methodology and computational cost from FedMA, which you dismissed for its layer-wise retraining?
- The ResNet compression experiment (Table 7)  is strong, but it compares against standard KD. How does your method compare to more advanced, layer-wise distillation techniques?

### Soundness
4

### Presentation
3

### Contribution
3
