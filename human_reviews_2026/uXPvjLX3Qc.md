# BaryBind: Binding All Modalities via Multimodal Wasserstein Barycenter Space

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Multimodal joint representation, which aligns multiple modalities in a shared latent space, has emerged as the foundation of recent multimodal understanding models. To scale beyond two modalities, existing models typically treat a specific modality  (e.g., text) as the anchor to bind other modalities via pairwise contrastive losses. However, the learned joint representation space tends to be sub-optimal and imbalanced, as the modality-specific anchor may inherit the modality bias and insufficiently capture the modality-agnostic semantics and holistic geometric structures within multimodal data. In this work, we are motivated by the intuition that multimodal representations arise from different shifts from an underlying modality-agnostic representation space. Based on this, we present **BaryBind**, a multimodal framework that aligns modalities in the multimodal Wasserstein barycenter (WB) space, which inherently models a modality-agnostic distribution by minimizing the average of Wasserstein distances to all modalities. We further construct a barycenter polytope, whose volume serves as a geometric metric for quantifying $n$-modality alignment.  This metric is integrated as a barycenter-anchored volumetric contrastive loss that contrasts the volumes of the $n$-dimensional polytopes, encouraging global alignment of non-anchor modalities to the barycenter while reducing inter-modality gaps. Extensive experiments show that BaryBind delivers more balanced zero-shot generalization performance in downstream tasks, e.g., cross-modal text/video retrieval and classification.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper aims to generate multimodal representations by aligning different modalities into a common latent space. The proposed method is evaluated on text-to-video (T-to-V), video-to-text (V-to-T), multimodal classification, and retrieval tasks, demonstrating promising performance gains over existing methods such as VAST.

### Strengths
- The authors formulate the problem within a multimodal learning framework that aligns diverse modalities in the Wasserstein barycenter (WB) space and propose three complementary loss functions to effectively optimize the network.

- The method is evaluated on multiple benchmark datasets, showing consistent and competitive performance.

- Comprehensive ablation studies are conducted to assess the contribution of individual components (e.g., each loss term) and to analyze the effect of different experimental settings such as varying the anchor modality.

### Weaknesses
- The authors claim that existing models typically fix a specific modality (e.g., text) as the alignment anchor to bind others via pairwise contrastive losses, which limits scalability beyond two modalities. Although BaryBind aims to align all modalities to a shared latent space via distribution matching based on the Wasserstein distance, it still appears to treat the text modality as the anchor through simple MLP mappings. Consequently, the proposed method may still be minimizing the discrepancy primarily between the text modality and others. It is unclear to what extent, BaryBind differs from the existing solutions to address the anchoring issue highlighted in the abstract.

- Missing evaluation on system efficiency during training: The reported experiments were conducted on two NVIDIA A100 GPUs. It would be helpful to provide a direct comparison with existing model to illustrate the trade-off between efficiency and effectiveness.

- The proposed method is not evaluated on text-to-audio (T-to-A) or audio captioning tasks. Since most experiments focus on T-to-V and V-to-T tasks, it would strengthen the paper to demonstrate that the approach generalizes across modalities by including results on text–audio datasets such as Clotho and AudioCaps. Otherwise, it remains unclear whether the generated representations are biased toward certain modalities.

- Some important related works are missing and should be discussed for completeness:

Explore the Limits of Omni-modal Pretraining at Scale, ICCV 2025.

ViT-Lens: Towards Omni-modal Representations, CVPR 2024.

### Questions
Some additional questions are given below:

- Is the dataset used for training the proposed method identical to that used in VAST? It appears that the current work employs VAST-150K, a subset of VAST-27M. Please clarify the rationale behind choosing this subset.

- It would be beneficial to include a comparison against recent open-source multimodal large language models (e.g., Qwen, Qwen-Audio) to contextualize the proposed model’s performance relative to emerging large-scale multimodal baselines.

### Soundness
3

### Presentation
3

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
This paper investigates multimodal alignment by leveraging Wasserstein barycenter concept in Optimal transport theory. The main idea of the proposed method, BaryBind, is matching the Barycenter space derived from multimodal distributions to encourage each modality embedding being aligned to barycenter. Experiments show that the proposed method outperforms baseline methods, showing possible potential to enhance the multimodal alignment method.

### Strengths
The paper is well written and organized. Leveraging Wasserstein barycenter space for aligning multimodal embeddings is interesting. The current experimental results are also interesting as only the proposed model with pretraining only one-epoch can surpass other baseline models.

### Weaknesses
While the approach that leveraging OT theory looks interesting, I do not see a clear problem statement that the authors tackle and the rationale behind of using Wasserstein barycenter, which makes it seem incremental. Moreover, I am doubt with the experiment settings if it is fair comparison and if it properly shows the effect of the proposed loss functions. Below, I list some of my questions regarding my concerns.

### Questions
1. First of all, what is the main challenge or limitations of existing methods? From my understanding, this works tries to make modality-agnostic multimodal alignment as the previous works use anchor. If so, the challenge needs to be formally analyzed.
1. Constructing Wasserstein barycenter space needs barycenter, where the paper uses anchor-modality embedding for constructing it. This does not match the claim "modality-agnostic" as the method somehow biased to anchor modality when it align multimodal embeddings. 
2. Regarding the above, using Wasserstein barycenter space as a contrasting point seems just aligning pair-wise distance from barycenter (some embedding of anchor modality) and all other non-anchor modalities (from eq(5)). This should be clearly compared mathematically and empirically. 
3. I do not see the clear reasoning to use Data-anchor matching (DAM) loss. The paper only states "To complement the barycenter-based alignment loss..." Where does the problem behind of "instance-level supervision that encourages the model to distinguish between matched and mismatched pairs" come from and why do we need this? This requires clear motivation and analysis.
4. It seems BaryBind requires full multimodal samples, meaning that we must have all samples for every modality, otherwise BaryBind does not work. This is very important in practice as such multimodal datasets are rare.
5. Are the baseline methods (e.g., imageBind, languageBind, VAST, etc) trained on the same pretraining dataset with one epoch? Have the author compared baseline methods with the same backbone, the same hyperparameter, but only different in loss function? To fair compare and claim the superiority of the proposed loss, all the experimental setup should be identical. 
6. Time complexity is only studied between the similarity measure. As the BaryBind has other modules, total training time (e.g., batch completion time) should be analyzed too.

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
4

### Summary
This paper introduces a novel approach to multimodal representation learning—BaryBind, whose core idea lies in addressing the issue of imbalanced representation space caused by anchor modality bias in conventional methods. Unlike existing approaches that typically align modalities around a specific anchor modality, BaryBind unifies the representations of different modalities by aligning them into a shared Wasserstein barycenter space. This method innovatively leverages the Wasserstein barycenter as a modality-agnostic semantic center, thereby effectively capturing semantics common to all modalities and achieving more balanced and robust alignment of multimodal representations.

### Strengths
The paper demonstrates significant strengths across several key dimensions:

**Originality & Conceptual Innovation** - The core contribution lies in fundamentally rethinking multimodal alignment objectives. By proposing the Wasserstein barycenter as a modality-agnostic semantic center, it shifts the paradigm from point-based anchoring to distribution-centered alignment. This conceptual breakthrough is further enhanced by the introduction of the barycenter polytope volume as a geometric metric, which transforms abstract alignment notions into computable quantities that naturally capture higher-order interactions beyond pairwise similarities.

**Theoretical Rigor & Technical Foundation** - The work establishes solid theoretical grounding through the dual formulation derivation of the MWB loss (Proposition 1), demonstrating careful mathematical development rather than heuristic design. 

**Experimental Validation** - The experimental design stands out for its comprehensive coverage and convincing demonstrations across multiple benchmarks. The evaluation strategy effectively substantiates the method's advantages while maintaining scientific rigor in comparisons with state-of-the-art approaches.

**Presentation & Clarity** - Despite the conceptual complexity, the paper maintains logical coherence and accessibility through well-structured exposition. The introduction successfully frames the limitations of existing approaches and the paper's contributions, while Figures 1 and 3 provide exceptional visual intuition for understanding the core workflow and methodological distinctions from baselines.

### Weaknesses
**Theoretical Limitations in Wasserstein Barycenter Approximation** - While theoretically grounded in optimal transport, the practical implementation relies on a lightweight MLP $T_\theta$ to approximate the mapping to WB space through dual formulation. This parametric approximation raises questions about whether the method truly learns a distribution minimizing Wasserstein distances to all modalities, or merely converges to point estimates $b$ that optimize the specific loss function. The discrepancy between the theoretical WB (a distribution) and the implemented point estimate deserves further validation.

**Geometric Metric Limitations** - The barycenter polytope volume $V$, though innovative as a global alignment measure, presents interpretability challenges. As a scalar quantity, it effectively indicates the degree of misalignment but cannot identify which specific modalities contribute to the problem. Furthermore, its dependence on the number of modalities $n$ limits cross-model comparability, undermining its potential as a universal metric. Normalization strategies such as $V^{1/n}$ could enhance its applicability across different modality configurations.

**Insufficient Experimental Validation** - The experimental scope doesn't fully support the claims of modality-agnostic representation and improved inter-modal interactions. Critical tests for robustness under modality absence (e.g., missing video data during inference) are lacking, which would powerfully demonstrate advantages over anchor-based methods. Additionally, direct evidence for enhanced non-anchor modality interactions remains limited - probing tasks or mutual information analysis between modalities like audio and video under BVC constraints would provide more convincing validation.

### Questions
**Comparative Analysis of WB Approximation Methods** - To address the theoretical concerns regarding Wasserstein barycenter approximation, future work should compare the current MLP-based approach with more advanced neural optimal transport mappings, such as those proposed in [Kolesov et al., 2024a] and [Tang et al., 2025]. This comparative evaluation would help validate whether different approximation techniques significantly impact final performance and provide insights into the trade-offs between computational efficiency and theoretical fidelity.

**Normalization Strategies for Polytope Volume Metric** - For the barycenter polytope volume to serve as a universal alignment metric, investigation into normalization methods is essential. Exploring geometric normalization factors like $V^{1/n}$ could enable meaningful comparisons across models with varying numbers of modalities $n$. Developing a standardized normalization approach would enhance the metric's practicality and interpretability in diverse multimodal learning scenarios.

**Enhanced Experimental Validation through Probing Tasks** - To substantiate claims of improved modality-agnostic representation and inter-modal interactions, future experiments should include targeted probing tasks. These could assess model robustness under modality conflicts (e.g., contradictory audio and video signals) or directly quantify cross-modal relationships through feature correlation analysis and mutual information measurements between modalities under BVC constraints.

**References**  
[1] Kolesov et al. Estimating barycenters of distributions with neural optimal transport. arXiv:2402.03828 (2024a)  
[2] Tang et al. Baryir: Learning multi-source unified representation in continuous barycenter space for generalizable all-in-one image restoration. arXiv:2505.21637 (2025)

### Soundness
2

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
4

### Summary
The paper introduces BaryBind, a new framework used for aligning multimodalities to a Wasserstein barycenter (WB) space, say, a distribution in latent space that minimizes the average Wasserstein distance to each modality’s latent distribution. In addition, the paper introduces a volumetric metric among the embeddings of the modalities around the barycenter. and proposed a volumetric contrastive‐style loss for ensuring tighter alignment and reducing inter‐modality gaps. Experimental results demonstrate that BaryBind achieves significant performance improvement over baselines on cross-modal retrieval and classification tasks, highlighting the effectiveness of the proposed approach.

### Strengths
- Clear and Well-Structured: The paper is well-organized, with detailed explanations of the preliminary, intuition, and methodology.

- Interesting Method: The use of optimal transport / Wasserstein barycenters as a latent‐space tool is theoretically interesting.

- Superiority in Alignment: The experimental results demonstrate that the proposed method achieves the best performance on the cross-modal retrieval and classification tasks compared to the baselines.

### Weaknesses
- Currently, the MWB, BVC, and DAM loss objective functions are equally weighted in the combined loss, but it remains unclear whether assigning different weights could lead to better performance. A study of this trade-off would provide deeper insight into the relative importance of multimodalities' alignment preferences.

- The paper does not include experimental comparisons with other recent multimodal alignment methods, such as TRIANGLE [1] and GRAM [2]. Including these baselines would provide a stronger empirical validation of BaryBind’s effectiveness.

- The paper would benefit from a more in-depth ablation analysis. While the authors provide clear theoretical intuition and a validation experiment for each proposed component, the empirical section lacks a deeper discussion and interpretation of how these components individually and collectively contribute to the overall performance.

- Figures showing toy embeddings (before/after alignment) would help in visualizing the effect of the volumetric loss. Such visualizations could help demonstrate how embeddings converge toward the Wasserstein barycenter, how inter-modality gaps are reduced, and whether the volumetric constraint indeed promotes tighter alignment.

[1] A TRIANGLE Enables Multimodal Alignment Beyond Cosine Similarity, NeurIPS 2025

[2] Gramian multimodal representation learning and alignment, ICLR 2025

### Questions
How robust is it to missing modalities, which are common in realistic data? Does the barycenter degrade gracefully if one modality is missing?

### Soundness
2

### Presentation
2

### Contribution
2
