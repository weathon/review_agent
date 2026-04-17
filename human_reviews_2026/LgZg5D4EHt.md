# Adaptive High-Dimensional Subspace Evolution Based on Broad Learning System and Error-Correcting Output Codes

- Decision: Reject
- Scores: 6, 2, 8, 2

## Abstract
High-dimensional data (HDD) commonly exhibit complex hierarchical structural characteristics; however, existing approaches typically employ fixed subspace evolution strategies that fail to adapt to the inherent hierarchical diversity across different datasets, resulting in suboptimal revelation of underlying discriminative patterns. Considering this critical limitation, we propose an adaptive high-dimensional subspace evolution algorithm (AHSE) featuring a dual-branch collaborative architecture: the series branch leverages Cholesky decomposition-based incremental Broad Learning System (BLS) to efficiently evolve cascaded subspaces tailored to distinct types of high-dimensional hierarchies; the parallel branch, built on multiple subspace evolution bases, utilizes post-hoc error-correcting output codes (ECOCs) for robust spatial encoding and evolutionary optimization. Both branches converge into a lightweight circuit, forming a closed evolutionary loop. Owing to the hierarchy-tailored evolution strategy, AHSE excels in various HDD tasks such as image pattern recognition, speech emotion recognition, and few-shot learning. Moreover, we offer a rigorous theoretical analysis of the mechanism and robustness guarantee of ECOCs on BLS, further promoting the integrity of AHSE.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Aiming at the complex hierarchical structure of high-dimensional data (HDD) and the core problem that the existing methods can't adapt to the diversity of data hierarchy by using fixed subspace evolution strategy, this paper proposes an adaptive high-dimensional subspace evolution algorithm (AHSE). AHSE adopts a double-branch collaborative architecture: the serial branch is an incremental breadth learning system (BLS) based on Cholesky decomposition, which customizes the evolution path of cascaded subspaces for high-dimensional data of different hierarchical types through feature priority sorting, subspace alignment and sample weight evolution; Parallel branch (PATH) is based on multi-subspace evolution basis, combined with post-error correction output code (ECOCs) to realize robust spatial coding, and the target space is purified by Flame optimization mechanism. The two branches are dynamically fused by lightweight SPOT circuit to form a closed-loop evolutionary system. This paper provides a rigorous theoretical analysis and robustness proof of ECOCs on BLS. Experiments verify the superiority of AHSE in many high-dimensional tasks such as image pattern recognition, speech emotion recognition, and small sample learning, giving consideration to performance and computational efficiency.

### Strengths
1. Adaptive subspace evolution with hierarchical adaptation is proposed, which breaks through the limitation of traditional fixed evolution strategy, and designs differentiated evolution modes for different structural data such as MNIST and Fashion-MNIST to accurately match the internal characteristics of the data.
2. SEED branch inherits the lightweight advantage of BLS and realizes efficient subspace iteration through incremental Cholesky decomposition. PATH branch uses the error correction characteristics of ECOCs to improve the anti-noise ability, Flame mechanism effectively solves the compatibility problem between ECOC codebook and BLS latent space, and the two branches cooperate to achieve the balance between efficiency and robustness.
3. SOTA or Top-2 performance is achieved in images (MNIST to TinyImageNet), speech emotion recognition (5 benchmark data sets) and small sample learning (5 high-dimensional small sample data sets), and the training time and reasoning FLOPs are significantly lower than those of ResNet and ViT models, giving consideration to generalization and deployment feasibility.

### Weaknesses
1. The evolution modes (exponential, cosine, linear) of different data sets need to be preset manually, and there is no mechanism to automatically identify data hierarchy types and dynamically select evolution strategies, which limits the applicability of the method to high-dimensional data with unknown structure.
2. The performance of extremely high-dimensional small sample scenes whose feature dimensions far exceed the number of samples (such as ten thousand-dimensional features and thousands of samples) has not been tested, which only verifies the robustness under weighted noise, and does not explore the influence of input data noise (such as Gaussian noise and feature missing) on the evolution process, so its practicability is limited.
3. The length and coding strategy of ECOC codebook depend on manual setting and fixed pool selection, and lack of adaptive optimization.
4. Compared with the mainstream high-dimensional data processing methods in recent years (such as self-supervised subspace learning and adaptive kernel method).

### Questions
Please refer to weaknesses.

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
This paper introduces AHSE (Adaptive High-Dimensional Subspace Evolution), an algorithm that aims to adaptively evolve feature subspaces in high-dimensional data. The model integrates three components: SEED: a series subspace evolution mechanism using Cholesky-decomposition-based incremental Broad Learning Systems (BLS); PATH: a parallel subspace evolution using Error-Correcting Output Codes (ECOCs); SPOT: a circuit that fuses SEED and PATH outputs dynamically. The paper claims that this dual-branch design enables hierarchical adaptation to different data structures, achieving superior results on diverse tasks including image recognition, speech emotion recognition, and few-shot learning.

While the paper presents an interesting and ambitious approach to adaptive subspace learning, the contribution lacks theoretical depth, conceptual clarity, and fair experimental validation for acceptance at a top-tier venue. A significantly revised and more focused version - with clear ablations, theoretical integration, and modern baselines - could be competitive in the future.

### Strengths
1. The attempt to address “adaptive subspace evolution” in high-dimensional data is an underexplored and potentially meaningful problem.
2. Comprehensive experiments: The paper includes a wide range of datasets and tasks.
3. The manuscript provides equations, pseudocode references, and hyperparameters, showing effort toward reproducibility.

### Weaknesses
1. The core ideas are extremely difficult to follow. Terms such as “evolutionary Cholesky decomposition,” “Flame optimization,” “hierarchy-tailored evolution,” and “closed-loop subspace evolution” are introduced with little intuition or justification. It remains unclear what adaptive subspace evolution concretely means in learning-theoretic or algorithmic terms. The method reads as a collection of loosely related heuristics rather than a coherent learning principle.
The presentation is overly verbose and obfuscatory. Many mathematical formulations restate standard operations (e.g., weighted least squares, feature ranking, or error-correcting output codes encoding) in unnecessarily complex notation.
 
2. Despite heavy terminology, the technical content is incremental:
- The SEED component largely combines feature selection, incremental regression, and Cholesky updates, which are all well-established.
- PATH builds on existing ECOC formulations with a “Flame” heuristic that seems ad hoc and lacks justification.
- SPOT’s fusion mechanism is simply a weighted combination of two outputs using validation loss ratios.
There is no fundamental new learning algorithm here — only an engineering combination of BLS, ECOC, and feature-ranking procedures.
 
3. The paper repeatedly asserts that it provides “rigorous theoretical analysis” and “robustness guarantees” for ECOCs on BLS. However, there are no theorems, proofs, or meaningful derivations in the main text. The “theoretical” results are said to appear in the appendix but are not summarized, contextualized, or experimentally verified. This undermines the scientific validity of the claimed contributions.
 
4. While many datasets are used, the evaluation does not meet top-tier standards:
- Unfair comparisons: BLS-based models are trained on pretrained ResNet or MoCo features, while deep baselines train end-to-end, making comparisons misleading.
- Weak baselines: Many chosen baselines (e.g., MLP, VGG-16, ResNet-34) are outdated. Modern efficient architectures (ConvNeXt, Swin, EfficientNet, ViT-L variants, or transformer hybrids) are absent.
- No variance reporting: All results are single-run accuracies, with no confidence intervals or standard deviations.
- Questionable scalability: The method is described as efficient, but the numerous steps (FPFE, SA, SWE, Flame, etc.) appear computationally heavy, and there are no FLOP or runtime analyses beyond summary tables.
The presented results cannot be confidently interpreted as evidence of superiority.
 
5.  The manuscript is bloated, overtechnical, and poorly structured. The authors use jargon-heavy language throughout, making it exhausting to read and nearly impossible to extract the core insight. Figures are decorative rather than explanatory; algorithms are fragmented across appendices; and critical details are buried in complex pseudocode.
Overall, the paper fails the clarity standard required for top tier-level publication.

### Questions
1. Can you explicitly define what is new in AHSE compared to prior BLS or ECOC-based frameworks?
2. How exactly does “adaptive evolution” differ from conventional feature selection or boosting?
3. What are the computational requirements? The architecture seems complex; please provide memory and runtime breakdowns.
4. Is there any theoretical justification (beyond empirical heuristics) for the “Flame” optimization mechanism?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This is a well-written and technically substantial paper proposing AHSE, an adaptive high-dimensional subspace evolution framework that integrates a serial evolution branch (SEED), a parallel ECOC-based branch (PATH), and a SPOT fusion circuit. The paper tackles the long-standing problem of fixed subspace evolution in high-dimensional data and introduces an architecture that dynamically tailors its evolution path based on data hierarchy.

The methodology is technically solid, with clear mathematical formulation and well-structured theoretical analysis—especially the derivation of ECOC robustness guarantees within the Broad Learning System (BLS) context. The experiments are broad and compelling, covering both small-scale and large-scale image datasets, speech emotion recognition, and few-shot learning. Ablation results (Table 3) and visualization of evolution modes (Figures 1–3) convincingly validate the adaptive design.

Despite its quality, several aspects could be refined. Some comparisons lack strict fairness (e.g., pretrained feature reliance for BLS-based models), and runtime efficiency would be clearer with standardized hardware benchmarks. The methodology’s generality beyond BLS-based frameworks also warrants further discussion. These are minor but relevant issues in an otherwise strong submission.

### Strengths
• The dual-branch SEED–PATH design, combined through SPOT, provides a principled mechanism for adaptive subspace evolution that generalizes across hierarchically diverse datasets.
• Theoretical analysis of ECOCs in BLS is rigorous and novel, with clear proofs and practical robustness implications.
• Experiments are comprehensive and multi-domain, demonstrating consistent superiority across image, speech, and few-shot settings.
• Ablation studies are detailed and provide clear evidence of each module’s contribution, particularly FPFE and Flame.
• The paper is very well written, conceptually coherent, and easy to follow despite its technical depth.

### Weaknesses
• Fairness in experimental comparisons could be improved, as AHSE benefits from pretrained features while DNN baselines are trained end-to-end.
• Computational efficiency claims rely on FLOPs and CPU time rather than uniform wall-clock measurements across hardware setups.
• Hyperparameter tuning criteria and seed variance are not reported, limiting reproducibility before code release.
• The adaptive evolution principle is framed specifically for BLS; its applicability to other model families (e.g., deep ensembles) remains untested.
• The theoretical analysis, while elegant, focuses on bounded-noise scenarios and does not consider stronger adversarial perturbations.

### Questions
Can you clarify how feature extraction fairness was ensured in comparisons against deep baselines? For instance, would AHSE’s advantage remain if all methods used the same frozen backbone (e.g., ResNet-34)?

The evolution schedules (γ, T, α) for different hierarchies are central to your approach. How are these parameters selected in practice—heuristically or via validation search—and how sensitive are results to them?

Could you provide wall-clock runtime comparisons on a shared hardware configuration to substantiate the efficiency claim beyond FLOPs and CPU-only metrics?

Have you evaluated how AHSE behaves under non-Gaussian or structured noise perturbations to validate robustness beyond the assumptions of Theorem 1?

While your framework is formulated around BLS, could the adaptive subspace evolution principle extend to deep or transformer-based encoders? If not, what are the main obstacles to doing so?

Finally, can you provide variance or confidence intervals (e.g., standard deviations across runs) for key results in Tables 1 and 2 to better understand reproducibility and statistical significance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes an Adaptive High-dimensional Subspace Evolution algorithm (AHSE) to address the limitation of static subspace evolution strategies in existing methods. AHSE is able to adapt to the inherent hierarchical diversity across different high-dimensional datasets. The architecture of AHSE contains a SEED branch that evolves subspaces using a Cholesky decomposition-based incremental Broad Learning System, a PATH branch that evolves multiple subspaces in parallel based on post-hoc Error-Correcting Output Codes for robust spatial encoding and evolutionary optimization, and a SPOT circuit that dynamically fuses the outputs of both branches to form a closed-loop evolutionary system. Extensive experiments on image classification, speech emotion recognition, and few-shot learning demonstrate that AHSE achieves comparable performance to state-of-the-art methods while maintaining high efficiency.

### Strengths
The paper's major contribution is formulating the subspace evolution problem as an adaptive evolution procedure. It is discussed that a fixed evolution strategy is suboptimal, and the subspace should be dynamically adapted according to the data's intrinsic hierarchical structure. The proposed AHSE is built upon a solid mathematical foundation on the basis of broad learning systems and error-correcting output codes. A rigorous theoretical analysis is presented in the appendix, providing mechanism and robustness guarantees for ECOCs on BLS under mild assumptions.

### Weaknesses
The content organization of this paper needs to be substantially improved, and the experiment section also needs improvements to be more convincing. 
1. The problem of subspace evolution is not properly introduced in this paper. Firstly, the problem itself is not a widely studied topic in literature, so a potential reader is not likely to have a good understanding of its basic concept. Secondly, the introduction section only discusses the flaws of existing static evolution strategies, but have not introduced the subspace evolution problem itself. Finally, the preliminary section has not formally define the problem, but only introduces BLS and ECOC. I have not understood the problem until reading the methodology section.

2. It is a bad idea to put a figure right underneath the title. And the contents of the figure is also confusing. The notion of data hierarchy and evolution pattern are hard to interpret.

3. Many contents in the paper contains references to the appendixes, and these contents are not self-contained. So the paper itself is incomplete without its appendix. If the page limits is too short for this paper, perhaps it should be better to submit the paper to a journal where the manuscript can be longer.

4. The experiments section compares AHSE with many widely-used models, but the details of the compared models is not clear. For example, VGG and Resnet have multiple variations, each of which have different number of layers and parameters. Besides, the results of ViT is obviously worse than a well-trained ViT model could get. Considering these aspects, the experimental results are not conviencing enough.

### Questions
Please see the weakness section above, and explain the concern about the experimental results.

### Soundness
2

### Presentation
1

### Contribution
2
