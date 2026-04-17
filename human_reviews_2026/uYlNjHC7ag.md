# Hierarchical Multi-Scale Molecular Conformer Generation

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Molecular conformer generation is a fundamental task for drug discovery and material design. Although deep generative models have progressed in this area, existing methods often overlook the hierarchical structural organization inherent to molecules, leading to poor-quality generated conformers. To address this challenge, we demonstrate that capturing the spatial arrangement of key substructures, such as scaffolds, is essential, as they serve as anchors that define the overall molecular distribution. In this paper, we propose a hierarchical multi-scale molecular conformer generation framework (MSGEN), designed to enhance key substructure awareness by leveraging spatially informed guidance. Our framework initiates the generation process from coarse-grained key substructures, progressively refining the conformer by utilizing these coarser-scale structures as conditional guidance for subsequent finer-scale stages. To bridge scale discrepancies between stages, we introduce a molecular upsampling technique that aligns the structural scales, ensuring smooth propagation of geometric guidance. Extensive experiments on standard benchmarks demonstrate that our framework integrates seamlessly with a wide range of existing molecular generative models and consistently generates more stable and chemically plausible molecular conformers.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces MSGEN, a hierarchical multi-stage framework for molecular conformer generation. The core idea is to generate molecular structures in a coarse-to-fine manner—first producing coarse substructures such as rigid scaffolds or heavy-atom backbones, and then refining them into full-atom conformations. Each stage is trained independently using a diffusion-based model, and the coarse-level geometry is passed to the next stage via an upsampling-based conditioning module.

The authors claim that this hierarchical design improves structural awareness, stabilizes generation, and produces more chemically valid conformers across standard benchmarks (GEOM-QM9 and GEOM-Drugs). Extensive experiments are presented across multiple backbone models (GeoDiff, ConfGF, ET-Flow, EBD), showing quantitative improvements in coverage (COV) and RMSD (MAT) metrics, as well as minor gains in ensemble property prediction.

Overall, the paper proposes a conceptually clear extension of existing diffusion-based conformer generation frameworks by introducing multi-stage conditioning, with an aim to better preserve molecular substructures and enhance geometric fidelity.

### Strengths
- **Clear Motivation and Formulation:**
    
    The motivation to introduce hierarchical structure into conformer generation is logical and clearly explained. The paper provides an organized mathematical formulation and schematic overview (Fig. 2) that makes the proposed pipeline easy to follow.
    
- **Compatibility with Existing Diffusion Models:**
    
    MSGEN is implemented as a general plug-in framework that can be attached to various diffusion or flow-based backbones (e.g., GeoDiff, ConfGF, ET-Flow). This design flexibility increases the practical usability of the method.
    
- **Comprehensive Experimental Coverage:**
    
    The authors evaluate their approach on multiple datasets (GEOM-QM9, GEOM-Drugs) and across different base architectures. 
    
- **Empirical Consistency Across Models:**
    
    The hierarchical conditioning leads to modest but consistent improvements over several backbones, suggesting that the framework is stable and reproducible.

### Weaknesses
**Point 1. Ambiguity in Stage-1 Scaffold Arrangement**

The first stage determines the spatial arrangement of isolated scaffolds (Figure 2) without knowledge of the full molecular context (e.g., linker length or flexibility). This arrangement may dominate the final conformation but cannot adapt once the subsequent stages add connecting atoms.

Consequently, the hierarchical formulation appears somewhat unsmooth, and it remains unclear whether this early arrangement can be corrected or whether it meaningfully affects overall generation quality.

**Point 2. Unfair Comparison and Step-Dependent Behavior**

The reported improvements may partially stem from the increased number of diffusion steps rather than from the hierarchical design itself. While the appendix (Table 13) includes equal-total-steps results, the main text does not, preventing fair comparison.

Moreover, GeoDiff + MSGEN with 5 k steps outperforms the 10 k-step version in recall (comparing Table 5 and Table 13), suggesting inconsistent behavior that is neither analyzed nor explained.

**Point 3. Limited Informativeness of the Upsampling Scheme**

The upsampling process simply places new atoms randomly around their topological neighbors. Since these coordinates are only used as conditioning input, it is unclear what meaningful geometric information they (newly added atoms) provide. If the network must relearn local geometry from scratch, the added step contributes little beyond stochastic initialization.

**Point 4. Limited Generality and Lack of Conceptual Novelty**

The hierarchical coarse-to-fine paradigm is well-established in biomolecular modeling. Diffusion-based protein generators (e.g., RFdiffusion, Chroma) already generate backbones first and refine them via side-chain packing or relaxation, while hydrogens are routinely added in post-processing. Therefore, separating hydrogens or rigid scaffolds into multiple stages offers little conceptual advancement. The independence of stage-wise training is also not unique—it parallels conventional coarse-to-fine pipelines widely used in structural biology.

**Point 5. Unconvincing Energy-Based Evaluation**

The task of conformer generation is to recover physically valid ensembles, not to discover novel coordinates. Hence, energetic fidelity is central to evaluation. Yet the proposed method shows larger energy errors than comparable add-on methods (such as Woo et al. [1]), which operates on the same dataset. Although ET-Flow uses a different dataset, its much lower energy errors (0.18 / 0.02 kcal/mol) highlight the expected accuracy. Overall, MSGEN does not convincingly improve energetic realism relative to recent baselines.

[1] Woo, J., Kim, S., Kim, J. H., & Kim, W. Y. (2024). Riemannian Denoising Score Matching for Molecular Structure Optimization with Accurate Energy. arXiv preprint arXiv:2411.19769.

### Questions
**Point 1. Scaffold Arrangement Correction:**

How is the relative arrangement of scaffolds in stage 1 adjusted once the full molecular context becomes available? Is there any mechanism allowing later stages to reposition or re-orient scaffolds when linkers are longer or more flexible than expected?

**Point 2. Fair Comparison and Step Allocation:**

Could you provide all main results under the *equal-total-steps* setting? Also, how were the diffusion steps allocated between stages, and why does the 5 k-step version yield higher recall than 10 k?

**Point 3. Information Content of Upsampling:**

What specific geometric information do the upsampled atom positions convey? Have you tested simpler or physically guided alternatives (e.g., random initialization, distance-geometry placement, or short MM relaxation) to quantify their effect?

**Point 4. Novelty Beyond Coarse-to-Fine Tradition:**

In what sense does MSGEN offer conceptual or methodological advances beyond existing hierarchical generation schemes widely used in protein modeling? Is there any plan to automate hierarchy discovery or to demonstrate benefits in domains where coarse-to-fine design is already standard?

**Point 5. Energy-Based Evaluation:**

How were energy metrics in Table 3 computed, and can you include a direct comparison with SOTA models under identical settings?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces MSGEN, a hierarchical multi-stage diffusion framework for molecular conformer generation that explicitly models chemical geometry across multiple spatial resolutions. MSGEN decomposes the molecular generation process into a coarse-grained diffusion stage and a fine-grained diffusion stage conditioned on the former, allowing for better-informed generation conditioned on structural geometry. Empirical evaluations demonstrate that this hierarchical approach significantly improves geometric accuracy, structural validity, and diversity over single-scale baselines, and that MSGEN represents a flexible extension for improving existing diffusion models.

### Strengths
1. The work additionally provides a motivating study evaluating the performance gains achieved via geometric structural conditioning in conformer generation. 
2. The approach outperforms all baselines on conformer generation metrics, and the authors further demonstrate that the method outperforms GeoDiff alone using the same number of total diffusion steps, positioning MSGEN as a flexible extension for existing diffusion-based conformer generators. GeoDiff + MSGEN is also faster at sampling time than GeoDiff alone.
3. The authors justify novel developments and implementation details, including conditional augmentation, molecular upsampling, and 2-stage step allocation with rigorous ablations.

### Weaknesses
1. The model requires different parameter sets for each stage in the molecular generation, meaning that model size scales linearly with the number of stages. For a more meaningful comparison, it may be best to scale the benchmark models to a proportional parameter count increase.
2. Regarding the 2-stage framework adopted in the work's main experimental results, the use of anchor-based upsampling to place hydrogens given the heavy-atom backbone seems somewhat unnecessary. Wouldn't inferring their positions instead using any generic toolkit (e.g. RDKit, and perhaps adding some Gaussian perturbation) provide a stronger prior for the second stage?

### Questions
1. In table 5, are all methods also performed with the same total number of steps?
2. Would it be possible to train on all stages with a shared backbone that receives the stage $k$ as additional conditioning information? Would this hurt performance significantly?
3. The structural conditioning enabled by a 2-step generation process, starting with the backbone, seems to play a role somewhat similar to self-conditioning (see reference for implementation example), where the model conditions on its own ground-truth predictions. Do any of the baselines implement a similar ground-truth prediction conditioning mechanism? If not, it may be worthwhile to compare MSGEN to a baseline equipped with it.

Watson, J.L., Juergens, D., Bennett, N.R. et al. De novo design of protein structure and function with RFdiffusion. Nature 620, 1089–1100 (2023). https://doi.org/10.1038/s41586-023-06415-8

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MSGEN, a novel hierarchical multi-scale framework for 3D molecular conformer generation. The authors posit that existing deep generative models often fail by overlooking the inherent hierarchical structure of molecules, where key substructures like scaffolds act as anchors for the overall geometry. MSGEN addresses this by generating conformers in a coarse-to-fine process, first generating a coarse-grained structure and then using it as conditional guidance for subsequent, finer-scale stages. The authors demonstrate that this framework can be integrated with a wide range of existing generative models, consistently enhancing their ability to produce more stable, accurate, and chemically plausible conformers, especially for complex drug-like molecules.

### Strengths
1. **Chemically-Grounded Motivation:** The paper's premise is strongly rooted in chemical principles. The preliminary study (Section 3) provides excellent justification by showing that a model provided with ground-truth geometric guidance (the heavy-atom backbone) dramatically outperforms other methods, confirming that substructure awareness is critical.

2. **Novel and Necessary Technical Contributions:** The framework introduces two clever solutions to problems specific to this domain, e.g., Molecular Upsampling and Conditional Augmentation, which I find to be novel and reasonable.


3. **8Thorough and Rigorous Experimentation**: The paper is supported by a comprehensive set of evaluations: Geometric and Chemical Evaluation, Generalization, Scalability. The evaluation is not limited to geometric metrics: The authors show that MSGEN-enhanced models produce conformers with lower (better) mean absolute errors on calculated chemical properties like energy and the HOMO-LUMO gap, indicating the generated structures are more physically realistic.

### Weaknesses
1. **Potential for Error Propagation:** As with any hierarchical system, errors from the initial coarse stage can be passed on and potentially amplified by the subsequent fine-grained stages. The paper's failure case analysis (Appendix G.3) acknowledges this, suggesting that for highly flexible molecules, small deviations in positional guidance may lead to misaligned ring orientations. Though they argue that this may be raised by GeoDiff, empirical evidence with other backbone models are not provided.

### Questions
1. Can you show that MSGEN combined with other backbone models can solve (to some extent) Weakness 1?

2. Did you examine different design choices of molecule upsampling? Current design could be heuristic.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work propose a hierarchy molecule generation method. It first generate scaffold, than intermediate structure, and final full molecules. Each stages generations is upsampled and guide the generation of the next stage. Experiments and ablation studies on drug dataset reveals the effectiveness of the method.

### Strengths
1. Clear illustration of the method. Figure 1 shows 3 stages in the hierarchy. Figure 2 shows generation process in each stage and upsampling process between stages. Notations are also clearly defined in Section 3, 4.
2. Strong experimental results. Table 2,3,4 shows significantly better score than previous generative models. Table 5 further shows that the hierarchical generation strategy can improve performance of different basemodels.

### Weaknesses
1. Missing related work. Similar to this work, which split the whole generation process into different stages, [1] also propose a hierarchical method. It first generate global representation, and then use the global representation as guidance to generate the full molecule. It also achieves good performance on Drug dataset. Therefore, it is necessary to compare hierarchy design and experimental performane between this work and [1].
2. 3 stages needs 3 times inference time and parameters compared to vanilla 1 stage model with the same model size and diffusion inference strategy. Is the comparison with baseline conducted under fair setting? I think the experiments should keep the whole model size, total training time (controlled by training steps), and total inference time (controlled by diffusion steps) similar to the strongest baseline. 


[1] Zian Li, Cai Zhou, Xiyuan Wang, Xingang Peng, Muhan Zhang, Geometric Representation Condition Improves Equivariant Molecule Generation, ICML 2025.

### Questions
1. The three stage design are largely by heuristic rather than theoretical justification. Therefore, more empirical results should be provided to justify the reason for the hierarchy. Can we design finer hierarchy with more stages? Is there any alternative hierarchy tried?

### Soundness
3

### Presentation
3

### Contribution
3
