# Efficient Prediction of Large Protein Complexes via Subunit-Guided Hierarchical Refinement

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6, 6

## Abstract
State-of-the-art protein structure predictors have revolutionized structural biology, yet quadratic memory growth with token length makes end-to-end inference impractical for large complexes beyond a few thousand tokens. We introduce HierAFold, a hierarchical pipeline that exploits the modularity of large complexes via PAE-guided (Predicted Aligned Error) subunit decomposition, targeted interface-aware refinement, and confidence-weighted assembly. PAE maps localize rigid intra-chain segments and sparse inter-chain interfaces, enabling joint refinement of likely interacting subunits to capture multi-body cooperativity without increasing memory. HierAFold matches AlphaFold3 accuracy, raises success rates from 49.9\% (CombFold) to 73.1\% on recent PDB set. While for large complexes, it cuts peak memory by $\sim$25\,GB on a 4,000-token target ($\sim$40\%), successfully models complexes with over $5{,}000$ tokens that are out-of-memory for AlphaFold3, and raises success rates by two-fold compared with CombFold.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the computational complexity issue in AlphaFold by presenting a hierarchical pipeline, refered to as HieraFold, which decomposes the end-to-end structure prediction task in a coarse-to-fine manner.
HieraFold first performs a coarse global prediction using a "lightweight" version of AlphaFold with a smaller diffusion module, and then locally refines critical subunits identified via the pAE matrix.
Experimental results on protein-protein and protein-ligand benchmarks demonstrate the effectiveness of the proposed method.

### Strengths
1. This paper addresses an important research problem: reducing the peak computational complexity of AlphaFold, which is critical for real-world applications involving very large protein complexes.

2. The proposed method for identifying critical subunits does not rely on expert curation, making it more generalizable and practical.

3. Experimental results on large complexes demonstrate substantial improvements over existing methods.

### Weaknesses
1. **Writing and organization.** Although generally well structured, the manuscript would benefit from improvements in writing. Specific issues include:  
   * Incorrect use of hyphens and dashes;  
   * Some paragraphs are overly wordy;  
   * The Introduction section discusses the literature extensively but there is no separate “Related Works” section. It is important to maintain the distinct roles of these sections rather than combining them;  
   * Line 249: “focus chai” should be corrected to “focus chain.”  

2. **Computational and time cost.** The multi-stage pipeline leads to higher computational overhead and longer runtime. A more detailed analysis of the trade-off between the ability to handle large complexes and the additional time and computation required would strengthen the paper.  

3. **Insufficient validation of design choices.** Some key design decisions are not thoroughly supported by ablation studies, including:  
   * The choice of pAE over other confidence metrics output by AlphaFold;  
   * Whether confidence-metric-based subunit identification indeed outperforms manually designed approaches;  
   * The necessity of the refining stages (Stages 2 and 3), i.e., how much improvement does the full pipeline provide over the coarse stage alone.

### Questions
1. A fundamental question: why do low-pAE regions naturally correspond to critical subunits? Please provide an intuitive explanation of the rationale behind this choice.  

2. In Figure 2, why is C1 retained while C2 and C3 are omitted?  

3. How does the performance gap between HieraFold and the baseline change as the size of the input complex increases?

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
2

### Summary
This paper presents HIERAFOLD, which exploits the modularity of large complexes via PAE-guided (Predicted Aligned Error) subunit decomposition, targeted interface-aware refinement, and confidence-weighted assembly. By leveraging coarse predictions and PAE-guided modular decomposition, the method automatically identifies subunits and interfaces, and refines each chain in the context of its key interacting partners. Experiments from various benchmarks clearly show that HIERAFOLD matches AlphaFold3 on standard benchmarks and, critically, extends tractable prediction to complexes exceeding 5,000 tokens, achieving substantial gains over prior divide-and-conquer approaches.

### Strengths
1. The challenges and motivation of the proposed method are clearly shown and demonstrated.

2. Various experiments are conducted to show the effectiveness of the proposed method.

3. The paper is well written and organized.

### Weaknesses
1. The decomposition is a good way to save memory usage. However, the paper only uses AlphaFold3 as the backbone model to test the method. An extension to other backbone methods will help better demonstrate the effectiveness.

2. More baselines should be added, such as OpenFold and MoLPC, to demonstrate the effectiveness of the proposed methods better.

3. The code is not publicly available.

### Questions
1. What is the computation cost of HIERAFOLD? Though the decomposition of long tokens into several small subunits is effective at reducing the memory cost. It ends up with a lot of extra computation. Is the extra computation cost acceptable compared to the saved memory usage?

2. How will the split k in PAE-Guided Subunit Segmentation for Biological Modularity affect the performance? And how much difference for the split part? Is the split showing a significant difference between the two split parts?

### Soundness
2

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
5

### Summary
HIERAFOLD: Hierarchical pipeline for efficient large complex prediction using PAE decomposition & refinement.

### Strengths
The work addresses a critical and well-known scalability bottleneck in modern protein structure prediction models, which suffer from quadratic complexity (O(N^2)) for large protein complexes. The originality of the method is high, centered on the Predicted Aligned Error (PAE)-guided hierarchical decomposition of large complexes into smaller, manageable subunits. This is a creative, biologically motivated, and highly relevant solution that directly exploits the inherent modularity of protein structures. The quality of the method is evidenced by the thoughtful design, particularly the targeted interface-aware refinement strategy, which focuses computational power precisely where inter-chain interactions are most critical, thus optimizing the accuracy-efficiency trade-off. The significance of HIERAFOLD is substantial, as it practically extends the application of highly accurate prediction methods to systems exceeding a few thousand residues, which is a major advancement for predicting large, biologically relevant assemblies.

### Weaknesses
1.Sensitivity to Initial Subunit Quality：The hierarchical pipeline is fundamentally limited by the quality of the initial prediction for the subunits. The current approach implicitly assumes the intra-subunit structure is largely correct and focuses its efforts on inter-subunit interfaces. The method lacks a mechanism to effectively identify or recover from errors in the internal structure prediction of the subunits themselves. This poses a significant failure mode, as local structural errors within a subunit cannot be easily rectified during the refinement or re-assembly steps, especially if they are far from the interface.

2.Decomposition Robustness and Structural Breaks: While efficient, the PAE-based decomposition strategy carries a risk of introducing structural discontinuities or breaks at the boundaries of the artificially created subunits, particularly in cases of highly dynamic, ambiguous, or tightly interwoven (non-modular) interfaces. This potential for introduced structural artifacts needs a more robust discussion and mitigation strategy.

### Questions
1.Addressing Intra-Subunit Errors: Given that the method assumes correct intra-subunit structure, how would HIERAFOLD perform if the base model prediction for one of the subunits was highly inaccurate (e.g., due to novel topology or unusual folding)? Could the targeted refinement be adapted to dedicate a small amount of computational budget for intra-subunit refinement based on a local confidence score (e.g., pLDDT)?

2.Decomposition Sensitivity Analysis: How is the threshold for PAE-guided subunit decomposition determined? Is it a fixed value, or is it dynamically tuned? Please provide a sensitivity analysis showing the final accuracy (e.g., TM-score, Interface TM-score) as a function of this PAE decomposition threshold to address the concern about structural breaks.

3.Assembly Mechanism for Flexibility: Can the "confidence-weighted assembly" mechanism explicitly handle the difference between rigid-body movements of subunits (which require global transformation) and local, flexible changes (which require local backbone/side-chain adjustments)? How does the assembly process prevent structural strain introduced during the re-assembly of independently refined subunits?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a new approach for predicting the structure of large protein complexes by decomposing them into modular subunits using Predicted Aligned Error (PAE) scores. The proposed approach uses a 3-stage process from coarse to split the segments, then performing fine prediction using existing models and the final alignment. It achieves similar accuracy to AlphaFold3 with lower GPU memory requirements for large protein complex.

### Strengths
- The paper addresses a difficult challenge of predicting the structure of large protein complexes.

- The proposed approach appears to improve both accuracy and reduce memory requirements.

- The three stages of coarse, fine, and assembly are modular and allow for examining individual stages. Overall, the approach is well constructed.

### Weaknesses
- The proposed approaches rely on existing approaches in protein modeling and do not advance the area significantly, nevertheless, they provide an application to an important problem. 

  - The reliance of PAE for segmentation is one of the main drawbacks of this approach. PAE is an estimated alignment error and may not fully capture the inter-domain interactions, leading to errors in clustering based only on PAE.

  - Since the main comparison in this paper was done between hierafold and combfold, it would be helpful to compare against the dataset used in the combfold paper.
  
  - Description of the dataset and performance variation with respect to homology, number of conformations, or known multi-state assemblies is missing. The paper would be strengthened if it explicitly demonstrated performance on disordered/flexible domains.

  - A better quantification or analysis of the identified sub-unit from segmentation is not provided. This will also allow to examine if common sub-structures are being identified as subunits across different proteins.

### Questions
Details of the dataset are very sparse. Provide more details, including the number and lengths of sequences across the 3 datasets, in the manuscript.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The method HIERAFOLD provides a memory-efficient and accurate solution to the prediction of large macromolecular complexes. It uses an optimised version of Protenix, an open-source reproduction of AlphaFold3, to generates 3D models of subparts, a subunit segmentation strategy based on PAE, and a combinatorial algorithm for selecting and assembling subparts. It performs favourably compared to AlphaFold3 and CombFold.

### Strengths
- The methodology and experiments are sound and clearly explained
- The success rate is much better than the state-of-the-art methods

### Weaknesses
- The main concern I have is whether this work actually fits within the scope of ICLR. It is not a machine learning contribution per se, as it relies on an already developed diffusion-based SOTA protein structure predictor. The contribution mainly focuses on algorithms for segmenting, selecting and combining subparts, without any representation learning involved.

### Questions
- Could the authors discuss other methods for segmenting proteins into domains, or rigid subunits? In particular those developed recently for annotating the AlphaFold Database?
- Could the authors comment on the influence of the content in intrinsically disordered regions on the performance of the method?

### Soundness
4

### Presentation
4

### Contribution
4
