# Fast and Interpretable Protein Substructure Alignment via Optimal Transport

- Decision: Accept (Poster)
- Scores: 8, 4, 8, 4

## Abstract
Proteins are essential biological macromolecules that execute life functions. Local motifs within protein structures, such as active sites, are the most critical components for linking structure to function and are key to understanding protein evolution and enabling protein engineering. Existing computational methods struggle to identify and compare these local structures, which leaves a significant gap in understanding protein structures and harnessing their functions. This study presents PLASMA, the first deep learning framework for efficient and interpretable residue-level protein substructure alignment. We reformulate the problem as a regularized optimal transport task and leverage differentiable Sinkhorn iterations. For a pair of input protein structures, PLASMA outputs a clear alignment matrix with an interpretable overall similarity score. Through extensive quantitative evaluations and three biological case studies, we demonstrate that PLASMA achieves accurate, lightweight, and interpretable residue-level alignment. Additionally, we introduce PLASMA-PF, a training-free variant that provides a practical alternative when training data are unavailable. Our method addresses a critical gap in protein structure analysis tools and offers new opportunities for functional annotation, evolutionary studies, and structure-based drug design. Reproducibility is ensured via our official implementation at https://github.com/ZW471/PLASMA-Protein-Local-Alignment.git.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper addresses the challenge of identifying and comparing local protein substructures, such as active sites, which is critical for understanding protein function but difficult with existing global alignment or sequence-based methods. The authors introduce PLASMA, a deep learning framework that reformulates residue-level substructure alignment as a regularized optimal transport (OT) problem. The method consists of a "Transport Planner" that uses differentiable Sinkhorn iterations to generate an alignment matrix and a "Plan Assessor" to compute an interpretable, overall similarity score. The paper presents both a trainable (PLASMA) and a training-free (PLASMA-PF) variant. Experimental results on benchmark datasets show that PLASMA outperforms existing structure-based (e.g., TM-Align) and embedding-based (e.g., EBA) methods in detecting motifs, binding sites, and active sites, especially in cases of low global similarity. The method is also demonstrated to be significantly more computationally efficient than baselines.

### Strengths
Overall, the formulations are clear.

The prosed method is efficient and effective.

The reformulation of substructure alignment as a regularized optimal transport problem is a novel and well-justified approach for this domain. This formulation naturally accommodates the partial and variable-length matches common in biological substructures, which the paper correctly identifies as a limitation of standard OT-based alignment methods.

The strong performance of PLASMA-PF, which requires no task-specific training, makes the method highly practical for scenarios where labeled data is scarce.

The qualitative case studies presented in Section 6.4 and Figure 5 are convincing. They effectively demonstrate PLASMA's ability to identify biologically meaningful local alignments between proteins with very low sequence identity and different global structures, and correctly contrast these interpretable results with the baseline EBA.

### Weaknesses
The approach relies on embedding models like protein language models, which may be biased to certain types of proteins. This reliance would potentially result in problems the downstream alignment task. It would be beneficial to test PLASMA on underrepresented proteins, e.g., orphan proteins.

In table 1, the results are tested over 3 different seeds. More repetition would make the results more robust.

The justification for the "Plan Assessor" design (Section 4) feels somewhat heuristic. Specifically, the confidence weight $\alpha$ is derived using a 2D convolution with a fixed identity kernel to detect diagonal patterns. The rationale for this specific operator over other potential methods (e.g., a learnable kernel, or a different path-detection algorithm) is not fully explored, nor is the sensitivity to the kernel size.

The paper does not provide an ablation study on the key hyperparameters of the Sinkhorn algorithm. The temperature parameter $\tau$ and the number of iterations $T$ (Section 3) are critical for controlling the sparsity and accuracy of the final alignment matrix $\Omega$, as the paper itself notes. While Appendix D.2 mentions hyperparameter setup, no results are shown to demonstrate how performance (e.g., accuracy and alignment sparsity) varies with these choices.

The formulation of PLASMA-PF is ambiguous. The paper states that it "bypasses the siamese network and directly computes costs from $LN(H)$". It is not explicitly stated how this cost is computed. It is unclear if it still uses the hinge-loss formulation from Equation 2 with $\phi_{\theta}$ as an identity function, or if it uses a different distance metric (e.g., cosine distance) entirely.

### Questions
Could the authors please clarify the exact mathematical formulation of the cost matrix $\mathcal{C}$ used in the parameter-free PLASMA-PF variant? How is the cost computed "directly... from $LN(H)$," and how does this computation relate to Equation 2?

Could the authors elaborate on the claim that the LMS metric cannot be meaningfully applied to the EBA baseline? Since EBA produces residue-level alignments, it seems this metric should be applicable and would provide a valuable head-to-head comparison of alignment quality, not just score-based classification performance.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper works on identifying and understanding protein structure for efficient and interpretable residue-level protein substructure alignment. The authors reformulate the problem as a regularized optimal transport task and leverage differentiable Sinkhorn iterations to solve the alignment problem. Besides, the authors use extensive quantitative evaluations and three biological case studies to demonstrate that the proposed method, PLASMA, achieves accurate, lightweight, and interpretable residue-level alignment. Moreover, the authors propose a training-free variant as an alternative when training data is not available.

### Strengths
1. The challenges behind protein alignment are well defined, and the motivation is clearly demonstrated.
2. Extensive experiments on various datasets, baselines, and backbone models clearly prove the effectiveness of the proposed method.
3. The paper is well written and organized, with code published.

### Weaknesses
1. Using optimal transport to align two graphs is not a novel idea, as can be seen in papers[1][2][3]. What are the differences and advances of PLASMA compared to other methods?

2. The authors claim that PLASMA can sufficiently address the variable length alignment challenge, but there are no experiments to prove this statement. What is the performance of PLASMA when the query and target protein have different lengths?

3. There is no ablation study to show the effectiveness of each component in PLASMA.

4. In Eq. 7, the authors want to introduce a loss that focuses exclusively on labeled substructures. Then how does the number of labeled substrcutres affect the performance of PLASMA? And is it necessary to let the model focus on these labeled substructures? What are the reasons to include this loss?

[1] Lee, John, et al. "Hierarchical optimal transport for multimodal distribution alignment." Advances in neural information processing systems 32 (2019).
[2] J. Tang, W. Zhang, J. Li, K. Zhao, F. Tsung and J. Li, "Robust Attributed Graph Alignment via Joint Structure Learning and Optimal Transport," 2023 IEEE 39th International Conference on Data Engineering (ICDE), Anaheim, CA, USA, 2023, pp. 1638-1651, doi: 10.1109/ICDE55515.2023.00129.
[3] A. T. Riahi, G. Woollard, F. Poitevin, A. Condon and K. D. Duc, "AlignOT: An Optimal Transport Based Algorithm for Fast 3D Alignment With Applications to Cryogenic Electron Microscopy Density Maps," in IEEE/ACM Transactions on Computational Biology and Bioinformatics, vol. 20, no. 6, pp. 3842-3850, Nov.-Dec. 2023, doi: 10.1109/TCBB.2023.3327633.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This submission introduces PLASMA, a method to obtain protein substructure alignment. By embedding each residue of a pair of proteins with a pLM, a soft alignment matrix coupled with an overall alignment score. The alignment matrix is obtained by passing the pLM embeddings through an optional siamese network yielding a cost matrix, to which Sinkhorn iterations are applied. From the resulting alignment matrix, an interpretable score is calibrated to surface regions of high structural similarity. Experiments show very high performance when benchmarked on substructure identification tasks.

### Strengths
The paper tackles a relevant problem in the field of protein representation learning. The paper is clearly written, the benchmarks are sound and convincing, and the methodology is original. PLASMA clearly demonstrates superior performance while simultaneously also being much faster than some of the previous methods. The VenuX benchmark is a strong set of experiments that clearly show generalization abilities. The testing of various pLMs show that the method is robust and adaptable.

### Weaknesses
Major comments:
- The claim that PLASMA is "the first deep learning framework for efficient and interpretable residue-level protein substructure alignment" is overstated. Clearly there is prior work for the kind of task tackled here (EBA, pLM-BLAST). Foldseek also falls under the category of models described here. epLSAP-Align (https://doi.org/10.1093/bioinformatics/btaf309) is also very similar and should be cited given it leverages the OT formulation on structures directly. Another related method is ActSeek, which should also be cited: https://pmc.ncbi.nlm.nih.gov/articles/PMC12343037.
- Although a preprint, I would already cite Folddisco as concurrent work, as it fulfills the same nice and needs https://www.biorxiv.org/content/10.1101/2025.07.06.663357v1. If the authors can benchmark against this method it would be great, but not strictly necessary. 
- More generally, I have a feeling that the writing could benefit from a tighter list of contributions in the introductions to clearly contrast the contributions of the work (mostly the machinery on top of the pLM and siamese networks as well as some other mentioned technical contributions like the losses and scores) to other works.

Most of the weaknesses are nitpicks, here are some things I would like to see improve in terms of presentation:
- I think the results of test_extra currently in Appendix G are the most relevant results since effective generalization is most relevant. Swapping this with table 1 is I think more appropriate, especially considering those results still show strong performance.
- It would be great to have a concrete percentage for the colors relative to TM-Align.
- The contrast ratio for the colors in figure 5 is pretty bad and barely readable when printed. The squares subsetting the proteins could be also bigger for the proteins.
- There is some interchangeable terminology which can be made more precise. "substructure," "local structure," "motif," "functional site," "active/binding site" are used interchangeably but they are not completely: motifs can be discontinuous, binding sites often involve side chains only, “substructure” could mean a whole domain.
- The description of the diagonal kernel could be made a bit more explicit by highlighting it surfaces continuous diagonal elements.
- "which causes troubles" needs to be reformulated, e.g. to "which causes problems".
- appendix references should be made more explicit, indicating more precisely what's in each of them.

### Questions
The plan assessor does a kxk identity deconvolution to detect motifs, but this assumes sequence continuity on both proteins, which would downweight cases where there are structural motifs that are non continuous. Is it possible to remove this component and obtain confidence values on the motifs without it? Can the authors show that diagonal patterns still emerge from diagonal motifs for pLMs?
- Is it possible to have ablations on the different components of the plan assessor? There are a few elements in there and it would be good to have some insights into the impact of those elements on performance.

### Soundness
3

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
PLASMA is novel protein local structural alignment algorithm that uses differentiable optimal transport method with trainable cost matrix. The method is very fast at inference and outperforms baselines on the presented examples. Unfortunately, only one (yet unpublished) benchmark has been used for evaluation, excluding more traditional choices used in the community.

### Strengths
- The manuscript is well written and motivated
- The SOTA is exhaustive and correct
- The baselines are well chosen
- Both interpolation and extrapolation results are presented
- The method provides interpretability by normalized similarity scores and alignment matrices
- Parameter-free version of the network is also presented and demonstrates a strong performance

### Weaknesses
- The individual components and ideas of the method are not novel
- The paper does not discuss nor demonstrates examples with non-sequential alignments. This can be the actual advantage of the presented algorithm. While traditional sequence alignment (like SW or NW algorithms) is order-preserving (they align sequence A on B while preserving the residue order in both A and B), the optimal transport (OT) is in general not order preserving. Please consider case of fold-switching proteins, for example.
- More traditional structural alignment benchmarks could have been also used, examples include BALIBASE 2, BALIBASE 3, HOMSTRAD, OXBENCH, SISYPHUS
- no multiple structural alignment is discussed
- Training/test splits based on InterPro families may have data leakage, as different families may still share identical local structural motifs or even domains.

### Questions
- please see Weaknesses
- "the sequence identity between training and test proteins is kept below 50%."  such a split may not be sufficient to exclude data leakage. Anyway, the authors provide performance data for TM-score similarities below 0.5.
- can you also show TM-align and FoldSeek results at different levels of TM-score similarity as additional baselines in Fig. 3B?
- the classical Sinkhorn algorithm has the complexity of O(N^2 log N) for a fixed tolerance. Can you please support your O(N^2) claim, provide more details on the chosen tolerance and provide additional numerical experiments?

### Soundness
3

### Presentation
3

### Contribution
3
