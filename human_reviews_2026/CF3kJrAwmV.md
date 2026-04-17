# BioBO: Biology-informed Bayesian Optimization for Perturbation Design

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 6

## Abstract
Efficient design of genomic perturbation experiments is crucial for accelerating drug discovery and therapeutic target identification, yet exhaustive perturbation of the human genome remains infeasible due to the vast search space of potential genetic interactions and experimental constraints. Bayesian optimization (BO) has emerged as a powerful framework for selecting informative interventions, but existing approaches often fail to exploit domain-specific biological prior knowledge. We propose Biology-Informed Bayesian Optimization (BioBO), a method that integrates Bayesian optimization with multimodal gene embeddings and enrichment analysis, a widely used tool for gene prioritization in biology, to enhance surrogate modeling and acquisition strategies. 
BioBO combines biologically grounded priors with acquisition functions in a principled framework, which biases the search toward promising genes while maintaining the ability to explore uncertain regions. 
Through experiments on established public benchmarks and datasets, we demonstrate that BioBO improves labeling efficiency by 25-40\%, and consistently outperforms conventional BO by identifying top-performing perturbations more effectively. Moreover, by incorporating enrichment analysis, BioBO yields pathway-level explanations for selected perturbations, offering mechanistic interpretability that links designs to biologically coherent regulatory circuits.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes _Biology-Informed Bayesian Optimization (BioBO)_, which addresses challenges in drug discovery that require genomic intervention experiments. The authors introduce two key innovations: (1) a multimodal gene representation to improve the surrogate model in Bayesian Optimization (BO), and (2) the integration of enrichment analysis (EA) into a biology-prior–augmented acquisition function through a principled πBO framework. This integration provides a theoretical “no-worse” guarantee and establishes a foundation for incorporating broader sources of biological knowledge in similar ways.

Experiments on real-world problems demonstrate the effectiveness of both proposed components — the multimodal embeddings and the biological priors. The results produced by BioBO are also biologically interpretable.

### Strengths
- The paper addresses a well-motivated and highly important problem.
    
- It integrates EA-based experimental design common in biology into the πBO framework,  bridging knowledge from two domains, and offers a principled non-worse theoretical guarantee.
    
- The ablation studies sufficiently demonstrate the contributions of the multimodal representations and the EA integration. **Furthermore, the genes selected by BioBO align with known biological interpretations of central regulatory genes, indicating strong real-world potential.**
    
- The multimodal surrogate model is modular, allowing a wide range of biological and machine learning encoders, which enhances its practicality for real-world applications.

### Weaknesses
- According to the appendix, the width of the error bars is ±1 s.e.m., which is less convincing compared to the standard ±1.96 s.e.m. used for 5% significance. Could the authors justify this choice?

- _(Minor)_ The generality of the method is limited. Although the proposed approach is termed “Biology-Informed BO,” it applies primarily to biological problems where EA is feasible. The authors might consider adjusting the algorithm’s name to better reflect its domain specificity.

### Questions
1. The multimodal design appears to involve simple concatenation of multiple gene embeddings. Is there a specific reason for choosing this approach to combine different embeddings?
    
2. There is little discussion of the individual contributions of each gene embedding or the rationale for selecting each. Could the authors provide more insight into how one should choose among available gene embeddings?
    
3. If I understand correctly, the so-called “biological prior” is derived from statistical patterns in the training data, rather than from external biological knowledge. Could the authors clarify how EA can still enhance BO performance under this setup, even without access to external priors?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes BioBO* a biology-informed Bayesian optimization framework for efficient design of genomic perturbation experiments. The goal is to prioritize which gene knockouts to test in costly CRISPR assays. BioBO integrates two main ideas:
1. **Multimodal gene embeddings (Fusion)** – combining Achilles (functional dependency), Gene2Vec (co-expression), and GenePT (text-derived) representations to improve surrogate modeling.
2. **Enrichment analysis (EA)** – incorporating biologically meaningful pathway priors into acquisition functions via the π-BO framework (Hvarfner et al., 2022), guiding the search toward biologically coherent gene clusters.

The method is evaluated on five CRISPR datasets from GeneDisco, showing 25–40% labeling efficiency improvement over standard BO, and stronger enrichment signals in discovered pathways. The authors also provide theoretical guarantees that integrating EA priors does not worsen regret asymptotically.

### Strengths
- **Strong biological motivation**: Addresses a key challenge in experimental design for genomics, where exhaustive perturbation is infeasible.
- **Principled integration of biology and machine learning**: The use of the π-BO formalism to incorporate enrichment priors is theoretically grounded and non-trivial.
- **Novel multimodal surrogate modeling**: Demonstrates empirically that fusing heterogeneous biological embeddings (Achilles + Gene2Vec + GenePT) yields better performance near optima.
- **Comprehensive experiments**: The evaluation on multiple datasets (IFN-γ, IL-2, Tau, NK, SARS-CoV-2) with ablations and sensitivity analyses is thorough and carefully executed.
- **Interpretability and biological validation**: The enrichment results (e.g., MYC and E2F target pathways) provide mechanistic insight into immune regulation and show that the discovered perturbations are biologically meaningful.
- **No-harm theoretical guarantee**: Retains the asymptotic regret bounds of standard BO, ensuring robustness to potentially noisy enrichment priors.

### Weaknesses
- **Limited novelty in BO theory**: The core algorithmic novelty is the integration of known biological priors within an existing π-BO framework; the BO component itself remains myopic and standard.
- **Empirical improvements modest in some datasets**: While fusion provides consistent gains, improvements are smaller or marginal in certain conditions (e.g., BioEI on IL-2), suggesting dataset-dependent benefits.
- **Dependence on predefined pathway databases**: The approach relies on GO and Hallmark annotations; biases or incompleteness in these resources may limit generalization, especially in less-studied cell types.
- **Simple fusion strategy**: Concatenation is effective but basic. More principled fusion (e.g., learned or attention-based integration) could enhance representation quality.
- **No analysis of computational cost**: Training BNN surrogates with multimodal embeddings and enrichment updates may be computationally heavier than standard BO, but runtime comparisons are not reported.

### Questions
1. **Prior calibration**: How sensitive are the results to the choice of temperature parameter *t* and β in Eq. 4 and Eq. 6? The appendix shows some analysis, but can the authors comment on interpretability of these hyperparameters across datasets?
2. **Fusion mechanism**: Have the authors considered learning the fusion weights (e.g., via gating or attention) instead of static concatenation? Would that risk overfitting given small training sets per iteration?
3. **Pathway dependence**: Since GO and Hallmark differ in coverage and granularity, could BioBO dynamically switch or ensemble multiple enrichment sources?
4. **Scalability**: How would BioBO scale to genome-scale design (>20k genes) when multiple priors and embeddings are used? Are there computational bottlenecks?
5. **Biological interpretability**: Can the authors provide further examples where enrichment priors led to selection of underexplored but biologically novel genes, beyond MYC/E2F modules?

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
4

### Summary
This paper proposes Biology-Informed Bayesian Optimization (BioBO) for designing genomic perturbation experiments. The method makes two main contributions: (1) integrating multimodal gene embeddings (Achilles, Gene2Vec, GenePT) to improve surrogate modeling, and (2) augmenting acquisition functions with biological priors from enrichment analysis using the πBO framework. Experiments on GeneDisco benchmarks show 25-40% improvement in labeling efficiency compared to standard BO, with the method identifying biologically coherent pathways. The authors demonstrate that multimodal embeddings improve surrogate model performance particularly near the optimum, and provide theoretical no-harm guarantees for their enrichment-augmented acquisition strategy.

### Strengths
1.	The combination of multimodal gene embeddings with enrichment-based priors in BO is novel for this application domain. The analysis showing that surrogate model performance near the optimum (rather than globally) correlates with BO performance provides useful insights.
2.	The experimental evaluation is thorough, covering five datasets across multiple modalities and acquisition functions with proper error estimation. The theoretical grounding via πBO provides principled guarantees. The correlation analysis between surrogate quality and BO performance is well-executed.
3.	The paper clearly motivates the problem and explains why existing BO methods for gene perturbation are limited. The two-pronged approach (better embeddings + biological priors) is logical and well-presented.
4.	The problem of efficient gene perturbation design is important for drug discovery. The 25-40% improvement in labeling efficiency could translate to meaningful cost savings. The interpretability aspect through enrichment analysis is valuable for practitioners.

### Weaknesses
1.	Limited Technical Novelty: The core techniques are straightforward applications of existing methods. Simple concatenation for multimodal fusion is the most basic approach; more sophisticated fusion architectures (attention mechanisms, learned fusion) are not explored. The enrichment prior construction (Eq. 6) is relatively simple. The paper primarily validates that combining these existing techniques works for this domain rather than introducing new methodology.
2.	Insufficient Biological Validation: Only one detailed biological case study (IFN-γ, Table 2) is provided. The biological interpretations feel somewhat post-hoc rather than predictive. The paper would benefit from: (1) validation with domain experts, (2) wet-lab validation of predictions, (3) deeper mechanistic explanations beyond pathway enrichment, and (4) analysis of failure cases where the method selects biologically implausible genes.
3.	Experimental Limitations: The experiments are restricted to GeneDisco datasets which represent a limited evaluation scope. Key concerns include: (1) DiscoBAX performing poorly contradicts Lyle et al. 2023 without adequate explanation, (2) high variance in some results suggests instability, (3) no comparison with other multimodal fusion approaches or alternative biology-informed methods beyond pure EA, (4) computational costs are not reported, (5) the choice of hyperparameters (beta, temperature, top-10%) lacks principled justification.
4.	Analysis Depth: The correlation analysis reveals interesting patterns but lacks deeper explanation. Why does global LL negatively correlate with BO performance? Why does fusion specifically help near the optimum? The paper states these observations but does not provide mechanistic understanding. The no-harm guarantee is asymptotic, but finite-sample behavior is not analyzed. The sensitivity to beta (Appendix C) shows significant variation that undermines the robustness claims.
5.	Scalability and Generalization: All experiments use the same 10,556 genes common across all embeddings, which is a restricted setting. How does the method perform with missing modalities? What about other organisms beyond human? The batch size is always 1, but practical screens often label batches; how does this affect the approach? The paper does not address these practical considerations.
6.	Presentation Issues: While generally clear, some aspects need improvement: Figure 3 and similar correlation plots are central to the argument but the negative global correlation is counterintuitive and poorly explained. Some claims are overstated (e.g., "dramatically stronger enrichment signals" in Section 4.5 when comparing different experimental settings). The main paper could better synthesize the extensive appendix results.

### Questions
1.	Can you provide more detailed analysis or ablation studies on why multimodal fusion specifically helps near the optimum but not globally? What properties of the fused representation drive this?
2.	The DiscoBAX performance is substantially worse than reported in Lyle et al. 2023. Can you investigate and explain this discrepancy? Is it due to implementation differences, hyperparameters, or dataset characteristics?
3.	How sensitive are the results to the choice of top-k% for enrichment analysis? You use 10% but provide no justification or sensitivity analysis for this critical hyperparameter.
4.	Can you provide computational cost comparisons between BioBO and baseline methods? How does the enrichment analysis overhead scale?
5.	For the biological interpretation (Table 2), can you provide additional case studies beyond IFN-γ? Are there cases where the method fails to identify biologically meaningful pathways?
6.	How does performance change when some modalities are missing? This is a practical scenario but not addressed.
7.	Have you validated any of the predicted gene perturbations experimentally or with domain experts? The biological interpretations would be more convincing with external validation.
8.	Why use simple concatenation for fusion rather than more sophisticated approaches like attention-based fusion or learned gating mechanisms? Have you experimented with alternatives?
9.	The beta parameter shows significant impact on performance (Appendix C). Can you provide principled guidelines for setting beta, or ideally, an adaptive approach?
10.	How does the method perform with batch acquisition (B > 1), which is more common in practical CRISPR screens?

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
3

### Summary
This paper proposes to enhance Bayesian optimization for genomic perturbation experiment design with multi-modal gene embeddings and enrichment analysis, essentially incorporating biological knowledge as prior to Bayesian optimization.

### Strengths
- The methodology component of the paper is generally easy to follow and clearly presented. Figure 1 is particularly helpful for a high level perception of the proposed framework.

- Ablation studies are presented in the experiment section, which helps the understanding of each component's contribution.

### Weaknesses
- Novelty: I think novelty of the proposed methodology is lacking. This work is more about how to incorporate biological knowledge and analysis into existing methods such as $\pi$BO. I think sometimes combining existing methods could be interesting, but if it's more of a realization of an approach in a specific domain, I think it's less interesting. To be honest I'm a little surprised that this work was submitted to this venue instead of some computational biology journal, since it has a quite heavy dose of biological content and analysis.

- Clarity & practicality: Although the methodology component of the paper is generally easy to follow and clearly presented, I find the biological experiment setup not so easy to follow (as a person who does not have abundant biology knowledge). From Figure 1, it seems that there is an online component of the UCB update (wet lab perturbations) being conducted, but I assume this is not actually carried out in the experiment section? How are the unlabeled perturbations evaluated online given a dataset in the experiment? And is the proposed method with wet lab perturbation actually practical given the significant efficiency concern with 20 cycles?

### Questions
Please see above

### Soundness
2

### Presentation
3

### Contribution
2
