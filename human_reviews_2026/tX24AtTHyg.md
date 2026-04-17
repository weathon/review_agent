# GitPatchDB: A Large-Scale GitHub Commit Databank for Vulnerability Patch Analysis

- Decision: Reject
- Scores: 4, 4, 8, 4

## Abstract
Machine learning based vulnerability detection relies on datasets that link vulnerabilities to their corresponding patches. However, existing resources such as Common Vulnerabilities and Exposures (CVE) often lack reliable patch references, e.g., many CVE entries do not provide patch commits, and a significant share of existing commits become inaccessible due to code repository changes. To bridge this gap and better facilitate vulnerability detection, we curate GitPatchDB, a large-scale, semantic-rich dataset that pairs CVEs with their corresponding patch commits, where each commit is formatted not only as code diffs but also as interprocedural program slices generated through program slicing and related program analysis techniques. To leverage this semantic-rich dataset, we further propose Contrastive Natural-language Programming-language Pre-training (CNPP), a novel approach that enables multimodal vulnerability patch search via contrastive learning. Extensive evaluations demonstrate that GitPatchDB paired with CNPP achieves 95.99% accuracy in vulnerability patch search, surpassing baseline methods by over 8% and establishing a new state-of-the-art performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
- This paper introduces GitPatchDB, a large-scale and semantic-rich dataset for vulnerability patch search.
- This paper proposes on-the-fly patch program co-analysis, that combines forward/backward program slicing with flow-sensitive pointer analysis.
- This paper proposes Contrastive Natural-language Programming-language Pre-training (CNPP), used for vulnerability patch embedding.

### Strengths
- This paper focusing on addressing an important question
- The motivating example of this paper is convincing

### Weaknesses
1. The novelty of "flow-sensitive pointer analysis". It has been proposed for many years, and there exists a large number of work [1, 2, 3] on this task.
Besides flow-sensitivity, field-sensitivity and context-sensitivity are also important in pointer analysis, the rationale of focusing only on flow-sensitivity needs explanation.

2. The definition of "multimodal". Generally, it means combining both text and non-text information for learning. In common knowledge, we do not call it "multimodal" when combining both natural language and code.

3. The evaluation setting. Ideally, it is proper to retrieve a vulnerability patch from all code commits in a specific repository instead of from a set of vulnerability patches. This paper can be substantially strengthened via adding a large number of negative samples for patch retrieving (Note that the proportion of vulnerability patch among all vulnerability patches is about 1:400 [VulFixMiner]).

#### Minor Issues
In Line 113, the sentence is incomplete.

[1] Hardekopf B, Lin C. Flow-sensitive pointer analysis for millions of lines of code[C]//International Symposium on Code Generation and Optimization (CGO 2011). IEEE, 2011: 289-298.

[2] Hardekopf B, Lin C. Semi-sparse flow-sensitive pointer analysis[J]. ACM SIGPLAN Notices, 2009, 44(1): 226-238.

[3] Wilson R P, Lam M S. Efficient context-sensitive pointer analysis for C programs[J]. ACM Sigplan Notices, 1995, 30(6): 1-12.

### Questions
1. The main question of this paper is the application scenario. This paper proposes a Contrastive Natural-language Programming-language Pre-training(CNPP) as vulnerability patch embedding, and evaluates it on a CVE-commit mapping task. Thus, why this task is important for open-source software community? 
2. What is the data-collection details of the proposed dataset, GitPatchDB? For example, how the CVE entries are collected and how their mapped patch commit is collected?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes GITPATCHDB, a large-scale benchmark that pairs CVE and the corresponding patch commits. Compared to existing benchmarks, GITPATCHDB enriches code diff in commit information with interprocedural program slices. The paper further presents a vulnerability patch searching technique based on contrastive natural-language programming-language pre-training. The results outperform baselines in top-10 accuracy.

### Strengths
The paper presents a large-scale dataset for CVE and patch pairs. 

The paper leverages program analysis to optimize the context of vulnerability patch.

### Weaknesses
One of the major technical contributions of GitPatchDB (e.g., pointer analysis and program slicing) does not lie in the AI domain, which instead is application of existing program analysis techniques.   

The improvement of the approach over existing embeddings (e.g., Qwen3-embedding without task-specific tuning) is marginal. Qwen3 even outperforms the proposed approach at Top2 to Top-4. The results could hurt the necessity of the proposed compound encoders. 

The construction process of the dataset and experimental settings are unclear. When the dataset GitPatchDB (i.e., with 14,575 samples) is constructed, how is the ground-truth labelled? As mentioned by the paper, the majority of CVEs do not have the linked patch commits. Therefore, it is unclear how the CVEs and patch commits get paired in the GitPatchDB. Moreover, as mentioned in line 365,  the dataset is split into training/validation/test set. It is unclear how the ground-truth labels are obtained. 

Unclear experimental designs. As mentioned in line 372, the retrieval effectiveness experiments use CVE descriptions and pre-patch slices. What do “pre-patch slices” refer to here? The input of retrieval should be the CVE descriptions, and the patches should be retrieval outputs.  

The datasets include the CVE spanning only to 2023, without including more recent data. 

Presentation error: broken sentence “rendering it unsuitable for”

### Questions
1.	How are the ground-truth labels (the pair relation between CVE and patches) in the 14,757 dataset obtained? 

2.	What is the overhead and the accuracy of the program analysis part? 

3.	What is setting for the retrieval effectiveness?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses a critical gap in vulnerability detection research: the lack of reliable links between CVEs and their corresponding code patches. To solve this, the authors make two core contributions:

GITPATCHDB: A large-scale, semantic-rich dataset with 14,575 total samples (12,629 CVE-commit positive pairs) spanning 3,071 open-source repositories and 10 programming languages. It enriches patch context via interprocedural program slicing and flow-sensitive pointer analysis, avoiding the limitations of syntactic-only diffs in prior datasets.

CNPP: A multimodal approach that aligns natural-language CVE descriptions with code patches via contrastive learning. It uses hierarchical BiLSTM+attention to handle long token sequences and fuses cross-modal information into a shared semantic space.

Extensive experiments show GITPATCHDB+CNPP achieves 95.99% Top-10 accuracy in vulnerability patch search, outperforming baselines like PatchScout (87.73%) and VCMatch (92.22%) by 3.77–8.26%. The dataset is open-sourced to foster reproducibility in software security and machine learning (ML) cross-research.

### Strengths
1. Originality:
Merging interprocedural program slicing and flow-sensitive pointer analysis into vulnerability patch dataset curation—prior datasets only support intraprocedural analysis or omit pointer alias resolution.
CNPP’s multimodal design addresses two unmet challenges: handling ultra-long token sequences (10K+ tokens) via residual BiLSTM+attention, and aligning natural language with code via contrastive learning, a novel application in vulnerability patch search.

2. Quality:
Dataset scale and diversity: 12,629 CVE-commit pairs (2010–2023) across 3,071 repositories and 10 languages, outperforming baselines.
Experimental rigor: Comprehensive evaluations include baseline comparisons, ablation studies, embedding analysis, and scalability tests, ensuring results are robust and generalizable.

3. Clarity:
Technical details are transparent: Algorithm 1 explicitly outlines dynamic co-analysis steps; Table 6 quantifies performance drops from feature/model ablations, making it easy to interpret the value of each component.
Motivating examples (CVE-2019-17498) link abstract techniques to concrete vulnerability semantics, aiding reader understanding.

4. Significance:
Addresses a critical reproducibility gap: GITPATCHDB is open-sourced with a clear license (CC BY-NC-SA 4.0) and includes full metadata (CVE descriptions, program slices), enabling other researchers to build on the work.
Bridges software security and ML: CNPP’s SOTA performance (95.99% Top-10 accuracy) demonstrates the value of semantic-rich data for ML-based vulnerability tasks, paving the way for automated patch auditing in industry.

### Weaknesses
Inadequate Validation of Patch Effectiveness
GITPATCHDB assumes that all curated CVE-commit pairs represent “valid fixes”, but it does not include a secondary validation step to confirm that the patches actually resolve the associated CVEs. In practice, some commits labeled as “vulnerability fixes” may be incomplete (e.g., only partially mitigating the flaw) or introduce new vulnerabilities. The absence of patch effectiveness checks means the dataset may contain noisy samples (invalid CVE-commit pairs), which could corrupt CNPP’s contrastive learning process.

### Questions
1. On negative sample design for contrastive learning: Your current negative samples are “random non-vulnerability patches”, but real-world ambiguity lies in “near-miss” negatives (e.g., patches for the same CVE type but wrong CVE, or incomplete fixes). Have you experimented with such fine-grained negatives, and if so, how much does CNPP’s Top-k accuracy decrease compared to random negatives? If not, do you believe the model’s current performance would hold when facing these more challenging negatives?
2. The dataset is open-sourced, but what maintenance plan do you have for updating it with new CVE-commit pairs?
3. For the dynamic co-analysis: How do you handle cases where pointer analysis fails to converge (e.g., extremely large codebases with complex aliasing)?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper offers GITPATCHDB, a large-scale dataset that pairs CVEs with their corresponding patch commits. The paper also introduces
Contrastive Natural-language Programming-language Pre-training (CNPP) for multimodal vulnerability patch search via contrastive
learning. The authors include evaluations showing that GITPATCHDB together with CNPP achieves 95.99% top-10 accuracy in vulnerability patch search.

### Strengths
- Significance: The paper tries to address an important problem. Anyone who has worked on vulnerability detection agrees that the datasets currently available in this area are not sufficient. As the authors mention they either lack the corresponding patch or they are focused on certain types of languages or vulnerabilities, as creating a reliable vulnerability issue and fix dataset on scale involves a lot of effort.

### Weaknesses
- Application of CNPP: one of the motivating reasons the authors provide for designing the CNPP is short context length for current embedding models (8K). However, new embedding models such as Voyage 3 support up to 32K context length. 
- Novelty: All the techniques used for program slicing, flow analysis, crafting embeddings are well-known. There is no existing technique also used in a new context that provides additional insights.

### Questions
- Have the authors looked into new embeddings models with more context lengths and the pros and cons of reusing a model vs. building and maintaining one from scratch? When comparing existing embeddings, have the authors considered Voyage 3?
- It is not clear whether all the instances in the GITPATCHDB does include the ground-truth patch or not? The authors mention the test set they used for the experiment includes the ground-truth patch, but how about all the instances? Can this dataset for example be used for evaluating a vulnerability fixing solution?

### Soundness
3

### Presentation
2

### Contribution
2
