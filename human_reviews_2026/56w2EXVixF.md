# PhenoKG: Knowledge Graph-Driven Gene Prioritization and Patient Insights from Phenotypes Alone

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Rare genetic disorders affect over 300 million people worldwide and remain difficult to diagnose, with current genomic approaches yielding definitive answers in only 30-50% of cases. Existing phenotype-driven methods often rely on expert-curated candidate gene lists and are sensitive to incomplete clinical data, limiting their real-world utility. We present PhenoKG, a knowledge-graph framework that enriches patient phenotypes with biomedical knowledge and can rank flexible amount of genes by their likelihood of being causative ($\sim$4,000). PhenoKG integrates graph neural networks with transformer-based encoders to capture patient-specific phenotype-gene relationships, and incorporates an optional reranking procedure that leverages recently validated clinical associations to extend the knowledge graph while maintaining robustness to noisy or incomplete input. Designed to operate with or without candidate lists, PhenoKG achieves strong performance across diverse diagnostic settings and consistently outperforms state-of-the-art methods on rare disease benchmarks.
Together, these results position PhenoKG as a step toward scalable, phenotype-first models for rare disease diagnosis, and open the path to integrating heterogeneous biomedical data for faster, more equitable genetic discovery.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tackles phenotype-driven rare-disease gene prioritization by building patient-specific subgraphs over a biomedical knowledge graph and encoding them with GATv2 plus transformer modules, followed by a contrastive ranking objective and an optional phenotype-based reranking step. Across simulated and real-world cohorts, the method reports strong MRR/Hits@K, outperforming several phenotype/KG baselines in both candidate-list and no-candidate settings. Overall the problem is important and the empirical results are promising.

### Strengths
- The proposed representation connects biomedical KGs with clinical phenotypes in a flexible, modular way.

- The approach is evaluated on multiple datasets (three real-world plus a simulated cohort) and two usage scenarios (with/without candidate lists), which improves the breadth and credibility of the results.

### Weaknesses
- Because the candidate list is predefined in many clinical settings, it would be helpful to discuss whether the framework can discover new, previously unlisted genes for a patient/phenotype (beyond reranking within a fixed set).
- The data preprocessing is fairly involved; this may filter out difficult edge cases and could limit realism. A concise flowchart with counts at each step would make exclusions transparent.
- Please add a table of basic dataset statistics (before/after preprocessing), e.g., average number of phenotypes per patient and the distribution of candidate list sizes.

### Questions
- Please report potential overlap between training and test distributions at both the entity level (genes/phenotypes appearing in both) and the association/patient-profile level (e.g., patients with very similar phenotype sets), and discuss how this overlap might affect generalization.

### Soundness
3

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
3

### Summary
This paper tackles the challenge of rare genetic disease diagnosis using patient phenotypes alone, aiming to infer disease-causing genes through the relationships among phenotypes, diseases, and genes. The authors propose PhenoKG, a knowledge-graph–based framework that integrates graph neural networks and transformer encoders to model phenotype–gene associations. It constructs a patient-specific subgraph from phenotype–gene associations within a biomedical knowledge graph (PrimeKG) and learns patient and gene embeddings for ranking ∼6,000 genes by their likelihood of being causative. The method also includes a reranking step using external databases (e.g., Monarch) to incorporate up-to-date phenotype–gene associations. Experiments on multiple datasets (FGDD, PhenoDis, MyGene2, and simulated cohorts) show that PhenoKG achieves state-of-the-art performance both with and without candidate gene lists, outperforming methods such as Shepherd, PhenoApt, and Amelie.

### Strengths
A key strength of the paper lies in its integration of graph neural networks with transformer-based encoders, which allows the model to effectively capture rich, patient-specific features as well as characteristics of potential causal genes. By leveraging these complementary architectures, PhenoKG is able to align patient and gene representations across modalities, enabling more precise modeling of phenotype–gene relationships and improving the prioritization of candidate causal genes.

### Weaknesses
One concern is the scalability of the approach. As the authors note, there are thousands of potential causal genes in this task, and computing embeddings for all genes could be computationally intensive. This may limit the practicality of PhenoKG in real-world scenarios where large gene sets need to be evaluated.

Another concern relates to the model’s performance consistency across datasets. While PhenoKG achieves state-of-the-art results on most benchmarks, it performs substantially worse than PhenoApt on the MyGene2 dataset. Although the authors provide some analysis of the dataset’s characteristics and potential reasons for this discrepancy, it raises questions about the generalizability of the approach and suggests that further investigation into dataset-specific performance is warranted.

### Questions
1. Candidate Gene Extraction: The paper briefly states that candidate genes are extracted “based on phenotype–gene associations,” but the specific algorithmic procedure (e.g., thresholds, ontology traversal strategy) is not detailed. How exactly are the candidate gene lists extracted when no clinician-provided list is available? Is there a fixed k-hop depth or a phenotype–gene similarity threshold?

2. Could the authors provide component-level ablations—for example, evaluating the effectiveness of the reranking module, and whether integrating external knowledge bases (e.g., Monarch) actually improves results?

3. Can the authors provide an example of a real patient case study showing how PhenoKG identified the correct gene and what features contributed most?

4. The authors mention that they imposed multiple restrictions for efficiency, such as limiting the knowledge graph to a 2-hop neighborhood around each patient and only considering genes that appear in at least 10% of the patient’s phenotype subgraphs as candidates. More details are needed to understand how these restrictions might affect the fairness and rigor of the evaluation—for example, clarifying what exactly “2-hop” refers to and how it is defined in patient knowledge graph. A clear explanation of these choices would strengthen confidence in the fairness and validity of the reported results.

5.  Could the authors show the results over some widely used datasets, such as DDD and the datasets used in LIRICAL, Phen2Gene and PhenoApt?  It can illustrate the robustness of the proposed method.

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
3

### Summary
The paper proposes PhenoKG, a knowledge graph (KG)–based framework for identifying causal genes given a patient’s phenotype and an external biomedical KG. It also introduces an optional reranking procedure that integrates recently validated clinical associations to expand the KG while maintaining robustness to noisy or incomplete input data. PhenoKG reportedly outperforms existing approaches across one simulated and three real rare-disease benchmarks. Ablation studies were performed to examine the effects of different loss functions, embedding initializations, and input graph configurations.


The paper addresses an important problem. However, I find that the methodological contributions are not clearly articulated, and the real-world impact of the proposed approach remains limited to benchmark demonstrations. A more thorough exposition and justification of the method’s design and novelty are required before publication. My detailed comments are as follows:



- Line 41: The distinction between disease-causal genes and genes associated with disease progression needs clarification. These two categories are known to have limited overlap (see “Limited overlap between genetic effects on disease susceptibility and disease survival”). 

- Line 70: why 6,000 genes, not 20,000?

- Section 2: It is unclear how PhenoKG is methodologically distinct from prior work. If it represents a completely new framework, the authors need to justify its overall design choices. Furthermore, it is not clear which specific component drives the reported performance improvements.

- Lines 74–83: What is the main methodological innovation? If the reranking strategy constitutes the novelty, ablation experiments should explicitly demonstrate its contribution. This is related to line 422.

- Lines 308–323: The reranking procedure is presented as a key innovation, but the description does not establish why it is theoretically or methodologically principled.

- Line 294: Can the authors provide statistical significance measures (e.g., p-values) for the inferred causal genes for each patient?

### Strengths
See summary above.

### Weaknesses
See summary above.

### Questions
See summary above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a new knowledge graph for phenotype. From a scientific, and even from a machine learning perspective, it is useful to have more data sources like this. However, the novel part of the paper is the dataset itself. The machine learning method itself is a combination of well-known approaches, and I do not identity novel insights.

### Strengths
* The presented dataset is an interesting one
* The experiments appear to be performed correctly

### Weaknesses
* I see little contribution to machine learning research.

### Questions
* Are there specific machine learning contributions I might have overlooked?

### Soundness
3

### Presentation
3

### Contribution
1
