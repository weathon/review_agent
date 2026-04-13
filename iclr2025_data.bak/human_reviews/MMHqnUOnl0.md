## Human Reviewer 1

### Summary
The paper introduces a pre-training strategy, **Hierarchical Encoding for mRNA Language Modelling (HELM)**, that incorporates the hierarchical structure of mRNA codons into language model training. By modulating the model's loss based on codon synonymity, HELM aligns learning with the biological structure of mRNA. Results show HELM outperforms non-hierarchical models by around 8% on various mRNA tasks, including antibody annotation and property prediction, while also enhancing generative capabilities for mRNA sequences.

### Strengths
- HELM integrates codon hierarchy into model training, capturing the inherent structure of mRNA data.
- Demonstrates a good performance improvement over existing models on multiple downstream tasks.
- The model can generate more biologically plausible and diverse mRNA sequences than existing approaches.

### Weaknesses
- The authors curate a dataset from OAS specifically to train their models. This may explain much of the performance gains observed in Table-1, as the model is trained and tested on data with similar statistical characteristics. An ablation study isolating the effect of the curated dataset would clarify the extent to which performance gains are due to the data source versus tokenization and architecture design.
- As the authors noted it, representing hierarchical relationships in Euclidean space might limit HELM’s ability to capture the full complexity of these structures, potentially affecting performance.
- Although the results are encouraging, they are primarily based on antibody and property prediction tasks, which could restrict the model's applicability to other mRNA-related contexts.
- Additionally, the strategy for splitting the data into training, validation, and test sets is crucial and should be detailed in the main paper. In sequence-based datasets, it is common practice to split data based on clusters after clustering sequences, which minimizes data leakage. The reported performance gains may be overstated if there is any leakage between the splits.


I am open to revisiting my score; however, I will wait for: (i) the authors' response to my comments, and (ii) feedback from other reviewers.

### Questions
- Are you planning to open-source the curated dataset ?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
The paper introduces HELM, a hierarchical encoding approach tailored for mRNA language modeling that leverages the codon-level structure of mRNA sequences. Unlike traditional models, HELM incorporates a hierarchical cross-entropy loss function to align with the biological structure of mRNA, particularly focusing on synonymous codon usage and its functional implications. The model is evaluated on multiple mRNA datasets, demonstrating improvements in downstream prediction tasks, generative diversity, and sequence annotation accuracy over non-hierarchical baselines. Overall, HELM represents a biologically-informed advancement in mRNA modeling, showing enhanced performance on tasks relevant to protein synthesis and gene expression.

### Strengths
1. Biological Prior Integration: The hierarchical encoding strategy effectively incorporates biological knowledge of mRNA structure, particularly the synonymous codon usage, which enhances the model’s performance on property prediction and generative tasks.
2. Diverse Evaluations: HELM's performance is thoroughly evaluated across multiple tasks, including property prediction, generative sequence design, and antibody sequence region annotation, showcasing the model’s versatility and relevance to bioinformatics applications.
3. Improved Representational Quality: By aligning model training with the codon hierarchy, HELM captures the underlying biological structure more effectively, achieving better clustering of synonymous sequences and improved predictive accuracy in key bioinformatics tasks.

### Weaknesses
1. Lack of Scaling Experiments: The paper primarily employs models with 50M parameters, which is relatively small compared to large language models (LLMs) in NLP. This limitation raises concerns about the necessity of HELM’s hierarchical encoding, as larger models might naturally learn these hierarchical relationships without explicit design. Experiments with larger models could help clarify if the hierarchical loss function offers unique advantages or if it becomes redundant with increased scale.
2. Limited technical contribution:  The primary contribution of this work lies in adapting hierarchical cross-entropy (HXE) to the mRNA structure, which, while valuable for bioinformatics, represents a limited advancement from a machine learning perspective. The adaptation of HXE to this domain may not provide significant innovation to the broader ML community.
3. Baseline suggestion: A simpler way to capture the hierarchical structure could be to introduce separate embeddings for different types of codons and combine these with the token embeddings as input. For instance, embeddings such as [start], [stop], [S], and [Q] (as shown in Figure 6) could be used to represent codon types without modifying the loss function. This approach would provide a straightforward baseline for evaluating the benefits of the proposed HXE solution.

### Questions
See W3.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper introduces Hierarchical Encoding for mRNA Language Modeling (HELM), a novel pretraining strategy that incorporates the hierarchical codon structure of mRNA into language model training. It addresses the limitations of existing models that overlook codon synonymity, which can lead to suboptimal performance in mRNA tasks. HELM modifies loss functions to prioritize errors between different amino acids over synonymous codons, enhancing the model's alignment with biological realities. Evaluations show that HELM outperforms standard language model pretraining and other state-of-the-art RNA models by approximately 8% across six diverse downstream tasks, including property prediction and antibody region annotation, while using fewer model parameters. Additionally, HELM demonstrates improved generative capabilities, producing mRNA sequences that better align with true data distributions. Overall, HELM effectively captures the hierarchical nature of mRNA, leading to enhanced performance in both predictive and generative tasks.

### Strengths
1. This study proposed a novel approach for embedding biological hierarchical structures into language models to enhance interpretability and accuracy in biological sequence analysis.
2. A comprehensive technical evaluation spanning multiple model architectures was presented, benchmarked across diverse datasets and use cases.
3. Clear and precise presentation of technical methods and experimental setup were provided to facilitate reproducibility and further research.

### Weaknesses
1. The performance improvement seems more attributable to data selection than methodology: - HELM uses antibody mRNA while baselines use different data types (ncRNA, pre-mRNA, diverse organisms mRNA).
2.  The paper fails to justify why antibody mRNA pre-training would generalize to: - Viral RNA sequences - Riboswitch sequences.
3.  The evaluation methodology is insufficient: a. Over-reliance on single metric (Spearman correlation) b. No analysis across sequence lengths c. No biological interpretation of prediction errors d. The evaluation lacks comprehensive stress testing on edge cases (e.g., extreme sequence lengths, unusual GC contents, rare codon clusters), raising concerns about the model's reliability in challenging real-world scenarios.
4. Methodological issues: (1) α parameter (0.2-0.6) lacks biological basis (2) Oversimplified codon bias analysis (3) Insufficient validation of hierarchical structure's biological relevance.
5. Limited practical applications: (1) Too narrow focus on antibody sequences and expression (2) Critical applications remain untested: a. RNA vaccine design applications b. Abnormal or mutated sequences c. Real-world biological scenarios.

### Questions
1. What is the biological justification for the α parameter range selection?
2. How does the model handle sequences with unusual codon usage patterns? Such evaluation is crucial because unusual codon usage patterns could significantly impact translation efficiency and protein expression levels, yet these scenarios are not addressed in the current validation.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper proposes an elegant and novel solution for incorporating codon hierarchical structure into language model training, which is hierarchical cross-entropy loss. Although the method is simple, the biological insight is deep and the performance is convincing. The authors also curate the hierarchical information and a pre-train dataset, which may benefit future research. Overall, I feel this would be an impactful work for the community.

### Strengths
1. The authors combine biological insight with language modeling.
2. The proposed HELM is solid and performs well for mRNA-related tasks. It achieves SOTA performance on several practical and impactful tasks, e.g., mRNA sequence design, and sequence region annotation.
3. The authors provide a curated dataset, and curated domain knowledge, which may benefit future computation-oriented research.
4. The authors provide solid benchmark experiments of various backbone architectures and tokenization methods.

### Weaknesses
1. The methodology itself is relatively simple.
2. It is not clear to what extent the model, the codes, and the data will be made public.

### Questions
I do have a strong concern for the reproducibility of this paper. Despite the fact that I think highly of this paper, I don't think the paper should be accepted without further declaration on reproducibility. Specifically, will the model checkpoint, source codes, and datasets be released?

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
6

### Confidence
3