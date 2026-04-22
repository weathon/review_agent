# CDBridge: A Cross-omics Post-training Bridge Strategy for Context-aware Biological Modeling

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 2, 6, 4

## Abstract
Linking genomic DNA to quantitative, context-specific expression remains a central challenge in computational biology. Current foundation models capture either tissue context or sequence features, but not both. Cross-omics systems, in turn, often overlook critical mechanisms such as alternative splicing and isoform reuse. We present CDBridge, a post-training strategy that unifies pretrained DNA and protein models into a context-aware framework without full retraining. CDBridge operates in two stages: (a) Seq-context learning, where a splicing-inspired token merge compresses long genomic regions into isoform-aware representations, and (b) Env-context learning, where a conditional decoder injects tissue embeddings to model expression under diverse biological contexts. To benchmark this setting, we introduce GTEx-Benchmark, derived from GTEx and Ensembl, which requires models to capture long-range exon dependencies, resolve isoform reuse, and predict tissue-specific expression levels. Across qualitative and quantitative tasks, CDBridge consistently outperforms prior methods that ignore central dogma constraints or context dependence, offering a scalable and biologically faithful solution for DNA-to-expression modeling.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents CDBridge, a novel post-training bridge strategy for context-aware biological modeling that addresses the critical challenge of mapping genomic DNA sequences to quantitative, tissue-specific expression levels. The proposed framework unifies pre-trained DNA and protein models through a two-stage approach: first, a sequence-context learning stage uses a splicing-inspired adaptive token merge to compress long genomic sequences into isoform-aware representations, and second, an environment-context learning stage employs a conditional decoder that integrates tissue embeddings to predict expression under diverse biological conditions. To rigorously benchmark this task, the authors introduce GTEx-Benchmark, a comprehensive dataset requiring models to resolve long-range exon dependencies, isoform reuse, and tissue-specific expression. Extensive experiments demonstrate that CDBridge consistently outperforms existing methods across both qualitative tasks like coding region segmentation and quantitative tissue-aware expression prediction, establishing a scalable and biologically faithful solution for cross-omics modeling.

### Strengths
This paper makes significant contributions to the field of computational biology by introducing a context-aware, post-training framework that effectively bridges the gap between DNA sequences and their functional protein outcomes. Its principal strengths are as follows:

1. Novel and Biologically-Grounded Methodology: The proposed two-stage bridge strategy is both technically elegant and biologically inspired. The splicing-like adaptive token merging in Stage 1 efficiently handles the extreme sequence length disparity, while the conditional decoder in Stage 2 adeptly models tissue-specific regulation, directly addressing the central dogma's one-to-many mapping challenge.

2. Strong and Comprehensive Empirical Validation: The authors provide compelling evidence for CDBridge's superiority through extensive experiments. It achieves state-of-the-art performance not only on the primary task of tissue-aware expression prediction but also on a suite of challenging downstream tasks including coding region segmentation, isoform retrieval, and DNA-protein association, demonstrating robust generalization to unseen tissues.

3. Effective and Efficient Post-Training Strategy: The framework smartly leverages powerful, pre-trained single-omics foundation models (e.g., Evo2, ESM2) and aligns them with minimal additional parameters. This post-training approach makes advanced cross-omics modeling more accessible and scalable without the prohibitive cost of full end-to-end retraining.

4. Enhanced Interpretability: The paper provides insightful analyses, such as visualizing the token merging process and tissue-aware activations, which show that the model's learned mechanisms align well with known biological principles (e.g., preserving exons, merging introns). This greatly enhances the model's transparency and trustworthiness.

### Weaknesses
While the paper presents a robust and compelling framework, there are minor limitations that could be addressed to further strengthen the work and its broader impact.

1. Simplified Modeling of Biological Context: The current model conditions on tissue type as a categorical variable, which, while effective, is a relatively coarse-grained representation of a cell's state. This approach does not explicitly capture more dynamic or granular contextual factors known to influence gene expression, such as developmental stage, disease status, or specific microenvironmental cues. Extending the context modeling to incorporate such continuous or multi-factorial signals could enhance the model's biological fidelity and applicability to more complex physiological and pathological conditions.

2. Dependence on High-Quality Annotations and Pre-trained Models: The performance of CDBridge is contingent upon the availability of high-quality, curated inputs, including precise isoform annotations from Ensembl and the pre-computed embeddings from large foundation models like Evo2 and scGPT. This reliance may limit the framework's generalizability to less-studied organisms, poorly annotated genes, or contexts where such powerful pre-trained models are unavailable or perform suboptimally, potentially restricting its utility in broader genomic applications.

### Questions
1. The adaptive token merging mechanism is a key component for handling long sequences. Could you provide more detail on how the merge ratio is sampled from a Gaussian distribution during training? Specifically, what were the chosen mean and variance, and what was the rationale behind this design choice over a fixed or learned ratio?

2. The model demonstrates strong performance on unseen tissues, which is a significant result. Could you elaborate on the compositional nature of these "unseen tissues"? For instance, are they entirely distinct tissue types, or do they share cellular subtypes or regulatory programs with tissues seen during training? This would help clarify the extent of the model's generalization.

3. In the ablation study (Table 4), the configuration with only Tissue Clust enabled showed a significant decrease in expression prediction performance. This could be due to insufficient gene embedding quality provided by scGPT or excessive information loss during pooling. Could the authors provide results using other single-cell foundation models or different pooling strategies?

4. In Table 4, the segmentation performance (AUC and F1) is identical for the 4rd, 5th, and 6th rows. This seems to indicate that the segmentation task is solely determined by the frozen Stage 1 components. Could you clarify the configuration for the 5th row? It appears to show the same segmentation results without the key Stage 1 components (ToMe and Learned Clust), which is counter-intuitive.

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
This paper presents CDBridge, a two-stage post-training strategy to model the "central dogma of molecular biology" by unifying pretrained DNA, protein, and RNA models. 

The method consists of a token merging strategy to fuse DNA and protein representations to predict functional masks, as well as a tissue-conditional decoder for predicting cell-level protein expressions.

Importantly, the authors introduce GTEx-Benchmark for the training and evaluation of tissue-aware central dogma modeling, which contains matched annotations of DNA, RNA, and protein.

CDBridge is validated on gene-expression prediction for seen and unseen tissues, as well as its ability to segment coding regions of DNA, retrieve isoforms, and classify whether DNA-protein pairs are functionally associated.

### Strengths
- The work is well-motivated and presents a new multi-omics model
- CDBridge generalizes to unseen tissues for gene expression prediction
- The figures describing the framework are well-made

### Weaknesses
The main weakness is the evaluation and details pertaining to it.  
- It is unclear how the evaluation of gene expression prediction was performed
- The single-omics foundation models in Table 2, as well as LucaOne, are too weak to be considered meaningful baselines.
- The behavior in Figure 5, while interesting, is not extensively studied, in that the  reason "non-coding regions are often merged" may be that:
  - almost all of the human genome is non-coding
  - the embeddings for coding regions may be more information-rich compared to non-coding regions
- Figure 6 shows selective token activation that is not strongly tissue-dependent, and therefore does not appear to reflect capacity for modeling tissue-aware gene expression.
- In Table 4 (the ablation study):
  - It appears that ignoring Stage 1 and only adding Stage 2 by itself saturates segmentation performance. 
  - The ablation study does not study whether there is a difference between the fixed and learned clustering in the presence of tissue clustering.

### Questions
- How long were the DNA sequences and how many were used for evaluation?
- How were the single-omics foundation models in Table 2 used to predict gene expression, considering that they "lack explicit modeling of tissue-specific regulatory signals"?
- Why is the Enformer performance much worse than reported in the paper?
- Can you compare to Borzoi (http://dx.doi.org/10.1038/s41588-024-02053-6), a successor to Enformer?
- What is the performance of a linear model baseline for gene expression prediction?

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
3

### Summary
CDBridge is a two-stage post-training strategy that unifies pretrained DNA (Evo) and protein (also from the Evo family) foundation models for context-aware modeling of gene expression. 
Stage 1 aligns DNA and protein embeddings using a splicing-inspired token merge mechanism (based on ToMe) to handle long-range dependencies.
Stage 2 uses a conditional decoder to inject tissue embeddings for environment-specific expression prediction.
A new GTEx-Benchmark is proposed to evaluate models on tissue-aware central dogma tasks. 
CDBridge achieves strong results in expression prediction (R² ≈ 0.39 vs. 0.0–0.32 for comparable baselines) and cross-modal tasks compared to prior models (significantly better in coding region segmentation, isoform retrieval, and  2-3 percentage points better in central dogma; measured in Acc, AUC, and F1).

Overall Assessment
CDBridge is a solid and biologically grounded contribution that advances cross-omics modeling by bridging DNA, RNA (as a latent bridge; not explicitly encoded) and protein representations through a context-aware, post-training framework. While the architectural novelty is moderate (stage 1 is based on ToMe/Token Merging; stage 2 is conditional transformer/standard cross attention; no new training objective or attention mechanism), the biological design insights and benchmark release make this work a meaningful step toward context-aware central dogma modeling. CDBridge’s novelty lies in the biological grounding of the architecture (splicing-inspired token merging, tissue-aware decoding), and the results show that the pretrained DNA and protein foundation models can be aligned post-hoc for expression prediction.

### Strengths
1. Biologically motivated architecture.
The splicing-inspired adaptive token merging and the explicit modeling of tissue context are well-designed and biologically meaningful. It nicely mimics exon selection and isoform reuse phenomena, showing awareness of biological grounding beyond standard transformer tricks.

2. Strong empirical performance.
The reported R² and Spearman correlations across tissues are competitive and consistently improve upon strong baselines, suggesting that the cross-omics alignment and conditional decoding are genuinely effective. The cross-omics downstream tasks also show significant improvements, comparing to the baselines.

3. Benchmark contribution.
The introduction of GTEx-Benchmark fills the gap in evaluating DNA-to-expression models, emphasizing isoform-aware, tissue-conditioned prediction. This benchmark could become valuable for future biological modeling studies.

4. Practicality and scalability.
The framework leverages pretrained DNA and protein models without full retraining, making it computationally feasible and modular for extension to new modalities (e.g., RNA, metabolomics).

### Weaknesses
1. Evaluation design ambiguity.
The dataset construction process for GTEx-Benchmark could benefit from more details (e.g., splitting strategies, tissue balance, sequence selection). It’s unclear how much the benchmark is novel versus derived from existing GTEx/Enformer configurations.

2. Ablation and interpretability is limited.
There is limited ablation showing the separate contributions of token merging, cross-omics alignment, and tissue conditioning. Moreover, while biological motivation is strong, interpretability analyses (e.g., learned exon attention, pathway enrichment) are missing.

3. Comparison fairness.
Some baselines (e.g., Isoformer, AlphaGenome) are not fine-tuned under the same supervision regime, and their task setups differ slightly. It is unclear if the performance gap reflects true model capability or evaluation setup differences.

### Questions
1. How sensitive is performance to the merge ratio and to the pretrained model choices (e.g., DNA encoder backbone)? 

2. How does the GTEx-Benchmark differ statistically from existing Enformer/AlphaGenome datasets?

3. How does the model handle unseen tissue embeddings (e.g., zero-shot tissue generalization)?

4. Can the author comment on how does the model differ from the following related papers (not referenced in your work) 
4a. Garau-Luis, Juan Jose, et al. "Multi-modal transfer learning between biological foundation models." Advances in Neural Information Processing Systems 37 (2024): 78431-78450.
4b. Liu, Zicheng, et al. "Life-Code: Central Dogma Modeling with Multi-Omics Sequence Unification." arXiv preprint arXiv:2502.07299 (2025).

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
3

### Summary
**Problem (P)**

The paper asks how to map whole-DNA to context-aware expression. Current systems either model tissue context from gene IDs without DNA, or model DNA without tissue, and most cross-omics pipelines ignore isoform reuse and splicing. The authors call out two challenges: long DNA must be compressed (because proteins generally have fewer amino acids), and the same DNA strand may map to multiple proteins depending on tissue. 

**Solution (S)**

CDBridge is a post-training two-stage bridge that keeps large DNA and protein models frozen and learns a light connector.
Stage 1 maps DNA embeddings to a protein prototype dictionary and an adaptive token-merge (ToMe) “splicing-like” compressor keeps informative regions, then a small decoder predicts coding-region masks.
Stage 2 is a conditional decoder, which injects tissue embeddings (from a single-cell FM) to produce tissue-specific isoform embeddings and a scalar expression prediction.

**Contributions (C)**

1. A two-stage bridge that aligns DNA and protein while conditioning on tissue for expression.

2. A splicing-inspired token merge that compresses long DNA while preserving functional tokens.

3. GTEx-Benchmark, pairing DNA, proteins, and tissue-resolved expression for three tasks: expression prediction, coding-region segmentation, and isoform retrieval. 

**Tasks and metrics (T)**
- Tissue-conditioned expression regression: R² and Spearman, in seen-tissue and unseen-tissue splits.
- Coding-region segmentation: Acc, AUC, F1 at token level.
- Isoform retrieval: Acc@3 and MRR. Also a Central-Dogma binary DNA–protein association task.

### Strengths
1. This design is highly practical and scalable. It allows the framework to leverage the billions of parameters and complex representations learned by FMs without the immense cost of full retraining. 

2. The paper uses domain-specific biological knowledge to inform its architecture instead of just apply generic ML techniques. The "splicing-inspired adaptive token merge" (Section 2.1) is a good example.

3. A common weakness in modeling is testing only on data similar to the training set. This paper's evaluation is stronger because it explicitly tests for generalization to entirely unseen biological contexts.

### Weaknesses
1. A methodological weakness in "Dataset Construction" (Section 3.1), where the authors state, "genes with DNA sequences longer than 200k base pairs are excluded." Many of the most complex and important human genes, like Dystrophin (over 2 M base pairs), are far longer than this 200k cutoff.
2. While framework's "post-training" design, while efficient, does it also inherit any/all biases, gaps, or blind spots from the FMs it bridges (Evo2 and ESM2). The author's view on this would be appreciated. 
3. The current approach, which essentially uses a single label (e.g., "Brain"), fails to capture finer-grained, critical factors that govern gene expression. Although acknowledged, could the authors clarify whether these factors were omitted primarily due to GTEx labeling constraints, modeling choices, or computational limits?
---
**Minor Issues (for the Authors; not weaknesses)**

1. The paper already notes that Isoformer (Official) numbers come from their paper with TSS alignment. Consider adding a brief sentence that scores are not directly comparable to your setting to avoid misinterpretation.
2.  "maps" --> "map", page 1
3. "remainslargely" --> "remains largely", page 2
4. “perform relative poor” --> “perform relatively poorly”, section 3.3
5. "actviated" --> "activated", section 3.3
6. "s while the regions belong the non-coding regions" This is an awkward sentence, figure 5
7. "Envir-context learning" or "environment-context learning" Please choose one and be consistent

I think I saw a couple more while reading. Kindly consider taking the help of LLMs to mitigate these.

### Questions
For my critical questions, kindly see the weaknesses. Given these clarifications in an author response, I would be willing to increase the score. 

Appendix C Table 5. Is stage 2 really 513 params?

### Soundness
3

### Presentation
2

### Contribution
3
