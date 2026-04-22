# Bimodal masked language modeling for bulk RNA-seq and DNA methylation representation learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Oncologists are increasingly relying on multiple modalities to model the complexity of diseases. Within this landscape, transcriptomic and epigenetic data have proven to be particularly instrumental and play an increasingly vital role in clinical applications. However, their integration into multimodal models remains a challenge, especially considering their high dimensionality. In this work, we present a novel bimodal model that jointly learns representations of bulk RNA-seq and DNA methylation leveraging self-supervision from masked language modeling. We leverage an architecture that reduces the memory footprint usually attributed to purely transformer-based models when dealing with long sequences. We demonstrate that the obtained bimodal embeddings can be used to fine-tune cancer-type classification and survival models that achieve state-of-the-art performance compared to unimodal models. Furthermore, we introduce a robust learning framework that maintains downstream task performance despite missing modalities, enhancing the model’s applicability in real-world clinical settings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
MOJO: Multi-Omics JOint representation learning is a novel approach to jointly model bulk RNA-seq and DNA methylation data. It does early fusion of both data modalities and learns a mixed representation via transformers blocks in a learned lower dimensional space, providing efficient training and inference for high-dimensional genomics data. The authors demonstrate strong performance of MOJO in downstream applications and modeling settings. They further interpret their model in the context of biological pathways and provide an interesting analysis on approaches to address missing data modalities.

### Strengths
- The authors present an efficient mechanism to do multihead attention in a lower dimensional space, demonstrating impressive gains in model training efficiency (e.g. Table 2). This is especially relevant for genomics data, as in the current setting, where the number of features is very large and can further scale if incorporating further genomic data types.

- Strong results across tasks for cancer classification, survival analysis, and zero-shot subtyping. Particularly interesting analysis with section 4.4 analysing the alignment of latent model features with known biological pathways.

- Nice study on missing data modalities that have strong implications for further experimentation in research or clinical settings. Both approaches for extended pretraining with missing data modalities using entire-modality masked tokens and the mutual information auxiliary loss are interesting and demonstrate strong results in this biomedical setting.

### Weaknesses
- For downstream objectives (Table 1 and Table 3), while the pretraining and downstream (classification or survival) objectives do differ, there is still potential data leakage of using some subset of the pretraining data in the test splits of the cross-validated classification results. This is a major flaw of the paper particularly when other RNA-seq and methylation data is present in the GDC alongside TCGA.

- Table 1: late-attention with a newly trained cross-attention module is comparable to probed MOJO. While the authors claim PEFT of pretrained MOJO does best, no comparable performance to a PEFT of pretrained BulkRNABert+MethFormer with a cross attention module is done (unless the illustration of the cross-attention approach in appendix B Figure 7 is incorrect and the modality specific encoders are tuned along with the cross attention?). If, in late integration, a PEFT of the modality arms and training the cross-attention match the PEFT of MOJO, then the results are further tempered.

- The analysis of embedding correlation with biological pathways is very interesting, however the author’s claim that pathways are “correctly enriched” on line 380 needs an appropriate reference.

### Questions
- Would be interesting to see an ablation where the convolutional dimensionality reduction is not used

- On line 149-151, why convolutions instead of fully connected layers for downsampling?

- Line 251 the authors say the results do not significantly differ - is this backed by a statistical test or is this loose use of the word? should be clarified

- Are data splits (for pretraining or downstream cross validation) stratified by cancer type and observed mortality?

- Is late integration in section 4.3 also restricted to concatenation? Should be explicitly stated

- Further clarification of the KNN method in section 4.3 should be provided (w.r.t if results are cross validated, what data was used for training, etc)

### Soundness
2

### Presentation
4

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
The paper introduces MOJO, a bimodal masked-language modeling framework for bulk RNA-seq and DNA methylation. Both modalities are aligned at the gene level and discretized into tokens (linear binning). A conv-downsampling + Transformer backbone is pre-trained with bimodal MLM and then probed/fine-tuned for pan-cancer type classification, survival analysis, and some zero-shot subtyping/clustering. To handle missing modalities at test time, the authors add an auxiliary mutual-information–based consistency loss during fine-tuning. The paper reports strong in-domain results and notable training efficiency.

### Strengths
- Clear, easy-to-reproduce pipeline: gene-level alignment + simple tokenization; architecture and objectives are standard and well explained.

- Training efficiency: the hybrid conv+Transformer design scales to long gene sequences with large batch sizes and fast steps.

- Solid in-domain performance: competitive results on TCGA for classification and survival; additional zero-shot analyses are included.

- Missing-modality robustness: the MI-based consistency objective substantially improves performance when one modality is absent.

- Initial interpretability: SHAP/GSEA-style analyses provide some biological intuition for the learned representations.

### Weaknesses
1. Evaluation loop / external generalization is limited. Pretraining and downstream evaluation are both conducted within TCGA splits, without independent validation on ICGC/GEO or cross-platform cohorts.

2. Coarse cross-modal alignment. Aggregating methylation to per-gene values via “near-gene CpG averaging” risks discarding promoter/enhancer/genic-region specificity and may mask true regulatory dependencies.

3. Modest novelty. The approach largely combines known ingredients (linear binning → MLM pretraining; conv downsampling + Transformer; standard probing/finetuning). Contributions feel incremental rather than conceptually new.

### Questions
1. External generalization. Beyond TCGA, can you replicate key results on independent cohorts (e.g., ICGC/GEO) and report cross-platform robustness (batch effects, sequencing platforms)? Please include both classification and survival, plus the zero-shot analyses, with identical preprocessing and fixed splits.

2. Alignment strategy. How sensitive are results to the per-gene methylation aggregation? Could you compare against region-aware or annotation-aware schemes (e.g., promoter/CGI/shore weighting, enhancer links, or graph-based aggregation), and provide a quantitative ablation (including different binning schemes or continuous embeddings without discretization)?

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
4

### Summary
This paper proposes MOJO, a bimodal model for learning joint representations from bulk RNA-seq and DNA methylation data. It utilizes a self-supervised bimodal masked language modeling (MLM) task for pre-training. The architecture combines a U-Net-like convolutional structure for efficient dimensionality reduction with a Transformer core to integrate the two modalities. Evaluated on the TCGA dataset, MOJO shows strong performance on downstream tasks like cancer-type classification and survival analysis, outperforming single-modality models.

### Strengths
1. This work pioneers the integration of bulk RNA-seq and DNA methylation, uniting complementary transcriptomic and epigenetic data for deeper mechanistic insight.

2. Its focus on clinically accessible bulk omics significantly enhances translational value, bridging foundational research with practical implementation.

### Weaknesses
1. An oversimplified architectural diagram (Fig. 1) that abstracts the core Transformer structure.
2. Incomplete benchmarking, which omits comparisons to single-cell foundation models (e.g., scGPT) adapted for bulk-data tasks.
3. An absence of critical ablation studies to validate the efficacy of the hybrid CNN-Transformer architecture or the necessity of Gene2Vec initialization.
4. Severe information loss in methylation data processing, where averaging ~450k sites to gene-level features discards crucial, region-specific regulatory patterns.
5. A lack of external validation, as the model's performance is demonstrated only within the TCGA cohort, limiting its proven generalizability.

### Questions
N/A

### Soundness
2

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
3

### Summary
This paper presents MOJO, a bimodal masked language model that jointly learns from RNA-seq and DNA methylation data using a hybrid CNN–Transformer architecture. It achieves strong performance in cancer classification and survival prediction, remains robust with missing modalities through mutual information loss, and demonstrates biological interpretability consistent with known cancer pathways.

### Strengths
The strength of this paper lies in its ability to maintain stable performance even with single-modality or missing-modality data through extended pretraining and mutual information loss. It also demonstrates biological plausibility by combining SHAP and GSEA analyses to verify that the model’s outputs align with known cancer-related pathways.

### Weaknesses
1.  Masked language modeling is the standard approach for pretraining in the biomedical domain; it has become a conventional method. The incremental improvement compared to existing approaches (such as BulkRNABert) mainly lies in the bimodal integration, but the technical innovation is limited. Moreover, the accuracy in predicting masked genes may not necessarily reflect biologically meaningful interactions. Is there any related literature to support this?
2. The model only performs a simple concatenation of RNA-seq embeddings and methylation embeddings. I suggest analyzing the similarity between the two modalities, for example, by using KL divergence.

3.  Table 9 shows that even with the addition of the MI loss, the F1 score when RNA-seq data is missing (0.916) is still lower than that of MethFormer (0.931).

4. There are already several existing studies that use multimodal approaches for survival analysis, such as DeepSurv, DeepHit, and RSF. This paper does not include comparisons with them, nor with other works like $\textit{Cross-Modal Translation and Alignment for Survival Analysis}$ (CVPR 2020) and $\textit{Long-term Cancer Survival Prediction Using Multimodal Deep Learning}$ (Scientific Reports, 2021).
5. In the binning strategy, how are the numbers of bins for RNA-seq and methylation  $\(B_{\text{rna}}$, $B_{\text{meth}}\)$ determined?

### Questions
No

### Soundness
3

### Presentation
3

### Contribution
2
