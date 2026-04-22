# RAG4DMC: Retrieval-Augmented Generation for Data-Level Modality Completion

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Multi-modal datasets are critical for a wide range of applications, but in practice, they often suffer from missing modalities. This motivates the task of Missing Modality Completion (MMC), which aims to reconstruct missing modalities from the available ones to fully exploit multi-modal data. While pre-trained generative models offer a natural solution, directly applying them to domain-specific MMC is often ineffective, and fine-tuning suffers from limitations like limited complete samples, restricted API access, and high cost. To address these issues, we propose RAG4DMC, a retrieval-augmented generation framework for data-level MMC. RAG4DMC builds a dual knowledge base from complete in-dataset samples and external public datasets, enhanced with feature alignment and clustering-based filtering to mitigate modality and domain shifts. A multi-modal fusion retrieval mechanism combining intra-modal retrieval with cross-modal fusion then provides relevant context to guide generation, followed by a candidate selection mechanism for coherent completion. Extensive experiments on general and domain-specific datasets demonstrate that our method produces more accurate and semantically coherent missing-modality completions, resulting in substantial improvements in downstream image–text retrieval and image captioning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tackles the Missing-Modality Completion (MMC) problem, where samples in a multi-modal dataset lack one or more modalities (e.g., an image without its paired caption). Instead of fine-tuning large generative models—often impractical due to limited complete samples, API-only access, or high cost—the authors propose RAG4DMC, a retrieval-augmented generation framework.

### Strengths
S1. Important practical problem: MMC occurs frequently in real-world pipelines, yet receives less attention than model-level fusion.
S2. Novel RAG formulation: combining internal and external KBs and explicitly tackling modality/domain shift via alignment + clustering is fresh in the MMC context.
S3. No fine-tuning of the large backbone: framework works even with “API-only” generative models, broadening applicability.
S4. Comprehensive evaluation: three datasets (general + domain-specific), multiple missing-rate settings, and downstream task benefits.
S5. Ablation on retrieval components and filtering validates each design choice.

### Weaknesses
W1. Scope limited to image-text dyads; unclear whether the approach scales to higher-order or fundamentally different modalities (e.g., audio, LiDAR).
W2. Generation quality is assessed only indirectly (via downstream tasks). Lack of direct fidelity/semantic metrics (e.g., CLIP-Similarity, FID) or human evaluation.
W3. External KB dependence: quality and domain bias of external data may dominate results; sensitivity analysis is brief.
W4. Computational footprint: retrieving from (possibly large) dual KB + generating multiple candidates can be expensive; runtime/memory costs are not reported.
W5. “First” claim could be overstated—earlier works have combined retrieval and generation for cross-modal reconstruction, though perhaps not under the MMC label; a clearer positioning is needed.
W6. Feature-alignment technique is only sketched; hyper-parameters and convergence behaviour are not well analysed.

### Questions
Q1. How does performance vary with the size and domain distance of the external KB? Please provide a scaling or ablation study.
Q2. What alignment method is used (e.g., Procrustes, MMD, contrastive loss)? Is it learned jointly with retrieval or pre-computed?
Q3. How many candidate completions are generated per query, and how sensitive are results to this number?
Q4. Could retrieval introduce harmful or biased content? Any safeguards or toxicity filtering applied?
Q5. What is the wall-clock cost (GPU hours) and latency per completion compared with direct generation?
Q6. Have you tested extremely high missing rates (e.g., 90 %) where internal KB is tiny? Does external data fully compensate?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes RAG4DMC to address the problem of missing modalities in multimodal datasets. Existing generation-based methods suffer from low reliability, while simple RAG-based approaches are ineffective due to noise and domain gaps in external data. To overcome these issues, the authors construct a dual knowledge base by combining internal complete data with external public data. They align and integrate the semantic spaces through cross-modal mapping, clustering-based filtering, and Procrustes alignment. For missing samples, a two-stage fusion retrieval is performed to find highly relevant examples, which are then used with BLIP2 and Stable Diffusion to generate restoration candidates. Finally, CLIP cosine similarity, BLEU, and NIQE metrics are used to select the most semantically consistent and high-quality result.

Experiments conducted on MSCOCO, Flickr30K, and RSICD demonstrate that RAG4DMC outperforms all baseline methods. Notably, as the missing rate increases, the performance gap between RAG4DMC and other methods widens, proving its robustness under data-deficient conditions. The authors further extend the concept of RAG beyond text generation to data-level multimodal restoration, showing that the generated complete datasets significantly enhance the performance of downstream models such as CLIP and LLaVA

### Strengths
- One of the key strengths of this paper is that it effectively overcomes the generalization limitations of fine-tuning approaches that rely on pre-trained generative models or a small number of fully observed samples. Notably, this work is the first to apply RAG to the Missing Modality Completion problem, leveraging retrieval-based semantic grounding to achieve more reliable and consistent restoration.

- The proposed RAG4DMC framework simultaneously utilizes both an internal knowledge base and an external knowledge base, addressing domain shift between the two through feature alignment and clustering-based filtering. These alignment and filtering procedures reduce noise and unify the semantic space, thereby improving retrieval quality and overall restoration accuracy.

- Through a multi-modal fusion retrieval strategy, the model fuses information across images and text, enabling more fine-grained search results. Additionally, the candidate selection stage ensures semantically coherent outputs, making the restoration process far more stable than simple generation-based reconstruction and enabling faithful modality recovery grounded in real data distribution.

- Even under various missing-ratio settings, downstream models trained on restored data from RAG4DMC achieved consistent and significant performance gains on tasks such as image–text retrieval and image captioning. These results demonstrate that the proposed approach remains robust and effective even in realistic scenarios with severe data incompleteness.

### Weaknesses
- RAG4DMC consists of multiple stages cross-modal mapping, knowledge filtering, cross-domain alignment, and retrieval-based completion which collectively result in high computational complexity. When applied to large-scale datasets, both training and inference may become slow, indicating potential inefficiency in large-scale or real-time environments.

- The framework focuses only on image and text modalities, leaving its applicability to other modalities such as speech or depth insufficiently explored. Further experiments are needed to verify whether the proposed approach can be effectively extended to additional modalities.

- Existing MMC methods generally suffer from limited transferability in out-of-domain scenarios, often requiring retraining to maintain performance on new domains, which incurs high computational cost. Although RAG4DMC mitigates this issue through a dual knowledge base and cross-domain alignment, the bidirectional mapping is learned primarily from internal data, making it difficult to guarantee complete domain transfer. Moreover, due to its reliance on BLIP2 and Stable Diffusion, the generation quality may degrade in domains that differ significantly from the pre-training data, potentially resulting in unnatural or inaccurate outputs.

### Questions
- The generator used in the proposed framework appears to be implemented based on Stable Diffusion, but it is unclear which specific version was used.

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
2

### Summary
This paper introduces RAG4DMC, a novel framework that applies retrieval-augmented generation (RAG) to the task of data-level Missing Modality Completion (MMC). The core contribution is a sophisticated system that constructs a dual knowledge base from both in-domain and external datasets, employing techniques like feature alignment and clustering-based filtering to mitigate domain and modality shifts. The proposed multi-modal fusion retrieval, which combines intra-modal retrieval with cross-modal re-ranking, is a clever approach to guide the generation process more effectively. The experimental results on both general and domain-specific datasets demonstrate that this method not only produces more accurate and semantically coherent completions but also leads to significant improvements in downstream tasks like image-text retrieval and captioning.

### Strengths
1. The paper is among the first to systematically adapt the RAG paradigm for data-level Missing Modality Completion, moving beyond feature-level imputation to generate complete, usable data samples.

2. The dual-knowledge-base design, which leverages both internal and external data, is a key strength. The proposed methods for feature alignment and clustering-based filtering to handle domain and modality gaps are well-motivated and appear effective.

3. The two-stage multi-modal fusion retrieval process is a significant contribution. By first using precise intra-modal retrieval and then refining with cross-modal signals, the method effectively mitigates the inherent modality gap, leading to more semantically relevant context for the generator.

4. The authors have conducted extensive experiments on a variety of datasets (MSCOCO, Flickr30K, and the domain-specific RSICD), evaluating the impact on multiple downstream tasks. The inclusion of several well-designed baselines and ablation studies (e.g., KFA-RAG, Combined-RAG) effectively dissects the contribution of each component of their framework.

### Weaknesses
1. The overall framework has many components and hyperparameters (e.g., clustering parameters, thresholds, fusion weights). This complexity might make the system difficult to tune and reproduce. While the appendix provides some details, a more in-depth analysis of hyperparameter sensitivity would be beneficial.

2. The construction of the knowledge base, particularly the clustering, filtering, and nearest-neighbor search for alignment, could be computationally expensive for very large-scale internal and external datasets. The complexity analysis in the appendix is helpful, but the practical implications on massive datasets remain a potential concern.

3. While the quantitative results are strong, the paper could benefit from more qualitative examples. Showing more side-by-side comparisons of the generated modalities from RAG4DMC versus the baseline methods would provide more intuitive evidence of its superior performance in generating semantically coherent and high-fidelity data.

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
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces RAG4DMC, a retrieval-augmented generation framework for data-level Missing Modality Completion (MMC). RAG4DMC overcomes challenges of fine-tuning large models and limited domain-specific data by structuring knowledge for more effective completion, leading to significant gains in downstream tasks. RAG4DMC builds a dual knowledge base —one from comprehensive in-dataset samples and another from external public datasets. It enhances performance by using feature alignment and clustering-based filtering to better manage differences in modality and domain.

### Strengths
* First framework for RAG tailored specifically for data-level missing modality completion problem. 
* Combination of internal complete samples with external public datasets to generate the dual knowledge base. 
* Two-stage multi-modal fusion retrieval strategy that leverages both intra-modal precision and cross-modal cues via pseudo-embeddings, ensuring the retrieval provides semantically consistent and highly relevant context for generation.
* Outperforms all baselines across general domain (MSCOCO, Flickr30K) and domain-specific (RSICD) datasets highlighting effectiveness in diverse tasks under high missing rates.

### Weaknesses
* High computational complexity of the proposed knowledge base construction especially the K-means clustering algorithm and iterative nearest neighbor search during cross-domain alignment. 
* In the setting it should be mentioned that the incomplete samples are encountered during the training phase (through missing rates) and the evaluation is performed on the complete samples.
* High sensitivity of the performance to specific thresholds associated with filtering and retrieval size.
* How does the quality of results associated with Direct generation baseline change with improvements in image caption generation models (for missing text) and text to image models (for missing image).
* The negative effects of missing modalities are increasingly significant in multimodal classification tasks such as audio-visual action recognition (for example, Mit-51, UCF-101, and Activity Net as examined in GTI-MM). This study does not address experiments related to audio-visual action recognition with varying rates of missing data.

### Questions
* Regarding Equations (20) and (21), were the fixed weights for candidate selection empirically optimized for both BLEU (image-to-text) and NIQE (text-to-image)? Additionally, is it appropriate to apply the same weighting scheme given the differences between these metrics?
* Under multi-modal fusion retrieval, it is not clear in terms of ablation if sim_{fuse} provides additional benefits over top-k retrieval results.

### Soundness
2

### Presentation
3

### Contribution
2
