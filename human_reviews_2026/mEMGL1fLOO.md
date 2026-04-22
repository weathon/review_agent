# DocPruner:  A Storage-Efficient Framework for Multi-Vector Visual Document Retrieval via Adaptive Patch-Level Embedding Pruning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
Visual Document Retrieval (VDR), the task of retrieving visually-rich document pages using queries that combine visual and textual cues, is crucial for numerous real-world applications. 
Recent state-of-the-art methods leverage Large Vision-Language Models (LVLMs) in a multi-vector paradigm, representing each document as patch-level embeddings to capture fine-grained details. 
While highly effective, this approach introduces a critical challenge: prohibitive storage overhead, as storing hundreds of vectors per page makes large-scale deployment costly and impractical. 
To address this, we introduce **DocPruner, the first framework to employ adaptive patch-level embedding pruning for VDR to effectively reduce the storage overhead**. 
DocPruner leverages the intra-document patch attention distribution to dynamically identify and discard redundant embeddings for each document. 
This adaptive mechanism enables a significant 50-60% reduction in storage for leading multi-vector VDR models with negligible degradation in retrieval performance. 
Extensive experiments across more than ten benchmark datasets validate that DocPruner offers a robust, flexible, and effective solution for building storage-efficient, large-scale VDR systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces an adaptative pruning technique to reduce the number of stored vectors in late interaction visual retrievers.
While prior pruning/merging methods do exist for text multi-vector retrievers [1] [2], and are even implemented for visual retrievers in the original ColPali paper and [3], this paper introduces an "adaptative" pruning technique, in which the pruning ratio is automatically adapted given a document. I believe this is an interesting problem to study, as information dense documents intuitively (and as shown in the ColPali paper) will not be as prunable as information sparse ones. The authors then provide a series of interesting findings from their experiments (ex: pruning vs merging).

[1]  A Study on Token Pruning for ColBERT, Lassance, arxiv.org/abs/2112.06540

[2] Reducing the Footprint of Multi-Vector Retrieval with Minimal Performance Impact via Token Pooling, Clavié, arxiv.org/abs/2409.14683

[3] Towards Storage-Efficient Visual Document Retrieval: An Empirical Study
on Reducing Patch-Level Embeddings, Ma, arxiv.org/abs/2506.04997

### Strengths
The question the paper is trying to tackle is of importance. With a simple and efficient design, this paper introduces a token importance score and uses it to inform an adaptive thresholding method. 
The method seems to work well across the board, and the paper is generally fairly simple to follow and read. The authors experimented across different models and datasets, painting a through picture of expected gains from the proposed method.

### Weaknesses
**Main Weakness**

Observation 2 claims pruning surpasses merging. However, from my understanding, these results are not decorrelated from the "adaptative" vs "non-adaptative" pruning, in which more complex documents are less pruned within a given corpus (sem-cluster, the best baseline, is non adaptive). I do not understand how this claim can be made without accounting for this factor. 

More generally, an independant study of (1) the pruning design (2) the adaptive mechanism could be warranted. 

(1) The pruning design ressembles Lassance [1] attention-based pruning mechanism and could be independently compared to the baselines (such as sem-cluster) by fixing the top k kept tokens. 

(2) The optimal document specific threshold (determined by your method) could be applied to semantic-clustering to make it benefit from doc-level adaptive clustering for a fair comparison. A simpler and more practical method could also be to apply semantic clustering at a corpus level (ex: fix 0.5 ratio, but enable clustering to be done once across document vectors instead of for each doc). 

Figure 5 partially tackles these questions but results intriguingly do not seem to indicate that the adaptative nature of the method is what helps most (see question 2). Furthermore, these results are computed with Jina-v4 backbone for which semantic-clustering (with fixed ratios) remains vastly superior. I feel proof of the benefits of adaptive thresholding (which should surely help) are not sufficiently demonstrated in this work.

As is, the experiments also do not give me the confidence that the proposed method for determining token importance (without adaptive selection) surpasses equivalent fixed ratio baselines (*evidence actually points otherwise if we piece together "attention-ratio" from Fig 5 and "Jina v4 semantic clustering" from Figure 2*). Regardless, applying adaptive thresholding with dynamic pruning ratio to these strong fixed-ratio baseline methods would be a strong addition to the paper and could lead to great results.

**Explorations around the token selection** 

Given it is the main contribution of this work, I feel the dynamic token thresholding criteria could be further explored and go beyond std dev based thresholding. At the very least, visualizing the distributions of token importances amongst a document, or number of kept tokens per document across a corpus could help justify these design choices.

**Remarks**

-  I feel Figures 2, 4, 5 are a bit redundant in the information they show (7 lineplots in total). Furthermore, some of the colors and opacity are a bit hard to see.
- The strong performance of JinaV4 with semantic clustering across the board, while justified in the paper, question the utility and the general nature of the proposed method and findings should probably be discussed under this light.
- Figure 5 comes without explanation for which model is used. Figure 22 wrongly states that multiple model variants are shown when only Jina v4 is shown (piecing together with figure 2).

### Questions
1. The adaptation factor k relies on varying the standard deviations from the mean token level importance scores. Would it be possible to plot the actual distribution and verify whether the distribution is in-fact Gaussian like (versus something bimodal etc ?) I guess this could inform the design of the thresholding? A histogram of the number of kept vectors in a given dataset could also be an interesting visualization.

2. From my understanding, in Figure 5, the token importance method is compared either with (a) keep the top K vectors (with K constant across documents) - "attention ratio" or (b) use adaptive pruning "DocPruner". We see that "attention ratio" is actually very competitive with DocPruner on all datasets but ESG Human (the most noisy one). Would this indicate that the "token importance" method is what brings most of the gains, wrt to the adaptive thresholding which is marginally useful here ? If this is the case, this puts the paper framing into question.

3. Could we have visual examples of documents with high pruning ratios (and the number of kept vectors) and inversely ? This would help build better intuitions about the method.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an efficiency optimization method for text-rich document retrieval. First, the importance of each patch in the document image is calculated based on the global token obtained from the VLM, resulting in an initial patch-level importance score. Then, parameters such as the mean and standard deviation are used to define the set of pruned patches. Finally, experiments comparing this method with other pruning methods demonstrate its superior stability.

### Strengths
1. This is a plug-and-play method that helps improve retrieval efficiency.

2. The method is simple and effective, and the experimental comparison with different types of cropping methods shows better results.

### Weaknesses
1. The paper's innovation is somewhat limited. Its core approach is to calculate the importance of each patch using a global token, a calculation method that has been widely used in other visual token cropping methods. Furthermore, I haven't seen any proof or rationale for setting a parameter formula based on mean and variance, and the contribution of each parameter remains unclear.

2. Document images contain a large amount of multimodal fine-grained information. Queries may involve information from different patches, and the information between different patches may be interrelated. Therefore, directly cropping patches may destroy the overall semantic continuity of the document image.

### Questions
See the Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes DocPruner, an adaptive embedding pruning framework for Visual Document Retrieval (VDR) that dramatically reduces storage requirements while preserving retrieval accuracy. Unlike existing multi-vector LVLM-based methods that store hundreds of patch-level embeddings per document, DocPruner exploits intra-document attention distributions to prune redundant embeddings selectively. This adaptive strategy achieves 50–60% storage reduction with minimal performance loss across ten benchmark datasets, providing an efficient, scalable solution for large-scale VDR deployment.

### Strengths
- The paper proposes the Adaptive Embedding Pruning, which realizes the first adaptive patch-level pruning framework that dynamically removes redundant embeddings based on intra-document attention.
- High efficiency is achieved with minimal performance loss, in which 50–60% is achieved in storage reduction while maintaining competitive retrieval accuracy across diverse VDR benchmarks.
- The proposed method is compatible with existing multi-vector LVLM-based retrieval models, enabling efficient and flexible deployment in large-scale VDR systems.

### Weaknesses
In my opinion, the paper focuses on the embedding pruning in the visual document retrieval tasks; however, the related works on general MLLM pruning are not discussed, as well as compared in the experiments, which weakens the presentation and evaluation of the paper to some degree.

### Questions
Please refer to the weakness part.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces DocPruner, which addresses the significant issue of storage overhead in modern patch-level retrievers with VLMs. It proposes several contributions :
1. An adaptive patch-level embedding pruning strategy to reduce the storage overhead based on the quantification of patch importance with global token attention.
2. An attempt to ground theoretically the approach with the information bottleneck principle.
3. An extensive and sound experimental validation and analysis.

### Strengths
**originality**
+ Good. The paper proposes a simple and straightforward approach for quantifying patch importance in the context of VLM-based retrieval. The proposed approach is simple yet possesses desirable properties, including query-agnosticity and document adaptivity. The authors also provide an analysis of their approach through the spectrum of the information bottleneck principle. 

**quality**
+ The soundness of the paper is very appreciable. 
+ Quality and completeness of the experimental study, including comparison with important related works. 

**clarity**
+ A very well-written paper with clear motivations, clear formalizations,  and a sound and complete experimental validation.
+ Quality of the positioning with respect to the state of the art

 **significance**
+ The proposed contribution answered one of the important shortcomings of multi-vector visual document retrieval. The latter is a very important paradigm in RAG-based applications, so the impact of the paper could be important.
+ The experimental validation shows the benefit of the proposed approach.

### Weaknesses
+ The quantification of patch importance relies on the idea of using a global token as an approximation of the document's overall semantics, and in particular, the EOS token.  The importance score is thus just the attention the patch receives from the global token. This choice makes sense, but it could be discussed and analyzed in greater detail. In particular, it is only valid for transformer-type and attention-based architectures. These architectures are the most common, but other types of architectures are also emerging. The proposed approach is therefore not architecture-agnostic.
+ The pruning strategy is based on classical statistical moments of the importance scores, namely the mean and standard deviation. The approach is limited to the level of a document page and does not consider the entire document or the entire collection, which is a limitation that could be addressed to improve the pruning strategy. 
+ The approach is strongly dependent on the choice of the adaptation factor, which is not easy to set up. 
+ Some important references are missing. In particular, some recent works have been proposed on  Token Pruning in Late-Interaction Retrieval Models such as [Zong et al,. 25](https://hal.science/hal-05037885/). Other related works are also the idea of meta-tokens [Xiao et al,. 2025](https://arxiv.org/pdf/2509.18095).


some typos :
 l289 : pathc => patch

### Questions
+ One of the advantages of DocPruner is the fact that it is a plug-and-play approach. In particular, the method used to calculate the importance scores of each patch in relation to the document, as well as the method used to perform adaptive thresholding, could be modified. What alternatives to the proposed approach could be used to better consider the document or the collection as a whole? 
+ The proposed approach strongly relies on the quality of the VLM embedding, and in particular, the quality of the EOS token. The EOS token is a topic of strong discussion in the community. It is a very strong assumption to consider that the EOS token representation encapsulates the document's overall semantics. It could be discussed more.

### Soundness
3

### Presentation
4

### Contribution
3
