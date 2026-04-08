## Human Reviewer 1

### Summary
This paper proposes the STBP framework for urban continual spatio-temporal forecasting. The method combines a general spatio-temporal backbone with a scalable contextual pattern bank, adapting to dynamically evolving spatio-temporal data through incremental pattern bank expansion while freezing the backbone network. This design effectively balances catastrophic forgetting mitigation, dynamic correlation modeling, and computational efficiency.

### Strengths
1. The task of continual spatio-temporal forecasting is interesting and has significant practical value for real-world applications.

2. The proposed method demonstrates substantial performance advantages over the compared baselines across multiple datasets.

### Weaknesses
1. While the application scenario is interesting, the proposed method lacks significant insight. The approach of freezing the backbone network while expanding a dynamic pattern bank to handle incremental node expansion appears to be a straightforward solution, and similar strategies have been proposed in prior work (e.g., EAC uses prompt pool expansion).

2. Insufficient baseline comparisons: Only six baselines are compared. More detailed experimental settings explaining how conventional spatio-temporal forecasting models (GWNet, STID, iTransformer) are adapted for this incremental scenario would help readers better understand the main results. 

3. The paper does not address a critical question: how much training data do existing baseline models require on new nodes to achieve acceptable performance? If baselines can reach satisfactory accuracy with only a short period of data (e.g., one week), the practical utility of the proposed continual learning approach would be significantly diminished. Conversely, if long-term data is necessary, this perspective could strengthen the motivation and method presentation.

### Questions
Please refer to Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper addresses the challenge of modeling dynamic and evolving spatio-temporal data in urban environments and proposes STBP, a continual forecasting framework. The method combines a general spatio-temporal backbone with a scalable contextual pattern bank, aiming to handle distributional drift and topological evolution without retraining the entire model. Extensive experiments conducted on three real-world streaming datasets demonstrate that STBP outperforms existing methods in terms of predictive accuracy, scalability, and resistance to catastrophic forgetting.

### Strengths
- Originality: The paper introduces a novel combination of frequency-domain modeling, lightweight linear graph attention, and a scalable prompt-based contextual parameter expansion strategy. This bridges the gap between spatio-temporal modeling and continual learning, making a meaningful contribution to the continual spatio-temporal forecasting field.

- Quality: The model architecture is well-structured with clear theoretical underpinnings. Each module is logically integrated to form a coherent pipeline. Specific designs address key challenges in CSTF, such as distributional drift, topological changes, and catastrophic forgetting.

- Clarity: The experimental setup is comprehensive, covering multiple representative real-world datasets. The appendix provides additional implementation details, which enhance reproducibility and interpretability.

- Significance: The proposed framework has practical value and potential applicability in domains such as traffic forecasting and meteorological analysis, where long-term adaptation to evolving topology and data distributions is essential.

### Weaknesses
1. Although the paper briefly acknowledges the challenge of cross-domain generalization, the main text lacks a detailed analysis of how STBP handles domain shifts, especially when there are significant structural differences between source and target tasks.

2. The current design tightly couples the contextual pattern bank with the backbone, which may limit the modularity of the backbone and restrict its transferability to other domains or tasks.

3. There is a typo in the title of Figure 5: "PESM-Stream" should be corrected to "PEMS-Stream".

### Questions
1. During parameter expansion of the contextual pattern bank, have redundancy or performance degradation issues been observed?

2. The use of frequency-domain modeling is claimed to help mitigate distributional drift. From a theoretical perspective, what are the advantages of this approach?

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
5

---

## Human Reviewer 3

### Summary
The paper proposes STBP, a framework that aims to unify STGNNs with continual learning. By combining a general spatio-temporal backbone and a scalable contextual pattern bank, the method addresses challenges of node expansion and concept drift through prompt-guided adaptation. Experiments on three public benchmark datasets show that STBP achieves competitive accuracy while maintaining efficiency and scalability.

### Strengths
1. Provides a general formulation for streaming spatio-temporal graphs that can be extended beyond traffic applications, with clear practical relevance to smart cities.  
2.  Conducts comprehensive experiments with diverse presentation formats, supporting reproducibility and interpretability.  
3.   Well-organized and clearly written.

### Weaknesses
1. The proposed method integrates multiple components, e.g., frequency-domain network, dual-stream linear graph attention, prompt-based continual learning, but each component is relatively straightforward and has been explored in related domains. The overall algorithmic novelty is limited.  
2.  While the authors propose a model-agnostic continual learning strategy, they do not demonstrate its generality across different backbone architectures, leaving its broader effectiveness unverified.

### Questions
1.  Figure 3 shows well-separated clusters in the contextual pattern bank $ P_{\tau} $, yet the method does not include explicit constraints to enforce such separation. Could the authors clarify what drives this differentiation?  
2.  Compared to similar prompt-tuning approaches such as EAC, what are the specific advantages and contributions of the contextual pattern bank?  
3. The output of FreNet is $ H_{\tau}^f $. How is it transformed so that the input of DLGA is $ H_{\tau}^s $? The author did not mention it in the paper.

### Soundness
3

### Presentation
4

### Contribution
2

### Rating
6

### Confidence
4