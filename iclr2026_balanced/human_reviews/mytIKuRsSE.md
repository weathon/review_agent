## Human Reviewer 1

### Summary
This paper addresses the overlooked problem of Dual-level Noisy Correspondence in multi-modal entity alignment, where both intra-entity attributes and inter-graph alignment pairs contain substantial noise. The authors propose RULE, a unified framework that estimates pair reliability through uncertainty and consensus modeling, performs noise-aware robust learning and multi-modal fusion during training, and further integrates a test-time reasoning module using an MLLM to refine alignment decisions. Experiments on several benchmarks demonstrate that RULE significantly improves robustness across a wide range of noise levels and achieves clear gains over state-of-the-art methods.

### Strengths
S1. The paper convincingly argues that Dual-level Noisy Correspondence is ubiquitous in real MMKG scenarios, yet largely ignored by prior MMEA works. The motivation is well-founded and supported by quantitative evidence of high noise ratios.

S2. RULE provides an end-to-end framework that integrates reliability estimation, noise-aware learning, multi-modal fusion, and test-time reasoning. The combination of training-time robustness and inference-time semantic reasoning is novel and compelling.

S3. Experiments cover multiple datasets, modalities, and a wide spectrum of noise levels (0%–70%). Results show large improvements over SOTA baselines in both low-noise and extremely noisy regimes, demonstrating the practical robustness of RULE.

S4. The evidential uncertainty modeling and greedy consensus estimation are well-justified. Visualizations (e.g., Fig. 3) clearly show distinguishable distributions between clean and noisy correspondences.

### Weaknesses
W1. RULE contains several components (uncertainty estimation, consensus reasoning, robust learning, fusion, test-time MLLM reasoning). While each part is motivated, the overall system introduces significant complexity, making it hard to isolate which component contributes most.

W2. Although TTR appears effective, the paper lacks a thorough analysis of computational overhead, scalability, and failure modes. Using a large MLLM at inference time raises concerns for deployment.

W3. Although TTR is conceptually strong, the paper could benefit from more real examples illustrating how MLLM reasoning corrects embedding-based mistakes, especially under ambiguous or incomplete attribute sets.

### Questions
1. How sensitive is RULE to the thresholds in pair division? Are there universal settings across datasets, or does each dataset require tuning?

2. What is the computational overhead of TTR? How many calls to the MLLM are required per entity, and is inference feasible for large-scale KG alignment?

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper addresses a critical issue in MMEA, namely Dual-level Noisy Correspondence (DNC), which refers to the misalignments at both the intra-entity, entity-attribute, and inter-graph, entity-entity, attribute-attribute, levels. This noise can severely degrade performance in multi-modal knowledge graph alignment tasks, which is a common problem in real-world benchmarks. To tackle this issue, the authors propose a framework called RULE, which robustly estimates the reliability of these correspondences using a two-fold principle, uncertainty and consensus. RULE mitigates the impact of noisy correspondences during training and incorporates a test-time correspondence reasoning module to enhance the robustness during inference. Experiment demonstrates that RULE outperforms existing state-of-the-art methods across several datasets and noise settings.

### Strengths
1. The identification and study of dual-level noisy correspondence is a novel and significant contribution to the MMEA task. It acknowledges that real-world knowledge graphs often contain significant noise both within entities and across graphs, which previous methods have largely ignored.

2. The proposed RULE framework leverages reliability estimation using uncertainty and consensus to combat noisy correspondences. The test-time correspondence reasoning module is a key innovation that improves entity identification at inference time, ensuring better alignment even when certain correspondences are noisy or unreliable.

### Weaknesses
1. This paper includes some parameter analysis like trade-off parameter λ, threshold β, and temperature τ, but there is no comprehensive discussion of how these hyperparameters affect the model’s robustness under various noise settings. For instance, how does RULE perform with different datasets or in cases where the noise level is very high, e.g., >50%?

2. RULE relies heavily on pre-trained CLIP models for image and text embeddings. While this is a reasonable approach, it could limit the flexibility of the model in scenarios where domain-specific or fine-grained embeddings are needed.

3. It seems lack of how the framework handles extreme cases, such as when the noise level exceeds 50%, or when the correspondence noise is extremely high in one modality (e.g., image attributes).

4. While the TTR module improves the robustness during inference, it may introduce additional computational overhead. The reasoning process involves complex reasoning steps and large language model queries, which could be slow and computationally expensive, especially for large-scale graphs.

### Questions
1. Could you provide a more detailed analysis of how the hyperparameters (λ, β, τ) affect the performance of RULE under different noise levels? Are there specific configurations where the model is more or less sensitive? It would be useful to explore whether hyperparameter tuning plays a significant role in noise resilience.

2. How do you think RULE compares to methods like graph matching or other representation learning-based approaches that also handle noisy correspondences? Some methods like [1] also addresses noise in multi-modal entity alignment, which names semantic consistency. It would be better if you can compare and analyze them.

3. I`m wondering how does RULE perform in extreme noise scenarios like >50% noise or heavily corrupted attributes in one modality? Would the model still perform robustly, or would it need additional modifications?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper makes the first attempt to explore dual-level noisy correspondences (DNC) in the field of multi-modal entity alignment (MMEA). To address this issue, the proposed RULE could not only achieve robust cross-graph learning and multi-modal fusion during training, but also incorporates a correspondence reasoning module to enhance test-time robustness and thus achieves more accurate entity alignment across heterogeneous multi-modal knowledge graphs. Extensive experiments demonstrate that RULE outperforms state-of-the-art MMEA methods, while showing significantly slower performance degradation as noise levels increase.

### Strengths
1. This paper is well-motivated and is clearly organized.
2. The intuitive illustrations in Figure 1 make the DNC phenomenon and its negative impacts clear.
3. The experiments are convincing, e.g., the visualizations (Figure 3b) of the reliability estimation, pair division process (Figure 4) further prove the effectiveness of the proposed RULE.
4. Most impressively, Appendix B not only analyzes the noise statistics in real-world datasets but also discusses the underlying causes of DNC. Correspondingly, the framework demonstrates strong performance under both noise-injected and inherent DNC settings, further validating the presence of DNC in real-world datasets.

### Weaknesses
1. Since both uncertainty and consensus are estimated through cross-graph relationships, I don’t understand why such relationships could be employed to estimate reliability of intra-graph attributes. A more intuitive and detailed explanation of this design would be very helpful for understanding.
2. In my understanding, the MLLM-based reasoning module aims to uncover underlying associations between attributes. I am interested in what kinds of associations can be further mined by the MLLM, for example, beyond the cases illustrated in Figure 1(c). Furthermore, I wonder whether the proposed MLLM-based reasoning module could be extended to other tasks, such as image-text retrieval.
3. Since the proposed RULE is a model-agnostic framework, which means RULE could be applied on various backbones. It would further strengthen the contribution if the authors could demonstrate the generalizability of RULE across more diverse backbones.
4. I want to see more details about the computational complexity of the proposed RULE. In particular, the MLLM-based reasoning module may introduce considerable time and memory. It would be valuable for the authors to provide a detailed analysis of the reasoning time, GPU requirements.

### Questions
Please see the Weaknesses.

### Soundness
3

### Presentation
4

### Contribution
4

### Rating
8

### Confidence
5

---

## Human Reviewer 4

### Summary
In this paper, the authors study a new and practical problem in MMEA called dual-level noisy correspondence (DNC), where both intra-entity and inter-graph correspondences might be noisy. To address this, the authors propose a robust framework (RULE) that estimates correspondence reliability based on a two-fold principle. With the estimated reliabilities, RULE performs noise-aware attribute fusion and discrepancy elimination, and further introduces a correspondence reasoning module to enhance test-time robustness. The authors conduct thoughtful experiment designs and the results show that RULE achieves significant improvements over existing methods. Overall evaluation, this is a good paper with interesting motivation and solid results.

### Strengths
This paper reveals a highly practical yet underexplored challenge (DNC) in MMEA. I appreciate the motivation of this work and believe that tackling DNC is valuable, as intra-entity and inter-graph noisy correspondences are often inevitable in real-world datasets. Particularly, the authors further provide statistical analysis of the noise distribution in real-world datasets in the Appendix, which is convincing and further highlights the necessity of addressing DNC. To address the DNC problem, the paper proposes a unified principle to estimate the reliability of correspondences across different levels. Based on the estimated reliability, RULE enhances robustness against DNC in both the training and inference phases, pioneering a full-process robust learning paradigm for MMEA. Overall, the paper tackles a meaningful and practical problem with a technically sound solution.

### Weaknesses
Although the paper is well-motivated and presents a technical-sound solution, I have several questions and suggestions for further improvement as follows:
- There are some prior efforts on learning with noisy correspondence (NC) in cross-modal retrieval. Since DNC could be viewed as a special case of NC in MMEA, I wonder whether existing NC-oriented methods are adequate for addressing the DNC challenge. I encourage the authors to clarify the distinctions and advantages of RULE over prior NC-oriented studies. Moreover, I notice that the paper ``Tackling Uncertain Correspondences for Multi-Modal Entity Alignment [A]” share some similar motivation with this submission, could the authors give some discussions?
- I am curious about how the DNC ratios (0%, 20%, and 50%) used in the experimental settings are determined. Are these noise ratios chosen based on empirical observations or following prior studies? It would be helpful if the authors could clarify the reasons behind these choices.
- As far as I know, uncertainty has been widely used as a principle for identifying noise input in classification tasks. However, I’m not clear about why uncertainty is insufficient to measure the reliabilities of the correspondences in MMEA. In other words, what motivates the introduction of the consensus principle?
- I am particularly curious about how the TTR module rectifies correspondences during test time. It is recommended that the authors include visualization examples to illustrate this process.

### Questions
- Provide more discussion on the differences between the proposed method and existing NC-oriented approaches and the related work [A].
- Explain the choice of noise ratios (0%, 20%, 50%) in the experiments.
- Justify the necessity of the consensus principle.
- Visualize the test-time reasoning process to offer more intuitive insights into how the TTR module rectifies correspondences.

### Soundness
4

### Presentation
3

### Contribution
4

### Rating
8

### Confidence
5