# BiHDTrans: binary hyperdimensional transformer for efficient multivariate time series classification

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 2, 8

## Abstract
The proliferation of Internet-of-Things (IoT) devices has led to an unprecedented volume of multivariate time series (MTS) data, requiring efficient and accurate processing for timely decision-making in resource-constrained edge environments. Hyperdimensional (HD) computing, with its inherent efficiency and parallelizability, has shown promise in classification tasks but struggles to capture complex temporal patterns, while Transformers excel at sequence modeling but incur high computational and memory overhead. We introduce BiHDTrans, an efficient neurosymbolic binary hyperdimensional Transformer that integrates self-attention into the HD computing paradigm, unifying the representational efficiency of HD computing with the temporal modeling power of Transformers. Empirically, BiHDTrans outperforms state-of-the-art (SOTA) HD computing models by at least 14.47\% and achieves 6.67\% higher accuracy on average than SOTA binary Transformers. With hardware acceleration on FPGA, our pipelined implementation leverages the independent and identically distributed properties of high-dimensional representations, delivering 39.4× lower inference latency than SOTA binary Transformers. Theoretical analysis shows that binarizing in holographic high-dimensional space incurs significantly less information distortion than directly binarizing neural networks, explaining BiHDTrans's superior accuracy. Furthermore, dimensionality experiments confirm that BiHDTrans remains competitive even with a 64\% reduction in hyperspace dimensionality, surpassing SOTA binary Transformers by 1–2\% in accuracy with 4.4× less model size, as well as further reducing the latency by 49.8\% compare to the full-dimensional baseline. Together, these contributions bridge the gap between the expressiveness of Transformers and the efficiency of HD computing, enabling accurate, scalable, and low-latency MTS classification.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces BiHDTrans, which is a binary hyper-dimensional transformer for multi-variate time series classification. The paper designs an HD encoder that converts real-valued features into binary HD vectors. The paper then modifies the transformer into a carefully designed binary version. Finally the paper implements the architecture on FPGA and achieves efficient inference.

### Strengths
The paper is technically challenging. Both training a binary transformers and deployment on FPGA are non-trivial and have significant contributions. If the entire process is reproducible, it could potentially have big impact on deploying LLM on edge devices.

### Weaknesses
* The selected tasks and datasets in the paper are quite simple. From Table 1, the small performance gap between full precision and the proposed method may be caused by the datasets are too simple. For Epilepsy dataset, the proposed method has a significant performance drop compared with full precision, which may hint the proposed method suffers from under performance.
* The proposed method has limited compatibility. It looks very difficult to add any additional transformers features, such as rope or alibi to the proposed  transformer.

### Questions
It sounds to me deploying a LLM using the proposed method has much more potential than multi-variate time series classification. What prevents this from happening?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a binary hyperdimensional transformer for multivariate time series (MTS) classification with FPGA acceleration. The model changes the conventional transformer computation (primarily the attention mechanisms) to HDC operations. Theoretical results show that binarizing the vector after projecting to high-dimensional holographic space loses less information than binarizing neural networks directly, and binarizing attention scores introduces information distortion that diminishes at higher vector dimensions. The algorithm, including an FPGA implementation of the pipeline, is evaluated against binary neural network models and other HDC models, showing competitive or improved accuracy and significantly better inference latency.

### Strengths
1. To the best of the reviewer's knowledge, this is the first work that fully integrates an attention mechanism within the HDC paradigm, improving the paper's originality. 
2. The paper provides strong theoretical support for the validity of binarization in BiHDTrans, which is further validated by empirical results. 
3. The paper demonstrates a practical path to fully binary, low-latency MTS classification with measured FPGA speedups (40×) while retaining accuracy competitive with full-precision one-block Transformers on these tasks. This directly targets IoT/edge deployment constraints.

### Weaknesses
1. Performance evaluation: While the experiment is run on several datasets, most of them are small. Additional datasets of a larger scale are recommended to stress-test the model, such as PTB-XL and Sleep-EDF. 
2. Ablation studies and sensitivity analysis: there is only one sensitivity analysis on the vector dimension. see questions for details.

### Questions
1. Replacing softmax with a hard binary mask removes temperature/normalization and can typically make transformer training brittle or overly sparse/dense depending on scale. What's the effective sparsity of $Ba$ during training and inference? 
2. What is the quantization level used for each dataset, and what quantization method is used? How sensitive is the performance with respect to quantization parameters? 
3. Using a different font for (or italicizing) the accuracies of full-precision model in Table $1$ is recommended, because the bolded results seem to be the best model performance excluding those of the full-precision model.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes to introduce hyperdimensional (HD) computing to the transformer architecture for the task of multivariate time series classification (MSTC). The authors explain the pitfalls of transformer architecture, especially in MTSC, and argue that HD computing can solve them. They prove several theoretical results showing 1) that binarizing a neural network within the HD computing setup can be more beneficial than doing so in the vanilla transformer architecture, and 2) information distortion within their framework remains bounded. 
The experimental results on a handful of UEA datasets show impressive computational gains and strong performance against other variations of compute-optimized transformers.

### Strengths
1.	An interesting exploration of applying HD computing to the transformer architecture with application to MTSC
2.	Several theoretical results showing why the proposed idea makes sense and should work reasonably in practice
3.	An experimental evaluation from both the accuracy and computational efficiency perspectives

### Weaknesses
1.	Lack of motivation for the MTSC task, specifically, where most models can run on edge devices. Although the authors do not mention it, SOTA TSFMs for classification (Mantis-8M (Feofanov et al. 2025), NuTime) are quite small and can be used easily on an edge device. It is not clear why we would want to use such involved techniques as HD computing in this setup.
2.	Evaluation is very limited, and no explanation for why only 7 datasets from the UEA repository were chosen is provided. There are now many MTSC datasets (see WOODS benchmark, for instance) that can be used to evaluate the proposed method as well.
3.	Outdated literature review omitting most of the recent advances in time series foundation models. As mentioned above, the authors do not mention the TSFMs proposed in the field in the last year or so, although they achieve strong performance in TSC.

### Questions
1. Can authors better motivate the need for using HD computing? For instance, one can take the MOMENT foundation model that is rather large and show the same findings with it? In this case, one would be able to see the usefulness of this approach. 
2. Is there any reason for using only 7 UEA datasets? 
3. Is there any particular reason to omit TSFMs from the introduction in Section 2? It seems that they are the most important use-case for the author's contribution, yet most references stop at 2023.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes BiHDTrans, a neurosymbolic architecture that performs self-attention entirely in high-dimensional binary (hyperdimensional) space. Queries/keys/values are constructed via HD binding, enabling end-to-end binarization without softmax and heavy FFNs. The paper argues (via concentration-of-measure style analysis) that binarizing in holographic space incurs less distortion than directly binarizing neural parameters. Empirically, BiHDTrans improves MTS classification accuracy over SOTA binary Transformers while delivering very large latency gains on an FPGA pipeline and retains accuracy under sizable reductions in HD dimensionality.

### Strengths
• The unification of hyperdimensional computing and Transformer-style attention is conceptually elegant and technically novel, offering a neurosymbolic route to full binary inference.
• Hardware results substantiate the efficiency claim, with real FPGA measurements showing major latency and energy gains.
• Theoretical analysis, while asymptotic, provides an intuitive justification for why HD binarization preserves representational fidelity.

### Weaknesses
• Evaluation is limited to classification; forecasting or anomaly detection would strengthen generality.
• Finite-dimensional behavior is not explored; empirical distortion plots could help quantify theory–experiment alignment.
• Training details (learning rates, temperature parameters, or binarization schemes) are somewhat under-specified for reproducibility.

### Questions
Can the authors provide an empirical plot of distortion versus dimensionality to substantiate the concentration-based theory? How sensitive is model accuracy to the number of attention heads and HD dimension? Could a small forecasting or anomaly detection task be included to demonstrate broader applicability?

### Soundness
3

### Presentation
3

### Contribution
3
