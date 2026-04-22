# When LLMs Encounter Open-world Graph Learning: A Fresh View on Unlabeled Data Uncertainty

- Avg Score: 4.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 4, 2

## Abstract
Recently, large language models (LLMs) have driven a systematic shift in the graph ML community through the adoption of text-attributed graphs (TAGs). Although a variety of frameworks have been developed, most fail to properly address the challenge of data uncertainty in open-world environments, which is vital for real-world deployment. A representative source of such uncertainty is the limited availability of labels in large-scale datasets due to high annotation costs, where unlabeled nodes may either belong to known classes or represent novel, unknown classes. While node-level out-of-distribution detection and conventional open-world graph learning attempt to tackle this problem, two core limitations remain:  Insufficient methods — TAGs integrate textual and structural information, yet existing approaches typically optimize semantics or topology in isolation for unknown-class rejection, limiting their effectiveness; \ding{173} Incomplete pipelines — handling unknown-class nodes is essential for model re-updates and long-term deployment, but most studies conduct only idealized analyses, such as assuming a predefined number of unknown classes, which restricts practical utility. To overcome these issues, we introduce the Open-world Graph Assistant (OGA), an LLM-based framework. OGA first performs unknown-class rejection via adaptive label traceability (ALT), harmoniously combining semantic and topological cues, and then applies the graph label annotator (GLA) for unknown-class annotation, allowing unlabeled nodes to contribute to model training. In essence, OGA offers a new pipeline that fully automates the handling of unlabeled nodes in open-world environments, and we establish a systematic benchmark covering four key aspects to validate its effectiveness and practicality through extensive experiments.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes OGA, a fully automated LLM-based data annotation framework designed to address unlabeled data uncertainty in open-world graph learning. Specifically, the framework introduces an Adaptive Label Traceability (ALT) module that integrates semantic and topological reasoning for unknown-class rejection (UCR), followed by a LLM-driven Graph Label Annotator (GLA) for unknown-class annotation (UCA). The proposed framework is theoretically grounded, and extensive experiments are conducted from different aspects to demonstrate the effectiveness of the proposed framework.

### Strengths
S1. The motivation behind each proposed module is clearly and thoroughly explained.
S2. The proposed integration of semantic and topological optimization within the ontology space is novel.
S3. The authors provide a detailed theoretical analysis of the proposed framework.
S4. The authors conduct comprehensive experiments from various key perspectives to thoroughly validate the effectiveness of the proposed framework.

### Weaknesses
W1. Lack of clarity in framework details. 
    W1.1. The paper emphasizes that one of its key contributions is leveraging a pre-trained graph-language encoder to obtain node representations. However, no description or implementation details are provided regarding this component. Therefore, it remains unclear whether the “graph-language encoder” refers to (a) an independent, pre-trained module used to initialize node embeddings, or (b) the integrated semantic–topology optimization in the ontology space.
    W1.2. Please include a description of the overall framework illustrated in Figure 1/2 in the figure caption.
W2. The paper emphasizes that leveraging a pre-trained graph-language encoder is a key advantage over graph-only encoders. If the “graph-language encoder” refers to a pre-trained module used to initialize node embeddings, this claim is not empirically supported. Please conduct ablation study on graph-language encoder vs. graph-only/language-only encoders.
W3. The authors claim that the proposed framework addresses the challenge of unlabeled data uncertainty, which is particularly critical in large-scale datasets due to high annotation costs. However, the scalability of the proposed framework is evaluated only on one large-scale graph (i.e., Arxiv), please evalute the scalability of the proposed framework on more large-scale datasets.

### Questions
Please see W1.1, W2, and W3.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a two stage framework that tries to perform unknown class rejection and then annotation when facing TAGs with both labeled and unlabeled nodes, where the unlabeled nodes might be from known or unknown classes. For the UCR, they aim to combine semantic and topological together with graph-language encoders and ontology representation learning. For the UCA, they first identify communities, then annotate low degree nodes with LLM and allocate high degree node class by referring to its neighbors, and finally perform some inter community fusion.

### Strengths
- The paper focuses on a realistic problem and setting that can be widely encountered in open world graph scenarios and point out the limitations of previous works and their practicability. 
- The argument that we need to add UCA on top of UCR is valid and reasonable, and the addition of UCA can be more beneficial to the training subsequently. Also, the adoption of LLM is a reasonable choice. 
- The experimental section includes clearly guided research questions and studies regarding ablation of models, hyperparameters and efficiency

### Weaknesses
- The baselines seem to include only some previous graph based OOD detection baselines, I am wondering if there are other some baselines that include LLM in the loop that can better demonstrate the superior of the proposed framework.
- Some of the designs might rely on some assumptions and other models. For instance, the UCR indicates that we need a pretrained graph language encoder to provide us with embedding, so the performance of UCR might be largely determined by the encoder choice. Also, the UCA design depends on the homophily structure of the graph. 
- The entire framework as well as the presentation of the whole paper is rather complicated and dense, which makes it hard to train with many hyperparameters and determine the key technical novelty.
- The evaluation of UCA annotated labels might be improved beyond the current semantic similarity and improvement from model to a potentially more quantifiable and interpretable metrics

### Questions
- Not sure if I miss this, what is the language-graph model used in UCR, is there an ablation with only GNN or only LLM?
- The current datasets are mainly regarding citation networks, there are still many other TAGs like in e-commerce and others, is there any reason that we don't include more sets of datasets for evaluation?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes OGA, a two-stage LLM-guided framework for open-world graph learning. The first stage detects unknown nodes based on class probability, and the second stage generates new labels with LLMs via efficient topology-aware prompting.

### Strengths
- The paper tackles a practically relevant yet challenging open-world graph setting.
- The authors propose additional techniques to reduce the number of LLM calls, instead of inferring across all nodes.
- OGA demonstrates strong results on multiple benchmarks.

### Weaknesses
- The overall framework feels overly complex, consisting of many submodules and three types of LLM prompting. It is not clearly justified why such multi-stage prompting is necessary instead of a simpler unified design.
- The handling of high-degree nodes may be problematic. For instance, if most of their neighbors are also unknown, the proposed degree-aware annotation could amplify uncertainty.
- The community merging process raises semantic concerns. Unknown-class labels might belong to completely different domains from existing known ones (e.g., biology (unknown) vs. machine learning-related categories (known)), in which case community merging could blur true class boundaries.
- The figure is too cluttered to straightforwardly understand the full pipeline.

### Questions
- What would be the performance of a zero-shot LLM classifier that directly predicts all labels? Could a well-designed retrieval-augmented LLM prompting achieve comparable performance without the two-stage complexity?
- Could the authors show the wall-clock time of OGA including all LLM calls, and compared it against non-LLM-based baselines?

### Soundness
2

### Presentation
3

### Contribution
2
