# Neural Message-Passing on Attention Graphs for Hallucination Detection

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6

## Abstract
Large Language Models (LLMs) often generate incorrect or unsupported content, known as hallucinations. Existing detection methods rely on heuristics or simple models over isolated computational traces such as activations, or attention maps. We unify these signals by representing them as attributed graphs, where tokens are nodes, edges follow attentional flows, and both carry features from attention scores and activations. Our approach, CHARM, casts hallucination detection as a graph learning task and tackles it by applying GNNs over the above attributed graphs. We show that CHARM provably subsumes prior attention-based heuristics and, experimentally, it consistently outperforms other leading approaches across diverse benchmarks. Our results shed light on the relevant role played by the graph structure and on the benefits of combining computational traces,  whilst showing CHARM exhibits promising zero-shot performance on cross-dataset transfer.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces CHARM (Catching Hallucinated Responses via Message-passing), a unified framework for hallucination detection in LLMs. CHARM represents attention maps and residual activations jointly as attributed graphs, where tokens are nodes and attentional flows define edges. Each node and edge is enriched with features derived from activations and attention scores. Hallucination detection is then formulated as a graph learning problem, addressed via a Graph Neural Network (GNN). The paper shows that CHARM subsumes existing heuristics such as Lookback Lens and LLM-Check and empirical superiority across multiple benchmarks. Experiments cover token-level and response-level hallucination detection across diverse datasets, with comprehensive ablations on graph structure, sparsity thresholds, and zero-shot transfer. CHARM consistently outperforms baselines and shows that incorporating structured, graph-based computation leads to measurable gains over attention- or activation-based heuristics.

### Strengths
- The authors presents a novel and unified framework that models LLM computational traces as attributed graphs, effectively integrating both attention scores and activation signals into a single structured representation.

- The proposed approach contributes a new methodological direction by connecting graph neural networks and hallucination detection, opening possibilities for broader applications in explainable AI and reliability of LLMs.

- It provides strong theoretical grounding by formally proving that CHARM subsumes well-known heuristic detectors such as Lookback Lens and LLM-Check, demonstrating clear expressive generality.

- The experimental evaluation is comprehensive, covering both token-level and response-level hallucination detection tasks across five datasets with diverse characteristics.

- CHARM consistently outperforms all baseline methods, including heuristic and activation-based detectors, showing clear empirical advantages in predictive performance.

- The ablation studies and analyses are thorough, investigating the role of graph structure, sparsity thresholds, message passing, and zero-shot transfer in a systematic and transparent manner.

- The paper is well written and organized, with clear mathematical formulations, logical flow, and illustrative figures that make the method easy to understand.

- The results are reported with honesty and nuance, including cases where CHARM’s performance varies in zero-shot transfer scenario, which enhances the paper’s credibility.

### Weaknesses
- The performance drop observed when combining activations with attention maps (Table 1) indicates that CHARM may be sensitive to feature selection and layer choice. A deeper discussion on this sensitivity, along with possible mitigation strategies, would strengthen the paper.

- The current framework still relies on heuristic choices to determine which layer of activations to include in the attributed graphs. Developing a more systematic or data-driven approach to layer selection would make the framework more complete and principled.

- There are a few minor writing issues; for instance, inline citations in the first paragraph of the Related Work section should be placed in parentheses.

### Questions
- How consistent are the results across different activation layers beyond layers 24 and 32? Could the model be extended to dynamically identify and select the most informative layers during training?

- What mitigation strategies could prevent the degradation in hallucination detection accuracy observed when incorporating additional features, such as activations, into the attributed graph?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper describes using a message passing network to use the attention of a transformer to hallucination detection. The method description is straightforward but the implementation and needed input and engineering work is complicated. Experiments across diverse benchmarks (contextual, factual, reasoning) and LLMs (LLaMa-2, Mistral) show that CHARM consistently outperforms existing methods. They evaluation on Natural Qeustions and reasoning datasets.

### Strengths
the paper uses internal signal with a well motivated method for attribution. I find this elegat and novel contribution to the best of my knowledge. 

The method via GANs moves from fixed, hand-crafted heuristics  to a learnable, expressive model. This allows CHARM to capture more complex and none obvious patterns in the computation that may signal a hallucination.

very good evaluation, very much appriciated, looks good. 

The finding that optimal features (attention-only vs. attention+activations) depend on the type of hallucination is a valuable insight.

open sourcing the code very much appriciated.

### Weaknesses
Like all methods that rely on internals of methods so CHARM too requires "white-box" access to the model's internal activations and attention matrices.

While GNN inference is reported as fast, it has quite some overhead and implmentation.

### Questions
How does the performance (both accuracy and latency) of CHARM degrade as sequence length increases into the tens of thousands of tokens?

The paper uses activations from a single, pre-selected layer (was it layer 24?). Why choosing a single layer? Could performance be improved by integrating activations from multiple layers ? did you try?

The conclusion mentions integrating logits as a future direction. How do the you envision this?

do the you believe the "signature" of a hallucination in the attention graph is universal?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
In this paper, the authors analyze hallucination detection using LLM-internals by employing message passing over a unified attributed graph, incorporating its activations and attention scores. In detail, the graph is set up by assuming tokens as nodes, edges representing the direction of tokens attending to future tokens in a causal manner, the node attributes represented by on-diagonal attention scores and activations, and edge attributes represented by off-diagonal attention scores. The paper proposes Charm, by using a Graph Neural Network over this graph, with token-level hallucination detection cast as a node classification task, and response level detection cast as a graph classification task.

### Strengths
1) The paper is well-written and introduces the core components of the proposed method in a clear and concise manner. The paper presents a structured and principled framework to detect hallucinations, by using a GNN to learn from an attribution graph that combines attention scores alongside activation states.

2) With the abstracted framework introduced, the authors show that prior techniques such as LookbackLens and LLM-Check can be represented by specific design choices and instantiations of the GNN used over the general attribute graph. 

3) Through empirical evaluations on both token-level hallucination detection and response-level detection, Charm is shown to achieve good empirical performance, and achieves significant improvements over existing baseline. Furthermore, the method is shown to require fairly low memory overheads with sparsification, without inheriting a large drop in performance simultaneously.

### Weaknesses
1) The proposed method is considerably more complex to deploy and utilize when compared to baselines: it involves building a potentially large graph for each generation with the number vertices being the same as the maximum token length, and then extracting all attention scores and activations for all layers. This is of course followed by supervised training the GNN on a set of truthful and hallucinated samples, whose distribution is close to the test samples seen at inference time. On the other hand, several baseline methods such as SelfCheck, INSIDE, LLMCheck do not utilize any trainable network components. Furthermore, given that CHARM explicitly trains the GNN, other training based methods such as ITI [1] form a strong baseline of comparison, since the degree of discriminability can be adaptively set for detection.


2) The attribution graph is dense prior to sparsification, and quadratic in the sequence length of  N tokens: $L * H * N * (N+1)/2$ attention scores are computed for an LLM of L layers and H attention heads per layer. While Table-4 does indicate that good performance is achieved even at high-sparsity levels, it is unclear how this would scale for longer sequence lengths. Could the authors kindly report the computation run-time and memory overheads over some fixed sequence lengths such as 1K, 10K and 100K? Furthermore, what is the impact on the generalization of the trained GNN when the context size potentially changes by a large manner at test time? 


3) Evaluation Metrics: The empirical evaluation relies exclusively on AUROC and AUPR. While these metrics give a good overall view of performance, in practice it is crucial to achieve high TPR at low FPR rates. Particularly with hallucination detection, it is quite important to report standard detection metrics such as TPR at 5% or 10% FPR and F1 score, to help assess and compare the efficacy of the proposed method?


4) Furthermore, the main experimental evaluations are reported only on LLaMa-2-7B and Mistral-7B, which are slightly dated. The evaluations could be significantly improved by including newer LLMs such as Llama3/4, Gemma and QwQ models. This would also help establish the generality of the method proposed.


5) The specific Response level detection datasets used in the paper (Movies, Winobias, Math) are not commonly used for benchmarking hallucination detection. More standard and established datasets such as HaluEval, TruthfulQA, FAVABench or RagTruth could have been utilized instead. 


[1] Inference-Time Intervention: Eliciting Truthful Answers from a Language Model, Li et al., NeurIPS 2023

### Questions
Kindly refer to the questions mentioned in the weaknesses section above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces CHARM, a framework that recasts LLM hallucination detection as a graph learning problem. It models an LLM's internal computational traces, specifically attention scores and activations, as a single attributed graph where tokens serve as nodes and attention maps define the edges. This formulation elevates the task from relying on isolated heuristics to a principled problem effectively tackled by a Graph Neural Network (GNN).

### Strengths
1. The paper proposes a novel approach to hallucination detection by modeling LLM computational traces (attention and activations) as attributed graphs. This formulation provides a unified view of different internal signals and allows for the application of established graph neural network architectures to the problem.
2. The proposed method is supported by both theoretical analysis and empirical evaluation. The authors show that their framework is expressive enough to subsume existing attention-based heuristics. The experiments, conducted across different benchmarks and detection granularities, demonstrate performance gains over several baselines.
3. The paper is well-organized and clearly written.

### Weaknesses
1. The choice of LLM layer corresponding to the activation values is not clearly justified. For example, in Tables 1 and 2, some baseline methods report results from multiple layers, while CHARM only uses the layer 24. It would be better to include results from different layers to provide a more comprehensive analysis. Moreover, as mentioned at the end of Section 5.2, CHARM uses the layer 32 only for the Math dataset. Does this imply that performance is highly sensitive to layer selection?
2. The paper adopts a relatively simple propagation mechanism, whereas in previous studies on GNNs, more sophisticated approaches such as GCN or GAT are widely used. The authors should clarify the rationale for not employing these advanced graph propagation methods.
3. The paper lacks visualizations. It would be helpful to include concrete examples illustrating the constructed graph, such as visualizing nodes of varying sizes and edges with different thicknesses, or employing heatmaps and other effective visualization methods. Such visualizations could greatly enhance the interpretability of the proposed method.

### Questions
See the Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
