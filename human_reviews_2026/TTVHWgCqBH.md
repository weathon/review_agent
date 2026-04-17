# Scalable Circuit Learning for Interpreting Large Language Models

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
A prominent research direction within mechanistic interpretability involves learning sparse circuits to model causal relationships between LLM components, thereby providing insights into model behavior. However, due to the polysemantic nature of LLM components, learned circuits are often difficult to interpret. While sparse autoencoder (SAE) features enhance interpretability, their high dimensionality presents a significant challenge for existing circuit learning methods to scale. To address these limitations, we propose a scalable circuit learning approach, CircuitLasso, that leverages sparse linear regression. Our method can efficiently uncover relationships among SAE features, showing how human-interpretable semantic features propagate through the model and influence its predictions. We empirically evaluate our method against state-of-the-art baselines on benchmark circuit learning tasks, demonstrating substantial improvements in efficiency while accurately capturing circuits involving LLM components. Given its efficiency, we then apply our method to SAE (high dimensional) features and obtain human-interpretable circuits for a grammatical classification task that has not been studied before in mechanistic interpretation. Finally, we validate the utility of our learned circuits by leveraging their insights to improve downstream performance in domain generalization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes CircuitLasso, a scalable circuit-learning method that reframes circuit discovery in LLMs as sparse linear regression over internal components (i.e., SAE features). By assuming a causal ordering aligned with the forward pass and linear relations, the method removes explicit acyclicity constraints and yields a block-lower-triangular Lasso objective. What's more, the authors provide a complexity analysis contrasting CircuitLasso with intervention-based methods, show accuracy/efficiency tradeoffs, then scale to SAE features to analyze GPT-2 small on CoLA, revealing interpretable multi-layer feature flow. Finally, they leverage learned circuits to ablate SAE features and improve domain generalization on Bias-in-Bios across Pythia-70M, Gemma-2-2B, and Gemma-2-9B, with favorable accuracy/runtime w.r.t. baselines.

### Strengths
+ It's interesting to see CircuitLasso to cast circuit discovery as a block-structured Lasso aligned with compute order, which avoids explicit acyclicity constraints and costly intervention loops.

+ From the experiments, it seems that CircuitLasso can reveal clear concepts (e.g., “-self”, punctuation, hunger/thirst) and their persistence/merging/dropping across layers, moving beyond polysemantic neuron noise (GPT-2/CoLA).

### Weaknesses
- My primary concern is the paper’s reliance on causal discovery without explicitly stating or validating its underlying assumptions. By using a Lasso-based approach while effectively overlooking key conditions, e.g., unconfoundedness and correct causal ordering, the method risks learning spurious or anti-causal circuits rather than the true causal structure. Without targeted interventional tests or invariance checks across environments, such errors may go undetected.

- In addition, a purely linear formulation may miss important nonlinear interactions among features. Therefore, the absence of residual diagnostics or small-scale nonlinear ablation study weakens the method.

### Questions
Please refer to my summary of weaknesses.

### Soundness
2

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
3

### Summary
The paper considers the problem of building circuits for interpreting the behavior of Large Language Model. The goal here is to construct a graph denoting causal dependencies between the various internal components of the model which faithfully represent its behavior. Unfortunately, while prior work devised efficient methods to do this for the neurons in a generic LLM, these circuits are hard to interpret since the neurons in a network are often polysemantic and may represent multiple distinct concepts. In order to disentangle these concepts, other interpretability research focuses on interpreting the behavior of individual neurons as sparse combinations of (hopefully) monosemantic concepts (sparse autoencoder or SAE features). However, these techniques are not compatible with prior techniques for circuit construction. The main contribution of the paper is a method to build circuits directly in the feature space of these monosemantic concepts rather than on the polysemantic neural representations.

The main technique in the paper is quite simple. The core idea of the approach is to solve a sparse linear regression problem between the SAE features of successive layers of the model. For each layer of the model, the approach trains a sparse linear regression model which predicts the SAE features at that layer as a linear function of those in the previous layer. By tracing backwards from the output of the model through the features with the largest influence in the previous layer (measured by the coefficients of the sparse regression vector and optionally, with the particular input), the method produces a circuit over the SAE features. The authors also experimentally validate the method in a range of setting, obtaining considerable efficiency gains. 

Overall, the approach proposed is natural and efficient. My main concern with the paper is the lack of positional awareness in the learned circuits. While averaging over the representations of individual neurons may enable scalability, this potentially comes at the cost of understanding the causal structure between different parts of the input. This makes a comparison to other interpretability methods, constructing position aware dependency graphs on neuron activations, challenging. I request the authors to expand more on the tradeoffs incurred by this choice. Despite this, I believe this is a good paper and would make a nice addition to the conference.

### Strengths
See main review

### Weaknesses
See main review

### Questions
See main review

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces CircuitLasso, a scalable method for circuit learning in large language models (LLMs), which leverages sparse linear regression (Lasso) to uncover and interpret relationships among sparse autoencoder (SAE) features. The method addresses challenges of interpretability and scalability posed by high-dimensional, monosemantic SAE features, and avoids computationally expensive interventions used in prior causal circuit discovery methods. CircuitLasso is empirically evaluated on INTERPBENCH and real-world LLMs, demonstrating efficiency and interpretability advantages, and yielding semantically meaningful circuits as well as measurable gains in downstream tasks such as domain generalization.

### Strengths
- The paper is generally well-structured, with a clear exposition of the evaluation criteria, algorithmic steps, and experimental setups. It is clear and novel that CircuitLasso reframes circuit discovery as a problem of sparse structure estimation, eliminating reliance on computationally expensive interventions. This makes large-scale analysis of SAE features feasible and efficient, especially as model size increases.

- The method operates directly on SAE latent features, yielding human-interpretable circuits. The CoLA case study presents intuitive, reproducible feature interpretations accompanied by a multi-prompt or single-prompt methodology that readers can replicate.

- The framework is evaluated on both synthetic benchmarks with ground-truth circuits and real-world LLM tasks, confirming its soundness, reliability, and general applicability.

- The learned circuits are shown to be functionally useful, enabling downstream applications like bias mitigation and domain generalization. Removing spurious SAE features leads to improved robustness, demonstrating tangible benefits beyond interpretability.

- The work builds on publicly available SAEs such as GPT-2 and Gemma-Scope, ensuring reproducibility, transparency, and ease of adoption for the broader interpretability community.

### Weaknesses
- CircuitLasso relies on a linear structural equation assumption (Section 3.1) and constrains edge directions to follow the model’s forward computation order. While these simplifications enable scalability and convex optimization, they limit causal faithfulness in highly nonlinear or feedback dominated regimes. The method may produce anti-causal or oversimplified paths, for example, hunger→eat vs. eat→hunger in Figure 2.

- Because the framework operates purely on observational activations, it cannot fully distinguish causal relationships from correlated or confounded dependencies. This is particularly concerning for LLMs trained on large, biased, or polysemous corpora, where co-occurrence artifacts are common. Without interventional or perturbation-based validation, the discovered circuits may reflect statistical associations rather than true causal mechanisms.

- Experiments use the five main models of the INTERPBENCH. However, the dataset has 86 different models containing 85 synthetic ones and a non-synthetic one. While the choice is justified, a stronger empirical case would include a subset diversity study and at least one non-synthetic diagnostic beyond CoLA for mechanistic tasks to establish faithfulness.

- The identification of important SAE features via coefficient magnitude or Hadamard-weighted edges is heuristic and may be unstable under feature collinearity or overlapping semantics.

- Furthermore, CircuitLasso inherits limitations of the underlying SAEs. If SAE features are not genuinely monosemantic or exhibit high reconstruction error, the resulting circuits could be misleading. The paper does not quantify how SAE fidelity affects circuit accuracy or interpretability.

### Questions
- The paper reports that direct SAE-feature ablation outperforms SHIFT’s decode-then-ablate route on larger models. Could the authors elaborate on the underlying reason, whether it could be reduced reconstruction noise, stronger feature disentanglement, or layer-specific sparsity?

- Could CircuitLasso be extended to capture nonlinear dependencies, for instance through group Lasso, kernelized regressions, or sparse neural DAGs? Where does the runtime advantage come from? Is it primarily a result of sparsity, or could algorithmic choices also apply to nonlinear variants?

- For anti-causal or ambiguous edges, could targeted activation patching or minimal interventional tests help refine causal direction without losing scalability?

- How robust are the learned circuits to the choice of SAE dictionary? Does retraining or altering the SAE dictionary significantly change the inferred circuit topology or key dependencies?

- In addition, an important question concerns the practical scalability of the proposed framework given the high computational cost of training sparse autoencoders. CircuitLasso requires substantial data, compute, and hyperparameter tuning, which may limit the framework’s accessibility and generalizability to other models or domains. Could the authors comment on how the overall pipeline might be made more practical in multiple LLMs?

- Experiments use a subset of 5 cases of the INTERPBENCH models for the main evaluation. However, the paper does not publicly specify the criteria used to select those specific cases from the full set of 86 (or the 16 main ones mentioned). Could the authors clarify how the subset was chosen and whether the selection introduces any bias in comparing circuit-discovery methods?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce Circuit Lasso, a circuit discovery method based on Structured Equation Models. They evaluate circuits on Interpbench and SHIFT.

### Strengths
The authors provide a highly efficient approximation to circuit discovery, as measured by runtime.

### Weaknesses
1. The CircuitLasso method operates on X, a stacked activation vector, there is no weight-based analysis or backpropagation involved. The matrix A introduced in equation 1 therefore only reveals correlational information, and cannot make claims about causal dependencies. 

2. The Circuit Discovery algorithms introduced by Marks et al. (SFC) and Conmy et al. (ACDC) are the main baselines for this work. This paper uses the SHIFT, a practical downstream application from SFC, for evaluation. This metric however only scores the identified set of relevant nodes and is invariant to the discovered set of edge weights. 
   The Faithfulness and Completeness metric, used by both Marks et al. and Conmy et al. are a complementary signal to SHIFT, and would be a useful addition to this work. A gap in the existing literature is that SFC ablates sets of neurons, not edges. This contains the implicit assumption that a node in the circuit is connected to all subsequent nodes.
   A strong improvement over existing literature would be to evaluate faithfulness and completeness based on edge ablation. I would improve my score if this evaluation was run, and SFC results would be compared to Circuit Lasso results in line with SFC Figure 2.

### Questions
How large are the acquired circuits? Does the method involve setting a threshold to choose the top K attributed nodes / edges?

### Soundness
2

### Presentation
3

### Contribution
3
