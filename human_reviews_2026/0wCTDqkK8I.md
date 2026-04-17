# Quantization with Purpose: Loss-Aware Bit Allocation for Gradient Compression

- Decision: Reject
- Scores: 4, 4, 8, 2

## Abstract
Gradient quantization is a critical technique for reducing communication overhead in large-scale distributed training. However, existing methods often employ fixed bit-width quantization or adaptive quantizers optimized with signal-level distortion metrics such as MSE, which poorly correlate with model performance. In this paper, we propose a novel layer-wise bit allocation framework for gradient quantization, formulated under a rate-distortion optimization (RDO) paradigm. Unlike prior approaches, our method introduces a loss-aware distortion metric that directly quantifies the impact of quantization on training loss, enabling task-aligned solution for bit allocation. A key insight of our work is the linear superposition property of cross-layer loss distortion, which we theoretically justify and empirically validate. This property allows us to decouple the original joint optimization problem and efficiently solve it via a Lagrangian optimization algorithm with linear complexity. Extensive experiments across vision and language tasks—using CNNs, ViTs, LSTMs, and Transformers—demonstrate the effectiveness of our approach. Moreover, our method integrates seamlessly with existing gradient compression techniques, yielding consistent performance gains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper propose a novel layer-wise bit allocation framework for gradient quantization, formulated under a rate-distortion optimization (RDO) paradigm. The method proposed introduces a loss-aware distortion metric that quantifies the impact of quantization on training. The paper provides some theory and empirical tests to validate the proposed approach.

### Strengths
- The paper introduces a loss-aware distortion (LAD) that estimates the impact of quantization on the training loss instead of relying on the gradient magnitude.

- The paper establishes that total loss distortion from jointly quantizing layers can be approximated by the sum of per-layer distortions, which allows decoupling and efficient optimization.

- A Lagrangian bit allocation procedure with linear complexity in the number of layers is proposed. 

- The framework is orthogonal to and compatible with other compression techniques.

- The experiments cover diverse architectures (CNNs, ViTs, LSTMs, Transformers) and tasks (vision and language). Demonstrating gains of the proposed approach

### Weaknesses
- Theoretical guarantees limited to optimality under decoupled RDO; convergence impacts are not formalized. Proof of convergence over the iterations and the convergence bounds are not reported. The algorithm optimizes bit allocation per step, but the effect on optimization dynamics is not theoretically characterized. Could you show that your approach maintains convergence guarantees similar to SotA compression techniques?

- Section 2.2.1, W_orig should not be the same for both. For your approach at iteration t, you should start from \tild{W_t} constructed using your compression in the previous iterations and not W_t 

- The paper introduces LAD but does not detail how it is estimated efficiently during training. Provide a precise estimator of the loss and how that affects your analysis.

- The linear superposition property requires a stronger theoretical scope and more testing. The paper mentions theoretical justification and empirical validation, but the conditions under which this holds (e.g., bound on the step size, smoothness of the loss, independence of layer quantization errors, small perturbation regime…) are not clearly mentioned

- Practical training often exhibits nonlinearity and interaction between layers.

### Questions
- Theoretical guarantees limited to optimality under decoupled RDO; convergence impacts are not formalized. Proof of convergence over the iterations and the convergence bounds are not reported. The algorithm optimizes bit allocation per step, but the effect on optimization dynamics is not theoretically characterized. Could you show that your approach maintains convergence guarantees similar to SotA compression techniques?

- Section 2.2.1, W_orig should not be the same for both. For your approach at iteration t, you should start from \tild{W_t} constructed using your compression in the previous iterations and not W_t 

- The paper introduces LAD but does not detail how it is estimated efficiently during training. Provide a precise estimator of the loss and how that affects your analysis.

- The linear superposition property requires a stronger theoretical scope and more testing. The paper mentions theoretical justification and empirical validation, but the conditions under which this holds (e.g., bound on the step size, smoothness of the loss, independence of layer quantization errors, small perturbation regime…) are not clearly mentioned

- Practical training often exhibits nonlinearity and interaction between layers.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents a rate–distortion optimization (RDO)–based framework for gradient quantization that employs a loss-aware distortion metric to capture the impact of quantization on training loss, enabling adaptive layer-wise bit allocation. By leveraging the linear superposition property of cross-layer loss distortion and solving via Lagrangian optimization, the method efficiently improves communication efficiency and overall model performance in distributed training.

### Strengths
1. Proposes a principled rate–distortion optimization (RDO) framework that makes gradient quantization more interpretable and theoretically grounded.
2. Introduces a loss-aware distortion metric enabling task-aligned and adaptive bit allocation.
3. Leverages the linear superposition property and Lagrangian optimization to reduce computational complexity and improve communication efficiency and model performance.

### Weaknesses
1. The paper lacks a dedicated Related Work section; the discussion of prior studies is scattered in the introduction without systematic comparison.
2. The paper contains few experimental figures, and the presentation of results is not sufficiently intuitive.
3. It is recommended to reorganize the paper’s structure and place the figures and tables in their corresponding sections.

### Questions
1. The accuracy improvement in Tables 1 and 2 is relatively small (around 0.3%–1%), is it sufficient to offset the additional computational cost?
2. For tasks involving stochastic perturbations (such as reinforcement learning or adversarial training), can the LAD metric still maintain stability?
3. The authors claim that the loss-aware distortion (LAD) metric aligns better with task objectives than MSE, but is there any theoretical evidence showing that LAD has a stronger correlation with final task performance?
4. The first-order Taylor expansion neglects the Hessian term, but is this assumption still reasonable in regions with a large learning rate or a steep loss surface?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the communication bottleneck in distributed deep learning training, a key challenge for large-scale models. It identifies limitations of existing gradient quantization methods: uniform bit-width allocation fails to account for layer-wise sensitivity, and signal-level metrics poorly correlate with model performance.
To resolve these issues, the authors propose a rate-distortion optimization-based layer-wise bit allocation framework. Key innovations include: 1) a Loss-Aware Distortion metric that quantifies quantization’s impact on training loss directly; 2) leveraging the linear superposition of loss distortion to decompose the intractable joint allocation problem; and 3) integrating Lagrangian optimization and a gradient-similarity-based dynamic trigger for efficiency.
Experiments on vision and language tasks show the framework outperforms static/heuristic baselines and enhances existing quantizers. Results support its ability to balance communication efficiency and model performance.

### Strengths
1) The paper effectively targets the critical limitations of fixed bit-width and signal-level metrics by proposing a loss-aware layer-wise bit allocation framework under the rate-distortion optimization paradigm, addressing the core gap between communication efficiency and model performance.
  
2) The paper presents a key insight that decouples the intractable joint bit allocation problem into independent per-layer subproblems solvable with linear complexity, making the framework applicable. 

3) The paper designs an efficient Lagrangian optimization algorithm for optimal bit assignment to balance distortion and communication budget and a lightweight dynamic reallocation trigger to monitor gradient norm similarity, which reduces unnecessary computational overhead by only updating bit allocations when gradient distributions shift significantly.

### Weaknesses
1) The dynamic reallocation trigger depends on fixed thresholds  without adaptive adjustment mechanisms, increasing deployment complexity
2) The Loss-Aware Distortion metric's resource cost for large-scale models remains unmeasured, risking computational bottlenecks
.

### Questions
NONE

### Soundness
3

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
4

### Summary
The authors propose a task-aware quantization method, unlike prior work that uses MSE to measure the difference between actual and quantized gradients. The authors formulate this problem as a rate-distortion optimization (RDO), then show that this joint optimization can be solved as a sum of independent problems to achieve layer-wise bit allocation. To reduce cost, this optimization is performed when a regime shift in the gradients is detected. The results show that this approach is effective when compared to uniform or greedy layer-wise bit allocations.

### Strengths
* The paper proposes a task-aware quantization that preserves model quality better than other baselines.
* The paper formulates the bit allocation problem as an RDO, and shows mathematically how this intractable joint optimization can be solved as a series of independent subproblems.
* The papers show experiments over a representative set of tasks.

### Weaknesses
* The paper motivates itself by the scale of foundation models (billions of parameters), but the proposed method targets weight and gradient compression, which primarily reduces communication and does not address the core memory/compute challenges of training very large models. Its applicability also appears limited to data parallelism.
* The paper emphasizes model quality, but it is not clear whether the method is cost-effective. Do the end-to-end speedups from compression outweigh the method’s computational overhead? Is it more efficient than a strong greedy compression baseline that achieves similar quality?
* The experiments use small models (ResNet18, 4 layers Transformer , 2 layers LSTM). To better match the paper’s motivation, the authors should evaluate larger models such as ViT-Large and BERT-Large.
* The baselines used in the experiments are not sufficient. The authors claim “Given that the optimization objectives and constraint conditions in prior studies (Markov et al., 2024; Yan et al., 2022) on bit allocation for gradient compression differ from ours” still other SoTA for gradient compression should be evaluated, and the difference in assumptions and constraints should be highlighted.
* The related work coverage is lacking. Xin et al. "Kimad: Adaptive Gradient Compression with Bandwidth Awareness" should be contrasted to.

### Questions
* What is the computational overhead of computing the loss-aware distortion metric and running the search across all layers?
* In your experiments, how frequently did regime shifts occur that triggered bit reallocation?
* What is the runtime cost of the trigger mechanism itself?
* What are the actual bit allocations obtain by your method and how much do they they differ from Uniform and Greedy?
* The perplexity in Table 2 using gradient compression is large. It seems that the model quality would be greatly affected by gradient compression. Would your method be able to recover ppl around 82? If so, in what configuration?

### Soundness
2

### Presentation
3

### Contribution
2
