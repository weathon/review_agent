# Learning A Linear Delay Surrogate Model for Timing-Driven Chip Global Placement

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
Timing-driven global placement (GP) is a critical step in chip physical design, where the objective is to determine the physical locations of millions of cells to optimize signal delays and satisfy timing constraints. Existing GP algorithms commonly rely on gradient-based optimization, which requires the placement objective to be differentiable with respect to cell coordinates. However, timing evaluation---particularly the delay computation---is inherently complex and typically non-differentiable, making it difficult to integrate into gradient-based GP algorithms. To address this challenge, we propose **LiTPlace**, a **L**earn**i**ng-based **T**iming-driven global placement framework, which learns a differentiable surrogate model to predict signal delays for timing-aware gradient-based optimization. To the best of our knowledge, the application of machine learning (ML) in timing-driven GP remains underexplored in previous works. At the core of LiTPlace is a graph neural network (GNN) inspired by the signal propagation in chip circuits, which predicts signal delays based on the netlist graph structure and the placement geometry. To ensure compatibility with gradient-based optimization, we design the GNN architecture so that its output is approximately a linear function of a set of geometric distance statistics, enabling efficient and stable gradient computation with respect to cell coordinates. Experiments on $28$ chip designs from widely used benchmarks demonstrate that LiTPlace significantly improves timing quality, achieving an average improvement of $19.2\\%$ in TNS and $7.7\\%$ in WNS, which are two key metrics to quantify the chip timing quality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
LiTPlace is proposed in this paper to predict delay with GNN, and is employed to serve as a differentiable surrogate integrated directly into gradient-driven global placement optimization. Experiments on publicly available chip design benchmarks demonstrate average improvements in both TNS and WNS, with negligible additional computational cost.

### Strengths
1. Proposes a differentiable delay prediction model that is directly integrated into gradient-based global placement, addressing a largely unexplored application of machine learning in timing-driven placement.

2. The proposed method achieves strong results when compared with GP solvers such as DREAMPlace and Efficient-TDP. The experimental results are convincing, and the paper is clearly written with a well-organized structure.

### Weaknesses
1. Due to the specialized nature of chip timing, the work may be difficult to fully understand for readers without a relevant background. Although the authors devote considerable effort to explaining the background, I personally still find chip timing concepts challenging to grasp. This is not a shortcoming of the authors.

2. LiTPlace is used within DREAMPlace to compute delays, but for the global placement problem, it is unclear whether there are other classical delay computation methods that could be used for comparison. The set of baseline methods appears limited.

### Questions
1. Since each model is trained on a subset of circuits and evaluated on the full benchmark suite, it would be helpful if the results explicitly indicate which chips were used for training, which for validation, and which for testing, along with the performance improvements observed for each set. Although the appendix provides this information, I believe it would be helpful to discuss the performance results for training, validation, and test sets directly in the main text.

2. The paper mentions that 'As ICCAD2015 and ChiPBench have different technologies, we train a surrogate model for each benchmark suite.' This raises questions about the generalization capability of the algorithm: for an unseen chip, can the current model be applied directly? It would be helpful to report the cross-benchmark performance, i.e., how a model trained on ICCAD2015 performs on ChiPBench, and vice versa.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper present This paper presents LiTPlace, a GNN-based framework aimed at improving timing-driven global placement (GP) in chip design. The paper proposes a new method using a differentiable GNN model to optimize timing in placement stage. LiTPlace predicts edge delays in chip layouts and integrates it as estimated timing objective directly into the optimization process. By integrating this GNN model in placement framework, it shows significant improvements in total negative slack (TNS) and worst negative slack (WNS), outperforming existing methods in both performance and computational efficiency.

### Strengths
The paper propose a differentiable method to directly optimize timing in placement without external tools such as OpenTimer to provide precise timing delay with STA. 

LiTPlace has extensibility as it can be seamlessly integrated into modern placement framework such as DREAMPlace ,NTUPlace or ElfPlace.

### Weaknesses
There appears to be a lack of geometric-information modeling. Prior studies on circuit property prediction—e.g., LHNN [1] and CircuitGNN [2]—have shown that geometric information is crucial for accurate prediction. However, LiTPlace seems to omit geometric features. Is this because geometric structure is not important for timing prediction? Could the authors provide a detailed explanation?

Reference:

[1] Wang et al. LHNN: Lattice Hypergraph Neural Network for VLSI Congestion Prediction. DAC, 2022.

[2] Shu et al. Versatile Multi-stage Graph Neural Network for Circuit Representation. NeurIPS, 2022.

### Questions
Comparisons with prior work. Did the authors compare detailed runtime and performance between DTDGP [1] and LiTPlace? The paper reports ~50% improvement in WNS and ~70% in TNS, with ~1.8× speedup over a DREAMPlace 4.0–like method—please clarify the setup and metrics used. In addition, TransPlace [2] uses a GNN for transferable placement and can address congestion-aware and timing-aware objectives simultaneously. Could the authors provide a detailed comparison between TransPlace and LiTPlace? LiTPlace does not appear to be the first ML model targeting timing-driven placement.

Contributation of incorporating prediction into GP objective. Incorporating a neural network estimator to provide a differentiable objective and integrating it into the placement objective has been explored for congestion-driven settings [3,4,9]. What, specifically, distinguishes timing-driven placement from other objectives in the context of ML for placement? Can methods developed for congestion-driven placement be applied to timing-driven placement with modest changes (e.g., adding features or changing the prediction head)? If there are few or no such differences, why introduce a new framework instead of modestly adapting existing methods and applying them to timing-driven optimization?

Model design for timing prediction. Prior neural network-based timing-prediction work includes TGNN [5], GNNTrans [6], LSTP [7], and EdgeGAT [8]. How does the GNN used in LiTPlace differ from these models (e.g., architecture, input features, training targets, and generalization behavior)?

Reference:

[1] Guo et al. Differentiable-Timing-Driven Global Placement. DAC, 2022.

[2] Hou et al. TransPlace: Transferable Circuit Global Placement via Graph Neural Network. KDD, 2025.

[3] Zheng et al. Mitigating Distribution Shift for Congestion Optimization in Global Placement. DAC, 2023.

[4] Liu et al. Global Placement with Deep Learning-Enabled Explicit Routability Optimization. DATE, 2021.

[5] Guo et al. A Timing Engine Inspired Graph Neural Network Model for Pre-Routing Slack Prediction. DAC, 2022.

[6] Ye et al. Fast and Accurate Wire Timing Estimation Based on Graph Learning. DATE, 2023.

[7] Zheng et al. LSTP: A Logic Synthesis Timing Predictor. ASP-DAC, 2024.

[8] Ye et al. Graph-Learning-Driven Path-Based Timing Analysis Results Predictor from Graph-Based Timing Analysis. ASP-DAC, 2023.

[9] Hou et al. RoutePlacer: An End-to-End Routability-Aware Placer with Graph Neural Network. KDD, 2024.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes LiTPlace, a learning-based timing-driven global placement framework for VLSI design. LiTPlace introduces a propagation-based GNN that learns a differentiable linear delay surrogate to predict signal delays as an approximately linear function of geometric distances between connected cells. This surrogate enables direct gradient-based optimization of timing-aware placement. Experiments show average improvements of 19.2% in TNS and 7.7% in WNS over baseline frameworks.

### Strengths
1.	The idea of learning an approximate linear delay surrogate model for timing-driven global placement is novel.
2.	Demonstrates consistent improvement across ICCAD2015 and ChiPBench benchmarks and baseline GP approaches.
3.	The proposed framework is much more computing-efficient than DREAMPlace.

### Weaknesses
1. The proposed GNN model is a little bit confused. The methodology explains how the node and edge features are calculated at different typological level (different nodes and edges across the layout). But it does not provide a precise mathematical formulation of the message-passing and aggregation process for each GNN layer.
2. The propagation from one typological level to the next is a sequential linear transformation. I believe the entire model can be collapsed into one global linear propagation function. It does not exploit nonlinear relational reasoning or deep representation learning that typically justify a GNN. So, it is unclear whether the use of a GNN framework is necessary here.
3. As the delay prediction is the most important part to enable accurate timing-driven GP. I cannot find the details of the delay model training and evaluation, such as the training dataset size, training hyperparameter settings, and delay prediction accuracy.  
4. While LiTPlace is compared against gradient-based EDA frameworks (DREAMPlace, Efficient-TDP), it ignores recent ML frameworks that also aim to learn placement objectives or surrogates such as TransPlace[1]
[1] Hou, Yunbo, et al. "TransPlace: Transferable Circuit Global Placement via Graph Neural Network." arXiv preprint arXiv:2501.05667 (2025).

### Questions
See weakness above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces LiTPlace, a learning-based, differentiable timing-driven global placement framework for chip physical design.
The key idea is to train a propagation-based Graph Neural Network (GNN) that acts as a differentiable delay surrogate, enabling signal delays to be directly incorporated into gradient-based placement optimization.
By designing the GNN to have a linear propagation structure, the authors ensure that predicted delays are approximately linear functions of geometric distances, allowing analytical gradients and efficient optimization.
Evaluated on 28 benchmark designs (ICCAD2015 and ChiPBench), LiTPlace achieves 19.2% improvement in TNS and 7.7% improvement in WNS, with negligible wirelength increase.

### Strengths
1. The paper first introduces a differentiable timing-driven placement framework. Previous GP methods (e.g., DREAMPlace) optimized differentiable surrogates such as HPWL or density but could not directly optimize timing. LiTPlace is the first framework to embed timing objectives within a differentiable optimization loop through a learned delay model.

2. The proposed GNN simulates the signal propagation behavior of static timing analysis (STA). It follows topological levels and performs level-wise message passing, effectively capturing timing dependencies between upstream and downstream nodes.

3. Integrating LiTPlace into several baselines (DREAMPlace, DREAMPlace 4.0, Efficient-TDP) consistently improves timing metrics while adding minimal runtime overhead.

4. The paper introduces pooled distance statistics (min, max, mean) and aggregated successor features, allowing the model to reflect both geometric and electrical factors while remaining differentiable and computationally efficient.

### Weaknesses
1. While linearity improves efficiency, real-world timing behavior can exhibit strong nonlinearities (e.g., RC coupling, congestion effects). The model’s accuracy under such conditions remains uncertain, and no theoretical or empirical error bounds are provided.

2. The framework relies on a fixed number K of critical paths for optimization, but the selection and update frequency of K are treated as hyperparameters. This may affect optimization stability and lacks adaptivity across different designs.

3. Experiments only compare with open-source baselines and not with commercial-grade EDA tools (e.g., Synopsys ICC2, Cadence Innovus). Thus, the industrial applicability and scalability of LiTPlace remain unverified.

3. Training requires STA-generated delay labels for millions of samples. Despite reusing existing placements, this remains expensive in industrial design flows and may limit scalability.

4. The authors do not provide the code.

### Questions
1. In real designs, multiple signals can share the same net or pin, leading to coupling and congestion. Does LiTPlace handle these interactions, or is each signal path modeled independently?

2. What is the typical range of K, and how often is the critical path set updated? Have sensitivity analyses been conducted to evaluate its impact on convergence and final timing quality?

3. I am a new researcher working on AI4EDA placement, but I recently found that the ICCAD 2015 dataset link is no longer accessible. I would greatly appreciate it if you could share the dataset on an anonymous GitHub repository.

### Soundness
3

### Presentation
3

### Contribution
3
