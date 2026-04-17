# Resolving Oversmoothing with Opinion Dissensus

- Decision: Reject
- Scores: 8, 4, 2

## Abstract
While graph neural networks (GNNs) have allowed researchers to successfully apply neural networks to non-Euclidean domains, deep GNNs often exhibit lower predictive performance than their shallow counterparts. This phenomena has been attributed in part to oversmoothing, the tendency of node representations to become increasingly similar with network depth. In this paper we introduce an analogy between oversmoothing in GNNs and consensus (i.e., perfect agreement) in multi-agent systems literature. We show that the message passing algorithms of several GNN models are equivalent to linear opinion dynamics in multi-agent systems, which have been shown to converge to consensus for all inputs regardless of the initial state. This new perspective on oversmoothing motivates the use of nonlinear opinion dynamics as an inductive bias in GNN models. In addition to being more general than the linear opinion dynamics model, nonlinear opinion dynamics models can be designed to converge to dissensus for general inputs. Through extensive experiments we show that our Behavior-inspired message passing (BIMP) neural network resists oversmoothing beyond 100 time steps and consistently outperforms existing continuous time GNNs even when amended with oversmoothing mitigation techniques. We also show several desirable properties including well behaved gradients and adaptability to homophilic and heterophilic datasets.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes BIMP (Behavior-Inspired Message Passing), a new class of graph neural networks inspired by nonlinear opinion dynamics. The authors establish a formal analogy between GNN message passing and multi-agent opinion consensus, theoretically showing that oversmoothing in GNNs corresponds to opinion consensus in social dynamics. Based on this insight, they design a nonlinear dynamical system that maintains diversity among node representations. The proposed BIMP model integrates communication and option graphs, introduces nonlinear saturation and bifurcation-controlled parameters, and achieves strong robustness against oversmoothing while remaining computationally efficient.

### Strengths
1. Theoretical originality:
    
    The paper establishes a clear and elegant analogy between oversmoothing in GNNs and opinion consensus in nonlinear dynamics, offering a novel theoretical lens for understanding message passing.
    
2. Well-motivated nonlinear model:
    
    The introduction of nonlinear saturation functions, bifurcation-controlled attention, and external inputs provides a principled way to prevent consensus, moving beyond heuristic anti-oversmoothing tricks.
    
3. Comprehensive analysis and proofs:
    
    Theoretical lemmas and theorems rigorously support the model’s stability, convergence, and dissensus properties.
    
4. Strong empirical validation:
    
    BIMP demonstrates stable Dirichlet energy over 1000 timesteps and consistent superiority across both homophilic and heterophilic datasets, outperforming recent continuous-depth GNNs such as GRAND, GraphCON-Tran, and KuramotoGNN.

### Weaknesses
1. Ablation clarity

   The authors should include an additional Dirichlet energy ablation experiment (on activation functions and inductive bias) to verify the impact of different functions and modules on oversmoothing.

Minor issues
* Please provide citations for the baseline models in the main text, and ideally include brief descriptions of these baselines.

### Questions
1. To what extent is the robustness to oversmoothing due to the nonlinear opinion dynamics itself, versus architectural choices (e.g., residual terms, attention normalization)?
    
2. Does the bifurcation-controlled parameter ( $u = \frac{d}{\alpha + 3}$ ) require careful tuning for different datasets, or is it generally robust across tasks?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the problem of oversmoothing in GNNs. It makes two significant contributions. First, it proposes a general framework to determine if a GNN model suffers from oversmoothing. In particular, this framework proves oversmoothing for linear discrete-depth GNNs and continuous-depth Laplacian GNNs. This is made possible by mapping a continuous depth GNN to opinion dynamics. Based on this result, the second contribution is a GNN model that does not suffer from oversmoothing. Additionally, the paper tests the performance of the proposed GNN on various datasets, showing slightly better accuracy than state-of-the-art models using similar computational power.

### Strengths
The paper presents an innovative connection between GNNs and opinion dynamics. The proposed GNN model is not susceptible to oversmoothing and outperforms state-of-the-art models.

### Weaknesses
Some relevant related works may be missing from the paper. For example, there is no reference to how this work relates to previous work on oversmoothing, such as *A Note on Over-Smoothing for Graph Neural Networks by Chen Cai and Yusu Wang* or *Residual Connections and Normalization Can Provably Prevent Oversmoothing in GNNs by Michael Scholkemper, Xinyi Wu, Ali Jadbabaie, and Michael T. Schaub*, which appeared in ICLR 2025. Additionally, there is insufficient reference to discrete-time opinion dynamics, of which there are many nonlinear examples that may adapt better to describe discrete-depth GNNs. See, e.g., *Consensus Dynamics: An Overview by Luca Becchetti, Andrea Clementi, and Emanuele Natale (SIGACT News 2020)*. The relation to these kind works is the subject of Question 3 below. 
Finally, it is natural to wonder how the external input B in your paper relate to skip connections (see Question 1).

### Questions
1. How does external input B in your paper relate to skip connections, which are known in the literature to prevent oversmoothing? 
2. Could the fact that GNNs based on Laplacian dynamics suffer from oversmoothing, even with external input, be an effect of the Laplacian dynamic rather than the linearity of the dynamic? 
3. How does your work compare to the references mentioned in the Weaknesses section?

### Soundness
3

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
The paper studies the oversmoothing problem in graph neural networks (GNNs) by drawing an analogy to consensus formation in multi-agent opinion dynamics. The authors show that several classes of existing GNNs can be interpreted as equivalent to linear opinion dynamics, which necessarily converge to consensus, and therefore inevitably oversmooth. Based on nonlinear opinion dynamics models, the paper proposes a continuous-depth GNN architecture named BIMP (Behavior-Inspired Message Passing), which is theoretically guaranteed to avoid oversmoothing under certain conditions. Experiments are conducted on common node-classification benchmarks to demonstrate resistance to feature collapse at large depths and competitive accuracy against baselines.

### Strengths
1. The analogy between oversmoothing and consensus dynamics is presented clearly and supported by formal statements, which may help unify prior perspectives on oversmoothing.
2. The paper states assumptions and propositions explicitly, with proofs delegated to the appendix, making the theoretical section readable.
3. The empirical plots demonstrate that the proposed model maintains stable Dirichlet energy and accuracy even when simulated for up to 10^3 layers, which supports the main claim of the proposed framework on the oversmoothing side.
4. The paper is easy to follow, and figures illustrating the bifurcation behavior are helpful in conveying the underlying intuition.

### Weaknesses
1. Necessity of focusing on oversmoothing is not fully justified.

The paper treats oversmoothing as a central obstacle in GNN design, but it is unclear why controlling depth-induced feature collapse is still an impactful problem in practice. Many modern architectures (e.g., graph transformers, attention-based models with residual paths, or shallow-but-expressive methods) already achieve strong performance with 2–4 layers, often outperforming deeper continuous-depth models regardless of oversmoothing behavior. If a 3-layer model can surpass a 128-step ODE-based model, the motivation for pursuing “arbitrarily deep but stable” GNNs needs stronger justification. The paper would benefit from either (a) a concrete real-world setting where depth improves performance and oversmoothing becomes the bottleneck, or (b) evidence that existing expressive architectures still fail due to oversmoothing rather than other factors (e.g., over-squashing, limited expressivity, memory constraints).

2. Benchmark coverage and presentation choices raise questions.

While the standard homophilic datasets are reported in the main text, several heterophilic results appear only in the appendix, despite being directly relevant to the claim that the model adapts to both regimes via tunable filtering. In addition, the benchmark suite does not explore domains where deep message passing is naturally required (e.g., long-range molecular graphs, multi-hop reasoning, or large-scale relational systems). Since the paper does not claim improvements in runtime, scalability, or memory, the restriction to small- to medium-scale citation graphs feels limiting.

3. Depth stability is demonstrated, but the practical benefit is unclear.

The paper shows that BIMP avoids Dirichlet-energy collapse and remains trainable at large depths, but it is not shown why this matters for downstream tasks. In the current benchmarks, deeper baselines typically do not fail catastrophically—they just plateau or degrade moderately. An illustrative failure case, where a strong baseline collapses due to oversmoothing and BIMP succeeds, would make the motivation more concrete.

4. Novelty is mainly conceptual rather than architectural.

Although the theoretical analogy is interesting, the empirical model resembles prior continuous-depth GNNs with a particular nonlinear choice and learned adjacency. The extent to which performance gains arise from the opinion-dynamics design rather than standard architectural components is not fully disentangled.

5. Scope limited to oversmoothing, not broader depth-related limitations.

Other known depth challenges—such as over-squashing, expressivity limits, and scaling to large graphs—are not addressed. Since oversmoothing is only one of several depth-related bottlenecks, the contribution feels narrow unless its relevance can be better contextualized.

### Questions
1. Can the authors provide a concrete example where oversmoothing is the dominant failure mode in modern GNNs?
2. How does BIMP perform on domains where depth is required (e.g., long-range molecular interaction graphs, knowledge graphs, large-scale recommender systems)?
3. Is the nonlinear inductive bias still necessary when skip-connections, normalization, or positional encodings are added? An ablation isolating the source of benefit would be useful.
4. How does BIMP compare against non-ODE architectures that already mitigate oversmoothing implicitly?
4. Since nonlinear continuous-depth models are not the only way to address the scenarios tested in the paper, how are the runtime / memory comparisons against strong non-ODE baselines (e.g., GCN with residuals, GraphGPS, GATv2, or GNN-SSM), especially on larger graphs.

### Soundness
2

### Presentation
3

### Contribution
2
