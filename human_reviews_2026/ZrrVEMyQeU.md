# WATS: Wavelet-Aware Temperature Scaling for Reliable Graph Neural Networks

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Graph Neural Networks (GNNs) have demonstrated strong predictive performance on relational data; however, their confidence estimates often misalign with actual predictive correctness, posing significant limitations for deployment in safety-critical settings. While existing graph-aware calibration methods seek to mitigate this limitation, they primarily depend on coarse one-hop statistics, such as neighbor-predicted confidence, or latent node embeddings, thereby neglecting the fine-grained structural heterogeneity inherent in graph topology. In this work, we propose Wavelet-Aware Temperature Scaling (WATS), a post-hoc calibration framework for node classification that assigns node-specific temperatures based on tunable heat-kernel graph wavelet features. Specifically, WATS harnesses the scalability and topology sensitivity of graph wavelets to refine confidence estimates, all without necessitating model retraining or access to neighboring logits or predictions. Extensive evaluations across nine benchmark datasets with varying graph structures and three GNN backbones demonstrate that WATS achieves the lowest Expected Calibration Error (ECE) among most of the compared methods, outperforming both classical and graph-specific baselines by up to 41.2\% in ECE and reducing calibration variance by 15.84\% on average compared with graph-specific methods. Moreover, WATS remains computationally efficient, scaling well across graphs of diverse sizes and densities. The implementation is available at \url{https://github.com/lxy1134/WATS.git}

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
WATS proposes a post-hoc, label-free calibration method for GNNs using graph wavelet features. A small MLP predicts a per-node temperature, guided by local spectral characteristics computed via heat-kernel wavelets. This allows finer calibration than global temperature scaling and mitigates overconfidence in deep GNNs.

### Strengths
1. Simple yet effective node-wise temperature scaling without retraining the base GNN.
2. The wavelet embedding captures multi-scale structural cues for confidence correction.
3. Extensive experiments across datasets and backbones with consistent ECE reduction.
4. Clear complexity and sensitivity analysis, including practical hyperparameter guidance.

### Weaknesses
1. Relies on the correlation between structure and calibration error, which may not hold for heterophilous graphs.
2. The wavelet scale parameter tuning could be expensive for a large graph

### Questions
1. How robust is WATS to heterophily or noisy topology?
2. Can wavelet-based calibration transfer to edge or graph classification tasks?
3. Does precomputing wavelets limit adaptability to dynamic graphs?
4. Could you briefly explain the intuition using wavelets to design a calibration algorithm in GNN?

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
3

### Summary
This paper introduces WATS, a post-hoc calibration framework for GNNs that learns node-wise learnable temperatures from graph wavelet-based structural features. By incorporating hierarchical structural signals, WATS improve miscalibration of GNNs compared to one-hop-based temperature scaling approaches.

### Strengths
- The application of graph wavelet transform in calibration domain is interesting.
- WATS substantially surpasses prior work in both small- and large-scale datasets.

### Weaknesses
- The discussion in Section 3.2 is a bit confusing. Even if model confidence exhibit high uncertainty in cases of disagreeing neighbors, a single node-level bias itself may not be problematic, since calibration error is inherently a population-level quantity, not measurable at a single-node level in practice. Could the authors provide additional explanation on this?
- According to Figure 1, deeper GCNs become more confident but less correct, which seems that increasing the receptive field rather worsens calibration. Furthermore, such phenomenon is likely attributable to over-smoothing rather than multi-scale connectivity or structural differences. This interpretation appears sometwhat misaligned with the design philosophy of WATS, which assumes that calibration errors arise when the calibration network fails to capture higher-order context. Could the authors clarify this?
- Minor) Found a typo in line 128.

### Questions
- Does similar trends of accuracy and confidence according to different numbers of layers in Figure 1 persist under heterophilous graphs?
- Could the authors show the wall-clock time analysis and memory consumption of the proposed method compared with baselines?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on the problem of calibrating model outputs for GNN architectures. Recent work shows that GNNs tend to be underconfident, which sets them apart from a lot of the NN literature and calls for bespoke strategies. A few strategies have already been suggested, that essentially try to predict the temperature parameter based on graph features. The central contribution of this paper is to replace this by a graph wavelet transform. The authors then show that their method performs well (where performance is measured using ECE) in several graphs, and outperforms competitors.

### Strengths
1) The paper's exposition is clear.
2) The results on experimental data look convincing.

### Weaknesses
As a disclaimer, I am not an expert on calibration (let alone on the calibration of GNNs), so here are a couple of points that I was hoping the authors could comment on.

1) Overall, the idea of using the graph structure to calibrate the uncertainty is fairly natural, but I wonder if the authors could try to phrase mathematically what they are trying to achieve.
For instance: I could imagine a setting where the GNN doesnt account perfectly for the signal on the graph, and as a consequence, the residuals are correlated.  For instance, let's simplify things by assuming that you're doing regression to predict $y_i = (\alpha X_i + \beta (\sum_{j \sim i} X_i/ d_i)  + \epsilon_i$  but that you're not using the graph structure, so $\hat{y}_i = \hat{\alpha} X_i$.

Consequently, using the wavelets that you are using is a much better idea than using predictors (like degree), as the wavelets will encode some deeper structural information (e.g where the errors are relative to one another) that will probably allow to calibrate more accurately the error, without having the overhead of expensive computations like GAT, etc.  
Another scenario could be that you've explained the graph structure away using your GNN, but the errors $\epsilon_i$ are not iid (network effects --- e.g. friends talk to one another and exert mutual influence on each other). In this case, maybe a way of formulating the problem is that each node has a neighborhood effect: 
$y_i = (\alpha X_i + \beta (\sum_{j \sim i} X_i/ d_i)  +  u_i + \epsilon_i$ 
where $u_i \sim N(0, \Sigma)$ where $\Sigma$ encodes dependencies between nodes. It is probably the case that the architecture you're suggesting is able to find dependencies that allow you to account for this random effect.
Is any of these close to what you were envisioning for the effect of the wavelets? Can you explain why the wavelet is a better idea?

2) The authors discuss the computational complexity of the method. It would have been insightful to compare the running time of each methods as well, on top of reporting the ECE.

3) It looks like the model doesnt crucially depend on K or s. What if, instead of the spectral wavelets, the authors used the Laplacian embedding of the graph --- that would prevent them from computing huge svd (just keep the top K) --- would we have similar results?

### Questions
(1) The characterization of conformal prediction as an "in-training approach [that]  uncertainty estimation within the model optimization process" seems off. CP acts as a wrapper, with no need to train any algorithm, so it doesnt interfer with the optimization process at all. It belongs to the post hoc methods.

(2) Could you explain this sentence: "GNNs tend to be systematically underconfident: their predicted confidence scores are consistently
lower than their true accuracy" --- what does "predicted confidence scores" mean, computed how?

(3) Tables 7-9 seem to show that the performance of the method is independent of the temperature parameter $s$ and $k$ -> any insight from the theory perspective?

4) For the ablation study, why not use the features? (Im not necessarily asking for more experiments, just curious to hear why it was not considered)? 


Notes: Line 68: ".Differs" --- the full stop there seems to be an error.
Line 269: "The hyper-parameter k sets the maximum receptive-field size" -> k should probably be K?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors aim to address the calibration issue in node classification. Existing methods only consider local topology when performing calibration, which limits their effectiveness. To overcome this issue, the author proposes using graph wavelets, which can capture information from more distant nodes, to determine the temperature for calibration. Experimental results demonstrate that the proposed method outperforms existing approaches on both GCN and GAT models.

### Strengths
- Proposes a method that can consider beyond one-hop neighborhood information
- Demonstrates the effectiveness of the proposed method across datasets with diverse characteristics
- The approach is lightweight, as it only requires running a 2-layer MLP on the validation set

### Weaknesses
- The method novelty of the proposed approach appears limited. Graph wavelets are already well-established, and in this work, they are applied to the calibration setting with only minor engineering and no specialized adaptation, which diminishes the originality of the method.
- The method involves training a MLP on the validation set, where the current split allocates 10% to validation and 20% to training. This is a relatively large validation portion, and in practical scenarios, such a large validation set may not be feasible. Consequently, the performance might degrade as the validation size decreases.
- Experiments are conducted only on relatively old architectures such as GCN and GAT, with no evaluation on recent GNN architectures, particularly those that include skip connections. Thus, it remains unclear whether the proposed method generalizes well to modern GNNs.
- Although the paper emphasizes the advantage of capturing beyond one-hop information, the evaluation is limited to 2-layer GCN and GAT, making it uncertain whether the method performs equally well on deeper architectures.

### Questions
Please see the Weakness.

### Soundness
3

### Presentation
3

### Contribution
2
