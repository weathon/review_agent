# EdgeCape: Edge Weight Prediction For Category-Agnostic Pose Estimation

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 4

## Abstract
Category-Agnostic Pose Estimation (CAPE) localizes keypoints across diverse object categories with a single model, using one or few annotated support images. Recent works have shown that using a pose-graph (i.e., treating keypoints as nodes in a graph rather than isolated points) helps handle occlusions and break symmetry. However, these methods assume a given pose-graph with equal-weight edges, leading to suboptimal results. We introduce EdgeCape, a novel framework that overcomes these limitations by predicting the graph's edge weights in order to optimize localization. To further leverage structural (i.e., graph) priors, we propose integrating Markov Attention Bias, which modulates the self-attention interaction between nodes based on the number of hops between them. We show that this improves the model’s ability to capture global spatial dependencies. Evaluated on the MP-100 benchmark, which includes 100 categories and over 20K images, EdgeCape achieves state-of-the-art results in the 1-shot and 5-shot settings, significantly improving keypoint localization accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents EdgeCape, which addresses the issue of fixed edge weights in graph structures for Category-Agnostic Pose Estimation (CAPE). The authors introduce an edge weight prediction module combined with a "Markov Attention Bias" mechanism, claiming to achieve SOTA results on MP-100.

### Strengths
1. The problem identification is precise. This is the first work to systematically identify the limitations of fixed unweighted pose-graphs in CAPE and propose "category-agnostic edge weight prediction" as a novel and practical research direction.

2. The proposed Pose-Graph Predictor effectively fuses global image context with keypoint features, using residual graph optimization to avoid learning structures from scratch while balancing stability and expressiveness. The Markov Attention Bias cleverly injects graph structural information into Transformer attention mechanisms, enhancing spatial relationship modeling.

3. Not only achieving SOTA on the standard MP-100 benchmark, but also validating method effectiveness through multi-dimensional tests including noisy graph robustness (Figure 6), cross-supercategory generalization (Table 8), and occlusion experiments (Figure 9). The results particularly demonstrate that EdgeCape significantly outperforms GraphCape even when A_prior is suboptimal, and even surpasses graph-free methods like CapeFormer.

### Weaknesses
1. Unfair Comparison with CapeFormer Due to Asymmetric Use of Structural Priors
The paper claims superiority over CapeFormer, a point-based CAPE method that does not use any pose-graph prior. However, EdgeCape relies on a user-provided $A_{prior}$ as input to its graph refinement module. While the authors justify this by stating that $A_{prior}$ is part of the *support data* in graph-based CAPE, the comparison remains misleading unless the robustness of this prior is rigorously tested. Crucially, Figure 6 shows that EdgeCape significantly outperforms GraphCape under noisy $A_{prior}$, suggesting that the method's strength lies not just in using a graph, but in correcting a potentially bad one.Therefore, to make the comparison with CapeFormer meaningful, the authors should evaluate EdgeCape under a degraded or randomized $A_{prior}$ (e.g., fully connected, self-loops only, or random edges), a setting where the prior provides little to no useful signal. If EdgeCape still outperforms CapeFormer in such a scenario, the claim of superiority would be far more convincing. As it stands, the reported gains may largely stem from access to privileged structural information rather than algorithmic innovation.

2. The paper shows successful cases (Figures 4, 5) but provides no examples where EdgeCape fails while GraphCape/CapeFormer succeeds. This raises concerns about robustness. For instance, on highly symmetric or severely occluded objects, might the predicted weighted graph introduce erroneous strong connections that mislead localization?

3. The ablation study (Table 3) is misleading. It only compares the presence/absence of two components. Is the predicted weighted graph $Ã$ actually better than the original $A_{prior}$? The authors should fix the decoder and compare performance using $A_{prior}$ versus $Ã$ as inputs. Otherwise, performance gains might entirely stem from Markov Bias or training strategies rather than the graph itself.

4. The novelty of the proposed method is limited. The core idea of EdgeCape (predicting edge weights and introducing graph structure into attention mechanisms) lacks substantial breakthroughs at the methodological level. The weighted graph prediction uses standard cosine similarity, and the Markov Attention Bias is just a re-packaging of multi-hop structure encoding, which belongs to engineering improvements rather than principled innovations.

### Questions
The authors casually mention *only ~2ms additional delay*. However, the Pose-Graph Predictor requires an extra decoder pass and matrix power computations ($Ã^k$). With $K=4$ and large keypoint counts, which is far from negligible in practical deployment. The authors must provide detailed FLOPs or parameter count comparisons.

### Soundness
4

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
This paper presents a one-shot, category-agnostic method for estimating joint positions and their topological relationships. The approach begins with a user-provided binary skeleton and predicts real-valued edge weights tailored to the test instance, rather than assuming uniform importance across links. The authors introduce a Markov Attention Bias that interprets the weighted adjacency as a stochastic matrix to inject multi-hop connectivity cues directly into Transformer self-attention. Experiments on the MP-100 dataset show state-of-the-art performance compared with strong baselines.

### Strengths
(1) The paper presents a novel architecture that transitions from fixed, unweighted pose graphs to data-driven, weighted graphs.

(2) The proposed graph-based model incorporates a Markov Attention Bias to better capture complex spatial dependencies among keypoints.

(3) The writing is generally clear, and the figures are well designed.

### Weaknesses
(1) Marginal performance improvement over baselines on the MP-100 dataset, especially in the 5-shot setting: in Table 1 and 2, the performance gain over the strongest baseline is less than 1%.

(2) Missing references and comparisons (Minor). The paper lacks comparisons with related works, such as AutoLink: Self-supervised Learning of Human Skeletons and Object Outlines by Linking Keypoints.

### Questions
Please respond to the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The submission focused on the task of category-agnostic pose estimation with one or few annotated support images. Specifically, the authors propose a novel framework named EdgeCape to predict the graph's wdge weights for optimal localization. The authors also employ Markov Attention Bias to modulate the self-attention interaction in multi-hop. The experiments are conducted on a large-scale benchmark datasets, which indicate the effectiveness of the proposed method.

### Strengths
1. The task of category-agnostic pose estimation is interesting and fundmental for extending the category number of pose estimation.

2. The idea of using graph-based model is reasonable and makes sense.

3. The proposed EdgeCape and Markov Attention Bias are novel and effective to model the structral information of objects.

4. The performances of proposed method are shown on large-scale benchmark dataset, and outperform baselines by a large margin.

5. The experimental analyses are extensive and in-depth.

### Weaknesses
1. How to deal with the edge missing issue if there are some keypoints being occluded? E.g., the same objects may have different graphs if the occlusion cases are different. More specifically, the query image has 3 occluded keypoints, while the support image has another 5 occluded keypoints.

2. What's the complexity of proposed method? It seems to be about O(K^2). Is it cost-effective?

3. Closely related works are missing in the related works.&#x20;

   > 1. @inproceedings{chen2025weakshot, title={Weak-shot Keypoint Estimation via Keyness and Correspondence Transfer}, author={Chen, Junjie and Luo, Zeyu and Liu, Zezheng and Jiang, Wenhui and Li, Niu and Fang, Yuming}, booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}, year={2025} }
   >
   > 2. @inproceedings{kim2025capellm,   title={CapeLLM: Support-Free Category-Agnostic Pose Estimation with Multimodal Large Language Models},   author={Kim, Junho and Chung, Hyungjin and Kim, Byung-Hoon},   booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision}, year={2025} }

### Questions
See Weakness.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces EdgeCape, a spectral pre-conditioning strategy for GNNs that rescales edge weights before message passing to improve numerical stability and convergence. The idea is simple and general, and it can be plugged into standard GNN layers. Experiments on common benchmarks show small but consistent accuracy gains.

### Strengths
The motivation is reasonable and relevant. 

The approach is lightweight, easy to integrate, and empirically helps across several architectures. 

The writing is clear, and the empirical section is well-organized with ablations.

### Weaknesses
- The conceptual novelty is limited: many prior works already consider spectral normalization, Laplacian smoothing control, or adaptive edge reweighting. EdgeCape largely reformulates these ideas as “pre-conditioning” without delivering substantial theoretical or algorithmic innovation.

- The theory is weak, providing heuristic spectral bounds but no solid convergence or generalization proof.

- The experimental gains are modest (1–2%), limited to small and medium datasets, and lack comparison to stronger modern baselines in 2025.

- The scalability and robustness on large or dynamic graphs are untested.

### Questions
See Weaknesses.

### Soundness
4

### Presentation
2

### Contribution
2
