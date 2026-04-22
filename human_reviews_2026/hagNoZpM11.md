# TEFormer: A Topology-Enhanced Transformer for Architecture Performance Prediction

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Evaluating architecture performance is a crucial step in neural architecture search (NAS) but remains computationally expensive. Performance predictors offer an efficient alternative by learning from a limited set of architecture-performance pairs. However, previous predictors tend to oversimplify the topological structure of neural architectures using adjacency matrices, node depths, or computation flow, which fail to fully capture topological features of architectures, leading to poor generalization. To address this limitation, we propose TEFormer, a Topology-Enhanced Transformer that integrates both local and global topological information beneficial to performance prediction. Specifically, we employ a topology-aware flow encoding module that incorporates local topological characteristics via a learnable structural encoding and a flow-based encoder. At the global level, we design a hierarchical attention mechanism to jointly model intra-flow and inter-flow interactions within the architecture. To further improve generalization, we propose an architecture augmentation strategy that synthesizes additional samples by interpolating similar architectures in the latent space. Extensive experiments on computer vision, graph learning, and automatic speech recognition tasks demonstrate that TEFormer consistently outperforms state-of-the-art predictors and exhibits superb performance across diverse search spaces.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes TEFormer, or Topology-Enhanced Transformer for Neural Architecture Search (NAS) performance prediction. TEFormer continues a line of work on flow-based predictors [1-3] for NAS. TEFormer augments these approaches by encoding the local and global features through a specialized attention mechanism and positional encoding. TEFormer is evaluated on several NAS-Benchmarks for computer vision and graph prediction.

### Strengths
- The paper makes advances in NAS performance prediction.
- The experimental setup and execution is solid. 
- The method is clear and easy to understand.
- Ablation studies are provided.
- The evaluation is not simply limited to NAS-Bench-{101, 201, 301}, but other benchmarks.

### Weaknesses
- The method is incredibly incremental. Table 6 proves this with how little, arguably not statistically significant change takes place during the ablation study. 
- TEFormer primarily stems from an existing, but very narrow line of work on Flow-based predictors [1-3] and the contribution is just slight increases in Kendall's Tau on DARTS which is not noteworthy when we've already been able to leap frog over the best DARTS architectures on ImageNet for several years [4]. 
- The statement in lines 224-225 "Considering that both forward and backward passes are essential for accurately modeling neural architectures (...), we encode the bidirectional computational flow", should be removed or heavily revised, as it is essentially ignoring other advances in performance prediction [5, 6, 7] that are tangential to [1-3]; it essentially reads as if the method of [1-3] is the only correct way to achieve performance prediction, which is not true.

### Questions
Can the local and global level features be used to extract information about the structure of good/bad architectures as [9, 10] do?

References:

[1] https://arxiv.org/abs/2004.01899

[2] https://proceedings.neurips.cc/paper_files/paper/2022/file/d0ac28b79816b51124fcc804b2496a36-Paper-Conference.pdf

[3] https://arxiv.org/abs/2403.12821

[4] https://arxiv.org/abs/1812.00332

[5] https://arxiv.org/abs/2506.04001

[6] https://arxiv.org/abs/2210.03230

[7] https://proceedings.neurips.cc/paper_files/paper/2022/hash/572aaddf9ff774f7c1cf3d0c81c7185b-Abstract-Conference.html

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes TEFormer, a novel Transformer-based predictor for neural architecture performance estimation. The core innovations include a topology-aware flow encoding module that incorporates bidirectional computation flows with learnable structural encodings based on random walks, a hierarchical attention mechanism to model both intra-flow and inter-flow dependencies, and an interpolation-based architecture augmentation strategy to combat data scarcity. The method is evaluated on multiple NAS benchmarks across computer vision, graph learning, and speech tasks, demonstrating highly competitive ranking performance and search results.

### Strengths
1. The idea of explicitly modeling bidirectional computation flows (forward and backward) is well-motivated and differentiates the work from many existing predictors that rely on static graph encodings. The integration of learnable structural encodings from random walks is a principled approach to capture rich topological information.
2. The authors evaluate on many standard NAS benchmarks and provide ablation studies and sensitivity analyses, showing consistent gains and some robustness to hyperparameters.

### Weaknesses
1. Some design choices are not sufficiently explained, especially from Eq. (2) to Eq. (4). The rationale behind these formulations is not intuitive.
2. In Section 4.2, the underlying motivation for computing attention only between nodes connected by a directed path or within the same topological group is not well-justified. The authors' explanations read more like descriptions of the rules' effects rather than a justification of the underlying design principles.
3. The interpolation-based augmentation strategy, presented as a major contribution, appears potentially risky. A more comprehensive evaluation is required to substantiate its value, such as conducting ablations in both chain-based and cell-based search spaces and studying the impact of the number of augmented samples.
4. The experimental comparisons lack several important baselines, such as [1], [2], [3], and [4]. Specifically, a direct comparison with the transformer-based predictor [1] is missing on CIFAR-10, and on ImageNet, only the cell-based results of [1] are compared, omitting its chain-based results. Furthermore, the results of [2], [3], and [4] appear to surpass those reported in this work.
5. For the results on CIFAR-10, it is necessary to report both the mean and standard deviation.

[1] PINAT: A Permutation INvariance Augmented Transformer for NAS Predictor
[2] CARL: Causality-guided Architecture Representation Learning for an Interpretable Performance Predictor
[3] HyperNAS: Enhancing Architecture Representation for NAS Predictor via Hypernetwork
[4] Computation-friendly Graph Neural Network Design by Accumulating Knowledge on Large Language Models

### Questions
see Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
TEFormer is a novel Transformer model designed for NAS performance prediction. It precisely captures the complex topological information of neural architectures by combining Topology-aware Flow Encoding and a Hierarchical Attention Mechanism. Additionally, the model employs an interpolation-based augmentation strategy in the latent space to enhance generalization in few-shot scenarios. TEFormer achieves SOTA performance across multiple NAS benchmarks and includes detailed ablation studies confirming the effectiveness of its components.

### Strengths
1.  The paper provides extensive experimental validation.
2.  The paper includes detailed sensitivity analysis and ablation studies.

### Weaknesses
1.Overall, the core of this paper's contribution is essentially the introduction of a new loss function, which is derived by combining several typical neural network modules, thus lacking novelty.
2. The paper lacks corresponding theoretical proof. I believe providing a relevant theoretical analysis for the proposed loss function would have been a significant addition to this paper.

### Questions
Can you try to theoretically analyze why this network architecture was chosen?

### Soundness
3

### Presentation
3

### Contribution
2
