# Transformers Adaptively Learn Molecular Structures Without Graph Priors

- Decision: Reject
- Scores: 2, 4, 2, 8

## Abstract
Graph Neural Networks (GNNs) are the dominant architecture for molecular machine learning, particularly for molecular property prediction and machine learning interatomic potentials (MLIPs). GNNs perform message passing on predefined graphs often induced by a fixed radius cutoff or $k$-nearest neighbor scheme. While this design aligns with the locality present in many molecular tasks, a hard-coded graph can limit expressivity due to the fixed receptive field and slows down inference with sparse graph operations.  In this work, we investigate whether pure, unmodified Transformers trained directly on Cartesian coordinates—without predefined graphs or physical priors—can approximate molecular energies and forces. As a starting point for our analysis, we demonstrate how to train a Transformer to competitive energy and force mean absolute errors under a matched training compute budget, relative to a state-of-the-art equivariant GNN on the OMol25 dataset. We discover that the Transformer learns physically consistent patterns—such as attention weights that decay inversely with interatomic distance—and flexibly adapts them across different molecular environments due to the absence of hard-coded biases. The use of a standard Transformer also unlocks predictable improvements with respect to scaling training resources, consistent with empirical scaling laws observed in other domains. Our results demonstrate that many favorable properties of GNNs can emerge adaptively in Transformers, challenging the necessity of hard-coded graph inductive biases and pointing toward standardized, scalable architectures for molecular modeling.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper tokenizes all the attributes of molecules and employs a vanilla Transformer to capture features. The authors assert that this approach does not rely on physical priors. Experiments demonstrate that many favorable properties of GNNs can emerge in this Transformer, achieving results comparable to state-of-the-art GNNs.

### Strengths
- This paper is well-written.

### Weaknesses
1.The main argument of this paper lacks innovation. The Transformer learning the representation of GNN is similar to the story of CNN and ViT: when data is scarce, ViT performs poorly, but it performs better when trained with large-scale data. This is because the Transformer is a fully connected model in all dimensions (sequence and embedding dimensions), with a high upper limit of expressive power, but it is prone to overfitting. Whether in vision or molecules, researchers introduce priors to reduce overfitting, thereby reducing training costs and improving generalization ability. Moreover, replicating ViT in the molecular field may not make sense, as existing datasets can only cover a small part of the molecular field. Therefore, the conclusions and results of this paper are likely due to large-scale data training, which has little significance in the field of science.

Graph Neural Networks are originally similar to Transformers (Transformer can be regraded as a fully connected graphs). If you use fully connected graphs for learning, the edge features can also learn the knowledge of cutoff.

2.I am concerned about the generalization of this paper. Although the Transformer has achieved remarkable results in NLP, its effectiveness in continuous space is still in doubt. Methods that divide 3D space into grids may reduce the model's generalization ability. For example, when inferring large protein molecules, GNNs with priors may show more stable results.

The Transformer tends to memorize the patterns of the training set rather than performing inference in many tasks [1], which is fatal for the mathematics in the molecular space.

3.The pure Transformer structure appears to be non-equivariant. Equivariance is a common characteristic of all molecules, so why give up this prior and increase the difficulty of model training? This requires more data or data augmentation. The Equiformer [2] and SE(3)-Transformer [3] models extract features using the Transformer while preserving equivariance, which I believe is a more rational and economical approach.

4.This paper lacks novelty in both algorithms and training. Its biggest contribution is a foundation model for molecules. I believe that such a contribution can only be valuable if it is open-sourced. Unfortunately, I couldn't find the open-sourced model and thus couldn't test it.。

[1] Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens

[2] EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations

[3] SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks

### Questions
See Weaknesses.

### Soundness
3

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
5

### Summary
This paper investigates the necessity of graph neural networks (GNNs) and their inherent graph inductive biases for modeling molecular systems, specifically for machine learning interatomic potentials (MLIPs) that predict energies and forces. The authors challenge the dominance of GNNs by proposing the use of a standard, unmodified Transformer architecture trained directly on atomic Cartesian coordinates, without any predefined molecular graph or explicit physical priors (like equivariance). Using the OMo25 dataset, they demonstrate that a 1B parameter Transformer can achieve competitive accuracy (energy and force MAE) compared to a state-of-the-art 6M parameter equivariant GNN (eSEN) under a matched training compute budget. Analysis of the trained Transformer reveals that it learns physically meaningful patterns, such as attention weights decaying with distance and adaptive attention radii based on local atomic density, effectively learning graph-like locality without being given a graph. Furthermore, the study shows that the Transformer exhibits predictable scaling behavior consistent with observations in other domains, suggesting potential for further improvement with scale, contrasting with reported difficulties in scaling GNNs . The authors argue these findings suggest graph biases might be learned implicitly from data, pointing towards standardized, scalable architectures for molecular modeling .

### Strengths
- The work addresses a core question about the necessity of explicit graph structure and associated inductive biases in molecular ML, challenging the prevailing GNN paradigm. Exploring the capabilities of general-purpose architectures like Transformers in this domain is timely and important.
- Using a standard Transformer with minimal modifications provides a clean testbed to study the emergence of relational patterns directly from data, without confounding factors from complex, hand-engineered modules.
- Achieving competitive accuracy against a SOTA GNN (designed with strong physical priors) using a generic Transformer under matched compute is a significant and perhaps surprising result. It strongly suggests that sufficient data and model capacity might compensate for the lack of explicit biases.

### Weaknesses
- While compute-matched performance is impressive, the Transformer (**1B** params) is vastly larger than the GNN (**6M** params). This raises questions about data efficiency: the Transformer might require significantly more data or computation per parameter update to learn the necessary patterns that GNNs get "for free" from their architecture.
- The experiments are primarily conducted on the OMo125 dataset. How well does the graph-free Transformer generalize to different chemical spaces, system sizes, or tasks (e.g., predicting properties other than energy/force) compared to GNNs on the standard benchmark like QM9 and MD17? GNNs' graph bias might aid OOD generalization , although the paper also notes potential GNN generalization issues.

### Questions
- Could the authors comment further on the data efficiency trade-off? Is the large parameter count primarily needed to learn the relational structure that GNNs encode explicitly?
- It would be valuable to see experiments on generalization, perhaps transferring a trained model to a related but distinct dataset (e.g., QM9) or evaluating on molecules significantly larger than those seen during training.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Graph Neural Networks (GNNs) have shown promise for modeling complex molecular dynamics and relationships. These remain the default architecture for drug discovery tasks. However, GNNs operate on pre-defined graphs which are hard-coded to make learning amenable. The paper empirically studies and compares the behavior of pure transformers, without graph priors, with GNNs for the task of Machine Learning Interatomic Potentials (MLIP). Authors show that pure transformers, when trained on discretized and binned tokens representing xyz cartesian coordinates, energy and forces, perform similarly as state of the art equivariant GNNs. Transformeres are found to be faster and more efficient and obey scaling laws of compute and budget. Furthermore, the paper takes an interpretability approach trying to understand representations learned by the model. Transformer captures local features in its early layers while later layers capture global layers. Additionally, the paper uncovers variation of attention with radius showing that the model adapts its attention pattern as per the radius per atom and packing of atoms in its neighborhood.

### Strengths
* The paper is well written and organized.
* The paper studies and disects the model from an intuitive perspective.

### Weaknesses
* **Contribution:** My main concern is the central contribution and direction of the paper. It is unclear as to what new knowledge and insights the paper is adding. The paper assesses and compares the vanilla transformer architecture with equivariant GNNs for MLIP. However, the paper limits itself to results, trends and insights that have already been established. It is well known that transformers scale and obey scaling laws, irrespective of the type and complexity of data. These results were obtained both theoreticaly and empirically [1, 2] and corresponding to applications to drug design [3]. The paper studies scaling laws and performance of transformers which has been well demosntrated in the past. Furthermore, inference speeds and latency studies of the transformer block have also drawn insights into its optimized operation [4]. At its core, the paper limits itself to a naive application of the transformer architecture to MLIP.

* **Comparison to GNN:** The paper compares a 1B transformer with a state of the art equivariant GNN. However, this comparison remains unclear. It is well known that GNNs also scale and obey scaling laws [5, 6], both in compute and budget. A more thorough evaluation would consider scaling both the transformer and GNN and comparing their scaling trends. Furthermore, while transformer a larger transformers is faster at inference and memory, GNNs are also found to be more efficient when distributing computation. Hence, a fair comparison would consider utilizing the same hardware and training setting for equal parameters and FLOP constraints for different datasets. An additional aspect to note is that GNNs are performant out of the box for both supervised and unsupervised data and handle sparse inputs effectively. An analogous comparison with the transformer model would help shed light on its limitations. Finally, GNNs are more compact (1-100M) even for state of the art models and do not require additional tricks such as KV caching and speculative decoding at inference. Thus, a comprehensive comparison between transformer and GNNs is found to be missing for MLIP.

* **Experiments:** The paper assesses how transformers scale and behave. However, the paper does not answer why they behave in the observed ways. We know that GNNs often fail due to oversquashing. But the paper does not study how transformers hallucinate or beave differently for molecular inputs. The paper aims to uncover their representational aspects but does not take an interpretability-based approach towards the architecture. Instead, authors crudely explain transformer outputs and layerwise representations. It would be worthwhile to evaluate any discovered circuits within the model or a grouping of neurons that specifically stores certain features. Furthermore, authors could help explain activation patterns and the distribution of weights within the residual stream for different molcular types.

[1]. Kaplan et al, Scaling Laws for Neural Language Models, https://arxiv.org/abs/2001.08361.  
[2]. Zhang et al, When Scaling Meets LLM Finetuning: The Effect of Data, Model and Finetuning Method, https://arxiv.org/html/2402.17193v1.  
[3]. Abramson et al, Accurate structure prediction of biomolecular interactions with AlphaFold 3, Nature, 2024.  
[4]. Austin et al, How To Scale Your Model, https://jax-ml.github.io/scaling-book/.  
[5]. Sypetkowski et al, On the Scalability of GNNs for Molecular Graphs, NeurIPS 2024.  
[6]. Shirzad et al, Exphormer: Scaling Graph Transformers with Expander Graphs, ICML 2023.

### Questions
Refer to weaknesses

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
This paper investigates the use of Transformer models for molecular property prediction, specifically energy and force estimation, without relying on the graph-based inductive biases traditionally used in Graph Neural Networks (GNNs). The authors train a Transformer directly on Cartesian coordinates and analyze its performance on the OMol25 dataset. This research challenges the prevailing approach of using GNNs with predefined graph structures and presents a compelling argument that Transformers, when appropriately scaled, can achieve competitive performance for molecular modeling tasks.

### Strengths
### Novel Approach:
The most significant contribution of this work is the application of unmodified Transformers for molecular modeling. By removing graph-based priors, the authors present a more flexible architecture that can potentially scale better and handle a wider range of molecular systems. This is a fresh direction that could influence future molecular property prediction models.

### Competitive Results:
Despite lacking explicit geometric and physical priors, the Transformer model achieves energy and force errors comparable to state-of-the-art equivariant GNNs on the OMol25 dataset, under the same computational budget. This demonstrates that the Transformer can effectively learn the relational and physical patterns in molecular data.

### Scalability and Efficiency:
The paper highlights the scalability of Transformers, noting their ability to perform well with large datasets and scale predictably with training resources. This scaling behavior, backed by empirical results, mirrors findings in other domains of machine learning, such as natural language processing, and suggests that similar approaches can be adopted in the molecular modeling community.

### Insightful Representation Analysis:
The authors perform a thorough investigation of the learned representations, especially the attention maps, and observe that the Transformer naturally learns relationships akin to those found in traditional GNNs. The distance-dependent attention and adaptability to atomic environments are key findings that suggest Transformers can capture complex, spatially localized interactions without needing predefined graphs.

### Potential for Broader Applications:
The paper emphasizes that the Transformer framework could generalize to other chemical systems and could eventually integrate with multi-modal data sources, further enhancing its versatility. The authors also suggest avenues for future research, such as integrating physical constraints and improving the generalization of the model across diverse molecular environments.

### Weaknesses
### Lack of Strict Physical Constraints:
While the Transformer model is successful in approximating energies and forces, it may not adhere to all strict physical principles, which is a limitation for some scientific applications where precise physical fidelity is required. The authors acknowledge this and propose future work to integrate constraints or fine-tune the model, but the current approach may not be suitable for all use cases without such modifications.
### Limited Evaluation in Larger Molecule Systems:
While the model is tested on the OMol25 dataset and achieves competitive results, this benchmark may not fully capture the complexities of larger molecular systems. A more extensive evaluation, particularly in dynamic simulations and larger-scale systems, would offer a clearer understanding of the model's general applicability and robustness across different types of molecular environments.

### Questions
1. Model Performance and Comparison

How does the performance of the Transformer model compare to that of GNNs when tested on other molecular datasets beyond OMol25?

What are the implications of the Transformer model’s speed advantages in terms of computational cost and practicality for large-scale molecular simulations?

2. Generalization and Scalability

While the Transformer achieves competitive performance with GNNs, how does its generalization ability perform when applied to new molecular geometries or diverse chemical systems that differ from those in the OMol25 dataset?

Given the scaling behavior observed with the Transformer, what might be the challenges of scaling this model beyond 1B parameters, and are there limitations to this scalability?

3. Physical Constraints

The paper discusses potential limitations in adhering to strict physical laws. What are the specific physical principles that the Transformer might struggle to respect, and how could these be addressed in future research?

Would a hybrid model combining the flexibility of Transformers with physics-based constraints offer a more accurate solution for molecular dynamics simulations?

### Soundness
4

### Presentation
3

### Contribution
3
