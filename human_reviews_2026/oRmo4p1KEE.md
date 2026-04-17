# QuadGPT: Native Quadrilateral Mesh Generation with Autoregressive Models

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
The generation of quadrilateral-dominant meshes is a cornerstone of professional 3D content creation. 
However, existing generative models generate quad meshes by first generating triangle meshes and then merging triangles into quadrilaterals with some specific rules, which typically produces quad meshes with poor topology.
In this paper, we introduce QuadGPT, the first autoregressive framework for generating quadrilateral meshes in an end-to-end manner. 
QuadGPT formulates this as a sequence prediction paradigm, distinguished by two key innovations: a unified tokenization method to handle mixed topologies of triangles and quadrilaterals, and a specialized Reinforcement Learning fine-tuning method tDPO for better generation quality. 
Extensive experiments demonstrate that QuadGPT significantly surpasses previous triangle-to-quad conversion pipelines in both geometric accuracy and topological quality. 
Our work establishes a new benchmark for native quad-mesh generation and showcases the power of combining large-scale autoregressive models with topology-aware RL refinement for creating structured 3D assets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces QuadGPT, an end-to-end autoregressive framework for generating native mixed quadrilateral and triangular meshes directly from a point cloud input. QuadGPT proposes a unified serialization scheme that handles both triangle and quad faces using a padding-based tokenization, along with an Hourglass Transformer architecture for efficient processing of long sequences. Furthermore, the model is refined using a reinforcement learning approach, which uses a topological reward function to encourage the formation of clean, production-ready edge loops. The authors demonstrate that this approach significantly surpasses prior state-of-the-art methods in both geometric fidelity and topological coherence on a large, curated dataset.

### Strengths
- The paper is clearly written and easy to follow. 
- The experimental results are impressive, demonstrating strong generalization capability on a wide range of meshes.

### Weaknesses
- The method itself is a combination of previous efforts (e.g., hourglass transformer and quad-dominance control form Meshtron, direct mesh tokenization from MeshXL, point cloud encoder from MeshAnything), with the biggest difference as the introduction of a mixed quad-triangle setting, which is a rather simple extension.
- There is no ablation on the dataset. It's hard to tell if the performance boost is mainly from better data quality, and the comparison with previous works trained on public datasets is unfair. 
- Missing references and discussions on the triangle-to-quad conversion algorithm, for example, Blossom-Quad [1] and Blender's built in algorithm.

[1] Blossom-Quad: A non-uniform quadrilateral mesh generator using a minimum-cost perfect-matching algorithm; Remacle, J‐F., et al.

### Questions
- The padding-based serialization seems pretty plain and inefficient for meshes with many triangles. Have the authors considered about using token compression techniques like BPT?
- I wonder if the authors have tried experiments on openly available dataset? It's unfair to compare with other models trained on different datasets (of potentially worse quality). It's understandable to use proprietary datasets for the best performance, but the author should at least do some ablation study on the dataset quality, which can provide some insights for future research (e.g., the TripoSG paper).
- How effective is the quad-dominance parameter for controlling the ratio of quad vs triangular faces? It also sounds pretty empirical to gradually anneal the data distribution with r from 0 to 1.
- What's the insight for native quad mesh generation? Especially, I wonder its difference from first generating pure triangular meshes with the same model (e.g. use r=0), and then apply the proposed triangle-to-quad algorithm.

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
4

### Summary
This work introduces QuadGPT, an autoregressive framework for direct generation of quadrilateral meshes. The input condition is a point cloud with normals. The authors propose a unified representation through padding, supporting mixed-element topologies (triangles + quads). The model is built upon an hourglass architecture. The pretraining loss is a standard cross-entropy loss. The model is further finetuned by a reinforcement learning approach with truncated direct preference optimization, rewarding coherent edge loops. Experiments show that QuadGPT generates higher quality quad meshes when compared to prior methods, both quantitatively and qualitatively.

### Strengths
- This work introduces an end-to-end learning-based framework for direct quad mesh generation from point clouds. This is challenging as meshes have complex structures with significantly large numbers of face and vertex elements, and forming coherent edge loops as in professional-crafted quad meshes is difficult.
- To promote clean topologies in the generated quad meshes, the authors introduce a reinforcement learning stage, optimizing a direct preference optimization objective rewarding long, coherent edge loops. To handle long sequences, the authors use a truncated, local window-based approach.

### Weaknesses
- The novelty of the proposed method is somewhat limited: the straightforward padding in the sequence representation to support triangles + quads, the hierarchical hourglass architecture from MeshTron [Hao et al. 2024], the direct preference optimization already proposed for mesh generation in DeepMesh [Zhao et al. 2025]. Overall, the proposed method seems to be a simple extension of those existing works to quad mesh generation.
- The experimental evaluation is less comprehensive. Comparisons to the triangle mesh generation baselines (e.g., MeshAnything, DeepMesh) may not be fair, due to the difference in training data and model capacity for long sequences (Fig 4).
- The proposed model has less controllability over the ratio of triangles and quads in the output mesh, though a conditioning mechanism with a quad-dominance parameter is introduced.

### Questions
- As mentioned in the weaknesses, the comparisons to triangle mesh generation baselines need to be strengthened. For example, in the training strategy, the authors already pretrained a model exclusively on triangle meshes (L247-248). Combining this model with triangle-to-quadrilateral conversion could be a strong baseline, reducing the difference of training data and model capacity used in other triangle mesh generation baselines.
- The authors introduced a learnable embedding for a quad-dominance parameter to control the target ratio of face types. However, L990 seems to indicate that this is not effective in practice, and there is no corresponding quantitative analysis.
- In L111, the authors claim that QuadGPT bridges the gap between text/image inputs and production-ready 3D artist meshes. However, the majority of the results in the main text are generated from point clouds.
- There is no promise of code and data release. Though the authors mention that a public API will be provided, access to the full model weights and training data is important for reproducibility and follow-up research.

### Soundness
3

### Presentation
4

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
1 The paper extends prior triangle-based mesh generation frameworks to quadrilateral mesh generation by introducing an additional vertex in the sequence representation.
2 The paper propose a topology-aware reinforcement learning fine-tuning method (tDPO) to enhance QuadMesh quality

### Strengths
1. Indeed, this paper extends existing triangle-based mesh generation methods to quadrilateral mesh generation.

2. The paper introduces a reinforcement learning strategy with one reward function that encourages long continuous edges and one penalty function that discourages fractures, aiming to improve mesh quality.

### Weaknesses
Although this paper is the first to extend autoregressive mesh generation to quadrilateral meshes, the extension—essentially adding one additional vertex—feels rather trivial and not strongly innovative.

The training process is also computationally expensive, requiring 64 A100 GPUs for 7 days, which makes it difficult for other researchers to reproduce the results. If the authors do not plan to release the code and weights, the paper’s academic contribution will be quite limited. Furthermore, the model is trained on proprietary licensed assets, which further reduces reproducibility and makes independent verification challenging.

Overall, this paper follows a typical data-driven approach—collect/label/clean a large dataset and train a large model. The methodology itself is not particularly innovative or inspiring, and it is unclear how future researchers could benefit from it.

### Questions
I could like the author to report NC and |NC| similar to other papers.

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
This paper presents an auto-regressive mesh generation model that can produce both triangular meshes and quadrilateral meshes. The model is trained with a multi-stage training strategy: 1) pre-training on triangular meshes only, 2) fine-tuning on a mixture of triangular and quadrilateral meshes, 3) reinforcement learning post-training to enhance the topology of the generated meshes.

### Strengths
1. This paper represents the first work on quadrilateral mesh generation. It proposes a hybrid representation for triangles and quadrilaterals, enabling the generation of artist-like 3D meshes input point clouds.

2. The paper presents a multi-stage end-to-end training framework, which incorporates both traditional next-token-prediction and sequence-level reinforcement learning supervision. The generation capability is improved in a targeted manner.

3. As observed in the metrics and qualitative study, the model achieves superior results. The paper also provides many potential applications of the generated meshes, highlighting its practical usefulness.

### Weaknesses
1. As the model supports both triangular and quadrilateral mesh generation, it would be better if treated triangular mesh generation as an evaluation task. It is interesting to study: 
    - whether training on the mixed representation helps the generation of triangular meshes, 
    - whether the proposed reinforcement learning also improves the generation of triangular meshes.

    It is also a natural request since previous works mainly focus on generating triangular meshes.

2. How well model follows the `quad-dominance parameter`? Now that we have mixed representations, *i.e.* triangular and mixed triangular and quadrilateral representations. Is the `quad-dominance parameter` sufficient for conditioning the model to generate the desired mesh representation? It would be better to show different generation results (triangular and mixed representations) for the same geometry.

### Questions
1. As one of the advantages of generating quadrilateral meshes is to save tokens (representing two triangles ~2 x 9 coords with one quadrilatero ~12 coords) It is also interesting to study how much the current model can represent complex geometries with less tokens compared to previous methods.

### Soundness
3

### Presentation
3

### Contribution
4
