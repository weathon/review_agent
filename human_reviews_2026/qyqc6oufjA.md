# ProcGen3D: Learning Neural Procedural Graph Representations for Image-to-3D Reconstruction

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
We introduce ProcGen3D, a new approach for 3D content creation by generating procedural graph abstractions of 3D objects, which can then be decoded into rich, complex 3D assets. Inspired by the prevalent use of procedural generators in production 3D applications, we propose  a sequentialized, graph-based procedural graph representation for 3D assets. We use this to learn to approximate the landscape of a procedural generator for image-based 3D reconstruction. We employ edge-based tokenization to encode the procedural graphs, and train a transformer prior to predict the next token conditioned on an input RGB image. Crucially, to enable better alignment of our generated outputs to an input image, we incorporate Monte Carlo Tree Search (MCTS) guided sampling into our generation process, steering output procedural graphs towards more image-faithful reconstructions. Furthermore, this enables improved generalization on real-world input images, despite training only on synthetic data. Our approach is applicable across a variety of objects that can be synthesized with procedural generators. Extensive experiments on cacti, trees, and bridges show that our neural procedural graph generation outperforms both state-of-the-art generative 3D methods and domain-specific modeling techniques.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
TLDR: Neural procedural graph representation for image-to-3D reconstruction using transformer + MCTS-guided sampling.

This paper proposes ProcGen3D, which reconstructs 3D objects from RGB images by generating procedural graph abstractions (skeleton-like nodes/edges with attributes) that are decoded into high-fidelity meshes via pre-existing procedural generators. Uses edge-based tokenization + GPT-style transformer with MCTS-guided sampling for better image alignment. Key insight: predicting intermediate graph representation avoids issues with predicting raw parameters (usually non-differentiable). Evaluated on cacti, trees, and bridges using Infinigen and CEM generators.

### Strengths
- Novel design choice: procedural graph as intermediate representation (skeleton-like abstraction) avoids non-differentiability issues of parameters.
- MCTS-guided sampling improves image alignment vs. pure autoregressive generation
- Compact representation (few hundred nodes for 10M+ face meshes)
- Well-motivated trade-off: parameters (unstable) vs programs (complex) vs graph (sweet spot)

### Weaknesses
- Critical limitation: requires pre-existing procedural generators (Infinigen for trees/cacti, CEM for bridges) - cannot generalize to new categories without building generators first
- Cannot process organic shapes that are very hard to represent with procedural programs (animals, humans, irregular forms) - major applicability constraint
- Missing discussion with program-code-generation methods (LL3M, MeshCoder) that generate procedural code directly - should give more discussion on this.
- MCTS adds significant computational cost at inference time.
- Training only on synthetic data; real-world generalization not fully validated
- Procedural generators are external dependencies - not created or learned by the method

### Questions
1. Can this approach generalize to categories without existing procedural generators? Specifically, how can organic shapes (animals, humans, irregular forms) be handled? What's needed to add a new category?
2. How do you compare with direct program generation method.

### Soundness
2

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
2

### Summary
This paper introduces ProcGen3D, a novel framework for image-to-3D reconstruction that learns to generate procedural graph representations of 3D objects, rather than conventional voxel, point, or neural field outputs. Each object is represented as a procedural graph capturing topology, structure, and attributes that can later be decoded into high-fidelity 3D assets via traditional procedural generators. The authors employ a GPT-style transformer to autoregressively predict edge-based tokens conditioned on image features, and enhance alignment with input images through Monte Carlo Tree Search (MCTS)-guided sampling during inference. Experiments on synthetic datasets (trees, cacti, bridges) demonstrate superior results over diffusion- and transformer-based baselines such as TRELLIS, Wonder3D, and TreeDFusion.

### Strengths
Innovative Representation:
The idea of learning procedural graphs as the latent 3D representation is novel and conceptually elegant. It bridges neural generative modeling and procedural graphics in a meaningful way.

Compact & Interpretable Outputs:
Procedural graphs are lightweight and structured, providing interpretable intermediate representations rather than opaque neural fields or dense meshes.

Effective Image Alignment:
The use of MCTS-guided sampling for test-time refinement is an original and well-motivated technique to improve consistency with image inputs without retraining.

### Weaknesses
Missing comparison:
The paper doesn't compare to DI-PCG, which I believe is a very important, relevant baseline

Limited real world examples:
Although the method claims to generalize to real world examples, number of results for the same is limited.

Limited Scope of Objects:
The evaluated categories—trees, cacti, bridges—are all graph-structured and hierarchical. It’s unclear how well the method extends to more complex or amorphous shapes (e.g., vehicles, furniture).

Computational Overhead of MCTS:
While MCTS improves alignment, it introduces extra test-time computation, and runtime comparisons are not clearly reported.

### Questions
How interpretable are the learned graphs — can users manually edit them meaningfully?

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
This paper proposes a new algorithm for 3D generation conditioned on an image prompt. Instead of NeRFs, SDFS, etc. it utilizes procedural graph abstractions as 3D representations. Those graph abstractions are then tokenized and a GPT-style Transformer is trained to do next-token prediction given a pre-trained image embedding. During inference time, the algorithm includes a Monte Carlo Tree Search sampling procedure to better enable the generation of fine details. Experiments are carried out on several categories of procedurally-generated objects, showing improvements over TRELLIS, Wonder3D, and TreeDFusion (on the tree category). The efficacy of the method is also validated on real-world images from those categories of objects.

### Strengths
- As far as I know, this is the first work on image-conditioned 3D generation using procedural graphs as the 3D representation. Procedural graphs have many advantages, such as generality and being able to represent details well and succintly.
- The construction of the Transformer-based generative model is sound.
- Experiments show good qualitative results.

### Weaknesses
The weaknesses of the paper fall into two main points. First, I think that the experiments are incomplete.
- The model is trained separately on each category of objects. This calls the experimental results' generality into question. Was training done on a more diverse dataset?
- Wonder3D and TRELLIS were both trained on a diverse set of objects. Therefore, comparing to them in a category-specific manner is not quite fair to them. It would strengthen the paper to include another category-specific baseline beyond TreeDFusion on trees - Get3D [1], for example?

Second, the presentation is unclear at some important points.
- No details of the model or training were provided in the appendix.
- The exposition of the MCTS is confusing at times. 1) How is a leaf state defined? 2) How are candidate child states computed? 3) How is $Q(s_i, s_{i+1})$ defined?

[1] Gao et al. "GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images", NeurIPS 2022.

### Questions
No question beyond those mentioned above.

### Soundness
3

### Presentation
1

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
The paper proposes a framework to create 3D objects from a procedural graph representation. The graph generation is performed by a transformer model, trained to autoregressively predict edge-based tokens from a single image. At inference time, the approach uses Monte Carlo Tree Search (MCTS) guided sampling. The model is evaluated on three procedurally generated categories – cacti, trees, and bridges – and outperforms strong baselines (TRELLIS, Wonder3D, TreeDFusion) on Chamfer Distance, Topo-Sim, LPIPS, and CLIP similarity.

### Strengths
1. Using a transformer model to learn the procedural graph generation is interesting and allows more compact representation.
2. MCTS-guided sampling is novel and experimentally performs better for complex local geometry.
3. The approach seems to generalize for real images based on qualitative evaluations.

### Weaknesses
1. The approach is limited to a single-view image which is not sufficient for capturing the full geometry of the object.
2. The overall generation will be limited by the limitations of autoregressive models, i.e., overall structure will be limited by the order of generation and errors can easily propagate. There is no discussion in the paper on the robustness of the generation process to small early mistakes.
3. The overall dataset is very limited with just three categories. The set of real-images is also very limited.
4. There’s no discussion of the runtime of the entire process. MCTS refinement is likely to add significant time to the overall generation process.

### Questions
1. The bridge in the example is symmetric but there’s no inductive bias in the model. How is the autoregressive model learning symmetric structures?
2. How robust is the autoregressive generation to small error propagation?
3. What is the inference-time of the entire generation process?

### Soundness
3

### Presentation
3

### Contribution
3
