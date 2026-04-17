# Auto-Regressive Surface Cutting

- Decision: Reject
- Scores: 6, 4, 10, 6

## Abstract
Surface cutting is a fundamental task in computer graphics, with applications in UV parameterization, texture mapping, and mesh decomposition. However, existing methods often produce technically valid but overly fragmented atlases that lack semantic coherence. We introduce SeamGPT, an auto-regressive model that generates cutting seams by mimicking professional workflows. Our key technical innovation lies in formulating surface cutting as a next token prediction task: sample point clouds on mesh vertices and edges, encode them as shape conditions, and employ a GPT-style transformer to sequentially predict seam segments with quantized 3D coordinates. Our approach achieves exceptional performance on UV unwrapping benchmarks containing both manifold and non-manifold meshes, including artist-created, and 3D-scanned models. In addition, it enhances existing 3D segmentation tools by providing clean boundaries for part decomposition.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes SeamGPT, an auto-regressive transformer model for generating surface-cutting seams on 3D meshes.
The authors reformulate surface cutting — a classic problem in UV parameterization and mesh segmentation — as a sequence generation task, treating each 3D coordinate of a seam segment as a discrete token predicted autoregressively.
A point-cloud encoder encodes sampled vertices and edges into a latent shape embedding, which conditions a GPT-style hourglass transformer decoder to sequentially predict seam coordinates.
The method is trained on a large dataset (≈560K artist-annotated meshes filtered from Objaverse/3D-Future) and evaluated on UV unwrapping and part segmentation benchmarks.

While the paper presents a polished system and thorough experiments, the core technical novelty is minimal. Most components (point-cloud encoder, quantization, autoregressive transformer, hourglass hierarchy) are directly adopted from existing works such as MeshGPT (Siddiqui et al. 2023) and Meshtron (Hao et al. 2024) with only superficial adaptation to a new task.
As a result, the overall contribution feels incremental and more of an application of existing sequence modeling methods than a fundamental advance in 3D geometry understanding.

### Strengths
The paper’s strength lies mainly in its problem framing rather than architectural innovation. It takes an established operation in computer graphics—surface cutting and UV unwrapping—and recasts it as an auto-regressive generation task. This conceptual reformulation is novel from a modeling perspective and has some intuitive appeal, as it parallels how artists progressively define seams in practice. The proposed workflow is well structured and technically sound: it integrates a point-cloud encoder to extract shape features and a transformer decoder for coordinate generation, both of which are established and well-tested components. The experimental section is extensive, evaluating across multiple benchmarks (FAM, Toys4K, AI-generated meshes), with both quantitative metrics and qualitative visualizations. The results consistently show that SeamGPT can produce fewer fragmented UV charts with comparable or slightly lower distortion compared to baseline methods. The inclusion of user studies with professional artists is also commendable, as it adds a perceptual validation dimension to the otherwise geometric evaluations. The paper is generally well written and easy to follow, and the authors provide sufficient implementation details for reproduction.

### Weaknesses
Despite a polished presentation, the paper’s technical novelty and empirical contributions are quite limited. The proposed framework is almost entirely composed of pre-existing components: the point-cloud encoder is borrowed from standard 3D transformer models, and the auto-regressive decoder closely follows prior work such as PolyGen, MeshGPT, and related sequence models for 3D geometry. The only distinct aspect—the reformulation of cutting seams as quantized coordinate sequences—feels incremental rather than conceptually new.

More concerning, the experimental results do not convincingly support the claimed superiority. In Tables 2 and 3, SeamGPT’s distortion metrics are often worse than or comparable to XAtlas, which is a purely geometric non-learning baseline. For example, on several FAM models (e.g., Bimba, Dragon, Happy Buddha) and most Toys4K categories, XAtlas yields lower distortion. The authors highlight fragment reduction, but this is a side effect of predicting fewer seams rather than genuine geometric improvement. The evaluation lacks a rigorous statistical analysis or significance testing, and the results fluctuate widely across categories.

### Questions
- Why does SeamGPT perform worse than Xatalas in distortion metrics on many test cases if its purpose is to improve UV quality?
- How sensitive are the results to the quantization resolution and seam-length control parameter?

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
This paper introduces a GPT-based auto-regressive model to infer and generate cutting seams derived from data. The surface cutting problem is turned into a next token prediction task and a language-like learning is conducted to infer seams. The experiments on several meshes demonstrate the seams inferred this way can better align artists' preferences.

### Strengths
- Overall, it's a good idea and a low hanging fruit to approach seam cutting through a GPT-like architecture. 
- I believe that the results are accurate and good quality can be achieved this way. 
- The paper clearly describes the approach and except some doubts on implementation (see below) the work seems to be reproducible.

### Weaknesses
- Abstract claims exceptional performance. This is not validated by the experiments. Please tone down.
- Missing baselines (See Q1). 
- UV texturing can introduce bad seams. There are some new works that discuss and alleviate this problem. See: 
Foti, S., Zafeiriou, S., & Birdal, T. Uv-free texture generation with denoising and geodesic heat diffusion. NeurIPS 2024. More on this in Q3 below. 
- Ordering (for example yzx) is rotation dependent and there seems to be no treatment of this. See Q4.
- Large triangles seem to be problematic for this work. Point cloud networks that operate on mesh vertices will fail if the surface is not resampled. Large triangles will also cause more errors in the seams which are directly defined over edges / vertices. See Q6. 
- In principle, edges have infinite number of points whereas vertices are finite. Even sampling on both does not seem justified. This must be studied in a controlled manner. See Q7.
- I'm not sure if qualitative examples in Fig. 3 are conclusive enough to justify the 'artistic' quality superiority.
- Appendix seem to report that SeamGPT is faster in runtime, whereas table 1 shows that it's slower. There seems to be some inconsistency or not consistent benchmarking of runtime.
- Paper uses graph convolutions as baselines for mesh processing. This is not okay. I suggest comparing to any mesh convolution based network. 
- Before making conclusive statements, I would like to see the results before Blender's minimum stretch algorithm is applied. One needs to gauge how much of the actual contribution is coming from this. In fact, this should be applied to other methods as well, for a fair comparison. 
- Social impact: This work is not theoretical. It has immediate practical application and can be used by artists. As such, I invite the authors to think a little about the implications of their work rather than dismissing this mandatory section. We owe this much to our community. 

Minor weaknesses:
- Ln. 41: flatten -> flattening
- Ln. 64: init -> initialize
- Ln. 203: S has to be ordered, not a set. This is true for all in Eq. 1. In fact I'm not sure if Eq. 1 is actually needed. It is covered in preceding paragraph. 
- Ln. 267: Isn't H20 96GB? (text says 98)
- Ln. 367: S was reserved for sequence, now used for depicting a 3D shape
- "Does pointer networks work?" section is not an ablation study. Please use the term correctly.

### Questions
1. What about a non auto-regressive method based on for example diffusions? Can we make a simple baseline and compare?
2. Will the authors make the filtered dataset public? (Maybe indices of the models?)
3. UV-parameterization naturally suffers from introducing arbtirary seams that are not meant to be in the original shape. So what about these seams that had to be there not because of semantics but because of the drawbacks of UV? 
4. How are rotations of the meshes handled? An equivariant network? It feels like data augmentation would just cause additional problems here. 
5. How is the quantization in Ln. 219 precisely done? 
6. What about large triangles? How are those handled? Are there different resampling strategies? Any of these ablated? 
7. Why are the points on vertices and edges evenly split? What about other ratios?
8. Does the paper compare to MeshGPT encoder? 
9. How about using a test set that is split from the training set in the experiments? Did the authors try this? 
10. In experiments, why is distortion an indicator of semantic superiority?  
11. How is the seam lines are used to partition the shape into P_i as in Ln. 374?
12. I don't see any reason why seams should align with semantic 3D parts of the objects. Could this be justified?
13. Can we have quantitative results corresponding to Fig. 5 and maybe compare with some other sampling methods? 
14. How is R chosen in practice? I mean the actual value.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
The paper formulates surface cutting as a next token prediction task, and designs a novel auto-regressive architecture that predicts seam coordinate sequences from a given mesh.

### Strengths
1. The paper proposes a new paradigm for surface cutting, which formulates surface cutting as a next token prediction task. The idea is novel, and the results are pretty good.
2. Surface cutting is a very important task in 3D understanding. It essentially finds the best way to geometrically segment a 3D surface into parts (with different criteria). With the part information, it potentially boosts a variety of downstream tasks, such as semantic segmentation (as demonstrated), texture editing, rendering, generation, animation, articulated objects, etc. Hence, the contribution of this paper to the community is significant in my opinion.

### Weaknesses
1. As the method is trained purely supervised by ground truth cuttings, the quality of the ground truth cuttings matters a lot, and the model might be sensitive to the poor samples. As the authors mentioned, a rigorous filtering process was applied to clean the data. Thus, scaling up the dataset may be laborious. 
2. Some details about the paper are not clearly described, which I will mention in the question section.

### Questions
1. How does the number of sampled points affect the performance?
2. How to choose K at line 244?
3. In the data augmentation, how large a portion will the masked region be? Is the method able to predict seams for a part of the object (instead of feeding in the whole point cloud, feed in a point cloud sampled from a part of the object)? 
4. The topologies of the objects shown in the paper are fairly simple. How are the generated cuttings for objects with complex topologies? Is this limited by the number of mesh faces?
5. Instead of controlling the segment count by the seam length, is it possible to do this hierarchically? For example, predict basic seams that cut the surface into a small number of segments, and then cut each of the large segments into smaller segments hierarchically.
6. Will the dataset and code be open-sourced?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SeamGPT, an auto-regressive model for 3D surface cutting and UV unwrapping. The method reformulates surface cutting as a sequence generation problem, where cutting seams are predicted token by token within a quantized 3D space. SeamGPT achieves strong results on both UV parameterization and part segmentation benchmarks.

### Strengths
The idea of framing surface cutting as an auto-regressive sequence prediction task is novel and well-motivated. The integration with PartField yields particularly clean and semantically coherent part boundaries, leading to visually impressive segmentation outcomes. The approach also demonstrates solid generalization across datasets and diverse mesh types.

### Weaknesses
The method section would benefit from additional toy visualizations to clarify the intuition behind the sequence representation and the quantization/tokenization strategy. Some architectural details (such as hierarchy levels and quantization schemes) could be illustrated more intuitively to help readers grasp the overall process.

### Questions
Could the authors add simple illustrative examples (e.g., 2D surfaces or flat cubes) to show how the auto-regressive process operates geometrically and how seams are tokenized?

Since vertex coordinates are quantized and generated sequentially, could BPT-style point compression or token sparsification be integrated into SeamGPT’s decoder to improve efficiency?

During data preparation, did the authors consider incorporating feature-line extraction or curvature-sensitive priors beyond UV seams to better capture subtle geometric cues?

The fandisk example shows missing curved boundaries, would integrating differential geometric features (such as curvature flow) help improve seam placement in such cases?

### Soundness
3

### Presentation
3

### Contribution
3
