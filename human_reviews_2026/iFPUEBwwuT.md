# Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 6

## Abstract
Existing auto-regressive mesh generation approaches suffer from ineffective topology preservation, which is crucial for practical applications. 
This limitation stems from previous mesh tokenization methods treating meshes as simple collections of equivalent triangles, lacking awareness of the overall topological structure during generation. To address this issue, we propose a novel mesh tokenization algorithm that provides a canonical topological framework through vertex layering and ordering, ensuring critical geometric properties including manifoldness, watertightness, face normal consistency, and part awareness in the generated meshes. Measured by Compression Ratio and Bits-per-face, we also achieved state-of-the-art compression efficiency. Furthermore, we introduce an online non-manifold data processing algorithm and a training resampling strategy to expand the scale of trainable dataset and avoid costly manual data curation.
Experimental results demonstrate the effectiveness of our approach, showcasing not only intricate mesh generation but also significantly improved geometric integrity.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new mesh tokenization algorithm to addresses the issue of topology preservation in auto-regressive 3D mesh generation. The algorithm first organizes vertices into hierarchical layers based on graph distance and a specific ordering. This layered structure is then compressed using self-layer and between-layer adjacency matrices. This canonical topological framework is designed to inherently guarantee crucial geometric properties, including manifoldness, watertightness, consistent face orientation, and part awareness, while also achieving state-of-the-art compression efficiency.

### Strengths
- The paper is well-written and clearly motivated.
- The proposed tokenization algorithm is interesting and achieves state-of-the-art compression efficiency, while also preserving manifold topology and watertightness.
- The experimental results demonstrate the effectiveness of this method.

### Weaknesses
- A primary limitation, acknowledged by the authors, is the large vocabulary size of over 10,000, which increases model complexity and memory requirements. This vocabulary must grow even larger to support higher spatial resolutions, further limiting the method's scalability. 
- Although the tokenization algorithm sets a new compression record, the improvement is marginal, particularly given the algorithm's complexity. 
- The paper lacks an analysis of failure cases, making it difficult to evaluate the method's robustness.

### Questions
- In mesh visualizations like Figure 7, I wonder if the baseline methods are also able to produce part-aware meshes? However, only the proposed method visualizes the parts with different colors. It's better to use the same visualization method for the comparison to be fair.
- The 256 quantization is relatively low for complex mesh generation. Have the authors tried to experiment with larger resolutions? What's the potential problems?

### Soundness
3

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
This paper presents a lossless mesh tokenization algorithm that achieves state-of-the-art compression efficiency while preserving geometric properties like manifold topology and face normal consistency. 
Besides showcasing compression capabilities, the authors train a surface pointcloud guided autoregressive transformer that predicts matching tokenized meshes.
They achieve state-of-the-art geometric accuracy when additionally employing custom batch sampling during training in combination with further data processing.

To encode a mesh, it is split into multiple manifold sub-components, which are tokenized individually and then combined.
For each component, the vertices are grouped into layers according to their graph distance to a reference vertex, and then sorted per layer with regards to the previous layer order and the local half-edge structure.
Each layer then gets tokenized as a sequence of compressed vertex positions, compressed intra-layer connectivity and compressed connectivity to the previous layer.
Thereby, each part uses an own tokenization scheme based on its observed typical structure, achieving strong compression.

### Strengths
The presented tokenization scheme achieves state-of-the-art bits-per-face ratio and compression ratio compared to previous mesh generation tokenization schemes. 
It is based on a detailed analysis of the structure of real-world mesh datasets.
Furthermore, its ability to preserve manifold topology makes it valuable for real-world applications.
This is reflected in the highly competitive geometric accuracy and quality scores in the generation experiment.

The paper is well written for the most parts and supported by very helpful figures for understanding the tokenization process.

### Weaknesses
My main concerns are w.r.t. the claims of the paper regarding watertightness and manifoldness of the inferred meshes, as well as the influence of the resampling step in the training strategy compared to other compression methods.

- The paper makes multiple claims about the properties of the generated meshes that are not obvious, yet are not further explained (e.g. watertightness (Table 1 and L073) or manifoldness (L312)). The autoregressive transformer is not incentivised to always predict a properly tokenized mesh with desirable geometric properties. Where do these properties come from? Is this achieved through additional validation during the generation process?  Since this is a major feature of the method, it should be explained in more detail.

- The resampling strategy seems to be crucial for competitive generation results (Table 2 and 3). Yet it seems, that the baseline scores were produced without a similar sampling during training. At least for the retrained EdgeRunner, employing the sampling would be helpful to understand the advantage of the training procedure versus the novel representation. (Would infrequent long sequences be more harmful for methods with lower compression ratios?)

On a more minor note, I have the following concerns:
- The effect of extended edge merging for handling non-manifold edges during pre-processing on the resulting sub-components and generation quality is unclear. Demonstrating the former qualitatively and the latter quantitatively could illustrate the contribution here.

- Writing: At several points, the text shifts between tokenization and generation contexts without explicitly mentioning it (L104, L134, Section 3.3). It may be helpful for the reader if the topic switch is made clearer here.

- Minor: L070 typo? "stems from ... methods treat(ing?)"

- Minor: Figure 2 uses the lowest layer as the innermost point. However, for Figure 3, this is flipped.

### Questions
See weaknesses.
* Is the SotA performance in terms of geometric accuracy mainly due to the resampling strategy or due to the compression scheme/representation?
* How are properties such as manifoldness guaranteed from arbitrary transformer-sampled token sequences?

### Soundness
2

### Presentation
3

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
This paper propose a new tokenization methods to improve the the compression ratio and bits-per-faces in mesh generation. New tokenization is based on vertex layering and local connectivity prediction.

### Strengths
I agree that the proposed method achieves the same compression ratio (0.22) as TreeMeshGPT. It also introduces a novel approach to connectivity compression, conceptually similar to sparse matrix compression, which may inspire future research. The preprocessing strategy for handling non-manifold meshes appears reasonable and may also benefit related studies.

### Weaknesses
The proposed method relies on a fixed layering size M, and meshes exceeding this vertex limit are discarded during training. This constraint prevents the model from generalizing to larger and more complex meshes, and it seems difficult to overcome. The compression ratio, which directly relates to computational resources, remains identical to TreeMeshGPT, showing 0 improvement. Although the new metric Bits-per-Face shows about a 10% improvement, the paper does not explain or demonstrate the practical impact of this new metric. Furthermore, the proposed local connectivity scheme appears overly complex and delicate, which may cause a long-tail effect during training. I would need to examine the implementation to be convinced that it can generalize to more complex meshes.

In summary, this is a complex and fragile approach that offers no tangible improvement while introducing additional limitations, making me hard to accept it to ICLR.

### Questions
In Table 2, the Chamfer Distance (CD) values of all baselines are 2–4 times larger than those reported in other papers (e.g., MeshMosaic, MeshWeaver). Could the authors provide an explanation for this discrepancy?
Additionally, the Chamfer Distance of the proposed method is reported as 0.079. If the meshes are normalized to the range [-1, 1], this corresponds to roughly a 5% error, which suggests the generated meshes might contain significant noise or failure. Could the authors clarify this point?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a novel tokenization algorithm for auto-regressive 3D mesh generation. Inspired by the process of weaving silk, the method organizes mesh vertices into hierarchical layers and establishes a canonical ordering. This structured approach, by its very design, guarantees the preservation of critical geometric properties such as manifoldness, watertightness, consistent face orientation, and part awareness—properties that are often violated by existing methods. The authors demonstrate that their approach not only ensures geometric integrity but also achieves a state-of-the-art compression ratio. The paper further contributes a practical online non-manifold data processing algorithm and a progressive resampling strategy to enable effective training on large-scale, uncurated datasets.

### Strengths
**1、Novel and Elegant Tokenization:** The core contribution—a tokenization scheme based on vertex layering and ordering—is highly novel and intuitive. The "weaving silk" analogy provides an elegant conceptual framework for ensuring local and global mesh consistency, which is a significant departure from prior methods that often treat meshes as an unstructured collection of triangles.

**2、Guaranteed Geometric Properties by Design:** This is the most compelling strength of the paper. Instead of relying on the model to learn geometric priors implicitly, the proposed framework enforces them explicitly through its algorithmic design. The guarantee of manifoldness, watertightness, and normal consistency is a major practical advantage, potentially eliminating the need for costly and often imperfect post-processing steps that are common in other generation pipelines.

**3、Strong Qualitative and Quantitative Results:** The paper provides convincing evidence to support its claims. Table 1 offers a clear and impactful qualitative comparison, highlighting the deficiencies of baseline methods. The quantitative results in Table 2, particularly the superior performance on Normal Consistency (NC and |NC|) metrics, provide strong empirical validation. Furthermore, the visual comparisons in Figure 7 are very effective, especially in showcasing the non-manifold artifacts in BPT and DeepMesh, which are absent in the proposed method's outputs.

**4、Practical Contributions for Large-Scale Training:** The inclusion of an online non-manifold processing algorithm and a progressive-balanced resampling strategy addresses crucial real-world challenges in training generative models on noisy, long-tailed web datasets. These contributions enhance the practicality and scalability of the proposed system.

### Weaknesses
While the paper presents a compelling framework, several points regarding its efficiency, scalability, and experimental scope warrant further discussion.

**1、Unexplored Scalability to Higher Polygon Counts:** The paper does not fully explore whether the proposed method can generalize to higher-polygon meshes, such as those with 10,000 or 20,000 faces. The methodology relies on a Maximum Vertices per Layer limit (m=200) to maintain efficiency. Does an increase in face count necessitate a corresponding increase in this limit, which could in turn lead to significant computational overhead? If so, are there potential solutions to this challenge? An analysis from the authors on this topic would be valuable.

**2、Unclear Scalability to Higher Quantization Resolutions:** It is unclear if the method can be efficiently generalized to higher resolutions, such as 9-bit. Would such an extension further increase the vocabulary size? Would it also necessitate an increase in the Maximum Layer Number or Maximum Vertices per Layer limits, thereby introducing substantial computational costs? The paper would be strengthened by an analysis of these questions, ideally supplemented with lightweight experiments to illustrate the potential trade-offs.

**3、Absence of Ablation on Tokenizer Complexity:** The proposed tokenizer is significantly more complex than those of its predecessors. While this complexity is key to the method's success, the paper would be strengthened by an ablation study to provide a more complete picture of the design trade-offs. A more complex token representation and a larger vocabulary can create a more challenging optimization landscape. An ablation study—for instance, simplifying the tokenization scheme—would have provided invaluable insight into the contribution of each component.

**4、Concerns Regarding Training Efficiency:** The reported training cost appears notably high when compared to key baselines. The authors report training their 500M model for 15 days on 16 H800 GPUs (240 GPU-days), whereas BPT trained a similarly sized model in 7 days on 32 L40 GPUs (224 GPU-days). Considering the significant performance advantage of an H800 over an L40, the effective computational budget for this work is likely several times higher than BPT's. While the method's geometric guarantees are a clear advantage, the numerical improvements on standard metrics (CD, HD), while positive, appear modest in comparison to the significant increase in computational budget. This raises questions about the cost-benefit trade-off of the proposed approach.

### Questions
**1、Regarding Scalability to Higher Polygon Counts:** The paper's analysis of scalability primarily focuses on the Maximum Vertices per Layer limit. However, an increase in mesh density (e.g., to 10k-20k faces) would naturally challenge both predefined limits. Could you elaborate on the relationship between face count and not only the Maximum Vertices per Layer but also the Maximum Layer Number? How would the current framework handle meshes that exceed these thresholds, and what architectural modifications might be necessary to support high-detail assets without incurring prohibitive computational costs?

**2、On the Rigidity of a Canonical Representation：** The proposed method generates a single, canonical token sequence for any given mesh. While this ensures consistency, could this deterministic representation be overly restrictive compared to methods that might allow for multiple valid tokenizations of the same shape? Does this potentially limit the model's ability to learn a more flexible and varied distribution of 3D shapes?

### Soundness
3

### Presentation
3

### Contribution
3
