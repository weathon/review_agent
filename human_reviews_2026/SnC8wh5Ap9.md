# Hyperbolic Hierarchical Clustering for Visual Representation Learning

- Decision: Reject
- Scores: 8, 6, 2, 6

## Abstract
We investigate the token mixer in vision backbones by revisiting clustering, one of the most classic approaches in machine learning. 
An effective token mixer is a fundamental component of modern vision backbones like vision Transformers, facilitating information exchange between image patches. Mainstream token mixers, which rely on convolution, attention, MLP, or their hybrids, primarily focus on navigating the trade-off between accuracy and computational cost. However, a significant drawback of these methods is their black-box nature; their encoding process is opaque and lacks interpretability. Diverging from these opaque designs, we introduce ClusterMixer, a transparent token mixer that is grounded in a clustering paradigm and interpretable by design. ClusterMixer explicitly formulates the token mixing process through a hierarchical clustering mechanism. To model the natural, tree-like relationships inherent in visual data, the clustering is performed in hyperbolic space, which is well-suited for embedding hierarchies with low distortion. Building on this innovation, we present HCFormer, a new backbone architecture that integrates ClusterMixer with a series of meticulously designed clustering strategies to ensure robust performance across tasks. Extensive experiments demonstrate that HCFormer consistently outperforms its counterparts across diverse tasks, including image classification, object detection, instance segmentation, and semantic segmentation. Considering its transparency and efficacy, we hope HCFormer can facilitate a paradigm shift toward interpretable backbones. Our source code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper builds on prior work that uses clustering for token mixing in visual tasks.  The main innovations are to use a hierarchical structure in clustering and to use a hyperbolic space for clustering at a coarser stage.  The work is supported by experiments in visual domains, identification, semantic segmentation and object detection and instance segmentation.  Ablations are included to show the value of these innovations.

### Strengths
The paper makes interesting and well motivated modifications to clustering based token mixing.  These seem to lead to real improvements in performance.  The clustering based approach is simpler and intuitive, compared to attention based methods.

### Weaknesses
No significant weaknesses, but there are a number of ways in which the clarity of the paper could be improved.
•	Eq. 2 is confusing because you sum a_{i,j} from 1 to M and from 1 to N.  I think something is wrong here.
•	In Eq. 3, what does Norm do?  I at first thought it took the norm of g_i, but that doesn’t seem to make much sense.  Does it normalize it?
•	Some small errors in grammar (eg., We first partitions)
•	Cluster Assignment, might mention clustering approaches that use soft assignment, eg., E-M.
•	This description of Center Estimation could be a little more detailed, maybe in supplementary
•	Descriptions of the results are slightly misleading because the new method is compared to published work that uses somewhat fewer parameters.  It does seem though that the proposed method improves performance.  I know that the number of parameters can’t be equalized, but in some cases the authors could point out that their method achieves the same performance as others with fewer parameters.  
•	In ablations, it is confusing because comparisons seem to be to the full model.  It seems like shifted windows should be compared to a hierarchical method with Euclidean distances?  I’m assuming that the Euclidean method that is compared to the hyperbolic one is still hierarchical?
I guess the other limitation of the paper is that the method is not completely original, building closely on prior work, and that the performance increase is somewhat small.  But I still find the results very interesting.

### Questions
One of the primary differences between cluster-based token mixing and transformers is that in the cluster-based method the embeddings are compared directly, whereas in transformers they are compared through a query and key, and also combined through a value matrix.  It would be interesting to have some discussion of this issue.  Does their work imply that values, queries and keys are not necessary?  Is there literature on this issue?  Could there be ablations that show that dropping the value, query and key does not hurt performance?  Or is there some reason that they are needed in transformers but not in clustering?  I don’t really expect the authors to address this in the rebuttal, but I find it curious.

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
4

### Summary
This paper proposes HCFormer (Hyperbolic Hierarchical Clustering Transformer), a novel vision backbone that replaces self-attention with a hierarchical clustering-based token mixer. The key idea is to first perform local Euclidean clustering at the patch level, followed by global hyperbolic clustering across window-level features. The authors argue that hyperbolic geometry (negative curvature) naturally captures hierarchical semantics, offering both representational compactness and computational efficiency. Experimental results on ImageNet, COCO, and ADE20K show consistent improvements over CoC and FEC under comparable parameter and FLOP budgets.

### Strengths
- **Conceptual novelty**: Introducing hyperbolic geometry into a hierarchical clustering-based vision transformer is a fresh and well-motivated idea. The connection between tree-like semantic structures and negative curvature is theoretically sound and intuitively clear.

- **Intuitive architecture**: The design cleanly unifies local Euclidean and global hyperbolic clustering. The structure is modular and can be plugged into standard ViT-style backbones.

- **Strong experimental coverage**: Evaluations span classification, detection, and segmentation. Gains of +0.7–1.0% Top-1 on ImageNet and consistent improvements on COCO/ADE20K demonstrate robustness.

- **Comprehensive ablations**: The paper includes clear ablations showing the independent and combined effects of hierarchical structure (+0.8–1.0%) and hyperbolic space (+0.5–0.7%).

- **Presentation quality**: The paper is very well written, with clear figures and consistent notation. Methodology and empirical sections are easy to follow.

### Weaknesses
* **Efficiency claim not fully supported.**
  While the paper theoretically reduces complexity from ($O(N^2)$) (self-attention) to ($O(NM)$) via hierarchical clustering, the *measured* FLOPs and throughput remain almost identical to CoC and FEC. The hyperbolic distance computation (arcosh + mapping functions) introduces overhead that cancels out the theoretical benefit.

* **Performance gains are modest.**
The reported gains are steady but incremental. In Table 1, HCFormer outperforms the prior FEC baseline by roughly +0.7–1.0 Top-1 % on ImageNet-1K under similar FLOPs and parameter budgets. The largest relative gap (Tiny → FEC-Small) is +2.3 points, while improvements shrink at larger scales (+1.2 points for Medium). Ablations (Table 4b–c) show that hierarchical clustering contributes about +2.3 points, whereas the hyperbolic geometry adds +1.6 points in a 6 M-parameter setting. However, the hyperbolic benefit is not shown for larger backbones or downstream tasks, and no analysis is provided on curvature sensitivity or numerical stability. Hence, while the geometric component is empirically beneficial, its effect remains secondary and under-validated

* **Limited analysis of hyperbolic embedding.**
  There is no exploration of curvature sensitivity, embedding radius, or gradient stability of the exp/log mappings. Without such analysis, the hyperbolic advantage remains somewhat qualitative.

* **Sec. 3.2 Cluster Mixer Writing Clarification**.
Section 3.2 primarily restates the CoC/FEC-style token–cluster mixing operation (soft assignment → aggregation → redistribution). The real novelty appears in Sections 3.3 and 3.4, which introduce hierarchical organization and hyperbolic distance. Not sure the claim of a new token mixer from the authors is legitimate.

### Questions
1. How sensitive are the results to the number of clusters (M) and the curvature parameter of the hyperbolic space?
2. Have you measured actual runtime efficiency (throughput) compared to FEC/CoC on identical hardware?
3. Could the same hierarchical clustering structure achieve similar benefits in Euclidean space with proper scaling or depth adjustments?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposes HCFormer, a MetaFormer-style vision backbone implementing ClusterMixer for token mixing via hierarchical clustering across dual geometries: Euclidean space for patch-level clustering within local windows and hyperbolic (Lorentz) space for window-level global abstraction. While claiming interpretability and efficiency advantages, the paper demonstrates competitive results on ImageNet, ADE20K, and COCO benchmarks but suffers from fundamental technical inconsistencies and unsupported theoretical claims.

### Strengths
1. Novel architectural concept: Dual-geometry approach (Euclidean for fine-grained, hyperbolic for hierarchical) is creative
2. Comprehensive evaluation: Experiments across ImageNet, ADE20K, COCO with consistent improvements over clustering baselines
3. Competitive performance, comparable to attention-based models

### Weaknesses
Major
1. Fundamental design flaw in dual-geometry integration: in the paper, author concatenates features from Euclidean and hyperbolic spaces directly in Eq. (10) without any alignment mechanism. This is mathematically problematic as these features exist in incompatible metric spaces with different scales and properties. The simple concatenation likely destroys the geometric structure that hyperbolic space is supposed to preserve, undermining the entire theoretical motivation. 
 - scale: Euclidean cosine similarity ranges in [-1, 1] while Lorentz distance has unbounded range. Without proper normalization, one component will dominate the Softmax computation.
 - Typically, when combining features from different manifolds, you need to either: a). Project both to a common space (e.g., tangent space)
 b) Use separate processing streams with late fusion, c) Apply learned projection/alignment layers
 - Cosine similarity (higher=better) vs Lorentz distance (lower=better) both fed to Softmax without explaining sign conversion
2. Why are window-level relationships hierarchical? No explanation or evidence provided
3. No evidence showing what hyperbolic geometry actually learns or why it helps
4. Only one visualization provided, no quantitative interpretability metrics (cluster purity, consistency, semantic alignment)
5. The paper completely omits standard practices for hyperbolic neural networks: No dimension reduction before hyperbolic operations (standard practice to reduce from C to C/r for computational efficiency [Khrulkov et al. 2020]:)  

Minor 
1. Eq. (10) notation error: g^W_L suggests window-Lorentz while g^P_E suggests patch-Euclidean, directly contradicting the text's claim of patch=Euclidean, window=hyperbolic
2. Eq. (2) indexing bug: Uses c'_i but should be c'_j
3. No exploration of cluster numbers, window sizes, or curvature parameter κ.

### Questions
1. How is your method actually implementable? Please provide the exact dimensions at each step: C → ? (reduction?) → hyperbolic (C+1?) → ? (projection?) → concatenation → FC. Without these details, the method cannot be reproduced.
2. What is the true parameter count? If FC reduces (2C+1) → C after concatenation, this adds significant parameters. Why isn't this in Table 1?
3. How do you handle high-dimensional hyperbolic operations? Processing 320-512 dims in hyperbolic space without dimension reduction contradicts standard practices. How is this numerically stable?
4. Why not test Euclidean-Euclidean dual-path? This critical control would reveal whether improvements come from hyperbolic geometry or simply dual-path processing.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces HCFormer, a novel vision backbone architecture that leverages interpretable clustering for token mixing, moving away from the black-box nature of convolution, attention, and MLP-based mixers. One of the main contributions is ClusterMixer, which explicitly mixes tokens via hierarchical clustering. To better capture the tree-like, hierarchical relationships in visual data, clustering is performed in hyperbolic space at the window level, while Euclidean space is used for fine-grained patch-level clustering. This dual-geometry approach enables HCFormer to efficiently and transparently aggregate information across both local and global contexts. The architecture achieves linear computational complexity, strong multi-task flexibility, and built-in interpretability. Extensive experiments show HCFormer consistently outperforms comparable models in image classification, and achieve decent results on semantic segmentation, object detection, and instance segmentation. However more results are comparison with CLIP and multi-object settings will make the paper stronger.

### Strengths
1) HCFormer replaces opaque token mixers (convolution, attention, MLP) with ClusterMixer, which uses explicit clustering algorithms for token mixing. This design provides transparency and interpretability, allowing users to understand how information is aggregated and propagated.
2) The model performs clustering in Euclidean space for local patch-level mixing and in hyperbolic space for global window-level mixing. This approach efficiently captures both fine-grained and abstract hierarchical relationships, improving semantic modeling and reducing distortion in feature aggregation.

3) By restricting clustering to local windows and using hierarchical strategies, HCFormer reduces the quadratic complexity of traditional clustering to linear, enabling efficient processing of high-resolution images and dense prediction tasks. The architecture is flexible and adapts well to various downstream tasks.
4) 
HCFormer demonstrates strong results across multiple benchmarks, such as  ImageNet-1K, ADE20k, COCO etc.

### Weaknesses
1) Comparison to Hyperbolic CLIP[1]; it would be good to have CLIP based baselines in the paper, since it will show general applicability of the method.
2) How to extend this to multi-object settings? For example, When we have more than 1 object, how will this method extend? For example, On OpenImages as discussed in [2].
3) In line 299 it’s said “ Euclidean similarity is computed at the finer-grained and computationally demanding patch-level, while hyperbolic similarity is estimated at the abstract” ; what happens if both are done using hyperbolic loss instead of euclideian loss? It’s a known fact that hyperbolic models are hard to train, it would be good to understand why this particular setup was chosen.
4) On object detection the gains are very incremental, any particular reason why these gains are incremental when linear probing is actually decent?
Refrences
[1] Hyperbolic Image-Text Representations
[2] Hyperbolic Contrastive Learning for Visual Representations beyond Objects

### Questions
I think there are few more experiments that can help the paper specially in CLIP side. And some of the results seem incremental which needs to be properly discussed.

### Soundness
3

### Presentation
3

### Contribution
2
