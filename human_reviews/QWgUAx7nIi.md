# Contrastive Graph Autoencoder for Geometric Polygon Retrieval from Building Datasets

- Decision: Reject
- Scores: 3, 8, 5, 5

## Abstract
Retrieval of polygon geometries with similar shapes from maps is a challenging geographic information task. Existing approaches can not process geometry polygons with complex shapes, (multiple) holes and are sensitive to geometric transformations (e.g., rotation and reflection). We propose Contrastive Graph Autoencoder (CGAE), a robust and effective graph representation autoencoder for extracting polygon geometries of similar shapes from real-world building maps based on template queries. By leveraging graph message-passing layer, graph feature augmentation and contrastive learning, the proposed CGAE reconstructs graph features of w.r.t input graph representations of polygons. The CGAE outperforms existing graph-based autoencoder on multiple polygon datasets in the task of similar shape retrieval of polygons. Experimentally, we show that our approach is capable of identifying and extracting similar complex polygon geometries with or without holes from polygonal datasets, based on template queries. Further experiments demonstrate that our approach is highly robust to geometrical transformations in contrast to existing GAE model, indicating the strong generalizability and versatility of CGAE in identifying complex real-world building footprints.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a method to embed a polygon, which can be applied to polygon retrieval. The core of the method is to train an auto-encoder together with a contrastive learning loss. Since both reconstruction loss and contrastive learning loss doesn’t require human labels, their method can be unsupervised and scalable. The paper validates their method on multiple datasets including both synthetic and real-world datasets.

### Strengths
The proposed method doesn’t require any human labels and thus is scalable, particularly useful to the applications in the GIS domain where unlabelled data are sufficient due to crowd-source platforms like OpenOSM. 

The proposed method also shows very promising results in the real world dataset.

### Weaknesses
The paper has several weaknesses which I’ll detail below: 

The paper has a very limited comparison with baselines. The results only show the comparison between the different variants of the proposed methods, despite the fact that there are multiple existing works that tried to solve this problem like Yan et al as mentioned in the paper. 

From my viewpoint, the impact of this paper on future research is quite limited. The proposed method is not new -- it's known that t contrastive learning together with autoencoder is better than one of them only, which limits its inspiration to future works. Also, the proposed method is not fully explored. For example, the most recent polygonal learning [1] lists two principled ways to encode the polygon geometry. The paper should also discuss, probably also try the better backbone and provide some insights. 

Some important implementation details are missing, especially for the dataset preprocessing. 
The reason why I think it’s important is because preprocessing might be able to help representation learning. For example, if the dataset is pre-aligned (rotation, scale, or translation), the representation learning would be much easier. Basically, different preprocessing might lead to very different performance. Thus, the paper should stress it in a very clear way. 

Some claims of the paper are not fully validated. The paper claims that the proposed method is robust to the node count, rotations, etc. However, I’m not sure how the method achieves this. The paper doesn’t validate it using experiments or proof. 

In short, I think the paper still requires a bit of work to polish before it is ready to be published. Although the presented method is effective in some datasets, the comparison is limited. Because of this, I’m not fully convinced that the proposed method is state-of-the-art. Also, the paper has other limitations that I detailed above.

But I’m glad to hear back from authors during rebuttal in case I misunderstand anything. 

[1] Mai et al. Towards General-Purpose Representation Learning of Polygonal Geometries.

### Questions
Please address the questions and concerns that I have above.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper considers a geographic retrieval problem, querying a "map" with polygons to search for similar shapes. The paper considers the state of the art and prior work using graphical autoencoders (GAEs).  They seek to provide an improved solution that is robust to holes and translations of the polygons, problems that have affected prior solutions. They pick up a method that came from point cloud application, and incorporate contrastive learning to gain robustness. The writing is clear and the contribution and novelty are explained. The paper presents experiments and compares to the prior GAE solution as a baseline.

### Strengths
The work builds logically on the prior art and is well motivated to add robustness.  The contrastive learning framework allows the authors to introduce controlled deficits, adding or removing nodes and/or edges.   

The authors work on known and available data sets and publish their code.  

The comparison with the GAE as baseline seems fair.  The method is a logical extension and handles cases that the baseline does not.

### Weaknesses
The paper would benefit from an algorithm statement (say, in a Figure, or a flow diagram with equation references).  This should include a listing of all the learnable parameters, and their dimensions.  For example, after eqn (3)  and eqn (4) there are MLPs, but not clear where these are specified. 

It isn’t clear if other (not necessarily ML-based) methods using geometry would work for these problems. The ideas of deformable templates and fitting are old, e.g., in the medical imaging literature.  However, the reviewer is not expert on these older methods. 

The proposed method handles new cases, which is an important generalization, and shows better Hausdorf metrics for retrieving similar shapes in data.  However, it isn’t clear if this is now a “solved” problem or if more work is needed to generalize.

### Questions
Please say more about how eqn (6) is interpreted as a probability (or an estimate of a probability).

Did you consider different augmentation ratios r other than 20%?

From a practical perspective, are the results good enough to develop a tool that is broadly applicable?  What challenges remain?

Can you comment on the relation to this work: “Graph Contrastive Learning with Implicit Augmentations”, Arxiv, Nov 2022, and graph augmentation in general?

After eqn (9), should the losses all be weighted equally?  Should this be tuned for the application?

What are the tradeoffs with the k-NN, and choosing k?  How does this potentially help with a search, and does growing k imply more "incorrect" or bad cases to be reported?

Section 2.1: This seems intuitive but perhaps good to exactly define “linear ring”.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The study presents a novel approach, the Contrastive Graph Autoencoder (CGAE), to effectively retrieve polygons with similar shapes from geographic maps, a task that has been challenging due to the complexity of polygon geometries and their susceptibility to transformations like rotation and reflection.

CGAE, utilizing advanced graph message-passing, feature augmentation, and contrastive learning, excels in encoding distinctive latent representations of polygon shapes, facilitating more accurate geometry retrieval. CGAE outperforms traditional graph-based autoencoders (GAEs) through experiments with real-world building map datasets.

### Strengths
1. In contrast to traditional models and state-of-art learning-based graph autoencoders, CGAE is independent of polygonal vertex counts
2. CGAE is capable of retrieving polygons with or without holes
3. CGAE is robust to polygon reflections and rotations
4. CGAE can effectively generalize to large polygon datasets

### Weaknesses
Lack of important experimental results that support the conclusions. See questions for details.

### Questions
1. The author claimed the proposed method could retrieve polygons with holes better than existing methods, but did not provide quantitative metrics like persistent diagram whose 0-th dimension topological feature represents the number of component and 1-th dimension topological feature represents number of holes. 

2. The author claimed the proposed method is free of polygonal vertex counts which is a major benefit over the traditional methods and other graph autoencoders, but I cannot find any experiment results in the paper that can support such claim. Basically CGAE is close to GAE in terms of architecture so why CGAE can scale significantly better than GAE? I think the author should provide more evidences and arguments to this point since it is a critical contribution claimed by the author.

3. Why is CGAE robust to polygon rotations and reflections given that the proposed contrastive loss mainly focuses on local perturbation?

4. Can you clarify this sentence

` CGAE generalizes effectively the geometric information learned from simple polygons to complex shapes, demonstrating a desirable model property, i.e., decoupling of shape detail (i.e., polygon vertex count) from classification accuracy.`

I'm a bit confused about what classification means here.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a way to encode building footprint polygons using a message-passing graph encoder trained with node, edge and contrastive losses. The method improves upon the cited prior work (Yan et al.) by handling polygons with holes, being more discriminative because of the additional losses, and being more robust to noise via additional augmentations.

### Strengths
The method is well-argued and technically sound, the technical contributions over the cited prior work (Yan et al.) are reasonable, and the results are good.

### Weaknesses
I am a little unsure about the magnitude of the contribution here. By and large, it rests on three features:

1. A new (compared to Yan et al., but not new overall) GNN backbone that can accommodate graphs with multiple connected components
2. An edge-preservation loss.
3. A contrastive loss.

These are perfectly reasonable and I have no specific criticisms of these choices. But I am not sure there is any insight that is specific to _polygons_, and as a result the method ends up as a way to retrieve arbitrary graphs (I think) and hence ends up in a much larger solution space of prior work on retrieval from collections of graphs, e.g.

Li et al., "Graph Matching Networks for Learning the Similarity of Graph Structured Objects", ICML 2019

or (for 3D layouts) Li et al., "GRASS: Generative Recursive Autoencoders for Shape Structures", SIGGRAPH 2017

Minor:
- Several parts of the text have "massage-passing" instead of "message-passing"

### Questions
Apropos the comments above: what in the method is specific to polygons? Would this work for arbitrary graphs? Could improvements be made by considering the graphs are specifically non-intersecting polygon boundary loops?

The authors evaluate on the Glyph benchmark but only compare to ablated versions corresponding to a baseline GAE method obtained by ablation (with different backbones). There must be other relevant baseline methods for graph/polygon/glyph/sketch retrieval?

Is there a specific need, in the studied domain of building footprints, to consider the precise topology of the input polygons? E.g. if a straight boundary segment is subdivided into several smaller segments by inserting vertices, the footprint is geometrically the same. But the descriptor will presumably change. Is this desired behavior or not? I understand the authors do additional augmentation to be robust to geometric and topological noise, but this sort of variation appears to be beyond the scope of that augmentation.

In this context, why not just encode the raster footprints instead of the non-regular vector representations? Building footprints are surely simple enough that the advantages of vector representations for complex geometry don't really apply.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
2 fair
