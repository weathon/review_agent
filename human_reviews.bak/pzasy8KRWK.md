# Physically Aligned Hierarchical Mesh-based Network for Dynamic System Simulation

- Decision: Reject
- Scores: 5, 8, 5, 6, 6

## Abstract
Dynamic systems evolve through complex interactions, where local events influence global behaviors, reflecting the interconnected nature of real-world phenomena. Simulating such systems demands models that effectively capture both local and long-range dynamics, while maintaining a balance between accuracy and computational efficiency. However, existing mesh-based Graph Neural Network (GNN) methods often struggle to achieve both high accuracy and efficiency, especially when dealing with large datasets, complex mesh structures, and extensive long-range effects. Inspired by how real-world dynamic systems operate, we present the Mesh-based Multi-Segment Graph Network (MMSGN), a novel framework designed to address these challenges by leveraging a physically aligned hierarchical information exchange mechanism. MMSGN combines micro-level local interactions with macro-level global exchanges, aligning the hierarchical mesh structure with the system’s physical properties to seamlessly capture both local and global dynamics. This approach enables precise modeling of complex behaviors while maintaining computational efficiency. We validate our model on multiple dynamic system datasets and compare it with several state-of-the-art methods. Our results demonstrate that MMSGN delivers superior accuracy and mesh quality, excels in managing long-range effects, and maintains high computational efficiency. Furthermore, MMSGN exhibits strong generalization capabilities, scaling effectively to larger physical domains. These advantages make MMSGN well-suited for simulating complex, large-scale dynamic systems across a variety of scenarios. Codes and data will be made publicly accessible upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper builds upon the EAGLE model and experiments different improved clustering techniques against three datasets for physics simulation on irregular meshes. The results show that some modifications perform better on some task, but not on others.

### Strengths
The paper is mostly well written and easy to follow. The related work section and the references to the state-of-the-art are relevant and in sufficient amount. The authors conducted an important number of ablations and announced public availability of the code and dataset, which is a good initiative, especially since the paper introduces a new benchmark.

### Weaknesses
**Major**
My main concerns are the contributions of the paper with regards to EAGLE and the clarity of these contributions in the paper. The approach is extremely close to the EAGLE model, and while results are indeed better, the better barely explain why, and I struggle to extract general knowledge that would be transferable to future research. To be precise:
- The fact that Multi-level is highly beneficial in models for simulating physics on meshes is well known from previous results (Liu et al, 2021, Janny et al 2023, Cao et al, 2023).
- The "micro-level" structure is a straightforward MGN model, exactly like in EAGLE.
- The "macro-level" structure is an attention over a fully connected mesh on the downscaled graph, exactly like in EAGLE. 
- The merge of micro and macro level information is done by concatenating both representations, like in EAGLE.

Eventually, I noted the following differences with Janny et al. 2023 :
- The clustering algorithm is improved. To me, this is the main contribution of the paper, I will come back to this.
- The position encodings seem improved, but Table 5 indicates that it does not have impact on the performances.
- Cluster representations are aggregated via average pooling, while EAGLE was using a GRU. This seems like a minor modification, and it is not discussed in the paper.
- Finally, the projection of the concatenated representation to the physical space is done with an MLP instead of another layer of GNN. Again, this difference is minor and not discussed in the paper.

Hence the only key contribution is the new clustering algorithm which uses physical prior to downscale the mesh while EAGLE was using a simple position-based clustering. This a valid and interesting way of improving this kind of model. Yet, the paper limits itself to describe when it works on three datasets, and fail (in my opinion) to extract rules, knowledge, insights that would advances the field of research.
- Algorithm 1 seems to be a straightforward K-means algorithm with handcrafted features modeling some aspect of the physical property of the domain.
- The authors tested different features, some of them working on some datasets, some did not without explanations. I could not find any insights on this phenomenon that would explain why and how the contribution actually benefits the learning of the dynamics.
- While the authors claims that the model works because the clustering "ensure that similar physical interactions are handled uniformly", I see no evidence that these clusters make indeed more sense for training an DL based simulator, apart from empirical results on three datasets.

Less importantly, I am not convinced that the community needs another medium-scale dataset of mesh-based physics simulations, and I struggle to see in what DeformingBeam differs from other existing tasks.

**Minor** (little to no effect on my review)
- The author used "dynamic" in several sentences to describe their clustering method. Yet, it is not based on "dynamic" (it is not taking time into account, which is what dynamics is about). I think the words "physical" and "dynamic" have been used interchangeably, but it is confusing for me (some sample lines where dynamics is used arguably misleadingly: line 78/79, line 183, line 216, ...)
- The related work section could benefits from few words about existing datasets for physics on meshes, since DeformingBeam is one of the contributions. In general, while the related work section presents well the state-of-the-art, it could better situate your contribution in the existing literature.
- l.220 "outputs a set OF graph" ?
- l. 806 : "we use the its shortest distance"
- l. 433 : "simulation.(SPACE)More"

**Motivation for my grade**
My rating is based on the lack of significant contributions of the paper. Most of the architecture is based on EAGLE. The only noticeable novelty is the better clustering technique, but the paper only presents experimental results on some datasets and fails to extract general knowledge that would benefit the community. 
I strongly encourage the authors to (1) prove me wrong if the contribution extends beyond the clustering technique and (2) strengthen the paper with a better analysis of the results, including when and why the proposed model performs better.

### Questions
- You tested several segmentation methods (i.e. handcrafted features) in table 3. It is not clear what are the conclusions of this experiment, since some methods work better on some datasets while some do not, and there is no clear reason why. Could you interpret these results ?
- Did you have any evidence that the segmentation is indeed correlated with the dynamics of the system (and not solely its geometrical description) ? 
- Does patterns emerges from the cluster's dynamics that would explain why your method performs well ?
- Did you spotted any behavior of your segmentation method that does not arise from EAGLE's purely geometric clusters that would explain or give an intuition about how the model benefits/uses these clusters ?
- You mention in the introduction that previous work "face drawbacks like manual effort or inaccurate mesh edges". Can you please develop why your method does not suffer from the same issues ?
- While your method is (I think) very close from EAGLE, why not evaluating on the corresponding dataset, which has been designed specifically for very similar models ? It has proven to be sensibly harder than CylinderFlow which exhibits a lot of regularities.
- Where did you sampled the initial condition that you provided to the models ? Is it randomly sampled in a longer simulation or does it corresponds to the very first timestep ? If not, it would be interesting to compare the models in a realistic scenario where the simulation is done from a "cold start".
- I am very confused by the description of the physics-guided segmentation (line 244 to 265). First, you mention that you apply METIS to obtain clusters, followed by SLIC to refine them (to be honest, I'm not familiar with these algorithms, I had to look them up), but it seems that you only used algorithm 1 (which seems to be a K-means with handcrafted features). Am I mistaken ? Is there a difference between your proposed approach and a K-means algorithm ? I think this could be made clearer in the paper.
- Line 225 to 230, you introduce a notation for overlapping clusters. This is not used in the main paper, and the ablation shows that this has negative effect on two datasets and positive on the last one. Why so ? Can we generalize from this experiment to understand which property of the underlying physics could benefits from $\delta>0$ ?

# Rebuttal
The authors provided a thorough and detailed reply to my review, accompanied by a substantial amount of supplementary results and additional analyses. These additions offer greater insight into how the contribution improves upon the state-of-the-art compared to Eagle.

As mentioned in my original review, I believe this is an interesting and valuable line of research. However, the paper contains numerous unsupported claims. For instance, the title refers to "physics-aligned" clustering, but the method does not incorporate any physics priors beyond basic geometric considerations. The rebuttal has addressed some of my concerns, but the main paper remains highly misleading and, in my opinion, poorly structured.

I have raised my score from 3 to 5 but still recommend rejection. The rebuttal lays the foundation for significant improvement, but the main paper should be thoroughly restructured to reflect these changes, particularly regarding the claims and findings.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents the Mesh-based Multi-Segment Graph Network (MMSGN), which is a novel hierarchical framework that aims at simulating dynamic systems with high accuracy and computational efficiency. The proposed model uses a two-level interaction mechanism (micro-level local interactions, macro-level global exchanges). By aligning the mesh structure and the physical properties, MMSGN effectively captures both local and long-range dynamics, overcoming common issues in existing mesh-based Graph Neural Networks (GNNs), such as oversmoothing and excessive computational load. The paper validates MMSGN on multiple datasets, showing that it outperforms several state-of-the-art methods in terms of accuracy, mesh quality, and efficiency, and introduces a new dataset (DeformingBeam) for evaluating mesh-based simulations.

### Strengths
- The combination of micro- and macro-level exchanges is interesting, innovative, and well-motivated. It also aligns with properties of real-world dynamic systems.
- The experiemnts indicate that MMSGN is able to achieve a balance of high prediction accuracy and computational efficiency. 
- The approach has been appropriately validated w.r.t. to different experiments and datasets. 
- The paper is well-written and easy to follow. Sections are meaningfully structured and enough details are provided. 
- The paper introduces a new dataset (DeformingBeam), which provides an additional benchmark for future research.

### Weaknesses
- While the hierarchical approach appears conceptually sound, MMSGN may be challenging to re-implement due to its multi-level structure. 
- The paper briefly mentions comparisons with other state-of-the-art methods but would  benefit from a more in-depth analysis (e.g. additional benchmarks) to further explore the capabilities and limitation sof MMSGN.
- Although the paper highlights scalability, it is unclear whether this extends to systems with highly irregular or extremely large-scale meshes. More details would be appreciated on this. 
- Not enough details are provided on how to incorporate diverse boundary conditions. A more in-depth discussion would be beneficial.

### Questions
- How does MMSGN handle complex boundary conditions? 
- Could it effectively simulate systems where boundaries have non-standard behaviors or physical constraints?
- Has the model been tested on highly irregular or unstructured meshes, and if so, how does it perform in these cases?
- Can the authors provide a more detailed analysis of the computational complexity of MMSGN? 
- Given its design, could MMSGN be applied to domains beyond traditional physics-based simulations (e.g. social network dynamics)?
- The authors may also want to add the following paper to the list of related work: H. Shao, T. Kugelstadt, T. Hädrich, W. Pałubicki, J. Bender, S. Pirk, D. L. Michels, Accurately Solving Rod Dynamics with Graph Learning, Conference on Neural Information Processing Systems (NeurIPS), 2021

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper presents the Mesh-based Multi-Segment Graph Network (MMSGN), a model designed to simulate dynamic systems by combining local and long-range information exchanges within a physically aligned hierarchical mesh structure. By segmenting meshes based on physics-informed features, MMSGN efficiently captures complex dynamics and outperforms baseline methods in both prediction accuracy and mesh quality across several datasets. Through empirical analysis, the model demonstrates robust generalization capabilities and scalability, making it suitable for large-scale, complex simulations​.

### Strengths
1. **Clarity of High-Level Concept**: The core idea—that deformations primarily remain local to the area of contact and propagate slowly across the entire structure so some clustered feature would be helpful—is intuitive and easy to follow. This perspective makes sense in scenarios where local interactions don’t significantly affect distant areas. I found the paper easy to follow. In addition, I think the paper provides good details on the setup of the experiments (including baselines).

2. **Physics-guided Segmentation**: I like the idea of using segmentation/clustering to "reduce feature space" locally, it draws some interesting connections to modern CNN and ViT structure in CV, and reduced-order modelling in engineering. In addition, Positional Encoding (PE) and Segment Encoding (SE) definitions are grounded in physical intuition by constructions.

3. **Performance**: Surprisingly fast compared to well-established baselines like MGN. However, I would like to learn some intuition behind this -- based on the neural network structure, I can't really tell where the performance boost comes from. Is it because the number of message-passing steps between nodes was reduced?

### Weaknesses
1. **Questionable Motivation in Localization**: The emphasis on local deformation may not always hold, especially in fields like computational physics or mechanical engineering, where elasticity often results in global, fast propagating deformation—particularly with low-stiffness materials. This raises concerns that the paper's foundation may not fully align with real-world mechanics.

2. **Limitations in Segmentation Approach**: The paper’s reliance on *purely geometric* (and arguably *topological*) clustering for segmentation may not capture clusters that reflect true mechanical behavior. For instance, with a bird flapping its wings, both wings would oscillate at similar frequencies, we could naturally group them as a single cluster in modal analysis. However, the geometric-only approach used here might yield clusters highly dependent on the initial setup, potentially missing important physical interdependencies. A physics-inspired method, like modal analysis, could provide clusters with greater physical relevance and reduce sensitivity to initialization. This is my major concern with this submission, the current segmentation approach is too *geometric* and already exhibits a **strong** prior on how every deformation must be highly local.

3. **Mesh Resolution Sensitivity and Segment Size Issues**: As number of segments grows, the number of finite elements within them decreases, leading to potential accuracy and performance downgrades. Appendix Table 4 reflects this, with errors increasing as segment size grows in some examples, but the paper lacks clarity on how it determines an optimal cluster count.

### Questions
1. **Mesh Continuity**: While the authors use *Mesh Continuity* as the mesh quality benchmark, this metric may not be standard in all visual computing applications or mechanical engineering without proper citations. I would recommend using aspect ratio, or "as-regular-as-possible" metric to measure the uniformness for every element (triangle/tetrahedra) on the mesh, as it is a more common mesh quality metric in FEM literature. Furthermore, the sole use of Hausdorff distance for geometric fidelity could be limiting; adding Chamfer distance would provide a more balanced measure of mesh accuracy.
2. **Underwhelming PE Impact**: Despite the geometric reasoning behind PE(e.g. understanding relative location between within segments), the benefit is marginal (around a 10% reduction in RMSE on the already low error). Could the authors elaborate on the observed impact of PE?
3. **Selection of Optimal Number of Clusters**: The paper includes an empirical analysis in Appendix Table 4, showing that the number of clusters impacts accuracy (with only two data points along the # of clusters dimension shown in the table). Given the algorithm's similarity to K-means, selecting an optimal number of clusters and their initialization will likely be crucial for accuracy and convergence speed. Could the authors expand on their criteria for selecting the optimal number of clusters, and explain how initialization sensitivity is managed in the clustering process?


**Misc**:
- **Missing Error Metrics**: Figure 3 lacks error colormap ranges, making it difficult to assess error variability.
- **Physics-Inspired Segmentation**: Consider exploring modal analysis in engineering and fracture modes in graphics for relevant physics-driven segmentation methods.


**Rebuttal**:

First off, I deeply apologize for missing the deadline to reply as I’ve been travelling. I have reviewed the revised manuscript and sincerely appreciate the additional statistics provided by the authors. I also agree with Reviewer YvcY that the statistical analysis convincingly demonstrates that segmentation is indeed highly useful. Kudos to the authors for addressing the issues raised in such a short amount of time.

I believe my earlier discussion with the AC is relevant here, so I will share part of it for context.

---

Here, I want to raise my concerns. My expertise lies primarily in FEM for mesh-based methods and neural field/PINN approaches for physics + ML, which informed my review from the perspective of norms in mechanical engineering and computer graphics.

While the authors provided additional statistics in their rebuttal, which I found convincing, they failed to adequately address or reinforce the concerns I raised:

1. **Literature review**: The paper does not sufficiently engage with physics-inspired methods, particularly modal analysis, which has a long history in computational physics and mechanical engineering and has seen renewed interest in graphics (e.g., [[Benchekroun et al. 2023]](https://www.dgp.toronto.edu/projects/fast_complementary_dynamics_site/) and [[Sellan et al. 2022]](https://www.dgp.toronto.edu/projects/breaking-good/), and I am not suggesting to cite those papers, just want to highlight what **"physics-informed features from geometry ONLY"** *should* be claimed). Using a graph-based clustering algorithm for a physics problem with no physical grounding feels underwhelming and disconnected from established practices.

2. **Clustering methodology**: The rebuttal essentially agrees that there is no physically motivated or elegant method to determine the optimal number of clusters for initial clustering [(reference)](https://openreview.net/forum?id=pzasy8KRWK&noteId=FGgmSZdP0U). This makes it hard for me to imagine this method being applied to any serious physics/engineering problems.

Although the paper incorporates some good physics intuition, I struggle to view it as motivated by a desire to develop a true physics solver. Instead, it appears to explore clustering on graphs—a valid direction for follow-up work, but not convincingly framed here as a physics-based contribution.

---
I also found the exchange between the authors and Reviewer YvcY regarding whether the segmentation is "physically-informed" highly interesting [(the exchange can be found here).](https://openreview.net/forum?id=pzasy8KRWK&noteId=UlcjK7vO4H). However, while the geometric segmentation may have been effective for a *specific* simulation setup, this is a *correlation*, not *causation*. Furthermore, I question whether the method would generalize to the same geometric setup under different boundary conditions or initial conditions (again, sorry for the late reply, it is no longer possible for authors to provide more updates on this).

To avoid confusion, I strongly recommend refraining from describing the segmentation approach as "physical." While this may seem like a matter of semantics, precise language is essential in technical writing to ensure clarity and avoid misrepresentation.

Given these considerations, I will not change my rating for the paper. That said, I greatly appreciate the professional conduct of the authors and the significant improvements made to the paper during the rebuttal process.

### Soundness
2

### Presentation
3

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
The paper presents the Mesh-based Multi-Segment Graph Network (MMSGN), a hierarchical model for simulating dynamic systems by capturing both local and global interactions in mesh-based structures. It addresses the challenge of balancing accuracy and computational efficiency, which is often an issue in existing Graph Neural Network (GNN) models used for dynamic simulations. The MMSGN framework utilizes a physics-guided hierarchical information exchange, merging micro-level node interactions and macro-level segment exchanges aligned with physical properties through a transformer. This approach allows for scalable, accurate modeling of complex behaviors and long-range dependencies, validated across several datasets, including a new DeformingBeam dataset designed to test long-range interactions and scalability.

### Strengths
+ The dual-level approach combining micro-level and macro-level interactions enables MMSGN to accurately model both short- and long-range effects, which improves accuracy across datasets.
+ The method can be used for both Lagrangian and Eulerian systems.
+ Segmenting the mesh based on physical properties through a transformer ensures that nodes within each segment exhibit similar behaviors, leading to better model convergence and reduced boundary discontinuities.
+ The model generalizes well to larger and denser mesh configurations, as demonstrated on the DeformingBeam large-scale dataset.
+ MMSGN preserves mesh fidelity and continuity better than competing models

### Weaknesses
- The model does not enforce strict physical constraints at contact points, potentially causing overlapping meshes in certain configurations.
- Achieving optimal performance requires careful tuning of segmentation and message-passing parameters, which may limit ease of application in new scenarios.
- The segmentation-based approach may lead to minor inconsistencies at boundaries, especially in cases involving diverse material properties.

### Questions
1. The paper does not discuss run time for the different configurations. It would be good to include some measure of the computational performance.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors present a Mesh-based Multi-Segment Graph Network (MMSGN) aiming to address the challenges in learning dynamic system, such as achieving high accuracy and efficiency in scenarios with complex mesh structures, and extensive long-range effects. The proposed method integrates micro-level local interactions with macro-level global exchanges, aligning the hierarchical mesh structure to reflect the system’s physical properties, enabling it to effectively capture both local and global dynamics. The method is validated  on several dynamic datasets and benchmarks with existing methods. The experiments show that the MMSGN demonstrates advantages on accuracy, mesh quality and managing long-range effects on test cases. The MMSGN also shows generalization ability, which can be applied to larger physical systems.

### Strengths
The paper is well written and organized. The idea of exchanging between micro-level and macro-level is intuitive. Inspiring by domain decomposition in physical simulation, the physics-guided mesh segmentation use not only the spatial distance but also the similarity between features for mesh nodes clustering, which looks sound. Experiments are solid and there are extensive comparisons to benchmarks. The proposed method is demonstrated to be effective on metrics including prediction error and mesh quality comparing to the benchmarks.

### Weaknesses
I didn't see obvious weaknesses for this paper. The authors claim that the proposed method can be applied on large cases. However in Table 2 in appendix, the largest scenario DeformationBeam (large) contains 4540 nodes, which is relatively not large comparing to real world cases (usually more than 10k or even 1M). It is unclear how the proposed method perform on these large scale scenarios.

### Questions
- In Mesh Segment Feature Dispatch part, how does the extracted micro-level information from MeshGraphNet integrates  with the macro-level features extracted from the mesh segments ?
- In Line 219 to Line 220, it mentions that "prior information I" includes boundary conditions, material properties etc. How the "prior information I" mentioned in line 249 calculated? How does these information utilized in the Physics-guided Segmentation part? It seems that there are only distance information(distance to obstacle nodes and distance to boundary nodes) included as described.
- In Line 348, it mentions that "predicted mesh in Lagrangian systems", just wondering what the predicted mesh indicates? Does it mean the method also output a mesh in addition to the current states on each nodes of a mesh?
- Does the segmentation performed every simulation step or just performed once? If is is only performed once, does the mesh segmentation still effective during the simulation especially for problems potential with large mesh deformation?
- How does the number of segments influence the performance of the proposed method?
- How does the boundary conditions imposed in this pipeline?
- How long is the training time and how much memories required for training of this method? 
- How will the method perform on a case which is more similar to a real world case (such as 1M)?

### Soundness
3

### Presentation
3

### Contribution
3
