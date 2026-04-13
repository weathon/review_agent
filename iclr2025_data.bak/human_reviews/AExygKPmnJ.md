## Human Reviewer 1

### Summary
In this paper, the authors present a state-of-the-art model for protein binding site identification based on the addition of virtual nodes to the usual EGNN.
Those virtual nodes serve multiple purposes:
- They alleviate some problems recurrent with GNN like smoothing, vanishing, or exploding gradient.
- They encode in their positions the central position of the identified binding site and in their final hidden representation some overall general information about the protein they were added on (like for example protein family). The final virtual nodes' hidden representation also contains some information (by training) about the confidence in those virtual nodes' ability to recover a binding site.
- In the case of graphs with nodes coordinates, those virtual nodes allow for more expressiveness of the network.

### Strengths
Particularly, and independently of its state of the art, this paper's novelty and interest lays on:
- An extension of EGNN for graphs with virtual nodes as well as a proof of their power for solving the binding identification task, thanks to an ablation study.
- A new 3-step message passing proven to be powerful compared to its homogeneous counterpart via an ablation study.
- A method to ensure at least approximate virtual node initial positioning invariance (data augmentation), and a discussion on potential ways to improve upon it.
Bonuses:
- The model is equipped with a self-confidence module, which is always useful at prediction time in real-case scenarios.
- The paper offers plenty of discussions about VN-EGNN vs EGNN even outside of the particular task of binding site identification.

### Weaknesses
The paper is pretty solid and the few questions that I think would strengthen the paper even more are listed below. They mainly have to do with showcasing more the use of the model at prediction time, clearly demonstrating strength , weakness and protocol associated to using the model. Some part are missing in that area: not sure how easy it would be for someone interested to use this model, to do so. Again mainly at prediction time, because I think the training is well described.

### Questions
Suggestions for improvement (suggestions are minors):
- Discussion about the reliability of the self-confidence score (most important comment for me): what typically is the range it outputs, what would be in practice still considered good confidence, how to work with multiple of those, and so on at prediction time. I am pointing this out because of 2 things. First, we only have metrics for DCC and DCA and never talk again about the confidence score. Second, the training loss consists of matching a known binding site to the nearest virtual node. Is it at training time that the virtual nodes clustering is done? Or is this a way to handle unassigned virtual nodes, or binding sites to virtual node multiplicity, at prediction time? I think a bit more details about how to handle those cases with examples, would be super helpful in the appendix.
- Overall a few sentences about the scaling in memory and computation according to protein size could be interesting, as well as quantitative limits in terms of number of binding sites and protein size. The case of  Holo4K is mentioned as a particularly hard case (even though their model still performs there), and even though the training is done on cases with way fewer binding sites, the model can predict up to 8 independent binding sites. Can we have a breaking of model performance by number of binding sites?
- Finally, do the authors have any ideas why they still needed 5 VN-EGNN  layers despite the huge theoretical and experimental (appendix section on expressiveness table) boost in expressiveness given by the inclusion of virtual nodes (and its decoupling to receptive field)? Here the virtual node addition should in principle give you the optimal receptive fields so adding layers should not matter much, except if in that case the model also needs to understand more geometry via N body correlation which is also offered by stacking 2 body message passing layers. But in that case and given what MACE showed I was more expecting 3 to 4 layers to at least have access to angles information. 5 seems a lot from an outsider's perspective.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
**Summary:** The authors present VN-EGNNs for fast and accurate protein binding site prediction.

**Recommendation:** I am recommending a weak accept at this time.

**Rationale behind Recommendation:** If the authors were to explain their dataset splitting criteria more thoroughly and include additional experiments e.g., with reflection-sensitive scalar node features or e.g., with a virtual nodes variant of another type-1 equivariant graph neural network (e.g., GVPs [1]), I will consider raising my score.

**References:**

[1] Jing, Bowen, et al. "Learning from protein structure with geometric vector perceptrons." The Ninth International Conference on Learning Representations.

### Strengths
- Binding site identification presents a great opportunity to showcase the importance of virtual node learning.
- The computational efficiency of this approach is clear.
- The confidence predictions are a nice addition for this problem.
- VN-EGNN's t-SNE virtual node embedding plots show that the model has learned somewhat informative embeddings for proteins.
- The authors' discussion of achieving approximate invariance to the initial virtual nodes' positions through random augmentations is interesting.

### Weaknesses
- The authors' claim that VN-EGNNs are the first equivariant GNN equipped with virtual nodes does not seem to be accurate. In particular, recent works such as those of [1] seem to have explored this idea across a variety of molecular systems. I still think the authors' contributions for the problem of binding site identification are notable, however, it's worth noting that other works have already begun to explore extending equivariant GNNs with virtual nodes for improved expressivity. Please consider discussing works such as [1] to distinguish the authors' approach to developing an equivariant virtual nodes GNN.
- In Appendix G.1, the authors only describe their redundancy reduction (i.e., clustering) employed for the scPDB dataset. Please also describe how the authors have ensured their training and test splits for the PDBBind, COACH420, and HOLO4K datasets were not overlapping, to ensure fair benchmarking on the respective test splits.
- Based on their benchmarking results, the authors claim that residue-level information suffices for conformational binding site prediction. However, the authors' results still have room for improvement, so does this suggest that some methodological components are still missing? One such idea is that the ESM embeddings the authors employ are not sufficiently sensitive to chirality. Sensitizing the scalar node features to reflections (but not translations or rotations) may be worth exploring to see if this additional inductive bias of structural chirality improves VN-EGNN's results or not. Please see [2] for some ideas on how to make geometric GNNs sensitive to chirality, to consider if this improves VN-EGNN's performance for binding site identification.
- Concerning line 416, having to know the number of true binding sites in a protein to evaluate such each binding site predictor makes the benchmarking results in this work and in previous works less practically relevant. Please discuss the limitations and alternatives of this evaluation approach more carefully for readers.
- Regarding the fifth sentence of the authors' abstract, to increase the accessibility of this work, I'd recommend the authors consider rewriting this to read as something like, "However, the performance of GNNs at binding site identification is still limited potentially due to a lack of expressiveness capable of modeling higher-order geometric entities, such as binding pockets."
- If possible, releasing source code to accompany this model would be beneficial for the research community.

**References:**

[1] Zhang, Yuelin, et al. "Improving Equivariant Graph Neural Networks on Large Geometric Graphs via Virtual Nodes Learning." Forty-first International Conference on Machine Learning.

[2] Morehead, Alex, and Jianlin Cheng. "Geometry-complete perceptron networks for 3d molecular graphs." Bioinformatics 40.2 (2024): btae087.

### Questions
**Questions:**

- On line 290, should this read as "such that all eigenvectors..."?

**Feedback:**

- I think unsupervised geometric learning of other protein-related properties is a promising direction for future work.
- Typo on line 40: should be "a ligand" and not just "ligand".

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper proposes an extended version of EGNN called VN-EGNN by introducing virtual nodes and applying extended message passing scheme. It focuses on protein binding site identification problems and claims achieving SOTA at locating binding site centers on COACH420, HOLO4K and PDBbind2020 datasets.

### Strengths
The problem studied in this paper is very important in the field of protein. The proposed VN-EGNN is very simple, and I believe that this type of method has very good scalability and can be further extended to other equivariant networks.

### Weaknesses
>**W.1 The entry point of the article is rather vague.**

The motivation of this paper is to develop a specialized design, but the methods adopted are more inclined towards general approaches. In the abstract, the authors state, 'The virtual nodes in these graphs are dedicated entities to learn representations of binding sites, which leads to improved predictive performance.' However, in the actual design, it resembles a general method and does not incorporate prior knowledge specific to the mechanism of virtual nodes (only using the loss function).
It is unclear whether the authors intend to emphasize the application value of this work (specialization) or its theoretical value (generality). If it is the former, more prior knowledge should be introduced for specialized design, and the biochemical significance should be analyzed; otherwise, this virtual node approach should be extended to other models, such as SchNet [a], PAINN [b], TFN [c], Allegro [d], and MACE [e].

[a] Schnet – a deep learning architecture for molecules and materials.
[b] Equivariant message passing for the prediction of tensorial properties and molecular spectra.
[c] Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds.
[d] Learning Local Equivariant Representations for Large-Scale Atomistic Dynamics.
[e] Mace: Higher order equivariant message passing neural networks for fast and accurate force fields.

> **W.2 The methodology of the article is inconsistent with that used in the actual experiment.**

The main purpose of this paper is to vacillate between designing a strict equivariant network and allowing the network to be approximately equivariant, which is very fragmented and confusing. In the section "Equivariant initialization of virtual nodes in VN-EGNN.", the virtual nodes are initialized at the center of the whole graph, which is a strictly equivariant operation. In the section "Data augmentation and approximate equivariance", the Fibonacci grid, an unequivariant operation, is used. This makes the paper inconsistent and many chapters lose their meaning, including the proof of equivariance (Appendix E) and the analysis of expressiveness (Appendix K), see comment W.5.

> **W.3 The contribution of the article is not enough.**

Adding global information such as virtual nodes (or even meshes) to the graph is a very obvious idea, and many previous works have studied it, including both non-equivariant and equivariant ones. The most classic non-equivariant work (such as MPNN [f]), and equivariant work includes using priors (such as MEAN [g], AbDiffuser [h]) and not using priors (such as FastEGNN [i], Neural P^3M [j]). The article claims that "To the best of our knowledge VN-EGNN is the first E(3)-equivariant GNN architecture using virtual nodes.", but ignores these pioneering works. Importantly, the core contribution of all the above articles is not virtual nodes, but treat the virtual nodes as only a useful engineering trick. The simple contribution of introducing virtual nodes is not enough to make it accepted (not to mention that such an introduction seems to have some hidden dangers). The author can consider the suggestions in the two directions in W1 to modify the article.
[f] Neural message passing for quantum chemistry.
[g] Conditional Antibody Design as 3D Equivariant Graph Translation.
[h] AbDiffuser: full-atom generation of in-vitro functioning antibodies.
[i] Improving Equivariant Graph Neural Networks on Large Geometric Graphs via Virtual Nodes Learning
[j] Neural P^3M: A Long-Range Interaction Modeling Enhancer for Geometric GNNs

> **W.4 The experiment is not convincing enough.**

(i) The article should compare relevant articles (such as those mentioned in W.3) and more lastest models (e.g. SphereNet [k], ClofNet [l], LEFTNet [m], ViSNet [n], Geoformer [o], SO3krates [p]) as baselines.
(ii) In addition, according to Appendix F, VN-EGNN performs random rotations during training, which is unfair to equivariant networks. This approach does not show that such invariance/equivariance is brought by VN-EGNN itself, but may be due to data augmentation. Instead of using this approach, it is better to discard the restrictions of the architecture and learn the potential equivariance completely through data-driven learning, like AlphaFold3 [q].
(iii) The dataset selected in the article contains substances of various conformations, which is equivalent to data augmentation. It should be ensured that the input conformations of the training set are similar to each other (such as protein dynamics in FastEGNN [i]), and then only the validation set and test set are randomly rotated to verify whether the model is strictly or approximately equivariant.
[k] Spherical Message Passing for 3D Molecular Graphs.
[l] SE(3) Equivariant Graph Neural Networks with Complete Local Frames.
[m] A new perspective on building efficient and expressive 3d equivariant graph neural networks.
[n] Enhancing geometric representations for molecules with equivariant vector-scalar interactive message passing.
[o] Geometric transformer with interatomic positional encoding.
[p] A Euclidean transformer for fast and stable machine learned force fields.
[q] Accurate structure prediction of biomolecular interactions with AlphaFold 3.

> **W.5 Problems caused by non-strict equivariance.**

(i) The equivariant model requires equivariance at each layer. Using the Fibonacci grid method to initialize virtual nodes will make the equivariance of the entire model no longer meaningful.
(ii) VN-EGNN uses non-equivariant initialization, which seems to enhance the expressive power of equivariant neural networks, but actually reduces the robustness of the model. Let's assume that the first virtual node initialized by the Fibonacci grid is $(x,y,z)$ and is not at the origin. The two graphs are $\\{(\pm 1, 0,0)\\}$ and $\\{(0, \pm 1, 0)\\}$. Obviously, the two graphs are geometrically isomorphic, but the virtual nodes introduced by VN-EGNN cannot guarantee the same output.

> **W.6 Relationship between Fibonacci grid and number of virtual nodes.**

Fibonacci grid is a commonly used technique for generating approximate equivariance, and plays an important role in eSCN [r] and EquiformerV2 [s]. However, Fig. 9 in eSCN also points out that to make the equivariance error very low, $18\times 18=324$ samplings may be required, corresponding to the virtual nodes of VN-EGNN. In fact, VN-EGNN only uses 4 or 8 virtual nodes, which makes people very worried about whether it will bring a large equivariance error. And if VN-EGNN really uses 324 virtual nodes, will the overhead become unacceptable and lose good scalability?
[r] Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs.
[s] EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations.

### Questions
See Weakness.

### Soundness
2

### Presentation
1

### Contribution
1

### Rating
3

### Confidence
5

---

## Human Reviewer 4

### Summary
This study improves E(n)-equivariant graph neural networks (EGNNs) framework to predict protein-ligand biding site through two innovations: 1) proposing virtual nodes; 2) applying an extended message passing approach. The performance of the proposed approach has been benchmarked with other baselines on three data sets.

### Strengths
Overall, I think the paper is well-written.
1. Although the topic is classical and the virtual node idea is also not novel, which has been explored in many other fields,  the improvement on the prediction performance using the two proposed ideas (virtual nodes and the improved message passing) are obvious. 
2. The proofs of the properties of the proposed approach are interesting and solid.
3. The research is comprehensive and clear, and the citations of the paper are detailed.

### Weaknesses
1. In the result tables, since you have calculated standard deviations, you can also calculate the p-values to measure whether the performance of your model is significantly different from the baselines in the different datasets.
2. Need some details about the prediction of the data set (HOLO4k) with domain shift. Was the same approach applied? why or why not can the proposed method handle the domain shift issue?


*************
During the discussion among reviewers and area chairs, it became apparent that the performance comparison between the proposed method and the baseline was unfair. The proposed method relies on augmented data, while the baseline does not. As a result, the score was adjusted to 5.

### Questions
1. I am wondering if you can generate a visualization plot to show if the predicted binding site(s) have high weights in your trained model(s), which can make the model more interpretable.

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
5

### Confidence
4

---

## Human Reviewer 5

### Summary
This paper focuses on enhancing binding site identification in proteins using extended E(n)-equivariant graph neural networks (EGNNs) with the introduction of virtual nodes. Traditional EGNNs have struggled with this task due to the absence of dedicated nodes for learning binding site representations and issues related to oversquashing in message passing. The proposed VN-EGNN method aims to address these challenges, demonstrating significant improvements in predictive performance across several benchmark datasets (COACH420, HOLO4K, and PDBbind2020). The paper provided a comprehensive overview of the problem, related work, and the proposed methodology.

### Strengths
1. The introduction of virtual nodes to enhance the learning of binding site representations addresses key limitations in traditional EGNNs, offering a novel method for protein binding site identification.
2. VN-EGNN demonstrates the state-of-the-art performance in binding site identification across multiple benchmark datasets.

### Weaknesses
1. In the ablation experiments, the ablation of virtual nodes should ensure the presence of heterogeneous message passing and the pre-trained protein embedding module, rather than the traditional EGNN.
2. In line 135, the coordinates of the virtual nodes are initialized randomly, and they ultimately converge to the coordinates of the ligands. So why are two initialization strategies for the virtual nodes mentioned in section 2.3? Could you provide more discussion on the initialization of the positions of the virtual nodes?
3. In the design of the loss function, Dice loss was used for binding site identification as a node-level prediction task. How does VN-EGNN perform on this task?

### Questions
see above

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
3