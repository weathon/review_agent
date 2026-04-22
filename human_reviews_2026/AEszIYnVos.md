# NECromancer: Breathing Life into Skeletons via BVH Animation

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 2, 6

## Abstract
Motion tokenization is fundamental to the development of generalizable motion models, yet existing approaches remain restricted to species-specific skeletons, such as humans, thereby limiting their applicability across diverse morphologies. We present NECromancer (NEC), a universal motion tokenizer designed to operate on arbitrary BVH skeletons. NEC is built upon three core components: (1) an
Ontology-aWare Skeletal Graph EncOder (OwO), which leverages graph neural networks to encode structural priors extracted from BVH files—including joint-name semantics, rest-pose offsets, and skeletal topology—into robust skeletal embeddings; (2) a Topology-Agnostic Tokenizer (TAT), which compresses motion sequences into a universal, topology–invariant latent representation, thereby decoupling motion dynamics from morphology; and (3) the Unified BVH Universe (UvU), a large-scale dataset that consolidates BVH motions across heterogeneous skeletons (humans, quadrupeds, and other species), enabling systematic training and evaluation under diverse morphologies. Experimental results demonstrate that NEC achieves high-fidelity motion reconstruction with substantial compression, while effectively disentangling motion from skeletal structure. This capability supports a broad range of downstream tasks, including cross-species motion transfer, motion composition, denoising, generation (plug-and-play with any token-based generator; e.g., MoMask) and motion–text retrieval (via an OwO-based CLIP variant). By grounding motion representation in BVH animation while removing species-specific constraints, NEC establishes a principled framework for universal motion analysis and synthesis across varied morphologies.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes a novel tokenizer named NECromancer (NEC) to represent the motions of arbitrary BVH skeletons. The tokenizer consists of two key components: Ontology-aWare Skeletal Graph EncOder (OwO) and a Topology-Agnostic Tokenizer (TAT). This work also contributes a new dataset named Unified BVH Universe (UvU). In the experiments, it shows that the proposed NEC outperforms standard VQVAE (Van Den Oord et al. (2017)) and RVQVAE (Guo et al. (2023)) tokenizers.

### Strengths
1. This work proposes a novel motion tokenizer and shows that it performs better than existing standard tokenizers. 

2. This work contributes a large-scale BVH benchmark for heterogeneous species and various skeletal topologies.

### Weaknesses
1. The dataset contribution is somewhat unclear. 
- The details of dataset construction are largely ignored. It may be partly because the section 3.2 is too short to describe how difficult the dataset construction is. 
- The current description gives the impression that the dataset is simple combination of existing three dataset with some transformations. 
- Also, the use of Truebones Zoo and text annotation is done in the following paper in a more thorough way. 
- W. Lee et al., How to Move Your Dragon: Text-to-Motion Synthesis for Large-Vocabulary Objects, ICML 2025.

2. Fig. 2 is hard to see due to too small text with too light color. 

3. The empirical evaluation is limited in that the compared baselines are only two - VQVAE and RVQVAE.
- For example, the following state-of-the-art baselines could be compared. 
- B. Jiang et al., Causal Motion Tokenizer for Streaming Motion Generation, ICCV 22025. 
- J. Zhang et al., Generating Human Motion From Textual Descriptions With Discrete Representations, CVPR 2023.
- C. Guo et al., TM2T: Stochastic and Tokenized Modeling for the Reciprocal Generation of 3D Human Motions and Texts, ECCV 2022. 

4. In the same vein, the qualitative results are highly limited. 
- No comparison with state-of-the-art models on reconstruction are compared in each dataset.
- Only a small Table (Table 1) is almost all of empirical evaluation of this work, as Table 2 is a rather straightforward ablation study on the proposed method. 

5. Qualitative results are somewhat pointless, as the key message of Fig.3-4 are unclear. 
- Generally, the figures are too small to recognize fine details of comparison. 
- In Fig.3, the NEC results are quite different with GT. With no baseline results, it is hard to know who much the NEC is good. 
- In Fig.4, the success of motion transfer is hard to be convinced.  
- Also, they could be cherry-picked.

### Questions
Please refer to the Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper aims to address the problem that existing motion generation models are limited to species-specific skeletons (e.g., humans). To this end, the authors propose a universal motion representation framework named NECROMANCER (NEC). The framework consists of three main contributions: 1) an Ontology-aWare Skeletal Graph EncOder (OwO) to extract skeleton embeddings containing topological and semantic information from BVH files; 2) a Topology-Agnostic Tokenizer (TAT) that compresses motion sequences of arbitrary skeletons into morphology-agnostic discrete tokens; and 3) a large-scale, multi-species BVH motion dataset named UvU for training and evaluation. Experimental results demonstrate that the framework can achieve high-quality motion reconstruction and cross-species motion transfer.

### Strengths
1.  **Addresses a significant problem**: This paper directly confronts a core limitation in the field of motion generation—the model's dependency on specific skeleton topologies. The proposed universal tokenizer, capable of handling arbitrary BVH skeletons, greatly expands the applicability of motion models and holds significant research and practical value.

2.  **Systematic contribution**: The contribution is comprehensive and solid. The authors not only propose a new model (NEC) but also build a new, large-scale, and diverse-species dataset (UvU) for it. The dataset itself is a valuable contribution to the community and can facilitate future research in universal motion modeling.

3.  **Solid experimental validation**: The paper thoroughly validates the effectiveness of its method through experiments across multiple tasks (reconstruction, retrieval, motion transfer). The comparison against baselines clearly shows the advantages of NEC in handling heterogeneous skeletons, with particularly impressive performance on non-human skeletons.

### Weaknesses
1.  **Strong dependency on data quality**: The `OwO` encoder relies on extracting semantic features from joint names. This means the model's performance is likely highly dependent on the standardization and consistency of joint naming within the BVH dataset. For data from the wild with messy or non-semantic names, the model's generalization capability might be compromised.

### Questions
1.  Regarding the `OwO` encoder, to what extent does it rely on canonical joint naming? If the input BVH files use non-semantic names (e.g., 'joint_1', 'bone_23'), how much would the performance degrade? Have any robustness tests been conducted in this regard?

2.  Could the authors further clarify the core novelty of the spatio-temporal module in the Topology-Agnostic Tokenizer (TAT)? Compared to existing spatio-temporal Transformers in motion modeling, what are its key differences and advantages?

3.  In the qualitative demonstrations of cross-species motion transfer, how does the model handle transfers that might be semantically or physically implausible (e.g., transferring a human dance motion to a fish)? Is there a mechanism to evaluate or ensure the plausibility of the transfer?

### Soundness
3

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
5

### Summary
The paper proposes a method to learn skeletal motion tokens by directly consuming the information stored in the BVH format. The ontology-aware skeletal graph encoder is responsible for encoding the skeletal structure. Specifically, each skeletal node (joint) feature is described with the CLIP embedding of its name projected through a fully-connected layer. The edges are described as the concatenation of the connected node features and their offsets encoded with the sinusoidal embedding, projected through a fully-connected layer. A graph attention layer computes the graph joint feature as a weighted sum of the edge-conditioned messages and the attention weight. The skeletal graph encoder is trained with a geometric loss over joint offsets, a topological loss with the least common ancestor prediction, and a semantic loss to encourage node features with semantic consistency. A topology-agnostic tokenizer is responsible for encoding the motion with a repeated sequence of a spatial block followed by a temporal block. The source motions are represented by per-joint translation and rotation (6D representation), which is projected through an MLP and then fused with the graph joint feature. This feature, with a virtual joint feature concatenated, is passed to a spatial block composed of a multi-head attention transformer modeling correlations between joints, then to a temporal block with a 1D convolution and a 1D ResNet. The virtual joint part is discretized with an RVQ to encode the motion token. The decoder is the reverse of the topology-agnostic tokenizer, with the graph joints feature of the target skeleton injected into the non-virtual joint features. The setup is trained with a heterogeneous dataset composed of HumanML3D, Objaverse-XL, and Truebone-Zoo with data filtering and augmentation applied. The paper demonstrates motion reconstruction and motion transfer with the learned features.

### Strengths
* The goal to unify the motion tokenization is ambitious.
* The combined dataset with the curation strategy could help the community

### Weaknesses
* Hard to interpret the provided videos
  * What do the "transfer" examples suppose to mean? All of them have the prefix "gt" (ground truth?). Which ones are the source motions, which ones are the transferred motions?
  * I see only quadrupeds in the transfer folder. Any non-quadruped transfer examples? Humanoid-to-quadruped or quadruped-to-humanoid?
* Questionable generalizability to different skeletal morphologies
  * For example, it looks like the joint semantic loss is taken over the same joint indices. This does not make sense as the joint ordering is arbitrary (e.g., children can be in any order), and there can be many intermediate joints (rigs can have a different number of spine joints and neck joints)
  * The semantic understanding of the joints must be paired with the spatial relations. As far as I can see, there is nothing in the model to learn this
  * As far as I see, there is nothing in the model and the training strategy to encourage the mapping of the same motion applied to different skeletal features. How would it know to move a fox with the motion token from a humanoid?
* Not enough ablations on architectural design choices, especially on the effectiveness of RVQ
* The paper can trim the UvU section. For example, the main method does not care about skinning other than for the visualization. Skinning is important, but since this does not matter for the main paper, why not move UvU to the appendix and add more details on the main architecture?
* (minor) BVH is just a file format. In theory, there is nothing in the method that hard-couples it with the BVH format. It could be any other 3D formats, such as FBX, USD, and glTF. In fact, BVH is not a suggested format for general rig assets. I would eliminate "BVH" from the paper title and most of the main text to minimize confusion. This also enhances the paper's general applicability

### Questions
Please answer questions in Weaknesses, mainly on the generalizability to different skeletons.

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
3

### Summary
This work aims to introduce a universal motion tokenizer that can encode and reconstruct motions from arbitrary skeletons formats. This allows unified modeling and transfer of motions across a wide variety of skeletons, in contrast with current methods which are typically limited to skeletons from the same family. Data is gathered from 3 sources: 1) human meshes from HumanML3D, 2) meshes from ObjaVerse, and 3) meshes from Trubones-Zoo. All meshes are put in a unified BVH format. The tokenization consists of two models: a graph encoder that provides a distinguishable encoding to each joint given the skeleton rest pose, and a unified tokenizer that can use the graph embeddings and BVH motion sequences to provide a unified tokenization of different types of skeletons. Experiments show that the proposed tokenizer is provides superior motion reconstruction relative to the naive approach that simply pads joints for skeletons of different lengths. Proof-of-concept experiments for motion transfer between different skeletons and unified skeleton-agnostic motion generation are provided.

### Strengths
* This work tackles an interesting and important problem. Unified skeleton representation has the potential to greatly expand the scope of motion generation models and break free of limitations imposed by scarcity of 4-D data for a variety of skeleton types.
* The proposed method of learning a unified token representation of motions is a reasonable and potentially useful direction to solve this problem.
* Experimental results show improved reconstruction over a naive padding-based approach for skeleton reconstruction.
* Videos in the supplementary material provide evidence that the proposed method can lead to natural reconstruction, transfer, and generation of motions across skeleton types.

### Weaknesses
* The motion transfer and generation directions are not explored very thoroughly, and only a few examples are provided.
* The details of tokenization in the paper are somewhat difficult to follow, especially the relation between the graph embedder and the tokenization model. Perhaps it would help to move Figure 3 and 4 into the supplementary material and provide more mathematical details of training these models in the main text.
* The training of the graph embedder is based on heuristic objectives and it is not clear whether the training method proposed is the optimal one.

### Questions
* Can you provide more examples of motion transfer and unified generation?
* Is it possible to ablate the importance of the graph embedder to demonstrate its importance within the tokenization model?

### Soundness
3

### Presentation
2

### Contribution
3
