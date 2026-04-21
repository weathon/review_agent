# Ligand Conformation Generation: from singleton to pairwise

- Avg Score: 3.00
- Decision: Reject
- Scores: 3, 3, 3

## Abstract
Drug discovery is a time-consuming process, primarily due to the vast number of molecular structures that need to be explored. One of the challenges in drug design involves generating rational ligand conformations. For this task, most previous approaches fall into the singleton category, which solely rely on ligand molecular information to generate ligand conformations. In this work, we contend that the ligand-target interactions are also very important in providing crucial semantics for ligand generation. To address this, we introduce PsiDiff, a comprehensive diffusion model that incorporates target and ligand interactions, as well as ligand chemical properties. By transitioning from singleton to pairwise modeling, PsiDiff offers a more holistic approach. One challenge of the pairwise design is that the ligand-target binding site is not available in most cases and thus hinders the accurate message-passing between the ligand and target. To overcome this challenge, we employ graph prompt learning to bridge the gap between ligand and target graphs. The graph prompt learning of the insert patterns enables us to learn the hidden pairwise interaction at each diffusion step. Upon this, our model leverages the Target-Ligand Pairwise Graph Encoder (TLPE) and captures ligand prompt entity fusion and complex information. Experimental results demonstrate significant improvements in ligand conformation generation, with a remarkable 18\% enhancement in Aligned RMSD compared to the baseline approach.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work present PsiDiff, a new method for molecule conformation generation by given target proteins. The authors propose to use TLPE and graph prompts to model the ligand-target interactions into the generation task.

### Strengths
- Conformation generation is an important task for drug discovery
- The authors give details theory study to ensure invariance

### Weaknesses
- Novelty: the target information has already been considered in methods like TargetDiff [1].

- Presentation: the explanation about how the "prompt graph" is build and used is not clear and hard to follow.

Ref:

[1] 3D EQUIVARIANT DIFFUSION FOR TARGET-AWARE MOLECULE GENERATION AND AFFINITY PREDICTION, ICLR 2023.

### Questions
- The concept of "Graph Prompt" is confusing. How is this related to the "prompt" in NLP?
- Is the "graph prompt" and "prompt graph" the same thing?
- It is not clear how the "prompt graph" is build. For nodes, while "The number of tokens equals the number of down-sampled target graph nodes"(line 199), how about their values? Meanwhile, how the edge set S are constructed?
- In line 212, it says $Z = Concat(F_L, P)$. How they can be concated together as they may have difference number of nodes and feature dims? It will be better to show the shape of all tensors.
- There are two insertion patterns. And seems they are both used in the method. Why they should be used at the same time?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a ligand conformation generation model that takes into account both ligand features and features of the target protein. They infuse the ligand's embedding with the embedding of the target protein using a seemingly novel method termed "Target-Ligand Pairwise Graph Encoder". They claim to outperform GeoDiff and TankBind in aligned RMSD to crystal ligand poses in the PDBBind2020 data.

### Strengths
1. The paper proposes a seemingly novel way of infusing protein embeddings into ligand embeddings for the purpose of ligand conformation generation.
2. The paper claims to outperform GeoDiff and TankBind in aligned RMSD to crystal ligand structures in PDBBind2020.

### Weaknesses
Related Work:
1. The paper criticizes models for predicting only a single binding pose but seems to ignore DiffDock's multiconformational capabilities.
2. The paper questions RDKit initialization in DiffDock without explaining why it is problematic.
3. The last sentence regarding the use of target information for molecular generation is unclear.
4. The related work section is not sufficiently detailed, making it difficult to understand the paper's unique contributions and how it stands apart from previous works.

Results:
1. The metric is the aligned RMSD to crystal ligand structures in PDBBind2020, where the structures correspond to the ligand bound to the target protein. This is essentially the same as blind docking, thus the results are not convincing without a thorough comparison to DiffDock (current SOTA in blind docking).

### Questions
1. The paper seems to suggest that other embeddings of the protein could be used to condition the molecular generation model (i.e. other than dMaSIF). What other embeddings could be considered? And, why was dMaSIF chosen?
2. How is the evaluation metric in this paper different from that of DiffDock?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose PsiDiff, a conditional diffusion-based model for ligand conformation generation, introducing a novel pairwise approach that incorporates ligand-target interactions and chemical properties. PsiDiff ensures rot-translational equivariance and employs a unique graph encoder, the Target-Ligand Pairwise Graph Encoder (TLPE), to implicitly extract ligand-target interactions throughout the diffusion process.

### Strengths
1. PsiDiff exhibits a sophisticated approach in embedding chemical properties and information within the diffusion model.
2. The methodology employed by PsiDiff in constructing graph prompt tokens, along with the strategic insertion into the ligand graph using two distinct insertion patterns, is noteworthy.

### Weaknesses
1. Problem in contribution and novelty: The authors assert that existing methods in molecular conformation generation have tended to neglect vital pocket-ligand interaction information, positioning their work on transitioning from singleton to pairwise modeling as a key innovation. However, this claim warrants a critical examination. The task undertaken in this paper bears a strong resemblance to docking, a field in which the incorporation of pocket information is a fundamental aspect. Given this context, the purported novelty of integrating ligand-pocket interactions in PsiDiff appears less distinctive, as it aligns closely with established practices in other machine learning based docking methodologies.
2. The data presented in Tables 1 and 5 highlight a pronounced enhancement in PsiDiff’s performance subsequent to force field optimization. In its absence, however, PsiDiff does not exhibit competitive performance levels, particularly in docking tasks (as shown in Table 5 for the 25th percentile), lagging substantially behind methodologies such as GNINA, GLIDE, and EquiBind/TANKBind. To provide a comprehensive evaluation and fair comparison, it would be advantageous to present results for other baseline methodologies after undergoing force field optimization.
3.  Some other recent competitive machine learning methods should be added as baselines,  like UniMol(https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/6402990d37e01856dc1d1581/original/uni-mol-a-universal-3d-molecular-representation-learning-framework.pdf),  Torsional Diffusion(https://arxiv.org/pdf/2206.01729.pdf), and DiffDock(https://arxiv.org/pdf/2210.01776.pdf), which gives much better docking performance compare to TANKBind as shown in https://arxiv.org/pdf/2302.07134.pdf.

### Questions
please refer to the weakness part.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
