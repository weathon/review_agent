# Rigidity-Aware Geometric Pretraining for Protein Design and Conformational Ensembles

- Avg Score: 3.00
- Decision: Accept (Poster)
- Scores: 6, 2, 2, 2

## Abstract
Generative models have recently advanced $\textit{de novo}$ protein design by learning the statistical regularities of natural structures. However, current approaches face three key limitations: (1) Existing methods cannot jointly learn protein geometry and design tasks, where pretraining can be a solution; (2) Current pretraining methods mostly rely on local, non-rigid atomic representations for property prediction downstream tasks, limiting global geometric understanding for protein generation tasks; and (3) Existing approaches have yet to effectively model the rich dynamic and conformational information of protein structures. To overcome these issues, we introduce $\textbf{RigidSSL}$ ($\textit{Rigidity-Aware Self-Supervised Learning}$), a geometric pretraining framework that front-loads geometry learning prior to generative finetuning. Phase I (RigidSSL-Perturb) learns geometric priors from 432K structures from the AlphaFold Protein Structure Database with simulated perturbations. Phase II (RigidSSL-MD) refines these representations on 1.3K molecular dynamics trajectories to capture physically realistic transitions. Underpinning both phases is a bi-directional, rigidity-aware flow matching objective that jointly optimizes translational and rotational dynamics to maximize mutual information between conformations. Empirically, RigidSSL variants improve designability by up to 43\% while enhancing novelty and diversity in unconditional generation. Furthermore, RigidSSL-Perturb improves the success rate by 5.8\% in zero-shot motif scaffolding and RigidSSL-MD captures more biophysically realistic conformational ensembles in G protein-coupled receptor modeling. The code is available at: https://github.com/ZhanghanNi/RigidSSL.git.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a pretraining method for an unconditional protein backbone structure generation model. Experimental results demonstrate that this approach enhances the performance of FrameDiff and FoldFlow2 in generating long proteins and high-quality proteins that are not purely helical.

### Strengths
1. The problem addressed by this method is highly important. A major issue with unconditional protein generation models is that the generated designable proteins are predominantly helical. This method alleviates this problem to some extent through pretraining.

2. The pretraining strategy used in this method is relatively general and trains fast.

3. The experimental section follows established conventions and provides strong support for the claims made in the paper.

### Weaknesses
1. Some descriptions are somewhat confusing. For example, it is unclear whether Phase 1 and Phase 2 represent two different strategies or if they are sequentially related.

2. The case study lacks illustrations or examples of designable proteins with a low proportion of helices.

### Questions
1. The batch size during the pretraining stage is set to 1, and no analysis related to batch size is provided. Most existing generation models demonstrate that a larger batch size often leads to better training performance. Why a larger batch size was not used in this work?
2. Can this approach naturally extend to the transformer decoder framework like AlphaFold 3?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a rigidity-based geometric pretraining method for protein backbone generation. For perturbation-based view construction on the large-scale AFDB dataset and molecular dynamics trajectories of the ATLAS dataset, they canonicalize proteins into a reference frame and use bi-directional flow matching to pretrain the IPA module, which is shared by FrameDiff, FoldFlow-2.

### Strengths
The SE(3) rigidity pretraining for protein backbone generation is reasonable. Perturbations on rigidity align with the protein's natural conformational fluctuations, which can be interpreted as a masking-like paradigm. The MD snapshots used for pretraining are novel and interesting.

### Weaknesses
**W1. It is hard to determine whether the performance gains stem from the introduction of new data (e.g., AFDB, ATLAS) or the proposed rigidity-based geometric pretraining method. (My main concern)**

The impact of RigidSSL-MD on diversity has been analyzed in lines 408–411 and Section 5, I think the improvements of diversity are attributed to the new data in the ATLAS dataset. 

For RigidSSL-Perturb, both FrameDiff and FoldFlow2 achieved improvements in designability and novelty. However, 

(1). FrameDiff was trained only on a small PDB dataset (\~20K), while RigidSSL-Perturb incorporates new data from the AFDB (\~432k). 

(2). The training length range for FoldFlow2 is 60-384, while RigidSSL-Perturb ranges is 60-512. For evaluations, the generated length is 100-600. This is unfair. 

I suggest authors:

(1). Show FoldFlow-2 + RigidSSL-Perturb results for 100-300 instead of 100-600.

(2). Add a comparison: train baselines Framediff and FoldFlow-2 on the same AFDB and ATLAS datasets using their original training objectives. In this case, comparison on 100-600 is OK.

(3). Use a table in the main body of the paper to present data sources, numbers, protein length ranges, training objectives, training time, and other relevant information of various methods (including training, pretraining, and fine-tuning methods). This will make the paper's contributions clearer.

**Note**: In my opinion, if my understanding is correct, previous CV/NLP tasks usually have scarce data labels, so we adopt self-supervised learning to utilize large amounts of unlabeled data. However, the unconditional protein backbone generation task here is different. We can directly utilize data from AFDB for training the original SE(3) flow matching. FoldFlow-2 [1] also validated that directly using new AFDB data (+PDB, \~160K filtered data in total) for training yields performance improvements. This is why I have the ‘W1’ concern. However, I still believe that the rigidity-based geometric pretraining could be beneficial. Directly learning noise to protein means simultaneously learning the fundamental rules of protein geometry and the complex task of de novo protein design, which may be challenging. RigidSSL decouples these challenges by first pretraining on large-scale data to learn a robust geometric representation that serves as a good initialization. This motivation is reasonable for me, but the authors may need to clarify the specific benefits of RigidSSL according to suggestions (1)-(3).

[1].Sequence-Augmented SE(3)-Flow Matching For Conditional Protein Backbone Generation

**W2. Referring to RigidSSL-Perturb as phase 1 and RigidSSL-MD as phase 2 is confusing. Particularly, they are two independent pre-training methods in the experiment part. What are the results of their combined use? Can you provide further experimental results and analysis?**

**W3. Can you show some results on the conditional generation experiments, e.g.,  motif scaffolding?**


I will raise my score if my concerns are resolved.

### Questions
**Q1. I'm curious whether the parameters of these equivariant models based on IPA is hard to scale? For rigidity-based geometric pretraining, thanks to dataset like AFDB, we can scale up the data size during the pretrain phase easily. But can we scale the equivariant models based on IPA module like Proteina[1], SimpleFold[2]?**


[1].Proteina: Scaling Flow-based Protein Structure Generative Models

[2].SimpleFold: Folding Proteins is Simpler than You Think

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a pretraining method for protein generation by viewing each residue structure as a rigid body and using flow matching. Specifically, it first defines an inertial frame as a reference frame for each protein and aligns protein backbone structure with the inertial frame; then, it perturbs each protein's translation and rotation by randomly sampling transforms from an Euclidean space and a special orthogonal group to form two views; next, it samples trajectory segments from a molecular dynamics dataset; finally, it uses flow matching to generate each view from the other views. The experiments are conducted on a protein generation task and the results outperform the state-of-the-art methods.

### Strengths
1. It utilizes the structural information available in large-scale protein datasets to pretrain a protein generation model in an unsupervised manner.

2. It achieves superior protein generation performance to the compared approaches.

### Weaknesses
1. This paper is more engineering-oriented. Though it achieves superior performance across several models on a protein generation benchmark, it seems to contain little new algorithms or architectures. Reference frame definition and flow matching are widely used across many areas, including, but not limited to, machine learning, computer vision, and computational biology.

2. The construction of two different conformation views is a little bit new, but the motivation for such a construction is unclear. We can directly sample two time steps in a dynamic trajectory and generate protein structures at one time step to the other.

### Questions
1. In Section 3.2.1, $g^0$ and $g^1$ represent the original state of one protein and the perturbed state by adding random noise; in Section 3.2.2, $g^0$ and $g^1$  represent different conformations sampled at different time steps along a dynamic trajectory. This inconsistency is confusing.

2. In Section 3.2.1, the other view is constructed by adding random noises sampled from Euclidean space and the SO(3) group. How to make sure the perturbed view is biologically valid?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes a pre-training procedure for diffusion and flow-matching based protein design methods. The pre-training method consists of two phases. The first phase, RigidSSL-Perturb, adds Gaussian noise to the translation and rotation component of the residual frames of a protein's backbone. The second phase, RigidSSL-MD, uses pairs of structures from the ATLAS dataset, which contains molecular dynamics simulations of proteins.

### Strengths
Incorporating MD simulations into a pre-training step of protein design methods is an interesting and novel contribution.

### Weaknesses
The experiments demonstrate that RigidSSL-Perturb outperforms the baselines for designability and novelty, while RigidSSL-MD outperforms the baselines in diversity. However, RigidSSL-MD is not the methods of choice with respect to designability and novelty. These results limit the appilcability of the approach, as there seems no practical advantage of RigidSSL-MD over RigidSSL-Perturb, which is merely a simple data-augmentation of the data with Gaussian noise.

The examples of generated structures in figure 3 for RigidSSL seem to solely consist of alpha helices, a common bias in generative models for proteins (compare e.g. with Wagner et al. 2024). Please also report alpha helix and beta strand content for each method.

References:
Wagner, S., Seute, L., Viliuga, V., Wolf, N., Gräter, F., & Stühmer, J. (2024). Generating highly designable proteins with geometric algebra flow matching. Advances in Neural Information Processing Systems, 37, 77987-78026.

### Questions
Is high designability of RigidSSL-Perturb potentially only achieved by a high amount of alpha helices?

The augmentation of RigidSSL-Perturb, which adds Gaussian noise to the coordinates and rotation, could destroy physical plausibility of the proteins. How is the noise level controlled to prevent this?

### Soundness
2

### Presentation
3

### Contribution
2
