# Unified Pose Embeddings: Utilizing Euclidean Space for Simplified Topology Alignment

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 2

## Abstract
Generative models for human motion synthesis have demonstrated remarkable capabilities across tasks such as text-to-motion generation, motion inbetweening, style transfer, and motion captioning. 
However, their adoption in industry remains limited, largely due to challenges in data representation. 
Industry applications often require diverse articulated skeleton topologies tailored to specific use cases, which are further constrained by limited data availability. 
Existing methods address these challenges by aligning datasets through shared subsets or unified representations. However, these approaches rely on error-prone alignment processes, limiting their flexibility and scalability.
In this work, we leverage Euclidean space to represent human poses, bypassing the need for alignment in configuration space and significantly simplifying the learning objective.
Using Euclidean space also frees us from the need to use a common subset representation and allows us to represent poses in any complexity we desire.
To disentangle pose and body shape, we introduce a simple yet effective learning strategy.
Our method achieves robust inverse kinematics with minimal data requirements, needing just over five minutes of motion capture data to integrate new topologies. 
We demonstrate the effectiveness of our topology-agnostic representation across three downstream tasks: motion retargeting, text-to-motion generation, and motion captioning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the long-standing challenge of representing motion across diverse human skeleton topologies in generative human motion modeling. It proposes a topology-agnostic motion representation that learns in Euclidean space rather than joint rotation space. Specifically, for each skeleton topology, joint rotation data is converted to Euclidean coordinates via forward kinematics, after which a learned projector maps the pose into a shared anchor space. An autoencoder is then used to encode the pose into a latent space and reconstruct it within the anchor skeleton. Finally, the reconstructed anchor pose is mapped back to the target topology, and a learned inverse kinematics network projects the reconstructed Euclidean pose back into the joint rotation space. The paper compares the proposed method with several baseline approaches across different applications, including motion reconstruction, motion retargeting, and motion generation, and reports consistent improvements on these tasks.

### Strengths
- The paper raises an interesting and timely question about how motion representation affects generalization in generative motion modeling, particularly by contrasting Euclidean-space representations with traditional joint rotation spaces.

- The proposed method is straightforward, conceptually sound, and easy to implement, making it both practical and broadly applicable.

- Extensive experiments are conducted across multiple applications—including motion reconstruction, retargeting, and motion generation—demonstrating the versatility and effectiveness of the proposed representation.

### Weaknesses
- Although the overall idea is straightforward, some methodological details are difficult to follow. For instance, the roles of \(x_{-}\) and \(x_{=}\) are not clearly explained, and it is unclear how these representations contribute to the overall learning pipeline.  

- In the experiments, the paper compares the proposed modular framework with baseline methods such as NKN, Skeleton-Aware Networks, and SAME. However, these baselines are designed as universal models that handle multiple skeletons within a single network, whereas the proposed approach relies on modular components trained separately. This difference raises concerns about the fairness and validity of the comparisons.  

- The architectural novelty of the method is limited, as it primarily combines standard components such as autoencoders and MLP-based inverse kinematics networks. While the simplicity of the design is appreciated, the paper would be stronger with deeper analysis or insight into *why* and *how* representing poses in Euclidean space leads to the observed performance improvements.

### Questions
- In Section 4.1, the paper studies the learned inverse kinematics results. Could the authors clarify whether there are any new technical contributions or innovations in how the inverse kinematics model is learned, beyond adopting an MLP-based approximation?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes to unify the humanoid skeletal pose representation with an embedding learned from joint positions, or Euclidean space, as in the paper. Different humanoid skeletal morphologies are canonicalized to the SMPL-H skeleton ("anchor" skeleton as termed in the paper) using an off-the-shelf retargeter or mapping to preset skeletons (without fingers, with simplified fingers, and with full fingers) to learn the mappings between the anchor skeleton and the source skeletons. From the canonicalized joint positions, the method learns the latent pose space with an encoder-decoder architecture. The challenge here is to learn the latent pose representation agnostic to the skeletal proportions (phrased as "body shape" in the paper) described by the "skeleton offsets" (translations relative to the parent joint). For this disentanglement, the encoder outputs both z_o ("skeleton offset" latent) and z (pose latent). An auxiliary decoder is then trained to recover the skeleton offset given z_o. The pose decoder is also trained to recover the joint positions given the pose latent and the augmented skeleton offset latent (output from another MLP given skeleton offset). The skeleton offset augmentation randomly scales up, down, and samples skeleton offsets from other characters to encourage the pose latent to be agnostic to the skeleton offsets. The setup is numerically evaluated in errors in IK, reconstruction, and retargeting. The applications in retargeting, text-to-motion, and motion-to-text (captioning) are demonstrated.

### Strengths
The goal to propose a unified pose representation is ambitious.

### Weaknesses
* Confusing writing
  * It is fine to use words like Euclidean space / Cartesian coordinates, but essentially, all this paper is doing is learn with joint positions
  * "Body shape" in this paper is not the shape (outer skinned geometry) as in SMPL and related papers. Instead, in the context of this paper, "body shape" refers to bone lengths or body proportions
  * L264 and L267: why put loss notation next to the predicted quantity???
  * L473: "absence of global transformation" ??? FK should put joint positions in the world space, and the joint positions should capture the global transformation
* Questionable premise and insufficient discussions on its limitations
  * The premise is that joint positions are better. But the autoencoder setup tries hard to eliminate the entanglement of the joint positions and the bone lengths (identity-specific feature). Then why not use joint rotations, which are not tied to bone lengths? There are previous papers (e.g., papers by Sebastian Starke and Daniel Holden) using the combination of joint positions and rotations for the pose representations.
  * Poses/motions learned with joint positions embedded in the world space become translation and rotation-dependent. E.g., a motion translated on the ground plane should be recognized as the same motion. The proposed setup breaks this expectation.
  * Positional representation will lose the original rotation information (twist along the bone direction). IK, whether learned or not, cannot recover this twist component, as this is ambiguous.
* Insufficient visual presentation of results
  * Skeletal poses and animations should always be presented with the skinning to make sure there are no twisting artifacts (stick figures of skeletons cannot tell the twist)

### Questions
Please answer the questions raised in the Weaknesses.

### Soundness
1

### Presentation
1

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
This paper proposes Unified Pose Embeddings (UPE), a topology-agnostic human-motion representation learned in Euclidean joint space instead of configuration space. The method uses (i) an anchor representation (SMPL-H joints) with optional reduced presets x^{-}, x^{=}, x^{\equiv} to cover datasets lacking paired anchors; (ii) a lightweight autoencoder that disentangles pose z from shape o via offset regression and shape augmentation; and (iii) a per-topology neural IK module (5-layer MLP, with optional gradient refinement) to map 3D joints back to rotations. Experiments report: strong IK accuracy across five datasets (Table 1–2), competitive retargeting on Mixamo with ~10 minutes of per-character data (Table 4, Fig. 6), text-to-motion with MDM/MARDM showing comparable recall and precision but somewhat higher FID (Table 5), and captioning results via k-means tokenization (Table 6). Claimed data efficiency: IK generalizes with ≈16k frames (~5.33 minutes at 50 Hz) for a new topology (Fig. 5).

### Strengths
- anchor representation + disentanglement + per-topology IK is easy to implement and extend; components can be trained independently.  
- convincing curves and table ablations (linear vs MLP-2/MLP-5, with/without post-opt). The stricter accuracy metric (max-limb threshold) is thoughtful.  
- x^{-}, x^{=}, x^{\equiv} reduce reliance on fragile full retargeting when anchors are unavailable.  
- shape-offset regression + augmentation measurably helps reconstruction/disentanglement (Table 3; PCA analyses).

### Weaknesses
- Although topology-agnostic is claimed, the pipeline anchors to SMPL-H and still requires learning g(\cdot) and IK r(\cdot) per topology; cross-domain generality is thus partly deferred to new training. Evidence on unseen, highly non-human or production rigs is limited.  
- On text-to-motion, recall improves but FID worsens vs native HumanML3D; attribution to conversion shift is plausible but leaves the core question—does Euclidean anchoring improve generative quality—only partially answered. Captioning trails MotionAgent/GPT on alignment (Table 6).  
- Authors note intra-structural style transfer is weaker (per-frame design; pose fidelity over style). Stronger baselines (e.g., SAME, AnyTop) are only partially covered, and the “cross-structural becomes trivial in Euclidean space” risks overstating difficulty reduction without broader rigs.  
- No stress tests for noisy joints, missing markers, frame-rate variance, or domain shift. The “5.33 minutes” data claim depends on 50 Hz and Human3.6M; sensitivity to sampling rate and motion diversity isn’t quantified.  
- Many results ultimately traverse anchor --> target or target --> anchor paths; cumulative conversion noise is acknowledged but not bounded with diagnostics (e.g., cycle errors per joint, hand articulation fidelity).

### Questions
1. How far can rigs deviate (extra twist bones, non-human proportions, non-tree constraints) before the preset/anchor approach fails? Any results on production rigs beyond SMPL/H36M/LAFAN?  
2. How does the “~5.33 min” generalization change with frame rate, motion diversity, or label noise? Please plot accuracy vs. minutes at 25/30/60 Hz and with missing joints.  
3. Can you report anchor-native evaluation (no conversion to HumanML3D) or a conversion-robust metric (e.g., Procrustes-aligned feature FID) to isolate representation benefits?  
4. What are wall-clock train times and parameter counts for g(\cdot) and r(\cdot) per new topology? Any amortization via meta-learning or shared adapters?  
5. Could a temporal/style head (or latent offset predictor) mitigate intra-structural style loss without sacrificing your pose-fidelity goal (cf. your Appendix A.9)?

### Soundness
2

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
In this paper, the authors propose to use the euclidean joint positions to represent the human pose,
which is a simple and effective way to represent the human pose across different skeletons.
A set of latent encoder and decoder is used and latent space is trained to be disentangled betweenthe body shape and the pose

### Strengths
I really like what the paper is studying and trying to solve.
My team actually had a similar discussion about the universal motion representation across different morphologies,
and we believed "end-effectors" positions can be a potential candidate for this.
This is similar to the idea of euclidean space in the paper but the authors push the idea further by introducing a disentanglement learning strategy.

1. The paper discuss a curical problem in animation,
how to handle retargetting across different proportions and morphologies.

2. The paper provides multiple resolution of skeleton representation which is flexible to use for different applications.

3. Disentanglement is studied in the paper, which opens the door to unsupervised motion feature learning.
This will allow us to learn the features from huge ton of skeleton morphologies and proportions.

### Weaknesses
1. The paper does not seem finished or ready.

The result section is very weak and we are seeing very limited visual results.
There are no meshed character visualized at all except for figure 1, which is also not very informative.
It's hard to draw any conclusion from the results.

2. The scalability of the number of characters or morphologies is not well studied.
A small subsets of characters of in mixamo (4 i believe?) is not enough to make a general statement about the scalability for the proposed method.

3. The algorithm does not consider character mesh, which is crucial to consider retargetting.
This is central in industry application and was briefly studied in [1, 2].
This needs to be studied before this retargeter can be useful for industry applications.

[1] Ho, Edmond SL, Taku Komura, and Chiew-Lan Tai. "Spatial relationship preserving character motion adaptation."
In ACM SIGGRAPH 2010 papers, pp. 1-8. 2010.
[2] Yang, Lujie, Xiaoyu Huang, Zhen Wu, Angjoo Kanazawa, Pieter Abbeel, Carmelo Sferrazza, C. Karen Liu, Rocky Duan, and Guanya Shi. 
"OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction." arXiv preprint arXiv:2509.26633 (2025).

4. There's no studied about keeping the semantic meaning of the motion.
For example a slow walking might be retargeted into fast walking for a smaller person,
which is common in traditional retargetting methods.

### Questions
please refer to the above section.
I think the project is going towards the right direction, but it's far from being ready to be viewed as a complete project.

### Soundness
2

### Presentation
3

### Contribution
2
