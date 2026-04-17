# Towards Uniformity and Alignment for Multimodal Representation Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 6, 2

## Abstract
Multimodal representation learning aims to construct a shared embedding space in which heterogeneous modalities are semantically aligned. Despite strong empirical results, InfoNCE-based objectives introduce inherent conflicts that yield distribution gaps across modalities. We identify and formally analyze two conflicts in the multimodal regime, both exacerbated as the number of modalities \(M\) increases: (i) an alignment–uniformity conflict, whereby uniform repulsion undermines positive-pair alignment, and (ii) an intra-alignment conflict stemming from the non-collinearity of multi-way positives. To address these issues, we propose a principled decoupling of alignment and uniformity. We then demonstrate a theoretical guarantee that our method mitigates the distribution gap by introducing a global Hölder divergence over multiple modality distributions. We show that our decoupled losses act as efficient proxies for minimizing this cross-modal divergence. Extensive experiments on retrieval and UnCLIP-style generation demonstrate consistent gains. Overall, this work provides a conflict-free recipe and theoretical guidance for multimodal learning that simultaneously supports discriminative and generative use cases without task-specific modules.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper presents a novel model for multimodal learning that promotes alignment of matching pairs and uniformity over the hypersphere for more modalities through the volume loss from the GRAM paper.
Results are convincing and the idea is interesting as it improves some limitations of previous methods.
Also, the idea of testing the model in the generative task sounds and is convincing.

### Strengths
The idea sounds and wisely mixes two crucial topics that are scaling multimodal learning to mroe modalities and reduce the modality gap between modality representations.

I appreciate the colored summary box at page 4, as it helps fix the main concepts.

The results in the generation task are convincing and interesting.

Overall, the paper is interesting!

### Weaknesses
W1) In the box and throughout the paper, the authors say that it is crucial to have intra-modality uniformity and conflict-free alignment. Subsequently, they introduce the uniformity loss U(Z) and the L_align. However, they further add to the total loss the centroid uniformity, why? In this way, if I understood correctly, the total loss has two uniformity terms (U(Z) and (U(C)), plus the align term and the gram volume. I am afraid that the contribution of the uniformity losses may be too strong and disrupt the alignment, but the authors provide no further information for this.

- Also, why is U(C) necessary?

- I know that asking for ablation studies is somewhat boring and standard, but I am really curious to understand the contribution of each of the losses. Moreover, the authors provided simple ablations only on U(C) and L_vol, but a deeper understanding on the contribution of each of the loss is crucial.

- More theoretical explanations on the reason why adding two uniformity terms would be appreciated.

W2) How the proposed losses combination is justified by the theoretical analysis from the divergence perspective? If I'm not wrong (and I might be by the way), the divergence proves the claims regarding the uniformity and the alignment term, not for the combinations of the four losses proposed.

W3) (minor) The authors should revise the notation as the theoretical part of the paper is a bit heavy. Moreover, sometimes the modality index is between brackets and on top, sometimes is without brackets at the bottom. This makes the paper a bit hard to read and confuse the notation.

Minor comments:
- the references of LanguageBind and GRAM are wrong (still arxiv), please update them with the correct citation.
- line 131, InfoNEC instead of InfoNCE.

Overall, even though my initial score is not so high, I can lean towards increasing the score if the authors provide theoretical and/or empirical evidences in response to my comments.

### Questions
Q1) Can the authors provide ablation studies for each of the losses from scratch on MSRVTT?

Q2) Why the \lambda_vol for the generation is set to 0.1? the authors say that it is to have more emphasis on anchor-based alignment, but the volume definition is based on the anchor as well.

Q3) Can the authors compute the cosine similarity between true pairs to further strengthen the plot with the tsne?

Q4) can the authors share the link for their implementation? I would be curious to see losses implementation.

Q5) I assume that the results in Table 1 are achieved via the multimodal encoder that both VAST and GRAM employ. Can the authors provide retrieval results before feeding the embeddings to the multimodal encoder (which, in the case of both vast and gram is the text encoder)?

### Soundness
4

### Presentation
2

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
This paper analyzes the inherent limitations of multimodal contrastive learning when using the InfoNCE objective. It identifies two sources of conflict that worsen as the number of modalities $M$ increases: the **alignment–uniformity conflict**, where uniformity forces oppose cross-modal attraction, and the **intra-alignment conflict**, caused by non-collinearity among multi-way positives. To address these issues, the authors propose **UniAlign**, which decouples intra-modality uniformity and cross-modality alignment via an anchor-based loss and a “volume” regularizer. They also introduce a new theoretical formulation — the *global Hölder divergence* — to interpret their objective as minimizing a cross-modal distribution gap. Experiments on retrieval and UnCLIP-style generation show modest but consistent improvements over GRAM and VAST baselines.

### Strengths
1. The paper tackles a relevant and well-known problem — the modality gap in multimodal contrastive learning — and extends previous analyses from the bimodal to the general multimodal case.  
2. The proposed framework is straightforward to implement, requiring no architectural modifications or additional modules.  
3. The empirical results are encouraging, showing that the proposed method consistently outperforms existing alternatives.

### Weaknesses
## Major  
1. **Theoretical clarity.** The theoretical section is difficult to follow, and several key definitions are vague or insufficiently motivated. In particular:  
   - In Eq. (3), it is unclear why $V_a$ represents the alignment force and $\Phi_a$ the uniformity force — this connection should be explained more carefully.  
   - Assumption 1 is introduced without justification; it is not obvious why it should hold or be meaningful in practice.  
   - In the boxed text of Section 3.1, the statement *“promote uniform coverage within that modality only”* is ambiguous and should be clarified.  
   - The same box refers to *“consensus magnitude”* as if it were a standard concept, but it is never defined or explained.  

2. **Idealized assumptions.** The theoretical analysis relies on simplified and somewhat unrealistic conditions (e.g., independence and isotropy of negatives, uniform temperature across modalities). A discussion of the robustness of the results under more practical conditions would improve the paper.  

3. **Loose theory–practice connection.** The derivation of the Hölder divergence (Eq. 16) is mathematically elegant but remains disconnected from the practical loss in Eq. (13). The claim that UniAlign effectively minimizes this divergence is heuristic, relying on a kernel-density approximation that is never empirically verified.  

## Minor  
1. Corollaries should follow from a theorem or proposition. Please revise the naming (e.g., use *Theorem*, *Lemma*, or *Proposition*).  
2. The method introduces several hyperparameters whose influence is not systematically analyzed. An ablation or sensitivity study would strengthen the experimental section.

### Questions
1. Could similar improvements be obtained by simply re-weighting InfoNCE gradients rather than introducing new loss terms?  
2. Can you empirically estimate the proposed Hölder divergence during training to verify that it indeed decreases?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper identifies two fundamental conflicts that hinder multimodal contrastive learning, the alignment–uniformity conflict and the intra-alignment conflict, and proposes a principled decoupling framework with a theoretical guarantee via a global Hölder divergence to achieve conflict free multimodal representation learning that improves both discriminative and generative performance.

### Strengths
1.The paper is clearly written, presenting the motivation, conflicts, and proposed solution in a well-structured manner.

2..The proposed method is theoretically grounded and experimentally validated, demonstrating robust performance across multimodal retrieval and generation tasks.

### Weaknesses
1.There are some writing errors, such as “InfoNEC” in Section 2.2.
2.Although anchor-based alignment eliminates cross-modal rejection, it introduces modal bias. I have a question: Could the selection of different anchor modalities lead to representation imbalance?
3.Your approach employs intra-modal consistency to prevent representation collapse. Was modal collapse considered during training?
4.The proposed global Hölder divergence is defined over multiple modality distributions. Is this divergence sensitive to the curse of dimensionality in high-dimensional embedding spaces?

### Questions
Specific issues can be found in the weaknesses.

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
The authors claim that standard contrastive learning over more than two modalities is limited by a conflict between the alignment and uniformity criteria. To this end, they propose an approach, UniAlign, which aims to resolve this conflict. They theoretically analyze the identified form of conflict and empirically evaluate their approach on generative and retrieval tasks.

### Strengths
1. The work addresses a potential roadblock in contrastive learning, i.e., the possibility that the alignment of positive tuples being interfered with by the uniformity criterion in contrastive learning.
2. The authors do a good job in conveying the intuition behind the problem that they are tackling.
3. For the kind of alignment-uniformity conflict presented in their Assumption 1, the authors theoretically prove that it increases as the number of modalities grow.

### Weaknesses
1. Although the authors argue for the existence of a cross-modality uniformity conflict which oppose alignment, there is no clear evidence for this, either in the existing literature or in this work.
For instance, in Line 104, the authors refer to learning, Yin et al. (2025), mentioning that "clearly demonstrate that uniformity across modalities (“inter-uniformity”) conflicts with the alignment term". However, I found no results as such upon going through that work. The claim can be described as: for each positive pair, the alignment term can be cancelled out by the uniformity forces when aggregated across modalities. However, at least intuitively, the probability of this happening seems extremely low, because (i) otherwise most multimodal contrastive methods just would not work; and (ii) it is well known that there exists clusters of similar positive pairs (even in the unsupervised case) which are spontaneously grouped together under contrastive learning [a, b], meaning that the probability that alignment of positive pairs would be reinforced by other similar positive pairs is much higher than the probability of interference from unrelated / negative pairs. Having said that, it is possible that the cancellations could get stronger as the number of modalities increase as the authors argue, however, to establish this fact there needs to be significantly more theoretical and empirical analyses, since there is plenty of very strong evidence pointing to the contrary [c, d].

2. No ablation studies are presented in this paper. Without them, it is difficult to evaluate exactly what is contributing to the performance improvements reported in Table 1.

3. The purpose of the volume-based complement loss is not clear. It seems that it encourages samples that are grouped together from multiple modalities into a tuple a dispersed. However, this would mean that samples that share the same semantic information would be pushed apart, which goes against the desired objective.

4. The L_align term does not seem to be any different from a standard contrastive learning objective, where one modality is considered as an anchor. However, there does seem to be a downside, which comes from imposing only the uniformity objective on the anchor modality. It would imply that samples that are semantically similar in the anchor modality will not be brought together, and consequently, neither would the samples from the other modalities, since such is the nature of the anchor to which they are aligned.

References:

[a] Parulekar et al., "InfoNCE Loss Provably Learns Cluster-Preserving Representations", COLT 2023. \
[b] Lu et al., "f-MICL: Understanding and Generalizing InfoNCE-based Contrastive Learning", TMLR 2023. \
[c] Girdhar et al., "IMAGEBIND: One Embedding Space To Bind Them All", CVPR 2023. \
[d] Wang et al., "Image as a Foreign Language: BEIT Pretraining forVision and Vision-Language Tasks", CVPR 2023.

### Questions
Please refer to the Weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2
