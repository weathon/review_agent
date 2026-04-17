# GRAM-DTI: Adaptive Multimodal Representation Learning for Drug–Target Interaction Prediction

- Decision: Accept (Poster)
- Scores: 4, 2, 4, 6

## Abstract
Drug target interaction (DTI) prediction is a cornerstone of computational drug discovery, enabling rational design, repurposing, and mechanistic insights. While deep learning has advanced DTI modeling, existing approaches primarily rely on SMILES–protein pairs and fail to exploit the rich multimodal information available for small molecules and proteins. Inspired by recent successes in multimodal molecular property prediction, we introduce GRAM-DTI, a pre-training framework that integrates multimodal small molecule and protein inputs into a unified representation. GRAM-DTI extends volume-based contrastive learning to four modalities, capturing higher-order semantic alignment beyond conventional pairwise approaches. To handle modality informativeness, we propose adaptive modality dropout, dynamically regulating each modality’s contribution during pretraining. Additionally, IC50 activity measurements, when available, are incorporated as weak supervision to ground representations in biologically meaningful interaction strengths. Experiments on four publicly available datasets demonstrate that GRAM-DTI consistently outperforms state-of-the-art baselines. Our results highlight the benefits of higher-order multimodal alignment, adaptive modality utilization, and auxiliary supervision for robust and generalizable DTI prediction. Our code is available at https://github.com/uta-smile/GRAM-DTI.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work introduces GRAM-DTI, a multimodal pretraining framework for Drug-Target Interaction. Firstly, GRAM-DTI combines multimodal modalities from molecule and protein into a unified representation using a novel volume-based contrastive learning to capture higher-order interdependencies when the modality number increases. Secondly, this paper proposes an adaptive modality dropout to dynamically filter out less informative modalities to prevent suboptimal representations. Thirdly, this work uses IC50 values as additional weak auxiliary supervision in the pretraining stage to make full use of the publicly available IC50 activity measurements. The GRAM-DTI achieves good performance on four datasets and several real-world application settings.

### Strengths
1. The motivations are interesting and attractive. Modality dropout could be a novel way to reduce the noise during the multimodal fusion, and the usage of new volume-based contrastive learning could capture more modality information when including more modalities.
2. The evaluation results show the performance improvement on four datasets, indicating the potential gain from the multimodal fusion. The ablation studies have been evaluated on most of the submodules, which is relatively comprehensive.

### Weaknesses
1. The choices of encoders are relatively out-of-date; for instance, there are stronger encoders for molecular modeling such as UniMol-v2, MLM-FG (2024), TranFoxMol (2023), MuMo (2025), ChemBERTa, HiGNN, 3D-MolT5, Mol-Llama, etc. A stronger encoder may bring better performance. The rationale of these selections is not specified. 
2. For the modality dropout, the top and least contributed modalities have been dropped to pursue a balance in multimodal learning and prevent overfitting. However, how to make sure that the important information has been preserved and more noise has been filtered out? For example, if a modality creates a larger gradient than others, it may contain more important information in this specific case. In this situation, forcing the model to learn more from irrelevant modalities can even bring in more noise. Also, the larger gradient may tell the model more information about this input data. If you want to prevent overfitting, adding a weight factor may be a better approach rather than removing the whole modality.
3. The removal of modality losses (compared with weighted modality loss) in the modality drop approach may not be a meticulous design. It may cause the gradient to become unstable and harm the pretraining efficiency and even the performance.

### Questions
1. Is the encoder for molecule $E_m$​ trainable or not? In the pretraining stage of Figure 1, it did not show whether the $E_m$​ ​is trainable or not.
2. Why does this model use these relatively old encoders instead of modern ones? What's the reason for choosing these? Is there any ablation analysis to prove the selection?
3. What are the four F in the pretraining stage in Figure 1? What model structure is used for these modules? 
4. For the datasets Inhibition and Activation, why only use AI-DTI and DTIAM baselines, and not stay consistent with the other two datasets?

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
4

### Summary
The study presents a new strategy for drug-target interaction prediction that integrates multimodality through a new pretraining strategy, that leverages volume-based contrastive learning and incorporates a regression target whenever a weak label is available. The method also incorporates a modality dropout strategy.

### Strengths
1. The volume-based contrastive learning loss for aligning multiple modalities seems like an elegant solution for the problem of multi-modal alignment in DTI tasks and more broadly in drug discovery tasks.
2. The modality dropout seems like a scalable solution for the exploding number of potential modalities and encoders that are able for molecular tasks.
3. The ablation study is comprehensive.

### Weaknesses
1. The improvement of the method over SOTA is unclear, the margins of improvement are always really low when compared to DTIAM < 0.02 in most cases. This is not a meaningful improvement. Furthermore, without error bars or confidence intervals is impossible to determine if this small improvements are even statistically significant or just an artefact of different random seeds.

### Questions
1. What ESM2 model was used: 8M, 35M 150M, 650M, 3B, or 15B?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a four-modal pretraining framework for DTI that aligns SMILES, molecule text, hierarchical taxonomic annotations, and protein sequences into a shared space using a gramian volume–based contrastive loss. It adds a gradient-informed adaptive modality dropout to prevent any one modality from dominating, and incorporates IC50 labels when available as weak auxiliary supervision discretized into three activity classes. Encoders are frozen, lightweight projectors are trained. Downstream, the model is evaluated on Yamanishi’08, Hetionet, TTD-Activation, TTD-Inhibition with warm/drug-cold/target-cold splits. I,t also reports zero-shot retrieval and ablations showing contributions from each loss and the dropout mechanism

### Strengths
- Using Gramian volume to enforce multi-way alignment is a clean way to go beyond pairwise CLIP-style losses. The objective and negative sampling scheme are clearly laid out
- he gradient-informed dropout (drop dominant or least-contributing modality with probability p drop is simple, implementable, and empirically helpful in ablations.
- IC50-based auxiliary labels provide biologically meaningful signal and choice of head is clear as it is expalined why classification was chosen over regression and how class imbalance is handled.
- Results cover DTI and MoA datasets with warm/cold-start splits, plus a zero-shot retrieval task and qualitative t-SNE alignment progression
- The five-way ablation (remove volume loss, remove bi-contrastive, remove IC50, remove dropout) makes it easy to see each piece’s contribution.

### Weaknesses
- You remove exact (SMILES, protein) pairs that overlap with downstream sets, but many entities may still be shared between pretraining and evaluation. That’s common and acceptable for self-supervised pretraining, yet quantified overlap and a performance breakdown with vs. without shared entities to de-risk confounds would be helpful, especially for the drug/target cold-start claims. Please report counts/percentages of shared drugs and proteins between pretrain and each downstream fold, and stratify results accordingly.

- The 1:10 negative generation follows prior work, but many “negatives” in DTI are unknowns rather than true non-binders, inflating AUROC and masking calibration issues. Consider similarity-aware negatives (e.g., scaffold- or family-matched), decoy generation, or at least sensitivity curves vs negative ratio, plus precision@k on class-imbalanced settings.

- You acknowledge scale is constrained by requiring full ⟨SMILES, Text, HTA, Protein⟩ tuples. Consider masked-volume training (optimize volume over the available modalities per sample) to exploit partial pairs and unlock larger corpora. Report how performance scales when adding partial-modality data

### Questions
- What fraction of drugs and proteins in each downstream fold appear in pretraining (even if the exact pair was removed)? How do results change when you exclude shared entities from evaluation?

- How sensitive are AUROC/AUPR to the negative ratio and to similarity-aware decoys? Any analysis of false-negative contamination?

- How does performance vary with batch size (negatives are in-batch), and do you use any tricks (e.g., memory banks) to stabilize volume estimation?

- Did you attempt partial fine-tuning (e.g., last layer of ESM-2 or MolFormer) and if so, what was the compute/accuracy trade-off?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents GRAM-DTI, a method for predicting drug-target interactions. The method uses pre-trained embeddings from foundation models to encode proteins, HTAs, molecules, and molecular functions. A Gram matrix-based contrastive loss is used to align modalities, and a gradient-based strategy for dropping modalities is used to promote equal contributions across modalities. Across various DTI datasets, GRAM-DTI outperforms relevant baselines in both AUROC and Recall@{1,10,100}.

### Strengths
- The method has overall strong performance across datasets. In particular, Recall@K is much improved over DTIAM across the board.
- The overall method is quite simple, which is a benefit for fast adoption and ease of implementation.
- Evaluation across warm and cold start conditions is useful and reveals the adaptability of the method.

### Weaknesses
- There are no sensitivity studies for the hyperparameters of the method (gradient history length, threshold multiplyer, loss weights).
- The negative sampling used seems weak. Keeping all but one modality to mine negative samples does not test mismatches across multiple domains.
- Failure modes are underexplored. Specific examples of drugs and targets that GRAM-DTI does not perform well on could be more insightful.

### Questions
- How does performance degrade when certain modalities are not available at pretraining time?
- How efficient is the method in wall-time compared to other methods for pretraining and training? The contrastive strategy also seems more memory-intensive, so details on what kind of hardware would also be nice.
- How comparable are GRAM-DTI with the baselines in terms of multi-modality? I.e. is GRAM-DTI privileged in any way?

### Soundness
3

### Presentation
4

### Contribution
3
