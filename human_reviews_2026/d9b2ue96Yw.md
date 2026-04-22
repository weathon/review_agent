# RAMM: Robust Adversarial Multimodal Learning for Protein Stability Prediction

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Multimodal representations that integrate protein sequence and structure offer powerful priors for modeling protein properties, yet adapting them to small, task-specific datasets often leads to overfitting. We present RAMM, a two-stage adversarial multimodal learning framework for predicting the stability effects of protein mutations. In the first stage, we fine-tune a multimodal encoder on large protein
datasets to capture general sequence–structure relationships. In the second stage, we train this encoder on target protein stability datasets while jointly optimizing an adversarial objective: a discriminator attempts to distinguish wild-type from mutant proteins, while the encoder learns to produce features robust to wildtype and mutation domain shifts that fool the discriminator. This adversarial game drives the system toward a Nash
equilibrium where the learned latent space becomes robust to distributional shifts introduced by mutations. Evaluations on low-sequence-identity benchmarks show that this approach improves generalization, achieving AUROC = 0.763 on the SKEMPI 2.0 classification task and RMSE = 1.39 kcal/mol on the S669 regression benchmark. These results highlight that adversarial deep learning can enhance the robustness of multimodal protein models for challenging biological prediction tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces RAMM, a two-stage framework for predicting the stability of proteins with mutations. The primary contribution is a method that reframes predicting mutation effects as a domain adaptation problem. The first stage involves self-supervised pre-training of a Fusion Autoencoder (FAE) that learns a unified representation from three protein data modalities. The second stage employs an adversarial fine-tuning strategy, where the encoder is trained to produce representations that are invariant to mutations. A set of discriminators is trained to distinguish between them. The authors claim that this adversarial objective regularizes the learning process, forcing the model to capture fundamental biophysical principles rather than spurious correlations. They validate their approach on a classification and a regression benchmark.

### Strengths
1. The idea of applying adversarial domain adaptation to make a protein representation model robust to mutations is novel and insightful.

2. The design of the two-stage approach is well-motivated and logically sound. Empirical results provide evidence for the effectiveness of the proposed method.

3.  The paper is well-written, and the proposed method is explained clearly.

### Weaknesses
1. The necessity for three modalities is not justified. The paper would benefit from a discussion of why these specific extractors were chosen and how the choice might impact overall performance. How about fusing a structure encoder and a sequence encoder?

2. The paper does not provide details on how crucial hyperparameters, such as the modality-specific weights (λ_m) in the adversarial loss, were selected. An analysis of the effect of these parameters would increase the robustness of the results.

3. No reproducibility statement. It is not clear how the baseline results are obtained. 

4. Just curious to see the comparison to [1]. Are there any relationships between the proposed adversarial training and the RL or the alignment approach in [1]?

[1] Boltzmann-Aligned Inverse Folding Model as a Predictor of Mutational Effects on Protein-Protein Interactions. ICLR 2025

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a two-stage training framework for protein representation learning. The first stage merges three pretrained encoders on protein sequence, residual-level graph, and 3D point cloud respectively into a common semantic space with a Fusion AutoEncoder (FAE); whereas the second stage adopts the adversarial learning principle to further train the encoder along with a discriminator. The paper claims that the trained encoder performs well on downstream tasks including SKEMPI 2.0 and the S669 benchmark.

### Strengths
* The paper addresses the protein mutation stability prediction task, which is a very important and challenging problem in proteomics and biology.  
* The encoder merges three different modalities with their corresponding pretrained encoders, generating a comprehensive representation of proteins while taking advantage of pretrained uni-modal encoders. 
* The proposed framework is evaluated on two different benchmarks, surpassing existing unimodal and multimodal baselines. An ablation study is also carried out, proving the contribution of the adversarial learning.

### Weaknesses
* The adversarial training process is very well explained in the paper, but I am still a bit confused and doubting the efficiency and necessity of the adversarial learning for this task. The paper states in the abstract that the adversarial learning stage is to train the encoder to “produce mutation-invariant features”; but also states towards the end of subsection 3.3 that the adversarial regulation is to compel the encoder “to learn a common, mutation-aware latent space”. These claims seem contradictory, and the paper lacks further mathematical explanation to justify them. 
* The novelty of the multimodal fusion appears limited. All three unimodal feature encoders are pretrained and frozen, each applied with one single linear layer for projection. The following autoencoder is more like a compression and fusion design instead of multimodal alignment. I question whether the cross-modal heterogeneous gap is very well addressed.

### Questions
Please address my two major concerns in the “Weaknesses” section first. I will reassess after the rebuttal. Two other miscellaneous questions are as follows:
* The paper picks XGBoost as the downstream classification head of their encoder. Maybe I missed it somewhere, but is this also the classification head for other baselines in Table 1? Also, have you tried with simpler heads like MLP? If so, how is the performance? 
* According to Table 3, it seems to me that DDGemb is a better model than the proposed approach, as it has much better PCC but only slightly worse RMSE. The interpretation in subsection 4.3.3 is not very convincing.

### Soundness
3

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
In this paper, the authors developed a two-stage adversarial model for predicting the stability effects of protein mutations.

First, they train a Fusion Autoencoder (FAE) to learn multi-modal representation from pre-trained seq, graph and point cloud embeddings. Then they take the encoder and fine-tune it in an adversarial way, where the Discriminator tries to predict the domain (i.e., whether the protein is a wild-type or a mutant), forcing the Encoder to learn a mutation-invariant latent space. Finally, they train a predictor on the difference vector between the wild-type and mutant embeddings for the downstream stability prediction task.

### Strengths
Their idea of using an adversarial learning process to enforce a mutation-invariant latent space, and use that for tasks like stability prediction is great.

### Weaknesses
1. For multimodal embedding, instead of simple concatenation, why not use contrastive learning?

2. FusionProt is a better multimodality fusion model. Its performance is only slightly lower than RAMM. Have you tried training adversarial model on top of FusionProt?

3. Why use 3 discriminators when the embeddings are already fused? What’s the performance with a single discriminator?

4. Have you considered using evolutionary profile/MSA (Multiple Sequence Alignment) as another input. This will help in determining which parts of the sequence are functionally important.

### Questions
N/A

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes RAMM (Robust Adversarial Multimodal Model), a two-stage framework for learning mutation-robust protein representations that integrate both sequence and structure information. In Stage 1, a Fusion Autoencoder (FAE) jointly encodes three modalities — sequence, backbone, and atomic point clouds — into a unified latent representation. In Stage 2, an adversarial domain alignment objective ensures that mutant and wild-type proteins map to similar latent distributions, encouraging mutation-invariant representations. The model is evaluated on SKEMPI 2.0 (binding affinity classification) and S669 (stability regression), achieving improved AUROC and RMSE compared to several baselines.

### Strengths
## Clear motivation and novelty: 
- The paper addresses a real gap in protein representation learning — current multimodal models often overfit to structural details and fail to generalize when fine-tuned on mutations. The use of adversarial alignment to enforce mutation invariance is original and well-motivated.
- The Fusion Autoencoder architecture effectively combines diverse protein modalities (sequence, backbone graphs, and 3D point cloud) into a coherent latent representation.
- The adversarial training is novel and very interesting. 

## Well-written: 
- The paper is clearly organized and connects ML ideas (domain adaptation, multimodal fusion) to biological motivation effectively.

### Weaknesses
Limited experimental depth: 
- The results are restricted to two datasets (SKEMPI and S669), and the experiments are relatively light for ICLR standards.

Poor Validation: 
- 5-fold cross validation is poor way to validate a model. This is probably why in your ablation study, the adversarial training has such a minor performance improvement. 
- I would recommend benchmarking using a high quality train/test split. I would use the training/test split from the Stability Oracle and Binding Oracle paper. Is there a reason why you excluded these methods as baselines? They also generate a mutation level representation in order to compute ∆∆G. An additional method for these tasks to add would be the EvoRank method and Mutate Everything which also demonstrates good performances on ∆∆G for stability and binding affinity. 
- Why do you not use cDNA-proteolysis based datasets like the Rocklin dataset? These datasets are 2-3 orders of magnitude larger and allow you to get more mutation and class balance. 

Ablation insufficiency: 
- It’s unclear how much each modality (sequence, structure, point cloud) or the adversarial loss contributes to the performance gain — an ablation table for each modality would strengthen the argument.

### Questions
To address class imbalance via data augmentation, why didn't you use thermodynamic permutation? 

In 4.2.2: When you say 80:20 split, how did you ensure there was not data leakage? 

 How are you addressing mutation imbalance? most of these datasets come from alanine scanning experiments, so how do you know that your model is not just becoming good at predicting from wild type to alanine mutations? 

Your confusion matrix in the appendix shows that you have much more data for stabilizing than destabilizing when I know these datasets are imbalanced and lean towards destabilizing. How do you have more stabilizing data than destabilizing data? 

To improve my score: 
- train using a much bigger training set and evaluate using high quality train/test splits to properly assess if the fusion AE and adversarial training actually translate to a downstream point mutation task like ∆∆G for stability and binding. 
- compare with the other baseline models I mentioned in the weakness section
- demonstrate performance based on mutation type so we can see if the model is able to generalize beyond mutations to alanine.
- ablate the 3 modality inputs

### Soundness
3

### Presentation
3

### Contribution
3
