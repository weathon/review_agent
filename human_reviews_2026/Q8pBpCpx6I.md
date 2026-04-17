# Accurate RNA 3D Structure Prediction via Language Model-Augmented AlphaFold 3

- Decision: Reject
- Scores: 0, 6, 6, 4

## Abstract
Predicting RNA 3D structure from sequence remains challenging due to the structural flexibility of RNA molecules and the scarcity of experimentally resolved structures. We ask how self-supervised RNA language models (LMs), trained on millions of RNA sequences, can best enhance AlphaFold 3 (AF3) for RNA structure prediction. Using an open-source AF3 reproduction, we run controlled experiments that fix data and hyperparameters while varying fusion position and method. We find large performance variations between fusion strategies, and without Multiple Sequence Alignment (MSA), they are generally not effective. When incorporating MSA, the most effective approach is additive fusion applied at the late stage of the conditional network, refining AF3’s single representations with RNA LM embeddings. On RecentPDB-RNA (47 newly released targets), our best model achieves an average TM-score of 0.438 and a success rate of 30\% (TM-score $\ge$ 0.6), significantly outperforming all baseline models. On 11 CASP16-RNA targets, it matches the best automated system trRosettaRNA. These results show that properly fused RNA LM features substantially advance RNA 3D structure prediction. We will release the data, code, and model weights to support open science, reproducibility, and the development of automated RNA structure prediction models.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper examines the integration of self-supervised RNA language models (RNA LMs) into AlphaFold 3 (AF3) to improve its RNA structure prediction capabilities. Different positions within the AF3 architecture and methods to fuse RNA LM embeddings were examined to achieve the best performance. Protenix, an open-source AF3 reproduction, which was trained on RNA data without using multiple sequence alignment (MSA), was fine-tuned on a custom, newly created RNA structure database, this time incorporating RNA LM embeddings and RNA MSAs. The paper reports a large performance variation based on the fusion position and methods, as well as including or not including MSAs. The authors report new state-of-the-art results on the custom RNA structure database, outperforming other deep learning RNA structure prediction methods, including the AF3 server. However, the performance boost mostly comes from augmenting the training dataset. On the CASP16 targets, the proposed RNA LM-augmented AF3 matches the best automated system trRosettaRNA, while still trailing behind the predicted structures of non-deep learning models with human interaction.

### Strengths
- Even though the idea of integrating self-supervised RNA LMs into RNA structure prediction models is not new and has already been investigated by replacing MSAs with them (DRfold2) or by augmenting MSA inputs (RhoFold+), examining the integration of an RNA LMs into AF3 is an interesting idea.
- The authors systematically evaluated different fusion methods and positions on the custom dataset, giving a clear idea of which strategy might improve the structure prediction performance.
- The authors promised to release data, code, and model weights to support reproducibility, which is important in the community dealing with a hard and delicate problem such as RNA structure prediction.
- Even though the proposed RNA LM-augmented AF3 model does not outperform the state-of-the-art automated system trRosettaRNA on the CASP16 targets, it boosts the performance of the server version of AF3. However, the way results are presented, it is not clear whether the boost comes from data augmentation, a different homolog search pipeline, or incorporating an RNA LM.

### Weaknesses
**Custom training and evaluation data:**
- For training data (finetuning data), the authors used the RNA3DB data released on 4 Dec 2024. This RNA3DB version takes into account all the existing RNA chains from the protein databank (PDB) released before 4 Dec 2024. For evaluation data, the authors created a custom dataset of RNA structures using only a time cut-off. The custom dataset was constructed of 67 samples from the PDB released between 4 Dec 2024 and 28 Apr 2025. The RNA structures in the evaluation dataset were not filtered by sequence and structure similarity to the training set. This means there is data leakage between the training and evaluation datasets. Other state-of-the-art tools make much more rigorous data splits. Why didn't you use the RNA3DB splitting strategy to make sure the RNAs in the custom evaluation dataset are structurally different from the training dataset? I suggest using the RNA3DB procedure, either including the newly released data or not, and using the *component #0* structures for the evaluation. That would provide a rigorous evaluation of the proposed method.

**Experimental setting and results:**
- From Table 1, it is clear that the biggest improvement from the original Protenix comes from data augmentation and possibly overfitting. The original Protenix used a time cut-off date of September 30, 2021 (same as AF3). From the table, it can be seen that adding only RNA LM embeddings either makes the performance worse or the performance in terms of TM-score is at most $ 0.003 $ higher, which is statistically insignificant. It is strained to say it is effective. Only after adding the MSAs, the results become better. Thus, it is not clear where the boost in performance comes from. Probably, it is the combination of both the MSA and RNA LM, but having only RNA LM is not effective. I think it would be beneficial to comment and explain why adding only RNA LM embeddings is not effective.
- (line 346) I have a problem with picking different models when reporting improvement in terms of percentage. For reported improvements, the authors picked interchangeably the worst of the two baseline models to report higher gains.
- In Table 2, the first column, it is not clear which versions of the concurrent models were used, their cut-off dates and whether the same MSAs as for the proposed model were employed. Additionally, even if the same MSAs were used, which were mined from the databases with temporal cutoffs before or 3 Oct 2022, the proposed model leveraged AIDO.RNA's embeddings, which were trained on the RNAcentral 24.0 dataset released on 25 Jun 2024. Reporting versions of the concurrent models used in comparison is the minimum I would expect. Also, I think it should be clearly stated and provided with the commands and inputs used for the inference of other tools. It would be fair to comment on the AIDO.RNA's pretraining dataset and its cutoff, and why wasn't the same cutoff date used for MSA search?
- In Table 2, the authors provided only the results of the proposed model finetuned on the augmented data, using MSA and RNA LM embeddings. It is crucial to see the effect of finetuning on the new dataset, which is not reported in Table 2. Additionally, since the benefits of using both MSA and RNA LM embeddings were not clear when evaluated on the custom evaluation dataset, I find it important to show their benefits separately for the CASP16 evaluation dataset. This way, it is not clear where the performance boost comes from; is it from data augmentation, using MSA, or employing RNA LM?
- In Table 3, the authors compared the proposed model with the finetuned Protenix; however, it is not clear whether this is the finetuned one with or without using MSA. This information is crucial since this way it is not clear whether the boost for low identity structures comes from using the MSA, RNA LM, or both. Please include this information to be more transparent. 

**Minor comments**
- The reference list is often missing capitalization of letters in the paper titles and journal titles. 

I recommend **rejecting** this paper. 

The key reasons are: 
1. Custom training and evaluation datasets are not able to test the generalization capabilities of the models for RNA structure prediction and are of little or no benefit. 
2. The way the results are presented is poor and lacks transparency. Even if there is improvement over the AF3 model, the improvement does not outperform the state-of-the-art methods and it is definitely not clear where it comes from.

### Questions
- (line 241) It is not clear whether only 39% of the training sequences had available MSAs during the fine-tuning phase, or whether you additionally used rMSA for the rest of the sequences?
- Why didn't you use the RNA3DB splitting strategy to make sure the RNAs in the custom evaluation dataset are structurally different from the training dataset?
- In Table 2, please comment on the possiblilty of overfiting of your model on the custom dataset.
- Please could you provide the following: 
  1. Report versions of the concurrent models used in comparison.
  2. Clearly state and provide the commands and inputs used for the inference of other tools.
- It would be fair to comment on the AIDO.RNA's pretraining dataset and its cutoff, and why wasn't the same cutoff date used for MSA search?
- In Table 2, for CASP16 results, could you provide results of the finetuned Protenix only using the augmented data, and the one finetuned using the augmented data and MSA?
- In Table 3, could you be more specific whether finetuned Protenix is the finetuned model with or without using MSA?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores methods to enhance RNA 3D structure prediction in AlphaFold 3 by incorporating embeddings from a pretrained RNA language model. The authors operate on the premise that the protein-centric training of AF3 leaves its RNA-specific representations underdeveloped. This model is benchmarked on the RecentPDB-RNA dataset, where it shows improved average TM-scores compared to the baseline AF3, and on 11 CASP16-RNA targets, where its performance is on par with trRosettaRNA.

### Strengths
1. The paper tackles a relevant and challenging problem in structural biology, and the hypothesis that RNA-specific LMs can help AF3 is well-motivated.

2. The experimental design is systematic. The controlled study of fusion positions and methods provides a clear ablation, which is a useful, if straightforward, contribution.

3. The analysis in Section 6.3, which breaks down performance by sequence length and identity, is helpful. It correctly identifies that the LM provides the most benefit in data-sparse scenarios (low identity, long sequences), which supports the original hypothesis.

### Weaknesses
1. The primary contribution of this work is empirical, not methodological. The paper does not propose any new fusion techniques but rather applies existing ones. The main takeaway is "additive fusion at position X works best," which is a useful engineering finding but lacks deeper algorithmic insight.

2. The performance claims seem somewhat inflated when considering the full picture. While the 21% relative gain on RecentPDB-RNA is notable, the model only matches the existing trRosettaRNA server on the CASP16 benchmark. This lack of a clear win on the CASP set calls into question how broadly the SOTA claims generalize.

3. The practical utility of the resulting model is questionable. The authors state that the confidence head was not fine-tuned. This is a significant omission, as it means the model cannot provide a reliable estimate of its own accuracy, a feature that is essential for real-world use.

4. The exploration of methods feels incomplete. For instance, the RNA LM is kept frozen. It is a missed opportunity not to investigate at least partial finetuning of the LM, which might have yielded different or superior results.

### Questions
1. The paper concludes that fusing at pair representations is ineffective. Was this conclusion based solely on the outer-sum method for generating $z^{rnalm}$? 

2. The paper does not provide an analysis of why simple additive fusion outperformed the more expressive cross-attention method. Is this simply because the RNA3DB training set is too small to properly train the attention adapter, or is there a more fundamental reason?

3. The results regarding MSAs in Table 1 are contradictory and unexplained. The Finetuned Protenix baseline actually performs slightly worse when MSAs are added (0.450 vs 0.441). In contrast, the RLM-aug Protenix model gets a significant boost from MSAs (0.431 vs 0.472). This suggests a non-trivial interaction between the LM features and the MSA features, which the paper fails to investigate. Can you explain this discrepancy?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper analyses the incorporation of RNA language model embeddings into AlphaFold 3 (or Protenix as an open-source reproduction of AF3). The authors identify five different positions in the AF3 architecture to enhance it with LM embeddings, and study three different strategies for embedding fusion (add, concat, attention-based) using RNA3DB. The best performing model, RLM-aug Proteinx, is further evaluated on 67 recent PDB samples, as well as 11 CASP16 RNA targets. Here, RLM-aug Protenix shows substantial performance improvements with SOTA results compared to state-of-the-art 3D folding engines on the recent PDB dataset and competitive performance with the best automated method for the CASP16 data. The authors further analyze the model performance across different sequence lengths and wrt. sequence identity between training and test samples.

### Strengths
- The authors provide a solid and well controlled experimental setup for evaluation of 3D predictions using their different embedding strategies.
- The model achieves SOTA performance on a PDB dataset of recent samples and general strong performance for RNA 3D folding.
- The incorporation of RNA-LM embeddings from massive self-supervised sequence learning into an existing SOTA 3D folding engine is promising, especially since the proposed model clearly outperforms existing approaches like RhoFold+ and DRFold2.
- Improving 3D RNA predictions is a challenging and important task.

### Weaknesses
1. While the authors thoroughly evaluate 3D structure predictions, there is nearly no assessment of the embedding quality (except for final 3D structure prediction quality). I’m not super familiar with embedding analysis of RNA-LMs, but a simple experiment could for example be to score embedding quality, e.g. using IsoScore [1] or other methods as recently done for DNA embeddings [2]. Then one could analyze if embedding “quality” correlates with structure quality. 
2. Another aspect is the usage of MSA. If I understand correctly, MSA was not available for all datapoints. Analyzing predictions with and without available MSA and potentially also relating to embedding information could provide further useful insights.
3. The authors use only a single RNA-LM for their evaluation. The only comment on the choice is that it is a “strong transformer-based encoder only LM”. Since the paper ‘only’ assesses different ways to incorporate RNA-LM embeddings into Protenix, different LMs could have been tested, although this would likely require effort to match embedding dimensions.

[1] Rudman, W., Gillman, N., Rayne, T., & Eickhoff, C. (2021). IsoScore: Measuring the uniformity of embedding space utilization. arXiv preprint arXiv:2108.07344.

[2] Awasthi, R., Mend Mend Arachchige, G. S., & Zhu, X. (2025). Unsupervised evaluation of pre-trained DNA language model embeddings. BMC genomics, 26(1), 710.

### Questions
1. In table 1, do the authors have any rationale why performance for all methods seems to decrease when using MSA, except for the final RLM-aug Protenix (and success rate for FT Protenix)?
2. Since the authors mention RNA flexibility as a challenge for 3D structure prediction, does the incorporation of RNA-LM features influence the sampling behavior of the model? While sampling in AF3 (and I expect it to be similar for Protenix) does not provide true conformation ensembles, it would still be interesting to see if the structure distribution of the five samples per test target changes with embeddings.
3. I can imagine that the embedding quality of the LM is higher if the model already knows similar sequences from the training data. While done for training data for 3D predictions, how does the training data of the LM relate to structure prediction quality? 
4. Since the performance drops quite substantially for longer RNAs, could training the LM on more longer sequences solve the problem? Or more general, does the length distribution of RNAcentral correlate with the observed performance across different lengths?  
5. The LM was trained on ncRNAs from RNAcentral only, how about performance on coding RNAs? How does performance look like for different RNA types? Are there datapoints available to analyze this kind of questions? Can this also be related to the embedding quality?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper targets RNA 3D structure prediction and systematically studies how to fuse an RNA language model (LM) into an AlphaFold-3–like architecture. Under fixed data and hyperparameters, it varies only the fusion position and mechanism, reporting notable gains on RecentPDB-RNA and claiming automated SOTA.

### Strengths
Data and hyperparameters are fixed; only fusion position and strategy (add/concat/cross-attention; single vs multi conditioning) are varied.

On RecentPDB-RNA, improvements in TM-score and success rate over the baseline are substantial, with stratified analyses by sequence length and identity.

### Weaknesses
Prior work (e.g., RhoFold+, DRfold2) already integrates LMs for RNA; contributions read as engineering optimization of where to fuse rather than new mechanisms.

Protenix is used instead of proprietary AF3, and its weaker baseline raises questions about transferability to real AF3.

12.9k training samples; tiny test splits (≈67 and ≈11) with no statistical tests or confidence intervals reported.

Performance degrades on low sequence identity (<0.5) and long sequences (>400 nt); automated methods still trail human expert solutions.

### Questions
Why freeze the RNA LM? What performance is forfeited by not finetuning it?

Can you disentangle LM vs MSA contributions with a full 2×2 ablation (with/without LM × with/without MSA) and report variance/CI?

Why does “single conditioning + add” at mid/late layers work best? Please provide theoretical or empirical evidence (e.g., attention/gradient flow/information bottleneck analyses).

Given the use of Protenix, what evidence shows the fusion strategy carries over to real AF3 (or other AF3 replications)?

What are the primary sources of error versus human-guided Vfold (data coverage, long-range constraints, inductive bias)? Can you provide an error taxonomy to localize the gap?

### Soundness
3

### Presentation
3

### Contribution
2
