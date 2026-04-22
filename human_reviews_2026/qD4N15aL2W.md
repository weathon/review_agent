# MTSSRL-MD: Multi-Task Self-Supervised Representation Learning for EEG Signals across Multiple Datasets

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 2, 0, 4, 2

## Abstract
Electroencephalography (EEG) supports diverse clinical applications. However, effective EEG representation learning remains difficult because scarce label annotations and heterogeneous EEG montages limit the scale of available datasets. In practice, single and small-scale datasets often result in models with poor generalization, particularly for underrepresented classes with limited samples, which are harder to learn reliably. These challenges become even more critical in the EEG-based sleep stage classification task, especially for minority stages that are not only scarce but also transitional with overlapping characteristics, which makes them prone to misclassification. In this work, we propose MTSSRL-MD (Multi-Task Self-Supervised Representation Learning for EEG Signals across Multiple Datasets), a unified framework that combines multi-dataset and multi-task self-supervised pretraining with a channel alignment module to alleviate the impact of scarce labels, heterogeneous EEG montages, and small-scale datasets that often cause poor generalization. This design enables the learning of EEG representations that are generalizable. Multi-dataset learning provides broader feature diversity that facilitates more robust cross-dataset generalization. A spatial-attention Channel Alignment Module (CAM) projects heterogeneous EEG montages into a shared channel space and provides spatial weights that highlight regions aligned with standard EEG montages, offering interpretability. Complementary self-supervised tasks—augmentation contrastive, temporal shuffling discrimination, and frequency band masking—provide temporal and spectral information that improve robustness on these underrepresented classes. Experiments on three heterogeneous EEG sleep datasets show that MTSSRL-MD consistently outperforms single-dataset SSRL baselines and even surpasses SeqCLR, a representative multi-dataset single-task SSRL method, particularly under low-label conditions, demonstrating the effectiveness of integrating multi-dataset and multi-task learning for EEG-based sleep stage classification. Besides classification performance, MTSSRL-MD achieves more efficient inference than single- and multi-dataset SSRL baselines. Moreover, the unified design of our proposed method allows the use of a single pretrained encoder to be fine-tuned across diverse datasets, highlighting efficiency and practical value for clinical research, suggesting strong potential for deployment in real-world settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a new method for multi-dataset self-supervised learning for dealing with heterogeneous datasets. This architectures use a Channel Alignment Module to project each dataset with a different number of channels to a space with the same number of channels. 
The CAM and the encoder are pre-trained using a three-loss supervised learning. Then, a fine-tuning is made for the encoder and a classifier with a classification loss. 
This method is evaluated on three sleep staging datasets and compared to different self-supervised methods and fine-tuning methods.

### Strengths
The paper is easy to follow.

### Weaknesses
The major weakness, in my opinion, is the motivation of the paper.

- The authors claim that:  "As a result, the sample size of available labeled data is small". In sleep staging, a huge database is available with the NSRR and several datasets, such as RobustSleepNet or U-Sleep, which used more than 10 datasets for their experiments, reaching more than 10.000 subjects. I agree that a missing label is something that is possible in the EEG field; sleep staging avoids this problem. Sleep staging remains a good modality for proof of concept, but using the method on other EEG tasks can help strengthen the claim.

- The authors motivate their method to be more generalizable across domains (datasets), but their method needs to see the domain for pretraining. This is a major bottleneck to claiming that the method is generalizable to a new domain. Since the CAM is dataset-specific, you cannot pretrain before having access to the new datasets.

- In Table 3, the authors compare MTSSRL-MD with ATCNET and DeepSleepNet. This Table makes me wonder if self-supervised learning is very useful since the scores seem very close for SleepEDF, lower for ISRUC, and better for ANPHY.  Can you try to compare with other supervised learning baselines, like CareSleepNet[1], U-Net[2], and SeqSleepNet[3] ?

- Very often, sleep staging only uses a few channels, even 1. Spatial information is not crucial for classifying sleep. Having the motivation to add more channels in sleep staging is somewhat contradictory since one doesn't need a lot of channels to have a good prediction. And it is exactly what is shown in the spatial attention weight, where a lot of datasets are not taken into account. 
Nevertheless, the heterogeneity in the EEG field is still an open question, and the proposed CAM is very interesting, and I would like to see some results on other modalities to see if it also helps.

[1] Wang et. al., CareSleepNet: A Hybrid Deep Learning Network for Automatic Sleep Staging, 2024
[2] Perslev et. al., U-Sleep: resilient high-frequency sleep staging, 2021
[3] Phan et. al., SeqSleepNet: End-to-End Hierarchical Recurrent Neural Network for Sequence-to-Sequence Automatic Sleep Staging, 2018

### Questions
- What is the final ratio that you get with the uncertainty weighting?
- It is not clear what the meaning of 5%, 10% or 100% of labeled fine-tuning is. Do you split by subjects and then use a percentage of training subjects?
- How did you choose the $c^*$  parameters ?
- The ablation study shows that using the uncertainty weighted loss is useful. I would like to see if the CAM is useful. Right now, it is not clear if it is the weighted loss that helps or the fact that you train on more datasets than the dataset-specific pretraining. Adding an MTSSRL-MD train only on one dataset or training MTSSRL-MD without CAM on a subsample of common channels between the three datasets to see if the results are still high.

- How is the split done? I didn't find the information.

### Soundness
1

### Presentation
3

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
The paper introduces MTSSRL-MD, a unified framework for multi-task self-supervised representation learning (SSRL) from EEG signals across heterogeneous datasets. The approach incorporates a Channel Alignment Module (CAM) to align montages, three self-supervised pretext tasks (augmentation contrastive, temporal shuffling discrimination, and frequency band masking), and an uncertainty-weighted multi-task loss. MTSSRL-MD is empirically evaluated on three public EEG sleep stage datasets improvements over baseline methods.

### Strengths
1. The paper's motivation is clear. It targets the difficulty in learning robust EEG representations due to label scarcity and dataset heterogeneity.
2. MTSSRL-MD is efficient in terms of parameters, memory, and latency, making it well-suited for real-time clinical deployment compared to prior models.

### Weaknesses
The primary concern with this paper is that the authors appear to be largely unaware of recent advancements in EEG foundation models [1].

1. The core mechanisms introduced in this work—namely, the channel alignment module, shared multi-channel encoder, and multi-task self-supervised loss—have been extensively explored in prior studies [2–4]. As a result, the paper’s technical novelty is limited.
2. The authors employ SleepEDF-20, ISRUC-S1, and ANPHY-Sleep as pretraining datasets. However, these datasets are relatively small and do not fully leverage the advantages of self-supervised learning. In contrast, recent EEG foundation models are typically pretrained on much larger datasets such as TUEG, which provide broader generalization and robustness. Consequently, the practical implementation in this work is weak and may limit the impact of the proposed approach.
3. In the experimental section, the only self-supervised baseline considered is SeqCLR (2020). Given the substantial progress in EEG foundation models in recent years, the authors should include more recent and competitive baselines, such as LaBraM [2] and Cbramod [4]. Without such comparisons, the evaluation is insufficient and does not convincingly demonstrate the advantages of the proposed method.

[1] Wu, Jiamin, et al. "Adabrain-bench: Benchmarking brain foundation models for brain-computer interface applications." arXiv preprint arXiv:2507.09882 (2025).

[2] Jiang, Wei-Bang, Li-Ming Zhao, and Bao-Liang Lu. "Large brain model for learning generic representations with tremendous EEG data in BCI." arXiv preprint arXiv:2405.18765 (2024).

[3] Yi, Ke, et al. "Learning topology-agnostic EEG representations with geometry-aware modeling." Advances in Neural Information Processing Systems 36 (2023): 53875-53891.

[4] Wang, Jiquan, et al. "Cbramod: A criss-cross brain foundation model for eeg decoding." arXiv preprint arXiv:2412.07236 (2024).

### Questions
1. Did the authors conduct a literature review beyond 2022 to ensure coverage of recent developments in EEG pretraining?
2. What specific innovations distinguish this work from existing methods that use similar architectures or objectives?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper proposes MTSSRL-MD, a unified framework for multi-task self-supervised representation learning (SSRL) across multiple EEG datasets. The framework aims to improve generalization under label scarcity and heterogeneous EEG montages by integrating three components:

(1) Multi-dataset learning, to leverage diverse EEG sources and mitigate overfitting to specific datasets;

(2) Channel Alignment Module (CAM), a spatial attention mechanism that projects heterogeneous montages into a shared channel space;

(3) Multi-task SSL optimization, combining contrastive, temporal order, and spectral masking tasks with uncertainty-weighted loss balancing.

Experiments are conducted on three heterogeneous sleep-staging datasets (SleepEDF-20, ISRUC-S1, and ANPHY-Sleep) to validate the framework.

### Strengths
The research direction is valuable, addressing the problem of cross-dataset EEG representation learning, which is a crucial step toward robust and scalable EEG analysis.

### Weaknesses
**1. Unclear problem framing.**

While the paper’s objective is generally understandable, the introduction lacks a clear articulation of the core challenges motivating the use of multi-task learning and explicit montage modeling. It remains unclear what specific limitations the authors refer to when stating that “These strategies show the promise of dataset fusion but also its limitations.” The specific limitations of previous strategies should be explicitly defined, and the authors should further clarify why explicit montage alignment is necessary or beneficial, especially considering the potential inter-subject variability and the fact that implicit alignment methods may already capture spatial dependencies to some extent.

**2. Limited novelty.**

Each component of the proposed framework has clear prior art, and the main contribution lies in the integration of existing ideas rather than introducing substantial algorithmic novelty.

**3. Task scope limitation.**

Although framed as a unified framework for EEG representation learning, the experiments focus solely on sleep staging. Evaluating on additional EEG tasks (e.g., seizure detection, emotion recognition) would better support claims of generality.

**4. Unclear advantage of CAM.**

While CAM is central to the proposed framework, the paper does not quantitatively isolate its contribution or demonstrate why explicit alignment outperforms other approaches. In particular, recent foundation models for EEG representation learning, such as MMM and CBraMod, have already proposed more systematic solutions for montage alignment and domain generalization. The paper does not include comparisons with these representative methods and therefore fails to show a clear advantage or competitiveness in handling montage heterogeneity.

**5. Outdated baselines.**

The latest baseline referenced in this paper is from 2022 (ATCNet). For a fair and up-to-date evaluation, the authors should include comparisons with stronger and more recent sleep-staging baselines, as well as SSRL baselines (particularly EEG foundation models), published between 2023 and 2025.

**6. Minor presentation issues.**

- Line 112: “Self-supervised learning (SSRL)” should be “Self-supervised representation learning (SSRL)”.
- Section 3.4.1 is a single subheading and should be merged or renumbered.
- Page 8–9 reference jump issue.
- Code is not provided.

### Questions
1. Can the authors clearly define the core challenge and explain how each proposed component directly addresses it?

2. The three SSRL tasks are commonly used in self-supervised learning. The paper should clarify what advantages this particular combination offers and how their complementary effects are demonstrated.

3. How transferable is MTSSRL-MD to other EEG domains such as seizure detection or emotion classification?

4. Why choose uncertainty weighting over alternatives like GradNorm or PCGrad for task balancing?

5. Can the authors theoretically or empirically demonstrate why this kind of explicit montage alignment yields stronger generalization than others?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes MTSSRL-MD, a framework for learning generalizable EEG representations across heterogeneous datasets for sleep stage classification. The method combines: (1) a Channel Alignment Module (CAM) to handle heterogeneous EEG montages, (2) three complementary self-supervised learning tasks (augmentation contrastive, temporal shuffling discrimination, and frequency band masking), and (3) uncertainty-weighted multi-task optimization. Experiments on three public sleep datasets (SleepEDF-20, ISRUC-S1, ANPHY-Sleep) demonstrate improvements over single-dataset SSRL and multi-dataset baseline SeqCLR, particularly in low-label regimes.

### Strengths
- The paper clearly articulates three critical challenges in EEG-based sleep staging: (a) label scarcity, (b) small dataset scale, and (c) montage heterogeneity. 
- Experimental validation is quite comprehensive (multiple baselines, multiple metrics, different amounts of labels)
- Considerable improvements over SeqCLR and single-dataset SSRL baselines, while maintaining competitive performance at full supervision and achieving superior computational efficiency
- Results are supported through visualizations of CAM spatial attention

### Weaknesses
- Limited novelty in individual components of MTSSRL-MD (e.g., three SRRL tasks, CAM) The main contribution appears to be the engineering combination of existing techniques rather than fundamental algorithmic innovations.
- Limited amount of comparisons (SeqCLR is the main method and rather old), graph-based and feature fusion approaches are discussed but not empirically compared. 
- Only three data sets are tested, all for sleep staging
- Ablation studies quite limited (only with respect to multi-task weighting strategies).

### Questions
- Can you provide a systematic ablation study that isolates the individual contributions of CAM and multi-task learning? Specifically, please report results for Basline, CAM-only, Multitask-only, full model. 
- Why were recent multi-dataset EEG methods not included in the experimental comparison?  Comparing only against SeqCLR is insufficient to claim state-of-the-art performance. 
- Have you evaluated MTSSRL-MD on any EEG tasks beyond sleep staging? E.g., seizure detection?
- When does MTSSRL-MD fail? Are there subjects or sleep patterns where the method struggles?
- Interpretability: Have you consulted with sleep clinicians or neurophysiologists to validate that the CAM spatial attention patterns (Fig 2) are clinically meaningful?
- How does MTSSRL-MD scale with increasing numbers of datasets? You tested 3 datasets - what happens with 5, 10, or 20 datasets?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a self-supervised learning framework for EEG for a multi-dataset, multi-task context. It uses a Channel Alignment Module to handle different EEG montages. While the paper is clearly written and addresses the important problem of cross-dataset generalization, its contributions are significantly undermined by a severe lack of relevant baseline comparisons and ablation studies being absent for one of its primary methodological proposals.

### Strengths
- The paper is well-written and clearly articulates its methods and aims. The goal of creating a unified SSL framework that is cross-domain is important to the field.
- The combination of multiple pretext tasks which have been shown to perform well is a solid choice; combining this with the uncertainty weighting of these multiple objectives is very sensible and is shown to improve effectiveness.

### Weaknesses
- **Severe lack of baseline comparisons**: A primary claim of the paper is the benefit of the multi-dataset approach, yet the paper does not compare against any recent multi-dataset SSL methods for EEG. Examples would have included EEGPT, BIOT, Labram, CBramod, among others. A single comparison to a workshop paper from 2020 (SeqCLR) is simply insufficient to judge the method's performance relative to the literature. Similarly for single-dataset baselines; These comparisons are also insufficient, relying on methods from 2021. To make the current paper relevant, it would need to either show that a combination of methods by Banville et al. (2021) is at least as good as current methods, or provide insight into how using and weighting multiple pretext tasks interacts with current/modern pretext strategies.

- **No ablation or analysis on the Channel Alignment Module**: A central component of the paper is the Channel Alignment Module yet no ablation studies are performed to validate its effectiveness. Although it aims to address precisely the same issue other cross-domain methods tackle (e.g. tokenizers in Labram or Cbramod), no comparisons either theoretical or empirical are provided.

- **Limited methodological novelty and/or insight**: The paper largely combines existing pieces (known pretext tasks, uncertainty weighting). The only potentially novel piece is the CAM, which is unverified. As the paper stands, its only clear contribution is showing that multi-task pretraining is useful, which has now been shown in multiple publications. The work fails to provide new and interesting insights for the field and feels incomplete as is.

- **Conflation of contributions**: The paper appears to marry the concepts of pretraining tasks and single vs multi-dataset pretraining. These are fairly independent from my understanding; SeqCLR can be performed on single datasets and singular Banville et al. methods can be applied in a multi-dataset setting. The paper would be stronger if it decoupled these contributions and analyzed them independently.


In its current state, the paper feels seriously incomplete. The lack of crucial baseline comparisons and any ablation or analysis of the proposed Channel Alignment Module makes it impossible to validate the paper's claims or understand its relevance. The paper does not offer sufficient new insight to warrant acceptance.

### Questions
1. To make the paper more relevant to the field, are you able to present analyses for pretraining with modern methods? (Either as comparisons or added in during multi-task pretraining)

2. Did you perform any analyses into the channel alignment module? 

3. Do the authors see any problems with disentangling the pretext tasks from single- vs multi-dataset training? If not, why was this not done?

Minor:
- Figure 2 Presentation: Figure 2 appears to be raw matplotlib output with very small fonts, making it difficult to read. Here the presentation could be improved.

### Soundness
1

### Presentation
2

### Contribution
1
