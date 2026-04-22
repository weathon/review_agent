# sleep2vec: Unified Cross-Modal Alignment for Heterogeneous Nocturnal Biosignals

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 2, 6, 4

## Abstract
Tasks ranging from sleep staging to clinical diagnosis traditionally rely on standard polysomnography (PSG) devices, bedside monitors and wearable devices, which capture diverse nocturnal biosignals (e.g., EEG, EOG, ECG, SpO$_2$). However, heterogeneity across devices and frequent sensor dropout pose significant challenges for unified modelling of these multimodal signals. We present sleep2vec, a foundation model for diverse and incomplete nocturnal biosignals that learns a shared representation via cross-modal alignment. sleep2vec is contrastively pre-trained on 42,249 overnight recordings spanning nine modalities using a Demography, Age, Site & History-aware InfoNCE objective that incorporates physiological and acquisition metadata (e.g., age, gender, recording site) to dynamically weight negatives and mitigate cohort-specific shortcuts. On downstream sleep staging and clinical outcome assessment, sleep2vec consistently outperforms strong baselines and remains robust to any subset of available modalities and sensor dropout. We further characterize, to our knowledge for the first time, scaling laws for nocturnal biosignals with respect to modality diversity and model capacity. Together, these results show that unified cross-modal alignment, coupled with principled scaling, enables label-efficient, general-purpose modelling of real-world nocturnal biosignals.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The author introduces a foundation model for physiological signal data recorded during sleep time, that is modality agnostic. The model is pretrained using a contrastive loss to encourage subject-specific consistency in the representation space, along with a proposed Dash-InfoNCE objective to help the model distinguish between different input modalities. The proposed framework is evaluated on varied downstreaming tasks including sleep stage classification and disease classification. During the modeling process, varied prior works including specialized models serve as baselines, and ablation studies are conducted including scaling law verification, leave-one-modality-out validation.

### Strengths
- It is the first of this kind of model specifically for sensing signals during sleep that can address heterogeneous input signals (in terms of different sensor signals and varied number of input channels). 
- The evaluation objectives including stage classification and disease classification are representative in terms of revealing the underlying value behind the proposed framework. 
- The design of loss function seems rigorous and ingenious that it adequately addresses the issue of inter-subject variability and how the model handles heterogeneity in the input modalities, as shown by the visualization presented by the author. 
- The problem setup and the aim of the work is clearly depicted.

### Weaknesses
- On the methodology side, though the visualization of the embedding seems promising, there is lack of ablation study on different pre-training approaches, such as using vanilla reconstruction loss, or varied masking rate, etc. 
- Though scaling law is presented, only scaling with model size is provided, but there is a lack of result on scaling with data size. Also the difference in the scaling law results seems not significant just from the bar plot. Probably statistical test results are needed to strengthen the claim of contribution on this aspect. 
- The data statistics is a bit insufficient Specifically, some important information are missing, such as total hours of data or distribution of length of a night, etc. 
- It remains unclear whether the model’s ability to handle heterogeneous inputs comes from the backbone design itself or the proposed loss function.

### Questions
- The presentation of the main result is a little bit unclear. Results on SHHS and WSC are provided. Is it the case that other datasets don't have the valid corresponding tasks? If that is the case, it should be clearly stated somewhere, as this would also help readers understand how the data were split (e.g., the pretraining data were divided based on certain factors, etc.).
- The evaluation criteria is a bit confusing, where the fine-tuning setting is not presented very clearly. Is all the baseline fine-tuned using the same setting? Does a different fine-tune strategy might potentially alter the overall observation? 
- There are varied backbone model frameworks that could handle heterogeneous input signal series. Is the observation of the representation separation consistent across different backbone model? It would be clearer if more empirical evidence were provided to clarify whether the backbone’s modality-agnostic nature itself enables it to handle heterogeneous inputs, or whether this capability primarily arises from the proposed loss function design.

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
This paper proposes a unified cross-modal pretraining framework, Sleep2Vec, designed for multimodal sleep representation learning. Specifically, the framework jointly enhances model generalization and robustness through cross-modal contrastive learning and data reconstruction tasks. The authors evaluate Sleep2Vec across multiple downstream tasks such as sleep staging, respiratory disorder detection, and periodic limb movement detection.

### Strengths
1.	Multi-task pretraining objective: Combines reconstruction and contrastive objectives to strengthen inter-modal coordination and representation learning.

2.	Cross-modal modeling innovation: The modality reconstruction task effectively addresses missing-modality scenarios.

3.	Comprehensive experiments: Covers a wide range of datasets and tasks, demonstrating strong adaptability.

### Weaknesses
1.	Limited originality: The approach lacks novelty, as many prior works have already explored missing-modality and contrastive learning in sleep research, such as CIMSleepNet (NeurIPS 2024), MultiConsSleepNet (IEEE JBHI 2025), and SleepSMC (ICLR 2025).

2.	Outdated baselines: The comparison methods are mostly old, missing fair comparisons with the latest relevant works mentioned above.

3.	Lack of ablation studies: The paper does not explicitly analyze the independent contributions of each module.

4.	Unaddressed modality inconsistency: Although cross-modal reconstruction is used, the generalization ability under missing-modality conditions during training or testing is not sufficiently demonstrated.

5.	No cross-subject experiments: The paper does not evaluate subject-independent generalization, which is crucial for assessing clinical applicability in real-world sleep studies.

6.	Unclear experimental interpretation: For instance, Table 1 does not clarify whether the first column corresponds to training or inference settings.

7.	No statistical significance analysis: The experiments lack multiple runs or reported standard deviations, making the result stability uncertain.

Related Work

[1] Shen Q, Xin J, Dai B, et al. Robust sleep staging over incomplete multimodal physiological signals via contrastive imagination[J]. Advances in Neural Information Processing Systems, 2024, 37: 112025-112049.

[2] Pan J, Yu Y, Li M, et al. A Multimodal Consistency-Based Self-Supervised Contrastive Learning Framework for Automated Sleep Staging in Patients With Disorders of Consciousness[J]. IEEE Journal of Biomedical and Health Informatics, 2024.

[3] Ma S, Zhang Y, Chen Y, et al. SleepSMC: Ubiquitous Sleep Staging via Supervised Multimodal Coordination[C]//The Thirteenth International Conference on Learning Representations, 2025.

### Questions
Please see Weaknesses

### Soundness
2

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
3

### Summary
The paper proposes a new foundation model for handling multi-modal PSD data. Instead of classic FM for sleep staging, the proposed sleep2vec can take as input all channels from the PSG study. To do this the authors use cross-modal alignment to have a shared representation. 
Sleep2vec is compared to various SOTA models for either channel-specific tasks or full channels over more than 30.000 subjects.

### Strengths
- The paper is very easy to follow. 
- The paper proposes a new representation learning for all the channels of a PSG.
- The method is tested over a big corpus of subjects comprising more than 30.000 subjects. 
- With the gating mechanism, we can see which channels bring more importance for the classification, giving good interpretability of the model.
- Good t-SNR visualization that gives insight into understanding the use of the proposed method.

### Weaknesses
- Figure 2 introduces the Intra-subject and Inter-subject segments. This is never used in the entire paper. This additional information, in my opinion, is likely to lead to a misunderstanding of the method.
- The motivation is that no model deals with the full channels of PSG. In Table 1, two competitors are proposed for a full channel setting. Does that mean the model can handle all the channels? What is the addition of sleep2vec? 
- The competitors presented in Table 1 are never introduced. I understand that it can be challenging to describe everything in detail, but a description can be provided in the appendix, and at least FM can be properly introduced, such as SleepFM and PFTSleep. That can give better positioning to the literature.

### Questions
- What is the computation time of a full channel setup compared to an EEG-only setup? Considering all the channels, the results are improving slightly. I'm wondering if the additional computation time is justified? The same question arises when comparing FM to non-FM computation time versus score. It can bring more justification to why we use FM instead of a specialized model.
- If the feature fusion is interesting, as it gives intuition of which channels are more useful, did you compare the three fusion strategies that you introduced (Concat, AVG, and gating)?
- In the appendix, you showed that increasing the number of parameters always increases the performance. When do you reach a plateau ? Maybe a threshold between time computation and performances can be found?

### Soundness
2

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
The paper introduces Sleep2Vec, a foundation model for nocturnal physiological recordings that learns a shared representation space across nine polysomnography related channels, including high rate neural and ocular signals and lower rate cardiorespiratory signals. The core idea is to treat concurrent nocturnal signals as complementary views of the same latent physiological state and to enforce alignment at the level of short epochs through a contrastive objective. The authors train on more than forty two thousand overnight recordings collected from several large public cohorts and harmonized through a common preprocessing pipeline. They propose a metadata aware contrastive loss named DASH InfoNCE that adjusts the weight of negatives by demographic attributes, recording site and subject night identity in order to reduce cohort shortcuts and to keep very similar samples from dominating the denominator. They report results on sleep staging for SHHS and WSC, on clinical outcome prediction for several conditions, and they include an ablation on leaving out individual modalities. They also attempt to characterize scaling behaviour along both the number of modalities and model capacity. The work aims to show that unified cross modal alignment for sleep becomes practical once enough heterogeneous data are brought together.

### Strengths
Strengths
The paper addresses an important practical problem in sleep medicine and mobile or low burden monitoring, which is the presence of many possible channel layouts and frequent missing sensors. Showing that one pre trained model can handle nine different signal types and remain robust when some are absent is a meaningful step toward realistic deployment across devices and centers. The methodological core is coherent. The use of two modality batches, masking, a single backbone and a shared projection head creates a simple recipe that other researchers can reproduce. The proposed contrastive objective is well argued. It uses age, gender and site to prioritize harder negatives, which is a clever way to turn epidemiological and acquisition metadata into useful training signals. The experimental section is quite extensive. The authors compare to specialized sleep staging networks on SHHS and show that the gap becomes small, sometimes negligible, despite the fact that their model was trained to be general and not tailored to sleep staging alone. They also evaluate on lower information channel groups such as IBI and respiratory signals, where the proposed model still outperforms prior foundation models and even reaches or exceeds specialized approaches in some metrics. The leave one out analysis is particularly useful since it quantifies the relative importance of each modality and confirms known clinical intuition that EEG and IBI carry more discriminative information for stage recognition than some of the other channels. The analysis of clinical prediction tasks with increasing numbers of modalities is also valuable as it suggests a predictable scaling trend.

### Weaknesses
Weaknesses
Although the paper claims better cross site generalization through metadata aware weighting, the current experiments do not fully isolate this effect. It would be more convincing to show a split in which one cohort is entirely held out during pre training and used only for evaluation, and to show that the gap between the standard InfoNCE and DASH InfoNCE enlarges in that setting. At present, the evidence comes from aggregate metrics and from the claim that site and demographic similarity are properly captured in the loss.
The paper positions itself as establishing scaling laws for nocturnal biosignals but the current analysis is still rather preliminary. The number of model sizes and modality subsets is limited and the curves are shown mostly for downstream performance without a deeper look at loss scaling during pre training, data efficiency or breakpoints where adding modalities stops helping. A more systematic study would make this part of the contribution stronger.
The reliance on a RoFormer backbone is sensible, yet the paper does not compare to other recent time series architectures that are specialized for long physiological sequences, for instance models with channel wise attention or structured state space models, which might be competitive or more efficient.
While the dataset section describes harmonization in detail, the paper does not quantify the amount of missing channels per cohort, nor does it show per cohort downstream results. Since the main motivation of the work is robustness to sensor dropout and to heterogeneous montages, a more explicit evaluation on realistic missing patterns would be helpful.
Finally, the work composes several ideas that have each appeared in some form in prior sleep foundation models, for example large scale contrastive pre training on ECG and respiratory signals and modality aware fusion. The novelty therefore resides mostly in the combination at scale and in the specific loss design rather than in a single transformative idea.

### Questions
1.	Can you report results in a leave one cohort out setting, for example training on HSP, SHHS, MrOS and MESA and evaluating only on WSC, with and without DASH InfoNCE, in order to show that the metadata aware weighting is particularly helpful when the target cohort is unseen.
2.	How sensitive is performance to the choice of the age kernel bandwidth and to the relative weights for same gender and same site samples. An ablation where these hyperparameters are perturbed would clarify to what extent the proposed loss needs tuning for new populations.
3.	In the two modality sampling scheme for pre training, have you tried curriculum strategies where the model first sees frequent and informative modality pairs such as EEG with respiratory effort and later progresses to rarer combinations. If so, did this accelerate convergence.
4.	For clinical outcome prediction, are the labels balanced across cohorts. If some conditions appear mostly in certain centers, the site aware part of the loss could risk encoding site related biases. Please clarify how this was mitigated.
5.	Could the authors provide a short comparison in compute cost between this approach and a variant that simply reconstructs masked modalities, since one of the selling points is that the alignment objective is more suitable for flexible inference.

### Soundness
3

### Presentation
3

### Contribution
2
