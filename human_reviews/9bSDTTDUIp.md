# ERM++: An Improved Baseline for Domain Generalization

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 5, 3

## Abstract
Multi-source Domain Generalization (DG) measures a classifier's ability to generalize to new distributions of data it was not trained on, given several training domains.  While several multi-source DG methods have been proposed, they incur additional complexity during training by using domain labels. Recent work has shown that a well-tuned Empirical Risk Minimization (ERM) training procedure, that is simply minimizing the empirical risk on the source domains, can outperform most existing DG methods. ERM has achieved such strong results while only tuning hyper-parameters such as learning rate, weight decay, and batch size. This paper aims to understand how we can push ERM as a baseline for DG further, thereby providing a stronger baseline for which to benchmark new methods. We call the resulting improved baseline ERM++, and it consists of better utilization of training data, model parameter selection, and weight-space regularization. ERM++ significantly improves the performance of DG on five multi-source datasets by over 5% compared to standard ERM using ResNet-50, and beats state-of-the-art despite being less computationally expensive. We also demonstrate the efficacy of ERM++ on the WILDS-FMOW dataset, a challenging DG benchmark. Finally, we show that with a CLIP-pretrained ViT-B/16, ERM++ outperforms ERM by over 10%, allowing one to take advantage of the stronger pre-training effectively. We will release code upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes ERM++, which introduces several techniques to improve the ERM baseline on multi-source domain generalization. The techniques consist of that for training data utilization, model parameter selection, and weight-space regularization. Experiments on 5 widely-used domain generalization datasets with different backbones show the effectiveness of the method.

### Strengths
1. The paper investigates many techniques for the ERM model to improve multi-source domain generalization, which helps other researchers in this field to find suitable methods for their research.

2. The paper provides rich experiments to show the benefits of the utilized techniques for domain generalization.

### Weaknesses
1. Although the paper includes many techniques to improve the ERM model on domain generalization, most of these technologies have been proposed or are widely known. The improvement with these techniques is not surprising and inspiring.

2. The paper argues to propose a general baseline for future domain generalization works with the existing techniques. However the experiments, such as Table 3 (a), Table 4, and Table 5 show that different techniques benefit different datasets or settings, which is not general to the domain generalization task.

3. Except for the averaged accuracy, it is better to provide more insight of different techniques on how they benefit or harm the performance of domain generalization on different datasets.

### Questions
See weakness.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
A training protocol containing three main components is introduced to compose ERM++ which is a new baseline proposed in the submission. By observing the critical of the training length, a two-stage training procedure is conducted by determining the training length with the source domain validation set performance while training on the source training set in the first stage and then in the second stage training on the whole set. Model weight initialisation and weight-space regularisation, namely MPA application, are also studied. For the weight space regularisation idea, I hold my option for the later discussion. ERM++ is evaluated on a large variety of DG benchmarks with significant improvement comparied with vanilla ERM.

### Strengths
1. The experiments are dense and comprehensive. The proposed baseline is evaluated on most existing domain generalisation benchmarks comprehensively. 
2. ERM++ explore existing practical technique tricks in DG with a detailed description.

### Weaknesses
1. The name is a little misleading as MPA is part of ERM++. If this is the case, it is natural to wonder what ERM + MPA performance will be like and by removing MPA from ERM++, then how ERM++ will perform. I think the closest setting is Table 4. Once MPA is added, for example comparing #1 and #2, the performance boosts significantly. The rest setting cumulates each tech one by one. But it is also important to know whether each one contributes independently. 

2. One of the main points made in DomainBed is that without a complicated algorithm design with fair hyperparameter running ERM is a very strong baseline. However, ERM ++ is way more complicated than ERM. 

3. Besides, since, it is justified by the submission that MPA works well with other introduced tricks, also it is good to know whether ERM++ is compatible with other advanced optimisation algorithms like SAM, GASM, SAGM, which is the benefit of using ERM. 

4. In terms of the training cost, ERM ++ is compared with other models such as MIRO and DIVA, but the comparison with ERM is more important.

### Questions
See the above sections.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work provides a detailed study of techniques used to improve the performance of empirical risk minimization (ERM) in domain generalization. Three categories of improvements are combined (data utilization, initialization, and regularization) to achieve state-of-the-art performance with lower computational cost than competing strategies.

### Strengths
1. The strategies used to maximize ERM performance for domain generalization are practically useful, and the resultant model is a competitive baseline for future work in the area.
2. Extensive experiments were performed with many different methods, architectures, and datasets. The ablation studies are also well done.
3. Careful analysis of the results were performed and edge cases were highlighted in the text. In particular, many of my initial questions were answered upon a closer read of the analysis, for example the discussion of VLCS performance in Section 5.1.

### Weaknesses
1. While the premise of the contribution - that ERM can match SOTA DG algorithms when appropriate data utilization, initialization, and regularization are applied - is important, the strong performance of ERM has been known since [1] and is not exactly novel. The main contribution of this paper is in applying recent “tricks of the trade” to further boost ERM numbers. While this may be helpful for practitioners, no new insight is offered as to why the proposed techniques are useful for ERM specifically or why ERM is a good baseline for domain generalization in the first place.
2. Many of the proposed improvements to ERM are not actually specific to ERM and can be applied to other state-of-the-art methods to improve performance. For example, it would be helpful to see a comparison where some methods in Table 2 are run with better initialization (say, AugMix) to see whether ERM still outperforms them. As is, the comparisons are made on somewhat unequal footing.
3. I believe the investigation of weight-space regularization is incomplete. First, why is ERM++ not run with SWAD, and is there any advantage of MPA in this scenario? Second, there is another category of weight-space regularization which is not included in the paper, namely sharpness aware minimization (SAM) [2] based techniques (e.g., SAGM [3]). SAM and SWA have been extensively compared [4] and found to each be beneficial in different circumstances. It would be interesting for the community to compare these two techniques in a DG setting, and I believe this experiment is necessary to claim a fully rigorous investigation of weight-space regularization for DG.

There is also a fair amount of confusing writing and typos, detailed in the next section.

### Questions
Here, I list some minor questions as well as suggestions for improving the writing.

1. The reference [5] cited in Table 2 should also be cited in the introduction.
2. The bar graph in Figure 1 is pixelated. If it was made with `matplotlib`, this can be fixed by setting the DPI or saving it as a PDF.
3. There is an inconsistent use of dataset names and abbreviations in the tables (e.g., TerraIncognita vs TI vs TerraInc vs TerraInco). I would recommend using the full name of the dataset everywhere, and perhaps reducing the font size when it doesn’t fit. The same goes for model names (e.g., Meal-V2 vs Meal V2 vs MV2).
4. There is an inconsistent use of spaces before citations, (e.g., Author(Citation) vs Author (Citation)). I encourage the authors to use the ~ LaTeX character to create a small space before the citations, and to keep this consistent throughout the text.
5. “Sketch” is misspelled in Table 6.
6. The headings in Table 7 are not explained. What are R0, R1, etc and P, I, Q, etc?

***Recommendation***

Overall, while this paper provides a useful benchmark on maximizing ERM performance for domain generalization, my concerns about novelty and the incomplete investigation of weight-space regularization cause me to lean slightly towards rejection rather than acceptance.

***References***

[1] Gulrajani and Lopez-Paz. In Search of Lost Domain Generalization. ICLR, 2021.

[2] Foret et al. Sharpness-Aware Minimization for Efficiently Improving Generalization. ICLR, 2021.

[3] Wang et al. Sharpness-Aware Gradient Matching for Domain Generalization. CVPR, 2023.

[4] Kaddour et al. When Do Flat Minima Optimizers Work? NeurIPS, 2022.

[5] Vapnik. An overview of statistical learning theory. IEEE Transactions on Neural Networks, 1999.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper present a new strong baseline, named ERM++, for the study of domain generalization (DG). By incorporating multiple previous results in training data utilization, parameter selection, and regularization, ERM++ achieved state-of-the-art performance on the DomainBed benchmark.

### Strengths
The obvious strength of the paper is the exciting results. The experimental settings are carefully described.

### Weaknesses
Despite presenting exciting results with elaborated experiments, the paper lacks technical insight into the effectiveness of various components, especially when they are used together. While I do see the merit of the engineering approach and agree that the field should appropriately acknowledge this as a baseline for large DomainBed, I do not think the current contribution of ERM++ is fit for a venue like ICLR. Thus, I cannot recommend acceptance for the paper.

### Questions
Why did the evaluation results for CMNIST and RMNIST not included in the paper?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
