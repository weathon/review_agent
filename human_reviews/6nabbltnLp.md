# Joint or Disjoint: Mixing Training Regimes for Early-Exit Models

- Decision: Reject
- Scores: 5, 5, 5

## Abstract
Early exits are an important efficiency mechanism integrated into deep neural networks that allows for the termination of the network's forward pass before processing through all its layers. 
Early exit methods add trainable internal classifiers which leads to different training dynamics. However, there is no consistent verification of the approaches of training of early exit methods and little understanding how training regimes optimize the architecture.  Most early exit methods employ a training strategy that either simultaneously trains the backbone network and the exit heads or trains the exit heads separately. 
We propose a training approach where the backbone is initially trained on its own, followed by a phase where both the backbone and the exit heads are trained together. Thus, we categorize early-exit training strategies into three distinct categories, and then validate them for their performance and efficiency. 
In this benchmark, we perform
both theoretical and empirical analysis of early-exit training regimes. We study the methods in terms of information flow, loss landscape and numerical rank of activations and gauge the suitability of regimes for various architectures and datasets.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper analysis the

### Strengths
1. Early exiting is a very important research topic to achieve efficiency. Focusing this topic is valuable.

2. This paper provide a understanding of the early-exit neural networks, which may be useful for other researchers.

3. The author do both image and language experiment.

### Weaknesses
1. The novelty is limited. The proposed joint / disjoint / mixed training sound naive. Although the authors provide some analysis for early-exit networks, the proposed methods and experiments looks have few relation with these these analysis.

2. Inproper baseline network and evaluation choice. And I think it is the biggest problem. I am curious why the author mainly follow the practice in SDN [11], rather than follow the practice of MSDNet [9]. It is apparent that MSDNet have a more clean and reasonable architecture for early-exit, a very clean training setting, and a more systematic evaluation method for early-exiting. 

    2.1)  The disadvantages of directly add early exits in resnet (as the practice in SDN) have been very througthly discussed in MSDNet paper. And MSDNet have a much stronger performance than SDN in a very clean training setting. I think the authors should do their experiments in more SoTA architectures.

    2.2) The line of MSDNet works [9, 7, 19, 32] provide a more reasonable evaluation method for early-exiting networks. They evaluation the networks in Budgeted Training and Dynamic Inference schemes. In the Budgeted Training scheme, they will calculate the threshold for each exits in the training set, and they use these thresholds to do evaluate in eval/test sets. However, the way SDN evaluate their model looks much naive. Furthermore, this submission "set 100 evenly spaced early-exit confidence thresholds" (as mentioned in line 319), is not very reasonable   compared with MSDNet.

3. The training setting is not clear and maybe infair. When the authors compare disjoint / joint / mixed training, it seems they have not keep the total training epoch (or some other method to evaluate training cost) the same. As a result, I am doubtful for their results.

4. The training hyper-parameter is also confuing. For example, in sec. D.3, the author claim they train 1500 epoches for efficientnet in line 791, while in line 791 they say they train efficientent for 200 epochs.

5. Lack of experiments.

    5.1) For image experiment, I think the results in imagenet-1k is very important. While the authors sometimes do experiments in CIFAR10, and sometimes in CIFAR100, limited ImageNet-1k results is provided. 

    5.2) I also does not understand the way they choose CIFAR 10 or 100 in some small ablations.

    5.3) The authors do not compare they method with related works.


Minors:

1) Line 379: Imagenette --> ImageNet

2) Line 773: Imagenette --> ImageNet

### Questions
1. How the author claim " Disjoint and mixed regimes produce similar models, while the model trained in joint regime lies in a different basin" in Fig. 2? What is the x axis and y axis means in Fig. 2? If the distance in Fig. 2 mean something, it looks like the distance between the three points is similar. If you think a very high loss "mountain" separate  joint and the other two points, I think the loss "mountain" may means nothing in this space.

2. How the MODE CONNECTIVITY findings motivate the authors to design these methods?

3. How the numerical rank is computed in each layer? Why the rank will have a ~3000 rank? What network this experiment use?

4.  How the NUMERICAL RANK findings motivate the authors to design these methods?

I would raise my rating if the author give a reasonable explanation and necessary additional experiments for my comments and questions in the weakness section and questions section.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper studies different learning regimes (joint, disjoint) that could be followed when training the base model (backbone) and the additional internal classifiers. In this regard, the paper proposes a “mixed” regime, which follows a warming-up type of approach, where the backbone is first trained, and then, the internal classifiers are added and trained together with the backbone.

On the more theoretical side, the paper analyses the learning dynamics behind these regimes. On the empirical side, experiments on image and text classification problems based on several backbones show the capabilities of the proposed method at different computational budgets.

### Strengths
- A a high-level, the paper is very clear. There are no  barriers getting on the way towards understanding the problem addressed by the paper and its proposed solution.
    
- The empirical validation of the proposed method covers different data modalities, i.e. images and text. As a beneficial consequence, different datasets (CIFAR’10/100, ILSVRC12, Newsgroups and ) and models/architectures. This helps the reader get a good overview of the capabilities of the proposed method.
    
- Results seem to be reported over different runs, i.e. 4 according to Sec. 4 (l.323)
    
- The empirical evaluation is complemented with a more theoretical analysis of the effect of the considered training regimes.

### Weaknesses
- W1: weak positioning; Good part of the related work (l.430-441) is centered on discussing Early Exiting Networks without focusing on the training regime aspect, the core of the contribution put forward by the paper.
    
- W2: While the proposed mixed regime seems to outperform the classical joint approach under some circumstances, the technical novelty seems to be relatively reduced and some what comparable to existing techniques used to train multi-component networks. A comparison wrt. to these could help position the proposed method and stress further its novel aspects.
    
- W3: From the reported results, the proposed method seems to be less suitable for the setting of interest, i.e. the one with reduced computational budget. Moreover, the improvement of the proposed mixed regime over the classical joint strategy following other exit strategies, e.g. the entropy exit criterion (Sec. 4.2, Fig, 10) does not seem be that clear anymore.
    
- W4: Some observations made by the paper seem rather anecdotal. For instance, in Sec. 3.1 (l.130-146) some observations are made regarding the relative locations between loss values from models  trained following the considered regimes. Similarly, in several places (l.125, l.267-269, etc.) there are some statements regarding performance of input samples with different level of difficulty (e.g. easier vs. difficult to classify). It is unclear however, how prevalent/frequent these observations hold in the different problems/models/datasets that are considered. A supporting quantification of this aspect would be a proper companion to these statements.
    
- W5: The proposed method seems to be currently tested only in classification problems. Experiments on regression problems would provide further evidence on the applicability of the proposed method.
    
- W6: The content of the paper is too verbal at times, a more formal presentation of the considered training regimes would make more clear what are the different factors that are behind and influence one or the other. This would also help throw further light into how training would be affected by the selection of one or the other regime.
    
- W7: In its current form, the paper provides almost no details on the classification problems (and related datasets) that were considered on the empirical evaluation. This would not only be desirable for unfamiliar readers, but it would also serve as a point to verify whether the paper follows the standard or its own protocols, and ensure reproducibility of the reported results.
    
- W8: There are some inconsistencies in how models/datasets are used in some of the reported experiments. For instance, in some cases only specific model/dataset combinations are considered (Sec. 4.1 Fig.6 & 7). In other cases,  a given model. e.g. ViT is only trained on CIFAR-10 (Fig. 8) and in other cases on CIFAR-100 (Fig. 9). Results from Sec. 4.3 Fig.11 are limited only to the ViT model and CIFAR-10 dataset. A similar focus occurs on (Sec. 4.4, Table 1). Given this, it is hard to assess to what level the difference in performance are generalizable accross other settings that the specific combinations reported in the paper.

### Questions
[Suggestion] Regarding W1 and W2, a positioning wrt. to the iterative approaches like those used in GANs (Goodfellow, 2014) , R-CNN based detectors (Ren, 2017), and other multi-component models (Haidar, 2023) would be beneficial in this context?

[Suggestion] Regarding W4, quantifying how prevalent the stated observations are present in the conducted experiments could provide better grounds to support such statements. In a similar manner, I would suggest defining the difficulty of the samples of the considered datasets, quantify where these sample groups exit in the models and find the relationship of this wrt. the considered regimes.

[Suggestion] Regarding W8, I would suggest conducting experiments on all the possible combinations of the considered datasets/models. Certainly the page limitations will not allow adding all of them in the body of the paper, but the additional/supporting results could be part of the supplementary material.

References

- Goodfellow et al., "Generative Adversarial Nets",  NeurIPS 2016
    
- Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks", Transactions of Pattern Recognition and Machine Intelligence (T-PAMI) 2017
    
- Haidar et al., "Training Methods of Multi-label Prediction Classifiers for Hyperspectral Remote Sensing Images", Remote Sensing 2023.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper presents a new method to improve model efficiency by early exits. Previous methods in this line usually train the backbone and head classifiers at the same time (joint scheme), or separately (disjoint scheme). This work argues that they will impair the performance, so they propose to train the backbone first and then both the backbone and head exit networks together, sort of a method in between of the previous two paths. Experiments across various architectures and datasets show the effectiveness of the method.

### Strengths
1. The early exits methods for model efficiency is practical and interesting. And their method has been motivated by grounded observations.
2. The method is associated with a theoretical analysis from the lens of mode connectivity. Although I think the "theoretical" part can be more grounded and rigorous, the intent and attempt are valuable.

3. Empirical results suggest the method is effective against other counterparts.

### Weaknesses
1. One problem with the experiments is that the paper does not include evaluations on relatively large-scale datasets like ImageNet-1K. Many papers have noticed that the conclusions on CIFAR is hard to generalize to ImageNet-1K, so the results on ImageNet-1K are encouraged.

2. Methodologically, the paper method looks too simple technically and too intuitive. One sign that the paper lacks *real* technical contribution is that it has zero equations - only one, if any, is in page 4 without indexing. The paper claims to "conduct theoretical analysis". Sorry to say it is hard to see where the "theory" is rigorously defined or introduced. With this missing, the paper has 9 pages, 1 page shy of the max 10 pages, which is of course okay, but somehow tells us that the paper appears to be rushed out.

3. In most results, the performance advantage over mixed training is quite marginal. Ie, the results are not strong.

4. Some of the results look strange. Why in Fig 7(a) does the disjoint scheme perform unusually better than the others at large FLOPs?

Minior writing or presentation issues:
- ”joint” regime -> “joint”， ”disjoint” regime ->“disjoint”  -- many of the quotes are in wrong format.

**==== Post Rebutal ====**

I thank the authors' response. Unfortunately, the presented new results are not convincing to me.  I mentioned before the results are not strong. The authors "respectfully disagree with this statement" and argued "in most of our results the mixed regime provides statistically significant improvements", with Fig. 9 as support. Also, they have the new ImageNet-1K results in Fig. 7.

The problem with these results is: they are all reported by the authors; critical details are unclarified, and the performance is far below the standard ones.
- The original Tiny-Vit on ImageNet-1K without pertaining can reach over 78% top1 accuracy (see https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810068.pdf, Fig. 1), but in this paper, the authors only report ~70% (see Fig 7 of this paper). I wonder if the experiment was conducted following the standards.
- Similar problem, in Fig. 1 of this paper, the reported Tiny Vit only reached ~54% accuracy on CIFAR100, which is unusually low too. And no details about how the Tiny Vit is adapted for the CIFAR100 dataset.

Without these critical details, the claimed performance advantage is hard to verify. And nearly all the comparison baselines are from the authors instead of the existing papers. There is no clear evidence so far that these results are trustworthy. Given this issue, and the shallow technical novelty, I maintain my score at weak rej.

### Questions
NA

### Soundness
3

### Presentation
2

### Contribution
2
