# On the memorisation of image classifiers

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 6, 3

## Abstract
The success of modern neural models has prompted study of the connection between memorisation and generalisation: such models generalise well, despite being able to perfectly fit (“memorise”) completely random labels.  To more carefully study this issue, recent work proposed an intuitive metric to quantify the degree of memorisation of individual training examples, and empirically computed the corresponding memorisation profile of a ResNet on image classification benchmarks.  While an exciting first glimpse into what real-world models memorise, these studies leave open a fundamental question: do larger neural models memorise more? We present a comprehensive empirical analysis of this question on image classification benchmarks. Intriguingly, we find that training examples exhibit a diverse set of memorisation trajectories across model sizes: while most samples experience decreased memorisation under larger models, surprisingly, there is a dichotomy between the remaining samples. In particular, per-example memorisation trajectories reveal that examples exhibit decreasing or cap-shaped memorisation, and some examples even exhibit increasing memorisation. We further show that various memorisation proxies fail to capture such fundamental characteristics as we vary model size. Lastly, we find that knowledge distillation — an effective and popular model compression technique — tends to inhibit memorisation, while also improving generalisation. Intriguingly, we find that the memorisation is mostly inhibited on examples for which memorisation increases as the model size increases, thus pointing at how distillation improves generalisation by limiting memorisation of such examples.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper studies model memorization with respect to model size. They define memorization score per sample as the change in model prediction for that sample when it is removed from the training set, versus when it is used for training. They study how such scores change as model size increases for CIFAR and ImageNet, and  show that higher capacity models tend to make memorization more "bi-modal" in the CIFAR case, ie for some datapoints memorization increases with model complexity but for others not. They further study how distillation affects memorization and show that it limits memorization of hard examples.

### Strengths
* The paper provides a study for memorisation with respect to model capacity, something that to my knowledge is novel and important to investigate.

* the memorisation trajectories analysis seems interesting

* measuring the effect that distillation has on memorization is also an interesting study to conduct

### Weaknesses
It seems to me that the "non-intuitive behaviour" observed for CIFAR where "memorization drops" for larger models is an artifact of how easy this task is and the definition of the memorization score from Eq 2. There memorization is defined as the in-sample minus the out-of-sample accuracy: for all models > ResNet 20, in-sample accuracy is at 100% for CIFAR and cannot increase further. 

There are a couple of more indicators of the above in the paper: this is not observed for any dataset where performance is not saturated (eg ImageNet, or even CIFAR with distillation for ResNets <=54). And this also explains the fact that the same analysis using cprox doesnt show the same behaviour. It is also clear to me from Fig 5, for the distillation case. Now that perfomance is saturated over ResNet56, the memorization score only start mildy dropping above that for CIFAR-100. 

It is therefore I think an exargeration to claim from this experiment that "memorization is dropping for larger models", as this behaviour is not also obaserved in other cases, eg  Fig2b for Imagenet or  even CIFAR with distillation, ie whenever performance is not saturated. Any claims on "less" or "more" memorization are unjustified. Therefore the titles of sec 3 and sec 3.2 is to me misleading. 

This extends to 3.3, where the "bimodal" finding only shows on CIFAR, possibly for the same reason that doesn't have much to do with actual memorization. Imagenet shows no such bi-modality. 

In conclusion, the non-intuitive findings of Section 3.2 and 3.3 seem to me to be an artifact of performance saturation rather than an effect of memorization. Therefore the overall claims of this section are to say the least highly exaggerated and, if the above are correct, misleading (the section title is "the unexpected tale of memorization"). To quote the authors from their conclusions section: "one should be careful with using certain statistics as proxies for memorisation".

Other notes, and suggestions to improve a study I still find very interesting to conduct properly:
- Currently there is no analysis on the most ubiquitous architecture: ViT. A small analyisis on whether this holds or not for basic ViT sizes (S/B/L) would make this submission stronger. Whether it holds there or not is not as important as to see what happens. 
- The analysis only uses basic data augmentation from ResNet, ie the one presented in He et al 2016. Since data augmenation is crucial for generalization, the study would be stronger if stronger data augmentation was used.

### Questions
Q1: Did you observe the memorization score drop in any case/dataset where performance is not saturated? Same for the bi-modality: Is there any clear indication that this is not an artifact of performance saturation?

Q2: How were the curves for Figure 1 (also Fig 21,22,23) obtained? are they really the curves after dropping that one specific image and retraining, or the protocol of K=400 still in place? If the latter is true, then any of the 400 images would give the exact same curve, right?

### Soundness
1 poor

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors perform a study of memorisation across different scales of image classifiers, concluding that there exist various subsets of samples whose memorisation properties are cohesive as the scale of the models change. The authors also propose distillation as a way to prevent memorisation.

### Strengths
- The analysis is extensive and well presented. 
- The paper contains a good revision of the literature about memorisation.

### Weaknesses
- The authors discuss how distillation could be used to prevent memorisation. Could the authors compare their work to [1], which suggests distillations should not be considered as a countermeasure to memorisation? The distillation discussion is indeed the most controversial part of the paper, I think.
- Did the authors consider studying the memorisation trends for Differentially Private trained models?
- The conclusions of the analysis are somehow trivial, but well presented and discussed. Although the novelty of studying how it scales through architectures is a bit marginal. 
- The paper is lacking the usage of more recent architectures (e.g. ConvNeXt, SwinTransformers etc.) therefore it is unclear whether the empirical findings generalise to modern architectures. 




[1] https://arxiv.org/pdf/2303.03446.pdf

### Questions
- Could the authors show a similar analysis for more modern architectures to verify the scaling occurs similarly for this type of models?
- Did the authors consider the idea of inserting canaries simulating the most problematic samples (e.g. by increasing the sensitivity of the loss by mislabelling for the category of samples that gets more and more memorise) and studying how these get memorised so as to draw a parallel between the "naturally" found memorised samples and the induced ones? This would "confirm" the author's discussions more rigorously (e.g. by showing intentionally misclassified samples behave in a certain way as scale increases).

### Soundness
2 fair

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studied the memorization effect of deep neural networks on training samples with different model sizes. It found that the memorization trajectory behaviors of examples can be divided into four types: constant, increasing, decreasing, and cap-shaped, which cannot be captured by some proxies for memorization. Further, it exploits these different behaviors of various examples to explain the generalization of larger models and distillation methods.

### Strengths
1. The studied problem about the memorization and generalization of DNN is important for understanding deep learning algorithms.
2. The highlighted memorization trajectories provided a new view to observe the behaviors of examples, which also explains the generalization of larger models to some extent.
3. The writing of this paper is very good, which makes readers without the knowledge in this area easy to understand.

### Weaknesses
1. The conclusions about generalization provided by this work are based on the hypotheses validated by experimental results, and it would be better if these conclusions could be proven through theoretical analysis under certain cases.
2. The statement that "The increasing examples are often multi-labeled or mislabeled" can be justified more clearly. For example, using a dataset without multi-labeled or mislabeled instances to conduct the experiments.
3. Could the authors discuss some reasons that these examples will be observed by four different types?

### Questions
See above weaknesses.

### Soundness
2 fair

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper studies the memorization in image classification networks. The authors consider a particular way to quantify memorization from prior work (for most of the experiments), and study various aspects of memorization; e.g., (i) how does the $overall$ memorization change with deeper networks, (ii) how does the distribution of memorization of individual examples look? (iii) what $kinds$ of memorizations take place in these networks etc. The authors also use an alternative proxy for memorization (again, from prior work) and study if the trend is similar with the earlier metric.

### Strengths
1. Understanding memorization will be important for the community to have a better understanding of why neural networks work/don't work. To this end, the motivation of trying to study memorization in a more principled/controlled setting will be helpful. 

2. The authors have tried to dissect the memorization on many different levels; from the overall memorization $\rightarrow$ trying to study the distribution of memorization scores for individual examples $\rightarrow$ understanding what kind of examples get memorized more with bigger models (and which do not). 

3. In the main paper, the results are presented on two popular datasets - CIFAR100 and ImageNet.

### Weaknesses
1. The main conclusion from Section 3.2 is that the larger models (more depth) might $not$ memorize more, as was previously hypothesized. This conclusion is drawn from the results on just one of the two datasets - CIFAR100. However, the results from ImageNet (Fig. 2b), tell the opposite - that bigger models do indeed memorize more than smaller ones. It is not clear how the authors are concluding that larger models memorize less. Based on the results, I certainly think that the title of this section should change and should be a question rather than an assertion.

2. The conclusions from the next section (3.3) are again confusing. In Fig. 3; I do not see the bi-modal nature of memorization scores. In most of the plots, for both CIFAR and ImageNet, the highest concentration is near 0 and then the remaining scores are (i) somewhat uniformly distributed across values between 0 and 1 for CIFAR (it often does not even have much concentration at 1); (ii) monotonically decreasing from 0’s bin towards 1’s bin. While I agree that with the increase in model depth the memorization score is shifting towards 1 (both for CIFAR and ImageNet); again, I do not think this is bi-modal depiction of scores (which I interpret as the bar graph having a U-shaped outline). 

3. If I understand correctly, the main conclusion from section 3.4, in simple terms, seems to be that hard examples (hard = having multiple labels or incorrect labels) tend to get memorized increasingly with increasing model size. If that is true, in what way do the results tell us anything new about the nature of hard examples, which do get memorized, as has been shown by prior works (cited by the authors themselves)?

4. There are many links to appendix all throughout the paper (e.g, in Section 3.4). A paper should be self sufficient in itself. Currently, because of the nature of investigation that the authors are interested in, it seems unlikely that a reader can get a comprehensive understanding of different conclusions simply from the main paper.

5. Overall, the paper feels like an amalgamation of multiple different analyses, but I cannot seem to figure out a coherent story out of it. Even beyond the confusion about the conclusions from multiple sections, it is not clear what a reader should take away from this paper.

### Questions
1. Fig. 5 right is not clear. How are the memorization scores of the distilled and independent student (x and y axis respectively) in the range of 15-20 when the same scores on the left plot (and elsewhere in the paper) are less than 1?

2. In Tables 1 and 2, alpha is the "change in memorization". How exactly is this change measured? Is it the change measured between the smallest and the largest model?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
