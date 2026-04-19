# Generative Classifiers Avoid Shortcut Solutions

- Decision: Accept (Poster)
- Scores: 8, 8, 5, 6

## Abstract
Discriminative approaches to classification often learn shortcuts that hold in-distribution but fail even under minor distribution shift. This failure mode stems from an overreliance on features that are spuriously correlated with the label. We show that generative classifiers, which use class-conditional generative models, can avoid this issue by modeling all features, both core and spurious, instead of mainly spurious ones. These generative classifiers are simple to train, avoiding the need for specialized augmentations, strong regularization, extra hyperparameters, or knowledge of the specific spurious correlations to avoid. We find that diffusion-based and autoregressive generative classifiers achieve state-of-the-art performance on five standard image and text distribution shift benchmarks and reduce the impact of spurious correlations in realistic applications, such as medical or satellite datasets. Finally, we carefully analyze a Gaussian toy setting to understand the inductive biases of generative classifiers, as well as the data properties that determine when generative classifiers outperform discriminative ones.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper revisits generative classifiers to address the problem of spurious features that affect discriminative classifiers trained with cross-entropy loss on out-of-distribution data. Leveraging class-conditional generative models, based on diffusion for image related problems and autoregressive models for text related problems, the authors show that generative classifiers outperform discriminative classifier both in-distribution and out-of-distribution on various benchmarks. The authors also propose some analysis on a toy task.

### Strengths
* While the concept of generative classifiers is not new, applying recent generative models to the challenging out-of-distribution generalization problem is timely and interesting. The paper is well written and has a strong overall message.

* The paper provides evaluation on several datasets covering both image and text-related problems. The improvement due to generative classifiers over discriminative baselines seems important in both out-of-distribution and even in-distribution scenarios.

* The authors provided intuitions why the generative classifiers performs better; in particular I found the analysis on the gradient norm and the toy evaluation well lead and insightful.

### Weaknesses
* Comparison to classifier with pretraining and data augmentation: although pretraining does not seem to improve the proposed model, it does have a big importance for standard classifiers; in particular, could the authors show (e.g. on one chosen image-based dataset) that finetuning a pretrained ResNet-50 classifiers (on imagenet) does not reduce significantly the gap with the considered approach ? Moreover, applying data augmentation is also a standard trick for image-based classification. Although its application to the generative classifiers is still unclear (as mentioned in the conclusion), was it included in the discriminative models reported in the paper ?
* Multi-class classification: the authors seem to have mostly included binary classification tasks; does the proposed model scale to multi-class classification problems ? If yes, including some evaluation on a multi-class problem could add some value to the experimental evaluation (e.g. DomainBed - https://arxiv.org/abs/2007.01434 - provides some multi-class image-based benchmarks where comparison can be made with several invariant based approaches).

### Questions
See above.

===Post rebuttal===

I thank the authors for the rebuttal; I recommend acceptance.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper shows that generative classifiers always perform better in the realistic OOD setting, which can be implemented easily based on existing class-conditional generative models. They conduct simulations to show that generative classifiers can model both core and spurious, instead of mainly spurious ones. They also present the data properties that affect when generative classifiers outperform discriminative ones.

### Strengths
1. The paper is novel and well-written.
2. The paper shows that generative classifiers have advantages compared to discriminative classifiers, both in ID and OOD settings. I think this is a fundamental challenge to existing discriminative learning, which is significant.
3. The experiments are sufficient to validate the proposed hypothesis.

### Weaknesses
1. I think some recent papers about the superiority of generative classifiers should be mentioned in this paper, including [a,b,c].

[a] Diffusion Models are Certifiably Robust Classifiers, NeurIPS, 2024

[b] Robust Classification via a Single Diffusion Model, ICML, 2024

[c] Revisiting Discriminative vs. Generative Classifiers: Theory and Implications, ICML, 2023

### Questions
1. In line 322, why do you choose epoch 5? Can this affect the final results?
2. In line 341, the increase of gradient norms makes me confused. It should decrease during the training because the gradients will converge to 0.
3. In line 364, the paper claims that parameter count does not seem to be responsible for the performance of the generative classifier. Does this mean, when considering downstream discriminative tasks, we can not enjoy the scaling law of generative foundation models?

If you can address the weaknesses and questions sufficiently, I will improve the rating.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper explored the ability of generative classifiers to handle domain shifts. They use diffusion models and AR models for images and text data and show significant improvement in OOD classification with their classifier compared to other baselines.

### Strengths
- The paper handles the important topic of OOD generalization in machine learning
- The method shows good performance on several benchmarks
- The paper is well written

### Weaknesses
- The main issue is limited novelty. Using generative models for classification isn't a new idea, and previous works already explored their robustness to OOD generalization and to adversarial attacks (and also the use of diffusion models as classifiers. For example Zimmermann et al "Score-Based Generative Classifiers", Grathwohl et al "YOUR CLASSIFIER IS SECRETLY AN ENERGY BASED
MODEL AND YOU SHOULD TREAT IT LIKE ONE", Chen et al "Robust Classification via a Single Diffusion Model" and Fetaya et al "Understanding the Limitations of Conditional Generative Models". The paper should also add them as related works.

- While I like the use of simple toy examples, I don't think there was any insight or useful observation from the whole section. For example, the observation that "As σ increases, the core feature becomes less reliable, and the generative classifier uses the spurious and noise features more" is quite expected and doesn't give any added value. This section also is only relevant to generalization so it mirrors the results in Ng&Jordan 2001. 

- Also regarding the Illustrative setting, the feature you name $x_{spu}$ is not a spurious feature as it has a direct causal relationship by the label (as you designed it so). It just isn't as useful for discrimination as the "core" feature. 

- Regarding model size in the experiments, table 2 is also on CivilComments so that only covers one dataset. Moreover, on this dataset you performed worse than the discriminative models on the ID while you performed better than the other models on 3/4 of image datasets. This shows that the CivilComments behavior might not transfer. Also, parameter count isn't an exact comparison.

- In two of the datasets, waterbirds and FMoW you can see in Fig. 2 that the ID and OOD seem to be very highly correlated when looking over all models. Looking at these plots, especially FMoW, it is not clear at all that the better OOD performance isn't just because your base model had better ID performance and not because of some innate property of generative models. 


Small remarks:
- L28 you write "Ever since AlexNet (Krizhevsky et al., 2012), classification has mainly been tackled with discriminative methods" but this isn't accurate as discriminative models have been the standard approach even before AlexNet, just mainly SVMs/decision trees/boosting etc.

### Questions
NA

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper explores the performance of generative classifiers as compared to discriminative classifiers under distribution shift. It first documents the performance of both types of models on 5 standard benchmarks for distribution shifts concluding that generative classifiers indeed perform better under the distribution shift but surprisingly also often in in-distribution classification. It argues that this advantage stems from the generative classifiers having to learn all instead of just the short-cut features for max-margin classification induced by the cross-entropy loss. It further explores the behavior of the two classifier types on a synthetic toy problem controlling for the ratios and variance of spurious, core and noise features and provides useful insights into the effects of these data properties through outlining "generatlization phase diagrams". This investigation leads to no free lunch conclusion stating that the advantage of generative vs discriminative classifier is a function of the data properties which is in line with the results of the benchmark experimentation.

### Strengths
The empirical investigation in the paper corroborates some rather well-accepted intuitions of potential benefits of generative classifiers in distribution-shift scenarios and thus provides useful support for future work in this direction. The systematic toy-problem exploration helps to understand the trade-offs and uses some tools that could be adapted for exploration of more complex scenarios in the future. Some further useful results are presented in the appendix (perhaps worth at least referencing in the main text?). The paper is well balanced in its conclusions, contributing to general understanding of the investigated phenomenons. The paper is well written and easy to follow.

### Weaknesses
It is unclear to me, how the conclusions documented on the rather simplistic and well controlled toy-problem could be translated to more complex practical problems under realistic scenarios. This is, however, acknowledged by the authors themselves.

### Questions
1. The superior ID performance of generative models in Table 1 suggests to me that perhaps the discriminative models are too weak. For example, the original Sagawa paper reported the ID accuracy of discriminative classifiers for the Waterbird dataset between 94-97% (depending on regularization) and close to 95% for the CelebA. If this were true, to comparison would be unfair and not very useful. This feeling of "unfair / skewed comparison" is further reinforced by the comment in page 7 - discriminative classifier for CivilComments is larger and it indeed performs better on ID. Can you comment and clarify, how you ensure the fairness of the comparisons? If the comparison is indeed not fair, what does this mean for the presented results and the conclusions we can take from it?
2. In page 5 you speak about "class-balanced accuracy". I am not sure how this is defined. Can you please clarify?
3. Could  not the evolution of the gradient norm in Figure 3 simply suggest that the discriminative classifier learns faster?
4. Section 6.3 Figure 6 - you fix $\sigma^2_{noise} = 0.36$. Why particularly this value? Is this somehow representative of the behavior? Would the graphs look similarly at different noise levels?
5. Section 6.3 Figure 6 uses the feature scale B=1? Or some other value?

### Soundness
3

### Presentation
3

### Contribution
2
