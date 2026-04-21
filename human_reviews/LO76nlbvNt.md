# Performance Control in Early Exiting to Deploy Large Models at the Same Cost of Smaller Ones

- Avg Score: 3.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 3

## Abstract
Early Exiting (EE) is a promising technique for speeding up inference at the cost of limited performance loss. It adaptively allocates compute budget to data points based on their difficulty by exiting at earlier layers when predictions are confident. In this study, we first present a novel perspective on the EE approach, demonstrating that larger models, when deployed with EE, can achieve higher performance than smaller models while maintaining similar computational costs. As existing EE approaches rely on confidence estimation at each exit point, we further study the impact of overconfidence on the controllability of the compute/performance trade-off. We introduce Performance Control Early Exiting (PCEE), a method that enables accuracy thresholding by basing decisions not on a datapoint's condfidence but on the average accuracy of samples with similar confidence levels from a held-out validation set. In our experiments with MSDNets and Vision Transformer architectures on CIFAR-10, CIFAR-100, and ImageNet, we show that PCEE offers a simple yet computationally efficient approach that provides better control over performance than standard confidence-based approaches, and allows us to scale up model sizes to yield performance gain while reducing the computational cost.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes a simple way to modify the decisions made by early exiting networks. Basically, the predicted “confidence” scores by the exit layer blocks are transformed into estimated accuracy scores, based on a precomputed histogram of mapped values. Since the histogram can be computed from an already-trained early-exit network, the authors describe it as a post-hoc approach. The method is evaluated on CIFAR-100 in the main paper for image classification.

### Strengths
I think that improving early exit networks, and more generally, dynamic neural networks, is an important and meaningful direction, and is worthy of investigation.

### Weaknesses
The contributions and novelty of the paper seems to be rather far from the level required at ICLR. In particular, early exiting networks have been around for a long time, and simply proposing to better map the predicted “confidence” scores to accuracy based on a pre-computed histogram from a held-out set is rather incremental. In short, the proposed process is very simple -- compute the average accuracy of samples at a certain confidence score, and use the average accuracy as a signal instead of the predicted “confidence” scores. This could be a small trick to use to better calibrate the early exit networks, but definitely not something enough for a full paper.
Since the proposed method is too simple, there is nothing much to comment about it. 

Besides this, there are also various serious weaknesses in the experiments. These weaknesses are likely not be able to fixed within the rebuttal window.


Firstly, the number of compared baselines is very small. Only two existing methods are really being compared with. Yet, early exit networks have existed for quite a long time, and there should be a long list of baselines to compare with. 

Secondly, the authors also used a single “backbone” early exit network MSDNet (albeit at three different sizes). This baseline is already quite dated, as it comes from a 2017 paper. Besides the fact that only one early exit network “backbone” is used (which is not convincing), the relevance of the experiment results are also in doubt, since they are applied to an old architecture. For instance, do these results transfer well to newer architectures?


Furthermore, only results from one dataset (CIFAR-100) is reported in the main paper. Yet, the reported results in Table 4 on CIFAR-100 does not even show very significant improvements.  For instance, for the accuracy results on MSDNet Small setting, the improvements over TS+Confidence is often not so significant.

Moreover, in Table 4, these results have been reported over a very small range of threshold values (delta), which is 0.65-0.71 for the small model and 0.67-0.73 for the large model. In fact, not only are these threshold values over a small range, they are also different between the different model settings, suggesting that these values have been cherry-picked. These results are not convincing to me. Why have these threshold values been selected? Also, can the authors provide results on a wider range of threshold values?


At the same time, here are some other suggestions to improve the relevance and significance of the paper:

Experiments can be done on more recent early exit networks, such as Transformer networks. For instance, at various parts of the paper, the authors discuss speculative decoding as another way of improving inference efficiency, and speculative decoding is most often applied on Transformers, and also often on text decoding tasks. However, Transformers were not even experimented on at all.

Since the proposed method is very general, it does not need to be only tested on image tasks. For example, it could (and also should) be easily tested on text or video-based modalities as well.

### Questions
Please refer to “Weakness” section for the concerns/questions.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This article proposes an accuracy-based early stopping method called PCEE to address the issue of accuracy degradation caused by overconfidence in confidence-based early stopping methods. PCEE achieves this by constructing a reliability diagram (confidence-to-accuracy mapping) to convert the confidence output at the early stopping point into accuracy, and determines whether to stop early based on whether the accuracy exceeds a threshold.

### Strengths
1. This article provides extensive experimental evidence of the existence of overconfidence phenomena in ViT and CNN models.

2. This article proposes a simple yet interesting idea of mapping confidence to accuracy to avoid the issue of overconfidence.

### Weaknesses
1. The paper employs a reliability diagram to implement an accuracy-based early stopping mechanism through constructing a confidence-to-accuracy mapping. If that is the case, why not directly output the accuracy of each layer's classification of the input samples as the basis for determining whether early stopping is needed? An ablation study on this would be beneficial.

2. Figure 3 shows the results of overconfidence in a 12-layer ViT; is there a similar phenomenon in CNNs, such as MSDNet?

3. In Table 4, although PCEE achieves better accuracy than the baseline oracle, the Avg Layers and Avg FLOPs it uses seem to be higher than those of the oracle, which does not adequately validate the accuracy of PCEE. In other words, although the accuracy of the Laplace or Confidence methods is lower, their Avg Layers and Avg FLOPs are also lower.

4. If the network does not employ early stopping, what is the output accuracy? How does it compare with the accuracy after implementing early stopping? Is there any loss in precision, and if so, how much?

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper introduces PCEE (Performance Control Early Exiting), a method that ensures a lower bound on accuracy by thresholding based not on a datapoint’s confidence but on the average accuracy of its nearest samples from a held-out validation set with similar confidences. This approach offers a simple yet computationally efficient alternative that provides control over performance, facilitating
accurate adaptation of EE methods for practical use. Comparisons have been carried out with the existing baselines on CIFAR-10 and CIFAR-100.

### Strengths
- The paper is clear and well-written
- The initial analysis regarding the model accuracy and confidence is informative.

### Weaknesses
- Authors mention that the proposed method is evaluated on the samples with similar confidence from a held-out dataset. How is it ensured that the held-out dataset has similar confidence trends as the test sample ?
- Is PCEE evaluated at each layer? How much computation overhead does it add in terms of time? Did the authors try to employ PCEE on some selected layers only?
- The application scope of the paper is limited. The authors have tested the approach in classification settings, that too only on 2 datasets. It is important to verify the effectiveness on other deep learning tasks as well. 
- From the Table 4, the performance of the proposed method does not seem to be consistent as the baselines seem to be overperforming in many cases.
- Overall, while the paper is explained nicely, there are major issues regarding the scope of application on more tasks and datasets and subpar results.

### Questions
Please check the weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2
