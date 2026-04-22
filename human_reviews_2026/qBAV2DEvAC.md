# Implicit bias produces neural scaling laws in learning curves, from perceptrons to deep networks

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 2, 4, 8, 8

## Abstract
Scaling laws in deep learning -- empirical power-law relationships linking model performance to resource growth -- have emerged as simple yet striking regularities across architectures, datasets, and tasks. These laws are particularly impactful in guiding the design of state-of-the-art models, since they quantify the benefits of increasing data or model size, and hint at the foundations of interpretability in machine learning. However, most studies focus on asymptotic behavior at the end of training. In this work, we describe a richer picture by analyzing the entire training dynamics:  we identify two novel \textit{dynamical} scaling laws that govern how performance evolves as function of different norm-based complexity measures. Combined, our new laws recover the well-known scaling for test error at convergence. Our findings are consistent across CNNs, ResNets, and Vision Transformers trained on MNIST, CIFAR-10 and CIFAR-100. Furthermore, we provide analytical support using a single-layer perceptron trained with logistic loss, where we derive the new dynamical scaling laws, and we explain them through the implicit bias induced by gradient-based training.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper first studies dynamical scaling laws in perceptrons (single neurons) under a teacher-student setting. The authors find that perceptreons undergo a particular pattern of generalization error vs weight norm (or loss smoothness, which is in a sense dual to the weight norm). In particular, the authors find power-law scaling curves during different training regimes. This is shown both theoretically and validated empirically. Inspired by this, the authors then empirically evaluate generalization error vs weight norm in deep networks. The authors empirically make two observations: 1) for multiple dataset sizes, learning curves are initially similar but then diverge later on depending on dataset size, 2) the scaling law exponent at convergence can be predicted from other statistics of the learning curve, a finding found in the perceptron as well.

### Strengths
On the positive side, the paper shows that looking at generalization error vs weight norm curves reveals interesting and consistent patterns across datasets, architectures and other training hyperparameters. The perceptron theory also appears sound and links to empirical observations on deep networks, which is notable. Empirically, the authors consider 3 architectures and 3 datasets which gives confidence in the robustness of their results.

### Weaknesses
The paper suffers from a few key weaknesses in my view. First, the perceptron theory introduced in this paper is not particularly novel or surprising: plenty of prior works have investigated dynamical scaling laws for linear networks in a student-teacher setting. Of course, the particular setting considered by the authors is different from prior work (for instance, the choice of a logistic loss with variable sharpness), but it's unclear what value this setting offers relative to prior work.

Second, (as the authors acknowledge in Section 4) the connection between the perceptron and deep networks is weak: the only real similarity in my mind is that they are both models with some notion of norm and are trained with gradient-based methods. This concern could be remedied if the authors could somehow show that deep networks somehow behaved like perceptrons more mechanistically (beyond just looking at their generalization behavior).

Now, these first two weaknesses could be put aside if the perceptron model provided strong empirical predictioons about scaling behavior in deep networks. Unfortunately, the results are lacking here as well: the paper's main result 1 is purely qualitative. There could plausibly be many other models of learning in a deep network that obey main result 1. Main result 2 is stronger, but again the results are not completely convincing: looking at table 1, for certain results, the predicted and actual $\gamma$ look outside the range of the variability $\sigma$. Moreover, main result 2 doesn't make prediction about the scaling laws of the networks from scratch; instead, it really just evaluates whether the scaling law obeys a certain property ($\gamma=\gamma_1 \gamma_2$) predicted by the perceptron model. Again, there could be other models of deep network learning that also obey this property.

In summary, at the current stage, the paper is hampered by its weak connection between the theory and experiments. I recommend making stronger mechanistic connections between the perceptron and deep network as well as stronger empirical predictions for the deep networks.

Minor comments:
- "perceprons" on lines 56-57
- Figures 3 and 4 are too small

### Questions
- Stronger mechanistic connections between perceptron and deep network
- Stronger empirical predictions for deep networks
- See minor comments above

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper attempts to connect the learning dynamics of neural networks to the evolution of weight norms. Inspired by a simple perceptron model, where the norm directly relates to the scaling of the test loss, the authors introduce a spectral complexity measure for the weights of deep networks. They show that the loss curves of equally sized models across different dataset sizes $P$ collapse when plotted against this spectral complexity measure.

### Strengths
This paper proposes an interesting hypothesis: that some complexity norm of a neural network should be directly related to performance and summarize scaling law behavior. This hypothesis is motivated by the perceptron model where they provide substantial evidence of this effect. They also examine a variety of norms for deeper networks in the Appendix to find which of them is most promising in summarizing scaling behavior.

### Weaknesses
While this paper studies a very interesting hypothesis, there are a few concerns. 

**Connection between Norms and Performance in Deep Models** It is not apriori obvious how the spectral complexity measure connects to performance. In the perceptron model, it is clear that only the weight norm and the weight overlap with the target direction fully capture the generalization performance, but this is not known for deeper models. 

**Online vs Offline Training** Is the relationship between norms and loss primarily driven by offline training? 

**Large Parameter Limits** It is unclear how the spectral complexity measures depend on total parameter count which is also often scaled jointly with training time (total processed tokens = batch size * steps ).

### Questions
1. Is the collapse of the risk curves at large $P$ surprising? Shouldn’t all curves collapse to identical dynamics as you approach gradient descent on the population risk? 
2. Have the authors thought about how to characterize models of different sizes? Many scaling law results consider jointly increasing parameters and data (steps). Are the complexity measures well defined when parameters diverge? My suspicion is that the loss should still be reasonable as you approach a mean-field infinite width limit but the complexity measures may diverge. 
3. Is there any distinction between the behavior of networks trained in the online regime where data points are not repeated (where I expect losses to be monotone with steps) and training in this multiple-pass setting? How many epochs are required to see these overfitting effects?
4. Do normalization layers (which don't encourage scale growth of weights) change the picture provided here?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Neural scaling laws predict test error as a function of training properties (architecture, data, compute, ...). However, they typically predict the test error at the end of training, not during training itself. This article provides scaling laws for neural networks (CNNs + ViTs trained on MNIST / CIFAR for image classification) that relate test error to the model's norm. The results show highly systematic and predictable behavior. The paper's specific insights from these experiments are neatly summarized at the beginning of Section 4.

Reviewing caveat: while I'm somewhat familiar with the area overall (neural scaling laws), I'm not familiar with the theory in this area and therefore can't provide an informed assessment of the math.

### Strengths
- outstanding figure design: very clear and compelling figures
- very well written article, a pleasure to read
- neat combination of theory and experiments across different model families
- deriving scaling laws that predict test error during training, instead of just at the end of it, could potentially be a useful contribution (though I have some questions below). Either way, the observation that generalization error is highly related to model norm is interesting.

### Weaknesses
1. **No ImageNet scale results**: since not all results from small and toy-scale datasets like MNIST and CIFAR generalize to larger settings, an easy way to improve the reader's trust in the presented results would be to include at least an ImageNet-scale training setting. With libraries like FFCV (https://github.com/libffcv/ffcv-imagenet), ImageNet training can be done within less than an hour.

2. Please correct me if my understanding is wrong, but **does relating the model's norm to test error means that we essentially can't do a prediction of test error ahead of time?** In classic scaling laws, a big advantage is that we can predict the (a priori unknown) test error as a function of settings like dataset size / compute that we can determine ahead of time. In the case of the model norm as investigated in this article, we might need to train the network to determine its norm and predict generalization error?

3. Related to #2, it remains a bit unclear (to me) why the findings matter, beyond a general scientific contribution that improves our understanding of how different properties relate to each other. To be clear if it's "just" a scientific contribution that's perfectly valid too, but the paper's impact could potentially be strengthened by clearly explaining - or better yet, demonstrating - **why the findings matter in practice** (for people looking to train their networks), if at all. For example, the article mentions that we can "use the norm as a measure of training time", but provocatively speaking, couldn't we also just use a wallclock for that purpose?

### Questions
Beyond the questions mentioned above:
1. Is the relationship between norm and generalization error of a correlational or causal nature? If causal, in which direction?
2. Related to question #1: what happens if the norm is regularized during training, does this increase / decrease generalization error as predicted by the scaling law or does the relationship break if one intervenes on the norm?


MISC:
- the authors describe testing "CNN, ResNet and ViT architectures" - this sounds a bit odd since ResNets are CNNs.
- caption formatting for Table 1 as centered is an unusual choice
- line 58: "at the corresponding fixed norm" - this sentence was a bit unclear to me, perhaps there's a way to rephrase.
- citations within a sentence should be "as shown by Authors (YYYY)" not "as shown by (Authors, YYYY)" if they're part of the sentence

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors find interesting connections between scaling laws and implicit bias in SGD to come up with new scaling laws for models with logistic loss, where learning curves as a function of the model’s increasing norm. They additionally show that these findings generalize across architectures and datasets in computer vision.

### Strengths
- the connection is interesting, elegant and theoretically well motivated
- the intuition builiding using the perceptron model is helpful
- theory is empirically confirmed across architectures and datasets
- I think this work has connections to several empirical observations in LLM scaling law fitting which are interesting, even though the authors stick to vision datasets.
- work is well presented

### Weaknesses
- The piece-wise scaling law proposed is not new and has been empirically demonstrated before: https://arxiv.org/abs/2210.14891 
- The paper would benefit from a literature review of related scaling law work. Eg.
  - https://arxiv.org/abs/2210.14891 
  - https://arxiv.org/abs/2304.15004
- Experiments are on a single model scale and it is not clear if it holds for LLMs and generative models where scaling law research is more useful
- From related work and observed practice (https://arxiv.org/abs/2502.18969), outliers early in training are dropped. Connecting this to the proposed form would greatly improve the draft and I'd consider raising my score if this is done.

### Questions
- How you think these findings would translate to LLMs? Do you think the norm justification will still hold, or you will need to change the form? In practice, early training points/low FLOP datapoints of LLMs are dropped while fitting scaling laws (https://arxiv.org/abs/2502.18969). 
- What determines where the elbow is? data quality/scale/model capacity etc?

### Soundness
2

### Presentation
4

### Contribution
3
