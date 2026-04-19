# Self-Normalized Resets for Plasticity in Continual Learning

- Decision: Accept (Poster)
- Scores: 5, 6, 8, 3, 5

## Abstract
Plasticity Loss is an increasingly important phenomenon that refers to the empirical observation that as a neural network is continually trained on a sequence of changing tasks, its ability to adapt to a new task diminishes over time. We introduce Self-Normalized Resets (SNR), a simple adaptive algorithm that mitigates plasticity loss by resetting a neuron’s weights when evidence suggests its firing rate has effectively dropped to zero. Across a battery of continual learning problems and network architectures, we demonstrate that SNR consistently attains superior performance compared to its competitor algorithms. We also demonstrate that SNR is robust to its sole hyperparameter, its rejection percentile threshold, while competitor algorithms show significant sensitivity. SNR’s threshold-based reset mechanism is motivated by a simple hypothesis test we derive. Seen through the lens of this hypothesis test, competing reset proposals yield suboptimal error rates in correctly detecting inactive neurons, potentially explaining our experimental observations. We also conduct a theoretical investigation of the optimization landscape for the problem of learning a single ReLU. We show that even when initialized adversarially, an idealized version of SNR learns the target ReLU, while regularization based approaches can fail to learn.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper presents a reset-based approach to continual learning, where a network's neurons are reset based on a function of their inactivity. A different threshold function is applied to each neuron. The proposed method is compared to other reset schemes as well as regularization-based approaches. Experiments show promising performance on several datasets (such as MNIST-adjacent ones). Some theory is presented in a setting for learning a single target ReLU.

### Strengths
The proposed method appears to outperform the selected baselines on a variety of datasets and exhibits good dependence on its sole hyperparameter. The experiments also appear to be rather thorough. For that reason, I think the proposed method may be practical.

The writing is also strong.

### Weaknesses
The experiments appear to be in a rather small setting, which is a slight problem since I believe the paper's main contribution is an experimental one.

Section 4 (theoretical analysis) studies an exceedingly simplified setting. There is no notion of distribution shift or changing tasks (as the expectation is taken with respect to \mu rather than a \mu_t), and the algorithm is full gradient descent, which is not only impractical in real-world settings but impossible without knowledge of the underlying distribution. The second (subtracted) term in the regret also appears to be 0. The restrictions on v in Theorem 4.1 seem rather restrictive and I don't know if they are necessary. It's also unclear if the algorithm suggested in Theorem 4.2 is executable - are the variables with 'min' and 'max' known a priori? 

No conclusion section.

If the authors provide strong answers to my concerns, I would be happy to raise my score.

### Questions
Section 2 - Each (x_t, y_t) is drawn from a separate distribution \mu_t? Is there any relationship between the different \mu_t?
Line 134 - This looks like dynamic regret, why is that (pessimistic) objective considered?
Line 136 -  Is this the definition of plasticity? That the averaged dynamic regret grows faster than a constant?
Line 144 - What is the formal definition of a 'dead' or 'inactive' neuron?
Line 153/5 - The definitions of Z and A are unclear. What exactly is s indexing? Which two consecutive activations for A?
Line 155 - 'let let' typo
Line 194 - what guarantees do you have on this approximation? \mu_t seems to be the distribution at a particular time, for which you only have one sample or a batch of samples? Why is considering 'firing times' while training on different, past, tasks/distributions a principled thing to do?

How exactly is A_i^{\mu_t} approximated? I would like more detail than footnote 2.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce Self-Normalized Resets (SNR) which is an adaptive algorithm that resets neuron weights when there is a drop in its firing rate in order to reduce plasticity loss. They also conduct a theoretical investigation of the optimization landscape for learning a single ReLU neuron. Their empirical results show that while existing baselines struggle in terms of sensitivity and detecting inactive neurons, SNR is robust to hyperparameter tuning, and can result in better performance.

### Strengths
- As resetting neurons is a well-known technique in addressing plasticity loss, the main novelty of this paper lies in its exploration of the question of 'how to reset' the neurons. If I understood correctly, by focusing on the *recent* activity of a neuron rather than activity within a uniform period of time, SNR tries to answer the question by selecting individual neurons to reset by being more sensitive to the distribution shifts.
- Their empirical results (especially Table 1) clearly show better average accuracy on a variety of benchmarks. In addition, SNR is better in terms of robustness to its hyper-parameter as compared to other baselines.

### Weaknesses
- While I understand the adaptive nature of SNR in selecting neurons, wouldn't it also result in over-dependence on the type of distribution shifts across tasks? The vision experiments, in particular, are primarily conducted on benchmarks where a task is uniformly sampled. As a result, the boundaries (or shifts) between consecutive tasks are relatively uniform as well. But what if instead of uniformly, the tasks arrive based on increasing/decreasing (or both) levels of noise and difficulty [1]? A controlled experiment could be conducted to analyze how the nominal firing rate of that neuron changes based on such distribution shifts. 
- The authors described their method SNR as an alternative way to pick neurons to reset them. However, in their experiments, they did not compare SNR with Redo and CBP in Table 2. Also, the standard deviation is crucially missing in Table 1. 
- The paper appears to be incomplete i.e., it ends abruptly after section 4 (with theorem 4.2) without providing any discussion points, limitations, and conclusion. 


There are several grammatical/clarity issues - 
- Line 215 - tacking the histogram
- Lines 57 - 60 can be explained better. 
- Line 269 - Shouldn't it be 1000 classes?
- Line 377 - seems redundant after the previous para

### Questions
- How would SNR's test accuracy (on the held-out dataset) compare to other reset methods? Recent works have been focusing on generalization performance showing that while reset-based methods may result in better training accuracy, they often struggle in terms of generalizability and are outperformed by regularization methods like L2 [1].
- Why do some of the methods in Table 1 results show a significant drop in performance compared to others? Is it dependent on the size/type of hyper-parameter search?

**I am willing to increase the score if the authors address the concerns and questions raised.**

[1] Hojoon Lee, Hyeonseo Cho, Hyunseung Kim, Donghu Kim, Dugki Min, Jaegul Choo, and Clare Lyle. Slow and steady wins the race: Maintaining plasticity with hare and tortoise networks. arXiv preprint arXiv:2406.02596, 2024.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work presents a new method for avoiding plasticity loss in continual learning settings (where neural networks lose their ability to learn new tasks after being trained on other tasks). The proposed method is called Self-Normalized Resets (SNR) and simply involves choosing a threshold time of zero activations, so that any neuron that does not activate (has 0 output) for the threshold number of training iterations has its weights randomly reset during training. SNR is found to outperform other methods in a number of small continual learning image classification tasks, and combines with L2 regularization to outperform on a small natural language task.

### Strengths
__Originality:__ The idea of resetting neurons to avoid loss of plasticity is not particularly novel or surprising (as evidenced by the cited related work), but the proposed method appears novel and most importantly, satisfies multiple desiderata (simple, effective, robust) well beyond previous methods.

__Quality:__ Although on relatively small tasks, the experiments are thorough and convincing. The comparisons are made on a variety of datasets and consideration is made for hyperparameter tuning. A number of known covariates for plasticity loss (L2 complexity of parameters, dead neurons) are plotted to explain the effectiveness of SNR versus other methods.

__Clarity:__ The paper is overall well written and easy to understand (see Weaknesses for a few areas that need improvement).

__Significance:__ The results are only on small datasets, but nevertheless show promise (scaling Permuted Shakespeare gives a positive trend in terms of the method's superiority). The robustness of SNR to hyperparameter choice is very striking. To highlight this, I actually would recommend putting at least 1 figure or table in the main text to summarize one of the tables in Appendix C: for example, plotting hyperparameter strength vs accuracy on one CL task for the compared methods.

### Weaknesses
This paragraph is confusing: "This reveals an interesting comparison with SNR. The schemes above will re-initialize a neuron after inactivity over a period of time that is uniform across all neurons. In the context of the hypothesis testing setup above, this will result in sub-optimal error rates across neurons. On the other hand, SNR will *reset a neuron after it is inactive for a period that is effectively normalized to the nominal firing rate of that neuron*, while still only specifying a single hyperparameter for the network." - referring to the italicized part of the last sentence, doesn't SNR also reset at a uniform rate of $\bar{T}$? The only difference I can tell is that in the previously described method, neurons are reset all at once every $r$ minibatches, and it is possible for a neuron to be inactive for anywhere between $r$ and $2r-1$ minibatches before reset. If $\bar{T} = r$ then SNR only differs in resetting neurons at exactly $r$ minibatches, and resetting individual neurons at different times during training. Could the authors please clarify this point?

In Permuted Shakespeare, what is the actual task being optimized (e.g. classification, next word prediction, etc.)?

Notes on the presentation:
- a discussion/conclusion section is missing
- some figures have too much noise to be easily readable (e.g. figure 3) and could either have a lower sampling rate for the displayed points, or use a confidence interval to cover the range of variability.
- typo on "hyperparemter(s)." (line 331)
- typo in appendix B "with the [second] element fixed as 1" (line 723)
- typo in appendix B: $w_0$ should be $w_{min}$ in the second part of "We sample $w_0$ uniformly from $[−1, w_0 ) \cup (w_0, 1]$" (line 755)

### Questions
One issue with the tasks chosen is that they are entirely disjoint and do not test for robustness to distribution shifts or catastrophic forgetting - e.g. training on task A and B alternately and trying to maintain accuracy on A. Although out of the immediate scope of neural plasticity, I think these factors are important to consider for practical settings. How does SNR interact with catastrophic forgetting, i.e. does it make it better or worse?

Is L2+SNR competitive with SNR alone for the tasks other than Permuted Shakespeare? If so, it would be nice to promote L2+SNR as a "default" method that works for all tasks.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a plasticity loss mitigation method called Self Normalized Resets (SNR). It proposes a new method to decide when a neuron should be reset. Specifically, for each neuron, it tracks the mean time between non zero activations of that neuron, models a geometric distribution with that mean, and resets the neuron if the probability that a random variable with this distribution is greater than the neuron’s current time without firing is less than some hyperparameter.

The paper performs some experiments on Permuted MNIST, Random Label MNIST, Random Label CIFAR10, Continual Imagenet, and Permuted Shakespeare, showing a slight improvement or matching previous baselines.

### Strengths
The paper’s new mechanism to select which neurons to reset seems interesting. The theory and intuition behind the method need to be explained more clearly, but the results do show a slight improvement on some settings for some datasets.

### Weaknesses
- This paper feels incomplete. The entire theory section in Section 4 is hard to follow and does not feel very motivated. It’s unclear why Section 4.2 is put in the paper. The paper then immediately ends after an equation, with no explanation of how it relates to the rest of the paper, and it is missing a conclusion section.
- The results in Table 1 are conducted over 5 seeds, but they are missing any error bars.
- Table 2 shows results on a task introduced in this paper, Permuted Shakespeare, but several of the baselines for earlier experiments are missing for this dataset.
- The caption for Figure 3 could be made clearer.
- The details on the hyperparameter search could be clearer.

### Questions
- How do you reset the neurons that you select? Is it the same procedure as in ReDo?
- How does this method do at improving generalization/testing accuracy across distribution shifts?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper proposes a particular heuristic (SNR) to identify neurons that need to be reset in order to recover plasticity. They show that this particular choice of resetting neurons works very well. The work motivates intuitively how SNR compares to more complex utility measures proposed previously, and show results on a bunch of continual learning problems.

### Strengths
The method is simple, easy to implement. And empirically on the problems it has been applied to, it outperforms the existing baseline, which are similar reset methods that use more complex conditions to reset (basically define complex utility functions that are used to make this selection).

### Weaknesses
I'm somewhat concerned about the significance of the results. I think the method outperforms the baseline, but most more detailed results tend to be on permuted MNIST or permuted Shakespeare. I know previous work relied on similar small scale problems, and it feels unfair to punish this work, while the others got away with it. But at some point I'm worried that we are reading too much into these numbers. 

The authors have run on other tasks (Random Label Cifar, Continual ImageNet) but we can only see an aggregate result on those. And why they highlight specific conditions, they are still borderline as dataset (at least the random label cifar). I think some previous papers also showed results on atari which I think is a more complex setting. And previous works have not scaled things a lot more either. 

A second weakness is in my view, not sufficient analysis. E.g. trying to compare some of those utility functions with the SNR heuristic. How often they disagree. Is the intuition that some of these utility functions don't take into account the firing rate of the unit the answer (it the paper there is a suggestion in this direction). The paper argues that L2init is needed for Shakespeare because of the attention layer. Has this been ablated (e.g. apply L2 init just to attention weights?).  Or is the L2init within the MLP block important. And if so why? 

Putting them together, I feel that the main sales pitch is that it is a simpler utility function and it got better numbers, but there is less on understanding why this is the right thing to do. And I'm worried at such it might not have the impact on the community it should. 

Finally I find the lack of conclusion and discussion a bit strange. I understand that there was a page limit, but I find the solution of not providing a discussion suboptimal. I would urge the authors to find other ways to reduce the paper size.  

Overall I'm happy to increase my score if some of the questions are answered.

### Questions
1. Could you move other things to the appendix to make room for a conclusion / discussion section? The results section is meant to include the discussion, but I don't think it goes sufficiently deep into the discussion aspect of it. 

2. Can you run an ablation showing what happens if you only L2 the attention weights?
 
3. Can you provide some more insights in how the heuristic works better than the others? Is it that is resets less, therefore is less disruptive. Or the amount of reset correlates better with the firing rate of the unit? Or something else? 

4. There has been a bit of discussion in the papers that you cite (e.g. works from Lyle et al.) on the causes of the loss of plasticity. Can you somehow connect a bit more the heuristic and the causes of plasticity? Right now it feels like the heuristic is just more reliable at predicting when a unit is actually dormant rather then firing rarely. But maybe is not as much connected to the problem it tries to resolve, that of loss of plasticity.

### Soundness
3

### Presentation
3

### Contribution
2
