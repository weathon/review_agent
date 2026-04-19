# Butterfly Effects of SGD Noise: Error Amplification in Behavior Cloning and Autoregression

- Decision: Accept (poster)
- Scores: 3, 5, 8

## Abstract
This work studies training instabilities of behavior cloning with deep neural networks. We observe that minibatch SGD updates to the policy network during training result in sharp oscillations in long-horizon rewards, despite negligibly affecting the behavior cloning loss. We empirically disentangle the statistical and computational causes of these oscillations, and find them to stem from the chaotic propagation of minibatch SGD noise through unstable closed-loop dynamics.  While SGD noise is benign in the single-step action prediction objective, it results in catastrophic error accumulation over long horizons, an effect we term *gradient variance amplification* (GVA).  We demonstrate that many standard mitigation techniques do not alleviate GVA, but that taking an exponential moving average (EMA) of iterates is surprisingly effective at doing so.  Furthermore, we illustrate the generality of the phenomenon by showing both the existence of GVA and its amelioration by EMA in autoregressive language generation.  Finally, we provide theoretical vignettes both exhibiting the benefits of EMA in alleviating GVA and illustrating the extent to which classical convex models help in understanding the benefits of iterate averaging in deep learning.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes that the training instability in the policy network is due to the gradient variance amplication during training

### Strengths
The idea is novel and the problem being dealt with is sufficiently important, namely, during training in reinforcement learning, the model does experience strong instabilities, and good methods to mitigate them would be of great importance

### Weaknesses
I am quite confused by the main claim of the work. 

The main claim is that the gradient variance is responsible for the instability and that it amplifies throughout training, but no numerical result in the paper really plots the gradient variance, and, of course, not a single experiment shows that this variance is amplified. It should not be difficult to plot the variance of the gradient at all, and the absence of such evidence makes it impossible for me to recommend acceptance.

### Questions
See weakness

### Soundness
1 poor

### Presentation
2 fair

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
The paper examines how noise in gradients affects systems with feedback loops. It argues that SGD error accumulates over longer horizons, leading to Gradient Variance Amplification (GVA), explaining sharp reward oscillations. GVA is shown to be the main cause of these oscillations, surpassing the influence of statistical and architectural factors. Empirical evidence shows that the empirical moving average (MVA) mitigates these oscillations, ensuring stability. Additionally, the paper revisits the theory of stochastic optimization for convex functions, assessing its explanatory power for empirical observations of EMA and various step size schedules.

### Strengths
a) The paper conducts meticulous experiments revealing that the variance in stochastic gradients is the true source of instability. It introduces exponential moving average as an effective solution to mitigate this issue. Additionally, this phenomenon is also demonstrated for other tasks with feedback loop, i.e., auto regressive processes for language generation. 
b) The paper is well-written and the main message is clearly presented.

### Weaknesses
a) The main problem of the paper in my opinion is the explanation provided for the benefits of EMA.  Proposition 3.1 says that land scape is very intricate and for every $\delta$ there is a separation between $J$ and the behaviour cloning loss. However, the *cliff*-type loss framework used to study this does not capture this behaviour, as it is small in a neighbourhood of radius $\epsilon $ and is very large outside. In my opinion, this framework it too tailor-made to study the benefits of EMA and does not reveal the real reason behind its working mechanism.

### Questions
It is mentioned in comments after proposition 3.1 that there is a good subset in parameter space that do not experience this worst-case error amplification. Does such good neighbourhood exists around any element in the parameter space or only around elements with specific properties ?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper examines a known problem in behavior cloning. This problem occurs during the training of such agents, where the validation criterion (return of an episode under long horizon rollouts) has large variance at all training steps, while the training surrogate loss has small, making it hard to perform model selection.

The authors perform a theoretical and empirical study of the origins of this problem. They argue that the observed variance is due to training instability, instead of insufficient dataset size, and they name the phenomenon as *Gradient Variance Amplification (GVA)*. They suggest that alternative training algorithms might not have the same issue. For this reason, they propose a very simple fix, which is to track the exponential moving average (EMA) of the parameter iterates of the SGD trained model. They demonstrate the effectiveness of the approach by performing experiments under various environments and by ablating design choices of the proposed EMA intervention.

In addition, they argue that GVA generally exists whenever agents are expected to operate by conditioning themselves in their past output (or effects of them). This definition aligns with conditional language modelling, and they argue with experiments that language generation quality in LLMs is similarly affected and that EMA can also help mitigate the problem there.

### Strengths
* The paper is well-organized, well-written and well-argumented. It introduces and motivates the problem sufficiently and discussed related work in depth.

 * They pose a single clear question at the heart of the problem: Do we just need more data? Or is there in the training algorithm that amplifies variance for the validation criterion? Their ablations convince that the second is the case (section 3.1/figure 2), and they provide with theoretical insight that the return in horizon H can be exponentially large for some suboptimal policy in a smooth neighborhood around the expert, even though for all of those suboptimal policies in the neighborhood the surrogate training loss (of behavioral cloning) is small.

 * They introduce various decision choices around implementing EMA as a candidate solution, and they ablate many of them, demonstrating that the problem is effectively mitigated in many RL environments.

 * They make connections to the relevance of the problem to other topics in ML, such as autoregressive language modelling.

### Weaknesses
1. Ablations regarding the design choice of $\gamma_t$ scheduling are missing. In the paper, a polynomial decay is used and ablated, but other schedules have also been used with EMA (like cosine). It would be nice to understand a bit better why this decay is important and how to design one which is tailored at the problem at hand.
2. Middle of Figure 4 misses y-axis values, which is important in order to know at which scale are we seeing the zoom at.
3. In the context of autoregressive language generation: Validation perplexity perhaps does not show the existence of GVA problem here in the most clear way. Ideally, some equivalent to a metric of generation quality of horizon H autoregressive rollouts should have been used instead.

### Questions
1. In **Proposition 3.1**, I guess that $\Delta$ refers to two different smooth “error functions” in each of the two inequalities.

### Typos

2. **Section 4.3**: “Here training loss is convex, but rollout reward is not.” The training loss is indeed convex, but the rollout reward is (still) concave, even if it is discountinuous. Is that right?
3. Later in **Section 4.3**: “SGD iterates:” $\theta_{t+1} = \theta_{t} - …$ instead of $\theta_{t+1} = \theta_{t+1} - …$
4. **Appendix A.1**/**Role of SGD noise**: “It is now well appreciate*d* that gradient noise facilitates the escape *from* saddle points.”

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good
