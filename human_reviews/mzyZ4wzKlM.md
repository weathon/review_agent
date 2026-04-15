# Expressive Losses for Verified Robustness via Convex Combinations

- Decision: Accept (poster)
- Scores: 8, 8, 6, 5

## Abstract
In order to train networks for verified adversarial robustness, it is common to over-approximate the worst-case loss over perturbation regions, resulting in networks that attain verifiability at the expense of standard performance.
As shown in recent work, better trade-offs between accuracy and robustness can be obtained by carefully coupling adversarial training with over-approximations. 
We hypothesize that the expressivity of a loss function, which we formalize as the ability to span a range of trade-offs between lower and upper bounds to the worst-case loss through a single parameter (the over-approximation coefficient), is key to attaining state-of-the-art performance. 
To support our hypothesis, we show that trivial expressive losses, obtained via convex combinations between adversarial attacks and IBP bounds, yield state-of-the-art results across a variety of settings in spite of their conceptual simplicity.
We provide a detailed analysis of the relationship between the over-approximation coefficient and performance profiles across different expressive losses, showing that, while expressivity is essential, better approximations of the worst-case loss are not necessarily linked to superior robustness-accuracy trade-offs.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a scheme for convexly combining verified and adversarial loss functions to train verifiably robust models.

### Strengths
1. The paper is generally well written and presented.

2. The contribution is generally clear.

3. The idea of expressive loss functions as interpolating between verified and adversarial losses is interesting.

### Weaknesses
1. My main concern is that the paper is not sufficiently novel to merit publication. Convexly combining loss functions is a rather obvious idea; indeed (8) is just a linear combination of two loss functions, something which has been around for ages.

### Questions
1. In table 3, the alpha parameter for the MTL-IBP method can get very low (e.g., $4 \cdot 10^{-3}$ for CIFAR-10 $2/255$). Does this not mean that the loss essentially just reduces to the adversarial loss?

2. Why do the optimal $alpha$'s vary so much between the $2/255$ and $8/255$ epsilons for CIFAR-10? As in Q1, the MTL-IBP loss boils down to just the adversarial loss for $2/255$, and changes to a $50/50$ split for $8/255$.

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors hypothesize that expressive loss functions yield better training for verified robustness. A family of loss functions $\mathcal{L}_\alpha$ is expressive if
- $\mathcal{L}(f(\theta, x_\text{adv}), y) \leq \mathcal{L}_\alpha(\theta, x, y) \leq \mathcal{L}_v (f(\theta,x),y)$ for all $\alpha \in [0,1]$, where the left/right inequality become an equality if $\alpha = 0/1$ respectively. 
- $\mathcal{L}_\alpha$ is monotonically increasing with $\alpha$

They support their hypothesis that expressive loss functions yield better training for verified robustness by showing that trivial expressive losses obtained via convex combinations between adversarial attacks and IBP bounds yield sota results. They further hypothesize that the notion of expressivity is crucial to get sota and argue as follows:
- They state that sota verified training algorithms rely on coupling adversarial attacks with over-approximations. They show that SABR is expressive - hence the good performance of SABR is inline with their explanation.
- Other expressive losses can be trivially designed via convex combinations, i.e.
	1. CC-IBP: combine adversarial and over-approximated network outputs with in the loss
	2. MTL-IBP: combine adversarial and verified losses
- Experimental evaluation of CC-IBP and MTL-IBP. Both attain sota, particularly on TinyImageNet and downscaled ImageNet. 
Further, they analyze the parameter $\alpha$ governing a robustness-accuracy trade-off. Better approximations of the worst case loss do not necessarily correspond to performance improvements. 

The authors experimentally compare CC-IBP and MTL-IBP to prior work and find that the proposed methods match or outperform the literature. They also study the effect of the over-approximation coefficient on the performance profiles of expressive losses. The take away here is that better approximations of the branch-and-bound loss do not necessarily result in better performance. 

Observed that standard accuracy decreases with alpha and verified accuracy increases with alpha. The adversarial and verified robust accuracies unter tighter verifiers may first increase and then decrease with $\alpha$, hence the need for careful tuning according to the desired robustness-accuracy trade-off. 

Finally, the assumption that better approximations of the worst-case loss results in better trade-offs between verified robustness and accuracy is investigated. The parameter $\alpha$ is chosen based on the performance on a hold out set consisting of 20% of the training set. The worst case loss is approximated using a branch-and-bound loss. The authors report that sometimes it is better for the BaB error to be positive and sometimes for the BaB error to be negative.

### Strengths
- The presentation is mostly good.
- Training certifiable networks is a relevant research problem. 
- The ideas are conceptually simple yet seem to be effective. 
- The work unifies and generalizes successfull approaches. 
- The authors provided code.

### Weaknesses
- It remains unclear how stable the results are (for example w.r.t. different seeds). 
- Writing could be improved in some parts of the paper, i.e. Section 6.3. 
- It remains unclear what "tricks" i.e. for regularization and initialization where specifically used.

### Questions
- What are the confidence intervals for the results in the paper with respect to different seeds? Are the trends consistent w.r.t. the randomness due to different seeds? 
- What specialized initialization and regularization techniques where used?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes "expressive" loss functions that interpolate between IBP and adversarial loss in a simple linear combinations and show good empirical performance on a variety of datasets.

### Strengths
The empirical results seem strong, especially considering the fact that the proposed methods are simple interpolations.

### Weaknesses
The presentation and writing needs a lot of work and it seems the paper is hurriedly written. Specific concerns are below.


	The mathematical definition of property P in Eq 1 is given in section 2 without any discussion of what it means or entails and why is it interesting/useful. 

	You could add atleast one example of how x_adv could possibly be generated in the background section.

	Explicitly write down what “verification” means before using it in section 2.1. I don’t know what the following statement means: “However, formal verification methods fail to formally prove their robustness in a feasible time” 

	 “As seen from Eq 1, network is provably robust if …logit differences … all positive” – why and how before even defining what verification is.

	“Incomplete verifiers will only prove a subset of the properties” – which properties ? only one is defined. 

	The unlaballed equation with relationship of \underline{z} and z. I would not write \underline{z} as an inequation when saying it a lower bound without defining it first say using an example.  

	Can you give an example when o and l are not equal ? the definition of z requires o to be atleast as big as l, and the definition of z also makes sense only if o and l are equal. Is o the size of the output before softmax evaluation or after? I am having a hard time reconciling dimensions of f( ) and z( ) for translation-invariance.

	Are the other methods also grid-searched for best hyperparameters  for their respective methods ?

	The runtimes are a bit misleading? Does the runtime also include hyperameter search cost including the cost for best interpolating parameter ?

### Questions
Please see the weakness section.

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work studies the certified training with over-approximation for robustness certification. Specifically, the authors introduce the idea of expressivity of loss functions and show that it can range from worst-case loss to verified loss, based on which two forms of loss are proposed. The experiments show the performance of the new losses and some findings regarding robustness and accuracy are given.

### Strengths
- The motivation of the paper is sound, and the underlying theory regarding certified training remains unknown and challenging.
- The paper is generally well-organized and easy to follow.
- The experiments are comprehensive and different datasets and attack radii are used for the evaluation.

### Weaknesses
- My biggest concern is that the contribution and novelty of the paper are incremental and minor, which is about the expressivity of losses. However, it seems that it somehow borrows the idea of the previous work SABR, which gives an effective loss ranging from adversarial loss and verified loss.  The difference between this work and SABR is not that clear and significant as SABR can induce expressivity by letting $\lambda=\alpha$ as shown in Sec. 3.
- Some key details are not given in the main text. E.g., it is not clear from the main text how the logit differences are associated with an adversarial attack for CC-IBP when it is compared to CROWN-IBP in Sec. 4.1, without which the contribution and novelty are further weakened in terms of the comparison.
- The insight and intuition of the relationship between CC-IBP and MTL-IBP are not clear. For example, does any case exist where one can be degraded to the other? If so, either theoretical or empirical results are needed to show it.
- For Table 1, the proposed method is with BaB as a complete method, I wonder if it is fair to compare with some incomplete baselines.

### Questions
See the Weakness part.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
