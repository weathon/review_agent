# Sharpness-Aware Minimization Efficiently Selects Flatter Minima Late In Training

- Decision: Accept (Spotlight)
- Scores: 8, 8, 6, 8, 6

## Abstract
Sharpness-Aware Minimization (SAM) has substantially improved the generalization of neural networks under various settings. Despite the success, its effectiveness remains poorly understood. In this work, we discover an intriguing phenomenon in the training dynamics of SAM, shedding light on understanding its implicit bias towards flatter minima over Stochastic Gradient Descent (SGD). Specifically, we find that *SAM efficiently selects flatter minima late in training*. Remarkably, even a few epochs of SAM applied at the end of training yield nearly the same generalization and solution sharpness as full SAM training. Subsequently, we delve deeper into the underlying mechanism behind this phenomenon. Theoretically, we identify two phases in the learning dynamics after applying SAM late in training: i) SAM first escapes the minimum found by SGD exponentially fast; and ii) then rapidly converges to a flatter minimum within the same valley. Furthermore, we empirically investigate the role of SAM during the early training phase. We conjecture that the optimization method chosen in the late phase is more crucial in shaping the final solution's properties. Based on this viewpoint, we extend our findings from SAM to Adversarial Training.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies Sharpness Aware Minimization (SAM), an optimization method that modifies the Stochastic Gradient Descent (SGD) in a way that the solution will prefer a flatter region (Eq. (1)).
In practice, its approximation (Eq. (4)) is implemented for computationally efficiency.
It is not well understood whether this approximated method has the intended effect of finding a flatter solution, and this is what this paper studies.

First, the paper empirically shows that when we switch from the SGD to the SAM during the updates, it increasingly improves the solution as we make the switching time earlier, both in terms of the loss and the sharpness (Figure 3). However, using SAM for the last few updates already accounts for the majority of the improvement, and the performance increment becomes very small after that. This suggests that applying SAM in a final few epochs may be nearly as good as doing so for the whole training.

Next, the authors theoretically study several properties of SAM. Theorems 4.1 and 4.2 show that SAM escapes from a solution with large sharpness exponentially fast. Furthermore, with a few additional assumptions, Theorem 4.3 shows SAM converges to the global minimum exponentially fast.

### Strengths
- The theorems are interesting and nicely presented with interpretations. They successfully explain how SAM escapes a sharp region and converges to a flat region.
- The paper cites relevant references and clearly discuss the differences.
- The paper is well written and easy to follow, except for a few points (please see the Weaknesses section).
- The convergence result in Theorem 4.3 does not rely on convexity.

### Weaknesses
- Eq. (4) and the update rule that the authors use in theory (the two-step update rule defined at the beginning of C.1) are not equivalent after eliminating $\theta_{t+1/2}$.

- The experiments and the theory part use different methods (Eqs. (3) and (4)).

- I could not understand
  - how to obtain the inequality in lines 1137-1138, and
  - the phrase "which implies that $\theta_{T+1} = 0$" in lines 1143-1144.
  - the equality in lines 1085-1086 and the inequality in lines 1087-1088.

- This paper only studies the squared loss.

---
### Relatively minor comments
- Theorems 4.1 and 4.2 rely on the linear approximation (5), so it does not explain the whole dynamics after escaping from a sharp minimum and before getting close to a flat minimum. Furthermore, combined with the squared loss, the analysis is in a classical strongly convex setup.

- The single-data setup of Example 4.1 is unrealistic, and I do not know how relevant it is. Moreover, Proposition 4.2 considers the case in which there is only one parameter and has almost no practical usefullness.

- Theorem 4.1 and 4.2 seem to be almost equivalent (contrapositive) to each other. Theorem 4.2 is slightly more detailed in that it provides the explicit inequality showing the exponential bound. Theorem 4.2 directly implies Theorem 4.1, so the latter should be a corollary.

- I am not totally convinced that Assumptions 4.1-4.3 will be actually satisfied in practical situations. To be fair, many supporting references are provided in this paper.

### Questions
### Major comments:
- Please address the first three comments in the Weaknesses section.

- I expected to see the loss going down in later steps in Figure 4 (b). Does the figure not show the entire steps?

- "(P3). SAM converges to a flatter minimum compared to SGD; Theorem 4.1" The result of Theorem 4.1 seems to be a little overstated here. If I understand correctly, Theorem 4.1 does not show SAM converges although Theorem 4.3 does. Note that the latter assumes additional conditions (Assumptions 4.2 and 4.3). Could the authors revise the sentence to a more accurate one?

### Minor comments:
- Are there any experiments using Eq. (4)?

- The paper uses the symbol $\approx$, but its mathematical definition is missing. Could the authors add the precise meaning?

- Proof of Proposition 4.1 could be improved by adding comments on which part the authors used the assumptions.

- Does this refer to Eq. (3)? If so, it should be clearly stated: "Our experiments are conducted using the original implementation of SAM"

---
**Edit**: During the discussion period, the authors addressed all of my concerns. I updated the Rating score from 6 to 8. (I was about to set it to 7, but there wasn't this option.)

### Soundness
3

### Presentation
2

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
The paper studies effects of SAM at the end of training upon an SGD baseline, and claims that using SAM for the final stages of training is sufficient to improve generalization. Considering a trajectory where SGD is switched to SAM at epoch $t$, the paper's theoretical analyses show that SAM rapidly escapes the SGD minima, while staying in the same valley. SAM then proceeds to converge rapidly to a new minimum that is flatter. These claims are supported by empirical evidence, and the authors posit a general conjecture that these loss objectives are only necessary toward the end of training.

### Strengths
While the empirical observations originate elsewhere, the theoretical analysis appears to be novel and interesting. 
While the analyses are limited to the interplay between SGD and SAM during the later phases of training, it is sound and, this line of reasoning can conceivably generalize to other learning algorithms of interest, allowing the potential significance to be large. 
The paper is clearly structured and easy to follow. The authors introduce a well-defined scope with thorough justification, both theoretically and empirically, and clearly discuss the motivation and role of assumptions. They then propose and support empirically a few general impacts of these insights to broader aspects of optimization, such as the role of SAM early-in-training and adversarial training late-in-training.

### Weaknesses
1. For some experiments (Fig 9) using SGD toward the end of training outperformed SAM in terms of accuracy, which is mildly contradicting toward an implicit claim that at the end-of-training, SAM outperforms SGD. Some commentary in the main section on these results would be appreciated. 
2. The size of the valley in Proposition 4.1 for P2, $(-2b,2b)$, appear extremely wide when the parameter is initialized between $(-b, b)$, which weakens the significance of this proof. The empirical evidence (Fig 4c) for this claim also appears limited. Can a more general visualization of the loss landscape be considered?

### Questions
1. In 'Support for P3', can you support why increasing the loss might lead to flatter minima and better generalization? The empirical evidence suggests that SAM decreases the training loss, in addition to test loss.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors explore how algorithm SAM converges to flat local minima and find a phase transition when switching SGD to SAM. By identifying the quick convergence property of SAM, the authors propose to use SAM in the later few epochs, which significantly reduces the computation of using SAM for flat local optima. The experimental results further verify the effectiveness of the proposed algorithm

### Strengths
1. The authors propose to use SAM in the last few epochs instead of starting from the beginning. The new method can reduce almost half of the computation of SAM while retaining the generalization ability of SAM.

2. The authors identify 2 phases of switching from SGD to SAM. With the theoretical analysis, the authors give 4 properties of SAM.

### Weaknesses
1. Some definitions are not clear.  
     (i) Definition 4.1:  Does it mean that SAM is used to optimize linearized function but is evaluated on the real loss or that SAM both trains and tests on the linearized loss?

    (ii)  What is $a$ in Proposition 4.1, which is used to give an upper bound of $\rho$?

2. Is the first claim in Theorem 4.2 the contrapositive of Theorem 4.1? If not, can you explain the difference between these two claims?

3. After Theorem 4.1 and Theorem 4.2, the authors claim that SAM escapes from the sharp minima exponentially fast, which sounds like a good result. However, in Theorem 4.2, the result says that the final loss is $C^t$ larger than the initial one, which seems to be a bad result.  What is the relationship of these two results or the exponentially fast claim should be derived from some other theorem?

4. When the authors want to prove that SAM can stay in the same valley as SGD, there is only one valley in Proposition 4.2. Thus, to show SAM stays in the same valley, the function should have at least two valleys.

### Questions
1. Can the authors give a clear definition of linear stability? It seems to be a very important definition in the paper.

2. Can the authors provide proof when the function has two valleys, SAM will not go from one valley to another? In my view, Proposition 4.1 can not support iterates generated by SAM stay in the same valley.

3. What are the advantages of SAM shown in Theorem 4.3? SGD converges at the same rate under the Assumptions 4.2 and 4.3.

### Soundness
3

### Presentation
3

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
The authors demonstrate that effects of sharpness-aware minimization (SAM) are independent of whether it is applied in early training. They show that when applied late, SAM-driven iterates first escape the narrow minimum found by SGD, and then find a wider minimum. The authors demonstrate that the earlier application of SAM has no discernible effect on the solution. Inspired by these results, the authors find a similar pattern in adversarial training (AT) as well.

### Strengths
- The paper's findings are novel, important, while also counterintuitive given previous research's emphasis on early training. The findings are likely to be of interest to research communities that focus on SAM in specific, and training dynamics and generalization in general.
- The authors make a clear and convincing case regarding the importance of adopting SAM in late (vs. early training) using empirical and theoretical findings.
- The paper is written well and is easy to follow, with the authors' visualizations being especially instructive. Theoretical results are promptly compared with previous results in the literature.
- The extension of the authors' results to adversarial training both supports their original results and also is an important contribution in and of itself.

### Weaknesses
- The divergence between paper's experimental and theoretical setting is notable especially in the choice of the loss function. This is acknowledged in the paper and is acceptable given the novelty of their results.
- Although the paper is written well overall, certain unclearly defined concepts, choice of notation, and design choices make it hard to follow at times. I provide more detailed feedback regarding these in the section below.

### Questions
- L000: The concepts of early vs. late training are used throughout the training, but is not given clear definitions. Please commit to an explicit definition (in accordance with L306), or discuss why this is not necessary/possible. Doing so earlier in the paper is important.
- L081: Efficient sounds like a technical ter
- L110: The notation for dataset and distribution are the same, $\mathcal{S}$. Please differentiate, and differentiate their use as subscripts to $\mathcal{L}$ as well (as this is almost exclusively used with distributions.)
- L115: Please define classification error explicitly.
- L177: "when switches from SGD to SAM"
- L177: The ordering of the x-axes in Figure 3 is confusing. I understand that the authors aimed for consistency between Figures 3, 5, and 6; but I think their choice makes the visualization less legible. I believe having x axes stand for changing time will make the graph easier to read. I ultimately leave this choice to authors' discretion.
- L286: Why change $T$ and $t$'s usage?
- L291: Typo after $\mathcal{L}''$
- L318: Is there a reason for usage of $\ell$ here?
- L352: After changing the use of $T$ and $t$ at L286, Theorems 4.1 and 4.2 changes this use one more time (from $T$ to $T + t$ to $0$ to $t$). Can these be made more consistent?
- L354: Please alert the reader that you will introduce Proposition 4.1 to resolve the conflict between the exponential escape and low-loss assumption
- L388: Do these results generalize to cases where $p>1$ and minibatch SAM?
- L406: Can we assume any valley around $\theta^*$ found by SGD to be sub-quadratic?
- L524: Are you making a claim about robustness-accuracy trade-off? I.e. do you expect robustness to increase without sacrificing vanilla accuracy if the AT is started late?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper delves into the behavior of Sharpness-Aware Minimization (SAM) in the later stages of neural network training, demonstrating that SAM can effectively evade the sharp minima typically identified by Stochastic Gradient Descent (SGD) and instead settle into flatter minima. It outlines a two-stage learning process in which SAM initially hastens the departure from sharp minima and subsequently swiftly reaches flatter minima, a pattern observed even when SAM is employed only during the final training epochs. The research indicates that the selection of the optimization algorithm during the late training phase has a substantial impact on the ultimate attributes of the solution, underscoring the significance of scrutinizing optimization tactics at this critical juncture.

### Strengths
This paper offers an in-depth examination of Sharpness-Aware Minimization (SAM) throughout the training process, with a particular emphasis on its late-phase behavior, an area that has been understudied in prior research.
The paper elucidates a distinct two-phase learning pattern of SAM, illustrating its ability to swiftly exit sharp minima and subsequently settle into flatter minima, thereby enriching the theoretical grasp of SAM's optimization trajectory.

The study underscores the pivotal impact of the optimization algorithm selected for the late training phase, proposing a potential optimization strategy that could reduce computational demands while sustaining optimal performance, thus offering practical relevance for deep learning applications in real-world contexts.

### Weaknesses
1.The theoretical results have limited applicability to real training scenarios. They do not sufficiently explain why transitioning the training method from SGD to SAM results in a large generalization gap compared to using the full SAM training method. The experiments in the paper demonstrate that switching from SAM to SGD often leads to poorer results.

2.In theory, the switch from SGD to SAM is expected to yield certain benefits; however, no experimental evidence is provided to support this.

3.The models discussed are primarily convolution-based, so it would be beneficial to include some transformer-based models as well.

### Questions
Could the authors explain how the neighborhood size affects the results when switching from SAM to SGD?

### Soundness
3

### Presentation
2

### Contribution
3
