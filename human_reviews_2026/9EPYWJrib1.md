# Optimizer Choice Matters For The Emergence of Neural Collapse

- Decision: Accept (Poster)
- Scores: 2, 8, 6

## Abstract
Neural Collapse (NC) refers to the emergence of highly symmetric geometric structures in the representations of deep neural networks during the terminal phase of training. Despite its prevalence, the theoretical understanding of NC remains limited. Existing analyses largely ignore the role of the optimizer, thereby suggesting that NC is universal across optimization methods.
In this work, we challenge this assumption and demonstrate that the choice of optimizer plays a critical role in the emergence of NC. The phenomenon is typically quantified through NC metrics, which, however, are difficult to track and analyze theoretically. To overcome this limitation, we introduce a novel diagnostic metric, NC0, whose convergence to zero is a necessary condition for NC. Using NC0, we provide theoretical evidence that NC cannot emerge under decoupled weight decay in adaptive optimizers, as implemented in AdamW. Concretely, we prove that SGD, SignGD with coupled weight decay (a special case of Adam), and SignGD with decoupled weight decay (a special case of AdamW) exhibit qualitatively different NC0 dynamics.
Also, we show the accelerating effect of momentum on NC (beyond convergence of train loss) when trained with SGD, being the first result concerning momentum in the context of NC.
Finally, we conduct extensive empirical experiments consisting of 3,900 training runs across various datasets, architectures, optimizers, and hyperparameters, 
confirming our theoretical results.
This work provides the first theoretical explanation for optimizer-dependent emergence of NC and highlights
the overlooked role of weight decay coupling in implicit biases of optimizers.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors address an important literature gap in the understanding of the conditions under which neural collapse emerges. The paper points out that the emergence of neural collapse can largely depend on the choice of the optimizer. In particular, they show that in adaptive optimizers, particularly those of second-order nature, the emergence of NC (in particular the weight-feature alignment NC3) might depend on the way in which weight decay is implemented – whether it is coupled or decoupled. The authors show both empirically and theoretically, that AdamW or SignumW consistently don’t converge to zero/small values of NC3 metric, while their coupled counterparts (as well as SGD with momentum, either coupled or uncoupled) converge to much smaller values. The main analytic tool (also used extensively in experiments) is the so-called NC0 metric (the global bias of the rows of last layer’s weight matrix), which is a necessary condition for the emergence of full NC (especially NC2 and NC3 together). The authors show that this metric converges to a non-zero constant for signGD with decoupled weight decay, while for the coupled weight decay it might converge to zero provided vanishing learning rate. The authors complement these main findings by several other related observations and ablations.

### Strengths
-	S1: Both theoretical and empirical evidence presented in the paper is convincing (modulo a few caveats mentioned in the weaknesses). The paper makes it clear that the presented phenomenon is indeed happening. Some of the ablations in the appendix (for instance the one with 2000 training epochs) reassure me even more. 
-	S2: The paper tackles an important topic of better understanding the conditions under which the neural collapse emerges. 
-	S3: The paper is mostly well written and easy to follow. 
-	S4: The paper also presents several non-central insights, such as the role of the momentum, analysis of the interpolation between the two Adam variants, discussion of the meaning of the NC metrics. 
-	S5: The authors seem to be open about the limitations of the paper and discuss them appropriately.

### Weaknesses
-	W1: If I correctly understand the mathematical essence of the paper, it seems that authors are misinterpreting their results. To me, the main distinction should not be coupled vs. decoupled weight decay but rather first- vs. second-order optimizers. It seems that the crucial parameter that really determines the limiting behavior is not so much the implementation of the weight decay, but rather whether the gradient computation corresponds precisely to the original loss function and thus, whether the zero-sum gradient property from line 263 holds for that optimizer or not. In particular, while authors correctly interpret Theorem 3.3, I believe that Theorem 3.4 should be interpreted as a proof that SignGD with coupled weight decay also doesn’t converge to zero, even in this idealistic ETF features scenario. I think there is a good empirical evidence that Adam itself doesn’t converge to NC too – in Table 2 we see that the best NC0 metric reached by Adam is orders of magnitude higher than those reached by both SGD and SGDW. Similar conclusions can be drawn from Figure 7 where even for pure Adam, NC3 metric seems to converge to positive value well above 0, or Figure 6, where we see that the smallest eigenvalue of Adam converges to small, yet non-zero value. In Figure 5, while NC3 is still improving for Adam, the NC0 seems to have converged. The SGDW, on the other hand, despite the decoupling, does not suffer from the above-mentioned issues, which suggests that the crucial aspect is indeed the division by empirical exponentially averaged variance in second-order optimizers like Adam. Now, the reason why there is such a big empirical difference between Adam and AdamW is that the presence of WD in the internal updates of the momentum and variance parameters seems to alleviate (yet not fully erase) this systematic bias. That’s why the empirical results seem to suggest that the difference between decoupled and coupled weight decay is bigger than that between first and second-order optimizers. Thus, I agree that this difference should be discussed and the paper provides compelling evidence for it, but I don’t think it should be narrated without proper discussion (or even prioritization) of the differences between the optimizer types. 

-	W2: Some of the empirical evidence gives mixed signals and gives the reader an impression that the authors did not properly calibrate the weight decay ranges for which the experiments were conducted. This can be seen by the fact that a lot of weight-decay vs. NC3 plots show concave behavior for the decoupled weight decay variants. This is true in Figure 3, but also in appendix figures 17, 20, 24, 25, 29, 30, 35, 40. This suggests that higher values of weight decay for the decoupled variants could reach as low values of NC3 as the coupled variants. Another strong evidence for this is in Figure 14 for NC0 metric, where we see a huge difference in the metric for the largest weight decay, again suggesting that the weight decay ranges should have been wider on the right end of the plots. While part of this phenomenon could be explained by the quadratic relationship between NC0 and regularization strength, the fact that NC3 metric is normalized properly makes it unclear to what extent are the curves concave because of this, and to what extent because of the actual convergence of NC3 to lower values.

-	W3: A fair comparison of weight decay and momentum values across experiments, or when comparing coupled and decoupled implementation, is a general issue in the paper. For instance, it is well known and it agrees with Theorem 3.1 of the paper, that the weight decay generally only influences the speed of convergence of NC metrics, but not necessarily whether they converge. Furthermore, it is also well-known that the training converges towards neural collapse also for zero values of weight decay, but the convergence is slower. Therefore, results such as the ones in Figure 4 should be calibrated to account for this, for instance by letting training run until a fixed small training loss is achieved. If the authors only meant that weight decay and momentum only affect the speed of convergence, then it should be more explicitly discussed in the paper, but these results are not novel. 

-	W4: Related to the previous point, I believe that the NC0 metric should be normalized by the norm of the weight matrix. If the NC0 metric remains constant but both the weight matrix and the final layer features grow to infinity (as is known to be the case in the zero weight decay case), than geometrically NC can still be asymptotically approached. Therefore, normalizing NC0 metric would provide for a more fair comparison across various weight decay values. 

-	W5: Finally, the strength of the message and/or contribution of the paper is relatively modest, especially because we see from the experiments that many conclusions the authors draw only hold for NC3 metric, but not for NC1/2 metrics. Thus, the coupling of weight decay / optimizer itself seem to be particularly important only for the NC3 metric. This limits the scope of the results, although I agree that the observation that full NC convergence might depend on the optimizer is still an important and insightful addition to our knowledge. 

-	W6: Some writing aspects should be improved. In particular:
-    Equation 1 doesn’t use bias in the last layer, but in appendix authors include bias in the definition.
-	The original definition of the NC metrics formulated them for the centered class means. The authors should discuss why they decided to omit this detail in their definitions. This also influences the line 143-144 in the proof where authors talk about the convergence of centered class means. 
-	Line 147: I don’t see why this should hold. We can always rotate M^* and counter-rotate P accordingly. 
-	The constant T in Theorem 3.1 is redundant. 
-	Line 273: you cite the wrong paper of Jacot. It should be Jacot 2024. 
-	The definition of within-class variability in line 637 is wrong.
-	There is a notation inconsistency between the definition of the \alpha parameter in Theorem 3.3 and its appendix version.

### Questions
-	In Figure 1 we see a little increase in the NC3 metric of Adam optimizer for the largest value of weight decay. Do you have a guess why this would be? 
-	Why is there such a small variance in the measured metrics in Figure 1 and similar figures, when you average them over a wide variety of learning rates and momentums? 
-	Have you tried to train Adam with as high values of weight decay as AdamW? What would happen? 
-	The NC0 values for SGD in Figure 8 seem weird. Do you have an explanation? 
-	In Figure 15 we see that the metrics go up for Adam for large values of weight decay. Why is that? 
-    How big of a difference there is in your opinion between real Adam and SignGD in terms of implicit biases? We know that Adam is known to converge in some regular cases, while SignGD does not seem to be likely to ever converge. For this reason, do you think that SignGD is a good approximation of Adam? 


**SUMMARY:**
While I think the content of this paper is publishable material and the contributions are insightful, I cannot recommend acceptance at this point due to fair amount of concerns I have. I would recommend the authors to re-structure and re-interpret the work and resubmit. Moreover, since the scope of the paper is a bit modest (particularly because it mostly only concerns the NC3 metric), my recommendation would be to submit to TMLR.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors provide new insight into Neural Collapse (NC). They present NC0, a new necessary condition for NC2 and NC3, and demonstrate that the choice of optimizer impacts neural collapse, particularly the momentum and weight decay parameters and their specific implementation.
Experimentally, the authors show that non-zero weight decay is necessary for NC0 and NC3 metrics to decrease. They also observe that these NC metrics decrease both by increasing weight decay (for a fixed momentum) and by increasing momentum (for a fixed non-zero weight decay). This behavior is theoretically proven for SGD with momentum and weight decay.
Finally, they show that neural collapse does not emerge with decoupled weight decay for optimizers like Adam and Signum. They prove this for signed SGD with decoupled weight decay, while also showing that NC0 does decrease for signed SGD when using coupled weight decay.

### Strengths
- This work investigates Neural Collapse from a novel perspective, supported by both experimentation and theoretical evidence.
- It opens up for further discussion and deeper investigation in the field.
- The work is clearly written.
- The authors critically discuss their findings and limitations.

### Weaknesses
- Some plots are difficult to interpret due to overlapping lines and similar colors.

### Questions
Given that the occurrence of NC is connected to better generalization, how would the authors interpret their result that decoupled weight decay prevents Neural Collapse, in connection with the fact that AdamW is an industry standard due to its performance gain over coupled weight decay?

### Soundness
3

### Presentation
3

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
The paper argues that how weight decay is implemented largely determines whether Neural Collapse (NC) emerges. It introduces NC0 as a necessary diagnostic and proves contrasting NC0 dynamics in a stylized model. The paper also validates the mechanism with large-scale experiments, showing NC metrics improve as the coupled component increases, while accuracy remains about the same.

### Strengths
1. An actionable mechanism that practitioners can immediately test and reason about.

2. NC0 provides a tractable and provable indicator, enabling convergence and impossibility statements.

3. Value the importance of optimization algorithm choices in NC.

### Weaknesses
1. Modeling gap to real Adam/AdamW. The formal results use SignGD and unconstrained features as a proxy. While the qualitative match to Adam/AdamW is persuasive, the absence of finite-step analysis with (β₁, β₂, ε) leaves open whether corner-casescould break the claimed dynamics.

2. NC0 is necessary, not sufficient. It could happen that NC0→0 but full NC (NC1–NC3) can still fail. Without parallel theory for the other NC metrics, one could over-infer the presence or absence of collapse.

3. External validity across regimes. The empirical case is broad but still focused on common settings. Very large batches, aggressive data augmentation, label smoothing, and heavy regularization may alter gradient noise scales and effective WD, potentially changing the coupling story.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
