# Double Descent Revisited: When Noise Amplifies and Optimizers Decide

- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
We examine how the double descent phenomenon emerges across different architectures, optimisers, learning rate schedulers and noise-robust losses. Previous studies have often attributed the interpolation peak to label noise. However, by systematically varying noise levels, optimizers, learning rate regimes and training losses, we demonstrate that, while noise can amplify the effect, it is unlikely to be the driving factor behind double descent.
Instead, optimization dynamics, notably learning rate and optimizers, strongly influence whether a visible peak appears, often having a larger effect than adding label noise. Consistently, noise-robust losses partially mitigate double descent in settings where the amplification effect of noise is stronger, however their impact is negligible when this is not the case.
Expanding on recent work, this study further confirms that noise primarily deteriorates the linear separability of different classes in feature space.
Our results reconcile seemingly conflicting prior accounts and provide practical guidance: commonly used learning rate/scheduler combinations/losses can prevent double descent, even in noisy regimes. Furthermore, our study suggests that double descent might have a lesser impact in practice. Our code is available at https://anonymous.4open.science/r/DDxNoise.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates the underlying causes of the double descent phenomenon in deep learning. The paper challenges the widely held belief that label noise is the primary driver of the interpolation peak observed in the double descent curve. Through a series of experiments, the paper demonstrates that optimization dynamics, particularly the choice of optimizer and learning rate schedule, play a more significant role in the emergence of double descent than the presence of label noise. Furthermore, the paper analyzes the geometry of learned representations and finds that even when the final classifier struggles, the model can learn robust representations.

### Strengths
The paper identifies that the optimizer and learning rate also play an important role in double descent, not solely the noisy label.

### Weaknesses
Despite the claim that optimizers decide the double descent is promising, the following concerns must be adequately addressed for the paper to be accepted.

1. The paper's logic is not rigorous enough. From the title "Optimizer decides" and line 65 "On the contrary, the optimization algorithm and
learning rate (with and without scheduler) appear to be strong drivers of DD" in introduction, it seems to argue that the optimizer is the dominant factor in double descent. However, the experimental results may not fully support this claim. In Section 4.1, while figures (a) and (b) lack a distinct peak, a clear plateau is observed at around the 10th layer. This might only indicate a less obvious double descent, not its absence. Furthermore, Figure 2 in [1] demonstrated that SGD can also produce a clear double descent phenomenon, so the results here cannot conclusively position the optimizer as the core driver. It is possible that different optimizers, similar to label noise, have varying strengths of influence. I think at best, Section 4.1 demonstrates that the optimizer plays a role **but not the key role** in double descent. The authors should clarify this point.
2. If the above point 1 is valid, meaning both the optimizer and label noise are drivers, then the novelty of this paper becomes questionable, as its experimental setup and the source codes are nearly identical to that of [1].
3. To convincingly prove that the optimizers "decide" the phenomenon, relying solely on experimental evidence is insufficient, as exceptions can always exist. A better approach would be to incorporate a theoretical component, demonstrating that under certain settings, the use of an optimizer with specific properties theoretically guarantees the emergence of the double descent phenomenon, even in the absence of label noise.

Reference.

[1] Gu, Y., Zheng, X., & Aste, T. (2023). Unraveling the Enigma of Double Descent: An In-depth Analysis through the Lens of Learned Feature Space. ArXiv, abs/2310.13572.

### Questions
See above.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents an empirical investigation of model-wise double descent across a wide range of settings. The authors conduct ablation studies on key factors, including noise level, loss function, optimizer, and learning rate, to understand the respective roles of noise and the optimization process.

Two primary findings emerge from their analysis. First, the double descent phenomenon is observed even in the absence of label noise (i.e., with clean data). Second, the choice of optimizer and learning rate significantly alters the double descent curve, sometimes causing the effect to vanish entirely. Lastly, they also check the properties of the representations and provide links to neural collapse phenomena.

### Strengths
The paper's extensive experimentation allows for a detailed discussion of the various factors contributing to double descent. The work is clearly written and accessible.

### Weaknesses
1. The analysis seems to be in line with previous work, and its contribution beyond the existing literature is unclear. The authors write, "Only a handful of studies have emphasized the role of optimization dynamics," but they provide numerous previous works that consider this perspective. Moreover, there is a huge literature on implicit bias in overparameterized networks which also suggests an explanation for DD via optimization dynamics.

2. The analysis assumes the datasets used are noise-free. I am not sure if this is true (at least for CIFAR-10), see [1]. Moreover, it is unclear to me what the added benefit is of experiments that show double descent when there is no explicit label noise. These findings are also present in [2].

[1] Wei, Jiaheng, et al. "Learning with noisy labels revisited: A study using real-world human annotations." arXiv preprint arXiv:2110.12088 (2021).
[2] Gu, Yufei, Xiaoqing Zheng, and Tomaso Aste. "Unraveling the enigma of double descent: An in-depth analysis through the lens of learned feature space." arXiv preprint arXiv:2310.13572 (2023).

### Questions
1. Can you clarify the statement: "DD is not caused by noisy data but ... is directly attributable to the optimization process"? Doesn't the noise level directly alter the optimization process? I would understand the following claim better: "DD is still present in non-noisy data".

2. Can you explain the added contribution over [2]? I think their statement matches the statement presented in this paper which is that noise amplifies DD. [2] also argues that there is label-noise in MNIST and CIFAR-10. Did you take into account any label-noise that was already present in the dataset labeling in your analysis?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper revisited the double descent phenomenon of deep neural networks, and analyzed the impact of the optimizer and noise. With verifying and rejecting prior hypotheses, the study confirmed that the choice of optimizer decides the phenomenon occurence and noisy regimes amplifies the phenomenon.

### Strengths
1. This paper revisited the optimization perspective in the DD phenomenon and evaluated from both loss functions / noise factors and geometry -based metrics, demonstrated in-depth understanding and analytics towards the problem.
2. The experiment design is complete and carefully implemented , with code open-sourced for re-production.

### Weaknesses
1. While the paper discussed the different role of optimizer and noise w.r.t. the DD phenomenon, it does not discussed how different optimizer handle noisy regimes from a fundamental perspective (only gently touched in the 4.3 section), which degrades the depth of analytics. 
2. As DD is first reported and mostly observed in deep learning and DNNs, the role of machine learning model and task characteristics is also critical to the analytics of DD. I believe the conclusion of "Optimizers Decide" is too vague in a bigger landscape.

### Questions
1. As discussed in the weaknesses, can you hypothesize how different optimizers promote learning in a noisy regime, which decides the DD phenomenon, according to your hypothesis? 
2. As a question, is it possible to extend your analytics to more recent large language models? I believe frameworks like NanoGPT are relatively accessible in comparison to CNNs, and either verifying the DD phenomenon or your hypothesis would be an extension to your existing works. 
3. While a unified theory remains absent towards the DD phenomenon, suggestions on future works and the remaining questions from this paper's perspective may also be a valuable contribution to the field.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper discusses the double descent phenomenon, which is the observation that test error can increase near the interpolation threshold and then decrease again as model capacity increases. While past work attributed DD primarily to label noise, this study argues that optimization dynamics (optimizer type, learning rate, scheduler) are the real reasons, with noise acting mainly as an amplifier. The authors also show that noise primarily affects the linear separability of different classes in feature space.

### Strengths
Strengths:

1. The authors challenge the prevailing idea that label noise causes double descent, showing instead that optimization dynamics are the main reason for double descent phenomena. 

2. The paper performs a well-controlled empirical sweep across optimizers, learning rates, and schedulers, isolating their effect on double descent.

3. Introducing geometry-based metrics (k-Nearest Neighbour and Nearest Centroid accuracies) provides a deeper probe into how learned features evolve across parameter regimes.

### Weaknesses
Weaknesses:

1. The paper lacks a theoretical framework explaining why optimizer dynamics cause double descent, and how noise is not a necessary condition for double descent phenomena. 

2. The analysis focuses mostly on learning rate and optimizer type, but other factors (batch size, momentum, weight decay) are not explored.

3. Experiments use only MNIST and CIFAR-10, which may not capture behaviors of larger or more realistic models.

4. The paper focuses almost exclusively on test-set curves (test loss and test accuracy) when illustrating the double descent phenomenon. However, by definition, double descent describes a mismatch between training and test behaviors. Specifically, the training loss continuously decreases or saturates at near-zero, while the test loss first decreases, then increases near the interpolation threshold, and decreases again afterward. Without showing training loss/accuracy, it is not possible to confirm that the observed “double descent” peaks are truly due to the generalization gap rather than optimization instability or noise.

5. The paper lacks quantitative metrics and statistical analysis measures

### Questions
Questions to the Authors:

1. Could you provide training loss and accuracy curves to confirm that the models actually interpolate (zero training loss) near the test loss peak?

2. Statistical Significance: Did you average the results over multiple random seeds?

3. Authors should discuss the theoretical reasons behind why noise is not a necessary condition for the double descent phenomenon. Why do optimizer dynamics cause double descent?

4. For ResNet, how does the double descent behavior change if you scale depth (number of blocks) rather than width?

### Soundness
2

### Presentation
3

### Contribution
1
