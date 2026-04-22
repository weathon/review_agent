# Activation Function Design Sustains Plasticity in Continual Learning

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6, 2

## Abstract
In independent, identically distributed (i.i.d.) training regimes, activation functions have been benchmarked extensively, and their differences often shrink once model size and optimization are tuned. In continual learning, however, the picture is different: beyond catastrophic forgetting, models can progressively lose the ability to adapt—loss of plasticity—and the role of the non-linearity in this failure mode remains underexplored. We show that activation choice is a primary, architecture-agnostic lever for mitigating plasticity loss. Building on a property-level analysis of negative-branch shape and saturation behavior, we introduce two drop-in nonlinearities—Smooth-Leaky and Randomized Smooth-Leaky—and evaluate them in two complementary settings: (i) supervised class-incremental benchmarks and (ii) reinforcement learning with non-stationary MuJoCo environments designed to induce controlled distribution and dynamics shifts. We also provide a simple stress protocol and diagnostics that link the shape of the activation to the adaptation under change. The takeaway is straightforward: thoughtful activation design offers a lightweight, domain-general way to sustain plasticity in continual learning without extra capacity or task-specific tuning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work presents two case studies to investigate the activation functions in the context of continual learning and identifies three principles for designing plasticity-friendly nonlinearities. Building upon these insights, the authors propose two new activation functions: **Smooth-Leaky** and **Randomized Smooth-Leaky**. Empirical studies on continual supervised learning and continual reinforcement learning demonstrate that the proposed activation functions achieve superior performance compared to existing alternatives.

### Strengths
- The case studies are interesting and thoughtfully executed, providing solid and well-supported conclusions. 
- The large-scale empirical studies on continual supervised learning and continual reinforcement learning clearly demonstrate the superiority of the proposed activation function over prior alternatives.
- The proposed activation functions show promise for broader application in domains that rely on network plasticity.
- The authors also provide code to improve reproducibility.

### Weaknesses
- As the experiments already include multiple runs, I recommend that the authors report not only the mean results but also measures of statistical significance—such as standard deviation or 95% confidence intervals—for example, in Table 1 and Table 2.
- The continual reinforcement learning evaluation appears limited to a single experimental configuration. Expanding this analysis to include additional environments or settings would provide stronger support for the generality of the proposed activation functions.
- *Case Study 1* and *Case Study 2* are important parts of the paper, but too many related results and figures are placed in the appendix. To enhance readability, the authors might consider reorganizing the content despite the page limit.
- The term $C^1$ transition first appears in the introduction (line 60), but without explanation.

### Questions
- Do the proposed activation functions result in greater wall-clock time compared to existing activation functions?
- Could the authors provide the results of the **Smooth-Leaky** activation function in the continual reinforcement learning experiments?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper argues that activation choice is a primary lever for sustaining plasticity in continual learning. It presents a property-level analysis (negative-branch responsiveness, saturation, “dead-band” width) of activation functions and proposes two drop-in $C^1$ activations, Smooth-Leaky and Randomized Smooth-Leaky, that maintain a non-zero derivative floor. The paper evaluates the activation functions on standard Continual Learning tasks.

### Strengths
The paper provides a substantial empirical study on the properties of many activation functions under the lens of continual learning. The case studies are well thought out and well conducted, they isolate negative-branch slope, saturation, and “dead-band width” as key predictors under shift. The study of desaturation dynamics under shocks is an important diagnostic of activation functions and ties well activation shape to recovery behavior after controlled shifts.
The suggested drop-in replacements are intuitive and easy to implement.

### Weaknesses
In general, the paper is verbose and somewhat difficult to parse. Table 2 reports statistical significance but does not provide confidence intervals or which statistical tests were used. There also appears to be an imbalance in the hyper-parameter sweeps: the two proposed activation functions are explored across 175 and 450 combinations, respectively, whereas other hyper-parameters are examined over only a handful of settings. This imbalance could bias the experimental results.

### Questions
What distribution over pre-activations is used for the effective negative slope in Figure C-1(per-layer? per-epoch? across tasks?), and how sensitive are the conclusions to that choice? Please report variability across layers/epochs and any normalization applied before computing.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors provide a thorough investigation on the role of activation functions in addressing plasticity loss in continual learning. They propose two activation functions, Smooth-Leaky and Randomized Smooth-Leaky, which involve maintaining a non-zero derivative floor, optimizing negative-side responsiveness and ensuring smooth transitions at the origin. They evaluate these activations across supervised continual learning benchmarks and RL environments, showing their effectiveness in maintaining plasticity.

### Strengths
- The main strength of the paper lies in its rigorous evaluation of activation functions across continual learning benchmarks (supervised and RL).
- Proposed activation functions, and the concept of a "Goldilocks zone" for negative-side responsiveness, offer a interesting perspective on mitigating plasticity loss in continual learning. ​The findings highlight the role of activation functions in maintaining plasticity.
- Metrics like saturation fraction and dead-band width provide valuable insights into activation function behavior under non-stationary settings.
- The paper is well-structured and clearly written, with visualizations, explanations.

### Weaknesses
- While the paper discussed impact unconstrained adaptive slopes (e.g., PReLU), it does not explore alternative methods to guide parameters into the "Goldilocks zone" which itself is primarily identified empirically, but lacks theoretical grounding or formal analysis. Therefore, explaining why this range is optimal would help the paper overall.
- The paper does not explore how activation functions interact with different optimizers or learning rate schedules. For example, adaptive optimizers like Adam or RMSProp may behave differently with the proposed activations. Are the proposed activations universally effective across all continual learning settings? Since the dynamics of plasticity loss may vary significantly between tasks, a more nuanced analysis of task-specific performance could provide deeper insights.
- The reinforcement learning experiments focus solely on MuJoCo tasks, Have authors tried running experiments on domains like sparse reward or high-dimensional tasks?

### Questions
Please refer to the comments in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors explore the role of activation function design on plasticity in continual learning. The authors introduce the Smooth-Leaky and the Randomized Smooth-Leaky activation functions.The authors evaluate their proposed activation functions in the supervised class incremental continual learning benchmarks and in RL environments and show that the randomized smooth leaky activation function consistently outperforms existing activation functions.

### Strengths
- The empirical results clearly show that the proposed activation function randomized smooth-leaky consistently attains superior average online task accuracy. 
- The experiments consider a comprehensive set of baseline or existing activation functions.
- The discussion and experiments regarding the generalization gap, specifically, that plasticity loss can manifest in reduced training loss and test loss in different ways, is a positive touch to the paper which is not usually evaluated in this space. As the authors point out, maximizing plasticity via maximizing training loss may result in over-fitting. While earlier papers have argued that solely analyzing test loss in continual settings can be confounded by issues of regularization and generalization. The focus that the authors put here is a strength of this paper.

### Weaknesses
- There is a paper by Lewandowski et al. "Plastic Learning with Deep Fourier Features" which also examines the role of activation function design on plasticity. It would be useful if the authors could comment on this line of work, and ideally, incorporate this paper's contribution as a baseline competitor. 
- It would be worthwhile to apply a few existing SOTA methods for plasticity loss such as reset-based: SNR and regularization-based: L2 Init, and to generate a table with results for a reasonable set of METHOD X + ACTIVATION FUNCTION Y. When given an additional continual learning intervention how important is the choice of activation function?

### Questions
- Given the recent popularity of the SwiGLU, why isn't this activation function evaluated?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper looks at the effect of various characteristics of activation functions on the plasticity of neural networks on supervised continual learning and RL benchmarks. These characteristics include things like the slope of the activation function for negative inputs, where they identify a “Goldilocks zone,” and dead-band width which is the portion of the activation function domain that has a small gradient. They propose two different activation functions, Smooth-Leaky and Randomized Smooth-Leaky which are modifications of the Leaky-ReLU activation function with a smooth transition region rather than a kinked one.

### Strengths
- The initial part of the paper with the case studies on the negative slope and dead band width are valuable experiments for the community.
- The diversity of metrics and analyses presented for the experiments is quite high.

### Weaknesses
- The presentation of the paper can use some work. Most of the results for the experiments in Sections 3 and 4 are not presented in the main paper, and are instead presented in an Appendix. This is fine for one or two auxiliary results, but here, most of the main results for a significant portion of the paper are presented elsewhere. I’d ask that either the authors figure out a way to present a coherent story with results in the main paper, either by moving some parts of the paper to the appendix (not just the graphs), or if that is not possible in the page limit, consider another venue.
- The paper introduces two activation functions Smooth-Leaky and Randomized Smooth-Leaky. These activation functions introduce 3 and 4 new hyperparameters respectively which are searched for in a grid, with 175 and 450 combinations respectively. Compare this to the baseline of Leaky-ReLU which had 12 configurations evaluated. Furthermore, these hyperparameters were individually tuned on each setting. The amount of extra tuning given to the new methods is an unfair advantage, and could definitely be the cause of any improvement in performance rather than something inherent about the methods themselves.
- The metrics used to measure plasticity in the RL section seem to be flawed. If we look at just the rewards, Sigmoid seems to significantly outperform the other activation functions on 2/4 environments. Yet the plasticity score seems designed in a way to remove that effect, taking the median across environments to result in a number where the proposed method does better. A better approach could be to do some kind of normalization of the return and presenting the mean. Taking the median across 4 environments specifically excludes the environment where none of the activation functions except for Sigmoid are able to learn.
- For generalizability, there’s another metric presented which is a non-standard metric. It looks at the gap between train performance and a test environment which is some perturbation of the train environments. The metric reported compares the difference between the gap after cycle 3 and cycle1. First, it’s unclear what the perturbations are. Second, this metric has no notion of what the actual reward obtained in each of the settings was. A small gap with low absolute rewards is less preferable to a large gap with high absolute rewards, but this metric doesn’t capture that. Finally, measuring generalizability is much easier in the supervised setting, and has been done in several prior works. The results on those settings should be reported before creating such a metric.
- Tables 1, 2, and 4 are missing standard deviations.

### Questions
- Could you further clarify the reasoning behind the criteria for dead units defined for non-ReLU activation functions? For ReLU it makes sense because there is a region where the value of the activation function is 0 with slope 0. That doesn’t seem to be the case with most of the other activation functions studied, and it seems a bit odd to talk about dead-unit fractions.

### Soundness
2

### Presentation
1

### Contribution
2
