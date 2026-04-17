# How Does Local Landscape Geometry Evolve in Language Model Pre-Training?

- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
The scale and expense of pre-training language models make efficient hyperparameter tuning essential, yet a principled guidance is still missing. Recent work shows that the geometry of loss landscape shapes training dynamics of neural networks and further informs hyperparameter choices. In this work, we analyze language model pre-training dynamics from a local landscape geometry perspective.
Our study reveals two distinct phases. In the *early* phase, sharpness of the local landscape is initially high, leading to instability and loss plateaus under large learning rates (LRs). Later, the landscape shifts from sharp to flatter regions. This dynamic explains the necessity of LR warmup and further suggests that larger peak LRs require proportionally longer warmup periods. In the *late* phase, the local landscape is governed by the gradient noise scale. Through diffusion-limit analysis, we prove a depth–flatness trade-off: high noise from smaller batches widens the loss basin, whereas reduced noise from larger batches deepens it. This theory motivates a dynamic batch-size (BS) scheduler that begins with a small BS and increases it late in training. Together, we provide a unified account of loss landscape evolution, which translates into actionable tuning strategies for large-scale pre-training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates how the local loss landscape geometry evolves during the pre-training of large language models. The authors identify two distinct phases in training dynamics.

Early phase: At the beginning of training, the loss landscape exhibits high sharpness. During this stage, the learning rate governs the training stability. Therefore, a **learning rate warmup** is essential to move from sharp to flatter regions safely.

Late phase: Once the model enters a more stable training regime, the batch size and the resulting gradient noise scale become the dominant factors shaping the local landscape. This implies the importance of a **dynamic batch-size schedule**.

### Strengths
**Systematic analysis**: The paper provides a systematic framework to analyze how the local loss landscape evolves during pre-training.

**Empirical insights**: It empirically identifies a two-phase transition in sharpness and links this dynamic to training stability and learning rate schedules.

**Practical implications**: The theoretical insights are translated into practical training strategies (learning rate warmup and dynamic batch-size schedule).

### Weaknesses
**Limited architecture and optimizer scope**: As noted in the paper’s limitations, the analysis is restricted to the LLaMA-2 architecture and the AdamW optimizer. It remains unclear whether the findings would generalize to other architectures or optimization algorithms.

**Lack of analysis on interaction effects**: The paper analyzes batch size ramping and learning rate decay in isolation. The interaction effects between BS ramping and LR decay therefore remain unclear.

**Theoretical simplifications**: Assumptions such as noise covariance being proportional to M, time-invariant M, equilibrium, and local quadratic approximation are introduced for theoretical convenience, but may not hold in actual training dynamics.

**Overstated novelty discussion**: The authors point out that prior works are “largely restricted to small-scale networks” and emphasize that their study provides the “first systematic study of how local landscape geometry evolves in large-scale language model pre-training.” To make this claim more meaningful, it would be helpful to more clearly articulate the qualitative differences between small-scale networks and large-scale language models. Moreover, discussing how the observed dynamics might change when scaling beyond the 93M and 170M parameter models used in this work would further strengthen the argument.

### Questions
Minor Errors

Line 129:  The Hadamard product are ->  The Hadamard product is

Line 132: near an local -> near a local

Line 134: at i-the example -> at i-th example

Line 146: gradient covariance at $\theta$ -> gradient covariance at $\theta^\star$

Line 157: Our experiments varies -> Our experiments vary

Line 67: landscapes gradually widens/flattens -> landscapes gradually widen/flatten

Line 266: in the most-case directions -> in the most directions

Line 279: This rational further suggests -> This rationale further suggests

Line 322: two key question remains -> two key questions remain

Line 172: Loss spikes and plateau -> Loss spikes and plateaus

Line 238:  only a sufficient small ->  only a sufficiently small

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors analyze the "local landscape" of the loss as training progresses. The paper is organized around two phases: "early" (when loss is sharp) and "late" when loss is flat. Within these two phases, they ask and answer two questions related to training dynamics. Based on these theoretical results, they offer heuristics/principles which they instantiate and show experimentally.

The questions/answers are:

* Why do short warmups produce instability? (Because the LR rises too quickly when the model is still too sharp) 
* Why do spike and plateaus only happen at the beginning of training? [Reviewer's note: they do not only occur then?] Because things are smoother later on.
* "Why is there a trade-off between widening and deepening the basin?" because temperature affects basin selection and different temperatures prefer different basins.
* How does BS impact this trade-off? BS impacts temperature.

### Strengths
At a high level, the paper reads really nicely. It's easy to follow: the math, the figures, the simulation results, and the real results are all presented in a coherent, digestible way. It would, in many ways, make a nice tutorial/review about some of the concepts in this space, linking theory to practice. 

I especially like the visualizations of sharpness along 1D slices. This is a nice visual tool, though I feel like the authors don't actually make that much use of it.

I think the experiments mostly support the claims made in the paper (mostly, though I have a quibble).

### Weaknesses
While it is nicely written, the authors claim much too much novelty. They apply existing tools to familiar settings but without much new insight. They don't seem to arrive at any substantially different conclusions. The math is sometimes a bit different (though usually not), but when it is, it provides seemingly no insight over what's already known.

Given that there is in fact quite a lot of prior art for many of these findings (which, admittedly, the authors do frequently cite), it would be good if they empirically compared their prescriptions to those other sources.

## Most results are known, both theoretically and empirically

To be blunt, I feel like the majority of the insights of this paper are fairly well known, both from an empirical and a theoretical perspective. 
For example, I think theorem 5.1 is a repackaging of known results (e.g. Jastrzębski et al. (2018) eqn 12 is I think the two-basin version of this).

To drive the point home, I asked ChatGPT: "does the batch size impact the type of basin found when using Adam?" and it produced a detailed, similar explanation with cites to existing theoretical and empirical work." (To be clear, I am not insinuating the authors had ChatGPT write this paper, just that the results are standard.)

I could dig up additional academic sources, but I'm fairly confident that I could pose ChatGPT most of the questions addressed in this paper and reach largely similar results, getting already published papers, both empirical and theoretical.

As another example, it's well known that BS and LR trade off with one another, including using theory from SDEs! One example: https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/

The batch size warmup is also known, with known theoretical grounding relating to CBS/gradient noise scale: https://arxiv.org/pdf/2505.23971

And even the "LR warmup needs to be longer for higher max LRs" is known, including a similar theoretical justification. The authors dismiss Kalra and Barkeshli  as being focused on resnets, but they also do studies on GPT-2 style transformers with much longer warmups than the authors claim in their description of the work?

## The principles/recipes are no more actionable than what is known/done

The "recipe" for LR warmup length literally just says... "the larger the peak LR, the longer the warmup should be". How much longer? proportional to LR change? proportional to LR^2? 

To me, a recipe would suggest a particular heuristic or criterion for, say, warming up the batch size. The authors say "Ramp the BS once loss reduction becomes marginal." but then seemingly the authors just use a fixed step/token count for when to ramp? How is this an improvement over what we know? (CF Merill et al 2025 https://arxiv.org/pdf/2505.23971 which actually do provide a criterion for when to scale that can be tracked during training)

### Questions
To be a bit less polemical, regarding the batch size experiments: the BS vs LR doesn't really match up super closely: increasing BS and decreasing LR both make the loss go down, great. The Princeton link I pasted above would say that Adam LR should scale with sqrt(BS) rather than linearly. If you ran your experiment using sqrt(BS) scaling, would it match more closely? I guess yes. 

Merrill et al explore batch size warmup from a CBS perspective, deriving a specific, trajectory-driven way of determining when to warm up batch size. How does your approach compare?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper admits a two-phase analysis on language model pretraining, to address two crucial decisions required: the learning rate warmup phenomenon and batch-size scheduling.
The authors note the common hyperparameter settings used in the literature, wherein a large number of warmup steps are used for the maximal stable learning rate (LR) applied, followed by a stable or annealing LR. 
Also, the use of arbitrary batch size schedules is reported in the literature.
The locally quadratic loss landscape analysis in the paper looks at two distinct training stages relying on SGD: (i) *early phase*: where the loss landscape is sharper and the imperative is `to flatten`, and thereby LR warmup, to begin optimization from a less sharp region, or attractor basins; (ii) *late phase*: where the loss landscape tends to a flatter region owing to SGD convergence, and the imperative is `to deepen` the loss basin reached for an improved loss, which is done by either LR decay or a batch size ramp up.

### Strengths
* Strong motivation and setup, to address two important empirical choices in language model pretraining, which are often the first hyperparameters adjusted for any new task, hardware.

* Fairly clear math and assumptions for the analytical explanation of the pre-training dynamics using the quadratic approximation of the loss landscape around the theoretical minima; empirical analysis supporting claims.

* Clear conditions given on the practical rule-of-thumb in setting the LR warmup and Batch size ramping, for a warmup-stable learning rate schedule.

### Weaknesses
* The terminology and goal of the analysis can be made cleaner; that is, the terms: sharpness, flat minima, wide basin, deep basin could be clarified independently early on before the lemmas and theorems.
  * It is hard to understand if the direction of analysis emerges as a result of wanting to go from sharp to flat early on (and wide to deep in the later phase), or vice versa. Especially since the locally quadratic assumption does not necessarily hold at initialization.

* Given LR decays are an important discussion in managing LR schedules and thus scaling and the laws fit on this data, the interaction of both LR decay and increasing batch size feels underexplored.

* Given the relatively limited empirical experimental scope, a broader *Limitations* section is warranted: especially for the unknowns such as the role of LR annealing choice; how the timing and the number of batch ramp ups matter.

### Questions
Below is an enumeration of various questions and suggestions.

Please note that the rating will go up contingent on the points below, with more weightage on the following points: 1, 2, 4, 5, 9.

1\. L76-81: Could the authors please explain (or elaborate here) why the deepening of the loss basin at *late-stage* training is crucial and different from the flat-minima conversation around SGD convergence?

2\. L106-114: Could the authors include [1] here and also contrast their early-late training phase interpretation with the loss landscape perspective given here?

3\. Equation 3: What is the role or effect of the $B$ in the denominator given the $1/n$ already included in $\Sigma ({\theta}^{*})$ (L145-146)?

4\. Figure 2: It appears that only the larger LRs (in the grid shown here) lead to spikes. Could the authors intuit why and how the finding here regarding the edge-of-stable LRs explains this phenomenon? 

* How does zero-warmup (not in Figure 2) actually affect the findings here, and also the actual effect on *moving away from sharp loss landscapes*? Given we expect noisy updates in the beginning few update steps in most cases, does no warmup and a small enough batch size have the same effect?

5\. L178: Could the authors make an overall comment on how the absence of LR decay influences the assumptions and findings, and thereby the practical recommendations? Given that recent literature suggests LR decay is crucial for a model before modeling its loss obtained for scaling law derivations, downstream performances, finetuning, etc.

6\. Figure 3: Could the authors please provide some guidance on how to read Figure 3? What exactly is being plotted with the orange line, and what are $u_1$ and $u_w$?

7\.1. Figure 4 (top): Which layer matrix are the eigenvalues being reported?

7\.2. Figure 4 (bottom): Does this perform a perturbation on all the weights?

8\. Lemma 4.2: Could the authors explain (or intuit) the definition of $z$ and therefore $z_l$?

9\. L270-291 and Figure 3: Does the figure suggest that anything *more* than the optimal LR warmup length leads to a worse loss? What, thus, is a *reasonable range* (L290) in practice?

10\. Figure 10: Could the settings here be marked differently (markers/linestyles) and not just with transparency?

11\. Personally, took me a long time to understand why we want to move away from a wide basin, until I realized we are talking about a region around the minima (a basin) and not the flat minima we converge to. Therefore, the finding of a small batch required for a wider basin felt counterintuitive and required re-reading of Section 5. The motivation, analysis, and conclusion can be presented much more cleanly, building up to a general batch size schedule. Additional comments on this or identifying future directions w.r.t. role of LR schedules could be more explicit. 

12\. L266, L469:  What does `most-case direction` mean?

---

References:

[1] Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs, Bergsma et al., 2025, arXiv:2502.15938 [cs.LG].

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to systematically study the geometrical evolution of the local loss landscape during LLM pre-training and correlate it with hyperparameter tuning strategies. The authors divide the process into two phases. In the first phase (early), authors reports a "from sharp to flat" evolutionary trend and, based on this, provides a geometrical explanation for the necessity of LR warmup via linear stability analysis. In the second phase (late), this paper argues that the landscape geometry is dominated by gradient noise. Through the continuous-time limit of Stochastic Differential Equations (SDEs) and the principle of free energy minimization, it reveals how batch size regulates a "depth-flatness" trade-off, thereby proposing a dynamic batch size scheduling strategy.

### Strengths
1. This paper intuitively documents and reports the "from sharp to flat" macro-dynamic trend in LLM pre-training, which is an important empirical finding that contradicts studies on small-scale models.

2. The proposed hyperparameter tuning strategies are logical, easy to implement, and experimentally shown to significantly improve training efficiency, contributing directly to reducing the cost of large model training.

### Weaknesses
1.  In Section 4, this paper attempts to explain the loss spikes and plateaus of the early training phase. The core argument is: first, it empirically observes that the initial landscape is very sharp (high Hessian eigenvalues), and then points out that a large LR on such a sharp landscape causes instability. To theoretically support this, the author borrows linear stability analysis from standard gradient descent (Lemmas 4.1 and 4.2). However, this analysis is strictly established on a **deterministic (noise-free), quadratic model centered around a local minimizer $\theta^{*}$** (Equation 4). **The key issue here is the applicability of this "extrapolation"**: While this paper does not claim the quadratic model *fits* the initial state, it *assumes* that the stability condition derived from this highly simplified, local-convergence model (i.e., $\eta < 2/\lambda_{max}(S)$) can effectively explain the dynamics of the earliest training phase (far from any minimizer, highly non-convex, and noisy). This is a strong assumption. The true early-stage dynamics are highly complex. Attributing the instability primarily to the simple linear interaction between LR and the local Hessian's max eigenvalue is likely an oversimplification, ignoring other non-linear or stochastic noise effects. Therefore, while the conclusion "high sharpness + high LR = instability" is intuitive and matches the data, using Lemmas 4.1 and 4.2 as its primary theoretical basis acts more as an insightful **analogy** than a rigorous proof for this specific phase. The validity of this explanation depends on the extent to which this local linear approximation dominates the early global dynamics, which is not sufficiently justified in this paper.

2.  The core theory in Section 5 (Prop 5.1 and Thm 5.1) relies on the SDE continuous-time limit, which assumes $\eta \to 0$. This contradicts modern LLM training practices (including the use of relatively large peak LRs in this paper's Section 4 experiments). Although the theory's key prediction (noise scale $\tau \propto \eta/B$) appears to match the experiments (Figure 8), this treats a heuristic approximation as a rigorous explanation.

3.  The study is limited to 93M and 170M parameter models under the LLaMA-2 architecture, which differs significantly from current mainstream model sizes. Whether its conclusions hold for much larger models remains unknown.

4.  This paper attributes the necessity of warmup to an external factor: the "landscape sharpness". However, a more direct and well-known explanation lies in the internal flaws of the Adam optimizer itself: its second-moment estimate ($v_t$) has high variance in the early stages, and its initial update degenerates into unstable "sign descent". The loss spikes observed in this paper are highly consistent with these known optimizer startup problems. This paper fails to clearly disentangle whether the observed instability originates from the landscape geometry or simply from the well-known startup deficiencies of the Adam optimizer.

### Questions
1.  Regarding the early-stage stability analysis: How do the authors justify that a noise-free, quadratic model (Eq 4), based on the neighborhood of a local minimizer, can effectively explain the instability phenomena in the earliest, far-from-equilibrium, and highly stochastic phase of training? Is there a more suitable theoretical model to describe this "chaotic initial" phase?

2.  Regarding the SDE limit and steady-state assumption: Given that LLM pre-training uses finite, large learning rates and is terminated long before reaching a theoretical steady state (stationary distribution), can the authors provide additional evidence or arguments to support the approximate validity of the SDE limit and free energy minimization theory in this scenario?

3.  Regarding the deeper reasons for blockwise heterogeneity: The authors attribute the ordering difference with Wang et al. to measurement methods and gradient sparsity. This raises a question: should we focus on the "intrinsic" Hessian geometry defined by the architecture, or the "effective" dynamic geometry jointly determined by data flow and the optimizer?

### Soundness
2

### Presentation
3

### Contribution
2
