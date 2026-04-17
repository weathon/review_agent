# What Is Preference Optimization Doing, How and Why?

- Decision: Reject
- Scores: 6, 2, 6, 2

## Abstract
Preference optimization (PO) is indispensable for large language models (LLMs), with methods such as direct preference optimization (DPO) and proximal policy optimization (PPO) achieving great success. A common belief is that DPO is supervised learning while PPO is reinforcement learning, yet deeper analyses for the reasons underlying these differences remain lacking. To fill this gap, we analyze their optimization dynamics, revealing distinct algorithmic behaviors and comprehending the causes of their differences.
First, we examine the target directions of gradient-based updates and find that DPO follows stable targets, whereas PPO follows dynamic targets that balance exploration and exploitation, thus validating the common belief from a new perspective. 
Second, we examine the roles of positive learning, negative learning, and loss reweighting, which are three key components in PO methods. Our analyses reveal that these components play fairly different roles. In DPO, positive and negative learning jointly shape the learning targets meanwhile mutually offset each other.
However, loss reweighting in DPO acts less as a reward signal but more as a regularizer to mitigate overfitting. In PPO, negative learning primarily supports exploration rather than determining the targets. Meanwhile, loss reweighting, related to absolute values of token-level advantages, indicates the distinct roles of token groups in updating targets. Given these findings, we conduct carefully designed ablation studies to further examine how controlling these dynamics impacts optimization efficiency and practical performance. The insights gained from our analyses not only deepen the understanding of PO methods but also inspire the development of more preference-aligned LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the training dynamics of preference optimisation when using a DPO and a PPO approach. To this end, they analyse the gradient alignment condition and look in more detail at the effect of the positive components, the negative components individually, as well as the impact of the gradient weight split into three tertiles. Moreover, they conduct ablation studies of dynamically adapting the positive and negative components, as well as the weighting, during preference optimisation training.

This paper gives a novel insight into the training dynamics of what is happening during PO in more detail, proposes some interesting hypotheses of the observed effects, and finally makes some initial proposals on how to improve the performance/gradient alignment when doing PO.

### Strengths
In my opinion, this paper introduces some new and novel insights into what is actually happening during preference optimisation, with a particular focus on the positive and negative components during training. I found their insight that DPO seems to overfit on the positive components and later focuses on the negative components, particularly interesting. I also appreciate their follow-up experiments in the appendix, which further validate their hypothesis.

In general, I find the paper mostly well written, and it's clear to follow. The experiments that are provided make sense to me and support the claims that the authors are making.

### Weaknesses
I think there are specific ways in which the paper could be further improved upon:
- The experiments lack any confidence interval, standard error, or at least standard deviation, which could indicate that the results are actually statistically significant. Especially in the win-rate experiments, reported in Figures 4b, d, e, and f (basically all the PPO ablations), the values of the win-rates seem to be within $\pm2$ pp, and I have a suspicion that they may not be statistically significant. Naturally, I understand that running the experiment multiple times is computationally expensive; therefore, the authors could consider, for example, bootstrapped confidence intervals of the test set. 
- slightly related to the previous weakness, the proposed improvements on DPO and PPO, displayed in section 4, do not really seem to make a strong (or any) improvement over normal DPO and PPO. While we gain valuable insight into the dynamics of learning, the proposed solutions based on these insights do not seem particularly compelling. (Yet I understand that this is only a subpart of the paper, and the primary focus is on the insights.)
- I understand that the authors focus on one single model to demonstrate all the findings of the PO dynamics. I wonder how much these results are a result of the backbone model itself? Have you tested these experiments on a slightly newer backbone architecture, of similar size (e.g. Qwen25-3B, Llama3.2-3B, Gemma3-4B), and are the results still consistent?
- In Figure 4, to me it is not immediately clear what Cases 1,2,3 are, even after trying to find them in the text. Maybe the legend can be renamed
- Figures 1c and 2c are hard to distinguish between the different shades of green, and the curves often overlap.

### Questions
- Why are the win-rates of the DPO so much higher than the PPO approach? I know this is not directly linked to the insights of your paper, but I feel like they should be in the same range to make comparable claims that apply to both.
- What are cases 1,2, and 3 in Figure 4?
- Are the results actually statistically significant?
- Are the results consistent when using different LLM backbones?

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
4

### Summary
This paper proposes a metric, *i.e.*, gradient alignment, to quantize the contribution of gradient descent to the log-probability of the final answer.
The optimization dynamics of two popular post-training algorithms, *i.e.*, DPO and PPO, are then analyzed.
The conclusions include

* DPO behaves like supervised fine-tuning as it has relatively stable targets
* As training progresses, negative learning dominates target shaping, while positive learning prevents collapse
* The implicit reward is not reliable but primarily serves as a regularizer to mitigate over-fitting
* PPO behaves like reinforcement learning as its exploration covers a broad range of conflicting responses
* Positive learning encourages discovery for new targets while negative learning fosters further exploration
* Loss re-weighting controls exploration

Multiple variants, *i.e.*, cDPO, cPPO, hPPO, are also proposed to ablate the effect of negative learning.

### Strengths
* The components of preference learning, *e.g.*, positive and negative learning and loss re-weighting, are analyzed thoroughly.
* Ablation study strengthens the persuasiveness of the conclusions and provides insights for future research.

### Weaknesses
I deem that several logical flaws hinders the soundness of the conclusions, so I lean to reject the paper.
I would like to raise my score if these concerns are well addressed.

* L105: Why does the distinction between SFT and RL lie in whether they have relatively stable targets?
I deem the difference between SFT and RL lies in whether they learn from demonstrations or rewards.
* L134 (Minor): I do not think the objective is inherently non-differentiable.
* L143: It is not very clear to me why the log-probability of final answer rather than the ground truth is considered.
L108 claimed that SFT is expected to steadily progress toward the targets, while the final answer is not necessarily the target.
* L157 (Minor): I deem the design of gradient alignment can be regarded as extension of [1], which may be cited and discussed.
* L160: It is claimed that the difference between SFT and RL lies in whether they have stable objectives, and here it is classified based on the value of gradient alignment.
I understand that positive gradient alignment indicates that the gradient descent increases the log-probability of the final answer.
Why does this also indicates a stable objective?
* L168: Only a single setting is performed so that it is not clear how well the conclusions can be generalized.
* L315 (Minor): I think $\hat{A}$ inherently can be negative without estimation and normalization.

[1] Estimating Training Data Influence by Tracing Gradient Descent, NeurIPS 2020.

### Questions
* L255: What is the unbiasedness of learning objective of DPO?
Why does that only hold at the optimal parameter?
* L285: Is there any evidence to support the proposal?
* L335: Is there any reference to support such definition for positive and negative learning and loss reweighting?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper analyzes the optimization dynamics of Direct Preference Optimization (DPO) and Proximal Policy Optimization (PPO) to elucidate the distinct roles of positive learning, negative learning, and loss reweighting. The authors introduce a "gradient alignment" metric to investigate how learning targets evolve during training. The analysis reveals that in DPO, positive/negative learning jointly shape targets while loss reweighting acts as a regularizer. In contrast, PPO uses negative learning to aid exploration, and its loss reweighting differentiates the roles of token groups in updating the policy. The authors substantiate these findings with ablation studies examining the practical performance implications of controlling these dynamic components.

### Strengths
1. The paper provides a deep, mechanistic explanation for the oft-discussed differences between DPO and PPO by skillfully analyzing their respective training dynamics.
2. The introduction of the 'gradient alignment' metric is a notable contribution, offering an effective method to quantify and inspect the optimization dynamics of preference alignment algorithms.
3. The findings are clear and insightful, providing actionable explanations for the distinct roles of positive learning, negative learning, and loss reweighting.
4. The paper's analytical claims are well-supported by sufficient empirical validation, including targeted ablation studies that connect the observed dynamics to practical performance.

### Weaknesses
1. The paper provides extensive empirical analysis, but it lacks a rigorous theoretical foundation to formally explain the underlying reasons for the observed phenomena.
2. The analysis could be strengthened by incorporating the distribution of key data properties. For instance, analyzing the distributions of the DPO reweighting term ($\omega$) and the PPO absolute advantage ($|\hat A|$), both globally and within subgroups, would provide a more complete picture of their impact.
3. The 'gradient alignment' metric is a first-order approximation that does not account for the adaptive, non-linear dynamics of optimizers like AdamW or the non-convex landscape. 
4. Minor Issues on Presentation:
* The conclusion (Section 5) is somewhat lengthy and could be compressed. This would create space to either expand the main analysis or move valuable insights from the appendices (e.g., parts of Appendix D) into the main paper.
* The experimental cases (e.g., "Case 1-3") in Figure 4 are not clearly explained in the text, making the results difficult to interpret fully.

### Questions
1. To strengthen the analysis on reweighting, could the authors show the distributions of the DPO term ($\omega$) and the PPO absolute advantage ($|\hat A|$)? It would be insightful to see this for the entire dataset and within the 'top', 'middle', and 'bottom' subgroups. This might also help justify the current split into three equal-sized groups, or perhaps suggest a more natural, data-driven way to segment the data.
2. The cDPO (controlled DPO) in Appendix E explores a gradual shift from positive to negative learning. As a clearer ablation to test the "role-switching" hypothesis, what would be the result of a hard switch? (e.g., training only with the positive learning component for the first half of training, and only with the negative component for the second half).
3. In lines 349-351, the authors state that in PPO, 'positive learning is stable in shaping the learning targets.' This is a key finding. Could the authors provide any further mathematical derivation or theoretical intuition to explain why this is the case, while negative learning's role is relegated to exploration?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies what preference optimization (PO) does by analyzing optimization dynamics for DPO and PPO through a gradient-alignment metric that measures the dot product between the PO objective gradient and the gradient of expected NLL on final responses. It reports that DPO behaves like supervised learning with targets implicitly shaped by both positive and negative learning, while PPO behaves like reinforcement learning with exploration near orthogonal targets; loss reweighting acts more like regularization in DPO and carries token-level information in PPO. The authors further test behavior-control variants (cDPO, cPPO, hPPO) and show illustrative win-rate gains on AlpacaEval with Pythia-2.8B.

### Strengths
- Well-structured decomposition of PO into positive/negative learning and reweights; which connects intuitively to training heuristics.
- The *gradient alignment* tool is simple, and allows concrete insights.
- Evaluates variants of PO (cDPO, cPPO, hPPO) from the insights acquired from the analysis.

### Weaknesses
- Insufficient breadth and scale of experiments
    - The paper uses a single base model (Pythia-2.8b) and narrow task sets. The claims in the paper about "what PO is doing" should be tested on larger models, multiple families, and varied domains.
    - Although the motivation of the paper seems promising, empirical proof of PO tendency should be backed up with much more depth.
- Lack of theoretical framing
    - Tightening the theoretical relation between $G$ and the performance can strenghthen the motivation of the paper, when extensive experiments is infeasible.

### Questions
- Refer to the weakness section.

### Soundness
3

### Presentation
4

### Contribution
2
