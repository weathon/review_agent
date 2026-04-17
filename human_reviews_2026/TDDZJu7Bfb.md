# On Monotonicity in AI Alignment

- Decision: Reject
- Scores: 6, 2, 6, 6

## Abstract
Comparison-based preference learning has become central to the alignment of AI models with human preferences. However, these methods may behave counter-intuitively. After empirically observing that, when accounting for a preference for response y over z, the model may actually decrease the probability (and reward) of generating y (an observation also made by others), this paper investigates the root causes of (non) monotonicity, for a general comparison-based preference learning framework that subsumes Direct Preference Optimization (DPO), Generalized Preference Optimization (GPO) and Generalized Bradley-Terry (GBT). Under mild assumptions, we prove that such methods still satisfy what we call local pairwise monotonicity. We also provide a bouquet of formalizations of monotonicity, and identify sufficient conditions for their guarantee, thereby providing a toolbox to evaluate how prone learning models are to monotonicity violations. These results clarify the limitations of current methods and provide guidance for developing more trustworthy preference learning algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper investigates "monotonicity violations" in AI alignment, where training a model on a preference for response 'y' over 'z' can counterintuitively cause the score of 'y' to decrease.

The main finding is a proof that while individual monotonicity can fail, a general class of methods (including DPO, GPO, and GBT) still guarantees **local pairwise monotonicity**. This means that near a stable solution, the *difference* in score between the preferred and rejected responses ($s_y - s_z$) will reliably increase.

The paper also formalizes a "bouquet" of different monotonicity types and identifies the stricter conditions required for stronger guarantees, which are often too demanding for complex language models.

### Strengths
1. Paper offers comprehensive theoretical take on monotonicity in post-training for LLMs on preference learning
2. Offers both weaker and stronger guarantees depending on assumptions

### Weaknesses
1. Exposition quite terse, some plots illustrating the intuition of local pairwise monotonicity would be helpful

### Questions
1. While I appreciate that this is a theory paper, what are the practical implications of this framework? Is there any insight we can use to improve the robustness during post-training?

### Soundness
3

### Presentation
3

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
The paper's goal is to study so-called "local pairwise monotonicity," in the context of RLHF: if a response y is preferred to response z, in terms of an improved score, then, will this difference or improvement (between the two responses) be increasing (monotonicity) in some given parameter (\theta); or equivalently, whether there's monotonicity in the likelihood ratio of the two scores’ corresponding probability distribution (soft max).   

The paper then goes on to study the above in three specific aspects: a) the addition of an unequivocal comparison, b) relative to an intensification of a comparison, and c) infinitesimal deviations from a critical point (e.g., an optimum or zero-gradient point).

### Strengths
The authors apparently know what they want to study.

### Weaknesses
The problem is, we have no idea what's the implication of their findings in terms of helping with either SFT or inference. There are no numerical or experimental results.

### Questions
It seems that all four theorems are consequences of certain degrees of continuity/smoothness in an \epsilon neighborhood such that the "local pairwise monotonicity" shows. If so, then once moving out of those neighborhoods the property won't hold. Then, isn't it true that the application domain of those results is basically vacuous?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
< Summary >

This paper investigates monotonicity properties in comparison-based preference learning methods used for AI alignment. 
The motivation is derived by well-known empirical observation that existing methods like DPO can counterintuitively decrease the score of the preferred response during training, despite increasing the margin between preferred and rejected responses.
To address this, the paper introduces a general framework encompassing Bradley-Terry (BT), Direct Preference Optimization (DPO), Generalized Preference Optimization (GPO), and Generalized Bradley-Terry (GBT) models and identifies what can be guaranteed and its condition. The authors formalize several flavors of monotonicity, including pairwise monotonicity, individual-score monotonicity, and individual-probability monotonicity. The findings suggest that while local pairwise monotonicity holds under mild conditions, stronger global guarantees require restrictive assumptions unlikely to hold in practice for language models.

### Strengths
< Strength >

- The paper addresses a practically significant issue in AI alignment. The empirical observation that preferred responses can have decreasing scores during training is widely-known concern for its reliability. The motivating example in Figure 1 effectively demonstrates this counterintuitive behavior across multiple Llama models.
- The general formulation in Section 3 successfully unifies multiple existing methods (BT, DPO, GPO, GBT) under a common loss structure and thereby enables unified study of monotonicity across a broad class of existing methods
- The mathematical rigor is generally sound, and the taxonomy of monotonicity types such as pairwise, local vs. global, score, probability, is well-organized and clarifies what different guarantees mean
- Theorem 5 provides necessary and sufficient conditions for one-step gradient descent monotonicity, which is import to actual training practice.

### Weaknesses
< Weakness >

- All 6 models tested are from the Llama family (3.1 8B, 3.2 3B, 3.2 1B with base/instruct variants) and this can raise concerns about generalizability. Testing on other architectures (e.g., Qwen) would strengthen confidence that findings aren't specific to Llama's particular parameterization.
- Although the paper is a theoretical analysis paper, it lacks a practical Interpretation or experiments. It would be great if the paper mention about what the theoretical guarantees mean for practitioners and show a small scale toy example. The paper doesn't clearly connect the mathematical properties to desirable empirical outcomes such as win rates or human evaluations
- The paper identifies when monotonicity fails, which is "unlikely to hold" for exponentially large response spaces). But it doesn't propose algorithmic modifications or training procedures to improve monotonicity beyond what current methods achieve. Section 5 provides sufficient conditions but no constructive way to enforce them during training.

< Minor issues >

- The paper introduces many symbols with minimal space. A notation table would improve readability.

### Questions
As a reviewer with a practical focus, I have a few questions

- The experiments span six models within the Llama family. To help assess the generality of the results, would you consider adding a small non-Llama baseline (e.g. Qwen-1B)? This could strengthen confidence that the findings are not family-specific.
- I would find it helpful if the authors could elaborate on the practical implications of these findings for real-world alignment applications. For instance, what should practitioners take away from knowing that local pairwise monotonicity holds? How might this inform decisions about algorithm selection, hyperparameter tuning?
- In standard DPO formulations, the “implicit reward” is often expressed via the ratio between the training and reference policies rather than directly as a logit score ​of $s_{y|x}(\theta)$. Could you clarify how score relates to the DPO implicit reward and how this mapping affects the stated monotonicity guarantees?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
A known problem in LLM fine-tuning is that pairwise monotonicity is not satisfied. This means that if in a pair $A > B$, the odds of outputting $A$ can decrease. The authors re-establish this in figures early on.

The authors define various forms of monotonicity in Sections 4 and 5, before proving a local pairwise guarantee and giving sufficient conditions for global pairwise, individual-score, individual-probability, and gradient-descent monotonicity. Section 4 in particular is where their contributions on pairswise monotonicity are detailed.

### Strengths
* Math is well-done; I did not make a "deep dive" but was able to follow the math and did not catch any errors or inconsistencies. A substantial piece of this paper is dedicated to theory, so this is fairly significant.

* The paper is well-motivated; the main novelty (monotonicity taxonomy and local pairwise guarantee) are clear and potential impacts with future work are clear.

* Overall, the paper is well-written. The Structure is clear and notation is consistent.

### Weaknesses
*The motivating example is good, but the figure is small. I would like to see a zero line, larger fonts, and clearer panel labels.

*The claim of a "toolbox" (as written in the abstract) feels somewhat strong to me. The authors do formalize several forms of monotonicity, which is appreciated, but there is not even a minimal empirical example or guidance on how to use and interpret results. I recognize space constraints but a clearer link between the sections would be appreciated.

### Questions
* Suggestion: as in weaknesses, I think a larger/better figure 1 would help.

* Some more anchoring in section 5 (as I stated in weaknesses) could help.

I could be persuaded to raise my score further primarily with more discussion on section 5; to me, this felt disconnected in the context of the paper. I understand why it is there, I understand why it is more "future-facing", but making the connections/flow and making it more clear and obvious the direct connection to the prior section and the value it provides in its current state would be helpful for me.

### Soundness
4

### Presentation
3

### Contribution
3
