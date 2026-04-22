# Optimal Scaling Needs Optimal Norm

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Despite recent progress in optimal hyperparameter transfer under model and dataset scaling, no unifying explanatory principle has been established. For Adam and Scion optimizers, we discover that joint optimal scaling across model and dataset sizes is conditioned on a single invariant: the operator norm of the output layer. Across models with up to 1.3B parameters trained on up to 138B tokens, the optimal learning rate/batch size pair $(\eta^\ast, B^\ast)$ consistently has the same operator norm value — a phenomenon we term norm transfer. This constant norm condition is necessary but not sufficient: while for each dataset size, multiple $(\eta, B)$ reach the optimal norm, only a unique $(\eta^\ast, B^\ast)$ achieves the best loss. As a sufficient condition, we provide the first measurement of $(\eta^\ast, B^\ast)$ scaling with dataset size for Scion, and find that the scaling rules are consistent with those of Adam. Tuning per-layer-group learning rates also improves model performance, with the output layer being the most sensitive and hidden layers benefiting from lower learning rates. We provide practical insights on norm-guided optimal scaling and release our Distributed Scion (Disco) implementation with logs from over two thousand runs to support research on LLM training dynamics at scale.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper studies optimal tuning of LLMs as LR, model size, dataset size, and batch size vary.  Specifically, they consider tuning of norm-based optimizers such as Scion, where the operator norm of different layers is controlled as part of the optimization process.  Over a number of training runs, they identify that, at optimal settings of LR and B, the operator norm of the output layer is fairly consistent as both model and dataset size scale.  Further, they derive empirical power laws for optimal LR and B.  They also show that per-layer-group tuning of LRs can also help with Scion.

### Strengths
It's interesting to consider optimal hyperparameter scaling rules for norm-based optimizers, since this has previously mostly been done in AdamW, with and without maximal update parameterization.

Since norm-based approaches give new metrics that you can monitor, it is well-motivated to monitor them, and to derive insights from them.

I also don't mind the idea of stress-testing at a constant learning rate (even though this is definitely not state-of-the-art), as doing so can make things obvious that might not be when we use LR decay.  Similarly, it's perhaps okay to not use weight decay (but again, this deviates from the SOTA).

Some of the observations are definitely intriguing and thought-provoking.  For example, recent work is showing collapse/consistency in training loss curves (after normalization) across model sizes (https://arxiv.org/abs/2507.02119, https://arxiv.org/abs/2509.25087).  Here the operator norm of the output layer --- the first layer to get gradient from the loss --- is also consistent in a kind of normalized sense across scales as well.  I wonder if there’s likewise a connection between the output norm and scaling laws.  Especially since you use a constant LR, power laws for loss should hold at every step.

### Weaknesses
My overall view is that the way things are presented here is a bit of an overclaim, or a bit misleading for practitioners.  I have concerns about the real cause-and-effect, and whether the insights gained here are actionable.
- For example, we say that scaling is “governed” by the output layer operator norm, and we plot it as the independent variable, but it’s just something we observe, right?  We don’t scale or control it directly.  So I’m not sure how to use this, and secondly, correlation is not causation, right?
- In terms of actionable insights: the output norm seems like something you measure after a large-scale training run.  If you're training various runs, you wouldn't choose the one with lowest output norm, you'd choose the one with lowest loss, right?
- Moreover, even if your other hypotheses prove valid, how can you actually operationalize this?  How do you know the region of low norm sensitivity a priori, where you can exchange η for B?  Does this low-sensitivity region also transfer across scales?  “we cannot fully rely on the output norm as a guide to selecting optimal hyperparameters” – as a practitioner, how can you rely on it at all?

At least in the main paper, I would have liked more discussion of the case where you actually do use LR decay.  Do you still see output norm transfer?

Looking at the results as a whole, I'm not sure I *do* actually see norm transfer.  For example, Figure 1(a) and Figure 1(b) have different optimal output norms.  Perhaps they get even more different in other situations.  So it seems to be that output norm alignment is neither necessary nor sufficient.  So what *can* we really say about it???

More nitpicky:
- Methods section: How can we say that spectral norms (or MUP) “guarantee” hyperparameter transfer?  We already know that the amount of data affects the optimal HP settings, right?  So under what conditions do the HPs transfer?  With the same amount of data?  With a certain tokens-per-parameter ratio?  So in what sense is it a guarantee?
- “We briefly explain the idea behind each of them below.” – each of what?
- Tuning on a proxy model citing “(OpenAI et al., 2024; Gunter et al., 2024; Dey et al., 2024; Meta AI, 2025; Zuo et al., 2025)” – can you make the semantics of these citations more clear?  Like, does GPT-4 use MUP, and if not, what does it tune on a proxy model specifically?
- Why do we italicize “a single optimal batch size” in 3.2?  Is that surprising or notable somehow?
- Typo: “Later, a deeper understand* has been built”

### Questions
- Why isn’t there the same optimal output norm in Figure 1(a) and 1(b)?  One curve (d=256, D=2^33) is even plotted on both plots, right?
- What is the significance of RMS-norming the inputs to all layers?  Do we still need the same control on the operator norm?  Does it affect, e.g., how LR correlates with output norm?  You mention this might be one reason we get depth transfer --- could it be a reason we get norm transfer???
- Can we explain via theory, or even via intuition, why different layers would have different optimal LRs, when using norm-based optimizers?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper looks at how the optimal learning rate and batch size scale across model dimensions (width, depth) and dataset size for the Scion optimizer. In particular, they investigate how the norm of the final layer seems to be preserved across the optimal hyperparameter configurations, which they call norm transfer. These experiments are based on grid searches for relatively small (69M) Llama 3 models with additional normalization layers and generally no momentum or weight decay.

### Strengths
* The paper is well written and clearly has a lot of effort put into it.
* The topic of the paper (HP scaling across multiple dimensions) is interesting and relevant to the community.
* Good breadth of experiments and ablations.

### Weaknesses
* Some aspects of the experimental configuration choices may limit the relevance of the finding to typical training setups (e.g. no weight decay, no momentum, no learning rate schedule).
* It seems very likely that the norm transfer is simply a correlation with some other measure, rather than directly causing any interesting behaviors.
* I feel the paper somewhat lacks a clear practical takeaway. The norm observations can not directly be used (and may not hold with weight decay which is standard practice) and the HP scaling rules may also not hold for more typical training settings.

### Questions
* Could you clarify exactly which scaling rules and results hold in a more standard training setting (LR warmup + decay to zero, momentum, weight decay)?
* I think the paper would be stronger if you made an attempt to explain why the norm should transfer and when it does not. With weight decay it seems less likely, especially if you consider different (LR, WD) pairs. For the PyTorch AdamW version where the total decay scaling is (1 - LR * WD), different configurations with a constant LR*WD value often give similar final performance while affecting the norm differently.
* Do you believe your findings hold for other optimizers, e.g. AdamW?

### Soundness
2

### Presentation
4

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
This paper studies the optimal learning rate and batch size scaling in LLM training. Using the scion optimizer, they find that the optimal learning rate and batch size share the same operator norm of the output layer. This condition, however, is necessary and not sufficient, as different learning rates and batch sizes share the same operator norm. Through large-scale experiments, they provide empirical scaling laws for learning rate and batch size as functions of the dataset. They recover the known result that the optimal learning rate is proportional to the square root of batch size.

### Strengths
* Large-scale experiments
* The work unifies learning rate transfer and learning rate-batch size scaling laws
* The norm transfer result is intriguing (but I am unsure about the implications)

### Weaknesses
* The work is done on Scion optimizer, which is not well adopted in the field yet. 
* The results are empirical, and it's unclear why norm transfer phenomena occur.

### Questions
* I would like to request that the authors help me understand the implications of this work --- as an optimal norm may not imply optimal performance, what is the impact of norm transfer?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates optimal hyper-parameter scaling for LLM training. The authors claim that the joint optimal scaling over model and dataset size is governed by a single invariant, the operator norm of the output layer. They study this optimal norm, specifically for Scion/Muon optimizer. The authors also mention that having an optimal norm is a necessary but not sufficient condition.

### Strengths
Strengths:
- The paper has a clear experimental methodology, with strong theoretical explanations about the approach. 
- The paper tests norm transfer across a range of width, depth and data scales.
- The paper poses some very interesting questions and theoretical gaps for the research community towards the end of the main text.
- The authors release their Distributed Scion implementation, which can be helpful for the broader research community.

### Weaknesses
Scope for improvement:
- Since the paper only looks at the invariant optimal norm for the Scion/Muon optimizer, the applicability is narrow and cannot be generalized to other widely used optimizers.
- Most of the experiments in the paper are performed on a 69M parameter model, which is really small. The authors should shed some light on why they didn’t use bigger models which are more representative of the current SOTA model size. I am also curious how the optimal norm changes across different data regimes for a fixed set of models.
- The paper evaluates optimality solely through training loss (cross-entropy), without downstream task benchmarks. While training loss is standard for scaling law research, validating that norm-optimized configurations also optimize downstream performance would strengthen the claims. Even for small models (69M-1.3B), evaluating trends on standard benchmarks (e.g., HellaSwag, LAMBADA) would confirm that norm transfer reflects meaningful capability improvements, not just training dynamics artifacts. This is particularly important given the paper's claim of discovering a 'unifying principle' for optimal scaling.
- The anonymous github repo (https:// anonymous.4open.science/r/disco_iclr2026-E11D) seems empty, which raises some reproducibility concerns.

### Questions
Points 1, 2, 3 from weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
3
