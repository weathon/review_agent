# Zero-Overhead Introspection for Adaptive Test-Time Compute

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 2

## Abstract
Large language models excel at reasoning but lack key aspects of introspection, including the ability to anticipate their own success and the computation required to achieve it. Humans use real-time introspection to decide how much effort to invest, when to make multiple attempts, when to stop, and when to signal success or failure. Without this ability, LLMs struggle to make intelligent meta-cognition decisions. Test-time scaling methods such as Best-of-N drive up cost and latency by using a fixed budget of samples regardless of the marginal benefit of each one at any point in generation, and the absence of confidence signals can mislead people, prevent appropriate escalation to better tools, and undermine trustworthiness. Learned verifiers or reward models can provide confidence estimates, but do not enable adaptive inference and add substantial inference cost by requiring extra models or forward passes. We present ZIP-RC, which equips models with zero-overhead introspective predictions of reward and cost. At every token during generation, ZIP-RC reuses reserved or unused logits in the same forward pass as next-token prediction to output a joint distribution over final reward and remaining length—no extra models, architecture change, or inference overhead. This full joint distribution is used to compute a sampling utility, which is the linear combination of the expected maximum reward, total compute, and latency of a set of samples if generated to completion. During inference, we maximize this utility with meta-actions that determine which prefix of tokens to continue or initiate sampling from. On mixed-difficulty mathematical benchmarks, ZIP-RC improves accuracy by up to 12% over majority voting at equal or lower average cost, and traces smooth Pareto frontiers between quality, compute, and latency. By providing real-time reward-cost introspection, ZIP-RC allows models to reason adaptively and more efficiently.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work is in the context of autoregressive Language Models with a finite vocabulary (i.e. popular transformer-based LLMs).
The paper aims to improve the model's introspective ability (e.g., predicting at each token, prediction final
reward and remaining length) without increasing the test time inference cost.
To enable this, the paper proposes to use a reserved or unused set of logits and thus claims to require no additional compute at inference time.
The paper examines this within the framework of test-time scaling, i.e., using test-time compute for improved performance.

### Strengths
1. The proposed approach of using unused logits in combination with the proposed optimization of a joint distribution different from the next token prediction loss (but over an estimated value of the correctedness and remaining tokens) seems very interesting and innovative to me.
2. The results of the proposed approach in comparison to the presented baselines show clear benefits on the Pareto curve.

### Weaknesses
## Major weaknesses
1. The paper's clarity in the main approach section is limited. Sections 4.1 and 4.2 are difficult to parse and would benefit from more careful dissemination, including model/approach figures.
2. The paper claims **zero** inference time overhead. However, computations described in Section 4.2, paragraph "Sampling utility." suggest that additional computations and parameters (alpha, beta, ...) are required (even though they may be small compared to the overall computation and parameter requirements of the LLM). It would be important to clarify this.
4. The paper's approach heavily relies on a reserved set of logits but does not describe where they come from. Would we save compute if we are not using these "reserved" logits? This also relates to the previous point: Do these reserved logits basically represent additional computation needed for this approach
3. I think the paper is missing an important baseline, which does not change the underlying LLM (i.e. one does not need the KL term), but just optimizes the same reward and cost (ZIP-RC) using a small MLP (one? or two layers), which has access to the same logits used for the proposed approach.


## Minor weaknesses
1. all formulas should be numbered so one can refer to them

### Questions
Please comment on / address the weaknesses above.

Additionally:
1. line 267: "ZIP-RC modifies only the sampling policy at inference-time." This seems misleading as the model does train and thus changes the parameters of the original models.
2. How many reserved logits are used in practice?
3. What is the initial number of samples? And how does this compare to baselines?
3. What other use case do the authors see for this framework (reserved logits for a different reward) besides the presented application of test-time scaling?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles an important limitation of current test-time scaling methods for large language models: their inability to adaptively allocate compute based on problem difficulty. The authors propose ZIP-RC, a method that equips models with real-time predictions of both final reward and remaining generation cost without any inference overhead. The key innovation is repurposing reserved vocabulary logits to output a joint distribution over reward and cost at every decoding step. This enables optimizing a sampling utility that explicitly trades off accuracy against compute and latency through meta-actions like dynamic pruning and adaptive sample allocation. On mathematical reasoning benchmarks, ZIP-RC achieves up to 12% accuracy improvements over majority voting while using equal or lower cost, and traces smooth Pareto frontiers across quality, compute, and latency dimensions.

### Strengths
1. The idea of using reserved vocabulary positions to produce auxiliary predictions with truly zero overhead is novel and elegant. Rather than requiring separate forward passes or additional models like most verifier approaches, ZIP-RC extracts rich signals from logits that would otherwise go unused. The joint modeling of reward and cost (rather than just scalar confidence) is a key insight that enables principled decision-making about the reward-cost tradeoff.

2. The authors clearly articulate why existing Best-of-N methods are inefficient—they generate all N samples to completion regardless of promise. The mathematical formulation of the sampling utility in Equation 1 provides an interpretable framework for balancing competing objectives through the α and β hyperparameters.

3. The paper is generally well-written with clear motivation and good use of figures to illustrate the method's behavior. The formalization as a token-level MDP and the discrete order statistics for computing expectations are technically sound.

### Weaknesses
1. The evaluation focuses exclusively on mathematical reasoning tasks. It's unclear whether ZIP-RC's benefits extend to other domains like creative writing, coding, or open-ended question answering where the reward structure and token length distributions may be very different. Mathematical problems have clear correctness labels and relatively predictable structure, which may make reward/cost prediction easier than in other domains.

2. The authors acknowledge that their method relies on having sufficiently diverse samples from the initial pool, but they don't actually address this limitation. If doubling the number of samples doesn't produce qualitatively different reasoning paths, ZIP-RC cannot improve performance. This is a fundamental constraint that limits when the method can be applied effectively. Some analysis of sample diversity in their experiments would strengthen the work.

3. The paper doesn't clearly specify how many vocabulary positions are reserved, how this impacts the model's generative capabilities, or what the training cost is. The KL penalty term αKL is mentioned but its practical effect isn't analyzed. More importantly, ZIP-RC requires generating ~100k labeled rollouts and training the predictor—this upfront cost should be quantified and compared to the inference savings.

### Questions
1. How many positions are reserved for the reward-cost predictor? How does this choice affect both the predictor's expressiveness and the base model's generation quality? Have you experimented with different numbers?

2. Can you provide any evidence that ZIP-RC works on non-mathematical tasks? Even preliminary results on a single coding or open-ended QA benchmark would significantly strengthen the generalization claims.

3. What is the wall-clock time and compute cost to train ZIP-RC compared to training the baseline models? How much data is needed for the predictor to become reliable?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes ZIP-RC, a zero-overhead inference-time mechanism that equips large language models with real-time predictions of task reward and remaining generation length. The key idea is to utilize a set of reserved vocabulary tokens as auxiliary prediction channels, allowing the model to output both next-token probabilities and reward/length estimates within the same forward pass. To prevent distribution shift, the method introduces a KL-regularization term that constrains the model’s behavior relative to a frozen reference policy.
On top of these predictions, the paper introduces an adaptive sampling and pruning strategy that dynamically reallocates compute across candidate trajectories, enabling the model to prune low-quality or unnecessarily long samples mid-generation. This yields controllable trade-offs between output quality, compute cost, and latency, improving inference-time efficiency.
Experiments on mathematical reasoning benchmarks (GSM8K, AIME24, AMC23, MATH-500) and small-to-medium LLMs (≤1.7B) show that ZIP-RC consistently outperforms majority voting and best-of-N baselines under equal or lower generation cost, demonstrating more favorable quality-compute Pareto frontiers.

### Strengths
1.The paper proposes the ZIP-RC mechanism, which leverages reserved tokens in the vocabulary to enable real-time prediction of reward and remaining tokens without adding extra forward passes. This idea is highly innovative. Combined with adaptive pruning, the approach significantly improves inference efficiency and is of substantial practical importance.

2.Experimental results show that ZIP-RC clearly outperforms the baseline methods without such modification, demonstrating the strong potential of the proposed approach.

### Weaknesses
1.Although reserved tokens are used to avoid additional computation overhead, this strategy may still introduce distribution shift to some extent. The paper would benefit from additional comparison or analysis on the extent of distribution shift before and after modifying the loss function.

2.The experiments are primarily conducted on relatively small models (mostly under 2B). It would strengthen the work to extend evaluation to larger-scale models to verify scalability. Additionally, the experiments currently focus mainly on mathematical reasoning tasks; further results on other domains (e.g., instruction following, coding, and open-domain dialogue) are needed to validate generalization ability.

### Questions
Address the shortcomings identified in the Weaknesses section, and include the required supplementary experiments to adequately resolve these concerns and strengthen the study.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes ZIP-RC, an adaptive inference method that equips model with zero-overhead inference-time predictions of reward and cost. ZIP-RC makes use of *reserved logits* in the forward pass to model a joint distribution over the reward as well as the remaining sequence. The authors show that ZIP-RC improves accuracy by up to 12% over majority voting at comparable computational cost.

### Strengths
- The promise of the paper, "Zero-overhead inference-time" control is a very promising and timely research direction

- The authors show that their approach is decently accurate at predicting the rewards during generation.

- The proposed approach achieves improvements, often significant, on a suite of reasoning benchmarks at the same cost as the baselines.

### Weaknesses
- I found the paper extremely difficult to read, and beyond the promise of a "zero-overhead inference-time prediction of reward" found it very hard to glean much if any insight on the core contributions of the paper beyond the fact it makes use of extra logits at every step.

- The related works section is really lacking giving how active an area of research inference-time control of LMs is.

- line 85, broken figure reference. 

- Paragraph 074-085 of the introduction misses the mark when it comes to introducing the approach, in my opinion. Specifically, starting line 080, the paragraph makes mention of sampling utility and coefficients $\alpha$ and $\beta$ that supposedly balance reward and cost. This level of detail at this point of the paper left me more confused than informed, with no big picture idea of what the core contribution is.

- I am unsure if the subsection on "Generation as a token-level MDP" offers much value. In fact, it seems that variables defined in that section were used differently in later section e.g. $P$.

- Lines 170-171: "ZIP interprets the slice on $\mathcal{R}$". I believe the previous sentence should be "slice on $\mathcal{V}" instead?

- It took me quite some time to realize that the approach is *not* inference time, but instead relies on some form of training. In my opinion, the authors should make sure this is clear very early on in the paper as well as potentially clarify the title which to me is currently a bit misleading.

- The presented approach fails to compare against many of the latest inference-time approaches in their experiments

### Questions
- Could you please give me a breakdown of your technical approach? What exactly does your approach do at training time? How do you modify autoregressive decoding to model the joint distribution of sequence continuations and their corresponding rewards? A diagram would've been extremely helpful.

### Soundness
2

### Presentation
1

### Contribution
2
