# Efficient Prediction of pass@$k$ Scaling\\in Large Language Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Assessing the capabilities and risks of frontier AI systems is a critical area of research, and recent work has shown that repeated sampling from models can dramatically increase both.
For instance, repeated sampling has been shown to increase their capabilities, such as solving difficult math and coding problems, but it has also been shown to increase their potential for harm, such as being jailbroken.
Such results raise a crucial question for both capability and safety forecasting: 
how can one accurately predict a model's behavior when scaled to a massive number of attempts, given a vastly smaller sampling budget?
This question is directly relevant to model providers, who serve hundreds of millions of users daily, and to governmental regulators, who seek to prevent harms.
To answer this questions, we make three contributions.
First, we find that standard methods for fitting these laws suffer from statistical shortcomings that hinder predictive accuracy, especially in data-limited scenarios.
Second, we remedy these shortcomings by introducing a robust estimation framework, which uses a beta-binomial distribution to generate more accurate predictions from limited data.
Third, we propose a dynamic sampling strategy that allocates a greater budget to harder problems.  Combined, these innovations enable more reliable prediction of rare risks and capabilities at a fraction of the computational cost.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper aims to efficiently and accurately predict model performance across massive repeated queries using only a limited number of trials. The authors 1) identify the limitations of two existing approaches for pass@$k$ estimation, i.e., linear regression and distributional fitting; 2) propose an alternative fitting method and a dynamic sampling strategy; and 3) demonstrate that the proposal brings significant improvements across various hyperparameters and datasets.

### Strengths
1. The motivation and the research problem are clearly introduced at the beginning of the paper. The problem is also significant for the practical deployment of LLMs at scale.
2. The authors contribute to the problem in multiple ways, including a formal theoretical review and the proposal of targeted solutions.
3. The experimental results are well presented and demonstrate that the proposed method outperforms the baselines by a clear margin.

### Weaknesses
1. Given that $k<B$, i.e., dozens to thousands of attempts, I don't think the setup corresponds to what is mentioned in L40, i.e., billions of daily interactions. 
2. The authors list the $k$-dependency as a drawback of linear regression for pass@$k$ estimation. However, as pass@$k$ inherently depends on $k$, there should be a more detailed justification for this claim.
3. In this paper, both linear regression and distributional fitting are criticized for being inapplicable to smaller $k$. While I agree that a robust scaling law should generalize across different values of $k$, the authors’ claim that these pass@$k$ predictions are most relevant to large-scale deployment may suggest a mismatch between the critique and their stated motivation.
4. The discussion section feels somewhat brief. Since you have pointed out the limitations of existing estimation approaches and designed your algorithm accordingly, I recommend you to provide a more detailed analysis clarifying how your proposals contribute to the efficiency and accuracy, and whether your method avoids the aforementioned limitations.

### Questions
1. For the mention of success rate in L36, I think the typical interpretation of this metric is that it's averaged across all attempts. You may refer to the definitions of empirical probability and expected maximum toxicity in the RealToxicityPrompts paper (Gehman et al., 2020) for a more rigorous statement.
2. Could you provide examples of the larger $k$ values mentioned in L194?
3. The proposed method seems to be a more flexible and efficient version of the distributional fitting approach. If my understanding is correct, I would like to see the individual contributions of these two improvements.
4. Could you elaborate on why you chose these ranges for $k$ and $B$ and how the conclusions drawn from this setting would inform the model providers and regulators?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tackle how to accurately predict pass@k for large k when you only have a small sampling budget. They show why popular approaches (e.g., log–log power-law regression and a discretized beta fit) give biased, high-variance extrapolations, and then propose to replace them with a beta–binomial fit to per-problem success rates and a dynamic policy that concentrates trials on the hardest problems, then shows this combo matches ground truth far better across AdvBench, MATH, and Code Contests benchmarks.

### Strengths
The paper: 
- Investigates a valuable, high-impact problem: how to predict pass@k under tight sampling budgets, which matters for safety (scaling of rare failures) and capabilities (planning compute for methods like RLVR). 

- Provides a clear critique of existing approaches: shows why log–log regression and discretized-beta fits are statistically flawed and yield poor high-k extrapolations. 

- Introduces a novel solution with strong empirical evidence: a beta–binomial estimator plus dynamic sampling that focuses trials on harder items; across AdvBench, MATH, and Code Contests, it tracks ground truth much better than baselines. 

- Stress tests & ablations clarify when it helps most: allocate more samples to hard problems, and in heavy-tailed cases, dynamic sampling usually beats (or at least matches) uniform across distributions.

### Weaknesses
- Certain assumptions in the proposed solution, that is i.i.d. attempts with a fixed per-problem success rate, a single Beta prior over difficulty, and assuming task stationarity (no changes in guardrails, prompts, or caching) over time, may reduce robustness of the proposed method and introduce bias.
- Evaluations cover a few benchmarks and ~hundreds of items. Although useful but limited for broad generalization.
- The paper primarily validates the methods using MSE to ground-truth pass@k curves, which is limited. Incorporating calibration measures and decision-centric metrics (e.g., risk at a target pass@k) would give a better picture of the proposed method’s advantages.
- Limited number of baselines considered for comparison. Adding richer-prior baselines, e.g., Beta mixtures or Dirichlet-process priors that better handle tail misspecification, would more clearly demonstrate the proposed method’s advantages.

### Questions
- Could you clarify why the error bars differ so much across LLMs, especially for the Gemini models compared with the others?
- What happens under non-stationarity (model updates, prompt/guardrail changes)? Do you have a rolling or re-weighting variant that stays calibrated?
- How does your method handle non-i.i.d. retries on the same problem. For example, when later attempts reuse earlier chain-of-thought or tool outputs (adaptive changes in success probability) or when samples are correlated (e.g., beam/diverse decoding sharing prefixes)?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
Current LLMs have a large number of users and their behavior in such conditions is not well studied. In particular, trying to understand performance for “pass @ k” where k is a large number is important. As such, this paper tries to predict the performance of models for “pass @ k” where k is large and experiments cannot be run to provide exact measurements. The authors model task success probabilities with a beta-binomial distribution, which allows accurate extrapolation from limited data. They also propose a dynamic sampling strategy that focuses more on hard tasks, improving prediction accuracy and efficiency. Overall, their method provides a practical, data-efficient way to estimate LLM performance or failure rates at scale.

### Strengths
A simple method for estimating the performance of models for pass @ k

### Weaknesses
Not clear how the method performs for tasks that are subjective.

### Questions
Do you have an intuition how well the method performs for tasks that involve safety, which tend to be subjective in nature (as opposed to the math/reasoning tasks you’re covering in the paper)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors tackle the problem of predicting LLMs' pass@k for verifiable problems, for a specific regime where the budget of LLM calls (for gathering data to predict pass@k) is fixed and k exceeds the average LLM budget per problem in the data set, thereby necessitating extrapolation. The authors propose making variable numbers of attempts per query by spending more of the LLM budget on problems with the low success rates. They present a theorem stating that if the probability of success is known exactly for each problem, difficult problems should receive a greater share of the budget for LLM calls, validating their approach.

### Strengths
- The paper is well written and interesting.
- The dynamic sampling strategy is sound and makes perfect sense for the problem: this is a valuable contribution to the literature.
- The theorem justifying over-sampling of difficult problems is mathematically clean.

### Weaknesses
- The framing of the significance of the problem seems misleading. For example, the cited research on brute-force jailbreaking actually varies the prompt on each attempt, so it's not directly comparable to repeated sampling. Similarly, it is not clear that "The relevance of [predicting pass@k] is only underscored by the massive scale at which these frontier AI systems are deployed, with some experiencing billions of daily
interactions." - does massive scale imply that customers are running the same query a million times? How does the rise of reasoning LLMs affect the significance of pass@k: given the increased latency of producing a thinking trace, does it still make sense to sample repeatedly from these models?
- Pass@1 is a fraught quantity: either an LLM is capable of answering a question, in which case it will get the correct answer reasonably quickly after some repeated trials, or the problem is beyond the LLM's capability. In the latter case, the probability of answering correctly is near zero. However, correctly estimating pass@1 for difficult problems with very low but nonzero success rates is important for accurately predicting pass@k as k -> inf. It appears that estimating pass@1 values with greater precision is the main benefit of your dynamic sampling approach, NOT the reduced variance of the pass@k estimator as described in your theorem (although that is an added benefit). The theorem assumes that pass@1 is known precisely, whereas in reality, estimating pass@1 seems to be the core problem, diminishing the significance of your theorem.
- Your evaluation estimates each problem's ground truth pass@1 with only 10,000 samples, effectively capping the resolution of detecting nonzero pass@1 at somewhere near 1e-4. Is this justified? I could easily see true pass@1 value to be smaller than 1e-4. This resolution limit affects all your posted results, so a discussion would be valuable. Do you expect many pass@1 values to be smaller than 1e-4 in practice?
- Your beta model for estimating the distribution of pass@1 may not adequately capture the heavy concentration of probability mass near zero. Perhaps a mixed continuous-discrete model with a discrete lump of probability at pass@1 = 0 would be a sensible approach. For example, see the paper https://openreview.net/forum?id=YCBVcGSZeR.
- This paper is not "scaling law research," which seems to imply a connection to the famous scaling laws from the pre-training literature. This paper is about pass@k and repeated sampling from LLMs. Thus, the related work section should not cover "scaling laws" in the abstract but stick to the paper's core question of repeated sampling from LLMs. Specifically, it would be relevant to discuss the other side of the coin of repeated-sampling: non-verifiable problems, where majority voting etc. must be applied. For example, see the paper https://openreview.net/forum?id=m5106RRLgx. Similarly, statistics papers on estimating very small probabilities could be relevant. As it stands, the positioning of the paper within the literature appears misleading and the related work section should stick closer to the knitting of the paper: repeated LLM calls.

### Questions
- Please see "Weaknesses". Overall, I respect your paper a great deal and consider your dynamic sampling methodology to be an important contribution to the literature. That said, I would appreciate more discussion or clarification on the issue of very small pass@1 values, and how to interpret your evaluation in light of such issues. I'm still unsure if the paper meets ICLR's bar in terms of substantial novelty and significance, and welcome your comments.
- Upon introducing Algorithm 1, I'd recommend clarifying that the arrays of "attempts" and "successes" are mutable arrays that start out empty and will be gradually extended (during Algorithm 2). As written, Algorithm 1 seems to require the input arrays to be fully formed. It also seems that you could describe Algorithm 1 more simply: essentially, the idea is to sample among all problems that have not yet observed a success and within all such "difficult" problems, the least-attempted ones are prioritized. Correct? I have this impression because in practice, the pool of problems with zero successes will likely never shrink to zero, since there will be SOME pass@1 values low enough never to yield a success.
- Some empirical illustration of the scale and magnitude of typically observed pass@1 values (specifically the ones near zero) would be appreciated.

### Soundness
3

### Presentation
3

### Contribution
3
