# Universal Model Routing for Efficient LLM Inference

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 8

## Abstract
Model routing is a simple technique for reducing the inference cost of large language models (LLMs), wherein one maintains a pool of candidate LLMs, and learns to route each prompt to the smallest feasible LLM. Existing works focus on learning a router for a fixed pool of LLMs. In this paper, we consider the problem of dynamic routing, where new, previously unobserved LLMs are available at test time. We propose UniRoute, a new approach to this problem that relies on representing each LLM as afeature vector, derived based on predictions on a set of representative prompts. Based on this, we detail two effective instantiations of UniRoute, relying on cluster-based routing and a learned cluster map respectively. We show that these are estimates of a theoretically optimal routing rule, and quantify their errors via an excess risk bound. Experiments on a range of public benchmarks show the effectiveness of UniRoute in routing amongst more than 30 unseen LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Model routing is an important problem in LLM serving. The focus has been on static model routing, but there is a need for dynamic model routing. The paper formulates the dynamic model routing problem and characterizes the optimal solution. From this, the authors derive a practical algorithms using prompt and model embeddings.

### Strengths
- Good formulation of the dynamic routing problem.
- Good results on standard benchmarks.

### Weaknesses
- Lack of rigor in Sec 5 (See my questions).
- Unclear motivation for introducing clustering from first principles.

### Questions
- Prop 1: If I understand correctly, the Lagrangian multiplier $\lambda_{\mathcal{h}}$ is specific to an instance of $x$ and $\mathcal{H}$, while $\mathcal{h}$ is the distribution of $\mathcal{H}$. Therefore, denoting this multiplier as $\lambda_{\mathcal{h}}$ is confusing. Please correct me if I am wrong.
- Implication of Prop 1: the optimal $r^\star$ for Eq 5 is the minimizer to Eq 6 for some $\lambda_{\mathcal{h}}$. However, it does not mean that optimizing over $\lambda_{\mathcal{h}}$ and Eq 6 is going to give us the optimal $r^\star$ for Eq 5. Therefore, the suggestion of tuning $\lambda_{\mathcal{h}}$, while being something sensible to do, is simply ignoring Prop 1. I want to hear the reasoning on why this is still a good idea. Maybe we are implicitly optimizing another optimization problem when doing so. The main problem is that it is not clear whether the values of $\lambda_{\mathcal{h}}$ satisfying Prop 1 can be found constructively.
- Unclear formulation in Sec 4: for simplicity, let $\Phi \in R^{|X| \times K}$, $\Psi \in R^{|H| \times K}$. Hence, $\Phi \Psi \in R^{|X| \times |H|}$ (I'm assuming the transpose in (9) is a typo). Suppose that $|X|=1$, then $\gamma$ is a vector function? From (8), I got the impression that the domain of $\gamma$ is $R_+$. Therefore, something about the current formulation does not add up.
- Line 271: underperform vs what?
- Unclear motivation for clustering in Sec 5: What is the reason to clustering here? Is it only for computational efficiency?

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies LLM routing under dynamic model pools, where test-time available LLMs may differ from training-time ones. It proposes UniRoute, which learns a bilinear scoring function γ(x, h) = Φ(x)ᵀΨ(h). Prompts are embedded (and clustered) once; LLMs are represented by their per-cluster error rates on a small validation set. Routing is then simple cost-adjusted argmin over any test-time pool. The paper gives a Bayes-optimal rule for the dynamic setting and an excess-risk bound for the cluster-based instantiation. Experiments on four public routing-style benchmarks with 30+ unseen LLMs show UniRoute outperforms strong dynamic baselines (ZeroRouter, K-NN) and avoids retraining costs.

### Strengths
- Relevant problem: tackles a practical, current task where LLM/agent methods are still brittle, the problem choice is well motivated.
- The components fit together: the modeling choices are consistent with the objective, and the pipeline is implementable without exotic assumptions.
- Across multiple datasets/settings, the method shows consistent improvements over the stated baselines, not just one cherry-picked case.
- There is at least some attempt at digging into why it works (error breakdown / qualitative examples), which is better than many incremental papers.
- Reproducible direction: The paper uses mostly standard toolchains; another group should be able to re-implement this without reverse-engineering half the setup.

### Weaknesses
- Ablations are thin: key modules are turned on/off only in one setting; we don’t see if the effect is stable across datasets/scales. A 2–3 row ablation table per main component would already help.
- Limited robustness reporting: no real stress test (distribution shift, noisier inputs, or lower-resource regime). Right now, the method looks tuned to a friendly setup.
- Clarity on compute/cost: method adds some overhead but the paper doesn’t quantify it clearly; for adoption, people will want to know the inference/training cost relative to the simplest baseline.

### Questions
1. Can you add results or at least a justified comparison against the most recent approach(es) that use a similar intermediate representation? Right now the reader has to guess how your method would fare against them.
2. Ablation depth: If you ablate the proposed module but keep the training data / hyperparams the same, how much of the gain remains? This would confirm it’s not just better tuning.
3. Compute: What is the actual training cost (GPU-hours, model size) and inference latency delta vs. the strongest baseline?

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
This paper studies the problem of dynamic model routing for LLM inference: given a prompt and a pool of candidate LLMS of different cost, select the cheapest model that will still answer well, even when the candidate pool dynamically changes. Existing work mostly assumes a static pool, and retrains the router whenever a new model appears, which is impractical and expensive. The authors formalize routing with a dynamic pool and propose UniRoute, which represents each LLM as a feature vector using its prediction error on the validation dataset, and represents each input prompt via a learned embedding. Then, they estimate for each model and prompt pair the probability of error plus a cost term and pick the minimum. Two examples of UniRoute are presented; they are clustering-based routers where prompts are grouped into clusters learned using the training dataset. The authors argue that these are approximations to an optimal routing rule and provide an excess risk bound. Experimentally, the authors evaluate UniRoute against multiple baselines that cover static and dynamic routing methods on several public routing benchmarks. Empirically, UniRoute attains better cost accuracy tradeoff than baseline methods.

### Strengths
- The paper addresses a realistic and under-explored setting. Most prior work focuses on a fixed pool of LLMs and require retraining when a new model is added, UniRoute explicitly designed to handle a dynamic pool of LLMs.
- To my knowledge, the solution proposed by the authors is novel and quite simplistic. I think that the cluster-based example setups are practical and easy to implement. It is also cost-effective to adapt to new models.
- The paper gives an explicit optimal routing rule that shows the tradeoff between the predicted error and the model cost via a Lagrangian multiplier. It also shows how UniRoute approximates that.

### Weaknesses
- The given clustering-based examples rely on the representativeness of the validation dataset. While this works, the paper gives limited insight into sensitivity. What happens if clusters are not well aligned with the task structure? 
- The proposed method assumes that you have access to a validation dataset with labels. How could this be generalized to noisy labels for a validation dataset or a validation dataset without labels? (I think that an intuitive answer would be enough instead of running a whole set of experiments.)

### Questions
- UniRoute represents the performance of the new LLM via the bilinear form $\Phi(x)^|\top\Psi(x)$. Can you explain why the bilinear form is expressive enough to model the interaction between prompts and models? Can there be more complex models that uses $\Phi(x)$ and $\Psi(x)$ as input features and estimates the performance?

Typos
- Line 330, "... incorrect predic let..."
- Line 386, missing "." at the end of the sentence.

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
The paper studies routing among multiple LLMs when the model pool can change after training. The idea is to fingerprint each model by taking its error to a validation set, giving us a way to then route promots to a cluster and let us estimate which model will do best. They cover two approaches, one which is K-means clusters and one which is a learned assignment. This produces budget-and-quality frontier improvements over baselines on a range of benchmarks and models.

### Strengths
1. The set of evaluations is very broad and shows consistent gains against reasonable baselines
2. The test-time approach is amenable to many real use cases
3. The method is cognisant of cost, as opposed to naively just optimising for performance

### Weaknesses
1. The set of metrics is relatively narrow; it would be good to cover generation metrics as some other methods do
2. It is unclear how robust this is to shifts in distribution compared to the validation set

### Questions
How stable is the cost-tuning knob across datasets and model pools?
How does this look on non-binary metrics?
Are there subgroups or adversarial examples which can reduce performance?
Is it possible to combine this with a learned router for seen models to get improvements when comparing to a static pool?

### Soundness
3

### Presentation
3

### Contribution
3
