# Multiple-Prediction-Powered Inference

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 2, 6, 4

## Abstract
Statistical estimation often involves tradeoffs between expensive, high-quality measurements and a variety of lower-quality proxies. We introduce Multiple-Prediction-Powered Inference (MultiPPI): a general framework for constructing statistically efficient estimates by optimally allocating resources across these diverse data sources. This work provides theoretical guarantees about the minimax optimality, finite-sample performance, and asymptotic normality of the MultiPPI estimator, and through experiments across three diverse large language model (LLM) evaluation scenarios, we show that MultiPPI consistently achieves lower estimation error than existing baselines. This advantage stems from its budget-adaptive allocation strategy, which strategically combines subsets of models by learning their complex cost and correlation structures.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper tackles cost-efficient estimation when a high-quality but expensive signal coexists with cheaper proxies. The method formulates variance-minimizing estimators with constraints. With known covariance \(\Sigma\) this becomes a tractable SOCP/SDP and is minimax-optimal. With \(\hat{\Sigma}\) from a small burn-in, it retains finite-sample bounds and asymptotic normality. Then, the authors show in experiments that, on LLM evaluation (arena wins, reasoning budgets, factuality), their proposed method achieves lower MSE and tighter CIs, adapting from cheap proxies at low budgets to accurate raters as budgets grow, in multiple settings.

### Strengths
1.  A major strength is that the paper frames the complex allocation problem in a manner that makes it solvable using standard optimization tools.
2.  The paper provides finite-sample bounds on the estimator's performance for the practical scenario where the covariance $\Sigma$ must be estimated from data.
3.  The effectiveness of MultiPPI is demonstrated across diverse and relevant large LLM evaluation scenarios.

### Weaknesses
1.  The practical algorithm relies heavily on an a priori estimate of the covariance matrix $\Sigma$, which is derived from an initial "burn-in" set of $N$ fully-labeled samples. The performance may degrade if the size $N$ is improperly chosen or if $\Sigma$ is ill-conditioned.
2.  The MultiPPI framework allows sampling from any subset $I$ of $k$ variables. However, enumerating or considering many such subsets and solving the corresponding SOCP/SDP problems can become computationally expensive as $k$ increases. The authors should include a discussion on the scalability of the MultiPPI estimator.
3.  The finite-sample bounds in Theorem 4.3 assume that $|X_i| \leq 1$, which is a restrictive assumption. Some evaluation signals (e.g., scores/logits) may violate this bound in other application scenarios.
4.  MultiPPI is designed to estimate a linear function of the mean. This may limit its applicability for estimating a broader scope of non-linear functions of the mean. (This is noted as a potential limitation, perhaps minor given the paper's focus.)

### Questions
Please see the Weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This work proposes a method to estimate the resource allocations for evaluation, e.g., assessing the qualities of model predictions in a task using large language models, under the budget constraint. Basic idea is to formulate the task setting as a gold human label with "auorators" by language models, and to trade the number of queries to the models and the expected qualities with the maximum budget constraint. It is now treated as an optimization problem assuming the accurate covariance of labeled samples are obtained.

### Strengths
This work introduce a task setting to maximize the expected evaluation qualities under the resource budget so that the number of queries to language models could be minimized. It is an interesting yet practical setting especially when language models are employed for evaluation, i.e., LLM-as-a-judge.

### Weaknesses
- Clarity is an issue. The task setting assumes $n$ samples with gold labels by human and $n$ samples without gold and thus estimated by autorators only, i.e., language models, in Equation 3. Further discussion introduces the cascade modeling in Equation 4. However, their relation to the proposed approach is not discussed in section 4. As a result, the contribution of this work is not clear.
- Given the task setting assumes incomplete labels in term of the lack of human ratings, it is not clear how that is reflected in the experimental settings. It is also not clear what optimizer was used in the experiment.

### Questions
See the comments regarding weaknesses.

### Soundness
2

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
3

### Summary
This paper addresses a critical challenge in modern AI model development: balancing the cost and quality of evaluation metrics. It proposes MultiPPI, a statistical framework that optimally allocates resources across diverse data sources (high-cost, high-quality "gold" metrics and low-cost, low-quality proxies) to estimate linear functions of population means (e.g., LLM win-rates, factuality accuracy) under a fixed budget. The framework provides theoretical guarantees (minimax optimality, finite-sample bounds, asymptotic normality) and validates performance across three LLM evaluation tasks, consistently outperforming baselines like classical sampling, PPI++, and vector PPI++.

### Strengths
The authors provide a complete and well-structured theoretical framework, which is rare in applied AI evaluation work:  
- Minimax Optimality: When the covariance matrix of data sources is known, MultiPPI achieves the minimax risk lower bound (Theorem 4.2), ensuring it is statistically optimal among all budget-feasible estimators. This anchors the method in fundamental statistical principles, rather than heuristic design.  
- Practical Guarantees: For real-world scenarios where covariance matrices are unknown, the paper derives finite-sample bounds (Theorem 4.3) that quantify performance degradation when using empirical covariance matrices. It also proves asymptotic normality (Theorem 4.4), enabling valid confidence interval construction—critical for practical decision-making (e.g., comparing model versions).  
- Optimization Tractability: The authors skillfully transform the resource allocation problem into solvable programs: second-order cone programming (SOCP) for single budgets and semi-definite programming (SDP) for multiple budgets. This ensures MultiPPI can be implemented with off-the-shelf tools (e.g., cvxpy), bridging theory and practice.

### Weaknesses
The Method Rely on Accurate Covariance Estimation

In practice, MultiPPI requires a set of "fully labeled samples" (containing both gold metric$X_1$ and all proxies $\(X_2,...,X_k\)$ to estimate the empirical covariance matrix $\(\hat{\Sigma}\)$. The paper uses N=250 or 1000 such samples, but :  
- Fully labeled samples may be scarce: Gold metrics like human annotations are often extremely costly—collecting 250 fully labeled samples could be prohibitive for small teams or niche tasks (e.g., low-resource language LLM evaluation).  
- Cost-effectiveness of estimation: The paper does not sufficiently explore whether an accurate covariance matrix can be robustly estimated with a smaller, more cost-effective sample.

### Questions
1. Are the golden samples, (i.e., the fully-labeled samples that contain both the high-quality metric $\(X_1\)$ and all low-cost proxy metrics $\(X_2, ..., X_k\)$), prepared to obtain an accurate covariance matrix sufficient for well estimating the mean of the high-quality score, $\(E[X_1]\)$,  thereby avoiding the need to compute the low-cost proxy metrics??  
2. Evaluation metrics are often used in the model training phase. If the model is modified, can the empirical covariance matrix $\(\hat{\Sigma}\)$ obtained under the old model still be applied to the new model?

### Soundness
3

### Presentation
2

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
This paper proposes Multiple-Prediction-Powered Inference (MultiPPI), an extension of Prediction-Powered Inference (PPI++) to settings with multiple predictive models of varying cost and accuracy. The authors frame the problem of estimating a population mean under budget constraints as an optimization over which predictors to sample and how to weight them. They derive a minimax-optimal solution assuming known covariance, propose a practical version using an estimated covariance matrix, and provide asymptotic and finite-sample guarantees. Experiments on LLM-evaluation tasks (Chatbot Arena, ProcessBench, factuality benchmarks) suggest improved efficiency over baseline PPI methods.

### Strengths
1. The formulation is clean and the mathematics appear correct.
2. Good Application. The use of realistic LLM cost scenarios is interesting for the LLM evaluation tasks.
3. The optimization framing (SOCP/SDP) is good and potentially generalizable.

### Weaknesses
I don't see much weakness of the paper. The main contribution is clear: MultiPPI is a straightforward generalization of PPI++: it replaces a single predictor with a vector and adds a *cost-weighted* sampling constraint. The estimator remains linear, the theory is a direct extension of standard control variates, and the minimax-optimality proof follows textbook arguments once the covariance structure is fixed. I am not sure if this contributions reaches the bar of ICLR though. I would also think that Cost-aware PPI would be a better name of the method, since the main contributions is not from aggregating multiple ML predictions, but  aggregate them in a cost-aware way.

### Questions
1. How sensitive are the theoretical results to covariance misspecification?
2. How sensitive is MultiPPI’s performance to errors in the covariance estimate? Could shrinkage or robust covariance estimation improve stability?

### Soundness
2

### Presentation
3

### Contribution
2
