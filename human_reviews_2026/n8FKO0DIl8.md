# Statistical Guarantees in the Search for Less Discriminatory Algorithms

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
U.S. discrimination law can impose liability on firms that fail to adopt a less discriminatory alternative (LDA), defined as a decision policy that achieves the same business objectives while reducing disparate impact on legally protected groups. Recent scholarship argues that this doctrine has direct implications for algorithmic decision-making in high-stakes domains such as employment, lending, and housing, potentially obligating firms to search for “less discriminatory algorithms” (Black et al., 2024). Regulators have
at times encouraged proactive LDA searches, reinforcing the expectation of a good-faith effort to identify equally performant models with lower disparate impact. Model multiplicity makes such searches plausible: retraining with different random seeds can yield models with comparable predictive performance but materially different disparate impacts. Yet firms cannot retrain indefinitely, raising a central question: when is the search sufficient to demonstrate good faith? We formalize LDA search under multiplicity
as an optimal stopping problem in which a developer seeks to produce evidence that further search is unlikely to yield meaningful improvements. Our main contribution is an adaptive stopping algorithm that provides a high-probability upper bound on the best disparate-impact gains attainable through continued retraining, enabling developers to certify (e.g., to a court) that additional search is unlikely to help. We also show how stronger distributional assumptions over the model space can yield tighter bounds,
and we validate the approach on real-world credit and housing datasets.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper seeks to give anytime-valid bounds on the expected marginal utility of randomly training an additional model in search of a Least Discriminatory Algorithm. The paper is organized in three levels of generality -- first, the authors tackle the full-information regime where the distribution and the population utilities are known. Then, the authors generalize their results, with high probability and under various assumptions, to unknown model sampling distributions. Finally, the authors generalize the results to finite data with an additional assumption that the 'selection effect' (the measurement error conditioned on the current empirical disparate impact) does not decrease with additional training runs.

### Strengths
1. The paper has a clear and precise motivation and framing of results. 

2. The paper's organization as a gradual movement towards generality, with some additional thoroughly explained assumptions, made the paper very easy to read and understand.

3. I do not have the expertise to know whether the proposed bounds on the probability of sampling a new minimum value of a random variable are significant, but to this reviewer they are interesting and potentially useful.

4. I am convinced that the proposed bounding approach and algorithm can be useful to firms seeking to meet regulatory requirements (which may not yet exist) around finding LDAs, supposing that their training procedures are capable of finding satisfactory models.

### Weaknesses
1. Relating to strength #4, I am unconvinced that knowledge that a firm satisfied a strong bound using the provided approaches would actually certify anything of value to a regulator or a court. It is entirely plausible that a firm could have an algorithm A that either is not sufficiently random, or is biased in some way (neither necessarily intentionally), so as to yield models which are very discriminatory, even when a non-discriminatory model may exist. Using the above approach could inadvertently allow a firm to certify that they had sufficiently searched for LDAs when in fact they hadn't.

2. Although the additional assumptions in the theorems are well-described, I am not left with a sense of whether or not they are plausible in real data distributions. 

3. Relating to weakness #2, I think that additional experiments including the tightened bounds using assumptions A1-A3 would be useful to the paper. As it stands, it is difficult to determine how much those additional assumptions help beyond allowing smaller values of $\bar{\mu}$ in the algorithm.

4. I think the experiments would benefit from many more iterations than the 60 plotted. Also, the starting values of both bounds at iteration 0 seem very small -- is there really such tiny expected marginal utility in training another model, even after only training a single model?

5. This is neither a weakness nor a strength, so I am putting it last. The framing of the paper as a search for LDAs does not capture the full generality of the results. As the authors suggest, the results apply to any loss function and to any repeated (randomized) model fitting procedure. The search for LDAs is one potential application of these methods, but I feel that this could have been emphasized adjacent to other potential applications instead of as the main thrust of the paper.

### Questions
1. How would a model developer determine which of the assumptions would be suitable for their data? How could they defend that choice to a judge or regulator?

2. The bounds on the ground truth in the experiment are very wide -- is it plausible to run the same experiment on a synthetic distribution with a known ground truth?

3. At the end of section 4, you state that, "Empirically, Algorithm 1 performs well in the sense that it "overshoots"... by tens of models..." How can I see this relationship from the plots? Is this by drawing a horizontal line from each point on the brown curve to the right until it touches the pink curve? 

4. Techniques exist in the literature to find a diverse set of models of a particular model class without randomized re-training. Some examples include the Rashomon set of decision trees, a sampling of the Rashomon set of XGBoost models, and a sampling of the Rashomon set using dropout in neural networks. What makes your approach desirable over these methods, which are more likely to directly find an LDA instead of randomly training and hoping for one?

### Soundness
3

### Presentation
3

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
The paper formalizes the practice of retraining to find less discriminatory algorithms (LDAs), models with comparable utility but reduced disparate impact, by casting it as an optimal-stopping problem. Given model multiplicity, the authors define a decision rule: stop retraining once the expected marginal improvement in disparity from training one more model falls below a cost/benefit threshold. 
They provide an adaptive stopping algorithm with anytime-valid, high-probability upper bounds on the marginal improvement, so that when the algorithm halts, one can certify that continued search is unlikely to pay off. Empirical studies are conducted to validate the method.

### Strengths
Turning “good-faith LDA search” into an auditable optimal-stopping problem with an explicit threshold $\gamma$ is a nice formulation. The adaptive rule provides high-probability upper bounds on the marginal gain from one more retrain, enabling a certificate that a search was “sufficient” at the data-dependent stopping time. The paper motivates the problem clearly, is well-written, and easy to follow.

### Weaknesses
1. **Bounding $\mu$ in Section 3.2.**  
   The paper presents several upper bounds on \( \mu(u) \) under different assumptions on the underlying density. It is not immediately clear how conservative these bounds are in practice. Could the authors comment on the **tightness** of these bounds (e.g., instances where they are known to be sharp vs. loose), and perhaps provide empirical or theoretical comparisons across the proposed choices?

2. **Online learning formulation (infinite-data regime).**  
   Consider the infinite-data setting where we observe i.i.d. $Q_t$ exactly but $P$ is unknown. Can we estimate $u_p^\star$ in a data-driven way at each round? For example, at round $t$ define $$ \hat g(u) = \frac{1}{t}\sum_{j=1}^t (u - Q_j){1}\{u>Q_j\},$$
   and set $\hat u_p = \sup_{u\in[0,1]} \hat g(u) \le \gamma$. One could update $hat g$ each round (or every few rounds) and stop when $U_\tau \le \hat u_p$. Would this constitute a **valid** approach within your framework, and under what conditions (if any) would it inherit your guarantees?

3. **Assumption 3.4.**  
   The intuition for Assumption 3.4 is not fully clear, particularly why it should hold for **any** $P$ and **any** $\hat P$? What is meant precisely by “regression to the mean is at least constant”? A more natural route might be to index the assumption by **sample size**, yielding an explicit bound between $P$ and $\hat P$ that vanishes as $n \to \infty$. Could the authors either (i) give **verifiable sufficient conditions** ensuring Assumption 3.4 under common validation-reuse protocols, or (ii) reformulate it in a sample-size–dependent way that makes its asymptotic validity transparent?

### Questions
Please see the weaknesses section.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper formalizes the search for less discriminatory algorithms (LDAs) as an optimal stopping problem grounded in the context of model multiplicity and fairness-aware model selection. It proposes an adaptive stopping algorithm (Algorithm 1) that provides a high-probability upper bound on the marginal benefit of retraining models, effectively giving a statistical “certificate” for when a firm can reasonably stop searching for fairer models. The theoretical framework integrates ideas from anytime-valid inference and optimal stopping theory, establishing guarantees that the expected marginal gain from retraining is below a user-specified threshold. Empirical evaluations on three fairness-sensitive datasets (Adult, Folktables, HMDA) and multiple model classes (logistic regression, random forests, neural networks) demonstrate the method’s ability to approximate the optimal stopping point with reasonable accuracy. The discussion highlights implications for algorithmic fairness compliance, proposing the approach as a tool to certify “good-faith” searches for LDAs in high-stakes domains.

### Strengths
- Theoretical novelty: The framing of LDA search as an optimal stopping problem is original and mathematically sound. The derivation of anytime-valid upper bounds for marginal gains extends prior work in statistical inference and stopping theory.

- Practical relevance: The work connects theoretical constructs to regulatory and compliance debates in algorithmic fairness, addressing a pressing question of how firms can demonstrate sufficient fairness efforts.

- Methodological rigor: The paper clearly delineates three regimes (full-information, infinite-data, finite-data) and progressively builds the theoretical results with appropriate assumptions and proofs.

- Statistical guarantees: The introduction of an adaptive, distribution-agnostic stopping rule that provides high-probability bounds strengthens the method’s interpretability and generality.

- Empirical validation: The experimental section, while modest, is well aligned with the theoretical claims. The algorithm’s overshoot relative to the full-information optimum provides empirical evidence for its reliability.

### Weaknesses
- Limited empirical scope: The empirical evaluation, though methodologically correct, uses small-scale settings with standard datasets. There is limited evidence of robustness in larger or more complex model retraining pipelines.

- Assumption strength: Several theoretical results depend on distributional assumptions that may not hold in realistic ML training scenarios with non-iid retraining or adaptive hyperparameter tuning.

- Connection to fairness metrics: While the framework generalizes beyond disparate impact, the empirical focus remains narrow. It does not analyze whether the stopping rule’s behavior changes under alternative fairness definitions (e.g., equal opportunity, demographic parity).

- Practical deployment considerations: The paper lacks discussion of computational cost, reproducibility challenges, and real-world compliance integration.

- Clarity of exposition: The writing is dense in the theoretical sections (e.g., Section 3) and could better motivate the intuition behind the derived bounds for a mixed audience of ML and applied fairness researchers.

### Questions
1. How sensitive is the stopping time to violations of Assumption 3.4 (non-decreasing selection effect)? Would the algorithm’s guarantees degrade gracefully under mild violations?

2. Could the method be extended to handle adaptive retraining (where the next model’s training depends on previous outcomes), which is common in fairness optimization pipelines?

3. How should practitioners select or validate the threshold γ in regulatory or organizational contexts?

4. Would the method still provide valid guarantees if retraining involved data reweighting or feature modification, rather than randomness in initialization or batch ordering?

5. How does the choice of fairness metric influence the stopping behavior and the empirical coverage rates?

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
3

### Summary
The paper proposes a statistical criterion to determine when to stop training (sampling) models on a finite dataset while trying to minimize a quantity of interest. The main statistical guarantee is that the algorithm will stop at a point where training an additional model will unlikely lead to an improvement. The paper focuses mainly on fairness metrics, but the approach is in theory applicable to any quantity of interest that is averaged over the test set e.g. model performance.

### Strengths
The paper is very easy to follow, despite containing a lot of theory. To accomplish this, the manuscript starts from an ideal scenario :known population risk $Q$ and known marginal distribution of said risk $P_0$. Then each assumption is subsequently relaxed. First, acknowledging that $P_0$ is unknown means that we must find an upper bound on expected improvement that holds with high probability. Further relaxing our knowledge of $Q$ to that of its empirical counterpart $\widehat{Q}$ is then explained to derive the final form of the algorithm. Structuring the paper this way is very pedagogical and helps the reader understand each step of the derivation.

The algorithm is well motivated based on Theorem 3.5. 

The correctness of the algorithm is demonstrated empirically on three datasets and model types.

### Weaknesses
## Extending Experiments

While the presented experiments highlight that the algorithm is correct (the upper bounds holds with probability at least 95% in general), they could be extended to highlight interesting trade-offs and other applications beyond fairness.

For instance, the appendix presents Algorithm 2 as an alternative that uses of subset of the trained models to estimate the upper bound in conditional expected improvement $\overline{\mu}$. However, the tightness of this algorithm is not compared empirically with Algorithm 1. There might be interesting trade-offs between algorithms 1 and 2 in terms of number of model trainings required for the bounds to go below $\gamma$. Algorithm 2 has a tighter $\overline{\mu}$ but it requires separate samples to first compute the quantile $C$ and the bound $\overline{p}_t(\delta/3)$ is looser because of the union bound. Consequently, it would be pertinent to add Algorithm 2 to Figure 1.

Another way the experiments could be extended is to apply Algorithm 1 to another use-case. For example, I think it is perfectly applicable to hyperparameter optimization via random search. Applying Algorithm 1 to find the hyperparameters of the Random Forests and MLP used in the experiments would highlight the versatility of the method. To avoid diluting the main message of the paper (which is about less discriminative alternatives), these additional experiments could be placed in a dedicated appendix.


## More details on Assumption 3.4

Assumption 3.4 is the key to replace population risk $Q$ with the empirical risk $\widehat{Q}$ in the algorithm. While this assumption is motivated by citing the existing literature, it would be better to assess whether it holds in the experiments. This should be doable since the experiments are designed so that population distributions are known. 

Figure 4 shows that mis-coverage is higher for more expressive models (RFs and MLPs). It would be interesting to see if assumption 3.4 is indeed less likely to hold for these models.

### Questions
In Equation 1, shouldn't the expectation be $\mathbb{E}\_{\mathbb{P}\_0\times \mathbb{P}\_0}[U\_{\tau} - U\_{\tau+1} | \widehat{U}\_{\tau}]$? This is because $\mathbb{P}_0$ is the marginal distribution of a single pair $(\widehat{U}, U)$, while in Equation 1 involves two pairs $(\widehat{U}\_{\tau}, U\_{\tau})$ and $(\widehat{U}\_{\tau + 1}, U\_{\tau+1})$? I assume that $U\_{\tau}$ remains a random variable when we condition on $\widehat{U}\_{\tau}$?

The theorems bound the expected improvement resulting from training **one** additional model. But let's assume that I can train $B$ models in parallel with no additional costs. How easy would it be to extend the theorems to bound the expected improvement from training $B$ additional models? Would it be as simple as applying a union bound? Maybe it is possible to do better than that, since the union bound will become artificially loose at $B$ increases?

The datasets used in the experiments are quite large. Would the approach provide a good coverage on smaller datasets e.g. COMPAS?

### Soundness
3

### Presentation
4

### Contribution
3
