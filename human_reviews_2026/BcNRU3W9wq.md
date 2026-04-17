# GPAR: Gaussian process based association rule mining

- Decision: Reject
- Scores: 4, 2, 0, 4

## Abstract
We introduce GPAR, a Gaussian process-based framework for association rule mining (ARM) that transforms rule discovery into a continuous latent variable modeling problem. Unlike traditional frequency-based methods which treat items as atomic symbols, GPAR represents items via feature vectors and fits a Gaussian process to latent membership variables. In the first stage, kernel hyperparameters are optimized via a Gaussian \emph{pseudo-likelihood}, learning a data-driven similarity structure over items. In the second stage, rule metrics are derived from the latent GP: co-occurrence probabilities are computed as orthant probabilities of a multivariate Gaussian via Monte Carlo integration. This yields principled probabilistic support, confidence, and lift, along with uncertainty estimates obtained by sampling the latent field. Importantly, GPAR enables \emph{zero-shot inference}: rules involving unobserved itemsets can be evaluated without reprocessing the transaction database by conditioning the GP on new feature vectors. Experiments on synthetic and real-world benchmarks demonstrate that GPAR recovers rare, high-lift patterns more reliably than classic baselines and assigns non-zero probability to plausible but unobserved combinations. While this expressiveness incurs higher computation, which restricts its applicability to small number of items regime, GPAR offers a robust alternative for feature-rich, high-stakes domains where accurate probabilistic estimation outweighs scalability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes GPAR, which models latent co-occurrence in transactional data using Gaussian Processes over item feature vectors and kernels. Frequent itemsets and rules are obtained via Monte Carlo estimates of joint occurrence probabilities. The method can generate rules for unseen items without retraining by augmenting the covariance. Overall complexity is dominated by $O(2^M(M^3+SM^2))$, so the approach targets small $M$. Experiments compare against Apriori / FP-Growth / Eclat on two synthetic datasets and one real dataset.

### Strengths
- **Probabilistic modeling & uncertainty.** GP offers principled uncertainty quantification and richer relational modeling than frequency-based methods.
- **Encodable priors.** Kernels allow injecting domain priors (e.g., complement/substitute relations) that classic ARM struggles to express.
- **New-item inference.** Can generate rules for unseen items without retraining—useful for dynamic catalogs/cold starts.
- **Kernel variety.** Beyond RBF, the paper discusses shifted-RBF, NTK, and NN kernels to capture nuanced non-linear patterns.

### Weaknesses
1. **No feature-level ablations / prior sensitivity.** Heavy dependence on feature mappings and kernel priors, yet experiments vary only kernels/thresholds; no ablation or sensitivity analysis on feature subsets or “good vs. bad” priors.
2. **Shifted-RBF non-PSD & semantic shift risk.** Generally non-PSD, requiring eigenvalue clipping; no quantification of how this projection alters relational semantics or spectrum—risk of “converges but mischaracterizes relations.”
3. **Scalability.** Complexity confines applicability to small $M$ (e.g., $M \le 15$), limiting use in large, high-dimensional domains.
4. **NTK evidence is thin; extrapolation risk.** NTK is evaluated only on a single synthetic dataset with mainly count/time metrics and no quality/robustness checks (cross-distribution validation, feature shuffling, counterfactuals). It remains unclear whether “new relations” are genuine or artifacts from kernel-prior extrapolation.

### Questions
see weakness

### Soundness
1

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
This paper proposes GPAR, a Gaussian Process alternative to frequency-based association rule mining. Each item is represented by a feature vector. A GP prior with kernel k(x_i, x_j) over item-level latent variables z is fitted on transactions. The probability that an itemset I occurs is then approximated as a Monte-Carlo estimate of the probability that all components of the latent vector z_I are positive.
Different kernels, e.g., RBF, shifted RBF, NTK, and an erf neural-net kernel, are explored. The authors claim GPAR can: (1) quantify uncertainty; (2) generalize to unseen items via kernel conditioning; and (3) infer new frequent itemsets without re-processing.
Experiments on two synthetic sets (10–15 items) and a UK accident subset (39 items) compare GPAR to Apriori, FP-Growth, and Eclat wrt runtime, memory, number of itemsets and rules, and top-rules tables.

### Strengths
1. The goal is ambitious. Feature-aware, uncertainty-quantified, new-item ARM in a unified model.
2. The discussion on Kernel choices is interesting. NTK and erf kernels are reasonable PSD choices. 
3. The work is quite transparent about scalability limits and includes complexity accounting.

### Weaknesses
1. The GP fitting approach is quite a stretch, and the benefits seem negligible.
2. I am not quite sure how the GP fitting objective would work for binary data. How do you get the posterior conditioning in rule probabilities? The likelihood term does not seem valid for binary data. 
3. It is not clear why confidence values get larger than 1. This should not happen.
4. It is not clear at all how new items are embedded. How do you embed new items without recomputing a global eigendecomposition? 
5. The fairness of the comparison in the evaluation is not clear. Since GPAR can produce rules with zero observed support, how do you make sure the metrics in the comparison against competitors are fair? 
6. Exponential enumeration with Monte Carlo estimation per itemset is very costly. There is no pruning or scalable inference strategy.
7. There are multiple internal inconsistencies (runtimes, duplicated content).

### Questions
See comments above.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
The paper presents a method based on Gaussian processes (GPs) for association rule mining, identifying from a set of transactions frequently occurring itemsets and understanding the relation in them, namely, which items precede the other ones. The goal is to construct such rules that capture meaningful associations. The authors frame this problem as latent variable models in which a GP prior is imposed on the latent variables, the inputs are feature vectors per item, and the observations are the transactions. The authors inspect several kernel functions and compare their method to classical algorithms.

### Strengths
* Novel formulation of association rule mining using Gaussian processes.
* In terms of the number of rules and the number of frequent itemsets generated, the method seems to be more flexible. 
* The method seems to uncover meaningful and complex patterns.

### Weaknesses
In my opinion, this paper is not ready and suitable for publication at ICLR. First, the paper deals with pattern discovery in transactional datasets, which is more suitable for other venues, in my opinion, but I leave that decision to the AC. Second, the novelty of this paper, in my opinion, is limited to applying a GP to a new setup. Third, there are formatting issues across the paper, such as incorrect citation style, broken references for citations (which appear as ?, e.g., in lines 584 and 627), the appendix header appears as a section header, and the same two paragraphs appear in two places (line 310 and line 337).
I will make a few comments regarding the method and experiments that seem most crucial to me.
Regarding the method:
* It seems that the authors addressed this problem as a multi-task problem sharing the same input, where each transaction is a different task and the transactions are independent. I infer that from Eq. 2. Is that right? No explanation or reference is given to that, and to me, this modeling choice seems odd. Why assume independence between the transactions? 
* The authors implicitly assume a Gaussian likelihood, which is also not justified, as the observations are vectors containing elements in the set {0,1}. Can the authors please clarify that point?

Regarding the experiments:
* The datasets are too simple for a non-theory paper (two syntactic ones and a real one).
* The evaluation metrics are also not clear. How do I know which method is better? Why the lift metric isn't presented for all methods? Is it better or worse that a method generates more rules? How does one know if the rules are good or not?

### Questions
.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work focuses on Association rule mining (ARM) task, which aims to discover relationships between items in transactional datasets. Traditional ARM algorithms such as Apriori, FP-Growth, and Eclat face critical challenges: poor scalability with high-dimensional or dense datasets due to exponential growth of possible itemsets, failure to capture probabilistic dependencies between items, and inability to leverage additional item features to model complex relationships. This work proposes Gaussian Process Association Rules (GPAR), a novel probabilistic framework for ARM that leverages Gaussian Processes (GPs) to model joint probabilities of itemsets.

### Strengths
The paper introduces a novel probabilistic framework (GPAR) that extends traditional association rule mining using Gaussian Processes, offering a principled approach to uncertainty quantification.
Incorporating item feature vectors and custom kernel functions (RBF, shifted RBF, NTK, NN kernel) enables GPAR to capture richer and more complex relationships between items than frequency-based methods.
Experimental results on both synthetic and real-world datasets show clear improvements in identifying rare or subtle itemsets and probabilistic rule discovery.
The paper is theoretically grounded, providing detailed mathematical formulations and clear comparisons with classic ARM algorithms.

### Weaknesses
Interpretability of probabilistic rules is lower than traditional if–then rules, making it harder for practitioners to apply results intuitively.

The approach relies heavily on well-defined feature representations and careful kernel tuning, which may limit general applicability.

### Questions
However, there are some doubts in understanding the technical details:
1/ In equation(3) and appendix D, what's the difference of $\mathcal{N}(\mathbf{z}_I; \mathbf{0}, K_I)$ and $\mathcal{N}(\mathbf{z}_I \mid \mathbf{0}, K_I)$.
2/ In section 5, Why choose the three kernel(shifted RBF, neural tangent kernel (NTK), and a neural network kernel with erf activation)?  There are no  comparisons with other kernel functions suitable for association rule scenarios (such as Matern kernel).
3/ The complexity bottleneck has not been addressed,  it has the same complexity problem as traditional methods.
4/ In section 7, author mention that GPAR is impractical for large M (e.g.M > 15). Why is it 15? Is there any basis for this?
5/ The convergence verification of Monte Carlo sampling, supplementing the "sampling number S- estimation error" curve (such as testing S=50/100/200/500 on Synthetic 1), proves the rationality of S=100.      

Regarding the experiment part, there are some doubts:
1/ In related work, author mention the Bayesian association rule mining, but why not compare it with BAR? Indeed, the methods of comparison are all very outdated and there are no mainstream methods of recent years for comparison.
2/ In related work, author mention the Huwel & Beecks (2023) combined Apriori with Gaussian Processes (GPs) ,why not make a comparison with it
3/ Why not add some comparisons with other kernel functions?
4/ The experiment only counted "runtime performance, memory usage, number of frequent itemsets and number of rules generated", without evaluating the practicality and accuracy of the rules.
5/ It is proposed that GPAR can achieve rule reasoning of new items by extending the core matrix without retraining, but this function has not been verified through experiments.
	There are suggestions for improving the writing the paper:
1/ The methods of literature research are too outdated and lack persuasiveness.
2/ The relevant work research is insufficient and there is a lack of the latest related work.
3/ On line 246,356,584,627,634,701,1097,1133, appear the garbled characters Question marks and exclamation marks.
4/ Try to bold the parts with the best performance in all the tables.
5/ In section 6, There are two repeated paragraphs. I don't know what the meaning is.(line 310~323,337~351).
6/ It would be best to draw an algorithm framework diagram.

### Soundness
3

### Presentation
3

### Contribution
3
