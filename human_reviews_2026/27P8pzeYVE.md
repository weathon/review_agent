# Painless Federated Learning: An Interplay of Line-search and Extrapolation

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
The classical line search for learning rate (LR) tuning in the stochastic gradient descent (SGD) algorithm can also tame the convergence slowdown due to data-sampling noise. In a federated setting, wherein the client heterogeneity introduces a slowdown to the global convergence, line search can be relevantly adapted. In this work, we show that a stochastic variant of line search tames the heterogeneity
in federated optimization in addition to addressing the slowdown in client-local optimization due to gradient noise. To this end, we introduce the Federated Stochastic Line Search (FeDSLS) algorithm and show that for convex functions, it achieves deterministic rates in expectation. Specifically, FEDSLS offers linear convergence for strongly convex objectives even with partial client participation. Recently, the extrapolation of the server’s LR has shown promise for improved empirical performance for federated learning. Considering that, we also extend FeDSLS to Federated Extrapolated Stochastic Line Search (FEDEXPSLS) to take advantage of extrapolation. We prove the convergence of FEDEXPSLS. Our extensive empirical results show that the proposed methods perform at par or better than the popular
federated learning algorithms across many convex and non-convex problems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduced two algorithms FEDSLS and FEDEXPSLS, proved their convergence property respectively in theory and offer empirical evidence to support the claims. FEDSLS is based on the famous FedAvg algorithm with Armijo line search applied on a client level in a stochastic fashion. FEDEXPSLS on the other hand is based on FedExP, which combines line search and extrapolation as an FL algorithm.

### Strengths
(1) The authors proposed two novel algorithms, and provide corresponding analysis under a set of assumptions, the assumptions are clearly stated and the theorems are proved rigorously. 

(2) The paper combines the line search with extrapolation and obtain state of the art results.

(3) Empirical results are there to further validate the theoretical claims.

### Weaknesses
(1) My primary concern is the set of assumptions used in the paper, the authors assume interpolation condition in all cases of their convergence guarantee. Although previous studies have provide certain justifications, such an assumption is not likely to be checked in general. In fact, I do not quite understand why would such an assumption is needed in the case of FedSLS and FedExPSLS. The authors mentioned FedExProx, and in that case this assumption is needed because of the algorithm connects to the parallel projection algorithm to solve the convex feasibility problem where interpolation holds. However, in this case the two algorithms are based on the FedAvg and FedExP, which does not require such an assumption. It this sense, it seems to me that the acceleration effect of line search is based on this condition, but not the trick itself.

(2) The author claimed that the empirical performance is improved, I suppose here the authors are referring to the iteration complexity, i.e., the proposed algorithms take less communication rounds to converge. This is not a fair comparison because the amount of work performed by each client in each round is different. For each client, we are replacing the original solver SGD with SGD-Armijo, which requires many additional forward passes to ensure that the criteria is met. This causes additional overhead, which could be huge. In my opinion, a comparison of the overall computational needed should be provided. 

(3) As I have mentioned, the algorithm seems to be a direct consequence of replacing the local solvers from SGD to SGD with Armijo, which is already studied by existing literatures. The extension seems to be pretty straight forward, and the benefits are not properly justified , despite its reliance on a very restrictive assumption that may not be needed.

### Questions
(1) In theorem 4, I do not get why this is a convergence guarantee, it seems that the iteration K appears on the numerator if we fix $\eta_{l_\max}$ properly, and in this case the RHS becomes increasing in K. Could the author clarifies how should we interpret this theorem?

(2) Could the authors explain why the proposed algorithm requires interpolation condition to converge. What changed compared to FedExP or FedAvg?

(3) Could the authors show what the overall computational complexity of the algorithm is, and compare to other FL baselines? Is there a way to quantify the number of additional overhead needed for the Armijo subroutine?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes FedSLS, which is like FedAvg except the clients perform Armijo line search to set the learning rate. Aside from the practical benefit of not having to set a learning rate, the Armijo test helps the convergence proof by allowing the analysis to control the client drift rather than assuming bounded heterogeneity which is typically done in practice. This allows the authors to achieve linear convergence in the strongly convex setting even with partial client participation. FedSLS in some sense follows the prior work on stochastic line search and applies it to the federated setting, where inter-client variance (heterogeneity) is somewhat like the variance in the centralized setting. In addition to the theoretical guarantees, the authors are able to show the method performs well on experiments, in particular FedExpSLS.

### Strengths
- The algorithm makes sense and appears simple to implement, and shows that line search is a natural thing to try in FL
- Linear convergence under partial participation is somewhat expected given the analogous result in the centralized setting, but it is good to see the authors are able to derive it.
- FedExpSLS looks pretty strong, with solid convergence results compared to other methods
- The line search overhead seems to not be too much, again showing the algorithm is not too costly

### Weaknesses
- The work assumes that there is an optimum that is shared across all the clients. This is a pretty strong assumption
- Communication round-based plots are good, but there should probably be wall-clock based ones (or total client iteration-based) because of the additional client compute used by the line search
- Tuning grids should be included in the experiments
- Ideally in optimization, training runs should be compared against prior work in a leaderboard-like manner. See for example https://kellerjordan.github.io/posts/speedrun/ in the centralized setting. Is there something like this for the FL optimization literature? Without it, it is hard to trust results are significant
- In particular, the paper mentions that FedAdam underperforms, but for the dataset/tasks considered, it usually performs quite well, see

Reddi, Sashank, et al. "Adaptive federated optimization." arXiv preprint arXiv:2003.00295 (2020).

- I wonder how pretraining or LLM assistance would affect the empirical results? Both have become relatively standard in FL deployments. See

Nguyen, John, et al. "Where to begin? on the impact of pre-training and initialization in federated learning." arXiv preprint arXiv:2206.15387 (2022).

Hou, Charlie, et al. "Private federated learning using preference-optimized synthetic data." arXiv preprint arXiv:2504.16438 (2025).

Wu, Shanshan, et al. "Prompt public large language models to synthesize data for private on-device applications." arXiv preprint arXiv:2404.04360 (2024).

### Questions
See weaknesses

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces two new algorithms, Federated Stochastic Line Search (FEDSLS) and Federated Extrapolated Stochastic Line Search (FEDEXPSLS), that adapt stochastic Armijo line search for federated optimization. By applying stochastic line search locally at clients, FEDSLS dynamically adjusts learning rates, mitigating both gradient noise and data heterogeneity. FEDEXPSLS extends this with extrapolation of the server learning rate to further accelerate convergence. Theoretically, both methods achieve deterministic convergence rates even under partial client participation. Experiments on CIFAR-10, CIFAR-100, FEMNIST, and Shakespeare benchmarks demonstrate that FEDEXPSLS consistently outperforms existing methods like FedAvg, FedExp, and FedProx in both training loss and test accuracy, with negligible computational overhead.

### Strengths
* Introduces the Armijo condition, which can be used to overcome the bias in SGD without the sample-wise interpolation for a single local solver. 
* Theoretical guarantees showing that FedSLS and FedExpSLS achieve deterministic rates in FL under standard assumptions along with the Armijo condition and the interpolation condition.
* Proposed methods are evaluated on multiple datasets (CIFAR-10/100, FEMNIST, Shakespeare) and models (ResNet-18, LSTM, etc.), consistently outperforming FedAvg, FedExp, FedProx, and FedAdam in both training loss and test accuracy,

### Weaknesses
**Lack of clear motivation and positioning:**
The motivation for the work feels underdeveloped. The introduction reads more like a problem formulation than a compelling argument for why this specific direction is needed. Many algorithms already aim to speed up federated learning, so it’s unclear why line search, and particularly Armijo line search, is the right tool for the job. The paper would benefit from a clearer explanation of what gap this approach fills and why combining line search with extrapolation is conceptually or practically appealing compared to existing methods.

**Limited discussion around the Armijo condition:**
Definition 3 is theoretically interesting, but it’s not clear when or how this condition holds in practice. For example, do deep neural networks approximately satisfy it? If not, what types of models or loss functions make this assumption reasonable? Some empirical evidence, perhaps showing the frequency of Armijo condition satisfaction during training, would help clarify the practical relevance of this assumption.

**Restricted experimental scope:**
The experiments are limited to small-scale benchmarks like CIFAR, FEMNIST, and Shakespeare. While these are common in FL papers, they don’t provide much insight into performance or scalability in more realistic cross-device settings. Including larger or more modern datasets such as StackOverflow, Reddit, or Google Landmark-v2 would make the results more convincing. Also, the surprisingly poor performance of FedAdam raises some concerns about hyperparameter tuning or implementation consistency.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2
