# Near Optimal Robust Federated Learning Against Data Poisoning Attack

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
We revisit data poisoning attacks in the federated learning system. There will be $m$ worker nodes (each has $n$ training data samples) cooperatively training one model for a machine-learning task, and a fraction (i.e.,  $\alpha$) of the workers may suffer from the data poisoning attack. We mainly focus on the challenging and practical case where $n$ is small and $m$ is large, such that each worker does not have enough statistical information to identify the poisoned data by itself, while in total they have enough data to learn the task if the poisoned data are detected. Therefore, we propose a mechanism for workers to cooperatively detect workers with poisoned data. In terms of attack loss, our mechanism achieves $\tilde{O}((\frac{1}{n})^{\frac{1}{2}}+(\frac{d}{mn})^{\frac{1}{2}})$ in IID setting and $\tilde{O}((\frac{1}{\gamma})^{\frac{1}{2}}+(\frac{1}{n})^{\frac{1}{2}}+(\frac{d}{mn})^{\frac{1}{2}})$ in non-IID setting, where $d$ is the VC-dimension of the learning model and $\gamma$ is a concentration parameter characterizing the non-IIDness. Alongside attack loss, our mechanism limits the adversary’s free-ride gain even when it cannot be directly quantified by the attack loss. We also propose the lower bound of the attack loss, and our proposed algorithm matches the lower bound when $m\rightarrow \infty$ both in IID setting and non-IID setting.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies robust federated learning (FL) under data poisoning in the regime with many workers \(m\) and few samples per worker \(n\). The authors focus on two goals: minimizing the attack loss (excess error over the attack-free optimum) and bounding a new metric, Effective Poison Rate (EPR), which measures how much of the overall data becomes *effectively* poisoned by the learning pipeline. 

They first prove minimax lower bounds on the unavoidable attack loss: in IID settings the loss is \(\Omega(1/\sqrt{n})\); for non-IID modeled with a Dirichlet distribution of concentration \(\gamma\), there is an extra \(\Omega(1/\sqrt{\gamma})\) penalty. Then they propose a two-phase defense. Phase 1 learns trustworthiness weights for workers by training a discriminator to maximize a dataset-variance objective aligned with an \(H\)-divergence view; workers with high deviation get down-weighted. Phase 2 runs weighted FedAvg using these trust weights. 

The authors prove upper bounds that match the lower bounds up to logarithmic factors: for IID, attack loss and EPR are \(\tilde{O}\!\left(1/\sqrt{n}+\sqrt{d/(mn)}\right)\); for non-IID, they add \(\tilde{O}(1/\sqrt{\gamma})\). Here \(d\) is a capacity term (e.g., VC dimension). Empirically, on MNIST and CIFAR-10 with label-flip and backdoor attacks, the method tends to improve test accuracy and decrease attack success compared to geometric median, Krum, and iterative filtering, while keeping server-side cost close to standard FedAvg. Overall I feel the paper is promising and quite practical.

### Strengths
- Clear objectives: I appreciate the joint focus on attack loss and the new EPR metric.
- Matching Lower and upper bounds align up to logs in both IID and non-IID cases, and the non-IID penalty is expressed cleanly via \(\gamma\).
-The algorithm and results are novel to my knowledge. Using a discriminator on client datasets (outputs) to drive weights avoids heavy dependence on high-dimensional gradient aggregation.
- Practical cost: Communication and compute remain similar to FedAvg; the method seems easy to plug into existing FL pipelines.

### Weaknesses
- Baselines are mostly classic robust aggregators. Including more recent FL defenses (NNM, Bucketing, ...) and also poisoning specific defenses (see questions below) would give a fairer picture of current SOTA.


- Many results rely on \(\alpha<1/3\). Whereas most other works only require a majority of honest clients. is there a fundamental reason for this?

- Experiments mainly test label-flip and standard backdoor. Stronger or adaptive poisoning (e.g., clean-label poisoning, gradient-aware poisoning including fall of empires and a liitle is enough) would strengthen the empirical story.

### Questions
The following two papers seem very related
   - An Equivalence Between Data Poisoning and Byzantine Gradient Attacks (S. Farhadkhani, R. Guerraoui, L. Hoang, O. Villemaud).  
   - On the Relevance of Byzantine Robust Optimization Against Data Poisoning (Sadegh Farhadkhani, Rachid Guerraoui, Nirupam Gupta, Rafael Pinot).  
   What is the connection with those results? In particular, how do your EPR notion and your lower/upper bounds relate to the equivalence and to the implications for gradient-robust optimization? Do your results identify regimes where gradient-robustness is provably insufficient but your discriminator-weighting remains effective?


Can your method work under secure aggregation or local differential privacy? If gradients/outputs are obfuscated, can the \(H\)-divergence alignment still be estimated reliably?

In cross-device FL with partial participation, \(\alpha\) can change across rounds. Do your theory and algorithm adapt if the set of corrupted clients varies over time?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
This paper proposes a mechanism for data poisoning attacks in federated learning that asymptotically achieves the theoretical lower bound of attack loss in both IID and non-IID settings when α<1/3. Additionally, the authors introduce a new metric, EPR, to quantify the effectiveness of data poisoning attacks.

### Strengths
1. This paper proposes a mechanism for data poisoning attacks in federated learning that asymptotically achieves the theoretical lower bound of attack loss in both IID and non-IID settings when α<1/3.
2. The authors introduce a new metric, EPR, to quantify the effectiveness of data poisoning attacks.

### Weaknesses
**The main contributions of this paper lie in the theoretical domain, an area where I do not feel adequately qualified. Therefore, I will focus my critique primarily on potential weaknesses from a practical perspective.**

1. From a practical perspective, data poisoning attacks are generally considered less threatening to federated learning (FL) systems than model poisoning attacks. The latter can directly manipulate model parameters to make malicious updates resemble benign ones, thereby evading detection more effectively. The paper does not clearly justify why focusing on data poisoning is necessary or impactful given this context.

2. The experimental evaluation includes only four baseline methods, all of which were proposed before 2022. This limited selection raises concerns about the adequacy and fairness of the comparison. Including more recent and stronger baselines would provide a clearer picture of the proposed method’s relative performance and novelty.

3. The proposed method is evaluated only on two small-scale datasets, MNIST and CIFAR-10. While these benchmarks are commonly used, they are insufficient to demonstrate robustness and scalability in realistic FL scenarios. Experiments on larger datasets such as ImageNet would strengthen the empirical claims, and extending the evaluation to other modalities, such as NLP tasks, could further validate the generality of the proposed approach.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

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
The paper addresses the challenge of defending against data poisoning attacks in federated learning by proposing a novel algorithm. The method first assigns a trustworthiness weight to each worker based on the variance of datasets, and then updates the global model using local gradients weighted by these trustworthiness weights. The authors provide a theoretical analysis of the attack loss, establishing an upper bound that matches the lower bound as the local dataset size approaches infinity. Empirical results further corroborate the theoretical analysis, demonstrating the algorithm’s robustness under data poisoning attacks and its superiority over representative robust aggregators.

### Strengths
1. The investigated problem of developing federated learning algorithms robust to data poisoning attacks is important and remains relatively underexplored.
		
2.  This paper conveys a key message that fine-grained algorithmic designs can defend against data poisoning attacks by exploiting the fact that poisoned workers still strictly follow the algorithmic protocol, rather than relying on coarse-grained robust aggregators.

### Weaknesses
1. The lower bounds of the attack loss presented in Theorems 3.1 and 3.2 appear to be not tight, as the upper bounds of the proposed algorithm in Theorems 5.1 and A.1 do not align with them, leaving at least a gap of $\tilde{O}\left(\sqrt{\frac{d}{mn}}\right)$. Could the lower or upper bounds be further improved to close this gap?
		
2. Related to my above comment, the authors should compare their derived lower bound with that in the case of model poisoning attacks, e.g., $\Omega\left(\frac{\alpha}{\sqrt{n}} + \sqrt{\frac{d_g}{mn}}\right)$ in Yin et al. (2018), and clarify why the lower bound under data poisoning attacks is asymptotically smaller than that under model poisoning attacks.
		
3. In Equation (8), what is the role of the probability simplex $a$? Furthermore, does the updated value $F_i(\theta, a)$ pose any risk to the privacy of worker $i$? The authors are encouraged to provide further discussion on these points.
		
4. In line 319, the authors claim that the variance of normal workers’ datasets is small when their datasets are correlated. In my view, this holds only under the i.i.d. setting. Will this statement also hold in the non-i.i.d. case? If so, please provide a detailed explanation, as this is a key insight underlying the proposed algorithm.
		
5. In Algorithm 2, the server removes a worker when its score exceeds a threshold $\eta$. How is $\eta$ chosen, either theoretically or in the experiments? Furthermore, why is the final size of the trust set $(1-2\alpha)m$ rather than $(1-\alpha)m$, given that only an $\alpha$ fraction of workers are poisoned?
		
		
6. The experimental baselines are limited. Several other algorithms defend against data poisoning attacks in federated learning [1] [2]. The authors should include comparisons with these methods in their experiments.

    [1] Jebreel, N. M., Domingo-Ferrer, J., Sánchez, D., \& Blanco-Justicia, A. (2024). Lfighter: Defending against the label-flipping attack in federated learning. Neural Networks, 170, 111-126.

    [2] Hallaji, E., Razavi-Far, R., Saif, M., \& Herrera-Viedma, E. (2023). Label noise analysis meets adversarial training: A defense against label poisoning in federated learning. Knowledge-based systems, 266, 110384.

### Questions
My detailed questions are listed in the above section; please refer to it.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the problem of defending against data poisoning attacks in Federated Learning (FL), focusing on the practical scenario where the number of workers $m$ is large, but each has only a small local dataset of size $n$. The authors make three key contributions:

Lower Bounds: They establish minimax lower bounds for the attack loss, proving it is $\Omega(1/\sqrt{n})$ in the IID setting and $\Omega(1/\sqrt{\gamma} + 1/\sqrt{n})$ in a Dirichlet-based non-IID setting (where $\gamma$ is the concentration parameter).

Novel Algorithm: They propose a two-phase algorithm. The core innovation is a trustworthiness weight update phase that trains a discriminator model to maximize a notion of dataset variance, allowing the server to assign weights to workers and effectively identify those with poisoned data. This is fundamentally different from prior robust gradient aggregation methods.

New Metric and Upper Bounds: They introduce a new robustness metric, the Effective Poison Rate (EPR), to quantify the adversary's gain. Their algorithm achieves upper bounds for both attack loss and EPR of $\tilde{O}(1/\sqrt{n} + \sqrt{d/(mn)})$ (IID) and $\tilde{O}(1/\sqrt{\gamma} + 1/\sqrt{n} + \sqrt{d/(mn)})$ (non-IID), which asymptotically match the lower bounds when $m \rightarrow \infty$, demonstrating near-optimality.

### Strengths
Novel Defense Strategy: The two-phase algorithm, particularly the trustworthiness weight update phase using a discriminator model, is a creative and well-motivated departure from existing methods.

Theoretical Completeness: Provides a full minimax analysis with matching lower and upper bounds under both IID and non-IID settings, a hallmark of strong theoretical work.

Practical Performance: Empirically demonstrates superior performance over several strong baselines on multiple datasets and attack types, validating the theoretical advantages.

New Metric: The Effective Poison Rate (EPR) is a useful and intuitive metric for quantifying the adversary's success beyond simple test accuracy.

### Weaknesses
Computational Overhead: The trustworthiness weight update phase requires training an auxiliary discriminator model, which adds non-trivial computational and communication cost before the main training can begin. While argued to be similar to FedAvg per round, the total cost of this extra phase is not thoroughly compared against baselines.

Strong Assumption on $\alpha$: The theoretical guarantees hold under the assumption that the fraction of malicious workers $\alpha < 1/3$. The performance of the algorithm when this assumption is violated (e.g., $\alpha \geq 1/3$) is not explored empirically or discussed in depth.

Clarity of Presentation: As noted above, the paper is challenging to read. The technical depth is high, but the presentation could do more to guide the reader through the complex ideas and proofs.

### Questions
The theoretical analysis and algorithm design assume $\alpha < 1/3$. What is the empirical performance of your method when $\alpha \geq 1/3$? Does the performance degrade gracefully, or is there a sharp breakdown point? Is there a pathway to relax this assumption in future work?

The trustworthiness weight update phase is a pre-training step. How does the computational and communication cost of this phase scale with the number of workers $m$ and the model/discriminator complexity? Could this initial phase become a bottleneck in very large-scale FL systems compared to baselines that perform robust aggregation during the main training?

Your method effectively identifies and down-weights suspicious workers. However, in a strongly non-IID setting, could "honest but unusual" workers (those with rare but valid data distributions) be mistakenly penalized by your variance-maximizing discriminator? How does your theory or experiments account for or mitigate this potential issue?

### Soundness
3

### Presentation
2

### Contribution
4
