# Tuning without Peeking: Provable Privacy and Generalization Bounds for LLM Post-Training

- Avg Score: 4.50
- Decision: Reject
- Scores: 8, 4, 0, 6

## Abstract
Gradient-based optimization is the workhorse of deep learning, offering efficient and scalable training via backpropagation. However, exposing gradients during training can leak sensitive information about the underlying data, raising privacy and security concerns such as susceptibility to data poisoning attacks. In contrast, black box optimization methods, which treat the model as an opaque function, relying solely on function evaluations to guide optimization, offer a promising alternative in scenarios where data access is restricted, adversarial risks are high, or overfitting is a concern. This paper introduces BBoxER, an evolutionary black-box method for LLM post-training that induces an information bottleneck via implicit compression of the training data. Leveraging the tractability of information flow, we provide non-vacuous generalization bounds, and strong theoretical guarantees for privacy, robustness to data poisoning attacks and extraction attacks. In experiments with LLMs, we demonstrate empirically that black-box optimization methods—despite the scalability and computational challenges inherent to black-box approaches—are able to learn, showing how a few iterations of BBoxER improve performance, generalize well on a benchmark of reasoning datasets, and are robust to membership inference attacks. This positions BBoxER as an attractive add-on on top of gradient-based optimization, offering suitability for deployment in restricted or privacy-sensitive environments while also providing non-vacuous generalization guarantees.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces BBoxER, a novel black-box post-training framework that enables secure fine-tuning of large language models without direct access to gradients or training data. Instead of learning from full supervision, BBoxER compresses the training signal into discrete comparison traces, transmitting only a few bits of feedback per iteration. Leveraging black-box optimizers such as CMA-ES, OneFifth, and D-CMA, it updates model parameters solely based on relative performance comparisons. Theoretically, the method establishes provable generalization bounds (via Hoeffding and Bonferroni inequalities) and robustness guarantees under data poisoning and differential privacy constraints (Theorem 2). Experiments on GSM8K, SVAMP, and GSM+ show competitive reasoning improvements despite the extremely limited information flow. Conceptually, BBoxER reframes model fine-tuning as information-constrained optimization, providing a principled bridge between privacy, robustness, and efficient adaptation in large-scale LLMs.

### Strengths
1. A novel research perspective of black-box retrofitting enables fine-tuning large language models without accessing gradients or original training data.
2. The method tightly integrates the information compression bottleneck with generalization theory, formally bounding the information flow to one bit per iteration.
3. Experiments on mathematical reasoning tasks (GSM8K, SVAMP, GSM+) show 5–7% accuracy gains under a no-gradient setting, confirming the method’s effectiveness.

### Weaknesses
1. Theoretical derivations contain omissions and jumps; Theorem 2 lacks a complete proof and key assumptions are not clearly explained.
2. The correspondence between theory and implementation is unclear; the link between continuous Gaussian perturbations and discrete branching assumptions should be clarified.
3. In Appdex C.2.1, the symbol $\pi_i$ is not explicitly defined in the paper, causing confusion and hindering understanding..
4. The code is not released and implementation details are missing, making the method hard to follow and reproduce.
5. The experimental scope is narrow, focusing only on mathematical reasoning tasks without broader task validation.
6. The privacy analysis is mostly qualitative; although extraction and membership inference attacks are discussed, there is no empirical support or comparison.
7. Several references are duplicated and should be cleaned up for clarity and consistency.

### Questions
See weakness.
1. Can simple Gaussian perturbations of model parameters be effectively captured by the theoretical generalization and robustness bounds? If so, how do they correspond to the finite branching assumption in the theory?
2. Why does the paper adopt an exponential form of parameter update (e.g., $exp(x)$)? Compared with additive perturbations, what are the theoretical or practical advantages of this approach, and were alternative strategies considered?

### Soundness
4

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
3

### Summary
This paper introduces BBoxER, a framework for LLM post-training using gradient-free Black-Box Optimization (BBO). The method avoids gradients by using comparison-based algorithms, which compress the dataset into a sequence of binary comparison results . The authors leverage this information bottleneck to derive theoretical guarantees for generalization, differential privacy, and robustness to poisoning/extraction. Empirically, they show BBoxER improves reasoning on Llama3.1-8B and resists membership inference attacks.

### Strengths
1. Linking BBO compression to generalization bounds that are independent of parameter count is a significant theoretical contribution.
2. Paper provides formal theorems for robustness against poisoning and extraction.
3. Paper shows detailed empirical results.

### Weaknesses
1. The method is not scalable. It only "works" by tuning a trivially small subset of parameters. This is a minimal tweak, not a scalable framework. Does the privacy stems from the fact that it barely changes the model?
2. The DP guarantee seems to be meaningless. The "general case" privacy guarantee is $(\epsilon=0, \delta \propto b/\sqrt{s})$. Using the paper's own numbers ($b=1200, s=7473$), $\delta \approx 13.9$. A $\delta > 1$ is a meaningless guarantee.
3. The paper claims to be "memory-efficient"  but ignores the astronomical computational cost. Each step of the BBO algorithm requires at least one full evaluation pass over the training data to get a single comparison bit. Could authors show the detail of running time and memory usage of this part?
4. Only evaluate on limited number of datasets.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper analyzes privacy and generalization error for post-training via comparison-based black-box optimization under discrete signal. They prove that due to the limited number bits contributed by each data record, such algorithm enjoys non-vacuous  generalization error bound that scale linearly with dataset size, and as corollary establish provable robustness to poisoning and resistance to extraction attacks, which are further confirmed by experiments of evaluating the proposed algorithm via Membership Inference Attacks.

### Strengths
- Extensive experiments showing that the proposed algorithm resist Membership Inference Attacks, unlike fine-tuned counterparts at equal utility.
- The problem of information leakage at post-training phase is timely.

### Weaknesses
- Lack of clarity: the main algorithm 1 has many undefined functionalities such as 'modified', 'tell', 'ask', and contains many completely terms $I_i, i=1, b$ that are unused after creation. Given the current state, it is almost impossible to precisely interpret what the algorithm executes as well as the novelty (which the authors claim in the introduction).
- Differential privacy guarantee in Section 4.2 is vacuous, meaningful DP requires delta to be polynomially small compared to dataset size (Vadhan 2017), while the DP guarantee in Section 4.2 is inverse proportional to sqrt(dataset_size), not strong enough to provide meaningful privacy guarantee. There is also no formal proof for the DP bound in the paper.
- No discussions on existing related works that show information leakage during post-training, under SFT and RL. (Such as Hayes 2024, Barbero 2025)


References:
- Vadhan, S. (2017). The complexity of differential privacy. In Tutorials on the Foundations of Cryptography: Dedicated to Oded Goldreich (pp. 347-450). Cham: Springer International Publishing.
- Hayes, J., Shumailov, I., Porter, W. P., & Pappu, A. (2024). Measuring memorization in RLHF for code completion. In The Thirteenth International Conference on Learning Representations.
- Barbero, Federico, Xiangming Gu, Christopher A. Choquette-Choo, Chawin Sitawarin, Matthew Jagielski, Itay Yona, Petar Veličković, Ilia Shumailov, and Jamie Hayes. "Extracting alignment data in open models." arXiv preprint arXiv:2510.18554 (2025).

### Questions
See weakness.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces BBoxER, a framework that uses comparison-based black-box optimization (BBO) for retrofitting models without requiring gradient access. BBoxER's core contribution is creating an information bottleneck by compressing the entire dataset into a small optimization trace of comparison outcomes. This compression mechanism allows the paper to provide strong theoretical guarantees for differential privacy, robustness to data poisoning. Experiments on LLMs demonstrate that BBoxER improves reasoning performance and empirically resists MIAs, proving it a practical method for achieving safe adaptation without sacrificing utility.

### Strengths
1. This work is the first to formally connect the finite "branching factor" of BBO algorithms to quantifiable safety guarantees (like generalization, privacy) for LLMs.
2. The paper is well-written and conceptually clear.

### Weaknesses
1. The paper emphasizes memory efficiency (no gradients) but lacks a critical discussion of computational cost. The BBO approach requires $b$ iterations (up to 1200 in experiments), with each comparison involving a full pass over the training dataset. This implies a potentially massive computational cost compared to standard SFT (2-5 epochs).  
2. The MIA robustness claim relies on a proxy metric (NLL difference), which is not an appropriate evaluation metric for MIA. It is recommended to use standard MIA methods such as LiRa [1] or other methods designed for LLMs [2,3,4], and report common MIA metrics such as TPR at low FPR.
3. The paper appears to directly leverage an existing BBO algorithm for post-training on LLMs. This is not a novel framework, and the authors are encouraged to reframe their contribution accordingly.

### Questions
What is the true computational cost of BBoxER compared to standard SFT?

### Soundness
3

### Presentation
4

### Contribution
2
