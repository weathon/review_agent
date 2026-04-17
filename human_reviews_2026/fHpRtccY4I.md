# ERIS: Enhancing Privacy and Communication Efficiency in Serverless Federated Learning

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Scaling federated learning (FL) to billion-parameter models introduces critical trade-offs between communication efficiency, network load distribution, model accuracy, and privacy guarantees. Existing solutions often tackle these challenges in isolation, sacrificing accuracy or relying on costly cryptographic tools. We propose ERIS, a serverless FL framework that balances privacy and accuracy while eliminating the server bottleneck and significantly reducing communication overhead. ERIS combines a model partitioning strategy, distributing aggregation across multiple client-side aggregators, with a distributed shifted gradient compression mechanism. We theoretically prove that ERIS (i) converges at the same rate as FedAvg under standard assumptions, and (ii) bounds mutual information leakage inversely with the number of aggregators, enabling strong privacy guarantees with no accuracy degradation. Extensive experiments on image and text datasets—ranging from small networks to modern large language models—confirm our theory: compared to six baselines, ERIS consistently outperforms all privacy-enhancing methods and matches the accuracy of non-private FedAvg, while reducing model distribution time by up to $1000\times$ and communication cost by over 94\%, lowering membership inference attack success rate from $\sim$83\% to $\sim$65\%—close to the unattainable $\sim$64\% limit—and reducing data reconstruction to random-level quality. ERIS establishes a new Pareto frontier for scalable, privacy-preserving FL for next-generation foundation models without relying on heavy cryptography or noise injection.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes ERIS, a serverless federated learning framework that combines gradient partitioning across multiple client-side aggregators with shifted compression. The authors claim simultaneous improvements in communication efficiency, privacy, and convergence while maintaining equivalence to FedAvg. The work presents theoretical convergence guarantees and information-theoretic privacy bounds, validated through experiments on image and text datasets ranging from small networks to LLMs.

### Strengths
S1- The paper provides formal convergence guarantees (Theorem 3.6) and information-theoretic privacy bounds (Theorem 3.7), with detailed proofs showing the approach maintains FedAvg equivalence.
S2-  Evaluation spans multiple datasets (MNIST, CIFAR-10, IMDB, CNN/DailyMail), model scales (62K to 1.3B parameters), and compares against six baselines under different privacy attack scenarios (MIA, DRA).
S3- The paper provides thorough ablation studies on the impact of compression and partitioning separately.
S4- The framework successfully scales to billion-parameter models where many privacy-preserving baselines fail to maintain utility.

### Weaknesses
W1- The paper conflates two orthogonal benefits throughout. Each client in ERIS uploads and downloads the same total amount of data (b' bits) as any compression-only method like SoteriaFL. The claimed "communication efficiency" actually refers to reduced distribution time through parallelization, not reduced per-client communication volume. Table 2 is particularly misleading—ERIS shows 1% vs SoteriaFL's 5% primarily due to more aggressive compression (different \omega values), not the partitioning scheme. 
W2- The core techniques are from prior work: (a) distributed aggregation exists in Ako (2016), Shatter (2025), C-DFL (2022); (b) shifted compression is directly from Li et al. (2022d). The main contribution is combining these with a straightforward proof that partitioning with disjoint/complete masks preserves FedAvg convergence (Theorem B.1), which is relatively obvious since it merely reorders aggregation operations. 
W3- The privacy guarantees only hold against honest-but-curious aggregators who do not observe network traffic beyond their assigned shard. A realistic adversary monitoring network communications can reconstruct the full gradient by observing all client transmissions, reducing privacy to compression-only protection. Under collusion (Corollary D.2), privacy degrades linearly—if adversaries observe all A channels, privacy advantages vanish. The paper needs to discusses these limitations.
W4- Despite aggregators being selected from clients (who "may vary in computational resources and connection stability," Section 5.2), the paper does not provide analysis of aggregator dropout/failure during training rounds,  evaluation of model sensitivity to aggregator unavailability or discussion of aggregator selection strategies.
W5- Figure 6 compares ERIS (A=50, with compression) against FedAvg (no compression), exaggerating gains. Fair comparisons should use identical compression ratios. 
W6- Theorem 3.6 shows convergence rate depends on ω (compression), matching SoteriaFL. Partitioning contributes nothing to convergence improvement, it only redistributes computation. This should be stated explicitly.

### Questions
Q1- Can you provide a detailed breakdown in Table 2 showing: (a) per-client upload bytes, (b) per-client download bytes, (c) compression ratio, and (d) distribution time, clearly separating gains from compression vs. parallelization?
Q2- What happens when aggregators drop mid-round? Does training continue with (A-1) aggregators, restart the round, or fail? Please provide analysis of robustness to aggregator failures.
Q3- Can you provide experiments where ERIS and SoteriaFL use identical compression ratios (same compression ratio)? This would isolate the benefit of distributed aggregation from compression.
Q4- How do privacy guarantees degrade when adversaries can observe all network traffic (not just content at aggregators)? Can you quantify information leakage in this realistic threat model?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes ERIS, a serverless federated learning (FL) framework designed to simultaneously improve communication efficiency, scalability, and privacy. The system replaces the central server with multiple decentralized aggregators, and combines this with a shifted gradient compression mechanism and gradient partitioning.

### Strengths
1. The paper provides clear utility and privacy analysis.

2. It is a new idea to have gradient partitioning scheme across multiple aggregators.

### Weaknesses
1. The paper claimed a few "the first": "ERIS is the first FL framework to simultaneously achieve decentralized aggregation, strong communication efficiency, and provable information-theoretic privacy guarantees without sacrificing model utility. ERIS is also the first to
extend privacy-enhancing federated training to modern LLMs, demonstrating feasibility at scale where prior methods fail to preserve utility and efficiency." I think these claims are ambiguous because no proof or metrics to show them. For example, I suppose there are lots of works for privacy-enhancing federated training even to LLMs.

2. The need for each client and aggregator to track the shifting reference vectors introduces extra memory and synchronization complexity.

3. I have concerns about theorem 3.6 as it is not tight. For example, if we use SGD as the inner optimizer, Eq(6) shows that it converges to an error that is independent to learning rate. This is not a classic rate, different from existing decentralized optimization and federated optimization. Furthermore, a table of convergence rate comparison with existing baseline algorithms in decentralized optimization and federated optimization would be appreciated. 

4. The paper attempts to deliver multiple contributions simultaneously, which makes the overall narrative somewhat overwhelming. For example, the claimed communication efficiency is mainly derived from shifted compression, a technique that is not new. As a result, the contribution in this dimension may appear incremental unless more concrete empirical or theoretical advantages are demonstrated.

### Questions
The contributions are about communication and privacy. I wonder if the authors can provide direct comparisons in each dimension with previous works. I suppose there are a lot baselines in decentralized optimizaiton and federated optimization.

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
3

### Summary
This paper introduces ERIS, a decentralized federated learning framework designed to achieve both communication efficiency and information-theoretic privacy guarantees without relying on a central server. The key idea is to partition model parameters into disjoint shards, each handled by a different client-side aggregator, and to apply a “shifted compression” mechanism that reduces communication cost while limiting information leakage. The authors provide convergence and privacy analyses, showing that ERIS maintains FedAvg-like utility bounds(Thm. 3.6) and scales privacy guarantees with the number of aggregators(Thm. 3.7). Empirical evaluations demonstrate improved privacy–utility–communication trade-offs compared to existing decentralized or privacy-preserving FL baselines.

### Strengths
- The paper presents a clear and well-motivated problem setup addressing privacy and communication challenges in decentralized FL.
- The overall framework is interesting: partitioning parameters across aggregators leads to linear scalability and better privacy by design.
- This paper includes solid theoretical analysis, providing convergence results and a clean information-theoretic privacy bound.
- The paper systematically evaluates both MIAs and DRAs, includes a Pareto analysis, and reports per-round communication/time numbers.

### Weaknesses
* The paper states ERIS is the first framework to simultaneously provide decentralized aggregation, communication efficiency, and provable information-theoretic privacy without sacrificing utility. However, it seems that the claims is too strong. Baslines in the experiments such as Shatter and SoterialFL as well as the other prior works (e.g., [Shen et al]) should be discussed and compared before the claim.
* - Shen, Meng, et al. "Secure decentralized aggregation to prevent membership privacy leakage in edge-based federated learning." IEEE Transactions on Network Science and Engineering 11.3 (2024): 3105-3119.

- Algorithm 1 and §3.2 say masks can be predefined/shared or dynamically sampled by each client. For an aggregator-specific shard, it’s more natural that the aggregator (or a coordinator) samples/defines $m_{(a)}^t$ and broadcasts them to all clients per round to guarantee consistent slicing across clients. As written, “dynamically sampled by each client” (line 202) invites inconsistent partitions unless there is a synchronization step. Please clarify the intended control flow (who samples? when? how are masks synchronized/broadcast?)
- The main text analyzes an honest-but-curious non-colluding adversary who only sees a shard; Appendix D mentions an extension to colluding adversaries and §5.2 acknowledges that privacy benefits diminish with collusion, scaling with the number of colluding nodes (Corollary D.2). This is important enough to surface earlier: how much privacy remains if a small constant fraction of aggregators collude? What if an aggregator colludes with a subset of clients?
- The experiments show performance degradation as $A$ grows (Figure 2) while Thm. 3.6 is agnostic to $𝐴$. An intuitive explanation is missing. Is this because increasing$A$ shrinks each shard’s dimensionality and, together with compression, increases effective variance and error-feedback lag on each shard’s reference vector? That can slow optimization or accumulate bias in finite rounds even if the asymptotic bound does not expose an explicit $A$ term.

### Questions
- Please see the weaknesses.
- If $A \rightarrow n$ (i.e., per-coordinate sharding), it there a stability/variance blow-up without increasing bandwidth? Any guidance on a practical range of $𝐴$ w.r.t model dimension and client count?

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
3

### Summary
This paper studies serverless federated learning (FL) that balances the trade-offs among multiple objectives: communication efficiency, network load distribution, model accuracy, and privacy guarantees. Unlike many solutions in the literature, the proposed ERIS FL framework can achieve good balances among all these objectives. The fundamental technique that ERIS leverages is a combination of a model partitioning strategy and a distributed shifted gradient compression mechanism. The authors demonstrate the superior performance both theoretically and through extensive experiments on image and text datasets.

### Strengths
1. I like the nice illustration of ERIS in Figure 1. The authors are able to clearly show how ERIS works during the client computation, shifted compression, model partitioning, and distributed training.

2. I also like it that the authors go beyond the small scale experiments on MNIST, FLMNIST many other papers would use and test the proposed methods on larger scale datasets and models such as 1.3B GPT-Neo.

### Weaknesses
My major concern lies in the literature review. The authors consider balancing the trade-offs among many objectives: communication efficiency, network load distribution, model accuracy, and privacy guarantees. This clearly falls into the multi-objective /task federated learning domain. The authors also conduct the utility and privacy trade-off analysis and plot the Pareto frontier of the solutions in the experiments such as those in Figure 4. All of these indicate that the authors might have already been aware of the multi-objective nature of the problem. However, I did not see a systematic review of multi-objective federated learning papers. For example, 

Yang, Haibo, et al. "Federated multi-objective learning." Advances in neural information processing systems 36 (2023): 39602-39625.

Kang, Yan, et al. "Optimizing privacy, utility and efficiency in constrained multi-objective federated learning." arXiv preprint arXiv:2305.00312 (2023).

Zhang, Xiaojin, et al. "Trading off privacy, utility, and efficiency in federated learning." ACM Transactions on Intelligent Systems and Technology 14.6 (2023): 1-32.

### Questions
Like what I have mentioned in the weakness section, I would like to see the authors have more discussion on the connections and differences with those literatures in multi-objective federated learning domains.

### Soundness
3

### Presentation
3

### Contribution
3
