# Tiered Gossip Learning: Communication-Frugal and Scalable Collaborative Learning

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 2, 2, 4

## Abstract
Modern edge deployments require collaborative training schemes that avoid both the single-server bottleneck of federated learning and the high communication burden of peer-to-peer (P2P) systems. We propose Tiered Gossip Learning (TGL), a two-layer push–gossip–pull protocol that combines the fault tolerance of P2P training with the efficiency of hierarchical aggregation. In each round, device-level leaves push their models to a randomly selected set of relays; relays gossip among themselves; and each leaf then pulls and averages models from another random subset of relays. Unlike other hierarchical schemes, TGL is fully coordinator-free, with communication and aggregation decentralized across nodes. It matches baseline accuracy with up to two-thirds fewer model exchanges, and surpasses it when exchanges are equal, across diverse datasets including CIFAR-10, FEMNIST, and AG-News. We provide convergence guarantees for TGL under standard smoothness, bounded variance and heterogeneity assumptions, and show how its layered structure enables explicit control of consensus-distance bounds. Thus, TGL brings together the strengths of FL and P2P design, enabling robust, low-cost mixing enabling large scale collaborative learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes **Tiered Gossip Learning (TGL)**. It is a two-tier, coordinator-free protocol. Each round has three stages: push (leaf→relay), gossip (relay↔relay), and pull (relay→leaf). The goal is strong mixing with small leaf degrees. The theory shows a contraction with product form \( \beta_{\text{TGL}} = \beta_{lr}\beta_{rr}\beta_{rl} \) and gives convergence under standard assumptions. They also derive a general upper bound \( \beta_{\text{TGL}} \le 1 - 1/e \). Under a relay-heavy “load-balancing’’ condition, the bound is tighter than the Epidemic Learning baseline. Experiments on image, text, and character datasets show better accuracy-per-communication than several decentralized baselines, and often approach FL with similar communication.

### Strengths
- The problem is interesting and practical:
- Solution is simple, easy to implement..
- Clean intuition: multiply three sparse mixing steps to get effective dense mixing.
- Theory links budgets to one multiplicative factor \( \beta_{\text{TGL}} \). This gives clear knobs to trade accuracy vs. communication.
- Experiments are broad and mostly convincing; Spectral-gap simulation supports the mixing part.

### Weaknesses
- The method likely introduces **significantly higher latency**, since each communication round includes three sequential stages (push, relay gossip, and pull). This means roughly **three times the latency** compared to direct node-to-node exchanges in fully decentralized protocols.

- The **name “Tiered Gossip Learning” (TGL)** can be somewhat confusing. Traditionally, *Gossip Learning* refers to an asynchronous decentralized protocol where nodes train and exchange models independently, as in [A]. In contrast, this work **extends the Epidemic Learning** framework, which already includes synchronized rounds and coordinated model exchanges. The paper should clarify this connection more explicitly and perhaps modify the naming to reflect this.

  [A] Róbert Ormándi, István Hegedüs, and Márk Jelasity. *Gossip learning with linear models on fully distributed data.* Concurrency and Computation: Practice and Experience, 25(4):556–571, 2013.

- The **theoretical contribution mainly analyzes the consensus rate** of the proposed scheme. The rest of the convergence proof closely follows from the Epidemic Learning paper, with minor adaptations. This is not necessarily a weakness, but it should be clearly stated in the main body so readers understand the level of novelty in the analysis.

- The method **assumes synchronous rounds and independent random sampling**. The paper does not discuss how the protocol behaves under **network delays**, **node churn**, or **partial participation**. Including an analysis or at least a discussion of these factors would make the paper more complete.

### Questions
Why is EL-Oracle not included in the experiments?

### Soundness
4

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
As decentralized training of machine learning models often suffer from fault tolerance and communication overhead, this paper propose a novel hierarchical federated learning algorithm termed "Tiered Gossip Learning" (TGL). The TGL consists of two-layer push-gossip-pull communication layer so that computational nodes do not communication with each other directly, but rather through intermediate relay nodes. By randomly selecting computation nodes and relaying nodes, TGL is able to have some robustness against failure nodes. Theoretical convergence rates are given and some empirical experiments are conducted to support the claims.

### Strengths
- Clarity: The paper is written in a clear way.
- Originality: Combining relay nodes and random sampling into a two layer communication topology may be original.
- Quality: Theoretical rates look correct, empirical evaluations are given. Overall this seems a complete work.

### Weaknesses
- TGL increases latencies due to the overhead of inter-relay communication.
- TGL increases bubble in training pipelines as Line 258-259 says "Only sampled leaves perform local training in that round" which is not ideal for distributed training in data centers.
- This paper does not give a clear threat model for faulty nodes. It seems this paper assumes as soon as a node fails, servers/relay nodes are automatically aware and move on to the next step. If this paper targets cross-device federated learning, then the system heterogenity of devices leads to some clients being more likely to be straggler and one may not simply exclude them. If this paper only consider simple homogeneous devices, then the novelty in terms of fault tolerance seem limited.

### Questions
I am not sure which distributed training scenario does this paper target? Is it cross-device FL / cross-silo FL/ distributed training inside a data center? The setups of TGL does not seem to fall into these common scenarios.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a new fully decentralized learning framework called tiered gossip learning, which includes a hierarchical aggregation protocol. Both theoretical and numerical analysis is  provided.

### Strengths
* The research topic is quite practical.
* It is always important to robustify decentralized algorithms with an optimized communication budget.

### Weaknesses
* Some of the claims are incorrect:
  * Lines 19: "TGL" is fully coordinator-free. The relays are aggregators of the leaf nodes.
  * In line 147, the authors claim that "but generally require orchestration by a central scheduler, and thus remain structurally closer to FL." However, the authors do not specify the benefits of their learning framework as compared to classical FL. From a communication-complexity perspective, a network coordinated by a central server is more efficient and stable than a gossiping framework. 
* Some of the statements are unclear and need clarification.
  * A concrete definition (using math and notations) of leaf nodes and relay nodes is never given.
  * The reason why the authors use $\text{CDR}$ to measure the consensus mixing efficiency is unclear. In fact, we can track $\text{CD}$ itself.
  * In Step 0 Local Training, the SGD protocol is not written out. Do we have 1-step local optimization or multiple steps?
  * One of the motivations is to make the training framework fault-tolerant. Yet, Theorem 4.4 in the convergence analysis does not have any characterizations on this part. In fact, the relays may come and go, and their communication graph might not be a complete graph, which the authors have experimented with in their numerical experiments. In fact, they talk about the spectral norm in lines 200 - 203. The reviewer cannot find $\lambda_2$ in Theorem 4.4.
  * It is quite interesting that expectations appear in both the denominators and the numerators in the lower bounds for different $\beta$'s. 
  * In assumption 4.2, the authors assume $\mathbb{E}_{\xi \sim D^{(i)}}[\|\|\nabla f(x, \xi) - \nabla f^{(i)}(x)\|\|^2] \le \sigma^2$. The reviewer is quite confused by this. We don't have a global stochastic gradient.
  * The experiment results are barely readable, in particular, for $-\log (\text{CDR})$.
 * Numerical results:
  * It is quite limited compared only to FL. The authors may also wish to compare with state-of-the-art hierarchical FL methods.

### Questions
* Can the authors justify their aggregator-free approach in the context of relays? I agreed that there is not a fixed central coordinator, but aggregator-free is an overclaim.
* Can the authors discuss in detail the traditional FL, hierarchical FL, and their approach? 
* Can the authors provide more intermediate results with discussions in the main text? Rushing to Theorem 4.4 is too fast.
* Can the authors explain why we have expectations appear both in denominators and numerators?
* Can the authors conduct another round of proofreading and fix the notations and typos?
* Can the authors discuss the technical challenges in their proofs?
* Can the authors provide theoretical results for the fault-tolerant part of the proposed methods, which is their main contribution as compared to traditional FL with only one central server?
* What does $\in {\mathcal{O}}(\cdot)$ mean in line 326 in Theorem 4.4?
* Can the authors illustrate how the spectral norm differs with different graphs?
* Can the authors provide more numerical experiment results as compared to the traditional hierarchical FL?

### Soundness
3

### Presentation
2

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
This paper proposes Tiered Gossip Learning (TGL) for large-scale collaborative learning over edge devices. TGL addresses the limitations of Federated Learning (FL), which suffers from a single point-of-failure (the central server) issue, and peer-to-peer learning (P2PL), which requires high communication degrees to maintain good mixing quality as networks scale. The primary motivation behind the design of TGL is to combine the fault-tolerance and decentralization of P2P systems with the efficiency and scalability of hierarchical aggregation.

TGL introduces a two-tier push-gossip-pull protocol with relay nodes in the upper tier and leaf nodes in the lower tier. Leaves push model updates to a small random subset of relays, relays gossip among themselves, and leaves pull updates from random relays. This coordinator-free architecture spreads aggregation across relays, avoids centralized failure risks, and significantly reduces communication overhead.

The paper proves convergence under standard smoothness, bounded variance, and data heterogeneity assumptions , offering explicit bounds on consensus distance and showing how TGL enables independent control over mixing at each layer. Empirical results on CIFAR-10, FEMNIST, and AG-news show that TGL achieves equal or better accuracy that decentralized baselines while using fewer model exchanges.

### Strengths
The paper tackles a real limitation of both FL and decentralized FL by using hierarchical aggregation. The proposed tiered gossip approach is clean and intuitive -- using relays to offload communication but without introducing a single coordinator.

The authors provide strong theoretical grounding under standard assumptions, along with explicit bounds to consensus contraction across stages. The spectral gap and consensus distance analysis seem sound. Empirical results also show non-negligible gains. The proposed algorithm is also relevantly compared with prior works such as ELL.

### Weaknesses
My major concern is regarding the novelty of the idea of hierarchical aggregation in general. There are several works that have tried addressing this problem, and a more detailed discussion is required over the pros and cons of the proposed TGL with respect to prior works. For example,

1. Client-Edge-Cloud Hierarchical Federated Learning (https://arxiv.org/abs/1905.06641)
2. ColRel: Collaborative Relaying for Federated Learning over Intermittently Connected Networks (https://openreview.net/forum?id=8b0RHdh2Xd0)
3. Multi-Level Local SGD for Heterogeneous Hierarchical Networks (https://arxiv.org/abs/2007.13819)

How do the proposed method compare with prior related works such as the above (and maybe some more).

Some other minor concerns include: 

1. In practice, the 3-stage aggregation procedure may introduce additional latency overhead. The authors discuss this asynchronous behavior  only briefly and is not benchmarked experimentally. Some benchmarks in this regard or a better analytical model that takes into account this relaying latency (depending on the quality of communication channels) will be appreciated.

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
2
