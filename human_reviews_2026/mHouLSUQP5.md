# MuonBP: Faster Muon via Block-Periodic Orthogonalization

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 2, 8, 6, 6

## Abstract
Gradient orthogonalization is a simple strategy that shows great utility in speeding up gradient descent. The Muon optimizer (Keller et al., 2024) combines gradient orthogonalization with first-order momentum and achieves significant improvement in data efficiency over Adam/AdamW (Loshchilov & Hutter, 2019) for language model training. However, when using model parallelism, gradient orthogonalization introduces additional overhead compared to coordinate-wise optimizers (such as AdamW) due to additional gather and scatter operations on gradient matrix shards from different devices. This additional communication can amount to a throughput hit of 5%-10% compared to Adam/AdamW. To remedy this, we propose Muon with Block-Periodic Orthogonalization (MuonBP), which applies orthogonalization independently to matrix shards on each device and periodically performs full orthogonalization to maintain training stability at scale. We show how to adjust the learning rate from the baseline to MuonBP and give convergence guarantees for this algorithm. Crucially, our theory dictates that we use two stepsizes: one for the blockwise orthogonalization steps, and one for the full orthogonalization steps. Our method is simple, requires minimal hyperparameter adjustments, and achieves competitive iteration complexity compared with baseline Muon while providing per-iteration throughput comparable to coordinate-wise methods such as AdamW. When training an 8B model with eight-way tensor parallelism and ZeRO optimizer state sharding, MuonBP achieves 8% throughput increase compared to Muon with no degradation in performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposed MuonBP, which basically trying to solve BlockMuon's loss divergence issue by adding some global synchronization via extra communication and adjust learning rate accordingly. the end results shows MuonBP loss curve getting lower than BlockMuon.

### Strengths
1.adding some global sync on top of BlockMuon to make loss curve converge better is a contribution.

2.PPL analysis is detailed and reasonable.

### Weaknesses
1. Research novelty is minor, 

2. Comparing loss(PPL) with running time is misleading. By looking at figure 3, it is hard to see that baseline Muon trains less iteration thus loss curve is higher than proposed MuonBP. If the author want to show the iteration time, they can create a table on this. In model training, the most important thing is model quality, which means after training same amount of Iterations (Not same amount of time), which model converges to the lowest PPL or loss values. The second priority is iteration time. 

3. The whole paper does not even include how many iterations each training is in their experiments, which is quite concerning. 

4. The authors lack of basic distributed training knowledge, for example L147-148, " If we use TP or FSDP2, we have to
do an additional all-gather across the TP/FSDP2 groups to gather the model parameters", which is Wrong. TP never gather model parameters.

### Questions
The whole paper uses fsdp and zero interchangeably, which makes it quite confusing. What zero stage is indeed used? zero 1,2,3. (please describe this more important and delete some unnecessary details like row-wise split, which is the by-default setting)

Also sometime the experiment says using Megatron? how could megatron-lm used together with simpleFSDP? they don't have the synchronization implemented and tested between TP and this DP.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes MuonBP, a communication-efficient variant of the Muon optimizer that combines block-wise orthogonalization with periodic full orthogonalization. It is a good attempt to scale Muon to modern training setup, e.g., model parallelism. The authors provide theoretical convergence guarantees, showing that MuonBP interpolates between BlockMuon (fully local) and Muon (fully global), delivering 8% throughput increase compared to Muon with no degradation in performance.

### Strengths
* **Clear motivation and practical importance.** The work directly addresses a key limitation of Muon, its poor scalability under model parallelism, by reducing communication overhead. It is a big issue for Muon as I know.
* **Theoretical grounding** The authors rigorously gives a bound for the propose method.
* **Realistic pretraining setting evaluation**. Experiments on multiple setups (FSDP2 + TP, ZeRO + TP, and scales from 160M to 8B parameters) demonstrate that MuonBP recovers Muon’s data efficiency with significantly better wall-clock performance.
* **Clear and good writing**: the paper is clear, well-motivated, and easy-to-follow.

### Weaknesses
Overall, it is a good paper that solves a real problem. I only have a few questions: 
* **Exploration beyond pretraining**. All experiments focus on pretraining, I am curious whether the author can envision the performance on SFT or even RL work.
* **How to adaptively tune the P**: The Author mentions that we might adaptively tune it based on observed properties. I wonder whether the author has some insight on how to monitor, especially in the practical training setting with multi-paralelesim

### Questions
n/a

### Soundness
4

### Presentation
3

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
The paper introduces MuonBP, a variant of Muon that replaces per-step global gradient orthogonalization with block-local orthogonalization aligned to model-parallel shards, and performs a periodic full orthogonalization every P steps. This preserves Muon’s data efficiency while removing most cross-device communication on off-period steps. The authors (i) formalize MuonBP and its communication model, (ii) prove a convergence rate that interpolates between Muon and fully blocked Muon and shows two distinct step sizes (for block vs full steps) are theoretically preferable, and (iii) demonstrate up to 8% throughput gains at 8B scale with comparable or slightly better perplexity vs Muon, while BlockMuon is unstable or worse at scale.

### Strengths
The Originality, quality and signifance are good because of the following details:
1. Clear motivation with concrete bottleneck: Muon’s additional all-gathers cost 5–10% throughput vs AdamW under model parallelism, which MuonBP directly targets.
2. Simple, practical algorithmic knob: Period P trades fewer collectives for slighly weaker per-iteration conditioning and it aligns blocks to TP/FSDP shards to eliminate extra communications on off-period steps.
3. Compelling empirical evidence: At 8B, 8% throughput gain vs Muon with no loss in perplexity. BlockMuon underperforms/unstable at higher LRs, validating the need for periodic global steps.
4. Wall-clock wins: For fixed perplexity targets, MuonBP reaches them 10–13% faster, and for fixed time budgets it achieves 5–7% lower perplexity.

The clarity is great because its presentation is easy to follow. The figures are intuitive.

### Weaknesses
1. Limited exploration of P scheduling. The paper largely uses a fixed P (often 5). It will be better to add ablations with adaptive P and report wall-clock to target perplexity.

### Questions
Is there any rule of thumb regarding how to choose a good value of P?

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
This paper addresses a key performance bottleneck in the Muon optimizer, which suffers from reduced throughput in model-parallel training due to the communication overhead of its gradient orthogonalization step. The authors propose Muon with Block-Periodic Orthogonalization (MuonBP), a variant that eliminates this overhead on most steps by performing orthogonalization independently on the local gradient shards on each device. To maintain the stability and convergence of standard Muon, a full, cross-device gradient orthogonalization is performed only periodically. The authors provide theoretical analysis justifying this periodic approach and the use of two distinct learning rates for block and full steps. Experiments on an 8B parameter model demonstrate that MuonBP matches Muon's performance while increasing throughput by 8%, effectively closing the throughput gap with coordinate-wise optimizers like AdamW.

### Strengths
1. The paper provides a solid theoretical analysis of the proposed MuonBP algorithm. The convergence guarantees (Theorem 2) insightfully connect the block-periodic approach to the two extremes (BlockMuon and standard Muon) and justify the novel use of two distinct learning rates, which adds rigor to the method.

2. The experimental validation is large-scale, convincing, and directly addresses the practical problem of throughput. By testing on models up to 8B parameters with 8-way tensor parallelism, the authors convincingly demonstrate that MuonBP achieves its goal of improving wall-clock throughput (by ~8%) without sacrificing the final model performance.

### Weaknesses
1. The core idea of using local updates to reduce communication is a well-studied direction in communication-efficient distributed training. While applying this idea to the Muon optimizer and achieving impressive experimental results is valuable, the concept itself is not entirely novel. A more fundamental solution to this problem would involve improving the all-gather operation itself to decouple the Newton-Schulz iteration from the communication, but this is admittedly a very difficult problem. Therefore, the paper's overall contribution is still commendable.

### Questions
1. I am curious about the key phenomenon in the experiments that explains why MuonBP is faster (in terms of wall-clock time, as seen in Figure 3) than standard Muon. Intuitively, MuonBP performs more local updates where the parameters are exposed to less global information (only the local shard gradient), which suggests it should not converge faster per step than standard Muon, which sees the full gradient. Could the authors elaborate on this surprising result?

### Soundness
3

### Presentation
3

### Contribution
3
