# Learning the Quadratic Assignment Problem with Warm-Started MCMC Finetuning and Cross-Graph Attention

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
The quadratic assignment problem (QAP) is a fundamental NP-hard task that poses significant challenges for both traditional heuristics and modern learning-based solvers. Current neural solvers for the QAP often struggle with poor scalability or limited flexibility when dealing with the coupled two-graph structure, and they underperform on real-world instances.  We propose PLMA, an innovative permutation learning framework to bridge this performance gap. PLMA features an efficient warm-started MCMC finetuning procedure to enhance deployment-time performance, leveraging short Markov chains to anchor the adaptation to the promising regions previously explored. For rapid exploration via MCMC, we design an additive energy-based model (EBM) over the permutation space, which enables an $O(1)$-time 2-swap Metropolis-Hastings sampling step.  Moreover, the network used to parameterize the EBM incorporates a cross-graph attention mechanism that directly models the coupled graph structure of the QAP, ensuring scalability and flexibility. Extensive experiments demonstrate the consistent superiority of PLMA over stat-of-the-art baseline methods across various benchmarks, highlighted by a near-zero average optimality gap on QAPLIB and remarkably superior robustness on the notoriously difficult Taixxeyy instances.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposed a probabilistic framework for learning QAP problems, which is proved in the paper that is equivalent to the original deterministic model. For the modeling of the distributions, the authors proposed an instance-based model that takes an instance as input and outputs the parameterized distribution of the assignment heatmap matrix. The model architecture is a GNN based on two graphs of the problem, with attention mechanism across the graphs. This framework thus allows pretraining and deployment time finetuning.

### Strengths
- Overall, the paper is well written, and the overall framework is well motivated. 
- The probabilistic approach for NP-hard QAP problems is novel and prospective. 
- The MCMC sampling is well designed and is a good approach for intractable distributions.
- The empirical results are promising in terms of both time and gap, as well as generalization on larger problems.

### Weaknesses
- I suggest adding an overall illustration of the framework, so that the concept is more straightforward on first seeing it. 
- Some concepts and details are not well explained to me, see my questions below.

### Questions
- For learning based methods on CO problems, is it common to do deployment time finetuning on instances?
- In equation 6, I still don't understand why $||\phi||_F^2$ term can be eliminated. Isn't it the output of the neural networks, and should be considered as well?
- As far as I understand, the node embeddings $h_{ini}$ are the same for all nodes? And the distinction across instances are in $D$ and $F$? But what exactly are $D$ and $F$? I think it is important but i didn't find where $D$ and $F$ are defined. Besides, how do I interpret $\bar{D}$ and $\bar{F}$?
- In equation 12, why do we need $C$ term and how is it tuned?
- In line 253: "This structure actually yields a low-rank approximation...." why is it low rank? 
- In e.g. Table 1, what is the reference solution based on which the gap is calculated?
- In Figure 1, we can see the ablation that without the attention it is worse, but what is the time overhead with the attention? If it is a lot of extra time then maybe removing the attention may also be acceptable. 
- In training as well finetuning, what is intuitively the $\hat{G}$? And how is the gradient $\nabla_{\theta} \Phi_{\theta}$ obtained?
- Finally, could you explain more on the pushforward transformation design? I can see it is to smooth the loss, but I don't get it how it is done. And it is also confusing what is conveyed in appendix B.1.

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
4

### Summary
The authors present a QAP solver based on the following. Pretraining minimizes the expected post-improvement cost via a pushforward map using a prior model parameterized as an energy-based model (EBM).  Fine-tuning starts from the pretrained model, runs the pushforward map, updates the EBM parameters, and returns the best solution found. The energy of a permutation is computed from a compatibility heatmap parameterized by a cross-graph attention network.

### Strengths
Well-structured paper with extensive benchmarks and clear awareness of recent developments in neuro optimization. Attention and graph-matching networks are not new per se; the novelty lies in their combination with the EBM and sampling framework. Main achievement seems to be competitive performance against a solid (although not champion) heuristic algorithm and better performance than previously proposed in its subfield.

### Weaknesses
Novelty is limited beyond the combination of existing methods. Benchmarking is narrow with respect to classical heuristics (Ro-TS alone may not suffice). Ablation studies are not entirely clear. Improvements appear incremental, and it is hard to assess their significance. Limited analysis of the contribution of pretraining.

### Questions
1) On what hardware was Ro-TS run? Were all runs performed on GPU? Was Ro-TS executed on CPU? Since the reported gap is similar to Ro-TS, runtime comparisons are only meaningful relative to it.

2) Are there stronger baselines (e.g., ITS, ILS)?
See for example:
- Misevičius, A. Letter: New best known solution for the most difficult QAP instance “tai100a.” Memetic Computing 11, 331–332 (2019).
- Misevičius, A. An implementation of the iterated tabu search algorithm for the quadratic assignment problem. OR Spectrum 34(3): 665–690 (2012).
- Hussin, M. S., & Stützle, T. Hierarchical iterated local search for the quadratic assignment problem. Hybrid Metaheuristics Workshop, Springer, 2009.

3) What are the “diverse instances” used for pretraining? Specify distributions. Any overlap with test families?

4) Why is the ablation study conducted only on the uniform dataset? No error bars are provided so the relative differences are not meaningful.

5) What about memory overhead of the proposed method?

6) Emphasizing time to reach the same or better gap as SOTA would be clearer? Mixed tables of gap and runtime are harder to interpret. Time-to-target (TTT) metrics are standard and more informative.

7) The key novelty is learning-based improvement. How much pretraining helps compared to classical heuristics by examining generalization for in-distribution and out-of-distribution instances?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces ``PLMA``, which combines ``EBM`` with a Cross-Graph Attention mechanism to learn ``QAPs``. Additionally, it incorporates a Warm-started Batched MCMC Finetuning mechanism during inference, enabling the model to efficiently adapt and optimize for specific instances after pretraining.

### Strengths
1. The idea of integrating energy-based modeling with ``MCMC`` for permutation learning is conceptually elegant and grounded in statistical physics principles. It provides a bridge between learning-based prediction and search-based optimization.

2. The dual-graph attention encoder effectively models inter-graph relations without constructing a dense association graph, which improves both interpretability and efficiency.

3. Experiments demonstrate strong empirical results on ``QAPLIB`` and ``Taixxeyy`` datasets, showing significant improvements in both optimality gap and runtime compared to classical metaheuristics.

### Weaknesses
1. It appears that the post-processing technique 2-swap is integrated into the ``Batched Warm-started MCMC Finetuning``, and the performance gains largely stem from this post-processing. Existing methods could also adopt the same technique, but the authors do not provide corresponding experimental data.

2. Previous studies have proposed similar approaches, such as iSCO[1] and RLSA[2], which also employed ``MCMC`` methods. However, the authors did not provide a comparative analysis with these works. Moreover, applying this technique to problems with complex constraints like ``CVRP`` remains challenging, and the current framework does not seem to offer a solution to this issue. 

[1] *Revisiting Sampling for Combinatorial Optimization, ICML2023*

[2] *Regularized Langevin Dynamics for Combinatorial Optimization*

### Questions
See ``Weakness``.

I would consider raising the score if the authors could demonstrate the application of their method to broader classes of permutation-based problems, such as ``CVRP``.

### Soundness
2

### Presentation
3

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
The paper introduces PLMA, a novel permutation learning framework for solving the Quadratic Assignment Problem (QAP), a well-known NP-hard combinatorial optimization task. The authors identify that existing neural solvers for QAP either suffer from poor scalability or lack flexibility, leading to suboptimal performance on real-world instances. PLMA addresses these issues through a two-stage learning paradigm: pre-training on diverse instances to learn general structural features, followed by a highly efficient, deployment-time finetuning stage.


Warm-Started MCMC Finetuning: A procedure that adapts the pre-trained model to specific test instances. It uses short, parallel Markov chains initialized from high-quality solutions found in previous steps, focusing the search on promising regions of the solution space.

Efficient Energy-Based Model (EBM): An additive EBM is designed over the permutation space, which allows for a constant-time, O(1), evaluation of 2-swap proposals in a Metropolis-Hastings sampler. This makes the MCMC exploration phase remarkably fast.

Cross-Graph Attention Network: A  neural network that models the two-graph (flow and distance) structure of the QAP. It separately encodes each graph and then uses a cross-attention mechanism to fuse information, avoiding the need for a large, computationally expensive association graph.

 Experiments: Demonstrates significantly better robustness than strong heuristics on the challenging Taixxeyy instances and QAPLIB benchmark.

### Strengths
State-of-the-Art Performance: Achieves a near-zero (0.10%) gap on the QAPLIB benchmark, outpacing heuristics in speed. It also shows exceptional robustness on difficult Taixxeyy instances where traditional methods fail.

 A novel two-stage learning process uses warm-started MCMC finetuning to avoid restarting searches, leading to faster convergence and superior solutions.

Thorough Validation: The design is rigorously supported by detailed ablation studies that confirm the importance of each core component.

### Weaknesses
Cross-Distribution Generalization: The model is pre-trained on synthetic instances (geometrically structured or uniform random) and then applied to real-world benchmarks like QAPLIB and Taixxeyy. It's unclear how much meaningful, transferable knowledge is actually carried over versus how much is rediscovered during adaptation.

It would be good to understand how much are the train and test instances structurally similar dissimilar. When does the model perform better and when it does not?

### Questions
1. Impact of pre-trained vs random policy? Currently the evaluation is based upon a pre-trained policy. Is there a study on impact of finetuning search on random policy? The paper does not provide a baseline (e.g., starting the finetuning from a random policy) to prove that the learned priors from pre-training are essential. Correct me if I am wrong. What kind of pre-training helps?


2. Could the authors clarify how were the baselines trained/tuned?

### Soundness
2

### Presentation
3

### Contribution
2
