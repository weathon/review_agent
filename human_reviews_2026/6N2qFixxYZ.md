# DES-LOC: Desynced Low Communication Adaptive Optimizers for Foundation Models

- Decision: Accept (Poster)
- Scores: 8, 8, 4, 4

## Abstract
Scaling foundation model training with Distributed Data Parallel~(DDP) methods is bandwidth-limited.
Existing infrequent communication methods like Local SGD were designed to synchronize model parameters only and cannot be trivially applied to adaptive optimizers due to additional optimizer states.
Heuristic approaches that keep states local or reset them lack guarantees and can be unstable in compute‑efficient batch regimes; conversely, Local Adam synchronizes all states uniformly and is provably convergent but triples communication costs.
We propose Desynced Low Communication Adaptive Optimizers (DES-LOC), a family of optimizers assigning independent synchronization periods to parameters and momenta, enabling lower communication costs while preserving convergence. Our theoretical analysis shows that while parameter synchronization dominates the asymptotic rate in-expectation, high-probability convergence guarantees require at least infrequent synchronization of the second momentum. Furthermore, we prove that more frequent momentum sync permits larger stable step sizes. Experiments on language models of up to 1.7B show that DES-LOC can communicate 170x less than DDP and 2x less than the previous state-of-the-art Local Adam, enabling 1.3x–2.1x wall‑clock speedups over DDP for 1-13B models on 100Gb/s links. Furthermore, unlike previous heuristic methods, DES-LOC is robust to worker failures offering a scalable, efficient, and fault-tolerant solution for foundation model training.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes DES-LOC, a family of desynced low-communication adaptive optimizers that assign independent synchronization periods to model parameters and optimizer momenta (first/second). Theoretical analysis shows (i) convergence for SGDM in non-convex settings and for Adam in weakly convex settings, and (ii) that more frequent momentum sync allows larger stable step sizes, while high-probability bounds require at least infrequent second-moment sync. Empirically (GPT-style LMs up to 1.7B parameters), DES-LOC achieves 170× less communication vs DDP and 2× less vs Local Adam, with 1.3–2.1× wall-clock speedups (aided by a wall-clock model for larger scales) and shows robustness to worker failures.

### Strengths
1. Originality. Decoupling sync periods for parameters and momenta, grounded in a half-life/step-size argument, moves beyond Local Adam’s uniform sync and earlier heuristics that lacked guarantees or failed under failures.
2. Quality. Convergence under standard assumptions with a clear theorem for SGDM and high-probability Adam bounds. Analysis explains why more frequent momentum sync permits larger stable steps and why $\beta_2$ cannot be left unsynced indefinitely.
3. Quality (empirics). Solid experimental design with explicit RQs, ablations over $K_x, K_u, K_v$ and comparisons showing DES-LOC matches Local Adam’s perplexity with 2× fewer state communications and drastically less than DDP.
4. Clarity. Clear algorithm block and figures.
5. Significance. Communication reductions (170× vs DDP; 2× vs Local Adam) and modeled wall-clock speedups directly target a primary FM training bottleneck (bandwidth), with fault-tolerance a notable practical plus.

### Weaknesses
There is no evidence for a larger-scale model.

### Questions
Could you add a $\ge$7B measured run?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces DES-LOC, an optimization algorithm designed for decentralized training of foundation models, with a particular focus on large language models. The main objective is to improve communication efficiency during large-scale training in order to reduce wall-clock time.
The proposed method builds upon existing approaches such as FedAvg, FedOpt, and more specifically LocalAdam, where workers perform local parameter updates before averaging the weights with a central server. This approach contrasts with classical Distributed Data Parallel (DDP) methods, where synchronization occurs at every training step.
DES-LOC modifies the LocalAdam algorithm by decoupling the synchronization frequencies of the parameters and the optimizer states. While LocalAdam synchronizes both every K steps, this can lead to unnecessary communication overhead. The authors argue, and empirically demonstrate, that the first and second moment estimates (i.e., the momentum terms) evolve more slowly than the model parameters themselves, allowing for less frequent synchronization of these states.
In practice, the authors recommend using default ratios of Ku = 3Kx and Kv = 6Kx, or alternatively determining these based on the half-lives of β₁ and β₂. The paper also provides a convergence guarantee under standard assumptions commonly accepted in the optimization community.
Extensive experiments on LLM training show that DES-LOC achieves performance comparable to LocalAdam while reducing communication costs by approximately half. Overall, the paper presents a well-motivated and empirically validated contribution to improving efficiency in decentralized large-scale model training.

### Strengths
- The paper is well motivated, and the overall writing is clear and easy to follow. The analysis based on the half-life times provides useful intuition for why less frequent parameter updates do not negatively affect model training.
- I particularly appreciated the qualitative analysis of the upper bound on model convergence (Theorem 1), which effectively shows that DES-LOC enables the use of a larger learning rate in practice.
- The authors’ claims are well supported by extensive experiments, including analyses of rate of change, convergence behavior in practical LLM training scenarios, and full model training comparisons (in wall-clock days) with DDP and other baselines, which are especially compelling.

### Weaknesses
- While the method is supported by both theoretical analysis and experimental results, its practical novelty is somewhat limited. The main contribution lies in decoupling parameter updates from optimizer updates, which constitutes a relatively minor modification of the existing LocalAdam algorithm.
- The authors did not conduct experiments under very low-bandwidth conditions. Demonstrating the method’s effectiveness in scenarios with limited network capacity (e.g., 1 Gb/s or 10 Gb/s) would further strengthen its practical relevance.
- The title of the paper refers to “foundation models,” but the large-scale experiments focus exclusively on LLM training. Although these results are already impressive, the authors could either adjust the paper’s scope to explicitly center on LLMs or include an additional experiment involving the training of a vision foundation model to better match the stated scope.

### Questions
- What would be the practical relevance of DES-LOC for training scenarios where workers are connected through significantly lower-bandwidth networks?
- Could the proposed desynchronization strategy also be applied to optimizers that use only a single momentum term (e.g., Muon)?

### Soundness
3

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
4

### Summary
The authors propose DES-LOC, a communication-efficient optimization algorithm that reduces the frequency of optimizer state communication compared to local Adam. The authors prove the convergence of DES-LOC in the non-convex case (for SGDM) and for Adam under weakly convex objectives. The author’s key insight over Local Adam is that the first and second moments can be synchronized at a rate on the order of their half-lives, much less frequently than their parameters need to be synchronized. In experiments pre-training language models, the authors show that significantly less than DDP and up to 2x less than Local Adam while achieving similar final performance.

### Strengths
- The paper is well-written and easy to understand. 
- I think experimental settings are well structured.
- From the math in the main paper, the convergence guarantees look sound (I did not check the appendix). 
- The idea of synchronizing optimizer states relative to their decay half-lives is intuitive to me, and I think it's an interesting direction of inquiry for the distributed optimization community.

### Weaknesses
- My main concern is the lack of comparison to existing methods such as DiLoCo [1] and MuLoCo [2]. Neither method requires synchronizing the optimizer states; therefore, they are trivially more communication-efficient than DES-LOC and have been shown to perform competitively to DDP in practice. I am aware of your experiments in Figure 6 showing a DES-LOC method with Nesterov momentum, but from my understanding, these still synchronize optimizer states. 
- Following from my concern above, since the communication efficiency benefits of DES-LOC are already realized by proposed methods in the literature [1,2], I’m not sure how strong the contribution is. Could the authors comment on this?
- I wonder if synchronizing the optimizer states in DES-LOC can provide a benefit beyond [1,2]. I would be surprised if it does not, and I believe experiments showing this would greatly strengthen your results.


[1][DiLoCo: Distributed Low-Communication Training of Language Models]

[2][MuLoCo: Muon is a practical inner optimizer for DiLoCo]

### Questions
- Are the comparisons to the DDP baseline FLOP-matched?
- Figure 3 (a), (b), (d) have the same rectangular pattern in the loss curve. Why is this the case?

**Suggestions**:
- I have trouble scrolling through your .pdf because the figures take long to load. Perhaps subsampling the point used for plotting can help.
- I have trouble distinguishing different loss curves in your plots. Perhaps smoothing could help.

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
4

### Summary
The paper provides a theoretical argument to synchronize local optimizer states (first and second moments) for fedavg/diloco. Specifically, it shows local momentum needs to be synced; however, it can be infrequent wrt outer updates, and this frequency depends on the momentum coefficients. It improves over vanilla diloco and reduces communication wrt syncing states in every outerstep.

### Strengths
1. The theory-backed argument to have infrequent momentum synchronization is insightful and makes sense. The connection to beta (momentum coefficient) and the synchronization frequency is intuitive. 
2. The results show the benefit of synchronizing momemtum terms with two optimizers and show no degradation wrt to local Adam.
3. Writing is very clear, and the toy example illustrated the need for momentum synchronization.

### Weaknesses
1. The main theorem is for SGD+momentum, however, the method is proposed for adaptive optimizers. This is a limitation as adaptive optimisers are popular and the frequency of synchronization is derived from the theory for SGDM and applied to adam/adopt.
2. The contribution over local Adam is marginal as local Adam already established that optimizer states needs to be synchronized and also connected the convergence bound to momentum coefficients. Please clarify the main contributions.
3. Even with momentum sync, the results are worse than DDP. This raises a question of whether diloco-type methods (weight averaging method) would match DDP (gradient averaging)?

### Questions
1. The method doesn't seem specific to adaptive methods. Would it be better to frame it more generally? Would it work with muon-type optimizers?
2. Is it correct to say local SGD (ie, fedavg/diloco) is sharing only model parameters? It actually shared the parameter differences (pseudo gradients) and uses them with an outer optimizer. Would it make sense to frame the method as parameter-sharing methods need optimizer states to be synced rather than relating them to diloco type methods?

### Soundness
3

### Presentation
4

### Contribution
2
