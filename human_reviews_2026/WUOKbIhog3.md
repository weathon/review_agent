# LEAN: Library-Based Adaptation for Continuous, Federated Fine-Tuning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
We consider the problem of learning to adapt a foundation model in a federated setting, particularly the most realistic and general setting: 1) When the local data sets are sampled from different distributions but we want to learn a globally adapted model, 2) Where local agents enter and leave the federation asynchronously at each time tick which is beyond the control of the learning algorithm, and 3) Where the goal is continuous adaptation so that after each time tick, the learned adapter generalizes accurately for all participants that have been seen during training.  We propose a simple idea called federated library-based adaptation (LEAN) for exactly this setting. In library-based adaptation, the system maintains a pool, or "library" of so-called "basis pairs."  Agents entering the federation check out basis pairs, update them, and check them in. Library-based adaptation is designed to avoid problems with more conventional methods, such as those based on averaging. In particular, we demonstrate LEAN outperforms traditional averaging baselines in both communication and computation cost efficiency across a broad range of important settings, including heavy data skewness and high asynchronicity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes LEAN (Library-basEd AdaptatioN), a novel framework for weak federated learning that replaces global parameter averaging with a basis-pair library. Each client checks out several basis pairs ((b, a)) from the server, locally fine-tunes them (similar to LoRA), and checks them back in. During inference, the server reconstructs updates by combining multiple basis pairs.

### Strengths
Novel and Intuitive Design Concept: The idea of exchanging basis pairs instead of averaging entire low-rank adapters is both elegant and practical. It provides an intuitive explanation of why LEAN might better handle heterogeneous data and asynchronous participation, avoiding the identifiability issues of LoRA averaging.

Clear Focus on Weak Federated Settings: The paper’s motivation is well aligned with realistic federated environments—characterized by asynchronous clients, resource constraints, and highly non-IID data. This problem framing is timely and relevant.

### Weaknesses
Writing and Logical Issues: The manuscript contains several grammatical and stylistic errors, “may not be be useful” (double “be”) and “Qwem-3 0.6B” (should be Qwen), which suggests a lack of careful proofreading. 
The text states blue birds have a mean (length, weight, wingspan) of (40, 20, 20) and red birds (20, 40, 40), indicating red birds have greater weight and wingspan. However, Figure 1 shows blue birds with all attributes larger than red birds. Could the authors clarify this discrepancy and revise the text or figure for consistency?

Incomplete Baseline Selection: The experiments only compare with FedAvg, FedProx, and FLoRA, omitting several recent and directly relevant baselines such as FedPEFT, FedPETuning, SLoRA, and FedIT.Since these methods explicitly address parameter-efficient or LoRA-based federated learning, their absence substantially weakens the empirical validity of the claimed superiority of LEAN.

Privacy and Security Risks Not Discussed: The basis pairs ((b, a)) stored and transmitted by the server could leak sensitive information about clients’ local data, potentially enabling membership or attribute inference attacks. Unlike aggregated LoRA weights, these basis pairs persist in the central library and are repeatedly reused, which may exacerbate privacy risks.
The paper lacks any discussion or mitigation strategy (e.g., secure aggregation, differential privacy), which is essential for a federated framework.

### Questions
Please see the weaknesses above.

### Soundness
3

### Presentation
1

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
The paper proposes LEAN, a novel, non-averaging approach for federated fine-tuning in challenging "weak federations" with high asynchronicity and data skew. Instead of aggregating models, clients asynchronously check out, refine, and check in rank-1 "basis pairs" from a central library. This circulation mechanism avoids the instability of averaging-based methods. Strong empirical results on vision and language tasks show LEAN is significantly more robust and communication-efficient than baselines like FLoRA in realistic, heterogeneous settings.

### Strengths
1. Clearly identifies and addresses the practical "weak federation" setting, with a compelling critique of standard averaging-based approaches.
2. The "check out/check in" library model is an elegant and creative departure from standard FL, naturally handling asynchronicity and sidestepping common PEFT pitfalls.
3. Provides comprehensive empirical validation on modern models, systematically demonstrating LEAN's superior robustness to data heterogeneity and asynchronicity.

### Weaknesses
1. No Theoretical Analysis: The paper is purely empirical, lacking any formal convergence guarantees or theoretical justification for why the method is stable and effective. For a top-tier venue like ICLR, the absence of any theoretical justification is a notable weakness.
2. LEAN introduces several new and important hyperparameters, namely the library size n, the checkout size m, and the proximal weight λ. The paper shows that n has a significant impact on performance (Table 1) but provides no systematic study or guidance on how to set n or m (or their ratio m/n) for a new task. A sensitivity analysis would be crucial for understanding the method's robustness and for enabling practical adoption.
3. The final adapted weight matrix is constructed as W + (m/n) * Σ(all basis pairs). If the library size n is large (e.g., 1280 in the experiments, to match the model's rank), the resulting adapter is a sum of many rank-1 matrices, which is not itself a low-rank matrix. This could introduce significant computational overhead (latency) at inference time compared to a standard single low-rank LoRA adapter..

### Questions
1. Can the authors provide any theoretical justification for LEAN's convergence and stability?
2. How sensitive is the method to the choice of library size (n) and checkout size (m), and how should they be set?
3. What is the inference cost of the final summed-pair adapter compared to a standard LoRA adapter?
4. Have the authors considered more intelligent strategies for sampling basis pairs from the library (e.g., based on staleness)?

### Soundness
3

### Presentation
2

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
This paper addresses federated adaptation of foundation models under realistic conditions involving heterogeneous data, asynchronous participation, and continuous adaptation. The authors propose Federated Library-Based Adaptation (LEAN), which maintains a shared library of adaptable basis pairs that agents update collaboratively. Experiments show that LEAN outperforms traditional averaging-based methods in both communication and computation efficiency under diverse and challenging federated learning scenarios.

### Strengths
1. Addressing continuous federated learning for foundation models through a library-based adaptation method is a novel and promising direction.

2. The paper is well organized and clearly written, making it easy to follow and understand.

3. The evaluation is comprehensive and effectively demonstrates the proposed method’s effectiveness across diverse federated learning settings.

### Weaknesses
1. The contribution of this paper to the federated learning community appears limited.

2. The scalability of the proposed method to larger-scale models (e.g., Qwen3-8B/32B) should be evaluated to make the experimental analysis more complete and convincing.

3. Additional discussion and empirical evaluation on the adaptability of the proposed method to multimodal foundation models would further enhance the contribution and broaden the applicability of this work.

### Questions
Please refer to the weaknesses discussed in the previous section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes LEAN, a library-based alternative to FLoRA for federated LoRA fine-tuning. Instead of stacking per-client adapters, LEAN keeps a shared global pool of rank-1 “atoms” (rows) on the server; clients check out a small set of indices (their rank budget, perhaps), update only those rows, then check in. The server materializes the model by summing all library outer products. The author claims that LEAN avoids row-wise averaging (the identifiability pitfall), supports native asynchrony (no round barriers), and is more robust under weak federations (heavy non-IID + asynchrony) with a proximal regularizer on local updates.

### Strengths
- "Identifiability" is an interesting problem framing. 
- The framework is straightforward and practical to implement.
- Compared with unbounded stacking, the approach offers more predictable communication cost.
- In several weak-federation regimes, LEAN outperforms—or is more stable than—the stacking baseline.
- Code is released (not yet reviewed in depth on my side).

### Weaknesses
- Overclaiming & limited novelty. In my view, LEAN collapses to FLoRA-style stacking whenever indices aren’t reused; so its edge really rides on reuse (for comms), a prox term, and native async. That feels like a modest step beyond FLoRA rather than a substantive new paradigm.

- Motivation not closed. The paper says stacking may falter in weak federations because “many low-rank models go in different directions.” But if LEAN degenerates to stacking without reuse, I don’t see a clear mechanistic reason it would converge better; the argument isn’t theoretically supported. 

- Sequential drift worries (my biggest technical concern). With reuse, clients sequentially edit the same row, inviting tug-of-war/negative transfer. There’s no reuse vs. no-reuse ablation, no routing study (random vs. similarity-aware), and no conflict diagnostics (e.g., cosine of successive updates). Also, there’s no theory on stability or convergence of the library dynamics.

- Missing “FLoRA + prox”. Since the proximal term clearly stabilizes LEAN, part of the gain could just be prox, not the library.

- FLoRA doesn’t ship dense full-rank  each round; it stacks LoRA factors, so cost scales with total rank. The paper’s narrative overplays FLoRA’s communication burden.

### Questions
1. Compare no-reuse (clone new rows → stacking) vs random reuse vs similarity-aware reuse at matched bytes/rank. Does reuse reduce variance or induce drift?

2. Add these baselines. If FLoRA gets a proximal term and/or runs without barriers, do LEAN’s advantages persist?

3. For reused rows, report cosine similarity between successive client updates, the frequency/extent of “row tug-of-war,” and how the prox weight modulates it.

4. Sweep library size n and inference pair count under fixed communication budgets; where is the sweet spot vs stacking?

5. Does assigning similar clients to the same indices help (or overfit)? Any aging/eviction to prevent stale or monopolized rows?

6.  Can clients use their recently edited rows at inference for personalization? What’s the privacy story for shared atoms (DP noise, secure aggregation, leakage tests)?

7. Larger models and datasets?

### Soundness
2

### Presentation
2

### Contribution
3
