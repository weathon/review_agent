# FedKLS: Federated KL-Driven Low-rank SVD Adaptation in Non-IID Data Distributions

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Federated learning faces two key challenges: handling non-IID client distributions and reducing communication costs in adapting large models. To address these issues, we propose FedKLS, a framework that combines KL-divergence-based personalization with low-rank SVD-based adaptations. FedKLS chooses spectral components in a dynamical manner by mapping the heterogeneity of client distribution to the singular value spectrum, then builds specialized LoRA-style adapters, which allow aggregation at scale and client-specific specialization. Extensive experiments on 20NewsGroup and Banking77 with DistilBERT and Qwen backbones show that FedKLS achieves competitive performance compared to state-of-the-art parameter-efficient fine-tuning baselines, including LoRA, PiSSA, MiLoRA, and full fine-tuning. In highly non-IID settings ($\alpha = 0.01$), FedKLS improves F1-score by up to 11--12\% and reduces total communication cost by about 3x over PiSSA, achieving the best trade-off between personalization and scalability. These results demonstrate the effectiveness of KL-guided spectral adaptation in federated fine-tuning of large models. Our implementation is available at: https://anonymous.4open.science/r/FedKLS.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a new framework named FedKLS to address several challenges regarding training models in federated learning with non-IID data and high communication costs by combining KL-divergence-based personalization with low-rank SVD adaptations. 

On a few benchmarks, experiments demonstrate that FedKLS significantly improves F1-scores in highly non-IID conditions and reduces communication costs compared to state-of-the-art methods.

### Strengths
The paper addresses an important problem of training models using highly non-iid data in Federated Learning systems.

In several results, the proposed framework outperforms a few other baselines.

Convergence analysis of the proposed method was explored, briefly.

A link to the code repo was provided. (I could not access the code, probably, due to some errors on the webpage of the repo).

### Weaknesses
The notation is confusing and redundant. It would be more clear to give Table 1 in the beginning of the paper, and then fix the notation used in the text and figures, accordingly.

There are a few major issues with the paper.

In half of the results, the proposed FedKLS outperforms the baseline with less cost. However, the remaining results, FedKLS underperforms the baseline with more cost. To support the superiority of FedKLS, additional analyses with various models and benchmarks should be given.

Also, the proposed convergence analysis only explores the bound on the gradient. Additional results on the convergence of optimized parameters should be given.

### Questions
Have you compared convergence rate of your proposed method and the other LLMs/VLMs on additional benchmarks?

In some results, FedKLS underperforms the baseline with more cost. Is this due to slow convergence or divergence of the models in those particular setups?

Could you please provide learning curves in the analyses?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
The paper proposes FedKLS, a federated PEFT method that (i) computes each client’s divergence from an IID reference using KL(P||Q) over label distributions, (ii) maps the normalized KL to a contiguous slice of the global SVD spectrum per layer (principal <-> generalization, minor <-> personalization), and (iii) trains/aggregates LoRA-style adapters (A,B) while freezing residual weights. On 20News, Banking77, and CIFAR100 (ResNet18), the authors report higher F1 and lower communication vs. LoRA, PiSSA, MiLoRA, and sometimes FFT, with the largest gains at strong heterogeneity.

### Strengths
* **Simple, intuitive personalization signal.** Using *one scalar per client* (KL divergence) to choose a spectral subspace is a neat bridge from distribution skew → representation subspace.
* **Communication-aware design.** Communicating only ((A,B)) is standard for PEFT, and the paper focuses evaluations on rounds, time, and GB, which is appropriate in FL.
* **Targeted hypothesis.** The principal/minor spectrum ↔ generalization/personalization hypothesis is well-motivated and aligns with prior SVD-based PEFT observations.
* **Scalability experiments.** Includes client-scaling and non-IID sweeps; attempts to measure practical costs (rounds/time/GB), not just accuracy.

### Weaknesses
1. Aggregation correctness is under-justified.

   Clients are initialized from *different spectral slices*, yet the server averages raw $(A, B)$ across clients. The appendix asserts this is “compatible” because shapes match and $W_{\text{res}}$ is frozen, but that does not resolve potential *directional misalignment* across adapters initialized from orthogonal subspaces; simple averaging can cancel useful updates. A principled fix would align in a shared basis (e.g., project $A_i B_i^\top$ onto the SVD basis and average block-wise), or perform weight-space aggregation of $\Delta W$ after basis alignment.

2. Convergence section has a likely inequality error and hand-wavy terms.

   The paper defines $\Delta_{\text{SVD}}$ as the best rank-$r$ truncation error, then claims FedKLS “further selects subspaces” with a bound

   $$
   \Delta_{\text{SVD}} \le \gamma \sum_\ell \lVert W_\ell - W_\ell^{(r)} \rVert_F^2
   $$

   with $\gamma \in [0,1]$. This implies smaller error than the best rank-$r$ approximation, which is generally impossible. Probably the direction should be $\ge$ (or a *new* error term should be introduced). The added $\Delta_{\text{KL}}$ term is not characterized (no dependence on $r, R, K, \alpha$, etc.), making the bound non-actionable.

3. Method specification gaps.

   * Which layers receive adapters? (attn Q/K/V/O? MLP in/out? convs in ResNet?) What rank per layer, scaling $\alpha$, initialization variance, and which dtype for comms?
   * Is the SVD computed once on the (frozen) global $W$ at $t{=}0$, or per round? If $W_{\text{res}}$ is frozen forever, say so; otherwise, re-SVD should be discussed.
   * The mapping uses a contiguous slice $[{\rm start}_i:{\rm end}_i]$. Why contiguous versus, e.g., temperature-weighted sampling or mixtures? Is there enforced overlap between KL-nearby clients?

4. Label-only KL & privacy/fairness.

   * Using uniform $Q(c)=1/C$ biases the mapping when the global distribution is non-uniform; a reference aggregated (privately) or public proxy would be more faithful.
   * Even a scalar may leak info about skewed clients; privacy accounting (e.g., DP noise on the scalar) is not addressed.
   * Assigning high-KL clients to minor components could starve them of high-energy directions; fairness across clients (variance of F1) is not reported.

5. Baselines selection.

   Comparison is only to centralized PEFT baselines used in FL. The paper omits personalization baselines (Ditto, pFedMe, FedPer/FedRep, pFedHN, MOON, FedBN, FedProx/Scaffold variants with personalization heads), which would better test whether spectral routing adds value beyond standard personalized FL.

6. Metric choices and reporting.

   The text says “maximum client-side F1 (averaged across clients) over 1000 rounds”. Reporting best-over-time is optimistic; please include last-round, AUC over rounds, std over seeds, and per-client distribution (median/IQR). Communication cost (GB) must define formula (uplink+downlink? optimizer states? compression? dtype?) and participation sampling.

7. Clarity issues / small technical slips.

   In Fig. 2 Step 1, “local data is represented by $U_r$ and $\Sigma_r$” is inaccurate—those are singular vectors/values of weights, not the data. Several figure captions/text passages blur data vs. weight SVD.

### Questions
Check weaknesses please.

### Soundness
3

### Presentation
2

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
The paper introduces FedKLS, a novel federated learning (FL) framework that combines KL-divergence-based personalization with SVD-based low-rank adaptation. FedKLS assigns spectral subspaces to clients based on the divergence of their data distribution from a global IID reference. Clients with small KL values receive adapters for generalizability, while those with large KL values receive more personalized components. This dynamic allocation of spectral subspaces results in performance improvements compared to state-of-the-art parameter-efficient fine-tuning (PEFT) methods.

### Strengths
The dynamic allocation of spectral subspaces based on KL divergence and data heterogeneity is a novel contribution in the context of FL. The paper demonstrates good convergence guarantees, ensuring that FedKLS retains the same order of convergence as FedAvg. By transmitting only low-rank adapters, FedKLS significantly reduces communication costs, which is crucial for federated systems operating under bandwidth constraints.

### Weaknesses
1.	The fixed mapping from KL divergence to spectral subspace indices is based on a heuristic, assuming a linear relationship between KL divergence and the number of singular values. This might not always be optimal, and the method would benefit from a more flexible mapping that can adapt to different scenarios.
2.	While the aggregation of low-rank updates is feasible due to the same shape of adapter matrices, there is no guarantee that updates from distinct subspaces (e.g., principal vs. minor singular components) will aggregate effectively. This could potentially lead to issues when clients work in highly divergent parts of the singular value spectrum.
3.	The KL divergence is computed only once at the beginning of training and assumed to remain static throughout. This assumption is restrictive, as client data distributions may evolve over time (e.g., due to concept drift). Periodic recalibration of KL divergence and reassignment of spectral subspaces could be necessary for long-term performance.
4.	The computational cost of the SVD step is not discussed in detail. For large models, computing SVD for each client could become a bottleneck, particularly on edge devices with limited computational resources. A detailed breakdown of the computational cost, especially for the SVD computation, is needed.

### Questions
1.	Did you consider using alternative metrics of heterogeneity, such as Wasserstein distance or gradient-based divergences (e.g., Fisher divergence)? How would these affect the spectral adaptation and personalization process in FedKLS?

### Soundness
3

### Presentation
3

### Contribution
3
