# Graph Mamba Operator for Learning the Dynamics of Particle-based Systems

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 4

## Abstract
Modelling complex 3D dynamics---from molecular conformations to particle interactions and human motion---requires capturing dependencies spanning long temporal horizons and non-local spatial interactions. Graph neural networks (GNNs) have shown promise in spatio-temporal settings but often suffer from instability and degraded accuracy in long-range forecasting. We propose the Graph Mamba Operator (GraMO), a neural operator that integrates state-space models (SSMs) with geometric learning to capture spatio-temporal correlations jointly. To jointly model complex dynamics, GraMO integrates a stable, SSM-based temporal backbone with an SSM-parameterized graph update to capture long-range spatial dependencies. Unlike stepwise predictors that accumulate errors over time, GraMO learns entire trajectories in a single forward pass. Across diverse benchmarks ranging from molecular dynamics to human motion capture, GraMO shows notable improvements in trajectory fidelity, stability, and robustness over strong baselines with relative improvements of over 26.3\% on motion capture benchmarks and 25.2\% on MD17 final-state prediction. Ablation studies reveal that temporal SSM components consistently improve performance, while spatial SSM updates show task-dependent benefits---helping with long-range interactions in large molecules but potentially hindering performance on systems with primarily local dependencies. Altogether, these results suggest that selective integration of SSM components with graph neural networks can improve performance on particle-based systems, with applications in molecular simulations, articulated rigid body dynamics, and particle systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper defines $\mathcal F_\theta=(\mathcal S\circ\mathcal T)^L\circ \mathcal E$, where $\mathcal T$ is a bidirectional selective SSM (Mamba) operating along time and $\mathcal S$ is a graph SSM that propagates non-local spatial information; the model maps a single snapshot to an entire future trajectory and is trained with A-MSE over $P$ uniformly sampled times. The encoder repeats the initial graph $P$ times and injects a learned or sinusoidal time embedding, and the theory section proves permutation equivariance of the spatial update and discusses stability under spectral-radius bounds. Experiments report A-MSE/F-MSE on N-Body, CMU MoCap subsets, MD17 molecules (heavy-atom only), and AdK equilibrium.

### Strengths
From an originality perspective, presenting a trajectory operator that composes temporal SSM blocks with a spatial SSM is a clean architectural statement, and the decision to predict the whole rollout in one shot is aligned with prior operator work but still practical for long horizons. The paper offers clear expository sections, including an explicit definition of S2S vs S2T metrics and detailed hyperparameters; this improves clarity and reproducibility. The theoretical part, though modest, correctly establishes permutation equivariance for the spatial update and gives a stability discussion via spectral-radius control, which, if enforced, would be meaningful for long rollouts. Empirically, the MoCap results suggest that the temporal SSM can help in human-motion settings where long-range temporal structure matters, which hints at potential significance if strengthened.

### Weaknesses
In terms of novelty, the contribution overlaps strongly with two existing streams. First, operator-learning with SE(3)-equivariant temporal convolutions already models state-to-trajectory mappings end-to-end (EGNO), where temporal kernels are parameterized in the Fourier domain; GraMO feels like swapping those kernels for SSMs. Second, there is now a growing pool of Graph-Mamba variants (Graph-Mamba, SpoT-Mamba, DyG-Mamba, etc.) that adapt selective SSMs to graphs; the manuscript would need to position itself precisely against these works and run them as baselines. At present, the paper reads as stitching together Mamba and EGNO, with the specific SSM parameterization being the main delta. 

On evaluation quality, several missing baselines materially weaken the claims. For geometric trajectories with long horizons, GeoTDM is the natural diffusion baseline with SE(3)-equivariant temporal kernels; for continuous-time modeling on graphs, GF-NODE is directly relevant; for MoCap generation/prediction, a diffusion model such as MDM is expected. None appear in the tables, so the comparative story is incomplete.

The symmetry story is underpowered for 3D dynamics. The method proves permutation equivariance but does not encode SE(3) invariance/equivariance in space or time; this is a known requirement for robust molecular and rigid-body dynamics and is already present in strong baselines (EGNN/SE(3)-Tr., EGNO). The paper should either add an SE(3)-equivariant variant or carefully delimit its scope to settings where coordinate frames are fixed or augmented. 

Some design choices complicate interpretation. The temporal block is bidirectional, which is fine for offline trajectory reconstruction but undermines any causal forecasting claims; the paper should make this explicit and, ideally, report a causal variant for fair comparison with rollouts. The input construction repeats the initial graph $P$ times and relies on time embeddings; this conflates learning actual dynamics with extrapolating from positional encodings and deserves sensitivity analyses to $P$, to embedding type, and to non-uniform time grids.

### Questions
First, how would results change if the temporal block were strictly causal, and could you include such a variant to compare with sequential EGNN rollouts; this could clarify whether bidirectional smoothing is a source of gains. Second, can you enforce the stability conditions you prove, for example, by parameterizing $W$ with a spectral-norm cap or adding an explicit damping factor $\alpha \hat A$, and show ablations on long-horizon drift. Third, can you evaluate on rMD17 with force/energy MAE (and, for AdK, report energies or contact/dihedral metrics) to align with standard practice. Fourth, where do you stand against Graph-Mamba baselines that use node ordering or walk-based scans; please run at least one representative method to demonstrate the benefit of your order-free spatial SSM. Fifth, could you include GeoTDM and GF-NODE as baselines on the molecular and motion tasks and a diffusion baseline like MDM on MoCap, with shared splits and identical horizons. Sixth, can you report efficiency (tokens/s, memory, params) and show how GraMO scales with $P$ and $L$; the efficiency pitch is currently unsubstantiated. Finally, because the encoder repeats $\mathcal G^{(0)}$ with time embeddings, can you test non-uniform or extrapolated time grids and report robustness across different $P$.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces the Graph Mamba Operator (GRAMO), a new architecture integrating State-Space Models (SSMs) with graph neural networks to learn the dynamics of particle-based systems. The model consists of a temporal SSM (Mamba) backbone to capture long-range time dependencies and a novel SSM-parameterized spatial graph layer for spatial interactions. The authors claim this S2T (state-to-trajectory) model achieves state-of-the-art results on several benchmarks, including MD17 and Mocap, and provide a theoretical analysis of its properties, such as permutation equivariance and stability.

However, due to a potential data leakage issue in 2 (out of 4) datasets, and a weird fixed/hardcoded starting frame issue in N-body dataset, I am very suspicious of the validity of their empirical reports. The seemingly unrelated proofs (that belong to generic GNN or Mamba paper) is another distraction point in paper writing. I am issuing reject now; if authors can fix the dataset bug and reproduce the affected experiments, also remove redundant proofs, I will re-consider.

### Strengths
1.  The central idea of integrating modern SSMs (like Mamba) with geometric graph learning is a compelling and timely research direction. Tackling long-range temporal dependencies in complex graph-structured physical systems is a significant and valuable problem for the community.
2.  The proposed architecture, which uses an SSM to parameterize the spatial message-passing layer, is an interesting concept, even if its empirical validation is flawed in the current manuscript.

### Weaknesses
The paper suffers from several critical flaws that undermine its central claims.

1.  The paper's headline results are derived from an invalid dataloader setup. Checking the codes in the anonymous repository reveals that:
    * For continuous trajectory datasets (MD17, MDAnalysis), the authors use "random initial points" and "in-trajectory" splitting. This is a severe form of data leakage. For sequential data, if a test sample (e.g., starting with initial time $30$) is close/adjacent to a training sample (e.g., at time $28$); assuming time step gap is $2$, your training sequence will contain $28,30,32,34,36,...$, while the test sequence will contain $30,32,34,36,...$. This setup allows the model to memorize short-term transitions from leaked data rather than learn the true long-range dynamics.
    * All results, extrapolation studies, and physics-conservation analyses on these datasets (MD17, MDAnalysis) are consequently rendered unreliable.
    * The N-Body is correctly splited by trajectories, but this one seems to hard code the initial frame to be $6,20,30$ for three variants, all of which discard many beginning frames for training. This is a very suspicious choice that needs justification. The results of this dataset also only shows a marginal **4.61\%** gain, adding on to the suspicion that the other datasets' gains are due to leakage.

3.  The "Theoretical Analysis" (Section 4.2) presented as a core contribution is not novel.
    * Proposition 4.3 (Permutation Equivariance) is a "sanity check" or a minimum requirement for any GNN, not a new discovery.
    * Lemma 4.2 (Spectral properties of $A$) is a textbook result from spectral graph theory and the basis of the original GCN, not a new finding.
    * The stability analysis (Prop 4.4) is a standard calculation for a simplified linear system, which ignores the actual non-linear "selective" nature of Mamba, hence not particularly insightful here.

4.  More mathematical errors:
    * The paper claims that setting $B=0$ and $C=0$ in the spatial operator ($S(t+1)=A^S(t)W+X(t)B$, $Y(t+1)=S(t+1)C$) recovers the GCN rule $Y=\sigma(A^XW)$.
    * A simple substitution shows this is false. If $C=0$, the output $Y(t+1)$ is always zero.

### Questions
1.  Can the authors correct all the "random" and "in-trajectory" splitting methodology to be "by-trajectory" splits for datasets (MD17, MDAnalysis), and re-run the corresponding experiments? This is essential to validate the main empirical claims of the paper.
2.  Please verify the derivation in Remark 4.1. Given $B=0$ and $C=0$, the output $Y(t+1)$ becomes zero. How does this recover the GCN message-passing rule as claimed?
3.  According to Table 5, the EGNN-spatial + GRAMO-temporal model (28.22) outperforms the full GRAMO model (34.12). Does this not imply that the novel spatial operator is the weakest part of the contribution and actually degrades performance compared to a baseline?
4.  Can the authors clarify what, precisely, is theoretically novel in Section 4.2? The analysis of permutation equivariance and the spectral properties of the normalized adjacency matrix are standard in the GNN literature.
5. Can you remove the abundant proof that belongs to previous Mamba works and only focus on your novel contributions?

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
4

### Summary
This paper proposes the Graph Mamba Operator, a neural operator for particle-based dynamics that interleaves (1) a bidirectional state-space model along the temporal axis and (2) an SSM-parameterized graph update along the spatial axis. GRAMO aims to predict an entire future trajectory from an initial state in one forward pass (state-to-trajectory, S2T), rather than rolling out step-by-step, thereby mitigating error accumulation.

### Strengths
1. The paper is generally readable with a clear operator view

2. The spatial update is clearly formulated as an SSM-like recurrence with stability, permutation equivariance, and multi-step Jacobian derived, which adds desirable guarantees.

### Weaknesses
- While SSMs and graph-SSM notions exist, the novelty hinges on the specific selective parameterization and stacking. The paper could sharpen comparisons to closely related graph-SSM works with similar token-mixing or ordering strategies.

- While qualitatively described as efficient, the paper lacks clear wall-clock/throughput comparisons versus EGNO/EGNN (for fixed accuracy) across horizons or system sizes.

- The reported Benzene F-MSE drops from ~55.7 (EGNO) to 2.06 for GRAMO (≈96% “gain”), an unusually large leap compared to other molecules; similarly dramatic gaps appear in A-MSE (22.06→1.08). This raises questions about preprocessing, evaluation horizons, or leakage; stronger cross-checks and per-molecule setup details are needed.  

- The evaluation omits larger-scale or more chaotic long-range systems and does not systematically compare with recent SE(3)-equivariant operator alternatives under equalized budgets.

### Questions
- MD17 setup reconciliation: Please provide a per-molecule table of (i) horizon $\Delta$T, (ii) timesteps P, (iii) train/val/test sizes, (iv) hydrogen removal, (v) graph augmentation, (vi) feature normalization, and (vii) random seeds. Also rerun Benzene with EGNO and GRAMO under a shared, verified script and report both A-MSE and F-MSE with error bars.

- Since SE(3)-equivariance is listed as future work, provide a small-scale variant with equivariant encoders/decoders (or an EGNN-spatial swap) to quantify sensitivity on MD17 and AdK. Even a partial result would help interpret cross-domain robustness.  

- Long-horizon stress test: Extend the extrapolation beyond 2× for at least one molecule and the mocap tasks, and include failure modes (where/when divergence starts) vs. EGNO. Adding energy/momentum drift plots over long rolls would strengthen the stability claim.  

- Ablation completeness: Report variants: (i) temporal SSM only vs. spatial SSM only vs. both; (ii) removing selectivity (fix B,C) to isolate the value of input-dependent parameters; (iii) replacing $\hat A$ with attention or higher-order adjacency to probe long-range capture. Clarifying these would solidify the design choices.

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
The author proposes the Graph Mamba Operator as a neural operator in modeling 3D dynamics. The method combines the SSM model with a graph to capture time and space dependencies. The method provides the benefit of learning the system dynamics in a single forward pass. The experiments on some benchmarks show an increase in performance. The paper also provides technical insight about the method and provides an efficiency analysis of the method.

### Strengths
1. First work to incorporate Graph into neural operator as layers of SSM in 3D dynamics, as I know.
2. The experiments on different benchmarks show the improvement in performance.

### Weaknesses
1. The authors introduce and comment on two "Key" related works in the paper, which contain "Graph Mamba" in the title and look the most related work for this paper. But the inclusion and comment seem to show that the authors did not read the paper, even the abstract, carefully, or they have a limited understanding of the area.
2. The experiments did not include some key baselines. The baselines could include more methods introduced in related work for each sub-area, such as other neural operators, other (neural) SSM methods, GraphODE, and Graph SSMs. Specifically, I noticed that in some parts of this method, the graph is copied into different layers with temporal embedding. What is the influence if we use one single graph all the time and let the SSM model learn the graph dynamics in the layers, and whether it could provide a better trade-off between performance and efficiency.
3. The idea of the work is new by including the graph mamba operator, but the clarity about the key difference and improvement of the work compared to previous work should be improved to make the contribution not incrimental.

### Questions
1. Could the authors provide a clear key version about the most related work, like a key roadmap?
2. Could the authors provide more experimental results to prove the effectiveness of the method?

### Soundness
3

### Presentation
1

### Contribution
2
