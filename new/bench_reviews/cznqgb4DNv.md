## Summary

The paper proposes Decentralized Sporadic Federated Learning (DSpodFL), a unified decentralized learning framework that models both local gradient computations and inter-client aggregations as arbitrary, potentially time-varying Bernoulli processes. The authors derive explicit convergence rates for strongly convex and non-convex losses under mild gradient-diversity assumptions, and provide delay-centric experiments showing training speedups over DGD, DFedAvg, and randomized gossip baselines.

## Strengths

- **Unified sporadic framework.** Eq. 2 and Table 1 demonstrate that DSpodFL jointly captures sporadic SGD, sporadic aggregation, heterogeneous resources, and time-varying dynamics in a single recursion, subsuming several prior decentralized methods as special cases.
- **Mild analytical assumptions.** Assumptions 4.1-(c) and 4.2-(b) employ $(\delta,\zeta)$-gradient-diversity bounds rather than the stricter uniformly bounded gradients assumed in several prior DFL works, tightening the analysis.
- **Coupled convergence analysis with explicit rates.** Lemmas 4.7–4.8 separate average model error and consensus error, unify them into the linear dynamical system in Eq. 7, and yield geometric convergence for strongly convex losses (Theorem 4.11) and $\mathcal{O}(1/K)$ rate for non-convex losses (Theorem 4.12).
- **Delay-centric experimental evaluation.** Measuring accuracy versus cumulative delay (Fig. 2) directly targets the operational metric practitioners care about, and DSpodFL consistently outperforms baselines across IID and non-IID splits on FMNIST and CIFAR-10.

## Weaknesses

### Fatal
None.

### Major

- **Assumption 4.4 does not support the constant-step-size guarantees.** Assumption 4.4 only requires that every edge in the asymptotic union appears infinitely often; it imposes no uniform lower bound on link probabilities $b_{ij}^{(k)}$ or on the spectral gap of $\mathbb{E}[\mathbf{P}^{(k)}]$. Yet Proposition 4.10 demands a constant learning rate satisfying $\alpha < O(1-\tilde{\rho}^{(k)})$ for all $k$, and Theorems 4.11–4.12 define $\tilde{\rho} = \max_{0\le k\le K}\tiltilde{\rho}^{(k)}<1$ and treat it as a uniform bound. Under Assumption 4.4 alone, the per-iteration spectral gap can approach 1, which would force the step size to zero and invalidate the asymptotic bounds in Eq. (10) and Theorem 4.12. The constant-step-size results therefore need an explicit uniform connectivity assumption (e.g., $b_{ij}^{(k)}\ge b_{\min}>0$ or $\tilde{\rho}^{(k)}\le \rho_{\max}<1$ for all $k$) that is currently missing.

- **Main-text experiments do not validate time-varying dynamics.** The abstract, introduction, and contribution bullets emphasize *time-varying* resource heterogeneity as a central motivation (e.g., “allowing these values to vary over the training process”). However, the main-body experiments in Figs. 2–4 sample static $d_i$ and $b_{ij}$ from fixed distributions and hold them constant for the entire run (Sec. 5: “held constant over iterations $k$”). While the paper refers to time-varying results in Appendix O, the core practical claim of handling dynamic variation is not demonstrated in the main body, leaving a significant gap between the stated contributions and the presented evidence.

### Minor

- **Imprecise terminology and inconsistent notation in the consensus definitions.** Definition 4.5 labels $\bar{p}^{(k)}$ as “the spectral radius of the expected mixing matrix,” which for a doubly stochastic matrix is 1; the intended quantity is presumably the second-largest eigenvalue modulus (or the spectral norm on the consensus orthogonal complement). The same definition introduces $\tilde{\rho}^{(k)}$ in Lemma 4.8 without reconciling it with $\bar{p}^{(k)}$. Additionally, the indicator variables are denoted $\tilde{v}_{ij}^{(k)}$ in Eq. 2, $\bar{v}_{ij}^{(k)}$ in Eq. 4, and $\hat{v}_{ij}^{(k)}$ in Assumption 4.3 and Definition 4.6, which needs to be aligned.
- **Baseline tuning limitations.** DFedAvg uses a single deterministic period $D$ computed from the average of $1/d_i$. A per-scenario grid search over $D$ would strengthen the comparison, though the current heuristic is reasonable.
- **Symmetric link activation assumption.** The analysis assumes $\tilde{v}_{ij}^{(k)}=\tilde{v}_{ji}^{(k)}$ without discussing how this symmetry is enforced in a fully decentralized, uncoordinated system (e.g., via pairwise rendezvous or shared random seeds).

### Trivial
None.

## Nice-to-Haves

- Move at least one time-varying experiment (e.g., sinusoidal or Markovian $d_i^{(k)}$, $b_{ij}^{(k)}$) from the appendix into the main text to substantiate the dynamic-heterogeneity claim.
- Include trace plots showing per-client computation/communication activity over iterations to make the sporadicity mechanism concrete.
- Clarify how symmetric gossip is implemented in practice, or extend the analysis to asymmetric activations.

## Removed Points

These points are flagged to be removed, treat them with caution:
- **“Appendix unavailable” criticism.** The paper notes time-varying experiments are in Appendix O; since appendices are stripped by the parser, this content exists in the original submission. The valid concern is that these results are absent from the main text, not that they are missing entirely.
- **“Mathematically incoherent” framing of Definition 4.5.** While the definition is sloppy, the intended consensus contraction coefficient is clear from context and standard in the cited literature (Koloskova et al., 2020). The issue is better described as imprecise terminology and inconsistent notation.

## Novel Insights

None beyond the paper's own contributions.

## Suggestions

- Add an explicit assumption that the spectral gap of $\mathbb{E}[\mathbf{P}^{(k)}]$ is bounded away from 1 uniformly in $k$ (or that communication probabilities satisfy $b_{ij}^{(k)}\ge b_{\min}>0$), and verify that all constant-step-size results follow from the amended assumption set.
- Align the indicator notation throughout ($\tilde{v}_{ij}^{(k)}$, $\bar{v}_{ij}^{(k)}$, $\hat{v}_{ij}^{(k)}$) and clarify in Definition 4.5 that the contraction coefficient is the second-largest eigenvalue modulus of $\mathbb{E}[\mathbf{P}^{(k)}]$ restricted to the consensus subspace.
- Include a main-text experiment where at least one resource parameter changes mid-training (e.g., a client’s compute probability drops abruptly) to validate the time-varying claim.

## Score and Decision

**Calibration anchors used:**
- **High:** `CMMpcs9prj.md` (avg 6.60, Accept Poster) – solid decentralized optimization theory with compression; our paper has a more novel framework but suffers from a missing uniform-connectivity assumption that this anchor did not have. `PpYy0dR3Qw.md` (avg 7.50, Accept Spotlight) – doubly-accelerated communication complexity with rigorous theory and experiments; our paper does not reach this level of rigor or experimental breadth.
- **Medium:** `C5w86qtcgY.md` (avg 5.25, Reject) – time-varying decentralized optimization with nearly matching bounds; criticized as incremental but theoretically sound. Our contribution is more novel, but the theoretical gap (missing uniform bound) makes it less rigorous. `nrctFaenIZ.md` (avg 5.75, Reject) – GradSkip with nice theory but limited experiments; our experiments are broader but omit time-varying validation in the main text.
- **Low:** `sSWGqY2qNJ.md` (avg 3.33, Reject) – unverified assumptions and oversimplified experiments. `jw8EoY1FvF.md` (avg 4.00, Reject) – impractical algorithm and restrictive bounded-gradient assumptions. Our paper is substantially stronger than these low anchors.

The paper sits between the medium and low bands: it offers a genuinely novel unified framework and milder assumptions, but the constant-step-size theorems lack a necessary uniform-connectivity assumption, and the main-text experiments do not support the time-varying claim that motivates the work. These issues are addressable in principle, but they undermine the core credibility as presented. Relative to the medium anchors, the theoretical gap is more fundamental than “incremental contributions” or “limited experiments,” justifying a score below the 5.25–5.75 cluster.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>