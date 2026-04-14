## Summary

DSpodFL is a unified algorithmic framework for decentralized federated learning (DFL) that jointly models heterogeneous and time-varying resource constraints by treating local gradient computation (indicator $v_i^{(k)}$) and inter-client communication (indicator $\tilde{v}_{ij}^{(k)}$) as arbitrary binary random variables. The framework subsumes DGD, DFedAvg, and Randomized Gossip as special cases. The authors derive convergence guarantees for both strongly-convex and non-convex objectives under both constant and diminishing learning rates, requiring only asymptotic graph connectivity—a weaker condition than static or B-connected graph assumptions used in prior work. Experiments on FMNIST+SVM and CIFAR10+VGG11 demonstrate improved accuracy-vs-delay tradeoffs compared to baselines across varying data heterogeneity, graph density, and network size.

---

## Strengths

- **Genuine unification of sporadic SGD and sporadic communication in the decentralized setting.** Prior work addresses one dimension or the other; DSpodFL is, to the reviewers' and authors' knowledge, the first framework to jointly model both in fully decentralized FL. The two-indicator design in Eq. (2) is clean and directly recovers DGD ($v_i^{(k)}=1$, $\tilde{v}_{ij}^{(k)}$ deterministic), Randomized Gossip ($v_i^{(k)}=1$, $\tilde{v}_{ij}^{(k)}$ random), DFedAvg ($\tilde{v}_{ij}^{(k)}$ periodic) as special cases.

- **Coupled error recursion that cleanly exposes the role of each sporadicity term.** Lemmas 4.7 and 4.8 decompose the analysis into average-model error and consensus error, with $d_{\min}^{(k)}$ governing optimization degradation from sporadic SGDs and $\tilde{\rho}^{(k)}$ governing consensus degradation from sporadic communication. The coupled linear recursion in Eqs. (7)–(8) is a technically sound and interpretable structure that allows closed-form convergence statements.

- **Weakest-known graph connectivity assumption for this class of methods.** Assumption 4.4 (asymptotic connectivity of the union graph with each edge appearing infinitely often) is strictly weaker than the static connectivity required by Sun et al. (2022) and Mishchenko et al. (2022), and weaker than $B$-connectivity (Nedić & Ozdaglar, 2009). This makes the results meaningful for genuinely dynamic peer-to-peer topologies.

- **Systematic empirical advantage across multiple axes.** Figures 2–4 show that DSpodFL consistently outperforms all baselines across data heterogeneity levels (IID to 1-label non-IID), graph radii, client counts (10 to 50), and resource distribution types (Beta, Uniform, Bimodal Gaussian). The magnitude of the advantage grows under higher heterogeneity (Figs. 3d, 4b), which is a meaningful structural finding rather than a cherry-picked result.

- **Generalized data heterogeneity assumptions.** By parameterizing gradient diversity with two parameters $(\delta_i, \zeta_i)$ rather than bounding the gradient norm at optimum by a single constant, Assumptions 4.1(c) and 4.2(b) subsume the stricter $\zeta = 0$ assumptions used in Sun et al. (2022) and Mishchenko et al. (2022), yielding tighter convergence bounds for systems with moderate inter-client heterogeneity.

---

## Weaknesses

### Fatal
None.

### Major

- **The "time-varying" convergence analysis collapses entirely to static worst-case bounds.** Theorems 4.11 and 4.12 use $\tilde{\rho} = \max_k \tilde{\rho}^{(k)}$ and $d_{\min} = \min_{k,i} d_i^{(k)}$, eliminating all temporal structure. The theory therefore provides no insight into whether favorable time patterns (e.g., averaging-out of heterogeneity) help or hurt—it treats dynamic variation adversarially. This is a meaningful gap given that "time-varying resource heterogeneity" is positioned as a primary differentiator from prior work (Table 1, abstract, introduction). A tighter analysis that exploits temporal averaging, or a clear argument that worst-case tightness is unavoidable, should be provided.

- **Time-varying experiments are confined to the appendix while main experiments use constant probabilities.** All main-body experiments (Figs. 2–4) draw $d_i, b_{ij}$ from fixed distributions and hold them constant over iterations. Appendix O provides time-varying results, but these are never visible to a main-paper reader. Because demonstrating performance under dynamic resource availability is the paper's core motivation, these experiments should be in the main body. Without them, the "time-varying" claim is unsubstantiated in the primary contribution.

- **The non-convex gradient diversity assumption (Assumption 4.2(b)) is non-standard and may not hold for realistic deep networks.** Requiring $\|\nabla F_i(\theta)\| \leq \delta_i + \zeta_i\|\nabla F(\theta)\|$ for all $\theta$ means every local gradient norm is globally bounded by an affine function of the global gradient norm. Under severe heterogeneity and far from stationarity, this is not a property that follows from standard assumptions on neural network loss landscapes. Since CIFAR10+VGG11 is the primary deep-learning experiment, the gap between the assumed regime (Assumption 4.2(b)) and the experimental regime must be explicitly acknowledged; otherwise Theorem 4.12's practical relevance is unclear.

- **The independence structure in Assumption 4.3(b) is unrealistic for the stated motivation, and overstates the framework's generality.** The paper motivates resource heterogeneity where slow processors and poor links co-occur at the same client. But Assumption 4.3(b) requires cross-client uncorrelatedness of $v_i^{(k)}$ and cross-link uncorrelatedness of $\tilde{v}_{ij}^{(k)}$, ruling out exactly this structural correlation. Meanwhile, the abstract states the indicators can be "arbitrary," which directly contradicts this assumption. The scope of Theorem 4.11/4.12 should be clearly communicated as covering *uncorrelated* sporadic schedules, not arbitrary ones.

### Minor

- **The synthetic delay metric is insufficiently justified as a proxy for wall-clock time.** $\tau_{\text{trans}}^{(k)} = \left[\sum_i \frac{1}{|N_i|}\sum_j \hat{v}_{ij}^{(k)}/b_{ij}\right]/\left[\sum_i \frac{1}{|N_i|}\sum_j 1/b_{ij}\right]$ and the analogous $\tau_{\text{proc}}^{(k)}$ define delay as a normalized sum of inverse probabilities. While the authors reference Appendix P.3 for justification, this does not correspond to standard queuing or networking delay models, and it is not obvious that this metric preserves relative performance orderings under actual network constraints (bandwidth, congestion, etc.). Since the central empirical claim is "improved training speeds," this weakens that claim.

- **Convergence bounds contain opaque constants ($\Gamma_0, \Gamma_1, \Gamma_2, \Gamma_3, A$) defined in appendices, making rate comparisons impossible from the main text.** Theorem 4.11 states the asymptotic optimality gap in Eq. (10) involves $A = \frac{\tilde{\rho}^2}{\mu}(\Gamma_2^* - 1)(1-\frac{1}{\Gamma_0^*})$ with $\Gamma_0^*, \Gamma_2^* > 1$ from Appendix F.3. The recovery of DGD rates is discussed qualitatively but never instantiated concretely. A simple explicit corollary matching a known theorem from a cited baseline would make the "unification" claim verifiable.

- **The relationship between Assumption 4.4 and a uniformly bounded $\tilde{\rho}^{(k)} < 1$ is not established in the main text.** Asymptotic connectivity guarantees repeated edge appearances but not a quantitative lower bound on mixing speed over finite windows. The analysis in Proposition 4.10 requires $\tilde{\rho}^{(k)} < 1$ at every iteration, but the main paper does not clearly argue how Assumption 4.4 ensures this. This gap should be closed, even with a brief argument.

- **All experimental baselines are special cases of DSpodFL.** DGD, RG, DFedAvg, and Sporadic SGDs are all subsumed by the proposed framework, so outperforming them under the DSpodFL delay metric (which is naturally aligned with the framework's degrees of freedom) is not entirely surprising. The paper would benefit from at least one comparison against a method outside the DSpodFL family—e.g., an asynchronous DFL method—to demonstrate that the advantages are not purely definitional.

- **Main experiments use only 10 clients by default.** Fig. 4a shows 50 clients with promising results, but DFL is most compelling at larger scales. Given that Fig. 3c already shows DSpodFL's advantage grows with $m$, extending the scalability analysis (e.g., $m=100$) would strengthen the practical case.

### Tiny

- **Objective function $F_i(\theta) = \sum_{(x,y)\in D_i} \ell_{(x,y)}(\theta)$ is a sum (not average) over local examples.** Combined with $F = \frac{1}{m}\sum F_i$, this weights clients with larger datasets more heavily, which is not the standard FL objective when dataset sizes are heterogeneous. The paper does not clarify whether $|\mathcal{D}_i|$ is assumed equal across clients or whether this weighting is intentional.

- **Definition 4.5 appears to contain a notation issue**, substituting $\mathbf{1}_m\bar{p}^{(k)}$ where $\mathbf{1}_m\bar{\theta}^{(k)}$ (the consensus mean matrix) seems intended. This should be clarified even if the concept follows Koloskova et al. (2020).

---

## Nice-to-Haves

- **Explicit proposition mapping DGD / DFedAvg / RG to specific choices of $v_i^{(k)}$, $\tilde{v}_{ij}^{(k)}$.** The reductions are described qualitatively in Sec. 3.2 and Fig. 1 but a short, precise proposition with exact conditions would convert the "unified framework" claim from qualitative to mathematical.

- **Empirical validation of convergence rates vs. theoretical predictions.** Plotting the predicted bound from Eq. (9) alongside empirical error across iterations would reveal whether the bounds are informative or vacuous, strengthening the link between theory and experiments.

- **Per-client consensus tracking in experiments.** Plotting $\|\theta_i^{(k)} - \bar{\theta}^{(k)}\|$ for representative clients would verify that DSpodFL actually achieves consensus under heterogeneous sporadicity, a key theoretical guarantee that is currently untested empirically.

- **Experiments with correlated computation/communication sporadicity.** Even if the theory does not cover this case, empirically characterizing how DSpodFL degrades under correlated schedules (e.g., clients with $d_i < 0.3$ also having $b_{ij} < 0.3$) would give practitioners useful guidance.

- **Evaluation on a larger-scale benchmark** (e.g., ResNet on CIFAR-100 or a language task), as SVM+FMNIST and VGG11+CIFAR-10 are limited benchmarks for a paper making practical heterogeneous-system claims.

- **Discussion of the mechanism for constructing $\mathbf{P}^{(k)}$ in practice.** The Metropolis-Hastings weights $r_{ij}$ in Eq. (4) are computed over the static graph $\mathcal{G}$, and the sporadic indicator only gates whether a link is used. This is a clean and implementable design, but a brief explicit statement in the main text confirming that double stochasticity is maintained without per-iteration coordination would preempt confusion.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **Reviewer claim that Assumption 4.1(c) is trivially satisfied by smoothness and thus provides no modeling value.** The reviewer argues that $\|\nabla F_i(\theta) - \nabla F_i(\theta^*)\| \leq \beta_i\|\theta - \theta^*\|$ by smoothness, making $\delta_i + \zeta_i\|\theta - \theta^*\|$ superfluous. This misses the point: the parameterization allows $\zeta_i < \beta_i$ at the cost of a non-zero $\delta_i$, reflecting gradient diversity at the optimum separately from global curvature. This is a well-known relaxation in heterogeneous FL (Lin et al., 2021) and leads to strictly tighter bounds than using $\zeta = 0$ alone. The assumption is meaningful and should not be criticized.

- **Criticism that the framework is "merely a stochastic template" rather than a deployable algorithm.** The paper explicitly and consistently presents DSpodFL as an *analysis framework* for a family of sporadic schedules. Criticizing the absence of an adaptive scheduler is scope creep—the paper never claims to provide one, and such a contribution would be a separate paper. This should not count as a weakness.

- **Criticism about doubly-stochastic weight construction requiring decentralized coordination.** The paper's design (fixed $\mathbf{R}$ via Metropolis-Hastings over the static graph $\mathcal{G}$, with the diagonal corrected by the sum of active-link weights) is fully described in Eq. (4) and Appendix B. Because $r_{ij}$ depends only on the static neighborhood sizes (known to each client), and the self-weight $p_{ii}^{(k)} = 1 - \sum_j r_{ij}\tilde{v}_{ij}^{(k)}$ is computable locally given knowledge of which links fired, this does not require global coordination beyond knowing one's own activated links. The concern about coordination overhead is not substantiated by the paper's actual design.

- **Criticism that comparing against baselines that are DSpodFL special cases makes the comparison unfair in favor of DSpodFL.** The baselines are standard, widely-used DFL algorithms (DGD, DFedAvg, RG), and showing that the more general framework outperforms restricted special cases under the same evaluation protocol is the natural way to validate a unification paper. Per the review instructions, comparisons unfair to baselines (not to the proposed method) should not be held as a weakness.

- **Reviewer criticism of the formatting/venue tag visible in the paper.** Not a substantive scientific concern.

- **Reviewer claim that the $d_{\min}^{(k)}\to 1$ recovery claim is only partial and thus the "unified" claim fails.** The paper explicitly states the conditions ($d_{\min}=1$, $\zeta=0$ for full DGD recovery) and demonstrates partial recovery in the main text (Lemmas 4.7, 4.8 discussions). While the recovery is approximate or conditional on parameter settings, this does not invalidate the unification claim at the algorithmic template level.

---

## Novel Insights

The most genuinely novel observation across the three reviews—and one not adequately highlighted in the paper itself—is the structural asymmetry in how the two sporadicity dimensions affect convergence: SGD sporadicity (via $d_{\min}$) enters *additively* in the asymptotic optimality gap (proportional to $(1-d_{\min})\delta^2$ in Theorems 4.11 and 4.12), while communication sporadicity (via $\tilde{\rho}$) enters *multiplicatively* through terms like $\frac{1+\tilde{\rho}}{1-\tilde{\rho}}$, which diverge as $\tilde{\rho}\to 1$. This implies that systems facing sparse communication (high $\tilde{\rho}$) suffer qualitatively more severe convergence degradation than systems with sparse computation at the same participation rate—a non-obvious and practically important result that deserves explicit discussion in the main paper. The empirical finding that DSpodFL's advantage grows under higher heterogeneity (lower Beta parameter, lower $\mu$ in bimodal settings) is consistent with this structure: the baselines that fix either SGD or communication schedules are disproportionately hurt by the multiplicative communication degradation when $\tilde{\rho}$ is high.

---

## Suggestions

1. **Move time-varying sporadicity experiments from Appendix O into the main body**, with at least one non-trivial temporal pattern (e.g., diurnal/bursty availability) that exercises the time-variation claimed as a primary contribution.

2. **Add an explicit discussion of why worst-case bounds are unavoidable** (or explore whether average-case or sum-of-rates bounds are achievable) for the time-varying setting. If the theory necessarily collapses to static worst-case, acknowledge this as a limitation and scope out tighter time-varying analysis as future work.

3. **Provide a concrete corollary** instantiating Theorem 4.11 or 4.12 under specific parameter choices to match a result from Koloskova et al. (2020) or Sun et al. (2022), making the "recovery of prior work" claim verifiable in the main text.

4. **Clarify the scope of Assumption 4.3(b)** by changing "arbitrary indicator random variables" in the abstract/introduction to accurately reflect the uncorrelatedness requirement, and add a short discussion of whether independent scheduling is a reasonable approximation in practice.

5. **Justify or replace the delay metric** $\tau_{\text{total}}$: either provide a derivation showing it is proportional to expected latency under a standard queuing or half-duplex channel model, or replace it with a more transparent physical delay model in the experiments.

6. **Include at least one non-subsumed baseline** (e.g., an asynchronous DFL method) to provide an external reference point for the empirical performance claims.

---

## Evaluation on Key Axes

- **Originality:** Moderately high. Jointly modeling sporadic SGD and sporadic communication in fully decentralized FL with time-varying heterogeneity is a genuine first, and the two-indicator framework is clean and general. The analysis techniques adapt standard decentralized optimization tools rather than introducing fundamentally new proof strategies, but the coupling they handle is novel.

- **Importance of research question:** High. Resource heterogeneity and time-varying availability are central challenges in practical peer-to-peer FL, and the absence of a principled joint analysis is a real gap.

- **Whether claims are well supported:** Partially. Convergence claims are mathematically rigorous for the stated assumptions. However, the "time-varying" and "arbitrary indicator" framings of the contribution are overstated relative to what the assumptions actually allow (static worst-case bounds, uncorrelated indicators). The strongest practical claim—improved training speed—relies on a synthetic delay metric not validated against real hardware.

- **Soundness of experiments:** Adequate but limited. The experimental design is internally consistent and covers a useful range of parameters. The main gaps are: constant-probability main experiments (time-varying in appendix only), exclusively DSpodFL-family baselines, and a delay metric without physical justification.

- **Clarity of writing:** Good. The paper is well-organized and the core idea is communicated clearly. Weak spots are the opacity of constants in theorems and the conflation of "framework" with "algorithm" in parts of the introduction.

- **Value to the research community:** Solid. The unified framework and convergence results will be useful to researchers building on decentralized optimization under heterogeneous resources. The practical value depends on whether the independence and non-convex assumptions can be relaxed.

- **Contextualized relative to prior work:** Mostly adequate. The paper correctly identifies and distinguishes itself from DGD, DFedAvg, RG, and centralized sporadic FL. The comparison with Even et al. (2024) is present but brief; a more detailed comparison of assumptions and rate expressions would benefit readers.