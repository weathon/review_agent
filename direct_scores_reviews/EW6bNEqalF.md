## Summary

This paper proposes two improvements to offline RL in Regular Decision Processes (RDPs): (1) a novel language metric $L_X$ grounded in formal language theory that replaces $L_\infty^p$-based distinguishability with a structured language-family test, and (2) a Count-Min-Sketch (CMS) approach to reduce memory requirements. The language metric is shown (Theorem 1) to achieve $\Omega(1)$ distinguishability in the T-maze family where $L_\infty$-based methods suffer $\mathcal{O}(2^{-N})$ distinguishability, and PAC sample complexity bounds are proven for both variants. The authors also identify and correct a mistake in prior work (Cipollone et al., 2023).

---

## Strengths

- **Exponential separation result (Theorem 1) is clean, concrete, and compelling.** The paper constructs an explicit family of RDPs where $L_\infty^\ell$-distinguishability is $\mathcal{O}(2^{-N})$ while $L_{\mathcal{X}_{2,1}}$-distinguishability remains $\Omega(1)$. This directly justifies the paper's motivating question and is not a generic argument—it pinpoints precisely which structural property makes the language family work (the probability of seeing a reward under action *North*).

- **T-maze scaling experiment (Figure 2) is the strongest piece of empirical evidence and directly matches the theory.** It shows linear vs. exponential scaling of time and RDP state count for language metric vs. CMS as corridor length grows, which precisely mirrors the Theorem 1 separation. The experiment runs to corridor length 100, providing ample quantitative evidence.

- **The two-dimensional hierarchy $\mathcal{X}_{i,j}$ is a creative bridge between formal language theory and RL distinguishability.** The $C_k^\ell$ operator combined with atomic symbol families $\mathcal{G}_1, \mathcal{G}_2, \mathcal{G}_3$ yields a principled and structured function class for an IPM-style distance, rather than an ad hoc construction. The unifying perspective in Definition 2 that recovers $L_\infty$, TV, and prefix metrics as special cases is elegant.

- **Correction of an error in Cipollone et al. (2023) demonstrates thoroughness.** The identification of an additional $\sqrt{H}/\mu_0$ multiplicative factor in the sample complexity is a substantive contribution to the theoretical foundations of the field.

- **Practical results in Table 1 are convincing on hard domains.** On T-maze(c), Cheese, and Mini-hall—domains requiring long-term memory—the language metric approach achieves substantially better policies than both FlexFringe and CMS, with smaller automata and faster runtime. FlexFringe fails entirely on T-maze(c) (reward 0.0 vs. 4.0).

---

## Weaknesses

### Fatal
None.

### Major

- **End-to-end offline RL evaluation is absent.** The paper's stated contribution is offline RL, yet all experiments evaluate only automaton learning quality (number of states, runtime) and the reward of the *derived* policy—without ever running the full offline RL pipeline (ADACT-H → RegORL planning → policy evaluation against a proper offline RL baseline). There is no comparison against RegORL with the original $L_\infty^p$ statistical test, which is the direct theoretical predecessor the paper claims to improve. Without this, the empirical offline-RL claim rests entirely on Table 1 reward numbers, which cannot be attributed to sample efficiency improvements.

- **No experiments validating the sample efficiency (PAC) claim.** The central theoretical contribution is improved sample complexity. Yet every experiment uses a fixed $K = 100$ episodes dataset. There are no success-probability-vs-dataset-size curves, no ablations over dataset size, and no empirical demonstration that the language metric requires fewer samples to recover the correct automaton or achieve a target reward. The sample efficiency improvement is entirely theoretical and empirically unsubstantiated.

- **Model-selection problem for $(i, j)$ is unresolved and could undermine applicability.** Assumption 1 requires the behavior policy to guarantee $L_{\mathcal{X}_{i,j}}$-distinguishability $\mu_0 > 0$ for the *chosen* language family. However, the practitioner does not know the hidden RDP and therefore cannot verify which $(i,j)$ satisfies this for a given dataset. The experiments always use $\mathcal{X}_{3,1}$, but no justification—theoretical or empirical—is provided for why this choice is safe across domains. The paper provides no characterization of when small $(i,j)$ fails to distinguish states, no adaptive search procedure, and no failure-mode analysis. This is not a minor reproducibility concern; if the chosen family is too coarse, distinct RDP states will be incorrectly merged and the resulting policy may be arbitrarily suboptimal.

- **No ablation over the hierarchy parameters $(i, j)$.** Since the central technical contribution is the hierarchy $\mathcal{X}_{i,j}$, the paper should empirically demonstrate the tradeoff between expressiveness and cost by varying both $i \in \{1,2,3\}$ and $j$. As it stands, only $\mathcal{X}_{3,1}$ is tested, making it impossible to assess whether the hierarchy design is necessary or whether $\mathcal{X}_{1,1}$ would suffice, or when larger $j$ matters.

### Minor

- **The $d_m^*$ dependence can be exponential in $H$, potentially negating the sample complexity gains.** Theorem 3's bound is $\tilde{\mathcal{O}}(C_\mathbf{R}^* \log(1/\delta) \log|\mathcal{X}| / (d_m^* \mu_0^2))$. The paper acknowledges that $1/d_m^*$ can be exponential in $H$ if some optimal-policy state is very hard to reach. This means the claimed exponential improvement via $\mu_0$ can be offset by an exponential penalty in $1/d_m^*$. The paper does not analyze whether these terms can simultaneously be favorable, which leaves the claimed exponential benefit ambiguous for general structured RDPs beyond T-maze.

- **The correction to Cipollone et al. (2023) is stated but not made self-contained.** The paper says "both their and our sample complexity has an additional multiplicative term $\sqrt{H}/\mu_0$," but this factor is not visibly present in the theorem statements shown in the main text (Theorems 2 and 3 show $\mu_0^2$ in the denominator, not $\mu_0^3$ or an explicit $\sqrt{H}$). The $\tilde{\mathcal{O}}$ notation hides only poly-logarithmic terms per the notation section, so a polynomial $\sqrt{H}$ factor should appear explicitly. This inconsistency needs clarification.

- **The offline RL objective is stated in two inconsistent forms.** Section 2 defines $\varepsilon$-optimality in expectation over $h_0$, but Section 2.3 states the goal as $V_\circ^*(h) - V_{\hat\pi}^*(h) \leq \varepsilon$ for each $h \in \mathcal{H}_0$. The notation $V_{\hat\pi}^*(h)$ (value of $\hat\pi$ with a star superscript) is also potentially misleading; presumably this is $V^{\hat\pi}$.

- **Theorem 2 does not specify the CMS parameters in the theorem statement itself.** The CMS approximation quality depends on $\delta_c$ and $\varepsilon$, but these are not enumerated in the theorem body, making the guarantee incomplete as stated in the main text.

- **No memory measurements despite CMS being explicitly motivated by memory reduction.** The paper introduces CMS to address memory requirements, but Table 1 reports only runtime and state counts. Peak memory usage is not reported for any algorithm, making it impossible to verify the memory efficiency claim empirically.

- **Typo in the two-dimensional hierarchy description.** Section 4.1 reads: "It is parameterised by $j$ for the granularity of the atomic symbols, and by $j$ for the sequential composition." Both dimensions are labeled $j$; one should clearly be $i$.

### Tiny

- The conclusion says the language approach "remov[es] the dependency on $L_\infty^p$-distinguishability parameters," which is slightly inaccurate—it replaces that dependency with $L_X$-distinguishability, which may be much more favorable but is still a distinguishability requirement.

---

## Nice-to-Haves

- **Discuss practical strategy for selecting $(i,j)$.** A heuristic such as iterating from $(1,1)$ upward until state counts stabilize, or using a held-out validation split, would greatly improve practical utility.
- **Robustness analysis when Assumption 1 is violated.** An empirical demonstration of graceful degradation (e.g., reverting toward $L_\infty$ behavior or harmless state over-splitting) would reassure practitioners.
- **Include comparison against neural offline RL methods (e.g., CQL/IQL with RNN history embeddings).** The paper's formal guarantee setting means this comparison is not required, but it would help contextualize practical performance relative to the broader ICLR community's typical baselines.
- **Provide a more general theorem characterizing when small-$(j)$ families suffice.** Theorem 1 is an existence result; a structural characterization (e.g., "whenever the distinguishing event depends only on $j$ elementary patterns in the trace") would strengthen the contribution.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **[Removed — scope creep]** Harsh Critic's suggestion that the paper should compare against "sequence-model-based offline RL / latent-state learning." The paper explicitly targets provably correct algorithms, and local optimization methods are acknowledged to lack formal guarantees. Evaluating this paper on whether it competes with deep learning approaches without guarantees is scope creep.
- **[Removed — scope creep]** Requests for comparison against Transformer/RNN-based policies from Review 2. Same reasoning as above; the paper's contribution is formal and theoretical, not a systems benchmark.
- **[Removed — methodology norms]** Requests for confidence intervals on all Table 1 quantities. Standard deviations are provided where results vary across runs; exact values are given where the outcome is deterministic. This is acceptable practice.
- **[Removed — cannot verify]** All criticisms about missing related works, as external reference availability cannot be confirmed.
- **[Removed — overly picky]** Harsh Critic's claim that the paper "relies too heavily on appendices for pseudocode." The pseudocode of ADACT-H is a standard prior algorithm; it is reasonable to place it in the appendix with a clear pointer. The novel parts of the algorithm (statistical test modification) are described in the main text.

---

## Novel Insights

The most genuinely novel conceptual insight in this paper—one not fully appreciated by the sub-reviewers—is that the exponential blowup in $L_\infty^p$-based RDP learning is not a fundamental barrier but an artifact of choosing too fine-grained a function class for the distributional comparison. By framing state distinguishability as an integral probability metric (IPM) over a structured language class, the paper shows that the right level of granularity—coarser than individual suffix strings but finer than total variation—can be both (a) computationally tractable via simple membership counting and (b) statistically powerful via $\Omega(1)$ separation in structured domains. This reframing connects automata learning with IPM theory in a way that may be useful beyond RDPs, e.g., in any setting where state-merging algorithms must compare distributions over structured sequential data.

---

## Evaluation Summary

| Axis | Assessment |
|---|---|
| **Originality** | High — the language metric hierarchy and its connection to the dot-depth hierarchy is a novel and principled idea, not a routine extension |
| **Importance of research question** | Solid — non-Markovian offline RL is genuinely important and the exponential sample complexity barrier is a real obstacle |
| **Claims well supported** | Partially — the theoretical claims are proven, and the T-maze scaling experiment is compelling; but the sample efficiency claim lacks direct empirical support |
| **Soundness of experiments** | Weak — experiments evaluate automaton quality and policy reward, but not the core sample efficiency claim; no comparison to ADACT-H with original test |
| **Clarity of writing** | Adequate — the narrative arc is coherent, but the notation for $L_\infty^\circ$, $L_\infty^p$, $L_1^\circ$, $L_1^p$, $L_X$ is easy to confuse, and Definition 1 / the hierarchy need more intuition |
| **Value to research community** | Moderate-to-high for the subfield; the IPM reframing and correction of prior error are useful contributions |
| **Contextualized relative to prior work** | Good — the paper clearly identifies what it improves over RegORL and positions itself correctly in the non-Markovian RL literature |

---

## Suggestions

1. **Add a direct ablation comparing ADACT-H (original $L_\infty^p$), ADACT-H+CMS, and ADACT-H+$L_X$ under identical datasets.** This is essential to empirically validate the sample efficiency claim and should be the primary baseline, not FlexFringe.
2. **Add at least one dataset-size sweep** (e.g., vary $K$ from 10 to 1000 on T-maze or Corridor) showing recovery probability vs. dataset size for each method.
3. **Provide an ablation over $(i, j)$** on at least two domains to justify the $\mathcal{X}_{3,1}$ choice and characterize when larger $j$ or $i$ is needed.
4. **Clarify the $\sqrt{H}/\mu_0$ correction** either by making it explicit in the theorem statement or explaining why it is absorbed into the $\tilde{\mathcal{O}}$ notation.
5. **Add peak memory measurements** to Table 1 to substantiate the CMS memory efficiency claim.
6. **Fix the notation typo** in the hierarchy parameterization description (both dimensions labeled $j$).
7. **Discuss Assumption 1 robustness** — what happens empirically when $\mathcal{X}_{i,j}$ is too coarse? Does ADACT-H over-merge states or under-merge?

MY FINAL SCORE: <pineapple>5.8</pineapple>