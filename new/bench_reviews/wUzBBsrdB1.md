Now I have all the information needed to write the final review. Let me compile it.

## Summary

This paper demonstrates that incorrect L0 settings in sparse autoencoders cause SAEs to learn incorrect features: too-low L0 leads to mixing of correlated (and anti-correlated) features into single latents, while too-high L0 produces degenerate solutions. Using toy models with ground-truth features, the authors show that the standard sparsity-reconstruction evaluation paradigm is fundamentally misleading—an SAE with mixed, polysemantic latents can achieve better reconstruction than the ground-truth SAE at low L0. They propose decoder pairwise cosine similarity (c_dec) as a proxy metric for detecting incorrect L0, and validate it on Gemma-2-2b and Llama-3.2-1b against sparse probing performance.

## Strengths

- **The demonstration that sparsity-reconstruction tradeoff plots are misleading (Section 3.3/3.4, Figures 4-5) is the paper's strongest contribution.** The key result—trained SAEs at L0=1 and L0=5 outperform the ground-truth SAE on variance explained by over 2x despite learning "horribly polysemantic latents"—directly challenges the dominant evaluation paradigm in the SAE literature. This has clear, immediate implications for how the field evaluates SAEs.

- **The initialization-to-ground-truth experiment (Section 3.1) is a clever methodological choice** that rules out local-minimum explanations and shows gradient pressure specifically drives feature mixing, not just poor initialization.

- **The analysis of anti-correlated features (Section 3.1, Figure 3) extends the feature-hedging story in an important direction.** Since negative correlations are pervasive in language data, the finding that anti-correlated features acquire negative components in each other's latents identifies a particularly damaging failure mode.

- **The toy model framework is well-designed**, following LRH assumptions with ground truth, enabling controlled manipulation of L0, correlation structure, and feature frequency. The systematic low/correct/high L0 exploration (Section 3.2, Figure 1) is thorough and reveals the asymmetric effect: low L0 corrupts every latent, while high L0 still preserves many correct ones.

- **The JumpReLU "sticking" observation (Section 3.6, Figure 7)**—that L0 stays near the correct value across a range of λ_s values—is a notable finding with practical implications for JumpReLU training.

- **The c_dec metric (Section 3.5, Eq. 4) provides a concrete, unsupervised signal** for L0 selection. Its minimization at the true L0 in toy models (Figure 6) and correspondence with peak sparse probing in LLMs (Figure 8) is a useful first step toward practical L0 guidance.

## Weaknesses

### Fatal
None.

### Major

- **No direct evidence of feature mixing in LLM SAEs.** The paper's strongest results are the toy model demonstrations showing *specific* feature mixing patterns (Section 3.1, Figures 2-3). For LLMs, the evidence is entirely indirect: high c_dec and poor sparse probing at low L0 are *consistent* with feature mixing but do not demonstrate it. The paper does not present a single case study of an LLM SAE latent at low vs. correct L0 showing that a specific latent has absorbed components of correlated features. This gap matters because alternative mechanisms could explain sparse probing degradation (e.g., important features simply being dropped rather than mixed), and those mechanisms would call for different diagnostics. Without directly inspecting LLM SAE latents, the paper cannot confirm the toy model mechanism transfers. The authors acknowledge c_dec is "not a perfect guide" (Section 6), but the abstract states the paper "shows" incorrect L0 causes SAEs to "fail to disentangle the underlying features" without distinguishing toy model demonstration from LLM inference.

- **c_dec's practical reliability for L0 selection in LLMs is underestablished.** In toy models, c_dec has a sharp, unambiguous minimum at the true L0. In LLMs, this clean behavior degrades: the Gemma-2-2b layer 5 c_dec curve (Section 4, Figure 8) shows a "long shallow region" with the global minimum somewhere inside it, and the authors resort to an ad hoc "elbow" heuristic (Section 4: "the 'elbow' in the cdec plots just before the jump due to low L0 is around L0 200"). This heuristic has no principled justification and could yield different L0 values depending on where one judges the elbow. Validation is against only one external benchmark (k-sparse probing from Kantamneni et al. 2025) on two small models at specific layers, with no comparison to alternative L0 selection methods (e.g., MDL-SAEs, AFA-SAEs). Additionally, the metric requires training a full sweep of SAEs, limiting practical utility. For a paper whose central practical contribution is a metric for L0 selection, this is insufficient evidence.

### Minor

- **The "most commonly used SAEs have L0 that is too low" claim (Abstract, Section 6) overgeneralizes.** This is supported by experiments on only Gemma-2-2b and Llama-3.2-1b at specific layers, plus a "cursory search of open source SAEs on Neuronpedia" (Section 6). The optimal L0 likely varies across model sizes, layers, SAE widths, and architectures. The claim should be narrowed to the models tested or supported by more systematic evidence.

- **The g=50 toy model's correlation matrix is "randomly generated" but not characterized (Section 3.2).** The paper does not describe how the correlation matrix is generated or report its properties (e.g., average correlation strength). Since c_dec's behavior could depend on the correlation structure, this omission limits understanding of when c_dec will be informative, particularly in real LLMs where correlations may be subtler.

- **Dead latents may bias c_dec.** The metric (Eq. 4) averages over *all* pairs of decoder vectors, including dead or near-dead latents. With h=32768 and L0~200, most latents are rarely active. Dead latents' decoder vectors contribute to c_dec but don't reflect feature structure. The paper does not discuss this potential bias. While Appendix A.9 explores alternative metrics, the main c_dec formulation's sensitivity to dead latents is unaddressed.

- **The "L0 can be simultaneously too low and too high" hypothesis (Section 4.2) is speculative.** This interesting claim is based entirely on the shape of decoder projection histograms, without independent validation. The statement that "there is likely a range of L0s where some latents are firing more than they ideally should while other latents are firing less than they ideally should" is reasonable but presented without evidence beyond histogram shape.

### Trivial
None.

## Nice-to-Haves

- Case studies of LLM SAE latents at different L0 values showing specific feature mixing (e.g., inspect a Gemma-2-2b latent at L0=10 vs. L0=200 to show the low-L0 latent mixes correlated concepts).
- Comparison of c_dec against alternative L0 selection methods (MDL-SAEs, AFA-SAEs) on LLMs to establish practical value.
- Testing on at least one larger model (e.g., Gemma-2-9b) or more layers to strengthen generalization claims.
- A method for setting L0 without training a full sweep; even a heuristic for scaling c_dec guidance would substantially improve practical impact.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Section numbering confusion (3.3/3.4 headings vs. content):** The harsh critic notes Section 3.3's heading says "MSE Loss Incentivizes..." but discusses the sparsity-reconstruction tradeoff, while 3.4's heading says "The Sparsity–Reconstruction Tradeoff" but contains the MSE comparison. There is also a self-reference "As we discussed in Section 3.3" from within Section 3.3 itself. While the headings are arguably swapped relative to content, both sections discuss closely related material, the arguments are logically sound, and the reader can follow the reasoning. This is a trivial presentation issue.

- **Missing complementary experiment (random initialization leading to same mixing):** The critic suggests initializing randomly and showing the same mixing pattern emerges. This would strengthen but not change the paper's argument—the ground-truth initialization experiment already proves the key point that gradient pressure drives mixing. The random initialization result is expected and not showing it is not a meaningful gap.

- **Training budget concerns (500M tokens):** The critic questions whether c_dec curves are stable with more training. 500M tokens is a standard training budget in the SAE literature and is sufficient for the purposes of this paper. This is a generic "more is better" criticism.

- **Request for per-latent firing rate analysis to explain JumpReLU vs. BatchTopK differences (Section 4.1):** The paper provides a plausible explanation (per-latent threshold adjustment in JumpReLU vs. global K in BatchTopK) and notes this is suspected. Further analysis would strengthen but is not required.

- **Comparison with alternative L0 selection methods demanded as essential:** While comparing c_dec against MDL-SAEs or AFA-SAEs would be valuable, the paper's main contribution is identifying the *problem* (incorrect L0 causes incorrect features) and proposing c_dec as a *first step* toward a solution. Demanding the paper fully validate against all alternatives is scope creep.

- **Explanation of high-L0 failure mechanism:** The paper clearly explains why low L0 causes mixing (MSE incentive) but only states high L0 causes "degenerate solutions" without detailed mechanism. This would be nice to have but the paper's main focus is on the low-L0 problem, which is the more important one (the paper shows it corrupts every latent, vs. high L0 preserving many correct ones).

## Novel Insights

The paper's most important insight—that the sparsity-reconstruction tradeoff evaluation paradigm is not just imperfect but *inverted* at low L0 (preferring incorrect SAEs over ground-truth ones)—has underappreciated implications beyond what the authors discuss. If reconstruction quality actively prefers feature-mixing solutions, then *any* SAE architecture optimized for the reconstruction-sparsity Pareto frontier will, by construction, tend toward polysemantic solutions when L0 is below the true value. This means the field's progress on improving the Pareto frontier may have been progress toward worse, not better, feature dictionaries. The asymmetric failure (low L0 corrupts *every* latent; high L0 preserves many) suggests the cost of underestimating L0 is dramatically higher than overestimating it—a practical guideline the paper could state more forcefully.

## Suggestions

- Add 1-2 case studies of LLM SAE latents at low vs. correct L0. Even a qualitative demonstration (e.g., showing a specific latent fires on mixed concepts at L0=10 but cleanly on one concept at L0=200) would transform the LLM story from indirect to direct.
- Narrow the "most commonly used SAEs have L0 that is too low" claim to "the specific SAEs we tested on Gemma-2-2b and Llama-3.2-1b have optimal L0 around 200, substantially higher than typical practice," and frame the broader claim as a hypothesis for future work.
- Discuss dead latent sensitivity of c_dec explicitly and consider whether an active-latent-only variant performs better.
- Provide a more principled criterion than "elbow" for identifying the correct L0 from c_dec curves, or at minimum acknowledge this gap clearly as a limitation.

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Decision | Relation to this paper |
|-------|-----------|----------|----------------------|
| "Scaling and evaluating SAEs" (tcsZt9ZNKD) | 8.2 | Accept (Oral) | Same domain; far more comprehensive (scaling to GPT-4, multiple new metrics, scaling laws). Our paper is narrower but makes a more focused conceptual point. Clearly below this. |
| "Sparse Feature Circuits" (I4e82CIDxv) | 8.0 | Accept (Oral) | Uses SAEs for downstream analysis; more applied and well-validated. Our paper is more about SAE methodology itself. |
| "A is for Absorption" (LC2KxRwC3n) | 7.5 | Reject | Studies a specific SAE pathology (feature absorption) with detailed analysis on Gemma Scope. Similar scope to our paper—identifies a failure mode and studies it. This paper got high individual reviewer scores but was rejected, likely due to limited scope. Our paper has comparable depth of LLM analysis but our pathology is arguably more fundamental (wrong L0 is upstream of absorption). |
| "Principled Evaluations of SAEs" (1Njl73JKjB) | 7.0 | Accept (Poster) | Proposes supervised evaluation; one case study (IOI). Similar narrow-but-important contribution profile. Our paper has broader scope but weaker validation. |
| "Rethinking Evaluation of SAEs" (HpUs2EXjOl) | 5.75 | Accept (Poster) | Critiques traditional metrics for ignoring semantic quality. Similar theme (metrics are misleading). Our paper has a stronger and more concrete demonstration. |
| "Compute Optimal Inference in SAEs" (ghH6YYDs15) | 4.67 | Withdrawn/Reject | Strong theoretical result about SAE insufficiency but limited real-model validation. Similar pattern to our paper (strong toy/theory, weak LLM), but our paper at least has LLM experiments. |
| "Interpretability Illusions" (v675Iyu0ta) | 5.6 | Reject | Toy model insights that may not generalize. Similar concern about toy-to-real gap. Our paper has more LLM validation than this. |
| "Metanetwork" (9L9j5bQPIY) | 2.5 | Reject | Only tested on single fully-connected layers. Our paper is clearly far above this. |

Our paper sits between the medium-scoring anchors (4.67-5.75) and the higher-scoring SAE pathology papers (7.0-7.5). The conceptual contribution (sparsity-reconstruction tradeoff is inverted at low L0) is genuinely important and well-demonstrated. However, the LLM validation gap (no direct feature mixing evidence, c_dec not fully validated) pulls it down from the 7+ range. Compared to "Rethinking Evaluation" (5.75), our paper has a stronger, more concrete demonstration. Compared to "Principled Evaluations" (7.0), our paper has weaker LLM validation but a more fundamental finding. I place this at 6.0—clearly above borderline papers with only toy validation, but below well-validated SAE methodology papers.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>