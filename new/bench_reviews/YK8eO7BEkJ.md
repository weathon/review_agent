## Summary

The paper presents an empirical study of how different normalization types (BN, LN, GN, IN, RMSN), their positions (before vs. after the SSM module), and their combinations affect the performance of Mamba blocks. Experiments on a long-sequence video dataset (Breakfast) and ImageNet‑100, with additional runs on ListOps and ImageNet‑1k, suggest that placing normalization after the SSM and using heterogeneous combinations (e.g., IN→SSM→LN, RMSN→SSM→BN) can improve accuracy and training behavior; the authors offer an intuitive explanation based on weight L2‑norm distributions across layers.

## Strengths

- **Timely and practically relevant question.** Normalization choices in Mamba variants are currently ad hoc; systematically exploring types, positions, and combinations in a unified block (Fig. 2) is clearly useful for practitioners.
- **Broad empirical sweep within the chosen setting.** The paper systematically enumerates 5 normalization types, their placement (before/after/both, via Tables 1–3), and all 5×5 type combinations (Table 4) on both a sequence and a vision task. This gives a reasonably complete map of this design space for the specific Mamba block and datasets used.
- **Clear empirical patterns.** The results consistently show that (i) adding normalization dramatically improves performance over None→SSM→None (Tables 1–4), (ii) after‑SSM normalization tends to outperform before‑SSM for both Breakfast and ImageNet‑100 (Tables 2–3), and (iii) certain heterogeneous combinations (e.g., IN→LN on Breakfast, RMSN→BN on ImageNet‑100) further improve over same‑norm configurations.
- **Some mechanistic insight.** The L2‑norm analysis (Fig. 4–5) shows that adding normalization after SSM reduces the growth and inter-layer disparity of weight norms, and that certain combinations (e.g., BN→IN) interpolate between the behaviors of single‑norm configurations. While tentative, this is a nontrivial observation that could inspire more rigorous follow‑ups.
- **Useful taxonomy of existing designs.** The related‑work section and Fig. 1 organize existing Mamba variants into “no norm / pre‑SSM / post‑SSM / both,” which is a neat, accessible summary of current practice.

## Weaknesses

### Fatal

None of the issues clearly render this “not even a paper,” but there are multiple major problems that substantially weaken the central claims.

### Major

- **1. Experimental setup and baselines are underspecified for a comparative study.**  
  The core claims rest on *relative* differences between normalization strategies, yet the paper omits key details needed to assess whether comparisons are apples‑to‑apples:
  - No description of the exact Mamba architecture used in experiments: depth, width, hidden sizes, head dimensions, or whether the block in Fig. 2 (with N1 at the input) is used verbatim in all settings, including those where only N2 is conceptually “after SSM” (Eq. 8 still references N1).
  - No training hyperparameters: optimizer, learning rate/schedule, batch size, number of epochs/steps, weight decay, dropout, etc., for Breakfast, ImageNet‑100, ListOps, or ImageNet‑1k.
  - No statement on whether any hyperparameters were tuned separately per normalization type, or shared across all methods. Given that different norms induce different effective scales, this is crucial.
  - No information on random seeds or number of runs per configuration (all tables show single numbers).
  Because the reported gains are often dramatic (e.g., 7.0%→68.8% on Breakfast; Table 1) and highly sensitive to optimization, the absence of this information goes beyond minor reproducibility nits and makes it impossible to judge whether a given normalization scheme is intrinsically better versus simply better tuned or more compatible with a fixed schedule.

- **2. Dataset protocols and metrics are not clearly defined, undermining interpretability.**  
  Section 4.1 only provides high‑level dataset descriptions. For Breakfast:
  - The paper never specifies the *task formulation*: is it frame‑wise classification, segment‑wise, or sequence‑to‑label? The tables list “Accuracy (%)” but not whether this is frame accuracy, clip accuracy, or some other metric.
  - There is no description of input representation (raw frames vs. precomputed features), temporal resolution, truncation, or padding for very long videos.
  - No information is given on train/val/test splits or evaluation protocol (e.g., original cross‑subject split vs. custom split).
  For ImageNet‑100 and ImageNet‑1k:
  - Input resolution, data augmentation, number of epochs, and optimizer are not disclosed.
  - It is not explicitly stated whether the reported numbers are top‑1 accuracy (likely, but should be clear).
  These omissions mean that the absolute numbers are hard to interpret, and it is unclear whether the observed patterns would hold under more standard, widely used protocols. This also affects the “validation” claims on ListOps and ImageNet‑1k (Sec. 4.5): we cannot ascertain that these runs match standard LRA or ImageNet training regimes, or even that training conditions are identical between “original” and “ours.”

- **3. No statistical treatment despite claims about “training stability.”**  
  The paper repeatedly frames its contribution in terms of “training stability” and “robustness” (Abstract, Introduction, Sec. 5), and the intuitive explanation section leans on notions of pathological loss landscapes and stable updates. However:
  - All quantitative results are single‑number accuracies, with no error bars, standard deviations, or indication of multiple seeds.
  - There is no reporting of failure rates (e.g., divergence, NaNs), gradient explosions, or sensitivity to initialization.
  - The L2‑norm plots (Figs. 4–5) are shown for a single 4‑layer model on ListOps and do not include any aggregate statistics across runs or architectures.
  Without repeated runs or explicit stability metrics, the evidence supports “certain configurations yield higher accuracy in these single experiments” rather than “these configurations are more stable.” This gap directly weakens the central, stability‑focused framing.

- **4. Overstated generality and lack of a concrete rule for the “combination intuition.”**  
  Contributions (2) and (3) emphasize explaining and guiding the choice of normalization combinations, and Sec. 4.6 introduces the “harmonic structure” intuition. However:
  - The main mechanistic evidence is one detailed case study (BN vs. IN vs. BN→IN on ListOps; Fig. 5) plus a few qualitative remarks about L2‑norm distributions for None/BN before vs. after SSM (Fig. 4).
  - The paper does not define a quantitative measure of “harmonic structure” (e.g., a variance or balance metric over norms or gradients across layers) or test whether such a measure predicts which N1/N2 pairs perform best.
  - There is no systematic link drawn between L2‑norm behavior and accuracy across *all* combinations in Table 4; potential counterexamples are not discussed.
  - The authors themselves note that this is “not intended as an essential explanation,” yet they use it as the main rationale for their combination recommendations.
  As a result, the work is best understood as an empirical grid search with some post‑hoc interpretations, not as a principled, generally applicable framework for normalization design in Mamba. The paper currently over‑markets its explanatory power.

- **5. Validation experiments are minimal and somewhat confusing.**  
  Table 5 is intended to show that the best‑found configurations generalize to other datasets, but:
  - On ListOps, “Original” is RMSN→SSM→RMSN (56.9%) and “Ours” is IN→SSM→LN (72.5%), which appears to reuse the best Breakfast configuration (IN→LN). This shows that one particular combination found on Breakfast also performs well on ListOps, but without comparing against the *best* single‑norm or position baselines under the *same* training setup, it does not establish that mixed norms systematically outperform other reasonable choices on ListOps.
  - On ImageNet‑1k, the reported gain is 70.8%→71.1% (LN→LN vs. RMSN→BN). Given no variance estimates and no training details, a 0.3% difference cannot be reliably interpreted as an improvement.
  - The explanatory paragraph in Sec. 4.5 is internally inconsistent: it refers to “for vision tasks, RMSN→SSM→RMSN represents the original Mamba’s normalization configuration, while IN→SSM→IN represents our proposed normalization configuration,” but Table 5’s “Ours” for vision is RMSN→SSM→BN, and “Ours” for sequence is IN→SSM→LN, not IN→IN. This mismatch suggests sloppy description of protocols.
  Overall, these validation experiments modestly suggest that some alternative normalizations can perform at least as well as the originals, but they fall short of robustly “validating” the proposed guidelines.

- **6. Limited task and model diversity relative to the scope of the claims.**  
  The main experiments focus on a single video dataset (Breakfast) and a relatively small image benchmark (ImageNet‑100), with a small 4‑layer model used for the ListOps analysis. There are no experiments on language modeling or on larger‑scale Mamba applications, despite the motivation being “long sequence modeling” broadly and even discussion of Mamba2 in the conclusion. While it is acceptable for a focused empirical paper to work at moderate scale, the discussion in the Introduction and Conclusion sometimes generalizes to “deep learning” and “large‑scale neural networks” at large. Given only modest‑scale experiments and no evidence on, for example, large language models or deep vision stacks, such broad generalization is not fully supported and should be tempered.

### Minor

- **7. Ambiguity about the exact block used in experiments.**  
  Sec. 3.1 and Fig. 2 show N1 always applied to x at the block input, with both branches consuming N1(x). The later formalizations of “before” vs. “after” SSM in Sec. 3.3 (Eqs. 7–8) and Sec. 3.4 simplify to forms without N1 in the after‑SSM case (Eq. 8), raising the question of whether the actual experimental implementation removes N1 entirely in “after only” settings or simply sets N1 to identity. This matters because the gating branch and residual interaction depend on N1 in the original block. The paper should clarify exactly what is held fixed and what is changed when comparing “before” vs. “after” configurations.
- **8. Lack of discussion of known normalization constraints.**  
  BN and IN can be sensitive to batch size, modality, and domain; GN and LN are often more robust in low‑batch regimes. Since the experiments apply all five norms on both sequence and vision tasks (including Breakfast and ListOps, which may have small effective batch sizes), a short discussion of whether any configurations required special handling (e.g., ghost BN, different ε, or batch sizes) would strengthen the interpretation of mixed results such as IN performing relatively poorly in some placements.
- **9. Multiple‑comparison issues not acknowledged.**  
  Table 4 tests 25 combinations per task. When searching over such a large grid with single runs per cell and no cross‑validation, some of the top configurations may benefit from chance. The paper does not acknowledge this or attempt simple robustness checks (e.g., re‑running top‑k combinations with multiple seeds).

### Trivial

- Minor inconsistencies and typos in references and phrasing (e.g., “Ju & Zhou, 2024:?”, “institution” instead of “intuition,” some repeated or duplicated figure captions). These do not affect the technical content.

## Nice-to-Haves

- Experiments explicitly targeting training dynamics: e.g., loss and gradient‑norm curves, convergence speed, and learning‑rate sensitivity for key configurations. This would directly support the training‑stability narrative.
- Additional analyses connecting L2‑norm statistics to performance across a broader set of normalization pairs: plotting accuracy vs. some function of the weight‑norm distribution across layers for multiple configurations could test whether the “harmonic” story has predictive value.
- Larger‑scale or more diverse tasks (e.g., at least one language modeling benchmark or a larger/deeper vision model) to probe whether the discovered patterns are consistent in the high‑capacity regimes where Mamba is often used.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **“Baselines may use non‑existent or unavailable models / configurations.”**  
  Any concern phrased as doubting the existence or availability of “original Mamba,” “VMamba,” or other cited architectures and datasets cannot be upheld, as all cited entities are assumed to exist and be available. The real issue is the lack of clear description of how those baselines were instantiated in this paper’s experiments, which is already captured under the major weaknesses.
- **“The authors fail to compare against specific missing related works or named techniques.”**  
  While comparisons to other stabilization methods (e.g., norm‑free training, specific named normalizations) could be interesting, we cannot assert that any particular method is “missing” from related work given limited external knowledge. The key, kept criticism is that they do not compare normalization against *non‑normalization* stabilization strategies in general, not that specific citations are absent.
- **“Validation experiments are invalid because the referenced original works might not have used these exact blocks or datasets.”**  
  The issue is not that previous works are untrustworthy or mismatched per se, but that this paper does not clearly explain how it re‑implements “original Mamba” or “VMamba” within its own framework. Doubts about external reproducibility or availability of prior models are removed.

## Novel Insights

The most genuinely novel insight, relative to existing Mamba work, is the empirical pattern that (i) adding normalization after the SSM module strongly reduces inter‑layer disparities in weight L2‑norms and can substantially boost performance over no or pre‑SSM normalization, and (ii) certain heterogeneous normalization pairs (e.g., IN→LN for sequence, RMSN→BN for images) outperform homogeneous ones for this specific Mamba block. Although the mechanistic explanation is limited, the combination of broad empirical sweeps across normalization types/positions and preliminary norm‑based analysis appears to be new for Mamba and could serve as a concrete starting point for more rigorous studies of training dynamics in SSM‑based architectures.

## Suggestions

- **Clarify and fully specify experimental setups.**  
  Add a dedicated “Implementation Details” subsection including:
  - Exact model architectures (layer counts, hidden sizes, where N1 and N2 sit in Fig. 2) used for Breakfast, ImageNet‑100, ListOps, and ImageNet‑1k.
  - Training hyperparameters for each dataset: optimizer, learning rate and schedule, batch size, epochs/steps, weight decay, dropout, warmup, etc.
  - Whether hyperparameters were shared across all normalization configurations or tuned per configuration; if shared, justify that choice.
- **Introduce basic statistical rigor.**  
  For at least the key configurations (e.g., None→None, best single‑norm, best “after‑only,” best combination per task), run 3–5 seeds and report mean ± standard deviation. Re‑evaluate whether the claimed improvements (especially the small 0.3% gain on ImageNet‑1k) are statistically meaningful.
- **Narrow and sharpen the claims.**  
  Rephrase contributions and conclusions to emphasize what is firmly supported: e.g., “For our chosen Mamba block and training regimes on Breakfast and ImageNet‑100, after‑SSM normalization and certain heterogeneous combinations yield consistent accuracy gains.” Avoid broad statements about “deep architectures” or universal stability guidelines unless backed by more diverse evidence.
- **Make the “combination intuition” more concrete or more clearly tentative.**  
  Either (a) define and evaluate a simple quantitative metric of “harmonization” (e.g., variance of layer‑wise weight norms) and test its correlation with accuracy across many configurations, or (b) explicitly label the L2‑norm discussion as speculative interpretation, decoupled from the main empirical claims, and reduce its prominence in the stated contributions.
- **Tighten and clarify the validation experiments.**  
  In Sec. 4.5:
  - Ensure that the textual description of “original” vs. “ours” exactly matches Table 5.
  - Clearly state whether you re‑trained the baselines under your codebase and hyperparameters, or simply changed normalization in an otherwise fixed implementation.
  - Where possible, compare your best combination not just against the default normalization of the original model, but against the best same‑norm and best single‑position settings you found under the same training regime.
- **If space permits, add at least one larger or more canonical benchmark.**  
  Even modest‑scale language modeling (e.g., WikiText‑103) or a deeper vision backbone on ImageNet‑100 would make the claimed generality more convincing and help disentangle sequence‑length effects from dataset idiosyncrasies.

In terms of originality, the paper is incremental but addresses a real gap (systematic normalization study in Mamba). The research question (how to normalize Mamba blocks) is practically important. However, the claims about best practices and stability are currently only partially supported: experiments are reasonably thorough in grid coverage but underspecified and statistically thin; the mechanistic story is suggestive but not rigorous. Clarity of writing is generally good, but technical and experimental clarity need strengthening. As it stands, the value to the community is that of a preliminary empirical ablation study rather than a mature, prescriptive guideline.

## Score and Decision

To calibrate, I considered several human‑reviewed papers:

- **“Beyond Standardization – Putting the Normality in Normalization”** (`9ut3QBscB0.md`), an empirical normalization paper with mixed reviews (3–8) and ultimately rejected. It had deeper normalization‑theoretic ambition but similar concerns about scale and evidence.
- **“Architecturally Aligned Comparisons Between ConvNets And Vision Mambas”** (`QBiFoWQp3n.md`), which, like this paper, is primarily an empirical architectural comparison; it was rejected with mostly 5s, partly due to limited scope and interpretability.
- **“Methods of Improving LLM Training Stability”** (`RL6R5ryuL5.md`), an empirical stability paper that lacked strong statistical analysis and large‑scale validation, receiving scores around 3–5 and being treated as rejected.

Relative to these, the current paper is somewhat more focused and cleaner than the weakest of them, but still below the threshold of those that were even borderline. The missing experimental specification and statistical treatment are serious for a paper whose entire value is empirical comparison and whose framing revolves around “stability” and “guidelines.”

I therefore place this paper around the lower‑middle of the scale: interesting and potentially useful as a workshop or exploratory submission, but not strong enough for acceptance in its current form.

MY FINAL SCORE: <pineapple>4.5</pineapple>  
MY FINAL DECISION: <orange>Reject</orange>