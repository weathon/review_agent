Now I have sufficient calibration data. Let me synthesize the final review.

**Calibration summary:**
- Highly related papers (text-informed TSF): mfc6FKgtQA (TGTSF, scores 6/3/6/5, withdrawn→reject), xW4J2QlqRx (ContextFormer, scores 5/5/5/5, reject), QE1ClsZjOQ (Dual-Forecaster, scores 5/5/5/3, withdrawn→reject), 4F1a8nNFGK (CiK benchmark, scores 5/6/6/3, reject), 01wMplF8TL (TITSP, scores 5/5/3/5, reject), NJqsHgxcKh (MetaTST, scores 6/5/3, withdrawn→reject). These all cluster around 3-6, rejected.
- Strong accepted TSF papers: bWcnvZ3qMb (FITS, scores 8/8/8/8, spotlight), Unb5CVPtae (Time-LLM, scores 8/8/8/8/3, poster), 4NhMhElWqP (DAM, scores 8/6/6/8, poster). These have genuinely novel insights, fair comparisons, and strong empirical evidence.

Now let me verify the key claims from the harsh reviewer against the paper:

1. **Overstated "hard barrier" theorem** - The paper does claim "a hard, mathematical barrier on forecasting accuracy" (abstract) and "a theoretical performance ceiling for conventional TSF approaches" (Sec 2.2). Proposition 2.1 states a standard result: conditional expectation is the best predictor under missing information, with irreducible error = BΣB^T. This IS a standard result from control theory/statistics, repackaged. The overclaim is real but the paper's contribution isn't purely theoretical - the framing is still useful.

2. **Unfair baseline comparison** - The paper explicitly compares FIATS (which gets U_f) against baselines that don't get U_f. This IS a crucial weakness - the paper never tests whether a standard model given the same influence information (even as simple features) could match or beat FIATS. This is consistently flagged in similar papers and is a valid structural concern.

3. **Missing covariate-aware baselines** - TimeXer and ChronosX are cited in the paper but not included as baselines. This is a fair criticism.

4. **FM Toy is circular** - The synthetic dataset is literally designed so that influence-aware models should achieve near-zero error. Showing this proves the dataset was designed correctly, not a general claim about TSF.

5. **Leakage concerns** - The paper discusses leak-free design in Section 4.1 but details are deferred to appendices. The weather/Atmospheric Physics relationship is potentially problematic since weather descriptions may encode information about the target variables themselves.

Let me now write the final review.

## Summary

This paper identifies the "self-stimulation" assumption in time series forecasting—predicting the future from historical observations alone—as a performance barrier. Through control-theoretic analysis, the authors formalize that ignoring external influences creates an irreducible error floor (Proposition 2.1) and that incorporating measurable influences reduces this bound (Proposition 3.1). To operationalize the proposed Influence-Aware Time Series Forecasting (IATSF) paradigm, they introduce a leak-free, temporally-synced benchmark with textual influences and develop FIATS, a lightweight LLM-free model with channel-aware mechanisms (CASM, CAPS) that explicitly models how external influences differentially affect each channel.

## Strengths

- **Well-motivated problem framing**: The paper correctly emphasizes that many real-world time series are driven by exogenous events and that standard models ignoring these influences produce "averaged-out" forecasts. The control-theoretic lens provides a clean, if not entirely novel, formalization of this limitation, making the argument more precise than purely empirical observations.

- **Thoughtful benchmark design philosophy**: The IATSF benchmark's emphasis on independently evolving influences, temporal synchronization, and leak-free construction addresses real methodological gaps in existing multimodal TSF datasets. The diversity of domains (toy systems, atmospheric physics, traffic, game analytics) is valuable.

- **Large empirical gains**: FIATS achieves dramatic improvements—near-zero error on FM Toy (vs. 0.136–0.909 for baselines), 36% MSE reduction on Atmospheric Physics, and 44.3% on NYC Traffic Speed. The scale of these improvements demands attention, even if the attribution is contested.

- **Interpretable architecture with theoretical alignment**: The CASM mechanism's design—using channel descriptions as queries and influence text as keys in cross-attention—is motivated by the control-theoretic analysis of channel-specific sensitivities (∇_U F). Attention maps provide direct interpretability of which influences matter for which channels, which is a meaningful design choice beyond ad-hoc concatenation.

- **LLM-free design choice**: By avoiding LLMs and using pre-trained text embeddings directly, FIATS keeps the model lightweight and makes it easier to attribute gains to influence modeling rather than model capacity.

## Weaknesses

### Major:

- **Unfair empirical comparison: FIATS receives influence information that all baselines are denied**. The central experimental claim—that IATSF dramatically outperforms existing approaches—rests on comparing FIATS (which receives future-aligned textual influences U_f) against models that receive only historical time series X_h. This answers the trivial question "does extra information help?" rather than "is influence-aware modeling paradigmatically superior?" In practice, weather forecasts, holidays, and scheduled events are available to any model and could be injected as exogenous covariates. Without at least one baseline that receives the same influence information (e.g., PatchTST or DLinear with text embeddings concatenated as extra channels, or existing exogenous-variable models like TimeXer/ChronosX), the 36–44% MSE reductions cannot be attributed to the IATSF paradigm or FIATS's architecture. This is the same critical flaw that reviewers identified in ContextFormer (xW4J2QlqRx, scores 5/5/5/5, rejected): "This paper should be compared with these frameworks that utilize external information, rather than solely with models that lack auxiliary information."

- **Overstated theoretical contribution**: Propositions 2.1 and 3.1 are standard results repackaged as breakthroughs. Proposition 2.1 states that the Bayes-optimal predictor using only X_h predicts the conditional expectation E_U[F(X_h,U)], with irreducible error BΣB^T—this is a textbook consequence of conditional variance decomposition under missing inputs, not a "hard, mathematical barrier" unique to TSF. Proposition 3.1 states that conditioning on any measured influence reduces conditional variance—this is the law of total variance. The paper presents these as discovering a "universally adopted yet flawed assumption" and proving a "theoretical performance ceiling," but the theory only says: if you have extra informative variables, incorporating them helps. This does not uniquely justify textual influences or the IATSF paradigm; any informative covariate would suffice. The "self-stimulation barrier" framing overclaims what the mathematics actually delivers.

- **FM Toy validation is circular**: The FM Toy dataset is explicitly designed as a system "where influences precisely control signal frequency, offering a theoretical error bound of zero for a perfect model." Demonstrating that FIATS achieves near-zero error on a dataset engineered to make influence-aware models succeed and self-stimulated models fail confirms the dataset was constructed correctly—it does not validate the broader claim that "the performance bottleneck is indeed the flawed self-stimulation assumption, not model scale" for real-world forecasting.

### Minor:

- **Insufficient architectural ablations**: The paper claims gains stem from CASM and CAPS specifically, but the only ablations (Table 3) remove entire information sources ("Zero News," "Zero Desc."). There is no comparison against a standard cross-attention mechanism (without CASM's channel-description-driven queries) or against simpler information injection (e.g., concatenated text embeddings as covariates). Without these, the attribution of gains to CASM/CAPS rather than to simply having influence information is unsupported.

- **Leak-freeness of the benchmark is asserted but not rigorously verified**: Section 4.1 claims leak-free design, but crucial details are deferred to appendices. For Atmospheric Physics, weather forecasts describe the same physical system as the target variables (solar radiation, air pressure are components of weather)—the distinction between "independent influence" and "future system state encoded in text" is not formally established. The paper would benefit from a more explicit leakage audit (e.g., showing that text embeddings do not contain excessive information about target variables beyond what genuine forecasts would provide).

- **Proposition 3.1's applicability to textual influences is loose**: The proposition assumes known covariance Σ_j and accurate observation of U_j. Textual influences are noisy, lossy, potentially biased observations of the true driving process. The paper states "any measurable influence information reduces forecasting uncertainty, even with incomplete influence knowledge" but this is too strong—poorly calibrated or adversarial influence signals could increase error. The theoretical-empirical bridge for text specifically is hand-waved.

- **No standard deviations reported**: Experimental results report only point MSE values with no error bars across multiple runs, making statistical significance impossible to assess—a practice that several reviewers of similar papers (QE1ClsZjOQ) have flagged.

### Trivial:

- The paper acknowledges delayed influences and chaotic dynamics as limitations, but does not discuss overfitting risk when leveraging rich text for relatively small numeric datasets (especially relevant for GAUD with 90 games).

## Nice-to-Haves

- A controlled experiment where PatchTST or DLinear receives the same text embeddings as additional input channels, creating a fair "same information, different architecture" comparison. This single addition would dramatically strengthen the paper.

- Evaluation of FIATS on standard TSF benchmarks (ETT, Weather) without influence inputs to show it does not degrade when influence information is unavailable.

- A quantitative evaluation of CASM's interpretability claims: on FM Toy (where ∇_U F is known), compare learned attention weights to ground-truth system sensitivities.

## Removed Points

These points were flagged for removal—treat them with caution:

- **"The theorem does not support the claimed uniqueness of the proposed paradigm"** (from Harsh Critic #1): While the overclaim is real, this specific critique is partially addressed by the paper—the paper's contribution is not just the theorem but the operationalization (benchmark + model). The theorem itself is indeed standard, which is captured in the "overstated theoretical contribution" weakness above.

- **"Incorporating pre-defined exogenous variables is a step forward... but this approach often lacks flexibility"** was raised by reviewers of similar papers as a defense against "exogenous variables already exist." The paper does discuss this distinction (Sec. 3.2) but the argument is qualitative, not empirically demonstrated. Kept as minor weakness about insufficient ablations.

- **"Standard deviations not reported"**: Included as a minor weakness per community norms, but per soft rules, single-run evaluation is common in large-scale TSF benchmarks, so this is downweighted.

- **"Missing related works"** (e.g., UniTime, TFT): Per hard rules, I cannot confirm these exist or are relevant, so removed.

- **"Parameter count comparison"** from Harsh Critic section notes: The paper claims "lightweight" but doesn't provide parameter counts. This is a minor gap but not structural—FIATS is described as using standard patch encoding + cross-attention, which is indeed lightweight relative to LLM-based methods. Not a major issue.

- **"GAUD is proprietary/not reproducible"**: The paper states data is properly anonymized and provides a code link. Per hard rules, we accept cited/linked resources exist. Removed.

## Novel Insights

The paper's most important insight is not the theoretical formalism (which is standard) but the empirical observation that even billion-parameter foundation models (Chronos-L, MOIRAI-L, Time-MoE-U) fail catastrophically on systems with strong exogenous drivers—yet are not given access to those drivers. This reframes the "foundation models vs. linear baselines" debate: perhaps the plateau isn't about model architecture at all, but about missing the right inputs. However, this insight is undermined by the lack of fair comparisons where those same foundation models receive the influence information.

## Suggestions

1. **Add a "same information" baseline**: Give PatchTST or DLinear the text embeddings (or simple weather categorical features) as additional input channels. This is the single most critical experiment for the paper's credibility. If FIATS still wins, the architectural claims are validated; if it doesn't, the contribution narrows to "provide the right data."

2. **Soften the theoretical framing**: Replace "hard, mathematical barrier" with "irreducible error under missing exogenous information"—this is accurate, honest, and still motivating. Similarly, replace "the primary path forward" with "a critical and under-explored path forward."

3. **Include TimeXer or ChronosX as baselines**: Both support exogenous variables and are directly cited in the paper. Their absence is conspicuous.

## Score and Decision

**Calibration comparison:**
- Most directly comparable papers (text-informed TSF): mfc6FKgtQA (6/3/6/5, withdrawn→reject), xW4J2QlqRx (5/5/5/5, reject), QE1ClsZjOQ (5/5/5/3, withdrawn→reject), 4F1a8nNFGK (5/6/6/3, reject), 01wMplF8TL (5/5/3/5, reject), NJqsHgxcKh (6/5/3, withdrawn→reject). Common fatal flaw: unfair baselines (comparing with-influence vs. without-influence), overclaims, information leakage concerns.
- This paper shares the same core weakness as ContextFormer (xW4J2QlqRx): comparing a model with extra information against models denied that information. Reviewers uniformly scored those papers 3-6 and rejected them.
- This paper is somewhat stronger than those comparators in having (a) explicit theoretical motivation (even if overstated), (b) a more carefully designed benchmark with explicit leak-free philosophy, and (c) a more interpretable architecture. But it is weaker in having even more extreme overclaims ("primary path forward," "hard mathematical barrier") and the circular FM Toy validation.
- Relative to accepted papers in TSF (FITS at 8s, Time-LLM at 8s), this paper lacks the crisp, well-isolated contribution and fair experimental setup that characterize those works.

This paper sits at the higher end of the rejected text-informed TSF cluster—it has genuine value in benchmark design and interpretability—but the unfair comparison setup and overclaimed theory prevent accepting the paradigm argument as currently presented.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>