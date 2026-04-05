=== CALIBRATION EXAMPLE 76 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is accurate and appropriately specific: the paper is about fragility of contamination detection in reasoning models, not contamination in general.
- The abstract clearly states the problem, two scenarios, and the headline empirical claims. It also makes the central thesis understandable: existing contamination detectors can be evaded in LRMs.
- The main concern is that the abstract is somewhat stronger than the evidence shown in the paper in two places: it says a “broad class of RL methods may inherently exhibit similar concealment capability,” but the experiments and theory mainly support PPO-style clipping/importance-sampling objectives, not a broad class of RL methods in full generality. It also says detection methods perform “near random guesses” in Stage II; this is true for many methods in the reported benchmarks, but not uniformly across all methods/bases in Table 5 and Appendix tables.

### Introduction & Motivation
- The problem is well-motivated and timely for ICLR standards: benchmark contamination in LLM/LRM evaluation is a real and important issue, especially given the importance of public reasoning leaderboards.
- The gap in prior work is reasonably identified: prior contamination detection literature focuses on memorization-style detection and not on reasoning models with hidden CoT or post-SFT/RL training dynamics.
- The contributions are stated clearly, and the two-stage framing is useful.
- However, the introduction somewhat over-extends the generality of the claim. The paper presents its findings as a fundamental vulnerability of LRM evaluation, but the evidence is primarily for specific open-source math/science reasoning benchmarks, specific base models, and specific RL/SFT recipes. The paper should more carefully delimit that scope in the introduction.
- A second issue is that the introduction treats “contamination detection methods” as a fairly unified target, but the paper later shows their behavior depends strongly on whether the detector sees questions, responses, reasoning tokens, or non-reasoning tokens. That nuance should be signposted earlier.

### Method / Approach
- The methodological setup is understandable overall: two contamination scenarios, 10 detectors, AUROC on member vs. non-member splits, and controlled SFT/RL pipelines.
- The Stage I and Stage II definitions are sensible and practically relevant.
- The main reproducibility concern is that the contamination protocol is not fully specified at the level needed to interpret the claims robustly. For example:
  - In Stage I, members are formed by duplicating contaminated examples and then RL is applied, but the exact relationship among contaminated SFT samples, RL prompts, and reward generation is not fully transparent in the main text.
  - The claim that GRPO is the “root cause” of concealment is stronger than what the setup fully isolates. The paper compares GRPO, RAFT, and RAFT++, but these differ in more than just clipping/importance sampling, and the theoretical analysis abstracts several details away.
- The theoretical section is directionally plausible but not fully convincing as a formal explanation:
  - Theorem 3.1 is stated for a “small natural gradient step” with a PPO-style loss, but the assumptions needed for the decomposition are quite idealized.
  - The derivation relies on a tabular setting and on simplifying assumptions about reward structure, state-action covariances, and the behavior of members vs. non-members. It is useful as intuition, but it does not yet fully justify the strong causal claim that clipping is the “root cause.”
  - In particular, the theory seems to explain contraction of NLL gaps under the optimization objective, but the link from that contraction to the empirical behavior of specific detectors is indirect and detector-dependent.
- There is a potential mismatch between theory and some empirical comparisons:
  - The paper argues RAFT does not conceal contamination while RAFT++ and GRPO do, and attributes this mainly to clipping/importance sampling.
  - But RAFT and RAFT++ differ not only by clipping; the way positive samples are chosen and weighted is also important, and the experiments should more explicitly separate these factors.
- Edge cases/failure modes are under-discussed:
  - The analysis does not fully address whether concealment depends on benchmark format, output length, verifier quality, or prompt template.
  - It would be important to know when RL might actually increase detectability rather than reduce it, since some tables in the appendix show detector behavior is not uniformly monotone across all methods/benchmarks.
- The method description is mostly adequate for a paper of this type, but the causal claims need tighter qualification.

### Experiments & Results
- The experiments are directly aimed at the paper’s claims, which is a major strength. The two-stage setup, multiple base models, multiple benchmarks, and multiple detection families are all well chosen for the question being studied.
- The use of 10 representative detectors is a strong point, as it reduces the chance that the result is an artifact of one detector family.
- The results do support the main qualitative claim:
  - Stage I: SFT contamination is often detectable initially, and RL training reduces AUROC substantially.
  - Stage II: extensive final-stage SFT contamination on advanced LRMs leaves much weaker evidence for existing detectors.
- That said, several limitations matter for an ICLR bar:
  1. **Baseline fairness and detector dependence.**  
     Some detectors are evaluated in settings that may favor or disfavor them depending on access to response tokens, reasoning tokens, or references. The paper does a good job in Appendix E.2 comparing question-only, question+response, reasoning-only, and non-reasoning-only signals, but the main text still risks implying all detectors are equally and fairly exposed to their best available inputs. Some detectors, especially reference-based ones, may be disadvantaged by the practical constraints of LRM contamination, so the paper should be more explicit that this is a realistic but asymmetric threat model, not a universal benchmark of detector quality.
  2. **Statistical robustness.**  
     The paper reports AUROC averages, but there are no error bars, confidence intervals, or significance tests. Given that many reported differences are moderate and some appendix results are noisy, the lack of uncertainty quantification weakens the strength of the claims.
  3. **Ablations.**  
     The main ablation around clipping vs. no clipping is useful, but more ablations would materially strengthen the conclusion:
     - varying the KL term separately from clipping,
     - varying reward sparsity / verifier strength,
     - varying number of contaminated members,
     - varying the degree of overlap between contaminated and non-member distributions,
     - comparing different RL algorithms beyond RAFT/RAFT++/GRPO.
     The current set is informative but not enough to claim a broad algorithmic phenomenon.
  4. **Potential confounding by model generalization.**  
     The paper correctly notes that generalization may confound memorization-based detectors, especially in Stage II. However, the experiments do not fully disentangle “contamination concealment” from “the model genuinely generalizes from contaminated CoT to similar unseen problems.” This is important because if the model is not memorizing in the detector’s sense, then detector failure may reflect an assumption mismatch rather than concealment per se.
  5. **Dataset scope.**  
     The strongest evidence is on math/science reasoning benchmarks and a smaller set of coding/general QA tasks in the appendix. The broader claims about LRMs and leaderboards are plausible, but the evidence base is still limited.
- Overall, the experiments are substantial and likely to be viewed positively by ICLR reviewers, but the paper would benefit from more careful uncertainty reporting and stronger isolation of confounders.

### Writing & Clarity
- The paper’s high-level argument is clear, and the stage-based organization helps readability.
- The main clarity issue is not grammar but conceptual precision:
  - The distinction between “contamination,” “memorization,” “generalization,” and “concealment” is important but not always cleanly separated.
  - Theoretical symbols are introduced in a dense way, and Theorem 3.1 is hard to parse without substantial effort. The result is likely correct as an intuition device, but the presentation makes it difficult to verify the assumptions and scope.
- Figures and tables mostly serve their purpose, especially the AUROC trend figures and the log-probability distribution plots.
- However, some claims in the text are only loosely supported by the figures, particularly statements about “root cause” and “eventually near-random performance.” The figures show a trend, but the phrasing can overshoot the direct evidence.
- The appendix adds useful detail, but the main paper relies heavily on it for key experimental design choices. For an ICLR audience, that is acceptable only if the main text more explicitly summarizes those design decisions and limitations.

### Limitations & Broader Impact
- The paper does include a limitations section, but it is very brief relative to the strength of the claims.
- The main limitation it acknowledges is that the paper does not propose a new defense, only exposes weaknesses of existing detectors. That is fine.
- Important limitations that are only partially acknowledged or not fully developed:
  - The findings are benchmark- and setup-dependent.
  - The post-LRM contamination result depends on access to long CoT-style outputs and the specific choice of advanced LRMs.
  - The theory does not yet prove that all PPO-style RL methods will conceal contamination; it mostly supports a subset of objectives under simplifying assumptions.
  - The paper does not sufficiently discuss false positives/negatives in a deployment setting where detectors may have limited access to references or training distributions.
- Broader impact is reasonable and important: the paper points to real risks for public reasoning leaderboards. That said, the ethics statement is too minimal for such a security/evaluation paper. The work is dual-use in the sense that it identifies how contamination can evade detection. The authors should discuss whether publishing these findings could facilitate misuse, and what safeguards or evaluation practices they recommend.
- ICLR typically values clear articulation of limitations and societal risks. The paper would be stronger if it more explicitly framed these as evaluation-security issues with concrete mitigation directions.

### Overall Assessment
This is a timely and empirically substantial paper with a clear ICLR-relevant message: contamination detection methods that work reasonably well for standard LLM settings can fail badly in reasoning-model pipelines, especially after RL or when contamination includes CoT. The experimental breadth is a major strength, and the two-stage framing is compelling. My main reservation is that the central claims are somewhat broader than the evidence fully justifies: the theory is suggestive rather than definitive, the “broad class of RL methods” claim is stronger than what is directly shown, and the Stage II result may partly reflect a memorization-versus-generalization mismatch rather than pure concealment. I would still regard the contribution as meaningful and likely interesting to the ICLR community, but it needs tighter qualification, better uncertainty reporting, and a more careful separation of concealment from generalization to fully clear ICLR’s bar for strong mechanistic and empirical claims.

# Neutral Reviewer
## Balanced Review

### Summary
This paper studies benchmark contamination detection in large reasoning models (LRMs) and argues that existing detectors are fragile in two realistic contamination scenarios: contamination introduced during SFT and then “concealed” by subsequent RL, and contamination applied as a final SFT stage to an advanced LRM with CoT. The main empirical claim is that many standard detectors drop from detectable AUROC to near-random performance, while the paper also provides a theoretical explanation tying the concealment effect to PPO-style importance sampling and clipping.

### Strengths
1. **Timely and relevant problem for ICLR.**  
   The paper addresses a concrete evaluation integrity issue that is highly relevant to current LRM development and benchmark-driven model selection. This is the kind of systems-and-evaluation concern ICLR often values, especially because the paper is about how training choices affect benchmark trustworthiness.

2. **Covers two practically important contamination regimes.**  
   The distinction between “pre-LRM” contamination during SFT→RL transitions and “post-LRM” contamination as a final SFT step is useful and well-motivated. This framing goes beyond a single narrow setting and helps connect the findings to multiple developer workflows.

3. **Broad empirical scope across detectors and models.**  
   The paper evaluates 10 detector types spanning generation-based, perturbation-based, reference-based, embedding-based, and reference-free methods, on multiple benchmarks and several base/advanced models. This breadth strengthens the claim that the issue is not isolated to one detector or one model family.

4. **Evidence that RL can reduce detector separability without eliminating performance gains.**  
   The results suggest an important and nontrivial phenomenon: subsequent GRPO training reduces AUROC while benchmark performance remains inflated. This is a substantive observation because it undermines a naive assumption that “more training” would erase contamination and thus preserve detectability.

5. **Attempts at mechanistic and theoretical explanation.**  
   The paper does not stop at observation; it argues that PPO-style clipping/importance sampling is the key driver of concealment and supports this with ablations comparing GRPO, RAFT, and RAFT++ variants. For ICLR, this combination of empirical pattern plus mechanism is a meaningful strength.

6. **Includes reproducibility-oriented details.**  
   The paper provides code, implementation notes, GPU resources, model choices, and many appendix results. While not sufficient on its own, this is better than a purely high-level empirical claim.

### Weaknesses
1. **The core contribution is more a negative result than a new method, and its practical impact is somewhat limited by that framing.**  
   The paper convincingly shows that several existing detectors fail under certain LRM contamination settings, but it does not introduce a new detector, benchmark protocol, or defense. For ICLR, negative findings can be valuable, but the acceptance bar usually expects either a clearly generalizable scientific insight or a strong methodological advance. The paper’s main takeaway is important, yet the “what should replace these methods?” part remains underdeveloped.

2. **The theoretical analysis appears simplified and may not fully justify the broad claims.**  
   The proof focuses on an idealized/natural-gradient-style analysis and makes simplifying assumptions about tabular settings and advantage structure. That can be fine as intuition, but the paper extrapolates from this to broad statements about a wide class of RL methods. The empirical evidence supports some of this, but the theory as written does not fully establish the generality claimed in the abstract.

3. **The causal story about clipping vs. importance sampling is not fully nailed down.**  
   The ablations suggest clipping is important, but the paper also discusses importance sampling and PPO-style objectives broadly. The experimental separation between these components seems incomplete, and some conclusions read stronger than the controlled evidence strictly supports. In particular, it is not fully clear whether clipping is the dominant factor in all settings or whether interactions with advantage normalization, rollout selection, and output entropy also matter.

4. **Potential confounds in the contamination-detection evaluation remain.**  
   The paper compares member and non-member halves of the same benchmark, but many detector families are known to be sensitive to response length, prompting, tokenization, and output format. The appendix helps somewhat, yet the evaluation still seems heavily tied to a specific choice of response generation setup, 8 rollouts, and token-level scoring. This makes it harder to know how robust the conclusions are to alternative inference budgets or prompting conventions.

5. **The “post-LRM” conclusion that contamination “barely leaves evidence” may overstate the evidence.**  
   Several detectors are indeed near chance, but some methods and some benchmarks still show above-random AUROC in places. The paper’s language is stronger than the tables sometimes justify. For an ICLR audience, a more calibrated claim would improve credibility.

6. **Limited discussion of false positives, calibration, and thresholded decision utility.**  
   AUROC is appropriate, but real contamination auditing also depends on calibration, operating points, and practical error rates. The paper focuses almost entirely on AUROC, so it is unclear how actionable these detectors are in realistic audit workflows.

7. **Reproducibility is helped by the appendix, but there are still missing details for exact replication.**  
   The paper lists many hyperparameters, yet some components remain under-specified at the level ICLR reviewers often expect for empirical reproducibility: exact detector implementations/threshold choices, confidence intervals or variance across seeds, and the full experimental protocol for some of the more complex pipelines.

### Novelty & Significance
**Novelty: Moderate to good.** The paper’s novelty lies less in a new algorithm and more in identifying a previously underexplored failure mode for benchmark contamination detection in LRMs, especially the role of RL in obscuring contamination evidence. The stage-based framing is useful and timely, and the combination of empirical failure cases plus a mechanistic explanation is more novel than a simple benchmark replication.

**Significance: Good if the claims hold robustly.** The work matters because benchmark contamination is central to trustworthy LRM evaluation, and the paper suggests that common auditing tools may be insufficient precisely in the regimes where leaderboard incentives are strongest. That is significant for the ICLR community, but the paper would be stronger if it more carefully delimited the scope of the claims and better distinguished broad impossibility from demonstrated fragility in the studied setups.

**Clarity: Mixed.** The high-level narrative is clear, but the presentation is dense and sometimes overclaims relative to the evidence. The appendix is extensive, which helps, but the main paper would benefit from sharper definitions, clearer separation of empirical findings from theory, and more restrained language.

**Reproducibility: Fair to good.** The paper provides code, multiple models, datasets, and extensive appendix details. However, the large number of experimental branches, some under-specified detector details, and limited uncertainty reporting reduce full reproducibility confidence.

**Overall ICLR significance assessment:** borderline-to-strong if the empirical results are robust and the claims are narrowed appropriately. It is a timely evaluation paper with meaningful implications, but it needs stronger methodological rigor and more careful claim calibration to meet ICLR’s typical standard for broad conclusions.

### Suggestions for Improvement
1. **Tighten the claims and distinguish demonstrated fragility from general impossibility.**  
   Rephrase broad statements such as “most detection methods perform near random guesses” to specify exactly which benchmarks, detectors, and model families exhibit this behavior. Avoid implying universal failure unless supported by stronger evidence.

2. **Add more rigorous uncertainty reporting.**  
   Report confidence intervals, standard deviations across random seeds, and, where feasible, hypothesis tests for AUROC differences. This would make the detector degradation claims substantially more convincing.

3. **Strengthen the causal analysis of RL objective components.**  
   Include cleaner ablations that separately vary clipping, importance sampling, advantage normalization, KL regularization, and rollout selection. This would better support the claim that clipping is the primary driver of concealment.

4. **Expand the evaluation beyond AUROC.**  
   Add calibration curves, precision/recall at realistic operating points, and perhaps a fixed-budget audit setting. This would make the work more relevant to practical contamination auditing rather than only ranking detectors by AUROC.

5. **Clarify the relation between memorization and generalization.**  
   The post-LRM story suggests that detectors fail because contaminated LRMs generalize to non-members with similar distributions. This is an interesting hypothesis, but it needs a sharper experimental test, ideally with explicit distribution-shift controls or matched non-member sets.

6. **Improve the presentation of the theory.**  
   Move the most technical derivation to the appendix and provide a simpler main-text statement of what is proved, under what assumptions, and what is only heuristic. This would make the contribution more accessible and prevent overinterpretation.

7. **Include stronger baselines for audit protocols.**  
   Since the paper argues that current detection is fragile, it would be useful to evaluate more robust alternatives or at least additional sanity-check procedures, such as hidden checkpoint auditing, prompt randomization, or cross-benchmark transfer tests.

8. **Provide a more actionable defense proposal.**  
   The paper’s suggested remedies are reasonable but high-level. A stronger contribution would include at least one concrete protocol or detector enhancement that partially mitigates the identified fragility.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a contamination-detection evaluation on at least one non-reasoning, non-CoT benchmark family with the same training recipe. ICLR will expect evidence that the fragility is specific to reasoning-model pipelines, not just a generic effect of further fine-tuning or stronger models.

2. Compare against stronger, more recent contamination detectors and black-box membership/sequence-level methods beyond the chosen 10, especially methods designed for generalization-aware detection. Without this, the claim that “existing detection methods” broadly fail is overstated relative to the actual baseline set.

3. Test whether the concealment survives across multiple RL algorithms and reward formulations, not just GRPO/RAFT/RAFT++. The paper’s core claim is about a broad class of RL objectives, so it needs evidence beyond one PPO-like family and one setting with clipping.

4. Add matched-control experiments with the same number of clean updates, same token budget, and same output-length distribution but without the RL objective. Right now the paper argues clipping/importance sampling is the driver, but the causal claim is not fully isolated from training duration, distribution shift, or entropy changes.

5. Evaluate on truly held-out, independently sourced contamination sets or synthetic benchmarks with known training membership labels. Using half-splits of the same benchmark can miss leakage effects and makes the “near random guess” conclusion less convincing for real leaderboard contamination.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify how much of the AUROC drop is due to reduced separability versus reduced detector calibration under distribution shift. ICLR reviewers will want to know whether detectors fail because contamination evidence is erased or because their thresholds/features break after RL.

2. Analyze the relationship between contamination concealment and answer correctness/generalization on member and non-member samples. The paper claims contamination is “concealed” while performance remains inflated, but it does not disentangle memorization, genuine reasoning improvement, and benchmark-specific overfitting.

3. Provide statistical significance and variance across seeds for the main AUROC and pass@1 results. Given the modest sample sizes on some benchmarks, the paper needs confidence intervals or multiple runs to support claims of monotonic decline and near-random performance.

4. Examine whether the effect depends on sample length, answer format, or benchmark difficulty. The post-LRM claim hinges on long CoT and “distributionally similar” non-members, but there is no controlled analysis showing which sample properties actually drive detector failure.

5. Test whether the observed log-prob convergence also appears for non-member prompts from nearby but distinct distributions. This is necessary to support the claim that generalization, not memorization, explains detector failure in advanced LRMs.

### Visualizations & Case Studies
1. Show per-example trajectories of detector scores before and after RL for matched member/non-member pairs. This would reveal whether RL truly collapses the margin universally or only on easy subsets.

2. Add a scatter plot of detection score versus answer correctness, response length, and token entropy. That would expose whether the detector is mostly tracking output confidence rather than membership.

3. Provide side-by-side case studies of member and non-member generations before and after GRPO/RAFT++. ICLR reviewers will want concrete examples where the model becomes harder to detect without visibly “forgetting” the contaminated content.

4. Visualize how AUROC changes as a function of RL step count and clipping strength on the same axes. This would make the causal role of clipping much more credible than isolated tables.

5. Include t-SNE/UMAP or embedding plots for both successful and failed detector cases, not just aggregate overlap. A few failure/success examples would show whether the embedding story is real or superficial.

### Obvious Next Steps
1. Build a detector that explicitly models reasoning trajectories or distributional generalization, since the paper’s main conclusion is that memorization-based detectors are no longer enough for LRMs. Without proposing or testing such a direction, the paper stops at a negative result.

2. Evaluate defense-side mitigations, such as checkpoint auditing or protocol-level contamination audits, under the same contaminated training pipelines. If the paper claims evaluation integrity is threatened, it should show which practical countermeasures remain viable.

3. Extend the study to other leaderboard-relevant reasoning systems and training recipes from different labs. ICLR will expect the fragility claim to generalize beyond Qwen/DeepSeek-style setups before treating it as a field-level warning.

4. Test whether release of intermediate checkpoints actually restores detectability in practice. The paper recommends checkpoint disclosure, but does not validate that this meaningfully helps under the specific concealment dynamics it identifies.

5. Probe whether contamination can be concealed under more realistic partial-contamination regimes and mixed clean/contaminated proportions. The current setup is strong enough to demonstrate a failure mode, but not enough to tell how easily it transfers to realistic leaderboard abuse scenarios.

# Final Consolidated Review
## Summary
This paper studies benchmark contamination detection in large reasoning models under two realistic scenarios: contamination during the SFT-to-RL transition, and contamination as a final SFT stage on already strong LRMs with CoT. The main claim is that many existing contamination detectors lose effectiveness badly in these settings, sometimes dropping from clearly above-chance AUROC to near-random performance, while benchmark performance remains inflated.

## Strengths
- The paper tackles a timely and important evaluation problem for LRMs, where leaderboard incentives make contamination a real threat rather than a hypothetical one.
- The empirical scope is unusually broad for this topic: 10 detector families, multiple benchmarks, two base-model families, and both pre-LRM and post-LRM contamination settings. The results consistently show detector degradation after RL and near-chance behavior under final-stage CoT contamination in many settings.
- The authors go beyond reporting failure cases and provide a mechanistic analysis tying the concealment effect to PPO-style clipping/importance sampling, supported by RAFT/RAFT++/GRPO comparisons and ablations.

## Weaknesses
- The central claims are stronger than the evidence in several places. The paper repeatedly gestures at a “broad class of RL methods” and a near-universal failure of detection, but the strongest evidence is really for PPO-style objectives with clipping/importance sampling in the specific recipes studied. That does not justify broad impossibility-style language.
- The theory is useful as intuition, but it is highly idealized and not a full causal proof. It relies on tabular/small-step assumptions and simplifies the advantage structure; it explains NLL-gap contraction, but the leap from that to detector failure is indirect and detector-dependent.
- The paper does not fully disentangle concealment from a memorization-vs-generalization mismatch, especially in the post-LRM setting. The authors themselves note that advanced LRMs may generalize to similar unseen problems, which could make memorization-based detectors fail even without “concealment” in the strongest sense.
- Statistical rigor is limited. The paper reports AUROC means extensively, but gives no confidence intervals, seed variance, or significance testing. Given that several effects are moderate and some appendix tables are noisy, this weakens confidence in the exact magnitude and monotonicity of the reported drops.

## Nice-to-Haves
- Add cleaner ablations that separate clipping, importance sampling, KL regularization, advantage normalization, and training length. Right now the clipping story is plausible, but not isolated as cleanly as it should be.
- Report uncertainty for the main AUROC and pass@1 results, ideally across multiple random seeds.
- Include more direct evidence that the post-LRM failures are due to generalization rather than detector calibration or distribution shift alone.

## Novel Insights
The most interesting insight is not just that contamination can be detected poorly, but that RL can actively make contamination harder to audit while still preserving inflated benchmark performance. That is a more worrying failure mode than ordinary memorization: the model becomes better at the task while becoming less distinguishable as a contaminated model. The second important observation is that, in advanced LRMs with long CoT, contamination may no longer present as simple rote memorization at all; instead, the model can assign high confidence to structurally similar unseen problems, which breaks the core assumption behind many existing detectors.

## Potentially Missed Related Work
- **Fu et al. (2024), survey on contamination detection assumptions** — relevant because the paper’s main claim is fundamentally about the limits of memorization-based assumptions.
- **Bordt et al. (2024), *How much can we forget about data contamination?*** — relevant since it studies contamination dynamics over training and could contextualize the “concealment through further training” result.
- **Wu et al. (2025), *Reasoning or memorization? unreliable results of reinforcement learning due to data contamination*** — relevant because it appears closely related to the same general problem of contamination affecting RL-based reasoning.
- **Samuel et al. (2024), limitations and oracle challenges in contamination detection** — relevant to the practical detectability limits the paper emphasizes.

## Suggestions
- Tighten the main claims: explicitly scope them to the studied benchmarks, base models, and PPO-style RL recipes, and avoid broad “all RL methods” language unless you can substantiate it.
- Add a sharper experiment that separates concealment from generalization, for example with matched non-member sets of varying distributional distance or synthetic membership-controlled data.
- Strengthen the causal story with more targeted ablations on clipping vs. importance sampling vs. KL regularization, and report variance across seeds for the headline AUROC results.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 6.0]
Average score: 6.7
Binary outcome: Accept
