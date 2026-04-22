# Position: The Hidden Costs and Measurement Gaps of Reinforcement Learning with Verifiable Rewards

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 6, 2, 2

## Abstract
Reinforcement learning with verifiable rewards (RLVR) is a practical and scalable approach to enhancing large language models in areas such as math, code, and other structured tasks. Two questions motivate this paper: how much of the reported gains survive under strictly parity-controlled evaluation, and whether RLVR is cost-free or exacts a measurable tax. We argue that progress is real, but gains  are often overstated due to three forces—an RLVR tax, evaluation pitfalls, and data contamination. Using a partial-prompt contamination audit and matched-budget reproductions across base and RL models, we show that several headline gaps shrink or vanish under clean, parity-controlled evaluation. We then propose a tax-aware training and evaluation protocol that co-optimizes accuracy, grounding, and calibrated abstention and standardizes budgeting and provenance checks. Applied to recent RLVR setups, this protocol yields more reliable estimates of reasoning gains and in several cases revises prior conclusions. Our position is constructive: RLVR is valuable and industry-ready; we advocate keeping its practical benefits while prioritizing reliability, safety, and measurement.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper provides a perspective on unseen costs associated with large performance gains reported when reinforcement learning with variable rewards (RLVR) is used to enhance large language models focused on math, coding, and other structured tasks. The authors argue that some of these gains might be overstated, and lay out a three-pronged thesis to support their claim. They propose a tax-aware RLVR protocol that is claimed to mitigate the identified pitfalls of RLVR.

### Strengths
(+) The paper is generally very well-written, and the authors lay out their case for the proposed protocol very nicely. 

(+) The paper presents `both sides' of the RLVR coin and provides perspective on the impacts of RLVR by examining recent literature. This perspective is given in terms of the potential of RLVR expanding model capabilities, improving generalizability, and whether RLVR indeed improves sample efficiency. 

(+) The premise of an RLVR tax is very interesting- in that the authors posit that gains and advances reported when RLVR is used must take into consideration associated systemic costs- akin to a tax. 

(+) The authors highlight the pitfalls of mismatched budgets and the brittleness of metrics when reporting performance of models/ methods in the literature. This is an especially important observation in a field where hundreds of papers appear daily, each claiming superior performance over current art. 

(+) The authors make a convincing case for development of a tax-aware RLVR protocol by highlighting challenges in evaluation standards, contaminated data, and unseen/ unaccounted costs associated with large performance gains.  

(+) The compositionality and ease of interpretability of the protocol reward makes it amenable for widespread use.

### Weaknesses
(-) Analysis is limited in its scope to open-weight models, and further, fine-tuned for math/ code/ QA with verifiable objectives. 

(-) The notion of a `tax' in this paper is somewhat limited in its scope to discussions on RLVR. In terms of the three ``recurrent pressures'' mentioned by the authors in Sec. 3 (Lines 190-193), this reviewer is curious as to whether such an interpretation (i.e., costs associated with huge performance gains of a technology) has been more widely applied to the examination of performance and capabilities of LLMs as a broader field, or even to other sub-domains beyond RLVR. 

(-) The takeaway from Sec. 3 seems very generic (Lines 265-269) in that such a co-optimization strategy to improve reliability while maintaining provenance is a solution approach that can be broadly used across multiple domains beyond RLVR. Similarly, in Sec. 4 and 5, the importance of standardizing evaluation procedures and impacts of contaminated data in terms of distributional shifts has been widely known across multiple experimental domains (not limited to LLMs and computer science). 

(-) Perhaps the most significant weakness in my opinion is that the main body of the paper contains very little results to support the claims made in Sec. 3, 4, 5. Many of the associated results have been deferred to the appendix. Moreover, the proposed protocol itself has not been explicitly evaluated (I might be mistaken in this observation, so authors please correct me if I'm wrong); rather the authors point to work in the literature where it is claimed to work. In this case, the paper amounts to providing a reward model as in Line 444-445 for the working of mechanisms described in paragraph (F) of Sec. 6.

### Questions
Please see Weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This is a good position paper on Reinforcement Learning with Verifiable Rewards (RLVR). It reveals three factors often overlooked in RLVR-related work: the RLVR tax, evaluation traps, and data contamination. It also reevaluates the performance of RLVR-related work under the same experimental settings and points out potentially viable solutions for each factor.

### Strengths
- This work points out aspects that researchers in the RLVR field may have overlooked, preventing reported results from being achieved at the expense of other performance metrics
- The authors evaluated the results of mainstream models using the same, fair, and reasonable experimental settings, which helps establish a more equitable evaluation understanding for researchers in the field

### Weaknesses
- In Section 6, this tax-aware RLVR protocol proposes several methods that can mitigate or resolve various RLVR tax issues. However, these are largely based on findings from existing work and lack experimental validation under the unified perspective presented in this paper. The practical effectiveness of these solutions is therefore questionable.
- The experiments in this paper mainly focus on data contamination, performance, and model output behavior, lacking a direct evaluation of impacts on privacy exposure and fine-grained instruction-following capability

### Questions
See weaknesses part

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper argues that while RL with Verifiable Rewards (RLVR) improve LLM performance on verifiable tasks (math, code, structured QA), headline gains are usually overstated due to three confounders: an “RLVR tax” (side-effects like overconfident hallucinations, instruction-following drift, and expanded safety/privacy attack surface), evaluation pitfalls (budget mismatch, fragile LLM-judge pipelines, calibration drift), and dataset contamination. The authors provide parity-controlled reproductions and a partial-prompt contamination audit showing that several celebrated gaps shrink under clean evaluation. They then propose a “tax-aware RLVR protocol” that co-optimizes accuracy, grounding, and calibrated abstention, and standardizes budgeting, calibration, and provenance checks.

### Strengths
1. Clear thesis: The paper cleanly frames three forces that can inflate RLVR claims, taxes, evaluation pitfalls, and contamination. and situates them within the current literature and practice.

2. Substantive diagnosis: The manuscript devotes substantial space to how each inflation source arises in practice (e.g., pass@k without budget parity, fragile LLM-judges, partial-prompt reconstruction), making the critique actionable rather than purely rhetorical.
Balanced stance. The paper acknowledges regimes where RLVR likely broadens capability (not just sharper search) and positions the protocol as compatible with those positive results.

### Weaknesses
1. Incrementality / evidence mix. Several observations (budget parity, judge robustness, contamination) echo concerns raised elsewhere; the novelty lies more in synthesis + packaging than in new empirical evidence.

2. The “tax-aware” protocol is promising, but for readers less familiar with RLVR and evaluation, it’s difficult to judge how much each protocol element actually fixes which failure mode and at what cost. There seem to be a lack of table or result supporting the proposed Tax-aware RLVR protocol.

### Questions
1. I would like to see a table comparison of ECE (and accuracy) computed (a) on the shared-attempt subset and (b) on each model’s own attempted set. How often do conclusions about “which model is better calibrated” flip between these two views? 

2. In Table 3, could similar suffix-reconstruction rates across models be partially explained by short, fact-seeking questions in SimpleQA? For example,  “On what day, month, and year was Algerian artist Mohammed Racim born?” could be capped to “ On what day, month, and year was Algerian artist? Would this result that both strong and weak models fail to infer what the actual question would be, thus bringing similar performance? Would using other dataset be a better choice?

3. Scope of parity issues beyond RLVR. You argue parity/control problems are common in RLVR. Do you observe the same failure modes (budget mismatch, judge brittleness, calibration neglect) in nearby post-training regimes?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper summarized key recent progresses in development of RLVR, specifically focusing on RLVR tax, evaluation pitfalls and data contamination. The author propose a comprehensive tax-aware RLVR protocol to calibrate RLVR methods.

### Strengths
1. The author gives a comprehensive review on recent  RLVR studies, pointing the RLVR tax, evaluation pitfall and data contamination problems.
2. The paper is well-written and easy to read.

### Weaknesses
1. The technique contribution is not clear, most results are already found in existing literature. The author might have to conduct further deeper analysis.
2. Lacking necessary experiments to demonstrate the effectiveness of the proposed RLVR protocol.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This position paper argues that RLVR reliably boosts verifiable tasks (math/code), but headline “reasoning” gains are often overstated due to three confounders:

(i) an RLVR tax (reduced abstention, increased overconfidence, instruction-following drift, larger safety/privacy surface) -- discussed in Section 3
(ii) evaluation pitfalls (budget mismatch, fragile LLM judges, calibration drift) -- discussed in Section 4 and
(iii) data contamination -- discussed in Section 5.

The authors sketch the future as a tax-aware training and evaluation protocol with componentized rewards (correctness/grounding/refusal), calibration-gated early stopping, matched-budget saturation curves, judge-robustness probes, and partial-prompt contamination audits and  parity-controlled reproductions.

### Strengths
S1 **Timely, well-scoped problem.** The paper isolates three concrete inflation sources—taxes, evaluation pitfalls, contamination—and keeps a constructive stance: preserve RLVR’s practical wins while fixing reliability and measurement. Fig. 1 clearly maps the agenda.

S2 **Strong coverage of recent work under a common umbrella.** The draft cites key 2024–2025 threads it builds on: pass@k/test-time scaling, judge reliability/surveys, code contamination, budget-parity normalizers (“SoberScore”), GHPO/difficulty-aware schedules, and “think-when-needed” protocols. 

S3 **Constructive position.** Many teams can adopt these protocols today.

### Weaknesses
W1. **Stance reads non-committal.** 
- The paper states too vague a position of “RLVR is valuable and industry-ready, but its benefits are easiest to preserve when reliability, safety, and measurement are treated as first-class objectives.” which could be said for anything. What wouldn't benefit from reliability, safety and measurement?
- Specifically, Section 6 aggregates wide range of controls (budget parity, seed averaging, judge stress tests, contamination hygiene) but don’t isolate which parts matter the most. My primary question: Which of many proposed elements move the needle the most, and by how much? The contribution seem more curatorial than guiding practitioners toward the highest-leverage controls.
- P.S. I’m treating the work as a position paper and ignoring the lack of experiments here that validate the protocols. I leave fit-to-venue to the AC.

W2. **Claim 1 (refusals vs. confident errors) needs isolation.** 
- Is this RLVR-specific? SFT on the same data appears to show similar effects (Table 1 and several cited papers). If so, the issue looks like domain or reasoning-training, not an RLVR tax.
- On hallucination/overconfidence: the critique is fair but targets the objective more than RLVR itself. For example, Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty (arXiv:2507.16806) adds an uncertainty loss. But I agree that evaluations should measure this explicitly.

W3 **Claim 2 (privacy/safety surface) lacks a clear threat model.** 
- The paper argues that long, explicit traces widen attack and leakage surfaces. Other work finds that traces aid monitorability (e.g., “Chain-of-Thought Monitorability”), which is more plausible to me.
- Maybe the crux of the difference is clarifying who sees traces, when, and how. Are traces exposed via API outputs? If users see only final answers, this vector seems weak. Open-weights models have stronger safety and privacy threat models. When are  RLVR traces the primary concern?

W4 **Claim 3 (evaluation pitfalls) vs. noise.** 
- Table 2 shows AIME/AMC gaps often within ±3–5 points apart from 1.5B models excluding nemotron — not too far apart from reported standard deviations (Mu et al., 2025; Hochlehnert et al., 2025). If effects sit in the noise, the original papers may have reported performance correctly. The evidence does not seem to support big mismatches.
- Table 3 suggests models reconstruct hidden tails and final answers in math (a successful reproduction of Wu et al., 2025b) and compares against a SimpleQA control. Qwen models leak more on math, but any leakage is concerning. Is the claim that SimpleQA shows no leakage and the variance stems from metric brittleness? Acc@80 and R@80 look comparable across datasets, which leaves it unclear whether the high contamination score reflects a brittle metric or real contamination in both Math and SimpleQA.

Basically, I worry that the claims do not match the evidence provided, and several issues seem general to reasoning/domain training rather than RLVR. Reported evaluation gaps look small. The contribution reads as a broad checklist without priorities; could the authors identify and justify the few controls that matter most?

### Questions
Could the authors address the weaknesses?

### Soundness
1

### Presentation
2

### Contribution
3
