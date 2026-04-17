---
job_id: 3dd986d4-885b-4782-baff-56ffdf5a3a73
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: zftquoHTgS.pdf
paper: SMARTSWITCH: Advancing LLM Reasoning by Overcoming Underthinking via Promoting Deeper Thought Exploration
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is squarely about test-time inference strategies for large language models, targeting reasoning and LongCoT behavior on math benchmarks, which fits ICLR’s core ML / representation learning / reasoning topics.

## Minimum Quality
Pass ✅.  
All required sections are present (Abstract, Introduction, Related Work, Methodology, Experiments, Results/Analysis, Discussion, Conclusion). The method is clearly specified, experiments are substantial on several standard benchmarks, and there is no obvious fatal methodological or theoretical flaw.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any hidden instructions or attempts to manipulate an automated reviewer in the main paper content.

---

# Expected Review Outcome:

## Summary

The paper studies an “underthinking” failure mode of Long Chain-of-Thought (LongCoT) LLMs, where models rapidly switch between shallow thoughts and prematurely abandon promising reasoning paths. It introduces SmartSwitch, a plug-and-play inference framework that monitors the generated text for thought-switch cues, scores the just-abandoned thought with a process reward model (PRM), and, if the score is high, backtracks and injects a “deepen” prompt to encourage further exploration of that path. Experiments on five math reasoning benchmarks and several LongCoT models (DeepSeek-R1-Distill Qwen series and QwQ-32B) show sizable accuracy gains, along with reductions in underthinking frequency, response length, and wall-clock inference time.

## Strengths

1. **Clear and well-motivated problem formulation (underthinking)**  
   - The paper articulates a specific failure mode of LongCoT models: frequent, premature thought switching that abandons viable strategies.  
   - The qualitative example in **Figure 1(a)**, showing 74 short thoughts from DeepSeek-R1 with minimal elaboration, nicely grounds the notion of “underthinking” and makes it concrete.  
   - The quantitative metric UF\(_L\) in **Equation (1)**, although simple, enables a systematic investigation of how often short thoughts occur across models and tasks.

2. **Systematic empirical evidence that underthinking is real and correlated with difficulty**  
   - **Figure 1(b)** shows that UF\(_L\) is nontrivial across six LongCoT LLMs and increases with the length threshold \(L\), suggesting that frequent short thoughts are widespread.  
   - **Figure 2(a)** shows that UF\(_{100}\) grows monotonically with human-annotated difficulty levels on MATH-500, and **Figure 2(b)** shows higher UF\(_{100}\) for incorrect vs correct responses across models. This is persuasive evidence that the defined underthinking behavior is not just a cosmetic metric but correlates with hard problems and failure.

3. **Simple, modular inference-time mechanism that is practically deployable**  
   - SmartSwitch operates entirely at inference time without fine-tuning, using a perception–intervention loop.  
   - The pipeline in **Figure 3** and the pseudocode in **Figure 5 / Algorithm 1** provide a clear picture of the system: detect linguistic cues (Table 10), segment the preceding thought, score with a PRM, and either backtrack + inject a deepen prompt or continue.  
   - The reliance on an external PRM (Universal-PRM-7B) with 32k context is realistic given current infrastructures; the method seems straightforward to plug into existing LongCoT LLM systems.

4. **Strong empirical gains across multiple models and benchmarks**  
   - **Table 1** shows consistent and often large pass@1 improvements on AIME24/25, AMC23, MATH-500, and GaoKao2023en for 1.5B–32B models.  
     - DeepSeek-R1-Distill-Qwen-1.5B: AIME25 from 20.0 to 36.7, AIME24 +11.1 points, AMC23 +10.0.  
     - DeepSeek-R1-Distill-Qwen-7B: AIME25 +23.3, AIME24 +11.2.  
     - QwQ-32B: AIME24 79.5 → 86.7; AIME25 63.3 → 73.3; AMC23 reaches 100%.  
   - The gains are especially impressive for smaller models and remain nontrivial for already strong 32B models, suggesting broad applicability.

5. **Evidence that SmartSwitch improves efficiency rather than just “thinking longer”**  
   - **Table 2** shows that on AIME24, SmartSwitch generally *reduces* average response length, especially for correct solutions (e.g., for R1-Distill-32B, from 12272 to 10284 tokens for correct responses).  
   - **Table 3** shows notable wall-clock time reductions despite PRM overhead, e.g., for R1-Distill-1.5B, AIME24 time goes from 3.23 to 2.14 min/q (−33.7%).  
   - This combination of higher accuracy and *lower* compute is important, and the analysis in Section 5.3 rightly emphasizes that SmartSwitch prunes wasteful exploration on unpromising paths.

6. **Direct evidence that SmartSwitch mitigates the targeted behavior (underthinking)**  
   - **Figure 4(a)** shows substantial reductions in UF\(_{100}\) for all DeepSeek-R1-Distill Qwen models and QwQ-32B when using SmartSwitch vs vanilla inference.  
   - **Figure 4(b)** further shows a reduced number of thought switches, again across all models, consistent with the stated goal of fostering deeper, more focused exploration.

7. **Reasonable baselines and ablations, particularly around the PRM**  
   - Section 5.4 and **Table 5** compare SmartSwitch to “Standard Prompting” and TIP (Wang et al., 2025), showing that naive “think deeply” instructions barely help and decoding penalties only modestly improve performance, whereas SmartSwitch gives a much larger gain.  
   - The ablation in **Table 4** (different PRMs and an “Always Intervene” strategy) is insightful: always intervening actually *hurts* performance on AIME25, reinforcing that selective, PRM-guided intervention is crucial. Universal-PRM-7B clearly outperforms other PRMs in this setting.  
   - **Table 6** shows that the authors have thought carefully about segmentation strategy; the Adaptive Paragraph (v4) method is empirically best across model sizes, and the text explains why grouping vs fragmentation matters.

8. **Good clarity and reproducibility**  
   - The method is described clearly in Section 4, with explicit mention of hyperparameters (PRM threshold 0.7, max 3 interventions, 200-token segmentation limit) and cue lists (Table 10).  
   - Evaluation details (datasets in **Table 9**, symeval-based verification, 32 random runs, inference settings, hardware) are sufficiently specified to reproduce the experiments.  
   - Qualitative case studies in **Figures 6–8** illustrate how SmartSwitch alters trajectories in concrete problems, including both cases where it fixes failures and where it only reduces computation while keeping the correct answer.

## Weaknesses

1. **The core underthinking metric is simplistic and somewhat brittle**  
   - Underthinking Frequency UF\(_L\) in **Equation (1)** is purely length-based: any thought segment shorter than \(L\) is labeled “underthinking”. This ignores content quality and the possibility of short but highly effective steps.  
   - The segmentation for UF analysis uses a separate LLM (DeepSeek-V3) via a custom prompt (Appendix F.3), which is itself heuristic and potentially inconsistent across models or marginal prompts. The metric therefore conflates the behavior of two models and might not exactly reflect the “thought units” of the target solver.  
   - This matters because several central claims about prevalence and correlation with difficulty (Section 3, **Figures 1–2**) rest on UF. While those plots are suggestive and likely directionally correct, the paper does not rigorously validate that “short segments as defined by our segmentation + threshold correspond to premature abandonment” (e.g., via human annotation or PRM-based labeling). As a result, the conceptual foundation of “underthinking” remains somewhat loose.

2. **Thought-switch detection for SmartSwitch is narrow and potentially misses many real switches**  
   - In the actual SmartSwitch pipeline, thought-switch detection is based on a small, hand-crafted list of lexical cues (Table 10), such as “Alternatively”, “Wait, let me try another method”, etc. This list is short and heavily English template-dependent.  
   - LongCoT reasoning traces often switch strategies without explicit markers (“Now I’ll try something else”) and may use idiosyncratic phrasing, especially across different base models or languages. SmartSwitch will not detect these shifts, so high underthinking episodes that do not use the exact patterns in Table 10 are invisible to the framework.  
   - The paper does not quantify recall of this detector. For example, in **Figures 6–8**, the presence of “Alternatively” yields SmartSwitch interventions, but it is unclear what fraction of thought-switching events in general corpora are actually captured. This affects both the robustness and generality of the method beyond the provided benchmarks.

3. **Strong reliance on a powerful external PRM and delicate hyperparameter tuning**  
   - SmartSwitch’s effectiveness is tightly bound to the quality and calibration of Universal-PRM-7B, which itself is a large specialized model with a 32k context window. Section 6 acknowledges this, but the practical implications are under-discussed. In many deployment settings, running a second 7B model repeatedly during inference is not trivial.  
   - The potential-score threshold \(\tau_{\text{score}}\) appears to be tuned on the *same* benchmarks used for reporting final results. **Table 8** shows very sharp performance sensitivity to threshold (for AIME24, 0.70 yields 40.0% vs 30.0% at 0.69 or 0.71 for the 1.5B model, similar sensitivity for others). There is no clear separation between tuning and test sets, nor a cross-validation or held-out benchmark used exclusively for hyperparameter selection. This risks optimistic performance estimates and overfitting of the framework to these specific contests.  
   - Similarly, the choice of “last-process” mapping in **Table 7** is justified empirically, but again on the same or very similar benchmarks. Collectively, these choices make SmartSwitch feel more like a carefully tuned stack specialized for AIME/AMC-style math than a robust general mechanism.

4. **Limited comparison to broader adaptive test-time reasoning methods**  
   - The paper compares only against (i) vanilla LongCoT, (ii) “Standard Prompting” with generic deep-thinking instructions, and (iii) TIP (Wang et al., 2025) in **Table 5**. While TIP is indeed closely related, the landscape of adaptive test-time compute / reasoning control is broader.  
   - There is no comparison or discussion vis-à-vis recent work on reasoning length calibration, adaptive depth, or over/underthinking tradeoffs (e.g., works on “reasoning on a budget”, step-level halting/continuation, or dynamic CoT length). This undercuts the claim that SmartSwitch is particularly competitive as a general inference-time control mechanism.  
   - At the experimental level, an obvious additional baseline would be a simple “PRM-guided linear decoding” scheme: periodically score the *current* thought without any backtracking and append a generic “continue / reflect” prompt when the ongoing thought has high potential. This would remove the somewhat complex backtracking-and-restart mechanics and might yield similar gains; the absence of such a baseline makes it harder to isolate the specific benefit of SmartSwitch’s interrupt–backtrack design.

5. **Scope is restricted to math reasoning; claims of generality are mostly speculative**  
   - All quantitative results are on math benchmarks (AIME24/25, AMC23, MATH-500, GaoKao2023en). While math is a natural testbed for LongCoT, the paper repeatedly suggests broad implications for “complex reasoning tasks” generally, including programming, scientific QA, and other domains.  
   - Since both the PRM (Universal-PRM-7B) and the underthinking metric are tightly math-oriented, there is no empirical evidence that SmartSwitch would behave well in more open-ended tasks where “promising reasoning path” is less crisply defined (e.g., multi-hop QA, legal reasoning).  
   - This is not a fatal flaw, but it suggests the paper’s contributions are narrower than the framing implies; essentially, the current evidence supports “math-reasoning-specific SmartSwitch” more than a broadly generalizable reasoning controller.

6. **Some aspects of the experimental methodology and reporting are under-specified or potentially confusing**  
   - **Table 2** reports “Response Length (Token Number)” where the values are very large (e.g., 14973.97, 16924.40). From the description, it seems these are *average total tokens across 32 samples per problem*, but this is not clearly stated. It would be more interpretable to report per-sample averages or normalized lengths; as written, a reader has to guess.  
   - The PRM scoring overhead is summarized in **Table 3** by wall-clock time, but there is no detailed breakdown of how many PRM calls are made per query, or how often SmartSwitch actually intervenes vs just evaluating and continuing. This hinders understanding of the true computational trade-offs.  
   - For the UF-based reduction analysis in **Figure 4**, the paper states that the metric uses a length threshold \(L=100\), but it is not fully clear whether thoughts are segmented using the same heuristic cues as at inference time or via the LLM-based segmentation of Section 3. Mixing segmentation methods could bias the comparison.

7. **Underdeveloped mathematical formalization of the SmartSwitch objective**  
   - The method is almost entirely heuristic. While this is acceptable for an inference-time engineering contribution, the paper occasionally edges into semi-theoretical claims about “promising but prematurely abandoned thoughts” without a formal notion of “potential” beyond PRM scores. There is no discussion of PRM calibration or how PRM scores relate to eventual correctness probabilities.  
   - For example, in Section 4.2 the potential evaluation treats the PRM score as a binary signal using threshold \(\tau_{\text{score}}\), but there is no loss or objective defined that SmartSwitch is even approximately optimizing. Some framing in terms of expected accuracy vs compute, or a simple decision-theoretic model (e.g., intervene when \(p(\text{correct} \mid T_{\text{prev}}) \times \Delta\text{utility} > \text{cost}\)) would provide more intellectual clarity and connect the method to existing work on adaptive test-time compute.  
   - At minimum, it would be helpful to provide some statistics on the empirical relationship between PRM scores and correctness of downstream answers (e.g., bin the PRM scores and plot empirical accuracy), to justify using a hard threshold.

8. **Missing or under-discussed closely related work**  
   - Several recent works directly tackle over/underthinking and optimal reasoning length or introduce dedicated benchmarks and analyses for this phenomenon. The paper cites Wang et al. (TIP) and Chen et al. (overthinking), but additional closely related works are not discussed (details below). This weakens the contextualization of the contribution and the argument that SmartSwitch addresses a novel or under-explored corner of the space.

## Potentially Missing Related Work

1. **Su et al., “Between Underthinking and Overthinking: An Empirical Study of Reasoning Length and Correctness in LLMs”, 2025**  
   - This paper systematically studies the relation between CoT length and correctness and explicitly frames the underthinking/overthinking tradeoff. It is directly relevant to Section 3 (analysis of UF and correctness) and Section 5.3 (efficiency vs accuracy). It should be cited in the related work on thinking effectiveness and in the discussion of UF, ideally with some comparison of their findings vs the UF-based trends here.

2. **Pan et al., “Large Language Models Think Too Fast To Explore Effectively”, 2025**  
   - This work analyzes how rapid decision-making and shallow exploration in LLMs limits reasoning quality, conceptually very similar to the “underthinking” phenomenon. It should be discussed in Section 2 (Thinking effectiveness in LongCoT reasoning) and related to the human-cognition analogy in Section 1/3, highlighting how SmartSwitch provides a concrete mechanism to slow down and deepen exploration.

3. **Aggarwal et al., “OptimalThinkingBench: Evaluating Over and Underthinking in LLMs”, 2025**  
   - Introduces a benchmark focused on over/underthinking behavior. This is highly relevant to the UF metric and **Figure 1–2** analyses. The authors should compare their UF-based characterization to metrics used in OptimalThinkingBench and explain why they chose their particular formulation. If feasible, evaluating SmartSwitch on that benchmark would greatly strengthen claims about addressing underthinking.

4. **Alomrani et al., “Reasoning on a Budget: A Survey of Adaptive and Controllable Test-Time Compute in LLMs”, 2025**  
   - This survey covers adaptive reasoning strategies and test-time compute control, which overlaps with SmartSwitch’s goal of improving both accuracy and efficiency. It should be cited in the related work on LongCoT and thinking effectiveness (Section 2) and used to position SmartSwitch among other inference-time control techniques (e.g., dynamic depth, early stopping, etc.).

## Questions

1. **Segmentation and UF metric validity**  
   - How sensitive are the UF\(_L\) trends in **Figures 1(b), 2(a), 2(b)** to the choice of segmentation LLM and its prompt? For example, if you vary the division prompt or use a simpler rule-based segmentation, do the same qualitative conclusions hold? Any quantitative robustness check would increase confidence that UF is not an artifact of the segmentation procedure.

2. **Coverage and recall of cue-based thought-switch detection**  
   - Have you measured how many actual strategy shifts (as judged by a human or by an LLM with access to the entire trace) are captured by the lexical cues in Table 10? Could you provide statistics such as “out of N manually annotated switches across K problems, X% are caught by our detector”? If recall is low, how might SmartSwitch be extended (e.g., with embedding-based similarity or PRM-based change-point detection)?

3. **Hyperparameter tuning procedure**  
   - For the potential score threshold \(\tau_{\text{score}}\) (Table 8) and mapping choice (Table 7), on which datasets were these decisions made? Were AIME24/AIME25 used both for tuning and final evaluation? If so, could you redo at least one benchmark with hyperparameters fixed based on a different held-out dataset (e.g., tune on MATH-500, evaluate on AIME25) to assess generalization?

4. **Ablation on intervention count and depth**  
   - You cap the number of interventions per problem at three. How sensitive is performance to this cap? Is there evidence that more than three interventions ever helps, or does performance plateau or degrade? An ablation varying this number (e.g., 0, 1, 2, 3, 5) on one benchmark would clarify whether the observed gains indeed come from a small number of well-placed interventions.

5. **Comparison to simpler PRM-augmented baselines**  
   - Have you tried baselines where the PRM is used without backtracking, for example: scoring the *current* partial thought periodically and, if the score is high, simply appending a deepen prompt inline rather than rewinding and restarting the pass? This would isolate the specific benefit of the interrupt–backtrack–restart design that defines SmartSwitch.

6. **Generalization beyond math**  
   - Do you have any preliminary evidence (even anecdotal) of SmartSwitch applied with a more general PRM (or even a heuristic scoring function) in non-math domains, such as code generation or multi-hop QA? If not, can you outline what you see as the main obstacles (e.g., lack of suitable PRMs, weaker alignment between PRM score and correctness)?

7. **Relationship between PRM score and final correctness**  
   - Have you empirically checked how PRM scores on thought segments correlate with eventual answer correctness, e.g., by plotting accuracy vs score bins? This would justify the decision-theoretic choice of thresholding at \(\tau_{\text{score}}\) and may suggest more principled strategies such as score-dependent intervention probabilities.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating
3: good.  
The method is technically plausible and reasonably described, with strong empirical support and careful ablations on key components. The main limitations are reliance on a specific PRM and somewhat heuristic metrics and detectors, but there are no glaring methodological errors.

## Presentation Rating
3: good.  
The paper is generally clear and well structured, with helpful figures (especially **Figures 1–4** and **6–8**) and detailed experimental settings. Some aspects (UF definition, length reporting, and cue coverage) could be clarified further.

## Contribution Rating
3: good.  
The paper offers a concrete, practically implementable inference-time mechanism that delivers sizable gains on strong LongCoT baselines, and it provides a focused empirical study of an underexplored “underthinking” failure mode. The ideas are incremental rather than deeply theoretical, but the empirical impact is meaningful.

## Overall Rating
6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

SmartSwitch is a solid, well-executed inference-time framework that produces consistent and sizable gains on challenging math reasoning benchmarks while also reducing compute and directly attacking a clearly observed “underthinking” failure mode. The reliance on an external PRM, heuristic detector, and task-specific tuning, along with somewhat simplistic underthinking metrics and limited baselines, prevent this from being a top-tier contribution, but the empirical results and clarity of the idea make it a worthwhile addition to the ICLR program.

## Reviewer Confidence
4: confident.  
I am familiar with LongCoT reasoning, PRMs, and adaptive test-time compute, and I have carefully examined the equations, figures, and experimental setup, though I have not independently re-run experiments.