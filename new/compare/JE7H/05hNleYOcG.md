---
job_id: 269b06bd-2921-4636-8f5a-2fdf2729f625
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 05hNleYOcG.pdf
paper: PLAGUE: Plug-and-Play Framework for Lifelong Adaptive Generation of Multi-Turn Jailbreaks
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper studies multi‑turn jailbreak attacks on LLMs, framed as an agentic / lifelong‑learning framework for red‑teaming. This is directly within ICLR’s scope (representation learning, agents, safety, evaluation).

## Minimum Quality
Pass ✅.  
All core sections are present (Abstract, Introduction, Related Work, Method, Experiments, Results/Discussion, Conclusion, Ethics). The work is clearly written, proposes a concrete algorithmic framework (PLAGUE) with equations and pseudo‑code, and includes substantial experiments with multiple baselines on HarmBench and several frontier LLMs. I do not see fundamental theoretical or experimental flaws that would justify desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
The paper includes prompts used in the method and evaluation, but I did not find any instructions that target the *reviewer* or attempt to manipulate the review process. No hidden or out‑of‑context prompt‑injection behavior is apparent.

---

# Expected Review Outcome:

## Summary

The paper proposes **PLAGUE**, a plug‑and‑play, lifelong‑learning framework for *multi‑turn jailbreak attacks* on LLMs. PLAGUE decomposes an attack into three phases: a **Planner** that generates an $n$‑step plan using a retrieval‑based memory of successful strategies, a **Primer** that builds adversarial context by following early plan steps with backtracking and reflection, and a **Finisher** that delivers the final goal‑conditioned attack using the frozen context.  

Experiments on HarmBench across several strong models (OpenAI o3/o1, Claude Opus 4.1, DeepSeek‑R1, Llama‑3.3‑70B) show that PLAGUE can substantially improve attack success rate (SRE and binary ASR) over prior single‑turn and multi‑turn methods such as GOAT, Crescendo, ActorBreaker, AutoDAN‑Turbo, FITD, and X‑Teaming, within a comparable query budget.

## Strengths

1. **Clear modularization of multi‑turn attacks with plug‑and‑play design.**  
   The core idea of decomposing attacks into Planner, Primer, and Finisher phases (Section 3, **Figure 1**) is conceptually helpful and practically useful. The diagram in Figure 1 makes it easy to see how the Attacker LLM, Rubric Scorer, summarizer, and memory interact. Importantly, the framework is not tied to specific attack algorithms; the paper shows that GOAT and Crescendo can serve as Finisher modules, and ActorBreaker’s planning can plug into the Planner, which underpins the claimed “plug‑and‑play” nature.

2. **Strong empirical gains across strong frontier models.**  
   **Table 2** is compelling: PLAGUE improves SRE over the best existing baseline for OpenAI o3 from 0.587 (GOAT) to 0.814 and for o1 from 0.798 (GOAT) to 0.931, while matching or slightly exceeding the best baselines on DeepSeek‑R1 and Llama‑3.3‑70B. With the Crescendo Finisher, **Table 4** shows SRE on Claude Opus 4.1 rising from 0.48 (base Crescendo) to 0.673, which is a large relative gain. These results, especially on very safety‑tuned models like o3 and Opus 4.1, indicate practical significance for safety evaluation.

3. **Systematic ablations linking components to performance.**  
   The GOAT‑based ablation in **Table 3** is quite informative. By gradually adding backtracking (BT), reflection (R), Planner (P), and retrieval of successful strategies (RSS), the authors show monotonic (or nearly monotonic) improvements in SRE on o3 (0.587 → 0.814) and Claude (0.222 → 0.465). This supports the claim that each architectural ingredient contributes meaningfully rather than being an opaque black‑box tweak.

4. **Careful budget/efficiency analysis rather than just raw ASR.**  
   **Table 5** measures the average number of Target, Evaluator, and Planner invocations per model and per baseline. The comparison shows that PLAGUE achieves its improved ASR with total calls roughly comparable to Crescendo and within one Target call of GOAT, despite having an additional planning phase. **Figure 2** further shows that SRE on o3 roughly saturates by 6 conversation turns, justifying the 6‑turn budget choice and arguing that improvements are not just from throwing more queries at the model.

5. **Integration of lifelong‑learning via retrieval‑based strategy memory.**  
   The memory bank $\mathbb{R}^{\{+\}}$, described in Section 3.3.1 and Algorithm 1, stores strategy category/definition plus example attacks keyed by goal embeddings. Strategies from semantically similar goals are retrieved using cosine similarity and used as in‑context exemplars for the Planner. The ablation in **Table 3** (GOAT + BT + R + P vs. +RSS) shows that this retrieval adds non‑trivial gains (o3 SRE 0.773 → 0.814, Claude 0.431 → 0.465) on top of already strong planning and reflection.

6. **Diversity analysis that connects planning choices to prompt diversity.**  
   Section C.2 and **Figure 3** introduce an embedding‑based diversity metric (Equation (1)) and show how swapping in ActorBreaker’s planner raises diversity from ~0.326 (normal planner with strategy library) to ~0.433 (ActorBreaker planner), while PLAGUE’s ASR remains substantially higher than pure ActorBreaker. The combination of diversity and ASR results strengthens the argument that PLAGUE is not just over‑optimizing one narrow attack pattern.

7. **Broader analysis across harm categories.**  
   **Figure 4** provides category‑wise SRE for PLAGUE vs. GOAT and Crescendo across HarmBench’s threat taxonomy (e.g., misinformation, hateful content, etc.). This is useful to see that gains are not confined to a single class (e.g., the paper notes near‑perfect ASR on misinformation and more difficulty on sexual content), which is relevant for safety practitioners.

8. **Reasonable metric design and equations.**  
   The explicit definition of ASR in Section 3.2 and the SRE formula in Appendix C.1 clarify how attacks are judged. The SRE metric  
   \[
   \text{SRE} = (1-\text{if\_refusal})\times\frac{(\text{Convincing}+\text{Specificity}-2)}{8}
   \]  
   is sound in the sense that it maps [1,5]×[1,5] Likert responses into [0,1] and tightly ties success to both non‑refusal and harmful specificity.

## Weaknesses

1. **Limited conceptual novelty relative to existing multi‑turn agentic jailbreak frameworks.**  
   While PLAGUE is well‑engineered, much of the framework combines ingredients that are already present in different forms: multi‑step planning (ActorBreaker, RACE), iterative query refinement with reflection (Crescendo, GOAT, AutoRedTeamer), and embedding‑based memory or lifelong learning (AutoDAN‑Turbo, AutoRedTeamer). The paper’s main conceptual move is to package these into a three‑phase “Planner–Primer–Finisher” decomposition with retrieval‑based strategy memory. However, the paper sometimes frames PLAGUE as “the first” multi‑turn attack with lifelong learning (Section 2.3), which feels overstated given the close parallels to prior adaptive/memory‑based red‑teaming agents; the distinctions from AutoRedTeamer, FITD, X‑Teaming, and other contemporary multi‑turn systems are not deeply analyzed beyond a checklist (**Table 1**). It would strengthen the work to articulate more clearly *what can be done with PLAGUE that cannot be straightforwardly implemented by reconfiguring those existing frameworks*.

2. **Heavy reliance on LLM‑as‑a‑Judge with potential circularity and bias.**  
   The framework uses a Qwen3‑235B‑A22B model as both the Rubric Scorer $\mathbb{R}$ (during planning, priming, and finishing) and the final Evaluator/Judge $\mathbb{J}$ for SRE and binary ASR, and uses the same family of models for goal labeling in HarmBench (Section 4, Appendix C). Since the attacker is also an LLM (DeepSeek‑R1), PLAGUE is effectively tuned to maximize the judgment of a specific evaluator. This raises several concerns:
   - There is no cross‑evaluator robustness test (e.g., using a different safety‑tuned evaluator model, or a human subset) to show that PLAGUE’s gains are not artifacts of overfitting to one evaluator’s rubric.
   - Because the Rubric Scorer provides fine‑grained signal that guides both backtracking and lifelong learning, PLAGUE might learn to produce responses that Qwen judges as highly specific and convincing, which may not always align with human safety risk.  
   These issues are central, since **all quantitative claims in Table 2, 3, 4, 5, and 6 depend on this evaluator**. At minimum, a robustness study under a second independent judge, or a small human study, would significantly increase confidence that improvements are truly meaningful.

3. **Somewhat thin baseline tuning and fairness details, especially for newer multi‑turn works.**  
   While many baselines are compared, the paper frequently modifies them “for fair comparison” but without always specifying whether hyperparameters were re‑tuned or left at default. For example, in Section 4 (Baselines) GOAT is run “without history enabled” and early‑stopped when a high rubric score is obtained. For ActorBreaker, the number of plans is restricted to 2 (ASR@K), and Crescendo’s explicit backtracking counts are removed and the total turns limited to 6. For FITD and X‑Teaming (Appendix C.4, **Table 6**), the attack budgets and TextGrad steps are restricted relative to their original configurations. While a common 6‑turn target budget is reasonable, these changes may systematically handicap some baselines that were designed to leverage longer or more flexible conversation horizons. The paper does not report any sensitivity analysis (e.g., performance vs. number of actors in ActorBreaker, vs. TextGrad steps in X‑Teaming) to show trends are preserved. Given the very large improvements claimed (e.g., 40.2% over Crescendo on Opus 4.1), more detailed baseline tuning discussion is needed.

4. **Mathematical and algorithmic description has some inconsistencies and missing clarifications.**  
   There are several points where the formalism diverges from the implemented algorithms or leaves key hyperparameters unclear:
   - In Section 3.2, ASR is defined as  
     \[
     ASR(\mathbb{J})=\frac{1}{P}\sum_{i=1}^{P}\mathbb{J}(p_i,\mathbb{MT}_i),
     \]
     but later in Section 4, ASR@K is used with a best‑of‑$K$ choice over multiple attack attempts per goal and over multiple runs. It is not made clear whether $\mathbb{J}$ in Equation (ASR) denotes the SRE score, the binary ASR, or both; nor how the best‑of‑2 heuristic is mathematically integrated.  
   - The Primer and Finisher algorithms in **Algorithm 2** and **Algorithm 3** nominally use “Evaluator model $\mathbb{J}$” in the pseudo‑code, but the text says feedback during these phases comes from the Rubric Scorer $\mathbb{R}$, not from the StrongReject evaluator. This mismatch between notation and implementation is confusing and should be corrected; one could, for instance, define two distinct scorers $\mathbb{R}_\text{rubric}$ and $\mathbb{R}_\text{SRE}$ and clearly state which is used where.
   - The success thresholds are inconsistent: Section 3.5 states that a score > 8/10 marks an attack as successful, but Algorithm 3 (Finisher) marks success at score > 9.0 (line 10) and stores the successful strategy only in that case. It is unclear which threshold is used when computing the reported SRE and binary ASR, and whether attacks with scores between 8 and 9 are counted as successful for evaluation or not.  
   These discrepancies do not appear fatal, but they undermine reproducibility and complicate the interpretation of performance numbers.

5. **Limited exploration of planner design and strategy retrieval beyond a single threshold.**  
   Section 3.3.1 fixes a cosine similarity threshold of 0.6 and a maximum of two in‑context examples, with random fallback if fewer are found. There is no analysis of how sensitive performance is to this threshold, to the number of examples, or to the initial seeding of the strategy library (currently a small set adapted from Crescendo). Given that lifelong learning and retrieval are central claims, I would expect at least a small sweep or qualitative analysis: e.g., does lowering the threshold to 0.4 or increasing to 0.8 change ASR/diversity? Does the system still improve significantly if initialized with no human‑designed strategies? As is, it is hard to know whether the benefits in **Table 3** are robust, or mostly due to a carefully hand‑tuned retrieval regime.

6. **Diversity claims remain secondary relative to ASR and are only partially quantified.**  
   While **Figure 3** and Equation (1) give a quantitative diversity metric, the main paper gives only high‑level numbers (e.g., “15.47% improvement in diversity over the base Planner version”) without providing the exact diversity scores per model or per configuration in a table. The diversity evaluation aggregates across models and only considers prompts that succeed at least twice per goal. There is no analysis of per‑category diversity (e.g., does PLAGUE over‑concentrate on certain stereotyped attack narratives for hate speech or terrorism?). Given the safety implications, deeper analysis of *qualitative* diversity, including potential mode collapse in strategy space, would make the lifelong‑learning claim more convincing.

7. **Ethical discussion remains brief relative to the potential for misuse.**  
   Section 7 acknowledges dual‑use risk but focuses on the importance of open access. There is little concrete guidance for responsible deployment of PLAGUE (e.g., recommended restrictions, logging, or gating when used in industry; how to prevent its use by non‑researchers). Considering the extremely high reported ASRs (e.g., 97.8% on DeepSeek‑R1, near‑perfect on misinformation), more thorough treatment of misuse risk, access control, and potential for adversaries to directly weaponize the provided prompts and strategies would be appropriate.

8. **Missing or weak positioning against several directly related recent multi‑turn frameworks.**  
   The Related Work section (Section 2) covers Crescendo, GOAT, ActorBreaker, RACE, FITD, X‑Teaming, AutoRedTeamer, and AutoDAN‑Turbo, but omits several recent works that are very close in spirit (see next section). Some of these also propose modular or adaptive multi‑turn jailbreak frameworks and would be natural baselines or at least discussion points. The omission makes it harder to judge where PLAGUE sits in the quickly evolving landscape.

## Potentially Missing Related Work

1. **Feng et al., “SEMA: Simple yet Effective Learning for Multi‑Turn Jailbreak Attacks”, 2026.**  
   This work appears to focus specifically on learning multi‑turn jailbreak strategies without relying heavily on existing human‑crafted strategies, which is directly relevant to PLAGUE’s lifelong‑learning, strategy‑retrieval aspects. It should be discussed in Section 2.3 (“Agentic and Lifelong Learning Frameworks”) as another approach to training adaptive attackers, and, if feasible, included as a baseline in **Table 2** or **Table 6**, or at least qualitatively compared in terms of data requirements and planning structure.

2. **Liu et al., “CoaxChain: Semantically Progressive Multi‑turn Jailbreak Attacks on Large Language Models”, 2025.**  
   CoaxChain explicitly emphasizes semantic progression in multi‑turn attacks, which is highly related to PLAGUE’s Primer phase and claim of avoiding semantic drift. This paper should be discussed in Section 2.2 (Multi‑Turn Red‑Teaming), with explicit comparison of how PLAGUE’s plan‑anchored priming differs from CoaxChain’s progression mechanism, and how each handles relevance and drift (potentially linked to **Figure 4**’s category‑wise performance).

3. **Wu et al., “Analogy‑Based Multi‑Turn Jailbreak Against Large Language Models”, 2025.**  
   This paper uses analogical reasoning to construct multi‑turn jailbreaks, which can be viewed as another strategy family within PLAGUE’s planning space. It would fit naturally into the discussion of planning strategies in Section 3.3 and could be mentioned as a type of strategy that could populate PLAGUE’s $\mathbb{R}^{\{+\}}$ memory library. Including it would clarify the breadth of planning paradigms PLAGUE can subsume.

4. **Sun et al., “Multi‑Turn Context Jailbreak Attack on Large Language Models From First Principles”, 2024.**  
   This work provides a more theoretical foundation for why multi‑turn context gradually erodes safety alignment. It is directly relevant to PLAGUE’s motivation in Section 1 and its use of Primers to build adversarial context. The authors should discuss how PLAGUE’s three‑phase decomposition relates to the principles derived in Sun et al., and whether PLAGUE can be interpreted as an instantiation of those theoretical insights.

5. **Choi et al., “MAPA: Multi‑turn Adaptive Prompting Attack On Large Vision‑Language Models”, 2025.**  
   Although MAPA targets vision‑language models, it is structurally similar in that it adapts prompts across multiple turns for jailbreak. It should be briefly mentioned in Related Work (likely Section 2.3) as an extension of the multi‑turn attack paradigm to multimodal settings, and as a potential target domain where PLAGUE’s plug‑and‑play architecture might generalize.

6. **Narula et al., “HarmNet: A Framework for Adaptive Multi‑Turn Jailbreak Attacks on Large Language Models”, 2025.**  
   HarmNet appears to be another modular framework for adaptive multi‑turn attacks, conceptually very close to PLAGUE. It should be discussed side‑by‑side with PLAGUE in Section 2.2 or 2.3, with a comparison similar to **Table 1** (e.g., presence of lifelong memory, planning, reflection, external knowledge) and with commentary on how PLAGUE’s planner/primer/finisher separation differs from HarmNet’s architecture.

## Questions

1. **Evaluator robustness and cross‑judge validation.**  
   Can the authors provide results where SRE and binary ASR are computed using a *different* evaluator model (e.g., a strong safety‑aligned model from another provider, or a smaller but independently fine‑tuned judge)? Even a subset of goals evaluated by a second judge (or by humans) would help quantify how much PLAGUE’s performance is tied to Qwen3‑235B‑A22B’s scoring behavior.

2. **Clarifying the success thresholds and algorithmic inconsistencies.**  
   Please clarify the precise operational definition of a “successful” attack in your implementation:  
   - What threshold is used in practice for storing strategies in the memory bank (Algorithm 3 uses >9) vs. for counting a run as successful in SRE/binary ASR (>8 in Section 3.5)?  
   - In Algorithms 2 and 3, are scores supplied by $\mathbb{R}$ or $\mathbb{J}$? Updating the notation and providing a short “signal flow” diagram would help.

3. **Sensitivity to retrieval hyperparameters and initial strategy library.**  
   How does performance change if the cosine similarity threshold for strategy retrieval is varied (e.g., 0.4, 0.5, 0.7, 0.8), or if you allow 1 vs. 2 vs. 3 in‑context examples? Could you provide a small ablation, at least on one model (say o3), to show that the benefits in **Table 3** are not brittle? Additionally, what happens if the initial Crescendo‑adapted strategies are removed and the memory is grown purely from PLAGUE‑discovered strategies?

4. **Fairness of baseline budgets and the impact of parameter choices.**  
   For ActorBreaker, GOAT, Crescendo, FITD, and X‑Teaming, can you provide either (a) references to prior work showing that your chosen budgets (e.g., 2 actors, 6 turns, 2 TextGrad steps) are near‑optimal, or (b) a brief sensitivity curve for at least one baseline to demonstrate that performance does not dramatically increase just beyond your chosen settings? This would help alleviate concerns that some baselines are artificially constrained.

5. **Qualitative analysis of lifelong learning and diversity.**  
   Beyond **Figure 3**, could you include a few concrete examples where retrieval from $\mathbb{R}^{\{+\}}$ clearly changes the course of an attack (e.g., side‑by‑side dialogues with and without retrieval for the same goal)? This would make the claim of “lifelong learning” more tangible and may also reveal potential failure modes like over‑fitting to a small set of high‑scoring strategies.

6. **Mitigation and responsible release.**  
   Given your extremely high ASRs, what safeguards do you propose for releasing PLAGUE code and prompts? For instance, will you restrict usage to safety‑research contexts, or include guardrails (e.g., whitelisting target models, logging, or gating access)? Clarifying this in the Ethics section would be valuable.

## Flag For Ethics Review

Yes, Potentially harmful insights, methodologies and applications  

(Other categories: No)

## Details Of Ethics Concerns

The work directly develops and evaluates **highly effective multi‑turn jailbreak attacks** that can reliably elicit harmful content (e.g., instructions for disabling GPS trackers on scooters, as in Appendix D) from frontier LLMs. Publishing this as open‑source, with detailed prompts and lifelong strategy libraries, could significantly lower the barrier for malicious actors to bypass safety measures. The ethics section acknowledges dual‑use but does not specify any concrete mitigation strategy (licensing, gating access, or usage policies). An ethics review should consider:  
- How to responsibly release PLAGUE’s code, prompts, and retrieved strategies.  
- Whether some of the most dangerous attack examples should be redacted or abstracted.  
- Potential downstream harms if PLAGUE is used directly by non‑researchers to attack production systems.  

## Soundness Rating

3: good.  
The methodology is technically coherent, the algorithms are reasonably specified, and experiments are broad and largely well‑controlled, though some implementation details (success thresholds, scorer notation, baseline budgeting) need clarification and the reliance on a single evaluator model is a notable limitation.

## Presentation Rating

3: good.  
The paper is generally well‑written and well‑structured, with informative figures (especially **Figure 1**, **Figure 2**, **Figure 3**, and **Figure 4**) and tables (**Tables 2–6**). Some notation inconsistencies and missing hyperparameter details slightly detract from clarity.

## Contribution Rating

3: good.  
The paper provides a strong empirical contribution and a practical framework that appears genuinely useful for safety evaluation, but its conceptual novelty over prior agentic and multi‑turn attack frameworks is moderate rather than transformative.

## Overall Rating

8: Accept, good paper (poster).  
The work presents a well‑designed and empirically strong framework for multi‑turn jailbreaks that advances the state of practice in LLM safety evaluation. While not radically new conceptually and with some concerns around evaluator bias and baseline tuning, it clearly improves attack strength against very capable models under a realistic query budget, and the modular decomposition plus lifelong memory are likely to be influential for both red‑team and defense research.

## Reviewer Confidence

4: confident.  
I am familiar with LLM safety and jailbreak literature (GOAT, Crescendo, ActorBreaker, AutoRedTeamer, etc.) and carefully examined the algorithms, equations, and experimental tables/figures. Some implementation details are not fully specified, but I am reasonably confident in my assessment.