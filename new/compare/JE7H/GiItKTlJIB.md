---
job_id: 398eef16-0af0-43a6-9d42-66d4d096000d
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: GiItKTlJIB.pdf
paper: How Much Chain-of-Thought Do LLMs Really Need for Physics?
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work studies chain-of-thought usage in LLMs on physics reasoning benchmarks using deletion-based probes, which clearly fits ICLR topics on reasoning, interpretability, and AI for science.

## Minimum Quality
Pass ✅.  
The paper includes Abstract, Introduction, Problem Setup/Methodology, Experiments/Results, Analysis & Discussion, Conclusion, and Related Work. The contributions are non‑trivial, the methodology is described in reasonable detail, and empirical results are substantive, even though there are notable weaknesses.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any hidden prompts, attempts to steer LLM reviewers, or other manipulative content in the main text.

---

# Expected Review Outcome:

## Summary

The paper investigates how much chain-of-thought (CoT) LLMs actually “use” when solving physics problems. The authors propose a deletion-based probing framework that intercepts CoT mid-generation, removes different fractions of tokens according to several strategies (end, random, physics-aware), and measures downstream task performance, answer length, and lexical overlap between deleted content and final answers. Experiments on three physics benchmarks (UG Physics, PhysReason, PhyBench) and three reasoning-focused open-source LLMs (Phi-4, Qwen-A3B, Magistral) show that accuracy is surprisingly robust to heavy deletions (roughly 40–60%) and that models often “cram” reconstructed steps into the final answer, raising concerns about the faithfulness of CoT in scientific problem solving.

## Strengths

1. **Interesting and timely question for AI-for-Science and CoT faithfulness.**  
   Focusing on physics as a structured, equation-heavy domain is a good choice to study CoT faithfulness, where correctness is not purely semantic but numerically and dimensionally constrained. The high-level question, “How much CoT do LLMs really need for physics?”, is well framed and relevant for both evaluation and deployment of reasoning models in scientific workflows.

2. **Deletion-based probing framework is simple, concrete, and easy to reproduce.**  
   The core experimental protocol, intercepting CoT mid-generation and deleting $k\%$ of tokens before continuing to decode, is clearly defined and could be readily applied by others. The distinction between three deletion strategies (from-the-end, random, physics-aware) is sensible and probes different aspects of CoT reliance. **Figure 3** illustrates this cleanly by contrasting “Annotated” (physics-structured) vs “Non-Annotated” deletions and showing that removing physics-specific tokens harms accuracy more.

3. **Systematic empirical exploration across multiple models and benchmarks.**  
   Evaluating three fairly strong reasoning models (Phi-4, Qwen-A3B, Magistral) on three separate physics benchmarks provides a decent breadth of coverage. The calibration study in **Figure 8** showing convergence of error bars around 5 runs per problem is a positive sign that the authors at least thought about sampling variance.

4. **Quantitative characterization of “cramming” using answer length.**  
   The paper does more than just report accuracy under deletion: it measures final answer length and documents a consistent X-shaped pattern where scores remain roughly stable up to a deletion threshold while answer length increases. **Figure 4** and **Figure 5** (and their detailed counterparts in **Figures 6, 9–11, 12–14**) clearly visualize this cramming pattern across deletion fractions and models, which is an insightful, easy-to-interpret empirical phenomenon.

5. **Overlap analysis with explicit metrics and equations.**  
   The information-overlap section is technically clear. Equations (1) and (2) define Jaccard similarity and Manhattan distance on bag-of-words in standard notation. **Figure 7** then reports how these metrics change as deletion fraction increases, separating end, random, and physics-aware deletions. The observation that overlap grows with deletion fraction in many conditions, but often does not rescue final accuracy, supports the claim that models can regenerate similar surface-level content without faithfully reusing the original scratchpad.

6. **Balanced discussion of faithfulness implications and limitations.**  
   Section 4.3 acknowledges that CoT is “informative and redundant” and explicitly distinguishes faithfulness from interpretability or explainability. The limitations section is honest about scope (physics only, three models, observable outputs only). The discussion of potential practical implications (e.g., trading off early stopping of CoT for compute savings vs. faithfulness) is reasonable and grounded in the results.

## Weaknesses

1. **Lack of explicit, formal description of the deletion mechanism and generation process.**  
   The core methodological object of the paper is the “intercept and delete” procedure, yet this is only described in words and figures, not as a formal algorithm. Critical details are underspecified:
   - At what token index is the CoT intercepted relative to the model’s internal decoding state? Do the authors stop generation once the model prints a delimiter (e.g., “Final answer:”), or do they truncate at a fixed length?  
   - When deleting $k\%$ of tokens, is the deletion applied over the *visible text string* or at the tokenizer level? If tokenization is used, different subword splits can change which physics symbols are destroyed, especially around equations.  
   - After deletion, is the modified prefix re-fed to the model as context, or does generation continue from the same hidden state despite truncated visible text? This is crucial; continuing from the same hidden state would not test reliance on *written* CoT at all, while re-feeding truncated text would.  
   Without clarity on this procedure, the interpretation of results is ambiguous. A minimal pseudo-code listing or formal notation (e.g., context $c$, scratchpad tokens $s_{1:T}$, deletion mask $m$, final prefix $\tilde{s}$) would remove this ambiguity and is currently missing.

2. **Heavy dependence on an external proprietary judge model with limited validation.**  
   All “Score” metrics throughout the paper come from Claude-4 Sonnet serving as a 0–1 grader (Section 2.4). This introduces at least three unaddressed issues:
   - The judge is not calibrated against human expert grading, which is especially important in physics where subtle units or algebra issues matter.  
   - The judge model’s reliability under perturbed CoT traces is unknown; deletion might produce odd formatting or partial derivations that confuse the judge independently of the actual correctness of the final answer.  
   - There is no basic sanity check (e.g., correlation with simple automatic correctness metrics for numerical answers on UG Physics).  
   This undermines the soundness of claims that “accuracy remains stable until ~40–60% deletion,” because this “accuracy” is not tied to ground-truth correctness in a validated way. Some analysis quantifying judge noise or comparing it to at least one automatic metric per dataset is needed.

3. **No explicit quantitative tables; difficult to read off key numbers and compare conditions.**  
   The paper relies entirely on figures and qualitative descriptions, with no results tables summarizing numeric performance (e.g., accuracy at 0%, 40%, 60%, 80% deletion per model/dataset). For instance, **Figure 6** and **Figure 11** show curves with shaded error bars, but one cannot directly read off, say, Phi-4’s UG Physics score at 60% random deletion vs Qwen’s. This makes it hard to rigorously support textual claims such as:
   - “Accuracy remains stable until approximately $40\%$ deletion…”  
   - “Accuracy declines steadily but less abruptly than in random or end deletion…”  
   Tables with concrete numbers and standard errors at key deletion points (0%, 20%, 40%, 60%, 80%, 100%) would make these claims verifiable and allow comparisons between models, datasets, and strategies. Right now, much of the argument relies on eyeballing plots.

4. **Methodological confound between deletion of “annotated physics content” and reliance on a second model for annotation.**  
   Physics-aware deletion (Section 3.2) uses Claude-4 Sonnet again to tag physics-related spans. This raises several concerns:
   - There is no quantitative description of annotation quality or inter-annotator agreement, nor any heuristic check (e.g., random inspection statistics on how often equations or units are mis-tagged or missed).  
   - Because the *same* vendor’s model acts both as annotator and as judge, systematic biases could creep in: if Claude tags certain tokens as “physics-critical,” its judging behavior may also implicitly weight similar patterns, conflating deletion effects with judge expectations.  
   - The ratio between annotated and non-annotated tokens (e.g., equations vs. prose) is never reported, so we cannot interpret curves in **Figure 3** and **Figure 12–14** quantitatively. Deleting $k\%$ of annotated tokens might correspond to vastly fewer or more *total* tokens than random deletion.  
   Without basic statistics on annotation coverage and quality, conclusions about the special role of “physics-aware” deletions are weaker than they appear.

5. **Lexical overlap metrics (Eq. (1)–(2)) only weakly capture faithfulness and are not tied to ground-truth reasoning.**  
   While the Jaccard and Manhattan bag-of-words metrics in Equations (1) and (2) are mathematically sound, they are only crude proxies for information recovery. Several issues are not addressed:
   - Bag-of-words ignores order and structure. In physics, flipping an equation (e.g., $F = ma$ vs. $a = F/m$) or reusing symbol names differently can result in similar vocabularies but very different derivations.  
   - The overlap is computed between *deleted CoT spans* and *final answers*, but there is no alignment with whether the recovered equations are actually used coherently in the derivation judged correct. High Jaccard does not imply causal relevance.  
   - The Manhattan distance is unnormalized; it depends on passage length and vocabulary size, and the “scaled” values shown in **Figure 7** are not clearly defined in the text (what normalization is used? over max distance per dataset? per model?).  
   As a result, statements like “overlap generally increases with deletion fraction, consistent with models attempting to reconstruct lost content” are suggestive but not compelling evidence about faithfulness. A more faithful probe would at least sample a subset of problems and manually or semi-automatically check whether specific deleted *equation types* or *constants* reappear in the correct place in the derivation.

6. **Insufficient baseline or counterfactual analyses to contextualize “cramming” claims.**  
   The central narrative is that models compensate for deletion by increasing final answer length and reintroducing missing content. However, there is no direct comparison to simpler baselines such as:
   - A prompt that *explicitly* asks the model to “be as short as possible” vs. “explain in detail,” controlling for length without deletion.  
   - A condition where CoT is never started and the model is asked to directly output the answer; this would help calibrate how much of the “crammed” reasoning is simply the model’s default style when not guided by a scratchpad.  
   - A “shuffled CoT” baseline where intermediate steps are permuted or replaced with unrelated physics text, to see if the model still reaches similar accuracy and length patterns.  
   Without these, the evidence that length increases are *specifically* a response to CoT deletion rather than just sampling variance or default verbosity remains weaker than it could be.

7. **Limited discussion of dataset specifics and evaluation protocol per benchmark.**  
   Section 2.1 gives only high-level descriptions of UG Physics, PhysReason, and PhyBench but omits important details:
   - Exact number of problems used from each benchmark, and how they are sampled or filtered.  
   - Whether problems with multiple correct answer forms (e.g., symbolic vs. numeric) are handled specially; this matters for judge model instructions.  
   - Any train–test contamination concerns: for reasoning-focused models like Phi-4 or Qwen-A3B, many physics benchmarks might have appeared in pretraining corpora, which could influence robustness under deletion.  
   These missing details make it difficult to assess how representative or challenging the evaluation actually is.

8. **No ablations or error analysis at the level of question types or physics subdomains.**  
   The paper treats each dataset as homogeneous but physics reasoning is highly heterogeneous (conceptual vs computational, mechanics vs electromagnetism, etc.). There is no attempt to stratify results by:
   - Problem type (conceptual explanation vs numeric calculation vs multi-part derivation).  
   - Difficulty level (especially for PhyBench, which is described as Olympiad-style).  
   - Error types (e.g., unit mistakes vs equation mis-application vs sign errors).  
   A modest qualitative or quantitative breakdown could significantly sharpen the conclusions about when CoT is truly helpful and when models can bypass it.

9. **Ambiguity in the interpretation of “stable accuracy until 40–60% deletion.”**  
   The text repeatedly claims that “accuracy remains stable until approximately 40–60% deletion,” referencing **Figure 4**, **Figure 6**, and **Figure 9**, but does not define a threshold for “stable.” Is this within 1 standard error? within 5 percentage points of the baseline? Visual inspection of **Figure 6** suggests that for some model–dataset pairs, there are non-trivial fluctuations even before the red dotted line. A more rigorous statistical criterion (e.g., confidence intervals overlapping baseline) should be specified and applied.

10. **Presentation issues: no tables, figure referencing inconsistencies, and missing details.**  
   While the figures are generally informative, the paper would benefit from better integration into the text:
   - Several figures (e.g., **Figure 9–14**) appear only in the appendix but are heavily relied upon in the main text for key claims (random/physics-aware deletion effects) without precise cross-referencing.  
   - Some figure captions are terse relative to their interpretive load (e.g., **Figure 11** is supposed to show random deletion effects but is only mentioned briefly).  
   - The lack of at least one summary table (see weakness 3) is unusual for an empirical paper and makes it harder for readers to extract key quantitative takeaways.

Overall, the work asks a valuable question and has some nice empirical observations, but methodological clarity, evaluation rigor, and quantitative reporting are not yet at the level I would expect for a strong ICLR main-track paper.

## Potentially Missing Related Work

1. **Yasunaga et al., “Large Language Models as Analogical Reasoners,” 2023.**  
   This work studies how LLMs perform analogical reasoning and proposes evaluation frameworks that probe models beyond surface correctness, directly relating to the paper’s focus on whether CoT reflects genuine reasoning. It should be discussed in the Related Works section alongside other reasoning and CoT faithfulness papers, and briefly compared in the Introduction as an alternative probing paradigm.

2. **Li et al., “Understanding Chain-of-Thought in Large Language Models via Topological Data Analysis,” 2025.**  
   This paper analyzes CoT trajectories using topological data analysis to assess properties of reasoning chains. It is directly relevant to the authors’ goal of understanding how models use CoT and could be referenced in Section 4.3 (Implications for CoT Faithfulness) as a complementary, representation-level approach to their deletion-based behavioral probe.

3. **Lei et al., “Reasoning in Large Language Models: From Chain-of-Thought to Massively Decomposed Agentic Processes,” 2025.**  
   This survey synthesizes recent work on reasoning and CoT, including issues of faithfulness and decomposition. It should be cited in the Related Works section to appropriately situate this paper in the broader reasoning literature, and possibly used in the Introduction to motivate why faithfulness in scientific reasoning tasks is particularly important.

## Questions

1. **Clarify the precise deletion procedure and generation pipeline.**  
   - Do you stop generation after the CoT and before any “final answer” marker, then delete tokens from the CoT text and re-run the model with the truncated CoT as part of the prompt? Or do you modify hidden states directly?  
   - Are deletions applied on tokenized sequences (e.g., subword tokens) or on characters / whitespace-delimited words?  
   A short algorithmic description would greatly increase clarity; please also confirm that no hidden-state continuation is used, since that would fundamentally change the interpretation.

2. **How reliable is the Claude-4 Sonnet judge, especially under perturbed CoT?**  
   - Did you perform any calibration experiments with a subset of questions graded by human physics experts or by simple automatic numeric equality checks (for UG Physics)?  
   - Can you provide statistics on inter-run variance of the judge for fixed model outputs, or a sanity check where you randomize the CoT but keep the same final answer?  
   Evidence that the judge is robust to CoT manipulation would make your main conclusions more convincing.

3. **What is the distribution and quality of “physics-aware” annotations?**  
   - Approximately what fraction of all tokens in the scratchpad are annotated as physics-specific per dataset?  
   - Did you manually inspect a sample of these annotations to estimate precision/recall on equations, constants, and units?  
   Providing these numbers and perhaps a short annotation error analysis would help interpret the physics-aware deletion curves.

4. **Can you add at least one summary table with concrete numbers?**  
   For the rebuttal, it would be extremely helpful to see a compact table with: for each model–dataset–deletion strategy triplet, performance (mean ± SE) at 0%, 40%, 60%, 80% deletion plus average final answer length. This would make your statements about robustness and cramming much easier to verify and compare.

5. **How often do models recover *semantically equivalent* reasoning, not just overlapping tokens?**  
   The overlap metrics are lexical. Can you provide a small-scale qualitative or semi-automatic analysis (on, say, 30 randomly sampled problems) where you check whether key deleted equations (e.g., $F = ma$, $v = v_0 + at$) are reinstated in the correct structural role in the derivation? This would strengthen the faithfulness story beyond bag-of-words similarity.

6. **Do your findings extend to non-physics tasks or to other reasoning paradigms (e.g., self-consistency, tree-of-thought)?**  
   I understand that experiments are scoped to physics. Still, any preliminary evidence or argument about whether deletion sweeps would behave similarly on math or commonsense tasks, or under alternative reasoning protocols like Tree-of-Thought, would help position the work more broadly.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The experimental design is conceptually reasonable and many findings are plausible, but key methodological details (exact deletion pipeline, judge calibration, physics-aware annotation quality) are missing or underdeveloped, which weakens the strength of the causal claims about faithfulness and cramming.

## Presentation Rating

3: good.  
The paper is generally well written, the figures are informative, and the narrative is coherent, though the absence of numerical tables and some underspecified procedures reduce clarity and reproducibility.

## Contribution Rating

2: fair.  
The central idea of deletion-based probing of CoT is interesting and the domain choice (physics) is appropriate, but the work remains largely descriptive with modest methodological novelty and limited evaluation rigor, making the overall contribution incremental rather than strong.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The paper addresses an important question and presents some intriguing empirical patterns (notably cramming and increasing overlap under deletions), but methodological ambiguities, reliance on an unvalidated judge, lack of concrete quantitative tables, and relatively shallow probing of faithfulness prevent me from recommending acceptance at ICLR in its current form. With clearer experimental specification, stronger evaluation of the judge and annotations, and more rigorous quantitative reporting, this line of work could become a solid contribution.

## Reviewer Confidence

4: confident.  
I am familiar with CoT faithfulness literature and LLM evaluation methodology, have carefully checked the math (which is simple and correct) and figures, and feel confident in the assessment, though some details (e.g., exact dataset implementations) are necessarily inferred from the text.