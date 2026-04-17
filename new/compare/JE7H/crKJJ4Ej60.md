---
job_id: 52db0840-0819-4a43-8069-540961455a31
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: crKJJ4Ej60.pdf
paper: Copy-Paste to Mitigate Large Language Model Hallucinations
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper tackles hallucination mitigation and contextual faithfulness in retrieval-augmented LLMs, combining preference optimization, prompting, and interpretability, which fits well within ICLR’s focus on representation learning, language models, reliability, and interpretability.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Methodology, Experiments, Results, Related Work, Conclusion) are present and the work is technically substantial with nontrivial experiments and analysis. I do not see fatal methodological errors or missing experimental design elements that would force a desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
The paper includes prompt templates for experiments, but there are no hidden instructions targeting reviewers or other manipulative content.

---

# Expected Review Outcome:

## Summary

The paper proposes *Copy-Paste*, a paradigm for retrieval-augmented generation that explicitly encourages models to copy lexical fragments from the provided context to improve contextual faithfulness and reduce hallucinations. The approach has two stages: (1) Copy-Paste-Prompting (CP-Order, CP-Link, CP-Refine) generates high-copying candidate answers under hard/soft constraints, and (2) CopyPasteLLM uses DPO on automatically constructed preference pairs to internalize a preference for high-copying, context-grounded responses. Extensive experiments on RAGTruth, FaithEval, ConFiQA, and PubMedQA show strong gains in both counterfactual and original settings with very few training samples, and the authors introduce a Context-Parameter Copying Capturing algorithm to analyze how CopyPasteLLM reallocates reliance between contextual and parametric knowledge during CoT reasoning.

## Strengths

1. **Clear and concrete main idea, strongly supported by results.**  
   The central hypothesis that increased lexical copying from context correlates with reduced hallucination is simple but well-motivated (Figure 1 bottom). The authors operationalize this via multiple prompting schemes and DPO, and then convincingly show that high-copying answers indeed improve contextual faithfulness and downstream accuracy, especially in counterfactual regimes where parametric priors are misleading.

2. **Very strong empirical performance with tiny fine-tuning data.**  
   Table 1 is compelling: CopyPasteLLM, trained using preference pairs derived from only 365 query–context examples, substantially outperforms strong fine-tuned baselines like Context-DPO (18k examples), Canoe (10k), and ParamMute (32k+) on FaithEval and ConFiQA counterfactual tasks. On FaithEval, accuracy jumps from ~68–80% for the best baselines to ~89–93% (+12.2–24.5 points). Table 3 further shows that in non-counterfactual settings CopyPasteLLM maintains or improves accuracy across PubMedQA and the ConFiQA subsets, especially for the harder MR/MC splits.

3. **Stage-1 prompting methods are systematically designed and evaluated.**  
   CP-Order, CP-Link, and CP-Refine are not ad hoc prompts but a clear design space: hard extractive ordering, extractive with small generative connectors, and an iterative writer–reviewer refinement with a quantitative copy score. Table 2 shows these methods yield notably higher faithfulness (AlignScore and MiniCheck) and lower hallucination rates than Attributed/Citations across four base models and three datasets. Figure 5 and Figure 6 nicely visualize how these methods increase copy coverage/density and still maintain high query relevance, particularly for CP-Refine.

4. **Thoughtful preference-construction and DPO pipeline.**  
   The Stage-2 pipeline (Figure 2, Algorithm 2) is carefully engineered: candidates from six generation modes are filtered by multiple metrics (faithfulness, copying degree, relevance, fluency), ranked via an Elo-style LLM-as-judge tournament that differentiates Twist vs Causal hallucinations, and then “stamped” with correct or incorrect final answers when possible. The ablation in Figure 12 shows that removing high-copying data (w/o Copying) or removing answer stamping (w/o Stamping) significantly hurts both Hit Rate and Accuracy, underscoring that these are not cosmetic choices.

5. **Interpretability contribution and analysis of knowledge source reliance.**  
   The Context-Parameter Copying Capturing algorithm (Algorithm 4) provides a concrete way to capture token-level reliance on context vs parametric knowledge, using parallel runs with and without context and top-\(K\) token inspection. Figure 3 and Figure 13 show that CopyPasteLLM shifts logit power from parametric to contextual tokens and peaks earlier in the response, while Figures 4, 14, and 15 reveal via UMAP that contextual hidden representations stay close to the base model but parametric ones become clearly separated. This supports the paper’s narrative that CopyPasteLLM “recalibrates” parametric knowledge confidence rather than radically altering contextual representations.

6. **Reasonably thorough analysis of behavior, not just headline numbers.**  
   The appendix sections are rich:  
   * Figure 7 and Figure 8 break performance down by knowledge type and reasoning difficulty.  
   * Table 5 and Figure 11 analyze response length and copying degree across baselines vs CopyPasteLLM, arguing that CopyPasteLLM achieves “rational” rather than blind copying.  
   * Figure 12 details training dynamics and stability across seeds, giving some confidence that the method is not brittle.  
   These analyses go beyond typical RAG papers and make the story more convincing.

7. **Clarity and organization.**  
   The paper is generally well written and easy to follow. The main pipeline is visually clear in Figure 2, the metrics \(\kappa\) and \(\delta\) are formally defined in Equation (1), and the mechanistic arguments in Appendix A link the copy-paste objective to attention anchoring and entropy reduction in a reasonably understandable way.

## Weaknesses

1. **Conceptual novelty is moderate; copying as faithfulness proxy is not entirely new.**  
   While the integration and empirical validation are strong, the core idea that high lexical overlap with context can mitigate hallucinations is closely related to earlier extractive summarization and citation-style generation work. CP-Order and CP-Link are essentially structured extractive QA with sentence reordering and light connective generation. The main methodological leap is to treat high-copy responses as DPO preferences. The paper would benefit from a more explicit comparison to prior “extractive RAG” / “copy-based decoding” paradigms beyond CoCoLex, and a clearer positioning of what is *conceptually* new relative to:  
   * Extractive summarization via LLMs (e.g., Zhang et al., 2023, which is cited but not deeply contrasted).  
   * Hard-copy decoding schemes or constrained generation typically used in data-to-text or legal RAG.  
   Right now, Section 5 acknowledges related classes of methods but somewhat glosses over how similar mechanisms have been explored.

2. **High-copying ≠ faithfulness; proxy is not formally or empirically stress-tested.**  
   The paper implicitly equates high copy coverage/density with contextual faithfulness. However, it is entirely possible to copy irrelevant or misleading context fragments (e.g., copying distractor sentences or outdated claims) and still score high in \(\kappa,\delta\) while being unfaithful to the *query* or to the “right” contextual evidence. Some signs of this appear in Table 2, where CP-Order and especially CP-Link can have worse fluency and inconsistent hallucination metrics despite very high faithfulness scores, suggesting the metrics may be partially conflated.  
   The paper partially addresses this by also enforcing query relevance and fluency in filtering, and by analyzing query relevance of copied spans in Figure 11. However, there is no explicit adversarial or counterexample study to probe failure modes where high copying leads to wrong answers or misleading emphasis. A more comprehensive error analysis would strengthen the claim that copying degree is a robust operational proxy for faithfulness, not just a beneficial heuristic on the tested datasets.

3. **Heavy reliance on automatic and LLM-based metrics for hallucination and preferences.**  
   Many of the key results hinge on AlignScore, MiniCheck, and an LLM-as-judge tournament (Qwen3-32B) for Twist/Causal hallucinations. Table 2’s “Hallu.” columns are entirely based on such judgments, and LLM-as-judge is also used to rank preferences for DPO (Algorithm 2). There is no human evaluation of hallucination or answer quality, and no robustness check showing that the conclusions are stable across different judges or scoring models. This is particularly concerning because CopyPasteLLM responses are structurally very different (high copying, sometimes stilted) from abstractive baselines, which could systematically bias both AlignScore and the LLM judge. At minimum, inter-judge consistency experiments or a second judging model would alleviate this concern.

4. **Some parts of the mechanistic story are more speculative than rigorous.**  
   Appendix A’s attention/entropy explanations are interesting but not entirely tight. For example, Equation (4),  
   \[
   \lim_{\text{copying}\rightarrow\text{max}}\sum_{j\in\mathcal{C}}\alpha_{t,j}\approx 1,
   \]  
   is asserted without a clear derivation; induction heads are empirically motivated but the argument is qualitative. Similarly, Equations (5–7) argue that CopyPasteLLM’s conditional entropy is “strictly lower” than base, yet no formal bound is proven, nor are the empirical logits distributions (Figure 3 / Figure 13) actually used to measure entropy. The high-level intuition is fine, but as a “mechanistic interpretation” it reads a bit hand-wavy and should be framed more carefully as a hypothesis supported by Figures 3–4 rather than as a quasi-theorem.

5. **Context-Parameter Copying Capturing design and assumptions need more scrutiny.**  
   Algorithm 4 selects top-\(K\) tokens, skips “meaningless” tokens, and then labels the first token that appears in context (and not yet in Tcts) as contextual knowledge, otherwise in the context-free answer as parametric. Several aspects are under-specified or debatable:  
   * The function `isMeaningless(x_j)` is not defined mathematically; removing function words may bias the analysis toward content tokens and miss systematic effects in how models use non-content tokens to structure reasoning.  
   * Limiting to *distinct* tokens via Tcts and Tpars means frequent tokens may be undercounted, which could distort the logit power computation in Equation (8).  
   * Using presence in the context-free answer as a proxy for “parametric knowledge” ignores the fact that many copied tokens are also high-probability in the context-free run. The algorithm tries to avoid overlap via Scom, but this is a somewhat brittle heuristic.  
   These choices might influence the striking separation in Figures 3–4. The authors should at least provide sensitivity analyses to \(K\), to the “meaningless” filter, and to the Tcts/Tpars uniqueness constraint, or acknowledge more clearly that these visualizations are exploratory rather than definitive.

6. **Evaluation breadth is good, but domains and task types are still relatively narrow.**  
   The datasets are QA-style RAG tasks: RAGTruth (daily life), FaithEval (science MCQ with counterfactual passages), PubMedQA (biomedical QA), and ConFiQA (Wikidata-based conflicts). This is a good variety within QA, but all tasks are essentially “choose the right fact from a provided passage”. It remains unclear whether CopyPasteLLM would help in more generative or structured settings (e.g., long-form summarization, multi-document report generation, code generation with API docs) where copying large spans verbatim may hurt readability or be infeasible. At least a qualitative example set or discussion of how copy-paste behaves for less extractive tasks would be valuable.

7. **Copying vs user utility not fully quantified.**  
   Table 5 shows CopyPasteLLM responses are shorter than typical abstractive baselines but longer than CoCoLex or ParamMute, and Figure 11 analyzes query relevance of copied fragments. However, there is no direct human evaluation of readability or satisfaction, nor task-specific metrics that explicitly measure utility rather than just correctness/faithfulness. For high-stakes domains like medicine, overly verbose or verbatim copying can be problematic (e.g., repeating confusing or hedge-laden clinical statements). The ethics section briefly mentions this risk, but more thorough discussion or simple readability metrics (beyond perplexity) would help.

8. **Some notational and implementation details are unclear or slightly inconsistent.**  
   A few examples:  
   * Equation (1) defines \(\kappa\) and \(\delta\) as averages over copy fragments; however, the description subsequent to Eq. (1) says “copy coverage: fraction of answer tokens that are covered by some copy fragment”; mathematically, if fragments are disjoint this is equivalent, but the algorithm (Algorithm 3) does not explicitly guarantee non-overlap. Clarifying whether overlapping fragments are merged would avoid confusion.  
   * The DPO loss in Algorithm 2 uses \(y_w\) and \(y_l\) but the text alternates between “chosen” and “rejected” responses; there is a minor typo in line 13 (`r_i^{\mathrm{down}}` vs `r_i^{\mathrm{descr}}`) which makes that line hard to parse.  
   * The paper states that each sample yields “roughly five preference pairs” (Section 3.2, line about “5× data efficiency”), but Appendix G mentions training each ablation on “365 preference pairs for 2 epochs (218 steps)”. It would be good to clearly state the exact number of *pairs* vs *base samples* used for main experiments.

Overall these are not fatal, but they chip away at the otherwise clean presentation.

## Potentially Missing Related Work

(All of the following appear to be uncited in the main text.)

1. **Béchard & Ayala, 2024 – “Reducing Hallucination in Structured Outputs via Retrieval-Augmented Generation.”**  
   Directly addresses hallucination reduction in RAG, particularly for structured output. It would be relevant to cite in Section 1 or 5 as part of the landscape of RAG-based hallucination mitigation techniques and to contrast their structural constraints with the copy-paste constraints proposed here.

2. **Zhang & Zhang, 2025 – “Hallucination Mitigation for Retrieval-Augmented Large Language Models: A Review.”**  
   A survey specifically on hallucination mitigation in RAG. This should be referenced in the Introduction and Related Work when motivating the remaining challenges in contextual faithfulness and positioning CopyPasteLLM among prompting, decoding, and fine-tuning approaches.

3. **Gupta, 2025 – “Retrieval-Augmented Generation and Hallucination in Large Language Models: A Scholarly Overview.”**  
   Another broad overview of hallucinations and RAG strategies. It would help contextualize the problem formulation in Section 2.1 and the methodological landscape surrounding RAG hallucination mitigation.

Including these works would better situate the paper within the growing literature on RAG hallucinations and clarify how copy-paste preference learning complements or differs from existing mitigation approaches.

## Questions

1. **Robustness of Context-Parameter Copying Capturing.**  
   Could the authors provide sensitivity analyses of Algorithm 4 to key hyperparameters and heuristics, such as the choice of \(K\), the `isMeaningless` filter, and the uniqueness constraint via Tcts/Tpars? For example, do Figures 3–4 qualitatively change if all tokens (including frequent and function words) are considered, or if multiple contextual/parametric tokens per step are allowed?

2. **Failure modes of high-copy responses.**  
   Have you examined qualitative failure cases where CopyPasteLLM copies irrelevant or misleading parts of the context (e.g., distractor sentences in RAGTruth, outdated or caveated statements in PubMedQA), leading to wrong answers despite high \(\kappa,\delta\)? A small table or figure illustrating such cases would help calibrate when copy-paste is *not* desirable.

3. **Generalization to more abstractive tasks.**  
   Do you have any preliminary evidence (even small-scale) on how CopyPasteLLM behaves on more open-ended generative tasks, such as multi-paragraph medical summaries or law case overviews, where heavy copying might reduce readability? If not, could you discuss concrete strategies for relaxing the copy constraint (e.g., dynamic thresholds on \(\kappa,\delta\)) in such settings?

4. **LLM-as-judge dependence.**  
   How sensitive are your Elo-based preference rankings and hallucination measurements to the choice of judging model? For example, if you substitute another strong model (e.g., a different family) for Qwen3-32B, do you still see similar relative rankings and CopyPasteLLM improvements? Any small-scale sanity check or correlation analysis would increase confidence.

5. **Entropy / attention calibration measurement.**  
   To support the mechanistic claims in Appendix A more strongly, could you directly measure token-level entropy of the next-token distribution and context-attention mass for base vs CopyPasteLLM (e.g., average fraction of attention on context tokens), and show whether they align with Equations (4) and (7) and the qualitative discussion?

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A. The work uses standard public datasets and focuses on reducing hallucinations. The ethics section appropriately notes potential over-reliance on copied content but does not raise obvious additional concerns.

## Soundness Rating

3: good.  
The experimental methodology is generally sound, with strong baselines, multiple datasets, ablations (Figure 12), and rich analysis (Tables 1–3, 5; Figures 3–5, 7–8, 11). Some mechanistic arguments are more speculative, and the reliance on automatic / LLM-based judgments for both evaluation and preference construction is a weak point, but the central empirical claims are well supported.

## Presentation Rating

3: good.  
The paper is well structured, the figures (especially Figures 1, 2, 3, 4, 5, 12) are informative, and most equations and algorithms are clearly described. A few notational typos and under-specified components (e.g., Algorithm 4 heuristics) could be cleaned up, but overall clarity is above average.

## Contribution Rating

3: good.  
Conceptual novelty is moderate but the combination of high-copy prompting, preference optimization, and interpretability is executed at a high level, with strong empirical impact and practical relevance for RAG systems. The data efficiency demonstrated in Table 1 and the mechanistic insights from Figures 3–4 make the contribution valuable to the ICLR community.

## Overall Rating

8: Accept, good paper (poster).  
Despite some conceptual and methodological caveats (proxy nature of copying, LLM-as-judge dependence, mechanistic speculation), the work makes a solid, well-executed contribution to contextual faithfulness in RAG, with impressive data efficiency and thorough analysis. It is clearly above the bar for an ICLR poster.

## Reviewer Confidence

4: confident.  
I am familiar with RAG, DPO, and hallucination literature, and I carefully examined the equations, algorithms, and key figures/tables. Some details of Algorithm 4 and the internal prompt engineering choices might still hide subtleties, but I am unlikely to have missed major issues.