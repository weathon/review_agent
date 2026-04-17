---
job_id: 16f917e1-476e-48e9-b558-2d8ac1a2bcd2
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: X2yzXtH4wp.pdf
paper: Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper focuses on LLM agents, interaction under underspecified instructions, and a new benchmark built on SWE-Bench, which clearly falls under datasets/benchmarks, interactive ML systems, and safety/robustness of large language models in complex software-engineering tasks, all squarely within ICLR’s scope.

## Minimum Quality
Pass ✅.  
All major sections are present (Abstract, Introduction, Method, Experiments/RQs, Results/Analysis, Related Work, Conclusion). The work is clearly written in English, provides concrete methodology and quantitative results, and there are no obvious fatal methodological or theoretical errors.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden instructions attempting to manipulate an automated review or otherwise interfere with the review process.

---

# Expected Review Outcome:

## Summary

The paper introduces **Ambig-SWE**, a variant of SWE-Bench Verified where each fully specified GitHub issue is paired with a GPT‑4o–generated underspecified counterpart, and an interactive evaluation framework in OpenHands where agents can query a user proxy that holds the full specification.  
The authors decompose performance under underspecificity into three capabilities: (i) detecting when an issue is underspecified, (ii) asking clarification questions to recover missing details, and (iii) leveraging those details to successfully resolve issues.  
They evaluate several proprietary and open-weight LLMs across three settings (Full, Hidden, Interaction), analyze resolve rates, interaction patterns, and question quality, and find that interaction can recover much of the performance lost to underspecificity, but most models are poor at detecting missing information and vary widely in how well they integrate user feedback.

## Strengths

1. **Well-motivated and clearly scoped problem**  
   The paper tackles a very practical and under-explored question: how LLM agents behave under *underspecified* software-engineering instructions, not just ambiguous natural-language requests. The definition of underspecificity is grounded in SWE-Bench Verified’s rubric, and the paper focuses on realistic multi-step code-editing workflows rather than toy tasks.

2. **Clean three-setting design with intuitive visualization**  
   The three settings in **Figure 2** (Full / Hidden / Interaction) are very clear and make the core experimental design easy to follow. Showing an example issue and PR in all three conditions, including which information is hidden where, nicely illustrates the causal structure of the evaluation and why the Interaction setting should ideally approximate Full.

3. **Decomposition into three capabilities is insightful**  
   Splitting the problem into (RQ1) leveraging interaction for problem solving, (RQ2) detecting underspecificity, and (RQ3) asking good questions makes the analysis much more actionable than a single aggregate score. For example, **Table 2** shows that Qwen 3Coder completely fails the detection task (FNR = 1.0 across all prompts), even though it performs competitively in resolving issues when forced to interact (Figure 3). This kind of decomposition is exactly what practitioners need when deciding how to train/evaluate agentic models.

4. **Solid empirical characterization across several strong models**  
   The study covers both proprietary (Claude Sonnet 3.5/4, Haiku 3.5) and open-weight models (Llama 3.1 70B, Deepseek‑v2, Qwen 3Coder 480B) within the same OpenHands environment, with per-model statistics on resolve rates, interaction patterns, and number of questions. **Figure 3** and **Table 1** together paint a coherent picture: interaction significantly boosts resolve rates vs Hidden for all models (Wilcoxon tests in **Table 4**), but the degree to which navigational questions help varies sharply, with Qwen sometimes performing worse *with* file locations.

5. **Question-quality analysis goes beyond shallow metrics**  
   The use of both embedding-based cosine distance and an LLM-as-judge to quantify information gain is thoughtful. **Figure 5** (cosine distance) and **Figure 6** (LLM-judge scores) show that models like Qwen 3Coder and Claude Sonnet 4 can extract similar amounts of information but with very different numbers of questions (**Table 6**), leading to nuanced insights: extraction volume is not enough, integration and efficiency matter. The authors also critically discuss why a naive “recovery distance” metric to the full issue (**Figure 7**) is misleading.

6. **Concrete, qualitative examples help ground claims**  
   **Figure 4** and **Table 7** provide side‑by‑side question/answer snippets for the same underspecified issue across models. This makes the high-level claims about Llama asking overly generic questions, Deepseek overshooting user knowledge, and Claude focusing on behavioral aspects much more convincing than aggregate metrics alone.

7. **Relevance and potential impact**  
   Ambig-SWE directly targets a pain point for real-world code agents: operating safely and efficiently under incomplete instructions. The paper surfaces non-obvious behaviors (e.g., Qwen 3Coder’s rigidity and reliance on internal knowledge, Llama’s tendency to ask too few, vague questions) that likely generalize to many deployment settings and should inform future training and evaluation of interactive agents.

## Weaknesses

1. **Synthetic underspecification and external validity concerns**  
   The central underspecified issues are generated by GPT‑4o from SWE-Bench Verified issues (Section 2.1, Appendix A.2.3). While the authors do a distributional comparison to naturally underspecified SWE-Bench issues and provide overlap metrics in **Table 3**, the construction procedure explicitly instructs the model to *remove* crucial information (“abstract enough that a code agent would not be able to solve the issue”). This risks producing a distribution of underspecified inputs that is more extreme and systematically biased than what real users typically write. In practice, users often omit *some* details but still provide partial stack traces, links, or conversational hints. Although the paper acknowledges differences (Page 3) and motivates not using natural underspecified issues due to missing ground-truth specifications, it remains unclear how well conclusions about detection behavior and interaction efficacy will transfer to realistic human-written underspecified tickets.

2. **Heavy reliance on proprietary LLMs as measurement tools**  
   Several aspects of the evaluation pipeline depend on proprietary OpenAI models:  
   - GPT‑4o generates the underspecified issues (Section 2.1).  
   - GPT‑4o is used as the user proxy in interaction (Section 2.2), which may bias the kind of answers and conversational style.  
   - GPT‑4o is used as LLM‑as‑judge for question quality (Section 5.1).  
   - OpenAI embeddings are used for cosine distance (Equation (1) in Appendix A.6, **Figure 5**, **Figure 7**).  
   Using one vendor’s stack for so many roles increases the risk that artifacts of that stack shape the results. For example, the proxy’s strict “I don’t have that information” behavior might be unusually cooperative and consistent compared to real users, and the same family of models is used to evaluate information gain. The paper would be stronger with at least a brief sanity check using an alternative proxy or embedding model, or a discussion of possible biases these dependencies introduce.

3. **Confounding of capability and interaction by differing step limits**  
   For RQ1, Claude Sonnet 4 and Qwen 3Coder are allowed up to 100 tool/action steps, while all other models are capped at 30 steps (Section 3.1). This makes their higher resolve rates in Hidden and Interaction settings harder to ascribe purely to better interaction or reasoning capabilities. In **Figure 3**, Claude Sonnet 4 and Qwen clearly dominate, but some of that advantage may simply stem from being allowed more attempts and exploration. Section 3.2 mentions average steps but does not analyze how much extra budget itself contributes to performance vs improved interaction quality. A controlled ablation (same step budget for all) or at least a per-model performance curve vs step limit would be needed to cleanly separate “better agentic behavior” from “more allowed computation”.

4. **Ambiguity in evaluation protocol for RQ2 (detection)**  
   In Section 4.1–4.2, the detection experiment measures whether a model chooses to interact when given either a Full or Hidden issue, under three prompt styles. However, the paper is vague about exactly *when* a run is counted as “interacted”: is any question to the user at any point in the trajectory counted, or only questions in the first three turns (which are later mentioned as a limitation in Section 7)? Also, in the Strong Encouragement prompt, models are told that asking questions is “critical”, which almost begs them to interact even when the issue is fully specified, making it unclear how to interpret False Positive Rate in **Table 2** under this condition. The resulting patterns (e.g., Sonnet 3.5’s higher FPR with stronger prompting) could be driven as much by prompt semantics as by actual underspecificity detection.

5. **Question-quality metric is under-specified and may conflate quantity and relevance**  
   The cosine distance metric in Section 5.1 and Equation (1) takes embeddings of “summarized task” and “cumulative knowledge after interaction”, but the paper never clearly defines how the latter text is formed. Is it the concatenation of the original summary plus all user answers? Or only user answers? Does it include the agent’s own intermediate reasoning? This matters because embedding distances are sensitive to length and content distribution. Moreover, as the authors themselves hint in Section 5.2 and Appendix A.8, a larger cosine distance can be driven by acquiring *irrelevant* details or stylistic differences, not just essential missing information. Although **Figure 7** is used to justify that recovery-to-full-issue distance is flawed, similar concerns apply to the chosen metric as well. Without a more precise operationalization (e.g., normalizing for text length, or filtering to only response spans containing new entities or code references), it is hard to interpret differences of ~0.02–0.03 between models in **Figure 5** as meaningfully reflecting “information gain”.

6. **Limited theoretical grounding for the statistical analysis**  
   The only statistical test discussed is the Wilcoxon Signed-Rank test (Appendix A.3.1), applied pairwise between settings for each model, with p‑values summarized in **Table 4**. While this is reasonable for checking that Interaction > Hidden and Full > Interaction, the analysis treats each issue–model pair as independent and does not account for multiple comparisons across six models and two comparisons each. At a significance level of 0.05, some of the smaller p‑values would remain significant after correction, but this is never discussed. More importantly, the paper does not report effect sizes or confidence intervals for resolve-rate differences (e.g., in **Figure 3**), which would be more informative than binary significance flags. This is not a fatal flaw, but it reduces the rigor of the claims about “significant” differences.

7. **Some important ablations and controls are missing**  
   Several design decisions are plausible but only weakly justified empirically:
   - The user proxy is limited to three interaction turns, while agents can take up to 30 or 100 steps; it is not clear how performance changes as the number of allowed questions increases or decreases.  
   - RQ2’s detection experiment only varies *prompt framing*; no explicit detection head or self-reflection mechanism is considered, so the conclusion “prompting is insufficient” is somewhat expected.  
   - The authors decide not to evaluate on naturally underspecified issues at all (Section 2.1, Appendix A.4), even though they have at least some annotations; reporting at least qualitative or partial quantitative trends there would strengthen the claim that Ambig-SWE behaviors transfer to real-world ambiguity.  
   Overall, the absence of these ablations limits how confidently one can generalize the main findings or attribute them to specific components.

8. **Some aspects of exposition could be tighter**  
   While the paper is generally readable, a few parts are confusing or redundant: e.g., Section 3.2 mixes discussion of relative performance, action-step efficiency, and data leakage in the same paragraph, making it hard to isolate each phenomenon. In Section 4.3, the narrative jumps between Qwen’s non-interactivity, Deepseek’s behavior, and Claude’s instruction-following without always tying back to **Table 2**. Equation (1) in Appendix A.6 is straightforward cosine distance but is written with oddly spaced typography (“C o s i n e D i s t a n c e”) and does not define how $E_{\text{before}}$ and $E_{\text{after}}$ are constructed from conversational turns.

Overall, these are not fatal issues, but they collectively suggest the need for more precise definitions and a few additional empirical controls.

## Potentially Missing Related Work

1. **Atkinson, C. F. (2025). “Human in the Loop Chain of Code Prompting for Deterministic Tool Development with Generative AI.”**  
   This work focuses on human‑in‑the‑loop, chain‑of‑code prompting for iterative tool development with generative models, which is closely aligned with interactive code generation under partial specifications. It should be discussed in Section 6 (Related Work) alongside Lahiri et al. (2023) and Fakhoury et al. (2024), and possibly cited when motivating the importance of minimal but targeted human intervention in complex code workflows.

## Questions

1. **On synthetic underspecification**: Can you provide more quantitative evidence that GPT‑4o-generated summaries capture the *same types* of missing information as naturally underspecified SWE‑Bench issues, beyond the overlap/recall scores in **Table 3**? For instance, did you manually categorize missing items (file paths, stack traces, configuration details) and compare distributions? Additional analysis here could significantly increase my confidence in external validity.

2. **On step budget confounding**: How much of the performance gap between Claude Sonnet 4 / Qwen 3Coder and the other models in **Figure 3** remains if you cap *all* models at the same number of steps (say, 30), or conversely, allow 100 steps to all? Even a smaller pilot on a subset of issues would help disentangle capability from computational budget.

3. **On the definition of $E_{\text{after}}$ in Equation (1)**: What exact text sequence is embedded for $E_{\text{after}}$? Does it include only user answers, or also the agent’s questions or internal thoughts? If you were to normalize for total token length or drop obviously irrelevant sentences (e.g., “Thanks, that’s helpful”), how robust are the relative rankings in **Figure 5**?

4. **On detection experiment labeling**: For RQ2, at what point in the trajectory is an issue classified as having “interacted”? Is a single question at any time sufficient, and is there a cutoff in terms of turn count? Clarifying this would make the interpretation of FPR/FNR in **Table 2** more precise.

5. **On user proxy realism**: Did you observe any substantial mismatch between how GPT‑4o answered questions as a proxy user and how real GitHub reporters respond in practice (e.g., verbosity, additional unsolicited hints)? If so, could you speculate on which findings might be optimistic compared to a real-user setting?

Clarifications on these points, particularly 1 and 2, could materially affect my assessment of the strength and generality of the conclusions.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The experimental setup is generally sound and carefully implemented within OpenHands, with reasonable baselines and clear comparisons across models and settings. Some methodological details (synthetic data design, step budgets, information-gain metric) could be better controlled or specified, but there are no obvious fatal flaws.

## Presentation Rating

3: good.  
The paper is mostly clear, with well-designed figures (especially **Figures 2, 3, 4, 5, 7**) and detailed appendices. A few sections could be tightened and some definitions (e.g., for the embedding-based metrics) clarified, but overall the exposition is above average.

## Contribution Rating

3: good.  
Ambig‑SWE is a useful benchmark extension and the decomposition of capabilities (detection, questioning, leveraging interaction) provides valuable diagnostic structure. While the methodology is not deeply theoretical, the empirical insights on interactive behavior under underspecificity are meaningful and likely to influence how future code agents are evaluated and trained.

## Overall Rating

8: Accept, good paper (poster).  
The work offers a well-executed, empirically rich study of a timely and practically important problem, with a clear benchmark contribution and nuanced analysis. Despite some limitations in synthetic data construction, proxy design, and ablations, the strengths substantially outweigh the weaknesses, and the paper is well suited for presentation to the ICLR community.

## Reviewer Confidence

4: confident.  
I am familiar with SWE‑Bench, interactive LLM agents, and ambiguity/underspecification work, and I have carefully checked the experimental methodology and metrics, though I did not re-run any code.