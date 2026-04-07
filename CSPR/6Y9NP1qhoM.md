## Summary

The paper studies misinformation injection in LLM-based multi-agent systems (MAS). It introduces MisinfoTask, a dataset of 108 complex, task-driven scenarios with carefully crafted misinformation goals and arguments, and proposes ARGUS, a training-free two-stage defense framework that first adaptively localizes high-risk communication edges in the MAS graph, then uses a corrective agent with goal-aware CoT reasoning to identify and rectify misinformation in multi-round interactions. Experiments across several LLM backbones, attack vectors (prompt injection, RAG poisoning, tool injection), and MAS topologies show that ARGUS substantially reduces a proposed “misinformation toxicity” metric and partially recovers task success rates under attack, outperforming Self-Check and G-Safeguard baselines.

## Strengths

1. **Clear problem formulation and focus on misinformation (not just generic jailbreaking).**  
   The paper makes a useful conceptual distinction between overtly malicious content and semantically benign but factually wrong misinformation in MAS (Section 1, Figure 1). The threat model around prompt, RAG, and tool injection targeting latent beliefs of agents is well articulated and timely, given the growing interest in LLM-based agents.

2. **Dataset contribution tailored to MAS misinformation.**  
   MisinfoTask is not just another QA benchmark: tasks are multi-step, tool-using, and decomposable into subtasks (Section 3.1, Appendix C), which is much closer to realistic MAS workflows. The dataset explicitly contains (i) a task description and reference workflow, (ii) a narrow misinformation goal, (iii) several persuasive but false arguments, and (iv) corresponding ground truths. Figure 9 and the JSON in Appendix C illustrate that instances integrate tools and realistic narratives, not just trivia, and Table 4 gives a clear breakdown of categories (conceptual reasoning, factual verification, procedural application, code/formal interpretation, logic analysis).

3. **Graph-based adaptive localization is technically interesting.**  
   Modeling the MAS as a directed graph and using edge betweenness centrality for initial deployment of the corrective agent (Equations (2)–(4)) is a reasonable and interpretable way to prioritize communication channels. The adaptive re-localization (Equations (5)–(9)) that combines semantic relevance to inferred malicious goals, channel frequency, and topological centrality is an appealing design: it uses feedback from the rectifier’s inferred goal set to focus later rounds on edges that semantically align with the attack’s intent.

4. **Goal-aware rectification leverages LLM reasoning in a structured way.**  
   Section 4.2 and Appendix B.4 describe a multi-step CoT-based procedure that first identifies suspicious sentences, then activates “internal knowledge resonance,” and finally performs root-cause analytical and persuasive reconstruction. Algorithm 1 operationalizes this flow. Compared to generic “self-check” prompts, this is a more structured and MAS-aware rectification mechanism, and the goal inference links nicely back into localization.

5. **Comprehensive empirical coverage across models, topologies, and attacks.**  
   The experimental section is relatively rich:
   - Table 1 reports MT and TSR across four LLMs (GPT‑4o‑mini, GPT‑4o, DeepSeek‑V3, Gemini‑2.0‑flash) and three attack channels, and compares against Self-Check and G-Safeguard. ARGUS consistently yields the lowest MT and highest TSR, often by a notable margin (e.g., GPT‑4o‑mini, Tool Injection: MT drops from 5.78 to 2.67, TSR jumps from 68.75% to 89.66%).  
   - Figure 2 clearly visualizes how attacks shift MAS outputs from the “vanilla” cluster (low MT, high TSR) to a region of higher toxicity and lower success.  
   - Figure 5 shows per-round MT under attacks with and without ARGUS; the contrasting upward vs downward trends nicely support the claim that ARGUS mitigates propagation over time.  
   - Figure 6 explores five topologies (self-determined, chain, full, circle, star) and shows that while all are vulnerable, ARGUS consistently lowers MT across them, suggesting some topology robustness.  
   - Table 2 and Table 3 ablations show that dynamic localization, CoT-based revision, and multi-turn correction all contribute materially, and that information relevance (weight γ) is the dominant but not sufficient component in the scoring.

6. **Dataset- and framework-level ablations and analyses.**  
   Beyond the main table, the authors provide:
   - Effects of agent count (Table 5), showing that ARGUS’s relative gain is larger when the MAS has fewer agents, and that more agents both help and hurt in different ways (higher TSR but more spread of misinformation).  
   - Hybrid attacks (Table 6), where ARGUS still improves MT and TSR under combined PI+RP, PI+TI, RP+TI.  
   - Per-task-category vulnerability (Table 7), which is insightful: conceptual reasoning and code/formal interpretation tasks are more fragile than factual verification tasks.  
   - Cost analysis (Table 8), showing that ARGUS increases API cost but stays in a reasonably bounded regime.

7. **Clarity of overall pipeline and figures.**  
   Figure 3 provides a high-level but coherent visualization: MisinfoTask on the left, three baseline attacks on the right, and the ARGUS pipeline in the center, showing (i) the graph-topology scoring (Score_topo, Score_freq, Score_rel) and (ii) sentence-level CoT correction with inferred misinformation goals feeding back into localization. Figure 1 similarly helps non-specialists quickly understand why “misinformation” is different from conventional harmful text and where ARGUS intervenes in the MAS.

## Weaknesses

1. **Reliance on LLM-as-judge with limited validation of MT/TSR metrics.**  
   The core evaluation metrics MT and TSR (Equation (1), Page 4) are entirely defined via a single LLM judge (GPT‑4o‑2024‑08‑06) scoring semantic similarity between outputs and goals. There is no human evaluation or inter-judge comparison, and no analysis of judge bias or consistency. More concerning, MT is defined using the semantic similarity between the conclusion’s output and the attacker-defined \(g_{mis}^k\), and TSR uses the same scoring function with respect to \(g_{task}^k\). Since the judge LLM and the defending LLMs are from the same family for two of the backbones (GPT‑4o‑mini, GPT‑4o), this creates potential coupling between defense behavior and evaluation. If the judge itself is vulnerable in similar ways, MT could be mis-estimated. At minimum, the paper should empirically check (e.g., with a second judge model or a small human-annotated subset) that MT correlates with actual human-perceived assimilation of misinformation, and that TSR is not overly sensitive to prompt phrasing.

2. **MisinfoTask scale and construction raise generalization questions.**  
   MisinfoTask has only 108 tasks (Table 4), and all were originally generated by GPT‑4o then manually curated (Appendix C). While the manual filtering is a plus, the relatively small size and reliance on a single base model raise concerns about overfitting of insights: (a) It is unclear whether the distribution of attack patterns and arguments reflects realistic, adversarially-chosen misinformation, or rather GPT‑4o’s own biases about “convincing” fallacies; (b) 108 tasks seem small for drawing strong conclusions on robustness across varied domains. The paper partially mitigates this by reporting per-category analyses (Table 7) and by varying topologies and attacks, but it still feels dataset-poor compared to other recent safety benchmarks. Some statistics on argument lengths, diversity of topics, or inter-annotator agreement on what counts as “misinformation” would strengthen the dataset claim.

3. **Mathematical formulation is partly underspecified or inconsistent.**  
   Several key equations are only loosely specified:
   - The combined score \(\texttt{Score}^r(e)\) used in Equation (9) is described qualitatively as a “weighted sum” of \(\texttt{Score}_{topo}(e)\), \(\texttt{Score}_{rel}(e)\), and \(\texttt{Score}_{freq}(e)\), but the exact formula and normalization are missing in the main text. The weights \(\alpha,\beta,\gamma\) are only given in Appendix B.5, and it is unclear whether individual components are normalized to comparable ranges (e.g., \([0,1]\)) before combination. This matters for interpreting Table 3: without explicit normalization, the ablation on “w/o α/β/γ” is hard to reason about analytically.  
   - Equation (6) is syntactically awkward:  
     \[
     \texttt{Rel}(m,V'_{goal}) = \max_{s\in m}\{\{0\}\cup \mathcal{S}(s,V'_{goal})\}\quad\text{s.t.}\quad \mathcal{S}(s,V'_{goal})\ge \theta_{sim}.
     \]  
     As written, the constraint applies to all sentences \(s\), but the max is over all \(s\). It would be clearer to define \(\texttt{Rel}(m,V'_{goal}) = \max_{s\in m: \mathcal{S}(s,V'_{goal})\ge \theta_{sim}} \mathcal{S}(s,V'_{goal})\) with 0 as a default if no \(s\) clears the threshold.  
   - Algorithm 1 updates \(\mathcal{E}_{r+1}\) inside the inner loop over edges \(e\in\mathcal{E}_r\), using \(\text{Score}^r(e)\) before it is fully computed from all messages; this suggests either a pseudo-code bug or missing explanation of when \(\text{Score}^r\) is computed. These mathematical and algorithmic ambiguities do not invalidate the empirical results, but they lower the methodological clarity.

4. **Defense design and baselines are somewhat narrow, limiting claims of generality.**  
   The paper compares ARGUS against Self-Check and G-Safeguard, both reasonable baselines given current MAS safety work. However:
   - There is no comparison to other debate- or committee-based defenses, despite citing multi-agent debate work (e.g., Chern et al., 2024). A naive “multi-debater veto” baseline using multiple agents to cross-check facts, or a two-stage retrieval-then-verification pipeline, could be surprisingly strong on misinformation tasks.  
   - For single-agent defenses, methods like RARR (Gao et al., 2023) and other revision-based approaches could be adapted as a generic rectifier on all messages, providing a closer “rectification-style” baseline than Self-Check (which just asks agents to self-critique) and G-Safeguard (which prunes graph edges).  
   Since the core claim is that goal-aware rectification plus adaptive localization are especially effective against misinformation, missing such baselines blurs how much gain comes from MAS graph reasoning versus simply having a reasonably strong fact-checking agent in the loop.

5. **Limited insight into failure modes and qualitative behavior of ARGUS.**  
   Beyond one case study (Figure 7) and aggregate metrics, the paper gives limited analysis of when ARGUS fails. For example:
   - Table 1 shows that for GPT‑4o-mini under RAG poisoning, G‑Safeguard sometimes *increases* MT (5.19 vs 4.95) while ARGUS strongly decreases it (3.91). It would be valuable to see concrete examples of interactions where graph pruning harms or helps, and where ARGUS’s CoT rewriting successfully steers conversation.  
   - Figure 4 reports the accuracy of \(a_{cor}\) in inferring misleading goals, but the axis and methodology are only minimally described. How is “accuracy” computed here, given that there may be multiple plausible goal descriptions? What is the error when the inferred goal is close but not exact, and how sensitive is localization to such errors?  
   - For hybrid attacks (Table 6), ARGUS still leaves MT relatively high (e.g., 4.13 under PI+RP). Understanding these residual failures would be important for practical deployment.

6. **Evaluation tightly coupled to a specific MAS platform and prompting setup.**  
   The MAS platform (Appendix B) uses ReAct-style agents with a specific planning–worker–conclusion pattern, and ARGUS is tightly integrated via a dedicated corrective agent and specific prompts (Figures 10–14). While this is a reasonable testbed, it is unclear how results would transfer to other MAS frameworks (e.g., AutoGen-style hubs, workflow-based orchestrators, or task-specific agents with minimal communication). In particular, \(k = N-1\) monitored edges (Appendix B.5) may be natural given their small graphs, but in larger MAS this may be unrealistic, and it is not clear how quickly performance degrades as \(k\) shrinks.

7. **Cost–benefit trade-off is underexplored.**  
   Table 8 shows that ARGUS raises cost per 10 instances from about \$0.43 (attack) to \$0.54, ~25% overhead. But the table also indicates that “w/o Intent Inference” has only slightly higher cost than the attack (0.45) while “w/o Edge Scoring” still costs 0.52. There is no corresponding ablation on performance vs cost for these variants, so it is hard to assess whether the full ARGUS configuration is justified in practice. Moreover, the paper does not report wall-clock latency or token counts, which would be highly relevant for multi-round, multi-agent deployments.

8. **Positioning relative to broader misinformation-in-MAS literature is incomplete.**  
   The related work section covers several MAS attack/defense works, but omits some close efforts that also address misinformation detection/correction and lifecycle management in multi-agent settings (see next section). This makes it slightly harder to place the contribution within the evolving ecosystem of misinformation-aware MAS frameworks and may overstate the novelty of the “goal-aware correction” idea.

## Potentially Missing Related Work

1. **Gautam, A., “Multi-agent Systems for Misinformation Lifecycle: Detection, Correction And Source Identification”, 2025.**  
   This work reportedly introduces a multi-agent system explicitly designed to handle the full misinformation lifecycle (detection, correction, source ID). It seems directly relevant to the paper’s focus on misinformation rectification within MAS. The authors should discuss how ARGUS differs from and complements that framework, particularly in terms of (i) goal-aware inference, (ii) graph-based localization, and (iii) treatment of source identification vs. channel-level localization. A natural place to add this would be Section 6 (MAS Information Injection / MAS Defense Strategies), and it could also inform the design of baselines or discussion of limitations.

2. **Sun, L., Wu, T., Zhang, Y., “A Defense Strategy for False Data Injection Attacks in Multi-Agent Systems”, 2023.**  
   This paper proposes strategies to defend multi-agent systems against false data injection, which conceptually overlaps with the misinformation injection setting studied here. While their focus may be on control-theoretic or cyber-physical MAS, the defense ideas (e.g., robust consensus, anomaly detection on communication channels) bear conceptual similarity to ARGUS’s edge-level scoring and localization. It would be helpful to cite and briefly compare against this work in Section 6, clarifying how ARGUS leverages LLM semantics and intent inference, whereas prior approaches may rely on numeric state estimation.

## Questions

1. **Metric robustness and human validation.**  
   Have you conducted any human evaluation or multi-judge experiments to validate that MT and TSR, as computed by GPT‑4o‑2024‑08‑06, correlate with human judgments of “accepted misinformation” and “task success”? If not, could you annotate at least a subset (e.g., 30–50 instances) to report correlation coefficients or agreement rates between human labels and LLM-based scores?

2. **Normalization and weighting in \(\texttt{Score}^r(e)\).**  
   Could you provide the exact formula of \(\texttt{Score}^r(e)\), including how \(\texttt{Score}_{topo}\), \(\texttt{Score}_{rel}\), and \(\texttt{Score}_{freq}\) are normalized before weighting by \(\alpha,\beta,\gamma\)? It would be particularly helpful to know their ranges and whether any clipping is used, to interpret Table 3’s ablations more precisely.

3. **Goal inference accuracy in Figure 4.**  
   How is “accuracy” defined for the corrective agent’s goal inference in Figure 4? Is it exact match against a canonical text description of \(g_{mis}\), or is an LLM judge used to score similarity above a threshold? Also, how does localization performance degrade when the inferred goal is only approximately correct?

4. **Sensitivity to the number of monitored edges \(k\).**  
   You mention in Appendix B.5 that \(k = N-1\) balances coverage and cost. Could you provide empirical results for smaller \(k\) (e.g., \(k=1,2\)) on a fixed MAS topology to illustrate the trade-off between MT/TSR and runtime cost? This would help assess scalability to larger systems.

5. **Comparison to alternative rectifier designs.**  
   Have you tried simpler rectification baselines where a single strong “fact-checker” agent examines all messages (without adaptive localization) or uses a different prompting style (e.g., RARR-like revision)? It would be informative to see whether the gain from ARGUS primarily stems from better localization, better rectification prompts, or both.

6. **Hybrid attacks and worst-case behavior.**  
   In Table 6, ARGUS still leaves MT relatively high (e.g., 4.13 for PI+RP vs 5.81 attack-only). Can you provide qualitative examples of these hybrid-attack cases where ARGUS partially fails, and analyze whether the failure is due to localization (missing some contaminated edges), rectification (not detecting misinformation), or both?

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A. The work deals with misinformation but uses synthetic tasks and focuses on defensive mechanisms. No clear additional ethics concerns beyond the usual cautions already acknowledged in the Ethics Statement.

## Soundness Rating

3: good.  
The methodology is generally sensible and empirically supported, but some mathematical definitions are underspecified and the evaluation hinges on a single LLM judge without human validation.

## Presentation Rating

3: good.  
The paper is mostly clear and well organized, with informative figures (especially Figures 1–3, 5–6) and tables (Tables 1–3, 6–8), though some equations and algorithmic details would benefit from more precise specification.

## Contribution Rating

3: good.  
Combined, MisinfoTask and ARGUS offer a meaningful contribution to the emerging area of MAS safety against misinformation. The dataset scale and baseline selection could be stronger, but the work is likely to be of interest to the ICLR community.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  
The paper tackles an important and underexplored problem, provides a dedicated dataset and a reasonably well-designed defense framework, and demonstrates consistent empirical gains. However, concerns about metric validation, dataset scale, somewhat underspecified math, and limited baselines prevent a more enthusiastic recommendation.

## Reviewer Confidence

4: confident.  
I am familiar with LLM-based agents and safety defenses, carefully examined the equations and experiments, and feel reasonably confident in this assessment, though additional information in the rebuttal (especially about metric validation and score normalization) could adjust my opinion.