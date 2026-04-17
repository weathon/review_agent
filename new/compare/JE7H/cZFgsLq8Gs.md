---
job_id: ce776f61-8806-473b-a52d-79914432cb88
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: cZFgsLq8Gs.pdf
paper: DeepScientist: Advancing Frontier-Pushing Scientific Findings Progressively
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper concerns autonomous scientific discovery using LLM agents, Bayesian optimization style search, and applications to AI tasks (agent failure attribution, LLM inference acceleration, AI text detection), which fits squarely within machine learning, representation learning, and infrastructure for ML research.

## Minimum Quality
Pass ✅.  
All core sections are present (Abstract, Introduction, Related Work, Method, Experiments/Results, Discussion, Conclusion). The work is substantial, technically nontrivial, and written in clear English. While there are weaknesses in rigor and evaluation, they are not at the level of fundamental invalidity.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden text, prompt injection, or instructions targeting automated reviewers in the paper content.

---

# Expected Review Outcome:

## Summary

The paper introduces **DeepScientist**, an LLM-based multi-agent system for autonomous scientific discovery that operates in a closed loop over long time horizons. The system formalizes discovery as a goal-driven Bayesian optimization problem over a large space of possible research "programs", using a persistent **Findings Memory** and an LLM surrogate model plus a UCB-style acquisition rule to select which hypotheses to implement and validate.  

DeepScientist is evaluated on three frontier AI tasks with strong human SOTA baselines (agent failure attribution, LLM inference acceleration, AI text detection), where it autonomously proposes, implements and evaluates new methods (A2P, ACRA, T-Detect/TDT/PA-TDT). These methods are reported to surpass human-designed SOTA by substantial margins, and the system’s exploratory dynamics and scaling behavior are analyzed using logs, statistics, and t-SNE visualizations.

## Strengths

1. **Clear and ambitious problem formulation.**  
   The paper explicitly frames automated scientific discovery as **goal-directed Bayesian optimization** over a space of research programs \( \mathcal{I} \), with an extremely expensive objective \(f(I)\). Section 3 and **Figure 2** make the three-stage loop (Strategize & Hypothesize → Implement & Verify → Analyze & Report) and interaction with the Findings Memory quite clear. This moves beyond earlier "one-shot paper generation" systems toward a more realistic, iterative view of research.

2. **Thoughtful system design with persistent Findings Memory.**  
   The notion of a structured **Findings Memory** that accumulates *both* successful and failed attempts, and is queried via retrieval to contextualize the surrogate LLM, is an important design choice. The implementation details in Appendix D (e.g., Research_Outline.md, structured JSON ideas, promotion from Idea to Implement to Progress Findings) show a serious attempt to build a reusable knowledge base, not just a flat list of trials. **Figure 4(a)**, which shows the funnel from ~5k generated ideas to 1.1k implemented ideas to 21 progress findings and finally 5 papers, empirically supports that this memory plus selection mechanism is central rather than decorative.

3. **Compelling empirical case studies on strong baselines.**  
   The three main case studies are reasonably challenging, state that they start from strong, recent SOTA methods (Table 1): All-at-Once for failure attribution, Token Recycling for inference acceleration, and FastDetectGPT/Binoculars for AI text detection. The methods discovered by DeepScientist (A2P, ACRA, T-Detect/TDT/PA-TDT) are themselves nontrivial and are documented with substantial technical detail later in the paper (Sections starting on Pages 25, 36, 48).  

   - **Table on Page 6 / Figure 3** shows large gains in failure attribution on Who&When: from 12.07/16.67 accuracy to 29.31/47.46 in hand-crafted and algorithm-generated settings (reported as +142.8% / +183.7%).  
   - LLM inference acceleration improves throughput on MBPP from 190.25 to 193.90 tokens/sec (small but in a heavily optimized regime).  
   - AI text detection AUROC on RAID improves from 0.800 (Binoculars) to 0.863 (PA‑TDT), with latency cut from 117 ms to 60 ms, as illustrated in **Figure 3(d)**.  

   These are concrete, task-appropriate metrics, and **Figure 1** juxtaposes human progress on RAID over several years with DeepScientist’s two-week trajectory (T‑Detect → TDT → PA‑TDT), making the story easy to grasp.

4. **Strong, interpretable downstream methods (A2P, T-Detect, TDT).**  
   The individual methods surfaced by DeepScientist stand on their own as reasonable research contributions:

   - **A2P** (Pages 25–33) presents a structured Abduct-Act-Predict framework for failure attribution, with clear causal grounding in Pearl’s SCM formalism. **Equation (1)** in that section formalizes the abduction step, and **Table 1** (Page 29) shows sizable gains over multiple baselines on both agent-level and step-level accuracy. **Tables 2 and 3** ablate causal components and step numbering in a mathematically explicit way.  
   - **T-Detect** (Pages 36–47) introduces a Student-\(t\)-based normalization over curvature scores. **Equation (3)** gives a precise formula \(\mathcal{D}_{t\text{-dist}}(x;\nu) = \frac{d(x)}{\sqrt{\frac{\nu}{\nu-2} V(x)}}\), and **Table 1** and **Table 2** show consistent AUROC and TPR@5%FPR boosts on RAID and HART.  
   - **TDT** (Pages 48–57) is a fairly sophisticated signal-processing formulation that applies a continuous wavelet transform over per-token discrepancy scores. **Equations (1)–(3)** in Section 3 define the CWT \(W(a,b)\) and the band energies \( \|W_{\text{band}}\|_F \), and **Figure 1** (TDT) nicely visualizes the path from 1D signal to 2D scalogram to 3D features. **Table 1** and **Table 3** show strong gains, especially on adversarial paraphrasing (HART Level 2).  

   These methods are not just hyperparameter tweaks; they have defensible mathematical structure and address specific failure modes of baselines.

5. **Nontrivial analysis of exploration dynamics and scaling.**  
   Section 4.3 presents substantial post-hoc analysis that goes beyond raw performance:

   - **Figure 4(b)** compares success rates with versus without the selection mechanism, showing that naïve random implementation of 100 ideas per task yields essentially zero successes, validating the importance of the surrogate and acquisition function.  
   - **Figure 4(c)** shows a violin plot of execution times for implemented trials across tasks, grounding the cost claims (~\(10^{16}\) FLOPs per LLM experiment).  
   - **Figure 5** provides a t-SNE visualization of 2,472 AI text detection ideas, with the trajectory Initial SOTA → T‑Detect → TDT → PA‑TDT overlaid. This gives compelling qualitative evidence that the system’s search is not random but follows a coherent trajectory through concept space.  
   - **Figure 6** shows a clear near-linear relationship between number of GPUs and number of "Progress Findings" in a one-week window, supporting the scaling-law claim.

6. **Evaluation of AI-generated papers using both automated and human review.**  
   The two-tier evaluation in Section 4.2 is thoughtful. **Table 2** compares DeepScientist’s 5 papers to dozens from other AI scientist systems under DeepReviewer-14B, showing higher scores and a simulated 60% accept rate. **Table 3** reports human committee reviews of the five DeepScientist papers, with mean rating 5.0 vs 5.08 for real ICLR 2025 submissions and Krippendorff’s \(\alpha = 0.739\) on rating, suggesting real human agreement. The detailed description of the review protocol in Appendix B is credible.

7. **Right level of self-critical reflection.**  
   The authors are unusually candid about limitations: 60% of failed trials are due to implementation errors, most remaining failures are non-improvements, and overall "success" is 1–5%. They explicitly call out that DeepScientist is more of an exploration engine than a substitute for human oversight, and Appendix C articulates sensible future directions (improving hypothesis quality, filtering, and implementation quality). This gives the work a grounded, non-hyped tone.

## Weaknesses

1. **Bayesian optimization formulation is informal and somewhat misleading.**  
   The core framing is "goal-driven Bayesian optimization", but the actual mathematical treatment in Section 3 is fairly loose:

   - The surrogate model \(g_t\) is an LLM that directly outputs three heuristic scores \(V=\langle v_u,v_q,v_e\rangle\) based on retrieved Findings Memory. There is no explicit probabilistic model, no notion of posterior mean or uncertainty, and no data-driven fit beyond prompts.  
   - **Equation (1)** defines the acquisition as \( \arg\max_I w_u v_u + w_q v_q + \kappa v_e\), yet labels both terms as "Exploitation Term" and uses \(v_e\) as a generic "exploration value", not as a variance estimate \(\sigma(I)\). This is called UCB, but it diverges substantially from classical UCB or GP-UCB in BO, where the second term is tied to epistemic uncertainty derived from a probabilistic model.  
   - There is no analysis of how well \(g_t\) correlates with true \(f(I)\), how the scores are calibrated across time, or whether the "exploration" component actually encourages coverage of under-explored regions.

   This gap matters because a key claim of the paper is that search efficiency, not brute force, explains the reported breakthroughs. Without a clearer probabilistic grounding or empirical calibration of \(g_t\) and \(\alpha\), it is hard to attribute success specifically to a principled BO strategy rather than to generic heuristic ranking plus large compute.

2. **Lack of strong baselines for the *system-level* search strategy.**  
   While the downstream methods (A2P, TDT, etc.) are compared against task-specific baselines, there is little rigorous comparison at the system level:

   - The only direct search baseline is the "no selection" random-implementation ablation in **Figure 4(b)**, which is quite weak. It would be more convincing to compare to, for instance, (i) random but *memory aware* sampling, (ii) a simple novelty-based selection, or (iii) existing AI Scientist pipelines (AI Scientist, AI Scientist-v2, CycleResearcher, etc.) run on the same three tasks under similar compute budgets.  
   - Section 4.2 compares AI-generated **papers** across systems using automated reviewers, but not their search strategies on the *same scientific problems*. We do not see whether, given 20k GPU-hours and the same baselines, AI Scientist-v2 or AI-Researcher would discover improvements of comparable magnitude.

   Since the central contribution is an *architecture and search method* for automated discovery, this weakens the causal link between design choices (Findings Memory, surrogate, UCB) and the observed SOTA improvements.

3. **Extent of human supervision and "autonomy" is under-quantified.**  
   The abstract and introduction emphasize "fully autonomous scientific discovery", but multiple parts of the paper reveal substantial human involvement:

   - Section 4 notes that "Three human experts supervise the process to verify outputs and filter out hallucinations." Appendix F describes manual verification of all experimental results and a secondary re-run of experiments to catch Claude agent failures.  
   - There is no quantitative measure of how many trials were discarded or corrected due to human intervention, how often humans altered hypotheses or code, or whether humans ever vetoed promising-but-risky ideas.  

   Autonomy vs supervised tooling is a key conceptual axis in this line of work. Without more detailed statistics on human edits, the claims about end-to-end autonomy are overstated. At minimum, the system should be characterized explicitly as "autonomous under human approval and verification loops".

4. **Generalization beyond LLM-heavy AI tasks remains unproven.**  
   All three main tasks are tightly in the LLM / NLP domain, with readily available code, fast feedback, and strong existing open research ecosystems. While this is already challenging, the paper also makes broader claims about "scientific discovery" and references robotics and physical sciences in discussion:

   - Section 4.3 rightly notes that high-cost domains (e.g., foundation model pretraining, pharmaceutical synthesis) are currently impractical for this style of trial-and-error.  
   - However, even within ML, there is no task that involves non-trivial data acquisition, robotics experiments, or cross-domain generalization. The additional Micronano-DeepScientist results on AlgoTune in Appendix E are interesting but use a significantly simplified variant and a different backbone (GLM-4.6), so they do not strongly validate the main system’s architecture.

   As it stands, the evidence supports "DeepScientist can advance SOTA on several LLM-centric AI benchmarks with large compute and human verification" more than the broader rhetoric about "modern scientific frontiers" in general.

5. **Heavy reliance on automated reviewers for paper-quality claims, with circularity concerns.**  
   Section 4.2 and Appendix E.2 extensively use LLM-based reviewer systems (DeepReviewer, o3-mini reviewer, AI Scientist-style prompts with Gemini/GPT-4o/GPT-5, CycleReviewer) to argue that DeepScientist’s papers are of high scientific quality and often top-ranked among AI scientist systems. While this is a useful signal, it raises several issues:

   - These reviewer systems are themselves LLM-based and may be biased towards the stylistic patterns of LLM-written papers, or towards the same underlying models used in DeepScientist.  
   - The human evaluation in **Table 3** is more meaningful but limited to three reviewers and five papers; reviewers also explicitly note weaknesses in experimental rigor and related work coverage.  
   - Using LLM reviewers as the primary basis for cross-system comparison risks circular reasoning: systems that better match the reviewer’s learned style and preferences will be rewarded, regardless of true scientific merit.

   Overall, the human evaluation should carry more weight in the narrative; the LLM-review scores should be presented more cautiously as auxiliary evidence.

6. **Methodological details are sometimes underspecified or inconsistent.**  
   While many implementation details are in the appendices, several important aspects remain unclear in the main text:

   - The surrogate model's scoring rubric: how are \(v_u, v_q, v_e \in [0,100]\) defined operationally? Are prompts fixed across tasks? Is there any attempt to calibrate or debias these scores over time?  
   - Retrieval into Findings Memory: Section 3 states a Top-K retrieval with K chosen so that records "fit in a 2×10^5-token context", but Appendix F later fixes \(K=15\). This is a surprisingly small K given thousands of findings, and there is no ablation on K or retrieval strategy.  
   - In **Figure 6** (scaling), there are no error bars, no mention of repeated runs, and no description of how random seeds or task mix were controlled. A 1-week experiment at each GPU count may be quite noisy.  
   - In the downstream method sections, some notations are sloppy. For example, in **TDT** Equation (3) on Page 52, the vector \(\mathbf{S}_{\mathrm{TDT}}(x)\) is written with \(\|W_{\text{vyn}}\|_F\) rather than \(\|W_{\text{syn}}\|_F\), presumably a typo; the text refers to "syntactic" features.

   These issues do not invalidate the main results but make it harder to fully trust the claimed efficiency and robustness.

7. **Mathematical grounding of some downstream methods is heuristic.**  
   While A2P and TDT are mathematically explicit, some derivations are ad hoc:

   - In **T-Detect** Equation (3), the normalization \(\sqrt{\frac{\nu}{\nu-2} V(x)}\) is justified by invoking the variance of a Student-\(t\) distribution, but there is no probabilistic derivation showing that \(d(x)\) and \(V(x)\) arise from a \(t\)-distributed statistic. The connection is more of a rescaling heuristic than a proper likelihood ratio or test statistic. The ablations in **Table 4** show modest gains (+0.6% AUROC), which are useful but small. A clearer statistical model (e.g., robust regression or explicit heavy-tailed prior) would strengthen the argument.  
   - In **A2P**, the SCM formulation with equations (1)–(3) is conceptually coherent but the actual implementation uses a prompt that asks the LLM to "abduct, act, predict" within a single forward pass. There is no guarantee that the induced posterior \(P(\epsilon|s_{0:t},a_t,Z(\tau)=1)\) or the intervention semantics match Pearl’s formalism. The empirical results in **Table 1–3** are strong, but the mathematical notation somewhat oversells the level of causal grounding.

   These concerns are more about match between notation and implementation than correctness, but they matter because the paper positions these methods as scientifically grounded advances.

8. **Ethical and societal implications could use deeper treatment.**  
   The Ethics Statement recognizes dual-use risks and the possibility of flooding the literature with autogenerated papers. Mitigations (not open-sourcing Analyze&Report, license requirements for human supervision) are reasonable first steps. However, given the scale of automation described, there is limited discussion on:

   - How to prevent subtle plagiarism or rediscovery of proprietary ideas when agents "access the internet for literature and code searches" (Stage II).  
   - How the community should evaluate AI-generated papers in peer review, beyond what is done here.  

   This is more of an omission than a fatal flaw, but given the potentially transformative nature of such systems, a more substantial engagement with existing work on AI scientist ethics would be beneficial.

## Potentially Missing Related Work

1. **Gottweis & Weng & Daryin, "Towards an AI Co-Scientist", 2025.**  
   This work discusses building AI co-scientists and addresses challenges in autonomous hypothesis generation and validation. While the paper cites several "co-scientist" style systems (e.g., Penadés et al., Swanson et al., 2025), this specific work is not referenced. It should be discussed in Section 2 under "Semi-Automated Scientific Assistance" or "Automated Scientific Discovery" to better situate DeepScientist relative to contemporaneous AI co-scientist proposals.

2. **Ghosal et al., "A Survey on the Possibilities & Impossibilities of AI-Generated Text Detection", 2023.**  
   Given that one of the three main tasks is AI text detection and that DeepScientist develops multiple detectors (T‑Detect, TDT, PA‑TDT), this survey is directly relevant. It should be cited in the AI text detection sections (e.g., Section 4.1, the T-Detect introduction on Page 36, and TDT's related work on Page 48) to contextualize where the proposed detectors sit relative to known limitations and impossibility results.

3. **Gonzalez, Dai, Hennig, "Bayesian Optimization for Scientific Discovery", 2021.**  
   This paper directly addresses the use of Bayesian optimization in scientific discovery. Since DeepScientist’s central methodological framing is "scientific discovery as Bayesian optimization" (Section 3), it is important both as prior art and as a reference point for what is gained or lost by using LLM surrogates and heuristic UCB. It should be cited in Section 3 and discussed in terms of how DeepScientist relates to BO approaches in more traditional scientific domains.

4. **Chen et al., "AutoSynth: Automated Hypothesis Generation via Large Language Models", 2023.**  
   AutoSynth focuses specifically on automated hypothesis generation using LLMs. DeepScientist’s Stage I (Strategize & Hypothesize) is closely related in spirit. Adding this to Related Work (Section 2, especially the "Semi-Automated Scientific Assistance" or "Automated Scientific Discovery" parts) and briefly contrasting the scale, memory usage, and evaluation would clarify the novelty of DeepScientist’s multi-cycle, memory-augmented design.

5. **Mangoni & Traversa, "AI Feynman: A Physics-Inspired Method for Symbolic Regression", 2020.**  
   AI Feynman is a prominent example of automated discovery of interpretable equations, closely aligned with the idea of AI systems making scientific findings. It should be discussed in Section 2 when talking about algorithmic or scientific discovery agents and in Appendix C where "derivable models" and symbolic approaches (AI-Descartes, AI-Hilbert) are discussed, as it provides a concrete, prior example of physics-inspired scientific discovery by AI.

6. **King et al., "A Robot Scientist that Intelligently Automates Scientific Discovery", 2009.**  
   This is a foundational work on robot scientists performing autonomous experiments in biology. While the paper cites more recent works on robot scientists and scaling laws (Zhang et al., 2025b), this classic paper is not mentioned. It should be added to Related Work (Section 2) and possibly to the Discussion when talking about extending DeepScientist to physical sciences and robotics, to acknowledge historical continuity.

## Questions

1. **Surrogate model calibration and BO behavior.**  
   Could the authors provide quantitative evidence on how well the surrogate scores \(v_u, v_q, v_e\) correlate with true performance improvements \(f(I)\)? For example, Spearman correlation between \(v_u + v_q\) and realized gains, or plots of success probability vs predicted score. This would greatly clarify whether the BO framing is justified.

2. **Alternative selection baselines.**  
   In addition to the "no selection" ablation in **Figure 4(b)**, have the authors tried simpler, more realistic baselines, such as (i) random selection among only top-K utility scores, (ii) novelty-based selection (e.g., embedding distance from prior ideas), or (iii) a purely exploitation-based rule \(v_u + v_q\) without \(v_e\)? Results for such variants would help isolate the contribution of the exploration term and Findings Memory.

3. **Quantification of human intervention.**  
   Can the authors quantify the amount and type of human supervision in each experiment? For example, number of runs where humans overrode the agent’s implementation, percentage of experiments re-run due to suspected bugs, and any instances where humans adjusted hypotheses. A small table summarizing human touchpoints per task would better align the autonomy claims with reality.

4. **Reproducibility under open models.**  
   The current system relies on Gemini-2.5-Pro and Claude-4-Opus, which are proprietary. Do the authors have preliminary results using only open models (e.g., Qwen-3, GLM-4.6) for the *main* DeepScientist runs, not just Micronano-DeepScientist? Even if performance is lower, it would be helpful to see how sensitive the approach is to the choice of reasoning backbone.

5. **Uncertainty and UCB interpretation in Equation (1).**  
   How do the authors justify calling Equation (1) a UCB rule when \(v_e\) is not an uncertainty estimate derived from a probabilistic model? Have they considered learning a predictive distribution over outcomes (e.g., via regression on historical results) so that a more classical UCB or Thompson sampling could be used?

6. **Statistical significance of SOTA gains.**  
   For the main gains reported in **Figure 3** and the method-specific tables (A2P, T-Detect, TDT), can the authors provide confidence intervals or significance tests based on repeated experiments or bootstrapping? Especially for modest gains like +1.9% tokens/sec and +0.063 AUROC, it would be important to know whether the improvements are robust beyond noise.

7. **Ethics and data access in Stage II.**  
   Since the ImplementationAgent can "access the internet for literature and code searches", what safeguards are in place to avoid copying proprietary code or violating licenses? Some detail on filtering or provenance tracking would be helpful.

## Flag For Ethics Review

No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The system and downstream methods are well engineered, the experiments are extensive and use strong baselines, and the key empirical claims (surpassing human SOTA on three tasks) are reasonably well supported by **Figures 1, 3, 4–6** and the various tables. However, the Bayesian optimization framing is mathematically loose, there is limited calibration of the surrogate model, and system-level baselines are thin. Some method derivations (e.g., T-Detect) are heuristic rather than fully probabilistic.

## Presentation Rating

3: good.  
The paper is long but generally well organized. The main text, **Figure 2** (architecture), **Figure 3** (task performance), **Figure 4–6** (statistics and scaling), and tables (Tables 1–3, and method-specific tables) give a coherent narrative from system design to empirical results to analysis. A few notational typos and underspecified implementation details detract somewhat, and the BO terminology could be more precise, but overall clarity is above average.

## Contribution Rating

4: excellent.  
The work demonstrates, at substantial scale, that an LLM-driven system with a persistent memory and a simple acquisition rule can autonomously discover methods that genuinely improve over strong human SOTA on nontrivial AI tasks. In addition, some of the discovered methods (A2P, T-Detect/TDT/PA‑TDT) are themselves substantive research contributions. The extensive analysis of exploration dynamics, failure modes, and scaling is also valuable to the emerging AI scientist field.

## Overall Rating

8: Accept, good paper (poster).  
Despite several weaknesses in theoretical rigor, baseline comparisons, and autonomy quantification, this is an ambitious and carefully executed piece of work that moves the AI scientist literature forward in a concrete way. The combination of system design, large-scale empirical evidence, and the quality of the discovered downstream methods is strong enough that the paper will be of significant interest to the ICLR community.

## Reviewer Confidence

4: confident.  
I am familiar with LLM-based agent systems, Bayesian optimization, and AI text detection, and I have checked the key equations, tables, and figures in detail. Some implementation aspects necessarily remain opaque due to reliance on external APIs, but overall I am reasonably sure of my assessment.