# ICLR Benchmark Results

Date: 2026-04-10 14:29
Critic/Merger: deepseek/deepseek-v3.2 (OpenRouter)
Neutral: deepseek/deepseek-v3.2, Related Work: deepseek/deepseek-v3.2:online (OpenRouter)

## fwYTXwoiCQ

- GT: Reject (avg 4.0)
- Predicted: N/A (4.0/10)
- Match: N/A

### Final Review

## Summary
This paper investigates whether large language models (LLMs) fully utilize training data for mathematical reasoning. Through experiments on models like Llama3 and Gemma3 with supervised fine-tuning and reinforcement learning on datasets including GSM8K and MATH, the authors find that adding more training data causes a significant portion (10–15%) of previously correct test answers to become incorrect. They attribute this to high predictive multiplicity (Rashomon effect), where models trained on the same data with different random seeds learn divergent functions, each solving only a disjoint subset of the test set.

## Strengths
- **Identifies and robustly demonstrates a counter-intuitive phenomenon:** The paper clearly shows that increasing training data can lead to forgetting of correct answers in math reasoning tasks, challenging standard scaling assumptions. This is evidenced across multiple model families (Llama3, Gemma3, Qwen2.5), training methods (SFT, RL), datasets (MAWPS, GSM8K, MATH), and inference techniques (greedy decoding, majority voting).
- **Comprehensive empirical coverage:** The experiments establish generality within the studied domain, with consistent results across varied settings, including tests without parameter-efficient fine-tuning (Appendix A.2) and with different model capacities (Appendix A.1).
- **Effective theoretical framing:** The paper connects empirical observations to established concepts of predictive multiplicity and Rashomon sets, providing a plausible conceptual explanation beyond mere correlation (Sections 4.1–4.2).

## Weaknesses
### Major:
- **Insufficient statistical support for variability claims:** The paper’s central argument about predictive multiplicity and seed-dependent differences relies on a very small number of random seeds—3 for supervised fine-tuning and only 1 for reinforcement learning experiments. This undermines the reliability of conclusions regarding the diversity of learned functions and the intersection of correct answers (e.g., Figures 5, 6, 7). For findings that hinge on randomness, more seeds are essential to ensure statistical significance.
- **Weak causal linkage between data addition and predictive multiplicity:** While the paper documents both forgetting with added data (Section 3) and high multiplicity in fixed-set training (Section 4.1), the direct mechanistic link between these phenomena is asserted rather than proven. The theory in Section 4.2 explains why large Rashomon sets exist but does not model why adding data should trigger shifts within this set, leaving open alternative explanations like optimization dynamics or catastrophic interference.

### Minor
- **Simplified theoretical analysis:** The combinatorial framework for Rashomon sets (Section 4.2) relies on strong assumptions (e.g., independence of per-sample strategies) and is not empirically validated beyond strategy counts. A more rigorous analysis of the loss landscape or model agreement would strengthen the contribution.
- **Superficial analysis of “strategies”:** The paper defines strategies as sequences of operations and reports counts, but lacks deeper investigation into what makes strategies different (e.g., semantic vs. syntactic variations) or how strategy choice correlates with correctness flips. This limits insight into the root causes of multiplicity.
- **Limited exploration of mitigation and practical implications:** The paper diagnoses a significant problem but offers no concrete solutions or experiments on how to mitigate it (e.g., via ensembling, regularization, or data curation). While the core contribution is diagnostic, addressing mitigation would enhance impact.

### Trivial
- **Narrow model scale range:** Appendix A.1 shows the effect persists up to 12B parameters, but frontier models are much larger. However, within the paper’s scope of studying the phenomenon, the models used are sufficient for initial demonstration.

## Nice-to-Haves
- A deeper qualitative analysis of examples where answers flip from correct to incorrect, including reasoning traces, to provide intuitive understanding.
- Experiments with more advanced test-time scaling techniques (e.g., verifier-based ranking) to further confirm the robustness of the phenomenon.
- Ablation studies on hyperparameters like learning rate or batch size to assess sensitivity.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Criticisms about model/dataset availability:** Any doubt about the existence or release status of cited models (Llama3, Gemma3, Qwen2.5) or datasets (GSM8K, MAWPS, MATH) is removed, as they are assumed available per hard rules.
- **Reproducibility nitpicks:** Requests for more detailed hyperparameters or implementation details beyond those provided in Sections 3.2.1 and 3.2.2 are removed as trivial per hard rules.
- **Unfair comparisons:** No such weaknesses were present.
- **Missing related works:** Suggestions to add more references are omitted per hard rules.
- **Formatting/style comments:** Any minor writing or presentation issues are excluded.
- **Strawman weaknesses:** Claims that the paper does not address multiplicity or linking are removed, as the paper explicitly discusses these in Sections 4.1–4.2.

## Suggestions
- Increase the number of random seeds to at least 5–10 for all experiments to bolster statistical claims about variability and predictive multiplicity.
- Conduct a controlled experiment where a single model is trained sequentially on data subsets to disentangle catastrophic forgetting from Rashomon effect-driven variability.
- Enhance the strategy analysis by clustering reasoning traces beyond operation sequences and examining how strategy distributions shift with added data or across seeds.

---

## GzvdIo1QiP

- GT: Reject (avg 1.0)
- Predicted: N/A (1.0/10)
- Match: N/A

### Final Review

## Summary
This paper presents a legal and policy analysis of AI-generated content (AIGC) watermarking from an Afrocentric perspective. It argues that current technical approaches are insufficient without considering Africa's unique regulatory context. Through case studies of Nigeria, Kenya, Egypt, and South Africa, it analyzes gaps in copyright and data protection laws and proposes a dual-purpose watermarking framework that attributes both generated content and its Indigenous training data, concluding with policy recommendations.

## Strengths
- **Important and Underexplored Geographic Focus:** The paper centers an analysis on African legal systems and the protection of Indigenous data, addressing a significant gap in the global AIGC governance discourse. This regional and ethical focus is timely and socially relevant.
- **Structured Comparative Legal Analysis:** The paper introduces and applies four clear metrics (provision for watermarking, provision for AIGC, institutional oversight, judicial opinion) to systematically evaluate and compare the regulatory landscape across four diverse African jurisdictions, providing a replicable framework.

## Weaknesses
### Major:
- **Fundamental Venue Misalignment:** The paper's core contribution is a descriptive legal survey and policy advocacy. It contains no novel algorithms, theoretical insights, empirical evaluations of models/methods, or technical frameworks. This places it outside the scope of ICLR, a conference focused on machine learning research. The work is better suited for law, policy, or interdisciplinary ethics venues.
- **Unsubstantiated Core Conceptual Claim:** The paper's proposed "dual-purpose" watermarking framework (for content authenticity and Indigenous data attribution) is presented as a conclusion but is not developed or evidenced. The paper provides **no technical pathway, mechanism, or feasibility analysis** for implementing this vision. The legal analysis in Sections 4-5 is descriptive and does not demonstrate how the identified gaps technically inhibit such a system or how to bridge them.
- **Lack of Technical Engagement and Validation:** The technical overview (Section 2) is superficial and non-critical. The "Challenges" section (Section 3) is generic and does not connect technical limitations (e.g., adversarial removal) to the subsequent African contextual analysis. There is no empirical data, case studies of AIGC harm in Africa, or technical experiments validating the claim that existing watermarking methods fail in African contexts (e.g., for low-resource languages or infrastructures).

### Minor:
- **Underdeveloped Narrative and Methodology:** The flow from introduction to analysis is uneven. The paper lacks a clear methodological description for how legal texts were selected and analyzed. The connection between the sparse technical background and the detailed legal sections is weak, creating a disjointed argument.
- **Limited Analysis of Unique Threat Models and Incentives:** While noting regulatory gaps, the paper does not deeply analyze unique adversarial threats (e.g., resource-constrained attacks) or the fundamental misalignment of incentives between Global North AI companies, African regulators, creators, and users. This undermines the practicality of its recommendations.

### Trivial:
- **Grammatical and Clarity Issues:** Some sentences are awkwardly constructed, which occasionally hinders readability.

## Nice-to-Haves
- A clearer diagram visualizing the proposed "Afrocentric" watermarking ecosystem involving creators, companies, regulators, and data flows could help clarify the envisioned architecture.
- A more structured comparative table summarizing regulatory requirements in the studied African countries versus jurisdictions like the EU, US, and China could strengthen the analysis of uniqueness.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Weakness (from Harsh Critic): "The claim that resources are 'one-sided' (favoring companies) is not backed by a systematic review..."** *Justification: The paper cites specific examples (e.g., OpenAI's disclosure practices) to support this claim. Demanding a full systematic review is outside the paper's scope and a methodological practice not standard for this type of analysis.*
- **Weakness/Experiment Request (from Spark Finder): "A core experiment should test whether standard watermarking methods fail... when applied to African cultural/language data..." and "Ablation on the 'double-sided watermarking' concept... A minimal experiment must simulate this concept..."** *Justification: These are demands for the paper to include technical experiments and become a different type of contribution. The paper is explicitly a legal/policy analysis; criticizing it for not being an empirical technical paper is scope creep. The core issue is venue misalignment, not the absence of these specific experiments.*
- **Weakness (from Spark Finder): "The paper must include a benchmark comparing the robustness... of current SOTA watermarking methods when deployed in the four case-study countries versus the Global North..."** *Justification: Same as above. This is a request to fundamentally change the paper's nature from legal analysis to empirical systems benchmarking.*
- **Nitpick on Reproducibility:** Any implied criticism about undisclosed hyperparameters or complete training logs is removed, as these are irrelevant for a non-technical, legal analysis paper.

## Suggestions
- **Consider Submission to a Different Venue:** The authors should seriously consider submitting this work to a venue specializing in AI policy, law, ethics, or African studies (e.g., FAccT, AIES, or relevant law/technology journals) where its contributions would be directly aligned with the venue's scope.
- **Strengthen the Technical-Policy Bridge:** If aiming for an interdisciplinary ML venue, the paper must integrate a substantial technical component. For example, it could propose a novel watermarking schema or metadata standard designed for the legal requirements identified, or provide a technical critique of existing methods based on African infrastructural constraints, supported by minimal proof-of-concept validation.
- **Improve Narrative Flow:** Reorganize the paper to create a stronger through-line: clearly state the problem, provide a more critical survey of technical watermarking limitations, explicitly link those limitations to the African regulatory and contextual analysis, and then derive both technical and policy recommendations from that integrated analysis.

**Overall Evaluation:** The paper addresses an important, overlooked topic with a structured legal comparison. However, it is **not a machine learning research paper**. It lacks the technical novelty, methodological rigor, and empirical evaluation required for ICLR. Its core contributions are in law and policy, not in advancing the field of machine learning. Therefore, it is **not suitable for acceptance at ICLR** in its current form.

---

## PemDVHC2KO

- GT: Reject (avg 2.0)
- Predicted: N/A (4.0/10)
- Match: N/A

### Final Review

## Summary
This paper introduces TiEBe, a benchmark designed to evaluate Large Language Models' factual recall of notable global and regional events across time, geography, and language. It comprises over 23,000 question-answer pairs, automatically generated from Wikipedia retrospective pages and their cited external sources, spanning 10 years, 23 regions, and 13 languages. The authors evaluate nine LLMs, finding significant regional performance disparities, strong correlations between model accuracy and countries' socioeconomic indicators (e.g., GDP, HDI), and notable degradation for low-resource languages.

## Strengths
- **Comprehensive Multi-Dimensional Evaluation:** The benchmark systematically evaluates LLMs along three critical and often isolated axes: temporal (10-year span), geographic (23 regions), and linguistic (13 languages). The inclusion of both English and native-language prompts for non-English regions is a particularly strong design choice that helps disentangle multilingual comprehension from factual recall.
- **Evidence-Based Quantification of Socioeconomic Bias:** The paper presents a robust, data-driven finding: a strong Spearman correlation (~0.73-0.77) between average model performance (on pre-2023 events) and national development indicators like GDP and HDI. This concretely quantifies a known concern about LLM bias and provides a measurable target for improving global equity in AI.
- **Scalable and Updatable Benchmark Construction:** By leveraging Wikipedia's structured retrospective pages—which are naturally updated—the authors create a benchmark that can be continuously refreshed with new events. This addresses a genuine need for evolving evaluation tools in continual learning research.

## Weaknesses

### Major:
- **Unvalidated Synthetic Data Quality Threatens Benchmark Integrity:** All 23,446 QA pairs are generated automatically by DeepSeek-V3 without systematic human validation of their factual correctness or faithfulness to the source documents (Section 3.2, Appendix B.1.1). The entire evaluation hinges on the quality of this synthetic data. While a 200-sample human check was performed for the judge, no equivalent validation is reported for the QA pairs themselves. If the generated questions or answers contain systematic errors, the benchmark's scores and all derived conclusions are unreliable.
- **Inadequate Validation and Potential Bias in the Evaluation Metric:** The paper relies on a single LLM-as-judge (DeepSeek-V3) to score all model responses. While a 200-sample validation shows 88.5% agreement with a human annotator, the judge is noted to be systematically stricter (Section 3.4, Table 2). This uncalibrated strictness may penalize verbose or nuanced answers. More critically, the judge's performance and potential bias across different languages, regions, and question types are not analyzed. The evaluation protocol for non-English answers is particularly problematic: the judge receives the English "expected answer" but must evaluate a candidate answer in a different language, creating a severe semantic alignment issue.
- **Failure to Quantify or Mitigate Data Contamination:** The benchmark is constructed from Wikipedia pages and their cited external news sources, which are highly likely to be part of the common pretraining corpora for the evaluated LLMs. The authors acknowledge this risk in limitations (Section 6) but do not measure its extent or attempt to control for it. Consequently, high performance may indicate memorization of training data rather than general factual recall, undermining the benchmark's core purpose. A simple analysis (e.g., comparing performance on events with vs. without easily retrievable source documents, or a dedicated "post-cutoff" clean test) is missing.

### Minor:
- **Confounded Analysis of Geographic Disparities:** The finding that model performance correlates with GDP/HDI is compelling, but the analysis does not control for a major confounding variable: the benchmark's own inherent geographic bias. Figure 6 (Appendix C.1) shows that Wikipedia retrospective pages have orders-of-magnitude more events for regions like the US and UK. The performance gap could partly reflect this uneven benchmark coverage rather than pure model bias. A partial correlation analysis controlling for event count per region would strengthen the causal claim.
- **Overly Simplified Temporal and Correlation Analysis:** The temporal analysis bins years into wide intervals (e.g., 2023-2025), masking nuanced trends, especially around model cutoff dates. The socioeconomic analysis uses country-level indicator data from 2015 (Appendix D) for events spanning 2015-2025, which may not accurately reflect the variable data distribution in LLM training corpora over that decade. Time-aligned indicators would be more rigorous.
- **Superficial Error Analysis:** The paper reports refusal rates and accuracy drops but does not diagnose the root causes. For instance, the high refusal rate for Sabiá-3 and the performance drop for Qwen2-72B in non-English languages are noted but not investigated. A qualitative categorization of error types (hallucination, incompleteness, refusal) per region/language would provide much-needed diagnostic insight.

### Trivial:
- The prompt for QA generation instructs avoiding questions about volatile information, but there is no verification this instruction was consistently followed. The distribution of question types is presented but not linked to model performance.

## Nice-to-Haves
- A controlled comparison of region-specialized models (e.g., Sabiá-3) versus generalist models on their target region's events to explicitly test if specialization mitigates disparity.
- An ablation study using a simple Retrieval-Augmented Generation (RAG) baseline with the provided source documents to establish a performance upper bound and contextualize the difficulty of the questions.
- Fine-grained, year-by-year performance plots to better visualize knowledge decay around model cutoff dates.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Strength: "The benchmark is ambitious in its multi-dimensional scope."** Removed as a generic strength that could apply to many papers.
- **Weakness: "The LLM-as-judge evaluation is uncalibrated and may introduce systematic bias."** This point was moved to the main Weaknesses section as it is a valid, substantive criticism, but the original phrasing from the harsh reviewer contained an overstatement ("undermines the reliability") that has been tempered.
- **Weakness: "Structural: The benchmark fundamentally measures memorization..."** The core of this criticism (data contamination) is kept as a major weakness. However, the absolute claim that this "invalidates the paper's core claims" is an overinterpretation and is softened. The benchmark still measures *recall*, even if contaminated; the issue is whether it measures *generalization*.
- **Weakness: Requests for "missing related works" or criticisms about "unreleased models."** Removed per hard rules. All cited models (e.g., DeepSeek-V3, Sabiá-3) are treated as existing and available.
- **Weakness: Nitpicks about undisclosed hyperparameters or large training logs.** Removed per hard rules on reproducibility.
- **Weakness: "The choice of GDP/HDI data from 2015... is problematic."** This point is partially addressed in the paper's methodology (using 2015 as a baseline) and is considered a minor, not major, weakness. The original criticism from the spark finder is weakened as it demands a level of precision not standard for this type of correlational analysis.

## Suggestions
- **Conduct a human validation study** on a statistically significant, stratified sample (e.g., 500-1000) of the generated QA pairs to report precision/recall and ensure factual correctness. This is essential for establishing benchmark credibility.
- **Re-run the non-English evaluations with a proper multilingual judge.** The judge should compare the native-language candidate answer to a verified native-language reference translation, not an English "expected answer."
- **Add a critical analysis section** quantifying the potential contamination issue. For example, compare model performance on events from the most recent time bin (post-all-model-cutoffs) versus earlier bins, or report the proportion of source URLs that appear in common crawl snapshots.
- **Strengthen the geographic disparity analysis** by reporting partial correlations between model performance and GDP/HDI *after controlling for* the number of Wikipedia events or retrieved sources per country (data already in Appendix C.1).

**Evaluation Axes:**
- **Novelty:** High. The synthesis of temporal, geographic, and linguistic evaluation into a single, updatable benchmark is a novel and valuable contribution.
- **Technical Soundness:** Medium. The core pipeline is clearly described and scalable. However, the lack of validation for the synthetic QA data and the unexamined biases in the LLM-as-judge protocol are significant technical flaws that undermine the reliability of the results.
- **Empirical Support:** Medium-Low. The experiments are extensive in scale but built on an unvalidated foundation. The correlations are interesting but potentially confounded. The empirical claims are not fully supported by the presented evidence due to the methodological gaps.
- **Significance:** High. The work addresses timely and important issues of LLM fairness, global representation, and continual learning. The demonstrated socioeconomic correlation is a significant finding that merits attention.
- **Clarity:** High. The paper is well-structured, figures are informative, and the methodology is easy to follow.

---

## ZS4fa5FgTD

- GT: Withdrawn (treated as Reject) (avg 2.7)
- Predicted: N/A (4.0/10)
- Match: N/A

### Final Review

## Summary
This paper introduces DyCO-GNN, an unsupervised learning framework for dynamic combinatorial optimization (DCO) that requires no training data. The method adapts the "shrink-and-perturb" technique to warm-start a GNN-based optimizer across temporal graph snapshots, aiming to accelerate convergence while preserving solution quality. Experiments on dynamic MaxCut, MIS, and TSP show consistent improvements over static and naively warm-started PI-GNN under varying time budgets.

## Strengths
- **Novel problem setting**: The work is the first to propose a learning-based, training-data-free approach for dynamic combinatorial optimization, addressing a clear gap between instance-specific learning and real-world dynamic problems.
- **Simple and effective method**: The adaptation of shrink-and-perturb (SP) to mitigate the local-optima trapping of naive warm-starting is straightforward yet yields robust gains across three CO problems and multiple real-world/synthetic dynamic graphs, often achieving better solutions than converged static PI-GNN in a fraction of the runtime.
- **Comprehensive empirical evaluation**: The paper tests DyCO-GNN on MaxCut, MIS, and TSP with different GNN architectures, time budgets, and sensitivity analyses (degree of change, SP parameters), providing solid evidence of its effectiveness within the evaluated scope.

## Weaknesses
### Major:
- **Limited core algorithmic novelty**: The central algorithmic idea—applying shrink-and-perturb (SP) to warm-start a neural optimizer—is directly borrowed from Ash & Adams (2020), a supervised learning technique. While the application to unsupervised dynamic CO is new, the methodological advance is incremental. For a top-tier conference, this adaptation alone may not suffice without deeper theoretical or mechanistic innovation.
- **Insufficient explanation for design choices and performance variation**: The paper evaluates three ways to apply SP (embedding layer, GNN layers, full network) with no single variant dominating across tasks/datasets (Tables 1–3). This inconsistency is not analyzed or explained, leaving users without principled guidance on which configuration to choose for a new problem. The paper also lacks a mechanistic analysis of *why* SP helps in this specific optimization context (e.g., how it affects the QUBO loss landscape or gradient dynamics).
- **Narrow exploration of dynamic scenarios**: Experiments are limited to edge additions/deletions (MaxCut/MIS) and a single moving node (TSP). More complex and realistic dynamics—node additions/deletions, simultaneous structural and constraint changes, or adversarial perturbations—are not tested, undermining claims of general applicability to DCO.
- **Weak theoretical connection to the main method**: Theorem 1 analyzes perturbation in the Goemans–Williamson (GW) algorithm for MaxCut, which uses SDP relaxation and randomized rounding. This provides only analogical support for DyCO-GNN, which performs gradient-based optimization of a relaxed QUBO objective via GNNs. The theorem does not offer direct insight into the GNN-based optimization process, leaving the method’s success as an empirical observation.

### Minor:
- **Limited baseline comparisons beyond the PI-GNN family**: While the paper focuses on instance-specific methods and includes some non-neural baselines in Appendix D.3, it does not compare against established dynamic CO algorithms or reoptimization heuristics from the optimization literature. This makes it difficult to assess the practical competitiveness of DyCO-GNN in the broader DCO landscape.
- **Hyperparameter choices presented as universal without full justification**: The SP parameters (λ_shrink=0.4, λ_perturb=0.1) are fixed across all experiments with a claim of "no further tuning." Although sensitivity analysis is provided in the appendix, the main text does not discuss the robustness of these choices or how they might need adjustment for different problems or dynamic regimes.

### Trivial:
- **Occasionally confusing metric notation**: In tables, the notation "Values closer to 1 are better (↑/↓)" is slightly ambiguous for TSP (where lower ApR is better), though the context clarifies the meaning.

## Nice-to-Haves
- Developing an adaptive SP mechanism that adjusts λ_shrink/λ_perturb or the layers to perturb based on snapshot similarity or gradient signals.
- Extending evaluation to dedicated dynamic CO benchmarks or synthetic dynamic graphs with controlled change properties (e.g., node arrivals/departures, large rewiring).
- Providing a deeper ablation study comparing SP to alternative stabilization techniques (e.g., learning rate resets, gradient clipping) to isolate the contribution of the specific SP formulation.

## Removed Points
*These points are flagged to be removed, treat them with caution*

**Strengths removed:**
- "The paper is well-written" – generic strength.
- "The topic is important" – generic strength.
- "The experiments are extensive" – already covered by specific empirical evaluation strength.

**Weaknesses removed:**
- "The ground truth acquisition using Gurobi with a 60-second time limit is unreliable" – The paper explicitly states Gurobi is used with a time limit, and this is standard practice for obtaining reference solutions; doubting the existence or availability of Gurobi violates the hard rule.
- "Missing comparison to all existing dynamic CO algorithms" – This is scope creep; the paper focuses on learning-based, instance-specific methods and includes relevant non-neural baselines in the appendix. Demanding exhaustive comparison to every traditional algorithm is unreasonable.
- "Hyperparameters like σ for the noise ε^t are not specified" – This is a reproducibility nitpick about implementation details; the hard rule removes such trivial hyperparameter complaints.
- "Formatting issues in tables make data hard to read" – Pure formatting/style nitpick, removed per hard rule.
- "The method does not scale to millions of nodes" – The paper evaluates graphs up to thousands of nodes, which is reasonable for a research submission; requesting arbitrarily larger scales is a generic one-size-fits-all weakness.

## Suggestions
- In the revision, add a concise discussion in the main text explaining the performance variation across SP application strategies (emb/GNN/full) and provide practical guidance on selecting a configuration based on problem characteristics.
- Expand the experimental section to include at least one more complex dynamic scenario (e.g., node additions/deletions) to better demonstrate generalizability.
- Strengthen the connection between Theorem 1 and the GNN method by adding a brief discussion on how the intuition from GW perturbations might translate to gradient-based optimization with SP, or explicitly note the theorem's role as analogical support.

---

## 80JylHgQn1

- GT: Accept (Oral) (avg 7.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Final Review

## Summary
This paper proposes a framework for generating semantically rich and expressive video avatars by simulating dual-process cognition. The core innovations are: 1) an MLLM-based agentic reasoning module ("System 2") that produces high-level textual guidance from audio, image, and optional text inputs, and 2) a specialized Multimodal Diffusion Transformer (MMDiT) architecture with a novel Pseudo Last Frame (PLF) strategy ("System 1") to fuse multimodal signals and mitigate interference. The method demonstrates strong quantitative performance, high user preference, and promising generalization to multi-person and non-human subjects.

## Strengths
- **Novel Cognitive Perspective and Technical Integration:** Framing avatar generation through a dual-process (System 1/System 2) analogy provides a fresh, motivating lens. The technical realization—integrating a multi-step MLLM planner with a redesigned MMDiT featuring the PLF strategy—is a comprehensive and well-executed contribution tailored to the avatar domain.
- **Extensive and Multi-Faceted Evaluation:** The paper validates its method rigorously with standard objective metrics (FID, FVD, Sync-C), novel motion dynamics metrics (HKC, HKV), detailed human subjective studies (pairwise GSB, artifact analysis), and supplementary MLLM-based semantic evaluation on challenging custom datasets. The strong and consistent user preference over academic and proprietary baselines is particularly convincing.
- **Demonstrated Generalization and Robustness:** The framework is shown to generalize effectively to complex, under-explored scenarios such as multi-person interactions and non-human characters, indicating its robustness and broader applicability beyond standard talking-head tasks.

## Weaknesses
### Major:
- **Insufficient Causal Evidence for the Core "Reasoning" Claim:** While ablation studies show that removing the MLLM module reduces motion dynamics (HKV) and increases perceived motion unnaturalness (MU), the paper does not conclusively isolate the benefit of *structured reasoning* from the effect of simply adding *any* high-level textual conditioning signal. A controlled comparison against a baseline using the audio transcript or a simple caption as text conditioning (bypassing the Analyzer/Planner) is missing, leaving the necessity and specific contribution of the "deliberative" agentic pipeline ambiguous. (Sec. 4.2, Tables 1 & 2a)
- **Evaluation Gaps for Semantic Coherence:** The claimed generation of "semantically rich and expressive" motions is supported primarily by proxy metrics (e.g., HKV) and overall user preference. While the MLLM-based evaluation in Appendix D.2 is a valuable step, it is not a standardized or validated benchmark and is used only supplementally. The paper lacks a targeted, quantitative protocol to directly measure alignment between generated motion and the high-level semantics of the input context (e.g., emotion, intent, narrative), which is central to its contribution. (Sec. 4)
- **Overstated Novelty Claims:** The assertion of being "the first to frame the video avatar problem through the cognitive science lens" (Sec. 1, Contributions) is overstated. Prior work has extensively used LLMs/MLLMs for planning and reasoning in video generation and agent simulation (e.g., MORA, StoryAgent, Anim-Director, cited in Sec. 2.3). The paper's primary novelty lies in the specific application to avatars and the integrated technical design, not in the foundational idea of using LLMs for cognitive simulation.

### Minor:
- **Incremental Nature of Some Technical Components:** The use of MLLMs for high-level guidance and the MMDiT architecture are built upon rapidly evolving existing literature. The PLF strategy is an elegant engineering solution but is presented more as a clever training trick than a principled methodological advancement; its mechanism (RoPE shift) lacks deep theoretical justification or comparison to a broader set of identity-preservation techniques. (Sec. 3.3)
- **Reproducibility and Benchmarking Concerns for MLLM Evaluation:** The MLLM-based evaluation relies on a proprietary model (Gemini-2.5-Pro) and specific, verbose prompts (provided in the appendix). This makes full independent replication and future benchmarking challenging for the community. (Appendix D.2)
- **Practical Latency Overhead:** The agentic reasoning module introduces a significant, fixed latency (~20-30 seconds) which, while argued as a justifiable trade-off for quality, impacts real-time applicability and should be more thoroughly discussed in the context of potential use cases. (Appendix F)

### Trivial:
- The paper is comprehensive and well-structured, with no trivial formatting or presentation issues.

## Nice-to-Haves
- A more systematic ablation of the PLF mechanism, analyzing the Pareto frontier of identity preservation vs. motion dynamics across different RoPE shift values.
- A failure case analysis for the agentic reasoning module, illustrating when and why the MLLM planner generates incoherent schedules and how this propagates to the final video.
- A computational breakdown of inference latency (MLLM call vs. diffusion sampling) to better contextualize the overhead.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Strength (Removed):** "The paper is well-written" – This is a generic strength that applies to many papers.
- **Weakness (Removed):** "The core claim of cognitive simulation is not substantiated" – This criticism is too absolute. The paper provides ablation evidence (reduced HKV, improved MU in user studies) linking the MLLM module to improved motion quality, even if the causal chain could be more directly proven. It is a limitation, not a complete invalidation.
- **Weakness (Removed):** "The Pseudo Last Frame strategy is inadequately justified and compared" – The paper includes ablations against a reference-attention baseline (Table 1, Table 2b) and visual analysis (Figs. 8, 9). Demanding comparison to every other identity-preservation technique is scope creep for this paper's contribution.
- **Weakness (Removed):** Criticisms about the MLLM models (e.g., Seed-1.5-VL) not being released or verifiable – The paper cites these models, so they are assumed to exist per the hard rules.
- **Weakness (Removed):** Criticisms about missing implementation details (exact prompts, hyperparameters) – These are provided in the appendix, fulfilling standard reproducibility expectations.

## Suggestions
- **Strengthen the Causal Argument for Reasoning:** Conduct a key ablation comparing the full agentic pipeline against a baseline that conditions the MMDiT on a simple text signal derived directly from the input (e.g., the audio transcript or a CLIP caption of the reference image). This would help isolate the added value of the MLLM's structured "reasoning" over generic text conditioning.
- **Propose a Standardized Metric for Semantic Alignment:** Leverage the MLLM-based evaluation protocol introduced in the appendix to define and report a quantitative "Semantic Coherence Score" on the main test sets. This would directly measure the paper's core contribution and set a valuable standard for future work.
- **Temper the Novelty Claims:** Revise the language in the introduction and contributions to more accurately reflect that the paper applies and integrates existing concepts (LLM-based planning, multimodal DiTs) in a novel way for the avatar domain, rather than claiming to be the absolute first to employ a cognitive lens.

**Overall Assessment:** This is a strong paper with a compelling narrative, solid technical innovations, and extensive empirical support. It makes a meaningful advance in pushing avatar generation toward higher-level semantic coherence. The weaknesses identified are primarily related to the strength of evidence for its central thesis and some overclaims, not fundamental flaws in the methodology or results. With revisions to provide more direct evidence for the role of reasoning and to temper novelty statements, this would be an excellent contribution.

---

## Dm6lP9YEsM

- GT: Reject (avg 5.0)
- Predicted: N/A (4.5/10)
- Match: N/A

### Final Review

## Summary
This paper introduces MASH, a framework that trains LLMs for selective help-seeking via reinforcement learning with a pay-per-search penalty. The core idea is that optimizing for efficient tool use naturally aligns search decisions with the model’s parametric knowledge boundaries, enabling abstention when search is disabled. Experiments on three QA datasets demonstrate improved tool productivity over prior efficient-search baselines and competitive abstention performance compared to methods explicitly trained for abstention.

## Strengths
- **Novel conceptual contribution:** The link between selective help-seeking and abstention is elegantly motivated and offers a unified approach to improve both tool efficiency and reliability without requiring pre-defined knowledge boundaries for training.
- **Comprehensive empirical evaluation:** The paper thoroughly evaluates multiple datasets, reward penalties, and model scales, and includes insightful analyses (warm-start ablation, oracle helper, out-of-distribution generalization) that deepen understanding of the method’s behavior.
- **Practical warm-start procedure:** The synthetic SFT data generation from a different model is a simple yet effective solution to encourage diverse search behaviors without baking in the base model’s knowledge boundaries, addressing a key exploration challenge in RL.

## Weaknesses
### Major:
1. **Conceptual mismatch in abstention evaluation:** The paper treats search-tag generation (when search is disabled) as abstention, which is a different behavior from verbalized uncertainty (e.g., outputting “I don’t know”). While this is a valid proxy, the direct comparison to methods like DPO and AFH that are trained to output explicit abstention phrases is not entirely fair, and the claim of “analogous” behavior is overstated without evidence that the underlying decision processes are similar.
2. **Incomplete ablation of warm-start contribution:** The reported improvements of MASH over the OTC baseline are confounded because MASH uses warm-start SFT while OTC does not. Without an ablation where OTC is also warm-started, it remains unclear whether the gains stem from the novel reward formulations or simply from better initialization, undermining the claim that MASH’s RL training extracts better search behaviors.
3. **Fragility of training:** The method requires dataset-specific tuning of reward penalties and heavily relies on warm-start to avoid degenerate policies (Table 4). This sensitivity limits robustness and general applicability, as the approach may not transfer easily to new domains without careful hyperparameter selection.

### Minor:
1. **Limited evaluation scope:** Primary experiments use a single base model (Qwen2.5-3B) and three QA datasets. While additional model scales are briefly explored in the appendix, broader evaluation across model families and task types would strengthen claims of generality.
2. **Inconsistent out-of-distribution generalization:** As shown in Section 4.5 and Appendix F, generalization to other datasets is mixed (e.g., models trained on multi-hop data struggle on single-hop questions), suggesting learned policies may overfit to dataset-specific patterns.
3. **Basic theoretical analysis:** The theoretical analysis in Appendix A merely restates the optimality condition of the RL objective and does not provide new insights (e.g., convergence guarantees or the effect of penalty severity on the decision threshold).
4. **Dependence on exact match and LLM judge:** Correctness evaluation uses exact match for training/validation and an LLM judge (DeepSeek-V3.1) for testing. The impact of the judge’s potential biases is not ablated, which could affect reported metrics.

### Trivial:
- None.

## Nice-to-Haves
- Comparison to state-of-the-art selective RAG methods (e.g., SEAKR, DRAGIN) to better contextualize selective search performance.
- Evaluation with varying retriever quality (e.g., a weak retriever) to assess robustness to noisy retrieval signals.
- Extension to other task types (e.g., long-form QA, fact-checking) to demonstrate broader applicability of the framework.

## Removed Points
- **Synthetic data with intentional errors:** The paper intentionally uses synthetic data with 35% errors from a different model to avoid aligning with the base model’s knowledge boundaries. This is a design choice explained in the paper, not a flaw.
- **Answerability threshold λ=0.1:** While arbitrary, this is common practice in prior work, and the paper acknowledges the lack of consensus. We retained this as a minor weakness but not a major issue.
- **Missing composite metric for abstention:** The paper reports both overall accuracy and precision on non-abstained questions, which is standard. Demanding a single composite metric is not necessary.

## Suggestions
- Conduct an ablation study training the OTC baseline with the same warm-start procedure to isolate the effect of the reward formulations.
- Perform a per-question analysis linking penalty severity to the model’s switch from parametric to search behavior based on estimated parametric accuracy.
- Include qualitative examples of successful and failed multi-hop trajectories to illustrate the learned search strategies and their limitations.

---

