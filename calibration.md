=== CALIBRATION EXAMPLE 1 ===

# Final Consolidated Review
Summary: This paper provides a broad survey-style overview of large language models, covering core architectures, representative applications, limitations, ethics, evaluation, and future directions. Its main value is as an introductory summary for readers new to LLMs. However, as submitted, it does not substantiate the title’s claim of being a systematic review: the paper contains no review protocol, search strategy, inclusion/exclusion criteria, or reproducible selection methodology. Beyond this central issue, the survey lacks a clearly articulated gap relative to prior surveys, offers limited synthesis beyond descriptive summaries, and has notably outdated/incomplete coverage for a contemporary LLM review. The comparative analysis is especially weak, relying on a small set of older models with mostly qualitative pros/cons rather than a meaningful cross-model analysis. Overall, the paper is understandable and touches many relevant themes, but it currently reads as a general narrative overview rather than a rigorous, differentiated, or systematic survey.
Strengths:
- Provides a reasonably accurate introduction to foundational LLM concepts such as transformers, self-attention, masked language modeling, generative pretraining, and sequence-to-sequence models.
- Covers a broad range of topics relevant to LLMs, including applications, hallucination, fairness, robustness, ethics, evaluation metrics, and future directions.
- Includes several foundational and relevant references, including Vaswani et al., BERT, GPT-style models, T5, BART, fairness surveys, and evaluation-oriented work.
- Acknowledges newer evaluation frameworks such as HELM and LMSYS Chatbot Arena, indicating awareness that LLM assessment extends beyond classic task metrics.
- Could be useful as an entry-level overview for non-specialists or readers seeking a high-level primer.
Weaknesses:
- The title claims a systematic review, but the paper does not present any systematic-review methodology. There is no search protocol, database/query description, inclusion/exclusion criteria, screening process, quality assessment, or reproducible paper-selection procedure.
- The paper does not clearly explain what it contributes beyond existing surveys already cited in the manuscript (e.g., Bommasani et al., Zhao et al., Chang et al.). Section 3.11 mentions prior reviews but does not articulate a distinct perspective, gap, or added value.
- Coverage is substantially incomplete/outdated for a modern LLM survey. The discussion centers mostly on BERT/GPT-2/GPT-3/T5/BART, while major developments such as instruction tuning, RLHF/RLAIF, retrieval-augmented generation, LoRA/QLoRA and other PEFT methods, quantization/efficient inference, and major recent model families are largely absent or only minimally touched.
- The comparative analysis is weak. Table 1 covers only four mostly older models and provides superficial qualitative pros/cons rather than systematic, quantitative, or benchmark-grounded comparison.
- The survey offers little novel synthesis. It mostly lists established facts about architectures, tasks, and challenges, but does not provide a new taxonomy, meta-analysis, unifying framework, or sharply defined research agenda.
- There is noticeable repetition and editing redundancy, especially in Section 3, where some technical points and paragraphs are duplicated.
- Some citations are incomplete or unresolved (e.g., placeholder '?'), which weakens scholarly reliability.
- Sections on future directions and open problems remain generic; they do not identify concrete unresolved questions, trade-offs, or methodological bottlenecks in a way that would guide researchers.
Nice-to-haves:
- A PRISMA-style review protocol, including search terms, sources, screening flow, and inclusion/exclusion criteria, if the authors want to retain the term systematic review.
- A more explicit positioning section comparing this survey against prior LLM surveys and explaining the paper's unique scope or contribution.
- A substantially expanded comparative table with recent model families, alignment methods, parameter-efficient adaptation methods, retrieval augmentation, multimodal models, and benchmark-based comparisons.
- A stronger synthesis layer: for example, a new taxonomy of LLM development/evaluation/deployment trade-offs or a structured map of open problems.
- Bibliometric or meta-analytic evidence such as publication trends, benchmark trends, or capability/efficiency trade-off summaries.
- Concrete case studies or domain-focused analysis (e.g., medicine, code, scientific reasoning, agentic use) rather than only broad topical mentions.
Novel insights: The most important filtered conclusion is that the paper’s central weakness is not merely lack of novelty, but a mismatch between claimed genre and actual content. As written, this is a narrative overview, not a systematic review. That distinction matters because the paper’s rigor, reproducibility, and contribution should be judged according to the standards implied by its title. Separately, the paper’s strongest realistic value is pedagogical rather than scholarly: it may help newcomers, but it does not yet provide the integrative or up-to-date synthesis that would make it a strong research survey.
Potentially missed related work:
- Instruction tuning and alignment methods such as InstructGPT, FLAN-style work, RLHF/RLAIF, Constitutional AI, and related preference-optimization methods could be considered for broader coverage.
- Retrieval-augmented generation and tool-augmented/agentic LLM frameworks may be useful additions if the authors aim for a comprehensive modern review.
- Parameter-efficient fine-tuning and deployment efficiency methods such as LoRA/QLoRA, prefix tuning, quantization, and speculative decoding may strengthen the practical-usage angle.
- Recent open and proprietary model families (e.g., LLaMA-series, Mistral/Mixtral, Claude, Gemini, PaLM-family) may help update the comparative analysis.
- Modern multimodal LLMs and vision-language assistants could deepen the current brief discussion of multimodality.
Suggestions:
- Either revise the title/claims to present the paper as a narrative survey, or add a genuine systematic-review methodology with transparent and reproducible study selection procedures.
- Add a clear contribution statement early in the paper that explains what this survey provides beyond prior surveys and why another review is needed now.
- Update the technical coverage to include major post-2020 developments, especially alignment, instruction tuning, retrieval augmentation, parameter-efficient fine-tuning, efficient inference, and modern multimodal LLMs.
- Replace the current comparative table with a more comprehensive and benchmark-grounded analysis, ideally including capability, scale, training/adaptation paradigm, evaluation tasks, and deployment trade-offs.
- Remove duplicated material and consolidate overlapping sections, particularly in the literature review and limitations discussions.
- Fix unresolved or placeholder citations and ensure references are complete and accurate.
- Strengthen the future directions section by identifying concrete research gaps, open questions, and trade-offs rather than high-level generic themes.
- If staying within a broad survey scope, add a stronger synthesis layer such as a taxonomy, framework, or bibliometric/meta-analytic analysis to create genuine scholarly value.

# Merger Subscores
Novelty: 2.0
Technical soundness: 3.5
Empirical support: 2.0
Significance: 3.0
Clarity: 5.5

# Actual Human Scores
Individual reviewer scores: [1.0, 1.0, 1.0, 1.0]
Average score: 1.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 2 ===

# Final Consolidated Review
Summary: The paper tackles an important and underexplored problem: extending decision-focused learning from single-objective settings to multi-objective optimization. The proposed MoDFL combines three losses intended to align prediction with downstream multi-objective decision quality: a landscape loss in objective space, a Pareto set loss in solution space, and a representative decision loss after scalarization. This is a reasonable and potentially useful formulation, and the empirical section includes comparisons to several adapted DFL baselines plus ablations suggesting each component helps. However, the paper has substantial limitations that weaken the strength of its claims. Most importantly, the method as instantiated relies on weighted-sum scalarization and DSLP-style differentiation, so the demonstrated scope is essentially smooth LP-style problems and convex/scalarizable fronts rather than general multi-objective optimization. In addition, the experiments contain inconsistencies that need clarification (notably identical Tables 1 and 2, and textual claims that do not match tabulated numbers), and the evaluation lacks variance/statistical reporting despite repeated runs. Overall, the paper identifies a real gap and offers a sensible first step, but the current presentation and empirical validation are not yet strong enough to fully substantiate the broader claims.
Strengths:
- Addresses a meaningful gap: bringing decision-focused learning into multi-objective settings is important and non-trivial because Pareto-optimal solutions are set-valued and objectives can induce conflicting gradients.
- The three-loss decomposition is conceptually well motivated: objective-space alignment (landscape loss), solution-space alignment (Pareto set loss), and representative decision quality (decision loss) capture complementary aspects of multi-objective prediction-and-decision mismatch.
- The method is not merely a naive weighted sum of existing single-objective losses; it proposes MOP-specific surrogates, especially the objective-space and Pareto-set components.
- Ablation results support that all three loss components contribute, with decision loss appearing especially important.
- The paper compares against a reasonably broad set of standard decision-focused baselines (TwoStage, SPO, BB, MAP, NCE, Pointwise, Listwise), which is useful for positioning against the DFL literature.
- The landscape-loss comparison against MMD and DSPM is a helpful attempt to justify the choice of sRMMD.
Weaknesses:
- The demonstrated applicability is substantially narrower than the paper's broad framing suggests. The differentiation mechanism is instantiated via DSLP, and experiments are on LP-style problems; the paper does not establish support for general nonlinear or mixed-integer multi-objective optimization.
- The decision loss relies on weighted-sum scalarization with uniform weights. This is a restrictive choice and can miss non-convex regions of the Pareto front; this limitation is not adequately discussed.
- The experimental section has serious consistency issues: Tables 1 and 2 appear identical despite supposedly corresponding to different benchmarks, and some textual claims in Section 5.3 do not match the table values. These must be clarified because they affect confidence in the empirical evidence.
- Results are reported as point estimates only, despite stating experiments were repeated 5 times. No standard deviations, confidence intervals, or significance testing are provided.
- The baseline adaptation to MOP is fairly naive: single-objective DFL methods are converted by uniformly aggregating per-objective losses. This is a reasonable baseline construction, but it may not be the strongest multi-objective adaptation, so claims of superiority should be tempered.
- Hyperparameter choices for the three loss weights and other components are given without justification or sensitivity analysis, even though performance likely depends on balancing losses on different scales.
- The study of increasing objective count is limited: the third objective is constructed as a weighted combination of the first two, so it does not convincingly demonstrate scalability to genuinely many-objective settings.
- The paper lacks computational analysis. Since the method adds Pareto-set computations, solver calls, and sRMMD/OT-based loss computation, runtime and scalability are important but unreported.
- The paper would benefit from stronger theoretical grounding: there is little analysis of when these surrogate losses should improve Pareto-front approximation or decision quality.
- The method appears to require access to Pareto-optimal solution sets for training instances (or their approximations), which may be expensive and should be discussed more explicitly as a practical requirement.
Nice-to-haves:
- Visualizations of predicted vs. true Pareto fronts and Pareto sets to make the improvements more interpretable.
- Training curves for the three loss terms and decision metrics over time.
- Failure-case analysis identifying when MoDFL helps least or fails.
- Evaluation on standard multi-objective benchmarks with 4+ independent objectives.
- Runtime/memory comparisons against baselines.
Novel insights: The strongest aspect of the work is the recognition that in multi-objective DFL, coefficient prediction errors should not be judged only through a scalarized downstream decision loss. Because MOPs are set-valued, there is value in separately aligning (i) the geometry of the objective landscape, (ii) the location of the Pareto set in solution space, and (iii) the quality of a representative deployed decision. This decomposition is a useful conceptual contribution even if the present instantiation is limited.
Potentially missed related work:
- As a suggestion, the paper could better connect to broader multi-objective ML literature, including Pareto-front learning, multi-objective optimization in machine learning, and preference-based/scalarization methods beyond weighted sums.
- It may also be useful to discuss classical multi-objective optimization baselines and scalarization alternatives such as Tchebycheff or ε-constraint methods when motivating the decision-loss design.
- If space permits, connecting to standard MOP benchmark ecosystems (e.g., DTLZ/WFG-style evaluations) would help readers place the contribution.
Suggestions:
- Clarify and correct the experimental presentation: explain why Tables 1 and 2 are identical or fix them, and ensure all text matches the reported numbers.
- Report mean ± standard deviation over the 5 runs, and ideally include significance testing for key comparisons.
- Temper the claims of generality and explicitly state that the current method is demonstrated for LP-style problems with DSLP-based differentiation and weighted-sum scalarization.
- Discuss the implications of weighted-sum scalarization, especially its inability to recover non-convex Pareto-front regions; alternatively, extend the approach to stronger scalarization schemes.
- Provide hyperparameter sensitivity studies for λ_l, λ_d, λ_ps, as well as the OT/sRMMD parameters.
- Add runtime or complexity analysis to quantify the overhead of the additional loss terms and solver usage.
- Strengthen the empirical section with genuinely many-objective benchmarks (at least 4 independent objectives) and, if possible, non-LP or mixed-integer settings.
- Include visualizations of Pareto fronts/sets and representative qualitative examples to show what the proposed losses improve.
- Clarify the training-time requirement for Pareto sets/approximations and discuss practicality when such sets are expensive to obtain.
- Add theoretical discussion or formal propositions where possible, including the normalization claim and conditions under which the surrogate losses are expected to align with downstream MOP quality.

# Merger Subscores
Novelty: 6.7
Technical soundness: 5.7
Empirical support: 4.9
Significance: 6.1
Clarity: 4.8

# Actual Human Scores
Individual reviewer scores: [3.0, 1.0, 3.0]
Average score: 2.3
Binary outcome: Reject

=== CALIBRATION EXAMPLE 3 ===

# Final Consolidated Review
Summary: The paper presents an ambitious integration of multimodal federated learning, fully homomorphic encryption, and quantum models, centered on a multimodal quantum mixture-of-experts architecture evaluated on both standard and biomedical datasets. The integration itself is interesting, and the experimental section spans several settings (centralized/federated, classical/quantum, with/without FHE). However, the paper’s main claim—that quantum computing specifically mitigates FHE-induced degradation—is not convincingly established. Both the methodology and appendix do not provide a sound mechanism linking client-side quantum modeling to reduction of server-side FHE noise accumulation, and the empirical comparisons do not isolate FHE-specific mitigation from a general change in model class. The experimental evidence also has important weaknesses, including lack of quantitative ablations, missing variance/significance for accuracy metrics, and no comparison to stronger privacy-preserving FL baselines. Overall, the system integration is novel and potentially useful, but the strongest causal/theoretical claims are overstated relative to the evidence provided.
Strengths:
- Interesting and timely systems-level integration of three active areas: federated learning, homomorphic encryption, and quantum/hybrid quantum-classical modeling.
- The multimodal MQMoE architecture is a plausible architectural contribution, especially for sensitive biomedical multimodal data such as DNA plus MRI.
- The empirical section is relatively broad in scope, covering multiple datasets and multiple training configurations (centralized vs federated; classical vs quantum; with and without FHE).
- The paper targets privacy-sensitive healthcare scenarios, which is a meaningful application domain for multimodal FL with encrypted aggregation.
- Results do show that the proposed quantum variants can sometimes recover part of the performance lost under FHE relative to the corresponding non-quantum encrypted setup, making the empirical direction worth further investigation.
Weaknesses:
- The central claim that quantum computing mitigates FHE-induced degradation is not theoretically substantiated. The paper does not convincingly explain how the proposed quantum components would reduce or compensate for CKKS noise introduced during encrypted aggregation.
- The empirical design does not isolate the claimed effect. Showing that QFL+FHE outperforms FL+FHE is insufficient, because QFL also differs from FL without FHE; the paper should directly compare the degradation gaps induced by FHE for classical and quantum models.
- The appendix argument based on SU(2)/unitary norm preservation does not establish mitigation of FHE noise; at best it suggests bounded evolution within the quantum model, not reduction of homomorphic encryption error.
- Some notation and modeling descriptions are conceptually unclear, especially around encrypting quantum states or quantum outputs with CKKS; this conflation between classical ciphertexts and quantum states weakens technical soundness.
- The ablation study is largely qualitative prose rather than a quantitative ablation with component removals and numerical results.
- Accuracy metrics are reported without confidence intervals or multi-seed statistical analysis; only runtime has uncertainty estimates.
- Several training accuracies in the federated experiments are at or near 100%, raising concerns about overfitting and making the generalization claims harder to assess.
- The paper does not compare against stronger privacy-preserving FL baselines such as FHE-FL methods from prior work, secure aggregation alternatives, or differential privacy baselines, so it is difficult to contextualize the gains.
- The computational overhead is substantial, and although reported, it is not deeply analyzed; the method is often much slower than classical FL while the accuracy gains are modest or dataset-dependent.
- The discussion of the confusion matrices contains at least one incorrect interpretation, which reduces confidence in the care of the empirical analysis.
- All quantum results appear to be obtained with PennyLane-based simulation on classical hardware; the paper does not adequately discuss the implications of simulator-only evaluation for the practical claims.
Nice-to-haves:
- A clearer threat model specifying what privacy/security guarantees are inherited from CKKS/FHE and what attacks are in scope.
- Complexity and communication-cost analysis, including ciphertext sizes and per-round overhead.
- Visualization of expert utilization and gating behavior to verify that the MQMoE experts specialize by modality as intended.
- Analysis of when the quantum approach helps versus hurts, especially on datasets where classical baselines remain stronger.
- Experiments under non-IID client distributions and with larger client counts to better reflect realistic federated settings.
Novel insights: The most promising aspect of the paper is not the claimed reduction of FHE noise per se, but the idea that a richer multimodal hybrid model may be more robust to the accuracy loss introduced by encrypted aggregation. Framed this way, the contribution becomes a systems/representation-learning hypothesis rather than a claim about quantum mechanics directly compensating for homomorphic noise. The current results are more consistent with that weaker interpretation than with the stronger causal claim in the paper.
Potentially missed related work:
- Direct empirical comparison to prior FHE-FL baselines such as FheFL and FedSHE would strengthen the paper substantially.
- Comparisons to other privacy-preserving FL alternatives, especially differential privacy and secure aggregation, would better contextualize the cost/benefit tradeoff.
- If the authors want to maintain claims about noise mitigation, more relevant theoretical or empirical work on error/noise behavior in hybrid quantum-classical learning under practical noise models would be useful to discuss.
Suggestions:
- Reframe the main claim more cautiously unless stronger evidence is added; e.g., claim improved encrypted FL performance with a quantum-enhanced multimodal architecture rather than direct mitigation of FHE noise.
- Add the key isolating experiment: compare performance drops from no-FHE to FHE for classical FL and quantum FL separately, and test whether the quantum degradation gap is consistently smaller.
- Replace the current qualitative ablation with a quantitative component study removing QC, FHE, and MQMoE individually, ideally across at least one multimodal and one unimodal benchmark.
- Clarify the methodology and notation around what exactly is encrypted (model parameters, activations, updates) and avoid suggesting that CKKS directly encrypts quantum states.
- Report mean and variance over multiple random seeds for all primary accuracy/AUC metrics and include significance testing for small gains.
- Add baselines against prior FHE-FL systems and at least one alternative privacy-preserving FL approach such as DP-based FL or secure aggregation.
- Include a deeper efficiency analysis, covering runtime, communication, and whether the observed gains justify the overhead.
- Discuss explicitly that the quantum components are simulator-based and temper claims about practical quantum advantage or deployment until hardware evidence is available.
- Correct the confusion-matrix interpretation errors and audit the empirical analysis for similar inconsistencies.
- If possible, include non-IID federated experiments and scaling beyond 10 clients to better support the FL relevance of the framework.

# Merger Subscores
Novelty: 7.3
Technical soundness: 4.2
Empirical support: 4.8
Significance: 5.5
Clarity: 5.0

# Actual Human Scores
Individual reviewer scores: [3.0, 5.0, 3.0, 3.0, 3.0]
Average score: 3.4
Binary outcome: Reject

=== CALIBRATION EXAMPLE 4 ===

# Final Consolidated Review
Summary: The paper identifies an important and plausibly underexplored issue in federated LoRA fine-tuning under heterogeneous quantization: aggregating adapters from clients using different quantization levels can hurt performance due to a quantization-induced bias. It proposes FedQLoRA, which adds a client-specific quantization-aware adapter to separate quantization error from task-relevant LoRA updates, and an iterative variant iFedQLoRA to better handle non-IID data. The core motivation is convincing, and the empirical results consistently favor the proposed methods over the included baselines across IID/non-IID settings, client counts, and heterogeneity analyses. However, the paper overstates its scope by framing the work as for LLMs while evaluating only DistilBERT-scale models, leaves some key methodological steps insufficiently justified, and omits several important empirical controls such as overhead analysis, stronger baseline validation/comparisons, and ablations showing gains are not simply due to added parameters.
Strengths:
- The paper identifies a genuine and practically relevant problem: mixed quantization across clients can make federated adapter aggregation perform worse than homogeneous low-precision settings, and Figure 1 provides a clear motivating example.
- The proposed solution is intuitive and reasonably well aligned with the problem: using a personalized quantization-aware adapter to absorb quantization error while only aggregating the LoRA adapter is a sensible design.
- The iterative variant for non-IID settings is a meaningful extension, and the distinction between quantization bias and data heterogeneity bias is useful.
- Empirical results are fairly broad within the paper's chosen setup: IID and non-IID partitions, 3/5/10 clients, varying quantization proportions, varying Dirichlet heterogeneity, and convergence curves are all included.
- Across the reported tables and figures, FedQLoRA/iFedQLoRA generally outperform the included baselines, with iFedQLoRA usually strongest in non-IID settings.
- The connection drawn to LoRA-aware quantization is interesting, and Proposition 1 helps conceptually relate the proposed quantization-aware adapter to existing quantization-aware low-rank ideas.
Weaknesses:
- The paper's framing around large language models is not adequately supported by the experiments: all results are on DistilBERT-scale models rather than contemporary billion-parameter LLMs, so the claims should either be narrowed or validated at larger scale.
- Only classification tasks are evaluated. Since LoRA/QLoRA are especially important for generation and instruction-tuning workloads, the practical relevance would be much stronger with generation-task evidence.
- The method adds a quantization-aware adapter in addition to the LoRA adapter, but there is no ablation controlling for parameter count (e.g., larger-rank LoRA with matched trainable parameters). As a result, it is unclear how much of the gain comes from the specific bias-separation mechanism versus simply more capacity.
- Communication, memory, and computation overhead are not analyzed, even though these are central considerations in federated learning and especially relevant because the method introduces extra adapters and, in the iterative version, extra optimization steps.
- Some core methodological steps are only partially justified. In particular, the estimation of the unquantized model using a LoRA-updated quantized model and the low-rank approximation of quantization error are plausible but not fully validated, leaving some concern about circularity/error propagation and when the approximation is accurate.
- A key cited related method, LoftQ, is discussed conceptually but not compared empirically in the federated setting, which weakens the positioning of the technical novelty.
- The included FFA-LoRA baseline performs extremely poorly in the reported tables relative to the other methods, and the paper does not convincingly rule out implementation or tuning issues. This makes some comparative gains harder to interpret.
- The experimental scope over quantization settings is still limited: mostly 2-bit and 4-bit mixtures are studied, with limited evidence for broader precision regimes commonly used in practice.
- No uncertainty estimates or statistical significance information are reported, which matters because several improvements over the strongest baselines are modest.
Nice-to-haves:
- Per-client breakdowns showing whether lower-bit or higher-bit clients benefit differently from the method.
- A study of approximation quality versus quantization-aware adapter rank, to directly test the low-rank quantization-error assumption.
- Visualization or qualitative inspection of what the quantization-aware adapter learns across clients.
- A discussion of deployment scenarios where clients share architecture but differ mainly in quantization precision, to better ground the practical setting.
- A brief limitations section discussing required knowledge of the client quantizer, scalability, and failure modes when quantization error is not well captured by a low-rank adapter.
Novel insights: The strongest insight in the paper is that heterogeneity in quantization level itself can create a systematic aggregation problem in federated adapter tuning, distinct from standard data heterogeneity. That framing is useful and likely transferable beyond this exact method. The separation between a shared task adapter and a personalized quantization-compensation adapter is also a promising design pattern for other forms of systems heterogeneity.
Potentially missed related work:
- As a suggestion, the paper could more explicitly position itself relative to broader federated quantization/communication-efficient FL literature (e.g., work on quantized communication or bias correction in FL), even if the setting is different.
- Since LoftQ is central to the paper's motivation and Proposition 1, a stronger experimental and conceptual comparison to LoRA-aware quantization methods would be helpful.
- If applicable, the authors may also consider discussing more personalized federated LoRA methods beyond the included baselines, to clarify what is specific to quantization heterogeneity versus personalization more generally.
Suggestions:
- Either revise the claims to match the evaluated model scale or add experiments on genuinely large models (e.g., 1B+ parameter LLMs) to substantiate the LLM framing.
- Add at least one generation/instruction-tuning benchmark to show the method matters in the settings where LoRA/QLoRA are most commonly used.
- Include parameter-matched ablations, such as larger-rank LoRA or an extra local adapter baseline, to demonstrate that the gains come from quantization-aware separation rather than just additional trainable parameters.
- Report communication, memory, and runtime overhead, especially for iFedQLoRA, and compare these costs against the achieved accuracy gains.
- Strengthen the empirical comparison set: validate the FFA-LoRA implementation/tuning more carefully and, if feasible, include a federated adaptation of LoftQ or otherwise explain why such a comparison is not possible.
- Broaden the quantization study to additional bit widths and mixture ratios, and clarify robustness when quantization methods differ across clients.
- Add uncertainty estimates over multiple runs or significance testing for the main tables.
- Clarify the assumptions behind the unquantized-model estimation and low-rank quantization-error approximation, ideally with empirical evidence on approximation fidelity or failure cases.

# Merger Subscores
Novelty: 7.0
Technical soundness: 6.0
Empirical support: 6.0
Significance: 6.0
Clarity: 6.5

# Actual Human Scores
Individual reviewer scores: [3.0, 5.0, 5.0, 3.0]
Average score: 4.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 5 ===

# Final Consolidated Review
Summary: The paper studies stochastic minimax optimization and proposes AdaFM, an adaptive variance-reduced method that couples the primal/dual stepsizes using historical estimator norms and uses a simple shared momentum schedule β_t=1/t^{2/3}. The main claimed contribution is to substantially reduce dependence on problem-dependent tuning while retaining near-optimal O(ε^-3) sample complexity in NC-SC and NC-PL settings, matching the best known rates of non-adaptive parametric VR methods up to small factors. The work addresses an important practical issue, and the idea of coupling x/y learning rates through accumulated estimator information is meaningful and specific to minimax optimization. The theory is fairly substantial and the experiments cover toy problems, deep AUC, and WGAN-GP. However, several claims are overstated: the method is not truly parameter-free in the strongest sense since γ, λ, and especially δ remain, and the ablations show δ materially affects behavior, with δ=0 failing in the toy study. Empirically, the paper demonstrates robustness relative to selected minimax baselines, but the support is limited by lack of variance/error bars and by omission of strong practical baselines such as Adam in GAN training. Overall, this is a technically interesting and potentially useful contribution, but the practical claims should be toned down and the empirical evaluation strengthened.
Strengths:
- Targets a real and important problem: variance-reduced minimax methods can be highly sensitive to tuning, and the paper provides concrete evidence of this sensitivity on WGAN-GP.
- Introduces a nontrivial adaptive design specific to minimax optimization, especially the coupled learning-rate rule where the x-update depends on both α_t^x and α_t^y, reflecting the need to be conservative when the inner maximization is not yet well solved.
- Uses a simplified shared momentum schedule β_t=1/t^{2/3}, avoiding the separate problem-dependent momentum choices used in prior VR minimax methods such as VRAdaGDA.
- Provides substantial theoretical analysis in both NC-SC and NC-PL settings, with near-optimal O(ε^-3) sample complexity claims that match strong parametric VR baselines up to small slack terms.
- Includes experiments beyond synthetic tests, covering deep AUC and WGAN-GP, and shows improved robustness over the chosen minimax baselines in several settings.
- Clearly positions the practical motivation around reducing dependence on unknown problem constants such as smoothness and gradient bounds.
Weaknesses:
- The paper's strongest wording around being 'parameter-free' or eliminating manual tuning is too strong. AdaFM still uses γ, λ, and δ, and δ is not innocuous empirically.
- The ablation results indicate that δ materially affects convergence behavior; in particular, δ=0 fails on the toy example. This weakens the claim that the remaining hyperparameters pose little difficulty and suggests the method reduces, rather than eliminates, tuning burden.
- Empirical support is limited by the absence of multi-seed results, error bars, or statistical variability estimates, which is important for stochastic optimization and GAN experiments.
- The GAN evaluation omits standard practical baselines such as Adam/AdamW, making it hard to assess whether AdaFM improves over what practitioners actually use, rather than only over selected minimax-oriented baselines.
- The paper does not sufficiently analyze or discuss the practical implications of its condition-number dependence, especially the worse κ dependence in the NC-PL result.
- Some positioning relative to TiAda is imprecise. The paper should more carefully distinguish 'adaptive minimax method' from 'adaptive VR-based minimax method' and clarify that the stronger novelty is achieving VR-style near-optimal complexity with adaptive tuning, not being the first adaptive minimax method overall.
- There is limited ablation of the key design choices beyond δ; for example, the benefit of the max{α_t^x, α_t^y} coupling in η_t^x is not isolated empirically.
- The bounded-gradient assumption is restrictive for modern deep models, and the paper gives only limited discussion of when this assumption is reasonable in practice.
Nice-to-haves:
- Report wall-clock overhead and memory cost relative to baselines, since AdaFM maintains cumulative estimator statistics.
- Show the evolution of η_t^x, η_t^y, and their ratio on real tasks, not just on the toy example.
- Add explicit failure-case discussion, e.g., behavior under poor γ, λ choices or extreme conditioning.
- Provide default hyperparameter recommendations and demonstrate one common setting across several tasks.
- Release code for reproducibility.
Novel insights: The strongest insight in the paper is not merely 'using adaptivity' but the minimax-specific coupling of primal and dual stepsizes via accumulated estimator information. This operationalizes a key structural intuition: the x-step should slow down when evidence suggests the y-subproblem is not yet sufficiently controlled. That coupling, together with a single decaying momentum schedule, is the most compelling conceptual contribution and seems to be what enables the theory to approach the best known VR sample complexity without relying on the usual problem-dependent schedules.
Potentially missed related work:
- As a suggestion, the discussion/comparison could be broadened to include more recent adaptive minimax methods such as AEGD-style approaches or PES-type results in related NC-PL settings, where relevant to the paper's problem formulation.
- For practical experiments, comparison to standard optimizers used in minimax training such as Adam/AdamW would strengthen contextualization, especially for GANs.
- If space permits, it may also be useful to connect more explicitly to adaptive VR methods in minimization (e.g., STORM+/AdaGrad-style lines) to sharpen what is inherited versus what is new in the minimax extension.
Suggestions:
- Revise the claims throughout to say that AdaFM substantially reduces hyperparameter burden, rather than fully eliminating tuning; explicitly acknowledge that δ is important in practice.
- Clarify the novelty claim relative to TiAda: the paper appears strongest as the first adaptive VR-based minimax method with near-optimal O(ε^-3) complexity, not necessarily the first adaptive minimax method overall.
- Add multi-seed experiments with error bars or confidence intervals for the main results, especially Deep AUC and WGAN-GP.
- Include stronger practical baselines for GAN training, at least Adam or AdamW, to better establish practical competitiveness.
- Add ablations isolating the main design choices, especially the coupled max{α_t^x, α_t^y} term and the shared β_t schedule.
- Explain theoretically and empirically why δ>0 is needed, and provide guidance for choosing it across tasks.
- Discuss the practical implications of the κ dependence in the two theorems and whether the worse NC-PL dependence is believed to be inherent or proof-related.
- Expand discussion of assumptions, especially bounded gradients, and comment on whether clipping or other mechanisms could support analysis in more realistic deep-learning regimes.

# Merger Subscores
Novelty: 7.7
Technical soundness: 7.9
Empirical support: 6.5
Significance: 7.4
Clarity: 7.2

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 8.0, 6.0]
Average score: 5.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 6 ===

# Final Consolidated Review
Summary: This paper studies federated training of GNNs on geo-distributed graphs with cross-client edges. The proposed Swift-FedGNN framework mainly performs parallel local training and only periodically lets a randomly selected subset of clients conduct cross-client training, using a double-aggregation design through remote clients and a trusted server to reduce communication and memory overhead. The paper combines an algorithmic contribution, a convergence analysis showing an O(T^{-1/2}) rate to a neighborhood, and experiments on ogbn-products and Reddit showing substantially lower communication/sample cost and faster wall-clock convergence than the chosen baselines while maintaining accuracy close to full cross-client training. Overall, the paper addresses an important problem and has a solid core idea with meaningful practical promise, but the empirical validation and some claims are not yet fully commensurate with the strength of the presentation. The main issues are the lack of formal privacy guarantees despite privacy-oriented wording, the mismatch between the GCN-based theory and GraphSAGE experiments, limited robustness/statistical evaluation, and insufficient analysis of the key efficiency-accuracy tradeoffs induced by the hyperparameters controlling correction frequency and participating cross-client clients.
Strengths:
- Addresses a well-motivated and important setting: federated GNN training on geo-distributed graphs with cross-client dependencies, where naive FL or standard distributed GNN training is inadequate.
- The core algorithmic idea is sensible and practically relevant: mostly local training with periodic cross-client correction on only a sampled subset of clients directly targets the main systems bottleneck.
- The communication-saving design is technically interesting: remote clients aggregate neighbor information locally, then the server aggregates again before sending to the training client, which plausibly reduces transferred volume and avoids caching cross-client neighbor data at clients.
- Theoretical analysis goes beyond standard unbiased-gradient assumptions and explicitly handles biased gradients arising from neighbor sampling and missing cross-client neighbors; the layer-dependent error characterization is a useful insight.
- The convergence result achieves an O(T^{-1/2}) rate to a neighborhood, matching the typical rate for sampling-based GNN training despite the more difficult federated setting.
- Experiments on two reasonably large and distinct node-classification datasets show strong wall-clock efficiency gains and markedly lower communication overhead than the selected baselines, while accuracy remains close to the full cross-client training baseline.
- The paper is generally well organized, with clear motivation, algorithm descriptions, and a helpful comparison to the most related prior federated/distributed GNN methods.
Weaknesses:
- The paper makes privacy-leaning claims (e.g., that the design helps preserve privacy or avoids node information leakage), but provides no formal privacy analysis or guarantee. Aggregated embeddings may still leak information, especially for small neighborhoods or under stronger threat models. These claims should be substantially softened or formally justified.
- There is a notable theory/practice mismatch: the convergence analysis is developed for GCN, while the experiments use GraphSAGE. Since the proof relies on a GCN-style propagation formulation, the paper does not fully connect its theory to its empirical results.
- The empirical evaluation lacks statistical rigor: results are shown without multiple seeds, error bars, or variance measures, making it hard to judge robustness for a stochastic training algorithm.
- The key tradeoff parameters I and K are central to the method, and Figure 6 mainly studies computation vs. sampling/communication ratios; however, the paper does not systematically show the corresponding accuracy tradeoffs for these settings. That leaves the central efficiency-versus-information-loss claim only partially validated.
- Experiments are limited to a single-machine simulation with shared-memory communication and a fixed simulated bandwidth. This is useful as a first step but does not capture realistic federated issues such as latency, heterogeneity, stragglers, or deployment overheads.
- The evaluation uses METIS partitioning, which tends to minimize edge cuts. Since the method is specifically about cross-client edges, the paper would be stronger with experiments under more challenging partitionings or varying cross-client edge ratios.
- The experimental scope is somewhat narrow: only node classification is evaluated, and only a small set of baselines is included. The current baselines are relevant, but broader comparison to other federated GNN systems/benchmarks would strengthen the claims.
- Method limitations are present but underemphasized. In particular, the current aggregation/offloading design mainly supports element-wise aggregation; the paper notes in a footnote that architectures like GAT would require sending raw features/activations to the server, which weakens generality and potentially privacy.
Nice-to-haves:
- Add experiments with GCN to align with the theory, or extend the analysis to GraphSAGE if that is the intended practical architecture.
- Report mean and standard deviation across multiple random seeds for convergence curves and final metrics.
- Include accuracy/efficiency ablations for different values of correction frequency I, number of cross-client training clients K, and fanout, not only communication/computation ratios.
- Test robustness under different partition schemes, especially random or controlled partitionings with varying cross-client edge densities.
- Include an explicit local-only baseline in the main paper to quantify the value of periodic cross-client correction.
- Add a clearer limitations discussion covering the trusted-server assumption, absence of formal privacy guarantees, residual error in convergence, and restricted support for non-element-wise GNN layers.
Novel insights: The strongest insight is that biased gradient errors in federated sampling-based GNN training arise from two coupled sources—neighbor sampling and missing cross-client neighbors—and that these errors scale with GNN depth because of the interleaving of aggregation and nonlinear transformation across layers. This gives a principled explanation for why infrequent cross-client correction can still work well in shallow practical models while becoming more delicate for deeper GNNs. The double-aggregation communication pattern is also a useful systems idea: it is not just a communication optimization, but a design that trades exact cross-client visibility for compressed intermediate exchange.
Potentially missed related work:
- It may be useful to discuss other federated GNN system/benchmark papers such as FedGraphNN and SpreadGNN more explicitly, even if their assumptions differ from this paper's geo-distributed cross-client-edge setting.
- If space permits, the authors could briefly position the work relative to alternative ways of handling missing cross-client information, such as graph completion, missing-neighbor generation, or distillation-style approaches, where relevant.
Suggestions:
- Tone down privacy claims unless formal guarantees are added; alternatively, provide a concrete threat model and a formal privacy/leakage analysis.
- Resolve the GCN/GraphSAGE gap by either adding GCN experiments that directly validate the theory or extending the analysis to GraphSAGE-like aggregation.
- Run multiple trials and report variance/error bars for final accuracy and wall-clock convergence.
- Augment the ablation study with accuracy-versus-efficiency plots as I and K vary, since this is the main design tradeoff in Swift-FedGNN.
- Evaluate under additional partitioning strategies or controlled cross-client edge ratios to show when the method is most beneficial.
- Discuss the practical meaning of the residual term in Theorem 5.6 and, if possible, relate it empirically to graph partition quality or cross-client edge density.
- Expand the discussion of architectural applicability, especially the limitation to element-wise aggregation under the current privacy/communication design.
- If feasible, include a small-scale multi-machine or more realistic networked evaluation to corroborate the simulated communication savings.

# Merger Subscores
Novelty: 7.3
Technical soundness: 7.0
Empirical support: 6.5
Significance: 7.6
Clarity: 7.8

# Actual Human Scores
Individual reviewer scores: [3.0, 6.0, 5.0, 5.0]
Average score: 4.8
Binary outcome: Reject

=== CALIBRATION EXAMPLE 7 ===

# Final Consolidated Review
Summary: This paper proposes HiCA for open-vocabulary object detection, combining two ideas: hierarchical prompts that inject superclass-level semantics into prompt-based region classification, and context-aware calibration that uses clustered global image context to adjust class logits. The core idea is sensible and the empirical results are generally solid: on both OV-COCO and OV-LVIS, HiCA improves over the authors' reproduced OADP and BARON baselines, with especially strong gains on base/overall metrics and moderate gains on novel categories. The ablations also support that the two components contribute differently: hierarchical prompts mainly boost base performance, while context-aware calibration provides an additional improvement on novel classes. 

That said, the paper has important weaknesses that should be addressed. The biggest issue is reproducibility and methodological clarity around the superclass hierarchy: the paper relies heavily on superclasses and a subordinate matrix, but does not clearly specify how superclasses are defined for each dataset, how many are used, or provide the class-to-superclass mapping. The context-aware calibration mechanism is also under-specified in key details, including how contexts are assigned at inference and the exact DG-layer configuration. In addition, the paper's framing emphasizes improved generalization to novel classes, but the empirical gains are more pronounced on base classes than on novel ones, so the claims should be better calibrated. Overall, this is a worthwhile and empirically promising contribution, but one whose presentation and reproducibility currently lag behind the quality of the core idea.
Strengths:
- The paper addresses a real weakness in prior OVD methods: over-reliance on direct region-to-class alignment and underuse of contextual information.
- The hierarchical prompt idea is intuitive and practically relevant: injecting superclass-level semantics is a reasonable way to bias the model toward broader transferable structure rather than only base-class identities.
- The method is evaluated on two standard OVD benchmarks, OV-COCO and OV-LVIS, and shows consistent gains over reproduced OADP and BARON baselines.
- Empirical improvements are meaningful, especially on base/overall metrics; e.g., on OV-COCO the reproduced OADP baseline improves from 29.9 to 31.2 mAP_N and from 51.7 to 57.2 mAP_B, and BARON+HiCA reaches 36.0 mAP_N.
- The ablations are useful and substantiate the roles of the components: hierarchical prompts produce the main base-class gain, while context-aware calibration adds novel-class improvement.
- The paper studies prompt variants, the balance parameter λ, and context-clustering/DG-layer hyperparameters, which gives some insight into the behavior of the system.
- Applying HiCA on top of two different detector/distillation baselines supports the claim that the modules are reasonably modular within this family of methods.
Weaknesses:
- A key reproducibility issue is that the superclass taxonomy is not clearly defined. The paper does not specify how superclasses are chosen, how many are used per dataset, or provide the class-to-superclass mapping needed to construct the subordinate matrix A.
- The context-aware calibration module is under-specified in important details. The exact DG-layer architecture is only described as an MLP, and the paper does not clearly explain the inference-time context assignment mechanism (e.g., nearest-cluster vs. soft assignment).
- The paper's main narrative stresses improving generalization to novel classes, but the gains are substantially larger on base classes than on novel ones in the main ablations. The method does help novel classes, but the evidence supports a more balanced claim than the current framing suggests.
- The benefit of hierarchical prompts for novel-class discrimination is only partial. The appendix explicitly shows that the similarity matrix is not strongly diagonal for novel classes, which is consistent with the relatively modest novel-class gain from hierarchical prompts alone.
- There is no ablation for context-aware calibration alone, so its standalone value is not isolated independently of hierarchical prompts.
- The method introduces nontrivial extra design choices and hyperparameters—superclass definitions, cluster count K, DG-layer depth, λ, and context assignment—without enough discussion of robustness or failure modes.
- Claims of 'state-of-the-art' should be phrased more carefully because the strongest improvements are demonstrated relative to reproduced baselines; the comparison to previously reported numbers is less clean than the wording suggests.
Nice-to-haves:
- Report per-class AP for novel categories, especially on OV-COCO, to show which classes benefit from hierarchical prompts versus context calibration.
- Include runtime, memory, or FLOPs overhead for hierarchical prompts and context-aware calibration separately.
- Provide qualitative detection examples showing when context-aware calibration changes a prediction in a useful way.
- Add failure-case analysis, especially for cases where superclass information or context is misleading.
- Evaluate sensitivity to alternative superclass hierarchies or granularity choices.
Novel insights: A useful insight from the paper is that the two proposed components appear to improve different aspects of OVD: hierarchical prompts mainly strengthen base/overall recognition by imposing coarse semantic structure, while context-aware calibration partially offsets the base-class bias and recovers some novel-class performance. This decomposition is valuable because it suggests that better open-vocabulary detectors may benefit from explicitly separating transferable semantic structure from context-dependent posterior correction, rather than treating prompting as a single monolithic mechanism.
Potentially missed related work:
- It may be useful to discuss stronger grounding/open-vocabulary detection systems such as GLIP and Grounding DINO for broader positioning, even if they are not direct like-for-like baselines.
- Related hierarchical or taxonomy-aware detection/classification literature could help position the superclass-based design more precisely.
- Work using richer language descriptions or LLM-generated prompts for open-vocabulary recognition may be relevant as an optional comparison point to distinguish hierarchy from simply adding more textual semantics.
Suggestions:
- Explicitly define the superclass taxonomy for each dataset, including the number of superclasses and full class-to-superclass mapping, and clarify whether this is manually defined or derived from existing ontology/meta-data.
- Specify the context-aware calibration pipeline more precisely: DG-layer architecture, hidden dimensions, activations, clustering details, and how the context index/vector is selected at inference.
- Temper the claims around novel-class generalization, or strengthen them with deeper analysis such as per-class AP, superclass coverage of novel classes, or cross-dataset generalization tests.
- Add an ablation for CA alone to isolate its contribution without hierarchical prompts.
- Analyze robustness to alternative superclass definitions or softer superclass-category relations, especially since the discussion claims superclass-category similarity can replace the binary subordinate matrix.
- Include variance over multiple seeds, particularly for K-means-based context clustering, to show stability.
- Provide qualitative visualizations of baseline vs. HiCA predictions in ambiguous-context cases to make the mechanism more interpretable.

# Merger Subscores
Novelty: 6.7
Technical soundness: 6.4
Empirical support: 7.2
Significance: 6.8
Clarity: 5.8

# Actual Human Scores
Individual reviewer scores: [5.0, 5.0, 5.0, 5.0]
Average score: 5.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 8 ===

# Final Consolidated Review
Summary: This paper studies limitations of unlabeled OOD detection through an information-theoretic lens. Its central claim is that when the surrogate objective used by SSL/unsupervised learning is independent of the in-distribution semantic labels, the learned representation can become “label blind,” leading to failure on OOD detection defined over unseen labels. The paper also proposes an Adjacent OOD benchmark, created by partitioning classes within the same dataset so that ID/OOD samples have substantial visual overlap, and evaluates several unlabeled, unsupervised, and zero-shot methods on this setting. The benchmark contribution is valuable and the empirical results consistently show that commonly used unlabeled methods can degrade severely in this more safety-relevant regime, often trailing a simple supervised MSP baseline. The main caveat is that the theoretical guarantees rely on strong independence/minimality assumptions, and the bridge from the strict theorem to realistic approximate settings is suggestive rather than fully established. Overall, the paper makes an important cautionary point and introduces a useful evaluation setting, but some claims should be narrowed to better match what is actually proved and demonstrated.
Strengths:
- The Adjacent OOD benchmark is a meaningful contribution that targets a realistic and underexplored failure mode: OOD samples that are semantically novel yet visually close to ID data because they come from the same dataset/class universe.
- The paper identifies a genuine gap in standard OOD benchmarking, where using disjoint datasets can let unlabeled methods succeed via non-semantic cues rather than label-relevant features.
- The information-theoretic framing is coherent, and the theorem/lemma/corollary progression makes the core failure mode conceptually clear under the stated assumptions.
- The empirical study covers multiple method families: self-supervised representation methods, an unsupervised diffusion-based method, and zero-shot CLIP-based methods.
- Results on Faces, Cars, and Food are consistent with the paper’s main cautionary message: several unlabeled methods are near random or substantially below the supervised MSP baseline on Adjacent OOD.
- The paper is constructive in not only criticizing existing unlabeled OOD settings but also proposing a benchmark and discussing implications for future label-efficient methods.
- The paper usefully highlights heterogeneity across datasets, including cases in the appendix where SSL performs better, which suggests the phenomenon depends on alignment between surrogate objectives and semantic labels.
Weaknesses:
- The theoretical guarantees depend on strong assumptions, especially the decomposition x=(x1,x2) into independent components and optimization to a minimal sufficient representation. These assumptions make the strict failure theorem clean, but they are restrictive and not verified for the real datasets used in experiments.
- The most rigorous theorem addresses the strict independence case (zero mutual information), which is an extreme setting. The extension to practical “approximate label blindness” relies on Fano-style intuition, but the paper does not provide a quantitative characterization linking small mutual information to actual OOD detection degradation.
- The empirical section demonstrates failures on Adjacent OOD but does not directly test whether the theorem’s conditions hold in practice, e.g., by estimating representation-label mutual information or otherwise measuring alignment between surrogate-task features and semantic labels.
- Some claims are stronger than what the theory supports. In particular, statements implying that one generally cannot ignore labels for OOD detection, or that no generally applicable unlabeled OOD detector can exist, should be more carefully scoped to the stated assumptions and worst-case setting.
- The paper does not sufficiently analyze cases where unlabeled methods do better, such as the CIFAR appendix results. Those results are important because they indicate that SSL can sometimes retain label-relevant information even in adjacent splits, and understanding why would sharpen the contribution.
- The experimental comparison omits informative hybrid baselines that would help isolate the role of labels, such as supervised contrastive or few-label/semi-supervised approaches, which are especially relevant given the paper’s discussion section.
- The analysis of CLIPN performance is plausible but largely qualitative; the pretraining-alignment explanation is not quantitatively substantiated.
Nice-to-haves:
- Include controlled experiments where the degree of overlap/independence between surrogate-task information and semantic labels is varied synthetically, to directly test how Adjacent OOD performance degrades.
- Add hybrid baselines using limited label information, such as supervised contrastive learning or few-shot/semi-supervised variants, to make the paper more actionable.
- Measure representation-label alignment empirically, for example via linear probing, MI proxies, or other diagnostics, to connect the theory to the observed failures.
- Analyze the CIFAR adjacent-OOD successes more deeply to identify dataset properties under which unlabeled OOD remains viable.
- Provide sensitivity analyses for the ID/OOD class split ratio and perhaps model architecture/capacity to assess robustness of the empirical conclusions.
- Quantify the CLIP pretraining-alignment hypothesis rather than only illustrating it qualitatively.
Novel insights: The paper’s most compelling insight is not merely that unlabeled OOD can fail, but that standard OOD benchmarks may systematically mask this failure by making ID and OOD too separable through dataset-level or style-level cues. The Adjacent OOD formulation reframes evaluation around semantic adjacency rather than dataset dissimilarity, which is a useful conceptual contribution for safety-oriented OOD work.
Potentially missed related work:
- None
Suggestions:
- Temper the strongest claims so they precisely reflect the proved setting: guaranteed failure under specific independence/minimality assumptions, plus empirical evidence that current unlabeled methods often struggle on adjacent OOD.
- Strengthen the bridge between theory and practice by adding empirical diagnostics of label relevance in learned representations.
- Add at least one label-efficient hybrid baseline to test the paper’s own hypothesis that a small amount of label information may overcome approximate label blindness.
- Expand the discussion of positive cases (e.g., CIFAR) to clarify when SSL objectives may align well enough with semantic labels to avoid failure.
- If possible, include a controlled synthetic study varying mutual-information alignment between surrogate and semantic labels, which would make the approximate-label-blindness claim much more convincing.
- Substantiate the CLIPN pretraining-alignment explanation with quantitative evidence if that comparison remains central.

# Merger Subscores
Novelty: 7.8
Technical soundness: 6.6
Empirical support: 7.0
Significance: 7.5
Clarity: 7.1

# Actual Human Scores
Individual reviewer scores: [6.0, 5.0, 8.0]
Average score: 6.3
Binary outcome: Accept

=== CALIBRATION EXAMPLE 9 ===

# Final Consolidated Review
Summary: This paper presents LocoVR, a VR-collected indoor locomotion dataset with 7,071 two-person trajectories across 131 home-like scenes, together with benchmark tasks for global path prediction, short-horizon trajectory prediction, and goal prediction. The dataset’s main contribution is its combination of scene diversity, precise geometry, and paired human motion in confined indoor environments. The empirical results are generally convincing that models trained on LocoVR transfer better than those trained on GIMO or THOR-MAGNI to the authors’ real-world test set and to an additional GIMO-based split, and the paper also includes useful ablations on dataset scale and input features. The strongest evidence supports LocoVR as a valuable resource for geometry-aware indoor locomotion learning with some social interaction content. The weaker part of the paper is the strength of its claims about proxemics/social behavior learning and VR-to-real validity: these are motivated well and illustrated qualitatively, but not quantified as rigorously as the geometry/generalization claims.
Strengths:
- Strong dataset contribution: 131 indoor scenes and 7,071 trajectories provide substantially greater scene diversity than prior real-motion indoor datasets discussed in the paper.
- The two-person collection setup is meaningful for indoor/home settings and captures interaction-relevant trajectories that are absent from single-person datasets.
- The paper provides a fairly comprehensive benchmark suite spanning three practical tasks: global path prediction, trajectory prediction, and goal prediction.
- Empirical results consistently show that training on LocoVR outperforms training on GIMO and THOR-MAGNI on the authors’ real-world LocoReal test set, and additional appendix experiments on GIMO also support better cross-scene generalization.
- Useful ablations analyze the role of dataset scale, second-person input, and heading information, helping support that the dataset’s diversity and multi-person signals matter.
- The paper is generally clear about the data collection pipeline, hardware, scene source, preprocessing, and benchmark setup.
- The authors explicitly acknowledge important limitations, including demographic homogeneity and the possible VR/real gap.
Weaknesses:
- The central claims about learning socially-aware/proxemic behavior are only partially supported. The paper provides qualitative examples and proximity statistics, but lacks dedicated quantitative social metrics such as collision rate, interpersonal-distance violations, yielding/pass-side accuracy, or time-to-collision outcomes.
- The VR-to-real transfer claim is promising but still limited by the evaluation design. LocoReal is relatively small (450 trajectories, 4 layouts, 5 participants), and the paper does not directly quantify behavioral differences between VR and real settings such as speed distributions, path efficiency, or interpersonal spacing.
- Baseline coverage is narrower than desirable for a dataset/benchmark paper. The evaluation uses U-Net and Ynet variants, but omits stronger modern social trajectory predictors, so it is hard to tell how much of the gains would persist across a broader model family.
- Because LocoVR has much greater scene diversity than the compared datasets, the experiments mainly demonstrate the value of scale/diversity rather than isolating whether VR collection itself yields better data quality. The included scale ablations help, but do not fully disentangle scene count, trajectory count, and collection modality.
- The dataset is restricted to two-person interactions. This is reasonably motivated for many home scenarios, but it limits applicability to richer multi-person indoor coordination settings.
- Participant diversity is limited to 32 able-bodied adults from one collection context, which constrains how broadly claims about human locomotion and proxemics should be interpreted.
Nice-to-haves:
- More detail in the main paper on scene selection from HM3D and scene-type composition (e.g., kitchens, living rooms, bedrooms).
- A compact taxonomy of social behaviors present in the dataset with frequencies (e.g., yielding, detouring, narrow-passage passing, following).
- Public release of trained baseline checkpoints in addition to data and evaluation code.
- More explicit failure-case analysis for the best-performing models.
Novel insights: The paper’s most credible insight is that, for indoor locomotion learning, scene diversity appears to matter at least as much as raw trajectory count. This is reflected by LocoVR outperforming THOR-MAGNI despite THOR-MAGNI having many trajectories but only four scenes, and by the authors’ own reduced-scale ablations. A second useful insight is that two-person room-scale data can expose behaviors that are not well captured by single-person indoor motion datasets, even without going to full crowd settings. However, the paper stops short of converting these observations into a rigorous quantitative account of social navigation behavior.
Potentially missed related work:
- None
Suggestions:
- Add quantitative social-navigation metrics to directly test the paper’s proxemics claims, such as collision/intersection rate, minimum-distance violations, yielding accuracy in narrow passages, or time-to-collision statistics.
- Strengthen the VR-to-real analysis with matched statistics between LocoVR and LocoReal (walking speed, path efficiency, clearance from obstacles, interpersonal distance), ideally in paired or closely matched layouts.
- Expand benchmark coverage with stronger recent social trajectory prediction baselines, even on a subset of tasks, to make the benchmark more broadly useful and convincing.
- Clarify the scope of claims: the paper strongly supports LocoVR as a scalable dataset for indoor geometry-aware and interaction-aware locomotion, but claims about proxemics and broad human generalization should be phrased more carefully unless supported by dedicated metrics.
- If space permits, move some key appendix material into the main paper, especially the rationale for the two-person setting and the statistics showing proximity/interaction frequency.
- In future versions, consider adding more diverse participants and, if feasible, more than two-person scenarios to improve ecological coverage for home and service-robot applications.

# Merger Subscores
Novelty: 7.8
Technical soundness: 7.2
Empirical support: 7.0
Significance: 7.6
Clarity: 8.0

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0, 6.0, 3.0]
Average score: 5.8
Binary outcome: Accept

=== CALIBRATION EXAMPLE 10 ===

# Final Consolidated Review
Summary: The paper proposes Fine-tuned Score Deviation (FSD), a simple and interesting method for pretraining data detection in LLMs. The central idea is to fine-tune the target model on a small set of domain-matched non-member examples and then use the change in an existing score (e.g., perplexity or Min-k%) between the original and fine-tuned model as the membership signal. The empirical observation motivating the method—that such fine-tuning tends to reduce scores more for non-members than for members—is novel and clearly presented. Empirically, the paper shows large gains over basic likelihood-based baselines on WikiMIA, ArXivTection, BookMIA, and BookTection, and includes several useful ablations on auxiliary data size, model size, and PEFT method. However, the paper’s broader claims are limited by several substantive issues: the method depends strongly on access to domain-matched non-members; it shows clear failure under domain mismatch; evaluation against stronger recent baselines is missing; and the experimental presentation lacks variance/statistical significance reporting. The paper also partly addresses temporal/distribution artifacts in WikiMIA, but the mitigated results are notably weaker, so the extent to which gains reflect true membership signals versus benchmark artifacts remains somewhat unresolved.
Strengths:
- The core insight is creative and potentially impactful: using score shifts induced by fine-tuning on non-members to amplify separability between members and non-members is a meaningful departure from standard one-shot scoring.
- The method is simple, easy to understand, and compatible with multiple existing scoring functions, which increases its practical appeal.
- Empirical gains over the included baselines are often large on key benchmarks. For example, Table 1 shows substantial AUC improvements across multiple models on WikiMIA and ArXivTection.
- The paper includes a fairly extensive empirical study: multiple model families, several datasets, four baseline scoring rules, and ablations on fine-tuning data size, model size, PEFT method, and some hyperparameters.
- The data-efficiency story is compelling: strong improvements are reported with only tens to hundreds of auxiliary non-member examples.
- The paper does acknowledge important limitations and includes a useful analysis of temporal shift artifacts in WikiMIA rather than ignoring them.
- The appendix contains informative additional analyses, including different fine-tuning methods and experiments on mixed-domain auxiliary data.
Weaknesses:
- A key practical limitation is the requirement for domain-matched non-member data. The paper shows this dependence is real: when the fine-tuning domain is unrelated to the test domain, performance can collapse or even degrade (Table 16, ArXiv evaluated with Wiki fine-tuning). This substantially narrows the generality of the method.
- The evaluation does not compare against several stronger recent pretraining-data/MIA baselines that are already cited or highly relevant, such as Min-k%++, neighborhood-based methods, and RECALL/reference-style approaches. As a result, it is unclear how competitive FSD is relative to the current stronger literature rather than only classic scoring baselines.
- The paper reports only point estimates, without variance across random seeds, confidence intervals, or significance analysis. Given random sampling of auxiliary data and stochastic fine-tuning, this weakens confidence in the exact magnitude and consistency of the gains.
- The temporal-shift concern is only partially resolved. Although the authors run deletion/replacement controls, performance drops noticeably after mitigation (e.g., Table 6), leaving open how much of the headline performance on datasets like WikiMIA may still benefit from residual benchmark artifacts.
- There is no comparison to a natural baseline that applies the scoring function directly on the fine-tuned model alone, without taking a deviation. That control would help isolate whether the gain truly comes from the deviation construction versus simply adapting the model to the target-domain non-members.
- The setup for choosing the decision threshold relies on a validation set with membership labels, but the practical construction of such a validation procedure in realistic deployment is not fully clarified.
- The paper gives little mechanistic or theoretical explanation for why fine-tuning on non-members should consistently induce larger score decreases for non-members than for members. The phenomenon is interesting, but currently mostly empirical.
Nice-to-haves:
- A computational cost analysis comparing FSD to plain scoring baselines would help quantify the trade-off introduced by LoRA fine-tuning.
- A robustness study with partially contaminated auxiliary fine-tuning data would strengthen claims of practical applicability.
- Qualitative failure-case analysis or ROC curves with confidence intervals would improve interpretability.
- More systematic study of domain similarity requirements could make the method more actionable in practice.
Novel insights: The strongest aspect of the paper is not just the empirical improvement, but the underlying observation that a small amount of post-release, domain-matched non-member data can act as a kind of contrastive probe: after adaptation, genuinely unseen examples in that domain move more than pretraining members under standard likelihood-based scores. This is a useful conceptual bridge between parameter-efficient adaptation and membership inference, and it suggests a broader perspective in which membership detection can be improved not only by designing better static scores, but by intentionally perturbing the model and measuring response asymmetries.
Potentially missed related work:
- Min-k%++ (Jingyang Zhang et al., 2024) as a stronger scoring-based baseline.
- Neighborhood comparison methods for LM membership inference, e.g., Mattern et al. (ACL Findings 2023).
- RECALL / relative conditional log-likelihood based approaches, e.g., Xie et al. (2024).
- More generally, reference-model or auxiliary-model based MIA methods could be discussed as potential comparison points, even if not directly identical in setting.
Suggestions:
- Add comparisons to stronger recent baselines, especially Min-k%++, neighborhood-based methods, and RECALL/reference-style methods, to better establish the method’s standing relative to current practice.
- Include a control baseline using the fine-tuned model’s score directly, without deviation, to verify that the deviation itself is the key ingredient.
- Report mean and variance over multiple random seeds for data sampling and fine-tuning, or provide confidence intervals/statistical tests for the main tables.
- Strengthen the discussion of domain dependence and failure under domain mismatch; this limitation should be more prominent in the main text, not only the appendix.
- Further reduce concerns about benchmark artifacts by using more artifact-resistant benchmark construction or better matching of member/non-member distributions beyond timestamp deletion/replacement.
- Provide more guidance on threshold selection in realistic scenarios where labeled validation membership data may be unavailable.
- If possible, add a mechanistic analysis explaining why the asymmetric score shift occurs, even if only empirically via training dynamics or representation analysis.

# Merger Subscores
Novelty: 7.9
Technical soundness: 6.9
Empirical support: 7.0
Significance: 7.4
Clarity: 8.2

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 6.0, 5.0]
Average score: 6.2
Binary outcome: Accept

=== CALIBRATION EXAMPLE 11 ===

# Final Consolidated Review
Summary: The paper proposes Process Advantage Verifiers (PAVs), arguing that step-level process rewards for reasoning should measure progress—formalized as advantage—under a prover policy distinct from the base policy, rather than raw future-success values under the base policy. The paper combines a conceptual argument, a tabular/NPG theoretical analysis of when a prover is helpful, and empirical results on Gemma 2B/9B/27B for MATH. The central empirical findings are substantial: compared to ORM-based reranking, PAV-guided search improves accuracy by roughly 8–10% and compute efficiency by 1.5–5×; in online RL, PAV rewards yield around 6× better sample efficiency and >7% accuracy gains over ORM-only RL. Overall, the core idea is novel and important, and the empirical gains appear meaningful. The main limitations are a noticeable theory-to-practice gap, limited evaluation breadth beyond MATH, and incomplete treatment of total training cost / robustness considerations in the main narrative.
Strengths:
- Novel and well-motivated core idea: defining process rewards as advantages (progress) rather than absolute Q-values is conceptually strong and directly addresses why prior automated PRMs often underperform.
- Clear insight that the prover should be distinct from the base policy, with an interesting and non-obvious conclusion that an intermediate-strength or even weaker-but-complementary prover can be more useful than a very strong one.
- Theoretical analysis provides a useful characterization of good provers in terms of distinguishability and alignment, and helps explain why both too-weak and too-strong provers can fail.
- Empirical results are strong on the target setting: PAVs improve test-time search accuracy and compute efficiency over ORM baselines, and the RL results are especially notable given prior reports of only modest PRM gains.
- Good ablations on prover choice support the claimed mechanism: BoK provers with moderate K work best, and stronger provers can indeed become less informative.
- The paper includes informative mechanistic discussion and examples, including why Q-based rewards can induce degenerate behavior and why advantage-based rewards better support exploration.
- Reproducibility is reasonably strong: the appendices contain substantial implementation detail, confidence-interval methodology, hyperparameter discussion, and extra analyses.
Weaknesses:
- There is a real theory-practice gap. The main theoretical results analyze an idealized tabular/NPG setting with oracle values and small-step assumptions, while the practical RL experiments use learned verifiers and REINFORCE-style training with tuned α values. The paper explains the intuition but does not fully connect the formal guarantees to the implemented algorithm.
- Evaluation is limited to a single task family/dataset (MATH). This makes it hard to assess whether the benefits of PAVs generalize to other reasoning domains such as GSM8K, code generation, or broader multi-step reasoning tasks.
- The weighting parameter α is important and tuned separately for search and RL, but the paper offers limited principled guidance for selecting it beyond validation-based search. The claimed robustness is encouraging but not fully systematized.
- The practical prover-selection story is still heuristic. Although the theory characterizes helpful provers via distinguishability/alignment, the actual choice in experiments relies mainly on Best-of-K heuristics and empirical tuning rather than an operational metric derived from the theory.
- Total cost tradeoffs are somewhat underemphasized in the main presentation. The appendices acknowledge that PAV training requires substantially more data/compute than ORM training, but this upfront cost is important for practical adoption and should be surfaced more clearly alongside the test-time and RL efficiency claims.
- Main-figure statistical presentation could be stronger. Confidence intervals are described in the appendix, but the main plots would be easier to evaluate if uncertainty were directly visualized.
Nice-to-haves:
- Add direct uncertainty bars or confidence intervals to the main figures.
- Include a compact table of symbols / notation to improve accessibility.
- Move some practically important appendix content (e.g., training-cost discussion, exploration baseline comparison) into the main paper.
- Provide more qualitative case studies of beam-search decisions and PAV failure modes on real MATH examples.
Novel insights: The strongest insight is not merely that dense rewards help, but that the right dense reward should capture stepwise progress under a complementary policy, not estimated eventual success under the current policy. This reframes automated PRMs from 'is this prefix promising under me?' to 'did this step improve solvability under a useful prover?'. The paper also highlights an important asymmetry: a stronger prover is not necessarily better, because overly capable provers erase the contrast needed for credit assignment. That is a valuable conceptual contribution likely to influence future work on process supervision and search-guided RL.
Potentially missed related work:
- None
Suggestions:
- Strengthen the theory-practice connection by discussing more explicitly how the NPG/oracle analysis should be interpreted for learned-verifier REINFORCE training, and if possible add controlled experiments that vary verifier error or approximate the theoretical quantities.
- Broaden the empirical evaluation to at least one additional reasoning domain beyond MATH to test generality.
- Provide a more operational recipe for choosing provers, ideally using empirical estimates of the distinguishability/alignment quantities suggested by the theory rather than only Best-of-K heuristics.
- Expand the analysis of α, either with stronger empirical sensitivity studies or with heuristic guidance tied to prover/base alignment.
- Report joint training+inference compute tradeoffs more prominently in the main paper, since PAVs have higher verifier-training cost than ORMs.
- Add more analysis of failure modes for PAVs themselves, not only for Q-based alternatives.

# Merger Subscores
Novelty: 8.7
Technical soundness: 7.9
Empirical support: 8.0
Significance: 8.5
Clarity: 7.8

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 8.0, 8.0, 6.0]
Average score: 7.1
Binary outcome: Accept

=== CALIBRATION EXAMPLE 12 ===

# Final Consolidated Review
Summary: This paper introduces the task of audio difference explanation: generating natural-language descriptions of how two audio clips differ. It contributes two benchmark datasets derived from AudioCaps and Clotho with three explanation tiers, and proposes ADIFF, a prefix-tuning style model with a separator/cross-projection module, position-captioning, and staged training. The paper’s strongest contribution is the task/dataset formulation and the fairly extensive ablation study. Empirically, ADIFF generally improves over the paper’s naive baseline and is competitive with or better than Qwen-Audio in many settings, though not uniformly (notably CLD Tier-2). Overall, this is a novel and interesting benchmark paper with a reasonable model contribution, but the main limitations are evaluation rigor and data quality: most training targets are LLM-generated rather than human-authored, statistical uncertainty is not reported, human evaluation is relatively limited, and some claims are stronger than what the evidence fully supports.
Strengths:
- The task itself is genuinely novel and well motivated. Comparing two audios and explaining their differences is a meaningful benchmark for comparative audio-language reasoning, with plausible applications in forensics, quality assessment, and editing workflows.
- The dataset design is thoughtful. The three-tier structure (concise / brief / detailed) is useful both conceptually and analytically, and the paper provides substantial dataset statistics for ACD and CLD.
- The paper does more than introduce a benchmark: it also provides a baseline family and a stronger model, which makes the contribution more actionable for the community.
- ADIFF shows consistent improvements over the naive baseline across datasets and tiers, and often outperforms finetuned Qwen-Audio variants, while the paper also honestly reports exceptions such as CLD Tier-2.
- The ablation section is relatively comprehensive: cross-projection, language-model scale, position captioning, stage-3 finetuning, and language-only effects are all studied.
- The language-only baseline is a useful inclusion, since it highlights that some tiers may be easier due to linguistic regularities rather than true audio grounding.
- Human evaluation is included in addition to captioning metrics, which is important for such an open-ended generation task.
- The paper is fairly clear about practical limitations such as compute constraints and the fact that only the test set explanations are human-verified.
Weaknesses:
- A central limitation is dataset supervision quality: the difference explanations are largely generated by an LLM from existing captions rather than directly authored by humans from listening to the audio. Only the test set is human-verified. This raises a real concern that the model may learn to mimic caption-derived textual contrasts rather than genuine human perception of audio differences.
- Evaluation rigor is weaker than it should be for the strength of the claims. Main quantitative tables report point estimates only, without multiple seeds, confidence intervals, or statistical significance testing.
- Human evaluation is helpful but limited: only five annotators are used, no inter-annotator agreement is reported, and variance/uncertainty measures are absent.
- The paper sometimes overstates its empirical conclusions. ADIFF is not uniformly better than all baselines in all settings; in particular, Qwen-Audio performs better on CLD Tier-2. Claims of broad or 'significant' superiority should be phrased more carefully unless statistically substantiated.
- The objective metrics are inherited from captioning and may not fully capture correctness of comparative explanations. The paper acknowledges metric limitations, but the mismatch remains a substantive issue for this task.
- The hallucination discussion is interesting but under-evaluated. The paper proposes using HTSAT event probabilities as a diagnostic tool, yet provides only anecdotal examples and no systematic quantification of hallucination frequency or detection effectiveness.
- Some evidence for the specific mechanism of cross-projection is suggestive rather than conclusive. The probing analysis based on nearest GPT-2 vocabulary matches is interesting, but remains somewhat speculative.
- The paper would be stronger with stronger generalization analysis, e.g., cross-dataset transfer or more systematic breakdowns by audio type or difficulty.
Nice-to-haves:
- Cross-dataset generalization experiments (train on ACD, test on CLD and vice versa).
- A human-authored subset of difference explanations for stronger validation of the task and to calibrate ceiling performance.
- Error analysis broken down by audio category (speech, music, environmental sounds; similar vs dissimilar pairs).
- Quantitative hallucination analysis and evaluation of the proposed hallucination-detection signal.
- More uncertainty reporting for both automatic and human evaluations.
- A clearer in-main-text positioning against prior audio difference captioning/change-captioning work, even though the appendix discussion is reasonable.
Novel insights: A useful insight from the paper is that the task decomposes into tiers with different failure modes: Tier-2 appears easier to match linguistically, while Tier-3 better exposes the difficulty of producing grounded, detailed comparative descriptions. The ablations also provide a practically relevant observation that, under limited compute and fixed token budgets, smaller decoder LMs can outperform larger ones in prefix-tuning style audio-language setups. This is a worthwhile empirical takeaway even beyond this specific task.
Potentially missed related work:
- As a suggestion rather than a criticism: the main-paper discussion could more prominently connect to prior audio difference captioning/change-captioning style works already cited in the appendix, especially Takeuchi et al. (2023) and Komatsu et al. (2024), to clarify the distinction from this task earlier and more visibly.
- As a suggestion: additional discussion of how ideas from image difference captioning transfer—or do not transfer—to audio would strengthen positioning, since such work is already partially cited.
Suggestions:
- Temper claims of superiority and significance unless supported by repeated runs or statistical testing.
- Add uncertainty estimates: multiple seeds for main automatic metrics, and confidence intervals or standard deviations for human ratings.
- Report inter-annotator agreement for the human study and provide more detail on sample size per condition.
- Validate the dataset more directly by collecting human-written explanations on at least a representative subset, then comparing them with the LLM-generated references.
- Include more systematic analysis of hallucinations, not just examples, and evaluate whether HTSAT-based diagnostics actually help identify incorrect generations.
- Clarify the necessity of cross-projection with stronger evidence, since gains after full finetuning appear smaller than in the frozen-LM setting.
- Expand evaluation to test generalization across datasets or audio domains.
- In the main text, foreground the distinction from prior audio difference/change-captioning work rather than leaving most of that positioning to the appendix.

# Merger Subscores
Novelty: 8.4
Technical soundness: 6.4
Empirical support: 6.1
Significance: 7.6
Clarity: 7.3

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0, 8.0]
Average score: 7.5
Binary outcome: Accept

=== CALIBRATION EXAMPLE 13 ===

# Final Consolidated Review
Summary: This paper presents SAM 2, a unified promptable segmentation model for images and videos, together with the Promptable Visual Segmentation task, a large-scale model-in-the-loop data engine, and the SA-V dataset. The work is ambitious and impactful: the dataset scale is substantial, the evaluation is unusually broad, and the model shows strong zero-shot performance across interactive video segmentation, semi-supervised VOS, and image segmentation. After filtering reviewer noise, the main concerns are not about whether the contribution is real—it clearly is—but about reproducibility and analysis depth: part of the training recipe depends on an unreleased internal dataset; some practically important system aspects such as memory/computation scaling, long-video behavior, and multi-object efficiency are under-analyzed; and failure modes could be characterized more systematically. The architectural ideas are more of a strong integration and scaling of known ingredients than a radically new modeling paradigm, but the overall paper makes a significant contribution through unification, data, and strong empirical validation.
Strengths:
- A major dataset contribution: SA-V is extremely large relative to prior public VOS datasets, and the paper provides a thoughtful data engine, annotation protocol, quality verification process, and dataset/model cards.
- Strong empirical support across a broad set of evaluations: the paper reports interactive promptable video segmentation on 9 dense zero-shot datasets, semi-supervised VOS on 17 datasets, image segmentation on 37 datasets, plus comparisons on established VOS benchmarks such as DAVIS, MOSE, LVOS, and YTVOS.
- The unified PVS formulation is compelling: it cleanly subsumes image promptable segmentation and first-frame-mask VOS, while enabling prompts on arbitrary frames and iterative refinement.
- The model appears practically effective: SAM 2 substantially outperforms the SAM+tracker baselines for interactive video segmentation and is competitive or state of the art on conventional VOS benchmarks, while also improving over SAM on image segmentation and reporting much higher throughput.
- The appendix contains meaningful ablations on data mixtures, scaling, memory design, positional encoding, and architecture capacity, which help support several design choices.
- The paper includes limitations, fairness evaluation, model/data cards, and release commitments, which improve transparency and likely downstream usefulness.
Weaknesses:
- Full reproducibility is limited because training uses a sizable unreleased internal dataset in addition to SA-V and SA-1B. Although the paper does provide some ablations showing strong performance from released data mixtures, the headline model relies in part on non-public data.
- Some practically important system analyses are missing or too limited: the paper reports FPS, but gives little characterization of inference memory footprint, scaling with video length, or compute/memory behavior as the memory bank grows in long videos.
- The paper acknowledges failure modes such as shot changes, long occlusions, crowded scenes, and similar-looking objects, but does not provide a sufficiently systematic quantitative breakdown of these failures or of performance versus video length.
- SAM 2 processes objects independently apart from shared image features. This design is simple and effective, but the paper does not analyze the computational trade-off or degradation as the number of simultaneously tracked objects increases.
- Architectural novelty is moderate relative to the scale of the contribution: the model is a strong unification of existing ingredients such as memory banks, cross-attention, and promptable decoding, rather than a fundamentally new video modeling primitive.
- A few key implementation details that matter for understanding the memory design are deferred to the appendix rather than made prominent in the main paper.
Nice-to-haves:
- Report inference GPU memory usage and scaling with video length / number of stored memories, not just FPS.
- Add explicit evaluation or plots of performance versus video length, especially for long-video regimes beyond the aggregate LVOS results.
- Include multi-object scaling experiments showing runtime/memory and accuracy as the number of tracked objects increases.
- Provide a more systematic failure taxonomy with quantitative frequencies and representative examples.
- Surface a few memory-bank hyperparameters and design choices more clearly in the main text.
Novel insights: The strongest insight of the paper is that the main advance comes from co-design across task, model, and data rather than from a single architectural trick. The work makes a convincing case that 'segment anything in videos' requires not just adapting SAM with a tracker, but changing the interaction paradigm (PVS), building a streaming memory-equipped model that can refine across frames, and collecting data targeted at open-world video segments including parts and reappearing objects. Another important insight is that this video-centric redesign does not merely preserve image performance; with the chosen encoder and training mix, it can improve image segmentation speed and accuracy as well. The paper also offers evidence that SA-V has value beyond SAM 2 itself, via the Cutie+SA-V experiment, suggesting the data contribution is genuinely broader than a single model.
Potentially missed related work:
- None
Suggestions:
- Clarify more prominently what performance is achievable using only publicly released training data (e.g., SA-V + SA-1B), and position the released model relative to that setting.
- Add quantitative analysis of performance as a function of video length, occlusion duration, and number of tracked objects.
- Report inference memory footprint and resource scaling, especially for long videos and multi-object use cases, to better substantiate deployment claims around real-time streaming.
- Expand failure mode analysis with category-wise breakdowns (shot changes, similar objects, fine structures, crowded scenes, long occlusions) and representative qualitative cases.
- If space permits, provide more insight into memory behavior, such as which memories are most used and how prompted-frame memories versus recent-frame memories contribute.
- For the interactive baselines, make configuration choices especially explicit in the main paper so readers can easily assess fairness of the SAM+XMem++ and SAM+Cutie comparisons.

# Merger Subscores
Novelty: 7.6
Technical soundness: 8.6
Empirical support: 9.3
Significance: 9.4
Clarity: 8.4

# Actual Human Scores
Individual reviewer scores: [10.0, 8.0, 8.0, 10.0]
Average score: 9.0
Binary outcome: Accept

=== CALIBRATION EXAMPLE 14 ===

# Final Consolidated Review
Summary: The paper makes a strong and useful conceptual contribution by framing several LLM safety failures through the notion of **shallow safety alignment**: current alignment appears to alter model behavior mainly in the first few output tokens, leaving later-token harmful continuations comparatively intact once a non-refusal trajectory is entered. This framing is supported by multiple empirical analyses, including refusal-prefix prefilling, per-token KL divergence between aligned and base models, and per-token dynamics under harmful fine-tuning. The paper then proposes two mitigation directions: (1) augmentation with safety-recovery examples to encourage refusal even after harmful prefixes have begun, and (2) a token-wise constrained fine-tuning objective that more strongly preserves early-token distributions during downstream fine-tuning. Both interventions show substantial empirical gains in the settings tested. The paper is clear, technically competent, and practically relevant. The main limitations are scope and robustness validation: experiments are mostly on two 7B models, the “deep” alignment intervention is implemented post hoc rather than in a full alignment pipeline, and there is no adaptive-attack evaluation. Overall, after filtering reviewer noise, this is a solid and insightful paper with meaningful empirical support, though some of its broader claims should remain calibrated to the tested regimes.
Strengths:
- Introduces a compelling unifying lens: the paper convincingly connects prefilling attacks, adversarial suffix attacks, decoding-parameter exploits, and fine-tuning attacks to the common issue of early-token-dominated alignment.
- Provides strong empirical evidence for the central phenomenon. In particular, prefilling refusal prefixes sharply reduces harmfulness even for base models, and per-token KL analyses show alignment differences concentrated in early positions.
- The fine-tuning attack analysis is especially strong: per-token loss, gradient norms, and KL divergence together give a plausible mechanistic explanation for why only a few optimization steps can undo safety.
- The proposed mitigations are practical and empirically effective in the evaluated settings. Safety-recovery augmentation substantially lowers ASR for prefilling, GCG, and decoding-parameter attacks; constrained fine-tuning greatly reduces safety collapse during both harmful and benign downstream fine-tuning.
- The constrained fine-tuning objective is better grounded than a purely heuristic proposal: the appendix gives useful limiting-behavior and gradient interpretations, and relates the objective to KL-regularized RL.
- The paper is generally clear about what is and is not claimed. It explicitly notes that shallow alignment is not meant to explain all jailbreaks and that the proposed defenses are not complete defenses.
Weaknesses:
- Generality is not fully established. The empirical study centers on Llama-2-7B-Chat and Gemma-1.1-7B-IT; there is no evidence yet that the same degree of shallowness, or the same mitigation efficacy, holds for larger, newer, or differently aligned model families.
- The defense evaluations are non-adaptive. Since the proposed methods specifically target early-token vulnerabilities, an adaptive attacker could plausibly optimize attacks that shift pressure to later positions or otherwise account for the defense mechanism.
- The "deep safety alignment" intervention is only a partial demonstration of the concept: the augmentation is applied by further fine-tuning an already aligned model rather than integrating into a full end-to-end alignment pipeline. This weakens any claim that the paper has shown how to build deep alignment in standard production alignment pipelines.
- Some interpretation claims are plausible but not fully proven. For example, concentrated early-token KL strongly supports the paper's thesis, but does not by itself uniquely establish that current alignment procedures explicitly 'spend most of their budget' there for the exact reasons hypothesized.
- Residual vulnerabilities remain meaningful. For example, augmentation reduces but does not eliminate GCG success, and the paper could do more to analyze what failure modes remain and whether they reflect implementation limits or more fundamental limitations.
- The constrained fine-tuning method operates in a restricted threat model where the defender controls the fine-tuning interface. This is practically relevant for hosted APIs, but not for open-weight settings where users can directly fine-tune models.
Nice-to-haves:
- Broader evaluation on larger and more diverse model families, including newer models and models aligned with different post-training procedures.
- Adaptive attack experiments tailored to the proposed defenses, especially attacks targeting later-token continuations or defense-aware fine-tuning.
- A stronger comparison in the main paper against prior harmful-fine-tuning defenses; the appendix comparison is useful and could be more prominent.
- More systematic failure-case analysis for the augmented model, especially examples where recovery does not occur or where attacks still succeed after the defenses.
- Additional validation of the GPT-4 judge with a human-labeled subset, particularly for borderline 'recovery mid-response' outputs.
Novel insights: The paper’s main insight is not merely that early tokens matter, but that this observation can unify several seemingly different safety failures and directly suggest two distinct mitigation directions: either deepen safety so the model can recover after unsafe prefixes, or explicitly preserve the early-token safety distribution during downstream adaptation. That synthesis is the paper’s most valuable contribution. The work also highlights an important asymmetry: shallow alignment can look strong under standard evaluation while remaining brittle under any perturbation that diverts the initial trajectory.
Potentially missed related work:
- None
Suggestions:
- Calibrate the broadest claims slightly more carefully: the evidence strongly supports shallow alignment as an important driver of several vulnerabilities, but not necessarily a universal explanation across models and attack classes.
- Add adaptive evaluations. This is the most important missing experiment for both the augmentation-based defense and the constrained fine-tuning objective.
- Expand model coverage to test whether the phenomenon persists across scale, architecture, and alignment method.
- Promote the comparison with existing harmful-fine-tuning defenses from the appendix into the main paper to better contextualize the contribution.
- Discuss the threat-model boundary more explicitly: constrained fine-tuning is most applicable when the platform provider controls the fine-tuning process.
- Include more analysis of residual attack success after augmentation, especially the remaining GCG failures, to clarify whether the limitation is conceptual or implementation-specific.

# Merger Subscores
Novelty: 7.4
Technical soundness: 8.0
Empirical support: 7.9
Significance: 8.0
Clarity: 8.3

# Actual Human Scores
Individual reviewer scores: [10.0, 8.0, 10.0, 10.0]
Average score: 9.5
Binary outcome: Accept
