=== CALIBRATION EXAMPLE 1 ===

# Final Consolidated Review
Summary: This paper is a broad narrative survey of large language models covering architectures, applications, limitations, ethics, benchmarks, and future directions. It is readable and reasonably well organized at a high level, but it does not substantiate the title’s claim of being a systematic review, nor does it offer a novel synthesis, methodology, or organizing framework beyond summarizing well-known material.
Strengths:
- Broad topical coverage across core LLM themes: model types, training methods, applications, limitations, ethics, evaluation, and future directions.
- The paper is reasonably structured, moving through a logical sequence of introduction, literature review, applications/limitations, evaluation, and outlook.
- It cites a number of recent survey and benchmark papers from 2023–2024, showing awareness of more recent literature beyond the original Transformer-era works.
Weaknesses:
- The paper claims to be a systematic review, but it provides no systematic-review methodology: no research questions, search strategy, inclusion/exclusion criteria, screening process, or quality assessment.
- The review is largely descriptive rather than synthetic. Most sections summarize individual models or techniques in a textbook-like way without identifying patterns, trade-offs, contradictions, or gaps across the literature.
- There is no clearly articulated contribution or scope distinction relative to existing comprehensive surveys such as Bommasani et al. and Zhao et al.; as written, the paper does not explain what new value it adds.
- Coverage of recent LLM developments is incomplete for a modern survey, with little or no discussion of major 2023–2025-era model families and alignment methods that readers would expect in a current review.
- Several parts are repetitive or shallow, especially the repeated Transformer/self-attention discussion and the brief one-paragraph treatments of adversarial robustness, hallucination, and ethics.
- The comparative table is not well developed in the provided text: it lacks a clearly justified comparison framework and does not offer quantitative, model-by-model evidence to support the comparison.
- The paper is missing quantitative literature analysis that would justify the 'systematic' framing, such as publication trends, counts by topic, or benchmark/evaluation summaries.
Nice-to-haves:
- Add a PRISMA-style or otherwise explicit review protocol, even if the paper remains a survey rather than a full systematic review.
- Include a literature timeline or taxonomy figure to show how LLM architectures and training paradigms evolved over time.
- Add a more concrete benchmark section with task-wise comparisons and a clearer explanation of which metrics are appropriate for which tasks.
- Strengthen the future-directions section by tying each proposed direction to a specific gap identified in the surveyed literature.
- If the intent is a broad overview rather than a strict systematic review, rename the paper accordingly and sharpen the scope to avoid overclaiming.
Novel insights: The main issue is not factual correctness of individual technical statements, but the mismatch between scope and method: the paper reads as a high-level survey of standard LLM topics, yet it labels itself a systematic review without providing any of the machinery that would make that claim credible. The most important improvement would be to turn the paper from an encyclopedic summary into an evidence-backed synthesis with explicit scope, selection criteria, and comparative analysis.
Potentially missed related work:
- Recent surveys and benchmark analyses on LLM evaluation and explainability beyond the cited Zhao et al. and Chang et al. papers would help sharpen the comparative framing.
- Recent work on resource-efficient LLMs and parameter-efficient adaptation would be useful for the efficiency/fine-tuning discussion.
- Recent surveys and technical reports on alignment, instruction tuning, and post-training methods would make the review more current.
- Recent open and closed model families that dominate contemporary LLM discussions, such as LLaMA-class models, Mistral, Gemini, Claude, and related ecosystem papers, would likely be expected in a current survey.
Suggestions:
- Either commit to a genuine systematic review with a documented protocol, or relabel the paper as a general survey and narrow the claims accordingly.
- State an explicit research question or review objective: what gap does this paper fill relative to prior surveys?
- Replace abstract summaries with synthesis: compare methods across papers, highlight recurring failure modes, and identify unresolved questions.
- Add quantitative literature analysis (for example, counts by year, task, method family, or limitation category).
- Expand the comparative analysis with a clearer rubric and, if possible, actual performance numbers or more principled comparison criteria.
- Ground ethics, hallucination, and robustness discussions in concrete empirical studies and specific mitigation methods rather than broad statements.
- Update the survey to include major recent model families and post-training/alignment developments, or clearly state the time horizon covered.

# Actual Human Scores
Individual reviewer scores: [1.0, 1.0, 1.0, 1.0]
Average score: 1.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 2 ===

# Final Consolidated Review
Summary: This paper proposes MoDFL, a multi-objective decision-focused learning framework that combines three losses: a landscape loss based on sRMMD to compare objective-space distributions, a Pareto set loss to measure solution-space proximity, and a decision loss obtained by scalarizing the multi-objective problem with weighted sums. The overall idea is timely and the empirical section suggests improvements over adapted single-objective DFL baselines, but the paper has several substantive issues in clarity, evaluation, and methodological justification that prevent a strong assessment.
Strengths:
- The paper tackles an important and underexplored problem: extending decision-focused learning from single-objective to multi-objective optimization.
- The three-loss design is conceptually coherent: objective-space discrepancy, Pareto-set proximity, and representative decision quality are all relevant signals for multi-objective decision learning.
- The experimental section includes multiple baselines, an ablation study, and a comparison of alternative landscape losses, which helps support the role of the proposed components.
- The use of differentiable surrogates such as sRMMD and DSLP is a reasonable way to make the framework trainable end-to-end for LP-style problems.
Weaknesses:
- The decision-loss component relies on weighted-sum scalarization, which is a well-known limitation for multi-objective optimization because it cannot represent non-convex parts of the Pareto front; the paper does not adequately acknowledge or mitigate this restriction.
- The justification for the combined objective and the fixed loss weights (λl=1, λd=2, λps=5) is weak; there is no sensitivity analysis or theoretical rationale for why this weighting should work broadly.
- The differentiation story is underexplained. In particular, the Pareto-set loss involves a minimum over Pareto-optimal solutions, but the paper does not clearly show how gradients are obtained for this term in practice.
- The empirical reporting has a serious consistency problem: Tables 1 and 2 are presented as results on two different benchmarks, yet they are identical. This must be clarified or corrected, as it undermines confidence in the experimental section.
- Although experiments are said to be repeated five times, the paper reports no variance, confidence intervals, or significance tests, so the claimed improvements are hard to judge statistically.
- The evaluation is limited to LP-style benchmark problems. The method’s applicability to genuinely non-convex, mixed-integer, or harder nonlinear multi-objective problems remains untested.
- Several metrics used in the tables (e.g., r, r1, r2, r3) are not clearly defined in the main text, reducing reproducibility and making it difficult to interpret the reported numbers.
Nice-to-haves:
- Add a sensitivity study over λl, λd, and λps to show whether the method is robust to loss weighting.
- Include mean ± standard deviation over runs, and preferably paired significance tests, for all reported results.
- Provide visualizations of predicted vs. true Pareto fronts to make the effect of the proposed losses more interpretable.
- Discuss the computational overhead of obtaining Pareto sets during training and how the approach scales with problem size and number of objectives.
- Add a short limitations section explicitly discussing the weighted-sum scalarization issue and the LP-centric nature of the current experiments.
Novel insights: The paper’s core idea is less about inventing entirely new optimization machinery and more about assembling three complementary training signals for multi-objective decision-focused learning. That assembly is sensible, but its success likely depends heavily on the specific problem class: the decision loss is tied to scalarization, while the Pareto-set and landscape losses require access to good solution sets or approximations. This means the framework may work best as a structured surrogate for relatively well-behaved multi-objective LPs, rather than a general solution to arbitrary multi-objective decision making.
Potentially missed related work:
- As suggestions, the authors may want to discuss scalarization alternatives for multi-objective optimization such as Tchebycheff or ε-constraint methods, since weighted sums have known coverage limitations.
- It may also help to cite classical multi-objective optimization literature on Pareto-front approximation, IGD-style metrics, and many-objective benchmarking to better situate the Pareto-set and landscape losses.
- For the differentiable sorting / ranking component, related work on differentiable ranking and optimal-transport-based set comparison may provide additional context beyond the cited sRMMD paper.
Suggestions:
- Fix the duplicated experimental tables and ensure each benchmark has distinct, correctly labeled results.
- Define all evaluation metrics in the main paper, especially the regret quantities and the Pareto-quality metrics used in the tables.
- Add variance estimates and significance testing across repeated runs.
- Include a hyperparameter sweep or robustness study for the three loss weights.
- Clarify the gradient computation for the Pareto-set loss and how the minimum over Pareto solutions is differentiated in practice.
- Broaden the experimental evaluation beyond LP-style problems, or explicitly frame the method as targeted to that regime.
- Consider comparing against stronger multi-objective scalarization baselines or explicitly stating why simple single-objective adaptations are the fairest comparison.

# Actual Human Scores
Individual reviewer scores: [3.0, 1.0, 3.0]
Average score: 2.3
Binary outcome: Reject

=== CALIBRATION EXAMPLE 3 ===

# Final Consolidated Review
Summary: This paper presents MQFL-FHE, a hybrid framework combining federated learning, fully homomorphic encryption (CKKS), quantum neural networks, and a multimodal mixture-of-experts extension for image and biological data. The paper’s main empirical claim is that adding quantum components can partially offset the accuracy loss introduced by FHE in federated settings, but the evidence is limited and uneven: improvements are modest, some results are mixed, and the strongest claimed contribution (MQMoE) is introduced architecturally but not actually evaluated. Overall, the paper is ambitious and well-motivated, but the experimental support and theoretical justification do not yet fully substantiate the headline claims.
Strengths:
- The paper addresses an important and timely problem: privacy-preserving federated learning with FHE, where aggregation overhead and accuracy degradation are real concerns.
- It provides a reasonably detailed system description, including CKKS parameter choices, a federated workflow, and Algorithm 1 for the multimodal pipeline.
- The paper explores a broad set of datasets spanning vision, genomics, and medical imaging, which at least demonstrates the intended multimodal scope of the framework.
- The results do show some empirical signal that QFL+FHE can outperform FL+FHE on certain datasets (e.g., CIFAR-10 and PCOS), so the central hypothesis is not entirely unsupported.
- The ROC-AUC analysis suggests the quantum-enhanced variant can improve class discrimination on the DNA and MRI tasks beyond what is seen in top-line accuracy alone.
Weaknesses:
- The strongest claimed contribution, MQMoE, is not actually evaluated in the reported experiments. The paper introduces the architecture, but there are no quantitative results showing that MQMoE improves performance.
- The claim that quantum computing mitigates FHE-induced degradation is only weakly supported. Improvements over FL+FHE are modest and not consistent across datasets; in some centralized settings the quantum variant is worse than the classical one.
- There is no convincing ablation that isolates the effect of the quantum component from architectural changes or simulation choices. As a result, it is hard to attribute any gains specifically to quantum computation.
- The theoretical explanation for why quantum layers would reduce FHE noise or accuracy loss is not rigorous. The SU(2)/Bloch-sphere discussion reads more like intuition than a derivation tied to the implemented system.
- The quantum setup appears to rely on PennyLane simulations rather than hardware, so the paper does not demonstrate any real quantum advantage; the computational claims are therefore much more tentative.
- Reproducibility is limited by missing implementation details, such as the exact PQC depth/layer count, client data partitioning, and the rationale for several hyperparameter choices.
- The method and results are somewhat inconsistent in how they define where the quantum component acts: some parts imply quantum-enhanced aggregation, while others suggest only a quantum local model or hybrid preprocessing. This makes the end-to-end pipeline hard to interpret precisely.
- The computational overhead is substantial, especially for QFL+FHE, and the paper does not provide a convincing cost-benefit analysis showing that the modest accuracy gains justify the added time.
Nice-to-haves:
- A quantitative ablation table separating the contributions of QC, FHE, and MQMoE would make the paper much stronger.
- Convergence curves over communication rounds would help show whether the quantum variant learns faster, more stably, or only ends with slightly better accuracy.
- Per-client performance breakdowns would clarify whether the method helps underrepresented or difficult clients, as claimed.
- A more explicit discussion of privacy-utility trade-offs would be valuable, including attack success rates or at least clearer security-cost analysis.
- If MQMoE is intended as a major contribution, it should be benchmarked against classical MoE and simpler multimodal baselines.
- Reporting confidence intervals or standard deviations for the key metrics, especially AUC, would help assess whether the observed gains are statistically meaningful.
- A more careful discussion of when simulation-based quantum models may or may not indicate practical benefits would improve the framing of the work.
Novel insights: The paper’s main conceptual novelty is not the individual use of CKKS, FedAvg, or PQCs, but the attempt to combine them with multimodal learning and an MoE-style architecture. However, the current evaluation suggests that the multimodal quantum MoE idea is more of a promising design direction than a validated contribution. The most credible empirical takeaway is narrower than the paper claims: quantum-enhanced local modeling can sometimes reduce the accuracy drop caused by FHE in federated training, but the effect is modest and not yet clearly attributable to a mechanism beyond architecture/simulation choices.
Potentially missed related work:
- FedQNN (Innan et al., 2024) for federated quantum neural networks in healthcare/genomics, as a natural baseline for the quantum component.
- FheFL (Rahulamathavan et al., 2023) and FedSHE (Pan et al., 2024) as important FL+FHE privacy-preserving baselines.
- CreamFL (Yu et al., 2023b) for multimodal federated learning, since it is the closest multimodal FL comparator.
- Federated quantum learning/challenges papers such as Ren et al. (2023) and Gurung et al. (2023), which may help frame practical limitations more honestly.
- Recent multimodal FL with encryption work such as Gong et al. (2024a), which is directly relevant to the multimodal privacy-preserving setting.
Suggestions:
- Add a real quantitative evaluation of MQMoE and make it the centerpiece if it is meant to be the main contribution.
- Run a strict ablation study with matched architectures to isolate the effect of QC versus classical multimodal learning and FHE.
- Clarify whether the quantum part is only simulated; if so, frame claims carefully and avoid implying demonstrated hardware-level quantum advantage.
- Strengthen the theory by connecting the FHE-noise story to the actual computation path used in the experiments.
- Add stronger baselines from the FL, FHE, and multimodal FL literature, especially FedQNN, FheFL, FedSHE, and CreamFL.
- Report standard deviations or confidence intervals and, where possible, statistical tests for the main metrics.
- Discuss runtime and communication overhead more explicitly, including whether the accuracy gains justify the extra cost.
- If the paper remains broad, narrow the scope and fully validate fewer claims rather than making many under-supported ones.

# Actual Human Scores
Individual reviewer scores: [3.0, 5.0, 3.0, 3.0, 3.0]
Average score: 3.4
Binary outcome: Reject

=== CALIBRATION EXAMPLE 4 ===

# Final Consolidated Review
Summary: This paper proposes FedQLoRA, a federated PEFT framework for clients that use heterogeneous quantized LLMs. The core idea is to separate quantization error from local task adaptation via a quantization-aware adapter, and to optionally iterate this separation with a dynamic adapter to mitigate non-IID data heterogeneity. The paper’s central novelty is identifying quantization bias as a distinct failure mode in adapter aggregation across mixed quantization levels, and it supports this with a mathematical derivation plus experiments on text classification datasets under IID and non-IID splits.
Strengths:
- The paper identifies a genuinely interesting and non-obvious problem: aggregation bias introduced by mixing adapters trained on clients with different quantization levels.
- The proposed quantization-aware adapter is a plausible and well-motivated mechanism for separating quantization loss from task-specific adaptation.
- The iterative FedQLoRA variant is a sensible extension for non-IID settings, since it tries to reduce heterogeneity bias by alternating between global and local adapter refinement.
- The experiments include both IID and non-IID settings, multiple client counts, and comparisons against several LoRA-based federated baselines.
- The empirical results consistently show that FedQLoRA/iFedQLoRA outperform the included baselines, and the iterative version is generally strongest in the non-IID setting.
Weaknesses:
- The experimental evaluation does not validate the paper’s LLM framing: it uses DistilBERT on text classification tasks, not true large-scale LLMs or generation/instruction-following benchmarks.
- The set of baselines is incomplete for the heterogeneity setting; standard FL methods such as FedProx, FedNova, or SCAFFOLD are not included, so it is hard to judge whether the method is better than generic FL approaches rather than only LoRA-specific ones.
- Several baselines, especially FFA-LoRA, perform implausibly poorly, which raises concerns about implementation quality or hyperparameter tuning and weakens confidence in the comparative results.
- The paper lacks key ablations, especially on the rank/capacity of the quantization-aware adapter and on whether the gains come from the new decomposition itself versus simply adding more trainable parameters.
- Theoretical support is incomplete: Proposition 1 is stated without a proof, and the iterative optimization is only supported empirically, with no convergence analysis or conditions.
- The paper does not report communication-cost measurements, even though communication efficiency is one of the main motivations for adapter-based federated learning.
- Important experimental details needed for reproducibility are sparse, including some LoRA hyperparameters and training protocol specifics.
Nice-to-haves:
- Add experiments on actual LLMs at meaningful scale, such as 7B-class models, to substantiate the claim that the method applies to federated LLM fine-tuning.
- Include generation or instruction-tuning tasks in addition to classification to demonstrate broader applicability.
- Report the magnitude of quantization bias directly, rather than inferring it only from downstream accuracy drops.
- Add a sensitivity study over the quantization-aware adapter rank to show the trade-off between accuracy and overhead.
- Provide a communication-traffic analysis comparing FedQLoRA with adapter-sharing baselines.
- Investigate and, if necessary, retune the weak FFA-LoRA baseline before drawing strong conclusions from that comparison.
Novel insights: The most important conceptual contribution is that the paper separates two different sources of degradation that are often conflated in federated PEFT: quantization bias caused by heterogeneous bit-widths, and heterogeneity bias caused by non-IID client data. This is a useful lens because it explains why a federated LoRA method can fail even when all clients share the same architecture, and it motivates a two-level correction scheme: a personalized quantization-aware adapter for client-specific quantization error, and an iterative global-local alternation for data heterogeneity. That said, the evidence currently supports the idea best at the level of a proof-of-concept on modest classification benchmarks, not yet as a demonstrated LLM-scale solution.
Potentially missed related work:
- FedProx
- FedNova
- SCAFFOLD
- FedPrompt
- FedPEFT / broader federated PEFT baselines
- Recent LLM federated fine-tuning frameworks beyond the cited adapter-sharing works, if relevant to the final camera-ready version
Suggestions:
- Validate the method on real large language models and more LLM-relevant tasks.
- Add standard federated optimization baselines for non-IID data.
- Provide proof sketches for the stated proposition and any other theoretical claims.
- Include a convergence discussion or analysis for the iterative algorithm.
- Report communication overhead and memory overhead explicitly.
- Add ablations for adapter rank and for the individual contribution of the quantization-aware adapter.
- Audit the strong performance gap of FFA-LoRA to ensure the baseline is correctly implemented and tuned.
- If the paper’s main claim is about quantization bias, measure that quantity directly in experiments.

# Actual Human Scores
Individual reviewer scores: [3.0, 5.0, 5.0, 3.0]
Average score: 4.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 5 ===

# Final Consolidated Review
Summary: The paper proposes AdaFM, an adaptive variance-reduced method for stochastic minimax optimization. Its main contribution is a STORM-like estimator/momentum design with learning rates adapted from historical estimator norms, yielding near-optimal O(ε^{-3}) sample complexity in both NC-SC and NC-PL settings and demonstrating improved robustness over RSGDA, VRAdaGDA, and TiAda on synthetic, deep AUC, and WGAN-GP experiments. The work is technically substantial and addresses a real practical issue: hyperparameter sensitivity in minimax VR methods. That said, the strongest concerns are about the overstatement of being “parameter-free,” the lack of broader baselines/statistical rigor, and some incomplete experimental validation of the claimed robustness.
Strengths:
- Addresses a real and practically important issue: variance-reduced minimax methods are often fragile to hyperparameter choices, and the paper directly targets this pain point.
- Proposes a concrete adaptive algorithm with a clear mechanism: momentum is fixed to a simple schedule (β_{t+1}=1/t^{2/3}) and learning rates depend on historical estimator information, which is well motivated by the error-dynamics analysis.
- Provides substantial theory in both NC-SC and NC-PL regimes, with near-optimal O(ε^{-3}) sample complexity claims that match the best parametric VR methods.
- Experiments cover multiple relevant settings (synthetic test functions, deep AUC, WGAN-GP) and consistently show improved robustness/convergence over the compared minimax baselines.
- The hyperparameter-sensitivity demonstrations for RSGDA/VRAdaGDA and the ablation on δ help support the core practical motivation of the paper.
Weaknesses:
- The paper overstates the “parameter-free”/“no manual tuning” claim. AdaFM still requires choosing γ, λ, and especially δ, and the appendix ablation shows that δ=0 can fail to converge.
- The experimental comparison is somewhat incomplete: it does not include common adaptive optimizers such as Adam/RMSprop for the WGAN setting, which would better contextualize the practical value of the method.
- The paper does not report error bars, multiple-seed variability, or other statistical measures, which weakens confidence in the experimental comparisons.
- The robustness claims are not fully stress-tested: δ is shown to matter, but its effect is only explored in limited settings rather than systematically across tasks and ranges.
- Theoretical proofs are extensive and intricate, but the presentation is hard to follow; the four-case analysis in particular would benefit from a higher-level roadmap or simplification.
Nice-to-haves:
- Provide a more precise description of what is meant by “parameter-free” versus “minimal tuning,” and state explicitly which hyperparameters still need to be fixed in practice.
- Report the total hyperparameter-search budget for each method to make the ease-of-use comparison more concrete.
- Add learning-rate trajectory plots to show how η_t^x and η_t^y evolve during training and how quickly the desired ratio is reached.
- Include more diverse WGAN experiments or additional architectures/datasets to strengthen the robustness claims.
- Add a short failure-mode discussion describing settings where AdaFM is sensitive or underperforms.
Novel insights: A central subtlety is that AdaFM’s practical advantage is not true elimination of all tuning, but rather a reduction in dependence on problem-specific schedules and delicate ratios. The paper’s strongest technical idea is to tie both the momentum decay and step sizes to historical estimator norms so that the x/y stepsize ratio self-corrects while remaining monotone. The evidence suggests this is useful, but the method still relies on at least one nontrivial scale parameter δ, so the meaningful claim is closer to 'substantially reduced tuning sensitivity' than fully parameter-free optimization.
Potentially missed related work:
- Adam (Kingma & Ba, 2014) and RMSprop-style adaptive optimizers as practical baselines for GAN/WGAN training.
- Recent adaptive minimax methods beyond the core baselines, especially more directly comparable parameter-agnostic or adaptive GDA variants that may help contextualize AdaFM's practical gains.
- More recent adaptive variance-reduction work in minimization that could provide additional perspective on the design choice of the momentum/step-size schedule.
Suggestions:
- Clarify the scope of the contribution by explicitly separating theoretical parameter-freedom from practical hyperparameter dependence.
- Add Adam/RMSprop baselines in WGAN-GP and, if feasible, in the other experimental settings.
- Report mean and standard deviation over multiple runs, or confidence intervals, for the main experimental curves.
- Expand the δ study to a wider range and across at least one real task beyond the current appendix ablation.
- Quantify tuning effort: number of grid points searched, total runs, and wall-clock cost for each method.
- Consider adding a condensed proof sketch or visual roadmap to make the theoretical analysis easier to digest.
- If space allows, include an explicit comparison table that separates algorithmic novelty, theoretical rate, and tuning burden across AdaFM, TiAda, VRAdaGDA, and RSGDA.

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 8.0, 6.0]
Average score: 5.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 6 ===

# Final Consolidated Review
Summary: Swift-FedGNN addresses an important and practically relevant bottleneck in federated GNN training: cross-client neighbor sampling and communication. The paper’s main idea is to make cross-client training periodic and limited to a subset of clients, while letting the remaining clients do local training, and it backs this with a convergence analysis and experiments on ogbn-products and Reddit. The empirical gains in communication cost and wall-clock efficiency are real, and the paper is generally well-motivated and clearly structured. The main substantive concern is a theory–practice mismatch: the formal convergence analysis is for GCN, while the experiments use GraphSAGE. In addition, the privacy discussion is architectural rather than formal, and the paper could better justify how to choose the correction frequency and clarify the implications of mixed local/cross-client gradients.
Strengths:
- The paper tackles a genuinely important bottleneck in federated GNNs: cross-client neighbor sampling and communication overhead, which is well motivated and empirically illustrated.
- The algorithmic framework is concrete and reasonably clear: periodic cross-client training with sampled clients, combined with local training in the other rounds, plus a two-stage aggregation design that reduces communication.
- The theoretical analysis is nontrivial and addresses biased stochastic gradients in GNNs without relying on overly strong unbiased-gradient assumptions.
- The experiments are relevant and fairly strong for the paper’s scope: they evaluate on ogbn-products and Reddit, include reasonable federated baselines, and show large communication reductions and faster wall-clock convergence.
- The paper provides sensitivity studies for several key hyperparameters, including correction frequency, number of cross-client training clients, fan-out, and number of clients.
Weaknesses:
- The most important issue is a mismatch between theory and experiments: the convergence analysis is developed for GCN, but the experiments are run with GraphSAGE. The paper does not convincingly explain why the stated guarantees should transfer.
- The privacy benefit is only argued at the design level. The paper does not provide a formal privacy analysis, so claims about privacy preservation should be stated more carefully.
- Theoretical quantities in the bounds are not very interpretable in the main text, and the paper does not give practical guidance for choosing the correction frequency I or the sampled-client set size K.
- The update protocol mixes local-training clients and cross-client-training clients in the same rounds, but the implications for gradient bias and FedAvg-style aggregation are not fully discussed.
- The experimental evaluation is helpful but still somewhat limited in breadth: it uses only two datasets and only one main GNN setting, so scalability across deeper models or more heterogeneous partitions remains unclear.
Nice-to-haves:
- An experiment using GCN would align the experiments with the theoretical analysis, or alternatively the theory could be extended to GraphSAGE.
- A more explicit ablation separating the contribution of periodic correction, two-stage aggregation, and client subsampling would help isolate which design choice matters most.
- A more formal statement of the threat model behind the privacy claim would make the paper easier to interpret.
- A Pareto-style plot of accuracy versus communication cost would summarize the trade-off more cleanly.
- Additional experiments under more heterogeneous client partitions would improve confidence in robustness beyond METIS partitioning.
Novel insights: The paper’s real technical novelty is not just periodic collaboration, but the combination of periodic client sampling with server-mediated double aggregation that avoids explicit feature sharing while still reintroducing cross-client neighborhood information. A useful takeaway from the theory is that approximation errors grow with GNN depth, which suggests the method becomes more attractive precisely when cross-client neighborhood effects matter more. At the same time, the practical value of the approach depends on how often cross-client correction is triggered; the paper shows the trade-off exists, but it does not yet turn that insight into a usable selection rule.
Potentially missed related work:
- FedGraphNN (He et al., 2021a)
- SpreadGNN (He et al., 2021b)
- FedGCN / federated GCN convergence-communication tradeoff work (Yao et al., 2023a)
- Subgraph federated learning with missing neighbor generation (Zhang et al., 2021)
Suggestions:
- Extend the convergence analysis to GraphSAGE, or add experiments with GCN to match the stated theory.
- Clarify the update rule for clients in K and M\K during cross-client rounds, and explain whether naive averaging of their gradients is theoretically justified.
- Add an explicit privacy/threat-model discussion, and avoid language that sounds like formal privacy guarantees unless such guarantees are proved.
- Provide practical heuristics or a tuning recipe for I and K, since the current theory exposes the trade-off but does not operationalize it.
- Add ablations that separate the effect of periodic cross-client training from the effect of the two-stage aggregation design.
- Consider evaluating on more heterogeneous graph partitions and, if possible, deeper GNNs to test the limits of the method.

# Actual Human Scores
Individual reviewer scores: [3.0, 6.0, 5.0, 5.0]
Average score: 4.8
Binary outcome: Reject

=== CALIBRATION EXAMPLE 7 ===

# Final Consolidated Review
Summary: This paper argues that unlabeled OOD detection can fundamentally fail when the SSL/unsupervised objective is independent of the downstream labels, and introduces an Adjacent OOD benchmark to expose this failure mode. The paper’s main value lies in the conceptual connection between label dependence, representation learning, and OOD behavior, along with the idea that current near/far OOD benchmarks may miss important safety-relevant cases. At the same time, the theoretical claims rely on strong assumptions and some proofs/interpretations appear overstated, while the empirical evidence is mixed and not always stronger than simpler or better-chosen baselines.
Strengths:
- The paper identifies an important and timely problem: whether unlabeled OOD detection methods can be trusted in safety-critical settings where labels matter.
- The Adjacent OOD benchmark is a useful and novel framing that targets a failure mode not well covered by standard far-OOD or near-OOD benchmarks.
- The paper provides a clear conceptual argument linking surrogate-task learning, information bottlenecks, and the possibility of label blindness in learned representations.
- The experiments are broad enough to include multiple datasets and method families (SSL, unsupervised, and zero-shot), which helps illustrate that the issue is not confined to a single approach.
- The paper’s discussion of when unlabeled OOD methods may still be acceptable is practical and helps contextualize the limitations rather than claiming they are universally unusable.
Weaknesses:
- The central theoretical results depend on very strong assumptions, especially the strict independence factorization of data into label-relevant and label-irrelevant variables; the paper acknowledges an approximate case but does not provide a corresponding theory.
- The information-bottleneck interpretation of modern DNN training is treated as broadly applicable, but this is a debated assumption and the paper does not engage much with that controversy.
- Several proofs and claims, especially around unavoidable overlap in real-world OOD data, feel more heuristic than fully rigorous and may overstate what is actually established.
- The experimental comparison against supervised OOD detection is limited: MSP is the main supervised baseline, which is relatively weak compared with stronger label-based detectors such as ODIN, Energy, Mahalanobis, or supervised contrastive variants.
- The empirical picture is mixed: on some adjacent CIFAR settings the unlabeled methods are non-trivial and sometimes competitive, so the paper’s broad narrative of near-universal failure is stronger than the evidence supports.
- The paper does not quantify how much label information is needed to move from label blindness to reliable detection, leaving the practical path forward underspecified.
- The zero-shot analysis is plausible but mostly post hoc and not developed into a systematic theory or study of alignment between pretraining data and target labels.
Nice-to-haves:
- A stronger empirical suite with supervised baselines beyond MSP, especially ODIN, Energy, Mahalanobis, and supervised contrastive methods.
- An explicit study varying label budget from zero-shot to few-shot to quantify how performance changes as label information is added.
- Quantitative analysis of the learned representations, e.g. probing or feature attribution, to substantiate the claim that SSL is capturing nuisance structure rather than semantic labels.
- Visualization of representation geometry or score distributions for ID versus adjacent OOD to show how the failure manifests.
- A more systematic characterization of what makes two classes ‘adjacent’ in practice, beyond the dataset-specific examples currently provided.
Novel insights: The paper’s most interesting insight is that unlabeled OOD detection can appear strong on standard benchmarks while still being fundamentally misaligned with the real safety question: whether a detector can reject semantically adjacent but label-shifted inputs. The Adjacent OOD framing usefully exposes this gap. A second important insight is that performance of unlabeled or zero-shot methods may depend less on the detector itself than on whether the surrogate objective or pretraining vocabulary accidentally aligns with the target labels; this helps explain why such methods can succeed on some benchmarks and fail badly on others.
Potentially missed related work:
- Supervised out-of-distribution detection methods such as ODIN, Energy-based OOD detection, and Mahalanobis distance would provide stronger label-based baselines than MSP.
- Supervised contrastive OOD work, especially Sun et al. (2022), is relevant because it explicitly studies how richer supervised representations improve OOD detection.
- Recent unlabeled OOD methods such as CADET and CSI are relevant comparators if the goal is to position Adjacent OOD as a stress test for the current SSL/OOD landscape.
- Work on few-shot or class-incremental OOD setups may be useful for contextualizing whether Adjacent OOD is truly novel or closely related to existing task-splitting benchmarks.
- Recent theoretical work on how unlabeled data can provably help OOD detection, and on how labels help OOD detection, is relevant to sharpen the distinction between this paper’s failure claims and complementary positive results.
Suggestions:
- Temper the strongest universal claims and more carefully distinguish strict theoretical failure from the approximate, empirical regime actually tested.
- Add stronger supervised baselines and report whether the proposed benchmark remains challenging once modern label-based OOD detectors are used.
- Develop a quantitative notion of approximate label blindness, ideally linked to mutual information or a measurable proxy, so the theory better matches the experiments.
- Include ablations over label budget and perhaps hybrid methods to show how much annotation is needed to overcome the failure mode.
- Expand the analysis of zero-shot methods with a systematic measure of pretraining-label alignment rather than a qualitative explanation only.
- If possible, add representation analyses (linear probes, embeddings, saliency, or clustering) to show what features are retained by SSL/unsupervised methods on the adjacent benchmarks.
- Clarify the relationship between this work and prior theoretical results on no-free-generalization and on unlabeled data helping OOD detection, emphasizing what is new and what is complementary.

# Actual Human Scores
Individual reviewer scores: [6.0, 5.0, 8.0]
Average score: 6.3
Binary outcome: Accept

=== CALIBRATION EXAMPLE 8 ===

# Final Consolidated Review
Summary: LocoVR is a substantive dataset paper introducing a large VR-collected indoor locomotion corpus with 7000+ two-person trajectories across 131 home scenes, plus a real-world test set (LocoReal) and evaluations on path, trajectory, and goal prediction. The paper’s main contribution is the dataset and collection pipeline rather than a new model. The experiments consistently show that training on LocoVR outperforms training on prior indoor datasets, and the ablations support the value of scale, multi-person context, and heading information. That said, the paper still leaves some important questions open: the social-navigation claims are only weakly quantified, the baselines are relatively limited, and the sim-to-real validation remains small in scope.
Strengths:
- Large-scale indoor dataset: 131 indoor home scenes and 7071 trajectory sequences is a meaningful scale jump over prior indoor motion datasets, especially for room-scale locomotion.
- VR-based collection pipeline is practical and well documented, including hardware, scene sourcing, alignment, tracking setup, and preprocessing details.
- The paper evaluates the dataset on three relevant downstream tasks (global path prediction, trajectory prediction, goal prediction), which makes the dataset’s utility concrete rather than purely descriptive.
- Ablation studies are a real strength: the paper tests the effects of dataset scale, multi-person input, and heading direction, and also includes additional experiments on scene representations.
- The dataset includes useful metadata beyond raw trajectories, such as head orientation and body tracker information, which increases its potential usefulness for future work.
- The paper includes a real-world test dataset (LocoReal), providing at least some evidence of transfer beyond the VR environment.
Weaknesses:
- The paper’s strongest claim is that LocoVR captures socially-motivated navigation, but this is only weakly quantified. Qualitative examples and a modest ablation with/without the other person’s trajectory are suggestive, but there is no direct metric for proxemic behavior, detour frequency, yielding, or collision avoidance.
- The evaluation baseline set is limited. The paper relies heavily on U-Net variants and Ynet; it does not benchmark against more modern social trajectory predictors or stronger transformer/GNN-based baselines, so it is hard to gauge how competitive the dataset is for current methods.
- The comparison against GIMO and THOR-MAGNI conflates several factors at once (scene count, trajectory count, multi-person setting, and task format), so the results do not cleanly isolate why LocoVR helps.
- The VR-to-real validation is useful but small: LocoReal has only 4 layouts in one physical room, which limits confidence in broad sim-to-real generalization claims.
- The participant pool is narrow (32 able-bodied adults, mostly young, from a single institutional setting), which limits demographic diversity and may affect the generalizability of proxemic behavior.
- The paper sometimes attributes performance gains to social motion learning, but some ablations show only small differences from adding the second person’s trajectory, suggesting that geometry/scene scale may be doing most of the work in some tasks.
Nice-to-haves:
- Provide a more direct quantitative analysis of social behaviors, such as minimum interpersonal distance, detour length, yielding frequency, or collision-avoidance success rate.
- Add stronger and more contemporary trajectory-prediction baselines, especially social/interaction-aware methods.
- Include a matched-scale comparison that better isolates whether gains come from VR collection, scene diversity, multi-person data, or sheer dataset size.
- Expand the real-world test set to more physical rooms or home environments to strengthen the VR-to-real claim.
- Show failure cases and per-scene breakdowns to clarify where LocoVR helps most and where it still struggles.
- Make fuller use of the 3D/body-tracker information, or explicitly justify why 2D representations are sufficient for the benchmark tasks.
Novel insights: The most important technical takeaway is that the paper’s gains appear to be driven at least as much by scene-scale diversity and geometric coverage as by explicit social-interaction modeling. The weak ablation effect from removing the second person’s trajectory suggests that ‘social motion’ is present in the data, but the current benchmarks do not strongly prove that the models learn a distinct social-navigation signal rather than mostly benefiting from larger and more varied indoor geometry. In other words, the dataset is clearly valuable, but the paper’s evidence supports it more strongly as a scalable indoor geometry/trajectory corpus than as a rigorously quantified social-navigation benchmark.
Potentially missed related work:
- Social LSTM / Social GAN / PECNet-style social trajectory forecasting methods could provide stronger comparative baselines for the trajectory task.
- More recent transformer- or graph-based trajectory prediction models would better reflect the current state of the art.
- Prior work on proxemics and socially-aware navigation datasets/benchmarks could be useful for framing the social-motion claims more precisely.
- Works that directly evaluate sim-to-real locomotion transfer in indoor environments would be relevant for the LocoReal discussion.
Suggestions:
- Quantify social proxemic behavior directly rather than relying mainly on qualitative examples.
- Add at least one strong modern social-trajectory baseline and one strong generic forecasting baseline.
- Perform a controlled matched-size study to separate the effect of dataset size from the effect of scene diversity and multi-person interaction.
- Broaden LocoReal or add another real-world test set to make transfer claims more convincing.
- Clarify how single-person baselines are adapted to the multi-person setting, especially for Ynet.
- Consider an analysis of demographic or interpersonal factors if the authors want to argue that the dataset captures social norms, not just motion patterns.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0, 6.0, 3.0]
Average score: 5.8
Binary outcome: Accept

=== CALIBRATION EXAMPLE 9 ===

# Final Consolidated Review
Summary: This paper proposes Process Advantage Verifiers (PAVs), which define dense process rewards as advantages measured under a distinct prover policy rather than as base-policy Q-values. The paper supports this with a didactic analysis, a tabular theory of complementary provers, and experiments on Gemma 2B/9B/27B on MATH showing improved test-time search efficiency and improved online RL sample efficiency relative to outcome-reward baselines. The central idea is clear, genuinely interesting, and backed by substantial empirical evidence, though the work remains mostly limited to one domain and leaves several practical questions about prover selection, compute accounting, and theory-to-practice alignment.
Strengths:
- The core insight is strong and well-motivated: process rewards should measure progress (advantages) rather than absolute value, and that progress should be measured under a prover policy distinct from the base policy.
- The paper provides a real theoretical contribution, formalizing 'complementary provers' through distinguishability and alignment, and it derives a meaningful policy-improvement bound in a stylized RL setting.
- The empirical evaluation is broad within its target domain, covering three model sizes (2B/9B/27B), both test-time search and online RL, and multiple prover choices/ablation settings.
- The results are substantively positive: PAVs improve test-time search accuracy and compute efficiency over ORM/Q-value baselines, and PAV-RL improves sample efficiency and final accuracy over ORM-RL.
- The paper is unusually practical for a theory-heavy submission: it spells out how to train PAVs, including the first-pit strategy, data-collection ratios, and hyperparameter tuning procedures.
- The observation that weaker or intermediate provers can outperform stronger ones is interesting and well-supported by both the theory and the ablations.
Weaknesses:
- The evidence remains confined to the MATH benchmark and Gemma models, so the generality of the main claim beyond math reasoning is still unproven.
- The theoretical results are derived in a tabular, oracle-access setting, while the practical system uses learned verifiers and large language models; the gap is acknowledged but not bridged.
- Prover choice is still somewhat ad hoc in practice: BoK with K=4 works well in the experiments, but the paper does not provide a principled method for selecting or adapting provers as the base policy changes.
- The compute-efficiency claims are directionally convincing but not fully transparent. The paper compares different search procedures with different scoring and expansion costs, and the end-to-end accounting is not fully worked out in the main text.
- Several strong claims about RL gains are only compared against ORM-RL; the paper would be stronger with additional baselines or at least a clearer positioning relative to other online RL methods.
- The hyperparameter alpha requires validation-set tuning, which is reasonable experimentally but limits out-of-the-box deployment and should be emphasized as an operational requirement.
Nice-to-haves:
- A more direct analysis of how PAV prediction error affects downstream search/RL performance would help quantify robustness.
- More evidence for the claimed exploration mechanism, e.g. entropy/diversity measurements or concrete trace visualizations where PAV and Q-value PRMs make different decisions, would make the intuition more tangible.
- A broader benchmark suite would help establish whether the complementary-prover phenomenon transfers to other reasoning domains such as GSM8K, multi-hop QA, or code.
- A cleaner end-to-end compute analysis that includes verifier training, inference, and beam-expansion cost would make the efficiency story easier to interpret.
- An automated or adaptive prover-selection strategy would make the method more practical, especially for settings where the base policy improves over time.
- A direct comparison to additional strong RL baselines would clarify how much of the gain comes from PAVs specifically versus dense online supervision more generally.
Novel insights: The most interesting insight is that dense step rewards should not be judged by whether they estimate correctness directly, but by whether they create useful progress signals relative to a complementary prover. This reframes process supervision from 'identify correct steps' to 'identify steps that change the future solvability landscape in a way that helps exploration.' The theory and experiments jointly suggest that an intermediate or even weaker prover can be more useful than a stronger one because it distinguishes among base-policy steps better, which is a non-obvious and valuable departure from the usual teacher-student intuition.
Potentially missed related work:
- AlphaLLM (Tian et al., 2024) for exploration-oriented search, which the paper discusses in Appendix K but not in the main related-work narrative.
- Recent RL-for-reasoning baselines such as GRPO-style methods or other policy-gradient variants would be useful comparison points for the online-RL claims.
- Work on uncertainty- or exploration-aware tree search for LLM reasoning may provide a more direct comparator for the exploration claims.
- Additional recent verifier-based reasoning methods beyond the cited ORM/PRM line could help position PAVs more clearly in the broader verifier literature.
Suggestions:
- Add a broader-domain evaluation, ideally at least GSM8K, to show the main idea is not MATH-specific.
- Report a sensitivity study for PAV quality: perturb verifier outputs or vary verifier accuracy to show how much downstream performance depends on the learned approximation.
- Include stronger and more diverse RL baselines, or clearly delimit the scope of the 6× claim to ORM-RL specifically.
- Provide an end-to-end compute table that separates training cost, verifier inference cost, beam-search cost, and RL update cost.
- Measure distinguishability/alignment statistics for the chosen provers to connect the theory more directly to the empirical choices.
- Consider an adaptive prover-selection mechanism or joint prover/base optimization, since the paper already identifies this as an open direction.
- Add a few qualitative examples showing how a PAV assigns credit to a step that an ORM/Q-value PRM would not prioritize, to make the core idea more concrete.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 8.0, 8.0, 6.0]
Average score: 7.1
Binary outcome: Accept

=== CALIBRATION EXAMPLE 10 ===

# Final Consolidated Review
Summary: This paper introduces a new task, audio difference explanation, and proposes two benchmark datasets (ACD and CLD) together with ADIFF, a prefix-tuning-based model with cross-projection and multi-stage training. The paper is clearly motivated and contains a substantial experimental section with ablations, objective metrics, and human evaluation. However, the core methodological and evaluative limitations remain significant: the datasets are built from LLM-generated explanations rather than direct human-written difference descriptions, the metrics are only loosely aligned with the task, and several claims are supported by limited or somewhat imbalanced comparisons. Overall, the work is novel and useful as an initial benchmark, but the paper would benefit from a stronger task-specific evaluation protocol and more careful positioning of its results.
Strengths:
- The paper identifies and formalizes a genuinely novel task that is not previously studied in this form: natural-language explanation of differences between two audio recordings.
- It provides two benchmark datasets (ACD and CLD) with three explanation tiers, which is a useful setup for studying granularity and comparative audio reasoning.
- The ADIFF architecture is sensible for the task: prefix tuning with a cross-projection module and a separator token directly targets the need to compare two audio inputs.
- The ablation studies are fairly extensive and cover architecture choices, language-model scaling, position captioning, and stage-3 finetuning, giving the reader concrete insight into which components matter.
- The paper includes both automatic metrics and human evaluation, and the human study evaluates multiple dimensions relevant to the task (correctness, granularity, readability).
- The hallucination analysis using frozen audio-event probabilities is a pragmatic and interpretable diagnostic contribution.
- The experimental appendix is detailed enough to support reproducibility, including training setup, prompts, and additional tables.
Weaknesses:
- The training and validation explanations are generated by an LLM from captions rather than being directly written by humans after listening to the paired audios, so the supervision is only indirectly tied to the actual audio differences.
- Only the test set explanations are human-verified; the larger training corpus may therefore contain LLM-induced errors or stylistic artifacts that the model can learn to imitate.
- The objective metrics are standard captioning metrics, which are only a rough proxy for comparative explanation quality and may reward generic or linguistically convenient outputs rather than correct audio differences.
- Several reported improvements are modest or mixed, especially when comparing against much larger Qwen-Audio variants; the paper’s framing occasionally overstates the consistency of the gains.
- The human evaluation is informative, but the sample size and statistical reporting are limited; the paper does not provide enough evidence about inter-annotator agreement or confidence intervals.
- The language-only ablation shows that non-audio cues can already achieve nontrivial performance, which raises questions about how much the model truly relies on audio content versus caption patterns.
- The discussion of why certain tiers are easier or harder is not fully convincing: the metric behavior and the linguistic properties of the tiers may confound the interpretation of task difficulty.
- The comparison to a 7B audio-language model is useful but not fully fair as a direct head-to-head method comparison, since ADIFF uses a much smaller backbone and the compute budgets differ substantially.
Nice-to-haves:
- Add a dedicated limitations section discussing dataset construction, evaluation mismatch, and the risks of LLM-generated supervision.
- Report inter-annotator agreement and confidence intervals for the human evaluation.
- Include more adversarial tests such as input-order swapping, semantically similar audio pairs, or out-of-domain audio to verify that the model truly performs comparison rather than templated captioning.
- Provide more qualitative failure cases, especially for Tier 2 and Tier 3 where subtle differences are important.
- If possible, include a stronger task-specific evaluation metric or a learned judge tailored to comparative audio descriptions.
- Clarify the relationship between the proposed tier difficulty and the observed metric trends, especially where Tier 2 appears easiest across models.
- Explore a stronger or larger language backbone for a fairer comparison with large audio-language models.
Novel insights: The paper’s most interesting empirical signal is that a simple language-only or weakly grounded setup can already do reasonably well on some tiers, suggesting that the task as currently instantiated is partly a test of caption-style pattern completion rather than pure audio comparison. At the same time, the cross-projection and position-captioning components do appear to help on the harder, more detailed tiers, indicating that the architecture is extracting at least some benefit from explicit comparison structure. This makes the benchmark promising, but also highlights that future versions of the task should more directly probe audio-dependent reasoning rather than textual regularities.
Potentially missed related work:
- Audio difference learning for audio captioning (Komatsu et al., 2024)
- Audio difference captioning utilizing similarity-discrepancy disentanglement (Takeuchi et al., 2023)
- Image difference captioning methods adapted as structural analogs for audio comparison, such as CLIP4IDC / related difference-captioning work
- Comparative reasoning pretraining for language models (Yu et al., 2023) as a conceptual bridge for structured comparison tasks
- Audio comparison and similarity/retrieval literature that could better motivate pairwise comparison evaluation
Suggestions:
- Add experiments that directly test whether the model depends on the audio inputs, for example by swapping audio order, using matched versus random pairs, or probing outputs on semantically similar clips.
- Evaluate the frozen-LM stage separately from the full finetuning stage to isolate the contribution of cross-projection more cleanly.
- If claiming superiority over larger ALMs, include a more balanced scaling comparison or phrase the claim more narrowly as an efficiency advantage.
- Introduce a task-specific evaluation protocol, even if only as an auxiliary metric, to complement BLEU/METEOR/SPIDEr.
- Expand the human study with more annotated examples and basic reliability statistics.
- Analyze why WavCaps hurts performance more deeply, rather than attributing it only to distribution shift.
- Show more failure cases where the model confuses perceptually similar sounds or hallucinates scene details.
- Clarify the dataset-generation pipeline and distinguish carefully between LLM-derived labels and human verification so the supervision source is transparent.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0, 8.0]
Average score: 7.5
Binary outcome: Accept

=== CALIBRATION EXAMPLE 11 ===

# Final Consolidated Review
Summary: SAM 2 is a strong and well-executed paper that extends promptable segmentation from images to videos via a streaming transformer with memory, and backs this with a large-scale video data engine (SA-V) plus extensive zero-shot evaluation. The paper’s main contributions are clearly articulated and empirically supported: the PVS task definition, the unified image/video model, and the dataset/annotation pipeline. The work is broadly convincing, but several substantive caveats remain around the dependence on internal data, the limited isolation of architectural gains from data gains, and the fact that some key claims (especially about interaction efficiency and 'segment anything' generalization) would benefit from more targeted analysis.
Strengths:
- Clear and well-motivated formulation of Promptable Visual Segmentation (PVS), which cleanly subsumes image segmentation and semi-supervised video object segmentation.
- A unified, streaming video/image segmentation architecture with memory attention, prompt handling, and occlusion prediction that is carefully designed and supported by ablations.
- A very large and diverse video segmentation dataset (SA-V) collected through a model-in-the-loop data engine, with substantial annotation-efficiency gains and quality verification.
- Extensive empirical evaluation across many benchmarks: interactive video segmentation, semi-supervised VOS, and image segmentation, with results reported on 17 video and 37 image datasets.
- Strong ablation suite covering data mix, data quantity, data quality, resolution, memory size, positional encoding, and memory design choices.
- Responsible AI artifacts are unusually thorough for a vision paper: fairness evaluation, model card, dataset card, annotation card, and compute/emissions reporting.
Weaknesses:
- The final model depends materially on an internal licensed video dataset that is not publicly released, which limits reproducibility and makes it harder to attribute how much of the gain comes from public data versus private data.
- The paper does not fully disentangle the contributions of the architecture versus the data scale/mixture; related ablations exist, but a cleaner isolation of architectural novelty would strengthen the claim that the memory design itself is essential.
- The automatic masklets form a large fraction of SA-V, but their error modes and their precise quality relative to manual annotations are not analyzed in depth, leaving some uncertainty about how much model-generated data may reinforce its own biases.
- The interactive evaluation is strong, but it is still oracle-driven click simulation; the efficiency claim would be more compelling with a small user study or a sensitivity analysis to non-oracle clicks.
- The 'segment anything' generalization claim is broad, but evaluation is still anchored in segmentation benchmarks rather than a deeper study of truly novel categories or more diverse out-of-distribution domains.
Nice-to-haves:
- A per-dataset breakdown of the interaction-efficiency gains, to show whether the 3× fewer-interactions result is uniform or driven by a subset of datasets.
- More granular failure-mode analysis on occlusion, disappearance/reappearance, long videos, crowded scenes, and similar-looking objects, especially for the memory mechanism.
- A deeper comparison of automatic versus manual SA-V masklets, including how often auto-generated annotations fail and what kinds of errors dominate.
- A runtime/memory breakdown that includes the overhead of memory attention and bank maintenance, not just the image encoder FPS.
- Additional evaluation on semantic or instance segmentation benchmarks would further test the breadth of the 'segment anything' framing.
Novel insights: The strongest technical insight of the paper is not just that memory helps video segmentation, but that a SAM-like promptable segmentation interface can be lifted into the video domain while preserving the interaction loop: prompts can arrive on any frame, masks can be refined iteratively, and the model can recover from failure using stored object context. The data-engine story is also central: the paper demonstrates that the model is not merely trained on more data, but that model-in-the-loop annotation can substantially change both annotation efficiency and the difficulty/coverage profile of the dataset. This makes the work simultaneously a model paper, a dataset paper, and a human-in-the-loop systems paper.
Potentially missed related work:
- Potentially relevant interactive video segmentation work such as CiVOS and EVA-VOS may be worth comparing more explicitly if not already covered in the main text.
- Recent click-based / promptable video segmentation variants and track-anything style systems may provide useful context for the interactive baseline design.
- Additional long-video or open-world video segmentation benchmarks and methods could help further position the claimed robustness to occlusion and extended temporal context.
Suggestions:
- Report a public-data-only version of the main model to isolate the contribution of SA-V and improve reproducibility.
- Add a more explicit ablation showing the impact of removing the memory bank / memory attention on long videos, not just the existing capacity sweeps.
- Provide a targeted analysis of automatic masklet quality and failure modes, including where model-generated labels help versus potentially hurt.
- Consider a small human-subject study or click-noise sensitivity analysis to validate the practical interaction savings under non-oracle user behavior.
- Broaden evaluation of the 'anything' claim by testing more clearly novel categories or out-of-distribution domains, ideally including semantic/instance segmentation settings.

# Actual Human Scores
Individual reviewer scores: [10.0, 8.0, 8.0, 10.0]
Average score: 9.0
Binary outcome: Accept

=== CALIBRATION EXAMPLE 12 ===

# Final Consolidated Review
Summary: This paper argues that current LLM safety alignment is often "shallow": it mainly changes the model’s behavior in the first few output tokens, making aligned models vulnerable to several jailbreak and fine-tuning attacks. The paper supports this with per-token analyses, then proposes two mitigations: a data-augmentation scheme that trains models to recover from unsafe prefixes and a token-wise constrained fine-tuning objective that better preserves safety during downstream adaptation. The core framing is compelling and the experiments show meaningful robustness gains, but the work is still limited in scope and leaves several open questions about generality, adaptive robustness, and the depth/safety mechanism itself.
Strengths:
- The paper offers a clear unifying framework—"shallow safety alignment"—that plausibly explains why several seemingly different jailbreak and fine-tuning attacks succeed.
- The empirical evidence for early-token concentration is strong: the refusal-prefix statistics, the prefill experiment, and the per-token KL plots all support the central diagnosis.
- The proposed defenses are concrete and technically grounded: the safety-recovery augmentation is simple to implement, and the constrained fine-tuning loss is accompanied by a careful theoretical interpretation.
- The experimental evaluation is broad within its chosen scope, covering multiple inference-time attacks, multiple fine-tuning attacks, and several ablations for both proposed methods.
- The constrained fine-tuning method shows substantial gains in robustness against harmful fine-tuning attacks while largely preserving utility on the reported benchmarks.
- The appendix provides useful implementation details and ablations that improve reproducibility and help clarify how the methods behave under different hyperparameters.
Weaknesses:
- The main characterization results are only shown on two 7B model families, so the claim that shallow safety alignment is a general property of aligned LLMs remains under-supported.
- The data-augmentation method is demonstrated only by further fine-tuning an already aligned model, not by aligning a model from scratch, so it is not yet clear how much of the benefit transfers to a full alignment pipeline.
- Figure 4 and the associated KL analysis show that the augmented model differs more at later tokens, but KL alone does not establish that the divergence is specifically toward safer behavior rather than just a different distribution.
- The defenses are not evaluated against strong adaptive attackers who know the defense design, leaving open whether the gains hold under an informed adversary.
- The constrained fine-tuning objective still has meaningful failure cases, especially for backdoor-triggered attacks and some downstream tasks where utility drops are nontrivial.
- The paper does not probe the model mechanistically, so the notion of "depth" remains mostly distributional and token-based rather than tied to internal representations or reasoning processes.
Nice-to-haves:
- A human evaluation on a subset of safety outputs would complement GPT-4 judging.
- A more explicit cost-benefit discussion would help quantify the training overhead and utility trade-offs of the augmentation and constrained objectives.
- A figure showing the Pareto trade-off between augmentation depth and utility would make the ablation results easier to interpret.
- Examples of failure cases for the proposed defenses would make the limitations more concrete.
- It would be helpful to test the methods under parameter-efficient fine-tuning settings such as LoRA, since those are common in practice.
Novel insights: The strongest insight is that the paper reframes several jailbreak and safety-regression phenomena as manifestations of one underlying optimization shortcut: safety alignment is often achieved by shaping only the first few tokens of a response. This is a useful and memorable lens because it turns a collection of attack papers into a single mechanistic story, and it naturally motivates defenses that either train recovery after an unsafe prefix or explicitly protect early-token distributions during downstream fine-tuning. The idea is conceptually richer than simply "add more safety data," though the paper would be stronger if it connected the token-level story to deeper mechanistic evidence.
Potentially missed related work:
- Zou et al., 2024, on short-circuiting harmful generation, which is conceptually close to the paper’s recovery-based mitigation.
- SafeLoRA (Hsu et al., 2024), representation noising (Rosati et al., 2024), and related defenses for harmful fine-tuning could serve as stronger comparison points for Section 4.
- Vaccine (Huang et al., 2024b) and other post-/during-fine-tuning defenses are relevant baselines; the paper mentions some of these in the appendix, but a fuller in-text comparison would help.
- Works on interpreting alignment or fine-tuning at the token level, including recent analyses of superficial alignment and per-token adaptation dynamics, are related context that could further support the framing.
Suggestions:
- Add experiments on additional model families and, if possible, larger models to strengthen the generality of the shallow-alignment claim.
- Evaluate the augmentation and constrained objective against adaptive attacks that explicitly target the proposed defenses.
- If feasible, run a small-scale alignment-from-scratch experiment with safety-recovery examples to show that deep alignment is not just a post-hoc patch on top of an already aligned model.
- Include stronger baseline comparisons for constrained fine-tuning, especially methods designed to preserve safety during downstream adaptation.
- Add mechanistic probes or representation analyses to distinguish true "deepening" of safety alignment from a distributional shift that merely improves benchmark outcomes.
- Consider a human-evaluation subset and a more explicit utility/compute trade-off analysis to better support deployment claims.
- Extend the constrained objective to LoRA or other parameter-efficient fine-tuning settings to improve practical relevance.
- Show concrete failure cases where the proposed defenses break, especially for backdoor-triggered attacks or longer-horizon harmful continuations.

# Actual Human Scores
Individual reviewer scores: [10.0, 8.0, 10.0, 10.0]
Average score: 9.5
Binary outcome: Accept
