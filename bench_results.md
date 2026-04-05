# ICLR Benchmark Results

Date: 2026-04-04 20:23
Critic/Merger: gpt-5.4-mini (OpenRouter)
Neutral: gpt-5.4-mini, Related Work: gpt-5.4-mini:online (OpenRouter)

## dCtkwjkK0E

- GT: Reject (avg 2.0)
- Predicted: N/A (3.4/10)
- Match: N/A

### Final Review

## Summary
This paper studies active learning for conditional flow matching in continuous-label shape design. The core idea is to derive dataset-level query rules from a piecewise-linear/closed-form view of flow matching, then use them to build a diversity-oriented query rule, an accuracy-oriented query rule, and a weighted hybrid.

## Strengths
- The paper targets a genuinely underexplored problem: active learning for conditional generative models rather than the more common discriminative setting. That is a timely and relevant direction, especially for expensive-label design and simulation tasks.
- The authors propose an interpretable data-centric perspective on the diversity/accuracy trade-off, and the experimental scope is reasonably broad for a pilot study: one synthetic benchmark plus three continuous-condition shape-design datasets.

## Weaknesses
- The theoretical foundation is too informal to support the strength of the claims. The piecewise-linear / closed-form flow-matching analysis is presented as if it yields a general principle, but the derivations are heuristic and rely on restrictive assumptions that are not convincingly validated in the experiments.
- The query objectives are underspecified and partly heuristic. In particular, the role of the RBF label predictor, the clustering used for entropy, and the exact optimization procedure for the multi-term scoring rules are not clearly defined, which makes reproducibility and fair interpretation difficult.
- The evaluation is not strong enough to justify the headline claims. The baselines are mostly classical active learning methods borrowed from discriminative settings, there are no confidence intervals or multi-seed statistics, and the metrics are narrow proxies that do not assess physical validity or downstream usefulness of generated designs.
- The empirical evidence for the central trade-off claim is incomplete. The paper argues that same-label additions increase diversity while label-spread additions improve accuracy, but this is not demonstrated with controlled ablations that isolate label spacing, label density, and the effect of the surrogate label predictor.

## Nice-to-Haves
- A clearer end-to-end active learning curve across budgets and random seeds would make the results much more convincing.
- Stronger comparisons against continuous-condition or generative-model-specific acquisition baselines would help position the method more fairly.
- A sensitivity analysis for the hybrid weight and the RBF/clustering choices would improve confidence in the practical robustness of the method.

## Novel Insights
The paper’s most interesting insight is that, for conditional flow matching, the acquisition problem may be better understood at the dataset level as a tension between label-space coverage and label consistency: one pressure seems to favor finer interpolation coverage, while the other favors tighter label alignment. Framed this way, the work offers a useful lens for thinking about conditional generative active learning, even though the current proof story is too brittle and the empirical support is not yet strong enough to make the mechanism feel established rather than plausible.

## Potentially Missed Related Work
- GALISP (Zhang et al., 2024) — relevant as a prior generative active learning approach for conditional generation, though the paper already cites it; it may deserve a more direct comparison or clearer distinction.
- Generative active learning methods such as VAAL / TAVAAL / BGADL — relevant because the current baselines are mostly discriminative AL methods, while these are closer in spirit to generative-query settings.
- Continuous-condition inverse design / conditional generative design work such as PCDGAN — relevant for positioning the shape-design setting and the continuous label space more carefully.

## Suggestions
- Add a clean algorithm box that specifies candidate scoring, label prediction, clustering/entropy computation, and hybrid selection end to end.
- Report multi-seed means and standard deviations, plus active-learning curves over budget, for all datasets.
- Include ablations that isolate each term in `QD`, the effect of `ω`, and the sensitivity to the RBF surrogate.
- Compare against stronger generative/continuous-condition acquisition baselines, not only standard discriminative active learning methods.
- Validate the central theory with controlled experiments that vary label spacing and label multiplicity independently, so the claimed diversity/accuracy mechanism is actually tested rather than just asserted.

---

## bm3rbtEMFj

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (4.8/10)
- Match: N/A

### Final Review

## Summary
This paper proposes ELMUR, a transformer policy architecture for partially observable long-horizon decision making that adds layer-local external memory, bidirectional token-memory cross-attention, and an LRU-based memory rewrite rule. The paper evaluates on T-Maze, POPGym, and the simulated MIKASA-Robo suite, and reports strong results on these benchmarks, alongside a simple theoretical analysis of exponential forgetting and bounded memory under convex updates.

## Strengths
- The core architectural idea is concrete and well specified: each layer has its own persistent memory track, with explicit read/write cross-attention and an LRU slot update rule. The paper provides algorithmic detail, equations, and a reasonably complete implementation sketch, which makes the method much more tangible than many “memory-augmented” transformer papers.
- The empirical scope is relevant and somewhat broad for the problem setting: synthetic long-horizon T-Maze, a diverse partially observable benchmark (POPGym), and simulated visual manipulation on MIKASA-Robo. The reported gains are especially strong on the memory-heavy manipulation tasks and on the long-horizon T-Maze setting.

## Weaknesses
- The novelty is incremental relative to prior memory-augmented transformers. ELMUR is mainly a combination of known ingredients — segment recurrence, explicit external memory, cross-attention read/write, and an LRU replacement policy — rather than a clearly new learning principle or memory mechanism. The paper does not make a fully convincing case that this is more than a careful recombination of established ideas.
- The headline claims are stronger than the evidence actually supports. “100,000× beyond the attention window” is demonstrated on a synthetic T-Maze setup with very short context and a highly controlled single-cue structure; it is not a general horizon guarantee. Similarly, the large aggregate improvements on MIKASA-Robo are benchmark-specific and should not be framed as broad real-world robustness.
- The theory section is modest and somewhat overstated. The forgetting result is essentially a consequence of repeated convex interpolation, and the boundedness result follows directly from convexity and bounded inputs. These are correct, but they do not establish retrieval quality, task-level credit assignment, or general long-horizon reasoning.
- The evaluation leaves important fairness and robustness questions open. The baseline set is not fully convincing for the robotics setting, especially given recent memory-heavy or long-context alternatives. In addition, the paper does not provide enough evidence that the gains are robust to tuning, compute budget, or alternative settings beyond the chosen hyperparameters.
- Several analyses are informative but remain post hoc. The probing, PCA, and attention visualizations are suggestive, but they are mostly shown on one clean task and do not decisively prove that the memory mechanism itself is the source of the gains rather than capacity, task simplicity, or tuning.

## Nice-to-Haves
- Add scaling curves over memory size, segment length, and number of layers on at least one synthetic and one robotics task, including wall-clock and GPU-memory measurements.
- Expand probing and visualization beyond RememberColor3-v0 to at least one POPGym task and one T-Maze condition.
- Provide a cleaner comparison table against the most relevant long-context and memory baselines, with matched parameter counts and training budgets.

## Novel Insights
The most interesting aspect of the paper is not “external memory” in the abstract, but the specific per-layer design: the model separates token processing from persistent memory and uses LRU to impose a disciplined write policy. This creates a neat mechanism where the model appears to learn a sparse, slot-specific storage pattern for the salient latent variable, which is qualitatively supported by the probing and attention visualizations. However, the same analyses also reveal how task-specific the story is: the evidence is strongest for single-event recall with clean temporal structure, and much weaker as proof of general long-horizon reasoning.

## Potentially Missed Related Work
- xLSTM — relevant as a strong recent long-context recurrent alternative with explicit memory behavior.
- Block-Recurrent Transformers — relevant because they study segment-level recurrence in transformers, close to the paper’s recurrence framing.
- Associative Recurrent Memory Transformer — relevant for explicit recurrent memory access patterns.
- Memformer — relevant as a direct external-memory transformer baseline.
- Transformer-XL / Compressive Transformer — relevant as canonical context-extension baselines for recurrence and memory.
- Stable Hadamard Memory — relevant as a memory-augmented RL baseline in partially observable settings.

## Suggestions
- Tighten the claims in the abstract and introduction so they are benchmark-specific rather than phrased as broad general results.
- Add a controlled ablation that isolates the external memory mechanism from the MoE FFN and from parameter-count changes.
- Report stronger baseline matching details: parameter count, context length, tuning budget, and runtime for all main comparisons.
- Include at least one harder multi-recall or multi-object task to test whether the memory mechanism generalizes beyond single-cue retention.

---

## M14YpuTejd

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (4.3/10)
- Match: N/A

### Final Review

## Summary
This paper critiques the emerging online map-based motion prediction protocol for autonomous driving and argues that the current nuScenes-based setup is misleading. It proposes OMMP-Bench, which introduces a spatially disjoint data split, revises evaluation to focus on moving non-ego agents, and adds a simple image-feature baseline to help agents outside the online map’s perception range.

## Strengths
- The paper identifies a real and timely evaluation problem in a nascent area: the default two-stage protocol can create a train/validation mismatch because the map model is evaluated on its training distribution during motion-model training but not at validation time. The argument is plausible and backed by Table 1 plus the reported spatial overlap in nuScenes.
- It also correctly points out that ego-only evaluation is a poor proxy for motion prediction, and that static agents make the metric less discriminative. The revised evaluation on moving non-ego agents, with separate close/far reporting, is a meaningful protocol correction rather than a cosmetic change.
- The benchmark-oriented contribution is concrete rather than purely conceptual: the paper defines a new split (367/397/86 scenes), evaluates existing online-map-based methods under the new protocol, and shows that the choice of map formulation and downstream conditioning materially affects results.

## Weaknesses
- The benchmark is still narrowly tied to a single dataset and a single task instantiation, so the paper’s claims are stronger than its evidence. Most conclusions are based on nuScenes only, with no additional dataset, no held-out geographic benchmark, and no sensitivity analysis showing that the new split is uniquely justified rather than one reasonable manual partition.
- The new split and revised metrics are sensible, but the paper does not convincingly isolate which correction drives the reported improvements. The gain in downstream motion metrics could come from changing data difficulty, removing leakage, or altering the agent subset; without stronger ablations, the benchmark diagnosis remains incomplete.
- The “boundary-free” baseline is only a heuristic fix and is under-specified in the main paper. Eq. (1) and the surrounding text do not provide enough detail to reproduce the feature extraction and fusion pipeline, and the paper does not compare against simpler alternatives such as enlarging map range or using BEV features more systematically.
- The recommendation to exclude static agents and always feed all map elements is not fully justified as a general benchmark rule. These are defensible choices for a focused analysis, but the paper overreaches in presenting them as default solutions without showing robustness across more model families or task settings.

## Nice-to-Haves
- Add a detailed appendix-style description of the exact spatial partitioning procedure, including scene selection criteria and overlap statistics, to make the new split auditable.
- Report uncertainty estimates or multi-seed variance for the main tables, since several improvements are modest.
- Include qualitative examples showing failure cases under the default protocol and the effect of the image-feature baseline on far agents.

## Novel Insights
The most interesting insight is not the baseline itself, but the paper’s reframing of what can go wrong in online-map-based forecasting: the benchmark can be misleading even when each individual component seems reasonable. In particular, the paper surfaces a three-way interaction between split design, map perception range, and evaluation choice, showing that good ego-only numbers can hide both leakage and missing-context failures for other agents. That is a genuinely useful diagnosis for the field, even if the proposed fixes remain more protocol corrections than methodological advances.

## Potentially Missed Related Work
- ViP3D and PIP — relevant as broader end-to-end prediction pipelines where upstream perception errors also affect forecasting, though the paper already mentions them and intentionally excludes that setting.
- UniAD and VAD — relevant for joint perception-prediction-planning evaluation, useful as adjacent protocol baselines rather than direct competitors.
- None identified beyond the paper’s cited online mapping and motion prediction references for the core benchmark setting.

## Suggestions
- Isolate the effects of the split, the moving-agent metric, and the range mismatch with targeted ablations or controlled oracle-map experiments.
- Add at least one broader generalization study: either another dataset or a carefully defined held-out geographic split on nuScenes.
- Provide full implementation details for the image-feature baseline and compare it against simpler range-extension or pooling-based alternatives.

---

## D5PJX02Jki

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (4.8/10)
- Match: N/A

### Final Review

## Summary
This paper revisits RoPE in its complex-valued form and proposes RoPE++, which restores the discarded imaginary component as an additional attention stream. The authors present two variants: RoPE++EH, which halves KV cache and parameters at fixed head count, and RoPE++EC, which keeps cache fixed and doubles heads, and report gains on short-context, long-context, and extrapolation-style benchmarks at 376M, 776M, and 1.5B scale.

## Strengths
- The core idea is technically clean and easy to state: the imaginary part of RoPE’s complex attention is not a new positional encoding in disguise, but a directly derived companion stream that preserves the same absolute/relative RoPE structure. The derivation in Section 3 is coherent and the two configurations are practically motivated.
- The paper does more than just report benchmark gains: it offers a plausible mechanistic story for why the imaginary stream helps long-context behavior, and backs this with characteristic-curve analysis, attention visualizations, and a noise-injection study showing that corrupting imaginary attention hurts long-context performance substantially.

## Weaknesses
- The main claim is under-isolated experimentally. RoPE++EC changes head count while RoPE++EH changes cache and parameter budgets, so the reported gains conflate the effect of “imaginary attention” with capacity, projection, and head-allocation changes. Without matched-capacity/matched-FLOP ablations, it remains unclear how much of the improvement is actually due to the imaginary component itself.
- The evaluation is still too benchmark-centric for the strength of the claim. Most long-context evidence comes from synthetic retrieval/reasoning suites like RULER and BABILong, and the short-context comparisons are broad but only indirectly tied to the proposed mechanism. There is little evidence on realistic long-document tasks or downstream usage scenarios where long-context capability actually matters.
- The paper’s extrapolation narrative is stronger than the evidence supports. It correctly admits that RoPE++ is not plug-and-play length extrapolation, yet parts of the text still suggest that the method “solves” or largely removes extrapolation issues for certain dimensions. The actual evidence is more limited: it indicates reduced distribution shift and slower perplexity growth, not a general guarantee of robust out-of-window generalization.
- The mechanistic analysis is suggestive but not decisive. Attention plots and Gaussian perturbations indicate that imaginary heads behave differently and seem more important for long-context performance, but they do not establish a causal link between sine-shaped characteristic curves and better long-range retrieval. The “dominant role” claim is still somewhat anecdotal.

## Nice-to-Haves
- A tighter comparison against more recent long-context baselines under one unified training recipe would make the gains easier to interpret.
- A compact algorithm box and clearer implementation details for how RoPE++ integrates with MHA/GQA would improve usability.
- Confidence intervals or multi-seed reporting would help, especially for the smaller short-context gains.

## Novel Insights
The most interesting insight is that RoPE’s discarded imaginary term is not merely redundant algebra: it produces a qualitatively different positional bias, one that appears to emphasize global or long-range context more than the real component. This suggests RoPE’s standard real-only implementation may be leaving useful positional structure on the table, and that long-context improvements can come not only from modifying the frequency schedule or interpolation strategy, but from recovering information already latent in the complex formulation.

## Potentially Missed Related Work
- **Barbero et al., 2024, “Round and round we go! what makes rotary positional encodings useful?”** — relevant for deeper analysis of RoPE behavior and could help contextualize the interpretation of real vs. imaginary components.
- **Wang et al., 2025; Lee et al., 2022** — relevant to complex-valued neural network formulations and the relationship between complex computation and attention, which the paper only briefly distinguishes from its own setting.
- **Recent long-context head-specialization / sparse-attention work such as DuoAttention and MInference** — relevant because RoPE++ changes head structure and claims efficiency/behavioral specialization.

## Suggestions
- Add a controlled ablation where the only difference is whether the imaginary component is used, while matching FLOPs, parameter count, head count, and KV cache as closely as possible.
- Include at least one realistic long-context downstream evaluation beyond synthetic retrieval benchmarks.
- Quantify robustness across multiple seeds and report variance, especially on the smaller short-context margins.
- Clarify in the main text that RoPE++ improves trained long-context performance and partial extrapolation behavior, but is not an inference-time plug-and-play extrapolation method.

---

## ahpO7S1Ppi

- GT: Reject (avg 3.5)
- Predicted: N/A (4.2/10)
- Match: N/A

### Final Review

## Summary
This paper proposes Pctx, a personalized context-aware tokenizer for generative recommendation. Instead of assigning each item a fixed semantic ID, the method conditions tokenization on user history so the same item can receive different IDs under different contexts, then trains a GR model over these personalized IDs with augmentation and multi-facet decoding.

## Strengths
- The core motivation is real and timely: static semantic IDs in GR do impose a single item-similarity geometry, and the paper makes a plausible case that this is mismatched to user-specific intent.
- The method is concrete rather than purely conceptual. It gives a full pipeline with context encoding, item-wise clustering, fusion with text features, quantization, redundancy merging, and training/inference adjustments.
- The empirical results are consistent across three datasets, and the ablations support that the personalized context, clustering/merging, and augmentation/multi-facet decoding all contribute to performance.

## Weaknesses
- The contribution is still fairly incremental. The key idea is a personalized extension of context-aware tokenization, but the paper does not establish a new principle beyond “use longer user history to drive token assignment.” Compared with recent GR tokenization work, this feels like a well-engineered refinement rather than a major conceptual step.
- The method is heuristic-heavy and under-specified in the main paper. Critical details such as centroid allocation, token merging rules, quantization behavior, and exact inference aggregation are pushed to appendices/code, making it hard to judge correctness and reproducibility from the paper alone.
- The evaluation is not strong enough for the breadth of the claims. All experiments are on three closely related Amazon review categories with standard leave-one-out next-item prediction; there is no runtime/memory analysis, no long-tail or cold-start breakdown, and no test of whether personalization helps most when it should.
- The interpretability evidence is weak. The GPT-4o-based “explainability” experiment is interesting, but it is subjective and not a substitute for a more grounded analysis showing that semantic IDs really correspond to distinct, stable user facets.

## Nice-to-Haves
- A clearer complexity and scalability report, including tokenizer construction cost, inference latency, and vocabulary growth.
- A sharper analysis of head vs. tail items and sequence length, since the method’s benefits likely depend on history richness.
- More principled evidence that different semantic IDs for the same item correspond to genuinely different user interpretations, not just different clusters.

## Novel Insights
The main interesting insight is that in GR, tokenization is not just a compression trick but an implicit similarity prior: fixed semantic IDs force the model to treat all users as if they share the same notion of item relatedness. Pctx’s useful twist is to move that prior into user history, so the same item can occupy different regions of the token space depending on context; this is a sensible direction and the case study illustrates it well. That said, the paper currently shows more that this is a plausible engineering improvement than that it reveals a fundamentally new mechanism for generative recommendation.

## Potentially Missed Related Work
- COST (contrastive quantization based semantic tokenization) — relevant because it is a recent tokenization method for generative recommendation and a more direct comparator than generic sequential models.
- RK-Means / learnable item tokenization variants — relevant because the paper’s gains may depend on the choice of quantizer, and these methods test the tokenization side more directly.
- Multi-identifier item tokenization work (e.g., MTGRec) — already discussed in spirit, but still relevant as a nearby line of work on one-to-many item IDs.

## Suggestions
- Add a controlled ablation that separates the effect of personalized context from item features and from simple extra token diversity.
- Report efficiency metrics and item-frequency breakdowns to show where personalized tokenization helps and what it costs.
- Move the core tokenization algorithm, centroid allocation, and merging procedure into a concise pseudo-code block in the main paper so readers can actually verify the method without reverse-engineering the appendix.

---

## X2yzXtH4wp

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (4.3/10)
- Match: N/A

### Final Review

## Summary
This paper introduces Ambig-SWE, an interactive variant of SWE-Bench Verified designed to study how LLM agents handle underspecified software-engineering tasks. It decomposes the problem into three stages — detecting missing information, asking clarifying questions, and integrating answers into a successful patch — and reports that interaction can substantially improve resolve rates, though most models remain brittle in recognizing when clarification is needed and in using it efficiently.

## Strengths
- The paper targets an important and timely failure mode for agentic LLMs: underspecified instructions in tool-using software engineering workflows. The problem is practically relevant and the benchmark framing is easy to understand.
- The three-way decomposition into detection, clarification-question quality, and downstream task completion is a useful diagnostic contribution. It goes beyond a single end-to-end metric and helps isolate where models fail.
- The authors evaluate both proprietary and open-weight models and provide qualitative examples of question/answer trajectories. This makes the behavioral analysis more concrete than a pure leaderboard-style benchmark paper.
- The paper is transparent about several limitations and provides code, prompts, and statistical tests, which is a real plus for a benchmark-style submission.

## Weaknesses
- The benchmark is built from GPT-4o-generated underspecified summaries of already verified SWE-Bench issues, not naturally occurring underspecified requests. That synthetic construction is a major external-validity weakness: the paper does not convincingly show that these summaries induce the same clarification behavior as real user prompts, only that they are shorter and less detailed.
- The interactive setup is confounded by design choices that make the task easier to interpret than real user interaction. In particular, the proxy can reveal file locations in addition to task details, and the proxy is far more cooperative and deterministic than real users. This makes it hard to know how much of the gain comes from genuine underspecification handling versus easier localization or an idealized user.
- The question-quality analysis is suggestive but still fairly weak as evidence. Cosine distance over embeddings and an LLM judge are only indirect proxies for “good questions,” and the paper itself notes that the embedding metric weights all information equally. As a result, the core claim that some models ask better questions than others is plausible, but not firmly established.
- Several conclusions are overstated relative to the evidence. The paper repeatedly gestures toward “real-world” underspecification and broader agentic workflows, but the experiments are confined to a controlled SWE-Bench-derived setting with synthetic hiding. The results are interesting, but the generalization beyond this benchmark is not yet demonstrated.
- The detection experiment appears highly prompt-sensitive. The large swings across Neutral/Moderate/Strong encouragement suggest the benchmark is measuring a mix of underlying capability and prompt compliance. That is a useful diagnostic finding, but it also means the detection task is brittle and not yet a clean capability measure.

## Nice-to-Haves
- Add a stronger validation study showing that the generated underspecified issues actually elicit similar clarification needs to naturally underspecified GitHub issues, ideally with human judgments.
- Separate navigational recovery from true requirement clarification more cleanly, since the interaction setting gives the proxy file-location knowledge and can blur the interpretation of gains.
- Report confidence intervals and effect sizes alongside the Wilcoxon tests, especially given the synthetic benchmark construction and the subset evaluation for Claude Sonnet 4.
- Include at least a small analysis on naturally underspecified issues, even without paired ground truth, to complement the synthetic benchmark.

## Novel Insights
The most interesting insight is that interaction helps, but not in a simple “ask more questions” sense. The paper’s qualitative analyses suggest that stronger models, especially Claude Sonnet variants, benefit from an exploration-first strategy: they inspect the codebase, then ask only for what cannot be inferred, while weaker or more rigid models either ask too much, ask too early, or fail to adapt after receiving answers. This means the key bottleneck is not just whether models can extract information, but whether they can decide what to ask, when to ask it, and how to incorporate the answer without wasting turns.

## Potentially Missed Related Work
- ClarifyGPT — relevant prior work on intention clarification for code generation.
- Learning to Ask / unclear-instruction clarification work — relevant for policies that trigger clarification when instructions are ambiguous or underspecified.
- AmbigNLG / CLAMBER — relevant benchmark-style work on ambiguity detection and clarification.
- Test-driven interactive code generation / interactive user-intent formalization — relevant prior interactive code-generation baselines that should be compared more directly.
- None identified beyond these clarification-oriented and interactive code-generation lines of work.

## Suggestions
- Add a controlled ablation that varies the amount and type of information removed from the issue summary, and compare against at least one non-LLM or retrieval-based clarification/localization baseline.
- If possible, validate the benchmark on a subset of naturally underspecified SWE-Bench issues, even without paired gold completions, to show the synthetic setup is not purely an artifact of summarization.
- Make the protocol table more explicit in the main paper: what the model sees, what the proxy sees, what the proxy may reveal, and how many turns are allowed in each condition.

---

## RpDJz00zNh

- GT: Reject (avg 4.5)
- Predicted: N/A (4.3/10)
- Match: N/A

### Final Review

## Summary
This paper proposes ConciseHint, a test-time method for reducing the verbosity of large reasoning models by repeatedly injecting a concise hint during generation, rather than only prompting before reasoning or fine-tuning the model. It also introduces a learned-hint variant, ConciseHint-T, and two simple heuristics: an adaptive injection interval based on current output length and a dynamic injection position intended to balance efficiency and accuracy.

## Strengths
- The core idea is practical and easy to understand: intervene during generation to steer the model toward shorter reasoning traces, instead of relying only on pre-reasoning prompts or retraining. The method is modular and the paper shows it can be combined with several existing efficiency methods.
- The evaluation is broader than token-count-only papers: it includes multiple models, several benchmarks of varying difficulty, ablations over interval and injection position, controllability via embedding interpolation, latency analysis, transition-word statistics, and extra results on CommonsenseQA and HumanEval.

## Weaknesses
- The main method is still largely heuristic. The adaptive schedule uses current output length as a proxy for difficulty, and the injection-position rule is hand-crafted. The paper provides intuition and ablations, but no principled justification for why these particular formulas should work or when they should fail. That matters because the method’s success depends heavily on these design choices.
- The strongest claims outpace the evidence. The paper frames ConciseHint as a largely unexplored “in-reasoning intervention” paradigm and suggests seamless plug-in compatibility, but the empirical support is limited to a small set of models, tasks, and baselines. In particular, the headline efficiency claims are mostly token-usage based; the actual end-to-end efficiency story is more complicated because repeated injection introduces prefilling overhead and the latency analysis is pushed to the appendix.
- Reproducibility and evaluation details are incomplete. Important protocol specifics are not fully spelled out in the main paper: exact prompt formatting, decoding and stopping behavior, how baselines were calibrated under identical settings, and how the concise training data for ConciseHint-T was built. This makes it difficult to fully audit whether the gains come from the method itself or from implementation choices.
- The learned-hint variant is under-validated. Training on concise GSM8K-derived data and then claiming transfer to harder out-of-domain tasks is plausible, but the paper does not sufficiently isolate whether ConciseHint-T learns a transferable “concise prior” or simply behaves like a tuned soft prompt on the tested setup.

## Nice-to-Haves
- Report full accuracy–efficiency Pareto curves, not just selected operating points, so readers can see whether the method truly improves the frontier.
- Add variance estimates or multiple-seed results, especially because some improvements are modest and the method is sensitive to hint strength and interval choice.
- Provide a more explicit failure analysis for cases where aggressive hinting causes premature stopping or degraded reasoning on harder queries.

## Novel Insights
The most interesting aspect of the paper is that it shifts the efficiency question from “how do we compress reasoning before generation?” to “can we modulate reasoning while it is being generated?” That is a legitimate and potentially useful direction, and the adaptive interval plus dynamic insertion idea does help avoid the obvious failure mode of over-hinting on hard problems. However, the evidence also suggests that the method’s apparent gains are tightly coupled to carefully chosen heuristics and to the specific serving setup, so the contribution is more of a practical control mechanism than a deep new reasoning principle.

## Potentially Missed Related Work
- TokenSkip — relevant as a controllable CoT compression method with a closer emphasis on test-time efficiency.
- CoT-Valve — relevant for length-compressible reasoning and budget-aware control.
- Thinkless / AdaptThink — relevant because they target when to think / adaptive reasoning control.
- DAST — relevant as a difficulty-adaptive reasoning method.
- AlphaOne — relevant as a test-time reasoning strategy with efficiency tradeoffs.
- O1-Pruner — relevant because it directly addresses pruning / length harmonization for o1-like reasoning.
- Dynamic early exit methods — relevant as a closely related efficiency baseline class.

## Suggestions
- Add a unified evaluation on the strongest and most relevant efficiency baselines under the same models, prompts, decoding settings, and compute budget.
- Include a clearer protocol for ConciseHint-T: dataset construction, size, filtering, train/validation split, and whether any benchmark-specific leakage is possible.
- Report wall-clock latency and throughput alongside token counts in the main paper, since the method repeatedly edits context and token reduction alone is not enough to justify efficiency claims.
- Replace the current heuristic justifications with a stronger sensitivity analysis over difficulty proxies, interval schedules, and insertion positions, so readers can judge when the method is robust and when it is brittle.

---

## ZNAY3ivd62

- GT: Reject (avg 4.0)
- Predicted: N/A (4.8/10)
- Match: N/A

### Final Review

## Summary
This paper presents GUI-Spotlight, an iterative GUI grounding system that uses explicit tools (`extract`, `crop`, `find color`) to progressively zoom in on a target region and then output click coordinates. The method is trained in stages with SFT followed by a modified GSPO-style RL procedure, and the authors report competitive results on ScreenSpot-Pro, UI-Vision, and OSWorld-G relative to much larger or more data-heavy baselines.

## Strengths
- The core idea is practically relevant and internally coherent: the model maintains image offsets across turns, uses explicit cropping/focus tools, and can iteratively refine a screen region before clicking. The pipeline is easy to understand at a high level and directly targets the hardest part of GUI grounding: precise localization on cluttered high-resolution screens.
- The empirical sample-efficiency claim is genuinely interesting if it holds up: the paper reports 52.8% on ScreenSpot-Pro with only 18.5K training samples, outperforming several 7B baselines trained on orders of magnitude more data. The ablations on RL variants and reward shaping also suggest the authors did nontrivial engineering to stabilize multi-turn tool-use training.

## Weaknesses
- The central causal claim is not cleanly isolated. The reported gains come from a bundle of ingredients: explicit tools, curated high-resolution data, aggressive filtering, staged SFT, modified RL, and reward shaping. As written, it remains unclear how much improvement is due to adaptive iterative focus refinement itself versus better data curation and training heuristics, which weakens the scientific claim.
- The method and training objective are under-specified in several important places. The modified GSPO objective, sample masking, bucketed sampling, and stopping behavior are hard to reconstruct precisely from the main text, and the exact prompting / serialization of multi-turn tool interactions is not fully clear. For an agentic RL paper, this is a real reproducibility problem.
- The evaluation is not strong enough to fully justify the breadth of the claims. Many baseline numbers are taken from leaderboards or prior reports rather than re-run under identical conditions, and there are no error bars or multi-seed variance estimates. Given that several improvements are modest in absolute terms, this makes the headline comparisons less convincing than they appear.
- The ablation story misses the most important isolations. The paper studies RL variants and reward weights, but does not convincingly separate the value of each tool, the value of the new dataset, and the value of iterative tool use versus simpler iterative inference heuristics. Without these controlled comparisons, the contribution risks reading as a systems recipe rather than a well-established algorithmic advance.

## Nice-to-Haves
- A clearer failure-mode analysis by UI type, target size, occlusion, and distractor similarity would strengthen the paper substantially.
- Tool-usage statistics such as average number of turns, success-conditioned stop rates, and where the policy tends to fail would make the “adaptive spotlighting” claim much more interpretable.
- A more explicit discussion of dataset overlap / contamination risks would improve confidence in the reported benchmark gains.

## Novel Insights
The most interesting aspect of the paper is not simply that it uses tools, but that it treats GUI grounding as a sequential refinement problem where the model is trained to decide both where to look and when to stop. That framing is promising for dense screens, where a single-shot coordinate predictor can fail badly, and the results suggest that a small number of well-chosen visual tools may be enough to make a 7B model competitive with much larger systems. However, the paper currently sells this as a clean method contribution while the evidence actually points to a stronger combination of engineering, data curation, and RL stabilization than to a sharply isolated algorithmic breakthrough.

## Potentially Missed Related Work
- None identified — the related work already covers most of the obvious adjacent GUI grounding, iterative refinement, and RL-for-grounding directions.

## Suggestions
- Add a controlled baseline using the same 18.5K curated samples with the same backbone but without tools, to directly test whether iterative focus refinement is the main driver of the gain.
- Report an ablation removing each tool individually and a simple iterative crop/re-query baseline under matched inference budget.
- Clarify the full RL objective, masking logic, and stopping rule in one unambiguous algorithmic specification, and include multi-seed variance for the main results.

---

## xFo13SaHQm

- GT: Accept (Poster) (avg 6.5)
- Predicted: N/A (5.2/10)
- Match: N/A

### Final Review

## Summary
This paper introduces **WithAnyone**, a multi-identity image generation system aimed at reducing the common “copy-paste” failure mode in identity-consistent generation. The main contributions are a large paired dataset (**MultiID-2M**), a new benchmark (**MultiID-Bench**) that explicitly measures the trade-off between identity fidelity and copy-paste artifacts, and a training recipe combining paired tuning, a ground-truth-aligned ID loss, and an extended-negative contrastive loss.

## Strengths
- **The paper identifies a real failure mode that standard identity metrics can hide.** The argument that maximizing face similarity can incentivize near-duplication rather than controllable identity preservation is well motivated, and the benchmark/metric design is directly aligned with that concern.
- **The dataset and benchmark are substantial and potentially useful to the field.** MultiID-2M is large-scale, with roughly 500k paired multi-ID images, 1.5M unpaired group photos, and ~1M reference images across ~3k identities; MultiID-Bench further provides a standardized testbed with explicit copy-paste evaluation rather than relying only on Sim(Ref).

## Weaknesses
- **The central empirical claim is still overly benchmark-dependent.** The paper’s “break the trade-off” narrative rests heavily on its own proposed benchmark and copy-paste metric, both tightly coupled to the same identity embedding family and dataset construction pipeline. That makes the conclusion more convincing as a result on this benchmark than as a broad statement about identity generation in general.
- **The evaluation is not yet as rigorous as the claim strength suggests.** Baselines are heterogeneous (open-source, API-based, and differently configured systems), variance/confidence intervals are absent from the main tables, and the benchmark is relatively small compared with the dataset scale. This weakens confidence in the exact ranking gaps and in strong SOTA framing.
- **Dataset and label quality remain under-characterized.** Identity assignment in MultiID-2M relies on ArcFace clustering and thresholded retrieval in crowded group photos, but the paper does not provide enough evidence on retrieval accuracy, label noise, or failure modes. Since the training signal depends on these pairs, this is a substantive omission.
- **Generalization is not convincingly established.** The method and benchmark are heavily centered on public figures and web-collected celebrity imagery. The paper shows some qualitative examples beyond the easiest cases, but it does not adequately demonstrate robustness to out-of-domain identities, harder occlusion settings, or non-celebrity user photos.

## Nice-to-Haves
- A fuller breakdown of how much each ingredient contributes: paired tuning, GT-aligned ID loss, extended negatives, and the final quality/style tuning stage.
- More systematic analysis of where the method fails: by number of identities, pose/expression change, reference quality, and occlusion.
- A clearer sensitivity study for the copy-paste metric and the negative-pool size, to show the results are not brittle to design choices.

## Novel Insights
The most interesting conceptual contribution is the paper’s reframing of identity preservation as a **two-objective tension**: high face similarity is not automatically desirable if it is achieved by copying the reference too literally. The proposed benchmark and loss design make this tension explicit, and the GT-aligned landmark trick is a clever engineering choice that allows identity supervision to be applied more broadly during flow-matching training. That said, the paper’s novelty is primarily in this re-framing and in the paired-data recipe, not in a fundamentally new generative paradigm.

## Potentially Missed Related Work
- **ID-Aligner / reward-based identity-preserving generation** — relevant as another line of work using identity feedback or reward learning for face customization.
- **Omni-ID / identity representation work for generative tasks** — relevant because the paper’s core depends on how identity is embedded and compared.
- **FaceID-6M / other large face customization datasets** — relevant as neighboring dataset efforts that help contextualize the scale and limitations of MultiID-2M.

## Suggestions
- Provide a stronger, apples-to-apples benchmark protocol with identical prompts, reference selection, and as many competitive baselines as possible under matched settings.
- Add a manual audit or quantitative study of identity-label noise in MultiID-2M, especially for multi-face group photos.
- Include external generalization tests on non-celebrity or out-of-distribution identities, and show failure cases alongside successes.
- Report variance or confidence intervals for the main benchmark results, and separate algorithmic gains from dataset gains more cleanly.

---

## j3htU5i01r

- GT: Reject (avg 4.0)
- Predicted: N/A (4.4/10)
- Match: N/A

### Final Review

## Summary
This paper proposes a compositional meta-learning framework that models tasks as sequences of reusable neural modules, with a learned gating network capturing how modules are combined. Training is done by maximizing a probabilistic generative model over task episodes, and test-time adaptation is performed by particle-filter inference over module sequences rather than gradient updates. The paper demonstrates recovery of ground-truth modules and transitions on synthetic rule-learning and motor-learning tasks, and shows one-shot inference under sparse feedback.

## Strengths
- The core idea is clean and genuinely appealing: separate within-module computation from between-module sequencing, then use inference over the learned task grammar at test time. The probabilistic framing is coherent and well aligned with the compositional meta-learning problem.
- The experiments do support the main proof-of-principle claim. In both rule learning and motor learning, the paper shows recovery of interpretable modules and history-dependent transition structure, plus successful held-out task inference with sparse feedback and no test-time parameter updates.

## Weaknesses
- The evaluation is narrow and heavily synthetic. The tasks are deliberately constructed to match the proposed inductive bias, with known module identities, fixed segment lengths, and clean compositional structure. This is fine for a proof of concept, but it does not justify the broader claims implied in the introduction and discussion.
- The baselines are not strong enough for the paper’s central claim. The comparisons are mainly against generic RNNs, task-identity RNNs, and gradient-based adaptation baselines. For a paper claiming that probabilistic compositional inference is the key ingredient, it is a notable omission that there is no closer comparison to latent-task, modular routing, or other inference-based compositional methods.
- The method’s practical cost and robustness are under-characterized. Test-time inference relies on particle filtering, and training appears sensitive to initialization and particle choices, but the paper does not provide a systematic study of particle count, inference variance, or runtime tradeoffs. That matters because the headline claim is “one-shot” adaptation without updates, which still incurs sequential inference cost.
- Several claims are broader than the evidence. The paper repeatedly gestures toward “any problem with sequential modular structure,” but the actual evidence is limited to two controlled settings with explicit compositional regularities. The work is best read as a proof-of-principle, not as a generally validated meta-learning framework.

## Nice-to-Haves
- A cleaner separation between the core method and task-specific engineering in the motor-learning setup would help. The motor domain introduces several special choices, such as resetting module state and using guided particle filtering, which are reasonable but make it harder to assess the generality of the base formulation.
- A more explicit robustness study across seeds and modest task variations would improve confidence, especially given the paper’s own acknowledgment of local minima and training instability.

## Novel Insights
The most interesting aspect of the paper is not just that it learns reusable modules, but that it treats adaptation itself as structured inference over a learned compositional grammar. That is a meaningful shift from the usual meta-learning view of “learning a good initialization” or “learning a fast updater,” because it turns sparse test-time feedback into a hypothesis-testing problem over module sequences. The strongest evidence in the paper is precisely where that framing matters most: when feedback is intermittent, the posterior over module histories propagates multiple candidate continuations until observations collapse the ambiguity.

## Potentially Missed Related Work
- Routing Networks / modular routing methods — relevant because they also learn to select among reusable computation blocks, though typically with test-time parameter adaptation or different training objectives.
- Recurrent Independent Mechanisms — relevant as a modular recurrent architecture that learns reusable dynamical components across tasks.
- Probabilistic task inference / latent task-variable meta-learning methods — relevant because the paper’s main contribution is an inference-based view of meta-learning.
- Modular Meta-Learning / Alet et al. — relevant because it is a closely related non-gradient search over module configurations, and serves as a natural conceptual comparator.

## Suggestions
- Add direct comparisons to closer modular and inference-based baselines, not just generic RNN and gradient-adaptation controls.
- Report sensitivity to particle count, runtime, and inference stability, since the test-time procedure is sequential inference rather than a cheap forward pass.
- Include at least one harder benchmark with less hand-crafted compositional structure to better support the paper’s generality claims.
- Make the distinction between the core probabilistic framework and the motor-task-specific implementation changes more explicit.

---


# Summary

Papers: 10 | Accuracy: N/A
## WhO6Km5Rku

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (2.4/10)
- Match: N/A

### Final Review

## Summary
This paper proposes QubitCache, a hybrid KV-cache compression scheme that keeps a small set of “critical” tokens in classical memory while encoding the remaining tokens’ attention patterns into quantum-inspired amplitude states. The goal is to reduce KV memory while preserving long-context performance, with reported gains on several benchmarks and a strong emphasis on multi-hop reasoning and relational structure preservation.

## Strengths
- The paper tackles a genuinely important problem for LLM inference: KV-cache growth is a real deployment bottleneck, and any method that can materially reduce memory without destroying long-context quality is practically relevant.
- The high-level idea is appealing and consistent with known properties of transformers: instead of treating compression as only token eviction, the method tries to preserve attention structure and uses a hybrid retained-token plus reconstruction strategy. The ablations do at least suggest that the attention-based selection matters more than random retention.

## Weaknesses
- The paper substantially overstates its scientific claims. It repeatedly implies “logarithmic compression beyond classical information-theoretic limits” and strong quantum advantage, but the implementation is a classical simulation with heuristic reconstruction. As written, the quantum framing reads more like metaphor than a demonstrated technical advantage.
- The method description is not yet precise enough to be reproducible or convincing. Key equations are hard to interpret, the exact tensor-level integration with standard KV attention is unclear, and several implementation claims in the text remain underspecified.
- The empirical story is not clean enough to fully support the strongest claims. The paper reports many numbers, but the benchmark mix is heterogeneous, the protocol is not always easy to audit, and there are no confidence intervals or significance tests. For a paper making fairly large claims about near-lossless 7× compression, this is a real gap.
- The core novelty is weaker than the presentation suggests. The system appears to combine known ingredients—heavy-hitter style token retention, sliding/segment management, interpolation, and amplitude-encoding-inspired compression—without yet establishing that the quantum-inspired component is necessary rather than decorative.

## Nice-to-Haves
- A clearer, step-by-step pseudocode version of the full inference pipeline, including what is stored per layer/head and how updates happen over time.
- More explicit reporting of latency, throughput, and overhead, not just memory and quality.
- A cleaner separation between what is theoretically claimed, what is classically simulated, and what would only matter on actual quantum hardware.

## Novel Insights
The most interesting idea here is not “quantum” per se, but the reframing of KV-cache compression as preservation of relational attention structure rather than preservation of raw tokens. That framing does align with some interpretability evidence and with the paper’s own ablations showing that attention-based selection is better than random selection. However, the current method does not yet demonstrate that quantum amplitude encoding is the essential mechanism behind those gains; at present, the performance may largely come from a strong hybrid heuristic plus reconstruction, not from a principled quantum-inspired advance.

## Potentially Missed Related Work
- PyramidKV / Quest-style long-context KV compression methods — relevant as stronger recent comparators for long-context cache compression beyond simple token eviction.
- Other classical low-rank or probabilistic cache approximation methods — relevant because they may match the memory-performance tradeoff without relying on quantum-inspired framing.

## Suggestions
- Tone down the quantum claims and present the method more honestly as a classical probabilistic attention-compression scheme inspired by amplitude encoding unless a real quantum-specific advantage can be shown.
- Add a rigorous, implementable algorithm description with pseudocode and exact shapes/operations.
- Report latency, throughput, peak memory, and variance across seeds alongside quality.
- Add stronger ablations isolating the effect of attention-based selection, reconstruction, segment size, and retention ratio.
- Include longer-context evaluations and more competitive recent baselines to make the main claim credible.

---

## ngOOlatCK6

- GT: Reject (avg 5.3)
- Predicted: N/A (5.4/10)
- Match: N/A

### Final Review

## Summary
This paper studies a single-node causal bandit setting with conditional interventions, where the agent knows the causal DAG but not the SCM. Its main contribution is a clean graph-theoretic characterization of the minimal set of candidate intervention nodes that must contain the optimal one: the LSCA closure of the parents of the target node, computed by a linear-time algorithm (C4). The paper then shows that pruning to this set can reduce search space and improve a simple UCB-based bandit’s regret in experiments.

## Strengths
- **The core characterization is nontrivial and elegant.** The paper identifies the minimal globally interventionally superior set with the LSCA closure of Pa(Y), which is a concrete and structurally meaningful answer to the search-space problem.
- **The algorithmic result is genuinely useful.** C4 computes the closure in \(O(|V|+|E|)\) time, so the theoretical reduction is not just existential; it is efficient enough to matter on large DAGs.
- **The paper connects theory to downstream decision-making.** The experiments do show that pruning the candidate node set can improve a UCB-style method’s regret and substantially reduce the number of nodes considered.

## Weaknesses
- **The setting is quite restricted.** The method assumes a known DAG, no latent confounding, and essentially full observability of ancestors for conditional policies. This makes the characterization clean, but it also sharply limits applicability; the paper does not address the harder settings that matter in many causal decision problems.
- **The empirical evaluation is narrow and somewhat self-serving.** The main comparison is against brute-force node search, with no meaningful baseline against other graph-pruning heuristics or causal/contextual bandit alternatives. The regret experiment therefore demonstrates that smaller action sets help, but not that C4 is the best or most informative way to obtain them.
- **The experimental protocol is not fully convincing as evidence of general usefulness.** The target node is chosen as the node with the most ancestors, which is favorable to the method and does not test arbitrary reward nodes. In addition, the regret metric uses an estimated best arm at the end of training, which is a weak proxy and can bias the reported curves.

## Nice-to-Haves
- A clearer ablation separating the effect of candidate-node pruning from the bandit algorithm itself.
- More diverse target-node selection, rather than always choosing the node with the most ancestors.
- A case study that visualizes how the LSCA closure is formed on a real DAG.

## Novel Insights
The most interesting insight is that, under the paper’s assumptions, the right notion of “nodes worth testing” is not simply parents or ancestors of the target, but the recursive closure induced by lowest strict common ancestors. This gives a precise structural explanation for when an upstream node can dominate several candidate parents simultaneously, and why naive heuristics like looking only at immediate parents or plain LCAs are insufficient. The equivalence to a deterministic atomic-intervention notion is also a useful simplification, though it mainly serves the proof rather than changing the practical problem.

## Potentially Missed Related Work
- **Lee & Bareinboim (2018, 2020)** — directly relevant because the paper positions itself relative to causal-bandit search-space reduction and mixed-policy/scope characterization.
- **Bender et al. (2005)** — relevant because the construction relies on lowest common ancestors in DAGs.
- **Subramanian & Ravindran (2022, 2024)** — relevant as closely related work on causal contextual bandits with context-dependent interventions, though the setting is not identical.

## Suggestions
- Add a stronger experimental baseline suite, especially simple graph-theoretic heuristics such as parents-only, ancestors-only, and LCA-based pruning.
- Evaluate on arbitrary target nodes, not just the most-ancestral node, to show the characterization is not tuned to favorable cases.
- Report regret against a more standard reference when possible, or at least justify the “estimated best arm” proxy more carefully.
- Include a compact illustrative example in the main text that walks through the LSCA closure and C4 step by step.

---

## USyGD0eUod

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (4.8/10)
- Match: N/A

### Final Review

## Summary
This paper asks a timely and important question for mechanistic interpretability: do common SAE quality and auto-interpretability metrics actually distinguish trained transformers from random ones? Across Pythia models from 70M to 6.9B and several randomization schemes, the authors find that aggregate auto-interpretability and several reconstruction-style metrics can look surprisingly similar for trained, step-0, and re-randomized models, while a Gaussian-embedding control behaves very differently. The paper’s main message is a real warning: high aggregate scores are not enough to conclude that SAEs have recovered learned, computationally meaningful features.

## Strengths
- **Strong sanity-check framing with a relevant null baseline.** The paper makes a good Adebayo-style argument for mechanistic interpretability: evaluation metrics should separate trained models from strong randomized baselines, not just look “good” in aggregate. The comparison across trained, step-0, re-randomized, and Gaussian-embedding controls is thoughtfully designed and substantively relevant.
- **Broad empirical coverage across model scales and several metric families.** The authors test Pythia models from 70M to 6.9B, and they do not rely only on one score: they report auto-interpretability AUROC, reconstruction-related metrics, explained variance, and qualitative feature inspections. The appendix examples also make the key phenomenon concrete: trained models more often yield later-layer, more abstract features, while randomized variants often stay token-fragment or surface-form level.

## Weaknesses
- **The headline claim is overstated relative to the evidence.** The paper repeatedly implies that automated interpretability metrics “do not distinguish” trained and random transformers, but the results are more nuanced: the Gaussian-token control is clearly separable, some layers and model sizes do show gaps, and the Step-0 variant is sometimes as strong as or stronger than trained models. The right conclusion is that these metrics are unreliable and insufficient in some settings, not that they uniformly fail.
- **The proposed “abstractness” proxy is underdeveloped.** Token-distribution entropy is interesting, but it is not yet a validated measure of abstractness or computational relevance. Right now it supports the qualitative story that trained models produce less token-specific features in later layers, but it does not yet justify replacing current SAE metrics.
- **The causal explanation remains speculative.** The toy analyses suggest that random networks may preserve or even amplify sparsity, but the paper does not identify which mechanism actually explains the transformer results. As a result, the paper is stronger as a diagnostic warning than as an explanatory account.

## Nice-to-Haves
- Add formal uncertainty reporting and hypothesis tests for the central trained-vs-random comparisons across layers, seeds, and model sizes.
- Validate the abstractness idea against stronger alternatives, such as human semantic judgments, causal interventions, or cross-token generalization.
- Broaden the benchmark beyond Pythia and RedPajama to test whether the phenomenon generalizes to other architectures and training regimes.

## Novel Insights
The most interesting insight is not simply that random transformers can sometimes produce interpretable-looking SAE latents; it is that aggregate evaluation pipelines can be fooled by relatively shallow token-level regularities, especially when the underlying activations still carry enough structure for SAEs and LLM-based explainers to produce plausible descriptions. The appendix feature dashboards make an important distinction visible: trained models increasingly yield latents whose activations span multiple tokens and more abstract concepts, whereas randomized models often remain anchored to single-token or substring-like patterns even when aggregate AUROC looks comparable. That gap suggests the community may be overvaluing score-based SAE benchmarks and under-measuring whether features are actually abstract, stable, and computationally meaningful.

## Potentially Missed Related Work
- **Adebayo et al., 2020, Sanity Checks for Saliency Maps** — directly relevant as the randomized-baseline sanity-check template the paper is extending.
- **Karvonen et al., 2024c, Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models** — relevant because it also probes SAE/dictionary-learning evaluation under more controlled structure.
- **Zhong and Andreas, 2024, Algorithmic Capabilities of Random Transformers** — relevant as related evidence that random transformers can exhibit nontrivial structure and behavior.
- **Dooms and Wilhelm, 2024, Tokenized SAEs** — relevant because the paper’s entropy analysis and token-level failure modes closely resemble the concerns raised there.

## Suggestions
- Tighten the framing in the title/abstract/conclusion to match the evidence: the paper shows that common SAE metrics can fail badly and are insufficient, not that they universally cannot distinguish trained from random models.
- Promote the abstractness analysis from a proof of concept into an actual benchmark: define it more clearly, compare it against alternatives, and report whether it predicts causal or semantic usefulness beyond token specificity.
- Include a compact table with effect sizes and confidence intervals for the main comparisons, so readers can see exactly where trained, randomized, and control variants separate and where they do not.

---

## U6ROetm5nW

- GT: Withdrawn (treated as Reject) (avg 4.5)
- Predicted: N/A (0.0/10)
- Match: N/A

### Final Review

## Summary
This paper gives a theoretical KDE data structure for the Gaussian kernel in the high-dimensional regime, obtained by reducing KDE to level-wise recovery and then instantiating that recovery with asymmetric ANN/LSH on the sphere. The main claimed payoff is a new query-time versus space tradeoff: in polynomial space the query exponent is improved to about \(0.05\), and in linear space the exponent is about \(0.1865\), improving on the prior data-independent KDE bound of \(0.25\).

## Strengths
- The paper identifies a real and nontrivial technical direction: replacing the symmetric LSH machinery in the Charikar et al. KDE framework with asymmetric ANN, which plausibly explains the improved exponents. This is a substantive algorithmic idea, not just a reparameterization.
- It states a genuine time-space tradeoff family rather than a single tuned bound, which is conceptually valuable. The tradeoff curve and the observed plateau around exponent \(\approx 0.05\) are interesting and suggest the framework has a real structural limit.

## Weaknesses
- The main quantitative results are obtained through heavy numerical optimization over nested expressions, not through a clean analytic derivation. The headline exponents \(0.051\), \(0.1865\), and the plateau near \(0.05\) are therefore less satisfying than they should be, and the paper gives limited insight into why these are the right values beyond a script-based evaluation.
- The presentation is very hard to follow. The core argument is buried under dense notation, many auxiliary definitions, and a long chain of reductions; the main mechanism is not communicated crisply enough for the reader to assess the contribution without constantly reconstructing the notation.
- The improvement is narrow in scope: it is specific to Gaussian KDE in the high-dimensional, \(\mu^*=n^{-\Theta(1)}\) regime, and the stronger bound comes at the cost of polynomially larger space. The paper is honest about this, but the practical significance remains limited without evidence beyond asymptotics.

## Nice-to-Haves
- A compact comparison table against prior KDE/LSH results would make the contribution much easier to parse: query exponent, space exponent, data-independent vs data-dependent, and kernel setting.
- A short intuition-first explanation of why asymmetric LSH helps specifically in the level-recovery reduction would improve readability substantially.
- If possible, a more analytic characterization of the exponent tradeoff, even if slightly looser, would be much more interpretable than pure grid search.

## Novel Insights
The genuinely interesting insight is that the KDE-via-ANN reduction has a hidden mismatch between where space is consumed and where query time is maximized, and asymmetric LSH can exploit that mismatch. In other words, the paper is not merely improving a nearest-neighbor primitive; it is rebalancing the KDE pipeline so that different distance scales can be assigned different space/query budgets, which is why a time-space tradeoff emerges and why the best query exponent is not achieved at the same point as the best space exponent. That is a real structural observation, even though the final presentation obscures it.

## Potentially Missed Related Work
- None identified — the paper already cites the main KDE/LSH line of work it builds on, including Charikar & Siminelakis (2017), Backurs et al. (2018, 2019), and Charikar et al. (2020).

## Suggestions
- Rewrite the main technical story in a much more digestible order: high-level idea, level-recovery reduction, asymmetric ANN instantiation, then the tradeoff theorem.
- Replace the script-driven exponent reporting with either a closed-form derivation or a more transparent numerical appendix that clearly states what is optimized, over what range, and how sensitive the result is to the grid.
- Add a short section explaining exactly where the space/query plateau comes from and why further space does not help beyond \(\delta \approx 3.15\).

---

## bH5M0ts8Y6

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (5.4/10)
- Match: N/A

### Final Review

## Summary
This paper proposes VINCIE, a video-driven framework for learning in-context image editing from native videos rather than handcrafted edit pairs. The core idea is to turn videos into interleaved text-image sessions with VLM-generated transition captions and RoE segmentation masks, then train a diffusion transformer with next-image prediction plus auxiliary segmentation prediction tasks. The paper also introduces MSE-Bench for 5-turn editing and reports strong empirical gains on multi-turn consistency, though much of the evidence is still tied to synthetic annotation/evaluation pipelines.

## Strengths
- **Compelling data-centric reframing of the problem.** The paper identifies a real bottleneck in in-context editing: coherent multi-turn training data is scarce, and native video is a plausible scalable source of contextual visual transitions. This is a genuinely interesting direction for ICLR.
- **Concrete multi-task training recipe with sensible auxiliary supervision.** The combination of next-image prediction with current/next segmentation prediction is well motivated for grounding and layout anticipation, and the ablations do show that segmentation-related supervision improves consistency and multi-turn success.
- **The paper does provide substantial empirical evidence of scalability and context benefits.** Results on MagicBrush and MSE-Bench, plus the scaling plot and ablations, consistently support the claim that longer contextual training helps later turns and that training scale matters.

## Weaknesses
- **The main causal claim is not cleanly isolated: gains are entangled with scale, initialization, and SFT.** The model is initialized from a large video foundation model and many of the strongest results use supervised fine-tuning on editing datasets. As a result, the paper does not convincingly prove that “videos alone” are the key reason for the gains.
- **The benchmark/evaluation story is fragile.** MSE-Bench is author-constructed from GPT-4o prompt imagination and then evaluated by GPT-4o, with only moderate correlation to human judgments. That does not invalidate the benchmark, but it substantially weakens the strength of the reported headline numbers.
- **The automatic data pipeline is under-characterized.** The paper relies on in-house VLM transition captions plus GroundingDINO/SAM2 masks, but gives only limited analysis of annotation noise, failure rates, and how often the generated RoEs or transitions are wrong. This matters because the whole method depends on noisy automatic supervision.
- **Some important ablations are missing.** The paper shows that segmentation/context help, but does not adequately isolate the effect of removing VLM transition text, varying RoE quality, changing video-source diversity, or matching compute across video vs pairwise alternatives. Without these, it is hard to know which component actually drives the improvement.

## Nice-to-Haves
- A clearer breakdown of results by edit category and failure mode, especially for background changes, camera motion, and abstract or non-local edits.
- More explicit compute and data-efficiency reporting, so readers can judge whether the gains come from the proposed learning paradigm or simply from very large-scale training.
- A sharper side-by-side comparison table against the closest recent video-conditioned and in-context editing systems under the same setup.

## Novel Insights
The most interesting insight is that video supervision appears to be useful not just as a source of visual consistency, but as a source of *sequential edit semantics*: the model can learn to represent change as a chain of grounded transitions, especially when segmentation is inserted as an intermediate target. That said, the paper’s strongest evidence suggests a more modest interpretation than the authors imply: the method seems to work because it combines a strong pretrained generative backbone, large-scale automatically annotated video sessions, and auxiliary mask prediction that regularizes multi-turn generation. The “emergent” story is plausible, but not yet proven cleanly enough to separate true video-driven generalization from the effects of scale and downstream adaptation.

## Potentially Missed Related Work
- **RealGeneral** — highly relevant recent video-to-editing / temporal in-context generation direction; should be compared directly on the same multi-turn setting.
- **UES** — related to temporal in-context consistency in video foundation models and relevant for assessing whether video-derived context is the key ingredient.
- **FramePainter** — another video-prior-based interactive editing line that is directly relevant to the claim of learning editing from video.
- **In-context Edit** — close in spirit on in-context image editing, and important for positioning what is actually new here.
- **Context Diffusion** — relevant for contextual image generation/editing and for distinguishing interleaved context modeling from prior work.

## Suggestions
- Add controlled experiments with matched compute and backbone comparing: video-only, pairwise-only, interleaved image-text only, and video+SFT settings.
- Report a stronger human evaluation on MSE-Bench, including inter-annotator agreement and statistical uncertainty, not just GPT-4o scores.
- Include a systematic noise analysis of the VLM annotation and RoE extraction pipeline, ideally with category-wise accuracy/recall and examples of failure.
- Expand ablations to remove transition text, RoE masks, learnable turn tokens, and to compare full vs block-causal attention under the same training budget.
- Add direct comparisons to the nearest prior video-conditioned/in-context editing methods on the same multi-turn protocol.

---

## khHNHzRjMy

- GT: Reject (avg 3.0)
- Predicted: N/A (3.1/10)
- Match: N/A

### Final Review

## Summary
This paper introduces EmoSign, a small ASL video dataset annotated by three Deaf native signers with sentiment labels, emotion intensities over 10 categories, and free-form descriptions of emotional cues. The paper also reports baseline results for several multimodal LLMs and argues that current models are weak at extracting affect from sign video and tend to over-rely on captions.

## Strengths
- The paper targets a genuinely important and underexplored problem: emotion understanding in ASL, which is relevant to accessibility and high-stakes communication settings. The motivation is credible and well grounded in the sign-language literature.
- The annotation process is stronger than most dataset papers in this area: labels come from Deaf native ASL signers with professional interpretation experience, and the dataset includes not just labels but also cue descriptions. The reported inter-annotator agreement is decent for a subjective affect task, especially for sentiment.

## Weaknesses
- The dataset construction is heavily biased by a VADER-based prefilter on captions, followed by selecting the 100 most positive and 100 most negative utterances. This means the benchmark is not a representative sample of ASL emotion in the wild, but a sentiment-filtered subset of an existing captioned corpus. That materially weakens claims that the dataset measures visual emotion understanding rather than caption sentiment shortcuts.
- The benchmark evidence is too thin for the paper’s strongest claims. Only four prompt-based MLLMs are tested, with no standard non-LLM baselines, no sign-language-specific baselines, no confidence intervals or significance tests, and no analysis by signer, length, or emotion frequency. On a 200-clip dataset, that makes the reported rankings and conclusions quite unstable.
- The caption modality appears to dominate the results, especially for emotion classification where caption-only is often similar to or better than video+caption. The paper’s claim that visual information “contributes meaningfully” is therefore only weakly supported; the more defensible conclusion is that current models mostly lean on text and still do not robustly use sign-video cues.
- The proposed “emotion cue grounding” task is not actually benchmarked in a rigorous way. Section 5.3 is qualitative inspection of a few examples, not an evaluable grounding benchmark. Presenting it as a task alongside the classification benchmarks overstates the empirical support.

## Nice-to-Haves
- A more representative sampling strategy that includes neutral and naturally occurring clips, rather than only the most positive and negative captioned utterances.
- Clearer annotation protocol details and a fuller release of the benchmark split and evaluation code once the dataset is public.

## Novel Insights
The most interesting scientific observation is not just that MLLMs struggle on ASL emotion recognition, but that they seem to treat captions as the primary evidence and use video mainly as weak or noisy support. This suggests a deeper issue than generic multimodal weakness: sign-language emotion understanding requires disentangling grammatical facial expressions, affective facial expressions, and motion cues in a setting where language and emotion are visually fused, and current general-purpose models are not equipped for that. The free-form annotator cue descriptions are a promising asset because they expose exactly which non-manual signals native signers use, but the paper does not yet convert that insight into a convincing quantitative benchmark.

## Potentially Missed Related Work
- FePh — the closest prior sign-language facial-expression dataset discussed by the paper; relevant as the main comparison point for emotion-related sign annotations.
- OpenASL / How2Sign / ASLLRP — relevant as source corpora and for understanding how much of EmoSign inherits caption and translation biases from prior ASL resources.
- Emotion-aware multimodal benchmark work such as MELD-ST or Emotion-LLaMA — relevant for stronger benchmark design and for grounding the motivation in broader emotion-understanding datasets.

## Suggestions
- Add standard baselines beyond prompt-based MLLMs: a text-only classifier on captions, a simple video-embedding classifier, and a fused baseline. This is the most direct way to test whether EmoSign is actually a hard multimodal benchmark or mostly a caption-sentiment benchmark.
- Release an ablation on the VADER-based filtering choice, ideally comparing performance on the current filtered set versus a more representative sample. Without this, the core dataset construction choice remains a serious confound.
- If grounding is a real contribution, annotate temporal or spatial cue spans and evaluate them with an objective metric rather than only manual case inspection.
- Report per-signer and per-emotion breakdowns, plus label prevalence and tie rates. With only 3 annotators and 200 clips, these details are necessary to judge whether the labels are stable enough for broader claims.

---

## ZBhZT307xx

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (5.3/10)
- Match: N/A

### Final Review

## Summary
This paper studies verifiers used in RL with verifiable reward for mathematical reasoning, comparing rule-based and model-based approaches across static verification and RL training. The main message is that rule-based verifiers miss a meaningful fraction of correct answers, especially for stronger models and harder answer formats, while model-based verifiers can improve static recall but may be fragile to reward hacking once deployed in RL.

## Strengths
- The paper addresses a genuinely important RLVR bottleneck: verifier quality directly determines training signal quality, and the study is timely given the rise of reasoning-focused RL systems.
- The empirical scope is reasonably broad for a diagnosis paper: it compares multiple rule-based verifiers, several general and trained model-based verifiers, multiple math datasets, and a general-science dataset, and it connects static verifier behavior to downstream RL outcomes.
- The static-vs-dynamic contrast is the paper’s most useful insight. It shows that higher verifier classification accuracy does not automatically translate into better RL performance, and it provides concrete evidence that some fine-tuned generative verifiers are exploitable by simple hacks such as empty symbols, gibberish, and prompt injection.
- The human-vs-GPT-4o validation for annotation quality is a legitimate and helpful attempt to support the evaluation pipeline, even if it does not eliminate all concerns.

## Weaknesses
- The core claims are still supported mostly by a narrow set of models, datasets, and training recipes. The paper draws broad conclusions about verifier robustness, but the RL evidence is concentrated on a limited Qwen-based setup and a small number of benchmark families, so generality is not yet established.
- The reward-hacking diagnosis is suggestive but not airtight. The paper compares training reward to GPT-4o “oracle” reward at checkpoints, but it does not fully rule out confounds such as annotation noise, prompt sensitivity, or distribution drift during training; the causal interpretation remains somewhat under-justified.
- The paper is much stronger at identifying failures than at explaining them. It shows that rule-based verifiers have false negatives and that generative verifiers can be hacked, but it does not give a deep mechanistic account of why these failures occur or why fine-tuning amplifies brittleness.
- Several of the most important conclusions rely on fairly small or partially controlled comparisons. There are limited seeds/statistical analyses in the main paper, and the RL gains from the hybrid verifier are modest enough that the paper would benefit from stronger robustness checks before making sweeping claims.

## Nice-to-Haves
- A more explicit ablation of the hybrid verifier design would be useful: rule-first gating, question/context inclusion, and decoding choices could all matter, and the paper currently does not isolate them cleanly.
- A clearer error taxonomy for rule-based false negatives would improve the paper substantially: formatting mismatch, mathematically equivalent variants, parsing failures, and dataset-specific conventions likely account for different fractions of the errors.
- A small mitigation study would add value, even if not required for the core claim: for example, output sanitization, adversarial training, calibration-based routing, or a confidence-aware hybrid could show whether the observed brittleness is practically addressable.

## Novel Insights
The most interesting insight is not simply that some verifiers are inaccurate, but that the relevant failure modes differ sharply between static classification and RL optimization. Rule-based verifiers appear precise but recall-limited, and their false negatives become more problematic as policy models become stronger and produce more diverse outputs; model-based verifiers improve static recall, yet their apparent advantage can disappear or reverse once the policy learns to exploit superficial patterns that fool the verifier. That distinction between “accuracy on held-out labeled data” and “robustness under adaptive pressure” is the paper’s real contribution, and it is a useful reminder that verifier evaluation for RL must be adversarial, not just benchmark-based.

## Potentially Missed Related Work
- Baker et al., 2025 — relevant for monitoring reasoning models and the risks of promoting obfuscation / reward manipulation.
- xVerify: Efficient Answer Verifier for Reasoning Model Evaluations (Chen et al., 2025) — directly relevant as a verifier baseline and for comparison on robustness.
- General-Reasoner / General-verifier (Ma et al., 2025) — relevant because the paper evaluates a trained verifier from this line and because it studies cross-domain verification.
- Works on RL with verifiable reward and verifier design in DeepSeekMath / DeepSeek-R1 / SimpleRL-Zoo — relevant background and closely tied to the training setup used here.

## Suggestions
- Add a controlled ablation that matches verifier accuracy while varying robustness, and vice versa, to separate “better static verification” from “better RL signal.”
- Report multiple seeds and confidence intervals for the main RL comparisons, especially where the gains over rule-only verification are small.
- Include a clearer breakdown of rule-based false negatives by failure mode, and representative examples for each major category.
- If space permits, test at least one simple defense against hacking patterns and report whether it changes the RL outcome.


---

## v05SW2X3IC

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (4.4/10)
- Match: N/A

### Final Review

## Summary
This paper proposes a learnable Gray-Wyner-style codec for two-task vision problems, with one common channel and two private channels, motivated by lossy common information. It includes theoretical bounds relating lossy Wyner and Gács-Körner common information, plus an entropy-model-based training objective intended to trade off transmit rate and receive rate in a neural codec.

## Strengths
- The paper targets a genuinely interesting and underexplored problem: separating shared vs. task-specific information for multi-task communication, rather than treating all downstream tasks as if they should share the same latent code.
- The Gray-Wyner framing is conceptually coherent and the proposed three-channel architecture matches the stated objective reasonably well; the synthetic and colored-MNIST experiments do show the intended rate-allocation behavior across common/private channels.
- The evaluation is broader than a toy demo: the paper tests synthetic data, controlled MNIST edge cases, and real vision tasks on Cityscapes and COCO, and compares against several natural architectural baselines.

## Weaknesses
- The theoretical development is difficult to verify and in places reads more like an intuition-driven sketch than a rigorous argument. Theorem 1 especially relies on interaction-information reasoning and separability conditions that are not cleanly operationalized, so the claimed relationship between the two lossy common-information notions is not convincingly established.
- The bridge from Gray-Wyner theory to the actual neural codec is still quite heuristic. The tensor-splitting/common-channel matching mechanism, the use of quantization surrogates, and the reliance on frozen downstream task models mean the learned system is only an approximate relaxation of the information-theoretic problem, but the paper does not sufficiently delineate that gap.
- The empirical evidence is directionally positive but not especially strong. The proposed method usually beats the Independent baseline, but the gains over simpler alternatives are often modest, and on the real vision tasks the reported improvements are not decisive enough to fully justify the paper’s stronger narrative about substantially reducing redundancy.
- Key ablations are missing. In particular, the paper does not systematically test the importance of the auxiliary matching loss, conditional entropy modeling of the private channels on the common code, or sensitivity to \(\beta\). Without this, it is hard to tell whether the Gray-Wyner formulation itself is doing the work or whether the gains come from extra flexibility and tuning.

## Nice-to-Haves
- A cleaner empirical link between the learned common-channel rate and the estimated common-information quantities would make the theory-to-practice story much more convincing.
- A more explicit sensitivity study over \(\beta\), \(\gamma\), latent dimensionality, and random seeds would help establish robustness.
- It would also help to compare against stronger multi-task compression baselines beyond the custom Joint/Independent/Separated variants, especially prior learned human/machine or multi-task codec methods.

## Novel Insights
The most interesting insight here is that the classical Gray-Wyner tradeoff is not just a source-coding curiosity but can be repurposed as a design principle for neural task representations: one can deliberately choose whether shared information lives in a common channel or leaks into private ones, depending on whether the objective is optimal transmit rate or optimal receive rate. The paper’s synthetic and MNIST results support the idea that the common code can be steered along this tradeoff, but the real contribution is still more of a promising framework than a fully validated principle of learned common-information extraction.

## Potentially Missed Related Work
- Choi & Bajic, 2022 — relevant prior work on coding for humans and machines with common/private channels
- Foroutan et al., 2023 — relevant comparison point for two-transform / shared coding designs
- Chamain et al., 2021 — multitask learned image compression without private channels
- Feng et al., 2022 — multitask / unified machine-oriented image coding
- Guo et al., 2024 — another recent multitask image compression baseline
- Dubois et al., 2021 — relevant from the representation-learning/compression perspective, though not directly supervised task separation

## Suggestions
- Add a hard ablation section: remove the auxiliary common-matching loss, remove common-conditioning in private entropy models, vary \(\beta\), and match parameter counts against baselines.
- Report uncertainty: multiple seeds, error bars, and a more careful explanation of how BD-rate is computed for heterogeneous task metrics.
- Strengthen the theory-to-practice bridge by directly plotting learned common-channel rates against estimated mutual-information/common-information proxies, and by clarifying which parts of the optimization are exact versus heuristic.

---

## JEN4nsDgh9

- GT: Reject (avg 3.5)
- Predicted: N/A (3.8/10)
- Match: N/A

### Final Review

## Summary
This paper proposes a benchmark for “taxonomy image generation,” i.e., generating images for WordNet concepts rather than ordinary caption-like prompts. It evaluates 12 open-source generation/retrieval systems on several concept subsets and reports human and GPT-4 pairwise preferences, plus a set of CLIP-based and image-quality metrics.

## Strengths
- The task framing is genuinely interesting and underexplored: connecting text-to-image generation with WordNet-style taxonomy concepts is a natural but previously neglected evaluation setting. The paper also goes beyond “easy” common-sense concepts and includes random hyponymy/hypernymy/mixed nodes and LLM-predicted concepts.
- The empirical comparison is broad and informative. Evaluating 12 systems spanning diffusion U-Nets, DiTs, and retrieval surfaces nontrivial differences from standard T2I benchmarks: in particular, Playground and FLUX are consistently strong on preference-based evaluation, while retrieval performs poorly.

## Weaknesses
- The core metric contribution is overstated and only weakly justified. Most of the “taxonomy-specific” metrics are CLIP-similarity variants wrapped in probabilistic language, but the appendix derivations do not rigorously establish the claimed links to likelihood, KL divergence, or mutual information. As written, the theory reads more like post-hoc intuition than a sound foundation, which is a serious problem for a benchmark paper.
- The evaluation protocol is still too fragile to support the stronger claims. GPT-4 is used as a judge, but the paper acknowledges first-position bias and does not use multiple runs or majority voting; the human-vs-GPT agreement is reported only at aggregate ranking level. In addition, several reported metrics are known proxies at best: FID is computed against retrieved images, and IS is largely orthogonal to semantic fidelity here. This makes it hard to tell how much the benchmark really measures taxonomy understanding versus generic image quality or CLIP/GPT-4 alignment.
- Dataset construction and experimental details are underspecified in important ways. The WordNet sampling procedure is confusing, the exact meaning of the sampling probabilities is not clear, and the LLM-prediction pipeline is only partially described. For a benchmark paper, the missing details around prompting, sampling, retrieval ranking, and label aggregation are nontrivial reproducibility gaps.
- The central claims are stronger than the evidence. The paper suggests that the benchmark can support “automating the curation of structured data resources,” but the experiments only establish that some T2I models can generate somewhat plausible depictions for a fixed set of WordNet concepts. There is no demonstrated downstream taxonomy-enrichment utility or expert-verified concept-image quality.

## Nice-to-Haves
- A cleaner separation of the metrics into concept fidelity, specificity, realism, and human preference would make the benchmark much easier to interpret.
- Stronger validation against human judgments at the instance level, not just model-rank level, would make the proposed metrics more credible.
- A second taxonomy or non-WordNet concept set would help show whether the benchmark is about taxonomy image generation in general, rather than WordNet-specific lexical quirks.

## Novel Insights
The most interesting insight is that taxonomy-oriented image generation is not just “ordinary text-to-image with different words”: model rankings shift substantially depending on whether the prompt includes definitions and on whether the evaluation emphasizes human preference, CLIP-style semantic alignment, or retrieval realism. The paper also reveals a striking mismatch between preference-based judgments and similarity-based diagnostics: some models look strong to humans yet are not the best under CLIP-derived scores, suggesting that current generic T2I metrics are poor proxies for concept-specific depiction quality. That said, this is precisely why the benchmark needs much tighter metric validation before the findings can be treated as definitive.

## Potentially Missed Related Work
- ConceptBed — directly relevant as a benchmark for concept learning in text-to-image models.
- Liao et al., “Text-to-Image Generation for Abstract Concepts” — relevant because it studies generation beyond concrete objects and provides a nearby evaluation framing.
- Baryshnikov & Ryabinin, “Hypernymy understanding evaluation of text-to-image models via WordNet hierarchy” — important prior WordNet/taxonomy-based image evaluation work that should be positioned more carefully as a precursor.
- Patel et al., “ConceptBed” and related concept-focused T2I evaluation papers — relevant for stronger baseline comparison and benchmark positioning.

## Suggestions
- Reframe the CLIP-derived metrics explicitly as heuristic taxonomy-aware diagnostics unless the authors can provide a much more rigorous derivation and validation.
- Add an ablation showing item-level correlation with human judgments for each proposed metric, plus sensitivity to CLIP backbone and prompt wording.
- Report a more rigorous GPT-4 judging protocol: exact prompt, number of repeats, positional randomization details, and robustness to order bias.
- Tighten the dataset specification: full sampling procedure, counts per relation type, prompt templates, and release-ready documentation for all prompts and concept lists.
- Add direct comparisons against prior concept/taxonomy T2I benchmarks and a small downstream use case to substantiate the “benchmarking for taxonomy curation” motivation.

---

## b6qQmQ2F13

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (5.4/10)
- Match: N/A

### Final Review

## Summary
This paper studies how to allocate a fixed memory budget when deploying reasoning LLMs, arguing that the usual “just use 4-bit weights” heuristic breaks down once KV cache growth dominates. Across Qwen3 and two additional reasoning model families, the authors map memory–accuracy Pareto frontiers over model size, weight precision, token budget, parallel sampling, and KV-cache compression, and find a scale-dependent regime switch: small effective models benefit more from spending bytes on weights, while larger models benefit more from spending bytes on longer generation and parallel sampling.

## Strengths
- The paper targets a real and timely deployment bottleneck: for reasoning models, KV cache can exceed weight memory, so conventional weight-centric compression guidance is incomplete. The framing is concrete and practically relevant, not just a minor variant of prior quantization work.
- The empirical sweep is broad and well structured. The paper explores more than 1,700 configurations across model size, precision, token budget, group size, and KV compression, and validates the main pattern on multiple benchmarks and on additional model families beyond Qwen3.
- The main conclusion is useful and actionable: the optimal memory allocation is not universal, and the choice between higher-precision/larger weights, longer generation, parallel sampling, and KV compression depends on effective model size and task type. That is a meaningful deployment insight if one is trying to serve reasoning models under tight VRAM.

## Weaknesses
- The paper’s core claim is empirical and threshold-based, but the evidence does not support a universal breakpoint as strongly as the writing suggests. The “8-bit 4B” cutoff clearly varies by benchmark and model family, so presenting it too much like a general law overstates the result. This matters because the paper’s practical prescription hinges on a boundary that is not stable across settings.
- The study is still narrow relative to the strength of the deployment claims. Most of the evidence comes from Qwen3, with a limited set of reasoning benchmarks and a limited set of KV-compression and verifier baselines. The broad message is plausible, but the paper does not fully rule out that the observed threshold is partly an artifact of the chosen architectures, prompting protocol, and benchmark mix.
- The work is almost entirely descriptive. It convincingly catalogs frontiers, but it offers only a weak mechanistic account of why math/code prefer higher precision while knowledge-heavy QA prefers 4-bit, or why the threshold lands where it does. Without a more principled explanation, the paper reads more like an extensive measurement report than a durable scaling law.
- The parallel-scaling and budget-forcing analyses are informative, but the interaction between batching assumptions, group size, and memory accounting is still somewhat under-explained. Since the paper’s conclusions depend on how weights are amortized and how KV cost scales, a clearer treatment of realistic serving conditions would strengthen the deployment relevance.

## Nice-to-Haves
- Add uncertainty estimates or repeated-run variance for the Pareto frontiers, especially where configurations are close.
- Include a compact decision table that maps memory budget and task type to a recommended strategy.
- Make the effective-size threshold more operational by deriving a simple predictor rather than relying on the “8-bit 4B” shorthand.

## Novel Insights
The main novel insight is that memory optimization for reasoning LLMs is not just “compress weights as much as possible”: once generation gets long enough, the KV cache becomes the dominant budget item and flips the best allocation strategy. The paper’s most interesting empirical pattern is that this flip is scale-dependent and task-dependent: smaller effective models are best improved by spending memory on weights, while larger models increasingly benefit from spending memory on more tokens and parallel samples, and the preferred KV strategy also changes with scale. That gives a more deployment-realistic view of inference trade-offs than prior non-reasoning quantization work, even if the exact cutoff is not universal.

## Potentially Missed Related Work
- H2O — relevant as a stronger KV-cache eviction baseline that could contextualize the eviction results.
- Scissorhands — relevant as a prior KV compression method for test-time inference.
- SnapKV — relevant as another important KV selection/compression baseline.
- KIVI — relevant for low-bit KV-cache quantization comparisons.
- PagedAttention — relevant for serving-time memory management and batching context.

## Suggestions
- Add a stronger, apples-to-apples comparison against a broader set of KV compression and serving baselines, and report uncertainty across seeds or sampling runs.
- Provide a more principled explanation of the scale threshold, ideally with a compact analytic or semi-analytic model that links effective size, token budget, and quantization error.
- Tighten the language around the “8-bit 4B” boundary so it is clearly presented as an empirical observation that shifts with task and model family, not a universal rule.

---

## cEXEmyW77N

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (4.4/10)
- Match: N/A

### Final Review

## Summary
This paper studies whether LLM-generated scientific reference lists can be distinguished from human bibliographies when represented as citation graphs. Using 10,000 focal papers from SciSciNet, the authors compare ground-truth references against GPT-4o- and Claude-generated reference sets, plus field-matched random baselines, under structural features, semantic embeddings, and GNN classifiers.

The central claim is that coarse citation topology is surprisingly close between LLM and human bibliographies, while semantic embeddings expose a much stronger separability signal. The paper is large-scale and carefully engineered, but its strongest conclusions are still more diagnostic than deeply explanatory.

## Strengths
- **Large paired dataset with a sensible comparison design.** The paper evaluates 10,000 focal papers with matched ground-truth and LLM-generated bibliographies, which is much stronger than comparing unrelated corpora. This pairing controls for focal-paper-specific confounds and makes the classification tasks meaningful.
- **Reasonable robustness checks beyond the main GPT-4o setup.** The authors repeat the analysis with Claude Sonnet 4.5, with SPECTER and OpenAI embeddings, with subfield- and temporal-matched random baselines, and with random-vector controls. These checks do support the claim that the semantic signal is not just a one-off artifact of a single generator or embedding model.

## Weaknesses
- **The main result is overstated relative to the structural evidence.** GPT-vs-ground-truth is not “indistinguishable” under structure-only features; the reported RF accuracy is about 0.61, which is clearly above chance. Structure is weaker than semantics, yes, but the paper’s rhetoric makes the topology result sound stronger and cleaner than it is.
- **The semantic signal is not well disentangled from simpler bibliographic correlates.** The classifier operates on title/abstract embeddings of papers in the bibliography, but the paper does not isolate whether the gain comes from true semantic mismatch, recency, venue prestige, citation-count bias, topical drift, author overlap, or other easy correlates. Without sharper ablations, the “semantic fingerprint” claim remains underspecified.
- **The feature pipeline is somewhat muddled and potentially confounded.** In the GNN structural setting, the paper uses graph-level quantities, including total edge count, as node features. That is an unusual design choice and makes the representation less clean than it should be, raising concerns about whether the GNN is learning a meaningful graph signal or exploiting a degenerate global-size cue.
- **Evaluation reporting is not crisp enough for a paper making strong detection claims.** The paper reports mean accuracy/F1 across runs, but the split protocol, selection of best hyperparameters, and the exact unit of variation are not always clearly stated. For an ICLR submission, the final test-set reporting and the validation/test separation should be much easier to audit.

## Nice-to-Haves
- A cleaner ablation separating title-only, abstract-only, similarity-to-focal, intra-bibliography cohesion, recency, and venue/author overlap features would make the semantic story much more convincing.
- A more deployment-relevant evaluation using AUROC, calibration, and low-FPR operating points would improve the practical detector story.

## Novel Insights
The most interesting insight here is not merely that LLM reference lists can be detected, but that the detection signal appears to invert across representation levels: topology behaves almost like a strong null-resistant prior, making generated bibliographies look human under coarse graph descriptors, while semantic embeddings uncover systematic differences that survive across generators and embedding backbones. That said, the paper has not yet shown that these differences are uniquely “semantic” in a deep sense; they may partly reflect broader bibliometric biases that happen to be captured by embeddings. The work is therefore best read as a solid empirical diagnostic of where the signal lives, not as a complete explanation of why LLM bibliographies differ from human ones.

## Potentially Missed Related Work
- **LLM-check** — relevant as a related detection approach, though it targets hallucination detection rather than full reference-list graph analysis.
- **Recent work on LLM-generated bibliographies by the same group** — highly relevant background because this paper builds directly on that pipeline and dataset construction.
- **Work on LLM-driven paper recommendation and citation bias** — relevant because several of the paper’s observed differences likely align with recommendation/bibliometric bias rather than pure semantic mismatch.

## Suggestions
- Add a compact ablation table that separates the main candidate signals: year, venue prestige, author overlap, citation count, and embedding similarity to the focal paper.
- Replace or justify the graph-structural GNN feature design, especially the use of graph-level edge count as a node feature.
- Report the final test-set results and split protocol more explicitly, ideally with confidence intervals and paired significance tests across seeds.
- Include a small failure-case analysis showing example bibliographies that are misclassified in both directions, to ground the detector’s behavior in concrete cases.

---

## Vgm77U4ojX

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (5.3/10)
- Match: N/A

### Final Review

## Summary
This paper proposes SIGMADOCK, a fragment-based SE(3) diffusion model for rigid-receptor molecular docking. The main idea is to decompose ligands into rigid fragments, diffuse their poses in SE(3)^m, and use triangulation-based geometric conditioning plus an SO(3)-equivariant score head to reconstruct docked poses. The paper reports strong results on PoseBusters and Astex, but the evidence is heavily conditioned on a custom ranking heuristic and a narrow redocking setup.

## Strengths
- The central modeling idea is genuinely interesting: replacing torsion diffusion with fragment-level SE(3) diffusion is a coherent inductive bias and directly addresses known ambiguities in torsion-to-Cartesian updates.
- The paper is methodologically substantial rather than a minor tweak: it includes FR3D fragmentation, triangulation conditioning, an equivariant score architecture, and extensive ablations showing that these components matter.
- The evaluation appropriately includes PoseBusters validity in addition to RMSD, which is important given recent critiques that RMSD alone can overstate docking quality.

## Weaknesses
- The headline performance claims are hard to trust at face value because the reported Top-1 results depend on a custom test-time ranking heuristic combining Vinardo energy and PoseBusters-style penalties. This makes it unclear how much of the gain comes from the generative model itself versus post hoc selection.
- The comparison protocol is not fully controlled. The paper contrasts itself against methods trained on different data, with different preprocessing, pocket definitions, and post-processing, while still making strong SOTA and “first to surpass physics-based docking” claims. That weakens the evidential value of the benchmark tables.
- The evaluation is too narrow for the strength of the claims. Everything is rigid-receptor redocking; the paper explicitly does not test cross-docking, apo docking, or flexible receptor settings. For a paper claiming broad practical relevance, that is a major limitation.
- The theoretical story is suggestive but over-argued. The product-measure and entanglement arguments are mathematically plausible, but they do not by themselves establish that fragment diffusion is easier to learn in practice or explain the empirical gains. The paper leans too hard on geometry as if it were a full causal explanation.
- Several important design choices remain heuristic and under-isolated: fragmentation merging, triangulation edge selection, pocket-center perturbation, and sample ranking. The ablations show these matter, but the paper does not cleanly separate which part of the pipeline is responsible for the final performance.

## Nice-to-Haves
- A cleaner ablation that separates “representation gain” from “test-time ranking gain” would make the paper much more convincing.
- A controlled comparison against a torsion-space version with the same backbone, training budget, and ranking procedure would help validate the core representation claim.
- Reporting uncertainty intervals or significance tests on the main benchmark numbers would improve trust.

## Novel Insights
The most interesting contribution is not merely “using fragments,” but the claim that fragment-level SE(3) diffusion sidesteps a structural mismatch in torsional models: torsional updates induce coupled, nonlocal Cartesian changes and ambiguous gauge choices, whereas rigid fragments give a more natural factorization for noise and sampling. That is a meaningful design insight and is supported by the ablation that triangulation conditioning and fragmentation merging both improve results. At the same time, the paper’s strongest empirical gains seem entangled with a fairly elaborate inference-time selection pipeline, so the real advance may be less “the diffusion model alone” and more “a carefully engineered docking system built around fragment diffusion plus ranking heuristics.”

## Potentially Missed Related Work
- DiffDock / DiffDock-L — directly relevant torsional diffusion docking baselines that should be matched carefully under the same protocol
- Uni-Mol Docking v2 — strong recent docking baseline, relevant for controlled benchmarking
- Re-Dock — relevant because the paper’s own formulation is closest in spirit to diffusion-based redocking
- TankBind and EquiBind — relevant geometric docking baselines for comparison under the same split
- PoseBusters benchmarking paper — already cited and highly relevant to validity-aware evaluation

## Suggestions
- Add a strict ablation table with identical training and sampling budgets for: model only, model + energy ranking, model + PB filtering, and full SIGMADOCK.
- Include a matched-comparison table where all baselines use the same train/test split, pocket definition, and seed budget, or clearly separate literature-reported numbers from directly reproduced numbers.
- Evaluate at least one harder setting beyond rigid redocking, ideally cross-docking or apo-to-holo docking, to support the generalization narrative.
- Add a torsion-space baseline with the same backbone and post-processing so the fragment-space advantage is actually isolated.

---

## iaoAKDRAJQ

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (6.1/10)
- Match: N/A

### Final Review

## Summary
This paper gives a unified theoretical comparison between adaptive optimizers and normalized steepest descent (NSD) through the lens of geometry. The main claim is that adaptive methods are governed by a stronger notion of smoothness, “adaptive smoothness,” which characterizes their nonconvex convergence and can also enable acceleration and improved stochastic guarantees under corresponding adaptive noise assumptions.

## Strengths
- The paper’s core conceptual framing is genuinely interesting: it cleanly separates two ways of exploiting non-Euclidean geometry, one via standard norm smoothness for NSD and one via adaptive smoothness for adaptive preconditioned methods. The diagonal example with \(\ell_\infty\)/\(\ell_1\) duality is a good illustration of why these are not the same analytic regime.
- The unified framework over well-structured preconditioner sets is a substantive technical contribution. It covers a broad class of methods, including AdaGrad, Adam/RMSProp-style variants, AdaGrad-Norm, and one-sided Shampoo/ASGO, and it supports both deterministic and stochastic analyses.
- The paper extends adaptive smoothness to the nonconvex setting and ties it to convergence of adaptive optimizers there. The additional matrix inequality for handling noncommutative preconditioners appears to be the key technical enabler and is plausibly of independent interest.

## Weaknesses
- The paper’s strongest claims are mostly theoretical separation results under strengthened assumptions, not direct algorithmic advances. In particular, the acceleration result depends on adaptive smoothness and additional structure, and the stochastic dimension-free result depends on adaptive variance. The paper does not establish that these assumptions are broadly realistic for modern ML objectives.
- The exposition is dense and hard to digest. The main story is understandable at a high level, but the paper introduces many layers of notation, several optimizer variants, and multiple theorem regimes, making it difficult to track what is new versus what is inherited from prior structured-optimizer analyses.
- The comparison with prior work is substantial but also reveals that much of the paper is an extension/unification of a fast-moving line of recent theory. The novelty is real but somewhat incremental: the paper synthesizes and extends recent results more than it opens a clearly new direction.

## Nice-to-Haves
- A compact summary table comparing deterministic nonconvex, convex accelerated, and stochastic NSD rates under standard versus adaptive assumptions would make the paper much easier to read.
- A simple toy example showing the gap between standard smoothness and adaptive smoothness, and when the factor-\(d\) separation is tight, would substantially improve intuition.
- A short discussion of when adaptive variance is plausible in practice, and how it relates to common covariance-bounded noise models, would help readers judge the scope of the stochastic result.

## Novel Insights
The most interesting insight is that the “geometry” used by adaptive optimizers is not just the same geometry with better bookkeeping: the relevant smoothness notion is genuinely different and strictly stronger than the standard norm smoothness used for NSD. That difference becomes meaningful in two places the paper highlights well: acceleration, where stronger adaptive smoothness can buy a true \(O(T^{-2})\)-type rate in the convex setting, and stochastic optimization, where the analogous adaptive variance condition can remove dimension dependence that standard variance cannot. The paper’s technical novelty is therefore less about inventing a new optimizer and more about formalizing a sharp analytical separation between two families of methods that are often informally conflated.

## Potentially Missed Related Work
- **Kovalev (2025a) — SGD with Adaptive Preconditioning: Unified Analysis and Momentum Acceleration** — directly relevant for the adaptive-optimizer/momentum side, and the present paper builds on its framework.
- **Kovalev & Borodich (2025) — Non-Euclidean SGD for Structured Optimization: Unified Analysis and Improved Rates** — relevant because the paper’s stochastic NSD analysis is closely related to this geometry-aware line.
- **Xie et al. (2025a, 2025b) — Adam Exploits \(\ell_\infty\)-geometry of Loss Landscape via Coordinate-wise Adaptivity; Structured Preconditioners in Adaptive Optimization** — central prior work; the present paper explicitly extends and unifies these results.
- **Balles et al. (2020) — The geometry of sign gradient descent** — relevant for the conceptual distinction between standard and adaptive geometry under \(\ell_\infty\)-type norms.

## Suggestions
- Add a single “results at a glance” table with the exact assumptions and rates for Theorems 3.1, 4.3, 4.5, 4.6, and 4.7.
- Include one concrete worked example and one small synthetic illustration to show the difference between standard and adaptive smoothness/variance.
- Make the scope boundaries explicit early: which optimizer families are covered exactly, which are only covered up to a meta-algorithm, and where diagonal/commutative structure is still essential.

---

## PFhrOUJZ5o

- GT: Reject (avg 5.0)
- Predicted: N/A (4.8/10)
- Match: N/A

### Final Review

## Summary
This paper introduces LAION-Comp, a 540K+ image dataset annotated with scene graphs over LAION-Aesthetics to improve compositional text-to-image generation. It also adds a lightweight scene-graph conditioning module for diffusion and flow-matching backbones, a new benchmark for complex scene generation, and an SG-based editing interface.

## Strengths
- The dataset is genuinely large for scene-graph supervision and targets an important weakness of current T2I systems: failure on multi-object scenes and non-spatial relations. The paper supports this with dataset statistics showing broader relation diversity than VG and with human verification indicating high annotation accuracy on sampled images.
- The benchmark and model evaluations are broad enough to show the idea has some practical value: the authors test on multiple backbones, compare against SG baselines, and report gains on their benchmark as well as COCO-Stuff, Visual Genome, and T2I-CompBench. The SG encoder is also lightweight relative to the generative backbones, which makes the approach more plausible for adoption.

## Weaknesses
- The central causal claim is not established cleanly: the paper argues that structured annotations, rather than architecture or extra training, drive the gains, but it does not provide the controlled ablations needed to support that. There is no matched comparison against caption-only recaptioning, no isolation of graph vs. encoder effects, and no convincing analysis of whether improvements come from better supervision or just a better-curated subset of LAION.
- Evaluation is too self-referential. CompSGen Bench is built from the held-out split of the same LAION-Comp source distribution, and the SG-IoU-style metrics depend on GPT-based extraction from generated images. That creates a serious circularity risk: the same family of models used for annotation and evaluation can bias the benchmark toward scene-graph-conditioned methods, making the reported gains less trustworthy than they appear.
- The modeling contribution is modest relative to the claims. The SG encoder is essentially a CLIP-encoded graph refinement module with a GNN and a scalar residual, which is a reasonable engineering choice but not a major methodological advance. The paper presents it as a more general foundation for compositional generation than the evidence justifies.
- The annotation pipeline remains only partially validated. The paper does report human checks and some error analysis, but the verification is limited relative to the scale of the dataset, and GPT-4o hallucination is acknowledged. This matters because the dataset quality, the benchmark construction, and several evaluation signals all depend on the same automated annotation stack.

## Nice-to-Haves
- A clearer stratified analysis by scene complexity, relation type, and object count would help show where LAION-Comp actually helps, rather than relying mostly on aggregate scores.
- The paper would benefit from a more explicit comparison between scene-graph conditioning and simpler alternatives such as free-form recaptioning, longer captions, or manual SGs on the same backbone and training budget.
- More detailed release artifacts would improve credibility: exact prompts, post-processing rules, failure distributions, and per-category annotation statistics.

## Novel Insights
The most interesting idea here is not the specific SG encoder, but the attempt to turn web-scale image data into structured supervision at a scale large enough to matter for modern generative models. That said, the paper’s strongest empirical story is also its weakest methodological point: the dataset, the benchmark, and the evaluation pipeline are tightly coupled, so the work currently demonstrates that structured supervision can help on structurally derived tests, but not yet that it cleanly generalizes as a robust solution to compositional generation. The editing results are a useful bonus, but they read more like an application of the same conditioning mechanism than a separate contribution.

## Potentially Missed Related Work
- LayoutGPT / LLM-grounded diffusion — relevant as a strong modern line of controllable generation that should be compared on the same backbone.
- GLIGEN, Ranni, MIGC, IFAdapter, RealCompo — relevant baselines for controllable multi-instance generation that are more directly comparable than older SG2IM papers.
- Scene-graph-based editing methods such as SGEdit — already partially covered, but still relevant as a direct precursor to the editing interface.

## Suggestions
- Add controlled ablations on the same backbone with matched data size comparing original captions, GPT-4o recaptions, and scene graphs.
- Evaluate on fully external compositional benchmarks with no source overlap, and report seed variance or confidence intervals.
- Release the exact annotation prompts, filtering heuristics, and error breakdowns, and validate the SG-based metrics against human judgments on a representative subset.

---

## hQZQVLJrH9

- GT: Withdrawn (treated as Reject) (avg 4.5)
- Predicted: N/A (4.6/10)
- Match: N/A

### Final Review

## Summary
This paper proposes a first-order geometric framework linking activation steering and training-data influence functions. The central claim is that, under local linearization and feasibility/alignment conditions, a steering vector can be represented as an influence reweighting over training examples, and conversely, with additional diagnostics for when steering is possible and when it is not.

## Strengths
- The unifying perspective is genuinely interesting: it connects two previously separate interpretability/control toolkits, and the paper does provide a coherent geometric language for comparing them.
- The paper goes beyond a slogan-level connection and gives constructive objects: an IAS vector via projection/pseudoinverse, an alignment diagnostic \(\omega(x)\), and a spectral criterion for choosing steering directions under a norm budget.

## Weaknesses
- The main equivalence is presented more broadly than the paper actually justifies. In the paper, the results are explicitly first-order, local, and conditioned on feasibility/subspace alignment assumptions; the abstract and introduction still read as if a near-universal duality were established. This matters because the practical and conceptual scope is much narrower than advertised.
- The empirical support is thin relative to the strength of the claims. The paper mostly reports one detoxification setup, one collinearity plot for first-order matching, one depth trend, and one vision spectral check. None of this convincingly validates the core claim that steering and influence are interchangeable in a useful data-level sense across models, tasks, perturbation sizes, or layers.
- The “identify responsible training examples” story is under-supported. The paper gives a mathematically defined signed measure, but it does not show that the top-weighted examples are actually more causally responsible than standard influence baselines or simple gradient similarity. Without such validation, the provenance claim remains speculative.

## Nice-to-Haves
- Add a clearer breakdown of when the theorem is exact, approximate, or inapplicable, ideally with a compact algorithm box for the full workflow.
- Include sensitivity analyses for damping, intervention magnitude, layer choice, and rank budget.
- Make the connection to behavioral effects more explicit, not just logit shifts, especially for detoxification and any provenance claims.

## Novel Insights
The most interesting insight is geometric rather than algorithmic: the paper reframes steering success as a question of subspace overlap between activation-reachable and parameter-influence directions. That view yields a practical diagnostic—\(\omega(x)\)—that could, in principle, tell practitioners when a cheap activation edit is likely to work and when it is fundamentally misaligned with the effect of training-data reweighting. This is a useful conceptual bridge, but the paper currently stops short of showing that the bridge is robust enough to support the broader claims it makes.

## Potentially Missed Related Work
- representer point methods / gradient-based data attribution — relevant for the “responsible training examples” claim and could provide stronger attribution baselines
- influence-function robustness papers — relevant because the paper’s first-order influence assumptions are precisely where prior work has identified fragility
- representation engineering / latent steering vector methods — relevant as closer prior art on activation steering constructions and diagnostics

## Suggestions
- Provide a direct end-to-end experiment: choose a steering vector, recover the implicated training examples, then verify by actually removing/reweighting those examples and measuring whether the behavior changes as predicted.
- Add robustness plots showing prediction error versus intervention magnitude and versus \(\omega(x)\), so the reader can see when the first-order regime truly holds and when it breaks.
- Benchmark the recovered training examples against standard influence methods and simple baselines to demonstrate that the proposed mapping adds real value beyond existing attribution tools.

---

## rBj2iVyrhh

- GT: Reject (avg 2.0)
- Predicted: N/A (3.6/10)
- Match: N/A

### Final Review

## Summary
This paper proposes CCAT, a two-stage training framework for multimodal imbalance: first pretraining a shared classifier with bidirectional cross-attention and a contribution regularizer, then freezing that classifier during modality-alternating training while using modality-specific LoRA adapters and a sample-level secondary update for severely imbalanced samples. The central claim is that alternating encoder updates alone do not prevent classifier bias toward fast-converging modalities, so constraining the classifier is necessary to better utilize weaker modalities.

## Strengths
- The paper targets a real and important failure mode in multimodal learning: alternating training can reduce encoder interference yet still leave a biased classifier, so focusing on classifier-level imbalance is a sensible and potentially useful direction.
- The framework is coherent and experimentally explored across three standard multimodal benchmarks spanning audio-visual and image-text settings. The ablation table indicates that freezing, LoRA, alternating training, and secondary updates each contribute to the final result, and the reported gains over prior baselines are substantial on the presented benchmarks.

## Weaknesses
- The method is under-specified in several places, which makes the core claim hard to verify or reproduce. In particular, the mutual-information-based contribution score, the regularizer that penalizes contribution disparity, and the exact placement/use of LoRA in the classifier are not clearly enough defined in the main text to understand what is actually being optimized.
- The theoretical justification is mostly intuitive rather than rigorous. The analogy between class imbalance and modality imbalance is interesting, but the paper overstates it as a “theoretical framework” without establishing a formal equivalence or proving that classifier bias is the decisive bottleneck.
- The experimental evidence is not yet strong enough to fully support the scale of the claims. The reported improvements are large, but the paper does not provide standard deviations or significance tests, baseline-parity details are limited, and the comparison protocol is not detailed enough to rule out differences in backbone choice, tuning budget, or training recipe across methods.
- The ablations do not cleanly isolate the value of the main design choices. The paper shows component removals, but it still does not convincingly separate the effect of classifier freezing from the extra capacity/adaptation introduced by LoRA and the extra sample-level retraining pass.
- The evaluation scope is narrow for the strength of the claim. All experiments are on paired two-modality benchmarks with standard encoders; there is no robustness analysis for missing/noisy modalities, no out-of-domain test, and no study showing how performance changes with imbalance severity.

## Nice-to-Haves
- Add direct measurements of modality balance, such as per-modality gradient norms, contribution entropy, or training-time contribution trajectories, to show that CCAT really reduces dominance rather than only improving final accuracy.
- Include a more explicit algorithmic description or pseudocode for the two-stage procedure, especially the contribution estimation and the secondary-update selection rule.

## Novel Insights
The most interesting idea is that modality imbalance is not just an encoder problem; once one modality converges faster, it can “capture” the classifier early and impose a persistent structural preference that later training struggles to undo. This classifier-centric view is the paper’s main conceptual contribution, and it is more compelling than another purely gradient-balancing heuristic. That said, the paper’s implementation choices—frozen classifier, LoRA adapters, and sample-level retraining—look like a reasonable engineering package around that insight rather than a deeply new optimization principle.

## Potentially Missed Related Work
- None identified

## Suggestions
- Provide a tighter method section with explicit formulas for the contribution estimator, regularizer, and LoRA insertion point, plus pseudocode for both training stages.
- Run a controlled comparison where the same backbone, preprocessing, and schedule are used for all baselines and for frozen-classifier versus non-frozen variants.
- Report mean ± std over more seeds and add sensitivity plots for the threshold β and LoRA rank.
- Add diagnostics showing whether classifier freezing actually stabilizes modality contributions over training, and whether the secondary updates target genuinely hard or imbalanced samples.

---

## Ksvv8x00eo

- GT: Withdrawn (treated as Reject) (avg 3.5)
- Predicted: N/A (4.4/10)
- Match: N/A

### Final Review

## Summary
CaTS-Bench is a benchmark/resource paper for context-aware time series captioning and reasoning. It combines 11 real-world datasets with numeric series, metadata, plot images, oracle-generated reference captions, a smaller human-revisited subset, and 460 multiple-choice Q&A items, then evaluates several proprietary and open-source VLMs with custom numeric metrics.

## Strengths
- The paper tackles a genuinely underexplored problem: describing numeric time series with rich context, rather than only synthetic patterns or shallow trend labels. The benchmark spans 11 real-world datasets across multiple domains and includes both captioning and diagnostic Q&A, which is a meaningful step beyond prior time-series captioning resources.
- The benchmark is broader than most existing alternatives and is designed with multiple modalities and multiple evaluation axes. In particular, the inclusion of metadata, plots, numeric sequences, and custom numeric/statistical metrics is a reasonable attempt to evaluate whether models can go beyond n-gram overlap and actually preserve quantitative content.
- The authors do make a serious effort to validate the semi-synthetic reference pipeline: manual factual checks on a large caption subset, a human detectability study, paraphrase-based robustness checks, and diversity analysis. Even if these are not fully decisive, they are stronger than the usual lightweight justification seen in benchmark papers.

## Weaknesses
- The benchmark’s core ground truth is still generated by a single oracle LLM, with only a small human-revisited subset. That is the paper’s biggest fragility: the benchmark may partly reward imitation of Gemini-style phrasing and oracle-specific tendencies rather than robust time-series understanding. The paper acknowledges this limitation, but the main claims remain stronger than the evidence warrants.
- The case for multimodality is weak. The paper’s own ablation suggests that removing the plot often has little effect, and sometimes even helps. This is a serious issue because it means the benchmark may not actually require vision in many cases; it may mostly test metadata-conditioned text generation and numeric extraction from the input series.
- The Q&A suite is useful as a diagnostic probe, but it is still template-driven and filtered using Qwen-based screening. That makes it less convincing as a general reasoning benchmark and raises the possibility that the questions are optimized around the construction pipeline rather than exposing broad reasoning ability.
- The custom numeric metrics are plausible but under-validated. The 5% tolerance is asserted rather than convincingly justified across heterogeneous domains, and the paper does not show that these metrics correlate with human judgments or are robust to different extraction rules. Without that, it is hard to know whether the scores reflect real caption quality or just numeric token matching behavior.
- Several claims are overstated relative to the evidence, especially around human indistinguishability and “reliable” references. The studies are useful, but they do not fully rule out style bias, oracle artifacts, or benchmark-specific shortcuts.

## Nice-to-Haves
- A larger human-authored or expert-verified gold subset across more domains would materially strengthen the benchmark’s credibility.
- More controlled modality ablations would help separate the value of metadata, raw numeric values, and plot images, rather than collapsing all non-visual inputs together.
- A sensitivity analysis for the 5% numeric tolerance and a validation of the custom metrics against human ratings would make the evaluation much more persuasive.

## Novel Insights
The most interesting insight is not simply that models are weak on time series captioning, but that many of them appear to succeed through textual priors rather than genuine visual grounding. The paper’s own modality ablations and attention visualizations suggest a troubling pattern: the plot image is often redundant, while the models lean heavily on metadata and language patterns. That is an important diagnostic result, but it also undercuts the benchmark’s multimodal framing and suggests that the field still lacks models that truly fuse numeric sequences, context, and visual evidence into a single coherent representation.

## Potentially Missed Related Work
- TRUCE — relevant as an earlier time-series captioning benchmark with truth-conditional framing; useful for a more direct shared-protocol comparison.
- TACO — relevant as a large-scale time-series captioning corpus and a close point of comparison for scale and caption quality.
- TADACap — relevant because it also tackles time-series captioning and domain-aware retrieval, making it important for positioning the novelty of CaTS-Bench.
- VisText — relevant as a chart-captioning benchmark; while not a time-series dataset per se, it is close in spirit for multimodal caption evaluation.
- PromptCast / other prompt-based time-series benchmarks — relevant for contrasting metadata/context conditioning with purely forecasting-style tasks.

## Suggestions
- Add a shared-evaluation comparison against TRUCE, TACO, and TADACap, ideally with the same models and prompts where possible.
- Report controlled ablations for metadata-only, numeric-only, plot-only, and full-input settings, so it is clear whether vision actually matters.
- Release a substantially larger human-authored or expert-edited subset, and use it as the primary evaluation set for at least one main result table.
- Validate the numeric metrics against human judgments and report sensitivity to the 5% tolerance threshold.
- Provide domain-wise and window-length-wise breakdowns, because averages over 11 datasets can hide where the benchmark is easy or brittle.

---

## C6WWMryELL

- GT: Reject (avg 5.5)
- Predicted: N/A (3.8/10)
- Match: N/A

### Final Review

## Summary
This paper tackles an important but under-measured problem in long-form LLM generation: not just whether a model can hit a target length once, but whether its outputs are stable across repeated samplings. It introduces VOLTBench, a heterogeneous benchmark spanning structured and unstructured tasks in English and Chinese, probes attention traces to characterize failure modes, and proposes SELB, a decoding-time logits-boosting method intended to improve length control and stability without additional training.

## Strengths
- **The paper targets a real gap: cross-sample volatility in long-form generation.** Most prior benchmarks assess one generation per prompt; this work explicitly measures variability over multiple runs using standard deviation/variation-style metrics. That is a meaningful and practically relevant shift, especially for deployment settings where token cost and reliability matter.
- **VOLTBench is broad and reasonably ambitious in scope.** It spans multiple task families (story, dialogue, diary, architecture, code, math, user/company profiles), two languages, and multiple instruction complexities, and it includes both structured and unstructured outputs. This makes the benchmark more informative than single-domain long-form evaluations.
- **SELB is simple and deployable.** The mitigation is a decoding-stage method with no additional training, which is appealing from an engineering standpoint. The paper also shows that it can substantially increase mean length and reduce volatility on the benchmark’s long structured settings.

## Weaknesses
- **The benchmark and the mitigation are too tightly coupled, which weakens the generality of the claims.** VOLTBench is clearly designed to expose the same failure modes that SELB then fixes, and many of the strongest results are shown on tasks with explicit section anchors and handcrafted constraints. That makes it hard to tell whether the method solves a general long-form generation problem or mostly a benchmark-shaped control problem.
- **The mechanistic story based on attention traces is suggestive but not convincing as evidence of causality.** The paper identifies “attention collapse” and “attention instability,” but the analysis is observational and based on a limited set of traces. This does not establish that the attention patterns cause volatility; at best, they correlate with it.
- **The evaluation is underpowered for a paper making volatility claims.** The benchmark uses only five samples per prompt, yet the paper’s central object is output variability. For a stability paper, that is a thin sampling budget, and the manuscript provides no confidence intervals or significance analysis to show the reported improvements are robust.
- **The reported gains are not always interpretable cleanly because length and quality are entangled.** In several tables, very high quality scores are reported on outputs that are short, incomplete, or structurally off-target, while in other places the paper celebrates longer outputs even when they overshoot the requested length. The paper needs a crisper distinction between “longer,” “closer to target,” and “better.”
- **The mitigation method is still more of a decoding heuristic than a principled solution.** SELB relies on hard token suppression, EOS banning, and forced structural boosts. Those interventions may work, but the paper does not yet show that this is robust across models, prompt styles, or decoding settings, nor does it sufficiently quantify possible side effects such as repetition or unnatural continuations.

## Nice-to-Haves
- A cleaner ablation separating title boosting, EOS suppression, banned-token suppression, and the hybrid free-form “keep-alive” logic.
- A stronger comparison against more directly relevant constrained-decoding and length-control baselines.
- A small human evaluation or audit of quality on the hardest settings, especially to validate that the method is not merely inflating output length.

## Novel Insights
The most interesting insight is that long-form generation failure is not just an average-performance issue but also a stability issue: some models may produce reasonably good long outputs in one run yet collapse, truncate, or skip structure in another. The paper’s attention-trace analysis suggests that these failures are preceded by recognizable internal dynamics rather than being purely random, and SELB’s success indicates that relatively lightweight decoding-time controls can have a large effect on output stability. However, the current evidence still looks more like strong empirical correlation and effective control engineering than a deep explanation of why long-form generation breaks.

## Potentially Missed Related Work
- **Constrained decoding / grammar-constrained decoding / structured decoding methods** — relevant because SELB is essentially an inference-time control method, and these baselines would better situate its novelty.
- **LongWriter, LongWriter-Zero, and inference-time training for long text generation** — the paper cites some of these, but they are especially relevant as closer competitors for the mitigation setting.
- **Prior long-form benchmarks with structure or length constraints** — relevant for positioning VOLTBench as extending, rather than replacing, earlier length-following and procedural generation evaluations.

## Suggestions
- Provide a rigorous ablation study and report confidence intervals over prompts and seeds.
- Add stronger baseline comparisons for inference-time length/structure control.
- Clarify metric behavior when outputs overshoot the target but remain high quality.
- Reframe the attention analysis as correlational unless causal evidence is added.
- Release the full prompt construction, parsing, and evaluation code so the benchmark can be independently audited.

---

## Me0n0iESJY

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (4.2/10)
- Match: N/A

### Final Review

## Summary
This paper presents a benchmark and study of model merging for multimodal LLMs, covering both capability merging across tasks such as VQA, geometry, chart, OCR, and grounding, and modality merging across vision/audio/video. It also proposes OptMerge, a data-free optimization-based merging method that denoises task vectors via low-rank approximation and tries to stabilize merged-vector optimization.

## Strengths
- The paper tackles a timely and genuinely underexplored problem: model merging for MLLMs, including a split between capability merging and modality merging. This is a useful framing, and the benchmark spans several distinct capability categories rather than collapsing everything into generic multimodal QA.
- The empirical scope is fairly broad for a merging paper. The authors evaluate multiple merge methods, multiple base families/scales, actual Hugging Face checkpoints, and a general downstream multimodal QA suite. The code/checkpoint release is also a real plus for reproducibility.

## Weaknesses
- The proposed method is only a modest refinement over prior optimization-based merging, and the novelty is limited. OptMerge largely reuses WUDI-style task-vector optimization and adds a collection of heuristics: low-rank truncation, mean initialization, and SGD for LoRA cases. The paper does not convincingly establish a new principle beyond “reduce noise and norm growth,” which makes the algorithmic contribution feel incremental.
- The method is heavily heuristic, with several ad hoc choices that are not well justified: the rank is set to roughly one-fifth of each task vector rank, the optimizer choice changes by setting, and the implementation relies on coefficient search for merging. This weakens the claim that the method is a principled data-free solution rather than a tuned recipe.
- The theoretical section is suggestive but not very convincing. The bound in Theorem 3.1 mainly formalizes the intuition that smaller parameter drift helps merging; it does not provide a sharp or especially predictive explanation of when merging succeeds or fails. The assumptions are strong, and the proof sketch reads more like a post hoc justification than a deep theory.
- The empirical gains are often small, and the paper does not provide variance, confidence intervals, or multi-seed robustness. That matters because many reported improvements are fractions of a point to a few points at most, so it is hard to tell whether the method is consistently better or just within noise in some tables.
- The comparison to mixture training is interesting but not fully clean. In particular, the “mixture training baseline” is not always a like-for-like baseline with the same training recipe and data setup, so the claim that merging can surpass multi-task training should be interpreted cautiously.

## Nice-to-Haves
- A stronger ablation that isolates each design choice would help: low-rank truncation, SGD vs. Adam, mean initialization, and the layer selection strategy should each be tested independently and across both full fine-tuning and LoRA settings.
- The benchmark would be more convincing with broader coverage beyond the current family of model architectures and a stronger reasoning-heavy setting. The paper already acknowledges this limitation, and additional evidence would help substantiate the generality claim.
- More analysis of failure cases would improve the paper: when do task vectors become too divergent, when does low-rank truncation remove useful signal, and when does merging harm out-of-scope capabilities?

## Novel Insights
The most interesting insight in the paper is that merging quality appears to depend less on how strong each expert is in isolation and more on how “merge-friendly” its parameter update is. In other words, smaller and more structured task vectors can be better merge candidates than aggressively optimized experts, because over-training increases interference and norm growth. This is a useful practical lesson, and it also explains why the benchmark design intentionally keeps fine-tuning drift modest.

## Potentially Missed Related Work
- AdaMMS — relevant because it is an MLLM-specific merging method with unsupervised hyperparameter selection, and the paper’s benchmark/method positioning should be compared more directly against it.
- UQ-Merge — relevant because it addresses multimodal merging with uncertainty-guided selection and is close in spirit to this work’s MLLM setting.
- EMR-Merging — relevant as a more recent tuning-free merging method that could strengthen the comparison set.
- Model merging in LLMs, MLLMs, and beyond — relevant as a recent survey that may contain closely related multimodal merging baselines and framing.

## Suggestions
- Add a clean ablation table that separates the effect of low-rank denoising, optimizer choice, initialization, and coefficient search, with multiple seeds.
- Report variance/error bars on the main benchmark tables, especially where gains are small.
- Tighten the comparison to mixture training so it is clear the baseline is matched in data, training recipe, and tuning budget.
- Expand the benchmark or add supplemental experiments on additional architectures and a harder reasoning-centric multimodal suite to better support the generality claim.

---

## GiaF5cFIpI

- GT: Reject (avg 3.5)
- Predicted: N/A (3.7/10)
- Match: N/A

### Final Review

## Summary
This paper proposes a real-time framework for adaptive neural stimulation in latent dynamical systems, combining streaming latent-space estimation, online stimulus-response modeling, and constrained optimization to choose high-dimensional stimulations that steer low-dimensional neural activity. The paper evaluates the framework on a toy system and several neural datasets, with emphasis on fast runtimes and robustness to changing stimulation-response mappings.

## Strengths
- The paper tackles an important and practically relevant problem: closed-loop stimulation under streaming, partially observed neural dynamics. The combination of latent-state tracking, response learning, and stimulation design is coherent and directly aligned with real neuroscience constraints.
- The framework is fairly comprehensive and explicitly addresses realistic issues that many prior papers ignore, including sparse stimulation, non-negativity, delayed responses, time-varying stimulus-response maps, and runtime feasibility. The appendix also shows the system can run fast enough for real-time use in principle.

## Weaknesses
- The main algorithmic novelty is limited. The paper largely combines existing ideas — streaming dimensionality reduction, online filtering, kernel regression, and constrained optimization — into one pipeline, but does not convincingly isolate a fundamentally new method that would clear ICLR’s bar on technical originality.
- The empirical validation is still weak for the paper’s central claim. Most “real-data” stimulation results are based on simulated stimulation injected into retrospective datasets, not true online closed-loop experiments, so the work does not actually demonstrate the claimed adaptive stimulation loop in vivo.
- The experimental comparisons are not strong enough. For stimulation design, the paper mainly compares against random, shuffled, and blind baselines, which is insufficient to establish that the proposed optimizer is meaningfully better than strong modern alternatives such as gradient-based control, Bayesian optimization, or active-learning-style targeting.
- Several key components are under-specified. In particular, the online kernel regression updates, bandwidth selection, optimization initialization/robustness, and the exact model-selection rule across latent spaces are not described at the level needed for easy reproduction or confidence in the results.
- The paper lacks the ablation structure needed to support its end-to-end claims. It is not clear how much of the gain comes from proSVD versus sjPCA/mmICA, from the stimulus-response model versus the dynamics model, or from the feasibility constraints versus the optimization objective itself.

## Nice-to-Haves
- A real closed-loop stimulation experiment, or at least an offline replay that uses intervention-corrected rollouts, would substantially strengthen the paper.
- More statistical reporting — confidence intervals, per-run variability, and significance tests — would make the reported improvements more convincing.
- A cleaner decomposition of the pipeline into isolated modules and ablations would help readers understand which pieces are actually doing the work.

## Novel Insights
The most interesting aspect of the paper is not any single module, but the attempt to unify streaming representation learning, adaptive perturbation modeling, and stimulation synthesis into one online loop that can adapt to nonstationary neural response maps. The parallel comparison of multiple latent spaces is also a useful idea, but the current evidence mostly shows that the framework can fit and react to logged stimulation effects, not that it can reliably discover causal control policies under true closed-loop biological feedback. In other words, the paper is more compelling as a systems prototype than as a fully validated method for adaptive neural control.

## Potentially Missed Related Work
- MiSO (Minai et al., 2024) — directly relevant prior work on optimizing brain stimulation to create neural activity states.
- Active learning of neural population dynamics using two-photon holographic optogenetics (Wagenmaker et al., 2024) — relevant for adaptive stimulation design and sequential experimental design.
- Bayesian target optimisation for high-precision holographic optogenetics (Triplett et al., 2023) — strong baseline direction for constrained stimulation search.
- Modeling and prediction of the dynamic responses of large-scale brain networks during direct electrical stimulation (Yang et al., 2021) — relevant for stimulation-response modeling in neural systems.
- Direct neural perturbations reveal a dynamical mechanism for robust computation (O’Shea et al., 2022) — relevant for causal perturbation of neural dynamics and latent-state interpretation.

## Suggestions
- Add at least one genuine closed-loop stimulation result, or clearly reframe the paper as an offline emulation study rather than a validated real-time control system.
- Include stronger optimization baselines and a more extensive ablation suite to show that each component of the pipeline is necessary.
- Clarify the online update rules and model-selection logic with precise pseudocode and hyperparameter details.

---

## cZFgsLq8Gs

- GT: Accept (Poster) (avg 4.0)
- Predicted: N/A (3.7/10)
- Match: N/A

### Final Review

## Summary
This paper presents DeepScientist, a multi-agent LLM system that frames autonomous scientific discovery as a Bayesian optimization loop over a persistent Findings Memory. It is evaluated on three AI research tasks and claims to discover methods that surpass the selected human SOTA baselines, while also analyzing the discovery trajectory, scaling behavior, and the quality of the generated papers.

## Strengths
- The paper tackles an important and timely problem: whether an autonomous system can do more than generate novelty and instead make measurable progress on demanding, human-defined AI tasks. The long-horizon, goal-directed framing is intellectually appealing and squarely relevant to ICLR.
- The system is described concretely enough to be more than a vague “agentic loop”: it includes a three-stage workflow, persistent memory, surrogate scoring, acquisition-based selection, implementation/verification, and report generation. The paper also reports substantial scale of operation, including thousands of generated ideas and many validated attempts.

## Weaknesses
- The central claim is overstated relative to the evidence. The paper repeatedly presents benchmark improvements on three AI tasks as evidence of “scientific discovery” and “first large-scale empirical demonstration” of frontier-pushing AI, but the tasks are still essentially method optimization on existing codebases with heavy human supervision and proprietary-model infrastructure. This is impressive engineering, but it does not yet justify the broad epistemic claim being made.
- The contribution is not cleanly isolated. It remains unclear how much of the gain comes from the proposed Findings Memory/Bayesian selection framework versus the sheer use of large foundation models, massive compute, careful baseline reproduction, and human verification. The paper does not provide the ablation evidence needed to separate these factors, so the causal story is weak.
- The evaluation is not rigorous enough for the strength of the claims. Key results are mostly single-run headline numbers without confidence intervals or run-to-run variance, and the comparison set is narrow: one selected human SOTA per task anchors the main story. For a stochastic, expensive, multi-agent system, this is not enough to establish robustness or rule out favorable selection of runs, tasks, or starting points.
- Autonomy is somewhat overstated. The paper does acknowledge human supervision and verification, but the practical dependence on humans, curated baselines, and proprietary models is substantial. That matters because it limits the interpretation of “autonomous discovery” and makes the system less general and less reproducible than the narrative suggests.

## Nice-to-Haves
- A cleaner separation between the system paper and the downstream discovered methods would improve readability and make the main contribution easier to assess.
- Stronger reporting of uncertainty, sensitivity to backbone choice, and compute-normalized comparisons would make the results more credible.
- A more transparent accounting of human intervention would help calibrate the autonomy claim.

## Novel Insights
The most interesting aspect of the paper is not simply that it finds better methods, but that it operationalizes autonomous research as a resource-allocation problem over a memory of prior successes and failures. That framing is plausible and potentially valuable: in effect, DeepScientist is trying to turn research into an iterative search process with explicit filtering, rather than a one-shot generation pipeline. However, the results also expose the main bottleneck in current AI science systems: the system can generate many ideas, but far fewer are actually valid, implementable, and persuasive, so the real advantage comes from aggressive triage plus human-grade engineering discipline rather than from a clearly demonstrated leap in scientific reasoning.

## Potentially Missed Related Work
- AI Scientist / AI Scientist-V2 — directly relevant prior autonomous-science systems; the paper builds on and should be compared more rigorously against them.
- AlphaEvolve — relevant as a high-scale coding/search-based discovery system with strong algorithmic optimization framing.
- PaperBench / Paper2Agent — relevant if the paper wants to position itself against automated research, reproduction, and agentic paper workflows.
- Agent Laboratory / MLE-Bench — relevant as closer baselines for agentic ML engineering and search under task constraints.
- AI-Descartes / AI-Hilbert — relevant for the paper’s own “goal-driven discovery” framing and discussion of constrained scientific search.

## Suggestions
- Add a controlled ablation that removes Findings Memory and replaces UCB selection with random or simple heuristic selection under matched compute.
- Report multiple independent runs, with confidence intervals, on the main three tasks.
- Compare against stronger search baselines and agentic pipelines under the same compute and model budget.
- Explicitly quantify the amount of human supervision and manual debugging involved in each task.
- Tone down the “scientific discovery” rhetoric unless the paper can show that the gains are not primarily benchmark-specific engineering improvements.

---

