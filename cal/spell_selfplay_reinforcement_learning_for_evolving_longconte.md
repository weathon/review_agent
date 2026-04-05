=== CALIBRATION EXAMPLE 69 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Does the title accurately reflect the contribution?**  
  Yes. “SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models” accurately signals the core idea: self-play RL for long-context LMs.
- **Does the abstract clearly state the problem, method, and key results?**  
  Mostly yes. It clearly identifies the bottleneck in long-context reasoning, describes the three-role framework, and reports benchmark gains.
- **Are any claims in the abstract unsupported by the paper?**  
  A few claims are somewhat stronger than the evidence presented:
  - “**label-free optimization**” is overstated because the paper still relies on substantial human-designed scaffolding: preprocessed corpora, task templates, reference answers generated from documents, and manual benchmark construction choices (see Appendix B and G). It is label-free in the sense of avoiding human annotations for training pairs, but not fully label-free in an absolute sense.
  - “**outperforms equally sized models fine-tuned on large-scale annotated data**” is only indirectly supported. The paper compares against base/instruct models and an RLVR baseline, but not against a strong supervised fine-tuning baseline on the same data scale in a controlled ablation.
  - The claim that SPELL “**enables scalable**” optimization is plausible, but the actual training cost is still substantial and the framework is limited to 16K training context (Appendix B, F.1).

### Introduction & Motivation
- **Is the problem well-motivated? Is the gap in prior work clearly identified?**  
  Yes. The introduction makes a strong case that long-context reasoning is under-served by existing RLVR methods, especially because rewards are harder to verify semantically and annotations are unreliable/costly. The gap between short-context RLVR and long-context reasoning is clearly articulated.
- **Are the contributions clearly stated and accurate?**  
  Largely yes: the paper claims a three-role self-play loop, an automated curriculum, and a verifier trained via self-consistency. Those are indeed the main contributions.  
  However, the introduction somewhat blurs what is genuinely new versus a recombination of known ideas from self-play, self-consistency, curriculum learning, and verifier training. For ICLR, the novelty is plausible but should be more crisply separated from prior art in self-play and difficulty filtering.
- **Does the introduction over-claim or under-sell?**  
  It leans toward over-claiming in two places:
  - It suggests a “fundamental challenge” is solved by self-play, but the paper’s method still depends on proxy rewards, external judges in some analyses, and task-specific prompt engineering.
  - The claim that SPELL “outperforms the leading gemini-2.5-pro in pass@4” is attention-grabbing, but this comparison is only shown for one model in Figure 3 and should be interpreted cautiously because the evaluation protocol, sampling budget, and benchmark mix matter a lot.

### Method / Approach
- **Is the method clearly described and reproducible?**  
  The framework is fairly well described, especially in Section 3 and Algorithm 1. The three roles, reward definitions (Eqs. 4–10), and sampling strategy are understandable.  
  That said, reproducibility is weakened by several missing operational details:
  - How exactly are “reference answers” constructed for the questioner from raw documents? The paper says raw documents are paired with reference answers, but the generation pipeline is only partially specified.
  - The grounding filter is important, but the exact criterion and failure rate are not fully quantified.
  - The verifier’s semantic judgment prompt and calibration behavior are critical to the method, yet their robustness is only partially evidenced.
- **Are key assumptions stated and justified?**  
  Some are, but others are implicit:
  - The method assumes the questioner can reliably generate answerable questions from documents and that the responder’s success rate around 0.5 is an effective learning frontier. This is motivated, but not deeply justified empirically beyond Figure 5 and the Gaussian reward analysis.
  - The method assumes majority vote over verifier rollouts is a reliable proxy for semantic equivalence. This is a strong assumption, especially because verifier errors can self-reinforce in a closed loop.
- **Are there logical gaps in the derivation or reasoning?**  
  Yes, a few:
  1. **Questioner reward definition (Eq. 7)**: the Gaussian reward is conceptually reasonable, but the paper does not fully justify why a symmetric peak at 0.5 is optimal for all task types, especially for heterogeneous tasks like financial math, general QA, and multiple-choice.
  2. **Unified optimization (Eq. 10)**: the method shares one policy across roles, but the paper does not clearly analyze whether gradients from the three roles conflict or how role prompts prevent catastrophic interference.
  3. **Dynamic sampling**: the claim that it reduces the training set to roughly 1/G is useful, but the exact sample selection bias introduced by discarding zero-variance groups is not analyzed. This could skew learning toward intrinsically ambiguous examples.
  4. **On-policy simplification (Appendix E.3, Eq. 12)**: the paper states the policy is “strictly on-policy” and the importance ratio is 1, but this seems to rely on a very specific training regime. It would be important to explain whether all role rollouts are generated before any update and whether the same policy version is used consistently across roles.
- **Are there edge cases or failure modes not discussed?**  
  Some are discussed in F.6, but the method still has important unresolved failure modes:
  - If the questioner learns to generate examples that are only superficially grounded but semantically ambiguous, the verifier could still amplify noise.
  - The system may overfit to document artifacts or prompt templates rather than robust reasoning.
  - For long-context tasks requiring retrieval across multiple distant spans, the questioner’s curated subsets may not reflect realistic query distributions.
- **For theoretical claims: are proofs correct and complete?**  
  There are no formal theorems, so this mainly concerns derivations. The mathematical definitions are mostly acceptable, though Eq. (12) in Appendix E.3 appears to be a simplification of the GRPO objective under on-policy updates that is more asserted than derived. The paper would benefit from a clearer derivation showing exactly which terms vanish and why.

### Experiments & Results
- **Do the experiments actually test the paper's claims?**  
  Mostly yes. The main table (Table 1) tests the core claim that SPELL improves long-context reasoning across models and lengths. Table 9 compares against long-context alignment baselines. Figures 3–5 and Tables 2–4 probe scaling, ablations, reward mapping, and judge choice.  
  However, some central claims remain only partially tested:
  - “label-free optimization” is not directly compared against supervised annotation-based long-context training under matched compute/data settings.
  - “self-play curriculum adapts to evolving capabilities” is supported qualitatively, but there is no direct comparison to alternative curriculum mechanisms beyond the static RLVR baseline.
- **Are baselines appropriate and fairly compared?**  
  Generally yes, and this is a strength of the paper. The RLVR baseline and long-context alignment baselines are reasonable. The authors also reimplement LongPO/SoLoPO/QwenLong-L1 in Table 9, which is good practice.  
  But there are still fairness concerns:
  - The paper uses a strong judge (gpt-oss-120b) in evaluation for semantic equivalence, and also uses it in some ablations. The interaction between training reward design and evaluation judge can make gains hard to disentangle.
  - The RLVR baseline is synthesized using DeepSeek-R1-0528 and filtered with a verifier, which may or may not be directly comparable in quality/diversity to SPELL’s dynamically generated curriculum.
  - Some comparisons to “instruction-tuned counterparts trained on extensive human-annotated data” are broad and not controlled.
- **Are there missing ablations that would materially change conclusions?**  
  Yes. Important missing or underdeveloped ablations include:
  1. **Role separation vs shared prompting**: Is performance due mainly to multi-role self-play, or could a simpler self-generated QA plus verifier pipeline match much of the gain?
  2. **Memory size sensitivity beyond L=3**: The paper only reports history memory removal, not a sweep over memory length or whether memory contents need to be recent solvable examples specifically.
  3. **Subset size m and data composition**: Since question generation depends on sampled documents, it would be useful to show sensitivity to m and to the mix of task types.
  4. **Verifier rollout count G for verifier/questioner separately**: Table 3 varies G, but not in a way that isolates which component matters most.
  5. **Judge quality**: The paper compares rule-based vs external judge, but not systematic sensitivity to the verifier’s accuracy or calibration over training.
- **Are error bars / statistical significance reported?**  
  Partially. The paper reports averages over eight runs in Section 4.1, which is good, but the main tables do not appear to include standard deviations or confidence intervals. For a paper making several 1–3 point claims, this is a notable omission. ICLR would benefit from variance estimates, especially since some improvements are modest.
- **Do the results support the claims made, or are they cherry-picked?**  
  The results broadly support the claim that SPELL improves performance. The gains are consistent across many tables.  
  Still, a few claims are stronger than the evidence:
  - “outperforms equally sized models fine-tuned on large-scale annotated data” is not rigorously substantiated.
  - “raises its performance ceiling” is supported by pass@k curves for one model, but this is not enough to generalize broadly.
  - Some ablation effects are small or inconsistent (e.g., in Table 4, the external judge variants produce similar average scores), suggesting the mechanism is more nuanced than the prose implies.
- **Are datasets and evaluation metrics appropriate?**  
  Mostly yes. The chosen long-context benchmarks are relevant and diverse. The combination of accuracy and semantic equivalence via LLM-as-judge is appropriate for free-form QA.  
  However:
  - Using maximum of CEM and LLM-as-judge for evaluation may inflate scores if either metric is noisy. The paper should more explicitly justify this choice and analyze disagreement cases.
  - The training data is built from Ultra-Fineweb and DocMath, but the paper does not fully establish that the question-answer generation process avoids leakage or overly synthetic artifacts that would simplify the tasks.

### Writing & Clarity
- **Are there sections that are confusing or poorly explained?**  
  The main exposition is clear, but some parts remain underspecified:
  - The precise construction of the questioner’s reference answers from documents is not fully transparent.
  - The role of the verifier in “calibrating non-verifiable rewards” is conceptually interesting but only partially explained mechanistically.
  - Appendix E.3’s simplified objective is hard to follow without a step-by-step derivation.
- **Are figures and tables clear and informative?**  
  Generally yes. Table 1 is useful and comprehensive; Tables 2–4 support the ablation narrative; Figures 3–5 are aligned with the claims.  
  The main clarity issue is not visual design but interpretability:
  - Figure 3’s pass@k claim is compelling, but the reader needs more detail on the benchmark aggregation and whether improvements are uniform across tasks.
  - Figure 4’s difficulty curves are helpful, but the measure “1 - pass@1 with an external responder” is indirect; the paper should explain why this is the right proxy for question difficulty.
  - Figure 5’s reward mapping comparison is informative, yet the narrative relies on a causal story that is not fully isolated experimentally.

### Limitations & Broader Impact
- **Do the authors acknowledge the key limitations?**  
  Yes, partially. Appendix B acknowledges the lack of theory, the 16K training limit, and some human intervention in preprocessing and prompt crafting.
- **Are there fundamental limitations they missed?**  
  Yes:
  - **Closed-loop self-reinforcement risk**: The verifier and questioner can co-adapt to each other’s biases. The paper discusses reward hacking, but not the broader issue of self-generated distributional drift.
  - **Dependence on strong base models**: The method may work best once the policy is already capable enough to generate useful questions and semantic judgments. This is only lightly acknowledged.
  - **Generalization beyond document QA**: SPELL may be tightly coupled to document-centric reasoning and may not transfer to tasks without clear latent document evidence.
  - **Evaluation contamination risk**: Given the heavy use of web-scale corpora and multiple benchmarks, the decontamination story is important but not deeply validated.
- **Are there failure modes or negative societal impacts not discussed?**  
  The ethics statement is generic. It does not address that self-play systems can amplify hidden biases, generate persuasive but incorrect long-form answers, or be used to automate training of increasingly capable reasoning models with limited oversight. For ICLR standards, that is acceptable but somewhat underdeveloped given the paper’s emphasis on autonomous self-improvement.

### Overall Assessment
SPELL is a strong and timely paper for ICLR: it tackles an important and underexplored problem—how to do RL for long-context reasoning without human labels—and presents a coherent self-play framework with broad empirical gains across many models and benchmarks. The experimental section is substantially richer than average, and the ablations generally support the design choices. That said, the paper’s central claims are somewhat stronger than the evidence in a few places, especially around “label-free” training, superiority to annotated-data fine-tuning, and the exact mechanism by which the three-role loop avoids self-reinforcing errors. The methodology is promising and likely publishable, but for ICLR’s bar I would want clearer controlled comparisons, more variance reporting, and a sharper accounting of what is truly new versus recombining known self-play and curriculum ideas.

# Neutral Reviewer
## Balanced Review

### Summary
This paper presents SPELL, a self-play reinforcement learning framework for long-context language models in which a single model alternates among three roles: questioner, responder, and verifier. The key idea is to generate training questions from document clusters, solve them with full-document context, and use a learned verifier plus rule-based checks to provide rewards, while a curriculum and history memory gradually increase task difficulty. The paper reports consistent gains across 12 open-source models and six long-context benchmarks, with especially notable improvements for stronger reasoning models and better test-time scaling.

### Strengths
1. **Addresses an important and timely problem for ICLR.** Long-context reasoning remains a major bottleneck for LLMs, and the paper targets a clear gap: reinforcement learning methods that work in short-context settings but do not transfer cleanly to long-document reasoning. This is a substantive and relevant problem for the ICLR audience.

2. **Novel self-play formulation with three roles in one model.** The questioner/responder/verifier setup is a meaningful extension beyond prior self-play or RLVR methods that typically rely on two roles or external verifiers. The paper makes a concrete case that semantic verification is needed because long-context answers can be correct without matching strings exactly.

3. **Strong empirical breadth.** Experiments cover 12 models spanning base, instruction-tuned, reasoning, dense, and MoE architectures, which is broader than many papers in this area. The reported improvements are consistent across multiple benchmarks and model scales, which supports the claim that the method is not narrowly tuned to one backbone.

4. **Competitive and relevant baselines.** The paper compares against a static RLVR baseline, as well as recent long-context alignment methods such as LongPO, SoLoPO, and QwenLong-L1. This is important for ICLR-style evaluation because it situates the method against both direct RL baselines and long-context adaptation methods.

5. **Ablations and analyses are reasonably thorough.** The authors provide ablations on the grounding filter, history memory, verifier updates, majority voting, and reward shaping, plus analysis of question difficulty, entropy, response length, and group size. These help explain why the method works rather than only showing aggregate results.

6. **Practical contribution with open code.** The paper states that code is available and includes substantial implementation details, which is valuable for reproducibility and potential follow-up work.

### Weaknesses
1. **Novelty is incremental relative to existing self-play and verifier-based RL work.** The paper combines several known ideas—self-play, curriculum learning, self-consistency verification, and GRPO-style optimization—into a long-context setting. The main novelty is in integration and application rather than a fundamentally new learning principle, which may make the contribution feel less than a strong ICLR acceptance if judged strictly on algorithmic originality.

2. **The verifier is partly self-referential and may be brittle.** The verifier learns from majority vote and rule-based consistency, but the majority vote itself is generated by the same model family. This raises concerns about confirmation bias and error propagation, especially in open-ended long-context QA where “semantic equivalence” is hard to define. The paper acknowledges verifier self-delusion, but the mitigation feels empirical rather than principled.

3. **Evaluation is broad, but the claims may outpace the evidence.** The paper claims superiority over “equally sized models fine-tuned on large-scale annotated data” and hints at surpassing stronger systems like Gemini 2.5 Pro in pass@4. However, the comparison set is somewhat uneven: not all baselines are trained on identical data or under equivalent compute, and some claims rely on selective metrics or specific settings (e.g., pass@4 rather than primary pass@1).

4. **Compute and training cost are not fully contextualized.** The method uses 8×80GB A100s and a multi-stage on-policy RL loop with multiple rollouts per sample, which is significant. While the paper mentions total training time, it does not provide enough detail to judge cost-effectiveness versus simpler alternatives or against the baseline compute budget.

5. **The method depends on nontrivial engineered components.** The curriculum, history memory, grounding filter, prompt templates, and task-format selection all appear important. This makes the approach more complex and potentially harder to reproduce or transfer outside the curated setup. The contribution may therefore be less “clean” than the headline suggests.

6. **Generalization claims are somewhat under-supported.** The paper shows transfer to short-context reasoning and to additional long-context benchmarks, which is good. However, the training and evaluation corpora are both document-centric and heavily curated, so it is unclear how well SPELL generalizes to broader domains such as open-domain dialogue, code, or mixed-modal long-context reasoning.

7. **Some experimental design choices could bias the comparison.** The training data is constructed from a curated corpus with clustering, deduplication, and decontamination, and the questioner generates tasks from the same data distribution used for evaluation-style long-context QA. This may favor methods that exploit document structure and make the results less indicative of truly open-ended self-play.

### Novelty & Significance
From an ICLR perspective, SPELL is a solid and relevant systems/algorithmic contribution rather than a breakthrough in learning theory. Its novelty lies in adapting self-play RL to long-context reasoning with a three-role single-model design, plus a curriculum and verifier mechanism that make the setup workable in practice. The empirical significance is fairly strong because the method improves many models and benchmarks, but the conceptual advance is moderate and somewhat dependent on engineering choices.

Clarity is generally good: the paper explains the roles, reward design, curriculum, and training loop clearly, and the ablations help. Reproducibility is above average because the paper provides many hyperparameters, benchmark details, and code availability, though the system is still complex enough that exact replication may be difficult. Overall, this looks like a worthwhile ICLR submission with meaningful practical impact, but likely below the bar for a top-tier acceptance if reviewers prioritize principled novelty over strong empirical gains and careful engineering.

### Suggestions for Improvement
1. **Strengthen the novelty argument against closest prior work.** The paper should more explicitly distinguish SPELL from prior self-play, self-rewarding, and verifier-training methods, especially in terms of what is genuinely new beyond combining known components.

2. **Provide a cleaner compute- and data-matched comparison.** Report training FLOPs, wall-clock, and number of generated tokens for SPELL and each baseline, and if possible compare at equal compute rather than only equal model size.

3. **Add stronger evidence for verifier quality.** Evaluate verifier accuracy against human judgment on a manually annotated subset, and analyze failure cases where the verifier disagrees with the rule-based judge but the model is actually correct or incorrect.

4. **Test robustness to prompt and curriculum choices.** Since the framework depends heavily on prompt templates and curriculum settings, include sensitivity analyses for these components, not just for group size and Gaussian width.

5. **Broaden the evaluation beyond document QA.** To support claims of general self-evolution, test on at least one additional non-document long-context task or a more diverse reasoning setting.

6. **Clarify the exact source of gains.** Add controlled experiments isolating the contributions of self-play, curriculum, history memory, verifier training, and dynamic sampling, ideally with cumulative ablations to show which components matter most.

7. **Discuss limitations more candidly.** In particular, acknowledge that the method still requires curated document corpora, task templates, and reward engineering, which limits the “label-free” claim in practice.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a stronger comparison to long-context-specific training methods under matched compute/data, not just reimplemented LongPO/SoLoPO and one RLVR baseline. ICLR will expect evidence that SPELL beats the most relevant alternatives when trained on the same corpus, with identical rollout budgets and context lengths, otherwise the claim of a superior long-context RL framework is not convincing.

2. Add an ablation that isolates the three central mechanisms jointly: questioner curriculum, verifier self-consistency, and role-specific dynamic sampling. Right now the ablations are fragmented; without a clean factorial study, it is unclear which gains come from the self-play design versus from generic RL stabilizers or increased training data.

3. Add training-cost-normalized results and wall-clock/compute-matched baselines. Since SPELL relies on multi-role rollouts and multiple verification samples, the paper needs a direct “performance per FLOP / per GPU-hour” comparison to show the gains are not just bought with substantially more compute.

4. Add tests on truly out-of-distribution long contexts beyond 100K and beyond the training corpus structure. The paper claims generalization to longer contexts, but evaluating only with middle truncation at 100K does not establish robustness when the model must actually process ultra-long inputs.

5. Add a baseline that uses an external judge/reward model with equal access to the same synthetic data and rollouts. The paper argues the internal verifier is essential; without a matched stronger-judge baseline trained under the same budget, that claim remains under-supported.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze verifier accuracy/calibration against human or strong external judgments on a labeled subset. The core claim is that the learned verifier produces reliable semantic rewards; without measuring its false positives/negatives and calibration, the self-play loop could be reinforcing its own mistakes.

2. Analyze failure modes of the questioner curriculum over time, not just average difficulty. ICLR reviewers will want to know whether SPELL truly generates harder but solvable questions, or whether it drifts toward artifacts, overly narrow patterns, or accidental reward hacking.

3. Analyze whether improvements come from better grounding or just from longer generations and larger outputs. The method changes response length and sampling dynamics; without controlling for these factors, the claimed reasoning gains may be confounded.

4. Analyze performance by benchmark type, context length, and answer style, especially where semantic matching matters. The paper’s claims depend on long-document reasoning, but the results are aggregated; you need breakdowns showing where the verifier helps and where CEM or semantic judging still fails.

5. Analyze stability across random seeds and training runs with variance/error bars. For a self-play RL method with multiple moving parts, single-number averages are not enough to trust the reported gains.

### Visualizations & Case Studies
1. Show example trajectories of one document cluster across several SPELL iterations, including the generated questions, responder answers, verifier judgments, and reward values. This would reveal whether the curriculum actually becomes more challenging and whether the verifier tracks semantic correctness.

2. Visualize verifier errors on borderline cases: semantically correct paraphrases, partially correct answers, and hallucinated answers with superficial overlap. This is needed to show the verifier is genuinely improving reward quality rather than simply following string overlap or template artifacts.

3. Show examples where SPELL fails or regresses on long-context tasks, especially with distractor-heavy documents. A few qualitative failures would expose whether the method is robust or just benefits from easier cases that happen to fit its synthetic distribution.

4. Plot reward, question difficulty, response length, and verifier disagreement on the same timeline for multiple seeds. This would reveal whether the system is converging to a healthy fixed point or exploiting shortcuts that only look stable in aggregate.

### Obvious Next Steps
1. Extend the method to a larger set of real long-context tasks with different supervision regimes, including retrieval-heavy and summarization-heavy settings. The current paper focuses on QA-style benchmarks, so the claim of a general long-context reasoning framework is still too narrow.

2. Replace or augment the synthetic document-cluster pipeline with more realistic corpora and task generation procedures. The current setup depends on curated clusters and prompt templates, so the next step is to show the framework works when task construction is less controlled.

3. Demonstrate scalability under a fixed compute budget at larger context windows, such as 32K/128K, using more efficient memory or chunking strategies. This is an obvious next step because the paper itself admits the current framework is limited by 16K training context.

4. Test whether the learned verifier can be reused as a standalone reward model for other long-context RL settings. That would clarify whether SPELL’s verifier is a useful reusable component or only works inside this specific loop.

5. Evaluate whether SPELL can be bootstrapped with weaker or partially noisy reference answers. If the method only works when the questioner is given clean answers extracted from documents, the self-play claim is materially weaker than it sounds.

# Final Consolidated Review
## Summary
This paper proposes SPELL, a self-play RL framework for long-context language modeling in which a single model alternates among questioner, responder, and verifier roles. The method is designed to generate its own training questions from clustered documents, solve them with full context, and use a learned verifier plus rule-based checks to provide rewards, with an automated curriculum to increase difficulty over time. The paper reports broad gains across multiple open-source models and long-context benchmarks, including some stronger scaling behavior at test time.

## Strengths
- The paper targets a real and important gap: RL methods that work for short-context reasoning do not transfer cleanly to long-document settings, where semantic verification is much harder.
- The three-role self-play formulation is a concrete and reasonably well-motivated extension of prior self-play/RLVR ideas, and the paper provides a complete training loop with curriculum, verifier self-consistency, and role-specific sampling.
- Empirically, the results are broad: SPELL improves many model families and sizes, including base, instruct, reasoning, dense, and MoE models, and the improvements are not isolated to one benchmark.
- The ablation and analysis sections are useful and generally support the design choices, especially the importance of grounding filters, history memory, verifier updates, and the Gaussian questioner reward.

## Weaknesses
- The novelty is more integration than invention. SPELL combines self-play, curriculum learning, self-consistency, and verifier training into a long-context pipeline, but the paper does not convincingly show a fundamentally new learning principle; it feels like a well-engineered recombination of known components.
- Several headline claims are stronger than the evidence. In particular, “label-free optimization” is technically true only in a narrow sense, since the method still relies on curated corpora, prompt templates, document clustering, decontamination, and reference-answer construction; likewise, claims about outperforming annotated-data fine-tuning are not shown in a controlled compute/data-matched setting.
- The verifier remains a major weak point. It is trained from self-consistency and rule-based consistency, so it is still self-referential and vulnerable to confirmation bias or drift; the paper acknowledges this, but the mitigation is empirical rather than principled.
- The experimental comparison is not fully convincing on cost and fairness. SPELL uses substantial compute and multiple rollouts per example, yet there is no strong performance-per-FLOP or wall-clock comparison against equally budgeted alternatives, making it hard to tell how much of the gain comes from the method versus from heavier training.
- The evaluation relies on aggregated scores and a max-over-metrics protocol for free-form QA, but the paper does not sufficiently analyze disagreement cases or provide variance/error bars, which is a concern for modest 1–3 point differences.

## Nice-to-Haves
- A cleaner factorial ablation separating the contributions of curriculum, verifier self-consistency, history memory, and dynamic sampling would make the mechanism much easier to trust.
- Reporting training compute, wall-clock, token counts, and variance across seeds would substantially improve credibility.
- A small manually judged subset for verifier accuracy/calibration would help validate the central reward mechanism.

## Novel Insights
The most interesting part of SPELL is not the self-play loop itself, but the way the paper tries to make self-play viable for long-context tasks where exact string matching is too brittle. The verifier acts as a semantic bridge between noisy long-form answers and RL reward, while the questioner’s Gaussian reward is explicitly shaped to keep difficulty near the responder’s competence frontier. This makes the method more of a closed-loop curriculum system than a simple self-generated QA setup, and that is the paper’s main technical insight.

## Potentially Missed Related Work
- **Self-Questioning Language Models** — relevant because it also uses a model to propose and answer questions with self-generated verification signals.
- **Absolute Zero Reasoner (AZR)** — relevant because it studies self-play with difficulty shaping and reward mapping, which is close to the questioner reward design here.
- **R-Zero** — relevant because it also centers on self-evolving reasoning with a challenge-solver dynamic and competence-aware task generation.
- **Mutual-Taught** — relevant because it is another self-evolving framework where policy and reward signals co-adapt, which is conceptually adjacent to SPELL’s verifier calibration.
- **LongPO / SoLoPO / QwenLong-L1** — relevant because they are the closest long-context alignment baselines and help situate SPELL among long-context-specific post-training methods.

## Suggestions
- Add a compute-matched comparison table with FLOPs, wall-clock time, tokens generated, and performance-per-budget against the strongest relevant baselines.
- Include a small human-annotated verifier evaluation to quantify semantic accuracy, false positives, and false negatives.
- Provide a more explicit derivation of the on-policy GRPO simplification and a clearer description of how reference answers are constructed from documents.
- Add a controlled ablation that isolates whether the gains come mainly from role separation, curriculum, verifier learning, or just more sampling.
- Report standard deviations or confidence intervals for the main results and scaling curves.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 4.0, 8.0]
Average score: 6.0
Binary outcome: Accept
