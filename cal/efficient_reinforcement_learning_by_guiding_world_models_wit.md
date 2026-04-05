=== CALIBRATION EXAMPLE 81 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is accurate: the paper is about using non-curated offline data to improve RL via world models and online fine-tuning.
- The abstract clearly states the problem, the core finding that naive world-model fine-tuning fails, and the proposed fixes: experience rehearsal and execution guidance.
- The main quantitative claims are plausible and aligned with the paper’s experiments, but the abstract is a bit stronger than the evidence in a few places. In particular, “nearly twice the aggregate score of learning-from-scratch baselines across 72 visuomotor tasks” and “outperforms prior methods that utilize offline data by a decent margin” are broad claims that depend on the exact benchmark aggregation, baseline alignment, and dataset filtering choices. ICLR reviewers will expect those claims to be very carefully grounded in the experimental protocol.

### Introduction & Motivation
- The motivation is strong and timely: the paper targets a real gap between curated reward-labeled offline RL datasets and the much larger supply of reward-free, mixed-quality, multi-embodiment data.
- The gap in prior work is identified reasonably well: representation learning alone, or world-model pretraining without careful fine-tuning, does not fully exploit such data.
- The contributions are clearly stated in C1–C4, and the paper is honest that the setting is more realistic than reward-labeled task-specific offline RL.
- That said, the introduction slightly over-claims novelty in framing. The idea of leveraging offline data via world models and then reusing it during fine-tuning is conceptually adjacent to existing pretrain-then-finetune RL and replay-based offline-to-online methods. The novelty appears to be mainly in the combination of non-curated data, retrieval-based rehearsal, and execution guidance, rather than a fundamentally new learning paradigm. That is still a meaningful contribution, but the introduction should be careful not to imply a larger conceptual leap than the paper actually provides.

### Method / Approach
- The method is mostly clear, and the overall pipeline is understandable: pretrain an RSSM world model on non-curated data, retrieve task-relevant trajectories, behavior-clone a prior policy from them, and use both rehearsal and mixed-policy execution during online fine-tuning.
- The formulation in Sec. 3.2 and 3.3 is sufficiently detailed at a high level, but reproducibility is weaker in the retrieval and guidance mechanisms than in the world-model pretraining.
- Key assumptions are not fully justified:
  - Experience rehearsal assumes the initial online observation is enough to retrieve task-relevant trajectories from a very large, heterogeneous offline dataset using encoder distance in Eq. (5). That is a strong assumption, especially for tasks with multimodal state spaces or low-level visual ambiguity.
  - Execution guidance assumes a behavior-cloned prior from retrieved trajectories is sufficiently useful early in training, but the paper does not characterize when this fails.
- There are logical gaps around the theoretical framing:
  - Eq. (5) uses Euclidean distance in learned feature space, but the paper does not justify why this metric is reliable under multi-embodiment shift or how sensitive the retrieval is to representation collapse.
  - The fine-tuning objective and the role of the replayed offline trajectories are described intuitively, but the connection between “reducing distribution shift” and actual policy improvement is mostly empirical, not theoretically established.
- Edge cases and failure modes are not discussed enough:
  - What happens when the initial online observation is not representative of the downstream task?
  - What if retrieved trajectories are only loosely related, or the task has sparse observations and many aliased states?
  - What if the behavior-cloned prior is worse than the RL policy after early exploration?
- The theoretical claims in Appendix B are not rigorous enough for ICLR standards:
  - Proposition 1’s proof is informal and relies on assumptions that are not proved (e.g., selecting states “such that” distance is below an epsilon threshold).
  - Proposition 2 largely restates a known result from Kakade & Langford (2002) and does not provide a novel theorem tailored to execution guidance. It is more of an analogy than a proof of the specific algorithmic choice.

### Experiments & Results
- The experiments do test the paper’s claims broadly: sample efficiency, comparison against offline-data baselines, comparison against model-based baselines, and task adaptation.
- The benchmark choice is strong: 72 visuomotor tasks across DMControl and Meta-World is substantial and relevant to ICLR.
- Baselines are mostly appropriate, but there are important fairness issues that affect interpretability:
  - For baselines that cannot handle multi-embodiment data, the paper preprocesses the offline data to include only task-relevant trajectories, while NCRL uses the full non-curated dataset. That is defensible as a “best effort” comparison, but it means the baselines are not evaluated under the same data regime as NCRL. The paper should more explicitly separate “algorithmic advantage” from “access to broader data.”
  - Some baseline implementations are modified substantially, e.g. ExPLORe is altered with ensembles and latent-space reward sharing. That may be necessary for stability, but it makes it harder to interpret comparisons as against the original method.
  - For iVideoGPT, the paper compares against original reported results in Fig. 4 and also an aligned variant in the appendix, which is good. However, the central claim in the main text leans on the more favorable original setup while the aligned comparison is relegated to the appendix.
- A major missing piece is a more systematic ablation on the two central components:
  - Experience rehearsal and execution guidance are ablated, which is good.
  - But the paper does not sufficiently isolate whether gains come from retrieval, from behavior cloning, or simply from adding more offline data during fine-tuning.
  - There is no ablation on retrieval quality beyond the robustness experiment in Fig. 14, and no comparison of retrieval metrics against downstream performance.
  - There is no meaningful sensitivity study for the number of retrieved trajectories, the retrieval threshold, or the offline/online mix ratio beyond the fixed default.
- Error bars are partially reported via 95% confidence intervals in some curves, which is good. However, the tables are mostly summarized without significance tests or variance across seeds beyond reporting means, and some headline claims depend on aggregate scores that may hide task-level failures.
- The results do support the general claim that NCRL is strong on these benchmarks, but some of the strongest claims remain somewhat selective:
  - The paper highlights “almost double the aggregate score,” yet the DMControl table shows especially dramatic gains on some tasks and weaker or even negative results on others, which suggests the aggregate can conceal uneven performance.
  - The manuscript presents NCRL as broadly superior to prior offline-data methods, but the strongest evidence is on task families closely aligned with the retrieval setup and with the offline corpus. More evidence would be needed to claim broad superiority across diverse real-world conditions.
- Dataset and evaluation metrics are reasonable for the domain. Still, because the offline datasets are assembled from specific sources with injected corruption, the conclusions are somewhat benchmark- and dataset-construction-dependent. ICLR reviewers will likely want to know how sensitive the method is to the choice of offline corpus.

### Writing & Clarity
- The paper is generally understandable, and the high-level story is coherent.
- The main clarity issue is that the method description sometimes mixes conceptual explanation with implementation details in a way that makes it hard to identify what is essential versus incidental. For example, the distinction between the world-model rehearsal effect and the prior-policy guidance effect is conceptually important, but the exact operationalization is scattered across Sec. 3.3, Appendix B, and Algorithm 1.
- Figures 1, 2, and 6 do communicate the intended ideas well, especially the distribution-shift story and the component ablation.
- Some tables and appendix results are dense, but that is not a problem in itself. The main issue is that the paper’s strongest claims rely on large collections of heterogeneous results, and the narrative does not always make clear which findings are central versus supportive.
- The theoretical appendix is harder to follow than the main method, not because of notation alone, but because the arguments are not sharply connected to the actual algorithmic details.

### Limitations & Broader Impact
- The authors do acknowledge several important limitations in Appendix D:
  - the use of RSSM rather than more scalable architectures,
  - limited generalization analysis,
  - in-domain rather than truly in-the-wild data,
  - simulation-only evaluation.
- That is a good start, but the limitations are still understated relative to the paper’s ambitions.
- The biggest missed limitation is that the method depends on having a large, task-relevant offline corpus already available in the target domain family. The paper’s claims about “non-curated” data are compelling, but the data are still curated at the benchmark/domain level and not truly open-world.
- Another important limitation is retrieval dependence: the method’s success may hinge on the availability of a good initial observation and a semantically meaningful feature space. This could be brittle in partially observable, highly dynamic, or low-visual-signal tasks.
- The ethical statement is generic and does not engage with concrete risks relevant to robot learning, such as misuse in autonomous systems, dataset provenance, or deployment safety. For ICLR, this is acceptable but not strong.
- Broader societal impact discussion is limited, but not egregiously so for this kind of systems paper.

### Overall Assessment
This is a strong and relevant ICLR submission with a clear practical contribution: it shows that non-curated, reward-free, mixed-quality, multi-embodiment offline data can be made useful for offline-to-online RL when combined with retrieval-based rehearsal and execution guidance. The empirical scope is impressive, and the results are often compelling. My main concerns are about how far the claims can be generalized beyond the benchmark setups, the fairness/interpretability of some baseline comparisons, and the lack of rigorous justification for the retrieval and guidance mechanisms beyond empirical success. The paper likely stands as a meaningful advance for sample-efficient visuomotor RL, but the authors should temper the breadth of the claims and strengthen the causal evidence that the proposed mechanisms, rather than the particular dataset construction or evaluation choices, drive the gains.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes NCRL, a two-stage offline-to-online reinforcement learning approach that exploits non-curated offline data: reward-free, mixed-quality, and multi-embodiment trajectories. The key idea is to pretrain a large task-agnostic world model on this data, then fine-tune it online using two mechanisms—experience rehearsal to mitigate distribution shift and execution guidance to bias early exploration using a behavior-cloned prior. The paper reports strong empirical gains over training-from-scratch and several offline-data baselines across 72 pixel-based visuomotor tasks, plus a continual-adaptation setting.

### Strengths
1. **Addresses an important and timely problem with realistic data assumptions.**  
   The paper explicitly targets reward-free, mixed-quality, multi-embodiment offline data, which is more realistic than assuming curated, reward-labeled task datasets. This is a meaningful direction for ICLR because it broadens the practical applicability of offline-to-online RL.

2. **Clear empirical scope and breadth.**  
   The evaluation is extensive: 72 visuomotor tasks spanning DMControl and Meta-World, with locomotion and manipulation, plus a continual task-adaptation experiment. This breadth is stronger than many world-model papers that only evaluate on a few tasks.

3. **Identifies a plausible failure mode and proposes targeted fixes.**  
   The paper does not just claim that world-model pretraining helps; it argues that naive fine-tuning fails due to distribution shift and then introduces experience rehearsal and execution guidance to address it. This is a concrete algorithmic narrative with a reasonable causal story supported by ablations.

4. **Ablation studies support the role of individual components.**  
   The paper includes ablations showing the contribution of pretraining, experience rehearsal, and execution guidance, as well as sensitivity analyses for the guidance schedule and retrieval robustness. This improves the credibility of the overall method.

5. **Reproducibility is reasonably emphasized.**  
   The paper provides code, compute details, algorithmic pseudocode, and hyperparameters. For an ICLR submission, that level of engineering transparency is a positive.

### Weaknesses
1. **The methodological novelty is moderate rather than clearly breakthrough-level.**  
   Experience rehearsal resembles replay/mixture strategies used in offline-to-online RL, and execution guidance is conceptually close to jump-start-style policy mixing or behavior-cloned priors. The combination with a world model and non-curated data is useful, but the paper does not convincingly establish a fundamentally new learning principle beyond a strong integration of known ideas.

2. **The theoretical analysis is weak and mostly informal.**  
   The proof sketches in the appendix appear to rely on simplifying assumptions and largely restate intuitive claims rather than provide rigorous guarantees. For example, the distribution-shift and performance-improvement arguments are not deeply formalized, and the results do not seem to yield actionable bounds that explain when NCRL should or should not work.

3. **Potential concern about fairness and comparability of baselines.**  
   The paper compares against several baselines with substantial implementation modifications and preprocessing choices. While some adaptation is necessary, the evidence suggests that baseline tuning is uneven and may advantage NCRL. ICLR reviewers typically scrutinize whether comparisons are equally well-engineered, especially for strong claims like “almost double aggregate score.”

4. **The use of a very large world model raises questions about efficiency and accessibility.**  
   The pretraining model is scaled to 280M parameters, and training costs are nontrivial. The method is presented as sample-efficient, but computational efficiency and memory footprint are not as clearly justified. For ICLR, “efficient” claims are typically expected to consider both sample and compute efficiency.

5. **The data setting is still somewhat bounded despite the “non-curated” framing.**  
   The offline data is non-curated, but it is still in-domain simulator data from two benchmarks. The paper acknowledges that it does not yet address truly in-the-wild data or real-world deployment. That limits the generality of the claim, especially given the strong framing around broad usability.

6. **Limited clarity on retrieval and guidance design choices.**  
   Retrieval is based on initial observation similarity in latent space, but the paper does not fully justify why this simple criterion should be robust across diverse tasks and embodiments. Similarly, execution guidance relies on a behavior-cloned prior from retrieved trajectories, but the exact failure modes of this mechanism are only partially analyzed.

### Novelty & Significance
**Novelty:** Moderate. The paper combines several known ideas—world-model pretraining, replay/rehearsal, behavior cloning priors, and offline-to-online RL—into a coherent pipeline for non-curated data. The specific setting of reward-free mixed-quality multi-embodiment data is appealing, but the algorithmic contributions are more incremental than radically new.

**Significance:** Moderate to high. If the empirical results hold up, the paper addresses a practically important bottleneck: how to reuse abundant but messy robot data for efficient online RL. The breadth of tasks and the improvement over strong baselines suggest meaningful practical value. However, for ICLR’s acceptance bar, the work would benefit from stronger evidence of generality, better computational analysis, and more rigorous methodological justification.

**Clarity:** Generally good at the high level. The motivation, failure mode, and two proposed fixes are easy to follow. However, some details of the method and theoretical arguments are harder to parse, and the paper would benefit from sharper explanations of what is new relative to prior offline-to-online and world-model methods.

**Reproducibility:** Fair to good. The authors provide code, hyperparameters, and algorithm pseudocode, which is positive. That said, the dependence on large-scale compute and the need for careful baseline tuning may make exact reproduction challenging.

### Suggestions for Improvement
1. **Strengthen the novelty claim by isolating what is fundamentally new.**  
   Add a clearer conceptual comparison against RLPD, JSRL, ExPLORe, MOTO, and prior world-model pretraining methods, explicitly separating “new setting,” “new mechanism,” and “new empirical finding.”

2. **Provide more rigorous and informative theory.**  
   Replace the current informal proof sketches with tighter statements, or clearly label them as intuition. If possible, derive conditions under which experience rehearsal and execution guidance provably help or fail.

3. **Improve baseline fairness and reporting.**  
   Include a stronger discussion of baseline tuning protocols, matched compute budgets, and any modifications made to baselines. A compute-normalized comparison would be especially valuable for ICLR readers.

4. **Report compute and data efficiency more comprehensively.**  
   Since the method relies on a 280M-parameter world model, include memory usage, training wall-clock, and performance-per-compute analyses, not only sample-efficiency curves.

5. **Ablate the retrieval and guidance mechanisms more deeply.**  
   Study alternative retrieval criteria, retrieval sizes, and guidance schedules. It would also help to compare against simpler variants such as uniform replay of offline data, fixed BC warm starts, or soft mixing of policies without episodic switching.

6. **Clarify generalization limits and extend evaluation beyond simulator settings.**  
   The paper already notes this limitation; strengthening it with cross-task or cross-embodiment generalization tests would make the contribution more compelling. Even a small-scale real-robot or more out-of-domain test would substantially increase impact.

7. **Tighten the presentation of the main empirical claims.**  
   The paper would benefit from concise tables summarizing average performance, sample-efficiency gains, and confidence intervals across the major benchmark groups, making the main message easier to assess quickly.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a true no-offline-data fine-tuning ablation for the full NCRL pipeline on all main tasks, not just a few curves. The paper’s core claim is that non-curated offline data improves sample efficiency; without showing the exact gain over identical DreamerV3/DrQ-v2 training when offline data is completely removed, the contribution is hard to isolate.

2. Compare against stronger offline-to-online baselines that are closer to the proposed setting, especially DreamerV3/DrQ-v2 with offline replay mixing, representation pretraining + online RL, and a world-model baseline that also reuses offline data during fine-tuning. ICLR reviewers will expect the method to beat the most relevant “use the same data, same backbone” alternatives, not just methods that are structurally mismatched or require reward labels.

3. Add a controlled comparison where the offline dataset quality is systematically varied from expert-heavy to noisy/non-expert across the same tasks. The main claim is that NCRL handles mixed-quality, reward-free data; right now the evidence is dataset-specific and could be confounded by one particular data collection recipe.

4. Include scalability experiments with much smaller and much larger offline datasets, plus retrieval budget sensitivity. The method depends on large-scale retrieval and rehearsal, so the claim of practicality is not convincing without showing how performance changes as dataset size and retrieval top-k scale.

5. Evaluate on at least one real-robot or sim-to-real benchmark, or explicitly show why the method transfers beyond simulation. For ICLR, a method claiming broadly usable offline-to-online RL from non-curated data needs evidence that the retrieval/guidance scheme is not simulator-specific.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify how much of the gain comes from better world-model learning versus better policy optimization. Right now the paper attributes failure to distribution shift, but it does not cleanly separate whether rehearsal helps because it regularizes dynamics, improves reward prediction, improves initial-state coverage, or simply acts as extra behavior cloning.

2. Analyze retrieval quality more rigorously beyond precision@k on a few tasks. The paper needs recall, false-positive failure cases, and correlation between retrieval quality and downstream return; otherwise it is unclear whether the retrieval heuristic is robust or just lucky on some benchmarks.

3. Measure the actual distribution shift reduction during training using multiple metrics, not only an illustrative t-SNE/Wasserstein-style visualization. The core explanation hinges on distribution mismatch, so reviewers will want quantitative evidence that rehearsal keeps online rollouts closer to useful offline support and reduces forgetting.

4. Study when execution guidance helps versus hurts, especially when the behavior-cloned prior is weak or suboptimal. The paper assumes the BC prior is beneficial early on, but it does not analyze failure modes where the prior biases exploration into bad regions and slows discovery of optimal behavior.

5. Report compute, wall-clock, and storage overhead of pretraining, retrieval, and rehearsal relative to the baselines. Since the method adds a 280M world model and large-scale retrieval, ICLR reviewers will expect a clear efficiency trade-off, not only sample-efficiency curves.

### Visualizations & Case Studies
1. Show side-by-side rollouts for success and failure cases with and without rehearsal/guidance. This would reveal whether NCRL actually explores better or just overfits benchmark-specific reward structures.

2. Visualize retrieved trajectories and their task overlap for several easy, medium, and hard tasks. The paper claims retrieval extracts task-relevant data from noisy multi-embodiment logs; examples of good and bad retrievals are needed to judge whether the heuristic is meaningful.

3. Plot world-model prediction error over training separately on offline-like and online-like states. That would directly test the claim that naive fine-tuning fails because of distribution shift and catastrophic forgetting.

4. Show the evolution of the initial-state distribution induced by execution guidance versus pure RL. If guidance works as claimed, early exploration trajectories should move toward higher-value regions and broader support more quickly.

5. Add failure-case case studies on tasks where NCRL underperforms or remains low. The current paper mostly highlights wins; ICLR standards favor understanding the boundary conditions of a method, especially for a broad claim across 72 tasks.

### Obvious Next Steps
1. Combine NCRL with stronger retrieval and world-model scaling baselines, and show whether the method remains beneficial when the backbone is already state-of-the-art. The paper argues the technique is architecture-agnostic, but that is not yet demonstrated.

2. Extend the framework to in-the-wild datasets and mixed-embodiment real robotic data, not just simulator-collected “non-curated” data. This is the most direct next step if the paper wants to justify the broader claim of using abundant uncurated data.

3. Test whether the same rehearsal/guidance ideas improve other offline-to-online algorithms beyond Dreamer-style model-based RL. If the method is genuinely general, it should transfer to at least one model-free or different model-based pipeline.

4. Replace the heuristic retrieval rule with a learned or uncertainty-aware retrieval mechanism and compare. The current method’s performance seems to depend on a fragile nearest-neighbor filtering step, which is a natural bottleneck for future work.

5. Evaluate continual adaptation on a more standard continual-RL suite with stronger forgetting metrics. The current adaptation experiment is too narrow to support a general continual-learning contribution.

# Final Consolidated Review
## Summary
This paper proposes NCRL, a two-stage offline-to-online RL method that leverages reward-free, mixed-quality, multi-embodiment offline data. It pretrains a large RSSM-based world model on this non-curated data, then fine-tunes online with two key mechanisms: experience rehearsal, which retrieves task-relevant offline trajectories to reduce distribution shift, and execution guidance, which mixes in a behavior-cloned prior policy early in training. The paper reports broad empirical gains across DMControl and Meta-World tasks, but the core story is still more an integration of known ideas than a clearly fundamental advance.

## Strengths
- **Targets a practically important and underexplored setting.** The paper moves beyond reward-labeled, task-specific offline datasets and studies reward-free, mixed-quality, multi-embodiment data, which is a more realistic use case for robot learning.
- **Large empirical scope with a coherent causal story.** The evaluation spans 72 visuomotor tasks across DMControl and Meta-World, plus a continual-adaptation experiment, and the ablations do support the claim that pretraining alone is not enough while rehearsal and execution guidance each help.

## Weaknesses
- **The main methodological idea is only a moderate recombination of existing components.** Experience rehearsal resembles replay-based offline-to-online methods, and execution guidance is close to behavior-cloned policy mixing / jump-start-style priors. The novelty is mostly in applying these ideas to non-curated data with a world model, not in a clearly new learning principle.
- **Theoretical support is weak and largely non-rigorous.** The appendix proofs are informal, rely on simplifying assumptions, and do not really establish when NCRL should work. The execution-guidance argument mostly repackages an old result rather than proving something specific to this method.
- **Some of the strongest empirical claims are harder to interpret than the paper suggests.** Baselines are not always compared under the same data regime: for methods that cannot handle multi-embodiment data, the offline corpus is preprocessed to include only task-relevant trajectories, while NCRL uses the full non-curated dataset. That makes the headline gains less cleanly attributable to the algorithm alone.
- **The retrieval and guidance mechanisms are not stress-tested enough.** Retrieval is based on nearest-neighbor matching from the initial observation in learned feature space, but the paper does not deeply analyze failure cases, sensitivity to retrieval size/quality, or what happens when the BC prior is weak. This is a real weakness because both components are central to the method.

## Nice-to-Haves
- A more explicit compute-normalized analysis would help, since the method uses a 280M-parameter world model and fairly heavy pretraining.
- A deeper study of how performance changes with retrieval budget and offline dataset scale would make the approach easier to assess.

## Novel Insights
The most interesting insight here is not that world models can be pretrained on messy offline data, but that naive fine-tuning of such models can still fail badly because the online distribution quickly collapses away from the support of the offline corpus. The paper’s two fixes are aimed precisely at this mismatch: rehearsal keeps the model anchored to relevant offline trajectories, while execution guidance biases early exploration toward regions where the learned model is more reliable. That diagnosis is plausible and borne out by the ablations, but the paper still leaves open how much of the improvement comes from true distribution-shift mitigation versus simply adding more useful data and a stronger initialization.

## Potentially Missed Related Work
- **RLPD / Ball et al. 2023** — relevant because it also studies replaying offline data during online RL, though under much more structured settings.
- **Jump-Start RL / Uchendu et al. 2023** — relevant because the execution-guidance idea is closely related to mixing a prior policy with the online policy.
- **MOTO / Rafailov et al. 2023** — relevant as another model-based offline-to-online approach, though it assumes reward-labeled data.
- **DreamerV3 fine-tuning / prior world-model pretraining work** — relevant because this paper’s core claim is about how to make world-model pretraining actually useful during online fine-tuning.

## Suggestions
- Strengthen the paper by adding a cleaner “same backbone, same data, same compute” comparison against the closest offline-to-online alternatives, especially variants that also reuse offline data during fine-tuning.
- Add a focused retrieval study: vary top-k, measure retrieval precision/recall more broadly, and correlate retrieval quality with downstream return.
- Include a clearer breakdown of where the gains come from: world-model pretraining, offline rehearsal, BC guidance, or their interaction.
- If space permits, replace the current informal theory with a smaller number of precise claims tied directly to the implemented algorithm.

# Actual Human Scores
Individual reviewer scores: [6.0, 10.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
