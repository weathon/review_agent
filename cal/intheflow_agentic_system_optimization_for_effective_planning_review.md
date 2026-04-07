=== CALIBRATION EXAMPLE 78 ===

# Harsh Critic Review
Now I have sufficient material to write a thorough review. Let me produce it.

---

## Section-by-Section Critical Review

### Title & Abstract

The title "In-the-Flow Agentic System Optimization for Effective Planning and Tool Use" is reasonably descriptive, though "in-the-flow" is an informal phrase that does not directly telegraph the technical content (on-policy RL inside a multi-module agent loop). The abstract is clear and specific in its performance claims (14.9%, 14.0%, 14.5%, 4.1% gains across four domains). However, the claim that a 7B system "even surpass[es] larger proprietary models like GPT-4o" needs important caveats: AGENTFLOW uses four Qwen-2.5-7B-Instruct instances simultaneously (planner + executor + verifier + generator), plus live internet search via Google and Wikipedia APIs, none of which GPT-4o baseline numbers are given with real-time search. This context is absent from the abstract and materially affects how readers will interpret the comparison.

---

### Introduction & Motivation

The motivation is well-constructed. The paper correctly identifies two limitations in prior work: monolithic TIR models scale poorly, and agentic systems are mostly static. The three contributions stated at the end of §1 are clearly delineated.

That said, the framing of Flow-GRPO as solving "long-horizon credit assignment" (§1, §3.2) is overstated. Broadcasting a single final-outcome reward to every turn does not solve credit assignment—it sidesteps it entirely by assigning identical credit to all actions, regardless of their contribution. This is a meaningful conceptual distinction the paper elides. Other work (StepSearch, RAGEN) explicitly tries to assign differential credit; the paper should honestly position its approach as "reward broadcasting" rather than "credit assignment."

---

### Method / Approach (§3)

**Architecture (§3.1).** The four-module design (planner, executor, verifier, generator) is cleanly formalized as an MDP. The evolving memory design—a structured, deterministic log rather than a raw KV-cache—is a sensible and reproducible choice. However, only the planner is trained; the executor, verifier, and generator are frozen prompted instances of the same model. The paper does not sufficiently motivate why only the planner is trainable or what would happen if other modules were fine-tuned. This is a notable design choice that warrants ablation.

**Flow-GRPO (§3.2).** The algorithm reduces to GRPO with a constant reward broadcasted across all turns. The implementation is straightforward and likely effective, but the claim of algorithmic novelty is weaker than presented. A constant per-trajectory reward applied token-level within a clipped PPO objective is precisely what many existing multi-turn RL systems already do. The paper should more precisely articulate what is new beyond the system-level framing.

**Theoretical analysis (Appendix B).** The "equivalence proof" (Theorem B.1/§B.2) is mathematically correct but trivially follows from the fact that the per-turn reward is constant; factoring a constant across terms and invoking linearity of expectation is not a non-trivial theorem. The convergence result (Theorem B.3) simply reapplies the standard TRPO/PPO monotonic improvement guarantee without any modification for the multi-turn setting. Neither result provides genuine insight into why Flow-GRPO specifically works for this agentic context. The appendix is misleadingly labelled "Theoretical Analysis" when it is standard bookkeeping.

**Training vs. inference mismatch.** A critical underdiscussed issue: training is conducted with T_max = 3 turns (§C.1), but evaluation uses T_max = 10 turns (§C.2). This is a significant distribution shift. The planner was never trained to plan under a 10-turn budget, yet all reported numbers use the longer horizon. The paper does not discuss whether training on 3-turn trajectories and evaluating on 10-turn ones causes degradation or whether the gains would be larger under matched conditions. Figure 7 shows that more turns help at inference, but none of those experiments report performance for the model *trained* at longer horizons.

**Reward signal dependency on GPT-4o.** The LLM-as-judge reward function (§E.3) uses GPT-4o during both training (§C.1) and evaluation. This (1) introduces significant API cost not borne by most baselines, and (2) means GPT-4o is not a truly independent baseline—it actively shapes what AGENTFLOW learns to optimize. The paper should discuss whether this gives AGENTFLOW a systematic advantage on tasks where GPT-4o's judgment aligns with its own style.

---

### Experiments & Results (§4)

**Compute parity.** The central fairness concern: AGENTFLOW instantiates four separate Qwen-2.5-7B-Instruct models (planner, executor, verifier, generator) plus tool engines. This implies roughly 4× the active parameter footprint compared to a single-model baseline. Every comparison labeled "7B" is therefore misleading—AGENTFLOW's effective inference compute is closer to 28B-equivalent. The paper acknowledges that "AutoGen and our AGENTFLOW method are agentic systems, which use Qwen-2.5-7B-Instruct for the LLM-powered agents and tools for fair comparison" but this statement itself reveals the issue: AutoGen uses the same backbone in the same multi-agent fashion. The question is why the gains over AutoGen (+14.9%, +19.9%) are so large—this gap should be the central ablation, not just noted as a headline result.

**Table 3 appears incomplete.** The ablation table on planner training strategies (§4.3, Table 3) shows only a single row in the parsed version (frozen baseline). The narrative references GPT-4o planner and SFT comparisons, but whether these appear in a full version of Table 3 or separately cannot be confirmed from the paper text. If Table 3 is truly one-row, this is a critical ablation gap.

**Baselines.** Only one agentic system baseline (AutoGen) is included. MetaGPT, OctoTools (which shares an author), and more recent trainable agentic systems (MAPoRL, MALT, OWL) are mentioned in related work but absent from evaluation. GAIA performance (33.1%) in particular warrants comparison against GAIA-specific state-of-the-art systems.

**Tool access asymmetry.** Several baselines (ZeroSearch, StepSearch, SimpleRL) simulate search rather than using real internet, while AGENTFLOW uses live Google/Wikipedia APIs. This is acknowledged nowhere in the experimental comparison and is a potentially large confounder on search-intensive benchmarks.

**Sample sizes.** Several benchmarks are sampled to 100 examples (2Wiki, HotpotQA, Musique, GPQA). For AIME24 (N=30) the numbers are small enough that a 1-2 problem difference corresponds to 3.3-6.7% accuracy change, making small improvements difficult to interpret statistically. Though the paper reports averages over 3 trials, confidence intervals and variance estimates would clarify whether gains are robust.

**SFT collapse.** The reported −19% average performance collapse under offline SFT (§4.3) is striking. SFT on GPT-4o trajectories collapsing this badly warrants deeper investigation. Was the data distribution checked? Did SFT succeed in imitating GPT-4o actions, or did it fail at the format/instruction-following level? Without this analysis the SFT ablation is anecdotal.

**Scientific tasks.** Training data includes only search (Search-R1) and math (DeepMath) data, yet the system is evaluated on GPQA and MedQA (both scientific). The gains here are attributed to generalization, but this could simply reflect the strong priors of the Qwen-2.5-7B-Instruct base. Ablation against the untrained AGENTFLOW (without Flow-GRPO) shows the system already scores 56.5% on science before training (Table 2), confirming most of the gain is architecture-driven, not training-driven.

**Training efficiency comparison (§4.6).** The comparison of Flow-GRPO vs. ToRL training curves (Figure 9b) shows ToRL "quickly stagnates and trends downwards." However, the two methods use different model architectures and system designs, making the comparison uncontrolled. This should not be cited as evidence of superior training efficiency unless the comparison is made within the same agentic framework.

---

### Scaling Analyses (§4.4)

The scaling results—both backbone size (3B→7B) and inference turns (3→10)—are genuine contributions and presented cleanly. The turn-scaling results in Table 4 show that actual turn usage still falls significantly below the allowed budget (T_max=10 → avg 5.42 turns for GAIA, 1.9 for AIME24), suggesting the verifier terminates trajectories conservatively. An analysis of verifier accuracy—does early termination ever cut off trajectories that would have succeeded?—would strengthen this section.

---

### In-Depth Analysis (§4.5)

The tool call ratio analysis (Figure 8) is one of the paper's most interesting contributions: the planner spontaneously shifts to task-appropriate tools after RL training. This is compelling evidence that the policy is learning something non-trivial. However, the mechanism is not analyzed—are these shifts driven by reward shaping or by implicit behavioral cloning from the frozen verifier? The case studies in Appendix F are illustrative but hand-selected; a more systematic analysis (e.g., success rate stratified by tool-choice pattern) would be more convincing.

---

### Related Work (§5)

The related work is competent and covers most relevant papers. However, the relationship to concurrent work on multi-turn RL (RAGEN, SPA-RL) deserves more precise technical differentiation, not just a citation. Specifically, RAGEN also trains agents on-policy in multi-turn loops; explaining what Flow-GRPO does differently would sharpen the contribution.

---

### Limitations & Broader Impact

The paper has no explicit limitations section. Key omissions:
1. **Cost**: The system requires 4× inference compute, live search APIs, and GPT-4o as judge—all expensive. This limits accessibility.
2. **Verifier reliability**: The verifier is a frozen LLM; if it makes errors (false positives in termination), the planner receives wrong signals. No analysis of verifier accuracy is provided.
3. **Search tool reliability**: Web search results are noisy and temporally unstable; AGENTFLOW's performance on knowledge-intensive tasks may degrade with stale search results.
4. **Transfer to new tools**: The claim that the framework "generalizes weakly to new scenarios" for monolithic models is asserted but never tested for AGENTFLOW itself—no experiment adds a new tool at inference time.
5. **Safety/misuse**: A system with live internet access and Python execution capability poses safety risks that are not discussed.

---

### Overall Assessment

AGENTFLOW presents a technically solid and empirically well-supported system for multi-module agentic reasoning with on-policy planner training. The paper's core insight—that a structured agentic pipeline with a trainable planner optimized via GRPO on final-outcome rewards outperforms both monolithic RL models and static agentic systems—is credible and supported by extensive experiments. The breadth of benchmarks is impressive.

However, the paper has several issues that require attention before acceptance. The fairness of the central comparisons is compromised by the multi-model inference cost not being disclosed, the use of live search APIs unavailable to some baselines, and the use of GPT-4o as a training reward signal. The claimed algorithmic novelty of Flow-GRPO over existing GRPO-based multi-turn approaches is overstated, and the "theoretical analysis" in Appendix B is standard bookkeeping dressed up as novel theorem proving. The critical train/inference horizon mismatch (T_max=3 vs. T_max=10) is unaddressed. The ablation of the training strategy (Table 3) appears incomplete, and the SFT collapse deserves proper diagnosis. The contribution is more an engineering contribution demonstrating how to integrate RL training inside a multi-module agentic system than a conceptual algorithmic advance—which is still a valuable contribution, but the paper should position itself accordingly. With proper disclosure of compute costs, honest framing of the algorithmic contribution, and a more controlled ablation study, this work could be a strong ICLR paper. In its current form, it sits just below the acceptance bar.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces **AGENTFLOW**, a trainable, **in-the-flow** agentic framework that coordinates four specialized modules (planner, executor, verifier, generator) via a shared evolving memory to solve complex reasoning tasks. To address sparse rewards in long-horizon planning, the authors propose **Flow-GRPO**, an on-policy algorithm that broadcasts verifiable trajectory-level rewards to every turn within the multi-turn loop and utilizes group-normalized advantages for stability. The system, backed by a 7B model, demonstrates significant performance gains across ten benchmarks, outperforming specialized baselines and larger proprietary models like GPT-4o.

### Strengths
1.  **Strong Empirical Performance with Efficient Backbones:** AGENTFLOW achieves state-of-the-art results on diverse benchmarks (search, agentic, math, science) using a 7B parameter backbone, significantly outperforming top-performing 200B-parameter proprietary models (e.g., GPT-4o). This demonstrates the efficacy of the proposed system architecture in unlocking capabilities beyond simple monolithic scaling.
2.  **Crucial Ablation on Training Strategy:** The ablation study in §4.3 provides compelling evidence for the necessity of **in-the-flow** learning. The finding that offline Supervised Fine-Tuning (SFT) leads to a "catastrophic performance collapse" (-19.0% average accuracy) while on-policy RL succeeds strongly validates the paper's central premise that decoupled training fails to capture live multi-turn dynamics.
3.  **Open Source and Reproducibility:** The authors provide extensive resources including code, model weights, and a visual dashboard/interactive demo. Detailed experimental setups, training hyperparameters, and tool metadata are available, facilitating reproducibility which is critical for RL in LLM research.

### Weaknesses
1.  **Incremental Algorithmic Novelty:** The proposed **Flow-GRPO** algorithm is described as a significant novelty, but mathematically it operates very similarly to standard GRPO/RPPO applied to multi-turn rollouts (assigning trajectory rewards to individual steps). While the broadcast mechanism is practically useful, the paper overstates the theoretical novelty (e.g., Section 3.2 claims to "convert multi-turn RL into single-turn updates," a property implied by standard RL objectives). The claim of a "formal proof" (Appendix B) is standard policy gradient analysis and does not introduce a new learning paradigm.
2.  **Inference Cost and Latency:** The system utilizes **four** modules instantiated as 7B models (Planner, Executor, Verifier, Generator). Compared to a monolithic agent, this implies roughly 4x the inference cost and memory overhead per turn. The paper does not provide a concrete cost-benefit analysis (e.g., cost-per-solved-task comparisons) against competitors like Search-R1 which use a single model interleaved with tool calls.
3.  **Evaluation Bias Concerns:** While using an LLM-as-judge (GPT-4o) is common, comparing a 7B system against GPT-4o using GPT-4o as the judge introduces potential model bias (the "judge" might favor GPT-4o's output style). Additionally, baseline comparisons for some specific tools (e.g., Google Search vs. internal tools) vary or require standardization that may favor the AGENTFLOW setup.

### Novelty & Significance
*   **Novelty:** The **system architecture** (in-the-flow, 4-module coordination) is novel and well-motivated by the limitations of current monolithic RL and training-free agents. However, the **Flow-GRPO** algorithm is more of an application/engineering adaptation of GRPO than a fundamental algorithmic contribution, which slightly dampens the novelty relative to the system design.
*   **Significance:** The work is highly significant for the community. It successfully bridges the gap between training-free agentic systems and monolithic reasoning models. The empirical demonstration that 7B systems can surpass larger closed models via structured training is an impactful contribution to the field of efficient LLM reasoning.
*   **Clarity:** Despite some formatting artifacts in the provided text, the core logic is explained clearly with good diagrams (Figure 2). The mathematical notation is consistent.
*   **Reproducibility:** High. Code and models are public, and training logs are described.

### Suggestions for Improvement
1.  **Clarify Algorithmic Contribution:** Refine the description of Flow-GRPO to be precise about its relationship to existing GRPO/PPO variants. If the contribution is primarily the *application* of broadcast rewards for stability in multi-agent systems, frame it as a "System-RL Integration" rather than a new fundamental algorithm. The equivalence proof in Appendix B is good but should explicitly acknowledge that standard PPO objectives imply similar properties.
2.  **Conduct Cost-Efficiency Analysis:** Add a section or analysis comparing the **cost-per-success** and **latency** of AGENTFLOW against baselines. Since 7B is cheaper than 200B, the aggregate cost of using 4 x 7B models vs. 1 x 200B (or 1 x 7B) is a critical metric for practical adoption that should be addressed.
3.  **Mitigation of Evaluation Bias:** To strengthen the claim of surpassing GPT-4o, consider using a diverse set of judges or a rule-based metric subset where LLM bias is minimized. Additionally, explicitly ensure that all baselines have access to the *same* toolset configuration (e.g., Google Search API versioning) to prevent confounding factors.
4.  **Failure Mode Analysis:** The paper emphasizes successes and gains. A balanced review would appreciate a detailed analysis of *when* the system fails (e.g., hallucinated tool parameters, infinite loops despite the verifier). A "Failure Analysis" section would increase confidence in the robustness claims.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Training vs. Inference Horizon Mismatch:** Training limits rollouts to 3 turns (Appendix C.1), but evaluation uses 10 turns (§4.4). Add experiments training with higher turn limits to verify the method actually learns long-horizon credit assignment rather than relying on inference-time scaling alone.
2. **Fair Tool-Augmented Baselines:** The comparison against GPT-4o does not specify if GPT-4o had tool access. Run GPT-4o with equivalent tool access (search/code) to ensure performance gains stem from the agentic architecture, not merely the availability of external tools.
3. **Verifier Capability Ablation:** The Verifier module is frozen (Section 4.1) yet controls termination. Evaluate performance with a stronger (e.g., GPT-4o) Verifier to determine if results are bottlenecked by the frozen verifier rather than the optimized Planner.

### Deeper Analysis Needed (top 3-5 only)
1. **Training Cost Breakdown:** Training relies on GPT-4o for reward judging at every step (Appendix C.1). Provide a detailed cost analysis (API calls vs. GPU hours) to validate claims of efficiency compared to baselines that do not require proprietary judges.
2. **Data Contamination Audit:** Training data mixes Search-R1 and DeepMath, while test sets include HotpotQA and AIME. Provide a rigorous deduplication report to ensure performance gains are not due to test set leakage during training data collection.
3. **Reward Noise Sensitivity:** Since rewards come from an LLM judge, they are stochastic. Analyze convergence stability under varying reward noise levels to prove Flow-GRPO's robustness claims are not artifacts of a specific judge configuration.

### Visualizations & Case Studies
1. **Credit Assignment Heatmaps:** Visualize attention or gradient norms across turns for successful vs. failed trajectories. This exposes whether the "broadcasted reward" actually guides early planning decisions or if the model ignores early turns.
2. **Termination Turn Distribution:** Plot the distribution of stopping turns for hard vs. easy tasks. If the frozen Verifier stops hard tasks too early, the Planner never learns to solve them, revealing a systemic bottleneck.
3. **Tool Switching Trajectories:** Show case studies where the Planner recovers from a wrong tool choice. Without these, it is unclear if the model learns strategic adaptation or just memorizes successful tool sequences.

### Obvious Next Steps
1. **Train the Verifier:** The Verifier dictates the MDP horizon but remains frozen. Extending Flow-GRPO to optimize the Verifier is necessary for a fully adaptive system.
2. **Replace Proprietary Judge:** Distill the GPT-4o judge into a local reward model to remove dependency on external APIs during training and ensure reproducibility.
3. **Long-Horizon Training:** Increase the training turn budget beyond 3 turns to match inference settings, ensuring the policy learns to manage context growth and fatigue over actual long horizons.

# Final Consolidated Review
## Summary
AGENTFLOW introduces a trainable agentic system with four coordinated modules (planner, executor, verifier, generator) that operates through an evolving memory, optimizing the planner via on-policy reinforcement learning inside the multi-turn execution loop. The proposed Flow-GRPO algorithm broadcasts a single trajectory-level reward to all turns, converting multi-turn optimization into tractable single-turn updates. Experiments across ten benchmarks demonstrate substantial gains over both monolithic tool-integrated RL models and training-free agentic systems.

## Strengths
- **Strong empirical results with efficient backbone**: The 7B-parameter AGENTFLOW achieves 8.2-18.0% gains over GPT-4o across search, agentic, math, and scientific tasks (Tables 1-2), demonstrating that structured agentic training can outperform much larger proprietary models on complex reasoning benchmarks.
- **Critical ablation validating on-policy training**: The SFT ablation (§4.3) showing catastrophic collapse (-19.0% average accuracy) when distilling GPT-4o trajectories, versus +17.2% gains from Flow-GRPO, provides strong evidence that offline training fundamentally fails to capture multi-turn dynamics—a key insight for the field.
- **Tool usage adaptation analysis**: Figure 8 demonstrates that the trained planner spontaneously shifts tool preferences (e.g., +42% Google Search usage on 2Wiki, +59.8% Wikipedia Search on MedQA), showing the policy learns genuinely non-trivial adaptation rather than memorizing fixed sequences.
- **Comprehensive benchmark coverage**: Evaluation spans four distinct reasoning domains (search-intensive, agentic, mathematical, scientific) with ten benchmarks, providing robust evidence of generalization beyond single-task optimization.

## Weaknesses
- **Training-inference horizon mismatch**: Training limits rollouts to T_max=3 turns (Appendix C.1) while evaluation uses T_max=10 (§C.2). This distribution shift is acknowledged but not analyzed—whether performance gains transfer from short-horizon training to long-horizon inference remains empirically unverified. The scaling analysis in Figure 7 shows inference turns help, but does not report results for models trained at longer horizons.
- **GPT-4o judge vs. GPT-4o baseline creates evaluation dependency**: The LLM-as-judge reward signal uses GPT-4o (§E.3, §C.1) during training, and GPT-4o also serves as a primary baseline. This dual role introduces potential confounds: the judge may favor reasoning styles similar to its own, and the training objective is shaped by GPT-4o's preferences. While standard practice, this should be disclosed as a limitation, and alternative judge ablations would strengthen confidence.
- **Compute cost not fully disclosed**: The system instantiates four 7B models (planner, executor, verifier, generator) simultaneously. This implies approximately 4× the inference compute of a single-model baseline. The abstract claims outperforming "larger proprietary models like GPT-4o" without acknowledging that AGENTFLOW's total inference compute (4×7B = 28B parameter-equivalent, plus tool API calls) is non-trivial. A cost-per-task or latency analysis would allow fair comparison.
- **Only the planner is trained—implications unexplored**: The executor, verifier, and generator remain frozen (§4.1). While motivated by simplicity, this design choice has consequences: a frozen verifier controls termination but cannot adapt to planner behavior; a frozen executor cannot improve tool calling. No ablation tests whether jointly training multiple modules would yield further gains or instability.
- **Small benchmark samples for some tasks**: For benchmarks like AIME24 (N=30) and sampled subsets (2Wiki, HotpotQA, Musique, GPQA at N=100), variance estimates become important for interpreting relative gains. While three trials are reported, confidence intervals would clarify whether observed differences are robust.

## Nice-to-Haves
- **Verifier capability ablation**: Testing with a stronger verifier (e.g., GPT-4o) would clarify whether the frozen verifier bottlenecks overall performance, particularly for early termination decisions.
- **Training at longer horizons**: Experiments training with T_max > 3 would verify whether Flow-GRPO genuinely solves long-horizon credit assignment or relies on inference-time scaling.
- **Failure mode analysis**: Beyond success cases in Appendix F, systematic analysis of when and why the system fails (e.g., verifier false positives, tool API errors) would strengthen robustness claims.

## Removed Points
- **Claim that baselines lack standardized tools**: The paper explicitly states in §C.3 that "we replace search tools in search-enhanced models with our Google Search tool" for fair comparison. This criticism is factually incorrect.
- **Claim of theoretical analysis being "trivial"**: While the proofs in Appendix B apply standard PPO/TRPO machinery, they correctly establish the formal properties of the approach. The analysis is conventional but not incorrect.
- **Formatting and notation nitpicks**: Issues with PDF parsing artifacts are not substantive paper problems.

## Novel Insights
Beyond the paper's contributions, a notable observation emerges from the tool usage analysis (Figure 8): Flow-GRPO training induces qualitatively different tool-selection behaviors across domains—shifting toward general search (Google) for knowledge-intensive tasks while preferencing specialized tools (Wikipedia, Web Search) for domain-specific queries like MedQA. This suggests the policy learns domain-appropriate strategies rather than a single heuristic, which contradicts the concern that broadcasting identical rewards to all turns might cause the model to ignore turn-specific contributions. Empirically, early turns do learn to plan differently.

## Suggestions
- Conduct and report training experiments with T_max=5 or T_max=7 to verify that short-horizon training transfers to long-horizon inference, or if longer-horizon training yields further gains.
- Add explicit compute cost analysis: report FLOPs, latency, or API call counts for AGENTFLOW vs. baselines to contextualize the efficiency claims.
- Clarify in the method section whether GPT-4o baselines had access to the same tool APIs during evaluation, or if they operated without tools. This determines whether the comparison is "tool-augmented system vs. tool-augmented system" or "tool-augmented system vs. base LLM."

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0]
Average score: 7.3
Binary outcome: Accept
