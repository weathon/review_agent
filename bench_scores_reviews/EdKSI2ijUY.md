## Summary
LMRL-Gym introduces a benchmark and open-source research framework for multi-turn reinforcement learning with large language models. It provides 8 tasks spanning interactive dialogue (20 Questions, Guess My City, Car Dealer) and RL capability tests (Maze, Text-Nav, Wordle, Chess, Endgames), each paired with offline datasets and online simulators. The framework includes implementations of BC, filtered BC, MC Returns, ILQL, and PPO baselines, enabling the community to develop and evaluate multi-turn RL methods beyond single-turn RLHF settings.

---

## Strengths

- **Paired FO/PO task variants are a concrete diagnostic design choice.** The Maze FO/PO and Text-Nav FO/PO pairs allow direct isolation of partial observability effects — something most prior text-game benchmarks do not systematically provide. The symbolic-vs-text Maze ablation (Appendix H) is a similarly concrete and insightful diagnostic that empirically shows where language wrapping degrades RL performance.

- **Simultaneous provision of offline datasets and online simulators.** Unlike most prior text-game or dialogue benchmarks that support only one modality, LMRL-Gym provides both. This directly enables fair head-to-head comparison of offline and online RL methods — a distinguishing feature that makes the benchmark more practically useful for algorithm development.

- **Empirical result that small RL-finetuned models close the gap to GPT-4 on capability tasks.** Table 2 shows that GPT-2-scale models trained with ILQL substantially outperform GPT-4 few-shot prompting on structured tasks like Maze, Text-Nav, and Wordle, providing concrete evidence that RL fine-tuning adds value beyond scale on goal-directed tasks. The divergence in dialogue tasks (where GPT-4 dominates) is a genuinely informative finding that motivates further algorithm development.

- **Inclusion of diverse capability targets in a single benchmark suite.** By explicitly mapping tasks to properties (trajectory stitching, credit assignment, partial observability, strategic decision-making, complex language), LMRL-Gym distinguishes itself from text-game benchmarks that test only task completion without diagnostic intent. This structuring is a meaningful contribution even if the mapping needs tightening (see weaknesses).

---

## Weaknesses

### Fatal
None.

### Major

- **No statistical reporting undermines the benchmark's core purpose.** Table 2 reports single normalized scores with no confidence intervals, standard deviations, or multi-seed evaluation for any method on any task. For a benchmark whose explicit purpose is to "gauge progress on algorithm design" and identify which RL algorithms outperform others, this is a critical gap. It is impossible to determine whether differences such as ILQL at 83.7 vs. PPO at 85.5 on PO Text-Nav, or BC variants at 47.2 vs. 48.0 on Chess, represent genuine algorithmic differences rather than noise. The benchmark cannot reliably rank methods without this information.

- **Inconsistency between Figure 2 capability labels and the prose undermines the benchmark's central diagnostic claim.** Figure 2 shows Chess with no credit assignment checkmark, but Section 4.2 explicitly states "The Chess, Endgames, Maze and Text-Nav tasks test credit assignment, because the RL algorithm must learn to assign credit to good actions." This is a factual internal contradiction. Additionally, Endgames does not appear in the Figure 2 table at all despite being discussed alongside Chess throughout Section 4.2. If the capability-to-task mapping — the key structural claim that distinguishes LMRL-Gym from generic benchmark suites — contains errors, the paper's diagnostic framing is weakened.

- **Simulator validity for dialogue tasks is only partially addressed.** The paper verifies naturalness through a 40-user study (Appendix A) but natural-sounding dialogue is not the same as strategic validity or resistance to reward hacking. The paper itself acknowledges the benchmark's goal is to test whether RL algorithms can learn to "accomplish tasks in an intentional and goal-directed manner" against the simulator. If the dialogue simulator can be exploited through non-human-like patterns, high-reward policies may not correspond to meaningful persuasion or information-gathering strategies. The paper mentions automatic checks for 20Qs/Guess (where correct guessing is verifiable), and acknowledges that "natural conversations... indicate the robustness of the Buyer model to hacking," but this chain of inference is weak — naturalness is a necessary but not sufficient condition for non-exploitability in Car Dealer. No targeted adversarial audit or reward-hacking analysis is provided.

- **GPT-4 scoring 0 on Chess and Endgames is unexplained and potentially misleading.** Table 2 shows GPT-4 at 0 on both Chess and Endgames, while GPT-2-scale RL models score in the 45–77 range. If this is because GPT-4 emits illegal moves that receive zero reward (a formatting/interface issue), it would reflect benchmark design rather than a meaningful capability gap. The paper does not explain the mechanism behind these 0 scores, which is essential for interpreting one of the most striking results in Table 2. If the 0 reflects interface mismatch rather than inability, the claim that "RL fine-tuning significantly outperforms GPT-4 on capability tasks" is substantially compromised for Chess and Endgames.

### Minor

- **The claim that offline RL "consistently outperforms" filtered BC is overstated.** Section 6 states offline RL "consistently outperform... the filtered BC policies," but Table 2 shows ILQL at 46.3 on Car Dealer, below filtered BC's 54.8. On Chess, differences are negligible (BC: 47.2, %BC: 42.9, MC: 46.5, ILQL: 47.3). The results are generally favorable for RL but not uniformly so; the narrative should be more calibrated.

- **No training curves for any method.** The paper notes PPO instabilities in Section 6 but provides no learning curves. For a benchmark paper aimed at guiding algorithm development, the absence of convergence information — including whether methods plateau, overfit the simulator, or destabilize — reduces the practical value for researchers choosing algorithms.

- **Dataset size imbalance across tasks complicates interpretation.** Table 1 shows sizes ranging from 1.24k (Maze) to 1M (Wordle). A method that performs well on Wordle may benefit from large data scale; a method that struggles on Maze or Car Dealer may reflect data scarcity rather than algorithmic weakness. The paper does not discuss or control for this in its cross-task comparisons.

- **Reward function details are deferred to appendices.** Reward structure (sparse vs. dense, shaping terms, how the Car Dealer reward is computed) is not summarized in the main text. For interpreting algorithm performance differences, reward signal density is critical context.

### Tiny

- **The speculative attribution of ILQL's superiority to trajectory stitching is asserted, not tested.** Section 6 states "ILQL's performance on these tasks is likely due to its unique ability to perform trajectory stitching," but no controlled ablation varying the proportion of suboptimal trajectories, or comparing ILQL vs. MC Returns on trajectory-stitching-specific setups, is provided. This is a plausible hypothesis, not an empirical finding.

- **No qualitative examples in the main paper of RL-trained agent behavior vs. BC.** Appendix I contains examples, but since verifying that reward improvements correspond to meaningful behavioral changes (rather than simulator quirks) is central to the benchmark's validity, at least one illustrative comparison belongs in the main text.

---

## Nice-to-Haves

- **Scaling experiments with at least one 7B-class model on a subset of tasks.** The paper explicitly notes the GPT-2/FLAN-T5 scale limitation; even a partial experiment would strengthen confidence that algorithmic rankings generalize beyond the smallest viable models.

- **Compute cost estimates for the full benchmark.** Given the accessibility motivation, a table indicating approximate GPU-hours for each method-task combination would directly serve the audience the paper targets.

- **Richer failure-mode analysis for the dialogue task gap.** ILQL underperforms MC Returns on all dialogue tasks but the cause (reward sparsity, value overestimation, simulator stochasticity, long text horizons?) is unanalyzed. A structured investigation, even qualitative, would make the finding actionable for future algorithm designers.

- **Controlled analysis validating the trajectory stitching diagnostic.** Varying the fraction of suboptimal trajectories in the offline dataset and measuring ILQL vs. MC Returns performance would provide empirical grounding for the trajectory stitching characterization in Figure 2.

---

## Removed Points
*These points are flagged for removal; treat them with caution.*

- **Critic: "The abstract overstates scope; the benchmark mixes language and game tasks."** The paper is transparent about this in Sections 1 and 4. The abstract accurately says the benchmark "covers tasks in open-ended dialogue and text games." This is not an overstatement.

- **Critic: "Insufficient comparison table in related work."** The paper provides adequate narrative differentiation in Section 2. Demanding a formal comparison table for every possible predecessor benchmark is a formatting preference, not a scientific requirement. Removed.

- **Critic: "The claim that no prior work evaluates offline RL in this way is too broad."** The paper's exact phrasing is "evaluating offline RL capabilities, which is not done by prior works" (Section 2). This is a reasonable scoping claim about the specific combination (multi-turn + offline datasets + simulators). The critic's call to "exhaustively establish" this is unreasonable given that the paper is making a combined-contribution claim. Removed.

- **Critic: "POMDP framing is conceptually muddy."** The paper defines states as full token histories (which are Markovian from the agent's perspective) and separately introduces partially-observable variants. This is standard in the LLM-RL literature and not muddier than necessary. The distinction the critic wants is present in the paper's FO/PO task variants. Removed.

- **Critic: "Baseline coverage too narrow; should add Decision Transformer, conservative offline RL."** The paper implements BC, filtered BC, online filtered BC, MC Returns, ILQL, PPO, and GPT-4. For a benchmark paper at ICLR this is a reasonable algorithmic sweep. Demanding additional baselines not currently standard in LLM-RL evaluation is scope creep. Removed.

- **Critic: "Pretraining contamination from cities, chess notation, Wordle not discussed enough."** The benchmark's explicit scope (Section 1, 4.3) is to evaluate RL algorithm improvements, not to measure world knowledge. Prior knowledge may affect BC performance but is orthogonal to whether RL improves over BC. This concern does not undermine the core contribution. Removed.

- **Critic: "PPO sample budget cap at <100k makes comparisons unfair."** The paper explicitly discloses this constraint (Section 5). Capped online RL as a baseline for offline methods is standard and, if anything, favors the offline baselines — making it a conservative comparison. Removed.

- **Strength: "The paper is well-written and the topic is important."** Generic; applies to any acceptable ICLR submission. Removed.

---

## Novel Insights

The spark finder's most useful observation — that the benchmark currently only provides rankings, not insights into *why* methods succeed or fail — deserves emphasis beyond what the paper itself articulates. The divergence between ILQL (strong on capability tasks, weak on dialogue) and MC Returns (strong on dialogue, weak on capability tasks) is one of the most substantive empirical findings in the paper, but the mechanism is completely unanalyzed. This gap points to a genuine open problem: TD-learning appears to break down when the action space is high-entropy natural language, but the specific failure mode (value overestimation, reward sparsity, stochastic simulator responses, compounding decoding errors) is unknown. A future version of this benchmark would be substantially more valuable if it included controlled experiments varying reward density, action entropy, and simulator stochasticity independently across tasks — precisely the kind of diagnostic experiment the paired FO/PO task structure is positioned to enable, but currently does not exploit.

---

## Suggestions

1. **Add multi-seed evaluation to Table 2 immediately.** Without variance estimates, the benchmark cannot fulfill its stated purpose of distinguishing algorithms. This is not an optional enhancement; it is a prerequisite for the core claim.

2. **Resolve and correct the Figure 2 vs. prose inconsistency on Chess/Endgames and credit assignment.** Determine whether Chess tests credit assignment (as the prose says) or does not (as Figure 2 shows) and update both to be consistent. Add Endgames to the Figure 2 table.

3. **Explain the mechanism behind GPT-4 scoring 0 on Chess and Endgames.** Report whether illegal move generation, output format failures, or true strategic failure is responsible. If it is interface-driven, implement legal-move retry logic or prompt standardization and report revised GPT-4 numbers. The current result, if interface-driven, misrepresents a key comparison.

4. **Include at least one qualitative trajectory comparison (RL vs. BC vs. best dialogue baseline) in the main paper** for Car Dealer, to demonstrate that reward improvement corresponds to meaningful behavioral change rather than simulator exploitation.

5. **Add training curves for PPO and ILQL** on at least two representative tasks (one dialogue, one capability) to expose convergence, instability, and saturation patterns — central information for practitioners adopting the benchmark.

---

## Paper Evaluation

| Axis | Assessment |
|---|---|
| **Originality** | Moderate. The specific combination of multi-turn dialogue tasks, structured RL capability tests with FO/PO pairs, offline datasets, and online simulators in one package is novel. Individual components are not. |
| **Importance of research question** | High. Multi-turn RL for LLMs is a genuine and underserved gap; infrastructure for this is valuable. |
| **Claims well supported** | Partially. Directional findings (RL > BC, ILQL wins on capability tasks, MC Returns wins on dialogue) are plausible but lack statistical support. The Figure 2 inconsistency undermines the benchmark's diagnostic framing. |
| **Soundness of experiments** | Moderate. Single-run results, unexplained 0 scores for GPT-4 on Chess/Endgames, and limited model scale reduce confidence in the conclusions. |
| **Clarity of writing** | Generally clear, but with internal inconsistencies (Figure 2 vs. prose) and key experimental details deferred to appendices. |
| **Value to research community** | High potential, conditional on the infrastructure working as described and the statistical gaps being addressed. The framework and datasets are the primary contribution. |
| **Contextualized relative to prior work** | Adequate. The paper correctly identifies its niche (multi-turn + offline RL) but the related work is more encyclopedic than analytical. |

Overall, this is a useful and timely infrastructure contribution whose empirical validation does not yet meet the standard the paper's claims require. The benchmark design — particularly the FO/PO pairing and the symbolic-vs-text ablation — reflects genuine methodological thought. But the absence of statistical reporting is a fundamental gap for a paper whose purpose is to measure algorithmic progress, and the Figure 2 inconsistency and unexplained GPT-4 results require correction before the diagnostic claims can be trusted.