## Summary
This paper proposes a dual-system framework to enhance LLM reasoning in complex social deduction games, specifically the 9-player Werewolf game. The architecture separates the pipeline into a **Listener** (for speech summarization and feature extraction), a **Thinker** (an RL-trained module for System-2 strategic reasoning, vote/skill planning, and speech instruction generation), and a **Presenter** (for speech generation). The authors release **FanLang-9**, a large dataset of ~18,800 recorded human game sessions, and demonstrate through online evaluations that integrating the Thinker module significantly improves win rates and deductive reasoning accuracy over prompting baselines (GPT-3.5/4 with Least-to-Most prompting). Additionally, a fine-tuned 6B model (WereLLM) augmented with the Thinker achieves performance comparable to GPT-4.

## Strengths
- **Significant Dataset Contribution:** The *FanLang-9* dataset, encompassing 18,800 recorded sessions (~6,000 hours of audio) and nearly 1.4 million characters of domain-specific corpus, addresses a critical data scarcity bottleneck in social deduction games and provides a substantial resource for the community.
- **Effective Performance Gains over LLM Baselines:** The integration of the Thinker module demonstrates tangible improvements. For example, Table 1 shows that GPT-3.5-T increases the total win rate to 47.4% compared to 36.7% for the GPT-3.5-LtM baseline. Furthermore, the 6B WereLLM paired with the Thinker matches the win rate of GPT-4-T (45.9% vs 46.3% in the top configuration), demonstrating the potential of domain-specific smaller models when coupled with structured reasoning.
- **Pragmatic Modular Pipeline:** The decomposition of the problem into a Listener, Thinker, and Presenter effectively circumvents context window limits and hallucination issues inherent in long-horizon dialogue games. By forcing explicit, structured state representation (language features) before strategic execution, the framework provides a clear and replicable engineering pattern for complex multi-agent interactions.

## Weaknesses

### Fatal
None.

### Major
- **Train-Test Distribution Mismatch in Thinker Training:** Section 3.3 explicitly states: *"During the Thinker’s training, we assume that the Presenter generates speech accurately... and the Listener... generate[s] a language feature that precisely matches the original speech instruction."* This means the Thinker is trained with oracle-level, deterministic features in a closed loop, but at inference, it must rely on noisy ASR outputs, LLM summarization artifacts, and retrieval errors from the Listener, as well as hallucination-prone generation from the Presenter. While Section 3.4 introduces a consistency filter at inference, this circular validation is entirely absent during training. Consequently, the Thinker’s claimed "robust, online deductive reasoning" is unverified because the policy network is optimized on a feature distribution that does not exist in deployment, creating a severe structural gap between training and inference.
- **Misalignment Between "System-2" Framing and Method Implementation:** The introduction and Section 3 define the Thinker as handling *"System-2 reasoning that is deliberate, analytical,"* contrasting it with the intuitive System-1 tasks of LLMs. However, Equations 2–4 and the surrounding text reveal the Thinker is a standard policy network $\pi_\theta(a|s)$ optimized via Behavioral Cloning and PPO over a discrete classification space. There is no iterative deliberation, symbolic deduction, or hierarchical planning mechanism; all complex logical parsing and information extraction are delegated to the LLM Listener. The paper conflates standard RL policy optimization with cognitive "System-2" reasoning, rendering the core theoretical framing disconnected from the actual implementation.
- **Inadequate Online Evaluation and Win Rate Interpretation:** The paper relies heavily on win rates in a 9-player asymmetric social deduction game to substantiate claims of "significantly improved deductive reasoning" (Table 1). However, in Werewolf, win rates are overwhelmingly sensitive to faction imbalance and opponent coordination rather than individual agent capability. The paper reports high Werewolf win rates (e.g., 74–81%) but fails to disentangle Good vs. Werewolf faction performance or employ Elo/league-style ratings to control for opponent strength. A high Werewolf win rate often indicates weak Good-faction coordination rather than superior Werewolf deduction. Without variance, confidence intervals across the ~600 rounds, or faction-level analysis, the experimental outcomes cannot fully support the central reasoning claims.

### Minor
- **Missing Critical Baseline in Main Text:** While the paper compares against GPT prompting variants, it relegates the comparison to the RL+LLM baseline by Xu et al. (2023b) to Appendix B.2. Xu et al. is a highly relevant work applying RL to Werewolf agents; moving this comparison to the main text is necessary to establish that the Thinker provides state-of-the-art reasoning beyond just prompt engineering improvements.
- **Lack of Inter-Annotator Agreement in Human Evaluation:** The human preference evaluation (Figure 4) uses rank scores from 10 evaluators but does not report inter-annotator agreement metrics (e.g., Fleiss' Kappa). Without agreement statistics, it is difficult to calibrate the reliability of the human alignment claims, especially given the subjectivity of "speech quality" and "deception" in the game.

### Trivial
- **External References for Metrics:** The "Behavior Score" metric cited in Table 1 relies on an external link rather than a definition or normalization in the main text, which hinders immediate reproducibility and evaluation.

## Nice-to-Haves
- **Feature Importance Analysis:** Analyzing attention weights or feature importance in the Thinker’s policy would help demonstrate that the model is performing genuine deduction rather than memorizing voting priors from the Behavioral Cloning dataset.
- **Closed-Loop Fine-Tuning:** Future work could benefit from end-to-end optimization where the full pipeline (Listener/Thinker/Presenter) is jointly fine-tuned or simulated, rather than assuming independent optimality of modules.
- **Distributional Shift Visualization:** Visualizing the distributional shift between the training features (perfect/instructions) and inference features (noisy LLM output) would help quantify the real-world impact of the train-test mismatch.

## Removed Points
- **Criticism of ASR error rate dropping "only" to 3.7%:** The reviewer questions if 3.7% Character Error Rate is sufficient. While high for some metrics, this is a standard ASR performance for domain data; the paper acknowledges this and does not claim perfect transcription. This is a minor implementation detail.
- **Nitpick on 5:1 training iteration ratio:** The reviewer notes the heuristic of optimizing Werewolves 5x more than Good players. The paper explicitly justifies this in Section 3.3 due to the adversarial nature and the difficulty of Werewolf disguise; this is a practical training stability choice, not a flaw.
- **Criticism of "undisclosed hyperparameters" / "missing proofs":** These fall under standard reproducibility nitpicks for empirical RL papers and are deferred to the appendix in the original submission.
- **Criticism claiming Circular Validation is a "structural flaw" separate from the independence assumption:** The reviewer treated the inference-time filter as a circularity that exacerbates the mismatch. This is valid but is already fully captured and emphasized in the Major Weakness regarding the train-test mismatch; separating it creates redundancy.

## Novel Insights
The paper introduces a practically motivated separation of concerns in LLM agents for social deduction: offloading the high-context, unstructured natural language understanding to a general-purpose LLM (Listener) while isolating the strategic decision-making into a specialized, RL-trained policy network (Thinker). The finding that a relatively small (6B) domain-fine-tuned model can match GPT-4 performance when augmented with this structured reasoning module highlights the potential of "hybrid" AI systems—where domain-specific optimization bridges the gap between generalist LLMs and specialized task competence.

## Suggestions
1. **Address the Independence Assumption:** In the rebuttal or revision, add an ablation where the Thinker is trained with realistic noise injected into the state features (simulating Listener errors) to verify robustness against the train-test distribution mismatch.
2. **Enhance Online Evaluation Rigor:** Report faction-disentangled win rates and, if possible, confidence intervals or Elo-style ratings to provide statistical backing for the reasoning claims.
3. **Clarify Methodological Framing:** Temper the "System-2" terminology or explicitly map the components of the PPO agent to the cognitive definition provided, acknowledging that the current implementation approximates System-2 via policy optimization rather than symbolic deliberation.

## Score and Decision
Based on calibration against similar works:
- Compared to **GameArena** (Average score ~6.5, Accepted), which focuses on benchmarking and has strong evaluation designs, this paper has a stronger engineering contribution and dataset but suffers from the major methodological weakness of the train-test independence assumption.
- Compared to **GameInstruct** (Average score ~4.2, Rejected), which was rejected for lack of novelty (merely swapping a game), this paper offers a novel architectural pipeline and a very substantial dataset contribution that saves it from rejection.
- The **independence assumption (Sec 3.3)** is a significant flaw for a method paper claiming "robust reasoning," but the empirical results (Tables 1 and 2) show the system works well enough in practice to warrant a borderline score rather than a rejection.

The paper provides a valuable dataset and a clear performance boost over baselines, but the structural decoupling of training and inference weakens the core reasoning claims.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Borderline</orange>