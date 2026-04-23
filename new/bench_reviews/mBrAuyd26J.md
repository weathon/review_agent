Now let me run calibration searches in parallel.Now I have enough context to write the final review. Let me compile everything.Now I have all the information needed to write the final review.

---

## Summary
The paper proposes a dual-system reasoning framework for LLM-based agents, where an external *Thinker* module handles domain-specific System-2 reasoning (deduction, strategy) while an LLM handles domain-agnostic System-1 tasks (NLU/NLG). This is demonstrated on a 9-player Werewolf game using RL-based self-play to train the Thinker, supplemented by the FanLang-9 dataset of 18,800 real human game sessions—the largest social-deduction-game dataset to date. Experiments cover deductive reasoning accuracy, human speech preference, and online win rates.

---

## Strengths

- **Large-scale, genuine dataset contribution (Section 3.1):** FanLang-9 comprises 18,800 real game sessions (~7,000 hours of gameplay, 6,000 hours of audio), a domain corpus of 1.4M characters, and 127 hours of annotated Werewolf speech for ASR fine-tuning. This substantially exceeds prior datasets (e.g., Deep Wolf used 48 logs) and is an independent contribution independent of the framework's performance.

- **Consistent within-group improvement from the Thinker (Table 1):** The most defensible comparison is same-LLM with vs. without the Thinker: GPT-3.5-LMM (36.7%) → GPT-3.5-T (47.4%) and GPT-4-LMM (37.9–42.5%) → GPT-4-T (41.1–46.3%), across all three mixture combinations. The Behavior Score also improves consistently. These within-group comparisons directly isolate the Thinker's contribution.

- **Multi-dimensional evaluation (Sections 4.1–4.3):** The paper evaluates across three qualitatively distinct axes—deductive reasoning accuracy (~1,200 instances), human preference ranking of speeches (10 evaluators, 2,000 speeches), and online win rates (~600 rounds per combination). This is more thorough than single-metric evaluations typical in this space.

- **Practical filtering mechanism (Section 3.4):** The Listener-based post-processing step—re-inferring language features from the Presenter's output and requiring consistency with the original speech instruction—is a well-motivated, implementable solution to LLM hallucination in constrained generation, with a graceful degradation to a template fallback.

---

## Weaknesses

### Fatal
None.

### Major

- **The deductive reasoning evaluation (Figure 3) compares a supervised classifier against zero-shot prompting—and is presented as testing reasoning capability.** Section 3.3 states clearly: the Thinker's identity model is *"labeled by the game environment in a self-supervised manner"* (i.e., trained with ground-truth role labels from 18k games). The GPT baselines receive no such supervision—they infer identities purely from prompting. Comparing a supervised classifier trained on thousands of labeled games to a zero-shot reasoner and claiming the former demonstrates superior *deductive reasoning* is misleading. The 40–50% vs. ~20–30% accuracy gap is the expected consequence of supervised vs. zero-shot learning, not a measure of reasoning quality. This weakens Section 4.1 as a standalone contribution. The paper should either reframe Figure 3 as "comparing a trained identity predictor to a prompted LLM" (a valid engineering comparison), or add a comparable supervised baseline for the GPT setting.

- **The training assumption of a perfect Listener/Presenter (Section 3.3) is structurally violated at inference (Section 3.4), and the magnitude of this mismatch is entirely uncharacterized.** The Thinker is trained assuming speech instructions pass through to language features losslessly. At inference, the filtering mechanism (Section 3.4) handles imperfect Presenter outputs, but the paper never reports: (a) how often the filter rejects outputs, (b) how often the template fallback is triggered, (c) whether this rate differs across model configurations (WereLLM-T vs. GPT-4-T). Since the Thinker's reward signal was derived under the perfect-pipeline assumption, a large gap could differentially benefit configurations whose LLMs are better calibrated to the Werewolf speech format (i.e., WereLLM-T, whose LLM was fine-tuned on the same 260k speech examples). Without this characterization, the comparative win-rate claims in Table 1 rest on an unvalidated assumption.

### Minor

- **No explicit train/test split documentation for the 300-game test set.** Section 4.1 states the 300 games are extracted from FanLang-9 after describing Thinker training on FanLang-9 (Section 3.3), but never specifies that these games were held out before training. For the identity model specifically—which is trained to predict roles from game trajectories—even modest overlap would inflate Figure 3 accuracy substantially. The paper should explicitly state when and how this split was made.

- **No statistical significance reported for any win-rate result.** Table 1 reports win-rate differences of 3–10 percentage points over ~600 rounds per configuration; Table 2 reports 5–7 percentage-point differences over 200 rounds with 13 humans. Werewolf outcomes are highly stochastic (random role assignment, multi-player dynamics), and no confidence intervals or p-values are provided. The claim that the Thinker "significantly boosts win rates" uses "significantly" in an informal sense; the statistical significance of the improvements is not established.

- **Human accuracy baseline in Figure 3 is poorly grounded.** Section 4.1 states: *"we assume that human players in the test set who are villagers would vote for the most likely werewolf."* In reality, human villagers vote based on their own (potentially mistaken) deductive reasoning—not ground truth—so the "human accuracy" figure is not a measure of human deductive ability but of human game behavior. This makes it an unreliable reference point for comparing reasoning systems.

- **Table 2 human evaluation is underpowered and missing a key ablation.** Only 13 human players participated across 200 rounds, with no information on their expertise level. More importantly, the paper does not include a Thinker-free condition (GPT-4-LMM) against humans. Without that baseline, it is impossible to attribute any AI advantage specifically to the Thinker when playing against humans. The current Table 2 only shows that Thinker-equipped AI is competitive with humans, not that the Thinker is responsible.

### Trivial
- The WereLLM-T vs. GPT-4 comparison in the abstract and introduction could be more precise: WereLLM-T received both Thinker training and fine-tuning on 260k Werewolf speech examples, while GPT-4 received neither. The claim of a "6B model achieving performance on par with GPT-4" understates this asymmetry.

---

## Nice-to-Haves
- An ablation varying Listener extraction quality (e.g., degrading the retrieval accuracy) would clarify whether the Thinker's benefit is robust to Listener errors—a question central to the system's deployability.
- A proper "Turing-style" evaluation—can human players detect which participant is the AI?—would directly address the paper's implicit human-level performance claim. The paper acknowledges this gap in Section 5 but does not attempt it.
- Sensitivity analysis on training hyperparameters: the 5:1 Werewolf/Good training ratio and population size of 4 are presented without justification; brief ablations would strengthen confidence in the training design.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic Issue 1 (Evaluation circularity, entire Table 1 invalidated)**: The harshest version of this critique—that *all* of Table 1 is circular because the Thinker was trained in the same self-play distribution—overstates the problem. The within-group comparison (GPT-X-LMM vs. GPT-X-T) is a valid head-to-head showing the Thinker's incremental contribution. The concern about comparing trained specialist vs. untrained generalist is real but addresses only the cross-group comparisons (e.g., WereLLM-T vs. GPT-4-LMM); this is retained as a framing concern (Trivial), not a fatal flaw. The harsh critic's framing that this "cannot be recovered" by Table 2 is excessive given that within-group comparisons remain valid.

- **Missing Listener extraction accuracy evaluation**: The harsh critic asks for ablations of Thinker performance under degraded Listener conditions. This is a reasonable suggestion (moved to Nice-to-Haves) but not a fundamental flaw that invalidates the paper.

- **Reward shaping details deferred to appendix**: The paper states reward shaping details are in the appendix. Per the rules, criticisms about missing appendix content are removed (the parser strips appendices).

- **Detailed prompts not in main text**: The harsh critic flags that detailed prompts are in the appendix. Same rule applies.

- **Strength: Figure 1 architectural comparison**: Removed as a listed strength—it is a presentation choice, not an evidence-backed contribution.

---

## Novel Insights

The paper's most genuine novel observation is that the communication bottleneck in Cicero-style action-based representations (predicted/intended action pairs) is too lossy for a game like Werewolf, where the gap between stated and intended content is intentionally large. The richer language-feature/speech-instruction protocol—essentially a natural-language structured intermediate representation—fills this gap and allows the RL component to be trained at scale independently of LLM inference costs. This decoupling insight has broader applicability: it suggests that any domain where LLM reasoning degrades (context overload, structured-logic requirements) could benefit from a similarly decomposed architecture, as long as an interpretable intermediate representation can be designed. The self-supervised identity model as an auxiliary task to reveal the Thinker's implicit beliefs is also a practically useful interpretability mechanism not commonly seen in RL game agents.

---

## Suggestions
1. Rename or reframe Figure 3 to distinguish between the supervised identity model and the GPT prompted baselines—call it "supervised identity classification vs. prompted inference" rather than a "deductive reasoning comparison."
2. Add a footnote or paragraph confirming the temporal/procedural order of the 300-game test-set extraction relative to Thinker training.
3. Add binomial 95% confidence intervals to all win-rate figures in Tables 1 and 2.
4. Add a GPT-4-LMM (no Thinker) row to Table 2 to enable attribution of human-facing gains to the Thinker specifically.
5. Report the filter rejection rate and template-fallback rate per model configuration (can be a single table in the supplementary).

---

## Score and Decision

**Calibration anchors used:**

| Paper | Avg Human Score | Comparison |
|---|---|---|
| Avalon/ReCon (fvhJu0FODp) | 3.75 (Withdrawn) | Same game type (Avalon), pure prompting, no dataset, weaker contribution; paper under review is clearly above this. |
| LLM Self-Play in DoND (tCfvktlrHI) | 4.75 (Reject) | RL + LLM for games, rejected for limited task scope and overclaiming; paper under review has a larger dataset and more evaluation dimensions but shares some "overclaim" patterns in Figure 3. |
| Entity-Deduction Arena (PfrpYGKGPL) | 5.50 (Reject) | LLM evaluation in deduction games; broadly similar evaluation limitations. |
| Hypothetical Minds (otW0TJOUYF) | 6.75 (Accept Poster) | MARL + LLM with Theory of Mind; multi-game evaluation, more rigorous methodology—paper under review is somewhat below this due to Figure 3 methodological flaw and limited human evaluation. |
| RL self-correction for LLM reasoning (CjwERcAU7w) | 8.0 (Accept Oral) | High-quality RL + LLM paper with rigorous evaluation; paper under review is clearly below this level. |

**Positioning:** The paper lands between the LLM self-play paper (4.75, reject, limited scope, overclaim) and Hypothetical Minds (6.75, accept, solid multi-task evaluation). The FanLang-9 dataset and the multi-dimensional evaluation push it above 4.75. However, the Figure 3 methodological flaw (supervised vs. zero-shot labeled as "reasoning comparison"), the uncharacterized training–inference gap, and the missing statistical significance tests are real issues that prevent it from reaching 6.75. The within-group Thinker ablations in Table 1 are solid engineering evidence.

**Overall assessment:** The paper's core engineering idea is sound and the dataset is a genuine community contribution. However, the strongest "reasoning improvement" claims rest on a methodologically flawed comparison (Figure 3), and the training–inference distributional gap is uncharacterized. These are addressable in revision but are not trivial concerns. The paper is better than the rejected LLM self-play paper but weaker than the Hypothetical Minds accept.

**Score: 5.0**

*Originality:* The Thinker decomposition idea is reasonably novel in the Werewolf/social-deduction context, though clearly related to Cicero. Medium originality.
*Importance:* Social deduction games are a valuable testbed; the dataset is an important contribution. Moderate importance.
*Claims vs. support:* The win-rate claims (within group) are adequately supported. The deductive reasoning claims are overclaimed given the supervised vs. zero-shot comparison. Mixed.
*Experimental soundness:* Mixed—multi-dimensional evaluation is good, but Figure 3 methodology is flawed and no significance testing is reported.
*Clarity:* Generally clear and well-organized.
*Community value:* The dataset and architecture ideas have value; the evaluation methodology needs improvement.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>