# Framing the Game: How Context Shapes LLM Decision-Making

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Large Language Models (LLMs) are increasingly deployed across diverse contexts to support decision-making. While existing evaluations effectively probe latent model capabilities, they often overlook the impact of context framing on perceived rational decision-making. In this study, we introduce a novel evaluation framework that systematically varies evaluation instances across key features and procedurally generates vignettes to create highly varied scenarios. By analyzing decision-making patterns across different contexts with the same underlying game structure, we uncover significant and specific contextual influence on LLM decision-making. Our findings demonstrate this variability is largely predictable, yet acutely sensitive to framing effects. These results underscore the urgent need for dynamic context-aware evaluation methodologies to ensure reliable LLM deployment in real-world applications, and provides initial directions for their construction.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies how narrative framing and context affect LLM decision-making in Prisoner's Dilemma scenarios. The authors develop an LLM-based generation framework to create varied vignettes with a consistent underlying Prisoner's Dilemma structure under real-world and imagined modes, varying topics and actor relationships. They examine 6 LLMs and find a significant context-dependent decision-making phenomenon that is largely predictable from these variables.

### Strengths
1. **Novel and meaningful perspective.**
This paper discusses an underexplored but fundamental question: how contextual framing shapes LLMs' decision-making, which provides a refreshing view for analyzing LLM rationality.

2. **Well-structured and scalable framework.**
The proposed vignette generation pipeline is scalable, flexible, and reproducible. By systematically varying contextual dimensions under the payoff matrix, the authors create a modular evaluation framework that can be easily extended to new games or social settings, allowing controlled and large-scale evaluation.

### Weaknesses
1. **Lack of deeper interpretation of contextual influence in decision-making.**
The experimental results show that actor type strongly influences cooperation rates (Allies > Neutral > Enemies, according to Figure 3), which the authors interpret as evidence of contextual instability in LLM decision-making. However, such variance could also be explained as an expression of human-like social preferences (e.g., trust and reciprocity) that naturally affect the cooperative behavior in real-world interactions, even though humans often deviate from the Nash equilibrium when facing allies or friends, reflecting contextual priors rather than irrationality.  Therefore, the observed pattern might not indicate poor rational consistency but reveal a contextually guided reasoning that LLMs exhibit implicit priors about social trust (e.g., allies → more cooperation; enemies → more defection). I think the authors can explore this interpretation and discuss whether an ideal LLM should adhere to the Nash equilibrium or emulate human social reasoning. If the authors can include human experiments and evaluate the alignment of LLMs and humans would substantially deepen the theoretical impact of the paper.


2. **Limited robustness analysis.**
In GAMA-Bench [1], the authors conducted a generalizability experiment by varying the payoff parameters and found that some models' decision-making patterns changed under different conditions. In this paper, the authors preserve a consistent payoff matrix (Figure 1) to generate those vignettes, but the contexts rarely provide the explicit and quantitative payoff descriptions. In Appendix A, only 1/8 vignettes specify the numerical and comparable gains, while others use affective terms like "low/high level of happiness". This raises two concerns for me: (1) Without explicit and comparable payoffs, the cooperation rates might be varied across different contexts, especially given that LLMs' decision-making does not exactly follow the Nash Equilibrium. (2) I still have a question about whether the observed behavioral variance reflects contextual sensitivity or noise introduced by inconsistent payoff.

3. **Lack of behavioral analysis.**
Sections 3.3.1 and 3.3.2 focus heavily on the cooperation proportions across topics, world types, and actor types, but the analysis is mostly descriptive. For example, the authors repeatedly restate surface-level variance patterns (e.g., cooperation in GP 21C > GP 5C, allies > neutral > enemies) without deeper behavioral interpretation. Their explanations remain high-level speculations (e.g., contextual influence and training-data overlap). I think a stronger contribution is to connect these phenomena to evidence from the models' reasoning traces, identifying which cues drive cooperation or defection biases with ablation experiments to evaluate such hypotheses. This helps the study turn from a report into a behavioral-mechanistic analysis, improving both rigor and insight.
The results in Section 3.3.3 provide a good motivation for studying the behavioral-mechanistic, because the authors show that models' decision-making is not random. However, this section does not provide insight into why these variables matter or what linguistic and moral features affect the predictions.

[1] How Far Are We on the Decision-Making of LLMs? Evaluating LLMs' Gaming Ability in Multi-Agent Environments. Huang et al.

### Questions
1. Cross-gaming Generalization: How does this phenomenon perform on other types of classical economy games (e.g., Public Goods Game), and do LLMs exhibit similar patterns?
2. Robustness Evaluation: How does the cooperation rate vary under different prompts (e.g., using the model to rephrase the generated context) and different payoff matrices?
3. Mechanistic and Behavioral Insights: What are the critical reasons or cues (e.g., moral, responsibility, or pragmatic) that lead to cooperation or variance? Any ablation-based analysis to support your claim?
4. Figure 2b) Should the arrows "Send decision prompt" and "Return decision (A/B)" reach "Decision LLM"? If not, what is "Decision LLM"?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Large language models are increasingly used for decision support, yet evaluations often overlook how context framing affects apparent rationality. This study presents an evaluation framework that systematically varies key features and procedurally generates vignettes to create diverse scenarios. Comparing decisions across contexts with identical game structures reveals strong context effects on model choices—variability that is largely predictable but highly sensitive to framing. The results highlight the need for dynamic, context-aware evaluation to ensure reliable real-world deployment and offer initial guidance for its design.

### Strengths
- The paper is well written and logically clear. Even readers who are not familiar with this field can understand it.

- The experiments are comprehensive and well-designed. They include multiple models, topics, and metrics, and cover both model behavior and the reasons behind behavior changes.

- For me personally, “How context shapes LLM decision-making” is a very interesting and practical topic. If we want to deploy LLMs widely in real-world production, they will face much more complex contexts than today. Understanding how the model’s decisions change is very important.

### Weaknesses
1. The paper lacks enough technical contribution. The authors demonstrate that context framing has a significant impact on LLM responses in the Prisoner’s Dilemma and advocate for dynamic evaluation strategies. Their evaluation framework (Figure 2) appears to use existing LLMs to generate different contexts. 

2. As the authors note in the limitations, the decision problem studied is simplified. This helps build an analysis framework, but it makes the real-world applicability of the conclusions unclear. Adding more realistic decision experiments (e.g., an agent in a company or bank setting) would help address this.

3. Reasoning models are now an important part of LLMs, but the paper’s evaluation of reasoning models seems very limited and lacks discussion. This makes the work feel incomplete at the current time.


Typo： Line 38 “of the art LLMs. including” → “of the art LLMs, including”

### Questions
See Weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a framework for evaluating how framing affects LLM behavior in strategic decision-making. Specifically, it instantiates a two-player, one-shot Prisoner’s Dilemma, uses Factorial-Survey–inspired procedural vignette generation, and assesses predictability with lightweight feature and embedding models (e.g., XGBoost).

### Strengths
1. **Clear formalization** with a strictly dominant-strategy Prisoner's Dilemma gives a neat normative baseline. 
2. **Factorial Survey to procedural generation pipeline** is articulated and reasonably motivated for stress-testing contextual sensitivity.
3. The paper **reports predictable framing effects and family-level differences**; e.g., correlation between MMLU and defection in some model families.

### Weaknesses
1. **Related work is narrow.** The literature review centers mostly on prior two-player static setups and does not adequately engage with the growing body of multi-player, multi-round/interactive evaluations. This weakens the motivation and overstates novelty relative to current trends.
2. **Limited generality and claims outstrip evidence.** All experiments instantiate a two-player, one-shot, symmetric 2×2 Prisoner’s Dilemma in normal form, with action set {Cooperate, Defect}; models are explicitly told not to consider repeated play. Despite describing the framework as supporting “an arbitrary, user-specified range of scenarios,” no evidence is shown beyond this single game class.
3. **Essentially a robustness/sensitivity study with limited methodological novelty.** The central result, where framing drives behavior; behavior is somewhat predictable—is interesting but familiar. Technically, predictability uses off-the-shelf XGBoost and standard sentence embeddings, with routine grid search. This reads more like a baseline robustness analysis than a new evaluation methodology. 
4. **LLM-based QC is reasonable in principle, but the current pipeline relies on a single judge (GPT-4o) with no human agreement, judge-swap, or sensitivity analyses.** As a result, it is unclear whether findings are robust to reasonable alternatives or partly circular (e.g., upstream biases encoded by the judge).

### Questions
1. If extended to multi-player/multi-round games or to non-dominance structures (e.g., Stag Hunt, Chicken, public-goods, voting), which findings do you expect to persist, and which do you expect to flip?
2. When a model recognizes the Prisoner's Dilemma yet behaves differently across framings, is that due to values, instruction-following vs. rationality trade-offs, or sampling noise? Any discriminative analysis planned?
3. Please detail your QC. As described, QC is LLM-based only, provide judge-swap, threshold sensitivity and generator–judge decoupling, and report coverage metrics and leakage checks for game-structure cues.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel, dynamically generated evaluation framework for probing how narrative context influences large language models (LLMs) in one-shot Prisoner’s Dilemma (PD) games. By procedurally generating thousands of varied vignettes along three axes—Topic (e.g. US politics, business, historical eras), World Type (real vs. imaginary), and Actor Relationship (allies, enemies, neutral)—the authors isolate framing effects while holding the underlying 2×2 PD payoff matrix constant. They evaluate 25 LLMs (including GPT-4o, Claude 3.5, Llama-3.3, etc.), demonstrate significant context-dependent variance in cooperation rates, show that these variances are largely predictable via simple XGBoost classifiers (using either contextual metadata or vignette embeddings), and analyze inter-model agreement, positional biases, and trends with benchmark capability (MMLU-Pro). They argue for more robust, context-aware evaluation protocols and release their vignette-generation code for reproducibility.

### Strengths
1. This paper conducts experiments on a wide range of open- and closed-source LLMs across diverse contexts, providing strong evidence that framing systematically alters model behavior.

2. Interesting topic and findings.

### Weaknesses
1. The paper focuses solely on the one-shot PD. While foundational, it remains unclear how generalizable the methodology and findings are to other strategic or decision-theoretic settings (e.g. coordination games, sequential dilemmas, multi-player interactions).

2. Vignette quality and neutrality are assessed automatically via GPT-4o using a 3-dimensional rubric. This introduces a potential circularity: an LLM judging its own procedural outputs. Reporting a small human validation sample to calibrate and validate rubric thresholds can address this issue.

3. While the paper documents that framing effects arise, it does not deeply probe why certain contexts drive higher cooperation or defection

### Questions
1. Have you experimented (or do you plan to) extending your framework beyond the one-shot Prisoner’s Dilemma to other canonical games (e.g. Stag Hunt, Hawk-Dove, public goods)?

2. The rubric for PD structure, clarity, and bias-neutrality is applied automatically by GPT-4o. Have you conducted any human evaluations to confirm the rubric’s judgments？

3. Your XGBoost models show that topic and actor type are strong predictors of cooperation. Can you illuminate why certain framings sway the models?

### Soundness
2

### Presentation
2

### Contribution
2
