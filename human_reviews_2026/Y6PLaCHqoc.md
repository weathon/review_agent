# Deceive, Detect, and Disclose: Large Language Models Playing Mini-Mafia

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Mafia is a social deduction game where informed mafia compete against uninformed townsfolk. Its asymmetry of information and reliance on theory-of-mind reasoning mirror real-world multi-agent scenarios, making it a useful testbed for evaluating the social intelligence of large language models (LLMs). To support a systematic study, we introduce *Mini-Mafia*: a simplified four-player variant with one mafioso, one detective and two villagers. We set the mafioso to kill a villager and the detective to investigate the mafioso during the night, reducing the game to a single day phase of discussion and voting. Remarkably, we find that the mafia win-rate $p$ in this three-agent system can be described by a simple theoretical model: $\text{logit}(p) = v \times (m - d)$, where $m$, $d$, and $v$ are intrinsic model parameters representing the mafioso deceive, the villager detection , and the detective disclosure capabilities, respectively. This model successfully predict any game combination outcome from intrinsic model parameters. Estimating these parameters from LLM gameplay data using Bayesian inference creates the *Mini-Mafia Benchmark*. Our experiments reveal counterintuitive results, including cases where smaller models significantly outperform larger ones. We also establish human baselines performance, revealing that LLMs excel at persuasive communication but lag in strategic reasoning for agentic interaction. Beyond benchmarking, Mini-Mafia enables quantitative study of emergent multi-agent dynamics such as name bias and last-speaker advantage, and contributes to AI safety by generating training data for deception detectors.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Mini-Mafia, a simplified four-player version of the social deduction game Mafia, as a benchmark for evaluating LLMs on three interactive capabilities: deception, deception detection, and strategic information disclosure. The authors design controlled multi-agent experiments where LLMs play fixed roles (mafioso, detective, villager) in structured discussions, isolating each capability via role-specific win conditions. A standardized two-stage evaluation procedure estimates win rates and aggregates model performance across backgrounds. Results show smaller models sometimes outperform larger ones, and reveal social phenomena such as name bias and last-speaker advantage. The authors suggest the benchmark also aids AI safety research by generating data for deception detection and tracking deceptive tendencies in LLMs.

### Strengths
1. Well-structured experimental framework isolating interactive capabilities under controlled conditions.
2. Additional analyses (e.g., name bias, last-speaker advantage) provide meaningful insights into emergent social dynamics.

### Weaknesses
1. Prior works have already explored deception games and benchmarks, such as Avalon and Werewolf. However, the paper does not clearly explain how Mini-Mafia differs from these existing setups or why this new variant is needed. The motivation for introducing a separate benchmark remains unconvincing.
2. Since the ultimate objective of such research is to enable LLMs to interact effectively with humans, the absence of any human-involved experiments significantly limits the benchmark’s practical relevance.
3. The setup may oversimplify real-world social reasoning, only one discussion round and fixed night actions limit depth of interaction.
4. Benchmark outcomes (win rates) depend heavily on prompt phrasing and turn order, which may confound interpretability.
5. No ablation or sensitivity study to confirm robustness against prompt wording or memory design.

### Questions
1. How does Mini-Mafia differ from existing deception-game benchmarks such as Avalon or Werewolf, and why is a new variant necessary?
2. If the ultimate goal is to enable LLMs to interact effectively with humans, why are there no human-involved experiments to validate the benchmark’s real-world relevance?
3. To what extent do prompt phrasing and turn order influence the benchmark’s outcomes?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes Mini-Mafia, a simplified version of the Mafia game, to evaluate the ability of LLMs to deceive, detect deception, and disclose information. These three skills directly map to the characters in the game. At a single instance, models are assigned a role in the game, play against each other and the win rates are the aggregated across multiple game configurations. The paper also reports interesting phenomena such as name bias and last-speaker advantage.

### Strengths
The mini-mafia benchmark provides a structures and reproducible way of measuring the three skills deceive, detect, and disclose with a clear mapping to the roles of the game. A large number of 100 games are run for each configuration and the analysis via hierarchical bayesian approaches is rigorous.

The emergent results of name bias and last-speaker advantage are interesting observations. The full prompts used to run the game are provided which helps reproducibility. The paper is overall well written and clear.

### Weaknesses
My primary concerns are the limited scale of the environment, applicability of the benchmark to the same skills in real-world scenarios and scope of the evaluation.

**Limited scale of the environment:**

The environment is limited to only 4 players and a single day phase (the night phase is fixed). This simplifies the social dynamics of the mafia game a lot. Extending the benchmark to multiple day-night phases, with a non fixed night phase and also scaling the number of players would significantly strengthen the paper. These extensions would allow for longer term strategies and more interesting dynamics.

**Real-world applicability of the benchmark:**

I am doubtful whether the insights obtained regarding the skills deceive, detect, and disclose can be transferred to any real-world scenarios. The current version of the game seems to oversimplify the complex dynamics and role descriptions that may be encountered in real-world tasks. This also raises another question whether win rates in the mafia game can be interpreted as the model's ability to deceive / detect / disclose?

**Evaluation scope:**

A size scaling study within multiple families would be helpful to further understand the correlation between size and the evaluated skills. Experiments scaling the number of discussion steps would be very interesting (and as mentioned before number of agents and day/night phases).

**Other points:**

* I liked section 4 with broader trends that were observed. However, section 3 mostly describes the tables and figures on a high level rather than providing a deeper analysis.
* Is the name bias really significant? The differences seem to be small.
* While the last-speaker advantage is an interesting observation, I find it too strong to conclude that procedural elements significantly impact social outcomes.
* How are the different backgrounds affecting the results? Across three different skills three different backgrounds were shown which makes it harder to compare.
* Typo: line 325 "Compare..."

Overall, I think this work is promising, but needs some more work (especially regarding the scale of the environment).

### Questions
I would be happy to increase my score if the authors can address some of the following points:
* How would results change if multiple day–night cycles were allowed, or if the detective could choose not to disclose immediately?
* How would results be affected by a longer discussion phase during the day?
* Why was the number of agents limited to 4? Have the authors tried scaling up?
* Could the benchmark be extended to cooperative dynamics (e.g., villagers coordinating or sharing partial information)? 
* Could the authors show correlations between model size, reasoning style, and success rates?
* Regarding the last-speaker advantage: can it be said that the more information the model has the larger the last-speaker advantage?

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
The paper introduces an evaluation framework for assessing LLMs’ abilities to deceive, detect deception, and disclose truthful information using a minimal, single night setting of a Mafia game. The conversation are between one mafioso who has killed a villager, one detective who identifies the mafioso, and a villager, who decides which of the two to believe.

The authors evaluate 10 candidate models against 5 background models, with the candidate model as one role and background models as the other two. Each candidate–background pairing is rolled out 100 times, and win rates are estimated via Bayesian inference, yielding corresponding means and variances. Model performance is then aggregated by computing z-scores across backgrounds and candidates.

The results show that no single model dominates all dimensions and small models sometimes outperform bigger models. Additionally, the authors discovered biases in  name and gender bias for trust attribution, as well as a more pronounced advantage for the last speaker

### Strengths
I like the paper’s novel approach of reducing the Mafia game simulation to a minimal, single-night conversation. Most existing social-deduction-based benchmarks focus on realism and implement multiple turns with more partners and roles, and models performances are estimated through aggregate outcomes over games. While such setups capture richer dynamics, they often reduce interpretability and make it difficult to isolate model abilities from their partners. However, this minimal Mafia setting offers a more interpretable and controlled environment, enabling a cleaner and simpler analysis of each model’s capabilities.

I also like the evaluation methodology, which is enabled by this simplified setup. By aggregating z-scores across background models, the authors provide a simple and unbiased method to estimate a model’s ability in a given role. This approach is practical and accessible for the community to reproduce and extend similar analyses.

The paper is also fluent and well structured, and ideas are well communicated.

### Weaknesses
First, while the mini-Mafia setup is conceptually sound, it is overly simplified and provides limited coverage of realistic game dynamics. As a result, it offers shallow insight into the deception, detection, and disclosure abilities the paper aims to evaluate. Because the setting only covers the first night, each role has minimal context to leverage for either deception or detection. This leads to short, repetitive, and low-content interactions, as reflected in several examples shown by the authors. A key challenge of deception lies in maintaining consistency with and taking advantage of prior facts and statements, which this setup does not capture. Similarly, deception detection and truthful disclosure depend heavily on contextual and temporal reasoning, which are largely absent here. To make the evaluation informative of modern LLM enabled dialogue system, the framework should at least include additional background scenarios or multi-turn extensions to enable richer interaction histories.

Second, the paper lacks semantic analysis explaining why certain models perform better at deception, detection, or disclosure. A more insightful analysis could involve categorizing persuasion or deception strategies (e.g., exploiting communication gaps as noted in Section 3.1) and examining their distribution across models. It would also strengthen the work to analyze game dynamics, such as how strategies evolve or how the villager’s trust shifts over discussion.

Third, the presentation could use more illustrative graphics and visualizations, which would make the framework and findings easier to understand and provide richer analysis.

### Questions
1. Beyond interpretability, what do you see as the main advantages of your framework compared to more comprehensive, realistic social-deduction benchmarks?
2. Given the minimal context available to each role, what factors do you think make a model appear more persuasive in this setting? From the examples, most persuasion strategies seem limited to direct self-identification, while villagers’ reasoning appears biased by game constraints (e.g., questioning why the detective investigated the mafioso—a system-determined action). How do you plan to analyze or mitigate these biases and encourage more diverse persuasion strategies?
3. As I mentioned in weaknesses, I suggest the framework to include additional background scenarios or multi-turn extensions to enable richer interaction histories. The authors could sample more conversation trajectories, find interesting middle points that require more complex deception/persuasion/detection abilities as additional starting points of the discussions.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new benchmark based on the social deduction game Mafia. The benchmark is a simplified 4-player variant with fixed night actions that reduce the game to a single day-time discussion and voting phase. The benchmark is designed to measure three capabilities, the ability of an LLM to: deceive the other players, detect the hidden player (mafioso), and disclose the hidden player. These capabilities are evaluated across 10 models, with the other players fixed to another model (out of 5 possible "background" models). Results are shown which highlight some interesting results, e.g. last word advantage.

### Strengths
* I appreciate trying to distill a complex social reasoning game into a controlled, repeatable benchmark. I think this is a valuable goal, and while I'm not necessarily convinced that this is the solution to get there, I do think there is value in exploring this direction.

* The main empirical results are pretty thorough across target/background models. The full results in Appendix C show that a significant amount of effort went into running these matchups. I also like the identification of some of the emerging phenomena. While I'm not totally convinced of their rigor (see weakness below), any good benchmark paper benefits from highlighting things like this.

### Weaknesses
* The simplification of Mafia to a single day-time discussion eliminates much of the social complexity that was useful in the original game. Is there really much opportunity for deception or detection outside of the most simplistic behaviors ("No, Bob did it!"), when players don't have the ability to reason about prior events or behaviors? I am not convinced on the utility of what is actually being measured here, and think the game is *too* simple.

* These are all relative scores, and there's no way to measure an absolute baseline, e.g. human player performance.

* The "serendipitous results" (sec. 4) are not explored very thoroughly. In particular, the name bias is only present for one player (Diana) and the example given in the paper is not very convincing. That example seems like it would more likely be an example of last word advantage. Similarly, last word advantage may be related to sycophancy, but no further investigation is done here. All in all, I think there's a lack of interesting results/analysis here. What do we really learn from this benchmark? Why should other people in the community use it?

### Questions
1. In the example in Sec. 3.2, why would we assume this is name bias instead of last word advantage?

2. Is last word advantage associated with sycophancy? Do the other models just listen more closely to the last model which "spoke"?

### Soundness
2

### Presentation
2

### Contribution
2
