# CATArena:  Evaluation of LLM Agents Through Iterative Tournament Competitions

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Large Language Model (LLM) agents have evolved from basic text generation to autonomously completing complex tasks through interaction with external tools. However, current benchmarks mainly assess end-to-end performance in fixed scenarios, restricting evaluation to specific skills and suffering from score saturation and growing dependence on expert annotation as agent capabilities improve. In this work, we emphasize the importance of learning ability, including both self-improvement and peer-learning, as a core driver for agent evolution toward human-level intelligence. We propose an iterative, competitive peer-learning framework, which allows agents to refine and optimize their strategies through repeated interactions and feedback, thereby systematically evaluating their learning capabilities. To address the score saturation issue in current benchmarks, we introduce CATArena, a tournament-style evaluation platform featuring four diverse board and card games with open-ended scoring. By providing tasks without explicit upper score limits, CATArena enables continuous and dynamic evaluation of rapidly advancing agent capabilities. Experimental results and analyses involving both minimal and commercial code agents demonstrate that CATArena provides reliable, stable, and scalable benchmarking for core agent abilities, particularly learning ability and strategy coding.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper *CATArena: Evaluation of LLM Agents through Iterative Tournament Competition* introduces a peer-learning-based evaluation framework for large language model (LLM) agents. It proposes an iterative tournament setting where agents compete, analyze each other’s strategies, and improve their code across multiple rounds. Each iteration provides agents with access to prior match logs and opponent code, allowing them to revise their strategies autonomously. This design aims to measure learning and adaptation beyond static task performance.

CATArena evaluates agents along three main dimensions: **strategy coding ability**, which measures initial code quality; **learning ability**, which quantifies improvement over successive rounds; and **generalizability**, which tests performance under modified game rules. The framework is implemented across four environments—Gomoku, Chess, Texas Hold’em, and Bridge—chosen for their structured yet diverse characteristics. Agents based on different LLMs, including GPT, Claude, Gemini, and Qwen, participate in multi-round tournaments.

### Strengths
The paper introduces an innovative iterative peer-learning tournament framework that evaluates LLM agents through multi-round competitive interactions. This dynamic setup goes beyond static benchmarks and allows the measurement of agents’ continuous learning and adaptation capabilities. Unlike traditional single-round evaluations, CATArena explicitly quantifies learning ability, strategy improvement, and generalization. It formalizes learning metrics such as global learning and counter-adaptation, enabling fine-grained tracking of performance progress across rounds. The framework naturally avoids score saturation, a common issue in static benchmarks. Its iterative structure allows performance ceilings to shift as agents evolve, supporting long-term benchmarking for advancing LLM agent capabilities.

### Weaknesses
1. The evaluation scope is somewhat limited: the benchmark focuses on variants of known games, but the differences from existing games are not substantial. Therefore, these tasks cannot be considered fully out-of-distribution (OOD), and they remain far from real-world applications. Please clarify whether the evaluation results of this benchmark reflect the general capability of the models or merely their domain-specific performance within the gaming context.

2. The multi-round learning setup involves long contextual inputs. I suspect that the performance degradation observed in some models during later rounds may be related to their long-context processing limitations. Please provide information on the context length for each round and explain how you ensured that models or agents were not affected by context length bias during evaluation.

3. It is unclear whether score changes across rounds genuinely result from models’ understanding of opponents’ strategies. The current paper does not directly demonstrate this. Could you provide a case study analyzing concrete examples of model output modifications or learning patterns that illustrate how the agents adapted their strategies?

4. Most experiments in the paper were conducted for only four iterations, with a limited number of matches per round, making it difficult to observe long-term learning trends or performance saturation points. Could you provide additional results showing model performance over longer iterations? If computational cost is a concern, this could be done using a smaller subset of stronger models.

5. The current testing framework seems unable to support adding new models independently. If I want to evaluate my own model, would it be necessary to rerun the entire experiment from scratch?

### Questions
The questions are already included within the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CATArena, a tournament-style benchmark for evaluating LLM agents through iterative peer-learning competitions. The benchmark includes four diverse games (Gomoku, Texas Hold'em, Bridge, and Chess) with variants, providing open-ended ranking tasks without explicit upper bounds. CATArena defines metrics for strategy coding, learning ability (global learning, counter-adaptation, self-improvement), and generalizability. Experiments compare 6 minimal agents (LLM + ADK framework) and 5 commercial code agents under both standard and variant game rules.

### Strengths
S1. Exploring LLM performance in competitive environments involving reasoning and multi-round iterations is a meaningful research problem. Seeking more specific metrics helps improve our understanding of LLM capabilities.
S2. The paper selects four games with unlimited improvement space (Gomoku, Texas Hold'em, Bridge, Chess) and introduces game variants to reduce memorization dependence.
S3. The experimental design and analysis are relatively comprehensive, including: (1) comparison between minimal agents and commercial agents; (2) testing under standard and variant rules; (3) 4 repeated experiments to assess stability; (4) ablation studies (ML track, multi-lingual track, comparison with LLM-Player).

### Weaknesses
W1. The paper's core contribution is evaluating LLM agents' "learning ability," but the definition and measurement validity of this concept lack adequate justification: Since LLM parameters are fixed, the exhibited "learning" is essentially strategy adjustment through historical information in context (in-context learning), which fundamentally differs from human parameter-update-based learning. The paper does not clearly distinguish these two concepts of "learning." 
W2.  The three learning metrics defined on page 5 (global learning Li, counter-adaptation Ci, self-improvement SIi) are mainly based on score differences and correlation coefficients, but why can these mathematical forms effectively capture "learning ability"? For example, Ci only compares the difference between An_i and Bn-1_i, which may be heavily influenced by game randomness. Additionally, the self-improvement metric SIi uses Pearson correlation, but with N=4 rounds, the sample size is extremely small (only 4 points), making the correlation coefficient statistically very weak and easily affected by noise.
W3. The paper claims on page 8 (Figure 3 and related text) that agents' strategy coding ability fundamentally differs from LLMs' direct reasoning ability, but the evidence is insufficient: The paper supports this by comparing action consistency between agent code and LLM-Player in endgame states, but low consistency could have multiple causes (such as code implementation bugs, different reasoning depths in LLM-Player), which cannot directly lead to the conclusion that "strategy coding is an independent ability."
W4. The paper states on page 10 that it will provide anonymous links in the future, but as a benchmark evaluation work, complete open-sourcing is crucial for effective review. Appendix D mentions that "agents sometimes fail to generate runnable code," but does not report the failure rate for each agent or explain how these failures are handled.

### Questions
Q1. Given that LLM agent parameters are fixed, their cross-round "learning" is essentially in-context learning through extended context windows. How do you distinguish this type of "learning" from gradient-based parameter update learning? Should this evaluation framework be more appropriately described as "contextual adaptation ability" rather than "learning ability"? Furthermore, the decomposition of learning ability into global learning, counter-adaptation, and self-improvement—what is the theoretical basis for this decomposition? What are the relationships and independence among these three dimensions?
Q2. The paper repeatedly claims the framework has "scalability" (in abstract and conclusion), but what specifically does this refer to?
Q3. When computing average rankings in Table 3, are all games and all ability dimensions given equal weight? What is the rationale for this choice?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The work enables LLMs to play complex card games by generating strategy code or directly outputing game moves, and evaluates them through pairwise comparison and batch competition, which together determine each model’s score in a tournament setting. In the first stage, the focus is on assessing the LLM’s capability for strategy coding. In the second stage, the evaluation centers on the LLM’s learning ability, which is further decomposed into global learning, counter adaptation, and self-improvement. The paper conducts a comprehensive comparison and analysis, leading to a series of findings, such as that Claude-4-Sonnet achieves the highest score among the minimal-agent settings.

### Strengths
1. The study performs multi-round dynamic evaluations, providing the LLM with access to historical game records. This design facilitates the assessment of the LLM’s learning, reflection, and evolution capabilities, making the evaluation more convincing than traditional single-round static assessments.
2. A novel metric computation framework is established to independently measure the LLM’s capabilities in global learning, counter adaptation, and self-improvement, rather than conflating these aspects into a single aggregated metric.

### Weaknesses
1. The tournament framework essentially remains an extension of the LLM Arena evaluation paradigm, with limited methodological innovation. It is therefore unclear what the substantive difference or novelty is between the proposed tournament setup and the previous pairwise blind comparison used in LLM Arena.
2. The comparative validity (i.e., the meaningfulness of comparison) across the three categories—Minimal Agents, Code Agents, and LLM Players—is limited, since their ways of participation differ substantially (e.g., generating executable code vs. producing game moves) and they have unequal access to tools and capabilities.
3. Overall, the range of evaluated LLMs is relatively narrow. The main results do not include different versions of the same model family, such as GPT-o1, Codex, GPT-5, or GPT-4o, which limits the comprehensiveness of the evaluation.

### Questions
1. What is the rationale for mixing the two modes of participation: (1) using LLMs to generate code that plays complex card games, and (2) allowing LLMs to directly output game moves? Does the work primarily aim to evaluate the LLM’s code generation ability, its capacity to understand game rules and make decisions, or its overall integrated competence?
2. Why was a card game chosen as the environment for the tournament? Given the inherent randomness in card games, why not adopt games with less stochasticity, such as board games or e-sports, for a more controlled and reproducible evaluation?
3. If a new LLM is to be included in the evaluation, does the entire tournament need to be rerun? Would this process be computationally or procedurally cumbersome?

### Soundness
3

### Presentation
3

### Contribution
2
