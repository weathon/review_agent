# How Far Are LLMs from Professional Poker Players? Revisiting Game-Theoretic Reasoning with Agentic Tool Use

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
As Large Language Models (LLMs) are increasingly applied in high-stakes domains, their ability to reason strategically under uncertainty becomes critical. Poker provides a rigorous testbed, requiring not only strong actions but also principled, game-theoretic reasoning. In this paper, we conduct a systematic study of LLMs in multiple realistic poker tasks, evaluating both gameplay outcomes and reasoning traces. Our analysis reveals LLMs fail to compete against traditional algorithms and identifies three recurring flaws: reliance on heuristics, factual misunderstandings, and a “knowing–doing” gap where actions diverge from reasoning. An initial attempt with behavior cloning and step-level reinforcement learning improves reasoning style but remains insufficient for accurate game-theoretic play. Motivated by these limitations, we propose ToolPoker, a tool-integrated reasoning framework that combines external solvers for GTO-consistent actions with more precise professional-style explanations. Experiments demonstrate that ToolPoker achieves state-of-the-art gameplay while producing reasoning traces that closely reflect game-theoretic principles.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles poker as a domain for strategic reasoning and analyzes the shortcomings of LLMs on poker. 
The authors examine what features of reasoning correlate with poor performance, finding gaps between action and knowledge and flawed or heuristic reasoning. 
Models are compared to a range of baseline solvers.
To improve models, they propose a training approach consisting of behavior cloning on curated data followed by RL using PPO. They find that this improves performance over the untrained model, but that it still lags behind solvers like CFT. 
To further improve the model, they propose ToolPoker, a method which enables LLMs to recruit external solvers and tools to improve.
They train ToolPoker also via a combination of BC and PPO and find that it further improves and closes much of the gap between models and CFP.

### Strengths
- The paper compares to standard baselines
- games are run across multiple runs 
- both open and closed-source models are considered
- qualitative analysis of reasoning is backed by quantitative results
- The LLM judge is externally validated
- BC+RL results show improvements, with ToolPoker showing further improvements

### Weaknesses
- ToolPoker improvements are not that surprising to me. Basically the gains here seems to boil down to externalizing the parts of the task that are more difficult for the model to external solvers. Implementationally, the training method might be useful, but I don't see what research question it is addressing that hasn't already been addressed by the BC+RL experiments. It seems like in the end, ToolPoker is largely introducing an engineered system that would be of limited use, since it offloads most of the strategic reasoning the model would do to other tools (so no longer getting at the motivating point of the paper on strategic reasoning) while also being costlier and less effective than baseline poker-playing solutions. 
- L083 the claim is made that ToolPoker is for imperfect-information games, but it is only evaluated on poker, and seems very explicitly engineered for poker with limited transfer to other games. 
- The reasoning analysis does not take into account that reasoning might not be faithful. Plenty of prior work has called into question whether reasoning traces from LLMs are faithful explanations of their behavior (see https://aigi.ox.ac.uk/wp-content/uploads/2025/07/Cot_Is_Not_Explainability.pdf for references). The analysis seems to hinge on reasoning being a causal explanation of the model's actions -- if that's not the case, it would explain the knowledge-action gap. 
- if ToolPoker uses CFR solver as a tool, why does it not outperform CFR in Table 5? L436 indicates it's a result of tool-calling errors, is there a way to reduce these? 
- The comparison of tool calling reasoning feels spurious. In this case, the tools are providing a lot of information that was not available to the model without tools.

### Questions
Minor comments:
- Tables 1, 3, 4 are unreadably small

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
This paper investigates the ability of LLMs to perform game-theoretic reasoning in the domain of two-player poker, a challenging incomplete-information game. The authors identify three fundamental reasoning flaws in current LLMs: reliance on shallow heuristics, factual misjudgments, and a marked "knowing-doing" gap between articulated reasoning and actions taken. Following attempts to mitigate these flaws using behavior cloning (BC) and regret-inspired reinforcement learning (RIRL), the authors propose ToolPoker in which LLMs interface with external solvers (e.g., CFR and equity calculators) via a unified tool API. ToolPoker combines imitation and RL to ensure both game-theoretic optimal (GTO) play and reasoning traces aligned with professional-level principles.

### Strengths
1. This paper is well-written and well-organized. The proposed method is simple and easy to follow.
2. This paper presents extensive experimental results with detailed analysis on the ablation study as well as limitations.

### Weaknesses
1. The novelty of the paper is limited. It mainly applies reinforcement learning to a large language model using a classic game-theoretic solver (e.g., CFR+) as the reward signal or direct PPO, without introducing any fundamentally new algorithmic contributions or insights. Essentially, the work repackages standard solver outputs within an RL fine-tuning framework, resulting in incremental rather than conceptual advancement. Also, while the related work section is decent, it omits several works that would be natural inclusions, such as [1]

2. While the core analysis and ToolPoker formulation are well-motivated by Leduc Hold'em and Limit Texas Hold'em, the current instantiation is restricted exclusively to these benchmarks. There is little empirical or conceptual discussion about scalability to larger, multi-player, or variable-rule settings. 

3. The system occasionally produces factual misunderstandings when tool outputs are unavailable or misinterpreted, revealing a continuing challenge for robust factual alignment.

4. Most experiments are performed with synthetic datasets or CFR-solver-based action labels, raising questions about how well ToolPoker would generalize to noisy or inherently human gameplay traces.

Reference:

[1] Fine-Tuning Large Vision-Language Models as Decision-Making Agents via Reinforcement Learning

### Questions
1. Can the authors provide more details on how the LLM-as-a-Judge scores were calibrated and validated? Specifically, are there estimates of both inter-rater LLM agreement and alignment with human judges beyond the 20-trace reference set? How sensitive are reasoning scores in Tables 2 and 4 to the prompt, model, or domain?

2. How does ToolPoker handle non-standard, noisy, or human-style inputs that diverge from solver-generated play? Has the method been evaluated on datasets that originate from human-expert games or crowd-sourced play, and how robust are the results in such settings?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the strategic reasoning capabilities of LLMs in the domain of imperfect-information games, using poker as a rigorous and interpretable benchmark. The authors systematically evaluate several LLMs across realistic poker environments, including Leduc Hold’em and Limit Texas Hold’em, assessing both gameplay outcomes and the quality of reasoning traces. The results show that existing LLMs exhibit three recurring flaws. To address these deficiencies, the authors first test a two-stage internal improvement pipeline—behavior cloning followed by reinforcement learning with step-level rewards. Although this approach yields more coherent, human-like reasoning, it remains insufficient for accurate game-theoretic play. Motivated by these limits, the paper introduces ToolPoker, a TIR framework that enables LLMs to call external poker solvers for GTO actions and quantitative support such as equity and hand ranges. ToolPoker unifies solver interfaces through a single API, constructs an expert-level reasoning dataset augmented with solver outputs, and trains models using a combination of supervised fine-tuning and PPO-based reinforcement learning

### Strengths
- The first systematic study analyzing LLM reasoning and action alignment in poker, identifying fundamental weaknesses in heuristic dependence, factual errors, and knowing–doing gaps.

- A detailed investigation of whether behavior cloning and step-level RL can internally mitigate these flaws, revealing their limited capacity to achieve GTO-consistent reasoning.

- ToolPoker integrates external solvers into LLM reasoning for imperfect-information games, with a unified API and solver-augmented training corpus.

### Weaknesses
1. The composite reward (Eq. 4) combines R_answer, R_format, and R_tool with tunable weights.  How each component quantitatively contributes to tool-learning behavior. Providing ablation or sensitivity analyses—e.g., varying α_f and α_t, or visualizing reward trajectories—would improve transparency and reproducibility. Moreover, discussing how the model avoids reward hacking, e.g., overusing the solver or formatting cues without deeper reasoning, would strengthen the credibility of the RL setup.

2. ToolPoker’s underlying design—LLMs invoking structured external APIs and fine-tuned with PPO—is conceptually close to frameworks like ReAct (Yao et al., 2023), Toolformer (Schick et al., 2023), and ReTool (Feng et al., 2025). The paper would benefit from explicitly contrasting how ToolPoker extends these approaches to imperfect-information and equilibrium-seeking contexts, possibly through a comparative discussion table or controlled ablation with ReTool baselines. Without this, the contribution risks being perceived as an application-specific adaptation rather than a fundamentally new paradigm. 

3. To reach the stated goal of establishing a “principled, general framework for tool-integrated strategic reasoning,” it would benefit from stronger theoretical justification. Why or when the hybrid architecture (LLM reasoning + solver calls) should converge toward a game-theoretic equilibrium, nor how tool-use uncertainty affects strategic optimality.  for instance, linking solver-invocation frequency to bounded rationality or expected regret

### Questions
1. Whether ToolPoker has any formal link to equilibrium convergence or regret minimization theory? Specifically, under what assumptions does the interaction between the LLM’s policy and the external solver guarantee or approximate GTO consistency?

2. How sensitive is the model’s performance to the weighting of the three reward components-answer, format, tool execution? Were any instabilities observed when tuning these hyperparameters?

3. Does ToolPoker include any mechanism to regulate when the solver should be called, or does it always invoke the solver at each decision step? If so, could you analyze cases where solver calls lead to redundant or conflicting information, and whether the model learns adaptive tool-use behavior over time?

4. The paper cites ReTool (Feng et al., 2025) but does not present a direct comparison. Given the conceptual similarity—both integrate external APIs into LLM reasoning—could the authors discuss key differences in design philosophy or experimental setup?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper uses poker to investigate game-theoretic reasoning in LLMs. It first compares vanilla LLM performance against several traditional baselines for game-theoretic reasoning, finding that LLMs noticeably underperform these baselines. An approach for improving LLM performance on poker is then introduced, with two main components: (1) behavior cloning, which consists of fine-tuning the LLM on expert-level trajectories augmented with reasoning traces and (2) RL fine tuning. This approach fails on its own, but these two high-level components are incorporated into the paper's last main contribution, ToolPoker, which allows the LLM to incorporate calls to external poker solvers into its reasoning trace (and is trained on a similar combination of behavior cloning and RL fine-tuning).

### Strengths
The paper is very clear in its presentation and generally high in quality: each newly introduced approach built on the last in a way that made the paper particularly easy to follow and made the motivation for the different components of the ToolPoker system apparent. The main points of significance and originality for the paper are that (1) it evaluates how LLMs reason about poker and presents qualitative and quantitative analyses of particular types of shortcomings in the reasoning process of vanilla LLMs and (2) applies a tool use framework to the poker setting.

### Weaknesses
The primary weakness of this paper lies in the novelty of the approach: while the paper does a very good job analyzing LLM performance on poker and explaining why the ToolPoker approach was developed, it is not clear if there is anything that sets ToolPoker apart from other tool-use frameworks, other than the task setting. In particular, explicitly comparing the strengths of ToolPoker with other approaches like ReTool (mentioned in the paper) would be helpful in evaluating the approach.

### Questions
1. What sets ToolPoker apart from other tool-use approaches?
2. Can ToolPoker be generalized to other imperfect-information games? What changes might have to be made (other than the tools called)?

### Soundness
3

### Presentation
4

### Contribution
2
