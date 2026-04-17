# GTA1: GUI Test-time Scaling Agent

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Graphical user interface (GUI) agents autonomously complete tasks across platforms (\eg, Linux) by sequentially decomposing user instructions into action proposals that iteratively interact with visual elements in the evolving environment. However, two main challenges arise: i) planning (\ie, the action proposal sequence) under expansive action space, where selecting an appropriate plan is non-trivial, as many valid ones may exist; ii) accurately grounding actions in complex and high-resolution interfaces, \ie, precisely interacting with visual targets. This paper investigates the aforementioned challenges with our \textbf{G}UI \textbf{T}est-time Scaling \textbf{A}gent, namely \name. First, we conduct test-time scaling to select the most appropriate action proposal: at each step, multiple candidate proposals are sampled and evaluated and selected by a judge model. It trades off computation for better decision quality by concurrent sampling. Second, we propose a model that improves grounding of the selected action proposals to its corresponding visual elements. Our key insight is that reinforcement learning (RL) facilitates grounding through inherent objective alignments, rewarding successful clicks on interface elements. Experimentally, \name achieves state-of-the-art performance on both grounding and agent task execution benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces the GUI Testing-time Scaling Agent (GTA1), which aims to solve challenges in planning and action localization for GUI agents. Through a testing-time scaling strategy, it samples multiple candidate action proposals at each step, from which a judgment model selects the optimal solution. This is combined with a reinforcement learning-based localization model that directly predicts interaction coordinates and is optimized via click rewards, leading to effective performance improvement.
Experimental results show that GTA1 achieves state-of-the-art performance on multiple benchmarks. For instance, on the ScreenSpot-Pro benchmark, the 7B model reached an accuracy of 50.1%, surpassing UGround-72B's 34.5%. On the OSWorld benchmark, GTA1-7B achieved a task success rate of 45.2%, outperforming all compared methods. These results demonstrate the effectiveness and robustness of the proposed method.

### Strengths
1.GTA1 effectively addresses the planning and localization challenges for GUI agents through a testing-time scaling strategy and a reinforcement learning-based localization model, achieving state-of-the-art performance across multiple benchmarks.

2.Extensive experiments have validated the effectiveness and generalizability of the method.

### Weaknesses
It is noted in the manuscript that the foundational model for GTA1-7B is UI-TARS-1.5-7B, while for GTA1-32B it is OpenCUA-32B. This raises the question of how the performance would be affected if qwen2.5-vl or qwen3-vl were used as the base models instead. Further experimentation is required to exhibit the superior generalization capabilities of the proposed methodology.

### Questions
1.I have observed that the training data originates from Aria-UI-Web, OmniACT, UI Vision, Widget Caption, and OSAltas-Desktop. Could you elaborate on the rationale for selecting these particular datasets? Furthermore, what considerations were made in terms of the volume and heterogeneity of the data?

2.it is stated that the training dataset underwent a filtering process based on the quality of bounding boxes. Could you provide the specific quantities of the dataset both prior to and subsequent to this filtering procedure?

3.This manuscript evaluates performance on benchmarks such as OS-World and WindowsAgentArena. What was the reasoning for not incorporating agent data into the training set? Additionally, could you compare the potential efficacy of a unified model that performs both planning and grounding, versus the current methodology which involves test-time scaling for planning followed by a separate grounding step?

4.It is mentioned that 70k data samples were utilized for training. Is it anticipated that performance would continue to enhance with a further increase in the scale of the data? I wonder if there is a scaling law here?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
GTA1 is a GUI agent that improves both plan selection and precise clicking on complex screens. At each step, it samples multiple action proposals and uses a judge model to pick the best one, trading extra compute for better decisions. For grounding, it trains a simple RL model that directly predicts click coordinates with a click reward, achieving SOTA results on ScreenSpot-Pro, ScreenSpot-V2, and OSWorld-G, and strong task success on OSWorld and WindowsAgentArena. The key contributions are test-time scaling for robust planning, a minimal RL recipe for grounding, and an analysis showing “thinking” isn’t needed in static UIs but can help in dynamic ones.

### Strengths
1. The paper tackles two concrete pain points—plan selection and precise grounding, and pairs them with two simple fixes: test-time scaling for planning and a minimal RL recipe for click grounding. The framing is clear and the approach maps tightly to the problems.

2. Test-time scaling actually pays off. Sampling K proposals per step and using a judge consistently boosts success while also cutting wall-clock via concurrent sampling.

3. Directly rewarding clicks inside the target element keeps the training signal simple and on-task, yet reaches SOTA on ScreenSpot-Pro, ScreenSpot-V2, and OSWorld-G. The simplicity is a strength.

4. The paper shows that explicit reasoning isn’t needed for static UIs and can even hurt, while helping in dynamic settings when trajectories/objectives are involved. This nuance helps practitioners know when to spend tokens on reasoning.

5. Results cover both grounding and full agent task execution (OSWorld and WindowsAgentArena), and scaling tests generalize across agents, horizons, and K settings, with qualitative analyses of cascading failures. The breadth makes the claims more convincing.

### Weaknesses
1. Test-time scaling samples K proposals each step and uses a judge, which can raise token and inference costs even if concurrent sampling cuts wall-clock time. The paper shows speedups and success gains but does not report detailed cost curves (tokens, latency) across K and horizons.

2.  The paper observes little average gain from “thinking” and attributes sample-wise differences to training instability. “Thinking helps only in dynamic UIs” needs stronger controls. The authors can test different “thinking” budgets, report variance to separate instability from genuine reasoning effects, and clarify when trajectories/objectives trigger reliable wins.

### Questions
1. Experiments use o3/GPT-5 as planners and a multimodal LLM judge. Can you test how performance transfers to an open-source stack?

2. Could you report the judge’s pairwise selection accuracy and show how judge errors propagate to end-task failure?

3. Can you share success–vs–cost curves (tokens, latency, $) across K and horizon, and results for adaptive policies (early exit, dynamic K per step)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a GUI agent that works in two stages: planning and grounding. The proposed solution improves each stage differently:
1. the planning stage through test-time scaling: sampling K times at each step and asking a judge to select the "best" one.
2. the grounding stage by fine-tuning a model using GRPO

### Strengths
- strong empirical results on relevant benchmarks
- simple method
- ablations and discussion on the efficacy of the different reward signals (for thinking, for location)

### Weaknesses
- the dataset curation step should be part of the "algorithm", otherwise the comparison with other methods is not 
- reduced novelty
- no ablation to understand the improvement brought by each of the two components in the agent task execution scenarios

### Questions
How is the training dataset built from the collection mentioned in the paper?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper’s main contributions are (i) a test-time scaling strategy that selects the next action and (ii) a GRPO-trained GTA model for coordinate grounding. Specifically, at each step, the planner samples multiple candidate actions, and a multimodal LLM judge selects the best one. For actions requiring coordinates, a GTA model trained with GRPO provides more precise grounding. The GTA grounding model achieves SOTA on grounding benchmarks (ScreenSpot-Pro, ScreenSpot-V2, OSWorld-G). On real OSWorld tasks, o3 with test-time scaling + GTA1-7B grounding attains a 45.2% success rate with a 100-step cap, surpassing CUA o3 at 42.9% with a 200-step cap.

### Strengths
- Strong empirical results across both grounding datasets and interactive environments, reaching SOTA on multiple benchmarks.
- The two-stage (planning+grounding) framework makes it straightforward to switch Planner/Judge modules or upgrade the grounding model.
- The light-weight data cleaning pipeline can be integrated with existing data pipelines.

### Weaknesses
- The description of test-time scaling is under-specified. For example, when the judge "picks the best candidate", is the output a score over candidates, or a rewritten action? Some results (e.g., Table 1) could be simplified to improve readability by moving some baselines to the appendix.
- Evaluations of test-time scaling (TTS) focus on ablations against the same agent without TTS. At least there should be equal-compute comparisons (e.g., self-consistency / majority voting / alternative sampling strategies).
- There are no confidence intervals. Reported gains over an OpenAI o3 baseline are ~2 points (Table 4). Without multiple runs and confidence intervals, statistical significance is unclear.
- The results of larger models are missing. GTA1-7B with GPT-5 and 32B configurations are "unavailable". If this is due to API cost, a proxy study (e.g., replaying 7B trajectories with a 32B grounding model) would strengthen the insights of the experiments. Likewise, results for open-source planners with test-time scaling + GTA grounding would be valuable.
- The paper appears to lack OSWorld ablations isolating the impact of the GTA grounding model.
- The proposed method largely adapts known ideas. Test-time scaling here is similar to best-of-N, and the GTA model is based on GRPO + data cleaning + reward shaping. It is reasonable that these techniques can boost performance, so more analysis and insights are required in the experiments (see suggestions above).

### Questions
- Under matched compute, how much does your test-time scaling improve over other test-time strategies?
- Does the judge score candidates or generate a new action, and how sensitive are results to this choice?
- If you increase the number of action proposals beyond 32, do gains continue?
- What is the performance delta on OSWorld with and without the grounding model? Please provide ablations quantifying its impact.
- How do open-source models perform under your test-time scaling, with and without the GTA grounding model? It would help to report accuracy-cost curves and confidence intervals for these settings.

### Soundness
3

### Presentation
2

### Contribution
2
