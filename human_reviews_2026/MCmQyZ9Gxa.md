# CubeBench: Diagnosing Interactive, Long-Horizon Physical Intelligence under Partial Observations

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4, 6

## Abstract
Large Language Model (LLM) agents, while proficient in the digital realm, face a significant gap in physical-world deployment due to the challenge of forming and maintaining a robust spatial mental model. We identify three core cognitive challenges hindering this transition: spatial reasoning, long-horizon state tracking via mental simulation, and active exploration under partial observation. To isolate and evaluate these faculties, we introduce \textbf{CubeBench}, a novel generative benchmark centered on the Rubik's Cube. CubeBench uses a three-tiered diagnostic framework that progressively assesses agent capabilities, from foundational state tracking with full symbolic information to active exploration with only partial visual data. Our experiments on leading LLMs reveal critical limitations, including a uniform 0.00\% pass rate on all long-horizon tasks, exposing a fundamental failure in long-term planning. We also propose a diagnostic framework to isolate these cognitive bottlenecks by providing external solver tools. By analyzing the failure modes, we provide key insights to guide the development of more physically-grounded intelligent agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces an interesting benchmark based on Rubik's Cube for evaluating the capability of LLMs. The authors introduced three tracks of tasks: full symbolic, visual, and partial visual. The authors have conducted solid evaluations of LLMs and have also discussed some insightful findings.

### Strengths
- The benchmark is useful in evaluating long-horizon, interactive, and spatial reasoning capabilities of LLMs.
- The presentation of this paper is good, with clear figures and well-written paragraphs.
- The evaluation of different LLMs seems to be extensive.

### Weaknesses
1. Lacks performance comparison to classical methods.
Unlike ARC [1], I think we should have some existing symbolic tools to solve the Rubik's Cube (correct me if I am wrong). What's the performance of them in this benchmark? And "why" do we need to consider the problem of "LLMs can solve them or not" if symbolic solvers are already very good at this task? (as against in ARC, we don't have performant classical program synthesis methods while human can easily achieve >90%). 

2. I am a bit confused by the visual and partial visual tier tasks.
- For visual tasks, if the image is simply generated with colored grids without any noise, isn't it the same as a symbolic state? As we won't have a grounding issue like discussed in LogiCity [2].
- For partial visual tasks, I think the agent can easily get the full state by simply turning the columns one by one and rotating them back to the init state, right? Since the Rubik's Cube is deterministic, if LLMs do this, isn't this the same as full-observability? Usually, POMDP is hard because it is very hard/impossible/very inefficient for the agent to get full states, like in LogiCity [2], so I am a bit confused by the setting here. Why is the full state hard to observe?

I will consider raising my score if these two concerns are resolved during rebuttal.

[1] Chollet, Francois, et al. "Arc prize 2024: Technical report." arXiv preprint arXiv:2412.04604 (2024).
[2] Li, Bowen, et al. "LogiCity: Advancing neuro-symbolic ai with abstract urban simulation." Advances in Neural Information Processing Systems 37 (2024): 69840-69864.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

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
The paper proposed an benchmark named CubeBench. It is a Rubik’s-cube benchmark for LLMs that probes three weak spots:
- 3-D spatial reasoning
- Long-horizon state tracking
- Active exploration under partial views

It offers three input tiers: (1) full 54-char state, (2) 2-D unfolded map, (3) sparse local views. The authors also try to give dense rewards and use an external solver.

Tests on 15+ top models show: everyone fails long scrambles, accuracy tanks when switching from symbols to visuals, dense rewards help only tiny tasks, and even with a solver models can’t reliably rebuild the cube. All code and task generators are open-sourced.

### Strengths
1.  An interesting task that can test  LLM's spatial reasoning, long horizon, and expoloration abilities. 
2.  Large-scale comparison across 15+ proprietary/open LLMs yields actionable insights for the community.
3.  Inclusion of dense-reward variants and solver tools makes the dataset usable for both RL and tool-augmented-agent research.

### Weaknesses
1.  Scope limited to a single puzzle; conclusions may not generalize to broader physical reasoning tasks which can be really useful in the real world.

### Questions
If an LLM/VLM works well on CubeBench, can it translate into tangible gains on real-world 3-D embodied tasks (e.g., physical-robot manipulation)? Or can you explain the correlation between CubeBench scores and actual physical-world capabilities?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces CubeBench, a benchmark designed to evaluate large language model (LLM) agents on spatial and long-horizon reasoning tasks under varying levels of observability. Centered on a Rubik’s Cube environment, CubeBench presents tasks of increasing complexity, defined by the cube’s optimal distance from the solved state. The benchmark comprises two main settings: a fully observable mode, where the agent has access to complete symbolic or visual information, and a partially observable mode, limited to local face or vertex views. The authors further explore how providing dense rewards and external solver tools affects performance, using these interventions to diagnose how effectively agents can represent, aggregate, and reason over spatial information across time.

### Strengths
- The choice of the Rubik’s Cube environment is well-motivated, offering a compact yet challenging testbed for spatial reasoning and long-horizon planning, while also enabling automatic verification of solutions.
 - The experimental evaluation is comprehensive, featuring multiple ablations across agent types and settings. The results clearly expose consistent failure modes—particularly in the vertex-level partial observation case—even when solver tools are available, while also highlighting interesting cases where agents learn to use tools in-context.
 - The paper is well-structured and clearly written, making the motivation, methodology, and results easy to follow.

### Weaknesses
-  The benchmark does not exclusively measure an agent’s intrinsic ability to mentally simulate and plan within a spatial model, since it permits writing code that performs explicit search. While this capability can be viewed as part of a general planning process, the paper should make this distinction clearer and discuss its implications for the interpretation of results.
 - The sample size per setting appears quite limited — only 20 states per difficulty–tier combination. It is unclear how many independent runs were conducted per model and configuration; if each setup was evaluated only once, the reported results may not be statistically robust or representative.
 - Reporting performance as #MM is less informative than expressing results as a percentage relative to the optimal path length, which would normalize for differing task difficulties and provide a fairer comparison across samples.
 - The pass rate presentation is inconsistent -- expressed as a percentage in Figure 1 but as absolute counts in Table 2 -- which can be confusing or misleading.
 - The claim of testing "physical intelligence" seems overstated, as the benchmark primarily probes spatial reasoning and planning rather than broader aspects of embodied or sensorimotor intelligence. A term like "spatial intelligence" would more accurately describe the benchmark’s current scope.

### Questions
- **No-code (pure reasoning) setting**: Could you add an evaluation mode where the agent is not allowed to write or call code (no programmatic search), and must output only a move sequence (e.g., as a JSON list)? This would isolate the model’s internal state tracking and planning ability from code-based search, which it might recall from pretraining data.
-  **Difficulty sensitivity and search usage**: Can you provide a plot of state depth vs. performance? Additionally, a histogram of state depth versus the number of moves (or relative number of moves) would help reveal when agents switch from direct planning to search-based strategies.
 - How many runs per state has the model performed? Can you make a few extra runs to verify the consistency of the results?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces CubeBench, a clever benchmark using the Rubik’s Cube to test LLMs on three tough abilities: spatial reasoning, long-horizon planning, and active exploration under partial observation. It defines three tiers — full symbolic, full visual, and partial visual states to progressively stress these skills.

The experiments are thorough and frankly pretty sobering. None of the tested models (GPT-5, GPT-4, open-source LLMs) managed any long-horizon solves 0% success across the board. Even short tasks were shaky: GPT-5 hits ~75% on symbolic cubes but drops to 20% with images and nearly 0% when only partial views are given. It’s a clean way to show that current models can describe cubes but can’t really think in 3D or maintain state.

Reward shaping helps a bit on short cases (0.2 - 0.55 success), but doesn’t fix planning. Adding a classical solver module boosts success to nearly 100% once perception is perfect, proving the core weakness is long-range reasoning, not local step choice. But even then, partial-view perception totally breaks them still 0%.

Overall, CubeBench is a strong diagnostic tool. It clearly isolates where LLMs fail — in sustained reasoning, 3D mapping, and exploration. The work is solid, though maybe a bit narrow (just Rubik’s Cube). Still, it makes a sharp point: today’s models can talk about the physical world, but they can’t mentally move in it yet.

### Strengths
This paper does a great job of cutting straight to the heart of the problem to clearly identifies three big cognitive gaps holding LLM agents back in physical reasoning: 3D spatial understanding, long-horizon planning, and exploration under partial observation. It’s refreshing to see these challenges framed so directly, with tasks designed specifically to isolate each one rather than mixing everything together.

The idea of using the Rubik’s Cube as a controlled testbed is honestly quite clever. It’s simple enough to simulate and evaluate, but complex enough to expose real weaknesses in reasoning. The three-tier setup (symbolic - visual - partial visual) works really well it’s systematic, and you can literally see where the models start to fall apart.

The experiments are solid and well-rounded. The authors test multiple top-tier LLMs (GPT-4, GPT-5, open models) and even include comparisons with smaller RL agents. The results are both surprising and sobering none of the models solved long-horizon cubes (0% success!), and even short ones get much harder as soon as the cube is shown visually instead of symbolically. The dense-reward experiments are a nice touch, showing that short-term feedback can help a bit but doesn’t fix the fundamental planning issue. And giving the agent an external solver was a smart diagnostic move success jumps instantly, proving the bottleneck really is long-term reasoning, not basic logic.

The paper is also cleanly written and easy to follow. The structure, figures, and summaries make it painless to digest what’s otherwise a pretty complex set of experiments. You can tell the authors put care into making the results interpretable rather than just dumping numbers.

Overall, this is a well-motivated and genuinely insightful piece of work. It’s not just another “benchmark paper” it actually deepens understanding of why LLMs struggle with embodied, physical reasoning. CubeBench feels like something the field could really build on for years, especially for testing new architectures that claim to reason spatially or plan over long horizons.

### Weaknesses
The use of the Rubik’s Cube is smart but narrow it’s a clean test of reasoning, but still a very specific domain. It’s not obvious if the lessons from CubeBench will generalize to other physical problems like robotic manipulation or navigation. A short discussion or even a small test on another environment would make the results feel less domain-locked.

The heavy reliance on GPT-5 is another weak spot. Since the model isn’t public or well-documented, readers can’t really verify or reproduce much of the analysis. The paper could be clearer about how GPT-5 was accessed and what capabilities it had, especially for the visual tiers. Including more open-source baselines, even weak ones, would make the results more trustworthy.

I also wanted a bit more about how the models behave. The paper hints that GPT-5 sometimes tries tens of thousands of moves or performs better with sequential observations than full maps, but doesn’t dive into why. Some short qualitative analysis what reasoning patterns or blind spots the model shows would add real depth.

Another missing piece is any attempt to train or adapt the models. The work focuses on evaluation and simple interventions (dense rewards, external solver), but doesn’t test whether fine-tuning or curriculum learning might help. That leaves an open question: are these failures fundamental, or just lack of experience with structured spatial tasks?

Overall, a few clarity nits: “GPT-5” appears without explanation, definitions of “short” vs “long” horizon are easy to miss, and some table cells are left blank without note. Minor, but they add small friction to an otherwise very clean paper.

### Questions
Do you expect CubeBench insights to generalize to broader physical tasks like manipulation or navigation? Or is it mainly diagnostic for cube-like puzzles? Some discussion on how this setup could extend to other domains would strengthen the paper’s relevance.

Why did models perform better with sequential face-by-face views than full 2D maps? Is it due to attention limits, easier focus, or prompt design differences? A short analysis of this counterintuitive result would be helpful.

Since the MLP agent learned via RL, did you consider fine-tuning an LLM similarly? Would more exposure or reward training help overcome the 0% long-horizon failures, or is the limitation inherent to transformer reasoning?

In the Standard-Solver setup, where exactly do models fail—misreading colors, wrong face mapping, or confusion in coordinate translation? A brief error breakdown would clarify which spatial step is the main bottleneck.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents CubeBench, a three-tier benchmark for diagnosing LLM agents’ spatial reasoning, long-horizon state tracking, and exploration under partial observation using Rubik’s Cube tasks. Tier-1 exposes a full symbolic state, Tier-2 a full visual unfolded map, and Tier-3 partial visual observations with view-change actions; evaluation follows a Thought-Code-Observation loop with strict budgets. Across many frontier models, all long-horizon settings achieve 0.00 pass rate. Dense reward shaping improves some short-horizon cases but does not fix long-horizon failures. Equipping agents with solver tools sharply improves performance and separates planning from spatial translation challenges. Overall, CubeBench provides a verifiable, decoupled diagnostic suite that yields crisp negative results and actionable insights for agentic research.

### Strengths
1. The three-tier design cleanly isolates (i) symbolic state tracking, (ii) visual/spatial reasoning, and (iii) exploration under partial observation.
2. The universal 0.00 pass rate on long-horizon tasks highlights a concrete direction where current LLM agents systematically struggle.
3. The paper contrasts sparse terminal rewards with thoughtfully defined dense-shaping variants (sticker/face/heuristic), enabling reproducible ablations.
4. End-to-end diagnostic arc. Framing the Rubik’s Cube across presentation (state/observation), optimization (with/without shaping), and tool use (standard/ideal solver) yields a coherent suite of ablations that pinpoint where capabilities break.

### Weaknesses
1. While the cube is a controlled, verifiable domain, it’s not yet clear why it’s the right anchor benchmark relative to other long-horizon planners (e.g., PlanBench, SPIN-Bench, ARC/ARC-AGI) or embodied 3D reasoning suites (e.g., VisiBench, ALFRED). A side-by-side rationale (capabilities stressed, evaluation verifiability, cost, and failure taxonomy) would better justify CubeBench as a first diagnostic stop rather than a narrow puzzle niche.
2.Tool-use evaluation may entangle multiple difficulties as well. “Standard-Solver” demands both perception→symbolic translation and strict format compliance; “Ideal-Solver” removes translation entirely. Without granular error audits, it’s hard to tell whether failures are due to spatial mapping, formatting, or simple string schema slips.

### Questions
1. Could you break out pass rates by each target depth (1,2,3,4,8,12,16,20) to show precisely where performance collapses?
2. Why Rubik’s Cube? Please articulate a principled rationale vs. existing planning/3D benchmarks: what capabilities are uniquely isolated here (beyond verifiability), and which insights demonstrably transfer to non-cube domains?

### Soundness
3

### Presentation
3

### Contribution
2
