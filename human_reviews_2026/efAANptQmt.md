# Automating Benchmark Design

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
The rapid progress and widespread deployment of LLMs and LLM-powered agents has outpaced our ability to evaluate them. Hand-crafted, static benchmarks remain the primary tool for assessing model capabilities, but they quickly become saturated. In contrast, **dynamic benchmarks** evolve alongside the models they evaluate, yet they are expensive to create and maintain.

To address these challenges, we introduce **BeTaL (Benchmark Tuning with an LLM-in-the-loop)**, a framework that uses environment-design principles to **_automate dynamic benchmark construction_**. BeTaL parameterizes key design choices in base benchmark templates and leverages LLMs to reason over this parameter space to achieve desired properties such as difficulty and realism, all in a cost-efficient manner.

We validate BeTaL by targeting specific difficulty levels across tasks. Using this framework, we create two new benchmarks and extend the popular agentic benchmark **τ-Bench**. Extensive evaluation across the three tasks and multiple target difficulty levels shows that BeTaL produces benchmarks much closer to the desired difficulty, with average deviations ranging from **5.3% to 13.2%**—a **2–4× improvement** over baseline methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a method for automated benchmark design with LLMs in the loop. The idea is that given a benchmark whose difficulty can be adjusted by changing some parameters, we can then automatically have LLMs find the right parameters so to achieve the right target difficulty, which is an important problem in the context of unsupervised environment design.

### Strengths
The idea to extend UED to automate full benchmark design with LLMs is interesting, and an important problem.

### Weaknesses
The environments being evolved are very toyish

The idea of UED from my understanding is to automatically create environments/tasks at the right level of difficulty for agents/policies to perform RL on. If the task is too easy or too difficult, the target agent/policy will not benefit from training in it. BeTaL attempts to solve the designer problem of creating tasks at the right difficulty, however it then does not attempt to have weaker models/policies train on those tasks, starting from the trivial ones, then increasing the difficulty as they improve. Given the narrative adopted by the paper and the analogies and comparisons with UED this seems like a missing experiment to me.

On many of the tasks the error bounds are very large, making it difficult to draw conclusions on performance differences. Running more seeds may help understanding actual performance differences. Since this is expensive, perhaps proving this in a single setup could be fine (eg: Run with #seeds >> 3 for a single model on all the benchmarks). Results on the t-bench are particularly weak.

I’m not particularly familiar with the space of automated benchmark design, but comparisons with bayesian optimization such as gaussian processes without LLMs in the loop might be a good simple baseline to try.

Editorials nits:
A few links for citations and figures are broken (lines 251, 399)

### Questions
1. Is it possible to run any curriculum learning experiments in real UED fashion, to check the validity of the method?
2. Is it possible to increase the number of seeds to improve the statistical significance of the results? Too high variance currently makes it a bit difficult to draw more conclusions.
3. How does the method compare to simpler bayesian optimization baselines without LLMs in the loop?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the problem of LLM evaluation, where static benchmarks quickly become saturated and dynamic benchmarks are costly to create and maintain manually. The authors propose BeTaL (Benchmark Tuning with an LLM-in-the-loop), a framework to automate the design of dynamic benchmarks. BeTaL starts with an under-specified environment template defined by a set of parameters. A designer LLM is prompted to reason over this parameter space and propose a specific configuration. This configuration is then used by a simulator to generate a set of problems. A target model (the model being evaluated) attempts these problems, and its performance is measured against a desired target performance level. The resulting performance gap is formatted as natural language feedback and included in the designer LLM's prompt for the next iteration. This loop repeats, and the goal is to output the parameter configuration that achieved the minimum gap. 

The authors demonstrate BeTaL on three tasks: Arithmetic Sequences, Spatial Reasoning, and $\tau$-bench Airline. Experiments show that BeTaL more effectively finds parameters that match a target difficulty level compared to baselines like random sampling and best-of-n method. The paper also shows that the difficulty of these generated benchmarks transfers across different evaluation models (e.g., from o4-mini to Gemini 2.5 Flash and Claude 3.7 Sonnet).

### Strengths
The paper tackles a clear and significant problem for the community: the saturation of static evaluation benchmarks and the high cost of manually updating dynamic ones. The goal of automating this process is well-motivated and valuable.

### Weaknesses
1. The framework's primary weakness is its reliance on access to parameterized and verifiable simulators. This assumption is extremely strong and does not hold for many, if not most, complex and realistic evaluation domains. While feasible for the toy-like "Arithmetic Sequences" or "Spatial Reasoning" grid world, this is a critical bottleneck for applying BeTaL to open-ended domains like agentic web tasks, robotics, or complex code generation, where a verifiable simulator is often as hard to build as the evaluation itself.
2. Following the first point, the experimental validation is confined to highly structured and relatively simple environments. It is not clear how the BeTaL framework would scale to tasks with high-dimensional, continuous, or combinatorially vast parameter spaces. The paper itself notes the evaluation is "limited to a small set of domains, leaving multimodal and more subjective tasks unexplored", but this weakness is central to the method's potential impact.
3. The "Designer" LLM's adaptation is purely based on in-context reasoning, where the history is fed back into the prompt. The designer model itself does not learn or update its parameters to become a better designer over time. This approach is capped by the base LLM's reasoning ability and may be inefficient, as the LLM must re-evaluate the entire history in-context at each step.
4. A crucial part of "automating benchmark design" is defining the parameter space itself. The paper briefly explores this in C4 , but the experiment shows the AI-designed space underperforms the human-designed one for "Easy" and "Trivial" benchmarks. This vital part of the problem feels more like an afterthought than a core component of the method.

### Questions
1. How do the authors envision BeTaL being applied to domains where a fast, parameterized, and verifiable simulator is not available? For example, evaluating the difficulty of mathematical proofs or the quality of generated code, where "ground truth" is complex and simulation is not applicable. Does the reliance on a simulator  fundamentally limit the method to toy or game-like environments?
2. BeTaL requires a strong LLM as the designer. Does that mean that the designed benchmarks are only be useful to evaluate weaker models instead of strong model such as in the same intelligence level of the designer?

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
The authors are responding to an existing need in the LLM community to provide challenging yet realistic evaluation methods to measure the performance of the ever evolving LLMs. The approach is well motivated and the topic of automated LLM benchmarking has been an important discussion point in both academic and industry settings. The authors "formulate the benchmark design process as an optimization problem, where the goal is to maximize utility or usefulness," introducing a "design process for automatically producing and evolving benchmark(s)," to ensure that the ever evolving LLMs can be matched with ever adapting evaluation methods.

### Strengths
The authors define the "benchmark design process as an optimization problem" and demonstrate that their "empirical results show BeTaL consistently obtains benchmarks with any given target difficulty, achieving a performance gap of as low as 0.4% and up to 5% in several settings, a significant improvement over baselines."

This is an important contribution, leading to environment setups that are well suited to a specific performance level and can be used to test a variety of different models. Such ability is significant, given that modern LLMs have a wide range of capabilities and differ from one another greatly, hence cannot all be measured by the standard of the best performing competition LLMs, which are often the ones leading to SoTA results on benchmarks.

### Weaknesses
While the paper provides a valuable contribution toward a benchmark contains tasks with different complexity levels, I believe to strengthen the authors claims' about its universality and adaptability, the benchmarks should have more been tested on target models of different sizes. It is a bit unclear to me what is the different between the 'target' and 'evaluating' models provided by authors in Section 4.3, stating that 'We use o4-mini as the target model in all the settings. We finally evaluate benchmarks developed by each method on three models: o4-mini, Gemini 2.5 Flash, and Claude 3.7 Sonnet," but all of these models are in the closed source, large scale model category, hence limiting the weight of the authors' claim as to how universal generated experiments are. 

I would like to see more details on the efficiency analysis of the proposed algorithm. While Figure 2 shows relative performance gain of BeTaL over the baseline, the authors do not provide any analysis of the convergence rate of the algorithm to generate the target performance level benchmarks vs the search parameter space, with the latter varying greatly between the different experiment setups.

Another necessary information missing is the rate of hallucinations by the model, demonstrating how often BeTaL generates out-of-domain configuration suggestions.

Authors in Conclusion mention “efficient task synthesis,” however later in Limitations directly state that the simulators themselves are assumed to already exist. That makes it unclear whether the authors method, based on Algorithm 1, does not exist to synthesize tasks (aka user environments), but only to optimize them to a target performance level, or whether it can do both.

### Questions
Could the authors clarify exactly the scope of the BeTaL algorithm. Is it accurate that its goal is to synthetize new parameters for existing environment to suit a target performance setting, rather than designing completely new ones?

In general, I believe that in order for the BeTaL to be most useful to the community, authors should provide more details about the environment choice step. What considerations should the people who use PeTaL follow to choose an optimal environment, is there an abundance of some predefined environments that the authors can recommend BeTaL users could start with, or is environment design and a setup still essentially a manual process, with BeTaL just meaning to optimize its configurations to a specific target LLM performance level?

### Soundness
3

### Presentation
3

### Contribution
2
