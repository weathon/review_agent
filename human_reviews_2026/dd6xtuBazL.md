# Plan2Evolve: LLM Self-Evolution for Improved Planning Capability via Automated Domain Generation

- Decision: Reject
- Scores: 6, 4, 2

## Abstract
Large Language Models (LLMs) have recently shown strong potential in robotic task planning, particularly through automatic planning domain generation that integrates symbolic search. Prior approaches, however, have largely treated these domains as search utilities, with limited attention to their potential as scalable sources of reasoning data. At the same time, progress in reasoning LLMs has been driven by chain-of-thought (CoT) supervision, whose application in robotics remains dependent on costly, human-curated datasets. 
We propose Plan2Evolve, an LLM self-evolving framework in which the base model generates planning domains that serve as engines for producing symbolic problem–plan pairs as reasoning traces. These pairs are then transformed into extended CoT trajectories by the same model through natural-language explanations, thereby explicitly aligning symbolic planning structures with natural language reasoning. The resulting data extend beyond the model’s intrinsic planning capacity, enabling model fine-tuning that yields a planning-enhanced LLM with improved planning success, stronger cross-task generalization, and reduced inference costs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Plan2Evolve, a self-evolving framework designed to enhance the task planning capabilities of Large Language Models (LLMs) by generating and learning from Chain-of-Thought (CoT) data. The approach bridges two key research areas: (a) the use of LLMs for Planning Domain Definition Language (PDDL) domain generation, and (b) the enhancement of CoT reasoning in LLMs.

This is achieved by repurposing LLM-generated PDDL domains as a source of scalable and verifiable training data. The process involves several stages: First, the framework first employs the base LLM to generate and self-correct a valid PDDL domain. Then, a symbolic solver is used to produce verified problem-plan pairs. Nextly, these symbolic pairs are converted to natural language pairs, forming CoT reasoning trajectories. Finally, the model is fine-tuned on these data.

Experimental results demonstrate that this method significantly improves the model's planning success rate, its ability to generalize to unseen tasks, and its overall efficiency. The study also underscores the critical importance of symbolic-language alignment and proves the framework's effectiveness compared to direct knowledge distillation.

The core contributions lie in demonstrating that an LLM's planning capabilities can enhance planning capabilities by: (1) self-generating scalable and verifiable data; (2) leveraging structured knowledge from tools like PDDL solvers; and (3) aligning formal symbolic structures with natural language reasoning.

### Strengths
1. The proposed method reduces the need for expensive, human-curated planning datasets, and enables models to achieve performance that surpasses even larger baseline models.

2. The approach is straightforward, effective, and highly reproducible.

3. The paper situates its work by identifying two relevant research areas—symbolic planning and Chain-of-Thought (CoT) post-training for LLMs. A key contribution is its novel approach to bridging these two fields.

4. The experiment is detailed, including comparisons and ablation studies that demonstrate the effectiveness of the proposed method.

### Weaknesses
Please refer to the question section.

### Questions
# On the Limitations of the PDDL-based Approach

1.1 The PDDL framework has inherent limitations in its expressiveness and cannot encompass the full spectrum of real-world planning tasks. How might this constrain the broader applicability and scalability of the Plan2Evolve framework?

1.2 The method relies on a symbolic solver to generate the valid plans that form the CoT data. This raises two concerns: First, could the LLM's learning process be seen as merely imitating the solver's logic? Second, since many real-world problems are not readily solvable by existing symbolic planners, does this reliance on a solver create a performance ceiling for Plan2Evolve?

1.3 The training dataset is solely based on the Blocks World task. Does this narrow training domain potentially limit the model's generalization capabilities on more diverse, unseen tasks? Have the authors considered whether expanding the variety of tasks in the training set could lead to further performance improvements?

# On the "Self-Evolving" Nature of the Framework

2.1 As noted in Q1.2, if the core planning logic is derived from an external symbolic solver, does the LLM primarily function as a translator between symbolic and natural language representations? This raises the question of whether the performance gains stem from the solver's embedded knowledge rather than a truely continuous "self-evolution" of the LLM's intrinsic planning abilities.

2.2 Line 424 highlights that the quality of symbolic-language alignment is critical for effective training. How does the model's own alignment capability change after undergoing SFT with Plan2Evolve? It would be insightful to see an evaluation of the model's text-to-symbol alignment performance (e.g., on benchmarks like [1, 2]) both before and after fine-tuning.

2.3 Could the fine-tuned Plan2Evolve model be used to generate a new, potentially higher-quality dataset for a subsequent round of training? Would an iterative, online learning-style approach, where the model continuously refines its own training data, lead to sustained performance improvements?

Clarification and additional experiments on these points would be very helpful in strengthening the paper.

*[1] TEXT2WORLD: Benchmarking Large Language Models for Symbolic World Model Generation. Hu et al.* 

*[2] PROC2PDDL: Open-Domain Planning Representations from Text. Zhang et al.*

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes a new method to enhance the planning capacity of LLMs. Specifically, an LLM is employed to generate symbolic representations in the form of Planning Domain Definition Language (PDDL) from natural language task descriptions. The resulting planning domain and task solution obtained from a PDDL solver are then translated back into natural language with chains of thought. These translated examples collectively form a dataset used for supervised fine-tuning to strengthen the planning capability of the LLM.

The paper is grounded in the observation that natural language explanations articulated through chains of thought can more effectively enhance the planning capacity of LLMs than fine-tuning directly on raw symbolic solutions. The curated training data translated from symbolic transitions capture stepwise symbolic reasoning patterns that often exceed the innate planning ability of LLMs. Overall, the proposed method is conceptually sound and easy to follow. Empirical results provide partial evidence of its advantage over LLMs that are not fine-tuned.

### Strengths
1. This work demonstrates that training on natural language translated from symbolic solutions can enhance the planning capacity of LLMs. In generating the planning domain, the authors design a two-stage procedure to produce an executable PDDL domain. During the symbolic-language alignment stage, they construct well-designed prompts from four perspectives to ensure that the generated language contains not just a linear narration of the symbolic solution but rich reasoning steps that enable the LLM to understand the task and its solution more holistically. These technical designs are commendable.

2. In the experimental section, the authors evaluate not only the planning capability within the designed environments but also the generalization performance on unseen tasks, which to some extent strengthens the contribution of this paper.

### Weaknesses
1. In the experimental section, the authors select only unfine-tuned LLMs as baseline methods and do not compare their approach with other LLM-based methods specifically designed for planning tasks. From my perspective, this is the main limitation of the work. Although the proposed method shows noticeable improvements, it remains unclear how much it contributes relative to existing approaches in the community. A more convincing evaluation would demonstrate that this method also outperforms prior fine-tuned LLMs developed for planning tasks.

2. The proposed method is implemented only on Qwen and not transferred to other base LLMs such as Llama or GPT-4o-mini. I suspect that some LLMs may lack the ability to generate executable planning domains solely from task descriptions. Nonetheless, it would still be valuable to see results showing how this method performs when built upon other base models.

3. The method is evaluated only on tasks with solution lengths of up to 60 steps. The paper does not examine its planning capability on longer-horizon tasks, which limits understanding of its scalability and robustness.

### Questions
1. Is it possible to fine-tune the LLM using RL instead of SFT within your proposed framework?

2. Do you have any plans to extend your model to other planning environments, such as RLBench [1], to further strengthen the contribution of your work?

[1] RLBench: The Robot Learning Benchmark & Learning Environment, RAL 2020.

### Soundness
2

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
5

### Summary
This paper presents Plan2Evolve, a framework for improving the planning capabilities of LLMs. The methods uses LLM-generated (using a strong LLM) symbolic planning domains (in PDDL), which are then used to produce problem/plan pairs, and then convert these symbolic traces into CoT explanations using the same model. The generated symbolic and natural-language traces form a training dataset for supervised fine-tuning (SFT), producing a specialized LLM.

Empirically, the paper shows that Plan2Evolve improves planning accuracy, cross-task generalization, and token efficiency on several benchmarks (Blocksworld variants and real-world robotic scenarios). Ablation studies analyze different choices for some of the parameters (e.g., teacher model).

### Strengths
- Both areas, (PDDL) planning and self-improvement of LLMs, are gathering attention recently, so the topic certainly fits the scope of ICLR
- The experimental results give solid results

### Weaknesses
(1) My main critique of the paper is that it is proposed as a self-evolving technique while it is not the case. There are several reasons why this is an overreaching claim:

- While the paper presents Plan2Evolve as a “self-evolving” framework, the experiments only show a single generation+fine-tuning iteration. The model is never shown to re-generate improved data using the evolved version. As such, the claimed self-evolving loop remains hypothetical rather than empirically validated. Demonstrating iterative self-improvement would have been a crucial step to substantiate the “self-evolution” claim. 
- More importantly, Plan2Evolve in practice requires strong teacher models to create the symbolic domains and reasoning traces. This dependency undermines the claim of self-evolving (it is not "self-" if it relies on a teacher!).

(2) My main issue is with the framework is the need for the LLMs to produce the domains by themselves. In practice, the success of Plan2Evolve depends heavily on the model’s ability to create valid and meaningful problems. It is unclear how far this can be pushed. Also, it is unclear how this process is done. This part was forwarded to the paper by Huang et al. (2025b), but I think it should be contained in the main text of this work. 

(3) The experiments give solid results (as I mentioned above) but some of the design choices for the framework would need extra experiments to be better motivated. As explained in (2), an important aspect of the framework is that the LLMs create the domains they will be trained on. But then, in the experiments, some of the domains used already exist. What am I missing here? On a similar note, the paper also mentions using FF to produce the training pairs, an off-the-shelf planner. If we rely on an off-the-shelf tool, how is this self-evolving?

(4) The high-level results of the paper --- using output from a stronger teacher to bootstrap reasoning traces helps reasoning ---   is known (e.g., see [1] below). The cool thing of this submission would be the self-evolving aspect of it but, as argued in the previous points, the claim of really being self-evolving is too weak.

(5) The paper is too hand-wavy for an ICLR paper. For example

- Section 3.1 is very sloppy. Predicates are described as Boolean classifiers (they are relations), actions lack explicit add/delete semantics, and goals are specified as complete states (which is quite limiting).
- Section 3.3 is very unclear and high-level; from the textual description it is not obvious what is going on
- The paper does not report any of the hyperparameters used during supervised fine-tuning. Neither Section 5.4 nor the Appendix provides training details such as optimizer, batch size, learning rate, training steps, or fine-tuning method. 
- In general, the paper would be helped by using some of the general guidelines proposed by [2]


[1] Muennighoff et al. "s1: Simple test-time scaling"\
[2] Katz et al. "Make Planning Research Rigorous Again!"

### Questions
- Can you clarify what exactly you mean by self-evolving? Could you please explain me why Plan2Evolve is indeed self-evolving despite the criticism that I raised above?

- Have you actually run multiple Plan2Evolve iterations? If not, can you provide evidence or analysis that an additional iteration would further improve results, to justify the self-evolving claim?

### Soundness
3

### Presentation
2

### Contribution
1
