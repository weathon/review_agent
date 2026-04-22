# Accelerated co-design of robots through morphological pretraining

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 2, 6, 6

## Abstract
The co-design of robot morphology and neural control typically requires using reinforcement learning to approximate a unique control policy gradient for each body plan, demanding massive amounts of training data to measure the performance of each design. Here we show that a universal, morphology-agnostic controller can be rapidly and directly obtained by gradient-based optimization through differentiable simulation. This process of morphological pretraining allows the designer to explore non-differentiable changes to a robot's physical layout (e.g. adding, removing and recombining discrete body parts) and immediately determine which revisions are beneficial and which are deleterious using the pretrained model. We term this process "zero-shot evolution" and compare it with the simultaneous co-optimization of a universal controller alongside an evolving design population. We find the latter results in _diversity collapse_, a previously unknown pathology whereby the population—and thus the controller's training data—converges to similar designs that are easier to steer with a shared universal controller. We show that zero-shot evolution with a pretrained controller quickly yields a diversity of highly performant designs, and by fine-tuning the pretrained controller on the current population throughout evolution, diversity is not only preserved but significantly increased as superior performance is achieved.  Videos and code can be found at: https://lukestrgar.com/codesign-mpt-project-page/

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a framework for accelerating robot morphology–control co-design through morphological pretraining. The authors propose training a universal controller across millions of simulated morphologies using differentiable physics, allowing rapid evaluation of new robot designs without retraining. The pretrained controller is then used in “zero-shot” and “few-shot” evolutionary experiments to explore and refine robot morphologies efficiently. The results show that the pretrained universal controller enables faster convergence and greater morphological diversity than training from scratch.

### Strengths
- The paper is clearly written and includes detailed descriptions of both the method and the experimental setup. The authors do a good job explaining their simulation and differentiable modeling pipeline, and the visuals and figures effectively communicate the core ideas and results.
- The motivation for leveraging morphological pretraining is sound, and the idea of a universal differentiable controller that enables rapid design-space exploration is intuitively appealing. The experiments convincingly demonstrate that pretraining can lead to both performance gains and increased design diversity during evolution.

### Weaknesses
- Despite the paper’s technical completeness, it reads more as a comprehensive experiment report than as a research contribution with clear conceptual novelty. Most components (differentiable simulation, universal controllers, and evolutionary co-design) are existing ideas in the literature, and it remains unclear what fundamentally new insight or principle this work establishes beyond scaling up prior approaches.
- The lack of comparison against existing co-design or differentiable evolution baselines (e.g., DiffAqua, BodyGen, or Metamorph) makes it difficult to quantify the method’s true advantage. Even a simplified baseline comparison would make the contribution more concrete.
While the concept of “morphological pretraining” is interesting, its novelty is mainly terminological. The paper would benefit from a clearer articulation of how it differs conceptually from pretraining universal controllers in prior work.

### Questions
- It would help if the authors clarified what specific insights this framework reveals that earlier differentiable co-design studies missed. For example, does morphological pretraining uncover a new pattern of transfer, scalability, or stability in evolution?
- Additionally, the paper could discuss how this approach might generalize to real hardware or to non-locomotion tasks, given that all experiments are simulation-based.

### Soundness
2

### Presentation
4

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents a timely contribution to the field of embodied co-design of artificial agents. The core idea is to leverage a process of large-scale, gradient-based "morphological pretraining" in a differentiable simulator to learn a single, universal controller that is agnostic to the robot's morphology. This pretrained controller is then used to accelerate evolutionary design in two novel paradigms: "zero-shot evolution" (using a frozen controller) and "few-shot evolution" (with generational fine-tuning). The authors identify and analyze a critical pathology termed "diversity collapse," where simultaneous co-optimization of morphology and control leads to a homogenization of designs, and demonstrate that their few-shot approach effectively mitigates this issue.

### Strengths
1. The identification of "diversity collapse" is a key intellectual contribution. It provides a formal explanation for a likely unstated observation in many prior co-design studies. 

2. The scale of the experiments is impressive (e.g., pretraining on 10M+ morphologies, evolving populations of 8192 robots).

### Weaknesses
1. The authors proposed the concept of "morphological pretraining". But they actually used a differentiable simulator to pretrain a single controller. This is a little bit confusing. I thought they were going to pre-train a single morphological design like what PreCo (Wang et al. 2023) does.

2. The task (phototaxis/magnetotaxis) is a single, well-defined objective. While complex due to the rugged terrain, it does not require higher-level behaviors like object manipulation, multi-agent interaction, or dealing with multiple, conflicting stimuli. 

3. The direct genotype encoding in a grid, while manageable for this study, may not scale efficiently to larger or more complex robots.

### Questions
1. Given the critical role of the differentiable simulator, what specific steps or modifications do you believe are most crucial to successfully transfer these co-designed robots to physical hardware?

2. The universal controller excels at a locomotion-based taxis task. Does it still work in manipulation tasks?

3. You show that few-shot evolution preserves diversity without explicit pressure. Could you speculate on the mechanism behind this? Furthermore, for practical design, how would you guide the search towards a specific type of diversity (e.g., prioritizing walking vs. rolling gaits) using your framework?

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
5

### Summary
The paper investigates the co-design of embodiment and control using a pretrained controller architecture. To this end, a policy was trained on 10 million morphologies - the resulting policy was then used to optimise the embodiment of the agent.
The paper investigates the use of such morphology-agnostic control policies when optimising the embodiment of agents in a zero-shot or few-shot manner.
While the technical novelty of the paper is reasonable, I think it provides great insight into the use of the generalisation abilities of morphology-agnostic controllers. Especially, shining a light on new open problems, such as the diversity collapse mentioned, is interesting to the community. Overall, I think the paper provides the robot leanring community with further insight into the co-design problem and the use of universal polcies in this setting.

### Strengths
- Interesting control problem, here a population of robots with different morphologies and configurations. Use of new technologies, such as differentiable simulators for fast design optimisation.
- The investigated morphology space is very large and complex - this is a clear strenght of this paper!
- Another particular strength of the paper is providing more insights into the general co-design problem with universal policies by providing deeper analysis.
- The literature review and discussion of related work is comprehensive and extensive, covering all important recent works.

### Weaknesses
- The method only considers one simulation environment.
- In Figure 6, it would have been interesting to see the best performance from all morphologies used in the pretraining stage (ie the best out of 10 million). It's not clear to me whether a better design can be found than through this massive grid search.
- An open question is also if the results can be translated to other robotics domains, or if no differentiable simulator is available. This could restrict the applicability of the methods and/or the generalizibility of the discussion points.

### Questions
See my question above about the best morphology out of the pretraining dataset.

### Soundness
4

### Presentation
3

### Contribution
3
