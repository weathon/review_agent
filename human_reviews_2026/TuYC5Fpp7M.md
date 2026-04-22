# What drives success in physical planning with Joint-Embedding Predictive World Models?

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
A long-standing challenge in AI is to develop agents capable of solving a wide range of physical tasks and generalizing to new, unseen tasks and environments. A popular recent approach involves training a world model from state-action trajectories and subsequently use it with a planning algorithm to solve new tasks. Planning is commonly performed in the input space, but a recent family of methods has introduced planning algorithms that optimize in the learned representation space of the world model, with the promise that abstracting irrelevant details yields more efficient planning. In this work, we characterize models from this family as JEPA-WMs and investigate the technical choices that make algorithms from this class work. We propose a comprehensive study of several key components with the objective of finding the optimal approach within the family. We conducted experiments using both simulated environments and real-world robotic data, and studied how the model architecture, the training objective, and the planning algorithm affect planning success. We combine our findings to propose a model that outperforms two established baselines, DINO-WM and V-JEPA-2-AC, in both navigation and manipulation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper systematically examines the design choices that determine success within the family called JEPA-WMs: model architecture, training objective, context length, rollout depth, proprioceptive inputs, and planner, 
via a large set of experiments in various benchmarks. From these studies the authors assemble a practical recipe and a concrete model that, under their evaluation protocol, improves over two established baselines (DINO-WM and V-JEPA-2-AC) on a range of navigation and manipulation tasks.

### Strengths
The paper’s main strength is its practical focus: rather than proposing another single algorithm, it provides a careful, broad empirical study that surfaces useful rules for building JEPA-WM systems, such as the value of proprioception, short multi-step rollouts, and specific conditioning choices. The experimental work is extensive and generally well controlled, with multiple ablations across planners, encoders, and predictor designs, and the paper documents training objectives, algorithms, and hyperparameters clearly enough to support reproduction. Finally, the results are significant for practitioners because they turn fragmented prior knowledge into actionable guidance for designing world-model-based planners.

### Weaknesses
The paper’s weaknesses may exist in the four points. First, the new method name “JEPA-WM” and the role of Eqs.(1)–(3) are ambiguous: it is unclear whether the manuscript proposes a novel algorithm or merely defines a unifying implementation recipe. Second, the experimental narrative lacks a clear prioritization and modular structure, so many results appear as a list rather than a justified study plan. Third, the encoder comparison (Fig.4 right) does not present a single, detailed fairness table (frame preprocessing, pooling, projection, freeze/finetune, normalization), making the claim “DINO > V-JEPA” hard to interpret causally. Finally, the predictor-conditioning section (Fig.6) does not explain where/how conditions are injected, the extra parameter/runtime costs, or the inductive biases, so the AdaLN finding cannot be mechanistically understood or easily reproduced. The details are given in the following questions.

### Questions
1. The paper states it does not propose a new algorithm, but the name “JEPA-WM” is new and readers may be confused. Please add a clear sentence in the Introduction saying why you chose the name JEPA-WM and whether Eq.(1)–(3) should be read as a unified implementation recipe rather than a novel algorithm; also mark at the start of Section 3 which parts are prior work and which parts are the paper’s original contributions.

2. This work mixes many modules and experiments, so you should explain the experimental prioritization and present the methods in a more structured way. Please add a short rationale about why some experiments were done first and others later, and reorganize Sections 1–4 so that the modular design and the experimental priority are explicit rather than leaving readers with a list of results.

3. Figure 4 (right) suggests DINO encoders can outperform V-JEPA in your pipeline, and I appreciate that you tried to equalize model size and predictor dimensions. To make this claim robust, please provide a single table that lists for each encoder the input frame count, frame preprocessing (e.g. duplication), token aggregation method, entry/exit projection details (layers and activations), freeze/finetune policy, and any normalization differences, and discuss how these choices might affect the observed differences.

4. Section 4 presents four predictor conditioning variants (results: Fig. 6) but the description is too brief to explain why those four were chosen or how each works. Please expand Section 4 to describe for each method where and how condition information is injected, the inductive bias you expect from that design, and why that design should connect to the empirical results so that others can reproduce your findings and understand the mechanism behind observations such as AdaLN’s advantage.

5. Please fix several presentation issues: place each figure close to the text that first cites it, use \citep{} where appropriate for parenthetical citations, and correct the reversed opening quotation mark in Figure 2 (e.g. ”open and move up”).

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
This paper studies design choices that make Joint Embedding Predictive World Models effective for planning in physical environments. The authors define a JEPA-based world model class with a frozen visual encoder and a learned predictor plus optional proprioception. They run a systematic ablation across planner choice, loss and rollout schedule, context length, encoder family and size, positional encoding, and conditioning strategy. The evaluation spans simulated navigation and manipulation tasks from Metaworld and offline datasets such as Push T, Wall, and PointMaze, plus real robot data from DROID and zero shot transfer to Robocasa. The study finds that DINO encoders outperform V JEPA encoders on these tasks, that two step rollout losses and modest context improve planning, that proprioception helps when aligned, and that AdaLN conditioning with RoPE is a strong architectural choice. The authors introduce an interface to use NeverGrad optimizers for planning and compare it with CEM, noting a trade off between exploration and precision. Combining the best choices, their model improves over DINO WM and V JEPA 2 AC on most tasks.

### Strengths
- Broad and careful empirical study of JEPA based world models for planning across both simulated and real robot settings. The scope covers navigation and manipulation with diverse datasets and evaluation regimes.
- Clear and useful findings that practitioners can act on. DINO encoders provide stronger fine grained spatial cues than V JEPA encoders for planning. Two step rollout loss and modest context help, with the constraint that the planning context should not exceed the training context. AdaLN conditioning with RoPE is consistently strong. Proprioception materially improves control when aligned between train and test domains.
- Introduces a practical NeverGrad planning interface that reduces hyperparameter tuning burden in new tasks and exposes a complementary exploration profile relative to CEM.
- Quantitative gains over credible baselines on most tasks. For example DROID Action Score improves from 37.4 to 49.9 over DINO WM and Robocasa Place improves from 25.9 to 37.1. The paper also reports seeds, evaluation episodes, and error bars, and explains how aggregate success is computed.

### Weaknesses
- Main novelty is empirical rather than algorithmic. The proposed model is a composition of known parts tuned within the JEPA WM family, which may limit the perceived conceptual advance.
- Real world evaluation is limited to offline action matching on 16 Franka videos and qualitative rollouts. No closed loop robot trials are reported, which limits conclusions about deployment readiness.
- Not all improvements are consistent. The model underperforms V JEPA 2 AC on Robocasa Reach. A deeper analysis of failure cases would help bound the method.

### Questions
- For the Robocasa Reach gap, what are the dominant failure modes. Is it due to action distribution shift, visual distractors, or sub optimal cost choice. A confusion matrix over sub tasks or a qualitative error analysis would help.
- How much do the results depend on the action to visual dimension ratio in the predictor. Can you run a control where sequence conditioning and feature conditioning are matched on this ratio to isolate the effect of conditioning scheme.

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
3

### Summary
The paper investigates which design factors make Joint-Embedding Predictive World Models (JEPA-WMs) effective for visuomotor planning. It analyzes how predictors trained in the embedding space of frozen visual encoders perform under different configurations. The study systematically ablates planner type, rollout loss length, proprioceptive input, encoder family, context length, predictor conditioning, and model size. A new Nevergrad-based planner (NG) is introduced, offering a CMA-ES-style optimizer requiring less hyperparameter tuning than CEM. Experiments across simulated and real-robot datasets (MetaWorld, Push-T, Maze, Wall, DROID, and Robocasa) show that short multi-step rollout losses, DINOv3 encoders, AdaLN+RoPE conditioning, and moderate context windows yield best results. Proprioception consistently improves planning success. The final configuration surpasses DINO-WM and V-JEPA-2-AC on most benchmarks.

### Strengths
1. The paper is well written and clear in presentation
2. The paper performs a systematic and granular exploration of multiple design factors in JEPA-based world models, isolating the effects of planners, context size, encoder type, etc. This level of experimental control is rare in world model research and provides actionable guidance for future work.
3. Evaluations span simulated control (MetaWorld, Maze, Push-T, Wall) and real-robot datasets (DROID, Robocasa). The authors' conclusions generalize across manipulation and navigation tasks.
4. The proposed configuration consistently outperforms DINO-WM and V-JEPA-2-AC across most tasks. The paper shows that that careful tuning of dynamics learning and planning can yield substantial performance gains even without architectural novelty.

### Weaknesses
Weaknesses + Questions here:

1. The paper provides a comprehensive empirical evaluation. Its contribution lies mainly in mapping hyperparameter effects within existing JEPA frameworks, making it more of a practitioner’s guide than a scientific leap. While this is not a dealbreaker for me, I think it hurts novelty a bit.
2. Despite the cost function being fully differentiable, the study does not include a gradient-based or hybrid planner comparison. In Fig. 3 (planning optimizers), the objective $L_p$ and unrolling $F_{\theta}$ are differentiable. Why not include a gradient-based planner through $F_{\theta}$? Any stability or exploding/vanishing gradient issues observed during backprop through Eqs. (2 - 4)?
3. Sec. 5.2 fixes a different planner+distance per dataset (e.g., NG-L2 for MetaWorld, NG-L1 for Robocasa, CEM-L2 elsewhere). What is the accuracy drop if you standardize one planner and one metric across all datasets?
4. Fig. 4 shows DINOv2/v3 outperform V-JEPA/V-JEPA-2 under your two-frame encoding. Is the advantage driven by local features, pretraining data, or the two-frame encoding choice? Please disentangle these factors.
5. Fig. 5 compares model size; larger encoders sometimes underperform. Is this due to harder latent-space optimization or a mismatch with CMA-ES covariance adaptation? Please clarify
6. The authors report strong results with AdaLN + RoPE. Does AdaLN primarily improve sample efficiency, training stability, or both? Do they have any ablations on AdaLN strength or RoPE variants?
7. I found some minor grammatical errors in the paper (not part of my evaluation).

### Questions
Questions above.

Overall I think the paper makes good empirical contribution that might help world model research in the future. The practical insights, careful study, and competitive results outweigh limited conceptual novelty. I would request the authors to participate in the discussion phase.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper aims to compare methods that use joint embedded predictive world models along the dimensions of planning optimizer, rollout prediction, impact of input from proprioception, the kind of observation encoder used, context length (i.e. the number of past observations and actions), predictor architecture (i.e. architecture that maps context to next observations), and model size.

Planner results: CEM tends to perform better than NeverGrad using L1 and L2 distances to the goal to guide planning

Rollout prediction: Results show two step prediction performs best, on average; however, there is a lot of overlap.

Input from proprioception: Proprioceptive input almost always helps performance.

Encoder: DINO performs better than V-JEPA.

Maximum context length, predictor architecture, and model size: These are less conclusive. A context length > 1 seems to help, but does not seem to help beyond that. Larger models do not show improved performance.

### Strengths
The paper aims to cover many factors that are related to success in joint embedded predictive world models. Based on all experiments, the paper proposes a method that combines the best elements amongst all experiments. This method performs better than others on the majority of tasks. I am not familiar enough with these environments to evaluate the significance of this.

### Weaknesses
The paper seems to have too wide of a scope. There are many questions that arise from the results that could probably be papers, on their own. However, because of the wide scope, there is not room to delve deeper into these questions to any meaningful degree. Here are my main points to illustrate this:

-	The planning architectures are compared only using L1 and L2 distances from the goal image. These distances can be very uninformative, in practice. For example, there is convincing work showing learning a value function using RL to guide search can work much better [1]. Therefore, comparing planning methods using only these metrics does not give one a good sense of which ones are truly superior.

-	Increasing the context length does not appear to help after 2. Why is this? Is the Markov property satisfied after 2? There are contrived examples one could create such that this is not the case. In that case, increasing the context length must become necessary. If it is not the case, then there is a flaw with context encoding. Further experiments are needed to make this value of 2 meaningful to the average practitioner.

-	Why does increased model size not lead to improved performance. The paper says training and validation curves do not indicate overfitting (where are they?). Is the practitioner then to stick to smaller models, in practice? This seems counterintuitive given the plethora of literature that shows improvements with more parameters in neural networks. Could it be a flaw in the architecture that prevents the model making use of these added parameters? Further research is needed before these results should be used as practical guidelines.

-	The performance when given proprioceptive input is evaluated, however, it is not clear which planner is used. How does the choice of planner interact with proprioceptive input and planner guidance?

Other comments:
Loss has \theta, but \theta is frozen for vis. Perhaps parameters should be given different symbols, for clarity.

Where are L_{vis} and L_{prop} formally defined?

[1] Tian et. al Model-based visual planning with self-supervised functional distances. ICLR, 2021

### Questions
Please see the questions listed in the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
