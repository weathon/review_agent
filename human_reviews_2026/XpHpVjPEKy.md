# Comp-LTL: Temporal Logic Planning via Zero-Shot Policy Composition

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
This work develops a zero-shot mechanism, Comp-LTL, for an agent to satisfy a Linear Temporal Logic (LTL) specification given existing task primitives trained via reinforcement learning (RL). Autonomous robots often need to safely and deterministically satisfy spatial and temporal goals that are unknown until run time. Prior work on learning policies to execute an LTL task incorporates the specification into the learning process, requiring retraining or fine-tuning if the specification changes. We present a more flexible approach--to create a pipeline to deterministically choose an execution set of composable safe task primitive policies that can be used to satisfy arbitrary LTL specifications without retraining or fine-tuning. Safe task primitives can be learned offline using RL with a reward function focused on penalizing unsafe actions and combined using Boolean composition at deployment. We focus on creating and pruning a transition system (TS) representation of the environment in order to solve for deterministic, non-ambiguous, and feasible solutions to LTL specifications given an environment with multiply-labeled regions and a set of safe task primitive policies. Our pruned TS is deterministic, contains no unrealizable transitions, and is sound. Combining the TS with the safe pretrained task primitives produces a sequence of composed policies that are guaranteed to deterministically satisfy an LTL specification. Training on a base set of safe tasks and composing at run time reduces total training time compared to non-composition approaches and has negligible processing time at run time. We verify our approach via simulation in grid-based and continuous environments, and compare it to other state of the art approaches, showing that Comp-LTL is safer, more adaptable, and quicker at satisfying unseen specifications at runtime.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes an approach that composes pretrained task primitives to satisfy Linear temporal logic (LTL) specifications, with the aim of avoiding retraining or fine-tuning whenever the specifications change. The core method constructs a transition system (TS), which is then pruned and made deterministic. The approach is compared with prior work.

### Strengths
The approach aims to address an important problem in learning for LTL: the need to retrain or fine-tune whenever specifications change. In addition, the figures in the paper are clear and illustrative.

### Weaknesses
- **W1.** The related-work section is not well organized or sufficiently extensive. For example, as far as I understand, your approach is quite similar to model-based planning without learning. The works by Qiu et al. (2023) and Jackermeier & Abate (2025) also appear closely related but were only briefly mentioned and were not explicitly compared with your approach.
- **W2.** The contributions are not very clear. As I understand it, zero-shot approaches for LTL already exist; the novelty of your approach seems to stem from implicit safety integration rather than from being zero-shot per se, and this is not emphasized—for example, in the abstract.
- **W3.** In my opinion, the assumptions are very strong: the environment is modeled as a deterministic Markov decision process (MDP) whose topology can be constructed, which makes the method strongly related to model-based planning approaches such as Kurtz & Lin (2023) rather than learning approaches.
- **W4.** Only the construction of a TS is explained in the technical approach section. The pretrained task primitives, arguably the core of the approach, are neither explained nor discussed, which makes the motivation for the TS unclear as well its function.
- **W5.** The experiments are not comprehensive; only two environments and two baselines are considered.


### References
1. Kurtz, Vince, and Hai Lin. “Temporal logic motion planning with convex optimization via graphs of convex sets.” *IEEE Transactions on Robotics* 39.5 (2023): 3791–3804.

### Questions
- **Q1.** See W1. Could you provide a more systematic related-work discussion and comparison? Possible categories: model-based planning for LTL; model-free and model-based learning for LTL; transfer learning and fine-tuning for LTL; zero-shot transfer for LTL. Could you explicitly state the advantages and disadvantages of your work relative to model-based planning approaches? What are the advantages and disadvantages of implicit vs. explicit safety, and why is this important?
- **Q2.** See W2. Also, what does your approach contribute beyond Kloetzer & Belta (2008) and Nangue Tasse et al. (2020)?
- **Q3.** What are your thoughts on W4?
- **Q4.** Could you provide results for additional environments and compare your approach with methods beyond reward machines (RMs) and skill machines (SMs), particularly with other zero-shot and model-based planning approaches?

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
The paper proposes a logic-based control framework that is _agnostic to the specific logical specification_, in the sense that changing the task objective, expressed as an LTL formula, does not require retraining any policy. This is achieved by combining a pre-trained set of primitive policies with a transition system (TS) that abstracts the environment into labeled regions. The TS is then composed with the automaton corresponding to the target LTL formula, inducing a product graph over which planning is performed. Consequently, adaptation to a new specification is achieved zero-shot, purely through symbolic planning on this graph, without additional policy learning.

### Strengths
- The paper is well written, and the contribution is clear.
- The proposed idea is novel. In particular, I find interesting the connection between temporal logic specification and zero-shot composition of policies for the similarity with the multi-task problem.
- The authors formally proove the soundness of the propose pruned transition system (TS).
- While in part I find the empirical evaluation limited, it shows promiseing results, and highlight that the the proposed framework can satisfy new logic specifications without retraining.

### Weaknesses
- The main concern is scalability. Constructing the transition system in realistic or continuous domains is likely intractable, especially since it requires identifying the regions where each atomic proposition holds.
- The need to train one policy for each atomic proposition $\sigma \in \Sigma$ does not scale well as the number of labels grows.
- As already anticapted, emprical evaluation is limited to toy domains. It is difficult to understand whether this method can be applied in more realistic environments.
- The framework assumes that the available primitives are sufficient to cover all relevant behaviors, which is a strong assumption in realistic settings (more on questions).

### Questions
- How computationally demanding is it to construct the transition system in real environments, particularly regarding the recognition of regions where specific propositions hold?
- Since the number of primitive policies required scales with the number of labels $\Sigma$, how does this behave in large or continuous domains? Could approximations or hierarchical abstractions make this approach feasible in practice?
- Also, suppose a simple navigation problem in a 2D environment. A goal can be any point (x, y), this means that I need infinite primitives to account for each goal?

Overall I find the idea interesting, my concerns are in the scalability and applicability of the method. I am open to discussion with the authors on the points raised.

### Soundness
4

### Presentation
4

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
The paper introduces Comp-LTL, a framework for zero-shot satisfaction of Linear Temporal Logic (LTL) specifications using pretrained RL task primitives. Instead of retraining policies for new specifications, Comp-LTL composes existing primitives via Boolean task algebra and a pruned, deterministic transition system (TS) that ensures feasible and sound planning. The method integrates safety through minimum-violation (MV) semantics and constructs a product automaton with Büchi automata for execution. Experiments in grid-based environments (Office World, Video Game) show that Comp-LTL achieves safer, faster, and more generalizable performance than baselines such as Reward Machines, Skill Machines, and Boolean Composition.

### Strengths
1. The integration of deterministic TS pruning and Boolean policy composition for zero-shot LTL satisfaction is original and well-motivated.
2. The paper provides clear theorems (determinism, soundness, feasibility) with proof sketches, demonstrating a solid theoretical foundation.
3. Comp-LTL requires no fine-tuning or retraining, showing strong adaptability to unseen specifications in grid-world environments.

### Weaknesses
1. The Q-learning algorithms used in the paper are value-based, discrete-action algorithms — meaning they assume a finite, enumerable action space. Authors claim that "Our approach agnostic to the method in which the policies are trained, as we show Comp-LTL is
successful with both tabular Q-learning and DQN policies." but the the Q-learning-based primitives in Comp-LTL cannot be directly applied to continuous-action robotic tasks. The experiments are confined to grid-based environments. Claims of generality would benefit from evaluation in continuous control or robotic settings.
2. While runtime and training times are reported, theoretical or empirical analysis of computational complexity (e.g., TS construction scaling with number of regions or propositions) is missing.
3. The claimed contribution of abstracting a geometric environment into a transition system with Boolean-composed task labels is not novel — similar abstractions were used in "Compositional RL from Logical Specifications" (NeurIPS 2020) and "Instructing Goal-Conditioned RL Agents with Temporal Logic Objectives" (NeurIPS 2023). The more distinct contribution lies in the pruning strategy ensuring deterministic, feasible TS representations and its integration with zero-shot safe composition.

### Questions
1. How would Comp-LTL perform on LTL formula having "Until" operator? And how about the performance on LTL with $\omega$-regular expression, which is a very important extension on LTL formula?
2. Would Comp-LTL still maintain zero-shot in continuous-action environments?
3. See above.

### Soundness
3

### Presentation
2

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
This work considers the problem of composing task primitives using LTL specifications. This is an established approach within the literature and this work aims to extend on this in two ways: 1) by incorporating an LTL pruning mechanism which simplifies the transition system defining the temporal sequence of tasks 2) by incorporating safety into the primitives themselves rather than relying on the LTL specification to guide the safety concerns which is done in prior work. The paper compares their approach to the state of the art approaches to using LTL specifications for task composition and shows that their approach is superior on a safety metric and in terms of learning speed.

### Strengths
## Originality
The work is grounded quite closely to the literature on using LTL specifications for temporal and spatial task composition. This is not inherently bad, and in fact by positioning the work clearly against these prior works it does make the differences stand out.

## Quality
The motivation of the work is clear and the hypothesis is grounded in prior work. The results that are presented are interpreted fairly.

## Clarity
The paper is well written and figures are clear and useful overall. The paper uses notation and symbols in a way that is typical of this line of work which makes it easier to follow.

## Significance
The work considers an important problem - safety within RL and also does support faster learning which is important as we expand our models into more difficult domains. Thus, there could be future work which builds on this paper and its stated claims.

### Weaknesses
## Clarity
Firstly the minor concern, the figure captions are very uninformative and this limits the benefit of the figures substantially. Figure 3a in particular really needs to be more descriptive both in terms of the caption and figure itself. The work also uses jargon which is not sufficiently defined such as "sound". When a word is used in this manner to mean something technical it is necessary to define it formally.

More importantly, it is very difficult for me to see the connection between the two main concerns of this work: the TS pruning and the approach of embedding safety directly into the primitives. This seems like two entirely distinct directions and makes the overall structure of the paper confusing.

## Quality and Significance
My first critique here is that the paper takes for granted that safety should be embedded directly into the task primitives rather than specified in the LTL. This is not obvious to me and undermines the entire direction as a result. I would greatly appreciate clarity on why we even want this in the first place. Secondly, the consequences of putting the safety behaviour into the training of the primitives is not given due consideration. My understanding is that this will make all of the task primitives sub-optimal and inflexible to cases where the constrains may be temporary. So while the domains with fixed constrains may be fine for this, the flexibility of the approach is limited greatly and by extension so is the applicability of the model. Remark 1 similarly notes a trade-off which emerges from the paper's approach to zero-shot satisfaction and the possibility of introducing sub-optimality and notes that RMs take a different approach by fine-tuning. But then why is the paper phrased as if it improves on RMs and SMs in this regard (for example on lines 471 to 473)? What is the point of being "full-zero shot" if the proposed method is also suboptimal at this just like the prior work which at least considers fine-tuning. 

Finally, Table 4 seems unreasonable to me and is poor experimental design. Perhaps I am missing something, but to compare the prior methods on a fairly arbitrary metric (number of additional symbols collected) which they were not trained to consider at all, while the proposed method explicitly optimises the metric and then claims to be superior, is fairly meaningless. I would appreciate more explanation on why this is even a fair comparison.

### Questions
I have asked some question in the review above which I would appreciate answers to. Additionally, what is the connection between the TS pruning and safety? Why report the computation time and the safety metric if this work is primarily concerned with safety? How should I interpret the speed-up relative to prior work when there is a trade-off as a result of this speed up (Remark 1)?

### Soundness
1

### Presentation
2

### Contribution
2
