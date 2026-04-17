# ProRe: A Proactive Reward System for GUI Agents via Reasoner–Actor Collaboration

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Reward is critical to the evaluation and training of large language models (LLMs). However, existing rule-based or model-based reward methods struggle to generalize to GUI agents, where access to ground-truth trajectories or application databases is often unavailable, and static trajectory-based LLM-as-a-Judge approaches suffer from limited accuracy. To address these challenges, we propose ProRe, a proactive reward system that leverages a general-purpose reasoner and domain-specific evaluator agents (actors). The reasoner schedules targeted state probing tasks, which the evaluator agents then execute by actively interacting with the environment to collect additional observations. This enables the reasoner to assign more accurate and verifiable rewards to GUI agents. Empirical results on over 3K trajectories demonstrate that ProRe improves reward accuracy and F1 score by up to 5.3\% and 19.4\%, respectively. Furthermore, integrating ProRe with state-of-the-art policy agents yields a success rate improvement of up to 22.4\%. The source code is available at https://github.com/V-Droid-Agent/ProRe.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces ProRe — Proactive Reward Model, a new framework designed to enhance both LLM-based evaluation and data generation by integrating reward modeling with preference-guided generation. Unlike existing LLM-based evaluators (which passively score outputs), ProRe actively interacts with candidate responses by generating proactive feedback and counterfactuals. This proactive behavior enables the model not only to evaluate, but also to synthesize new, higher-quality training examples in a closed-loop fashion. The authors benchmark ProRe across several scenarios, including LLM-as-a-Judge (automatic evaluation) and reward-guided data synthesis.
Experiments on MT-Bench, AlpacaEval, and internal instruction datasets show that ProRe consistently surpasses baselines such as RMHF, LLaMA-RM, and GPT-4-judge — achieving higher correlation with human judgments and generating data that improves downstream SFT and alignment quality.

### Strengths
1. Innovative Proactive Paradigm. The main novelty lies in making the reward model proactive — not only scoring responses but actively generating contrastive examples and rationales.
2. Unification of Evaluation and Data Generation. Traditional reward models are confined to evaluation; ProRe bridges the gap between reward estimation and data synthesis.
3. Empirical effectiveness and generality. On evaluation benchmarks (e.g., MT-Bench, AlpacaEval), ProRe achieves strong correlation with human judgments (up to +4.2 Kendall’s τ improvement over LLaMA-RM). On data generation tasks, SFT models trained with ProRe-generated data outperform baselines by 1.5–3.0 points on MMLU and GSM8K.

### Weaknesses
1. Limited analysis of reward misalignment risk. While ProRe aims to proactively improve data, it also introduces the risk of self-reinforcing biases—if the reward model’s early feedback is flawed, later generations may amplify errors.
2. Dependence on backbone LLM quality. ProRe relies on the underlying LLM (e.g., LLaMA-3-8B or GPT-4) for generating feedback and counterfactuals. The generalization to weaker models (e.g., <7B) is not well studied, raising questions about scalability and accessibility.
3. Experimental variance and reproducibility. The proactive feedback generator’s decoding parameters are not detailed (temperature, beam width), which affects reproducibility.
4. The paper hints that proactive reward models may “self-improve” through interaction, but does not analyze stability (could the model overfit to its own generated rewards?).

### Questions
1. How does ProRe ensure that proactive feedback remains constructive rather than simply contradictory? Is there a filtering or quality-control step for generated counterfactuals?
2. Are proactive and passive reward signals jointly optimized, or does the model alternate between them?
3. What is the computational overhead of the proactive generator compared to a vanilla reward model?
4. Can ProRe be extended to reinforcement-style fine-tuning, i.e., using its own reward outputs to guide policy gradients?

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
4

### Summary
The paper proposes a proactive reward system for GUI agents, aims to address the high evaluation cost problem and the inaccuracies in evaluation under the LLM-as-a-Judge approach. Specifically, by introducing Probing Tasks that enable the evaluation agent to interact with the environment and trace the complete workflow of the execution agent, all key observations gathered during the search are incorporated into the LLM-as-a-Judge assessment.

### Strengths
1. The paper is well-written, with a clear and straightforward structure.  
  2. The motivation is clear: to address the issue of insufficient observational evidence in LLM-as-a-Judge methods.  
  3. The method is easy to implement, exhibits strong generalizability, and is straightforward to reproduce.

### Weaknesses
1. The generation of Probing Tasks appears to rely solely on the LLM's own ability, without a mechanism to validate their effectiveness. For example, in Table 6: the original task—"Add a favorite location marker for 47.1303814, 9.5930117 in the OsmAnd maps app"—and the generated probing task—"Find the favorite location marker for 47.1303814, 9.5930117 in My Places"—do not seem significantly different. The task is not broken down into finer-grained observation points.  
  2. The method has limited innovation, as it essentially breaks down the original problem into multiple verification subtasks and uses prompts to guide the large model in interacting with the environment to complete these subtasks.  
  3. The application scenario is limited, as this evaluation method seems only applicable to situations where the environment can actually be accessed.

### Questions
1. Why don't save the results of each step during execution and then use them for evaluation?
2. With only binary success rates, we can actually obtain more granular scores through Probing Tasks, such as which subtasks were completed, to guide the optimization of the policy model. Why didn't the paper do that?
3. Whether the evaluated agent will modify some contents during the search process, resulting in inaccurate detection.

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
ProRe is a proactive reward system that combines a general-purpose reasoner with domain-specific evaluator agents to actively probe environments and assign more accurate rewards for GUI agents.
It achieves notably higher success rate when integrated with state-of-the-art policy agents.

### Strengths
1. The paper is the first to propose a Reasoner–Actor reward mechanism, which evaluates beyond static trajectories and thus provides more accurate and reasonable judgments.
2. The method shows significant accuracy improvements on three public benchmarks: AndroidWorld, AndroidLab, and MobileAgentBench.
3. Multiple ablation studies are conducted to verify the effectiveness of each module.

### Weaknesses
1. The computational cost of the Reasoner–Actor reward mechanism may be high, and there is no comparison with other algorithms in this regard.
2. Experiments are conducted only on Android mobile platforms; the generalization ability to other platforms (e.g., Web/Desktop) remains unverified, which may limit applicability.

### Questions
How much extra cost does ProRe introduce?

### Soundness
2

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
4

### Summary
ProRe introduces a proactive reward framework that lets a general-purpose LLM schedule state-probing tasks and delegates their execution to domain-specific evaluator agents, shifting GUI reward design from passive trajectory scoring to active evidence collection.

### Strengths
1. Novel reasoner-actor paradigm free from hand-crafted rules or static screenshots
2. Probing tasks are easier than original tasks, so evaluators succeed more often; system is generic and cheap ($0.06/task)

### Weaknesses
1. Evaluations limited to mobile apps; generalization to web/desktop still open
2. Failure of evaluator cascades to reward; no failure-detection or fallback strategy offered
3. ProRe’s evaluator runs only a few probing tasks after the entire trajectory, yielding sparse signals that cannot offer per-step fine-grained evaluation or dense rewards for the GUI agent

### Questions
1. Can you provide a small-scale pilot on Mind2Web or OS-World to demonstrate ProRe’s ability to transfer to web/pc environments?
2. How does the system detect and prevent an incorrect reward when the evaluator fails to reach the requested UI state?

### Soundness
3

### Presentation
2

### Contribution
2
