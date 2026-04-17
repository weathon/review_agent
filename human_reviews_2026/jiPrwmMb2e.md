# Control Reinforcement Learning: Interpretable Token-Level Steering of LLMs via Sparse Autoencoder Features

- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Large language models exhibit emergent misalignment behaviors during test-time generation, necessitating dynamic control mechanisms for safe deployment. Inspired by sparse interpretable representations, sparse autoencoders (SAEs) can disentangle monosemantic features from superpositioned dense activations, offering a natural interface for controlling language model behavior through interpretable feature manipulation. This work introduces Control Reinforcement Learning (CRL), a framework to unify reinforcement learning with SAE features for interpretable token-level language model control. CRL enables interpretable branch tracking by isolating feature contributions at each generation step, revealing which features drive behavior changes across diverse benchmarks including question answering, bias mitigation, and reasoning tasks. To balance exploration and exploitation, the framework employs Adaptive Feature Masking (AFM) to encourage diverse yet effective feature discovery while maintaining interpretability. Through token-wise feature analysis, CRL provides mechanistic insights into model behavior, revealing task-specific feature contributions across diverse benchmarks including question answering, bias mitigation, and safety tasks. The framework is compatible with supervised fine-tuning, providing complementary control when applied to SFT models. Results demonstrate that interpretable steering can serve as both a control method and analysis tool, establishing a practical pathway for controllable AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Control Reinforcement Learning, a framework for dynamically steering LLMs through SAE features. Instead of static activation edits, CRL learns a policy to select which SAE features to activate at each generation step, guided by reinforcement learning rewards. The authors claim this allows interpretable, token-level control while modestly improving performance across tasks such as MMLU, GSM8K, and HarmBench. The approach also provides diagnostic insights into critic bottlenecks, layer-specific effects, and semantic coherence of SAE features.

### Strengths
1. **Novelty of the proposed method**: I think the main strength lies in framing reinforcement learning over SAE features, which differs itself from other activation-based or gradient-based method.
2. **Breadth of evaluation**: The authors conduct extensive experiments across reasoning, factual, and safety benchmarks, showing the generality of their approach.
3. **Clarity of methodology**: The paper is well-structured and technically sound, with a clear explanation of the training process and design choices.

### Weaknesses
1. **Limited comparison with existing feature control methods**: A key weakness is the lack of comparison to established feature control approaches such as activation-based or gradient-based interventions. I find this omission makes it hard to understand what specific advantages CRL provides beyond existing techniques. A more direct experimental or conceptual comparison would help clarify novelty.
2. **Computational complexity and scalability**:Training a PPO agent over sparse feature activations introduces nontrivial computational cost. I think it would help to include runtime analysis or efficiency comparisons to justify the added complexity relative to simpler steering methods.
3. **Under-analyzed reward design**: The reward function is central to the method but not well justified or analyzed. Its stability and sensitivity to tuning are unclear, which weakens confidence in reproducibility.
4. **Limited interpretability validation and generalization evidence**: Interpretability results are qualitative and narrow in scope. The authors should demonstrate that the learned feature control generalizes across different tasks or datasets.

### Questions
Please refer to weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes to learn by reinforcement learning a policy to steer LLMs by intervening on the activations of sparse features, as identified by SAEs trained on the residual stream activations. Experiments show modest improvements. The use of SAE features enable a degree of interpretability.

### Strengths
[S1] The idea of learning a policy to control feature strength is interesting and, to my knowledge, novel.

[S2] Some of the proposed analysis yield interesting insight, e.g. the different effects of controlling features across layers.

[S3] Results are presented in a measured way, without overstating them.

### Weaknesses
[W1] The paper suffers from a general lack of clarity. Crucial terms, like 'coefficients', are used without introduction or being obvious from the context (e.g. there are no coefficients in Eq. 3, which defines the activation interventions). Many sections in the body are not understandable without going back and forth with the appendix, as if content was moved there without checking the impact of doing so on the narrative flow. See below a detailed list of issues.

[W2] There are reproducibility issues: an Algorithm 1 is mentioned in the reproducibility section, but it does not seem to be anywhere. There is no description of the policy and value network implementations besides them being MLPs. Task-specific rewards are announced in 3.4, but are not in Appendix A as promised.

[W3] There is no experimental baseline. How well would CRL work if using random features instead of SAE features?

[W4] It is unclear whether reported results in Table 1 are obtained directly on the test set, or if the intervention layer is determined based on held-out data.

Additional points:

L22-24: Adaptive Feature Masking (AFM) is mentioned in the abstract and in the contributions, but never again in the paper?

L132-133: Why is the full problem a POMDP? Does this have to do with the fact that the influence of the KV cache of previous tokens through attention is not taken into account? This should be clarified, and an attempt to quantify the impact of this approximation should be made.

L159-161: This part is unclear. What are the coefficients being referred to? Should there be additional coefficients in Eq (3)?

L240-241: Coefficient averaging has not been introduced at this point.

Figure 2: what is in the left pane, and what in the right?

L263-265: Constrained and unconstrained decoding patterns have not been introduced. What is the connection between constrained/unconstrained decoding on one hand and factual question answering on the other?

L 268 and elsewhere: disambiguous --> unambiguous

L268-269: Can this norm increase be visualized?

L270: what is the 'coefficient 18 analysis'?

L285-286: Correct, incorrect, corrected and misguided are not introduced/defined.

In Figure 3 "generation step" means token position, in appendix it means layer.

Figure 4 caption: blue <--> green.

L480: "SAE-learned directions operate in non-linear interaction spaces rather than simple superposition": what is the evidence supporting this?

L481: "steering effects do not transfer well to models after supervised fine-tuning": was this shown anywhere?

L863-864: Fig. 9 right and Fig. 10 right do not seem to show this.

Appendix A.5 is empty

Fig. 14 caption: what does coefficient 18 mean?

### Questions
See above.

Also: why PPO rather than DPO or GRPO?

### Soundness
1

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces a framework called "Control Reinforcement Learning" (CRL) , aimed at interpretable, token-level dynamic steering of LLMs . The core of this method is to use "monosemantic features" extracted from LLM activations by Sparse Autoencoders (SAEs) as a control interface. The CRL framework trains a RL policy network that, at each generation step, observes the current token's residual stream activation and dynamically selects an SAE feature. Then, the decoded vector of this selected feature is added back into the model's residual stream to "steer" the model towards a better output. The method aims to solve the "emergent misalignment" problem that occurs during LLM inference and has achieved modest performance improvements on various benchmarks (such as MMLU, GSM8K, BBQ bias, and HarmBench safety) , while providing interpretability by tracking feature contributions.

### Strengths
1. The paper addresses a very forward-looking and critical problem in the field of LLM alignment: how to achieve dynamic, interpretable, token-level control over model behavior.

2. Using monosemantic features extracted by SAEs as the action space for RL is an innovative idea 

3. The paper provides qualitative evidence to support its claim of "interpretable steering"

### Weaknesses
I am not an expert in the RL area. Therefore, the following comments are based on my current understanding, and I welcome any corrections to potential misunderstandings on my part.

1. The abstract mentions the use of "Adaptive Feature Masking" (AFM) to balance exploration and exploitation, but this key component is never mentioned or defined again in the paper's main body, experiment, or appendix.

2. Section 3.2 defines the intervention as $\tilde{x}_{t}=x_{t}+a_{t}W_{dec}$ which implies a steering coefficient of 1. However, Experiment Section 4.2 dedicates significant analysis to "steering coefficients" ranging from 10 to 100. This coefficient c, which is critical in the experiments, is not defined in the methodology.

3. The paper only compares CRL's results against the "Before" model (i.e., the original, non-intervened model). It completely lacks a comparison against standard fine-tuning methods (like SFT or DPO) on the same task data. 

4. The paper admits in its conclusion that these steering effects "do not transfer well" to models after supervised fine-tuning (SFT). This severely limits the method's practical application value in real-world systems that require continuous updates and iteration.

### Questions
No

### Soundness
2

### Presentation
2

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
The paper proposes Control Reinforcement Learning (CRL): a PPO-trained policy that, at each generation step, selects SAE feature(s) at a chosen layer and adds the corresponding decoder vector to the residual stream, thereby "steering" the model token-by-token. The state is the current residual activation; the action is a binary selection over the SAE dictionary; rewards are task-specific. Reported gains on Gemma-2 2B are modest but non-trivial on some tasks (e.g., HarmBench +5.61-pt, BBQ-Ambig +3.55-pt; others are small). The paper also analyzes (i) layer/coefficients ("sweet spots" in later layers), and (ii) critic behavior (bottlenecks for single-token tasks vs gradual divergence in long-horizon reasoning).

### Strengths
1. Clear control interface over interpretable features. Formulating steering as an MDP over SAE features with per-token actions is neat and practically implementable. The action/state definitions and the steering equation are explicit.

2. The paper gives useful empirical guidance: later layers tolerate larger coefficients; early-layer large coefficients tend to break behavior ("sweet-spot" effect), aligning with residual-norm growth across depth.

3. Feature "impact" and diversity metrics provide some transparency into which SAE features are used when steering helps or hurts.

### Weaknesses
1. It is dissatisfying to see that most headline improvements are small. Also, there are no confidence intervals, multiple-seed runs, or bootstrap tests. I couldn't be confident about the stability of the gains.

2. The paper itself notes that on single-token QA without constrained decoding, a substantial portion of MMLU gains comes from eliminating invalid outputs (e.g., "*", whitespace) rather than improving knowledge. That weakens the claim that CRL improves reasoning/knowledge rather than format adherence.

3. Since CRL's core novelty is adaptive selection of interpretable features, it needs stronger ablations vs. simple static/greedy heuristics (e.g., always add the top-k SAE features by activation, or by a supervised classifier over features), and vs. logit-space steering matched for compute. The paper doesn’t convincingly isolate the benefit of PPO-based selection over such cheap alternatives.

4. The authors report critic "bottlenecks" (corrected vs. misguided nearly indistinguishable on MMLU), suggesting value estimation struggles when rewards are sparse and binary. That weakens the paper's promise that CRL delivers reliable token-wise interpretability. If the critic can't separate outcomes, the per-token attributions are noisy.

### Questions
1. First, can you provide CIs / multiple seeds and report per-task variance; which gains survive across seeds?

2. If resource permits, could you add non-RL baselines (e.g., pick top-k SAE features by current activation so that total intervention norm equals CRL's)? Also, could you add a format-sanitizer that only enforces valid answer formats to quantify the fraction of gains due to formatting?

Conditional on satisfactorily addressing the above points, I am open to increasing my rating.

### Soundness
3

### Presentation
4

### Contribution
2
