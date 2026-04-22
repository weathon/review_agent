# Nonlinear Steering for Token-Efficient Reasoning in LLMs via Flow Matching

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4

## Abstract
Large Reasoning Models (LRMs) excel at complex reasoning tasks, but their efficiency is often hampered by overly verbose outputs. Prior steering methods attempt to address this issue by applying a single, global vector to hidden representations—a rigid approach grounded in the restrictive *linear representation hypothesis*. In this work, we introduce *FlowSteer*, a nonlinear steering method that goes beyond uniform linear shifts by learning a complete *transformation between the distributions* associated with verbose and concise reasoning. This transformation is learned via *Flow Matching* as a velocity field, enabling precise, input-dependent control over the model's reasoning process. Across diverse reasoning benchmarks, *FlowSteer* simultaneously achieves superior accuracy and token efficiency over leading inference-time baselines. Our work demonstrates that modeling the full distributional transport with powerful generative techniques offers a more effective and principled foundation for controlling LRMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes using Flow Matching for LRM steering. The authors explored distribution alignment with flow matching, as well as methods for robust training. They followed the SEAL methodology of using verbose and concise CoTs to train their model. The results show improvements in both model quality and reduced length of CoTs.

### Strengths
- The paper is generally well written, with sufficient details provided (e.g., examples of reasoning, training details, etc.).
- The idea is novel and interesting for further research.

### Weaknesses
- My main concern with this paper is the stated dichotomy between the SEAL intervention protocol and training with RL (L451). More concretely, the method ultimately learns steering vectors based on two sets of representations (verbose and concise). From my point of view, this does not differ much from an RL setup, as we still rely on sampling from a non-trained model and training interventions.  From this perspective, I find it premature to state that the linear representation hypothesis ignores complexities in a model (L043). This also reduces the depth of the work, as it remains unclear whether one should bother with sophisticated methods or simply use linear interventions (without relying on a pre-defined set of verbose and concise representations).

### Questions
See weaknesses

### Soundness
3

### Presentation
4

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
The paper introduces a method for steering large language models toward concise, accurate reasoning by learning a nonlinear transformation of hidden states via flow matching. The method models a velocity field that transports internal representations from verbose to concise reasoning styles. It incorporates robust training techniques and a novel guidance mechanism based on Gaussian score matching to improve reliability and coverage. Tested on math and coding tasks, it outperforms existing inference-time baselines.

### Strengths
* The method is well-theoretically grounded, and the motivation to learn nonlinear transformation of hidden representations sounds clear and logical
* The method has a low overhead in computing resources and time

### Weaknesses
* The paper's probabilistic guidance relies on approximating hidden state distributions as diagonal Gaussians—a simplification that may not fully capture the true geometry of LLM representations. The authors justify this assumption by citing prior work showing Gaussian-like behavior in the final-layer activations of RNNs and vision models [1, 2]. However, it remains unclear whether this finding reliably extends to Transformer-based language models pretrained on diverse textual data. Is this assumption too strong, given the potential multimodality or anisotropy of hidden states in large-scale LLMs?
* The method’s effectiveness depends on access to a training dataset containing paired verbose and concise reasoning traces, from which the flow model learns to steer representations. In this work, the authors use the MATH dataset to construct these training pairs, and then apply the learned steering across multiple domains including code generation without further adaptation. While the method generalizes reasonably well in their experiments, it’s unclear how well the method would perform on other domains.
* The obvious way to solve the problem of moving from verbose to concise reasoning is RL. For the honesty of the experiment, it is also possible to train only steering vector using RL, as it is done in the article [3]. Although this method is not inference-time, it requires very little resources and time.
* The results presented in the paper do not include standard deviations or any assessment of statistical significance.

[1] Hashemi et al, Gaussian-based runtime detection of out-of-distribution inputs for neural networks.

[2] Zhang et al, Finegrained neural network explanation by identifying input features with predictive information. 

[3] Sinii et al, Steering LLM Reasoning Through Bias-Only Adaptation

### Questions
See weaknesses

### Soundness
3

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
3

### Summary
The paper introduces a method for steering reasoning models using flow matching. The setup resembles SEAL, but rather than steering in a single direction, Flow Steer learns to match the distribution between long and short answers by partitioning reasoning traces into three phases: execution, reflection, and transition.

### Strengths
- The core idea is interesting and produces better results than baselines in most settings.

- The evaluation appears sound, with a comprehensive set of baselines.

### Weaknesses
- Section 3 introduces several modifications to simple flow matching—median–IQR normalization, Huber loss in place of MSE, and probabilistic guidance—yet the ablation studies cover only the last. For example, what happens if standard z‑score normalization $ \tilde{x} = (x - \mu)/ \sigma $ is used?
- The method relies on many heuristics/tricks but shows only marginal gains on many tasks/models; without clearer benefits. I can’t see it being widely adopted in modern pipelines.
- Data efficiency is unclear. How many training samples are required to train the flow-matching component, and from what scale does it outperform linear steering?

### Questions
- Replacing MSE with Huber loss appears to drop Gaussian-based optimality guarantees. Probabilistic guidance, however, assumes a Gaussian approximation and seems to enhance steering performance. Can the authors reconcile this tension and justify the loss choice empirically and/or theoretically?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors note that linear steering vectors transform representations in a way that discards higher-order distribution statistics, creating a mismatch. To address this, they propose Flow Matching, which is less restrictive. They highlight two challenges: massive activations and low-velocity zones, and tackle the former with normalization, a Huber loss, and optimal-transport coupling between source and target, and the latter with probabilistic guidance. Their strategy yields higher performance on several benchmarks than prior methods and achieves better distributional alignment.

### Strengths
* The paper is well written and easy to follow.
* It tackles the distribution-mismatch problem in a novel and interesting way.

### Weaknesses
### Methodological Problems

* **Small gains**

  In my experience, the gains reported in Table 1 for math benchmarks are often not statistically significant. Please add standard deviations to the table.

* **Narrow evaluation**

  The evaluation uses only Qwen-based models, which are known to yield observations that do not generalize broadly in math reasoning [1,2]. Please include evaluations on LLaMA models.

* **Incomplete efficiency evaluation**

  Section 4.5 compares only to the vanilla model, reporting reduced time per answer. What is the reduction relative to SEAL, which, per Table 1, often reduces token counts comparably to FlowSteer?
  Also, how much time/resources are required to train FlowSteer?

### Motivational Problems

I am concerned about the complexity of the proposed approach given the marginal gains -- training, substantial code, and an ODE solver are required. Could similar results be achieved by training, for example, an affine map, a LoRA, or a single full layer? Moreover, while I see the benefit of reducing distribution mismatch, I’m not convinced it is the primary obstacle preventing steering methods from achieving high performance. [3] show that when steering vectors are trained directly on the objective of interest, they can match full-weight training. The paper would benefit from a deeper discussion of its motivation and limitations in light of these two concerns.

## References

[1] Shao, Rulin, et al. “Spurious rewards: Rethinking training signals in RLvR.” arXiv:2506.10947 (2025).

[2] Liu, Zichen, et al. “Understanding R1-zero-like training: A critical perspective.” arXiv:2503.20783 (2025).

[3] Sinii, Viacheslav, et al. "Steering LLM Reasoning Through Bias-Only Adaptation." arXiv preprint arXiv:2505.18706 (2025).

### Questions
* Please clarify what a single source–target pair comprises and how each component is computed.

### Soundness
3

### Presentation
3

### Contribution
2
