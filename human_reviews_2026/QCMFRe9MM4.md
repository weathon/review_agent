# Planning with Unified Multimodal Models

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
With the powerful reasoning capabilities of large language models (LLMs) and vision-language models (VLMs), many recent works have explored using them for decision-making. However, most of these approaches rely solely on language-based reasoning, which limits their ability to reason and make informed decisions. Recently, a promising new direction has emerged with unified multimodal models (UMMs), which support both multimodal inputs and outputs. We believe such models have greater potential for decision-making by enabling reasoning through generated visual content. To this end, we propose Uni-Plan, a planning framework built on UMMs. Within this framework, a single model simultaneously serves as the policy, dynamics model, and value function. In addition, to avoid hallucinations in dynamics predictions, we present a novel approach self-discriminated filtering, where the generative model serves as a self-discriminator to filter out invalid dynamics predictions. Experiments on long-horizon planning tasks show that Uni-Plan substantially improves success rates compared to VLM-based methods, while also showing strong data scalability, requiring no expert demonstrations and achieving better performance under the same training-data size. This work lays a foundation for future research in reasoning and decision-making with UMMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper argues that unified multimodal models (UMMs) can improve long‑horizon planning by treating a single generative model as policy, dynamics, and value, and by “reasoning through generated visual content.” It further proposes self‑discriminated filtering, where the same generative model filters its own dynamics predictions to reduce hallucinations.

### Strengths
- Clear articulation of the limitation of language‑only reasoning for decision‑making and a plausible motivation for multimodal reasoning.
 - The idea of integrating policy, dynamics, and value into a single UMM is conceptually coherent and could simplify interface complexity.
 - Self‑discriminated filtering targets a genuine pain point (hallucinated dynamics) and aims to avoid reliance on external validators.

### Weaknesses
- Self‑discriminated filtering may be circular: Having the generator judge its own rollouts can lead to correlated errors. Without an independent criterion (calibration, consistency checks across models, or grounded constraints), filtering may either accept coherent hallucinations or over‑reject plausible futures.
- How is uncertainty addressed in partially observable environments?
- Relation to prior model‑based RL and multimodal planning: The positioning relative to existing unified architectures (e.g., world models that produce latent visuals, diffusion planners with internal consistency checks) is underdeveloped.

### Questions
- How does the framework differ from world models that generate latent visuals and from diffusion‑based planners with internal consistency checks? 
- How does a UMM that reasons via generated visuals compare to a MuZero‑style latent‑space world model?

### Soundness
3

### Presentation
3

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
This work introduces Uni-Plan, a planning framework using unified multimodal models, as opposed to LLMs or VLMs. They propose one model to act as a policy, dynamics model and value function and introduce a self-discriminated filtering mechanism to filter out hallucinations. They evaluate their planning system against open source and closed source VLM-based planning methods, outperforming open-source benchmark and getting close to the closed source GPT-5-Thinking model, whilst their model has a much smaller number of parameters (14B).

### Strengths
1) The authors choose a comprehensive set of baselines for comparison, with both open-source and closed-source models. I particularly appreciate the results showcasing a similar performance to GPT-5-Thinking-Tool, a closed-source model, with a much more significant training dataset and compute budget. This way, authors provide an open alternative to the research community.
2) Compared to related work, such as VLP (video language planning), it proposes a single model for policy, dynamics and value function, thus reducing inference time. As expressed in the conclusion, there is, however, further work to be done to improve inference times, especially if this method aims to be applied to real-world planning scenarios.
3) Proposes a self-discriminator filtering mechanism to address and reduce hallucinations and strengthen the dynamics model quality. I also appreciated the additional object-count consistency check to further validate observation predictions.

### Weaknesses
1) Given it's a key component in the self-discriminated filtering mechanism, I would've liked to see a more detail on the inverse dynamics inference method used.
2) The authors state that the system addresses long-horizon planning tasks, but I could not see any definition of the choice of hyperparameter `H` for time-horizon and in all 3 environments it looks like the planning task is in the order of tens of steps or less. It would be good if authors could elaborate more on what they regard as long-horizon.
3) Even though I really appreciate the inclusion of the real-world planning example with the dual-arm robot in Section 3.1, I fear only one example in Fig 3 would not suffice to make a claim the planning framework applies reliably to such scenarios. I would've expected a quantitative analysis on the task (FID between imagined state and real execution).
4) Similarly, I would've liked to see numbers on the benefit of self-discriminating filtering across all environments (not just language task) in Fig 5. I'm not sure if they might be included in the appendix.
5) In Fig 4, where authors showcase finetuned BAGEL is a strong dynamics model, this is illustrated only as 1 short example (4 steps) - I believe a quantitative evaluation across all environments and multiple runs would make the claim stronger.

### Questions
-  *Suggestion*: There should be a small correction in section 3.3 - authors state Uni-Plan consistently achieves higher performance with fewer trajectories, but in Fig 6, on the language table, the performance on Uni-Plan with 500 traj is on par with (if not a bit lower than) the performance of Qwen2.5 with 1K trajectories. In the abstract and the conclusion, the claim is corrected to state "outperforming VLMs when trained with the **same** amount of data".
- *Suggestion*: It would be good to add the number of parameters for BAGEL-VLM in `Table 1` for clarity and completeness in the comparison.
- *Question*: Could the authors elaborate more on how the hyperparameters A, D, B and H are chosen? I see in the appendix the beams B are 2 for all environments - what is the rationale for choosing that value?

### Soundness
2

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
3

### Summary
This paper presents Uni-Plan, a planning framework built on unified multimodal models (UMMs) that jointly handle text and image reasoning for decision-making. Unlike prior language- or vision-language-based planners that rely mainly on textual reasoning, Uni-Plan employs a single UMM as the policy, dynamics model, value function, and self-discriminator. A key innovation is the proposed self-discriminated filtering mechanism, where the model validates its own predicted dynamics by comparing inferred inverse actions with executed ones, effectively reducing hallucinated transitions. Experiments on long-horizon planning benchmarks demonstrate that Uni-Plan achieves about 30% higher success rates than open-source VLM-based methods, matches the performance of the GPT-5-Thinking model, and shows strong data scalability without requiring expert demonstrations.

### Strengths
1. The paper convincingly demonstrates that a unified multimodal model can serve simultaneously as (i) a policy, (ii) a dynamics model, (iii) a self-discriminator, and (iv) a value function. This is a very interesting idea, and the paper provides solid evidence of its effectiveness.
2. Under this unified setting, the UMM framework significantly boosts the performance of the base VLM, showing clear advantages in planning and reasoning.

### Weaknesses
The experimental settings are relatively simple. All real-world and simulated environments are fully observed and easy to predict. I believe the main challenges in this domain lie in semantic understanding, which is largely powered by the base model. For example, a sufficiently strong base model such as GPT-5-Thinking-tool could outperform even a carefully designed architecture like the one proposed in this paper.

### Questions
1. Could you further demonstrate the dynamic model’s ability in more unseen environments, or under more challenging real-world conditions?
2. Could you provide additional real-robot experiments, such as pick-and-place tasks involving occlusion or unseen objects?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a high-level planning method that uses foundation models, specifically unified multi-modal models (UMMs). The UMM is used in four distinct roles: 1) as a policy, 2) as a dynamics model, 3) self-discriminator, and 4) value function.

The proposed method (Uni-Plan) utilizes an image of the environment's initial state and a textual description of the goal to produce a multi-step plan, where each step is executed by a low level controller (the learning of this controller is outside of the paper's scope).

At each step, the method involves 1) sampling A actions from the policy, 2) generating D next observations, and 3) filtering the D next observations down to 1 observation by tracing which action could have led to the next observations and matching it to the actual action. The authors do this for B beams.

### Strengths
1) The novel contributions include: 1) using a UMM, 2) fine-tuning with non-expert trajectories, and 3) using a UMM as a self-discriminator. These are all interesting contributions.
2) The proposed method shows high performance with much fewer trajectories—these results are promising!

### Weaknesses
1) The authors should cite the published version of the papers in their references (e.g., "Do as I can, not as I say: Grounding language in robotic affordances" was published at CoRL in 2022; "Inner monologue: Embodied reasoning through planning with language models" was published in CoRL in 2022; etc.).
2) The images of figure 3 are slightly out of focus and there is a black edge at the top of each photo that makes them appear unaligned horizontally.
3) No true empirical comparison to prior planning approaches with LLMs/VLMs. Experiments only vary the model type (VLM vs. UMM) and model size. Authors claim to have prior planning approaches as baselines, but I would argue that their baselines are VLM-based plan generators not prior approaches. For example, a good baseline for comparison would be SayCan [Ahn et al., 2022].
4) No statistical rigour.

### Questions
1) How many seeds/trials did you run? I see no mention of this and I see no reported standard error or standard deviation.
2) Feel free to address any weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
