# Multi-Resolution Skills For HRL Agents

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Hierarchical reinforcement learning depends on temporally abstract actions to solve long-horizon tasks.
We propose Multi-Resolution Skills (MRS), a simple and scalable approach that constructs a discrete set of skill modules, each specialized to predict subgoals at a fixed temporal horizon (e.g., 8, 16, 32, 64 steps).
Skill encoders share parameters, causing a minimal increase in model size while allowing each module to generate plans at a distinct temporal resolution.
A learned meta-controller selects among these resolution-specific skills based on the task context; the meta-controller and skill policies are trained jointly with a single end-to-end objective in a single training phase.
We evaluate MRS on DeepMind Control Suite, Gym-Robotics, and long-horizon AntMaze tasks.
While maintaining computational efficiency, MRS consistently outperforms single-resolution baselines, yields meaningful gains over the HRL baselines in long-horizon navigation, and remains competitive with the non-hierarchical state-of-the-art (SOTA) on standard benchmarks.
Ablations show that the multi-resolution design drives the improvement, suggesting temporal partitioning of skills is a useful inductive bias for HRL.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Multi-Resolution Skills (MRS), a hierarchical reinforcement learning (HRL) framework built upon the Director agent. The core idea is to replace Director's single Goal VAE with a set of parallel Conditional VAEs (CVAEs), each specialized to predict subgoals at a fixed, predefined temporal horizon (e.g., 8, 16, 32, 64 steps). These CVAEs share parameters for efficiency. A meta-controller policy, trained jointly and end-to-end with the skill policies, learns to select the appropriate temporal resolution based on the current state. The method is evaluated on DeepMind Control Suite, Gym-Robotics, and AntMaze tasks, showing improvements over the Director baseline in several cases and competitiveness with non-hierarchical SOTA methods like DreamerV3.

### Strengths
1.  **Empirical Gains on Challenging Tasks:** The paper demonstrates performance improvements over the Director baseline, particularly on tasks requiring longer-horizon reasoning or dealing with sparse rewards, such as the Gym-Robotics environments and the Egocentric Ant Maze tasks (Figures 4, 6). Achieving success in these domains from pixel inputs is a valuable contribution.
2.  **Well-Executed Ablation Study:** The ablation study presented in Figure 7 provides convincing evidence that both the multi-resolution structure and the learned meta-controller (skill interleaving) are crucial for the observed performance benefits, strengthening the core claim of the paper.
3.  **Practical HRL Framework:** The method builds upon a strong existing agent (Director) and maintains practicality through parameter sharing in the CVAEs and a single-phase, end-to-end training procedure. The inclusion of the $\infty$-length skill to mitigate policy collapse (Sec 4) is also a thoughtful practical consideration.
4.  **Exploration & Unsupervised Skill Learning:** The experiments showing effective skill learning using only the exploratory objective (Sec 5.2, Fig 8) and the emergence of interesting behaviors (Appendix C) are promising demonstrations of the framework's potential beyond standard task rewards.

### Weaknesses
Despite the empirical successes, several concerns prevent a stronger recommendation at this time:

1.  **Trivial Incremental Novelty:** The core architectural modification extends the single Goal VAE used in the Director baseline (Hafner et al., 2022) to N parallel CVAEs, each tied to a specific temporal horizon. While the parameter sharing and learned selection mechanism are well-implemented, this extension feels somewhat incremental from an architectural standpoint. It raises the question of whether simply providing multiple, fixed temporal skills constitutes a sufficiently novel contribution for ICLR without deeper justification. It's also un-convincing learning multi-res action with hand-crafted temporal dependencies through multi-head CVAEs is a better idea than established methods, as stated in point 2. Given the prevailing Diffusion models nowadays, encoding multi-res through a multi-head CVAE seems suboptimal, slightly off-track to appear on a top conference. I do acknowledge the emprical value and importance of this topic. But this kind of novelty needs hard rework to hold up to ICLR standard. I encourage authors to explore more SOTA archs.
2.  **Inconsistent Empirical Significance:** The evaluation is performed on a selected subset of DMC tasks. While MRS shows clear benefits on some tasks, the performance gains over the strong Director baseline appear marginal or comparable on roughly half of the standard benchmark tasks presented in Figure 4. A broader evaluation across more DMC tasks would be needed to make a stronger claim about general applicability and superiority. The performance difference compared to the non-hierarchical DreamerV3 is also often small, raising questions about the practical benefits of the added hierarchical complexity in these cases.
3.  **Lack of Theoretical Grounding:** The paper lacks theoretical justification for several aspects:
    * **Optimality:** There are no guarantees that the hierarchical policy learned by MRS converges to the optimal policy of the underlying MDP. The introduction of multiple CVAEs, a meta-controller, and intrinsic worker rewards fundamentally alters the optimization landscape and underlying process.
    * **Convergence:** Only a trivial policy gradient (Not Policy Gradient Theorem) is mechanismly derived through log grad trick for the proposed architecture, no proof of convergence for the overall learning algorithm (jointly training world model, CVAEs, manager, worker) is provided. Given the complexity, convergence is not guaranteed.
    * **Algorithm Details:** The paper states it uses Soft Actor-Critic (SAC) principles, but the implementation seems closer to an actor-critic method with an entropy bonus, missing key SAC components like soft Q-functions and soft Bellman updates in the critic training. The true distribution and variational distribution also not given. How SAC-like algo is derived is unclear. This requires clarification.
    * For strengthening the theoretical underpinnings, the authors might consider frameworks that explicitly handle the State Abstraction / SMDP->MDP relationship in options, potentially drawing inspiration from works like DbC (Zhang 2021) and DAC(Zhang 2019), HiT-MDP (Li 2023).
4.  **Limited Engagement with Related Temporal Abstraction / Option Literature:** The related work section provides a reasonable overview but could benefit from a deeper engagement with the broader options and skill discovery literature. Specifically, positioning MRS relative to methods that also learn temporally extended actions or deal with MDP formulations of options would strengthen the paper. Relevant works to consider include:
    * *DAC (Zhang & Whiteson, 2019):* Proposes an alternative MDP augmentation for learning options.
    * *Invariant Representation Learning (e.g., Zhang et al., 2020):* Formal proof of convergence and bounded optimality.
    * *Off-Policy Option Learning (e.g., Wulfmeier et al., 2021):* Discusses challenges and techniques for learning options efficiently from data.
    * *HIT-MDP (Li et al., 2023):* Provides a formal proof of equivalence between an MDP formulation and the SMDP option framework.

### Questions
1.  Could the authors provide results on a wider range of DMC tasks to better assess the generality of the performance improvements?
2. Can authors propose a more elegant architecture to learn temporal dependencies end-to-end in a unified architecture (diffusion-like), or any improvements to drop the cumbersome of training multi-head CVAE?
3.  Can the authors clarify the connection to SAC and justify the specific theoretical points raised above?

Overall, this paper presents an interesting and empirically promising approach to multi-resolution HRL. The results on sparse-reward and long-horizon tasks are encouraging, and the ablations support the design choices. However, concerns regarding the incrementality of the core idea, the consistency of empirical gains across standard benchmarks, the lack of deeper theoretical justification, and the limited positioning within the broader options literature currently place it marginally below the acceptance threshold for ICLR.

I would be willing to reconsider my score if the authors can provide one of stronger empirical evidence across a wider task suite / compare to stronger baslines (diffusion based models) / offer more theoretical insights or justifications for their approach, and more thoroughly contextualize their work within the existing HRL and options literature during the rebuttal phase.

### Soundness
2

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
4

### Summary
This paper proposes a method for hierarchical reinforcement learning (HRL) that extends Director (Hafner et al., 2022) with policies that predict sub-goals at multiple time horizons. The sub-goal policies are denoted as skills. The extension of Director involves replacing the Goal VAE with conditional VAEs that condition on the current goal and are trained to predict possible future states at a given horizon. The approach is evaluated in several RL simulation benchmark tasks and compared with Director and either DreamerV2 or V3. Improvements over these baselines models is demonstrated.

### Strengths
- The paper is well written and easy to follow. 
- The provided experimental results demonstrate improvements over the baselines (Director, Dreamer).
- An analysis of the performance of individual learned policies is provided which gives additional insights on the method.

### Weaknesses
- The paper misses comparison to other state-of-the-art HRL baselines such as HIRO (Nachum et al., 2018) or HiPPO (Li et al. Sub-policy Adaptation for Hierarchical Reinforcement Learning. ICLR 2020).
- The paper should discuss differences and relation to HiPPO, since it also proposes to learn subgoal policies for varying time horizons.
- The CVAE is trained online while the policy is adapting. Why can the CVAE cope with the domain shift? Please discuss.
- The motivation to learn only heads to a common backbone network for the CVAEs for multiple resolutions seems odd. How does the model perform with separate networks? Is there a benefit of sharing the same backbone in terms of performance?
- Eq 4, how is the discrete sampling of c_t implemented and made differentiable?
- l. 226 mentions “abstract state transitions”, however, there is no abstraction (actions or states) performed by the model. When abstracting states, one would assume that a new state space would be found, e.g., lower-dimensional or discrete. Please revise.
- l. 300 mentions that MRS retains compute efficiency of Director. Please quantify compute efficiency of the models (MRS, Dreamer, Director).
- l. 314, please consistently compare with DreamerV3.
- Sec 5.2 reports on modifying the manager policy after training and using random and single skill variants. This should not be named ablation, since it does not correspond to taking away parts of the full method which would involve retraining! Please discuss the difference.
- It is surprising that the model can learn backflips or somersaults. How were these skills selected and the corresponding behaviors generated? In Appendix C, somersaults are not visible. How can the model achieve such complex behavior just from an exploration objective? What is the incentive for the model to learn this? Please explain/discuss.
- l. 324 directly refers to Fig. 18 in the appendix which violates the page limit. A reference just to the appendix would be ok though.
- “While we do not draw any parallels…” – the paper should not try to suggest this by stating it in this way.
- The paper refers to the method as a “skill discovery framework”. The method learns subgoal policies. There is no definition of skill in the context of the paper. Also the paper does not learn a discrete set of skills or skills parametrized by a discrete latent skill variable. 
- l. 474, what are “recall capacity” and “option” in this context? The first bullet point is unclear.
- l. 485, the statement about the applicability of the multi-head policy gradient formulation is too vague and rather trivial. Please remove.

Minor comments:
- Eq. 1 should be provided directly after the first reference in the text.
- Fig. 3 caption: “sug-goals” => “sub-goals”
- Sec 3.4.2, please make the equation fit within the line width
- l. 376, “peack” => “peak”
- Fig. 8, what does “MSRD” and “MRSD” stand for?
- Adding plots from (Hafner et al. 2022) might violate copyright by the authors. Please obtain permission to reuse the plots or re-run the evaluation and make own plots.
- l. 476, “4” => “Fig. 4” ?

### Questions
- Major concerns arise from the unclear definition of the term “skill”, the missing comparison to state-of-the-art HRL methods, and the vague description of the qualitative skill evaluation (l. 418). See “Weaknesses” for further details on these issues and respective questions. 
- Please also address the remaining issues/questions listed in “Weaknesses”.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This article proposes an extension to the Director hierarchical RL architecture, where instead of having a single temporal horizon for target subgoals, there are several different subgoal networks that each propose a subgoal with a different temporal horizon. The higher-level policy network selects which of these the agent should chase. The resulting architecture is pretty straightforward and the method is tested across several domains, where it performs slightly better than Director, and competitive with Dreamer when Dreamer works, and works when Dreamer does not.

### Strengths
- The proposed idea is simple (which is a very good thing) and seems to work well, and would be a useful contribution to the literature.
- Multiple temporal resolutions for skills is an under-explored idea that has clear benefits and would help alleviate the brittleness somtimes associated with these systems.  
-  The neural implementation is well-thought-out and sensible.
- The set of domains tested is thorough.
- The computational and memory overhead of the method is pretty minimal. 
- The writing is pretty clear.

### Weaknesses
The major objection I have to the paper is that the experiments - while achieving wide coverage in terms of domains - are inadequate. For one thing, the only hierarchical  method compared against is Director. The authors claim that Director is SOTA (which is not a word, BTW, at least not in formal scientific writing) but comparisons against least HIQL (Park et al., Neurips 2023) and HAC (Levy et al., ICLR 2019) are warranted. HIQL is likely the best performing current method, and HAC is an older method but is closely related because it has a multi-level hierarchy at which each level has different temporal horizon. Missing these - which are just the first two I thought of - suggest that the authors have not done a very thorough survey of the related work.

Secondly, 3 seeds are nowhere near adequate for an experimental comparison. I think 10 seeds is a meaningful minimum.

My second objection is much less serious, but it is that, while the actual English in the paper is relatively clear, the paper has been produced very carelessly. It is aggressively irritating to constantly run into citations that should be parenthetical, but where the authors names have just been dumped into the text mid-sentence. There is almost no way that anyone who even looked at the PDF a single time would not have noticed this error, which occurs throughout. The paper is full of other, similar but less annoying, signs of carelessness, like using the acronym SOTA as if it was a word, use of "HRL" In the title, failure to punctuate equations, etc. This kind of thing does a disservice to what is otherwise good work. It also (and this is much worse) does a disservice to the (hopefully many) people who will eventually read it.

### Questions
Can you please run a comparison with at least HIQL and HAC, and explain the relationship between your work and multi-level hierarchies like HAC that are also increasingly abstract in time? And please increase the number of seeds.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes Multi-Resolution Skills (MRS), a hierarchical reinforcement learning framework that generates plans at multiple temporal resolutions. It is composed of two components: manager and worker. The worker is trained to reach goal state s_{t+l}. The manager is trained to generate the goals for different temporal steps (l) and selecting among these resolution-specific goals based on the task. Empirical results show that MRS outperforms single-resolution baselines on long-horizon navigation tasks and achieves performance comparable to non-hierarchical state-of-the-art agents on standard continuous-control benchmarks. Ablation studies confirm that the performance gains is indeed from the multi-resolution design.

### Strengths
* The manager and worker model are trained with shared layers (all except the resolution specific layer) to minimize the increase in model size.
* The framework uses CVAE so the predicted goal in future state is conditioned on the current state, this is to try and constraint the goal to to achievable futures. 
* The framework includes an escape/reset mechanism that is a ∞-horizon skill that is designed to be invoked in unusual or recovery states (e.g. after falling).

### Weaknesses
* The ablation study can benefit from including experiments comparing workers trained separate (i.e. an individual model for each temporal resolution) to show that the design choice of having shared layers do not significantly negatively impact the performance. 
* The work uses a single ELBO across horizons, but do not show clearly whether the decoder would be blurring for larger time step. If reconstruction is poor, the “goal” may be a fuzzy or unrealistic state, which would weaken the whole “constrain to achievable futures” story
* CVAE is trained online with data from the current worker, so it only recalls what the current worker can already do. That makes skills conservative. This means “skill = abstract action” is actually “skill = whatever the current policy happened to do.” That’s narrower than the claims.
* Compute/reporting transparency could be improved. They say experiments take “2 days … 5M steps on an RTX 5000” (App. B), but not for baselines. So we can’t tell if MRS is actually more sample-efficient or just more update-efficient.

### Questions
* In figure 2, the symbol for output of the decoder is s_*. Why not use the same notation \hat{s}_{t + l} as in figure 3 a)? 
* In figure 4, there seem to be an issue with the y axis on the quadruped run, pendulum swingup, cheetah run graph (shows .000 where it is supposed to be 1000) and on the push, puck n place graph (shows .100 where it is supposed to be -100).

### Soundness
3

### Presentation
3

### Contribution
3
