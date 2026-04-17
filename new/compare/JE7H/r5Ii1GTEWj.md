---
job_id: c6ffdd01-29be-4944-9957-90d52c47a222
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: r5Ii1GTEWj.pdf
paper: Motion-R1: Latent-Intent Motion Generation with Physical Consistency
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is clearly about text-to-motion generation with reinforcement learning and representation learning components, well within ICLR’s scope.

## Minimum Quality
Pass ✅.  
All core sections (Abstract, Introduction, Related Work, Method/Approach, Experiments/Results, Conclusion) are present and in English. While there are notable technical and empirical weaknesses, they do not rise to the level of an automatic desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden prompts, instructions to reviewers, or other manipulative content in the provided manuscript.

---

# Expected Review Outcome:

## Summary

The paper introduces Motion-R1, a text-to-motion “policy generation” framework aiming to combine latent-intent reasoning with physically consistent motion synthesis. The approach has three pillars: (1) a Motion2Motion dataset of 7,132 text–motion dialogue samples annotated via an ERA-CoT entity–relationship analysis procedure; (2) an enhanced Group Relative Policy Optimization (GRPO) algorithm with a JS-divergence regularizer and a structured reward function for action and skill text generation; and (3) a low-level RL-based kinematic controller that is supposed to convert high-level motion descriptions into physically feasible trajectories. Experiments mostly evaluate the language side (action and skill generation) of the fine-tuned Qwen2.5-3B model, with JS versus KL divergence comparisons and one qualitative example of physical skill execution.

## Strengths

1. **Interesting problem formulation at the intersection of reasoning and physics-based motion.**  
   The attempt to bring R1-style reasoning optimization into text-conditioned motion generation and then bind it to a physics-based controller is conceptually appealing. The stated goal of moving beyond single-turn text prompts toward multi-turn, latent-intent-aware motion generation is timely and relevant.

2. **Motion2Motion dataset is at least a concrete artifact.**  
   The Motion2Motion dataset aggregates 7,132 annotated samples with actions, skills, and long-form “conversation” text (e.g., JSON example in Appendix A). ERA-CoT tries to systematically decompose dialogues into entities and relations, which is potentially reusable beyond this specific model.  
   - **Figure 2a/2b** provide some visibility into the dataset’s content distribution (skills word cloud and top-50 frequent words). While basic, this is still useful to understand the vocabulary and action diversity targeted.

3. **Clear mathematical objectives for the GRPO fine-tuning part.**  
   The high-level formulation of the GRPO objective in **Equation (3)**, the group-based advantage normalization in **Equation (4)**, and the JS divergence in **Equation (5)** are standard but clearly written. Likewise, the multi-component reward function in **Equations (6)–(10)** outlines the design intent (action precision, skill coherence, structural/format compliance). This gives a reasonably transparent view of how text outputs are scored.

4. **Empirical evidence that JS divergence helps over KL in the given setting.**  
   - **Table 1** (action generation) and **Table 2** (skill generation) consistently show small but non-trivial gains of “Our (JS)” over “Our (KL)” across metrics (e.g., SS from 0.2111 to 0.2178, Jaccard from 0.0531 to 0.0616, recall from 0.0876 to 0.1013).  
   - **Table 4** (GSM8K in Appendix B) shows a similar pattern, with JS outperforming KL under both 4-bit and 16-bit settings.  
   Together this supports the claim that JS regularization can be beneficial for structured generation in this implementation.

5. **Some qualitative demonstrations of skill understanding and motion execution.**  
   - **Table 3** and **Figure 3** jointly illustrate that the proposed model can parse a long text description (“forced entry” scene) and produce a “Kick the Door” skill which is then executed as a sequence of door-kicking frames, while the Anyskill baseline reportedly fails to do so.  
   - **Figures 4a and 4b** (GPT-4-based evaluation) show that the model is preferred for rationality and relevance in action/skill predictions, suggesting it is at least doing something qualitatively sensible.

6. **Positioning within R1 / RL-fine-tuning literature.**  
   The paper appropriately connects to GRPO and DeepSeek-R1, and tries to articulate why group-relative optimization and symmetric JS regularization may be better suited for structured, format-constrained outputs like XML/JSON motion policies.

## Weaknesses

1. **Very weak empirical validation of the *motion* aspects and physical consistency claims.**  
   Despite the title and narrative emphasizing physical consistency and low-level RL, almost all quantitative experiments in Section 4 are on *text generation metrics* or GSM8K math reasoning.  
   - There is no evaluation on standard text-to-motion datasets (e.g., HumanML3D, KIT-ML) or motion metrics (FID, diversity scores, acceleration smoothness, foot sliding, physical violation counts).  
   - The only motion-related empirical evidence is a single qualitative sequence in **Figure 3** (door kicking) and some high-level schematic in **Figure 1**. There are *no* quantitative comparisons to motion baselines like MDM, MotionDiffuse, MotionGPT/MotionGPT-2, MotionAgent, Anyskill, AMP-based controllers, etc.  
   This creates a large gap between the paper’s claims (“physically consistent latent-intent motion generation”) and the presented evidence, and makes it impossible to judge the physical consistency or motion quality beyond anecdotal cases.

2. **Low-level RL / kinematic optimization section is generic and under-specified, with no clear integration to the high-level text model.**  
   Section 3.3 describes a generic style+goal RL objective:  
   - **Equations (11)–(14)** are essentially the AMP / adversarial motion prior style of reward and a standard discounted RL objective. There is no detailed description of the state/action representations, joint limits, physics engine, character morphology, or how the high-level XML motion description is converted into goals \(g\) and task reward \(r_G\).  
   - It is also unclear how the discriminator \(D(s_t, s_{t+1})\) is trained: where do “expert demonstrations” come from, what dataset is used, and how does this tie to the annotated Motion2Motion dataset?  
   - The experiments do not report success rates of tasks, style scores, or any ablations for this low-level controller, even though it is framed as one of the three core pillars in **Figure 1**.  
   As written, this part reads more like a textbook summary of adversarial motion control than a concrete, reproducible component of the proposed system.

3. **Mathematical formulation issues and notation ambiguities in the GRPO and reward sections.**  
   - In **Equation (3)**, the PPO-style clipping is written as  
     \[
     \min\Big(\frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\epsilon, 1+\epsilon\Big) A_i
     \]
     which is problematic: the standard PPO objective is \(\min(r_i A_i,\ \text{clip}(r_i,1-\epsilon,1+\epsilon) A_i)\), i.e., the *min* is taken over two scalar terms that already include \(A_i\), not three scalar arguments \(r_i, 1-\epsilon, 1+\epsilon\). In the current form the gradient behavior w.r.t. \(A_i\) is unclear and, strictly speaking, dimensionally inconsistent if \(A_i\) can be negative. The authors need to clarify if this is a notational shortcut or a different algorithm, and if different, why it is correct.  
   - In **Equations (6)–(10)**, the symbols \(\alpha, \beta, \gamma\) are used as reward weights but \(\beta\) was *also* used as the JS regularization strength in **Equation (3)**, which is confusing. Then \(\alpha_t, \beta_t, \gamma_t\) are introduced in **Equation (10)** without explanation of how they relate to \(\alpha, \beta, \gamma\), or how they are scheduled over time.  
   - Operators like \(\Phi_{\text{action}}\), \(\Phi_{\text{skill}}\), \(\mathcal{S}_{\text{BERT}}\), and \(\Psi\) in **Equations (7)–(9)** are left abstract; there is no specification of architectures, training regimes, or even whether they are frozen or learned jointly. Since these functions define the rewards that drive RL, their ambiguity makes the proposed optimization impossible to reproduce and raises questions about the stability and correctness of the learning signal.

4. **Motion2Motion dataset construction is underspecified and only weakly analyzed.**  
   While Section 3.1 outlines ERA-CoT and provides some formulas, there are several concerns:  
   - **Data source and motion representation**: It is not clear where the 7,132 “human motion samples” come from (existing motion capture datasets, simulations, images, text-derived pseudo-motions, etc.), what their representation is (SMPL, joint angles, trajectories), or how they are aligned with the conversations. The JSON example in Appendix A only lists text fields (`action`, `skills`, `conversation`), no actual motion. If the motions themselves are not part of the dataset, this significantly undermines the claim that the dataset supports physically consistent motion generation.  
   - **ERA-CoT formulas**: **Equations (1)** and **(2)** use notation like \(R'_i\), \(v*th\), and \(V(i,j,k)\) without detailed definitions; the “Self-Consistency” procedure is only described qualitatively and seems to rely heavily on GPT-4-style LLMs, but the number of samples, prompts, and validation protocol are missing.  
   - **Dataset analysis**: Apart from the word cloud and frequency chart in **Figure 2**, there is no deeper statistical characterization (dialogue length distribution, number of skills per sample, taxonomy coverage) or comparison to existing instruction datasets (MotionGPT’s text corpus, Anyskill’s descriptions, etc.). Without such analysis, it is hard to judge whether Motion2Motion really fills a new gap or is just another medium-size LLM-generated dataset.

5. **Evaluation protocol is narrow and leans heavily on LLM-as-judge without safeguards.**  
   - **Tables 1 and 2** report metrics (SS, KMR, IC, Jaccard, precision, recall), but there is no description of how these are computed from text outputs. For example, what is “Information Completeness” exactly, and is “Semantic Similarity” computed with BERTScore or another model? There are no standard deviations or significance tests, so it is unclear whether differences like CPS 0.2117 vs 0.2176 are meaningful.  
   - **Figures 4a and 4b** use GPT-4 as an “impartial evaluator” for rationality and relevance, but the evaluation protocol is missing: number of prompts, randomization of model order, sampling temperature, number of judges, and whether pairwise comparisons or Likert scales were used. Using a single proprietary LLM as the only semantic judge is risky, especially when that same LLM family is also used in dataset construction.  
   - More critically, none of these metrics measure *physical plausibility* of the resulting motion; even the comparison against Anyskill in **Figure 3** seems to only show one scenario, with no aggregate statistics, and does not isolate whether improvements derive from better language understanding or better low-level control.

6. **Novelty on the RL / physics side appears limited relative to existing work, with key related work missing.**  
   The idea of combining language-conditioned motion generation with physics-based controllers and RL has been explored in several recent works that are not cited or discussed, including:  
   - Yue et al., *RL from Physical Feedback: Aligning Large Motion Models with Humanoid Control* (2025), which integrates physics-aware evaluation with text-conditioned motion.  
   - Han et al., *ReinDiffuse: Crafting Physically Plausible Motions with Reinforced Diffusion Model* (2025), which couples diffusion models with RL to enforce physical plausibility.  
   - Ren et al., *Towards Realistic Human Motion Prediction with Latent Diffusion and Physics-Based Models* (2025), and Xie et al., *Physics-Based Human Motion Estimation and Synthesis from Videos* (2021), which blend generative and physics-based motion modeling.  
   - Song et al., *Deep Reinforcement Learning for Modeling Human Locomotion Control in Neuromechanical Simulation* (2021), which provides a thorough treatment of RL locomotion control.  
   - Mir et al., *HuMouS* (2024) and Hu et al., *Efficient Text-driven Human Motion Generation via Latent Consistency Training* (2024), which address controlled and text-driven motion in latent spaces.  
   These efforts are very close in spirit to the paper’s low-level RL plus high-level text conditioning, yet the manuscript does not position itself against them, making the incremental contribution hard to assess.

7. **Overstated claims and ambiguous framing of “latent-intent” and “reasoning”.**  
   The paper repeatedly states that it is the “first attempt to explore the R1 paradigm for physically consistent latent-intent motion generation” and implies substantial gains in reasoning. However:  
   - The reasoning improvement evidence boils down to moderate text metrics and GSM8K performance in **Table 4**, with no direct link to better motion, physical feasibility, or multi-turn dialog coherence.  
   - The notion of “latent intent” is never formalized; in practice, the model is trained to output actions and skills from text, which is standard supervision, not clearly a latent-intent inference mechanism. ERA-CoT is described as extracting entity relations, but there are no ablations showing that ERA-CoT or the reasoning-specific parts (versus ordinary supervised fine-tuning) actually matter.  
   Overall, the paper’s rhetoric around reasoning and intent is stronger than what is substantiated by experiments.

8. **Clarity and reproducibility issues.**  
   Beyond the specific mathematical and dataset gaps listed above, a number of implementation details are missing: model architectures for reward embedding networks (\(\Phi_{\text{action}}, \Phi_{\text{skill}}\)), hyperparameters for GRPO (group size \(G\), clip \(\epsilon\), JS weight \(\beta\)), the exact XML schema and parsing rules used for \(R_{\text{format}}\), and full training schedules for both high-level and low-level policies.  
   **Figure 1** sketches the overall pipeline (“ERA-CoT”, “Improved GRPO”, “Low-level Optimization”) but the text does not fill in enough detail to make this a practically implementable system. This is especially problematic given that code is only promised but not available at review time.

## Potentially Missing Related Work

The following works are highly relevant but not cited or discussed; they should be incorporated into the related work and comparison sections:

1. **Yue, J., Wang, Z., Wang, Y. (2025), “RL from Physical Feedback: Aligning Large Motion Models with Humanoid Control.”**  
   Directly targets physics-aware alignment of large motion models with humanoid control via RL. This is conceptually very close to Motion-R1’s goal of binding a language model to a physics-based controller and should be discussed in Sections 2.1 and 3.3 and, ideally, serve as a comparison point for the low-level optimization framework.

2. **Han, S., Wang, Y., Li, J. (2025), “ReinDiffuse: Crafting Physically Plausible Motions with Reinforced Diffusion Model.”**  
   Combines diffusion models with RL to enforce physical plausibility. It is relevant to the claim of being the first RL-based paradigm for improving physical consistency and should be cited in Section 2.1 and contrasted with the proposed GRPO approach.

3. **Ren, Z., Jin, M., Nie, H. (2025), “Towards Realistic Human Motion Prediction with Latent Diffusion and Physics-Based Models.”**  
   Explores blending latent diffusion with physics-based modeling for realistic motion, which is in line with the paper’s objective. It should be added in Section 2.1 when discussing physics-based and latent-space motion generation.

4. **Mir, A., Ding, J., Bakr, E. M. (2024), “HuMouS: Human Motion Synthesis with Fine-Grained Control using Latent Space Manipulation of Cycle-Consistent Diffusion Models.”**  
   Focuses on fine-grained motion control, directly relevant to latent-intent motion generation and controllability; appropriate to cite in Sections 1 and 2.1 and compare to Motion2Motion’s skill-level control.

5. **Hu, M., Zhu, M., Zhou, X. (2024), “Efficient Text-driven Human Motion Generation via Latent Consistency Training.”**  
   Proposes a text-driven motion framework built around latent consistency. It is relevant to the “latent intent” and reasoning angle and should be discussed in Sections 1 and 2.1.

6. **Xie, X., Xu, H., Wang, J. (2021), “Physics-Based Human Motion Estimation and Synthesis from Videos.”**  
   Addresses physics-based motion synthesis, relevant to the paper’s physical consistency goal. It should be added in Section 2.1 when describing physics-based motion generation.

7. **Song, S., Kidziński, Ł., Peng, X. B. (2021), “Deep Reinforcement Learning for Modeling Human Locomotion Control in Neuromechanical Simulation.”**  
   Gives a comprehensive RL approach to locomotion control. It should be referenced in Section 3.3 to ground the low-level policy training in prior RL locomotion literature.

8. **Zhang, Q., Ma, J., Liu, P. (2026), “MeshMimic: Geometry-Aware Humanoid Motion Learning through 3D Scene Reconstruction.”**  
   Deals with scene-aware motion and physical plausibility; relevant to any claims about environmental dynamics in Section 3.3.

9. **Hu, T., Jampani, V. (2026), “HumANDiff: Articulated Noise Diffusion for Motion-Consistent Human Video Generation.”**  
   Concerns motion-consistent video and could be mentioned in Section 2.1 as related work on motion consistency and articulation.

10. **Ouyang, R., Li, H., Zhang, Z. (2026), “Motion-R1: Enhancing Motion Generation with Decomposed Chain-of-Thought and RL Binding.”**  
    Appears to be closely related conceptually (R1-style reasoning for motion with RL binding). Ignoring it risks confusion and weakens the novelty claim; it should be positioned carefully in Sections 1 and 2.2 and differentiated in terms of method and evaluation.

## Questions

1. **Clarification of the GRPO objective and clipping behavior.**  
   Can you clarify whether **Equation (3)** is a typo or intentionally different from PPO? If different, please provide the exact implemented formula and explain how gradients behave when \(A_i < 0\). Including a pseudo-code snippet or algorithm box would help.

2. **Concrete details and ablations for Motion2Motion and ERA-CoT.**  
   - What are the exact sources and formats of the 7,132 “motion samples”? Do they contain 3D motion data, or only text?  
   - Could you provide statistics on dialogue lengths, average number of skills per sample, and coverage of your proposed taxonomy?  
   - Have you compared training with and without ERA-CoT-style annotations to quantify its impact?

3. **Low-level RL pipeline and its experimental evaluation.**  
   - How exactly do you map the XML motion description produced by the high-level model to the goal \(g\) and reward \(r_G\) in **Equation (11)**? Are there intermediate planners or symbolic parsers?  
   - Which character model, physics engine, and action space are used? How long are episodes, and what is the training curriculum?  
   - Can you provide quantitative metrics for the low-level controller (e.g., task success rate, energy expenditure, joint-limit violations, foot-sliding) and compare them to a baseline controller (e.g., AMP without GRPO-driven plans)?

4. **Evaluation design and significance of JS vs KL improvements.**  
   - For **Tables 1 and 2**, could you report variance or confidence intervals over multiple random seeds, and clarify how SS, KMR, IC, and Jaccard are computed?  
   - Are the ~3–8% relative improvements of “Our (JS)” over “Our (KL)” statistically significant? Any sensitivity to \(\beta\) or group size \(G\)?

5. **Broader motion benchmarks and comparisons.**  
   Do you plan to evaluate on standard text-to-motion benchmarks (e.g., HumanML3D, KIT-ML, BABEL) and compare quantitatively to MDM, MotionDiffuse, MotionGPT(-2), MotionAgent, and Anyskill? Even a subset of these would greatly strengthen the claim that Motion-R1 improves both semantic fidelity and physical plausibility.

Addressing these points with additional experiments and clarifications would significantly improve my confidence in the technical soundness and actual benefits of the proposed framework.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The high-level ideas are reasonable and many components are standard, but crucial details are missing, the GRPO objective is potentially misspecified, and the motion/physics claims are not adequately supported by experiments.

## Presentation Rating

2: fair.  
The paper is readable and figures/tables are generally interpretable (e.g., **Figures 1–5**, **Tables 1–4**), but there are many underspecified components, overloaded notation, and a lack of precise experimental descriptions that impede full understanding and reproducibility.

## Contribution Rating

2: fair.  
The conceptual framing of R1-style reasoning plus physics-based control for motion is interesting, and the dataset / JS-divergence ablations have some value, but the practical novelty and impact are limited by weak motion evaluation and insufficient differentiation from very close prior work.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The paper tackles an important and timely problem with an appealing conceptual pipeline and shows some evidence that JS divergence improves structured text-based action/skill generation. However, there is a substantial mismatch between strong claims about physically consistent latent-intent motion generation and the presented empirical support, especially on the motion and control side. Missing details, ambiguous math, and limited positioning relative to recent RL + physics-based motion literature further reduce confidence. With significantly stronger motion experiments, clearer formulations, and explicit comparisons to closely related works, this line of work could become more compelling.

## Reviewer Confidence

4: confident.  
I am familiar with text-to-motion, physics-based character control, and RL-fine-tuning of LLMs, and have carefully checked the equations and experimental descriptions. Some implementation details are missing, but the main issues identified are unlikely to stem from misunderstanding.