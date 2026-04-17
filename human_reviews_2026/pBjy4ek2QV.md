# Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
Reinforcement fine-tuning (RFT) often suffers from reward over-optimization, where a policy model hacks the reward signals to achieve high scores while producing low-quality outputs. Our theoretical analysis shows that the key lies in reward misspecification at the high-reward tail: the inability to reliably distinguish excellent responses from merely great ones. This motivate us to focus on the high-reward region. However, such tail examples are scarce under the base LLM. While off-policy exemplars (e.g. from stronger models or rewrites) are easier to obtain, naively training on them yields a misspecified reward for the policy we aim to align. To address this, we study rubric-based rewards. By design, rubrics can leverage off-policy examples while remaining insensitive to their artifacts. To elicit rubrics that capture the high-reward tail, we highlight the importance of distinguishing among great and diverse responses, and introduce a workflow to implement this idea. We empirically demonstrate that rubric-based rewards substantially mitigate reward over-optimization and deliver effective LLM post-training improvements.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the problem of reward over-optimization in the reinforcement fine-tuning (RFT) of Large Language Models (LLMs). This issue occurs when a policy model learns to achieve high scores from a proxy reward model without actually producing high-quality output, essentially "hacking" the reward. To address the issue, the authors propose a rubric-based reward refinement that distinguishes excellent responses from good ones, and distinguishes among a diverse set of off-policy responses.

### Strengths
The paper is well-motivated to address the reward hacking issue by designing a rubric-based reward system. The experiments are conducted across different settings (1/4 pairs, good v.s. great v.s. great and diverse pairs). There are also analyses on refinement types to give more detailed examination of the proposed method.

### Weaknesses
The technical components—LLMs as proposers/verifiers and RFT—are not very novel 

The paper's presentation could be improved. The initial motivation in Section 3 is well-reasoned but contains excessive detail better suited for the appendix. In contrast, Section 5.3 would benefit from more in-depth analysis of refinement types and their specific details.

### Questions
1. I'm curious if there are ways to attribute the language model performance gain to specific refinement types. For example, certain refinements, such as "Defining explicit scope, boundaries, or constraints" mentioned in Table 3 might be particularly useful and could be explicitly incorporated into the rubric-writing instructions.


2. Writing rubrics and generating evaluations can also be regarded as actions and optimized through reinforcement learning. With a rubric-writing LLM-as-judge, we could train it jointly with the base LLM. This could create an interesting multi-agent learning paradigm to co-evolve the rubric-writer and underlying LLMs. Any thoughts on such extensions?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an iterative rubric refinement framework (“Refinement-through-Differentiation”, RTD) to construct reward models that focus on the high-reward tail of LLM responses, aiming to mitigate reward over-optimization in reinforcement fine-tuning (RFT). The authors provide a theoretical justification that reward model accuracy in the top quantile of responses is the dominant factor for alignment success, and empirically show that rubric-based rewards—refined using high-quality off-policy responses—outperform baselines in both general and medical domains.

While the motivation is compelling and the empirical results appear promising, I have significant concerns regarding bias propagation, response diversity vs. superficial style differences, and the robustness of the “pair” construction strategy. These issues undermine confidence in whether the observed gains truly reflect improved capability discrimination rather than alignment to a specific LLM’s stylistic or judgmental preferences.

### Strengths
1. **Theoretical grounding**: The analysis in Section 3 provides a clean and insightful characterization of why high-reward region fidelity matters—this is a valuable contribution to the reward modeling literature.
2. **Practical relevance**: The rubric-based approach offers a data-efficient, interpretable alternative to Bradley-Terry models, especially in domains where human annotation is scarce (e.g., healthcare).
3. **Empirical rigor**: The ablation studies (Table 1, Figure 4) systematically validate the two proposed principles and show consistent gains across metrics.

### Weaknesses
1. The entire pipeline—initial rubric generation, top-2 selection, response scoring, and final evaluation—relies on GPT-4.1 as both proposer and judge. If GPT-4.1 systematically favors certain response styles or reasoning patterns, the refined rubric will encode and amplify this bias, leading to a self-consistent but potentially misleading reward signal.  



2. The paper defines “1 Good Pair” as two responses from Gemini-2.5-Flash-Lite, and “1 Great Pair” as two from Gemini-2.5-Pro. However, sampling twice from the same model (even with temperature > 0) often yields highly similar outputs, especially on factual or constrained tasks. In such cases, the differences may be trivial (e.g., word order, minor elaboration), leading the proposer LLM to “chase noise” rather than extract meaningful quality distinctions.  



3. When selecting top-2 from 16 diverse models (e.g., GPT-4o vs. Gemini-2.5-Pro), the responses often differ dramatically in length, tone, use of bullet points, emoji, hedging language, etc. While the rubric design guidelines discourage style-based criteria, the proposer LLM (GPT-4.1) may still interpret stylistic differences as quality signals (e.g., “more detailed = better”). The medical case study (Appendix I) is convincing, but it’s a single example, i hope the authors can provide broader quantitative evidence that style differences are not driving top-2 selections.


4. It remains unclear which models dominate the top-2 selections in the final iterative rounds. For instance, does GPT-5 or o1 consistently appear in the top pair? If one model (e.g., GPT-4.1 itself) dominates, the rubric may effectively become a proxy for that model’s behavior. Could the authors provide statistics on the distribution of models represented in the final top-2 pairs across all evaluated prompts?

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper addresses the critical issue of reward over-optimization in LLM post-training by proposing a rubric-based reward modeling approach. The authors first provide a theoretical analysis demonstrating that the accuracy of the reward model in the high-reward region is the key determinant of alignment performance. To tackle the sample inefficiency of obtaining such examples, they introduce an iterative workflow that refines scoring rubrics using off-policy responses from strong models. Experiments show that their method can mitigate reward over-optimization and improve model performance across diverse domains.

### Strengths
1. The paper provides a rigorous theoretical analysis that clearly links reward misspecification in the high-reward tail to the phenomenon of reward over-optimization. 
2. The authors proposed a novel and practical iterative rubric refinement strategy, which effectively leverages off-policy data to create more robust reward signals.
3. The experiments are solid. They have conducted experiments across diverse tasks and demonstrated the effectiveness of the proposed approach. The results show clear improvements in win-rate.

### Weaknesses
1. The use of a simple weighted average for aggregating rubric scores may limit the expressiveness of the reward function, although the paper acknowledges that more sophisticated aggregation schemes are an area for future work.

### Questions
1. Have you evaluated the generalization of your per-prompt rubrics to unseen prompts?

### Soundness
4

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
This paper investigates the problem of reward over optimization in large language model post training and identifies that the critical factor is not general reward model accuracy but its inability to reliably distinguish excellent responses from merely good ones in the high reward tail. To address this, the authors propose a rubric based reward framework that replaces traditional Bradley Terry models with structured, interpretable rubrics consisting of weighted binary criteria evaluated by another LLM. They introduce an iterative "Refinement through Differentiation" (RTD) method that progressively refines rubrics by comparing high quality and diverse off policy responses generated by stronger models. Experiments across general and healthcare domains demonstrate that focusing on these high reward regions significantly improves alignment performance boosting win rates, enhancing reward accuracy in the top quality range, and delaying reward over optimization thus validating rubric based reward modeling as a robust and explainable approach to mitigating misspecification in LLM post training.

### Strengths
- Clearly identifies that the core of reward over optimization lies in the ranking accuracy at the high reward tail  rather than overall error, and provides formal theoretical support.
- Proposes a complete pipeline combining rubric-based reward modeling, high-quality off-policy samples, and RTD iterative refinement, operationalizing “fine-grained differentiation in the high core region” into an automated and executable method.
- From theoretical proof of the high tail’s importance, to using stronger and more diverse responses for rubric optimization, to empirical results showing that improved high tail accuracy leads to more stable RL performance with less over-optimization.

### Weaknesses
- The generation and evaluation of rubrics heavily depend on stronger closed models such as GPT-4.1, essentially a “strong teacher supervising a weaker student.” It lacks systematic validation from human experts or multi-source reviews, which risks the propagation of biases from the teacher model, especially in sensitive domains such as healthcare.
- Heavy reliance on GPT-4.1 as the sole judge and labeler introduces an additional fragility: if GPT-4.1 systematically prefers the wrong answers that are the better response in certain patterns, these errors are directly baked into both the rubrics and the evaluation pipeline, and the method offers no robust mechanism to detect or correct such failure modes.
- The RTD + multi-round refinement + large-scale off-policy sampling pipeline incurs substantial computational and implementation cost; moreover, the trained model may learn to mimic rubric patterns or templates instead of genuinely improving reasoning ability. The paper lacks a systematic analysis of this potential “rubric-induced reward hacking.”
- Experiments are confined to a single model family (Qwen3-8B), a single judger GPT-4.1, and a few datasets, without comprehensive cross-scale, cross-lingual, or cross-domain validation. Furthermore, the paper does not include reasoning-intensive tasks (e.g., multi-step mathematical reasoning, code generation, or tool use), leaving it unclear whether high tail rubrics are equally effective at improving reasoning performance and mitigating over-optimization in such settings, making the conclusions more of a “strong case study” than a fully general solution.

### Questions
1.  Can the rubric-based rewards that have been constructed be effective in different fields (such as general, medical, and reasoning tasks)?
2. If the generation and verification of rubrics rely on closed-source models such as GPT-4.1, then how can we detect and correct any systematic errors in the evaluation model? 
3. For Figure 3, the green line 4 Great Pairs appears to have a better overall trend compared to the orange line 4 Great & Diverse. It is not that 4 Great & Diverse is superior to 4 Great Pairs. Please explain.
4. Why only use one model (GPT-4.1) as the proposer instead of using different models to provide more comprehensive evidence?

### Soundness
2

### Presentation
3

### Contribution
2
