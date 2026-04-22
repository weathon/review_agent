# Social-R1: Enhancing Social Intelligence in LLMs through Human-like Reinforced Reasoning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 4, 8

## Abstract
Recent advances in reinforcement learning with verifiable rewards (RLVF) have elicited strong reasoning abilities in large language models (LLMs) on objective tasks such as math and coding, yet social intelligence—the capacity to perceive social cues, infer others’ mental states, and interact effectively—remains underexplored. We argue that progress has been hindered by the simplicity and homogeneity of existing social datasets, which incentivize shortcut solutions over genuine Theory-of-Mind (ToM) reasoning. To address this, we introduce \textbf{ToMBench-Hard}, a challenging, multi-dimensional multiple-choice benchmark that rigorously evaluates ToM (e.g., perspective-taking, belief revision, and deception), exposes limitations of current LLMs, and provides verifiable outcomes for reinforcement learning. Training with RLVF on ToMBench-Hard using only outcome-based rewards already yields clear improvements. Motivated by the role of human-like mental processes in social cognition, we further collect diverse reasoning trajectories and train a social thinking reward model that scores trajectory quality—rewarding accurate perception of social cues and ToM-consistent inference prior to answer generation. We combine these signals in \textbf{Social-R1}, a reinforcement learning framework for social reasoning that integrates outcome and trajectory-level rewards. Across SocialIQA, SimpleToM, EmoBench, and MotiveBench, Social-R1 consistently outperforms strong reasoning LLMs; notably, Social-R1-4B surpasses LLaMA3-70B on all benchmarks despite the latter having more than ten times as many parameters. These results show that outcome-based RLVF substantially improves LLMs’ social reasoning while process-level thinking rewards provide additional gains, underscoring the importance of supervising the reasoning trajectory to foster human-like social intelligence in language models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Social-R1, a post-training framework for social reasoning that augments outcome-based RL with a trajectory-level “thinking reward.” The authors introduce a harder Theory-of-Mind benchmark (ToMBench-Hard), collect ~6.3k scored reasoning traces to train a thinking-reward model, and optimize with GRPO using a linear mixture of format, outcome, and thinking rewards. Experiments report gains over outcome-only training and competitive results for a 4B model against larger open/closed baselines on multiple social-reasoning benchmarks.

### Strengths
1. Clear motivation for combining process-level and outcome-level supervision in social reasoning.

2. A harder ToM benchmark aimed at multi-party, context-dependent scenarios with a measurable human–model gap.

### Weaknesses
1. Coverage imbalance: Despite claiming broad coverage of ToM sub-skills, the dataset distribution is heavily skewed toward certain categories (e.g., Intention) with much lower representation for others (e.g., Desire, Knowledge, non-literal communication). This undermines the claim of comprehensive coverage and risks biasing both training and evaluation toward high-frequency sub-skills.

2. Ground-truth reliability vs. human ceiling: Human accuracy on several tasks hovers around ~0.9 rather than near 1.0, yet the benchmark adopts single canonical labels. In nuanced social-semantics settings, this gap raises doubts about the uniqueness and reliability of “ground truth,” especially when humans themselves are not at ceiling.

3. Anomalies in Table 1 challenge construct validity: (i) For multiple Qwen variants, “disable thinking” scores exceed “thinking” scores, contrary to the expected benefit of explicit reasoning in ToM; (ii) Qwen3-4B with “disable thinking” achieves unusually strong Intention performance, surpassing many larger or closed models; (iii) Weak correlation between ToMBench-Hard and ToM-RL across the same models. These patterns call into question the benchmark’s construct validity and content coverage.

4. Underspecified thinking-reward data collection: The description around how the ~6,300 trajectories were produced is confusing and incomplete (what questions, which models, how many per item, sampling settings, prompt formats, selection/cleanup criteria, and scoring protocol). Without these details, it is unclear whether the collected trajectories faithfully represent the training distribution, whether there are prompt/model-specific biases, and how consistent the scoring actually is.

5. Potential circularity in reward modeling: Using one proprietary model to draft “gold” trajectories and another to score candidate trajectories risks imprinting upstream model priors and stylistic preferences into the reward signal, weakening the independence and interpretability of the proposed thinking-quality measure.

6. Insufficient justification for reward fusion with GRPO: The final reward is a linear mix with a sigmoid-transformed thinking component, but there is no principled rationale for this specific combination or its scaling. Given group-normalized advantages, fluctuations or noise in the thinking reward can materially shift credit assignment, creating a pathway for overfitting to the reward model rather than improving genuine social reasoning.

7. Ambiguous attribution in Figure 2: The reported gains track closely with the increasing influence of the thinking reward, but the reliability of that reward is uncertain. The evidence does not disentangle genuine reasoning improvements from potential overfitting to the reward signal’s idiosyncrasies.

8. Precedent and scope for PRM+GRPO remain unclear: The manuscript does not position the “process reward merged into GRPO” setup relative to prior art or delineate conditions under which this integration is expected to be stable vs. brittle. As a result, it is difficult to judge whether the approach is robust beyond the particular setting studied here.

9. Train–test separation concerns: The text includes phrasing that suggests training on the test set in at least one place, and the overall pipeline trains on data from the same family as the evaluated benchmark. The write-up does not provide a crisp account that rules out leakage across RL rollouts, reward-model training, prompt tuning, and final evaluation.

### Questions
The same as the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper aims to improve the social intelligence of Large Language Models (LLMs) by addressing the limitations of existing benchmarks and the lack of process-level supervision in training. The authors introduce two primary contributions: 1) ToMBench-Hard, a new multiple-choice benchmark for Theory-of-Mind (ToM) reasoning, designed to be more challenging and resistant to shortcut solutions; and 2) Social-R1, a reinforcement learning framework that integrates an outcome-based reward for the final answer's correctness with a trajectory-level 'thinking' reward for the quality of the reasoning process. This thinking reward is provided by a reward model trained on human-annotated reasoning trajectories. The authors show that their method improves performance on several social reasoning benchmarks, with a trained 4B model reportedly outperforming a 70B baseline.

### Strengths
Significance: The paper addresses a significant and timely problem: enhancing the social reasoning capabilities of LLMs. This is a crucial direction for developing more human-aligned and capable AI systems.

Originality: The core idea of integrating process-level supervision alongside outcome-based rewards is a logical approach. Grounding the design of the thinking reward model in the Social Information Processing (SIP) theory is a commendable effort to connect machine learning techniques with established cognitive science frameworks.

Clarity: The paper is generally well-written and clearly structured. The motivation for the work and the description of the proposed framework are easy to follow.

Resources: The creation of the ToMBench-Hard dataset is a potentially valuable contribution to the community, as it could help researchers better evaluate the true social inference abilities of future models.

### Weaknesses
1. Lack of Data and Code for Verification: For a paper where a novel dataset (ToMBench-Hard) is a central contribution, the absence of supplementary materials containing the data is a critical flaw. Without access to the dataset, it is impossible for reviewers to independently verify its quality, assess its claimed "hardness," or check for potential artifacts and biases. This opacity undermines the foundation of the paper's experimental claims, as the results are entirely contingent on a resource that cannot be inspected.

2. Circularity in the Reward Model's Supervision: The social thinking reward model (TRM) is trained on data annotated by GPT-5. This introduces a significant methodological concern. The paper's own results (Table 1) show that GPT-5 struggles with ToMBench-Hard (scoring ~60%), suggesting it is not a reliable expert on this complex social reasoning task. Using a flawed model as the "judge" to generate supervision signals creates a circular loop: the Social-R1 framework may be learning to mimic GPT-5's specific reasoning patterns, rather than a more general or genuinely human-like form of social intelligence. The validity of the core "thinking reward" is therefore questionable.

3. Potentially Misleading Main Comparison: The headline claim that a 4B parameter model trained with Social-R1 surpasses a general-purpose LLaMA3-70B model is compelling but potentially misleading. The Social-R1-4B model has undergone intensive, specialized fine-tuning on the in-domain ToMBench-Hard dataset. In contrast, the LLaMA3-70B baseline is evaluated in a zero-shot or few-shot setting without this task-specific training. A more rigorous and fair comparison would involve applying the same fine-tuning procedure (at least with outcome-based rewards) to the LLaMA3-70B model to disentangle the benefits of the Social-R1 method from the benefits of simple in-domain specialization.

### Questions
My primary questions for the rebuttal period stem directly from the weaknesses identified above. Addressing them would be essential for a re-evaluation of the work.

 I am open to significantly revising my score upwards. A satisfactory response from the authors along with the discussion and opinions of the other reviewers, will be critical in shaping my final score.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces Social-R1, a novel reinforced reasoning framework designed to significantly enhance the social intelligence and Theory-of-Mind (ToM) capabilities of Large Language Models (LLMs). The core contribution includes presenting ToMBench-Hard, a challenging multiple-choice benchmark that rigorously tests complex ToM skills like deception and belief revision. Crucially, Social-R1 employs a combined reward mechanism: rewarding both the final outcome and the quality of the reasoning trajectory (thinking process), trained via a social thinking reward model. This methodology proved highly effective, enabling the smaller Social-R1-4B model to outperform the much larger LLaMA3-70B across all tested social intelligence benchmarks.

### Strengths
1. The introduction of the ToMBench-Hard benchmark, which is specifically designed for complex social scenarios and Theory-of-Mind (ToM) tasks (like deception and belief revision). It effectively prevents models from achieving high scores through simple pattern matching or statistical shortcuts, providing a more authentic and rigorous testing ground for evaluating LLM social intelligence.
2. Combining outcome-based rewards with trajectory-based thinking rewards is the correct direction to go.
3. The trained Social-R1-4B model (a smaller 4-billion-parameter model) comprehensively outperformed LLaMA3-70B (a much larger, state-of-the-art model with ten times the parameters) across all evaluated social intelligence benchmarks.

### Weaknesses
1. This paper only uses single modality, i.e., text. But multi-modality is significant for social reasoning.
2. The model performs well on textual benchmarks. But generalizing to real-world application needs to deal with multiple modalities.
3. Despite ToMBench-Hard being more challenging than existing benchmarks, it remains in a multiple-choice format. The multiple-choice nature is essentially a closed-ended evaluation, which may not fully capture the model's true social competence in open-ended, free-form interaction, or generative tasks. The model might be proficient at distinguishing between provided options but fail to exhibit original, appropriate social responses in broader social contexts.
4. Reinforcement Learning with Verified Rewards (RLVF) training, especially when combined with an additional reward model (the Social Thinking Reward Model), is typically more expensive and complex than traditional Supervised Learning (SL) and Instruction Tuning. This high computational cost may limit the application and reproducibility of the Social-R1 framework across the broader research community or for institutions with limited resources.

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Social-R1, a reinforcement learning framework for improving social intelligence in LLMs. The work makes three main contributions: ToMBench-Hard, a 900-question Theory of Mind benchmark spanning 6 dimensions; a social thinking reward model trained reasoning trajectories that are designed according to Social Information Processing (SIP) theory; and Social-R1, a reinforcement learning framework training framework that combines outcome-level and trajectory-level rewards using GRPO optimization. The authors find that the Social-R1 consistently outperforms strong reasoning LLMs across SocialIQA, SimpleToM, EmoBench, and MotiveBench and that this framework enables much smaller models to surpass or match larger models on these benchmarks. These results provide evidence that process-level thinking rewards should be used to supplement outcome-level rewards for human-like social intelligence in language models.

### Strengths
Overall, this paper is well-motivated and well-written, presenting a benchmark that is valuable for its comprehensiveness and complexity. ToMBench-Hard addresses real limitations in existing ToM evaluations through careful human curation across six ability dimensions (spanning emotion, desire, intention, knowledge, belief, and non-literal communication), achieving meaningful difficulty calibration where humans score 87% while state-of-the-art models struggle below 64%. The work makes an important empirical contribution by demonstrating that RLVF, previously confined to objective domains like math and coding, can effectively improve social reasoning and that process/reasoning-level rewards provide additional gains to solely outcome-based rewards. The results are compelling, particularly the finding that Social-R1-4B outperforms models with 10x more parameters across all benchmarks with consistent out-of-domain transfer.

### Weaknesses
The authors state that existing benchmarks have "exploitable patterns" and enable "superficial shortcuts" but don't provide much insight into what these are. Manipulations of perceptual access and asymmetric information may increase difficulty, but its not clear that this will prevent shortcut solutions or just allow models to index on different kinds of features (like transparency). Addressing this claim in a more systematic way would significantly strengthen the paper: 
- Categorize specific exploitable patterns in existing benchmarks (Hi-ToM, ToMBench, etc.) and show that models exploit these patterns or provide citations to this effect (apologies if I missed this)
- Explain the design choice in ToMBench-Hard that addresses this and perform ablations showing that removing that design element reintroduces the shortcut
- Experiments verifying that the models aren't using new shortcuts.

Readers could benefit from more insight into the authors process in constructing reward model and RL optimization, as well as more analysis of the reward function itself:
- The reward design seems highly dependent on the authors choice of SIP and this choice isn't explained. Why not simulation theory (Goldman, 2006) or Theory-Theory (Gopnik & Wellman, 1994)? Explaining why SIP was chosen, and if possible, experiments comparing reward models informed by these other competing theories of ToM abilities in humans would strengthen this work.
- It seems like the outcome and format rewards plateau but the thinking reward does not -- can the authors provide any intuition for this? possible connection to the binarization of the thinking reward? Is there anything interpretable about the weighting coefficients that are learned that could explain the behavior of the reward function.

### Questions
In addition to addressing the points from weaknesses some minor points: 
- There are a number of minor capitalization (line 156) and spelling issues, as well as missing/incorrect words especially in section 3.2 that made the text there a bit difficult to read. 
- Section 3.3: Why was the reward model initialized from Qwen3-4B? Do the authors think the results would change with a different choice of reward model? 
- I assume the sigmoid function only applied to the thinking reward in $R_i$ because the format and outcome rewards are already binary, but this could be made a bit more explicit.

### Soundness
3

### Presentation
3

### Contribution
3
