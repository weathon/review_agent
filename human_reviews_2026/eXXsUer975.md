# Motion-R1: Enhancing Motion Generation with Decomposed Chain-of-Thought and RL Binding

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 2, 6, 6

## Abstract
Text-to-Motion generation has become a fundamental task in human-machine interaction, enabling the synthesis of realistic human motions from natural language descriptions. Although recent advances in large language models and reinforcement learning have contributed to high-quality motion generation, two major challenges remain. Existing approaches often fail to capture the temporal and causal complexities inherent in natural language, leading to oversimplified or incoherent motions. Additionally, RL-based methods are frequently overly complex, hindering their scalability and adaptability across various motion generation tasks.
    To address these challenges, we propose **Motion-R1**, a novel framework that combines decomposed Chain-of-Thought reasoning with reinforcement learning to enhance both the quality and interpretability of generated motions. Specifically, we introduce the **Decomposed CoT Data Engine**, which leverages an automated pipeline to synthesize high-quality reasoning data, allowing the model to better capture the temporal dependencies and causal relationships of human motion. We also propose **RL Binding**, a reinforcement learning strategy that incorporates multi-modal text-motion alignment into the RL reward function, guiding the model to produce motions that are both semantically accurate and motionally realistic. Extensive experiments across benchmark datasets demonstrate that Motion-R1 achieves state-of-the-art performance, with a 3.5\% improvement in MM-Dist on HumanML3D and improvements in R-Precision and FID on KIT-ML and BABEL, surpassing existing methods across key metrics and highlighting its superior capability in handling complex motion generation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces a novel framework called Motion-R1 for text-to-motion generation. It combines Decomposed CoT Data Engine and RL Binding. First it uses LLM to produce step-by-step motion descriptions. This helps enable better causal modeling. Then RL binding achieves multi-modal alignment rewards(motion similarity, semantic similarity, and format rewards) to improve the motion realism and semantic fidelity without human annotations. The method uses HumanML3D, KIT-ML, and BABEL datasets for evaluation, achieving competitive results across multiple metrics.

### Strengths
- Motion generation tasks have been struggled with generating long-sequence motions effectively. This paper innovatively addresses this issue by integrating CoT with motion generation.
- The RL Binding mechanism considers multiple reward signals. It prevents the generated motions from over-optimizing toward one specific aspect.
- The overall framework is highly innovative and interpretable, with a very natural idea. 
- The experimental results are highly convincing, demonstrating competitive performance across three mainstream datasets.

### Weaknesses
- The experiments are insufficient in terms of quality evaluation, as it provides few visual cases for **comparison** of different methods.

- The experimental section does not report the time consumption for generating a single motion.

- The article lacks a comprehensive review of methods combining human motion generation with reinforcement learning. Some approaches that employ an RL + motion generation framework, such as **AToM**, **InstructMotion**, **ReinDiffuse**, **MotionRL**, and **RLPF-MA**, were not mentioned in the article. I am aware that some of these methods are currently on Arxiv and have not yet undergone peer review. I am just offering a friendly reminder for the authors to **clearly outline the development trajectory of RL + Motion generation in the related work section**, so as to better highlight the innovative contributions of your method.

### Questions
- What is the time consumption for generating a single motion sequence with your method?

- Would using different LLMs have an impact on the results of your method?

- How does the innovation of your method compare to the five RL+Motion generation methods (**AToM**, **InstructMotion**, **ReinDiffuse**, **MotionRL**, and **RLPF-MA**) I mentioned?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents Motion-R1, a text-to-motion generation framework that integrates a Decomposed Chain-of-Thought (CoT) Data Engine, which uses LLM-generated reasoning traces to break complex instructions into stepwise sub-actions, and an RL Binding strategy, which applies multimodal alignment rewards for motion, semantic, and format consistency to guide reinforcement learning without human preference labels. Experiments on HumanML3D, KIT-ML, and BABEL show that Motion-R1 achieves strong performance, with a 3.5% improvement in MM-Dist on HumanML3D and gains in R-Precision and FID on KIT-ML and BABEL.

### Strengths
1. Paper writing: the paper writing, figures and overall structure make the paper easy to follow.

2. Technical novelty: The RL Binding replaces human preference modeling with automatic motion/text similarity rewards is somewhat novel, but its effectiveness are not probably evaluated (see weaknesses).

3. Good quantitative results: Based on the reported results in Table 1, Motion-R1 achieves consistent gains across major metrics (yet, the improvement is not significant and worse in some metrics).

### Weaknesses
1. Unfair and incomplete comparison with Motion-Agent: The paper compares Motion-R1 only against MotionLLM, which is merely one internal component of Motion-Agent (Wu et al., 2024). Motion-Agent integrates MotionLLM with GPT-4o for reasoning, task decomposition, long-sequence composition, and interactive motion editing. Thus, the comparison omits the very agent capabilities that Motion-R1 aims to emulate with its CoT Data Engine and RL Binding. Claims of superiority are therefore not substantiated. Notably, this is not the first work that propose motion decomposition in the text-to-motion generation literature.

2. No evaluation against GPT-4o or similar reasoning LLMs: Both the CoT Data Engine and RL Binding could be replaced by prompting a powerful multimodal model like GPT-4o, which can already perform temporal decomposition and semantic alignment in a zero-shot way.
It is unclear whether Motion-R1’s improvements derive from genuine algorithmic novelty or simply from additional fine-tuning.

3. Lack of evidence for RL Binding effectiveness: While RL Binding is presented as the key to text–motion alignment, the ablation in Table 2 is limited, as it simply removes this component from the framework. A fairer comparison would replace it with a non-RL or purely supervised variant (using existing text–motion pair datasets for training, as done in MotionGPT, MoMask, or MotionLLM). Without such comparisons, it is impossible to determine whether RL Binding materially improves alignment or merely introduces minor regularization.

4. No mechanism or evaluation for temporal smoothness between sub-motions: Because Motion-R1 decomposes a description into multiple sub-actions, a crucial question is how continuity and smooth transitions between these segments are enforced. The paper does not describe any temporal-smoothing module, blending loss, or transition constraint, nor does it evaluate transition quality. This raises practical concerns: even if each sub-motion is semantically correct, discontinuities between them could produce unrealistic or jerky results.

5. Scope and novelty: The proposed framework mainly recombines existing elements: LLM-based reasoning, VQ-VAE motion tokens, and RL optimization, without introducing a fundamentally new generation paradigm. Its novelty claim relies on integration rather than on new algorithmic insight, and the unfair comparisons and evaluations weakens its empirical significance.

### Questions
Please address my comments regarding the weaknesses of the paper.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Motion-R1 is a lightweight RL framework for text-to-motion generation that circumvents costly human-annotated rewards. An LLM-powered decomposed chain-of-thought engine first converts high-level captions into temporally ordered, fine-grained reasoning sequences, yielding abundant synthetic supervision. Reinforcement learning is then reduced to a simple binding stage that optimizes a composite reward combining motion-similarity against ground-truth and semantic-similarity to the text. Extensive experiments on HumanML3D, KIT-ML and BABEL show that Motion-R1 attains state-of-the-art R-Precision, FID and MM-Dist while requiring no human preference data.

### Strengths
1. Elegant and Effective Methodology: The core contribution—combining an automated CoT data generation pipeline with a streamlined RL mechanism—directly addresses a clear limitation of prior end-to-end models, which often struggle to interpret and execute multi-step or complex instructions. 
2. Eliminates the Need for Human Annotation: RL Binding obviates the need for costly, time-consuming, and often subjective human preference labeling. By cleverly using the existing ground-truth data (text and motion) to formulate the reward function, the framework becomes more scalable and efficient.
3. Strong Empirical Performance: The paper provides robust empirical evidence to support its claims. Motion-R1 achieves state-of-the-art results on multiple standard benchmarks. The quantitative tables show consistent improvements, and the qualitative visualizations (Figure 3) are particularly compelling, clearly illustrating the model's superior ability to handle complex, out-of-distribution prompts compared to a strong baseline (MotionLLM).

### Weaknesses
Dependency on External LLM Quality: The performance of the entire framework is fundamentally tied to the reasoning and decomposition capabilities of the LLM used in the Decomposed CoT Data Engine. The paper acknowledges that the LLM can produce "noisy or suboptimal plans," which could introduce errors into the training data. This dependency might limit the reproducibility and robustness of the approach if a different or less capable LLM is used.
Inconsistency LLM output across multiple decompositions: A prompt can be decomposed into different numbers of steps with varying content and granularity across multiple generation attempts. This phenomenon can be readily reproduced by reviewers and readers. Such inconsistency introduces noise and ambiguity into the training data, as the model is taught that a single instruction can map to many different, and sometimes conflicting, action plans. This could lead to the model learning an "averaged" and incoherent motion policy or producing unpredictable results at inference time.

### Questions
1. How does the framework address the high variability in CoT decompositions for complex prompts like "doing tai-chi"?
2. Does the author have specific mechanisms to enforce consistency or select a single "best" decomposition from many variants? What are the criteria for this selection?
3. Have the authors considered releasing the generated CoT dataset or the filtering scripts to allow for further analysis of this issue by the community?
4. Have the authors analyzed the failure cases that arise specifically from this data inconsistency? For instance, when trained on varied decompositions for the same concept, does the model generate hesitant, "averaged," or nonsensical motions? Showing such examples would provide valuable insight into the model's limitations.
5. Have the authors quantified the impact of external LLM quality on the final motion generation quality? For example, was a study conducted comparing the model's performance when trained on the "raw" noisy CoT dataset versus a manually cleaned or consistently-decomposed subset? This could help measure the model's sensitivity to the quality of the generated data.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Motion-R1, a novel framework for text-to-human-motion generation. It addresses key limitations in temporal reasoning and motion quality of existing methods through two core innovations: firstly, a Decomposed Chain-of-Thought Data Engine that leverages large language models to automatically break down complex language instructions into structured, step-by-step action plans, enhancing the model's understanding of temporal and causal relationships; secondly, an RL Binding strategy that integrates multi-modal alignment directly into the reinforcement learning reward function, using motion similarity, semantic similarity, and format rewards to guide the model towards generating motions that are both realistic and semantically faithful to the text, without relying on costly human annotations. Extensive experiments on multiple benchmark datasets including HumanML3D, KIT-ML, and BABEL demonstrate that Motion-R1 achieves state-of-the-art performance across key metrics such as R-Precision, FID, and Diversity, validating its superior capability in generating high-quality, semantically-aligned, and diverse motion sequences.

### Strengths
1. Skillfully combines the Chain-of-Thought paradigm from natural language processing with reinforcement learning, providing a novel and effective framework for applying large language models to complex motion generation tasks.
2. Not only achieves performance improvements but also provides transparency into the generation process through structured CoT reasoning steps, enhancing model interpretability and controllability—crucial aspects for practical deployment.
3. Directly tackles two core challenges in text-to-motion generation, temporal causal reasoning and motion quality optimization, by proposing practical solutions, demonstrating the authors' deep understanding of the field's fundamental problems.

### Weaknesses
1. The overall framework performance is highly dependent on the quality of CoT data generated by LLMs. Errors in LLM's understanding of certain actions could be amplified in subsequent processes, yet the paper lacks systematic analysis of such error propagation.
2. The three reward functions (format, motion, semantic), while intuitive and effective, may not cover all important aspects of motion generation, such as physical plausibility, energy consumption, style consistency, and other nuanced quality dimensions.

### Questions
1. Despite using DeepSeeker R1 for filtering, the CoT data generated by LLM inevitably contains errors or subjective biases. Has there been a manual sampling evaluation of the automatically generated CoT data? What is the approximate error rate? What observable effects do these noises have on model training?
2. The paper proposes three rewards: format, motion, and semantics. How are the weights of these rewards determined? Is it through grid search, empirical setting, or adaptive methods? Is there any experiment showing the sensitivity of model performance to these weights?
3. The paper demonstrates excellent performance on OOD instructions. Are there any specific types of instructions (such as highly abstract, requiring physical knowledge, or multi-role interaction) that Motion-R1 still struggles to handle? Can its failure modes be systematically defined?

### Soundness
3

### Presentation
3

### Contribution
3
