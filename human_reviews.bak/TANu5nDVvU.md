# SMART: Self-Learning Meta-strategy Agent for Reasoning Tasks

- Decision: Reject
- Scores: 3, 5, 3, 6

## Abstract
Tasks requiring deductive reasoning, especially those involving multiple steps, often demand adaptive strategies such as intermediate generation of rationales or programs, as no single approach is universally optimal. 
While Language Models (LMs) can enhance their outputs through iterative self-refinement and strategy adjustments, they frequently fail to apply the most effective strategy in their first attempt. This inefficiency raises the question: *Can LMs learn to select the optimal strategy in the first attempt, without a need for refinement?*
To address this challenge, we introduce *SMART*: **S**elf-learning **M**eta-strategy **A**gent for **R**easoning **T**asks, a novel framework that enables LMs to autonomously learn and select the most effective strategies for various reasoning tasks. We model the strategy selection process as a *Markov Decision Process* and leverage reinforcement learning-driven continuous self-improvement to allow the model to find the suitable strategy to solve a given task. Unlike traditional self-refinement methods that rely on multiple inference passes or external feedback, *SMART* allows an LM to internalize the outcomes of its own reasoning processes and adjust its strategy accordingly, aiming for correct solutions on the first attempt.
Our experiments across various reasoning datasets and with different model architectures demonstrate that *SMART* significantly enhances the ability of models to choose optimal strategies without external guidance (+15 points on the GSM8K dataset). By achieving higher accuracy with a single inference pass, *SMART* not only improves performance but also reduces computational costs for refinement-based strategies, paving the way for more efficient and intelligent reasoning in LMs.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper tries to solve a problem for language model: Can LMs learn to select the optimal
strategy in the first attempt, without a need for refinement? The strategy I believe indicates the prompt strategy like CoT, PoT, step-by-step, etc. The authors design a strategy policy that selects the strategy based on current state and optimize it to select the best strategy on the first attempt.

### Strengths
1.	The topic is interesting since it will reduce the number of attempts that LM needs before achieving successful solution. 
2.	Even though there is some details that need to be clarified, the approach is generally easy to understand.

### Weaknesses
1.	Authors formulate the problem as MDP, but the policy is designed based on history of states and actions. It conflicts with conventional understanding of MDP in RL field. If the problem has Markov state, there is sufficient to make decision based on state. Otherwise, the state is mistakenly designed and it contains no full information for decision making. 
2.	The algorithm is named with “meta-strategy”. People might wonder the meaning of “meta”. Does the policy adjust to different tasks or it just choose appropriate strategy action based on current state of the problem?
3.	The objective is modified to use (4) for policy optimization. Only current outputs are used for update. It becomes a supervised learning with maximum likelihood to reproduce the pattern of correct samples, in stead of reinforcement learning through trial and error. 
4.	Following the above question, if the policy is trained on only correct (h_1, a_1/a_t), how it is able to make correct refinement in evaluation where h_t is not seen in the training set. One possible reason I guess might because the authors force “the model samples with a different strategy than the one present in its history” (Line 197). So perhaps randomly selecting the rest strategies in the refinement of SMART may still improve the performance. Authors can refute my guess by adding a comparison experiment. 
5.	Does SMART increase the LM parameters and computation because an additional policy LM is needed, in addition to the original pretrained LM to produce problem solutions? If so, I think a discussion is necessary in the paper.

### Questions
1.	Equation (1) has a mistake. It is s_1 that samples on \mu, not a_1. 
2.	In Algorithm, step 2 makes step 31 totally meaningless. I am also confused if Implicit Biasing should happen before Policy Update or not?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces SMART, a novel framework that enables language models to autonomously learn and select the most effective strategies for reasoning tasks. It models strategy selection as a Markov Decision Process and employs reinforcement learning for continuous self-improvement, aiming to achieve correct solutions on the first attempt. The approach significantly enhances the ability of models to choose optimal strategies without external guidance, improving both accuracy and computational efficiency.

### Strengths
see questions

### Weaknesses
see questions

### Questions
The paper presents an approach to improving reasoning in language models through self-learning strategies. The experimental results seem significant. However, the key details of the method are missing and I cannot fully understand the contribution of this paper.

1. In the methodology section, the authors have emphasized the formulation of MDP, which is a crucial aspect of SMART. However, there is a noticeable absence of critical methodological details. For instance, the refinement process lacks clarity on the specific method (or prompts) used to facilitate the model's self-improvement. Whether additional training conducted to enhance the model's evaluation capabilities? How rewards are derived?  How do data from multiple refinements contribute to improving the model's success rate on the first attempt?

2. In the experimental section, the authors refer to three methods: Chain of Thought (CoT), Least to Most (L2M), and Program of Thought (PoT). How these methods are integrated and utilized within the SMART framework? Why does Table 3 only provide a comparison with CoT and not with the other methods?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes SMART, a self-training framework that aims to improve problem-solving performance by searching for solutions through self-reflection and subsequently fine-tuning the model to predict the correct solution directly. Specifically, SMART addresses the problem of automatic selection of problem-solving strategies, such as Least-to-Most (L2M) and Chain-of-Thought (CoT), by leveraging reinforcement learning (RL). Empirical results demonstrate that SMART outperforms baseline methods that do not employ self-training.

### Strengths
1. **Relevance**: The problem of efficient self-training is important for improving post-training capabilities of large language models (LLMs). A successful self-training method can enhance LLM performance on complex tasks, such as multi-step reasoning, making the research direction highly relevant.
2. **Clarity**: The overall presentation of SMART, including the RL framework for strategy selection, is clearly explained, and the experimental methodology is easy to follow.
3. **Diverse Empirical Evaluation**: Experiments are conducted using several LLM architectures (e.g., Gemma 7B, Mistral 7B) and multiple benchmarks, which adds credibility to the empirical results.

### Weaknesses
1. **Limited Novelty**: The core idea of SMART resembles existing self-training methods that utilize rejection sampling (https://arxiv.org/pdf/2308.01825). The concept of generating correct samples through self-reflection has already been widely explored, such as [ReST](https://arxiv.org/pdf/2308.08998) (Gulcehre et al., 2023) and [Re-ReST](https://arxiv.org/pdf/2406.01495), with the only new aspect being the emphasis on strategy selection rather than general solution generation. This difference does not appear significant enough to warrant a new framework.
2. **Lack of Related Work Discussion**: The paper does not adequately discuss related self-training approaches such as [ReST](https://arxiv.org/pdf/2308.08998), [ReST^{EM}](https://arxiv.org/pdf/2312.06585), and other comparable methods like STaR. Including a detailed discussion of these related works would help contextualize SMART’s contributions and clarify how it advances beyond these methods.
3. **Unfair Empirical Comparison**: SMART is compared only against baselines without self-training, which makes the comparisons somewhat unfair. Including direct comparisons against other self-training methods would offer a more meaningful assessment of SMART’s effectiveness.
4. **Contribution of Strategy Selection**: The paper lacks a detailed analysis of the impact of strategy selection. Previous self-training methods, even when limited to a single strategy like CoT, have shown significant improvements. Therefore, it is unclear how much of SMART’s gains can be attributed specifically to strategy selection as opposed to other aspects of the self-training process.
5. **Limitation to LoRA Results**: The experiments only report results from LoRA fine-tuning, which may not fully capture the performance gains that could be achieved by fine-tuning the entire model. While this is acceptable given possible resource constraints, including results from full-model fine-tuning could strengthen the empirical evaluation.

### Questions
1. **Comparison with Existing Baselines**: How does SMART compare against other self-training baselines like ReST and ReST^{EM}? A direct comparison with these methods would provide stronger support for SMART’s effectiveness.
2. **Ensuring Strategy Selection**: How does SMART ensure that during data generation, one of the three strategies (L2M, PoT, CoT) is selected? Without proper constraints or instructions, LLMs might struggle to consistently use the intended strategies. Does SMART simply try each of these strategies until a correct solution is found, or is there a more sophisticated mechanism in place?
3. **Algorithm Implementation**: In the algorithm description, should the implicit biasing step occur before the policy update? This order seems more intuitive and would ensure the update incorporates the biased data correctly.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces SMART, a novel reinforcement learning framework enabling language models (LMs) to autonomously select optimal reasoning strategies for complex tasks on the first attempt, reducing reliance on iterative refinements. By modeling strategy selection as a Markov Decision Process, SMART applies continuous self-improvement, enabling LMs to adapt strategies dynamically. Experiments across datasets show significant performance gains and computational efficiency, with SMART achieving up to +15 points in accuracy on the GSM8K dataset, outperforming traditional refinement methods.

### Strengths
1. SMART provides a novel framework by modeling strategy selection in LMs as a Markov Decision Process, adding a unique layer of self-learning.
2. The paper demonstrates high-quality experimentation, validating SMART’s effectiveness across multiple datasets and model architectures, such as GSM8K, SVAMP, and ASDiv.
3. The paper  is clearly articulated, particularly in its two-stage process design, making the approach easy to understand and apply.
4. By enabling LMs to select optimal strategies without iterative refinements, the work significantly reduces computational costs.

### Weaknesses
1. The method is overly complex without sufficient justification for why simpler methods cannot achieve similar results. There should be a more detailed baseline comparison.

2. The experimental section lacks clarity, particularly regarding the setup and specific configurations for each dataset, making the results difficult to interpret and replicate. Besides, the paper's discussion on the generalization capability of SMART across different datasets is weak, as it does not convincingly demonstrate robustness beyond minor accuracy gains.

3. The use of only 7B models raises concerns about the scalability of the proposed method, with no evidence provided for its effectiveness on larger models.

4. Theoretical claims about strategy optimization are vague and insufficiently supported by quantitative evidence, especially regarding the selection mechanism's consistency across different reasoning tasks.

### Questions
1. The method introduced appears intricate, particularly with its reliance on the MDP. Could you provide more concrete evidence or reasoning to support the necessity of this complex setup over simpler alternatives, such as supervised learning with strategy labels or existing reinforcement learning baselines?

2. Experimental details on configurations and hyperparameters are sparse. You should specify the settings for each dataset and model, especially for reward design and training.

3. The generalization to OOD datasets shows limited gains. What factors might limit this, and could SMART be adapted to improve performance on diverse datasets?

4. Only 7B parameter models were tested. Can you scale SMART to larger models, like 70B parameters?

5. The strategy selection process is crucial but lacks detailed evidence of consistency across tasks. Could you provide quantitative results showing its reliability on different reasoning tasks?

### Soundness
3

### Presentation
3

### Contribution
3
