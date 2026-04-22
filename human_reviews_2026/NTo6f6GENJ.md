# Revisual-R1: Advancing Multimodal Reasoning From Optimized Cold Start to Staged Reinforcement Learning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 6, 4, 6

## Abstract
Inspired by the remarkable reasoning capabilities of Deepseek-R1 in complex textual tasks, many works attempt to incentivize similar capabilities in Multimodal Large Language Models (MLLMs) by directly applying reinforcement learning (RL). However, they still struggle to activate complex reasoning. 
In this paper, rather than examining multimodal RL in isolation, we delve into current training pipelines and identify three crucial phenomena: 1) Effective cold start initialization is critical for enhancing MLLM reasoning. Intriguingly, we find that initializing with carefully selected text data alone can lead to performance surpassing many recent multimodal reasoning models, even before multimodal RL.2) Standard GRPO applied to multimodal RL suffers from gradient stagnation, which degrades training stability and performance. 3) Subsequent text-only RL training, following the multimodal RL phase, further enhances multimodal reasoning.
This staged training approach effectively balances perceptual grounding and cognitive reasoning development. 
By incorporating the above insights and addressing multimodal RL issues, we introduce \textbf{ReVisual-R1}, achieving a new state-of-the-art among open-source 7B MLLMs on challenging benchmarks including MathVerse, MathVision, WeMath, LogicVista, DynaMath, and challenging AIME2024 and AIME2025.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents ReVisual-R1, a novel post-training methodology aimed at enhancing the reasoning capabilities of MLLMs. This paper proposes a principled three-stage training curriculum, consisting of: (1) a text-only cold-start initialization, (2) a multimodal reinforcement learning phase, and (3) a text-only reinforcement learning refinement. The effectiveness of the proposed approach is evaluated on a range of multimodal and textual reasoning benchmark datasets.

### Strengths
1. The methodology proposed in this paper is easy to understand, and the algorithmic descriptions are presented in a clear and accessible manner.
2. Compared to other reasoning enhancement approaches for MLLMs, this paper also conducts comparative experiments on text-only benchmarks.

### Weaknesses
1. This paper exhibits some instances of over-claiming. Although the authors assert that their approach achieves SOTA results among 3B and 7B open-source MLLMs (lines 24-25), to my knowledge, several representative models, such as **MIMO-VL** and **Keye-VL* demonstrate significantly superior performance compared to the method proposed in this work. Additionally, in lines 42-43, regarding "The direct application of text-centric RL," I believe the methodology presented does not fundamentally depart from existing RL algorithms, such as **GRPO** and **DAPO**. While achieving SOTA or radical innovation is not strictly necessary for a publication, the actual results and contributions in this paper deviate from the claims made by the authors, which raises concerns.
2. The contributions of this paper are largely similar to those in prior works. For example, Contribution 1 has already been discussed in **LMM-R1**, and Contribution 2 can be found in papers such as **DAPO**. Thus, the novelty of this paper is relatively limited, and it leans more towards engineering practice rather than academic innovation.
3. The paper should clarify whether its primary contribution lies in proposing a novel training paradigm or in the specific model trained using this methodology. These two aspects are fundamentally different. If the focus is on the training paradigm, then experiments limited to Qwen2.5-VL7B/3B are insufficient. It would be more compelling to demonstrate the effectiveness of this approach across a broader range of models, such as LLaVA and InternVL.


Team, Xiaomi Mimo, et al. " Mimo-vl technical report."  arXiv preprint arXiv:2506.03569 (2025).

Team, Kwai Keye, et al. "Kwai Keye-VL Technical Report." arXiv preprint arXiv:2507.01949 (2025). 

Shao, Zhihong, et al. "Deepseekmath: Pushing the limits of mathematical reasoning in open language models." arXiv preprint arXiv:2402.03300 (2024).

Yu, Qiying, et al. "Dapo: An open-source llm reinforcement learning system at scale." arXiv preprint arXiv:2503.14476 (2025).

### Questions
1. Please clarify the issues mentioned in the Weakness section, as they are not sufficiently explained in the current draft.

2. There are a few minor issues:

   - In Figure 1 (bottom),  ‘Traj A Tra B Tra C’ --> ‘Traj A Traj B Traj C’.

   - Some arrow symbols in the figure appear to be partially obscured; please fix this.

   - On line 459, merge the citation marks for consistency.

   - The paper does not discuss limitations, failure cases, or future work, which makes it difficult for others to follow up. Please add a section addressing these points.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a three-stage training paradigm named ReVisual-R1 to enhance the performance of MLLMs on complex reasoning tasks. The paradigm consists of:

1. **Text-centric Cold Start**: Initializing the model with high-difficulty, structured pure-text reasoning data to establish a strong foundation for language-based reasoning;
2. **Multimodal RL**: Introducing a PAD mechanism to alleviate gradient stagnation caused by sparse rewards in standard GRPO under multimodal settings;
3. **Text-only RL**: After multimodal training, applying pure-text RL as a "polishing" phase to further refine linguistic expression and logical consistency without compromising visual grounding capabilities.

The method significantly outperforms existing open-source 3B/7B models on multiple multimodal and text-only reasoning benchmarks, including MathVerse, AIME24/25, and MATH-500.

### Strengths
1. **High-quality data construction pipeline**: The authors introduce the **GRAMMAR** dataset, employing a well-designed pipeline involving rule-based filtering, difficulty-level grading, embedding-based clustering (NV-Embedding + HDBSCAN), topic annotation, and hierarchical sampling. This ensures both data diversity and high reasoning complexity.
2. **Well-motivated training paradigm**: The three-stage curriculum learning approach (cold start → multimodal RL → text-only RL) is logically sound. Experiments validate the necessity and sequence sensitivity of each stage, particularly highlighting the unique value of **text-only RL as a polishing phase**.
3. **Openness and reproducibility**: Detailed training configurations and hyperparameters are provided, and the effectiveness is demonstrated across both 3B and 7B model scales, offering strong value to the research community.

### Weaknesses
1. **Lack of multi-seed ablation studies**: The main experiments and ablation studies do not report variance across multiple random seeds, making it difficult to assess whether the observed performance gains are statistically significant—especially on high-variance benchmarks like AIME24/25.
2. **Empirical rather than analytical justification for text-centric cold start**: While the authors support their claim with metrics such as response length and pass rates to demonstrate higher complexity in text data, they do not deeply investigate why multimodal cold-start data fails to effectively stimulate reasoning capabilities (e.g., due to vision-language alignment noise, annotation quality, or task design limitations).

### Questions
1. **Is the complexity of the multimodal RL data insufficient?**  
   The authors observe that an additional pure-text RL phase after multimodal RL is necessary to achieve optimal performance. Does this imply that the current multimodal RL dataset still lacks sufficient reasoning depth or diversity to support end-to-end training on its own? If a multimodal RL dataset with complexity comparable to the text-only RL data were constructed, could the subsequent text-only RL phase be eliminated?

2. **How is the training balance between multimodal RL and text-only RL determined?**  
   The paper does not specify the training balance between the two RL stages. Is the final performance sensitive to this balance? 

3. **Does the text-only RL phase degrade visual capabilities?**  
   Although the authors claim that text-only RL “without eroding the newly acquired visual grounding”, the experimental evidence supporting this assertion appears limited.

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
This paper presents ReVisual-R1, an open-source multimodal large language model (MLLM), with the following key contributions:
GRAMMAR Dataset Construction: The authors curate a diverse dataset covering textual and multimodal problems, with a focus on reasoning complexity and data diversity.
Staged Reinforcement Optimization (SRO): A two-stage RL training framework is proposed, including Multimodal RL (MRL) and Textual RL (TRL), which strengthen cross-modal reasoning and pure textual reasoning capabilities, respectively.
Prioritized Advantage Distillation (PAD): A novel method to prioritize informative samples during training, mitigating the gradient stagnation problem in GRPO and improving learning efficiency.
Experimental results show that this approach achieves significant improvements over existing open-source models and is competitive with some commercial models, demonstrating the effectiveness of the proposed data strategy and training methodology.

### Strengths
Comprehensive and well-structured: The paper provides a complete workflow from problem formulation, cold-start analysis, dataset construction, staged RL training, to ablation studies and generalization evaluation. Innovative approach: The combination of dataset curation (GRAMMAR) with staged RL training (SRO) and sample prioritization (PAD) represents a novel methodology in open-source MLLM research.

Empirical effectiveness: Extensive experiments across multimodal mathematics, logic, and textual reasoning benchmarks show clear performance gains at both 7B and 3B model scales.

Rationale behind staged training: The design of MRL followed by TRL is reasonable—first grounding visual-textual reasoning and then refining textual reasoning and linguistic fluency, allowing the model to generate longer, reflective CoT reasoning.

### Weaknesses
Training cost not analyzed: The computational overhead, training time, and data annotation cost of SRO + PAD are not quantified, which is important for assessing the method's practicality and scalability.

Dataset contribution ablation missing: The ablation studies focus on MRL/TRL phases and PAD, but the contribution of the GRAMMAR dataset itself (e.g., sample quantity, difficulty levels, multimodal vs. textual content ) is not independently analyzed.
Self-constructed dataset and generalizability: The use of a custom dataset, while common in multimodal reasoning research, may limit reproducibility and comparability. 

Discussion on how this dataset design affects generalization would strengthen the work.
Potential limitations in generalization: Although the model is evaluated on several knowledge-intensive benchmarks, its performance in completely different domains or real-world open-ended scenarios remains untested.

### Questions
Training cost and efficiency: What are the computational overhead, training time, and memory requirements of SRO + PAD compared to standard GRPO? 

Dataset ablation: How do different components of the GRAMMAR dataset (text vs. multimodal, difficulty levels)  contribute to the overall performance?

How would the model perform if trained or evaluated on other open-source multimodal reasoning datasets?

### Soundness
3

### Presentation
3

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
This paper introduces ReVisual-R1, a 3B/7B multimodal large language model (MLLM) that achieves sota performance on challenging multimodal and text reasoning benchmarks. The key innovation is a three-stage curriculum: (1) a text-only cold-start with complex reasoning data, (2) multimodal RL with a novel Prioritized Advantage Distillation (PAD) to combat gradient stagnation, and (3) a text-only RL polish.

### Strengths
- First to show that a text-only cold-start outperforms multimodal cold-starts for MLLM reasoning; PAD is a new fix for GRPO gradient stagnation.
- Impressive performances on both multimodal and text reasoning benchmarks.

### Weaknesses
- All baselines are Qwen-VL derivatives; no comparison with other backbone families (e.g., LLaVA-NeXT, InternVL, Flamingo-style).
- Interpretability of text-only data employed in the training process of MLLMs.

### Questions
- Have you tried the proposed curriculum on LLaVA-NeXT or InternVL models? If not, do you expect PAD to remain effective when the vision encoder differs significantly?
- After text-only cold-start SFT and text-only RL, visual related ability could degrade. Did you measure vision-only tasks (e.g., OCR, object count) before & after text-only training to confirm no catastrophic forgetting?
- How about the performance of sft training of mixture on multimodal and text data compared to the text-only setting?

### Soundness
3

### Presentation
3

### Contribution
3
