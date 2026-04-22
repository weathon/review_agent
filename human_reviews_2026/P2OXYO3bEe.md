# Tina: Tiny Reasoning Models via LoRA

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 2, 8, 6

## Abstract
How cost-effectively can strong reasoning abilities be achieved in language models? Driven by this question, we present Tina, a family of tiny reasoning models achieved with high cost-efficiency. Tina shows that substantial reasoning performance can be developed using only minimal resources, by applying low-rank adaptation (LoRA) during reinforcement learning (RL), to an already tiny 1.5B parameter base model. This minimalist approach produces models that are competitive with, and sometimes surpass, SOTA RL reasoning models built upon the same base model. Crucially, this is achieved at a tiny fraction of the computational cost employed by existing models. In fact, the best Tina model achieves a >20\% reasoning performance increase and 43.33\% zero-shot Pass@1 accuracy on AIME24, at only \$9 USD cost (i.e., an estimated 260x reduction). Our work reveals the surprising effectiveness of efficient RL reasoning via LoRA. We validate this across multiple open-source reasoning datasets and various ablation settings starting with a single, fixed set of hyperparameters. Furthermore, we explore the hypothesis that this effectiveness and efficiency stem from LoRA rapidly adapting the model to the structural format of reasoning rewarded by RL, while largely preserving the base model's underlying knowledge. In service of accessibility and open research, we fully open-source all code, training logs, model weights, and checkpoints.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes Tina -- a family of tiny reasoning models that achieve strong performance with extreme cost-efficiency offering a minimalist yet powerful path to efficient reasoning in small LMs. By applying LoRA during RL, Tina boosts reasoning ability at just $9 training cost which is very much cheaper than comparable models.

### Strengths
1. Novelty: One of the earliest works to demonstrate that GRPO based RL post-training can be used with LoRA
2. Cost-efficiency: Achieves SOTA or near SOTA reasoning performance at a tiny fraction of typical RL costs
3. Scalability & accessibility: Tiny models are easier to deploy and experiment with; open-source release ensures reproducibility
4. Comprehensive validation: Multiple datasets and ablations strengthen claims about effectiveness and generality

### Weaknesses
1. As mentioned in the paper, while the novelty is limited, theis is one of the few works to systematically investigate applying LoRA with GRPO for reasoning and while some tricks on training dynamic have been shared, the technical contribution looks limited.

### Questions
1. Authors, an ideas on extension of the work to tool-calling scenarios and multi-turn rollouts?

### Soundness
3

### Presentation
4

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
This paper proposes to use Low-rank adaptation(LORA) finetuning with a  group relative policy optimization (GRPO) approach in order achieve reasoning capacities with tiny models.

### Strengths
* Pursuits an interesting, potentially really important idea
* Easy to implement
* Large amount of experiments

### Weaknesses
The Idea is very straightforward (could also be seen as a plus!)

The paper is missing to report baseline benchmark results of small (but still larger models) for comparison (e.g., you could use something like a deepseek R1 model with ~7B parameters), e.g., in Table 2 and 3.; In the current state it is very hard to judge if the results could lead to any reasonable use case.

Measuring the training costs in $ is ephemeral and can change very quickly even without hardware changes (for economical reasons, exchange rates, etc...); For a scientific paper, I would recommend reporting FLOPs or at least GPU hours instead or on top.
Reporting the cost breakdown for different ablations and the evaluations to me seems not to be relevant for the main paper; Overall the main paper seems to be a bit stretched out to fill the pages and could be written way more concisely.

What am I missing is an in-depth discussion of the advantages and disadvantages of using very small models for reasoning and their place in an agentic AI. While I absolutely see reasons to use small models, reasoning appears to be *the* task, in which at least decently sized models seem to be a good investment; 

The main paper is missing details about the actual training setup ("Tina models build upon the open-source training recipes and datasets
of STILL-3 (RUCAIBox STILL Team, 2025), DeepScaleR (Luo et al., 2025), and Open-RS (Dang and Ngo, 2025). " What does "build upon" mean here? Which datasets have been used? Is it a fair comparison if you just collect more datasets for fine-tuning?

### Questions
The paper goes into large efforts to show what can be achieved with small training costs. 
An alternative to using tiny models could be to uses small models with quantization; How would these compare?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a study on using LoRA and RL to improve the reasoning capability of small language models. Through extensive experiments, the authors show us a surprising conclusion that LoRA, compared to full-parameter tuning, is more efficient and effective when using RL to improve the reasoning capability of small language models. To explain this phenomenon, the authors propose the Rapid Reasoning Format Adaptation Hypothesis, suggesting that LoRA enables models to quickly adapt to reasoning formats rewarded by RL while largely preserving the base model’s knowledge. They conduct rigorous ablation studies and detailed training-dynamics analyses to support this claim. Overall, the paper is well-organized, technically sound, and reproducible, offering clear insights into the mechanisms of parameter-efficient RL and setting a valuable precedent for future research.

### Strengths
1. This paper is well written and clearly presents the necessary background, experimental setup, and extensive ablation studies that support the authors’ claims.

2. The experimental setup is clean and effective. The paper adopts a minimalist yet powerful combination of LoRA and GRPO reinforcement learning, eliminating noise from architectural or dataset improvements, which is essential and helps focus on the most important research question the paper aims to answer.

3. The authors propose the Rapid Reasoning Format Adaptation Hypothesis to explain why they observe that RL with LoRA is more effective for reasoning, and rigorous ablation studies support this. This is very beneficial for the entire research community.

### Weaknesses
1. Although the paper evaluates Tina across multiple training recipes (STILL-3-1.5B, DeepScaleR-1.5B, and Open-RS), all experiments ultimately rely on the same underlying base model, DeepSeek-R1-Distill-Qwen-1.5B. This raises concerns about the generality of the claimed findings. It remains unclear whether the observed “rapid reasoning adaptation” and cost-efficiency would still hold for other base models that are not distilled from DeepSeek-R1 or outside the Qwen2.5 family. For instance, would similar trends be observed for models such as Qwen3-1.7B or other architectures of comparable scale? A broader evaluation would strengthen the paper’s generality.

2. The authors attribute LoRA’s effectiveness to its ability to rapidly adapt the reasoning format under RL while preserving the base model’s knowledge. However, using a sufficiently small learning rate in full-parameter fine-tuning could potentially yield similar effects: slow, stable updates that preserve existing knowledge while adapting reasoning behavior. It would be valuable to analyze whether such small-learning-rate full-parameter fine-tuning exhibits the same adaptation dynamics or efficiency gains as LoRA, and to clarify the conceptual and empirical differences between the two.

### Questions
See Weaknesses.

### Soundness
4

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
This paper introduces TINA, demonstrating that applying LoRA during reinforcement learning to a tiny 1.5B model can achieve SOTA-competitive reasoning performance at a tiny fraction of the computational cost.

### Strengths
- The authors' research finding that LoRA on small language models can achieve performance on par with full-parameter RL training is highly valuable. I believe this will benefit researchers who are "compute-scarce" and those who are new to the field, allowing more people to experiment with and master RL scaling.

- The hypothesis proposed by the authors makes sense. It holds similar viewpoints with the concurrent blog-"LoRA Without Regret", suggesting that the amount of information gained during the RL process is less than that in SFT. Therefore, a parameter-efficient method like LoRA can be effectively used to "organize" the base model's existing knowledge into the "structured format" preferred by a RL reward scalar.

### Weaknesses
Please correct me if my understanding is wrong or biased:-)

Overall, TIna is a very "timely" paper. Its primary contribution does not lie in proposing a new methodology, contributing new data or benchmarks, or releasing a powerful SOTA model. Instead, its main contribution comes from "updating the reader's cognition." The authors are loudly proclaiming to the community that "LoRA + small language models can still achieve significant reasoning gains via RL," essentially encouraging everyone to "Go try LoRA RL training now!" If I had reviewed this paper earlier this year for a conference like COLM or NeurIPS, I would have given it a clear accept. However, at this current point in time, I feel compelled to raise two additional questions:

- Could the authors provide deeper insights? For example, whether other PEFT methods (e.g., DoRA), or different model families (e.g., Llama, MiMO, or even VLMs), or non-mathematical reasoning tasks (e.g., puzzles, games) are applicable to TINA? Or perhaps some deeper theoretical insights?

- LoRA RL Training is already supported by several existing RL frameworks, such as verl, and they are often scalable, supporting training on larger models and more tasks. Given this, what do the authors believe is the unique, standalone value of the open-sourced code and model weights provided by TINA?

### Questions
The first citation in this paper appears to be sensitive. It indirectly points to a blog post that is directly related to this work. I am not sure if this potentially violates the double-blind review process.

### Soundness
3

### Presentation
4

### Contribution
2
