# QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6

## Abstract
Reinforcement learning (RL) has emerged as a central paradigm for training large language models (LLMs) in reasoning tasks. Yet recent studies question RL’s ability to incentivize reasoning capacity beyond the base model. This raises a key challenge: how can RL be adapted to solve harder reasoning problems more effectively?
To address this challenge, we propose a simple yet effective strategy via Question Augmentation: introduce partial solutions during training to reduce problem difficulty and provide more informative learning signals. 
Our method, QuestA, when applied during RL training on math reasoning tasks,  not only improves pass@1 but also pass@k—particularly on problems where standard RL struggles to make progress. 
This enables continual improvement over strong open-source models such as DeepScaleR and OpenMath Nemotron, further enhancing their reasoning capabilities. We achieve new state-of-the-art results on math benchmarks using 1.5B-parameter models: 72.50\% (+10.73\%) on AIME24, 62.29\% (+12.79\%) on AIME25, and 41.67\% (+10.11\%) on HMMT25. Code, data and model are available at https://anonymous.4open.science/r/questa932.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces QuestA, a data-centric augmentation technique designed to enhance reasoning performance during RLVR by injecting partial solution hints into training prompts. The approach is conceptually simple yet theoretically grounded, with analyses explaining why partial-solution augmentation is effective. Experimental results demonstrate its strong performance and incorporating partial hints not only accelerates learning but also mitigates the entropy collapse.

### Strengths
1. The method is simple yet effective, and achieves new SOTA pass@1 results on challenging math benchmarks for 1.5B-parameter models.
2. The method is thoroughly evaluated through comprehensive ablation studies and extensive analyses across multiple datasets, model architectures, and training curricula, consistently demonstrating performance improvements across diverse settings.

### Weaknesses
1. The method’s reliance on high-quality, step-wise solutions for augmentation raises concerns about scalability to domains lacking such curated data. For instance, can QuestA generalize to real-world science Q&A or open-domain reasoning tasks where solution steps are unavailable?  
2.  The paper would benefit from a qualitative error analysis. There is limited discussion of why the “Partial-0” (no hints) setting yields no improvement, or why certain tasks exhibit smaller gains.
3.   In my personal opinion, although the paper offers a theoretical perspective on why partial-solution augmentation improves RL efficiency, this section feels somewhat unnecessary and potentially confusing. The core idea is already straightforward and intuitive, and adding theoretical formalism may detract from its practical clarity—especially since the method can be naturally interpreted as a form of prompt optimization.

### Questions
1. Could the authors elaborate on practical limitations—such as applicability to problems without step-wise gold solutions or the impact of poor-quality hints? Is there empirical evidence of robustness to incorrect or misleading hints?  
2. How sensitive is QuestA to the form of hints used (e.g., solution block, chain-of-thought, or intermediate step)?  
3. How does the choice of $p$ trade off between training speed and final performance? Can the method adaptively tune this during RL?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper explores improving reasoning in large language models (LLMs) using reinforcement learning (RL). It highlights limitations of RL in enhancing reasoning beyond the base model and proposes a novel strategy called Question Augmentation (QuestA). QuestA introduces partial solutions during training to simplify problems and provide better learning signals. Applied to math reasoning tasks, QuestA significantly boosts performance metrics like pass@1 and pass@k, especially on challenging problems. The method achieves state-of-the-art results on math benchmarks using 1.5B-parameter models.

### Strengths
1,The authors demonstrate QuestA's effectiveness through rigorous experiments on math reasoning benchmarks, achieving state-of-the-art results with notable performance gains on metrics like pass@1 and pass@k.
2. The open resources makes it a valuable contribution to the field of LLMs, particularly in reasoning-intensive domains like mathematics.

### Weaknesses
1. The experiments are conducted on 1.5B-parameter models, which may not generalize to larger models. The scalability and adaptability of QuestA across different model sizes remain unclear.
2. The relationship between data difficulty and performance was mentioned in earlier papers on mathematics long ago, such as [1]. However, this paper does not discuss. And, in this subproblem, it is also an obvious insight, somewhat lacking innovation.

[1] WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct

### Questions
refer to Weaknesses

### Soundness
2

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
5

### Summary
A simple yet effective strategy via Question Augmentation is introduced for  partial solutions during training to reduce problem difficulty and provide more informative learning signals.

### Strengths
1.Improves pass@1 but also pass@k—particularly on problems where standard RL struggles to make progress.
2.Achieve new state-of-the-art results on math benchmarks using 1.5B-parameter models: 72.50% (+10.73%) on AIME24, 62.29% (+12.79%) on AIME25, and 41.67% (+10.11%) on HMMT25.
3.There is reasonable proof for the the proposed theory .

### Weaknesses
1.The benchmark is better if code dataset is evaluated.
2.The algorithm of RL is not advanced.

### Questions
1.It's a good data augmentation for advance the reasoning during RL training.
2.It is more experiments on other RL algorithm.

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
4

### Summary
This paper introduces QUESTA, a simple yet effective data-centric strategy to enhance the reasoning capabilities of LLMs through reinforcement learning. The paper identifies a critical trade-off in RL training for reasoning: while easy problems can cause a decline in reasoning diversity (pass@k), training on hard problems is inefficient due to sparse reward signals. The proposed method is introduced to navigate this trade-off, aiming to mitigate this inefficiency and effectively expand the model's reasoning capacity. Experiments show the method enables 1.5B-parameter models to achieve better results on challenging math benchmarks, in some cases surpassing a much larger 32B model. 

A key limitation is the method's reliance on an empirically-tuned curriculum. This hand-crafted 50-25 schedule presents challenges for the method's transferability to new settings. Furthermore, its effectiveness on larger-scale models remains unverified, as such models may not benefit from this specific scaffolding strategy.

### Strengths
- The paper clearly identifies and demonstrates the critical trade-off between training on easy versus hard prompts in RL, providing a strong motivation for the proposed method.

- The paper introduces an elegant approach that does not require complex changes to the underlying model architecture or RL algorithm. The idea of using partial solutions as hints is intuitive and proves to be effective.

- The method yields consistent performance improvements. The 1.5B model trained with QUESTA shows a clear gain over the standard RL baseline and, on the AIME25 benchmark, even surpasses a model over 20 times larger, highlighting the efficiency of the approach.

- The paper includes a theoretical analysis that formalizes why augmenting questions with partial solutions can improve the sample efficiency of RL, adding depth and rigor to the empirical findings.

### Weaknesses
- The experiments are limited to 1.5B models. It is unclear if the method would provide similar gains on larger models that already possess stronger reasoning capabilities.

- The curriculum for providing hints (50-25) appears to be a key component of the method's success. The ablation study only compares this strategy against a fixed 50% hint, which is insufficient to understand the sensitivity of the model's performance to other potential curriculum designs or hyperparameter choices.

- "We apply augmentation using the solution block rather than the reasoning chain-of-thought"; this is a design choice presented without any rationale or comparative experiments to justify this decision. It is plausible that hints derived from the CoT could be more effective.

### Questions
The curriculum for the hint percentage (from 50% to 25%) is a key component. Could you elaborate on how this schedule was chosen and have you considered alternatives, such as a more gradual decay or an adaptive schedule based on model performance?

### Soundness
3

### Presentation
3

### Contribution
3
