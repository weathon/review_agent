# RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 4, 8, 6

## Abstract
Fine-grained visual reasoning remains a core challenge for multimodal large language models (MLLMs). The recently introduced ReasonMap highlights this gap by showing that even advanced MLLMs struggle with spatial reasoning in structured and information-rich settings such as transit maps, a task of clear practical and scientific importance. However, standard reinforcement learning (RL) on such tasks is impeded by sparse rewards and unstable optimization. To address this, we first construct ReasonMap-Plus, an extended dataset that introduces dense reward signals through Visual Question Answering (VQA) tasks, enabling effective cold-start training of fine-grained visual understanding skills. Next, we propose RewardMap, a multi-stage RL framework designed to improve both visual understanding and reasoning capabilities of MLLMs. RewardMap incorporates two key designs. First, we introduce a difficulty-aware reward design that incorporates detail rewards, directly tackling the sparse rewards while providing richer supervision. Second, we propose a multi-stage RL scheme that bootstraps training from simple perception to complex reasoning tasks, offering a more effective cold-start strategy than conventional Supervised Fine-Tuning (SFT). Experiments on ReasonMap and ReasonMap-Plus demonstrate that each component of RewardMap contributes to consistent performance gains, while their combination yields the best results. Moreover, models trained with RewardMap achieve an average improvement of 3.47% across 6 benchmarks spanning spatial reasoning, fine-grained visual reasoning, and general tasks beyond transit maps, underscoring enhanced visual understanding and reasoning capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper tackles the problem of improving reasoning of multi-modal large language models. They use the REASONMAP dataset and show that directly performing GRPO on the dataset is not very effective for multi-modal reasoning because of high difficulty of questions and sparse rewards. To alleviate this, they construct REASONMAP-PLUS, an augmented dataset which easier VQA style queries with denser reward. They also show improvements in training MLLMs using GRPO by adding a curriculum (easy->hard) questions and by adding partial rewards to alleviate the problem of sparse rewards. They name this new recipe as REWARDMAP which shows performance gains on the test set of REASONMAP and gains of around 3.5% on other reasoning benchmarks as well.

### Strengths
1. The paper shows gains using REWARDMAP on both in-distribution and out of distribution test sets, suggesting that the curriculum learning + denser reward works well for multi-modal reasoning training 
2. They construct an augmented dataset REASONMAP-PLUS with around 4k VQA pairs across different types like local counting, global counting, and true/false
3. The difficulty and dense rewards are assigned programatically without the need for any reward models or human labelling which makes the method cheap and efficient

### Weaknesses
1. Incremental novelty: both curriculum learning (https://arxiv.org/abs/2506.06632, https://arxiv.org/abs/2501.12599, https://arxiv.org/abs/2502.14768) and denser rewards to improve reasoning are approaches which have been tried in several existing works, while this paper shows a combination which works well, the approach as a whole feels incremental. 
2. Reward shaping not generalizable: The detail reward (credit for route names, transfers) is tightly linked to transit-map structure; it’s unclear how the same scheme would extend to charts, floor plans, or natural images without such information. Likewise, the difficulty weights rely on a dataset-specific notion of map/question difficulty (e.g., transfer counts). It would be good to show a more generalizable approach to reward design. 
3. Reward shaping not ablated well: The reward seems ad-hoc, there are no clear ablations for how the value of alpha was chosen, nor are there ablations on how W_diffculty is decided, or why it is multiplied to the other rewards. It would be good to provide some additional experimental results as to how this particular reward structure was chosen. 
4. Limited OOD improvements: While the authors show performance on other benchmarks like MM-Star as well, the performance on them shows negligible gains. It might be more informative if there is a task wise breakdown of the performance which might help analyze the exact tasks where performance improves vs where it stagnates.

### Questions
Please refer to the weakness section for questions.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studys the problem of sparse reward in standard reinforcement learning for spatial reasoning in structured and information-rich setting.  To address the problem, the authors construct REASONMAP-PLUS, a dataset with dense reward signals and propose a multi-stage RL framework for improving the corresponding capabilities in current MLLM. Experiment results demonstrate its effectiveness across various reasoning tasks and general tasks.

### Strengths
1. The research topic of this work is interesting, as sparse reward is indeed a significant challenge in current vision tasks, particularly in structured visual input tasks as highlighted in the paper.
2. This work introduces the REASONMAP-PLUS dataset, which could be beneficial for future related research.
3. The experimental results seem promising, demonstrating the effectiveness of the proposed method on transit map planning tasks and showing its transferability to other tasks.

### Weaknesses
1. The method parts of the paper presents certain ambiguities. For instance, how does the Difficulty-Aware Weighting mechanism operate? Are weights assigned to each sample within the groups of GRPO, or to each sample across the entire training dataset? Furthermore, what does Multi-stage RL training entail—does it refer solely to the sequencing of data based on difficulty, or does it also encompass different training phases?
2. The novelty of the approach appears limited, as the core methodology remains largely aligned with that of GRPO. While the design of the detailed reward function is interesting, it only considers specific components of the response (e.g., destination stops, route names), thereby constraining its overall flexibility.

### Questions
1. In the acquisition of the detailed reward, how is the correctness of each sub-part determined? Is an external LLM employed to extract the relevant content, or is the response constrained to a fixed format?
2. Please refer to the weaknesses highlighted above.

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
3

### Summary
This paper proposes RewardMap, a multi-stage RL framework that mitigates sparse-reward problems in fine-grained visual reasoning for multimodal LLMs. Building on ReasonMap, the authors introduce ReasonMap-Plus, a dense and difficulty-graded dataset for visual question answering on transit maps. RewardMap integrates (1) a difficulty-aware reward function combining format, correctness, and detail rewards, and (2) a multi-stage GRPO training curriculum progressing from perception to reasoning tasks. Experiments on RewardMap, ReasonMap-Plus, and six external benchmarks show consistent gains and reduced visual confusion, demonstrating improved visual understanding and reasoning performance.

### Strengths
1. New dataset and training paradigm: the paper introduces ReasonMap-Plus, which is a novel dataset that helps conducting RL using dense reward signals.
2. Models being trained on the dataset seem to perform well on the proposed benchmarks

### Weaknesses
1. Lack of baseline VLMs on ReasonMap and ReasonMap-Plus. What would more recent and powerful VLMs like GPT-4o and GPT-5 perform on these benchmarks?
2. Would be better to train other VLMs other than Qwen2.5-VL, such as InternVL models to see if the training and data could actually be generalized.

### Questions
N/A

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
4

### Summary
This paper proposes REWARDMAP, a multi-stage reinforcement learning (RL) framework for improving fine-grained visual reasoning in multimodal large language models (MLLMs). It builds on the REASONMAP benchmark and introduces REASONMAP-PLUS, a companion dataset providing dense supervision via VQA-style tasks organized by difficulty. The approach combines a difficulty-aware reward design with curriculum-based RL under GRPO to mitigate sparse reward issues. Experiments show consistent but small gains (around 3.5%) on REASONMAP, REASONMAP-PLUS, and several external benchmarks. Ablations suggest that reward shaping and staged training both contribute to improvements.

### Strengths
- Addresses a critical problem of sparse rewards in RL for multimodal reasoning with a practical and elegant solution.  
- Well-engineered multi-stage RL approach combining dense and sparse supervision effectively.  
- Comprehensive evaluation with ablations that isolate reward design and curriculum effects.  
- Clear writing, reproducible setup, and solid empirical results across multiple benchmarks.

### Weaknesses
- Conceptual novelty is limited. The method is primarily a well-engineered combination of known ideas: reward shaping, difficulty weighting, and curriculum learning under GRPO. There is no new RL algorithm or theoretical contribution.  
- Reported performance gains are numerically small relative to the additional dataset, compute, and engineering effort. It is unclear whether such modest improvements are statistically significant.  
- The reward weighting scheme (α, γ_e/m/h, β_0/1) is ad-hoc and lacks sensitivity analysis. Without this, claims about “difficulty awareness” remain anecdotal.  
- Evaluation is restricted to Qwen2.5-VL, raising concerns about generalization and overfitting to a single model architecture.  
- REASONMAP-PLUS is narrowly scoped to transit maps; claims of general fine-grained reasoning ability are not convincingly supported.  
- The paper lacks deeper insight or analysis into why multi-stage RL works, there is no visualization or error decomposition that would make the mechanism interpretable.  
- The average improvement on external benchmarks (+3.47%) is within noise for modern MLLMs; effect size is weak.

### Questions
1. How sensitive are results to the hyperparameters \( \alpha \) and difficulty weights (\( \gamma_e, \gamma_m, \gamma_h \))?  
2. Could the proposed curriculum RL pipeline be applied to other structured domains (e.g., diagrams, charts)?  
3. How do you ensure no data leakage between REASONMAP-PLUS easy/hard tasks and REASONMAP test sets?  
4. Why were RLHF or DPO-based baselines not compared for reward densification?  
5. Can you provide error-type or reasoning-chain analyses to explain where REWARDMAP helps most?

### Soundness
2

### Presentation
3

### Contribution
3
