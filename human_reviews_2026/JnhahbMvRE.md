# HBO: Hierarchical Balancing Optimization for Fine-Tuning Large Language Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Fine-tuning large language models (LLMs) on a mixture of diverse datasets poses challenges due to data imbalance and heterogeneity. Existing methods often address these issues across datasets (globally) but overlook the imbalance and heterogeneity within individual datasets (locally), which limits their effectiveness. We introduce Hierarchical Balancing Optimization (HBO), a novel method that enables LLMs to autonomously adjust data allocation during fine-tuning both across datasets (globally) and within each individual dataset (locally). HBO employs a bilevel optimization strategy with two types of actors: a Global Actor, which balances data sampling across different subsets of the training mixture, and several Local Actors, which optimizes data usage within each subset based on difficulty levels. These actors are guided by reward functions derived from the LLM's training state, which measure learning progress and relative performance improvement. We evaluate HBO on three LLM backbones across nine diverse tasks in multilingual and multitask setups. Results show that HBO consistently outperforms existing baselines, achieving significant accuracy gains. Our in-depth analysis further demonstrates that both the global actor and local actors of HBO effectively adjust data usage during fine-tuning. HBO provides a comprehensive solution to the challenges of data imbalance and heterogeneity in LLM fine-tuning, enabling more effective training across diverse datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Hierarchical Balancing Optimization (HBO), a novel framework designed to address data imbalance and heterogeneity challenges in fine-tuning LLMs. HBO employs a bilevel optimization strategy with two key components: a Global Actor for balancing data sampling across datasets and Local Actors for optimizing sampling within individual datasets based on difficulty levels. The paper demonstrates HBO’s effectiveness across three LLM architectures on multilingual and multitask setups, showing consistent performance improvements over existing baselines.

### Strengths
The hierarchical optimization framework with global and local balancing is novel and addresses a critical gap in LLM fine-tuning.
The experiments are extensive and well-designed, covering multiple models, tasks, and evaluation metrics. The ablation studies and sensitivity analyses provide deep insights into the effectiveness of HBO.
The paper is clearly written, with detailed explanations of the methodology and results. Figures and tables are well-designed and enhance understanding.
The proposed method is broadly applicable and demonstrates consistent performance gains, making it a valuable contribution to the field.

### Weaknesses
While HBO demonstrates strong performance improvements, the additional 15% training overhead may limit its applicability in resource-constrained environments. The authors could provide more discussion on strategies to mitigate this overhead.

Although the paper evaluates different reward functions, the focus is primarily on L2 norm and perplexity ratio. Exploring alternative reward mechanisms (e.g., task-specific metrics) could further enhance HBO’s applicability.

The experiments are conducted on sampled datasets (e.g., 20% of total data). It would be helpful to discuss how HBO scales to larger datasets and whether the performance gains remain consistent.

### Questions
Have the authors considered task-specific reward functions or alternative optimization techniques for the global and local actors? If so, how do these compare to the current design?

### Soundness
3

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
This paper presents Hierarchical Balancing Optimization (HBO), a bilevel optimization framework designed to mitigate data imbalance and heterogeneity when fine-tuning LLMs. HBO introduces two interacting components, i.e., a Global Actor that adjusts sampling probabilities across datasets and several Local Actors that manage sampling within datasets based on difficulty levels. Both actors are guided by rewards derived from the LLM's training state: gradient norms for global balancing and perplexity ratios for local balancing. The authors evaluate HBO on three model backbones (Llama-3.1-8B, Qwen2.5-7B, EuroLLM-9B) and nine tasks across multilingual and multitask setups, reporting consistent performance improvements over both heuristic and dynamic baselines.

### Strengths
1. The paper targets an important problem, i.e., data imbalance and heterogeneity in LLM fine-tuning,which is relevant to current multi-task and multilingual training paradigms.
2. The hierarchical bilevel optimization formulation is conceptually interesting and provides a unified framework for global and local data balancing.
3. The paper is well-written and easy to follow.

### Weaknesses
I have the following concerns. *If the authors could properly address them during the rebuttal phase, I am willing to raise my score.*
1. The technical novelty is somewhat limited. While the hierarchical structure and bilevel setup are well-motivated, they mainly combine known techniques such as policy gradients and dynamic sampling into a straightforward framework, without introducing fundamentally new optimization principles.
2. This paper lacks strong theoretical or analytical justification for why the proposed reward formulations (gradient norm and perplexity ratio) should lead to optimal or stable data allocation.
3. Although the experiments are comprehensive, the reported improvements are modest (often within 1–2 points), raising questions about whether the additional implementation overhead (15% extra training cost) is justified.
4. Some important recent baselines [1,2] are missing. The paper would benefit from either a comparison with these methods or a discussion of their relevance.
5. The authors may want to provide deeper analysis of failure cases or scenarios where HBO underperforms, as well as qualitative insights into how the hierarchical balancing actually affects learning dynamics beyond probability curves.

[1] How Abilities in Large Language Models are Affected by Supervised Fine-tuning Data Composition. ACL 2024.

[2] Boosting Multi-Domain Fine-Tuning of Large Language Models through Evolving Interactions between Samples. ICML 2025.

### Questions
Please see Weaknesses.

### Soundness
2

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
This paper introduces Hierarchical Balancing Optimization (HBO), a novel framework designed to address data imbalance and heterogeneity during the fine-tuning of Large Language Models (LLMs). The authors argue that existing methods typically focus on balancing data across different datasets (globally) but neglect imbalances and variations within individual datasets (locally). HBO aims to solve this by enabling the LLM to autonomously adjust data sampling probabilities at both levels. It employs a bilevel optimization strategy featuring a Global Actor, responsible for balancing sampling across datasets, and multiple Local Actors, each optimizing sampling within a specific dataset based on predefined difficulty groups. These actors are trained using reward signals derived from the LLM's training state: the global reward is based on the L2 norm of gradients (indicating learning progress), and the local reward uses the ratio of current to initial perplexity (indicating relative improvement on difficulty groups). The framework is evaluated on several LLM backbones across multilingual and multitask settings, reportedly outperforming various baseline sampling strategies.

### Strengths
1. The paper tackles a significant and nuanced challenge in LLM fine-tuning by explicitly addressing hierarchical data imbalance and heterogeneity (both global, across datasets, and local, within datasets), which is often overlooked by simpler methods.

2. The proposed HBO mechanism, utilizing a bilevel optimization framework with distinct global and local actors guided by rewards derived from the model's own training state, is a novel and sophisticated approach to achieve autonomous, dynamic data balancing.

3. The empirical results are strong and comprehensive, showing consistent improvements over heuristic and existing dynamic sampling baselines across multiple LLM backbones (Llama, Qwen, EuroLLM), diverse tasks (multilingual, multitask), and including insightful analyses like the dynamic evolution of sampling probabilities.

### Weaknesses
1. The framework introduces substantial complexity compared to standard fine-tuning or simpler dynamic sampling. Implementing and tuning the bilevel optimization setup, managing multiple actors (one global, potentially many local), and ensuring stable training with the Reinforce algorithm likely requires significant expertise and effort.


2. The reported computational overhead, while quantified (~15%), is non-negligible and could be a barrier to practical adoption. This additional runtime cost for actor updates and reward computations might be prohibitive, especially for resource-intensive LLMs or scenarios requiring frequent retraining.


3. The method's performance relies heavily on the specific choices for reward functions and the difficulty grouping metric. While the chosen rewards (L2 gradient norm, perplexity ratio) and the SuperFiltering metric  show good results in the experiments, their universal optimality or robustness across different datasets, tasks, or model architectures is not guaranteed. The framework might be sensitive to these design choices.

4. The paper would be improved by comparing this RL-based actor approach to other recent dynamic strategies for multi-domain data, including those based on sample interactions[1] or alternative optimization objectives [2]

[1] Boosting Multi-Domain Fine-Tuning of Large Language Models through Evolving Interactions between Samples. ICML 2025.

[2] Mixture-of-Skills: Learning to Optimize Data Usage for Fine-Tuning Large Language Models. EMNLP 2024.

### Questions
1. Could the authors provide more specifics about the SuperFiltering metric  used to partition data into difficulty groups? How is difficulty defined and measured by this metric, and how robust is the local balancing performance if a different difficulty metric were used?

2. The cyclical patterns observed in the local sampling probabilities (e.g., Figure 2c) are interesting. Is this behavior consistently observed across different datasets and models? What mechanism within the HBO framework drives this apparent emergent curriculum?

3. Please review the formatting of Equation 8; the "where" clause defining PPL seems awkwardly placed on the same line and might be clearer if moved below or reformatted.

### Soundness
3

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
3

### Summary
This paper introduces Hierarchical Balancing Optimization for fine-tuning large language models (LLMs) on diverse datasets by addressing both global and local data imbalance and heterogeneity. HBO uses a bilevel optimization framework with two types of actors: a Global Actor, which adjusts data allocation across different datasets, and Local Actors, which optimize data usage within individual datasets based on difficulty levels. Extensive experiments show that HBO consistently outperforms existing methods, offering significant improvements in model performance. The proposed method provides a comprehensive solution to the challenges of data imbalance and heterogeneity during fine-tuning.

### Strengths
1) HBO effectively addresses both global and local data imbalances, providing a more comprehensive solution to the challenges of fine-tuning LLMs on diverse datasets.

2) The bilevel optimization framework with Global and Local Actors allows for fine-grained control over data sampling, leading to improved model performance across various tasks.

3) Extensive experiments demonstrate HBO's strong applicability across multiple LLM backbones and tasks, consistently outperforming existing baselines and offering superior accuracy gains.

### Weaknesses
1) My main concern is the proposed method adds more computations based on MoS. The reinforcement learning framework, as well as some reward, is similar to the MoS method. This work adds more actors and the grad norm reward, more insights of this field could be added.

2) This paper primarily compares three sampling balancing methods: MoS, MultiUAT, and MultiDDS. However, many of the results are similar to uniform sampling. What is the next step of this field could be discussed.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
