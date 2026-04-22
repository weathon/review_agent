# ssToken: Self-modulated and Semantic-aware Token Selection for LLM Fine-tuning

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6, 4

## Abstract
Data quality plays a critical role in enhancing supervised fine-tuning (SFT) for large language models (LLMs), and token-level data selection has emerged as a promising direction for its fine-grained nature. Despite their strong empirical performance, existing token-level selection methods share two key limitations: (1) requiring training or accessing an additional reference model, and (2) relying solely on loss information for token selection, which cannot well preserve semantically important tokens that are not favored by loss-based metrics. To address these challenges, we propose **ssToken**,  a **S**elf-modulated and **S**emantic-aware **Token** Selection approach. ssToken leverages readily accessible history models to compute the per-token loss difference with the current model, which serves as a self-modulated signal that enables the model to adaptively select tokens along its optimization trajectory, rather than relying on excess loss from an offline-trained reference model as in prior works. We further introduce a semantic-aware, attention-based token importance estimation metric, orthogonal to loss-based selection and providing complementary semantic information for more effective filtering. Extensive experiments across different model families and scales demonstrate that both self-modulated selection and semantic-aware selection alone outperform full-data fine-tuning, while their integration—ssToken—achieves synergistic gains and further surpasses prior token-level selection methods, delivering performance improvements while maintaining training efficiency. 
Source code is available at https://github.com/jianke0604/ssToken.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes ssToken, a token-level data selection method for supervised fine-tuning (SFT) of large language models. The key contributions are: (1) a self-modulated selection strategy using retrospective excess loss (REL) that eliminates the need for training a separate reference model, and (2) a semantic-aware attention-based metric that complements loss-based selection. Experiments across models ranging from 3B to 14B parameters demonstrate improvements over full-data fine-tuning and prior token selection methods.

### Strengths
1. Well-motivated approach: The paper clearly identifies two practical limitations of existing token selection methods (reference model requirement and sole reliance on loss) and proposes targeted solutions.
2. Novel self-modulated paradigm: Using the history model instead of a separate reference model is elegant and practical. The formulation of REL is intuitive and eliminates additional training costs.
3. Complementary signals: The integration of attention-based semantic information with loss-based selection is well-motivated. The paper demonstrates that these signals are indeed orthogonal and mutually reinforcing.
4. Comprehensive experiments: Evaluation across multiple model families (LLaMA and Qwen), sizes (3B-14B), and 10 benchmarks provides good coverage. Ablation studies on γ and ρ are thorough.
5. Practical implementation: The lightweight attention extraction design compatible with FlashAttention is a thoughtful engineering contribution.
6. Consistent improvements: ssToken shows stable gains across different settings, which is valuable for practical adoption.

### Weaknesses
1. The reference models in baseline methods are trained on only 10k samples (20% of data), which may not provide fair comparison. A stronger reference trained on more data might close the gap.
2. The claim of "maintaining training efficiency" (Fig. 2) might be misleading since tokens still participate in forward pass. The computational overhead reduction is minimal.
3. The paper acknowledges that "history and current models share identical parameters at the beginning of training, which makes early-stage token selection nearly random" (line 210-212). However, there is no analysis of how many steps are needed before REL becomes effective, nor any experiments showing performance with different initialization strategies.

### Questions
1. Can you provide some analysis of when attention to prompt is vs. isn't a good indicator of token importance?
2. Can you show learning curves of REL values over training to demonstrate when self-modulation becomes effective?
3. Have you tried adaptive selection ratios (e.g., selecting different ρ per sample based on quality)?
4. What happens if you use ssToken for continual learning or multi-epoch training?

### Soundness
2

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
The paper proposes ssToken, a token-level data selection method for SFT that (i) removes the need for an externally trained or stronger reference model and (ii) augments loss-based selection with a lightweight semantic-aware signal derived from attention. 

Instead of computing “excess loss” against a fixed reference model, the authors consider Retrospective Excess Loss (REL) computed against a history model, which is futher combined with attention-based token importance, another measurement of token weight by summing up attention scores over all prompt tokens. Lastly they consider the convex combination of the two signals and only train on top-x fraction of response tokens.

Empirically, they showed ssToken outperforms both full-data SFT and prior token selection baselines across various base models and benchmarks.

### Strengths
1, The paper proposed an innovative prompt-focused attention metric
2, Details are well discussed, such attention-layer’s preference, hooks+recompute approach for FlashAttention compatibility, 
3, The paper is well structured (motivation → method → ablations → results) with comprehensive figures and explicit algorithmic details
4, The topic of token-level data selection is an increasingly important topic for better data-efficiency. The methods introduced in the paper are easy to adapt and implement.

### Weaknesses
1, While the author claims the attention based measurement is lightweight and no reference training is needed, it still helps to quantify the additional costs of 1) forward passes to compute REL using history model (2) attention recomputation, and especially when scaled to larger models and longer contexts
2, The paper notes that if history agrees with the current model at the beginning then the selection is “nearly random”, and suggests EMA, but eventually admits that EMA doesn’t yield gain and leads to additional overhead. What’s the take away on this discussion? The authors are expected to explain why this is the case, and whether this is due to small SFT. Does adaptive history update help on longer training horizons or larger corpora?
3, The attention-based approach focuses exclusively on attention from prompt, and overlooks contributions from other response tokens. This omission can be problematic in reasoning-heavy tasks such as mathematical problem solving, where the model must follow a chain of logic across intermediate steps. In such cases, the next step often depends more on information contained in previous reasoning steps than on the original prompt. Ignoring attention flows among response tokens may therefore hurt model’s ability to capture these intra-response dependencies.
4, A more concerning point is that all experiments were performed with LoRA, which is a non-standard approach for fine-tuning LLMs. This leaves the audience wondering if the method works for full tuning and or even pretraining.

### Questions
1, How many extra forwards/backwards per step are required to compute REL and the attention metric? Is REL computed with an additional pass through the history model each step/batch, or are there cached losses? 
2, You mention an optional EMA but ultimately fix the base model as history due to small data. For larger/longer SFT, do you expect EMA to stabilize selection and reduce early randomness? Can you share a small-scale ablation to illustrate the trend?
3, In prompt based attention approach, what happens if the response context is long? Will position bias become a problem? Does your analysis cover this regime?  
4, Which exact layer index do you use across models? Do model families disagree on which depth is best?
5, Have you considered head-weighted aggregation (e.g., entropy-based or gradient-based head importance) rather than simple averaging?
6, can you justify why omitting response tokens' contributions in the attention approach? 
7, Do the conclusions change for full-parameter SFT vs LoRA? Can you add ablation studies to support this?

### Soundness
2

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
This paper proposes ssToken, an adaptive token selection approach to address two major issues in existing traditional approaches: (i) adopting a stronger model as the reference model for the token selection is not practical, and (ii) the token-level losses do not caption the semantic importance of tokens in the given contexts.  The proposed method doesn't rely on an additional reference model and using its history to improve the token selection and introduces a new semantic-aware token importance estimation metric. The empirical evaluation shows that the proposed method achieves the siginificant improvemnts in existing approaches.

### Strengths
This paper is well written. It clearly presents the current disadvantages of traditional approaches and those statements are convincing. 

The experiental results are solid. It has covered multiple modern LLMs including LLaMA-3.2 & 31 and Qwen-2.5 in different scales. Also, the results are evaluated in multiple benchmarks.  

The ablation studies cover all necessary components.

### Weaknesses
I am not in this field so I am afraid that I may not be able to identify any key weakness of this work. So, I set my score 8 (a clear acceptance). I do have a few questions just for clarification (see Questions section). They do not affect my scores.

### Questions
In Table 1, the performance of TokenCleaning is lower reported. In their paper's Table 1, *Token Cleaning: Fine-Grained Data Selection for LLM Supervised Fine-Tuning*, the AVG score of LLaMA-3.2-3B is 53.00 (for self-evolving clearning). It seems that the difference is caused by the difference in the evaluation set; so, I have a few questions: 
1. Why is a different evaluation dataset, instead of the same one used in TokenCleaning's Table 1, used in this submission? 
2. How do you choose the evaluation set? Is it common to use a different evalaution set in this field? I noticed that all baseline approaches in their own papers (e.g. TokenCleaning, Pho) are using different evaluation sets (and even training set or models for some baseline papers). How can I really know which approach is better when there is no standard way to compare different methods?   
3. Also, should the LLM generation process have some randomness? As the next token prediction is based on a sampling procedure instead of simply taking the largest possible token. However, I found all three of this submission,  the TokenCleaning paper, and the Pho paper, didn't report the confidence interval in the main experimental result table. It seems to be a common rule in this field but I don't understand that.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes ssToken, a self-modulated and semantic-aware token selection method for LLM supervised fine-tuning. It addresses limitations of existing methods (needing extra reference models and relying only on loss) by using historical models to compute Retrospective Excess Loss (REL) and an attention-based semantic metric. Experiments on 3B-14B models (LLaMA, Qwen) across 10 benchmarks show ssToken outperforms full-data fine-tuning (up 4.3%) and prior token methods (up 2.8%), maintaining efficiency. It has a limitation of needing manual ρ setting, with future work on adaptive ρ.

### Strengths
1. It creatively combines self-modulated signals (via the model’s historical trajectory for REL) and semantic-aware attention metrics, solving prior reliance on external models and loss-only selection.
2. It has rigorous methodology with grounded components and lightweight implementation, plus sufficient validation across models/benchmarks and ablation studies ensuring quality.
3. It follows a clear problem-method-result structure, with precise technical definitions and transparent limitations, enabling easy understanding.
4. It boosts LLM fine-tuning performance without extra costs, advancing semantic-informed self-evaluative data selection with academic and practical value.

### Weaknesses
1. It relies on manual tuning of the token selection ratio ρ, and without an adaptive mechanism to adjust ρ based on model capacity or data quality, it adds overhead for practitioners and limits generalization across diverse model families or domains.
2. The optional EMA-based update of the history model is not fully explored—experiments only use a fixed base model as the history model, leaving unaddressed whether adaptive history model updates could bring more stable guidance in large-horizon training scenarios.
3. It lacks in-depth analysis of why deeper attention layers outperform shallow ones in semantic-aware selection beyond citing prior studies, and no discussion on how layer selection might vary across different task types (e.g., QA vs. reasoning) limits methodological completeness.

### Questions
1. You used manual token selection ratio (ρ) – did you test any initial adaptive ways to adjust ρ (e.g., linking to model loss progress)? If not, what’s your hypothesis for designing such a mechanism?  
2. You mentioned EMA-updatable history models but used fixed ones – what EMA hyperparameters (e.g., α) did you test? Do you plan to try EMA in longer training where fixed models may lag?  
3. You chose deeper attention layers for semantic selection – could you test if layer choice differs by task (e.g., QA vs. reasoning) to strengthen the method’s generality?  
4. You noted compatibility with FlashAttention – can you quantify the computational cost of your attention calculation for sequences longer than 2048 tokens?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces ssToken, a token-level selection algorithm for fine-tuning Large Language Models (LLMs). Unlike previous token-level selection methods that rely on a reference model, ssToken employs a historical model as a proxy to dynamically compute the loss difference with the current model. The authors also develop an attention-based metric to assess token importance. These two metrics are then combined to select important tokens. Experiments conducted across various models and benchmarks demonstrate the effectiveness of ssToken.

### Strengths
1. ssToken does not require a pre-trained reference model, making it more cost-effective for token selection.

2. In addition to the loss-based metric, the authors also developed an attention-based metric to evaluate token importance.

3. The experiments are comprehensive and demonstrate the strong performance of ssToken.

### Weaknesses
1. There is a contradiction between Equation (2) and Equation (3) that requires further explanation. In Equation (2), tokens are ranked highly if the current model’s loss on the token is large. In contrast, Equation (3) prioritizes tokens for which the current model’s loss is small. The authors should provide more insight into the rationale behind this difference.

2. Is the calculation in Equation (3) performed per training step or per epoch? How does the frequency of this calculation affect token selection performance?

3. Could the authors provide some theoretical justification or analyses for why using the current model (rather than a fixed reference model) for loss calculation leads to better performance?

### Questions
See **Weaknesses**

### Soundness
2

### Presentation
3

### Contribution
2
