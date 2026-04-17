# Parallel Scaling Law: Unveiling Reasoning Generalization through A Cross-Linguistic Perspective

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 6

## Abstract
Recent advancements in Reinforcement Post-Training (RPT) have significantly enhanced the capabilities of Large Reasoning Models (LRMs), sparking increased interest in the generalization of RL-based reasoning. While existing work has primarily focused on investigating its generalization across tasks or modalities, this study proposes a novel cross-linguistic perspective to investigate reasoning generalization. This raises a crucial question: Does the reasoning capability achieved from English RPT effectively transfer to other languages?
We address this by systematically evaluating English-centric LRMs on multilingual reasoning benchmarks and introducing a metric to quantify cross-lingual transferability. Our findings reveal that cross-lingual transferability varies significantly across initial model, target language, and training paradigm. Through interventional studies, we find that models with stronger initial English capabilities tend to over-rely on English-specific patterns, leading to diminished cross-lingual generalization. To address this, we conduct a thorough parallel training study. Experimental results yield three key findings: $\textbf{First-Parallel Leap}$, a substantial leap in performance when transitioning from monolingual to just a single parallel language, and a predictable $\textbf{Parallel Scaling Law}$, revealing that cross-lingual reasoning transfer follows a power-law with the number of training parallel languages. Moreover, we identify the discrepancy between actual monolingual performance and the power-law prediction as $\textbf{Monolingual Generalization Gap}$, indicating that English-centric LRMs fail to fully generalize across languages. Our study challenges the assumption that LRM reasoning mirrors human cognition, providing critical insights for the development of more language-agnostic LRMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the Parallel Scaling Law to systematically model and empirically study how the volume of parallel training data influences the cross-linguistic generalization of reasoning abilities in large language models. The research aims to uncover the mathematical relationship and underlying mechanisms that govern the scaling of these multilingual reasoning capabilities.

### Strengths
This paper proposes the Parallel Scaling Law to systematically investigate how increasing parallel data specifically enhances complex reasoning generalization across multiple languages. The authors conduct extensive experiments on two LLMs across ten languages, providing a comprehensive analysis of cross-lingual transfer performance.

### Weaknesses
- The research scope is a bit narrow, focusing primarily on STEM subjects. However, reasoning tasks often depend on cultural or common-sense knowledge implicit in the prompt language. The paper seems to ignore or inadequately address how parallel data addresses the cultural grounding mismatch between languages, rendering the direct comparison of reasoning skills problematic.


- The study uses translated versions of standard datasets into target languages. Translation quality and cultural or linguistic adaptation issues may cloud the results. It is unclear whether there is impact caused by translation artifacts, as the solutions to problems may depend on specific linguistic or cultural contexts.

- It is unclear about the details of language consistency. How to calculate the reward of language consistency and how to measure the Off-tag are not clearly explained in the paper. Does this paper consider the impact of code-switching? Is there any relation between the Off-tag and accuracy? 

- For the cross-lingual transfer, this paper only consider transfer from English to other languages. It would be more convincing if the authors could provide more comprehensive experiments, such as transfer from other languages to English or between non-English languages. Qwen also has strong Chinese capabilities, will it produce the similar conclusion when transferring from Chinese to other languages?

- It is unclear about the impact of hyperparameter in Equation (7). How to choose the value of $\lambda_1$, $\lambda_2$ and $\lambda_3$? Is there any relation between the value of α and the performance of cross-lingual transfer? There needs more analysis about the impact of hyperparameters.

### Questions
- Llama 3.1 has better cross-lingual generalization ability compared to Qwen 2.5. Is there any influence caused by the training data or model architecture that contributes to this improvement? As the Llama is English-centric but Qwen is English and Chinese-centric. Does Qwen will perform better transfer on other Sino-Tibetan languages?

- What is the 'reasoning component' mentioned in line 285? Does it refer to a specific module or mechanism within the model architecture?

- How to ensure the quality of translated datasets? Are there any evaluation metrics or human evaluations conducted to assess the quality of the translations? Is there any cleaning process applied to the translated datasets to remove potential noise or errors introduced during translation?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper systematically investigates the cross-lingual generalization of reasoning capabilities in English-centric Large Reasoning Models (LRMs). Through observational, interventional, and parallel training studies, it reveals that reasoning skills acquired via English reinforcement learning do not fully transfer to other languages, highlighting a Monolingual Generalization Gap. The authors propose a Just Go Parallel training strategy and uncover two key phenomena: the First-Parallel Leap and a Parallel Scaling Law, which show that cross-lingual transferability improves with the number of parallel languages following a power law, but with diminishing returns. These findings challenge the assumption that reasoning is language-agnostic and offer practical insights for developing more linguistically universal reasoning models.

### Strengths
This paper introduces a cross-lingual view of reasoning generalization, introducing the MTI metric to quantify how well English-trained models transfer to other languages.
2. A three-tier pipeline—observational, interventional, and parallel-training—progressively isolates factors and proves that small parallel data deliver big gains.
3. Key findings reveal three phenomena: adding just one parallel language triggers a disproportionate accuracy and MTI jump (the First-Parallel Leap); further parallel data obey a power-law with exponent <1 and rapidly diminishing returns (the Parallel Scaling Law); and English-only training underperforms this law, exposing a Monolingual Generalization Gap that reveals reliance on language-specific shortcuts rather than truly language-agnostic reasoning.

### Weaknesses
Q1: In the Observational Study, you mention using prompt hacking techniques to control model generations for evaluation. This process appears to be training-free. However, in the MTI metric, how do you distinguish between the training language set and the unseen language set? Additionally, in line 158, the phrase “Even with the same training data”—could you clarify which specific data this refers to?


Q2: For Section 3.2, including results from one additional model(e.g. mistral_instrcut or other sft models) would make the observed trend more convincing and provide stronger empirical support for your argument.

Q3: The Parallel Scaling Law results are interesting. However, could the observed patterns be influenced by the total number of training samples? If I understand correctly, as the number of languages increases, the total amount of training data also grows proportionally. It would be more sound to control for total training size—keeping it constant while varying the number of languages—to better isolate the effect of multilinguality itself.

### Questions
In Section 3.2, could there be a phenomenon akin to a multilingual alignment tax? While weaker models may show larger gains in multilingual generalization after training compared to stronger models, their performance ceiling might still be lower. In other words, even though the relative improvement is larger, the final performance could remain below that of inherently stronger models.

### Soundness
3

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
4

### Summary
This paper investigates cross-lingual reasoning generalization in Large Reasoning Models (LRMs), particularly those fine-tuned with Reinforcement Post-Training (RPT). It shows that cross-lingual transferability varies significantly depending on the initial model, target language, and training paradigm. Through observational, interventional, and parallel training studies, the authors find that models with strong initial English performance tend to over-rely on English-specific linguistic patterns, resulting in weaker generalization to other languages. They propose two laws of multilingual reasoning: 1. First-Parallel Leap — adding a single parallel (non-English) training language yields a disproportionately large improvement in transferability. 2. Parallel Scaling Law — cross-lingual reasoning performance follows a power-law relationship with the number of parallel languages used during training, with diminishing returns as the number increases.

### Strengths
1. Strong experimental design — The paper systematically isolates variables via observational and interventional studies. The use of off-tag and MTI (Multilingual Transferability Index) metrics offers a clear quantitative measure of transfer performance
2. Insightful findings — The analysis of initial model types, families, and sizes reveals counterintuitive patterns: smaller or less English-specialized models transfer better, and RL-based fine-tuning provides strong advantages in low-resource languages
3. Conceptual clarity — The introduction of First-Parallel Leap and Parallel Scaling Law provides interpretable and theoretically grounded generalization principles.

### Weaknesses
1. Limited generalizability — The study focuses exclusively on mathematical reasoning, leaving open whether the proposed laws extend to other domains like code generation or open-domain logic.
2. Marginal absolute gains — For hard tasks like GPQA, the baseline accuracies (~20–30%) are close to random; thus, small improvements may not represent meaningful generalization effects.
3. Overemphasis on fitted power laws — While visually appealing, the scaling relationships may be too simplified; no theoretical justification beyond empirical fitting is given.

### Questions
1. A control baseline where the total data size is constant, but the dataset remains monolingual (e.g., 2× English vs. 1× English + 1× other language) might be helpful to see the improvement of multi-lingual training under a constant dataset size. This would clarify whether the improvement is due to cross-lingual exposure or data scale.
2. Does the Parallel Scaling Law hold if the additional languages are typologically similar (e.g., Romance languages only)?

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
This work studies whether (and how) reasoning abilities learned through reinforcement post-training on English can transfer to other languages. It introduces a multilingual transferability index, with an observational comparison of open-source LRMs while forcing models to reason in a specific language, finding that transfer varies by model, language, and training paradigm. They find that RL transfers better than SFT in general, and helps especially with low-resource languages. They also introduce a study of training on cross-lingual pairs solving the same problem, finding that this boosts performance and MTI in a predictable manner by a scaling / power law, with diminishing returns.

### Strengths
1. This work takes a quite refreshing lens on cross-lingual generalization by developing a concrete index for transferability.
2. The study examines a reasonable range of model sizes, which use different post-training recipes for reasoning (thus, LRMs).
3. The findings in Section 3 on model size are mostly intuitive, that a smaller model is sufficient for MATH500 but stronger models transfer better for more difficult benchmarks.
4. The first-parallel leap phenomenon (if validated through more experimentation) seems actionable, and can guide curriculum / multilingual RPT recipe design.

### Weaknesses
1. For the observational study and parallel scaling law, all models examined are from a single family of models (Qwen2.5), it would be best for diversity to not just be in model size, but to show that this applies to several other model families, especially since the intervention study does consider Llama-3.1-8B. 
2. The claim in l283-284 that “a less specialized model may be better suited for broad cross-lingual transfer” does not seem to be consistent with the Qwen2.5 results that the more specialized model (math) has a higher MTI. 
3. I am a bit concerned about the effects of the language consistency reward along with the prompt hacking, it could be possible that low-resource gains should be attributed to the language-consistency reward.

### Questions
1. In the parallel study, were the steps, rollouts, total tokens, and prompts held constant across the languages? If not, can you add a fixed-budget ablation so that the parallelism claim ca be isolated from the effects of data quantity. 
2. In line 223, GRPO stands for Group Relative Policy Optimization, not Rollout.

### Soundness
3

### Presentation
3

### Contribution
3
