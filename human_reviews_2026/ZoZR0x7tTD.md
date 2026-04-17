# Multilingual Routing in Mixture-of-Experts

- Decision: Accept (Poster)
- Scores: 6, 8, 2, 6

## Abstract
Mixture-of-Experts (MoE) architectures have become the key to scaling modern LLMs, yet little is understood about how their sparse routing dynamics respond to multilingual data.  In this work, we analyze expert routing patterns using parallel multilingual datasets and present highly interpretable layer-wise phenomena.  We find that MoE models route tokens in language-specific ways in the early and late decoder layers but exhibit significant cross-lingual routing alignment in middle layers, mirroring parameter-sharing trends observed in dense LLMs. In particular, we reveal a clear, strong correlation between a model's performance in a given language and how similarly its tokens are routed to English in these layers. Extending beyond correlation, we explore inference-time interventions that induce higher cross-lingual routing alignment. We introduce a method that steers the router by promoting middle-layer task experts frequently activated in English, and it successfully increases multilingual performance. These 1-2% gains are remarkably consistent across two evaluation tasks, three models, and 15+ languages, especially given that these simple interventions override routers of extensively trained, state-of-the-art LLMs. In comparison, interventions outside of the middle layers or targeting multilingual-specialized experts only yield performance degradation. Altogether, we present numerous findings that explain how MoEs process non-English text and demonstrate that generalization is limited by the model’s ability to leverage language-universal experts in all languages.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the behavior of different languages ​​in the routing mechanism of a mixture-of-experts (MoE) model in multilingual scenarios and proposes a method to improve multilingual performance by intervening in routing. This paper proposes a soft intervention approach, which adjusts expert scores before routing, and a hard intervention approach, which forcibly activates certain experts. Experimental results demonstrate that the proposed method effectively improves the model's performance in other languages.

### Strengths
1. The authors conduct an in-depth exploration of the behavior of different languages ​​in the MoE's large language model and summarize four findings.

2. There is a close connection between the proposed method and the findings.

3. The proposed method can improve the performance of the model in other languages ​​without further training.

### Weaknesses
1. Lack of further explanation for how the proposed method work.

2. Experiments lacking qualitative analysis.

### Questions
1. The authors do not explain why closer approximation to English leads to improved performance. Can similar effects be observed by making a low-resource language closer to a high-resource language?

2. There are no qualitative experiments demonstrating the inference process after intervention, or whether using an English expert leads to changes in the output language during inference.

3. The authors' proposed method involves many hyperparameters. To what extent do the choices of these hyperparameters affect the results?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This work conducts an in-depth study on the routing pattern  of MoE models on multi-lingual data. Authors conclude five findings via visualization and one more finding about how to leverage the observations to further improve the model.

### Strengths
1) The writing is very clear. It is great to highlight the findings in this paper as it is indeed a bit easy to get lost when reading the detailed analysis and the description of experimental setups.
2) The finding 6 is a good evidence helping support the correctness of the findings via visualisation. It is also interesting that simply changing the routing decision by aligning with english can improve the performance. 
3) Consider MoE is becoming a default choice of most LLMs. It indeed motivates the community to rethink the strength and weakness of using such a sparse model. For the long term purpose, as the compute is getting cheaper and cheaper, we might roll back to dense for better and more robust universal representations.

### Weaknesses
1) It would be helpful to provide more insights about how these findings are connecting to each other, and maybe draw a figure to explain this in the final draft.
2) Minor: To my best knowledge, this work (https://arxiv.org/abs/2402.01739) should be one of the earliest work taking a closer look at this problem. I understand that the model has been a bit too weak and some designs have been outdated, but it would be helpful to discuss this work as a reference properly.

### Questions
NA

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Paper presents a descriptive analysis of how MoEs process multilingual data and investigates the router patterns of tokens for different languages and presents the trends like tokens are routed in language-specific ways in the early and late layers, but exhibit significant cross-lingual alignment in the middle layers, mirroring phenomena observed in dense models. The paper's core contribution is demonstrating a strong correlation between a model's performance in a given language and the similarity of its token routing to English in these middle layers. Paper then presents a simple intervention to leverage this phenomena to sterr model to activate English-identified task-specific experts for non-English inputs, however this method yields only a marginal boost in performance.

### Strengths
1. The paper is clearly written and easy to follow, with effective visualizations and well-designed, information-theory informed probes to analyze routing patterns in multilingual data. The analysis  using an entropy-normalized JS divergence metric to study routing divergence, complemented by routing entropy and consistency analyses is good to me.

2.The work includes comparisons across model scales and multiple languages, enhancing the generality of the findings.

3. . The figures, especially Figure 1 and Figure 2, are excellent visualizations that clearly communicate the central hypothesis of the U-shaped routing divergence and its correlation with model performance.

4. The finding of complete modularity between task-specialized and language-specialized experts (Finding 5) is particularly strong and clean.

### Weaknesses
1. The proposed method improves performance by only 1–2%, which is not practically meaningful. The analysis and intervention seem too complex for such minimal payoff, which makes it hard to justify the paper’s overall significance.

2. The proposed “steering” approach feels like a post-hoc, brute-force fix. Forcing a model to use certain experts is more of a hack than a genuine solution. The method shows that steering can slightly help, but not that it’s a meaningful or scalable way forward.

3. The observation of middle layers of LLMs form a shared language space has already been well established for dense models. Showing that MoEs follow the same pattern is interesting but not surprising. The paper essentially confirms existing ideas rather than offering something genuinely new.

### Questions
1. The paper suggests a causal link between routing alignment and multilingual performance, but the evidence feels somewhat weak. A stronger case would involve showing that performance systematically improves as the degree of steering increases, or that steering towards unrelated experts consistently hurts performance. Is there enough evidence to make a strong causal claim here?

2. In Table 2, the intervention appears to slightly degrade performance in English. How do you interpret this? Could it be that forcing a fixed subset of experts interferes with the model’s naturally optimized routing for its primary language?

3. Regarding Finding 5, the clean separation between multilingual and task-specific experts is quite striking. Does this separation remain consistent across different expert selection thresholds (τ)? If the threshold were lowered, could there be experts that participate in both language and task representations, and what might that mean for the design or effectiveness of your intervention?

### Soundness
2

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
The paper investigates multilingual behavior in MoE LLMs. Using models such as Qwen3-30B, Phi-3.5-MOE, GPT-OSS-20B, and OlmoE, the authors analyze expert routing patterns across languages and layers. They find that early and late layers are language-specific while middle layers exhibit cross-lingual alignment. They propose inference-time “routing interventions” that encourage the activation of experts often used for English, reporting 1–2% improvements on multilingual benchmarks (MGSM and Global-MMLU).

### Strengths
- Timely topic: Multilingualism in MoE architectures is underexplored; the paper touches on a relevant question for LLM scaling.
- Comprehensive dataset use: Employs multiple evaluation datasets (FLORES, BELEBELE, MGSM, Global-MMLU).
- Interpretability focus: The routing-divergence visualizations and entropy analyses are informative.
- Reproducibility: The methodology is relatively transparent, with sufficient detail to replicate the experiments.

### Weaknesses
- Marginal contribution: The reported 1–2% improvements are minor and lack depth of analysis. No theoretical insight or convincing causal mechanism is demonstrated. The “steering” interventions are heuristic and depend heavily on ad hoc hyperparameter tuning (λ, τ, and layer ranges). No systematic ablation or generalization evidence is provided.

- Interpretation overreach: The claim of “causal relationships” between routing alignment and multilingual performance is not empirically supporte (correlation is misinterpreted as causation).

- Lack of broader impact or insight: The paper neither improves multilingual modeling methods nor contributes significant interpretability tools.

### Questions
- The findings that early and late layers are language-specific, while middle layers exhibit cross-lingual alignment, also hold for dense-model architectures, as noted by the authors. You may want to reference this recent paper on the topic: https://aclanthology.org/2025.findings-acl.1385
- Overall, my assessment is toward acceptance, as it is a timely topic.

### Soundness
2

### Presentation
3

### Contribution
3
