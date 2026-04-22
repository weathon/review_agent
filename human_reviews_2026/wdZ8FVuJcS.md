# Be Affective, Not Just Cognitive - Towards Imparting Pertinent Empathy in Dialogue Agents

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Empathetic Response Generation (ERG) has gained significant attention in diverse areas but still faces challenges that hinder its effectiveness. These challenges include $1$) the lack of affective empathy in existing works, where they exhibit cognitive empathy (feel $\textit{for}$ user); $2$) generate generic responses, where agents address an emotion with monotonous replies; $3$) have limited user relatability. To tackle these issues, we propose incorporating affective empathy in models through additional pre-training. We introduce a benchmark dataset and its collection mechanism, that helps curate an $8.5$GB dataset, enabling the agent to truly feel $\textit{with}$ user. Using this pre-trained model, our framework EMPATH enhances ERG by reducing generic responses. This is achieved by a novel loss function that involves both conversation history and golden response. EMPATH also enhances user relatability by accounting for multiple emotions and their underlying causes via explainability. Through extensive experimentation, we demonstrate the effectiveness of our dataset on our proposed framework and other existing approaches. Additionally, we depict EMPATH's superior performance in ERG on benchmark datasets across various metrics.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
EMPATH effectively enhances affective empathy and response specificity through its synthetic dataset and emotion-aware loss design, achieving significant performance gains across several benchmarks. The explainability-based emotion cause extraction improves interpretability and conciseness compared to traditional clause-based methods. However, the paper lacks verification of the synthetic dataset’s generalization to real-world human dialogues, and the hyperparameter optimization process for the proposed loss remains insufficiently detailed. Furthermore, the statistical significance of reported improvements is not analyzed, and reproducibility information is incomplete.

### Strengths
1.The study introduces a large, well-curated synthetic dataset that substantially expands resources for empathetic dialogue modeling and effectively supports the learning of affective empathy.

2.The explainability-based emotion cause extraction improves interpretability and user relatability by achieving higher conciseness and accuracy than existing approaches.

3.The proposed emotion-aware contextual loss successfully reduces generic responses by balancing emotional alignment, contextual relevance, and empathy validation.

### Weaknesses
1.The real-world validity of the synthetic dataset remains untested. No experiments compare the model’s performance on synthetic versus human-written empathetic data, which raises concerns about its generalization to real user conversations.

2.The paper does not include statistical significance analysis of the results. All metrics are presented as single values without variance or confidence intervals, leaving uncertainty about whether the reported improvements over baselines are meaningful.

3.The process for tuning hyperparameters (λ₁/λ₂/λ₃) in the emotion-aware contextual loss is insufficiently described. The paper does not explain how the loss weights are optimized and lacks ablation studies to assess the contribution of each component.

4.The framework’s behavior in non-empathetic contexts is not examined. There is no evaluation of whether EMPATH can avoid expressing empathy when it is inappropriate, which may lead to misalignment with user intent.

5.The study does not consider multilingual or cross-cultural dimensions of empathy. All experiments are conducted in English, with no analysis of how cultural differences in emotional expression may influence model performance.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a full pipeline for endowing dialogue systems with affective (not only cognitive) empathy. The authors (i) construct a large “empathic statements” corpus by sampling ChatGPT across ~500 rounds (upper bound ≈125k sentences; after filtering ~111k unique) and then applying two-stage paraphrasing with PEGASUS-Large to reach ~81 M sentence, with Perspective-API filtering and 1% human-in-the-loop active learning; (ii) continue pre-training a T5-base model on this corpus to obtain E-T5; and (iii) introduce EMPATH, a decoding/learning framework that combines multi-emotion detection, explanation-based emotion-cause extraction via Integrated Gradients, and a triple loss (emotion-distribution KL, contextual cosine similarity, and an empathy-classifier MSE) within an iterative generation loop.

Evaluation: Empathetic response generation (ERG) is reported on two datasets (EmpatheticDialog, CHASE). Emotion-cause extraction (ECE) is evaluated on RECCON. EMPATH outperforms diverse baselines on automatic metrics (e.g., perplexity, BLEU, Distinct-n, SES) and human preference ratings. Ablations indicate all modules contribute.

### Strengths
1. Resource contribution. The paper releases (and documents filtering for) an 80M-scale corpus explicitly targeting affective empathy, which is rare and potentially valuable to the community.

2. Methodological novelty. Coupling multi-emotion awareness with explanation-based cause extraction and a unified triple loss is conceptually clean and addresses the “template empathy” failure mode seen in prior work.

3. Empirical coverage for tasks. ERG and ECE are both covered, with ablations that clarify the effect of affective pre-training (E-T5), multi-emotion gating, explanation-based causes, and the loss design.

### Weaknesses
1. Baseline fairness. E-T5 enjoys a substantial extra pre-training advantage (~8.5 GB) whereas several baselines are only fine-tuned. There is no comparison to size-matched strong baselines (e.g., FLAN-T5-base) or to frontier LLMs run at high temperature for diversity.

2. Diversity claim under-evidenced. The stated motivation that PEGASUS paraphrasing yields more diverse outputs than ChatGPT is not empirically verified via Distinct-n, self-BLEU, or MAUVE comparisons.

3. Human evaluation scale. Only six graduate annotators are used, and inter-rater reliability is not reported, limiting the strength of human-study conclusions.

### Questions
1. Please clarify the fairness of the baseline comparisons.

2. Please specify the exact model version and hyperparameters used for ChatGPT, Claude, and Gemini (precise model versions,  decoding parameters such as temperature/top-p/max tokens, context length, prompts, and seeds).

### Soundness
2

### Presentation
2

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
The paper proposes a novel framework, EMPATH, to enhance empathetic response generation (ERG) in dialogue systems. It addresses limitations in existing models that exhibit only cognitive empathy and produce generic, unrelatable responses. The authors construct an 8.5GB affective empathy dataset using ChatGPT and Pegasus-based paraphrasing to pre-train a T5-based model. EMPATH combines affective and cognitive empathy, uses explainability for multi-emotion cause extraction, and introduces an emotion-aware contextual loss. Extensive experiments show superior empathetic, contextually relevant, and diverse responses across benchmark datasets.

### Strengths
- The paper correctly identifies an important and underexplored limitation in current ERG systems—the absence of affective empathy in responses.
- The authors perform extensive evaluation using both quantitative metrics (BLEU, ROUGE, Distinct-n, etc.) and human assessments, providing evidence of performance gains.

### Weaknesses
- Writing:
  - The paper is difficult to follow due to the mixed presentation of methodology and implementation details.
  - Many design choices in dataset construction and model formulation are not well motivated. Readers are often left uncertain why a particular step was necessary or how it contributes to affective empathy.
  - Some sentences are very confusing, e.g., “we additionally pre-train the T5 model due to the open-source unavailability of LLMs and its strong performance.” Overall writing quality significantly hinders comprehension.
- Evaluation: Although the paper poses the lack of affective empathy as a key research question, the evaluation does not directly measure it. The presented metrics (BLEU, Distinct-n, etc.) and corpus-level analysis do not show whether responses are indeed more affective rather than merely cognitive. Human evaluation focusing on empathy type or emotional depth would strengthen the claims.
- The core claim—that affective empathy can be explicitly modelled is not contrasted against simpler or more intuitive alternatives. For instance, prompting LLMs with explicit instructions to generate affective (and avoid purely cognitive) responses could serve as an important baseline. Without such comparisons, it is unclear whether EMPATH’s complexity is necessary.

### Questions
See the above comments.

### Soundness
2

### Presentation
2

### Contribution
3
