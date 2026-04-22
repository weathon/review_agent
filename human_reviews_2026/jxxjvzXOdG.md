# Can Reasoning Language Models Think More Creatively? A Study of Reasoning Ability and Overconfidence

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Recent advances in Large Language Models (LLMs) have been largely attributed to improvements in reasoning abilities. Reasoning models trained with Supervised Fine-Tuning (SFT) and Reinforcemenet Learning (RL), such as Policy Optimization (PO), demonstrate significantly superior performance compared to base models.  However, recent studies raise questions about the ability of these reasoning models to achieve creative thinking beyond that of the base model. In this study, we compare the creative problem-solving abilities in mathematics of two types of models: reasoning models and math models that have been further trained on simple mathematical corpora. Our comparison spans two representative open-source LLM families, DeepSeek and Qwen. The results indicate that reasoning models are less effective in generating creative solutions. We attribute the reasoning models’ limited ability to generate creative responses to Overconfidence (OC)—the tendency of models to exhibit excessive confidence in their own outputs. For example, within the DeepSeek family, reasoning models exhibit 15% higher OC compared to the math model, and within the Qwen family, the gap rises to 80%. Notwithstanding their heightened OC, they fail to generate creative responses as intended. We hypothesize that the high OC may stem from overly aggressive probability adjustments for certain tokens during SFT and PO. To examine this hypothesis, we introduce the notion of a High Entropy Segment (HES), defined as a region in which entropy varies sharply. Within these segments, reasoning models tend to exhibit greater heterogeneity compared to other models. Lastly, we measure the proportion of time steps where the model does not generate the most probable token, and observe that reasoning models show a substantially lower rate than math models. This is largely because their distributions contain a substantially greater share of tokens whose probabilities exceed 80% at each step. Our findings will be of great help in understanding and improving reasoning models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this analysis paper, the authors analyze the creativity of Qwen and Deepseek after different training stages using a model-based evaluation approach. They find that further post-training can harm their creativity. The authors also quantify token entropy and finds that models exhibit lower entropy after post-training.

### Strengths
1. The investigation into the diversity of language model outputs is important, as it is closely related to the generalization of language models.
2. Overall, the analysis is comprehensive and multi-faceted.

### Weaknesses
1. The fact that entropy can be lower after post-training has been investigated and is a well-known fact in the field [1-3], which makes the analysis less novel.
2. The creativity evaluation is a bit subjective as it is model-based and reference-based. The authors do not clarify why the reference solution is less creative. Also, the lack of qualitative studies makes it hard for the reader to understand.
3. Besides the analysis, guidance on how to help design better language model training methodologies is also lacking
4. The use of the terms "reasoning model" and "math model" is a bit strange, as a reasoning model can also perform well on math tasks. I understand the naming habit of open-source models before o1 was released in 2024/09, but it would be better not to treat them as parallel categories.

References

[1] The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models, in arxiv 2025

[2] One-shot Entropy Minimization, in arxiv 2025

[3] Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning, in arxiv 2025

### Questions
See the weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper focuses on the trade-off between reasoning ability and creative problem-solving in Large Language Models, comparing two types of models: reasoning models (trained via SFT/RL, including DeepSeek-RL, Qwen-Inst) and math models (only pre-trained on mathematical corpora, including DeepSeek-Math, Qwen-Math). The core findings suggest that reasoning models exhibit higher accuracy in typical mathematical tasks but significantly lower creativity than math models.

### Strengths
- The study addresses a critical yet underexplored gap in LLM research—whether the pursuit of advanced reasoning ability (via SFT/RL) compromises creative thinking. This is highly relevant given the widespread application of reasoning models in fields like mathematics and coding, where both accuracy and novel problem-solving are valuable. 
- The cross-model family design (DeepSeek + Qwen) enhances the generalizability of findings, while the introduction of HES and OAR adds mechanistic depth beyond simple OC ratio measurements. Comparing entropy, token length, and OC across model types (reasoning vs. math) and evaluation scenarios (self-evaluation vs. external evaluation) provides a comprehensive view of the relationship between model training paradigms and creative output.
- The identification of OC as a barrier to creativity, along with the potential of distillation models (e.g., Qwen-Distill) to balance reasoning and creativity, offers actionable insights for improving LLM design

### Weaknesses
- Severe Writing and Table Clarity Issues. e.g.,
1. Inconsistent Model Naming: Section 3.1 clearly defines model aliases, but Table 1 uses inconsistent full names. This forces readers to repeatedly check Section 3.1 to map model types (reasoning vs. math), creating unnecessary confusion.​
2. Unreadable Table Structures: Table 2 is hard to read and we can't have a straightforward comparison between different models in Table 3.
- Insufficient Control and Baseline Design: The inclusion of InternLM2-Math-20B in Table 1 lacks justification (no explanation of its training paradigm in the main text) and creates a baseline mismatch. Unlike the DeepSeek/Qwen models, which have clear "math vs. reasoning" pairs, InternLM2-Math-20B’s classification as a "reasoning model" is unsubstantiated, weakening the paper’s core comparison between model types.​
- Incomplete Explanations: While the paper attributes low creativity to OC, it does not fully address why OC arises during SFT/RL. For example, it mentions "overly aggressive probability adjustments for certain tokens" but provides no analysis of which tokens (e.g., fork tokens vs. common tokens) are affected or how RL loss functions (e.g., GRPO) drive this bias.​

### Questions
- Confusion in Section 5.1: Why is InternLM2-Math-20B classified as a reasoning model? What is the purpose of including InternLM2-Math-20B? 
- Confusion in Section 5.2: Why is 28.9% considered a "high" OC ratio? How to explain OC ratio drops in non-self-evaluation? 
- Confusion in Section 5.3: Why link token length to creativity? 
- Priority of Creativity vs. Accuracy: Why is overconfidence criticized if it boosts accuracy?
- Consequences of Overconfidence: What harms does OC cause beyond low creativity?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the internal behavioral patterns of reasoning-oriented large language models (LLMs) by introducing two quantitative metrics: token-level entropy and overconfidence (OC). Through comparative experiments between math-pretrained models (e.g., Qwen-Math, DeepSeek-Math) and reasoning-enhanced models (e.g., Qwen-Inst, DeepSeek-RL), the authors find that reasoning models exhibit lower overall entropy and higher OC, indicating excessive certainty and reduced creative variability. The study further introduces High Entropy Segments (HES) to analyze local uncertainty spikes, suggesting that these segments may correspond to superficial or unstable creative attempts rather than genuine reasoning diversity.

### Strengths
- The paper provides well-defined and interpretable quantitative metrics (entropy and OC) to assess model uncertainty and reasoning confidence, addressing a relatively underexplored aspect of LLM interpretability.

- The experimental analysis is systematic and statistically grounded, using proper correlation metrics and consistent evaluations across two model families (DeepSeek and Qwen).

### Weaknesses
- While the results are internally coherent, the implications for training methodologies (SFT and RL) are not fully articulated. It remains unclear what practical insight the findings offer for improving reasoning model design beyond observational diagnosis.

- The definition and evaluation of “creativity” rely entirely on LLM-based judges. The paper does not convincingly justify that current LLMs possess the reliability or semantic sensitivity to distinguish truly creative reasoning from surface-level diversity.

-  Model selection is somewhat confusing. The DeepSeek “RL” and Qwen “Inst” variants are presented as reasoning models, but they lack clear long-chain-of-thought (LongCoT) abilities typically associated with advanced reasoning architectures. The experimental lineup therefore might not represent the reasoning paradigm in its full sense.

- There is limited discussion of causality: are high OC and low entropy truly caused by RL fine-tuning, or simply correlated with model style and decoding parameters?

### Questions
Minor stylistic issue: line 141 refers to DeepSeek-Coder with a typo.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focus on the diversity loss or homogeneity of reasoning models compared to pre-trained models in terms of math problem solving.
The author introduced a concept of **High Entropy Segment** to measure the ratio of the mean TE (token-level entropy) of a window compared to  the mean of the entire generation.  The take-away is that, although reasoning models often exhibited (over-)confidence in
the creativity of their generations, the actual proportion of creative solutions was relatively low.

### Strengths
1. The theme to investigate diversity loss or homogeneity of reasoning models is imo very pertinent and deserves more attention to the LLM community.
2. The author introduced a concept of **High Entropy Segment** to measure the ratio of the mean TE (token-level entropy) of a window compared to  the mean of the entire generation. I can envision future works report the time-sensitive heterogeneity of their model using this metric

### Weaknesses
Two major weaknesses:
1. The overall observation is that reasoning models is less creative in math problem solving than a base model pre-trained on math corpus.
A very important hypothesis of the paper is that this is because of **overly aggressive probability adjustments for certain tokens -- referred to as fork tokens**. The authors introduced High Entropy Segment (HES) only to corroborate such observation, yet, no quantitative or qualitative analysis is conducted on these certain tokens to showcase or support the hypothesis. 
The claim *This is largely because their distributions contain a substantially greater share of tokens whose probabilities exceed 80% at each step.* needs more support either by quantitative or qualitative analysis.
I am afraid the take-away message is `reasoning models tend to mysteriously exhibit greater heterogeneity` with an `unverified hypothesis`. This make the contribution less convincing to me.

2. Paper presentation is largely unclear. I found the following confusing:
* Experimental models mentioned in Section 3.1 does match the results shown in Table 1, Section 5. No results for qwen-distill. Where does InternLM2-Math-20B come from

* The results of jaccard, HESR are presented in Table 3 without definition of these terms. (later they were explained in section 6 but the authors need to provide definitions the first time these terms appear. otherwise it is surprising and confusing to the readers when the first time encountering these terms are in the results table. 

* The paper seem to not emphasize its effort in LLM as a judge, but a decent proportion of the results, analysis, and discussions are pertinent to the variance of math&reasoning models as a judge. 

* Same as previous point, def of math model is presented in Figure 1, but abstract and intro did not provide a crystal clear definition

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
3
