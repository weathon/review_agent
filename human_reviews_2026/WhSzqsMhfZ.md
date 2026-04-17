# Evaluating and Improving Cultural Awareness of Reward Models for LLM Alignment

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Reward models (RMs) are crucial for aligning large language models (LLMs) with diverse cultures. Consequently, evaluating their cultural awareness is essential for further advancing global alignment of LLMs. However, existing RM evaluations fall short in assessing cultural awareness due to the scarcity of culturally relevant evaluation datasets.
To fill this gap, we propose Cultural Awareness Reward modeling Benchmark (CARB), covering 10 distinct cultures across 4 cultural domains.
Our extensive evaluation of state-of-the-art RMs reveals their deficiencies in modeling cultural awareness and demonstrates a positive correlation between performance on CARB and downstream multilingual cultural alignment tasks.
Further analysis identifies the spurious correlations within culture-aware reward modeling, wherein RM's scoring relies predominantly on surface-level features rather than authentic cultural nuance understanding.
To address these, we propose Think-as-Locals to elicit deeper culturally grounded reasoning from generative RMs via reinforcement learning from verifiable rewards (RLVR) and employ well-designed rewards to ensure accurate preference judgments and high-quality structured evaluation criteria generation. 
Experimental results validate its efficacy in mitigating spurious features interference and advancing culture-aware reward modeling.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors focused on the important problem of cultural awareness in reward models (RM). Their contributions include proposing a multilingual, culturally aware reward model (CARB), covering 10 typologically diverse cultures and 4 key cultural dimensions. By analyzing over 20 classifier-based and generative RMs, the authors demonstrate that generative RMs outperform classifier-based ones in capturing cultural nuance. Their results further highlight that value alignment as the most difficult domain. The paper also contributes to improving the RMs with Think-as-Locals, a dual reward function to improve the training of culturally aware RMs.

### Strengths
S1: The paper focused on a timely and the important problem of cultural awareness in reward models. 

S2: The proposed CARB benchmark explicitly designed for cultural awareness in reward models, filling a genuine research gap, unlike prior multilingual benchmarks ignoring this aspect. CARB covers 10 typologically diverse cultures and 4 key cultural dimensions.

S3: The paper provides comprehensive evaluations on both classifier-based and generative RMs (over 20 models), yielding valuable insights on comparative performance. Nice causal and spurious correlations analysis provide interesting insights of why current RM failed at cultural awareness. 

S4: The proposed dual reward function (correctness + criteria appropriateness) is justified and experimentally validated to show effectiveness.

### Weaknesses
W1: Although some human validation exists (L173), the dataset’s independence from model-generated bias and with human cultural experts in judging CARB as well as RM judgments.

W2: While carefully filtered and validated by some humans, the CARB benchmark heavily depends on LLM generated data. Including authentic human generated examples would strengthen the benchmark.

W3: [Minor] Despite the coverage of 10 cultures, CARB favours high(er)-resource languages. Low-resource cultures (e.g., African or Indigenous) remain underrepresented, including some of the lower-resource cultures could further strengthen the work and show generalization.

### Questions
Comment: Values in Fig. 3 are a bit difficult to see, consider using a darker colour for the grid.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses an under explored yet crucial dimension of aligning large language models (LLMs): cultural awareness in reward modeling. The authors first diagnose the shortcomings of existing reward model (RM) evaluations, which mainly focus on general capabilities and lack culturally grounded benchmarks. To bridge this gap, they introduce CARB (Cultural Awareness Reward modeling Benchmark), covering 10 cultures across 4 cultural domains. The authors then show that CARB scores correlate strongly with downstream multilingual cultural alignment tasks, based on which the paper claims CARB is effective as an evaluation benchmark for RM. Subsequently, the authors show that current RMs struggle to model authentic cultural nuances, often relying on surface-level correlations rather than meaningful cultural reasoning.
To mitigate the identified issues, the authors finally propose Think-as-Locals, a reinforcement learning from verifiable rewards (RLVR) approach, designed to elicit deeper, culture-aware reasoning and reduce spurious feature reliance.

### Strengths
- The paper is well written and easy to follow, with a clear logical flow: each section identifies a limitation and proposes a corresponding solution. 

- The topic is timely and important, addressing cultural inclusivity/awareness, a crucial but underexplored aspect of global AI alignment.

- The strategy "Think-as-Locals" seems to be an effective RLVR method to elicit deep cultural understanding in generative RMs.

- The experiments are quite extensive with large-scale multilingual evaluation and ablation studies.

### Weaknesses
**Major Weaknesses**

(W1) The authors use GPT-4o to translate English prompts into other languages, and report that three human annotators manually refined them. However, according to Section B.1, these annotators are independent undergraduate and graduate students. However, the authors do not report if the human annotators are native speakers (or even fluent speakers) of the studied languages. This raises concerns about the quality and cultural authenticity of the prompts. 

(W2) The same concern applies to the human annotation agreement results. If the annotators are not from the respective cultural backgrounds, the evaluation may not fully reflect genuine cultural preferences. A more rigorous evaluation involving native speakers/members of the cultural regions would increase the reliability of CARB and better ensure that it captures authentic human judgments.

(W3) In Section 5, GPT-4o is used to rate generated responses for cultural relevance, faithfulness, and helpfulness. These are complex subjective dimensions, and it is debatable whether GPT-4o can serve as a reliable judge for such evaluations. Even if using LLM-as-a-Judge might be necessary for an efficient evaluation, the authors should provide a human meta evaluation of the LLM-judge in the different cultural settings.

(W3) While the Think-as-Locals method appears effective, it likely increases computational cost, as rewards depend on the full reasoning traces rather than final outputs.


**Minor Weaknesses**

- The selected set of cultures is essential to this work and its list should not be hidden in the Appendix (Table 4).

- While CARB covers 10 cultures, its cultural coverage remains limited compared to the global landscape; especially with regard to lower resource languages. Extending the benchmark to more low-resource or underrepresented regions (e.g., from Africa) would enhance representativeness.

- Figure 5 shows that the linear relationship is weak and statistically insignificant for M-RewardBench but strong for CARB. The paper should provide a clearer explanation or hypothesis for this difference.

- Including an analysis of failure cases (e.g., where Think-as-Locals still misinterprets cultural cues) would enhance the understanding of this approach.

- Cultural evaluation is a rapidly growing area and the authors list a lot of relevant works. Some additional, recent works could also be relevant to their section on related work: https://aclanthology.org/2024.acl-long.345/ https://aclanthology.org/2024.acl-long.862/ https://aclanthology.org/2025.naacl-long.402/ https://arxiv.org/abs/2505.21693

- The proposed reward R_{appr} is claimed to be motivated by intrinsic probability, but no intuitive explanation is given. A brief intuition behind this design would improve understanding.

- The authors mention "CARE" in line 275. I first thought this was a typo (of their "CARB"). But actually, it is a different dataset which is only properly introduced with reference in line 408.

**Typos **

Line 143: structured. -> structured, (comma instead of full stop)

### Questions
- How scalable is CARB in practice? Can it be easily extended to new cultural domains or low-resource languages with limited data?

- How does CARB perform when applied to unseen cultures or languages in post-training or adaptation scenarios?

- Did the human annotators have a suitable background to be able to judge and annotate the diverse cultural regions? (see W2)

- Does the LLM-as-a-Judge align with human judgments? (see W3)

- How does the training time of the more complex procss compares to using a standard reward computed on the final answer only? (see W4)

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
4

### Summary
This paper introduces CARB (Cultural Awareness Reward modeling Benchmark), a benchmark designed to evaluate and improve the cultural awareness of reward models (RMs) used in aligning large language models (LLMs). CARB includes 10 cultures and 4 domains covering cultural commonsense, values, safety, and linguistics, using a Best-of-N evaluation on human-curated prompts and LLM-generated responses. Results show that current RMs perform inconsistently across cultures, often relying on spurious correlations such as surface linguistic features rather than genuine cultural understanding, though CARB scores correlate positively with downstream multilingual cultural alignment performance. To address these limitations, the paper also introduces Think-as-Locals, an RLVR-based framework that encourages RMs to explicitly generate culturally grounded evaluation criteria before judgment. Experiments demonstrate that Think-as-Locals mitigates spurious biases and enhances the cultural sensitivity and reasoning depth of reward models.

### Strengths
**S1.** CARB covers 10 cultures with 4 cultural domains, and the final dataset contains 8,576 prompts with 24 LLM-generated responses, which is very diverse.

**S2.** The study involves large-scale and comprehensive evaluations of a lot of different reward models (such as classifier-based and generative reward models), cross-lingual comparisons, and correlation studies with downstream multilingual alignment tasks.

**S3.** The paper introduces new framework called Think-as-Locals that is shown to encourage culturally grounded reasoning and effectively mitigates spurious correlations in reward modeling, as it outperform strong baselines across multilingual cultural benchmarks, even compared to other method with retraining (i.e. RM-R1).

### Weaknesses
**W1.** Here I group all weaknesses related to CARB as a dataset:
- CARB relies heavily on GPT-4o for prompt generation and translation, which may introduce translationese or Western-centric phrasing [1]. While three human annotators refined the prompts, the dataset spans ten cultures, which makes me assume that only a subset of languages or cultures received native-level review.
- The construction of “rejected” responses using embedding similarity and the validation of human annotations with GPT-4 risk circular reasoning, as embedding distance and model agreement do not necessarily capture true cultural contrast or correctness.
- During generation of the dataset (Figure 14), it appears to assume a one-to-one correspondence between language and culture, which would overlook multilingual or multicultural contexts where a single language represents diverse cultural norms.
- Some cultural values and commonsense knowledge evolve over time, which potentially misrepresenting shifting cultural norms.
- Finally, the cultures being considered are limited to languages that are relative high/medium-resourced languages. However, I would say 10 cultures/languages are quite a lot already if the dataset is assessed by native people.

**W2.** Here I group all weaknesses related to Multilingual Alignment Performance evaluation:
- While I agree that m-reward-bench does not provide multicultural evaluation, I don't think correlation of performance between one benchmark and another is a good indicator of a well-correlated benchmark in terms of cultural knowledge, as there could be many confounding factors that represent such correlation. In addition, it only measures linear correlation, doesn't necessarily mean there is no correlation, furthermore in terms of causality that CARB represents multicultural benchmark. 

Suppose correlation were a good measure. Notice that include-base-44 itself consists of academic and professional exams, which also includes cultural-agnostic questions, AlpacaEval (which is the source of OMGEval) contain a lot of questions that are not culturally related (as shown in small number of localizations in the original paper), and BLEND's answers depend on the annotations being given (not representative of the culture of a country as a whole), so another reliance on GPT-4 as a judge wouldn't be great. If correlation were a good measure, this wouldn't remove the possibility that the performance could be correlated to the cultural-agnostic questions.

- Section 5 claims strong “multilingual” correlation between CARB and downstream alignment, yet Section 6.2 explicitly shows that reward models exhibit cross-lingual inconsistency by assigning divergent scores to the same semantic content across languages. Wouldn't this show that CARB’s predictive validity about assessing non-shallow cultural features to be wrong?

**W3.** While the result for the reward model framework looks promising on m-reward-bench and CARB, there is a lack of evidence that the reward model can be used to successfully post-train a policy model (RLHF or DPO). 

## References

[1] Yan, J., Yan, P., Chen, Y., Li, J., Zhu, X., & Zhang, Y. (2024). Gpt-4 vs. human translators: A comprehensive evaluation of translation quality across languages, domains, and expertise levels. arXiv preprint arXiv:2407.03658.

### Questions
Beyond my comments in weaknesses, here are my questions:

**Q1.** Related to **W1**, what kind of metric did you use to measure the inter-annotator agreement between GPT-4o since it's in percentage?

**Q2.** Most of the reward models used are LLM-as-a-judge(s) or reward model that is trained in English, how about specific multilingual models that (possibly) learn cultures during training (from recent related multilingual works), such as mR3 [1], Multilingual Nemotron [2], m-prometheus [3]?

**Q3.** After RLVF, seems like there is much more improvement in CARB compared to m-reward-bench, and I wonder if it's due to in-domain evaluation as the curated training dataset also involves CARB?

## References

[1] Anugraha, D., Hung, S. Y., Tang, Z., Lee, A. E. S., Wijaya, D. T., & Winata, G. I. (2025). mR3: Multilingual Rubric-Agnostic Reward Reasoning Models. arXiv preprint arXiv:2510.01146.

[2] Wang, Z., Zeng, J., Delalleau, O., Egert, D., Evans, E., Shin, H. C., ... & Kuchaiev, O. (2025). HelpSteer3: Human-Annotated Feedback and Edit Data to Empower Inference-Time Scaling in Open-Ended General-Domain Tasks. arXiv preprint arXiv:2503.04378.

[3] Pombal, J., Yoon, D., Fernandes, P., Wu, I., Kim, S., Rei, R., ... & Martins, A. F. (2025). M-Prometheus: A Suite of Open Multilingual LLM Judges. arXiv preprint arXiv:2504.04953.

### Soundness
2

### Presentation
2

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
This paper introduces a high-quality dataset called the Cultural Awareness Reward Modeling Benchmark (CARB). By benchmarking a series of reward models on CARB, the authors identify spurious correlations in culture-aware reward modeling, which shows that RM scores often depend on surface-level features rather than genuine cultural understanding. To address this problem, they propose Think-as-Locals, a method designed to elicit deeper, culturally grounded reasoning in generative reward models through reinforcement learning with verifiable rewards. This approach helps mitigate the influence of spurious features and advances culture-aware reward modeling.

### Strengths
1. The paper introduces a high-quality benchmark, Cultural Awareness Reward Modeling Benchmark (CARB), which covers 10 distinct cultures and typologically diverse languages across four domains. This benchmark is a valuable contribution for evaluating the cultural awareness of reward models.
2. The authors conduct comprehensive experiments using a range of models and datasets, enabling a thorough comparison of different models’ capacities for cultural awareness.
3. The paper proposes Think-as-Locals, a method that improves reward model performance by over 10% while reducing correlations with surface-level features.

### Weaknesses
1. The paper attempts to cover too many aspects within a single work, without providing sufficient description or analysis for each section. For instance, Table 2 compares the performance of many models, but the accompanying explanation is brief and lacks analysis of why certain models outperform others.
2. The analysis in Section 4 appears redundant and somewhat unrelated to the main narrative of the paper; it could be moved to the Appendix.
3. Think-as-Locals is the core method proposed in this paper, yet the authors do not provide enough justification or explanation for its design choices.

### Questions
1. Line 159 – What are the reference embeddings referring to?
2. What is the definition of Classes 3, 4, and 5 in Figure 2?
3. Table 3 is somewhat confusing. Why does it include comparisons with many models that were not trained using the proposed method, instead of focusing only on models trained by the authors?

### Soundness
4

### Presentation
3

### Contribution
4
