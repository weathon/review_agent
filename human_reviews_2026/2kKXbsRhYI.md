# Long Chain-of-Thought Reasoning Across Languages

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
While large reasoning models have shown remarkable ability to generate long chains-of-thought (CoTs) in English, we still lack understanding of how these long-form reasoning abilities transfer to the vast majority of the world’s languages. In this work, we systematically investigate four key stages of model development–scaling, pretraining, post-training, and inference–to understand how long CoT capabilities extend beyond English. We compare two reasoning settings across nine non-English target languages: En-CoT, where models process target-language inputs, but reason in English; and Target-CoT, where models both process inputs and generate long CoTs in the target language. We find that scaling reasoning model size improves multilingual task performance in En-CoT, but Target-CoT performance lags behind. This gap widens for tasks requiring long, multi-step CoTs such as mathematical reasoning. Shifting to pretraining, we find that adding a specialized reasoning stage enhances En-CoT performance but degrades Target-CoT, whereas broad multilingual pretraining improves both modes simultaneously. Given the scarcity of high-quality reasoning traces in languages other than English, we explore synthetic data curation approaches for post-training. We demonstrate that fine-tuning on reasoning traces
automatically translated from gold English traces outperforms fine-tuning on target-language traces distilled from large reasoning models. Finally, we report disparities in inference efficiency between languages and uncover language-specific failure modes
in CoTs. We release models, datasets, and code to foster further research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper systematically investigates long CoT reasoning in large language models across nine non-English languages. It distinguishes three reasoning settings—En-Only, En-CoT, and Target-CoT—and examines how multilingual reasoning evolves through model scaling, pretraining, post-training, and inference. The study finds that scaling enhances cross-lingual comprehension but fails to improve target-language reasoning; specialized reasoning pretraining can even degrade non-English performance, whereas broad multilingual pretraining benefits both. Post-training on translated reasoning traces outperforms distillation in target languages, especially for mid- and low-resource cases. Error analysis reveals distinct failure modes between English and non-English reasoning, underscoring that effective multilingual reasoning requires targeted interventions beyond simple cross-lingual transfer.

### Strengths
1. The paper conducts an extensive assessment of long chain-of-thought reasoning across nine non-English languages spanning high-, mid-, and low-resource settings, providing a solid empirical foundation for future multilingual reasoning research.

2. The study disentangles the effects of scaling, pretraining, post-training, and inference, offering clear and interpretable insights into how multilingual reasoning capabilities emerge and evolve.

3. The results reveal several unexpected yet important trends—such as specialized reasoning pretraining degrading target-language reasoning, and translated reasoning traces outperforming distilled ones—offering actionable implications for improving multilingual LLMs.

### Weaknesses
1. While the empirical results are comprehensive, the paper provides limited theoretical insight into why Target-CoT performance remains low even as models scale. A deeper analysis—e.g., examining the role of tokenizer design, cross-lingual alignment, or representational interference—would strengthen the interpretation of the observed trends.

2. The experiments primarily focus on mathematical reasoning tasks, which may understate the linguistic challenges inherent to multilingual CoT. Including broader reasoning or commonsense understanding benchmarks would make the conclusions more general and impactful.

3. The amount of English reasoning data greatly exceeds that of the target languages, which could introduce residual bias despite the normalization analysis in Appendix B.4. More controlled data scaling or balanced sampling experiments would help validate the robustness of the conclusions.

### Questions
1. **Line 112–116:** Could the authors clarify whether the trends observed for long chain-of-thought (Long CoT) reasoning are consistent with those for shorter CoT reasoning across languages? In particular, do the same cross-lingual gaps persist for shorter reasoning chains?

2. **Line 155:** The paper mentions using *Gemini 2.0 Flash* for translation. How do the authors ensure translation quality and linguistic diversity, especially across mid- and low-resource languages? Are there quantitative or human evaluations to validate translation accuracy?

3. **Section 4:** The scaling experiments primarily rely on long reasoning models such as DeepSeek-R1. Have the authors observed similar multilingual patterns in other model families (e.g., open-weight LLaMA or Qwen models) that are not explicitly optimized for long CoT reasoning?

4. **Line 258:** For high-resource non-English languages such as Chinese, Target-CoT performance drops even after scaling. Could this be partly due to bias introduced by the 20k English SFT data used during fine-tuning?

5. **Suggestion:** Including results from LLaMA-based models in the main paper (rather than the appendix) would strengthen cross-model comparisons and make the findings more broadly convincing.

---

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
This paper presents a systematic study of long chain-of-thought (CoT) reasoning across nine non-English languages, examining how reasoning ability in large language models generalizes beyond English. The authors analyze four development stages—scaling, pretraining, post-training, and inference—and show that while model scaling and multilingual pretraining enhance comprehension of non-English inputs, reasoning within target languages lags far behind. Specialized reasoning pretraining (e.g., math-focused) improves English reasoning but often harms target-language reasoning. Fine-tuning with translated reasoning traces proves more effective than direct distillation, particularly for mid- and low-resource languages, and even small amounts of target-language data significantly improve performance. Error analysis reveals that English reasoning failures are mostly logical, whereas non-English reasoning is hindered by language-specific generation and conceptual errors.

### Strengths
- The paper tackles a novel and meaningful question—how long CoT reasoning transfers across languages—addressing a major gap in multilingual LLM research.

- Comprehensive evaluation across nine languages and multiple resource levels offers strong empirical grounding.

- The finding that translated synthetic data can substitute for large English datasets is practical and impactful for multilingual model training.

### Weaknesses
- The study relies solely on Qwen-family models (Qwen2.5, Qwen2.5-Math, Qwen3), limiting generalizability; results might be model-specific rather than universal.

- Although reports describe these Qwen models’ data composition, the exact pretraining and fine-tuning details remain opaque; using them as representative backbones may reduce methodological rigor.

### Questions
No.

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
This paper investigate the long-chain reasoning across languages, examining four stages of model development: scaling, pretraining, post-training, and inference. To achieve this, this paper compare three reasoning settings across nine non-English target languages: En-Only, En-Cot, and Target-Cot. The key findings are: (1) scaling reasoning model size improves multilingual task performance in En-CoT, but Target-CoT performance lags behind. (2) adding a specialized reasoning stage enhances En-CoT performance but degrades Target-CoT, whereas broad multilingual pretraining improves both modes simultaneously. (3) fine-tuning on reasoning traces automatically translated from gold English traces outperforms fine-tuning on target-language traces distilled from large reasoning models.

### Strengths
1. This paper first investigates the long cot in LLMs across languages.
2. The expriments offer  some insights for furture improvement.

### Weaknesses
1. The quality of the experiments remain to be improved.
2. Some experiment settings are strange.
3. The analyses could be more thorough.

All can refer to the questions below.

### Questions
1. In section 6, is it reasonable to translate English to low-resource language? How to ensure the translation quality?
2. Why conducting SFT in section 5? In section 5, the point is ``multi-lingual pre-training''.
3. In section 5, comparing Qwen-3 to Qwen-2.5 is not fair. Except the languages of pre-training data, the size, pre-training task, pre-training strategies are all different. Controlling variables for experiments is the optimal method.
4. In lines 181-182, I do not observe a ``consistently approach'' trand as model size scales. 7B LLMs with En-Cot are closest to corresponding En-Only baselines.
5. In section 6, comparing Translated-s1k and Distilled-s1k to OpenThoughts3-20k is unfair and unreasonable.
6. There are some issues of the format of Table 1.
7. This paper states many viewpoints in Introduction with only one citation. Although some of them are accepted by most researchers, citing their sources are very important.

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
The paper investigates whether the "long chain-of-thought" abilities of large reasoning models actually transfer beyond English. It disentangles two evaluation modes across nine non-English languages. The first is En-CoT, where inputs are in the target language but the reasoning chain is in English, and the other is Target-CoT, where both input and reasoning are in the target language. Scaling improves multilingual performance primarily in En-CoT, while Target-CoT is left behind, especially on math tasks that require long multi-step chains. A controlled comparison reveals that math-specialized pretraining enhances En-CoT yet often harms Target-CoT, whereas broad multilingual pretraining improves both modes. For post-training, modest target-language supervision built by translating gold English traces generally outperforms distilling target-language traces from a strong teacher model. The paper also analyzes inference efficiency and error profiles, finding language-specific failure modes when reasoning in the target language.

### Strengths
The paper is well-written and framed clearly with three setups and a comprehensive evaluation of nine languages, covering high/middle/low resource languages.

Also, the scaling study is carefully controlled and highlights that Target-CoT never reaches English-reasoning levels, even at 32B; switching to target-language reasoning at 32B still performs lower than a 7B English baseline.

Besides, the post-training section is practical. It shows that with only ~1k target-language traces, translated from high-quality English rationales, models substantially outperform target-language distillation in aggregate and become comparable with much larger English-only SFT. It especially benefits mid/low-resource languages. 

In addition, the analysis of inference efficiency shows that accuracy is negatively correlated with response length in tokens and that target-language SFT mitigates cross-lingual efficiency gaps. The byte-based view further investigates tokenizer effects. 

Finally, the error analysis is quite insightful. Most of the errors in En-CoT are inference flaws, while the errors in Target-CoT are more reflected in output generation and conceptual levels. It also provides a qualitative example of a situation where Target-CoT fails while En-CoT succeeds.

### Weaknesses
Although the authors benchmark translators and justified the usage of  Gemini-2.0-Flash in Appendix B.5, it may still be promising to further measure the quality of the translated datasets with existing translation quality estimation metrics (e.g., xCOMET, MetricX). These scores will directly show that the translated datasets are reliable and trustworthy.

The evaluated model only covers one language family, that is, the Qwen series. Although Deepseek-Distilled-R1 is trained mainly on English and Chinese data, it still shows capability in multilingual reasoning. So I may suggest testing at least one Deepseek model and seeing if a similar phenomenon also happens there.

For the improvement part, the work only adopts supervised fine-tuning but does not try reinforcement learning, which has already become a popular strategy nowadays. Specifically, it may teach structured reasoning in non-English languages via verifiable rewards.

### Questions
Based on the weaknesses, I may post the following questions and suggestions:

(1) Adopt existing translation quality estimation metrics, like xCOMET, MetricX, to directly measure the translation quality.

(2) Evaluate with at least one DeepSeek series reasoning model, say, DeepSeek-R1-Distill-Qwen-7B or other preferences.

(3) Adopting RL post-training to see if it benefits Target-CoT, like using GRPO with a format reward of 0.2 if the language of the thinking traces matches the (low-resource) question language.

### Soundness
3

### Presentation
4

### Contribution
3
