# Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Language model post-training has enhanced instruction-following and performance on many downstream tasks, but also comes with an often-overlooked cost on tasks with many possible valid answers. On many tasks such as creative writing, synthetic data generation, or steering to diverse preferences, models must cover an entire distribution of outputs, rather than a single correct answer. We characterize three desiderata for conditional distributional modeling: in-context steerability, valid output space coverage, and distributional alignment, and document across three model families how current post-training can reduce these properties. In particular, we disambiguate between two kinds of in-context learning: ICL for eliciting existing underlying knowledge or capabilities, and in-context steerability, where a model must use in-context information to override its priors and steer to a novel data generating distribution. To better evaluate and improve these desiderata, we introduce Spectrum Suite, a large-scale resource compiled from $>40$ data sources and spanning $>90$ tasks requiring models to steer to and match diverse distributions ranging from varied human preferences to numerical distributions and more. We find that while current post-training techniques elicit underlying capabilities and knowledge, they hurt models' ability to flexibly steer in-context. To mitigate these issues, we propose Spectrum Tuning, a post-training method using Spectrum Suite to improve steerability and distributional coverage. We find that Spectrum Tuning often improves over pretrained and typical instruction-tuned models, enhancing steerability, spanning more of the output space, and improving distributional alignment on held-out datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper investigates how the post-training of models (through instruction-tuning) negatively affects a specific set of desiderata, such as the ability of the in-context examples to steer the large language model towards a specific perspective, or diversity in the outputs for tasks with many valid responses. The authors prepare a dataset of tasks, called Spectrum Suite, to evaluate these desiderata and compare them between base and instruction-tuned models, finding that instruction-tuning breaks the steerability of LLMs for the generation tasks (while increasing performance for classification tasks or tasks with one/few specific valid answers). Building on this dataset, the authors introduce Spectrum Tuning, a post-training approach that provides the benefit of instruction-tuning withouth breaking the LLM desiderata (i.e., steerability and diversity of outputs).

### Strengths
The paper deals with important problem and I believe it can lead to many possibilities for intersting discussions and future work not only in post-training, but also in areas of interpretability.

The authors run an extensive set of experiments, providing a comprehensive evaluation on "failure" modes of instruction-tuned models, while proposing a solution for mitigating it. In addition, authors provide important resources (Spectrum Suite dataset).

### Weaknesses
**Paper structure/writing**

My biggest and only issue with the paper is how it is structured and the overall presentation. Overall, the paper is not really well put together -- although all the information is there, it is kind of "all over the place" and there are many explanations of motivation, connections between ideas, and details missing.

First of all, the abstract and introduction do not really introduce or motivate the problem and it is not really clear why it is important to deal with it -- only 2 sentences are dedicated to this (Liens 33-37) and afterwards the contributions and the description of the desiderata is introduced. I would suggest expanding the motivation in introduction and abstract as it would improve the paper quite a lot. In addition, I would suggest having a dedicated section for the Desiderata and providing more in-depth description or discussion on them. Similar problems are in the dataset and method description, which are quite short and deserve more detailed description and explaining the motivation behind them.

For the experiments section, the details for how the individual metrics are calculated are missing -- for example, there are lot of references to "yield" in Section 4, which is not explained and makes it more problematic to understand the results. Similar case is with "diversity" or "calibration" as it is not clear how these metrics were calculated. In addition, when performing Spectrum tuning and comparing with instruction-tuned models, it is not clear whether the evaluation is done on a completely different set of tasks or not.

Furthermore, almost all of the figures are quite small and full of information, making them hard to read and easily understand -- for example, in Figure 4 the legend (what the colours mean) is small and very well hidden.

(minor) 

Lines 89-90 "see 1 above" -- it is not really clear what this is referring to; is it the "exhibit natural person-to-person variation" from the list at the beginning of the section?

Appendix D seems to be empty, or at least does not refer to any additional figures



**I acknowledge that many of the misunderstandings and difficulties understanding the paper are a result of limited space and can be easily fixed during rebuttal period, and will update my score if addressed**

### Questions
See weaknesses for details, but mostly relate to details:

How are the metrics, such as "yield", "calibration", and "diversity", calculated?

Is the Spectrum tuning trained on one set of tasks and then evaluated (and compared with instruction-tuning) on a separate, unseen set of tasks?

### Soundness
3

### Presentation
1

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
The paper considers the scenario where questions in a task have no strict correct answers, and a model needs to give an answer that matches the distribution of a series of examples. The experiments show that instruction-tuned models perform worse than the pretrained models under this setting, and that a model specifically trained for this task performs better than pretrained models. They also find that the trained model achieves a better diversity-quality trade-off in out-of-domain questions that have multiple possible answers.

### Strengths
1. It is the first paper that considers the in-context steerability tasks
2. It provides insightful results about the degradation of IT models on this specific task
3. It provides a large-scale dataset for the task

### Weaknesses
1. Experiment setting seems over-complicated. It does not make sense to me to evaluate on the first few outputs, if the purpose is to evaluate in-context steerability.
2. Section 4 is missing an ablation study on temperature. It is claimed that the spectrum tuned models is Pareto optimum comparing to IT tuned models with temperature 1. But it would clearly make more sense if you use a couple of different temperature values for the IT models and see whether spectrum tuned model with a specific temperature is better than all of them, or plot a curve for both methods if the advantage is not that obvious.
3. Missing a baseline that instruction-tunes the model on the spectrum-suite training set. The paper only shows that a model, trained on the spectrum-suite training set with spectrum tuning, performs better on the spectrum-suite test set, comparing to models that are not trained on this dataset (PT and IT). This does not demonstrate the effectiveness of spectrum tuning.

### Questions
1. Are the instruction-tuned models taken from some existing checkpoints?
2. In the main experiment, how does the spectrum-tuned model perform under zero-shot settings? (i.e., we only evaluate on the first output)
3. What is the rationale in changing system/user/assistant to description/input/output?

### Soundness
2

### Presentation
3

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
This paper argues that standard instruction-tuning improves instruction following but harms three properties that matter when many outputs are valid: (i) in-context steerability, (ii) diversity and coverage of the valid output space, and (iii) distributional alignment. It introduces SPECTRUM SUITE (>40 sources, >90 tasks) formatted as description/input/output sequences, and SPECTRUM TUNING, a simple SFT variant that incoperates task description and ICL examples into the training process. Empirically, instruction-tuned (IT) models are strong on validity but collapse in diversity, while pretrained (PT) models are diverse but under-valid; Spectrum-tuned models raise yield and often give Pareto-style gains on diversity–validity and improve calibration and JS-divergence vs. PT on several held-out datasets, without degrading standard capabilities.

### Strengths
1. This paper tackles fundamental, challenging questions in the LLM post-training stage, most notably the loss of diversity after instruction tuning.

2. The experimental evaluation is comprehensive, incorporating human judgments in key sections; several empirical observations are novel and likely of broad interest.

3. The writing and presentation are clear, crisp, and well-structured.

4. Spectrum Tuning delivers notable gains over instruction tuning. While the source of improvement (training paradigm vs. dataset) is not fully disentangled, the work meaningfully advances the SFT stage.

### Weaknesses
**General problems**

1. Technical and implementation details are insufficient. Please specify training configurations, the datasets used, and key statistics. **Most importantly, did instruction-tuning and Spectrum Tuning use the same dataset, or datasets of comparable scale?** Otherwise, the gains could be due to a better dataset rather than Spectrum Tuning itself.

2. While the paper addresses important problems and paints a broad picture, many of the issues are largely orthogonal. A unified, principled analysis is missing to explain why these seemingly orthogonal issues can be resolved by a single approach.

**In-context learning**

1. ICL provides training-free adaptation to downstream tasks, often to enforce output format without SFT. If the model is already fine-tuned for the target task, is ICL still necessary? A fine-tuned model should already know the required format and task specifics. I recommend authors to look for more justifications (through existing studies perhaps) for this aspect.

2. The baseline results are puzzling: the instruction-tuned LLM consistently underperforms the raw pre-trained model. This contradicts prior studies and common community expectations.

**Diversity and Space Coverage**

3. The diversity concerns of conventional SFT are valid, but substantial prior work has addressed this area. The paper does not adequately cite, discuss, or compare with approaches such as rejection fine-tuning (RFT) [1] and entropic distribution matching [2], among others.

**Distributional alignment**

4. This section reports results without explaining why Spectrum Tuning improves distributional alignment. Is the effect due to broader data coverage, the addition of task descriptions and ICL examples, or something else? If the latter, why would the instruction-tuned model’s probability distribution be spikier than Spectrum Tuning’s?

[1] Scaling Relationship on Learning Mathematical Reasoning with Large Language Models

[2] Entropic distribution matching for supervised fine-tuning of LLMs: Less overfitting and better diversity, ICLR 2025.

### Questions
1. It is not very straight forward for me why adding description embedding before the instruction can improve the diversity of the generation results.

### Soundness
2

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
This paper discusses three important properties of language models for adaptable inference. (1) In-context steerability: the ability to adapt to new distributions given ICL examples. (2) Valid output space coverage: generating diverse yet valid responses, and (3) Distributional alignment: matching a target output distribution (Calibration). To test these three properties (especially in-context steerability), the author first constructs the datasets Spectrum Suite, which contains data that includes natural person-to-person variations that requires the model to adapt to certain distribution given ICL examples. Using this dataset, the author found that instruction tuning, while gives good ICL elicitation and with high valid response rate, will hurt the in-context steerability and output diversity. The author further proposed the Spectrum Tuning paradigm, which let the model learn via predicting each of the sequential ICL outputs to achieve better generalizbility. The proposed methods shows improvement on all of the three properties, showing potential of increasing inference adaptability of language models with this new training paradigm.

### Strengths
1. The problem definition of this paper is interesting. While most papers focus on direct comparison of accuracy, the authors proposed the three properties of output that largely impact the user experience in real-world, which is often lack discussed in the benchmark results.
2. The authors proposed the ICL steerability and elicitation with clear definition. The motivation of why ICL steerability is important for inference adaptation is also clear.
3. The Spectrum Tuning method is extremely simple yet effective following the experiment results. Showing improvement on all diversity, ICL steerability and calibration.

### Weaknesses
Overall I think the paper well demonstrated the effectiveness of the Spectrum Tuning method. However, there're certain points about the experiment that lack clarification.

1. Spectrum Suite is an important dataset for this paper, since the author use it to evalaute the three properties. However, how this dataset is constructed is not very clear, which can hurt the soundness of the experiment results. Specifically, at line 78 to 86, how the author identified those subjective tasks is unclear. This is important in the sense that the author uses these tasks to evaluate the ICL steerability.
2. While the reason how Spectrum tuning shows improvement on ICL steerability is trivial, how it help with better diversity and calibration is unclear to me. Can the author proposed some justification about this?
3. Following 2, while Spectrum Tuning models show better diversity compare to Instruction tuning in Figure 4, the validity is lower than instruction tuning. This indicates that Spectrum tuning might not overall be a superior tuning mechanism for the valid-diveristy rate, since it can just be a model that is underfit and has higher diversity compared to instruction tuning.

### Questions
1. Transition from line 53 to 54 is abrupt. Shouls add a few sentences to talk about the motivation of choosing to investigate these abilities instead of jumping right into it.

### Soundness
3

### Presentation
2

### Contribution
3
