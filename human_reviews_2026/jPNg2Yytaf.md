# Fit-LoRA: Fit Your LoRAs to Pruned LLMs Without Additional Training or Data

- Decision: Reject
- Scores: 4, 0, 6, 2

## Abstract
Personalization of LLMs via fine-tuning has become a popular way to enhance performance on downstream tasks. However, the model adaptation obtained after fine-tuning is specific to the base model. Any modifications made to the structure of the base model require users to fine-tune on the downstream task again. During deployment, a base model may be modified using pruning to obtain several LLM scales tailored to specific compute requirements. In this scenario, it becomes challenging to keep up with personalization, since each derived model must be individually fine-tuned. To address this challenge, we explore the possibility of leveraging the base model's fine-tuned knowledge to personalize any derived models. In this paper, we present Fit-LoRA, a framework that enables fine-tuning knowledge transfer between a base LLM and derived LLMs of smaller scales without needing any training or access to the original fine-tuning data. We validate our approach by conducting extensive experiments covering representative datasets such as BoolQ, SST-2, MRPC, RTE, and WinoGrande, across various model architectures including Llama-2, Llama-3.1, Mistral, and Gemma-2. Furthermore, we show the effectiveness of our approach by demonstrating its compatibility across multiple types of state-of-the-art LLM pruning methods, including depth pruning, structured pruning, and sparsification.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces FIT-LoRA, a training-free and data-free framework that enables the reuse of LoRA adapters on pruned or compressed versions of large language models (LLMs). It supports depth pruning, structural pruning, and sparsification, effectively addressing the incompatibility issues that arise when a fine-tuned LoRA adapter is applied to a model that has been modified through pruning.

### Strengths
1. The paper is the first to demonstrate the transferability of LoRA adapters between base and pruned large language models (LLMs)
2. The proposed method shows strong compatibility across multiple architectures, with experiments conducted on LLaMA, Gemma, and Mistral models, demonstrating its general applicability.
3. Extensive experiments on BoolQ, SST-2, MRPC, RTE, and WinoGrande show that FIT-LoRA nearly matches or sometimes exceeds task-specific LoRA fine-tuning. The method scales well up to 13B-parameter models, maintains cross-architecture consistency.

### Weaknesses
1. The author mentions the concept of “Train once, fit anywhere” in Figure 2, but after reading the paper, it remains unclear to me why such a scenario is necessary or what specific problem it aims to address.
2. The paper assumes a workflow where fine-tuning is performed before pruning. It would be helpful if the authors could elaborate on why this order is preferred over the more conventional approach of pruning first and then fine-tuning. Moreover, the experments that compressed models fine-tuned with LoRA consistently achieve higher performance (scores) deserves further analysis or justification.
3. The paper’s approach to depth pruning, structural pruning, and sparsification relies on straightforward, intuitive operations. While this simplicity is an advantage in practical implementation, the work would be strengthened by a deeper analysis or ablation study clarifying why such simple operations are sufficient for effective knowledge transfer.

### Questions
Please refer to the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper proposes a method which combines in a straightforward manner previous methods performing Low-Rank Adaptation and different types of pruning (structured, unstructured, layer). Comprehensive experiments are conducted, which however do not show a clearly consistent benefit of the combined approach.

### Strengths
- Comprehensive experimental evaluation on many tasks.

### Weaknesses
- Straightforward combination of existing approaches.
 - No clear benefit experimentally.

### Questions
- What is the distinctive contribution of the method, except a straightforward combination of prior ones?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
Authors propose Fit-LoRA, which demonstrate that LoRA adapters can transfer between backbones before and after pruning.

### Strengths
1. Paper is clearly written and generally easy to follow.
2. The method is simple and intuitive, and this paper conduct experiments to validate this method. Although the method itself may not be new/novel, this work constitutes a good-to-know report that community would find beneficial. 
3. Experiments are solid.

### Weaknesses
1. Authors did not consider quantization. If considering quantization as “pruning bits”, it indeed makes sense to add quantization experiment in. For example, applying LoRA trained on Qwen3-4B (BF16) to Qwen3-4B-FP8.
2. Regarding framing, I don’t consider this work as “proposing a novel approach”, although it’s framed this way. I find the work’s value in exploring LoRA’s transferability across scales/pruning. I suggest authors frame their work as more of a valuable empirical observation like [1].
[1] LoRA Without Regret. Schulman et al. 2025

### Questions
[1] explores cross-scale weight parameter transfer for vision transformer pretraining. They proposed weight selection and also discuss the case where dimension and depth mismatch. I encourage authors to discuss the connection of their paper with this paper. For example, could authors transfer LoRA from LLaMA-3.1-8B to LLaMA-3.2-1B?

[2] discusses the transferability of finetuning task vector between different checkpoints of an LLM. I find the reason behind the working mechanism of Fit-LoRA is also related.

[1] Initializing Models with Larger Ones. Xu et al. ICLR 2024
[2] Param  for Direct Weight Mixing: Post-Train Large Language Model at Zero Cost. Cao et al. ICLR 2025

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper propose a training-free framework, Fit-LoRA, that aims to modify adapters (specifically LoRA) when we compress the base models with a finetuned LoRA. The paper mainly considers depth pruning, structured pruning, and sparsification for the base model compression part, and Fit-LoRA is effectively a direct selection/match over depth, channel, and no change for sparsification. The experiment is conducted on

### Strengths
- The paper is easy to follow and the overall idea of Fit-LoRA is understandable upon first read. 
- Fit-LoRA are tested with multiple modern LLMs and compression algorithms such as ReplaceME and MaskLLM.

### Weaknesses
- Although training-free, the approach of Fit-LoRA is too simple if not trivial. In this case, I won't consider Fit-LoRA have a major methodology innovation as the motivation / explanation behind Fit-LoRA on line 236-240 is also insufficient and ungrounded. For example, we could have a structured compression algorithm that is not layerwise independent (for example, each layer corrects the error of a previous pruned layer), and in this case, each layer output is not necessarily aligned with prior base model. The compression algorithm can still perform well, but not necessarily Fit-LoRA. **The effectiveness of Fit-LoRA is implicitly dependent on the compression algorithm**. 

- **The requirement of being training-free is much more restrictive than the requirement of cannot finetune with proprietary data, and such requirement is unrealistic.** We often still have online update or cheap calibration with auxiliary public data and if the performance degradation from compression is concerning, a cheap calibration to recover the performance loss is better than do-nothing. 

- The benchmarks SST2, BoolQ, RTE, MRPC, WinoGrande are all too *easy* for modern LLM. For example, Llama3 can have zeroshot acc of SST2 as 90% and a finetuning acc of 95% is of marginal improvement. **The commonsense (Arc-C, PIQA, OBQA, etc.), arithmetic and math reasoning (GSM8K or similar) should be definitely added.**
  - This might have *major* influence on the benchmark performance of Fit-LoRA. **It is quite likely that the simplicitiy of Fit-LoRA can work with easy benchmarks but will suffer from serious degradation for harder ones.**

These 3 weaknesses are critical for the validity and generalizability of Fit-LoRA so I will vote for reject.

### Questions
- I am not confident that there are absolutely *no* prior works on modifying adapter after base model compression. Even this holds, there are still prior works on *transferring* adapters from 1 task to other task with training. A discussion of such should be added to the related works, or if appropriate, as a baseline to the experiment.

### Soundness
2

### Presentation
3

### Contribution
2
