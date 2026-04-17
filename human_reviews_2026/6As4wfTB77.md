# Learning to Interpret Weight Differences in Language Models

- Decision: Accept (Poster)
- Scores: 8, 6, 4

## Abstract
Finetuning (pretrained) language models is a standard approach for updating their internal parametric knowledge and specializing them to new tasks and domains. However, the corresponding model weight changes ("weight diffs") are not generally interpretable. While inspecting the finetuning dataset can give a sense of how the model might have changed, these datasets are often not publicly available or are too large to work with directly. Towards the goal of broadly understanding model weight changes in natural language, we introduce Diff Interpretation Tuning (DIT), a method that trains models to describe their own finetuning-induced modifications. Our approach uses synthetic, labeled weight diffs to train an introspection adapter, which can be applied to a compatible finetuned model to make it self-describe the weight changes. We demonstrate in two proof-of-concept settings (reporting hidden behaviors and summarizing finetuned knowledge) that our method enables models to describe their finetuning-induced modifications using concise and accurate natural language descriptions.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose a novel preliminary method to investigate the behavioral changes of language models after finetuning. To do so, the authors propose Diff Interpretation Tuning (DIT), a method that utilizes a trained adapter to be applied to a finetuned model to elicit its behavior changes stemming from finetuning. The authors compare DIT to prompt-based eliciting techniques on two synthetic case studies. Moreover, the authors discuss the applicability of the idea to generalize to real problems. 
The contributions are: the conceptual idea and first prototype of the DIT framework; showing that DIT outperforms simple prompt-based techniques on synthetic case studies; and an honest discussion about its limitations with respect to generalizability and future work.

### Strengths
The strengths of the paper are: 

* The paper presents a novel and original idea towards making fine-tuned language models more interpretable.
* The paper is mostly clearly written, is easy to follow, and well-structured. 
* The paper shows evidence for the first prototype of the idea to argue for its potential.

### Weaknesses
The weaknesses of the paper are:

* Given its explicit preliminary nature, the paper still lacks more concrete evidence for and a discussion of the applicability of DIT on real-world use cases. In particular, it would be helpful to discuss the limitations that might arise when scaling to apply the idea in practice, such as whether we would have access to various finetuned models and their trained-for behavior for applications of interest. 
* From the paper and its evaluation studies, it is unclear to what extent more sophisticated prompt-based baselines could match the performance of DIT. What does the performance look like if you write a more elaborate prompt that includes the base question? (or if it is hard to come up with a prompt, what if one uses prompt tuning).

### Questions
* The general formulation of DIT and Problem 2.1 does not seem to require an adapter. However, the realization seems to require an adapter-based approach. Are there alternative ideas or approaches that would work without an adapter? How could we apply/encode non-adapter weight diffs?
* In Equation 3,  why is applying the weight diff commutative?

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
5

### Summary
The authors propose a new LoRA “introspection adapter” that, when applied to a finetuned model, prompts the model to describe behavioral changes introduced by its own weight diff; they show that on two proof-of-concept settings this is highly successful at recovering the training objective.

### Strengths
- Very exciting simple new method to interpret weight differences.
- Well and clearly written paper. There were several concerns that I initially had during reading, which were addressed later on – this made it a very pleasant and engaging reading experience.

### Weaknesses
**Major:**

- **W1.** Your problem statement is very close to the problem of “model diffing” (i.e. trying to understand the difference between two models). You should refer to that literature and cite some of those works, e.g.:
    - https://transformer-circuits.pub/2024/crosscoders/index.html
    - https://transformer-circuits.pub/2024/model-diffing/index.html
    - https://arxiv.org/abs/2504.02922
    - https://arxiv.org/abs/2311.12786
    - https://arxiv.org/abs/2402.14811
    - https://arxiv.org/abs/2501.03012
    - https://aclanthology.org/2020.aacl-main.11/
    - https://arxiv.org/abs/2504.02904
    - https://arxiv.org/abs/1908.08593
    - and many more.
- **W2.** Your investigation is very narrow and appears to break down in OOD settings, which is generally a big weakness of your paper – as you correctly note in the limitations / 6.1. However, the introduction currently slightly oversells the generality. While you do mention “proof-of-concept”, I think you should acknowledge these limitations more explicitly early on. Providing less toy-ish experiments would significantly improve the paper and impact.
- **W3.** LLM judges are notoriously noisy. I would like to see some experiments on the stability of the reported similarity metrics – e.g. how much variance do we get from rerunning the judge model? A slightly more stable comparison might be to use semantic embeddings to compare the predictions.
- **W4.** In general, your baselines seem relatively weak. An agent-based baseline, where the agent can iteratively refine its questions, would be a fairer and more competitive comparison.
- **W5.** The mechanistic analysis does not really tell us much beyond the observation that the finetuning signal seems to live more in later layers. There is already some literature showing this (see w10 below). Do you have any other clear takeaways from that analysis? It is a bit unclear what a reader should take away from that section?

**Minor:**

- **w7.** Baseline “Base Question”: do you only sample a single token here? If no: sampling with temperature 0.0 is clearly suboptimal. Greedy decoding will lead to suboptimal generation and might therefore unfairly impair this baseline. I would suggest to resample *k* times and report point estimates (or, even better, use it to report confidence).
- **w8.** You never evaluate whether your models actually perform their learnt task well. This seems like a crucial detail that is missing (although it is somewhat expected, since the interpreter model can figure it out).
- **w9.** I feel like you should cite https://arxiv.org/pdf/2509.13316 as verbaliser-related work as well.
- **w10.** Chapter 6.3: there is a range of literature that shows that higher-level semantics mostly appear in middle to later layers (e.g. https://arxiv.org/abs/2502.02013v2, https://arxiv.org/pdf/2502.16570v2). Further, there are concrete works showing that finetuning mostly affects later layers, which could be referred to here:
    - https://arxiv.org/abs/2004.14448
    - https://publikationen.sulb.uni-saarland.de/handle/20.500.11880/37254
    - https://arxiv.org/abs/2109.08406
    - https://openreview.net/forum?id=YWbEDZh5ga
    - https://arxiv.org/abs/2305.17446
- **w11.** How your data looks is a bit unclear until the reader digs through the appendix. I would suggest adding a short sentence in the main text about how exactly you generate it (for the hidden-topic models) and linking to the appendix more clearly (e.g. chapter 3.1 “generating training data…” should point to the detailed description).
- **w12.** Missing methodological details: which modules do you finetune when applying LoRA?

**Expectation Management:** I would like to see this paper accepted. If you can address all of my major points, either fixing them or providing a convincing response, I will increase my score to 8. I won't lose sleep over the minor points, but addressing them would improve the paper overall so I'd suggest to still address them. However, I don't think a higher score of 10 is appropriate due to the lack of generalisation and the proof of concept setting.

### Questions
- **Q1.** Do you have any intuitions for why your method collapses on Qwen3 4B full training but not on Gemma 2? I would also have been interested in the generalisation the other way around – since this is probably what matters more: do interpreter LoRAs on full finetunes generalise to LoRAs?
- **Q2.** I would be really curious whether this also works for methods like Subliminal Learning, where the training data does not naturally encode the semantics. Based on the paper below, I would expect it to work as well, but I would be interested to hear your thoughts. If it works reliably, your method might be very interesting for finetuning providers to detect potentially unwanted behaviour that cannot be detected from data analysis alone.
- **Q3.** Did you evaluate the thematic overlap of test and training set?
- **Q4.** Regarding not being able to detect the trigger: I generally agree with your explanation. Another thing I suspect is that the trigger is only ever present in the user message and hence not something that the model is directly optimised on (gradients still flow to those tokens, but the user message itself is masked). This would connect to your point that the model only implements the *check* and does not really implement the *trigger identity* itself. I would be curious to see what happens if you do **not** mask the user message and whether this allows you to recover the trigger message – this would also give us important insights into what effects masking certain tokens during finetuning has.
- **Q5.** Any specific reason why your SEP trigger only has 3 digits but the prompts contain 6 digits? Is it just to be able to have multiple “triggers” for a single question?
- **Q6.** [This point should not be considered for the review]: in a recent paper (https://arxiv.org/abs/2510.13900) – that came out after the ICLR deadline – the authors propose that finetuning in the style that you used leaves very clear traces in the activation differences that can be read with basic methods. This raises the question of whether your interpreter model mainly learnt to apply a more sophisticated “logit lens” to this signal, and it slightly updates my prior downward on whether this will generalise to more complex finetunes. They also propose that mixing in data somewhat removes the signal, so I am curious what you think about this, and whether your approach still works when mixing in unrelated data. I think addressing this would make your paper stronger if you could show it (but I'm not considering it for the rating).

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
The authors propose to train a LORA adaptor on multiple fine-tuned versions of the same base model, where the adaptor is trained to output the task the fine-tuned model was trained on. Once the adaptor has seen enough fine-tuned versions of the base model, the hope is that it can generalize well to other fine-tuned versions of the same base model. Evidence showing this is interesting because it can demonstrate the possibility of interpreting the weight difference (weight difference between the fine-tuned and pre-trained model). The authors show positive evidence to this using LLMs upto 7B. This can have important implications to the safety and privacy communities as if the pre-trained model is public, having the knowledge of the pre-trained version of a private model can help the attacker train multiple fine-tuned versions and later analyze the semantics of the fine-tuning dataset used to fine-tune the privately fine-tuned model, given that the private model is open sourced (which although might be unlikely).

### Strengths
* Although, I believe that this paper is more of a proof of concept, it can provide a good starting point to future work trying to investigate the extent of knowledge possible to extract from the weight differences alone. If it is possible to extract a lot of information about the fine-tuning dataset, then it could mean that open-sourcing the fine-tuned models without open sourcing the dataset, could act as a good proxy to understand the semantics of the fine-tuning dataset.
* The setup used by the authors seems quite extensively framed i.e. the headlines of the news articles and the topics used for fine-tuning seem non-trivial. I also liked the idea of using trigger tokens, as it provides a good proxy of the maximum performance one can achieve via prompting. Framing a comprehensive evaluation framework is very important in this type of work.

### Weaknesses
* The main limitation is that the proposed method is quite restricted to the same base model and several of its fine-tuned versions. This means, that each time someone wants to understand the semantics in weight diffs, they need to train several of the fine-tuned versions of the pre-trained which might be computationally intractable for very large models. Further, if we increase the task complexity of the adapter, intuitively more samples of the fine-tuned versions would be needed (as is the case with fine-tuning for news summarization vs hidden topics) . Therefore, there seems to be an inherent limitation with the existing approach.
* The generalization ability of the adaptors is also not completely clear. We might need to train an extensively large number of adaptors in order to hope to achieve some generalization behavior which is very important.. It would be interesting to understand how generalization to different domains of questions scale with the number of fine-tuned models.
* There is a degradation in performance in Fig. 4.3 on training adaptors on full rank fine-tuned versions of the base model. This is intuitive as now the information about the fine-tuning dataset would be spread on a larger number of parameters in the fine-tuned model, making the task to detect the semantics in the fine-tuning dataset more difficult for the adaptors. Is there some way to increase the expressivity of the adaptors (i.e. the rank of LORA adaptors from 16) can this help in improving the performance on full rank fine-tuning.

### Questions
* It would be great if the authors could share ablations on the effect of changing the rank of the lora adapters trained on the fine-tuned models. Is there any reason to set the rank of the adapter as 16?
* How do the methods employed by authors compare with data attribution methods? One can also use data attribution methods like influence functions to get signals on the hidden topics or headlines of news articles. It would be very helpful if the authors can compare their method with some of the existing data attribution methods. A naive baseline could be to analyze the influence of multiple topics on the model’s performance for the pre-trained as well as fine-tuned model and then define a metric as the difference between the individual influences. The samples having significantly larger influence on the fine-tuned model as compared to the pre-trained one could potentially give insights on the fine-tuning dataset. Collecting multiple such samples and then using an LLM as a judge to label the common semantics between the could help identify the news article headlines or hidden topics.

### Soundness
3

### Presentation
3

### Contribution
3
