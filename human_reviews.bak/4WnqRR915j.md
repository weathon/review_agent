# Llemma: An Open Language Model for Mathematics

- Decision: Accept (poster)
- Scores: 8, 6, 6

## Abstract
We present Llemma, a large language model for mathematics. We continue pretraining Code Llama on the Proof-Pile-2, a mixture of scientific papers, web data containing mathematics, and mathematical code, yielding Llemma. On the MATH benchmark Llemma outperforms all known openly released models, as well as the unreleased Minerva model suite on an equi-parameter basis. Moreover, Llemma is capable of tool use and formal theorem proving without any finetuning. We openly release all artifacts, including 7 billion and 34 billion parameter models, the Proof-Pile-2, and code to replicate our experiments.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposed continual training with math/code data and achieved very competitive results on the related tasks. The evaluation is comprehensive and ablation studies are sound.

### Strengths
1. This paper showcases that if we continue pretraining an LLM on a specific domain, we are able to get good performance. The authors established a good way to do domain adaptation.
2. Data and training code are open-sourced, expect to have high reproducibility 
3. Ablation study is comprehensive.
4. Writing is well-organized

### Weaknesses
- The authors are comparing their results with `Minerva`, however since the datasets and its mixture, model architecture, training methods are different, we don't know which part contributes to the good performance. 
- We don't know if there training from scratch or starting from other LLM (like llama2 base) could be as impactful as well

### Questions
- One thing would be interesting to know is that if pretrained from scratch, will it be better or not? 
- Do we still need fine-tuning in this case

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this submission, the authors released a new large-scale dataset for training mathematically-specified
large language models. Additionally, based on the dataset, a new base model called LLemma is trained and released, which outperforms existing models like Code LLaMA and Minerva. Moreover, the dataset and the baseline are openly released, which helps to boost the research in AI4Math.

### Strengths
1. The paper is well-written and easy to follow. The details of data collection and training instructions are provided. Moreover, both data and model are released. 

2. The experimental part is solid. I especially like the ablation study of data mixture components given in Table 4, which helps to reveal the contribution of different data sources.

### Weaknesses
1. Technically, this submission is not so interesting, and I cannot learn anything new in the aspect of methodology. The claims and experimental results are natural — fine-tuning Code Llama on a larger mathematically specific dataset helps to improve its performance. Note that this is a very personal opinion:) It is OK if this work targets the topic of datasets and benchmarks.
2. I hope that the authors can discuss more about their future plans in the section of Conclusion. In the field of AI4Math, the performance of the current LLM is still limited. What will the authors do in the next step? Will they continuously enlarge the dataset and/or model? Is this the final solution to AI4Math?
3. The comparison between Code Llama and the proposed Llemma is not fair. It demonstrates the usefulness of the Proof-Pile-2 dataset. It would be nice if the authors could finetune Llama2 directly based on the Proof-Pile-2 dataset and show the model performance.

### Questions
Why did the authors train the 7B model for 200B tokens and the 34B (the larger) model for 50B (the fewer) tokens?

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper present a new model, namely LLEMMA, an open language model for mathematics. The model is used to solve math-related problems such as proving or calculators. The major contribution of the paper is opensourcing the model, as well as the training data and codes. The evaluation results show that the performance is encouraging.

### Strengths
1. The model and code are opensourcing.
2. It has proposed the Proof-Pile-2 data and will be released as well.
3. It has shown a training method that works well by fine-tuning a llama2 based pretrained model.

### Weaknesses
1. It doesn't have much ablation study or analysis about the data configuration. For example, why we should use which part of data.
2. It doesn't have much novelties, other than a model that experimentally looks ok.

### Questions
1. Have you experimented with different combination of training data?
2. How did you perform instruction fine-tuning?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
