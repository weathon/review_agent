# FTFT: efficient and robust Fine-Tuning by transFerring Training dynamics

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 5, 3

## Abstract
Despite the massive success of fine-tuning large Pre-trained Language Models (PLMs) on a wide range of Natural Language Processing (NLP) tasks, they remain susceptible to out-of-distribution (OOD) and adversarial inputs. Data map (DM) is a simple yet effective dual-model approach that enhances the robustness of fine-tuned PLMs, which involves fine-tuning a model on the original training set (i.e. reference model), selecting a specified fraction of important training examples according to the training dynamics of the reference model, and fine-tuning the same model on these selected examples (i.e. main model). However, it suffers from the drawback of requiring fine-tuning the same model twice, which is computationally expensive for large models. In this paper, we first show that 1) training dynamics are highly transferable across different model sizes and different pre-training methods, and that 2) main models fine-tuned using DM learn faster than when using conventional Empirical Risk Minimization (ERM). Building on these observations, we propose a novel fine-tuning approach based on the DM method: Fine-Tuning by transFerring Training dynamics (FTFT). Compared with DM, FTFT uses more efficient reference models and then fine-tunes more capable main models for fewer steps. Our experiments show that FTFT achieves better generalization robustness than ERM while spending less than half of the training cost.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper demonstrates that (1) smaller models can also be used as reference models in data map (DM) methods as described in Swayamdipta et al. 2020 and (2) transfer is possible between models with different pre-training methods on NLI and hate speech detection benchmarks. They also conduct ablation studies on how fast models learn with ERM vs DM. I cannot recommend this paper for acceptance as the novelty is very limited over Swayamdipta et al. 2020.

### Strengths
Although the novelty is limited, ablations and benchmarking shown are quite detailed. Presentation is very clear. Related literature is very well-reviewed.

### Weaknesses
Contribution is extremely limited over Swayamdipta et al. (2020) who proposed the original method of data maps combined with the work of Sar-Shalom & Schwartz (2023) who demonstrated that a DM constructed by ELECTRALarge can be used to improve the robustness of DeBERTaV3Large. This work reads more like a tech report rather than an ICLR paper. Insights are practically useful, but does not go beyond systematic benchmarking.

### Questions
I do not have any questions -- presentation is clear.

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose a method for finetuning pretrained language models, FTFT, that finetunes a smaller reference model which is then used to select examples for training the target model on a downstream task. The authors demonstrate that smaller models can be used for constructing a DataMap of samples without significant reductions in performance.

### Strengths
1. The authors conduct a systematic investigation of various reference model sizes and compare against reference models trained with an alternative discriminative pretraining method.
2. The authors demonstrate that using smaller reference models and training on the resulting DataMap does not result in performance reduction as compared with an ERM baseline trained over the entire dataset; and results in improved performance on OOD robustness datasets.

### Weaknesses
1. One of the primary contributions is the sample efficiency of models trained on a smaller data map selected via ambiguity.  However, there is limited comparison or discussion of related work on sample efficient methods of training such as curriculum learning and dataset pruning [1, 2] .
2. The DataMap selection criteria is limited to example ambiguity -- and does not compare against other criteria such as "hard-to-learn", example forgetability [3] 
3. Evaluations are limited to finetuning of models for language classification -- unclear whether results would generalize to other domains or task settings (e.g. image classification, language generation).

References:
1. Sorscher, Ben, et al. "Beyond neural scaling laws: beating power law scaling via data pruning." Advances in Neural Information Processing Systems 35 (2022): 19523-19536.
2. Paul, Mansheej, Surya Ganguli, and Gintare Karolina Dziugaite. "Deep learning on a data diet: Finding important examples early in training." Advances in Neural Information Processing Systems 34 (2021): 20596-20607.
3. An empirical study of example forgetting during deep neural network learning. In ICLR, 2019.

### Questions
* Significance of performance gains over baseline DataMap are unclear without variance across random seeds in Tables {1, 2, 4}?
* Table 4; How is the cost of one "ELECTRA-Small with ERM" calculated (i.e. FLOPs, GPU-Hours, power consumption?) Does this account for the cost of finetuning the reference model and scoring the samples DataMap?
* Why is 33% chosen for the top q% to create the Data Map? What is the distribution of the ambiguous and hard to learn examples? Do the values have a large degree of skewness?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes FTFT, an efficient fine-tuning algorithm that selects a core set of examples to fine-tune a large model by using the training dynamics of a small reference model. The authors observe that such an algorithm can achieve better OOD performance with a slight drop in ID performance when compared to the conventional ERM algorithm. The authors conduct extensive experiments to find the right reference model to select the core set, where the selection can be made based on model size and family. Finally, the authors show the efficiency gains of their method compared to ERM by comparing the behavior of the model's OOD performance over training time.

### Strengths
The strength of the paper lies in its easy-to-understand explanation of the algorithm. The authors begin with a clear description of the existing literature on the data map methods and the underlying issue of these methods. With a proposed hypothesis of using a small model to provide the necessary data map, the authors test multiple candidates that can act as the reference small model. Finally, extensive experimentation shows the efficacy of their method on multiple ID-OOD dataset pairs.

### Weaknesses
I have a few questions regarding the experimental setup.

(a) How efficient is FTFT compared to ERM in terms of total flops? Since FTFT first trains a small reference model to select the ambiguous set of examples, it has to incur the flop necessities of training the small reference model. A rough estimate of the flop counts for both methods will be useful.

(b)  How does FTFT perform when compared to existing algorithms that aim to improve the OOD performance of trained models? Examples of such methods include invariant risk minimization algorithms [4], DRO [5], and WiSE-FT [6]. Comparison to a couple of them will strengthen the results of the FTFT method. 

(c) How sensitive is FTFT to training hyperparameters of the small reference model and the target model? Does the ambiguous core set selected using the small reference model change with its training hyperparameters?

(d) I observed that the core set selected with base models (ELECTRA-base and DeBERTaV3-base) performs better OOD than training with a core set selection from large models. Can the authors provide more insights into the behavior?


There are a number of papers that I believe should be part of the related works section to give readers a full overview of the literature. 
For example, [1] dynamically weighs training domains in the pre-training dataset of a large language model, using the training dynamics of a small language model. Other citations may include works that train a proxy model to select the right set of data to train the target model. [2, 3]


1: DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining. Xie et al.' 23

2: Selection via proxy: Efficient data selection for deep learning. Coleman et al.' 19

3: SVP-CF: Selection via Proxy for Collaborative Filtering Data. Sachdeva et al.'21

4: Invariant Language Modeling. Peyrard et al.'21

5:  Distributionally robust language modeling. Oren et al.'19 

6: Robust fine-tuning of zero-shot models. Wortsman et al'21

### Questions
Please see my questions in the previous section.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
To reduce the training cost of Data map. This paper develops a variant of Data map by swapping in a smaller model for data selection. The authors experimented on DEBERTA, ELECTRA and TinyBERT with a few datasets, showing training cost improvement and mixed results in performance.

### Strengths
1. Simple and clear idea.
2. The motivation and reasoning are well explained.

### Weaknesses
1. The proposed method has limited novelty compared to Data map.
2. The result is mixed. Out of the experiments in Table 2, only half of them show successful transfer. Suggesting the scale of the reference model still needs to be relatively close to the Main model. Besides, even in the remaining rows, the result is inconsistent across datasets.
3. Cost saving is only in fine-tuning time rather than inference time. However, for LLMs, fine-tuning cost is much less of an issue than pertaining or inference cost.

### Questions
N/A

### Soundness
3 good

### Presentation
2 fair

### Contribution
1 poor
