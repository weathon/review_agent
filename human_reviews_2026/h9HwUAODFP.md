# In-Context Learning of Temporal Point Processes with Foundation Inference Models

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 2, 6, 6

## Abstract
Modeling multi-type event sequences with marked temporal point processes (MTPPs) provides a principled framework for uncovering governing dynamical rules and predicting future events. Current neural approaches to MTPP inference typically require training separate, specialized models for each target system. We pursue a fundamentally different strategy: leveraging amortized inference and in-context learning, we pretrain a deep neural network to infer, *in-context*, the conditional intensity functions of event histories from a context consisting of sets of event sequences. Pretraining is performed on a large synthetic dataset of MTPPs sampled from a broad distribution over point processes. Once pretrained, our Foundation Inference Model for Point Processes (FIM-PP) can estimate MTPPs from real-world data without additional training, or be rapidly finetuned to specific target systems. Experiments show that FIM-PP matches the performance of specialized models on multi-event prediction across common benchmark datasets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
*disclaimer* i did not use llm at all for reviewing this paper. 

This paper proposes a foundational model approach for event prediction commonly modelled by marked TPPs.  The approach is modeling the conditional intensity given history context in the pretraining, specifically modeling the parameters of a modifed hawkes process intensity that inclusion excitation and inhibition effect.  Authors provides a zeroshot and fine tune version. The fine tune performs better on next event prediction on 4 real benchmarks. For multi event prediction task results are less prominent.

### Strengths
A few strengths include 1) a new foundational model perspective on modeling conditional intensity.  The paper can be improved by mentioning/citing papers with other angles for foundational model for marked TPPs, ( I am sure there are a few ) and highlight how theirs differ from others) 2). Presentation is very clear, easy to follow 3). I think the thoroughness of experiments is reasonable – may include a few other datasets from benchmarks such as EasyTPP.

### Weaknesses
A few weaknesses include 1). The expressiveness of the modeling of conditional intensity form (can you think of something more expressive? 2.) I think the fine tune version is much prominent in downstream tasks, so it kind of diminish the claim that the transfer learning approach setting one does not have to train on individual datasets. The zeroshot version is okay. For fine tuning the authors should think about computation costs as well compare to baselines. 3). The author should consider add Transformer based baseline for a fair comparision since the proposed approach is transformer based.

### Questions
1.	Have the authors look at other transfer learning based approach for TPP models? Whats the benefit of modeling conditional intensity  over directly predicting time and type in the transfer learning setting? 

2.	For multi event prediction,  I think the current approach is still autoregressive, the FIM-PP models are less convinced to me that they will be better than CDiff / Dual -TPP models – since the latter ones are for long horinzon prediction. It will be beneficial to provide some theoretical insight?

### Soundness
2

### Presentation
3

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
Until now, there have been no general-purpose foundation models for temporal point processes (TPPs). This paper introduces FIM-PP, the first foundation model for TPPs. FIM-PP can perform on real-world data as zero-shot model and can also be fine-tuned quickly. The paper's main contribution is this first attempt at a foundation model for TPPs. While many foundation models exist for vision and text, none existed for this task. Quantitatively, the model shows competitive performance in a zero-shot setting and outperforms the baselines when fine-tuned (Fig 3 and Table 1).

### Strengths
- As the authors note, this work is the first to propose a foundation model for temporal point processes (TPPs), thereby introducing a new and challenging task that sets an important direction for future research.
- The paper is well-organized and clearly written, making it easy for readers to follow the proposed methodology and understand the results.
- The foundation model demonstrates strong performance in estimating intensity functions - event time predictions (Table 1).

### Weaknesses
1. The claim of establishing a "foundation model" for Temporal Point Processes (TPPs) is not fully substantiated by the experimental evaluation. A core capability of any foundation model is its versatility across diverse data types and domains. The evaluation is limited to five datasets (Taxi, Taobao, StackOverflow, Amazon, Retweet), which primarily cover only two broad domains: Social/E-commerce and Transportation. To truly validate the model's generalized knowledge, the evaluation should be expanded to include critical TPP application domains such as Healthcare (MIMIC, EHR) and Finance (Financial Transactions), as noted in the TPP literature (e.g., [1], [2], [3]). The current limited domain coverage makes it difficult to definitively assess the model's claimed generalizability.

2. The model exhibits a noticeable weakness in event type prediction, which is a critical measure for TPP models. Specifically, the results on the Taxi dataset show a significant performance gap in event type prediction. The fine-tuned model's performance (e.g., 0.69) is notably lower than that of established baselines (e.g., 0.91). While strong time prediction performance is valuable, the poor type prediction suggests the model struggles to accurately capture the inter-dependencies between different event marks, which is essential for complex marked TPPs.

3. A key argument for using a foundation model is its rapid adaptability to new, specific tasks, as highlighted in the abstract ("rapidly fine-tuned to target systems"). The paper misses a quantitative analysis of this fine-tuning efficiency. Metrics such as the time or computational resources required for fine-tuning (e.g., convergence speed, epochs to reach peak performance) should be presented and compared against baselines to validate the model's advantage in rapid task adaptation.

[1] Shchur, Oleksandr, Marin Biloš, and Stephan Günnemann. "Intensity-free learning of temporal point processes." ICLR (2020).

[2] Mei, Hongyuan, and Jason M. Eisner. "The neural hawkes process: A neurally self-modulating multivariate point process." NIPS (2017).

[3] Liu, Zefang, and Yinzhu Quan. "Tpp-llm: Modeling temporal point processes by efficiently fine-tuning large language models." ICLR 2025 Workshop on Foundation Models in the Wild.

### Questions
1. Regarding the first weakness, the use of only five datasets may limit the ability to fully evaluate the foundation model’s generalization capability. Does the model perform well across diverse domains and dataset characteristics? If possible, it would be helpful to see the model’s performance evaluated on datasets with varying domains and dynamics.
2. Concerning the statement, “In contrast to some other neural methods, … we design our model to parametrize TPPs per mark conditioned on the joined history of all marks,” is this modeling approach original to your work?
3. With respect to fine-tuning the foundation model, how quickly can the reported performance be achieved?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes FIM-PP, a marked temporal point process foundation model that extends the foundational inference model (FIM) pretraining paradigm. Specifically, FIM-PP simulates synthetic data from several predefined parametric intensity functions and leverages an autoregressive attention-based encoder-decoder framework, enabling in-context learning on novel datasets to estimate the parameters of a Hawkes process with exponential decay. Experimental results over long time horizons demonstrate that the proposed approach achieves competitive zero-shot and fine-tuned event prediction performance on four datasets, with mixed results for next-event prediction compared to dataset-specific baselines on two datasets.

### Strengths
- The paper is well written and easy to follow.
- The adaptation of FIM to marked temporal point processes using an autoregressive attention-based encoder-decoder framework that enables in-context learning is both interesting and novel.
- Experimental results over long time horizons demonstrate that the proposed approach achieves competitive zero-shot and fine-tuned event prediction performance on four datasets.
- The paper provides both the model and data, enabling reproducibility.

### Weaknesses
- The performance on novel datasets appears to depend heavily on the quality and diversity of the generated synthetic data. A focused discussion on how the synthetic data is generated, namely, which parametric intensity functions are assumed would be valuable. Additionally, it would strengthen the submission to include an experimental or theoretical analysis quantifying uncertainty at inference time, particularly in relation to the similarity between inference dataset and the synthetic data used during training, similar to approaches in Gaussian processes.

- FIM-PP consistently estimates the parameters of a Hawkes process with an exponential decay kernel for any novel dataset. However, it is unclear how this approach would generalize to other triggering kernels, especially those with localized effects. The paper should also discuss and include benchmarks against methods such as [1,2], which assume more flexible triggering kernels for Hawkes processes.

- The proposed approach appears computationally intensive compared to dataset-specific methods, especially given the mixed performance gains observed in zero-shot tasks. The paper should provide a detailed discussion of the computational complexity of the proposed method, including the number of parameters, memory requirements, training and inference time, and how the model scales with the number of marks. It is also unclear why the number of marks is capped at 22, particularly since many clinical MTPP datasets, such as MIMIC, contain more than 22 marks.

- The results for next-event prediction, when compared to dataset-specific baselines on two datasets, are mixed and warrant further discussion.

- The proposed approach does not explore transferring the learned FIM-PP representations to alternative dataset-specific models for initialization, which could enable the incorporation of dataset-specific modeling assumptions. Exploring this aspect would strengthen the submission.

**References**
- [1] Isik Yamac et al., "Hawkes Process with Flexible Triggering Kernels", MLHC  2023.
- [2] Pan Zhimeng et al., "Self-adaptable point processes with nonparametric time decays", NeurIPS 2021.

### Questions
- Are all model parameters retrained during fine-tuning of FIM-PP?
- Could you clarify why FIM-PP achieves mixed next-event prediction results compared to dataset-specific baselines? Additionally, how does the model's performance vary as the prediction time window increases beyond N=1? What is the largest value of N for which the model can still predict with reasonable performance?
- Table 2: Could you provide complete results for all baselines across all datasets?
- Are the representations learned by FIM-PP transferable to alternative dataset-specific models for initialization, in order to support dataset-specific modeling assumptions?

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
3

### Summary
This paper introduces FIM-PP (Foundation Inference Model for Point Processes), a Transformer-based model for inferring conditional intensity functions of Marked Temporal Point Processes (MTPPs) through in-context learning. Instead of training a dedicated model for each event dataset, the authors pretrain FIM-PP on a large synthetic corpus of Hawkes processes, learning to infer conditional intensities directly from event histories and contextual sequences. Once pretrained, the model can be applied to new datasets, zero-shot or fine-tuned with minimal data.

### Strengths
1. The idea is interesting.

2. The paper pioneers the application of foundation inference models and in-context learning to temporal point processes—an original and timely contribution aligning with the broader movement toward general-purpose dynamical system learners.

3. The writing is good.

### Weaknesses
1. The pretraining prior is entirely Hawkes-based; real-world processes often deviate from this form (e.g., non-linear, periodic, or exogenous influences). The authors do not quantify or discuss this distributional gap.

2. While the concept of amortized inference for MTPPs is appealing, the paper provides little theoretical justification for why in-context learning generalizes to unseen, potentially non-Hawkes dynamics.

3. Transformer-based FIMs may struggle with long event sequences and large contexts. The paper lacks a complexity or runtime analysis.

### Questions
1. How does FIM-PP perform on non-Hawkes synthetic processes (e.g., power-law kernels or renewal processes)?

2. What is the sensitivity of zero-shot performance to the number of context sequences provided?

### Soundness
2

### Presentation
3

### Contribution
3
