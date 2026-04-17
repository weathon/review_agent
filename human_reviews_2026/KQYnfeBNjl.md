# SAE as a Crystal Ball: Interpretable Features Predict Cross-domain Transferability of LLMs without Training

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4

## Abstract
In recent years, pre-trained large language models have achieved remarkable success across diverse tasks. Besides the pivotal role of self-supervised pre-training, their effectiveness in downstream applications also depends critically on the post-training process, which adapts models to task-specific data and objectives. However, this process inevitably introduces model shifts that can influence performance in different domains, and how such shifts transfer remains poorly understood. To open up the black box, we propose the SAE-based Transferability Score (STS), a new metric that leverages sparse autoencoders (SAEs) to forecast post-training transferability. Taking supervised fine-tuning as an example, STS identifies shifted dimensions in SAE representations and calculates their correlations with downstream domains, enabling reliable estimation of transferability \textit{before} fine-tuning. Extensive experiments across multiple models and domains show that STS accurately predicts the transferability of supervised fine-tuning, achieving Pearson correlation coefficients above 0.7 with actual performance changes. Beyond this, we take an initial step toward extending STS to reinforcement learning. We believe that STS can serve as an {\color{black} interpretable} tool for guiding post-training strategies in LLMs. Code is available at \url{https://github.com/PKU-ML/STS}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper shows that SAE activations on specific domains (in ICL) can predict gains from SFT on the same domain, with high correlation.
It studies what portion of features that are activated during ICL also get bolstered by SFF, and it then does ablation studies to check their impact on model accuracy. It checks the portion of overlapping features between ICL and SFT, and it makes sure that results hold in RL (with a new formulation). The paper finally proposes an application to designing the data mixture used for SFT based on SAE feature activations).

### Strengths
The main strengths of this papers lie in the novel insight that SAE activations in ICL can predict improvements from SFT. The link between ICL and SFT was present in the literature, and the authors locate themselves in the literature well.
The paper is comprehensive, and it covers different modalities (RL/SFT) and applications (data mix for SFT). It correlates features with SFT gains well (0.7/0.8 correlation), and it shows the (very strong) effects of ablating features.
The paper also studies overlaps and uncovers significant portion of feature shifts (increase in activation under ICL) are the same as what SFT causes.

### Weaknesses
The main weaknesses are:
- no sample standard deviations for statistics you calculate
- no R2 for Figure 4's linear regressions, it would be interesting to know the fraction of the improvement from SFT you predict
- the SFT you do is can be ineffective (Figure 5), so predicting it is less interesting
- SAEs are lossy, and so is your predictive method (compounded lossyness)

I am open to increasing my score if the (first 2 or 3) weaknesses above are addressed

Minor:
- the contributions in the intro are too verbose

### Questions
Can you unpack the most important features in terms of neurons (that you could later fine tune selectively doing something like a selective LoRA)? Have you tried a sparsity penalty on SAE weights (to prune weights to save compute)?

Have you tried computing sample standard deviations and R2s?

How do you perform compared to trying to predict improvements form SFT based on model activations directly (using a probe)?

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
3

### Summary
This paper investigates the problem of the unpredictable transferability of post-training to downstream tasks. Post-training on data can introduce model shifts that improve the performance of some tasks while degrading others. To address this issue, the paper proposes a Sparse Autoencoder (SAE)-based Transferability Score (STS) to forecast post-training transferability. The method leverages supervised answers as demonstrations for in-context learning and identifies the SAE dimensions that exhibit the largest changes, which correlate with downstream task performance. Experiments across multiple models and domains demonstrate that the proposed transferability score accurately predicts the effects of both supervised fine-tuning and reinforcement learning (RL) tuning.

### Strengths
S1: The paper focuses on an interesting problem of forecasting changes in downstream task performance without additional training.

S2: The proposed fine-tuning-free approach for estimating transferability is both novel and useful.

S3: The investigation of SAE dimension shifts under both fine-tuning and in-context learning (ICL) is insightful.

S4: The paper evaluates the proposed STS method across multiple major open models (Qwen2.5-7B, Llama3-8B, Gemma2-9B) and demonstrates consistently high correlations.

### Weaknesses
W1: The study is limited to a single training dataset (LIMO) and a single evaluation benchmark (MMLU-Pro). Broader domains (e.g., dialogue, code generation) and larger model scales remain underexplored, even though these are areas where STS would likely be most valuable.

W2: There are potential reproducibility issues, as details on the SAE architectures, training procedures, hyperparameters, and prompt templates are either missing or insufficiently explained.

W3: Methodologically, STS relies heavily on the monosemanticity assumption of SAEs that each latent corresponds to a distinct human-interpretable concept. It also depends on access to high-quality demonstrations to estimate feature shifts (based on the weaker correlations observed in the RL experiments).

### Questions
Q1: Figure 2: What exactly do “raw model dimensions” refer to?

Q2: What does the ICL prompt template look like?

Q3: Line 243: The statement “the shifted features before SAE are more uniformly distributed” could be clarified. Does this mean less sparsity, or something else?

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
2

### Summary
This paper proposes to leverage (the change in) SAE activations to predict fine-tuning performance.

The premises are (well-studied either empirically/theoretically in prior work):

1. The sparse activations in SAE are domain/task-specific.
2. ICL can approximate taking a few gradient steps during the SFT process.

The setup is that, given a fine-tuning/source dataset (e.g., Math) and eval datasets (e.g., Engineering/Law), we want to predict the change in performance on the eval datasets from fine-tuning on the source dataset. The authors propose to use SAE to perform this prediction as follows:

- Identify the top-changed SAE activation dimensions before/after adding source-domain ICL examples in zero-shot prompting (ICL simulates fine-tuning).
- Identify how often those dimensions activate when feeding in the target domain examples, and use this as a measure of transferability (since SAE activation is associated with task relevance)

They found that the proposed measure

1. Correlates with the absolute change in performance (not the improvement) after fine-tuning.
2. Has the potential to be used in applications such as dataset mixture setting, e.g., to mitigate catastrophic forgetting due to fine-tuning.

### Strengths
1. The reviewer finds the topic interesting and timely, and the proposed technique could be useful in LLM training workflows.
2. The experiment design is generally sound, and the application demonstrated in section 5 is well-motivated.

### Weaknesses
The reviewer feels that this paper has the potential to have greater impact, but is limited by its current presentation and the scope of the experiments.

1. Details on the experiment setup/results are lacking.
- It appears that all experiments are performed once; ideally, it should be repeated over different random seeds (e.g., for initialization and train/test split) and report the mean + std.
- Since SAE should be interpretable, a qualitative analysis of the identified SAE activations would be nice as a sanity check to see whether the selected dimension semantically aligns with the task.
- Following the comment above, another direction is to annotate all SAE features and identify those related to the target task, and then see how much of the relevant features are "recalled" in the top-changed SAE activations.
- The paper would greatly benefit if more source training domains can be evaluated.

2. The fact only absolute change in performance is reported, rather than the signed improvement/decrease in performance, is rather disappointing. Would it be possible to also predict the sign of the change using the proposed approach?
- Furthermore, in the paper, it was unclear to the reviewer that "accuracy shift" meant absolute change rather than the signed improvement/decrease.

### Questions
See above.

### Soundness
2

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
This paper proposes a metric for predicting how supervised fine tuning will improve performance without performing the fine-tuning. The metric called STS trains a sparse autoencoder on hidden activations in the LLM to obtain sparse "monosemantic" latent features. The central assumption is that SAE dimensions that shift during in-context leraning also shift during fine-tuning. A correlation between these shifted features is the core of STS. Results on presented on a math dataset for several public models with ablations on SAE size, layer, etc.

### Strengths
* Addresses an interesting and timely problem - forecasting post-training transfer effects for LLMS
* Builds on recent interpretability work with SAEs
* Correlations in the experiments are consistent suggesting the metric captures a genuine phenomenon

### Weaknesses
* Limited conceptual novelty. Reframes existing ideas on representation drift and feature correlation under transferability and SAEs. Lacks a theoretical link between SAE features and fine tuning.
* Limited empirical evidence. While the experiments presented are indictive of some trend, the central hypothesis is only tested on one dataset and adaptation direction. The scope is too narrow to lend credible evidence to a correlation between ICL feature drift and SFT
* Lack of baselines. No comparison is made against more simple correlations such as cosine distance or linear probe similarity.
* Overstatement wrt principled prediction

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
