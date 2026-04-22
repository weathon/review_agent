# Enhancing Trustworthiness of Fine-Tuned LLMs via Regularized Subset Selection

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Supervised fine-tuning (SFT) improves large language model (LLM) perplexity but can also degrade trustworthiness—leading to the generation of untruthful, biased, or unsafe content during user interactions. These issues are often traced back to specific phrases or patterns in the training data. However, correcting them usually requires expensive retraining or new data collection. In this work, we propose a two-stage, compute-efficient repair of the post-SFT models that enhances trustworthiness while preserving the downstream performance. In the first stage, we identify the training samples responsible for failures on trustworthiness metrics like truthfulness, stereotypical bias, and machine ethics—and select a small, diverse subset of these examples using a determinantal point process (DPP)-based regularization. In the second stage, we repair the model under the framework of proximal Bregman response function (PBRF) using a gradient ascent update, which enhances trustworthiness while preserving downstream task performance (perplexity). We evaluate our method on multiple LLMs of varying sizes and demonstrate up to 21\% improvement in trustworthiness metrics with minimal impact ($\leq1$ %) on perplexity. Our method provides a computationally efficient approach to enhance post-SFT models and offers a practical alternative to hours of retraining required for model repair

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a two-stage, compute-efficient method for repairing post-SFT (supervised fine-tuned) large language models (LLMs). The approach first identifies a representative subset of detrimental training samples and then enhances model trustworthiness through a gradient ascent update. Experiments demonstrate that the proposed method improves model trustworthiness with minimal negative impact on downstream performance, as measured by perplexity.

### Strengths
1. The paper is clearly written and well-organized, making the proposed method easy to follow.
2. Extensive experiments show that the approach effectively improves trustworthiness without substantially degrading the model’s general capabilities.
3. The use of gradient ascent for targeted unlearning provides a cost-efficient alternative to retraining-based methods.

### Weaknesses
1. According to the authors, the optimization procedure is adapted from [1], and the diverse subset selection is based on [2]. However, it remains unclear how these methods are specifically modified or integrated to address the LLM repair problem. Without clearer differentiation from prior work, the contribution risks appearing incremental.
2. The evaluation focuses primarily on metrics that are directly related to loss functions (e.g., log-odds and perplexity). To more convincingly demonstrate that general model capabilities are preserved, the authors could include standard benchmark metrics such as MMLU or GSM8K accuracy.
3. The proposed method assumes full access to the SFT dataset, which limits its applicability to real-world settings where proprietary fine-tuning data are unavailable or the LLM undergoes a further RL process. A discussion on how this method could be adapted or approximated in such cases would strengthen the paper's practical relevance.
4. The proposed method, *when used reversely, may intensify the ethics problems of LLMs*.

I'm willing to raise my score if the authors can further explain their contributions compared to previous works, e.g., how they modify existing methods to adapt to LLMs.

> **References**
>
> [1] If influence functions are the answer, then what is the question?.
>
> [2] Studying large language model generalization with influence functions.

### Questions
1. What would happen if the proposed repair process were applied iteratively or multiple times? Would the increases in perplexity accumulate to the point where retraining becomes necessary?
2. Could the authors provide more details on the computational cost of each stage, particularly the time spent on identifying detrimental points and selecting diverse subsets? As SFT datasets continue to grow, these steps might dominate the total computation time.
3. How feasible is the proposed method for models whose SFT data are not available? For open-source LLMs that have undergone extensive safety alignment, could the method still yield measurable improvements in trustworthiness?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a post-SFT repair method to improve LLM trustworthiness (truthfulness, ethics, bias) without retraining from scratch, while maintaining downstream performance. The approach identifies training examples that most degrade trust metrics and selects a small, diverse subset using influence-based scores and a DPP regularizer. Then, a proximal Bregman response function–based gradient-ascent update increases the loss on these samples while approximately preserving performance.

The method uses a differentiable log-odds trustworthiness surrogate and EK-FAC for scalable inverse-Hessian approximations. Experiments on Pythia and Qwen models show consistent improvements in trust metrics (up to ~20 percent) with minimal perplexity increase (<2 percent), outperforming simple gradient-ascent baselines and competing favorably with DPO.

### Strengths
- The paper provides a principled way to post-hoc improve a SFT model’s performance on trustworthiness metrics while maintaining performance on the downstream task. 
- The experiments are thorough with six models (Pythia & Qwen families), three trust aspects (truthfulness, stereotypical bias, machine ethics). Comparisons to other ‘repair schemes’ SGA/GA/GA+KL show strong performance in terms of improving trustworthiness with preserving perplexity.
- Well-designed ablations (e.g Section 4.6) to present the direct effects of different choices (e.g. DPP).

### Weaknesses
- The paper claims that both terms in Eq. (6) are submodular, enabling a greedy approximation, but it does not prove or cite the 
submodularity of the attribution term, log sum_j gamma^j .

- DPO is compared on perplexity but not on the same trust metrics; a fuller comparison (same compute budget) would clarify trade-offs.

- The evaluations aren’t based on model generations but on log probabilities on fixed data sets.  This weakness is acknowledged by the authors. Instead of only considering perplexity (which aligns with the loss used in the SFT phase), further downstream tasks could be considered for a more whole analysis.

### Questions
Could you please follow up on the submodularity result?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a new method for increasing perplexity while not degrading perplexity too much. The authors present it as a two-stage process: identify subset of training points that harm trustworthiness, then use proximal Bregman response function.

### Strengths
- Paper is well written and clear, tackling the problem of improving trustworthiness in large language models
- Paper contains empirical results from 6 models across 3 tasks.

### Weaknesses
- Authors don't consider other metrics like privacy and robustness (adversarial, OOD)
- No other trustworthiness methods as baselines in the main paper

### Questions
- Why were the specific trustworthiness metrics chosen?
- In Table 2, why does truthfulness decrease with a common subset?
- Could the authors provide runtimes for their method and also measure for existing methods?

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
3

### Summary
The paper proposes an efficient and lightweight two-stage repair method to enhance the trustworthiness of fine-tuned models without sacrificing downstream task performance. The method first identifies detrimental samples in the training data that cause a decline in trustworthiness using data attribution techniques. Subsequently, it performs gradient ascent on this subset within the Proximal Bregman Response Function (PBRF) framework to precisely "unlearn" the negative influence of these samples. The PBRF framework ensures that the model parameters do not deviate excessively from the original model, thereby preserving its core capabilities.

### Strengths
Important and Practical Problem: The issue of Supervised Fine-Tuning (SFT) undermining model trustworthiness is a common pain point in applying LLMs to real-world scenarios, especially in user-facing applications. The proposed lightweight repair method is more cost-effective than traditional retraining or RLHF, making it highly practical.
Solid and Novel Methodology: The work skillfully integrates advanced techniques from multiple fields. It combines data attribution (approximated by EK-FAC), diversity-based subset selection (DPP), and constrained model updating (gradient ascent under the PBRF framework) to form a logically coherent and technically sound solution.
Comprehensive and Convincing Experimental Design: The experiments cover models of different architectures (Pythia, Qwen) and sizes, and utilize standard trustworthiness benchmarks (e.g., TruthfulQA, DecodingTrust).

### Weaknesses
The experiments (Table 2) indicate that using a "common subset" to simultaneously improve all trustworthiness dimensions is less effective than targeted repair for each dimension individually. In practice, this implies that users may need to run the repair process separately for each dimension of concern (e.g., bias, safety), which increases operational complexity.
Strong Dependence on Paired Evaluation Data: The method's core relies on high-quality, paired trustworthiness evaluation data (proponent/opponent) to compute attribution signals and optimization objectives. In some vertical domains (e.g., healthcare, law, financial compliance), constructing such datasets is an expensive and time-consuming task. Furthermore, the definition and quantification of "untrustworthiness" are often contentious and context-dependent in open-ended tasks, which can lead to unstable metrics and insufficient coverage.

### Questions
The paper is presented with exceptional clarity and is very thorough, leaving me with no major questions. The authors have done a commendable job of anticipating potential queries and addressing them proactively through their detailed methodology and comprehensive experiments.

### Soundness
3

### Presentation
3

### Contribution
3
