# DADM: Hallucination Detection in LLMs via Distance-Aware Distribution Modeling

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Despite the remarkable advancements, large language models (LLMs) still frequently generate outputs that contain factually incorrect or contextually irrelevant information, commonly known as hallucinations. Detecting these hallucinations accurately and efficiently remains an open challenge, especially without relying on labeled datasets. Current  methods primarily depend on internal activation or consistency of multiple responses for one prompt, limiting their effectiveness in capturing global semantic and distributional structures of truthful outputs. Besides, methods that estimate latent subspaces directly from mixed-quality data, suffer from noise contamination and imprecise geometric representations. To address these limitations, we propose a novel Distance-Aware Distribution Modeling (DADM) framework that operates in two stages: first, we apply an iterative distance-based process to select consistently truthful samples; second, we model the global distribution using normalizing flows, enabling accurate likelihood estimation by maximizing the likelihood of truthful samples and minimizing the likelihood of hallucinated samples. This two-stage design ensures both robust sample purification and expressive modeling of truthful generations, leading to interpretable confidence scores and more reliable hallucination detection. Extensive experiments on benchmark datasets demonstrate that our method consistently outperforms prior unsupervised approaches across multiple LLM settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposed a new approach to detect hallucination with unlabeled data. The approach consists of two stages: high-confidence sample selection and flow model training. The authors first iteratively estimated the center of a set of representations and selected top-m highly confident truthful samples based on a distance metric. Then, the authors trained a flow model to predict whether a representation of a generation is truthful or not. The authors conducted experiments on four datasets with two LLMs, showing that their proposed approach outperforms other baselines. The authors also conducted ablation studies to justify their design choices.

### Strengths
1. The writing is clear and easy to follow.
2. The proposed method does not require labeled data, which would be easier to deploy in the real world.
3. The authors conducted extensive experiments to justify their design choices.

### Weaknesses
1. **Basic assumption.** The proposed method based on one assumption: "hallucinated responses tend to be more arbitrary, diverse, and semantically inconsistent, which causes their representations to spread out more widely and lack a coherent structure" (Line 135-136). Is there any reference to support this claim? As I can imagine that hallucinations can be fluent and plausible, like saying an event happened in 2021 while actually happening in 2022. In such cases, the hallucinated response may still have a highly semantically similar representation to factual content. Actually, in [1], the author has shown that the representation of truthful and hallucinated generations is highly overlapped, questioning the soundness of this paper.
2. **Baseline and LLMs.** A strong baseline, TSV [1], is overlooked. The authors may want to include it in the experiment. In addition, OPT-6.7b is relatively outdated. The authors could conduct experiments on some newer LLMs like Qwen3, to make the result more convincing.
3. **Generalizability.** The proposed method finds the high-confidence samples and trains a flow model on a single dataset. It is unclear how the model can generalize to out-of-distribution data. The author could report the cross-dataset performance to demonstrate the generalizability.

[1]: Steer LLM Latents for Hallucination Detection (2025)

### Questions
1. Line 106: $X_{hallu}\to X_{hal}$.
2. The current experiments were mainly conducted on short-form QA. I wonder whether DADM can be applied to long-form QA, which is a more realistic setting.

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
This paper proposes a two-stage unsupervised framework called DADM (Distance-Aware Distribution Modeling) for detecting hallucinations in large language models (LLMs). The authors argue that existing unsupervised methods, such as HaloScope, which directly construct a representation subspace from noisy data containing hallucinations, suffer from performance degradation due to data contamination.
To address this issue, the DADM framework operates in two stages:
Stage 1: Distance-aware sample selection.
This stage applies an iterative, distance-based outlier detection process to automatically select a high-confidence core subset of truthful samples from unlabeled LLM-generated outputs.
Stage 2: Distribution modeling.
After obtaining the purified truthful subset, this stage employs Normalizing Flows to model the feature distribution of the subset, learning a precise confidence score by maximizing the likelihood of truthful samples while minimizing that of potential hallucinations.
The main contribution of this paper lies in its innovative two-stage “purification–modeling” design, which mitigates the noise contamination problem in existing approaches by first identifying a clean data subset, leading to more robust hallucination detection. Experiments on multiple benchmark datasets and LLMs demonstrate that DADM outperforms existing unsupervised methods.

### Strengths
Clear problem orientation:
The paper accurately identifies the core weakness of existing unsupervised hallucination detection methods—particularly HaloScope—in handling noisy data, and proposes a well-targeted solution.

Extensive experimental coverage:
The study conducts comprehensive experiments across multiple models (e.g., OPT, LLaMA series) and benchmarks spanning factual QA and commonsense reasoning, accompanied by detailed ablation studies that demonstrate the method’s effectiveness.

Significant performance improvement:
Experimental results show that DADM consistently and substantially outperforms all baseline methods across all test scenarios, with particularly notable gains in the AUROC metric.

### Weaknesses
Implicit dependence on a labeled validation set:
The method requires a labeled validation set to tune key hyperparameters (the retained sample ratio (m/n) and the number of feature extraction layers), which contradicts the paper’s claim of being “fully unsupervised.” The authors should more candidly discuss this prerequisite and its limitation on the method’s practicality.

Questionable validity of the core assumption:
The approach relies on the assumption that truthful samples cluster together while hallucinated samples are scattered. This assumption may not hold in tasks requiring creativity or diverse responses, thereby constraining the applicability of the method.

High hyperparameter sensitivity:
As shown in the ablation studies (Fig. 2 and Fig. 3), the model’s performance is highly sensitive to the (m/n) ratio and the number of feature extraction layers. Improper settings can cause sharp performance degradation, resulting in high tuning costs in practical deployment—contradicting the goal of providing a robust solution.

### Questions
On the dependence on the validation set:
How do the authors view the issue that, in practical scenarios (e.g., in a completely new domain), a labeled validation set may be unavailable for tuning (m/n) and the number of feature extraction layers? Is there a heuristic, label-free approach to set these key parameters?

On the applicability of the core assumption:
In what types of tasks or data distributions do you think DADM might underperform? For instance, if a task’s correct answers are inherently diverse (multi-modal), making truthful samples fail to form a single compact cluster, would DADM’s first stage mistakenly exclude some truthful samples as outliers?

On the reliability of the evaluation:
Given that the BLEURT-based labels used for evaluation are themselves noisy, how do you think this noise might affect the two stages of DADM? In particular, during the first stage, if some hallucinated samples are incorrectly labeled as “truthful” by BLEURT, could they contaminate the selected high-confidence core subset?

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
3

### Summary
The paper proposes Distance-Aware Distribution Modeling (DADM) for unsupervised LLM hallucination detection. DADM (1) purifies generations via an iterative, distance-based selector that retains consistently truthful samples and (2) fits a normalizing-flow model to the purified set, yielding likelihood-based confidence scores that maximize likelihood for truthful outputs and minimize it for hallucinations. This two-stage design captures global semantic/distributional structure while avoiding noise contamination from mixed-quality data, producing interpretable scores and reliable detection. On multiple benchmarks and LLM settings, DADM consistently outperforms prior unsupervised baselines.

### Strengths
1. The topic is interesting and addresses an important problem.
2. The paper is well written.
3. The paper is supported by solid theory.

### Weaknesses
1. The baseline model should add some new models, like LLaMA 3.2, Qwen2.5, Qwen 3.

2. For the benchmark discussion, note that several recent studies address both hallucination and maintain performance (even some improvement) on general scenario. I recommend the authors add some benchmarks like math etc.

### Questions
See above

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
3

### Summary
This paper proposes a novel unsupervised framework called Distance-Aware Distribution Modeling (DADM) for detecting hallucinations in large language models. 
DADM operates in two stages: In stage one, an iterative distance-based process is proposed to select a high-confidence subset of truthful samples using a covariance-modified Mahalanobis distance. This stage progressively filters out likely hallucinations, resulting in a coherent set of high-confidence responses. In stage two, a normalizing flow model is trained on the selected truthful subset to model the underlying distribution of truthful responses, enabling accurate likelihood estimation for hallucination detection.
The paper demonstrates that DADM consistently outperforms prior unsupervised methods across multiple datasets and LLMs. The key innovation lies in addressing the limitations of prior work (like HaloScope) that suffer from noise contamination in subspace estimation by first obtaining a clean subset of truthful responses.

### Strengths
1. The two-stage approach addresses a fundamental limitation of prior work. The key insight of first selecting a clean subset of truthful samples before modeling the distribution represents a creative solution to the noise contamination problem in existing subspace-based methods.
2. The findings of the paper are quite well supported by the experiments and the results are consistently strong across multiple datasets and LLMs, with statistically significant improvements over baselines.

### Weaknesses
1. Overly idealistic truthful subset assumption. The method presumes that truthful responses form a single, tight cluster. This assumption is unreasonable. For open-ended scenarios, multiple semantically distinct but equally correct answers form several clusters in the feature space. The proposed method might discard some small truthful clusters as the outliers. Moreover, consistent hallucinations might also form a tight cluster that gets selected as the truthful subset, which leads to the amplified mistake in the proposed second stage.
2. Hard label training with complement as negative is ill-posed. Treating the selected subset as positive hard labels and suppressing the complement as negative samples introduces substantial label noise.
3. Truthful sample ratio experiment lacks oracle baselines. Considering two ablation studies: (a) only use 0.1-0.7 ratio of BLEURT-true samples as the truthful subset for stage 2 modeling without complement treated as negative samples. (b) use BLEURT-true samples as the truthful subset with complement treated as negative samples. These two experiments would clearly disentangle the performance bottlenecks.

### Questions
1. How does your method handle scenarios where (a) truthful answers are multi-clustered, or (b) systematic hallucinations form their own dense clusters?
2. The Stage 2 objective treats the complement of the selected subset as negative samples, actively suppressing the likelihood. Since the complement inevitably contains many valid truthful responses, isn’t your model being explicitly trained to penalize the diversity of truthful answers?
3. Could you add the oracle ablations: (a) train stage 2 on only 0.1-0.7 fractions of BLEURT-true subset, (b) train stage 2 on 0.1-0.7 fractions of BLEURT-true subset and the complement as negative samples? This quantifies the upper bound and the impact of noisy labels.
4. Have you tried multi-layer feature aggregation or layer-invariant representations to reduce the sensitivity to layer choice?
5. The paper mentions the use of the softplus function in the "Hallucination Suppression" section. However, the softplus function is not explicitly utilized in the described framework. Could you clarify this?
6. The entire evaluation is based on the BLEURT scores, which are merely a proxy for human judgment and have their own biases. Recent work [1] conducted the “LLM-as-a-judge” evaluation, which is considered as a more reliable and human-aligned evaluation method. Could you add a additional ablation to verify the validity of your method?


[1] https://arxiv.org/abs/2503.01917

### Soundness
2

### Presentation
3

### Contribution
2
