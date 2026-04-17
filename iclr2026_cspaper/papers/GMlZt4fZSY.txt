# Mo B I L Ellm-R1: Exploring The Limits Of Sub-Billion Language Model Reasoners With Open Training Recipes

Chia-Jung Chang Meta AI
Wei Wen Meta AI
Chen Lai Meta AI
Rick Cao Meta AI

| Changsheng Zhao∗ Meta AI cszhao@meta.com   |
|--------------------------------------------|

Yuandong Tian Meta AI

Raghuraman Krishnamoorthi Meta AI

Yangyang Shi Meta AI

Vikas Chandra Meta AI

| Ernie Chang∗ § Meta AI erniecyc@meta.com   |
|--------------------------------------------|

Zechun Liu 
∗ †
Meta AI zechunliu@meta.com

## Abstract

The paradigm shift in large language models (LLMs) from instinctive responses to chain-of-thought (CoT) reasoning has fueled two prevailing assumptions: (1) reasoning capabilities only emerge in sufficiently large models, and (2) such capabilities require training on massive datasets. While the first assumption has already been challenged by recent sub-billion-parameter reasoning models such as Qwen3-0.6B and DeepSeek distilled variants, the second remains largely unquestioned. In this work, we revisit the necessity of scaling to extremely large corpora (>10T tokens) for reasoning emergence. By carefully curating and resampling open-source datasets that we identify as beneficial under our designed metrics, we demonstrate that strong reasoning abilities can emerge with far less data. Specifically, we show that only ∼2T tokens of high-quality data are sufficient, and pre-training with 4.2T tokens on the dataset resampled from these ∼2T tokens, followed by a established post-training procedure, enables the development of MobileLLM-R1, a series of sub-billion-parameter reasoning models that substantially outperform prior models trained on fully open-sourced data. For example, MobileLLM-R1-950M achieves an AIME score of 15.5, compared to just 0.6 for OLMo-2-1.48B and 0.3 for SmolLM-2-1.7B. Remarkably, despite being trained on only 11.7% of the tokens compared to Qwen3's proprietary 36T-token corpus for pretraining, MobileLLM-R1-950M matches or surpasses Qwen3-0.6B across multiple reasoning benchmarks. To facilitate further research in this direction, we have made the models (https://huggingface.co/collections/facebook/mobilellm-r1) and code (https://github.com/facebookresearch/MobileLLM-R1) publicly available, along with the complete training recipe, data sources, and data mixing ratios.

## 1 Introduction

Large language models (LLMs) such as GPT (Achiam et al., 2023), Qwen (Yang et al., 2025; 2024), and DeepSeek (Guo et al., 2025) have demonstrated remarkable progress in explicit reasoning. Advances have been driven by scaling model size, expanding training data, and applying post-training techniques such as supervised fine-tuning (SFT) and reinforcement learning (RL). Reasoning LLMs are capable of tackling complex problems by following long chains of thought that incorporate reflection, backtracking, and self-validation. At the same time, reasoning traces have evolved from prompt-based chain-of-thought (CoT) in-
∗Equal contribution, authors listed in alphabetical order. §Led overall data curation efforts †Corresponding author

![0_image_0.png](0_image_0.png)

1 context learning (Wei et al., 2022) to models explicitly optimized on long reasoning traces to generate multi-step reasoning sequence (Jaech et al., 2024). However, this paradigm poses increasing challenges for real-world deployment. Large models already strain resourceconstrained devices (Liu et al., 2024), and long-context reasoning further exacerbates memory usage as KV cache growth sharply increases the footprint (Sadhukhan et al., 2025). Looking ahead, one can envision a future with personal assistants, smart homes, and robots increasingly relying on on-device reasoning for complex tasks. In such a world, deployability and portability will become inevitable trends for the next generation of LLMs. This motivates our central question: *Given strict capacity constraints, what is the most effective recipe to endow small reasoning models with* strong capabilities and unlock their hidden potential? Developing small reasoning models poses unique challenges beyond simply scaling down large ones. For large models, expanding the corpus often drives stronger generalization. In contrast, small language models are far more sensitive: noise in the data can easily overwhelm their limited capacity, making data quality and curation paramount. As models shrink, neurons must encode more overlapping knowledge, increasing the risk of interference and conflicts (Zhu et al., 2025)—superposition provides an intuitive lens for understanding this challenge (Elhage et al., 2022). Mitigating these risks requires carefully optimized data, objectives, and training procedures. While extensive research has explored how post-training objectives and data curation can elicit reasoning from pretrained models (Wang et al., 2025; Li et al., 2025), far less attention has been paid to a more fundamental question: *How can* we endow pretrained models with the latent potential for reasoning in the first place? This work addresses this gap by investigating two critical questions: (1) What kinds of data are most effective for instilling reasoning capability, and (2) How can diverse forms of reasoning—such as coding, mathematics, and logical problem-solving—be embedded into a compact model without overwhelming its limited capacity?

Through capability-aware data curation and probing into the latent factors that govern reasoning, we achieve highly token-efficient pretraining compared to prior work. With only 4.2T training tokens, just 11.7% of Qwen's 36T, our MobileLLM-R1-950M model, matches or surpasses Qwen3-0.6B Yang et al. (2025) on multiple reasoning benchmarks, placing itself on the Pareto frontier of accuracy–training-token efficiency trade-off curve (Figure 1). Beyond introducing a high-performing small-scale reasoning model, we share both the insights and the pitfalls encountered along the way, offering a first-hand glimpse into the complex yet fascinating mechanisms behind reasoning models. Our contributions are as follows:
- We introduce benchmark-free, self-evolving data optimization for pre-training data curation, a principled dataset-level weighting approach that leverages cross-domain influences to tailor the data mixture. This facilitates robust reasoning generalization on held-out benchmarks, achieved without exposing them during training or data mixture optimization.

- We further propose a data–model co-evolution strategy to adapt to rapid changes in model capacity during mid-training. We show that this process converges as most samples reach zero or negative influence, indicating that the dataset's information has been largely exhausted and offers minimal further improvement.

- Compared to existing fully open-source models, MobileLLM-R1-950M model attains 5× higher MATH
accuracy than Olmo 1.24B (Allal et al., 2025) and 2× higher than SmolLM2 1.7B (Allal et al., 2025), while significantly outperforming both on code benchmarks despite having fewer parameters.

- We have disclosed the complete set of open-sourced datasets employed in our study and have released all trained models and accompanying code to enable full reproducibility and foster future research.

## 2 Pre-Training: Balance Of Capabilities

The notion of *reasoning* in large language models (LLMs) remains both complex and contested. While the term is sometimes used to describe a model's ability to engage in structured, multi-step inference (Wei et al., 2022; Kojima

![1_image_0.png](1_image_0.png)

Figure 2: Overall training pipeline of MobileLLM-R1.
et al., 2022), it has also become a proxy for improved performance on challenging benchmarks (Srivastava et al., 2022). In this work, we adopt a pragmatic stance: we treat gains on reasoning-centric benchmarks as reasonable evidence of enhanced reasoning *behaviors*, while remaining cautious about equating such gains with genuine reasoning ability in the cognitive sense (Bender & Koller, 2020). Concretely, this entails selecting informative datasets that most effectively enhance the target capability (Section 2.1) and optimizing their combination ratios to maximize knowledge acquisition within the fixed token budget (Section 2.2). Figure 2 illustrates the training pipeline with the full procedure in Appendix A.

## 2.1 Selecting Informative Datasets For Target Capability

To systematically assess which pre-training distributions most effectively support downstream reasoning behaviors, a naïve approach would be to pre-train separate models on all combinations of candidate datasets, followed by midtraining and post-training, and then measure performance on reasoning benchmarks. However, this strategy is both computationally prohibitive and prone to overfitting to specific benchmarks. Instead, we design a leave-one-out (LOO) analysis. We train models from scratch on the entire set of pre-selected high-quality datasets, excluding one dataset at a time. We then trace negative log-likelihood (NLL) on curated capability-probing datasets throughout training. Each *capability-probing dataset* can be viewed as defining a token distribution that implicitly induces the necessary preconditions for reasoning to emerge. Importantly, these distributions are heterogeneous: when learned, they contribute unequally to different reasoning-related capabilities, such as *code* understanding, *general knowledge*, and *mathematical problem solving* (Chen et al., 2021; Hendrycks et al., 2021; Cobbe et al., 2021).

## 2.1.1 Curation Of Representative Datasets And Capability-Probing Dataset

Curating the *capability-probing datasets* is critical: it must be representative of the desired capabilities and sufficiently comprehensive to cover each reasoning category. We describe the process of preparing capability datasets as follows. Hierarchical Rejection Sampling. To derive a compact *capability-probing datasets* for each domain, we employ a hierarchical rejection sampling pipeline that integrates multiple classifier- and model-based filters. The objective is to construct a small yet representative target dataset for each capability, such that it can serve as a faithful proxy for reasoning performance while dramatically reducing overall volume during evaluation. For each corpus in Table 5, we first apply the FINEWEB-EDU classifier (Penedo et al., 2024) to select samples with high educational value, retaining only those with classifier scores above 4. Next, we incorporate model-based evaluation by scoring each remaining sample using the Ask-LLM paradigm (Sachdeva et al., 2024). The evaluation prompt asks the model to judge whether a sample should be included in a reasoning-probing dataset, framed as a binary classification task ("1" for inclusion, "0" for exclusion). Rather than relying solely on the hard prediction, we record the probability assigned to "1" as a graded measure of the model's confidence in the example's reasoning relevance. For all Ask-LLM scoring, we select the top 10% samples within each dataset. This step complements classifier-based quality filtering by directly capturing signals of reasoning relevance, consistent with recent findings that costly, fine-grained quality samplers can outperform simple maximum-coverage approaches in terms of data efficiency (Sachdeva et al., 2024; Pang et al., 2025; Chen & Zhou, 2025). Next, we apply a domain-specific prompt to Ask-LLM for each capability with specific emphasis on code, math, general knowledge or combined. Finally, we perform semantic deduplication across corpora, shrinking each dataset in Table 5 to a subset of roughly 10,000 examples. This yields the *representative datasets* DRi, each containing highly representative samples for its corresponding corpus. We categorize them into three domains according to their composition: Code (C), Math (M), and Knowledge (K):
- C = {StarCoder, StackExchange, Nemotron-Code, Cosmopedia, Natural Reasoning, pes2o} - M = {OpenWebMath, FineMath, Algebraic Stack, Nemotron-Math, Cosmopedia, Natural Reasoning, pes2o}
- K = {FineWeb-Edu, Wikipedia, Arxiv, Cosmopedia, Nemotron-Science, Natural Reasoning, pes2o}
Note that a single dataset may contain data relevant to multiple domains, in which case its representative subset is included in more than one domain. In this way, we construct three filtered, domain-specialized *capability-probing* datasets, DP
C,M,K, by combining the representative subsets from all datasets assigned to each domain. We use (C,M, K)
to denote a *mixture of original datasets* prior to down-sampling.

## 2.1.2 Disentangling The Impact Of Data Sources

We then evaluate the impact of different pretraining corpora on the emergence of reasoning ability by measuring the negative log-likelihood (NLL) on the *capability-probing datasets*. To isolate the contribution of each corpus, we perform

![3_image_0.png](3_image_0.png)

$$(1)$$

rigorous leave-one-out ablation studies, systematically removing individual datasets and measuring the resulting change in NLL across the three *capability-probing datasets* corresponding to Code, *Math*, and *General Knowledge* capabilities.

Group Impact via Loss Delta. We define the impact of a dataset Dj on a reasoning capability as the change in loss it induces on the corresponding *capability-probing dataset* DP
C,M,K . Let ˆθ denote parameters trained on the full dataset D = ∪iDi, and ˆθ−j denote parameters trained with Dj removed. The *group impact* of Dj on DP
c, c *∈ {C*,M, K} is

$$\Delta{\mathcal{L}}({\mathcal{D}}_{j},{\mathcal{D}}_{{\mathcal{C}},{\mathcal{M}},{\mathcal{K}}}^{\mathcal{P}})\;=\;\mathbb{E}_{z\sim{\mathcal{D}}_{{\mathcal{C}},{\mathcal{M}},{\mathcal{K}}}^{\mathcal{P}}}\left[\ell(z;{\hat{\theta}}_{-j})-\ell(z;{\hat{\theta}})\right],$$
ˆθ), (1)
where ℓ is the evaluation loss. A positive value indicates that removing Dj increases the benchmark loss (i.e., Dj is beneficial), while a negative value suggests that its presence may hurt performance. Leave-One-Out Ablations. We operationalize Eq. 1 by training models under leave-one-out settings and measuring the resulting differences in loss across benchmarks. Together, these analyses highlight not only *which* sources matter most, but also *how much* marginal benefit additional data from a given source provides. This methodology allows us to disentangle the contributions of heterogeneous data sources to reasoning-related performance in code, knowledge, and mathematics. Figure 3 presents the results of our leave-one-out (LOO) experiments across the three evaluated capabilities. To ensure fairness, tokens from each dataset are sampled with equal probability, and no example is repeated during pretraining. Without this normalization, larger datasets such as FINEWEB-EDU would otherwise dominate exposure. We find that excluding FINEWEB-EDU results in the largest degradation across all capabilities, including knowledge, math, and code. We attribute this to its web-based composition, which provides broad and diverse coverage across domains. This result highlights the central role of large-scale web data as a form of "glue" that binds heterogeneous domains together. In contrast, domain-specific datasets primarily strengthen their respective domains: STARCODER substantially improves code performance (and, interestingly, math), while math-focused corpora primarily benefit math. However, their transfer to general knowledge is limited. An unexpected observation is that STARCODER benefits math more than OPENWEB- MATH benefits code, a reversal of the commonly held view that mathematical data contributes disproportionately to coding ability Lewkowycz et al. (2022). Finally, WIKIPEDIA appears to contribute little to math or code compared to web or domain-specific data, yet remains necessary as a structured and reliable source of factual knowledge.

## 2.2 Datamixing Via Cross-Capability Self-Influence

In Section 2.1.2, we demonstrate that the pre-selected datasets yield measurable utility, as evidenced by reductions in NLL on *capability-probing datasets*. Building on this, we study token budget allocation: given a fixed training budget, how should tokens be distributed across heterogeneous datasets to maximize downstream reasoning performance? Uniform sampling provides a natural baseline but ignores the varying marginal utility of different datasets. Our key insight is that more informative datasets should receive proportionally larger sampling ratios. To operationalize this, we leverage the *influence score* to quantify each dataset's contribution and guide principled token re-weighting.

![4_image_0.png](4_image_0.png) 
Generally, let θ
∗ denote parameters obtained by training on dataset D, xi a training example, xtest a example from the target set, which in our case is *capability probing set*, and L(x, θ) the loss function. The *influence score* of xi on the test loss can be approximated as I(xi, xtest; θ) = −∇θL(xtest; θ
∗)
⊤H
−1 θ
∗ ∇θL(xi; θ
∗), (2)
where Hθ
∗ is the Hessian of the training loss at θ
∗. While directly computing the Hessian matrix Hθ
∗ is computationally prohibitive for large models, *AutoMixer* (Chang et al., 2025) proposes an efficient approximation method that bypasses explicit Hessian inversion and makes influence score calculation scalable. We extend the *AutoMixer* framework by treating influence scores as quantitative proxies linking individual training samples to capabilities. Concretely, the influence of a sample on the validation loss of a capability-probing dataset measures the *connection strength* between the sample and the corresponding capability. Rather than using benchmark test sets, we employ samples from *capability-probing datasets* and compute influence scores separately for Code (C), Math (M), and Knowledge (K) domains.

For each training sample xi from a source dataset, we compute its influence on the validation loss of all three capabilityprobing datasets. We term this "self-influence" when training and validation samples originate from the same capability and "cross-influence" if they target different capabilities. Because the source datasets are substantially large, we develop an efficient influence estimation algorithm that operates on the *representative dataset* (defined in Section 2.1.1) of each source in Table 5, yielding a computationally scalable surrogate that faithfully preserves cross-capability contribution signals. Concretely, if xi ∈ DRStarCoder ⊂ DP
C, we evaluate Self-influence: I(xi, xtest ∈ DP
C; θC,t), Cross-influence: I(xi, xtest ∈ DPM; θM,t), I(xi, xtest ∈ DPK; θK,t) (3)
Here, checkpoints θC,t, θM,t, and θK,t are obtained by training separate models to convergence on the full training sets of domains C,M, K, yielding domain-specialized parameters. Following the *AutoMixer* protocol, a single checkpoint is insufficient to capture the full training dynamics. We therefore compute influence scores at T = 10 evenly spaced checkpoints, weighting each score proportionally to its training step to emphasize later-stage training. These weighted scores quantify the evolving influence of example xi on the Code, Math, and Knowledge domains throughout training.

Then, the joint influence of a sample is computed as

$${\mathcal{I}}_{\mathrm{joint}}(x_{i})=\sum_{c\in\{{\mathcal{C}},{\mathcal{M}},{\mathcal{K}}\}}\sum_{t=1}^{T}\alpha_{c,t}\cdot{\mathcal{I}}(x_{i};\theta_{c,t}),$$
$,x_{test}\in$

$$|\rangle$$

$$\mathbf{\Pi}$$

αc,t · I(xi; θc,t), (4)
where θc,t is the checkpoint t for capability c, and αc,t are blending factors reflecting acquisition speed across checkpoints. We assign linearly increasing weights αc,t ∝ t across the T checkpoints, and maintain uniform weights across capabilities c.

Each source dataset g is then assigned a sampling weight (wg):

$$w_{g}=\frac{\rho_{g}}{\sum_{g^{\prime}}\rho_{g^{\prime}}},\quad\rho_{g}=\frac{1}{N_{g}}\sum_{x_{i}\in g}\mathcal{I}_{\mathrm{joint}}(x_{i})\cdot s_{i},\tag{1}$$

with Ng the token count of dataset g and sithe length of sample xi. The resulting mixture respects the global budget N
while prioritizing datasets whose samples show strong self- and cross-capability connections.

![5_image_0.png](5_image_0.png)

In this setup, we derive a closed-form solution for the data mixture ratio, enabling effective utilization of the limited token budget while enhancing each dataset's contribution to model performance. Using the representative datasets and capability probing datasets sampled from the training corpus, it makes influence score computation tractable and exposes how strongly each source dataset (Table 5) contributes to Code, Math, and Knowledge capabilities. This formulation enables principled weighting at the dataset level , grounding the mixture in empirically measured cross-domain influences rather than heuristic allocation. As illustrated in Figure 4, the resulting mixture consistently outperforms uniform sampling on Code, Math, and Knowledge benchmarks—none of which are accessed during training or mixture construction—demonstrating the potential for benchmark-free, self-adaptive data optimization.

## 3 Mid-Training: Knowledge Compression

After the model has been exposed to broad knowledge during pretraining, the mid-training phase focuses on compressing this knowledge and maximizing performance on target tasks. We design each mid-training phase with a limited budget of 100B tokens. Unlike pretraining, mid-training induces dramatic shifts in weight distributions and necessitates a more sophisticated, co-evolving model–data mixture strategy. To this end, we propose a novel mid-training paradigm that enables self-boosting: the model trained on a given data mixture is used to compute influence scores for samples, which are then leveraged to dynamically remove negative influence samples and adjust the data sampling ratios for the next phase. As training progresses, the influence scores of data samples increasingly concentrate around zero or negative values, indicating near-complete utilization of the informative content in the dataset and convergence of the process.

Notably, this self-evolving scheme requires no access to external benchmark datasets, yet it substantially improves performance on target benchmarks relative to uniform sampling.

We build upon the Dolmino dataset, which has been shown in the OLMo 2 (OLMo et al., 2024) to be an effective mid-training corpus. To enhance domain specialization, we augment Dolmino with additional mathematics and programming data, aiming to strengthen the model's math and coding capabilities. Given a training example x i from the mid-training dataset and a probe example x lest from capability probing dataset DE M.K , we calculate the influence score I ( x i , x test ; θ ). Here, rather than relying on separately trained models for domain-specific corpora, we leverage the pretrained model θ to capture the dataset requirements at the current stage of training. The data–model co-evolution proceeds iteratively through the following steps:
(1) Sample-level influence for rejection sampling.  Intuitively, this step acts as a filtering mechanism: only training examples that positively contribute to the target capabilities are retained, while neutral or detrimental samples are discarded. Given the raw mid-training dataset D(raw), at compression phase t we define the retained dataset as:
Dt = {xi ∈ D(raw) : I(xi; θt) > 0}, (6)
where θt is the model state at phase t. This rejection sampling can be interpreted as an iterative data distillation process:
the model continually refines its training distribution, focusing only on samples that yield positive transfer toward the target probing dataset. (2) *Dataset-level influence for adaptive data mixing.* Beyond sample-level filtering, we aggregate influence scores to the dataset level, enabling adaptive control of the mixing ratio among mid-training datasets, according to Eqs. 4 and 5). (3) *Train the model on the curated data and repeat the iterative process.* The compressed dataset with the updated mix ratio is used for continued mid-training:
θt+1 = MidTrain(θt, Dt). (7)
and the updated model θt+1 provides refined influence scores for the next stage. This iterative compression continues until no additional samples yield a positive influence score. In practice, we find that two stages suffice to produce a well-compressed dataset that balances generality with targeted capability improvements. Intuition: Distributional Compression of Influence. The compression phases can be viewed as iteratively distilling the mid-training dataset in alignment with the model's evolving capacity throughout training. In early phases, the influence scores are more varied because the model θt is still under-trained. However, as t increases, the model becomes better aligned with the target distribution, and its estimates of sample importance are narrowed down (See Figure 5). This recursive interplay produces increasingly refined datasets: uninformative (or negative-influence) samples are discarded, thus amplifying the impact of high-value samples. Conceptually, compression phases mimic an iterative denoising process, where each step sharpens the signal from D(target) against the noisy background of D(raw). We terminate the iteration until the distribution of influence converges to approximately zero. Figure 5 shows histograms of influence scores for general knowledge and math across stage 1 and stage 2 of training. During stage 2, the distribution of influence scores undergoes a pronounced "compression": the range of values narrows, and extreme contributions become less pronounced. Intuitively, as the model becomes more capable, the influence of individual data samples converges toward zero, indicating diminishing impact on downstream reasoning performance. We further highlight the effect of this influence compression in Figure 6. Subsampled mid-training data consistently outperforms the original mid-training set under both standard crossentropy training and knowledge distillation. Notably, the original data experiences a pronounced performance dip around 30K steps, whereas the subsampled data maintains higher downstream performance throughout training. A
similar trend occurs with knowledge distillation using the LLaMa3-8B teacher model, though the performance gap is slightly smaller than under pure cross-entropy. These results indicate that compressing influence scores effectively identifies and preserves the most informative samples, leading to more robust and stable performance trends.

Figure 6: Comparison of the impact on the MMLU benchmark between the original mid-training data and the subsampled data, with and without knowledge distillation.

![6_image_0.png](6_image_0.png) 

## 4 Experimental Results

Using the datasets from Sections 2 and 3, we obtain MobileLLM-R1-base. Given that our primary goal is to elucidate how data curation in pre-training and mid-training builds strong small reasoning models, we leverage established supervised fine-tuning (SFT) datasets. We first apply Tülu-3-SFT (Lambert et al., 2024) dataset for instruction alignment and OpenScienceReasoning-2, OpenCodeReasoning-2 (Ahmad et al., 2025) and OpenMathReasoning (Moshkov et al., 2025) for reasoning-oriented SFT to extend context and elicit long chain-of-thought reasoning. We validate our training process and data choices, compare our model against prior work trained on the same reasoning SFT datasets.

Post-training process ablation: Our ablation studies (Table 1) reveal several key insights into the two-stage posttraining pipeline. (1) Instruction-following supervision (Tulu-SFT) provides crucial alignment signals that make subsequent reasoning adaptation significantly more effective than starting directly with reasoning data. (2) Domainspecific reasoning corpora (math, science, code) yield consistent gains on their respective benchmarks, while scientific reasoning data further exhibits strong cross-domain transfer to math and code. (3) Symbolic reasoning improvements often trade off with factual knowledge retention, as introducing math or code data reduces MMLU performance, particularly in smaller models with limited capacity. (4) Decoupling alignment and reasoning proves essential: a staged approach (Tulu first, then reasoning data) consistently outperforms joint training, especially on math and general reasoning benchmarks. Comparison with baselines on identical reasoning SFT: To disentangle the contribution of curated pre-training and mid-training data from that of high-quality post-training data, we conducted an ablation study. Specifically, we finetune all baseline instruct models, as well as the MobileLLM-R1 general supervised fine-tuned model (trained for 2 epochs on the Tulu dataset), on the joint reasoning SFT corpus (OpenMathReasoning + OpenScienceReasoning-2 + OpenCodeReasoning-2) for one epoch. Our results in Table 2 show that, even under identical supervised fine-tuning, models with stronger pre-training and mid-training exhibit more robustly embedded knowledge, which in turn facilitates the elicitation of reasoning capabilities during post-training. In that sense, MobileLLM-R1 consistently outperforms prior models trained on fully open-source corpora, such as OLMo-2 and SmolLM, on reasoning benchmarks. Notably, our 140M and 360M checkpoints achieve substantial gains over SmolLM baselines, while our 950M model surpasses both OLMo-2 1.48B and SmolLM-1.7B, despite its significantly smaller size.

## 4.1 Final Results

![7_image_0.png](7_image_0.png)

Figure 7: Evolution of reasoning capability during training, measured by perplexity reductions on reasoning-focused benchmarks: HumanEval for coding and GSM8K for math.

While the model undergoes the pre-training, mid-training, and post-training stages, we track its reasoning ability by measuring perplexity on two reasoning-focused benchmarks: HumanEval and GSM8K. Our results in Figure 7 show that the perplexity on math sees a significant drop early during the second phase of pre-training. Interestingly, the same model, when subjected to the second phase of mid-training with limited data, exhibits a dramatic perplexity decrease in HumanEval. This suggests that the knowledge acquired from math training is transferable to coding, enabling the model to develop coding abilities subsequently. In the following, we position our trained MobileLLM-R1 within the context of prior state-of-the-art models and compare their performance. We present results for two sets of models: the base models, evaluated after pre-training and mid-training, and the final models after the complete training pipeline. The experimental settings and full training pipeline can be found in Section A Base Model Figure 8 compares base reasoning models across multiple benchmarks. We group them into fully opensource models (OLMo OLMo et al. (2024), SmolLM Allal et al. (2025), MobileLLM-R1), with weights, data, and training recipes available, and partially open-source models (Qwen Yang et al. (2025), Gemma Team et al. (2025), LLaMA Dubey et al. (2024)), which released model weights and partial training procedures. Compared to fully open-source models, MobileLLM-R1 consistently outperforms both OLMo and SmolLM across all parameter scales. For example, at the 140M scale, MobileLLM-R1 achieves 16.3% GSM8K and 15.9% HumanEval, dramatically

| OpenCodeReasoning-2 datasets respectively. Stage 1 Stage 2 MATH GSM8K LCBv6   | MMLU        |      |      |      |      |
|-------------------------------------------------------------------------------|-------------|------|------|------|------|
| Ablation on Stage 1                                                           |             |      |      |      |      |
| w/o Tulu-3                                                                    | M + C + S   | 56.2 | 68.2 | 13.1 | 44.0 |
| w/ Tulu-3                                                                     | M + C + S   | 57.8 | 68.5 | 13.7 | 43.7 |
| Ablation on Stage 2                                                           |             |      |      |      |      |
| Tulu-3                                                                        | M           | 57.4 | 68.2 | 0.0  | 43.1 |
| Tulu-3                                                                        | C           | 16.2 | 31.0 | 12.0 | 39.9 |
| Tulu-3                                                                        | S           | 23.8 | 62.2 | 3.4  | 45.6 |
| Tulu-3                                                                        | M + C       | 58.4 | 65.6 | 10.9 | 40.4 |
| Tulu-3                                                                        | M + S       | 60.0 | 66.9 | 0.6  | 45.0 |
| Tulu-3                                                                        | C + S       | 29.4 | 65.3 | 14.3 | 44.4 |
| Tulu-3                                                                        | M + C + S   | 57.8 | 68.5 | 13.7 | 43.7 |
| Joint Ablation                                                                |             |      |      |      |      |
| Tulu-3 + (M + C + S)                                                          | -           | 56.2 | 53.1 | 14.9 | 44.0 |
| Tulu-3                                                                        | (M + C + S) | 57.8 | 68.5 | 13.7 | 44.0 |

Table 2: Evaluation of reasoning capabilities elicited by

different models when fine-tuned on the same reasoning supervised finetuning (SFT) dataset. Baseline models use their instruct checkpoints; our model uses intermediate Tulu3-SFT checkpoints, denoted with *. All models are trained for one epoch on the joint reasoning SFT corpus (OpenMathReasoning + OpenScienceReasoning-2 + OpenCodeReasoning-2).

Model Size MATH GSM8K LCBv6 SmolLM2-135M-Instruct 135M 3.2 1.6 0.6 MobileLLM-R1**-140M*** 140M **4.8 3.7 1.1** SmolLM2-360M-Instruct 362M 5.2 7.4 3.4 MobileLLM-R1**-360M*** 359M **19.2 23.8 4.0** OLMo-2-0425-1B-SFT 1.48B 53.0 58.8 11.4 SmolLM2-1.7B-Instruct 1.71B 41.4 50.5 7.4 MobileLLM-R1**-950M*** 949M **57.8 68.5 13.7**

![8_image_0.png](8_image_0.png)

![8_image_1.png](8_image_1.png)

surpassing SmolLM2-135M (1.8% and 0.0%, respectively). Compared to prior partially open-source models, such as Qwen3-0.6B, MobileLLM-R1 achieves comparable or superior results despite being trained on substantially fewer tokens (4.2T for MobileLLM-R1 vs. 36T for Qwen3). Notably, MobileLLM-R1-950M attains the highest HumanEval score (46.3%) among all sub-1B models, significantly outperforming Qwen3-0.6B (30.5%).

Post-trained Model Figure 9 presents the performance of post-trained models. Notably, on LiveCodeBench, small models below 400M parameters struggle to produce reliable outputs. In contrast, MobileLLM-R1-360M achieves 5.1 points, surpassing even models with over 1B parameters, such as SmolLM2-1.7B, Gemma3-1B, and LLaMA3.2-1B. Remarkably, MobileLLM-R1-950M demonstrates a substantial accuracy gain over Qwen3-0.6B on LiveCodeBench and even matches the performance of much larger state-of-the-art models, such as DeepSeek-R1-Distill-Qwen-1.5B. Across Math and AIME benchmarks, MobileLLM-R1 consistently outperforms other fully open-source models and achieves scores comparable to the partially open-source Qwen3 series. See Appendix B.1 for detailed comparisons.

## 5 Related Work

The advent of GPT-3 (Brown et al., 2020) highlighted the transformative potential of large language models (LLMs),
spurring research on both proprietary models (e.g., Claude (Anthropic, 2024)) and open-source alternatives (e.g.,
LLaMA (Touvron et al., 2023a;b; Dubey et al., 2024), Gemma (Team et al., 2024a;b; 2025), Qwen (Team, 2024; Yang et al., 2025)). Scaling laws have been central to understanding how larger models elicit emergent capabilities and potential performance singularities, while efficiency-accuracy trade-offs in smaller models are increasingly studied to optimize computational resources. MobileLLM Liu et al. (2024) pioneered on-device LLM deployment, followed by high-performance small models such as OLMO OLMo et al. (2024) and SmolLM Allal et al. (2025), reflecting a broader trend toward transparency, including open-sourcing weights, training data, and full pipelines. Research has also shifted from instinctive response to explicit reasoning thinking, exemplified by OpenAI's O1 (Jaech et al., 2024) and DeepSeek-R1 Guo et al. (2025). Qwen Yang et al. (2025) is another example of a high-quality reasoning model, with smaller variants achieving state-of-the-art benchmark results. Prior studies suggest reasoning emerges only after extremely large-scale pretraining Yang et al. (2025). In contrast, our work shows that a small model can attain strong reasoning abilities using only 4.2T pretraining tokens—comparable to Qwen trained on 36T tokens. We release our full data collection and training pipeline with a detailed rationale for each data selection, ensuring maximal transparency and reproducibility.

## 6 Conclusion

We present a data-centric framework to maximize reasoning in small language models under limited parameters and tokens. We introduce benchmark-free, self-evolving data optimization, a principled dataset-level weighting method that leverages cross-domain influences to dynamically tailor the data mixture. This approach enables strong performance on code, math, and knowledge benchmarks without exposing any benchmark data during training or mixture construction.

Trained on 4.2T tokens drawn from ∼2T curated open-source data, MobileLLM-R1 achieves state-of-the-art results among small models with a fully open-sourced recipe, and matches Qwen3-0.6B with only 11.7% of its 36T-token training data. Our findings challenge the conventional belief that small reasoning models require massive data, instead underscoring the pivotal role of data quality, token efficiency, and principled data curation.

## Ethics Statement

Our work investigates methods for optimizing the training and deployment of small-scale language models. The models and datasets used in this study are publicly available and widely adopted in the research community. We did not collect new human subject data, nor did we rely on sensitive or proprietary sources. Potential risks primarily concern the general risks associated with language models. While these risks are not specific to our contributions, we acknowledge them and emphasize that our focus is methodological rather than on downstream deployment. We encourage future applications of our techniques to carefully assess ethical considerations in their respective domains.

## Reproducibility Statement

We follow the ICLR reproducibility guidelines. All datasets used are publicly available, as we revealed in Section A.3. We describe data processing procedures, model architectures, training configurations, and hyperparameters in detail in Sections A. We will release code and trained model checkpoints to support full reproducibility.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.

Wasi Uddin Ahmad, Somshubra Majumdar, Aleksander Ficek, Sean Narenthiran, Mehrzad Samadi, Jocelyn Huang, Siddhartha Jain, Vahid Noroozi, and Boris Ginsburg. Opencodereasoning-ii: A simple test time scaling approach via self-critique. *arXiv preprint* arXiv:2507.09075, 2025.

Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlícek, Agustín Piqueres Lajarín, Vaibhav Srivastav, et al. Smollm2: When smol goes big–data-centric training of a ˇ small language model. *arXiv preprint arXiv:2502.02737*, 2025.

Anthropic. Claude. https://claude.ai, 2024. Large language model. Emily M. Bender and Alexander Koller. Climbing towards NLP's Everest: Avoiding the pitfall of understanding. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 5185–5198, 2020.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020.

Ernie Chang, Yang Li, Patrick Huber, Vish Vogeti, David Kant, Yangyang Shi, and Vikas Chandra. AutoMixer: Checkpoint artifacts as automatic data mixers. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 19942–19953, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi:
10.18653/v1/2025.acl-long.979. URL https://aclanthology.org/2025.acl-long.979/.

Fei Chen and Wenchi Zhou. Quality over quantity: An effective large-scale data reduction strategy based on pointwise v-information.

arXiv preprint arXiv:2507.00038, 2025. URL https://arxiv.org/abs/2507.00038.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, et al. Evaluating large language models trained on code. In *arXiv preprint* arXiv:2107.03374, 2021.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, et al. Training verifiers to solve math word problems. In *arXiv* preprint arXiv:2110.14168, 2021.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. *arXiv e-prints*, pp. arXiv–2407, 2024.

Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, et al. Toy models of superposition. *arXiv preprint arXiv:2209.10652*, 2022.

Quentin Garrido, Randall Balestriero, Laurent Najman, and Yann Lecun. Rankme: Assessing the downstream performance of pretrained self-supervised representations by their rank. In *International conference on machine learning*, pp. 10929–10974.

PMLR, 2023.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In *International Conference on Learning Representations (ICLR)*, 2021.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. *arXiv preprint arXiv:2412.16720*, 2024.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hannaneh Hajishirzi. Tülu 3: Pushing frontiers in open language model post-training. 2024.

Aitor Lewkowycz, Anders Johan Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Venkatesh Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models. *arXiv preprint arXiv:2206.14858*, 2022. URL https:
//arxiv.org/abs/2206.14858.

Dacheng Li, Shiyi Cao, Tyler Griggs, Shu Liu, Xiangxi Mo, Eric Tang, Sumanth Hegde, Kourosh Hakhamaneshi, Shishir G Patil, Matei Zaharia, et al. Llms can easily learn to reason from demonstrations structure, not content, is what matters! arXiv preprint arXiv:2502.07374, 2025.

Zechun Liu, Changsheng Zhao, Forrest Iandola, Chen Lai, Yuandong Tian, Igor Fedorov, Yunyang Xiong, Ernie Chang, Yangyang Shi, Raghuraman Krishnamoorthi, et al. Mobilellm: Optimizing sub-billion parameter language models for on-device use cases. In *Forty-first International Conference on Machine Learning*, 2024.

Ivan Moshkov, Darragh Hanley, Ivan Sorokin, Shubham Toshniwal, Christof Henkel, Benedikt Schifferer, Wei Du, and Igor Gitman.

Aimo-2 winning solution: Building state-of-the-art mathematical reasoning models with openmathreasoning dataset. arXiv preprint arXiv:2504.16891, 2025.

Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, et al. 2 olmo 2 furious. *arXiv preprint arXiv:2501.00656*, 2024.

Xi Pang et al. Fine-grained data selection for llm supervised fine-tuning. *arXiv preprint arXiv:2502.01968*, 2025. URL https:
//arxiv.org/abs/2502.01968.

Guilherme Penedo, Hynek Kydlícek, Anton Lozhkov, Margaret Mitchell, Colin A Raffel, Leandro Von Werra, Thomas Wolf, et al. ˇ
The fineweb datasets: Decanting the web for the finest text data at scale. *Advances in Neural Information Processing Systems*, 37: 30811–30849, 2024.

Noveen Sachdeva, Benjamin Coleman, Wang-Cheng Kang, Jianmo Ni, Lichan Hong, Ed H Chi, James Caverlee, Julian McAuley, and Derek Zhiyuan Cheng. How to train data-efficient llms. *arXiv preprint arXiv:2402.09668*, 2024.

Ranajoy Sadhukhan, Zhuoming Chen, Haizhong Zheng, Yang Zhou, Emma Strubell, and Beidi Chen. Kinetics: Rethinking test-time scaling laws. *arXiv preprint arXiv:2506.05333*, 2025.

Aarohi Srivastava et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. In *arXiv* preprint arXiv:2206.04615, 2022.

Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024a.

Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, et al. Gemma 2: Improving open language models at a practical size. *arXiv* preprint arXiv:2408.00118, 2024b.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical report. *arXiv preprint arXiv:2503.19786*, 2025.

Qwen Team. Qwen2 technical report. *arXiv preprint arXiv:2407.10671*, 2024. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023b.

Zengzhi Wang, Fan Zhou, Xuefeng Li, and Pengfei Liu. Octothinker: Mid-training incentivizes reinforcement learning scaling.

arXiv preprint arXiv:2506.20512, 2025.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. *arXiv preprint arXiv:2409.12122*, 2024.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. *arXiv preprint arXiv:2505.09388*, 2025.

Hanlin Zhu, Shibo Hao, Zhiting Hu, Jiantao Jiao, Stuart Russell, and Yuandong Tian. Reasoning by superposition: A theoretical perspective on chain of continuous thought. *arXiv preprint arXiv:2505.12514*, 2025.