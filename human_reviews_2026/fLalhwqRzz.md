# Benchmarking and Advancing Quantization-Aware Training for Reasoning Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 8, 2, 2

## Abstract
Reasoning models have excelled at complex tasks such as coding and mathematical competitions, yet their reasoning processes suffer from low inference efficiency. Quantization is a popular way to boost efficiency, but prior work shows that it causes large performance drops in these models. To address this, we comprehensively benchmark the quantization-aware training (QAT) for reasoning models. Our key findings are: (1) knowledge distillation serves as a versatile objective for reasoning models trained with either supervised fine-tuning or reinforcement-learning algorithms; (2) post-training quantization (PTQ) provides a strong initialization for QAT, improving accuracy while reducing training cost; (3) QAT with reinforcement learning is feasible and yields additional gains for the quantized model; and (4) aligning the domain of QAT training data with the PTQ calibration data further improves the performance. Building on these insights, we propose Reasoning-QAT, an optimized QAT workflow tailored to reasoning models. Empirical results show that Reasoning-QAT outperforms state-of-the-art PTQ methods across multiple LLM backbones and reasoning datasets. For instance, on the DeepSeek-R1-Qwen-Distill-1.5B model, Reasoning-QAT surpasses FlatQuant by 2.92\% under W4A4KV4 quantization and GPTQ by 4.74\% under W3G128 quantization, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Quantization can produce large drops in reasoning benchmark performance, particularly for small models. This work comprehensively benchmarks quantization-aware training (QAT) for reasoning models, offers some insights into best practices for QAT, and introduces Reasoning-QAT, an optimized QAT workflow tailored to reasoning models which outperforms state-of-the-art PTQ methods across multiple LLM backbones and reasoning datasets.

### Strengths
* The authors provide the first comprehensive benchmark of quantization-aware training, according to them, and I have no reason to doubt it. This is valuable.
* The authors provide sufficient experimental evidence to convince me that, at least under certain conditions, Reasoning-QAT improves on the baselines, and they selected strong recent baselines for comparison, good base models, good choice of benchmarks.
* The ablation Table 5 is very nice.
* I think the layout of this work is rather nice. I like the way the authors highlighted and numbered their research questions, and the esthetic effect of the colors and figure placement is good.

### Weaknesses
* The inconsistency of the experimental settings between figures and tables is vexing; it makes it harder to figure out what is really going on in this paper. For instance, Table 1 compares SFT and KD to an RTN baseline, but Figure 1 uses GPTQ as a point of comparison, but keeps the model architectures the same. There's no reason not to include both RTN and GPTQ in both Table 1 and Figure 1. In Table 4, Qwen3-4B W4A4KV4 is missing QuaRot (only shows FlatQuant), all other models show QuaRot for W4A4KV4. Why does Table 4 compare W4A4KV4 to W3G128 instead of comparing 4-bit to 4-bit or 3-bit to 3-bit? Is it because the authors are trying to control for the difficulty of the quantization task? This would benefit from a clearer explanation.
* The W4A4KV4 version of Reasoning-QAT is dependent on FlatQuant's specific architecture and transformation matrices. This is a potential confound, as FlatQuant is also a baseline.
* While the subject is worthy of some study, the authors may be somewhat overstating the centrality of QAT. While the research literature does show that quantization can cause large performance drops in reasoning models, there are a number of mitigating factors, including model size (Liu et al. (2025a) showed that the 32B model maintained 96.77% of baseline performance at 3-bit while the 1.5B model collapsed to 67.15%), PTQ algorithm (shown by the authors as well as others), quantization bit depth, and the nature of the reasoning task, with mathematical reasoning being particularly severely affected. I do not object to the authors studying a niche topic, but I think it is important for the authors to highlight somewhere (for instance, in an appendix Limitations section) that in fact, under many circumstances, PTQ with a reasonable compression scheme such as AWQ will perform just fine, even on reasoning tasks.
* (nit) Authors should define acronyms in their figures and tables captions, e.g. W4G128 in Figure 1
* (nit) There are too many figures and tables in the main paper, it's cluttered and it makes it hard to follow the main story. Table 2, for example, would be fine in the appendix as far as I can see, Table 5 tells a similar story but is much more comprehensive.
* (nit) Figures 2, 3 and 4 are too small and hard to read, again, because the authors are trying to cram everything into the main paper. It isn't all needed.
* (nit) Why is the workflow diagram on Page 7? Shouldn't that be towards the beginning in a methods paper?

### Questions
* If you used a different W4A4 PTQ method would the QAT approach transfer?
* Why not show "QAT without FlatQuant" for W4A4?

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
5

### Summary
The goal of this study is to analyze and improve Quantization-Aware Training (QAT)  for reasoning models. Authors first try to benchmark how much quantization hurts reasoning models in two settings W3G128 and W4A4. They systematically ablate various design choices for doing QAT and then propose a reasoning QAT recipe that minimizes quantization loss. The 4 recommendations involves 
* Initialization using a Post-train Quantization (PTQ) checkpoint eg GPTQ
* doing Knowledge distillation (KD) as opposed to reasoning SFT
* QAT GRPO on top yields marginal benefit too
* Matching data domain while doing PTQ calibration in first step helps

### Strengths
- Very comprehensive evaluations
- Solid experiments followed by concrete recipe
- Lots of good learning from the well formulated Research Questions like impact of PTQ caliberation data, KD > SFT etc.
- Results show meaningful reduction in quantization accuracy loss for both the quantization settings
- Paper is extremely well written and a delight to read.

### Weaknesses
- There is rich literature on logit distillation as opposed to doing reasoning SFT by distilling from stronger teacher which is not adequately referenced
- Gains from RL are unclear and it is within error for reasoning benchmarks
- avg of 3 is not enough for evals especially for AIME25 
- no experiments with larger models so unclear how results will translate.
- Authors posttraining is very math focused and not a general postraining which will target multiple domains and also chat and instruction following. Its unclear if the results will translate over as it is much more tricky to get the balance right in that scenario.

### Questions
- I would be curious if authors try on-policy KD where you first sample a response from the quantized student for a given prompt and then minimize KL For that prompt, response compared to the teacher. There is evidence in literature that it works better [1]

- dataset for RL, DAPO is probably better

[1]https://arxiv.org/abs/2306.13649

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the performance degradation of reasoning-focused Large Language Models (LLMs) under extreme low-bit quantization. The authors find that standard Post-Training Quantization (PTQ) methods are insufficient for these models. They propose Reasoning-QAT, a three-stage workflow combining PTQ-based initialization, knowledge distillation (KD), and a "cold-start" reinforcement learning (RL) phase. Experiments show this method significantly recovers performance on several reasoning benchmarks compared to PTQ baselines.

### Strengths
Addresses a Significant Problem: The work tackles the critical and timely challenge of deploying reasoning models efficiently, focusing on extreme low-bit quantization where existing methods fail.  

Strong Empirical Validation: The proposed Reasoning-QAT workflow is shown to be effective and robust, consistently outperforming strong PTQ baselines across some models and benchmarks.  

Provides Actionable Insights: The paper produces practical guidelines, such as the superiority of KD for QAT and the critical need for a "cold-start" before applying RL, which are of immediate use to practitioners.

### Weaknesses
Limited Novelty: The core weakness is the lack of algorithmic innovation. The proposed workflow is a pipeline of existing techniques (QAT, KD, RL) applied to a new domain. While the execution is effective, it does not introduce a new fundamental concept.  

Overstated Contribution: The paper's claim to be a "comprehensive benchmark" is not supported by the limited scope of the experiments, which explore only one RL algorithm and a single pipeline configuration. This overclaiming undermines the paper's credibility.  

Lack of Mechanistic Insight: The paper successfully shows what works but provides little explanation for why. For example, it does not offer a deep analysis of why SFT fails catastrophically on RL-trained models or explore the potential dual role of quantization noise as a regularizer in the RL phase.

Ambiguous Framing: The paper's narrative is caught between being a "benchmark" paper and a "novel method" paper. This dual identity weakens its focus. If the primary goal is to be a benchmark, the scope needs to be broader. If it is to propose a new method, the novelty needs to be more clearly articulated against prior art, and the analysis needs to be deeper.

### Questions
Clarification of Contribution: Is the primary contribution intended to be the Reasoning-QAT workflow or the benchmark itself? If it is a benchmark, can you justify the exclusion of other common RL algorithms and QAT methods?

Mechanism of SFT Failure: Do you have a hypothesis for the severe performance degradation when applying SFT to the RL-trained model? Could this be a form of catastrophic forgetting of the RL policy?

Role of Quantization Noise: The work treats quantization noise as purely detrimental. Did you consider that this noise might serve as a form of exploration-enhancing regularization during the RL phase, as suggested by other recent work like QERL?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This provides a systematic study of applying QAT to reasoning large language models such as DeepSeek-R1 and Qwen. It benchmarks QAT under different training objectives, initialization methods, reinforcement learning integration, and data choices. Based on their findings, they propose Reasoning-QAT, a three-stage workflow that combines PTQ initialization, knowledge distillation, and cold-start reinforcement learning. Experiments show that Reasoning-QAT consistently outperforms state-of-the-art PTQ baselines.

### Strengths
1) The paper studies an increasingly important problem of applying QAT to LLMs and provides an empirical benchmark.
2) The paper is generally well-written with clear organization.

### Weaknesses
1) The paper’s originality is limited. The proposed “Reasoning-QAT” pipeline merely stacks existing techniques (PTQ warm-start, KD, and RL fine-tuning) without introducing any new algorithmic innovation, theoretical formulation, or quantization method.
2) The authors report accuracy improvements but offer little explanation for why the observed effects occur. For example, there is no detailed investigation into the interaction between quantization noise, KD signal, and RL reward dynamics.
3) It will be better if authors can investigate more architectures besides DeepSeek/Qwen-family models.
4) The authors benchmark various QAT methods; however, the proposed Reasoning-QAT is primarily compared against PTQ baselines. Since QAT inherently benefits from retraining, such a comparison is not entirely fair and weakens the convincingness of the claimed improvements.

### Questions
1) The proposed Reasoning-QAT mainly combines PTQ initialization, KD, and RL fine-tuning. Can the authors clarify what specific interactions or dependencies exist between these stages beyond simple sequential training?
2) Can authors compare Reasoning-QAT with SOTA QAT methods?

### Soundness
2

### Presentation
3

### Contribution
2
