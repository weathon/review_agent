# ReGenesis: LLMs can Grow into Reasoning Generalists via Self-Improvement

- Decision: Accept (Oral)
- Scores: 8, 6, 8, 8

## Abstract
Post-training Large Language Models (LLMs) with explicit reasoning trajectories can enhance their reasoning abilities. However, acquiring such high-quality trajectory data typically demands meticulous supervision from humans or superior models, which can be either expensive or license-constrained. In this paper, we explore how far an LLM can improve its reasoning by self-synthesizing reasoning paths as training data without any additional supervision. Existing self-synthesizing methods, such as STaR, suffer from poor generalization to out-of-domain (OOD) reasoning tasks. We hypothesize it is due to that their self-synthesized reasoning paths are too task-specific, lacking general task-agnostic reasoning guidance. To address this, we propose **Reasoning Generalist via Self-Improvement (ReGenesis)**, a method to *self-synthesize reasoning paths as post-training data by progressing from abstract to concrete*. More specifically, ReGenesis self-synthesizes reasoning paths by converting general reasoning guidelines into task-specific ones, generating reasoning structures, and subsequently transforming these structures into reasoning paths, without the need for human-designed task-specific examples used in existing methods. We show that ReGenesis achieves superior performance on all in-domain and OOD settings tested compared to existing methods. For six OOD tasks specifically, while previous methods exhibited an average performance decrease of approximately 4.6% after post training, ReGenesis delivers around 6.1% performance improvement. We also conduct an in-depth analysis of our framework and show ReGenesis is effective across various language models and design choices.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes ReGenesis which applies task-agnostic reasoning guidance to facilitate reasoning path synthesis, achieving great performance in a self-improvement manner. Experiments on in-domain and out-of-domain tasks show the strong generalization ability of the proposed data-synthesizing and training framework. Further analysis indicates the importance of each component in ReGenesis and demonstrates the robustness across different base models.

### Strengths
- The framework of ReGenesis is clearly shown, especially in Figure 1.
- The generalization ability of the proposed method is impressive, even in a self-improvement manner.
- Empirical studies in this paper are comprehensive.
- The paper is well-written

### Weaknesses
1. Missing some baselines: How will ReGenesis perform when compared with few-shot CoT prompting (w/o FT) on in-domain and out-of-domain tasks, and how will it perform compared with GPT-3.5/4/4o? Also, if Mistral-7B is fine-tuned on gold trajectories, which means removing the self-improvement setting, how large will the performance gap be between this and ReGenesis?
2. Why is A+S not included in Table 4?

3. StrategyQA and OpenbookQA are not challenging for LLMs nowadays. What would happen if training is done on StrategyQA and testing is performed on CommonsenseQA, a more challenging out-of-domain dataset? Furthermore, ASDIV, SVAMP, and AQUA are quite simple for GSM8K. If training is conducted on these datasets and testing is done on GSM8K, what would happen?

### Questions
1. It appears that the generalization of ReGenesis mainly stems from task-agnostic guidance. However, during actual model training, only specific trajectories are provided, without including this guidance. So, where does the model's generalization come from? Fundamentally, task-agnostic guidance simply ensures that the model can synthesize more correct trajectories. These additional trajectories enhance the model's capabilities. If ReGenesis, STaR, and LMSI were trained using the same number of synthetic trajectories, would there be differences in their effectiveness?
2. Could the authors provide a detailed analysis and explanation as to why giving hints to models when generating reasoning paths might have adverse effects?

### Soundness
3

### Presentation
4

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
The paper introduces a novel framework called ReGenesis. It aims to improve the reasoning capabilities of large language models (LLMs) without relying on external supervision. Unlike prior self-synthesizing techniques that generate task-specific reasoning paths, ReGenesis creates general task-agnostic reasoning paths, promoting broader applicability. ReGenesis begins with abstract general guidelines, adapts them to specific tasks, and generates reasoning paths, refining the model's capability on both in-domain and out-of-domain (OOD) tasks. Experiments show that ReGenesis significantly enhances LLM performance across various domains, demonstrating its effectiveness in developing reasoning generalists capable of transferring knowledge to new tasks.

### Strengths
ReGenesis introduces a structured approach for LLM self-improvement by creating task-agnostic, generalizable reasoning paths, which is a substantial departure from the task-specific paths in prior methods.

ReGenesis shows superior performance in OOD tasks, addressing a major limitation of existing self-synthesizing methods. The model's flexibility across multiple reasoning domains suggests broader applicability.

The authors conducted thorough evaluations across various datasets, including mathematics, logical reasoning, and common sense. This diversity reinforces the model’s effectiveness and robustness.

Detailed ablation studies on various components, filtering methods, and language models strengthen the framework's design choices, validating each element's contribution to ReGenesis’s success.

### Weaknesses
1. I believe the paper presents a valuable approach by establishing task-agnostic general reasoning guidelines. However, I am curious whether employing a more advanced model, such as GPT-4, to formulate these guidelines might yield better results than having them generated by the 7B model itself.

2. I think many benchmarks in Table 3 might not fully qualify as out-of-domain (OOD) tasks. Previous work has often used GSM8K’s training set as demonstrations for datasets like ASDIV and SVAMP, which lack dedicated training data, in in-context learning (ICL) setups. In my understanding, true OOD tasks should go beyond simply math, commonsense, or logical reasoning to include other tasks like code generation, summarization, etc.

3. I suggest adding an “average” column to both Table 2 and Table 3 for a clearer overview of the results.

### Questions
1. Many tables look weird, such as Table 20 and Table 21, which have no borders.

2. Please use vector graphics as much as possible.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a method named ReGenesis to enhance the reasoning abilities of LLMs by allowing them to self-synthesize reasoning paths without human intervention or task-specific examples. Unlike prior methods that struggle with OOD generalization due to reliance on specific task examples, ReGenesis employs a progressive approach that builds from abstract reasoning guidelines to task-specific structures and paths. This approach encourages generalization across tasks, improving performance both in-domain and on OOD reasoning tasks.

Key contributions include:

1. A novel framework that allows LLMs to self-improve by generating diverse, task-agnostic reasoning paths.
2. Demonstrated effectiveness of ReGenesis on multiple datasets, with notable OOD performance improvements over existing methods.
3. An extensive analysis showing ReGenesis's adaptability across various LLMs, design configurations, and reasoning tasks.

### Strengths
1. The work provides a fresh insight into a key limitation in reasoning synthesis for LLMs, observing that existing self-synthesized reasoning methods often experience substantial performance drops in out-of-domain (OOD) settings.
2. The authors propose a novel abstract-to-concrete synthesis route, progressively transitioning from general reasoning guidelines to task-specific reasoning paths.
3. This paper presents comprehensive experiment, including extensive comparisons with competitive baselines like LMSI and STaR.

### Weaknesses
1. Although the paper includes a comprehensive ablation study (Tables 4 and 5) to examine the components of the ReGenesis framework, the fundamental reasons behind the performance improvements remain ambiguous. The framework is designed to construct reasoning solutions that incorporate *general task-agnostic reasoning guidance*. However, it is unclear how the ReGenesis-generated reasoning chains retain this generalizability across tasks. According to Table 25, the generated reasoning chains appear more specific and detailed, including elements like *Reasoning Structures*. While this added detail likely contributes to performance gains, the paper lacks clear evidence on how such specific structures support generalization across diverse tasks rather than being overfitted to in-domain scenarios.


2.  While the paper presents ReGenesis as highly effective, there is limited discussion on cases where the framework may fail or yield lower-quality reasoning paths. Especially in cases of OOD tasks, which may vary significantly in structure or complexity, it is important to understand where and when ReGenesis might underperform or produce suboptimal reasoning paths.  For instance, in Table 3, when the model was fine-tuned on NumGLUE, it exhibited a performance drop on ASDIV, an OOD math reasoning task. This raises questions about the potential limitations of ReGenesis when the source and target tasks have significant differences in structure or reasoning requirements.

### Questions
1. Could the authors clarify whether the process of editing and expanding seed guidelines remains consistent across different models used in the experiments? Additionally, how sensitive is ReGenesis to the selection of inappropriate guidelines? For instance, if an unsuitable guideline is chosen, could this negatively impact the reliability of the final solution, particularly in OOD scenarios?
2. In Table 2, ReGenesis demonstrates a substantial performance improvement on GSM8K, surpassing other methods by over 1%—a margin comparable to the performance increase from LLaMA3.1 8B to LLaMA3.1 70B. Such a substantial gain warrants further investigation. Could the authors provide a deeper analysis of the underlying factors contributing to these performance gains?
3. The experiments in Table 3 are conducted by fine-tuning on one of five in-domain datasets at a time. Would mixed training across multiple in-domain datasets improve the generalization performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
ReGenesis enhances LLMs' reasoning by self-synthesizing generalizable reasoning paths without human supervision, outperforming prior methods on in-domain and out-of-domain tasks. It improves OOD performance by 6.1%, while existing methods show a decline, proving its effectiveness across models and configurations.

### Strengths
- The problem of building a reasoning generalist is important, compared to other works focusing on dataset-specific improvements in LLMs.

- Comprehensive experiments highlight ReGenesis’s superior performance in both in-domain and out-of-domain tasks.

### Weaknesses
- The major concern is that there is a mismatch between the motivation and the proposed method. The paper pointed out that the bottleneck of self-synthesizing methods is that they suffer from poor generalization to out-of-domain (OOD), and hypothesized that it is because "their self-synthesized reasoning paths are too task-specific, lacking general task-agnostic reasoning guidance". However, even though they proposed a multi-level prompting method starting from general guidelines, in the end, they still used task-specific reasoning paths to train the model. While the authors suggest their method can enhance the diversity of reasoning chains, which helps with OOD performance, there is no clear analysis.

- The core challenge of creating a reasoning generalist may lie more in synthesizing diverse reasoning problems rather than just generating diverse reasoning chains. ReGenesis still relies on access to a dataset of questions and even ground-truth answers, limiting its potential for fully autonomous reasoning and diminishing its broader impact.

### Questions
- Can you compare the size of synthetic data you use for all methods? You mentioned that 32 reasoning paths are sampled for STaR, but how many of them are used for training after filtering?

- Why is there such a large difference between STaR and the proposed method on GSM8k? (46.3 vs 63.6)

### Soundness
3

### Presentation
3

### Contribution
3
