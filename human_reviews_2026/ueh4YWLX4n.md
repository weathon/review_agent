# AutoJudger: An Agent-Driven Framework for Efficient Benchmarking of MLLMs

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Evaluating multimodal large language models (MLLMs) is increasingly expensive, as the growing size and cross-modality complexity of benchmarks demand significant scoring efforts. To tackle with this difficulty, we introduce ***AutoJudger***, an agent-driven framework for efficient and adaptive benchmarking of MLLMs that tackles this escalating cost. AutoJudger employs the Item Response Theory (IRT) to estimate the question difficulty and an autonomous evaluation agent to dynamically select the most informative test questions based on the model’s real-time performance. Specifically, AutoJudger incorporates two pivotal components: *a semantic-aware retrieval mechanism* to ensure that selected questions cover diverse and challenging scenarios across both vision and language modalities, and *a dynamic memory* that maintains contextual statistics of previously evaluated questions to guide coherent and globally informed question selection throughout the evaluation process.
Extensive experiments on four representative multimodal benchmarks demonstrate that our adaptive framework dramatically reduces evaluation expenses, i.e. AutoJudger uses only 4\% of the data and 10\% computational cost (for evaluating 7B model) to achieve over 90\% ranking accuracy with the full-benchmark evaluation results on MMT-Bench.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes an efficient benchmarking method for multimodal large language models (MLLMs), which leverages Item Response Theory (IRT) to estimate question difficulty and employs an autonomous evaluation agent to dynamically select the most informative test questions based on the model’s real-time performance.

### Strengths
1. The work is comprehensive, supported by extensive experiments validating the effectiveness of the proposed approach.
2. The paper focuses on efficient MLLM benchmarking, which is an important and practical problem for the community.

### Weaknesses
1. While the proposed framework is interesting, the paper lacks a deeper analysis of the fundamental factors contributing to its effectiveness. Beyond the ablation study, it remains unclear which core design choices are primarily responsible for the improvement. This makes it difficult to disentangle whether the observed benefits stem from `essential ideas` or the `complex agentic workflow`. It would be helpful to include a more prototype-level or simplified implementation as an additional baseline, to better isolate and discuss the key contributing factors, while reducing the emphasis on the agent workflow itself.
2. The comparison could be more comprehensive. If prior studies on efficient MLLM benchmarking exist, including them would make the contribution clearer and the improvement more convincing.

### Questions
1. I am curious whether there have been previous attempts or existing methods for efficient benchmarking of MLLMs？
2. The baselines in the paper appear relatively simple, while the proposed method introduces a much higher level of complexity. Could the authors clarify the necessity of such a complex design? Is it possible that a simpler approach might achieve similar effectiveness?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents AutoJudger, an agent-driven framework designed to address the escalating computational cost of benchmarking MLLMs. The core methodology leverages Item Response Theory (IRT) to pre-estimate question difficulty. This information is then used by an autonomous agent to adaptively select a subset of informative questions based on the real-time performance of the model under evaluation. The framework incorporates a semantic-aware retrieval mechanism to ensure question diversity and a dynamic memory module to maintain contextual statistics, guiding a coherent and globally informed selection process.
The authors report that this adaptive approach significantly reduces evaluation costs, achieving over 90% ranking consistency with the full-benchmark results while using as little as 4% of the data on benchmarks like MMT-Bench. The work's primary contribution is a shift from static, fixed-subset evaluation to a dynamic, model-specific testing protocol that aims to balance efficiency with reliability.

### Strengths
- The paper's primary originality lies in the principled reframing of MLLM benchmarking from static subset sampling to a dynamic, agent-driven "interview" process. This is a significant conceptual shift. The creative synthesis of established concepts from different fields—Item Response Theory (IRT) from psychometrics, semantic-aware retrieval, and a dynamic memory module—into a cohesive framework to solve this problem is highly novel and insightful.

- The paper is exceptionally well-written and clearly structured. The core "interviewer" agent paradigm is an intuitive and effective metaphor that guides the reader through the complex process. Figure 2 provides an excellent, comprehensive overview of the entire dynamic evaluation loop, making the complex interactions between components easy to understand and follow.

### Weaknesses
- The framework's effectiveness is heavily reliant on the capability of the "interviewer" agent (Qwen2.5-VL-7B). This conflates the performance of the framework's mechanics (IRT, memory) with the reasoning power of a specific, strong MLLM. A weaker agent might make suboptimal choices, and the agent's inherent biases could lead to systematically skewed question selections.

- The premise of using IRT requires pre-estimating question difficulties, which the authors accomplish by collecting responses from 60 offline models on the full benchmarks. This represents a substantial, and for many researchers or new benchmarks, prohibitive, upfront computational barrier. While amortized, it limits the framework's practical applicability to new or evolving datasets.

### Questions
Plz answer my concerns in the weakness section

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces AutoJudger, an agent-driven framework designed to address the escalating cost of evaluating MLLMs. The core method uses Item Response Theory to pre-calculate a difficulty score for all benchmark questions and to estimate a model's ability in real-time during the evaluation. An MLLM-based agent acts as an "interviewer," adaptively selecting the most informative questions by analyzing the model's current ability, the difficulty of the questions, and a dynamic memory that tracks statistical coverage across different topics. The framework was evaluated on four representative multimodal benchmarks, including MMMU, MMT-Bench, and SEEDBench, demonstrating that it can maintain over 90% ranking accuracy with the full benchmark results while using as little as ~5% of the data.

### Strengths
- The paper proposes a novel, agent-driven framework to address the practical problem of expensive MLLM evaluation costs. The AutoJudger framework is built on a principled and well-suited foundation, adapting Item Response Theory to define the problem difficulty and dynamically estimate the model's ability.

- The design and implementation of the framework are solid and reasonable. The core components—real-time ability estimation, semantic-aware retrieval, an agent-based selection module, and a statistical dynamic memory—work together logically to achieve efficient and adaptive evaluation.

- The method demonstrates good empirical results, showing it can maintain high ranking accuracy (e.g., >90% on MMT-Bench) while using a small fraction of the original data (e.g., 4%). The robustness of the framework is well-supported by ablation studies, including testing different agent backbones.

### Weaknesses
**The most important issue:** The primary metric, "Ranking Accuracy" (Sec 4.1), seems to be a significant limitation in the evaluation. This ordinal metric only tells if the relative order of models is preserved and does not measure if the cardinal score gaps are retained. A framework that shrinks a 20-point performance gap (on the full benchmark) to a 1-point gap would still achieve 100% ranking accuracy, but users won't trust such a framework even if it's much more efficient and cheap.

Also, the framework's practicality is questionable when evaluating smaller models. The agent itself is a medium-sized MLLM (e.g., 7B), introducing a substantial, fixed computational overhead per step. As shown in Table 4, this overhead is proportionally massive (e.g., 4.85x per step for a 3B model), which undermines the goal of cost-saving for this common use case (Consider that many researchers in academia do experiments with small models).

### Questions
The paper takes a valuable direction and proposes an interesting automatic evaluation framework for MLLMs. However, the evaluation only discusses how the ranking can be preserved without measuring whether the score gap can also be retained under reduced evaluation costs, making the conclusion less convincing.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an autonomous MLLM evaluation pipeline that dynamically selects the most informative questions from the question pool, with the aid of agents. In the core of the pipeline, it measures the difficulty (or score) of a question using Item Response Theory based on the model's response, subsequently it selects a small set of questions from the entire benchmarks. Finally, the selected question subset is viewed as a budget-friendly evaluation benchmark proxy. 

Trimming down the evaluation benchmark to lower the cost is indeed an interesting idea. Nevertheless, the cost saved by trimmed benchmarks seems trivial compared with the massive cost of training or post-training.

### Strengths
1. Nicely presented paper, easy to follow, and well organized. The problem is interesting; the evaluation cost of MLLMs indeed often gets ignored. 
2. The AutoJudger framework seems reasonable to me, and the idea of utilizing an autonomous framework to mine difficult questions for MLLM evaluation is worth promoting, especially since some recent works have pointed out that some benchmarks may suffer from data leakage or lower-quality question issues.

### Weaknesses
1. The need to trim down benchmark size **needs more solid evidence.** For example, the CO2 emission on a full benchmark, or a more straightforward measure such as GPU rental cost? 
   * I know section 4.4 has already mentioned the cost, but it will be nice to see a number in $. 
   * As mentioned in the summary, the evaluation cost can be trivial compared with the massive cost of training (including post-training/fine-tuning) of a model. 
2. It is good to see that the author provides how different methods perform under different compression ratios; however, **it is unclear why the compression ratio is capped to 5%.** For example, 10% or 15% compression is a good step up compared with utilizing the full benchmark already. 
   * I consider this a weakness because I doubt that randomly selecting 10% or maybe 15% questions from a benchmark can yield a satisfying result already. I strongly recommend that the author clarify why it is capped at 5% and how it behaves if the number is set to a higher value, say 15% or 10%. 
3. The IRT-based difficulty assessment requires offline evaluation results of different models. I wonder if this will compromise the contribution of the proposed method. In fact, I respectfully **disagree** with the claim that the cost of question difficulty estimation is **negligible** simply because it is precomputed (Section H in the appendix), especially since it involves evaluating 60 models on full benchmarks.
   * Also, will the evaluation result on outdated models be useful for the difficulty assessment?

### Questions
see weakness

### Soundness
2

### Presentation
3

### Contribution
1
