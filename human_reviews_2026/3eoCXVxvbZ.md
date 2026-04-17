# AU-Harness: An Open-Source Toolkit for Holistic Evaluation of AudioLLMs

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 0

## Abstract
Large Audio Language Models (LALMs) are rapidly advancing, but evaluating them remains challenging due to inefficient toolkits that limit fair comparison and systematic assessment. Current frameworks suffer from three critical issues: slow processing that bottlenecks large-scale studies, inconsistent prompting that hurts reproducibility, and narrow task coverage that misses important audio reasoning capabilities. We introduce **AU-Harness**, an efficient and comprehensive evaluation framework for LALMs. Our system achieves a speedup of up to 127% over existing toolkits through optimized batch processing and parallel execution, enabling large-scale evaluations previously impractical. We provide standardized prompting protocols and flexible configurations for fair model comparison across diverse scenarios. Additionally, we introduce two new evaluation categories: LLM-Adaptive Diarization for temporal audio understanding and Spoken Language Reasoning for complex audio-based cognitive tasks. Through evaluation across 380+ tasks, we reveal significant gaps in current LALMs, particularly in temporal understanding and complex spoken language reasoning tasks. Our findings also highlight a lack of standardization in modalities of user-provided instructions existent across audio benchmarks, which can lead to performance differences of up to 7.1 absolute points on challenging complex instruction following downstream tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces an open-source framework designed to address critical challenges in the evaluation of Large Audio Language Models (LALMs). The authors identify three main limitations with existing evaluation toolkits: throughput, reproducibility, and task scope.

To overcome these issues, AU-Harness is engineered for efficiency, achieving a significant speedup over current systems. The framework also standardizes evaluation protocols by providing consistent prompting to ensure fair and reproducible comparisons between different LALMs. Furthermore, it expands the evaluative scope by incorporating a more diverse and comprehensive set of 27 audio-based reasoning tasks, aiming for a more holistic assessment of model capabilities. 

The paper presents empirical results from evaluating 3 different families of LALMs, including Qwen2.5-Omni-7B, Phi-4-Multimodal, and Voxtral-Mini-3B, to demonstrate the framework's effectiveness and to highlight performance differences between those models.

### Strengths
1.	The motivation behind the paper is strong.
2.	Improved runtime efficiency strengthens the evaluations of LALMs.
3.	The framework structure is complete and ideally standardized, which allows other researchers to follow and build upon this work easily.

### Weaknesses
1.	The “380+ tasks” claim in the abstract is confusing. For instance, as mentioned in this paper, Dynamic-SUPERB Phase-2 [1] provides up to 180 tasks, where they do provide a list of them in the appendix. In contrast, this paper lists datasets and task categories but does not provide a comparable breakdown explaining how the ‘380+’ figure is derived.
2.	Limited numbers of LALMs, as only 3 open-source models are provided, while potentially omitting other viable options. 
3.	See Questions.

**Reference**

[1] https://arxiv.org/abs/2411.05361

### Questions
1.	What experimental conditions, such as hardware specifications and software versions, were used to obtain the efficiency claims? A detailed breakdown is necessary to validate them.
2.	How were the specific tasks chosen for inclusion in the overall evaluation suite? Are these tasks collectively sufficient to ensure a holistic judgment of a LALM's capabilities?
3.	Is it sufficient to evaluate the reasoning tasks by just converting text instructions into audio context using TTS? In other words, does this experimental setup primarily test the model's transcription (ASR) capability rather than its true, deeper reasoning capacity?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces AU-Harness, a framework for evaluating LALMs. The proposed framework provides speedup compared with existing toolkits. The paper also provides standardized prompting protocols and flexible configurations, along with two new tasks for comprehensive evaluation.

### Strengths
- The speedup provides better efficiency for researchers in this area.

### Weaknesses
- The paper lacks sufficient novelty. The main novelty is the two newly proposed tasks, while other aspects are more like engineering optimization instead of scientific innovation. 
- The definition of "exisiting toolkit" is not appropriate. It seems like the so-called toolkit refers to the official implementation of the benchmark papers. However, it should be noted that the implementation of these papers should not be viewed as "toolkit" but "sample codes" instead. Therefore, for benchmark developers, it is natural that they do not focus on efficiency optimization. Therefore, the authors' statement that takes the efficiency as the weakness of exisiting works is not appropriate.
- While the paper argues that AU-Harness is holistic, the evaluation scope is quite limited. and the paper misses many important references.
- The paper only compares with three exisiting benchmarks, which is insufficient.

### Questions
I have listed my concerns in the weakness section. I think the paper is not novel enough, and it also misses many references to relevant prior work.

The evaluation landscape in this paper is quite limited. For example, while the paper evaluates reasoning capability, the evaluation only focuses on the content-based reasoning, which only requires the semantic understanding for reasoning. However, the acoustic-based reasoning that combines acoustic information for reasoning is also essential that distinguishes LALMs from text-based LLMs[1]. Therefore, I believe the "holistic" in the paper title is overclaimed.

[1] Yang et al., "Towards Holistic Evaluation of Large Audio-Language Models: A Comprehensive Survey", EMNLP 2025

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper introduces a new toolkit for benchmarking speech-aware language models (speech-in, text-out). Compared to existing toolkits, it emphasizes faster benchmarking speed, highly customizable configuration, and two additional tasks: speaker diarization and operational tasks. For the latter, the authors refer to spoken tasks that involve function calls, code generation, and instruction following.

Overall, there is no new problem formulation, no research question being addressed, and no key findings—only engineering work (though they do provide the code). Despite the engineering effort, the paper does not propose any novel technical contribution that enables faster benchmarking. To the best of my knowledge, the authors rely on existing infrastructure (e.g., vLLM) to boost the speed, rather than proposing a new parallelization tool or paradigm that would contribute to the broader field. The newly proposed tasks are not entirely novel either: speaker diarization has already been discussed in DynamicSUPERB, and instruction following in SpeechIFeval. The task coverage also does not match that of DynamicSUPERB. The only new contribution is the dataset for function calls and code generation, but this contribution is minor and would be more appropriate for a dedicated paper on the agentic capabilities of speech language models.

Despite ICLR welcoming benchmark papers, I believe this refers to works that formulate a new problem or direction, introduce novel datasets and metrics, and analyze existing models to reveal key insights, rather than purely software engineering papers. As a result, it is difficult to identify sufficient scientific contribution in this work. I suggest a reject.

### Strengths
- The benchmarking speed is really faster than existing benchmarks

### Weaknesses
- This is a software engineering work built upon existing frameworks, tools, and knowledge. It is known that applying the latest parallelization tools can improve speed; this work is simply the first to implement it. The results are good but expected.
- Two of the four newly added tasks are not novel (speaker diarization and instruction following). The former is discussed in DynamicSUPERB, and the latter in SpeechIFeval and VoiceBench.
- I do not consider software configurability a scientific contribution.
- The benchmarking speed is indeed faster, but the engineering contribution alone does not seem significant enough.

### Questions
No

### Soundness
2

### Presentation
2

### Contribution
1
