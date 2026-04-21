# ChipVQA: Benchmarking Visual Language Models for Chip Design

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3

## Abstract
Large-language models (LLMs) have exhibited great potential to as-
sist chip designs and analysis. Recent research and efforts are mainly
focusing on text-based tasks including general QA, debugging, design
tool scripting, and so on. However, chip design and implementa-
tion workflow usually require a visual understanding of diagrams,
flow charts, graphs, schematics, waveforms, etc, which demands
the development of multi-modality foundation models. In this paper, we propose ChipVQA, a benchmark designed to evaluate the
capability of visual language models for chip design. ChipVQA includes 142 carefully designed and collected VQA questions covering five chip design disciplines: Digital Design, Analog Design, Architecture, Physical Design and Semiconductor Manufacturing. Un-
like existing VQA benchmarks, ChipVQA questions are carefully
designed by chip design experts and require in-depth domain knowledge and reasoning to solve. We conduct comprehensive evaluations
on both open-source and proprietary multi-modal models that are
greatly challenged by the benchmark suit. ChipVQA examples are available at
https://anonymous.4open.science/r/chipvqa-2079/.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes a comprehensive benchmark including 142 VQA questions covering five chip design disciplines: Digital Design, Analog Design, Architecture, Physical Design and Semiconductor Manufacturing. It is the first benchmark suite in the field of multi-modal chip design knowledge. Moreover, some experiments are implemented to test this benchmark, and it is proved that the benchmark is challenging for the capabilities of current VLMs.

### Strengths
ChipVQA is the first work which constructs a multiple fields and multi-modal benchmark in chip design and tests the performance of mainstream VLMs on the benchmark. Results reveal that this benchmark is challenging for current capabilities of VLMs.

### Weaknesses
Although the authors emphasize the quality of their benchmark and say that these questions are developed by “seasoned chip design experts, each with over ten years of industry experience” and majority of some previous benchmark primarily cover content up to the level of undergraduate engineering courses, as a reviewer who majored in Integrated Circuit and System during undergraduate studies, I think the most questions of their benchmark showed in the anonymous repositories only have the level of undergraduate courses in majors related to integrated circuit. Moreover, most of the questions are common for students in majors related to Integrated Circuit. The authors only collect them together from textbooks, course exams, manuscripts and so on, and some of the images of the questions are rough, hastily written manuscripts which make it even difficult for VLMs to recognize the content.

Apart from the benchmark quality, there’s not much novelty in this work, because authors mainly test some mainstream VLMs on this benchmark. In the part where the authors demonstrate their discovery, the third, the forth, and the fifth are obvious. And the first and the second also doesn’t have much value, considering that some image paired with questions are just hastily written manuscripts.

### Questions
1.Whether the benchmark has some questions, which are not included by the anonymous repo, with higher quality?
2.Whether the authors try to redraw the manuscripts of analog circuits by software like Visio to test the performance of VLM?
3.Whether the authors test senior undergraduate students in majors related to Integrated Circuit on this benchmark as a comparison? I think only after this, they can say that majority of some previous benchmark primarily cover content up to the level of undergraduate engineering courses and their benchmark is an exception.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The authors proposed a new benchmark, ChipVQA, to evaluate the existing VLMs ability to understand and reason in chip design, which is a specific and important area. ChipVQA is considerably challenging even for the most advanced VLM model, GPT-4o. Meanwhile, the collected QAs span various chip design areas from abstract architecture design to realistic semiconductor manufacturing.

### Strengths
1. The authors proposed a new benchmark, ChipVQA, to evaluate the existing VLMs ability to understand and reason in chip design, which is a specific and important research area.
2. ChipVQA is considerably challenging even for the most advanced VLM model, GPT-4o.
3. The collected QAs span various chip design areas from abstract architecture design to realistic semiconductor manufacturing.

### Weaknesses
1. The limited number of QAs in this benchmark fails to adequately represent the chip design field and does not provide sufficient potential to support VLM development.
2. The experiments are insufficient to demonstrate the superiority of this benchmark.

### Questions
I have some concerns about this benchmark.
1. The number of QAs in this benchmark is only 146, which is too small to represent the five major class problems in the chip design. The small scale will make the evaluation results difficult to reflect the ability of VLM. The rare QAs belonging to specific small problems may lead to results that are not robust.
2. The small benchmark could only be used for fast evaluation, which cannot be used to improve the chip design ability of VLM. Although the authors’ claim “demonstrates promising potential to enhance LLM/VLM problem-solving capabilities with minimal training overhead” in the conclusion, the small number of QAs in the benchmark makes this work not promising enough.
3. The experiments cannot sufficiently support the effectiveness of this benchmark. The authors said “unlike existing benchmark efforts targeting at most undergraduate level engineering question Yue et al. (2024),”. However, the authors never compare the performance of VLM on the other benchmarks, such as Yue et al. (2024), with ChipVQA to justify its superiority.
4. Experiment 4.1 seems to verify a common conclusion, that more knowledge will help VLM understand and reason. However, the low pass rate in Table 3 actually highlights the hardness of ChipVQA. Experiment 4.2 also seems to verify a common conclusion, that the higher image resolution will help improve the answer quality of VLM.
5. The quotation marks are used in mistake in multiple places, such as “”Derive” on page 5, etc.

In conclusion, I think a large-scale high-quality VLM benchmark for chip design will be more attractive to the researchers. The authors could enlarge the limited number of QAs in the benchmark and provide more sufficient experiments and insights of applying VLM in chip design problems.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a benchmark in terms of chip design. Such benchmark is characterized as VQA tasks, parallel to existing text-based tasks.

### Strengths
- Proposing a multi-modality VLM benchmark is meaningful and has basic value, which can greatly help the communities of chip design and LLMs.
- The benchmark is notably challenging due to the diverse types of visual content, allowing significant room for research and improvement.
- This paper offers five distinct data categories in data collection and introduce their details, enhancing the diversity of the benchmark.

### Weaknesses
- ChipVQA contains only a total of 142 samples, which limits its scalability and utility. Specifically, there is respectively only one sample each for certain categories (flow, equations, neural nets), reducing the benchmark's effectiveness for evaluating specific content types.
- As a benchmark, this paper lacks a further dev/test splitting, which restricts the flexibility for developers to conduct training and fine-tuning.
- While the paper describes the benchmark as multi-modal, it only incorporates text and images. Although there are various types of visual samples, such as diagrams and graphs, all are treated as images in the experiments.
- This paper lacks the discussion of an alternative aspect of VLMs, such as CLIP [1], which emphasize visual capabilities over language components.
- Some typos: A missing space after "ChipVQA" in line 175, and a labeling error where "Figure 1" should read "Figure 3."
- The authors did not follow the citation instruction, as citations should be in parenthesis when the authors or the publication are not included in the sentence.

[1] Radford, Alec, et al. "Learning transferable visual models from natural language supervision." International conference on machine learning. PMLR, 2021.

### Questions
- What do the abbreviations ‘MC’ and ‘SA’ mean in Table 1?

### Soundness
2

### Presentation
2

### Contribution
2
