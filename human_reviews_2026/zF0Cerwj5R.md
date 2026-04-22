# LENS: Multi-level Evaluation of Multimodal Reasoning with Large Language Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 6, 8, 4

## Abstract
Multimodal Large Language Models (MLLMs) have achieved significant advances in integrating visual and linguistic information, yet their ability to reason about complex and real-world scenarios remains limited.
Existing benchmarks are usually constructed in a task-oriented manner, without a guarantee that different task samples come from the same data distribution. Therefore, they often fall short in evaluating the synergistic effects of lower-level perceptual capabilities on higher-order reasoning. To lift this limitation, we contribute Lens, a multi-level evaluation benchmark of multimodal reasoning with with 3.4K contemporary images and 60K+ human-authored questions covering eight tasks and 12 daily scenarios, forming three progressive task tiers, i.e., perception, understanding, and reasoning. One feature is that each image is equipped with rich annotations for all tasks. Thus, this data set intrinsically supports evaluating MLLMs to handle image-invariable prompts, from basic perception to compositional reasoning. In addition, our images have been   collected manually from social media, with $53$% published after Jan. 2025. We evaluate 15+ frontier MLLMs such as  Qwen2.5-VL,  InternVL3,  GPT-4o  and two reasoning models  QVQ-Max and Kimi-VL. Most models were released in 2025, and none of them achieve an accuracy beyond $60$% in the reasoning tasks. Furthermore, we propose the Self-Driven Multi-Expert Collaborative Framework (SMEC), a framework designed for MLLMs that simulates a panel of experts discussing and exchanging viewpoints via self-generated role-specific prompts. The experimental results confirm the existence of synergistic effects in a hierarchical task structure, where low-level tasks facilitate the reasoning of MLLMs on more complex, high-level tasks. Statistical analysis and ablation studies further demonstrate the comprehensiveness of our dataset and the superiority of our methodology. Project page:  https://github.com/Lens4MLLMs/lens. We conducted the ICCV 2025 MARS2 Multimodal Reasoning Challenge on Lens. https://mars2workshop.github.io/iccv2025/

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper introduces Lens, a benchmark for evaluating MLLMs on multi-modal reasoning and SMEC, a framework for MLLMs to solve the tasks with self-generated expert opinions. The Lens consists of eight tasks under three main categories: perception, understanding and reasoning. Images (3.4K) of the benchmark are collected manually from variety social media sources. The results of the experiments on Lens show that low-level visual tasks can affect the higher-order reasoning process. When the models apply SMEC for Scene Knowledge Inference task, the accuracy of the models increases with more iterations.

### Strengths
•	The data is manually filtered and annotated by human verifiers.

•	The process of SMEC is explained very detailly.  

•	Synergistic effects evaluation provides detailed examination of cross-tasks performance.

### Weaknesses
•	The overall writing quality of the paper could be improved. Some sections lack clarity, and certain sentences are difficult to understand.

•	While the paper highlights multi-level evaluation and integrated tasks as key contributions, these concepts are not sufficiently explained or exemplified in the paper.

•	The paper does not clearly justify the need for introducing a new benchmark. Several existing benchmarks already include some of the tasks used in the paper. It remains unclear why extending existing benchmarks with additional tasks would not be sufficient to analyze cross-task interactions. 

•	The authors did not explain why the community needs a new benchmark. Other benchmarks have already included some tasks. By adding other task-related questions to the existing benchmarks and data, the same effect can be achieved. In that case, the evaluation of the synergistic effect does not require generating a new benchmark. 

•	It remains unclear how the proposed SMEC framework distinguishes itself from prior work in terms of novelty and contribution. The related work section does not address prior efforts involving expert-based or modular task-solving frameworks. 

•	The experimental validation of SMEC appears limited, as its performance is only evaluated on a single task (Scene Knowledge Inference). To validate its effectiveness, it would be valuable to test SMEC across other tasks and other datasets.

### Questions
•	In which task is the interleaved image-text feature used? Figure 7 does not show the example. 

•	The second link on foot page opens Instagram webpage, not developer agreement. The reviewer didn’t understand the relation of the link and text in the paper. 

•	Do the authors get legal permissions from social media platforms X, Instagram, Weibo and RedNote to use their data? Although some processes, like erasing facial information, have been done for privacy concerns, the images are still copyrighted. For example, the copyright regulations of RedNote state that “RedNote's trademarks, logos, and content are protected by intellectual property laws. You may not use our intellectual property without explicit permission.” According to these cases the authors should get permission from the platforms to use their data for research purposes. If permission has been granted for data use, including documentation of this in the Appendix, is recommended. 
RedNote: https://red-note.co/terms-of-service 

•	It would be valuable to report human performance on the Lens benchmark and the performance comparison between humans and models. 

•	The Related Work section does not clearly articulate how Lens differs from existing benchmarks, particularly in terms of visual and reasoning capabilities. It would be helpful to clarify what unique contributions or challenges Lens introduces that are not addressed by prior datasets.

Suggestions:

•	The organization of the paper could be improved. For instance, placing the 'Related Work' section in the Appendix makes it harder to assess the motivation and novelty, whereas some less critical elements like Figure 2 are included in the main text

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces "Lens," a new multi-level benchmark for evaluating multimodal large language models (MLLMs). The authors argue that existing benchmarks fail to assess how foundational perceptual skills (like object detection) contribute to higher-order reasoning. Lens addresses this by providing 3.4K contemporary images, each annotated for a hierarchy of tasks progressing from perception to understanding and reasoning. The paper evaluates over 15 recent MLLMs and finds that even top models struggle with the reasoning tasks, scoring below 60%. To address this, the authors also propose SMEC, a framework where an MLLM uses self-generated prompts to simulate a "panel of experts" to collaborate on complex reasoning, showing improved performance. A key finding is the confirmation of synergistic effects, where improving low-level perceptual tasks directly facilitates better performance on high-level reasoning.

### Strengths
1. Hierarchical Benchmark Design。 The "Lens" benchmark introduces a valuable multi-level structure (Perception, Understanding, Reasoning) where all tasks are annotated on the same set of images. This is a strength as it uniquely enables the evaluation of synergistic effects。

2. Contemporary and Relevant Dataset. The dataset is built from 3.4K contemporary images manually collected from social media, with a large portion (53%) published after January 2025. This freshness is crucial for fairly evaluating modern MLLMs and mitigating the risk of data contamination from older, widely used training sets.

3. Constructive Contribution with SMEC. Beyond just identifying a problem, the paper proposes a solution with the "Self-Driven Multi-Expert Collaborative Framework" (SMEC). This framework offers an innovative, tool-free method for enhancing MLLM reasoning by simulating a panel of experts. The positive results from SMEC add significant value to the paper.

### Weaknesses
1. Limited to Static Images. As acknowledged by the authors, the benchmark is confined to static images. This scope does not capture the complexities of real-world multimodal reasoning, which often involves video, temporal sequences, audio, or long-form narrative understanding.

2. Potential Impracticality of SMEC. The SMEC framework relies on an iterative, multi-step process of generating expert prompts and synthesizing answers. This implies a significant increase in computational overhead and latency at inference time, which could make the approach impractical for real-time applications. The paper demonstrates effectiveness but does not deeply analyze this efficiency trade-off.  For example, if using several small models, how about just using a larger powerful model? Under the same compute, which method is better?

### Questions
There is a formatting problem of the citations in the paper. To cite a paper, \citep should be used instead of \cite.

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
This paper presents Lens, a large-scale, multi-level benchmark for evaluating multimodal reasoning in large vision-language models (MLLMs). 

Unlike prior task-specific datasets, Lens uses a unified image set - 3.4K high-resolution, non-commercially licensed social media image - paired with over 60K human-authored questions across 12 real-world scenarios. Each image supports eight tasks organized into three progressive tiers: perception, understanding, and reasoning, enabling a structured evaluation of how lower-level perceptual skills contribute to higher-level reasoning, while minimizing the effects of data distribution across task categories.

Comprehensive experiments on 15+ state-of-the-art MLLMs, including GPT-4o, Qwen2.5-VL, InternVL3, and reasoning MLLMs like QVQ-Max, and Kimi-VL, reveal strong interdependencies between low-level tasks, like perception and high level tasks like reasoning.

Lastly, the paper also introduces Self-Driven Multi-Expert Collaboration (SMEC), a novel framework in which an MLLM simulates a panel of specialized agents via self-generated, role-specific prompts. SMEC can dynamically refine or even redefine these roles as needed to address a given instruction without external supervision. The authors demonstrate that this approach outperforms established baselines such as majority voting, self-reflection, and direct prompting.

### Strengths
1. Hierarchical evaluation:
The benchmark’s three-tiered structure -  spanning perception, understanding, and reasoning - with a consistent data distribution across tasks, enables a more causal interpretation of how lower-level perceptual abilities influence higher-level reasoning (if any).

2. Scale of human annotations:
In terms of human-authored data, the dataset is impressively large and aligns with the current scale of contemporary multimodal benchmarks.

3. Up-to-date and well-curated dataset:
The dataset is well-defined and systematically structured, with 70% images sourced after November 2024 and 50% after January 2025. The images are high-resolution and reflect contemporary visual content, ensuring relevance to real-world scenarios beyond academic testbeds.

4. Strong empirical analysis:
One of the paper’s strongest aspects lies in its thorough analysis and experiments - particularly the dataset exploration in Fig. 4, cross-task synthetic analysis in Fig. 6, and detailed discussions in Sections 2.3, 4.3, 4.4, 4.5.

5. Interesting new SMEC framework:
The proposed Self-Driven Multi-Expert Collaboration (SMEC) is an interesting  idea where the same MLLM, prompted with diverse role-specific instructions, can collaboratively address complex reasoning tasks. It consistently outperforms direct prompting, self-reflection, and majority voting. This approach also raises an intriguing direction for future work on redundancy in self-reflection or voting setups - showing that structured role specialization may better harness the diverse knowledge and reasoning abilities embedded within a single MLLM?

6. Clarity and presentation:
The paper is clearly written and easy to follow, with well-organized appendices that are rich in detail and effectively referenced throughout the main text.

### Weaknesses
Weaknesses and Questions

1. Limited scenario diversity: While the dataset spans three main scenarios -  education, city, and home - it would benefit from broader coverage. For example, incorporating workplace, outdoor, or social-interaction settings could better capture real-world multimodal reasoning. Additionally, introducing samples grounded in logical, mathematical, or scientific domains, following the same hierarchical design principles, would strengthen the benchmark’s comprehensiveness.

2. Number and organization of tasks per tier: The benchmark currently features a relatively limited number of tasks within each tier. It is unclear whether this was an intentional design choice (e.g., grouping subtasks to facilitate downstream synergy analysis between task tiers) or simply a byproduct of human annotation diversity.

a) Did the authors design subtasks with cross-tier synergy in mind?
b) How do they envision scaling the benchmark to include additional domains (e.g., logic, mathematics, science)?

3. Clarifications on SMEC framework:
The Self-Driven Multi-Expert Collaboration (SMEC) framework is conceptually strong but would benefit from further detail and analysis. Section 3 and the appendix provide helpful descriptions, yet several aspects remain underspecified:

a) How exactly is the diversity metric defined and measured?
b)  Why did the authors choose 3500 samples subset, and how representative it is of the entire benchmark? Do results in Table 3 scale to the entire dataset?
c) What are the types of roles the model tends to generate, and which roles contribute most to performance gains? 
d) are roles domain-centric  (e.g., geometry, culture, ethics) or task-centric roles (e.g., perception vs. reasoning)? Which is better and what scenario? 
e)Could the framework be extended to multi-model setups, where different MLLMs act as specialized experts?

### Questions
Follow-up questions presented along with the weaknesses

### Soundness
3

### Presentation
4

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
The paper presents LENS, a multi-level benchmark for evaluating multimodal large language models on perception, understanding, and reasoning tasks within a unified dataset. LENS contains over 60K human-written questions built on 3.4K real-world images, enabling analysis of interdependencies among visual reasoning levels. The authors further propose SMEC, a self-driven multi-expert collaboration framework that improves reasoning performance without external tools. Experiments on 15+ open and closed-source models reveal strong correlations between perception and reasoning performance and highlight persistent challenges in complex reasoning.

### Strengths
1. LENS unifies eight well-defined tasks over shared images, allowing systematic analysis of inter-level dependencies.
2. Ensures methodological transparency and ethical compliance through detailed dataset documentation and privacy handling.

### Weaknesses
1. The boundaries between perception, understanding, and reasoning tasks are not clearly defined, and some tasks (e.g. SRC as a reasoning task) overlap in scope.
2. Lacks qualitative or fine-grained error analysis, which limits understanding of how and why models fail across different levels.
3. Font size in Figure 6 should be improved for better readability.

### Questions
1. Could the authors clarify the rationale for categorizing SRC as a reasoning task, given that its scope appears to overlap with understanding-level tasks?
2. Would it be possible for the authors to include or discuss more detailed qualitative error analyses to better illustrate common failure patterns and model limitations across different task levels?

### Soundness
2

### Presentation
3

### Contribution
2
