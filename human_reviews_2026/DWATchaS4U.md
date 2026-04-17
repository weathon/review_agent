# PSBench: Editing Image via  GUI Agents in Photoshop

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Photoshop is a powerful and widely used professional software for image editing, design, and creative production. Its complex multi-level menu structure, extensive set of graphical processing tools, and reliance on precise manipulations make automated operation and agent interaction particularly challenging. Despite recent progress in GUI agents, existing datasets and methods are primarily designed for web-based interfaces and short-horizon, low-complexity tasks in operating systems, falling short in addressing the fine-grained control, multi-step decision-making, and semantic understanding required in professional graphic software. To this end, we propose the first benchmark specifically tailored for image editing in Adobe Photoshop environment, with a particular focus on its core principle of non-destructive editing through layers. The benchmark consists of  600 human-annotated tasks, spanning three difficulty levels. Easy and medium tasks are distilled from official Photoshop tutorials, capturing the most common basics.
Hard tasks are directly collected from the most popular Photoshop tutorials in Youtube, ensuring both challenge and real-world relevance. Task categories cover fundamental functionalities such as canvas adjustment, layer manipulation, and filter application, each accompanied by dedicated fine-grained evaluation metrics. Through various experiments in PSBench, we find that current leading MLLMs, like Qwen2.5-VL, GPT-5 and Gemini-2.5-Pro, exhibit generally low task success rates but can demonstrate remarkable planning ability. Further via a human-in-loop experiment, we find that MLLMs can serve as highly effective Photoshop assistants, substantially boosting novice users’ task success rates while dramatically reducing their operation time.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces PSBench, a benchmark for evaluating the Photoshop capabilities of GUI agents. The benchmark contains 600 samples categorized into easy, medium, and hard modes, accompanied by expert-annotated editing trajectories. Evaluation on SOTA MLLMs shows that most models struggle even with tasks in the easy category. The authors also conduct a human-in-the-loop study, showing that MLLMs can help novice users complete the tasks more accurately and efficiently.

### Strengths
- The proposed benchmark is novel and has practical value, as Photoshop (or, more generally, controllable image editing) represents a common real-world task.

- The benchmark is highly challenging (SOTA models perform poorly even in the easy category) leaving substantial room for future model development.

- PSBench is evaluated on a wide range of SOTA models, and the authors provide a solid error analysis of model failures.

### Weaknesses
- Limited details are provided about the 600 individual tasks. The paper mentions 16 categories (Appendix E), but per-category statistics are missing. The authors claim that the tasks are diverse, yet no evidence supports this. Showing the distribution of task counts across categories and difficulty levels (easy / medium / hard) would strengthen this claim.

- The evaluation functions lack human validation for GPT-4o as a vision-language judge. Since VLMs can hallucinate, it is unclear how trustworthy these evaluations are. Alignment with human evaluation (even on a subset) is needed.

- The proposed metric NDEC, which compares the agent trajectory to the gold trajectory, seems oversimplified. While comparing two trajectories is difficult, a six-criterion checklist is too coarse-grained. The evaluation protocol is also vague, as it relies on an expert’s subjective judgment (e.g., whether a feature like Smart Objects is “used properly”). A clearer, more fine-grained evaluation protocol would improve reproducibility and soundness.

### Questions
**Question**

- In Table 2, the time horizon is defined as “the number of UI actions per task, reported as the average operation length for hard tasks.”
Why not report the average operation length across all tasks (easy, medium, and hard)? The current number might be inflated, making the comparison unfair.

**Comments on Writing**

- Line 292: “NDEC metric” is mentioned, but its formal definition appears only at line 303.

- Lines 399–404: “All models achieve overall NDEC scores above 70%, indicating that their generated action sequences not only accomplish the intended edits but also largely adhere to nondestructive editing principles.”
 However, according to Table 3, the models do not seem to accomplish the intended edits.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces the first benchmark for image editing within the Adobe Photoshop environment, emphasizing its core concept of non-destructive layer-based editing. Experiments show that leading MLLMs (e.g., Qwen2.5-VL, GPT-5, Gemini-2.5-Pro) achieve low task success rates but exhibit strong planning abilities. Moreover, human-in-the-loop experiments demonstrate that MLLMs can effectively assist novice users, significantly improving task success and reducing operation time.

### Strengths
The paper is well-structured and easy to follow, with a clear presentation and well-designed experiments. A well-designed benchmark for image editing is much needed to systematically evaluate the capabilities of agents in this domain.

### Weaknesses
1. The paper lacks a clear definition of task difficulty levels in Photoshop. Please clarify whether the difficulty is determined by the number of operation steps or by the intrinsic complexity of each operation. For example, “cropping” and “applying filters” clearly represent different types of tasks, and the criteria for difficulty classification should be explicitly defined. In addition, please provide a detailed statistics and categorization of all Photoshop operation types included in the study to justify the task design better.

2. Since the trajectory in Photoshop is not a fixed path, please further explain how the model handles diverse operation sequences and how the performance is evaluated under such variability.

3. The current MLLM model accuracy is quite low (up to only 18%), while end-to-end editing models demonstrate significantly better performance. Although the authors mention that the proposed method can effectively assist novice users in the human-in-the-loop experiments, the related results are not sufficiently demonstrated. It is recommended to further analyze the causes of low accuracy and to discuss whether the benchmark dataset used in this work is truly effective and representative.

4. Regarding the human-in-the-loop experiment, please specify the number and background of participants. The scale and diversity of participants play a crucial role in validating the effectiveness of the benchmark.

### Questions
It is strongly suggested to include a complete visualization of the agent workflow to help readers better understand the overall system mechanism and benchmark datasets. Specifically, please consider: 
Visualizing the agent’s input and output at each step; 
Presenting intermediate results along with their corresponding evaluation metrics.

### Soundness
3

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
This paper introduces PSBench, the first specialized benchmark for evaluating GUI agents in Adobe Photoshop, a professional image editing application. The benchmark comprises 600 human-annotated tasks divided into three difficulty levels (Easy, Medium, Hard), each with 200 tasks covering both layer-related and non-layer-related operations. Tasks range from simple atomic operations to complex real-world workflows derived from popular YouTube tutorials. Each task includes an original image, target output, and expert-annotated "gold trajectory" demonstrating non-destructive editing practices. The authors propose a novel evaluation metric, Non-Destructive Editing Consistency (NDEC), which measures adherence to professional Photoshop workflows using a 6-criteria checklist. Comprehensive experiments on seven state-of-the-art MLLMs (GPT-4o, GPT-5, Claude-4-Sonnet, Gemini-2.5-Pro, etc.) reveal extremely low success rates—the best model (GPT-4o) achieves only 17.46% on non-layer tasks and 3.80% on layer-related tasks. In contrast, end-to-end image editing models achieve 90-96% overall success. Human-in-the-loop experiments demonstrate that AI assistance significantly improves novice user performance, suggesting collaborative approaches as a promising direction.

### Strengths
1、The benchmark addresses a significant gap in GUI agent evaluation by targeting professional software with complex interfaces and long-horizon tasks, moving beyond existing web and OS benchmarks. The focus on Photoshop is highly relevant given its widespread use in professional image editing.

2、The Non-Destructive Editing Consistency metric is a valuable contribution that goes beyond simple outcome-based evaluation to assess workflow quality and professional practice adherence. This process-oriented metric is particularly appropriate for complex software domains.

### Weaknesses
1、 With only 600 tasks total (200 per difficulty level), the benchmark is relatively small compared to modern datasets. The task distribution is limited to specific Photoshop operations and may not cover the full breadth of professional editing workflows. No discussion of task diversity metrics or coverage analysis is provided

2、The NDEC metric, while innovative, is based on a fixed 6-criteria checklist that may not capture all aspects of professional practice. The paper does not validate whether these 6 criteria comprehensively represent non-destructive workflows or discuss inter-annotator agreement on NDEC scoring. The binary checklist approach may miss nuanced differences in workflow quality.

3、 The paper does not evaluate specialized GUI agent models (AutoGLM-OS-9B, OpenCUA-32B, UITARS) or agent frameworks (CoACT-1) that reportedly outperform general-purpose MLLMs on OSWorld. 

4、Comparing GUI agents with end-to-end image editing models (Table 6) is somewhat misleading since these models solve the task through completely different mechanisms (direct image generation vs. software manipulation). The 90%+ success rates of end-to-end models primarily demonstrate that PSBench tasks can be solved via image generation, but this doesn't provide actionable insights for GUI agent development.

### Questions
see Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
