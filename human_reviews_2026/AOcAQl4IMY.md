# VideoCogQA: A Controllable Benchmark for Evaluating Cognitive Abilities in Video-Language Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Recent advances in Large Video-Language Models (LVLMs) have led to promising results in multimodal video understanding. However, it remains uncertain whether these models possess the key cognitive capabilities for high-level tasks, especially those requiring symbolic and abstract reasoning. Existing benchmarks predominantly rely on real-world, annotated videos, which suffer from a lack of control over content and inherent difficulty, limiting their diagnostic utility. To address these limitations, we introduce \textbf{VideoCogQA}, a scalable and fully controllable benchmark inspired by game-based environments, designed to assess the cognitive abilities of LVLMs. By generating synthetic videos through a programmatic engine, VideoCogQA offers precise control over visual elements, temporal dynamics, and the video task difficulty, effectively isolating cognitive reasoning from prior semantic knowledge. The dataset consists of tasks involving abstract concepts, symbolic elements, and multimodal integration, with varying levels of difficulty based on Python-based game scenarios. Experimental results show that even state-of-the-art (SOTA) models, such as Qwen2.5-VL-72B, achieve an average performance of 48.8% on tasks involving abstract concepts. Additionally, performance drops by 15% as task complexity increases, highlighting the challenges LVLMs face in maintaining consistent performance. Through this work, we hope to show the limitations of current LVLMs and offer insights into how they can more effectively emulate human cognitive processes in the future.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses a critical gap in evaluating Large Video-Language Models (LVLMs): the lack of controllable benchmarks to assess high-level cognitive capabilities (e.g., symbolic reasoning, abstract concept understanding) beyond basic semantic comprehension. Existing benchmarks rely on real-world annotated videos, which suffer from limited content control and difficulty in isolating cognitive reasoning from prior semantic knowledge. To solve this, the authors propose VideoCogQA, a scalable and fully controllable benchmark built on programmatic synthetic videos inspired by game environments (e.g., maze navigation, sky battles).

VideoCogQA uses a Python-based pipeline to generate 800 videos and 3,280 questions across 10 game scenarios, with three difficulty levels (Easy/Medium/Hard) controlled via code parameters (e.g., grid size in Chameleon Grid, enemy count in Sky Battle). It evaluates six cognitive dimensions: Object Perception (OP), Action Perception (AP), Temporal Reasoning (TR), Spatial Reasoning (SR), Game-environment Perception (GP), and Full-modal Perception (FP)—expanding beyond the scope of existing video benchmarks

### Strengths
The work introduces a novel paradigm for LVLMs evaluation by leveraging programmatic synthetic videos to isolate cognitive reasoning from prior semantic knowledge—addressing a fundamental limitation of real-world video benchmarks (e.g., MVBench, Video-MME) that rely on contextual cues (e.g., playground scenes for action inference). Inspired by cognitive science (game-based human cognition studies), VideoCogQA’s 10 game scenarios and six cognitive dimensions (especially GP and FP) expand the scope of video-LM evaluation beyond existing frameworks, which focus primarily on semantic understanding. The combination of Python-driven controllability, GPT-4 QA generation, and fine-grained difficulty tuning is a creative integration of existing tools to solve a new problem.

### Weaknesses
1. Insufficient Annotation of Frame Sampling Requirements:  A major limitation of the synthetic pipeline is the lack of frame sampling annotations for each video scenario. As noted in the reviewer’s comment, tasks like Maze Runner (8×8 maze requiring 14 steps to solve) may be unsolvable with small frame samples (e.g., 8 frames), as critical steps would be missed. The paper mentions evaluating models with their "official default inference settings" (Section 4.1) but does not:
  - Define the minimum number of frames (N=8/16/32/64) required to solve each task/difficulty level.
  - Analyze how frame sampling impacts performance (e.g., whether Qwen2.5-VL-72B’s 54.1% average accuracy drops further with 8 frames vs. 64 frames for Maze Runner).
This omission weakens the interpretability of results—poor performance on a task could stem from model limitations or insufficient frame sampling, not just cognitive gaps.
2. Ambiguous Human Evaluation Setup: 
The human performance benchmark (90.3% accuracy) lacks critical details, making it hard to compare with model performance:
The paper states human accuracy is the "average of two independent annotators" (Section 4.1) but does not clarify:
  - Whether humans accessed the full video or only sampled frames (consistent with model inputs). If humans used full videos, their 90.3% accuracy may understate the gap (models are at a disadvantage with limited frames).
  - The viewing protocol: Did annotators see questions before or after watching the video? Did they watch once or multiple times? A  "question-first" setup (common in QA tasks) would likely yield near-100% accuracy for humans, so the 90.3% error rate needs explanation (e.g., ambiguous questions, fast-paced videos).
Without this clarity, the human baseline cannot effectively contextualize model limitations.
3. Coarse-Grained Correlation Analysis: 
The paper computes correlation coefficients at the dataset level (e.g., VideoCogQA vs. VideoMME, Table 5) but not at the cognitive dimension level. Dataset-level correlations mask whether VideoCogQA’s individual cognitive dimensions are valid proxies for real-world capabilities. A dimension-specific analysis would better validate the benchmark’s diagnostic utility.

### Questions
1. Frame Sampling Requirements: For each of the 10 scenarios (e.g., Maze Runner, Time Sequence) and difficulty levels, could you provide the minimum number of frames (N=8/16/32/64) required to answer questions correctly?
2. Human Evaluation Details: Could you specify the human annotation protocol:
  - Did annotators use full videos or sampled frames (matching model inputs)?
  - Did they view questions before or after watching the video? How many times could they watch?
  - What caused the 9.7% human error rate (e.g., question ambiguity, fast video pace)?
This would strengthen the human baseline as a meaningful comparison for model performance.
3. Dimension-Specific Correlation: Could you compute correlation coefficients between VideoCogQA’s six cognitive dimensions (OP/AP/TR/SR/GP/FP) and corresponding dimensions in real-world benchmarks? 
4. Error Attribution: Could you add a supplementary analysis distinguishing between perception and reasoning errors?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes VideoCogQA, a synthetic, controllable video benchmark designed to probe cognitive abilities in Large Video-Language Models (LVLMs), in which videos are generated programmatically, difficulty is tuned via explicit code parameters, and multiple-choice questions are created from GPT-4-authored templates. The benchmark totals 800 videos and 3,280 questions over ten game-inspired scenes with easy/medium/hard settings.  The work claims novelty in explicit controllability and difficulty. 

Empirically, strong LVLMs (e.g., Qwen2.5-VL-72B, GPT-4o) still trail humans by a large margin on symbolic tasks, performance drops with difficulty, and results correlate highly when varying frame sampling and when compared to several real-world video benchmarks.

### Strengths
1. Controllability & difficulty. Clear, code-level knobs (e.g., grid size) allow precise difficulty control, improving diagnostic value.
2. Breadth of skills. Ten diverse scenes spanning object/action perception, spatial/temporal reasoning, game environment understanding, and audio-visual mapping.  
3. Well-Documented Human–Model Gap. The paper clearly reports a substantial gap between human and model performance across all tasks and scenarios.

### Weaknesses
1. Lack of random baseline. With 3–5 options, the performance of random choice can be 20–33%. This paper does not foreground a random baseline. 
2. Lack of connection to Real-World Tasks. The paper does not extensively discuss the connection between VideoCogQA and real-world tasks. The current justification, based primarily on frame sampling, is insufficient. It remains unclear whether performance on specific VideoCogQA tasks correlates with performance on real-world tasks. Clarifying whether success on particular tasks within VideoCogQA is predictive of performance in specific real-world scenarios would significantly strengthen the practical relevance of the benchmark.

### Questions
1. What is the average of choices per question in the dataset, and what is the corresponding random baseline accuracy?
2. Is model performance on VideoCogQA correlated with performance on other existing benchmarks?

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
This paper introduces a benchmark called VideoCogQA, designed to evaluate the cognitive abilities of video-language models. The benchmark automatically generates question–answer (QA) pairs by creating synthetic videos from simple games and combining them with predefined text templates. It also allows for controllable difficulty adjustment. When evaluated using existing state-of-the-art (SOTA) models, the performance remained relatively low at around 48.8%, indicating significant room for improvement. Thus, VideoCogQA serves as a valuable benchmark for identifying the current limitations of video-language models and establishing new research goals aligned with those limitations.

### Strengths
​
- The paper demonstrates that synthetic videos can be automatically generated from a game simulation engine, and that LLM-based instruction templates are created for each game according to predefined question categories. This approach enables dataset generation at scale, without being constrained by data size. To support this, the authors propose a Python-based video synthesis pipeline.


- The authors introduce VideoCogQA, a scalable and fully controllable benchmark. This benchmark is well-organized, consisting of six categories (OP, AP, TR, SR, GP, FP) and three difficulty levels (easy, medium, difficult).


- The paper conducts and analyzes extensive experiments across various Large Vision-Language Models (LVLMs).

### Weaknesses
- Lack of Details on Dataset Distribution


   - The paper does not provide sufficient details or analysis regarding the dataset distribution. It would be beneficial to include a detailed breakdown of the number of samples per category, organized by game and by difficulty level. The current explanation in Section 3.2 is largely textual and difficult to fully understand.


   - It would also be helpful to report how each game covers the different question categories, and how the VLM (Vision-Language Model) performances vary across these categories.


   - Additionally, a comparative analysis of the data distribution between VideoCogQA and existing benchmarks would strengthen the evaluation and contextual understanding of the dataset.


- Limited Relevance to Real-World Scenarios


   - While generating synthetic videos from games to evaluate cognitive abilities is an innovative idea, it remains unclear how such synthetic settings translate to real-world problems. There is uncertainty about whether this approach truly enhances real-world understanding.If the benchmark includes a training split, one way to validate its practical relevance would be to fine-tune models on VideoCogQA and evaluate them on other benchmarks to assess transferability. However, in the current setting, the paper should either demonstrate or justify the real-world applicability of the benchmark in another way.

### Questions
Please provide your responses with reference to the weaknesses mentioned above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes VideoCogQA. VideoCogQA is a synthetic, fully controllable benchmark for testing video–language models on object/action perception, temporal/spatial reasoning, gameplay stats, and simple audio–visual links. Models trail human performance and get worse as tasks grow harder; when videos are replaced with clean code-log text, accuracy jumps, suggesting the main bottleneck is visual perception rather than language reasoning. Scaling helps but doesn’t close the gap, pointing to the need for stronger spatiotemporal encoders and better symbolic perception.

### Strengths
1. The benchmark adds Game-environment and Full-modal, explicitly targeting symbolic/abstract attributes (size, color, shape) and temporal/spatial relations.

2. The authors programmatically synthesize videos with parameterized difficulty and log code-level events, then generate QA templates with GPT-4 and human filtering.

### Weaknesses
1. Question templates originate from GPT-4 and are then filtered; more auditing of prompt templates and filtering criteria may strengthen validity claims and reproducibility.

2. The “~90% human” number isn’t well documented. We don’t know how many people were tested, how much time they had, whether they could replay the video, or how consistent the labels were. That makes the human ceiling hard to trust and compare against models.

### Questions
1. If you swap in a stronger vision stack—say, a structured front-end with detection/tracking/attributes, or a higher-capacity spatiotemporal backbone—does overall accuracy go up, and would that change your conclusion about the main bottleneck?

2. Do difficulty levels align with human-perceived difficulty (item-response theory or psychometrics)? Any per-item discrimination analysis?

3. If a model is trained on the synthetic tasks, does it transfer to natural-video QA, and in which categories?

### Soundness
2

### Presentation
3

### Contribution
2
