# EchoMind: An Interrelated Multi-level Benchmark for Evaluating Empathetic Speech Language Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 8

## Abstract
Speech Language Models (SLMs) have made significant progress in spoken language understanding. Yet it remains unclear whether they can fully perceive non lexical vocal cues alongside spoken words, and respond with empathy that aligns with both emotional and contextual factors. 
Existing benchmarks typically evaluate linguistic, acoustic, reasoning, or dialogue abilities in isolation, overlooking the integration of these skills that is crucial for human‑like, emotionally intelligent conversation. We present EchoMind, the first interrelated, multi‑level benchmark that simulates the cognitive process of empathetic dialogue through sequential, context‑linked tasks: spoken‑content understanding, vocal‑cue perception, integrated reasoning, and response generation. 
All tasks share identical and semantically neutral scripts that are free of explicit emotional or contextual cues, and controlled variations in vocal style are used to test the effect of delivery independent of the transcript.
EchoMind is grounded in an empathy‑oriented framework spanning 3 coarse and 12 fine‑grained dimensions, encompassing 39 vocal attributes, and evaluated using both objective and subjective metrics. 
Testing 12 advanced SLMs reveals that even state‑of‑the‑art models struggle with high-expressive vocal cues, limiting empathetic response quality. Analyses of prompt strength, speech source, and ideal vocal cue recognition reveal persistent weaknesses in instruction‑following, resilience to natural speech variability, and effective use of vocal cues for empathy. 
These results underscore the need for SLMs that integrate linguistic content with diverse vocal cues to achieve truly empathetic conversational ability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces EchoMind, a benchmark designed to evaluate the empathetic capabilities of Speech Language Models (SLMs) in dialogue, addressing the limitation of existing benchmarks that test linguistic, acoustic, and reasoning abilities in isolation. The benchmark employs an empathy-oriented framework including 39 vocal attributes, and its evaluation of 12 advanced SLMs reveals that almost all models struggle with highly expressive vocal cues, restricting their ability to generate contextually and emotionally aligned empathetic responses.

### Strengths
The paper introduces EchoMind, a large-scale benchmark designed to evaluate empathy in Speech Language Models (SLMs) through tasks of understanding, reasoning, and conversation. It initiates a systematic direction for assessing empathetic capability in SLMs, which is largely overlooked in prior speech or multimodal benchmarks. The authors back this up with extensive experiments on 12 advanced SLMs, and provide comprehensive analyses covering prompt sensitivity, synthetic vs. human speech robustness, and upper-bound performance under ideal vocal cue recognition.

### Weaknesses
See questions.

### Questions
1. The approach to synthesizing *hoarse* voices via voice cloning seems questionable, as such vocal characteristics are typically physiological rather than stylistic. Similarly, generating environmental-sound speech by simply mixing clean TTS outputs with background noise may not accurately reflect real-world acoustic conditions. Could the authors clarify how these methods ensure realism and avoid artifacts that might bias evaluation results?
2. Using vocabulary-level metrics such as BLEU or ROUGE to compare model responses against gold references does not seem to capture empathetic ability, since model generates responses in a free-form way. Could the authors justify the inclusion of these metrics or explain how they relate to empathy-oriented evaluation?
3. Table 5 shows large discrepancies between human and model-as-a-judge scores, particularly for GPT-4o-Audio, which raises concerns about the reliability of automatic judgments. Could the authors provide more details on the size of the sampled subset and discuss why such divergences occur?
4. The abbreviations *C1–C4* and *P1–P3* are difficult to interpret without repeatedly referring back to the text. Providing clearer, descriptive names would make the evaluation setup easier to follow.

### Soundness
3

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
4

### Summary
This paper introduces EchoMind, a benchmark designed to evaluate the empathetic capabilities of Speech Language Models (SLMs) through a multi-level framework consisting of understanding, reasoning, and conversation tasks. The benchmark uses controlled vocal-style variations of semantically neutral scripts across 39 vocal attributes spanning speaker information, paralinguistic cues, and environmental sounds. The authors evaluate 12 advanced SLMs and find that even state-of-the-art models struggle with highly expressive vocal cues, particularly in generating empathetically aligned responses. While the paper addresses an important gap in evaluating emotional intelligence in SLMs, it suffers from several methodological limitations and presentation issues that prevent acceptance at this time.

### Strengths
- The focus on empathetic capabilities in SLMs addresses a critical gap in current benchmarks, moving beyond pure linguistic understanding to emotional intelligence.
- The benchmark uses controlled vocal-style variations of semantically neutral scripts across 39 vocal attributes spanning speaker information, paralinguistic cues, and environmental sounds.

### Weaknesses
- The study introduces multi fine-grained tasks, but results are only reported at the coarse levels (understanding, reasoning, conversation). Showing results for each sub-task would reveal which empathy aspects, like tone recognition or emotional adaptation, remain most challenging.

- While the paper lists 39 vocal attributes, their individual contributions to empathy are not analyzed. Exploring which features (e.g., pitch, tempo, timbre) drive performance would make the work more insightful and interpretable.

- The human–TTS comparison is interesting but stops at showing a performance gap. It’s unclear whether the issue comes from model robustness or unrealistic TTS cues(39 vocal attributes is reasonable?). Diagnostic experiments could clarify this.

- The human evaluation is on too small a sample (6 cases per vocal-cue type for 3 models) to draw strong conclusions.Expanding the dataset and adding inter-rater agreement (e.g., Cohen’s Kappa) would strengthen the conclusions.

- GPT-4o is used both to generate  “ground truth” and as a model under evaluation. This creates circular bias and may inflate its scores. An independent labeling model is recommended.

-  Typos: Line 291  two ”, ,” ; Line 263: "shown as Table 2"-->"as shown in Table 2"

### Questions
- What are the exact prompts used for GPT-4o to generate scripts and MCQs?
- How do you justify calling this an "empathy" benchmark when it primarily tests acoustic feature detection?
- Why not include more analysis of which specific vocal attributes are most challenging?

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
4

### Summary
The paper presents EchoMind, the first interrelated, multi-level benchmark designed to evaluate the empathetic and socially intelligent capabilities of Speech Language Models (SLMs). It aims to integrate understanding, reasoning, and conversational response tasks grounded in vocal cues, testing models across 39 vocal attributes. The benchmark includes both synthetic and limited human-recorded data and evaluates 12 SLMs using objective and subjective metrics.

### Strengths
This is the first benchmark explicitly targeting speech-based social intelligence and empathetic understanding. The effort to formalize empathy-related evaluation dimensions in SLMs is both timely and valuable for the community.

### Weaknesses
- While the paper motivates the work by criticising existing benchmarks for evaluating capabilities in isolation, the proposed benchmark does not actually integrate these skills in a meaningful way. The three “levels” (understanding, reasoning, and conversation) are still evaluated independently, using separate metrics. What is presented feels like more dimensions, not genuine integration. A stronger demonstration would involve an Arena-style extrinsic evaluation assessing the real downstream impact of empathy-aware dialogue systems.
- The reported high inter-dimensional correlations undermine the paper’s central claim about interrelated evaluation. If the metrics are highly correlated, this suggests that the tasks may not capture distinct or complementary aspects of empathetic ability — further challenging the notion of “multi-level” integration.
- The benchmark relies heavily on synthetic data generated by LLMs and TTS systems, with only two human speakers contributing real recordings—and both are non-native English speakers. This raises serious concerns about data diversity, realism, and ecological validity. Additionally, although the paper states that synthetic data was “manually checked,” there is no evidence of systematic human evaluation or quality control beyond a few ad hoc examples.
- The human evaluation results are not statistically validated and, in fact, appear inconsistent with the model-based GPT-4o evaluations in half of the tests. It is, therefore, misleading to conclude “alignment” between human and model-based assessments. Without statistical tests or inter-rater reliability analysis, the claim lacks empirical support.

### Questions
See my comments above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose EchoMind, a multi-level benchmark for evaluating empathetic capabilities of Speech Language Models, assessing whether models can perceive non-lexical vocal cues—beyond just spoken words—and respond with emotional intelligence. 
The benchmark proposes 3 coarse-grained dimensions (speaker, paralinguistic, environmental) and 39 vocal attributes. It evaluates existing models with three tasks: understanding (content + voice perception), reasoning (integrated inference), and conversation (empathetic response generation), assessed via both objective metrics andModel-as-a-Judge/human evaluation. Testing 12 SOTA Speech LLMs reveals that while models excel at content understanding, they struggle with vocal-cue processing, and even state-of-the-art systems fail to generate emotionally aligned responses.

### Strengths
- Novel benchmark for evaluating empathetic capabilities of Speech LLMs. High quality taxonomy covering 39 attributes across speaker, paralinguistic, and environmental dimensions, providing comprehensive coverage of non-lexical vocal cues essential for human-like conversation.
- Rigorous evaluation in multiple setting, using both automatic and human evaluation at both text and audio levels. Moreover, it introduces specialized empathy metrics (EmoAlign, Vocal Empathy Score) and validates it on many state-of-the-art Speech LLM, revealing  limitations in vocal-cue integration and instruction-following.

### Weaknesses
Minor
- Majority TTS-generated data (646/1,137 scripts) with only 2 professional voice actors for human recordings, potentially missing natural variation and introducing artifacts that don't reflect real-world.

### Questions
URO-Bench and EChat-eval seems the most similar, could you elaborate more about the differences and the coverage?

### Soundness
3

### Presentation
4

### Contribution
3
