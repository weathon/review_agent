# ParaS2S: Benchmarking and Aligning Spoken Language Models for Paralinguistic-aware Speech-to-Speech Interaction

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Speech-to-Speech (S2S) models have shown promising dialogue capabilities, but their ability to handle paralinguistic cues—such as emotion, tone, and speaker attributes—and to respond appropriately in both content and style remains underexplored. Progress is further hindered by the scarcity of high-quality and expressive demonstrations. To address this, we introduce a novel reinforcement learning (RL) framework for paralinguistic-aware S2S, ParaS2S, which evaluates and optimizes both response content and speaking style directly at the waveform level. We first construct ParaS2SBench, a benchmark that evaluates the naturalness of input–output pairs in terms of content and speaking style using expressive and challenging queries. For the automatic judge, we propose a PolyTone training strategy and a multi-stage framework, preventing the style hallucination of end-to-end audio LLM judging. Our judge correlates well with human preferences and is scalable, enabling the model to interact and learn from unlabeled speech via RL. Experiments show that existing S2S models fail to respond appropriately to paralinguistic attributes, performing no better than pipeline-based baselines. Our RL approach (ParaS2SAlign) achieves an 10% relative improvement in the appropriateness of response content and speaking style on ParaS2SBench over supervised fine-tuning (SFT), surpassing all prior models while requiring substantially fewer paired demonstrations than pure SFT. Our findings highlight the need for a scalable and accurate automatic evaluator for speech-to-speech interaction.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a novel reinforcement learning (RL) framework to enhance the capability of paralinguistic cues for speech-to-speech (S2S) dialogue model. It clearly introduce the topic of natural speech communication which conveying more than just words but also paralinguistic cues such as emotions, tone, speaker attributes and how it impacts the speech response. To address this, the author constructs a benchmark which can automatically evaluates the S2S model as ParaS2SBench for content and style appropriateness, then with the benchmark system, it can use RL approach like GRPO to improve the response content and style fitness. 

The major contributions lies on the concrete description of this topic, construct appropriate S2S evaluation as benchmark, demonstration of RL and SFT effectiveness and cost via experiments. The benchmark is characterized by imaginative design and rigorous focus on its objectives, while the exploration of the experiments are also inspiring.

### Strengths
1). The construction of this benchmark system is quite clear and scientific. It designs various of domain for different query, contrasting speaking style, and the most key part is the scenario-controlled queries which is designed to control the neutral text content filtering those doesn't convey too much additional paralinguistic information. According to the appendix table for query example and prompt, the data curation is full of authors' thought on this topic and looks very interesting.  

2）In the experiments part, besides the validation of the effective of RL framework and SFT analysis, it propose so many questions with experiments which are very instructive. Such as the amount of annotations can RL save? should we invest more costs on SFT or reword model, how is the generalization to real speech? These are realistic and serious problems for this S2S model need to handle and could be continuous and broad interest for the research problem.

### Weaknesses
The most weakness of this paper from my perspective is the depth of the experiments part. The author provides rich and many good inspects questions here which is good, however, it seems there isn't any experiment or analysis which shows the most important and critical point view from the author. It may weaken the paper's persuasiveness and confuse reader about the core information obtained from the experiments.

### Questions
It would be much friendly if the author could select the most crucial experiments to introduce more profoundly, analysis/conclude them clearly in the main paper session 5 and put the other experiments details in the appendix paper. Anyway, there are some part which is not fully introduced due to the length to the article and need to read the appendix to get the full results. 
For the generalization to real speech experiment, is there any human evaluation to compare with the bench score? This is quite curial to this RL framework.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ParaS2S, a reinforcement learning framework that enables speech-to-speech models to generate responses with appropriate content and speaking style. Using the new ParaS2SBench for automatic content and style evaluation, ParaS2S improves paralinguistic awareness by 11% over supervised fine-tuning while requiring far fewer annotations.

### Strengths
The motivation to evaluate paralinguistic responses in speech language models is both natural and important. This reviewer appreciates the authors’ effort in advancing research on this topic. Moreover, the overall workload presented in this paper appears substantial.

### Weaknesses
(1) The core contributions of this paper are somewhat unclear. It mainly includes two parts: a new benchmark for evaluating speech response style and content, and an alignment technique for tuning speech language models. However, each contribution appears incomplete on its own. The benchmarking part omits many relevant speech and speech-to-speech models, while the proposed alignment method lacks sufficient novelty and empirical validation.

(2) The citation format should follow the ICLR template by using \citep instead of \cite, as the current style blends citations into the text and reduces readability.

(3) Table 1 reports several numerical results to show that the evaluation aligns with human judgments, but the justification is not rigorous. The paper should clarify what criteria define “closeness” to human evaluation and why they are reasonable. For instance, in the Emotion S2S model, the gap between GPT and human evaluations does not appear negligible.

(4) The novelty of the alignment approach is limited. If the paper argues for the necessity of GRPO, this claim should be empirically supported, and comparisons with existing methods such as SpeechAlign are essential.

(5) The GRPO alignment is evaluated only on the Kimi-Audio base model. A more comprehensive study should include multiple base models to demonstrate that the proposed strategy generalizes beyond a specific setup.

In conclusion, while the paper presents a decent amount of work, it remains incomplete by publication standards. The authors are encouraged to focus on a single, well-developed contribution—either the benchmarking framework or the alignment technique.

### Questions
(1) What is the primary contribution of this paper — the new benchmark or the proposed alignment strategy? The authors should clarify which aspect represents the core focus of the work.

(2) Why is the proposed alignment strategy not compared with SpeechAlign? The two methods appear quite similar, except for the adoption of the GRPO technique. Since SpeechAlign also employs DPO and evaluates comparable aspects of speech language models, a direct comparison is necessary to highlight the distinction and advantage of the proposed approach.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The study introduces a methodology for enabling Speech-to-Speech (S2S) language models to recognize and respond to critical paralinguistic elements, such as emotion, intonation, and speaker characteristics, which extends far beyond simple content transmission.

The authors present two core components:

* **ParaS2SBench:** An automated benchmark designed to evaluate how effectively S2S models align with both the **content** and **style** of an utterance.
* **ParaS2SAlign:** A learning framework that utilizes Reinforcement Learning (RL) to achieve model alignment directly at the waveform level.

The benchmark scores show a high correlation (>0.7) with human evaluations and the RL approach achieves a notable 11% performance improvement over Supervised Fine-Tuning (SFT), alongside a five-fold enhancement in label efficiency.

### Strengths
- The paper puts forward a novel benchmark and dataset, with a welcome commitment to their public release. 
- The authors provide a valuable analysis of the respective impacts of RL and SFT on the modeling of non-verbal conversational aspects within the proposed framework.

### Weaknesses
* **On the Reward Model:** A point of consideration emerged regarding the reward model. I'm respectfully curious about the potential for it to be somewhat overfitted to the specific synthesis engines used for evaluation, namely the GPT-based TTS and CosyVoice. I would be interested to hear the authors' perspective on its generalization capabilities to other speech styles.
* **On Data Synthesis:** Additionally, as the audio corresponding to the evaluated scenarios appears to be entirely synthetic, a slight query arises regarding potential constraints on the diversity and complexity of the model's expressive output. I wonder if this might impact the model's ability to capture the full spectrum of nuances present in organic, human-to-human interaction.

### Questions
1.  **Readability:** I noticed a minor formatting point where the inconsistent use of parentheses for citations occasionally impacted readability. Clarifying this convention throughout the manuscript might be beneficial for readers.
2.  **Data Composition:** Could the authors please specify the total number of distinct speakers represented in the training data and the ParaS2SBench benchmark, respectively?
3.  **Confidence Intervals:** I would find it very helpful to see the 95% confidence intervals for the reported GPT and human evaluation scores, as this would further strengthen the statistical significance of the findings.
4.  **Performance on Existing Capabilities:** Finally, a point of great interest is the trade-off with existing abilities. I would be grateful if the authors could provide an analysis of any performance degradation on foundational capabilities (e.g., as measured by VoiceBench) after the application of SFT and, subsequently, the full RL alignment process.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces an innovative benchmark and reinforcement learning framework for paralinguistic-aware speech-to-speech (S2S) models, effectively addressing the current limitations in handling paralinguistic information such as emotion, tone, gender, and age. The authors design an automated data curation and speech synthesis pipeline, and leverage a reward model for efficient training and evaluation. Experimental results demonstrate that the RL approach achieves significantly better content and style appropriateness than conventional supervised fine-tuning, with much lower data and annotation costs. Overall, this work is forward-looking and practical, providing valuable tools and references for the development of paralinguistic-aware S2S models.

### Strengths
1. The proposed paralinguistic-aware S2S reinforcement learning framework is highly practical, effectively enhancing the model's ability to understand and generate paralinguistic information such as emotion and tone, which provides valuable tools and methods for the advancement of speech dialogue systems.
2. The experiments are thoroughly designed, covering various paralinguistic factors and realistic scenarios. The results comprehensively validate the significant improvements in content and style appropriateness achieved by the proposed method, making the findings highly convincing.
3. The paper is well-structured and clearly articulated, with a rigorous logical flow. It progresses coherently from problem background, method design, to experimental validation and result analysis, making it easy for readers to understand and follow.

### Weaknesses
1. The methodological innovation of the paper is limited, as it merely applies GRPO in a straightforward manner.
2. The presentation lacks intuitiveness; it is difficult to fully convey the paralinguistic features of audio through text alone. It would be better if there were a demo page or web-based showcase.
3. Some references are missing, such as [1]: 
[1] Omnichat: Enhancing spoken dialogue systems with scalable synthetic data for diverse scenarios. arXiv preprint arXiv:2501.01384.

### Questions
1. In your data curation and speech synthesis process, how do you ensure that the generated paralinguistic styles (such as emotion, age, gender, etc.) are sufficiently authentic and diverse?
2. After SFT and GRPO, does the model’s original capability decrease compared to the results reported for Kimi-Audio?

### Soundness
3

### Presentation
3

### Contribution
3
