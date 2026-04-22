# Shanks: Simultaneous Hearing and Thinking for Spoken Language Models

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2, 4

## Abstract
Current large language models (LLMs) and spoken language models (SLMs) begin thinking and taking action only \textit{after} the user has finished their turn.
This can create a high latency for waiting until the model ends the thinking process.
Consequently, thinking \textit{after} receiving the full input is not suitable for speech-to-speech interaction, where real-time and low-latency interaction is important.
We address the above issue by drawing inspiration from the fact that humans can naturally \textit{``think while listening''}.
In this paper, we propose \textbf{Shanks}, a general inference framework that enables SLMs to generate unspoken chain-of-thought reasoning when listening to the user input.
Shanks streams the input speech in fixed-duration chunks and, as soon as an input chunk is received, generates an unspoken reasoning based on all previous speech and reasoning; in the meantime, the user is still speaking.
Shanks uses unspoken reasoning to perform intermediate calculations, make API calls to complete the task, and determine whether to interrupt the user.
We demonstrate that Shanks enhances the real-time user-SLM interaction in two scenarios:
(1) When the user is presenting their solution to a math problem, Shanks can listen to and reason over the user's speech and make interruption when the user makes a mistake.
Shanks interrupts the user 37.1% more accurately compared with a baseline that interrupts the user without thinking.
(2) In a task-oriented dialogue setting, where the user's request needs to be completed by calling hotel and flight booking APIs, Shanks can complete 63.2% of the API calls before the user even ends their turn.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces SHANKS, an inference framework to enable SLMs to perform thinking while listening. The paper focuses on two tasks. 1) Interrupting the user when the model detects a mistake in the user's reasoning. 2) Making simultaneous API calls while the user is still speaking. Experimental results demonstrate that SHANKS can effectively handle both tasks.

### Strengths
- The paper is well-written and easy to follow.
- Experimental results demonstrate the effectiveness of SHANKS on the proposed two tasks.

### Weaknesses
- Over-claiming. The title “thinking while listening” is broad and ambitious, but the paper only addresses two specific tasks. A more accurate title might be “SLM for Educational Error Correction ...” It would also be more compelling to see an SLM with more general “thinking while listening” capabilities.
- Questionable setting. For the API-call scenario, it is unclear whether users would actually care about API-call latency. Users may prioritize task success rate over latency. Furthermore, some complex tasks may require several minutes to complete, while user queries may finish very quickly (e.g., in 10 seconds), making the latency reduction be trivial.
- Lack of Self-Containment. The paper fine-tunes a thinker-talker SLM but does not discuss how the talker is adapted to perform speech generation. In Qwen-2.5-Omni, the talker’s input includes hidden states from the thinker. The SHANKS approach introduces additional reasoning tokens, which may affect the hidden states of subsequent response tokens. This suggests that adapting the talker may not be trivial. Additionally, the evaluation appears to be performed only on the thinker’s text output, effectively reducing the SHANKS pipeline to a speech-to-text LLM.
- Lack of Real-Time Latency Evaluation from the User’s Perspective. The time for an API call is not a fixed number, so if the API call gets stuck, the user may never get the response.

### Questions
- In scenario 1, when the model interrupts the user, can the user interrupt back? Does the model continue listening to the user’s speech after the interruption? Can the model handle multi-turn dialogues?
- In Table 1, why is the valid interrupt ratio typically lower than the interrupt ratio in the “wrong” subset? Are there scenarios when an interruption occurs in the wrong subset but it is invalid?
- According to Section 5.5, in the “SHANKS + call-after-listen” setting, the early call mechanism is the same as in the “SHANKS” setting. Why, then, is the early call accuracy in “SHANKS + call-after-listen” higher than in “SHANKS” as shown in Table 2?

### Soundness
2

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
5

### Summary
The paper proposes a pipeline for Speech-to-Text (presumably) multimodal LLMs that performs intermediate reasoning while listening, in contrast to most existing LLMs or SLMs that execute reasoning or thought processes only **after** receiving the full input. The proposed method demonstrates its effectiveness particularly in mathematical speech correction and tool-calling tasks, thereby validating the utility of the introduced pipeline.

### Strengths
Personally, I found this paper quite engaging. It is written in an accessible way, and the figures are intuitive and easy to understand. I was particularly impressed by how the proposed approach partially aligns with human-like reasoning patterns.

### Weaknesses
Rather than noting weaknesses, I have summarized a few personal questions and minor concerns below:

### Questions
1. **Use of LLM-as-a-Judge in Mathematical Tasks**
   The paper employs an LLM-as-a-judge framework to evaluate reasoning and interruptions in math-related tasks. As you know, reasoning tasks are inherently more challenging than general ones, and I still have slight concerns about whether current LLMs have reached a level of convergence sufficient for reliable evaluation. Perhaps I missed it, but I would like to know which API or model was used as the judge and whether its evaluations correlate with human judgments. In other words, I would like to see evidence that the LLM is capable of evaluating such novel tasks accurately.

2. **Clarification on the Speech Modality (Speech-to-Speech vs. Speech-to-Text)**
   From the figure, the architecture appears to have a speech-to-speech structure with a decoder attached. However, to my knowledge, *Qwen2.5-Omni* does not currently support speech-to-speech SFT, as the encoder of the codec module used for decoding is not publicly released. In my experience, even when tuning the core thinker module with LoRA (especially with full tuning), compatibility with the talker module often breaks, leading to generation failures. Should I understand the proposed system as *speech-to-text* instead? If so, it might be helpful to clarify or revise the figure and corresponding text. On the other hand, if it truly is *speech-to-speech*, I am curious how you addressed potential representation mismatches between the two modules that could degrade generation quality.

3. **Generation Stability in Variable-Length Interleaving**
   As I understand it, prior interleaving approaches, those that handle listening and reasoning concurrently or operate in a full-duplex manner (e.g., OmniFlatten, SyncLLM), typically generate a *fixed-length* assistant response chunk corresponding to each user input chunk to maintain generation stability. This fixed-chunk design, in my view, helps mitigate endless generation issues and allows manual truncation when necessary. In contrast, your method appears to train on *variable-length* reasoning trajectories aligned with the preceding input chunks. I am curious whether you encountered any generation instability (e.g., occasional repetition or degradation in 1 out of 100 samples, etc.) during this variable-length reasoning chunks generation.

4. **Comparing “Thinking While Listening” vs. “Thinking After Listening”**
   This is not a criticism but rather a personal curiosity. I personally believe that enabling reasoning or assistant speech *during listening*, as your pipeline does, represents the future direction of speech LLMs. From that perspective, its conceptual counterpart would be the conventional *turn-based* approach, where the assistant reasons and responds *after* the user’s utterance ends. While you did compare the two in the tool-calling experiment, I am particularly interested in how they differ in more reasoning-intensive tasks such as mathematics. Specifically, it would be fascinating to compare “thinking while listening” and “thinking after listening” pipelines in terms of reasoning accuracy, interruption prediction quality (i.e., whether interruptions are appropriate), and the validity of the interruption content. I understand that latency might be impossible to measure, and I do not view lower performance of the after-listening variant as a weakness, but including such an analysis could enhance the completeness and informativeness of the paper.

Overall, I found this paper very enjoyable and thought-provoking. Thank you for your work. If additional experiments addressing the above questions are provided, I would be happy to raise my score.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Shanks, a general inference framework that enables spoken language models (SLMs) to determine whether to interrupt the user's input speech when the user is making mistakes. Shanks starts the reasoning as soon as the input speech from the user is received. This way it reduces the time the user needs to wait after their speech is done, and also the user can correct their misunderstandings in their intermediate steps. Experimental results on math problems and task-oriented dialogues show substantial improvement over the baselines.

### Strengths
- The paper is clearly written and easy to follow.
- The core concept, “thinking before speaking,” is interesting. The extension of this idea to conversational AI, particularly through the proposed “interrupt” functionality, is creative. Such mechanisms could be beneficial for users studying logical or structured domains like mathematics or science, where step-by-step reasoning feedback is valuable.
- The authors also constructed a speech dataset derived from existing text-based reasoning datasets to train and evaluate their models. It would be helpful for the research community, but I'm not entirely sure how these datasets support future research beyond their current experiments.

### Weaknesses
- The core idea of this paper: "Shanks: Simultaneous Hearing and Thinking for Spoken Language Models" is already described in the other paper: "STITCH: Simultaneous Thinking and Talking with Chunked Reasoning for Spoken Language Models". In STITCH, the authors also had the same motivation as this paper, which is "...humans can think while speaking and propose STITCH-R".
The main difference might be the interruption function of SHANKS. The paper does not include direct experimental comparisons with STITCH models (STITCH-R and STITCH-S) despite the conceptual overlap. It is unclear what additional benefit or novelty SHANKS provides, considering that SHANKS performance on a task-oriented dataset performs worse than call-after-listen.

- I’m not convinced about the usefulness of SHANKS in other contexts, such as mental consulting or conversational counseling. In these settings, users often describe the main issue only at the end, and much of the earlier speech may be digressive or irrelevant. SHANKS does not explain how it identifies and prioritizes the most important context segments—it simply begins reasoning as soon as the input speech starts. This approach could cause significant computational overhead and wasted inference cycles without improving understanding.

- In educational contexts, premature or incorrect interruptions by SHANKS could lead to confusion or misunderstandings rather than improving interactivity. Some human experiments, like how humans actually find this useful, would help to strengthen the paper's results.

- While SHANKS shows strong performance improvements on math reasoning datasets, its results on task-oriented dialogues (e.g., flight or hotel booking scenarios) are mixed. In particular, the call-after-listen baseline often performs better than SHANKS itself. Moreover, user inputs in such scenarios can be long and continuously updated. For example, time-sensitive information like flight details may change quickly, so initiating the reasoning process too early can lead to outdated or irrelevant inferences.

### Questions
My questions and suggestions are listed throughout the strength and weakness sections.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
SHANKS (Simultaneous Hearing and Thinking with Chunked Input Speech) is an inference framework that enables spoken language models to generate internal chain-of-thought reasoning while the user is still speaking, rather than waiting until the user finishes their turn. The system works by streaming user speech in fixed 4-second chunks and generating hidden thinking tokens after each chunk, conditioning on all previous speech and reasoning. The authors fine-tune Qwen-2.5-Omni on synthetic data generated by GPT-4o and evaluate on two applications: (1) educational tutoring where the model interrupts users making mistakes in math problem-solving, achieving 63.9% valid interruptions compared to 26.8% for a no-thinking baseline, and (2) task-oriented dialogue where the model makes API calls for travel booking while the user speaks, successfully completing 63.2% of required calls before the user finishes. The paper claims this is the first work to explore generating unspoken reasoning during user speech and demonstrates that this approach reduces response latency and enables more natural real-time interaction.

### Strengths
### **Strength 1: Novel Problem Formulation with Clear Practical Motivation**

The paper addresses an underexplored direction in spoken language model research: enabling models to generate internal reasoning while the user is still speaking, rather than waiting for complete input. This problem formulation is well-motivated by natural human behavior---people think while listening, allowing for timely reactions and reduced response latency. While incremental processing and chain-of-thought reasoning exist independently in prior work, the specific combination of streaming chunked speech input with unspoken CoT generation for real-time action-taking (interruptions, API calls) represents a relatively unexplored setting. The paper clearly articulates why this capability matters for real-time interaction and provides a concrete framework (chunked streaming with interleaved thinking) that could inspire future work in making speech-based AI systems more responsive.

---

### **Strength 2: Demonstrates Technical Feasibility with Measurable Effects**

The paper successfully implements the proposed system and shows it functions as designed across two diverse applications. The quantitative results demonstrate clear technical effects: in the API calling task, the system successfully completes 63.2% of required calls before the user finishes speaking (compared to 0% for traditional approaches); in the math tutoring task, thinking-based interruption achieves 63.9% valid interruptions compared to 26.8% for the no-thinking baseline. The authors provide detailed methodology for training data construction, including all prompts used for synthetic data generation, making the work reproducible. They also create evaluation protocols and datasets for this setting, providing infrastructure for future research. While the evaluation is entirely synthetic (a limitation discussed in weaknesses), the paper does establish proof-of-concept that this approach is technically implementable and can produce measurable differences in timing and decision accuracy within controlled settings.

### Weaknesses
### **Weakness 1: Entirely Synthetic Evaluation with No Human Validation**

The paper's core claim is about improving real-time user interaction---reducing latency, enabling timely interruptions, and enhancing user experience. However, the evaluation is entirely synthetic: training data generated by GPT-4o, speech synthesized by TTS, and all quality judgments made by GPT-4o as evaluator. There are zero human studies.

This is a critical gap because:

- **UX claims require UX validation**: The paper positions this as solving practical interaction problems (educational tutoring, customer service), but provides no evidence that real users benefit from or prefer this system over traditional turn-taking.

- **Interruption acceptability is inherently subjective**: Whether an interruption feels "timely and helpful" versus "annoying and premature" cannot be determined by LLM judges alone. Different users may have vastly different preferences for interruption behavior.

- **LLM-as-judge limitations**: While LLM evaluation is useful for rapid iteration, relying on it exclusively for subjective judgments (especially about human experience) leaves the core utility claims unsubstantiated.

The evaluation demonstrates the system works in synthetic settings but provides no evidence that it actually improves human-AI interaction in practice. At minimum, a comparative user study (SHANKS vs. traditional turn-taking) measuring user preference, perceived helpfulness, and task completion experience would be necessary to validate the paper's central claims.

---

### **Weakness 2: Limited Analytic Depth and Under-Motivated Design Choices**

Beyond the evaluation gap, the paper’s contributions are primarily at the *inference-system* level rather than algorithmic. While this is a valid contribution type, the paper does not yet analyze the trade-offs or justify key design choices, which limits its scientific depth.

- **Conventional learning formulation.** Training uses cross-entropy fine-tuning on synthetic data with interleaved speech and reasoning chunks. There are no new objectives or architectures. The novelty lies in the inference structure, but its dynamics (e.g., reasoning coherence over time) are not quantitatively studied.  
- **Arbitrary hyperparameters.** The 4-second chunk length is chosen for GPU throughput rather than from a principled latency-versus-accuracy or user-experience analysis.  
- **Limited baselines.** Both baselines are author-defined. The cascade baseline (ASR + stronger text LLM) outperforms end-to-end SHANKS, suggesting that backbone reasoning quality may drive the main improvements rather than the streaming design itself.  
- **Train/test overlap.** The API-calling experiment splits ComplexFuncBench 50/50 for training and testing, leaving open questions about generalization.  
- **Missing analyses.** The paper omits studies of reasoning stability across chunks, failure cases, compute cost, and robustness to real speech phenomena such as noise or disfluency. Including these would elevate the work from a functional demo to a deeper scientific contribution.  
- **Hybrid requirement.** The strongest results combine SHANKS with post-speech reasoning, implying that thinking-while-listening is helpful but not sufficient on its own.

Overall, SHANKS is a compelling **proof-of-concept** showing that simultaneous reasoning and listening is technically feasible. To reach ICLR-level maturity, future work should include ablations on chunk size, latency–accuracy trade-offs, real-speech robustness, and human evaluations.

### Questions
Question 1: Why does the cascade baseline outperform end-to-end SHANKS, and what does this tell us about where the value actually lies?

Question 2: Have you conducted any human evaluation, and if not, what barriers prevented it?

Question 3: Your best results require combining SHANKS with traditional post-speech processing. Doesn't this suggest thinking-while-listening alone is insufficient?

### Soundness
2

### Presentation
2

### Contribution
2
