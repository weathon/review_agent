# TPI-VA: Third-Party Interruption-Aware Voice Assistant

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4

## Abstract
While recent progress in Spoken Language Models (SLMs) has enabled increasingly natural voice-based interactions, they remain vulnerable to third-party interruptions (TPI). To address this challenge, we present a holistic framework for building and evaluating TPI-aware voice assistants. We first introduce TPI-Train, a large-scale dataset of 80K instances spanning 26 realistic interruption scenarios. For evaluation, we introduce TPI-Bench, which includes TPI-Test for measuring response strategies under interruptions and Janus-Test for probing whether models can distinguish true multi-speaker utterances from acoustically single-speaker yet textually misleading speech. To ensure reproducible and interpretable assessments, we also design two complementary metrics: Response Strategy Following (RSF) and Overall Helpfulness (OH). Experiments demonstrate that models fine-tuned with our approach achieve robust performance on TPI-Bench while preserving general dialogue capabilities on VoiceBench, effectively avoiding reliance on textual shortcuts. Human evaluations further confirm that both our dataset and trained models align with human preferences, establishing the first comprehensive solution for TPI-aware voice assistants. Our dataset will be publicly available, Demo samples: https://tpi-va.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The study's major success lies in establishing the first comprehensive TPI-Awareness framework and developing the innovative Janus-Test benchmark, which rigorously isolates acoustic perception to prevent semantic shortcut learning. The use of a composite training strategy with Hard-Negatives is validated as effective in achieving genuine acoustic discrimination, with the final model's responses being highly aligned with user preferences. However, the study's weaknesses center on generalization and depth: the current response strategy diversity is limited to "actionable" and "ignorable" categories; the reliance on synthesized speech for the TPI-Corpus may restrict the model's robustness against complex, real-world acoustic dynamics; and the system lacks demonstrated capability in handling long-term dialogue context.

### Strengths
1. **Novel and Comprehensive Framework for TPI-Awareness:** The paper introduces and formalizes the concept of **TPI-awareness** (Third-Party Interruption-Awareness), establishing the first holistic framework for tackling this crucial, yet underexplored, real-world challenge in conversational AI. The definition, based on two core capabilities (*Discerning Speaker Interruption* and *Situation-Discriminative Response*), provides a clear and actionable path for future research.

2. **Creation of Rigorous Benchmarks to Isolate Acoustic and Semantic Cues:** The authors develop **TPI-Bench** (comprising TPI-Test and the innovative **Janus-Test**) specifically to diagnose the critical failure mode of "shortcut learning." The Janus-Test is a strong methodological contribution as it cleverly forces models to rely on **acoustic evidence** rather than misleading textual cues, enabling a truly robust evaluation of TPI-aware models.

3. **Effective Training Strategy Validated by Ablation and Mechanism Analysis:** The work proposes a novel **composite training approach** that incorporates carefully constructed **Hard-Negatives**. Experimental results, particularly the successful performance on the Janus-Test and the visualization of clearly separated embedding clusters, provide compelling evidence that this strategy is effective in mitigating semantic shortcut learning and enforcing genuine acoustic discrimination.

### Weaknesses
1. **Limited Strategy Diversity, Lack of Richer Response Modes:** Although the paper proposes the binary framework of "Actionable" ($C_A$) and "Ignorable" ($C_I$), this categorization of response strategies is still relatively limited. In complex real-world multi-party dialogues, VAs may require richer response modes such as **seeking confirmation from the primary speaker**, **temporarily maintaining silence**, **logging the third-party input without immediate action**, or **explicitly asking for clarification of intent**. The current framework lacks exploration and benchmarking for these more nuanced, socially sophisticated response modes, limiting the model's practical utility and naturalness in complex human-AI interactions.

2. **TPI-Corpus Reliance on Synthesized Speech, Limited Acoustic Realism:** The entire TPI-Corpus and TPI-Bench are built using **speaker-adaptive Text-to-Speech (TTS)**. While WER verification ensures high transcription quality, synthesized speech often fails to capture the full acoustic complexity of genuine human-human interruptions (e.g., shifts in emotion, accelerated speaking rate during overlaps, spontaneous dynamics). This limitation may cause the model's performance to degrade when generalizing to the subtle acoustic cues present in **real-world, unscripted speech**.

3. **Lack of Capability in Handling Long-Term Dialogue Context:** The paper primarily focuses on the immediate handling of **single interruption events** ($U_{p \to tp} = (U_p, U_{tp})$). However, in practical applications, a third-party interruption might be a persistent issue or require reference to **earlier dialogue history**. The model does not demonstrate the ability to **maintain third-party identity and context** across multiple turns (e.g., remembering a constraint set by the third-party in an earlier turn). A benchmark for evaluating TPI-awareness in complex, long-term multi-party dialogues is needed.

### Questions
Please refer to weaknesses.

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
This work focuses on the robustness of voice assistants in third-party interruption (TPI) scenarios, proposing:

1. The TPI-Corpus (containing approximately 80,000 training samples covering 26 interruption scenarios) and the TPI-Bench evaluation set (comprising two parts: TPI-Test for real dual-speaker interruptions, and Janus-Test for text-identical but single-speaker re-synthesized interruptions to expose text shortcut issues)

2. Evaluation across two dimensions: RSF (Response Strategy Following, indicating adherence to processing strategies) and OH (Overall Helpfulness).

3. This work builds upon the Qwen2.5-Omni-7B foundation, progressively incorporating VoiceAssistant-400K and 8k “hard-negative” single-speaker samples to suppress text shortcuts.

Results show that without significantly compromising general speech task capabilities (evaluated using VoiceBench), the model better follows strategies on the TPI-Test and is less misled on the Janus-Test. Additionally, human preference experiments reveal model outputs align closely with reference answer preferences, corroborating the effectiveness of the strategy and training approach.

### Strengths
- TPI represents a key obstacle to enhancing the practicality of voice assistants in real-world settings. This paper accurately identifies and systematically addresses this issue, demonstrating significant practical relevance and research value.
- The ingenious design of the Janus-Test stands as one of the most prominent highlights of this paper. By controlling for consistent textual content while varying acoustic features, it provides a direct and powerful tool for detecting shortcut learning issues. This significantly enhances the persuasiveness of evaluation conclusions, demonstrating that the model possesses auditory capabilities.
- The experimental section is comprehensively constructed. Table 1 (e.g., Qwen-it performs well on the TPI-Test but fails on the Janus-Test) reveals the essence of the TPI problem, clearly demonstrating the contribution of each training data component to model performance, and strongly supports the core argument.

### Weaknesses
- The entire TPI-Corpus is constructed using text generated by LLM and speech synthesized by TTS systems. The synthesized speech may not fully capture the complex prosodic, emotional, and temporal dynamics present in real human interruptions (e.g., the urgency in the interrupter's voice, the precise timing of overlapping segments, etc.). Supplementing the corpus with a small-scale TPI test set recorded by real humans would further enhance the reliability of the conclusions.
- Simply categorizing interruptions into “actionable” and ‘ignorable’ types is an effective simplification, but more complex scenarios may exist in practice. For instance, in “ignorable” situations, highly intelligent assistants might choose to acknowledge the interruption during their response (e.g., prompting the speaker to continue or confirming whether the interruption should be ignored) rather than completely disregarding it. The binary nature of the current framework may limit exploration of more granular interaction strategies.

### Questions
- In both dataset construction (e.g., scenario dialogue generation) and final evaluation, this work employed Qwen3-235B. Could this introduce bias in the LLM's evaluation of models trained on this dataset, potentially leading to inaccurate experimental results? Could the authors provide evaluation results using other LLMs as judge models, or demonstrate consistency between LLM and human evaluations on smaller datasets?
- The current approach to handling “ignorable” interrupts is to completely disregard them. Have you considered alternative, more nuanced handling strategies, such as providing a brief acknowledgment or pause before responding to the primary user? What considerations led to the decision to ignore them entirely (e.g., simplifying the task, or deeming this the optimal strategy)?
- Does the fine-tuned model still produce similar results on the TPI-Test when using speech data from other languages (non-English), non-LibriSpeech domain-specific speakers, and non-Chatterbox domain-specific TTS models?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work studies the ability of voice assistant systems to handle third-party interruptions using acoustic and textual cues. To support this, the authors construct a large synthetic dataset of interruptions by injecting simulated interjections from third speakers into user-VA dialogues, and propthat integrates audio features and is trained with hard negative examples designed to challenge text-only shortcutting. First, they show that baseline models struggle to handle these synthetic interruptions when speaker identity is manipulated, and that training on synthetic interruptions significantly improves performance on a evaluation set sampled from the synthetic corpus. Then, they introduce a paired contrast set designed to isolate acoustic grounding, and show that hard negative training further improves robustness by encouraging models to rely on speaker change cues rather than transcript semantics alone.

### Strengths
- The work introduces a synthetic data pipeline for constructing interruptions which can be used to finetune models to improve their robustness to third party interruptions.
- The work shows that they are able to finetune models on this dataset without significant degradation to other voice assistant capabilities.
- The Janus-Test counterfactual setup is a clever way to probe whether models are actually using speaker change information versus relying on semantic shortcuts.

### Weaknesses
The work does not provide any clear separated validation for the effectiveness of their synthetic data pipeline. The paper generates interruptions synthetically and solely validate the utility of finetuning on this data by evaluating on data generated using the same synthetic data generation process. This leaves me significantly unclear whether this dataset helps with real world interruptions.

While the synthetic data process seems likely valuable for training, I would expect to see one of the following things:
(1) TPI-Test could be validated by heavily quality filtering using human judgements to confirm that this subset is one that humans agree is high quality and realistic of real world interruptions. 
(2) TPI-Corpus (train split) could be validated using a relatively small sample of real human interruptions gathered from the web (for example, filtered from large web-scale audio corporate such as YODAS2). 

Real world interruptions involve far more auditory cues than simply changing speaker voice (some speakers may be closer or further away from the microphone for example) and it seems necessary to do at least some amount of validation of the data using either human judgements on the proposed test data or additional human evaluation on the trained models. Otherwise, I worry that this dataset could only be improving the ability of the model to respond to the synthetic data distribution.

### Questions
- What is just Qwen2.5-Omni-it? The other models seem to be defined, but this one isn't defined within the paper. https://huggingface.co/collections/Qwen/qwen25-omni doesn't show this being one of the release Qwen 2.5 Omni models, but the lack of definition seems odd if this is newly defined in this paper.

- Was there any form of non-synthetic validation done of the data generation process? Given that the core argument is that this should help the model deal with real auditory

- Is there any form of synthetic variation for the TPI's created other than synthetic speaker swaps? It seems possible to add other forms of realistic noise induced by synthetic interruptions.

- Are all synthetic interruptions strictly turn based (as shown in the figures)? Real world interruptions seem likely to have overlapping speech, but this isn't mentioned in the work.

- Do the LLM-based helpfulness judgments correlate with human judgments in any way? Were any human evaluations of the model responses collected to ground those metrics?

### Soundness
2

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
This paper presents a systematic study of voice assistants’ performance in third-party interruption scenarios, introducing the first large-scale TPI-aware dataset and a comprehensive evaluation framework. The work is innovative in dataset construction, task definition, and evaluation methodology, and it can significantly advance the practical deployment of voice assistants in complex multi-speaker environments. The overall structure is clear, experiments are thorough, and the paper offers substantial academic and practical value.

### Strengths
1. The dataset is very useful and provides a solid foundation for research on third-party interruptions in voice assistants.
2. The experiments are comprehensive and effectively demonstrate the method’s effectiveness.
3. The paper is well-structured and clearly written

### Weaknesses
1. The evaluation is not comprehensive enough, as results for some state-of-the-art closed-source dialogue models (such as GPT-4o-audio, Gemini 2.5 Pro, etc.) are missing.
2. The case presentation is somewhat insufficient; while the paper showcases some cases from the dataset, it lacks output cases from comparative models, making it difficult to intuitively understand the differences in metrics.
3. Although the paper proposes a valuable dataset and evaluation strategy, there is a lack of methodological innovation in model training.

### Questions
1. In constructing the TPI-Corpus, how do you ensure diversity across different scenarios and speakers?
2. User preferences for handling interruptions may vary. What directions for improvement do you see for your work in this aspect?
3. In Section 3.4 you mention that the open-source LLM’s performance is close to closed-source models. Is there a quantified comparison for this task? Also, how large is the gap between LLM-generated and human-generated results?

### Soundness
3

### Presentation
3

### Contribution
3
