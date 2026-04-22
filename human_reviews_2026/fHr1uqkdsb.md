# Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 8, 4

## Abstract
When reading, we often have specific information that interests us in a text. For example, you might be reading this paper because you are curious about LLMs for eye movements in reading, the experimental design, or perhaps you wonder ``This sounds like science fiction. Does it actually work?''. More broadly, in daily life, people approach texts with any number of text-specific goals that guide their reading behavior. In this work, we ask whether open-ended reading goals can be automatically decoded solely from eye movements in reading. To address this question, we introduce goal decoding tasks and evaluation frameworks using large-scale eye tracking for reading data in English with hundreds of text-specific information seeking tasks and auxiliary annotations of task-critical information. We develop and compare several discriminative and generative multimodal text and eye movements LLMs for these tasks. Our experiments show considerable success on selecting the correct goal among several options, and even progress towards free-form textual reconstruction of the precise goal formulation. We further tie model performance to cognitively interpretable aspects of human gaze behavior. These results open the door for further scientific investigation of goal driven reading, as well as the development of educational and assistive technologies that will rely on real-time decoding of reader goals from their eye movements.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper tackles the task of decoding open-ended reading goals from eye movements. Using the OneStop dataset, the authors propose two tasks-target selection and target reconstruction—and design multimodal models combining gaze and text features. Results show that gaze–text integration slightly improves performance, indicating that eye movements encode semantic goal information. This paper provides a new research paradigm for the combination of eye movement and language models, but further improvement is needed in terms of interpretability, cross-text generalization, and comparison with more powerful baselines. Overall, the paper is interesting, but its current form cannot totally support all authors' claims.

### Strengths
- The authors claim that they are the first study to decode open-ended, text-specific reading goals from eye movements, framed as dual tasks of selection and reconstruction.
- Integrating text and gaze features markedly improves target selection; RoBERTEye-Fixations achieves 49.3% accuracy vs. 33% baseline.
- DalEye-Llama attains 76.3% QA accuracy on unseen participants (vs. 68.1% for human distractors), validating the gaze–goal correspondence.

### Weaknesses
1. Lacks cognitive interpretation of gaze behavior and its link to goal decoding.
2. Sharp performance drop on unseen texts (Kappa 0.478 → 0.069) unexplained.
3. No comparison with fine-tuned multimodal LLMs (e.g., GPT-4o, LLaVA-1.5).

### Questions
1. What causes DalEye-Llama's performance to drop dramatically on new text? Is it due to limitations in feature transfer or model overfitting?
2. The paper mentions "inherent noise" in eye tracking data but does not discuss mitigation strategies. Have you tested noise reduction or fixation filtering to improve model stability?
3. The generation model is only compared to GPT-4o in zero-shot mode. Could results with a fine-tuned GPT-4o (using the same text and gaze inputs) clarify advantages in efficiency or robustness?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents a novel and important investigation into decoding open-ended, text-specific information goals from eye movements during reading. The work is rigorous, introducing a clear task framework (goal selection and reconstruction) and systematically evaluating a range of discriminative and generative models. The experiments are comprehensive, using a large-scale, high-quality dataset (OneStop) and thoughtfully evaluating model generalization. While the generative task remains highly challenging, the strong performance of the best discriminative model, especially on the difficult "same critical span" task, provides compelling evidence that eye movements contain fine-grained information about a reader's goal. This work opens promising new directions for both scientific inquiry and practical applications.

### Strengths
1.This is the first work to systematically address the decoding of arbitrary, text-specific information goals from eye movements. It moves beyond previous work that classified pre-defined procedural reading tasks (e.g., reading vs. skim-reading) to a more challenging and practically relevant semantic decoding task. The contribution is significant and opens a new research direction.

2.The experimental design is exemplary. The data splits are carefully constructed to evaluate generalization to "New Participants," "New Texts," and "New Text & Participant," providing a clear and honest assessment of model robustness. The subdivision of the selection task into "Different" and "Same" critical spans offers nuanced insight into task difficulty.

3 The paper provides a thorough benchmark, covering simple heuristics, adapted state-of-the-art discriminative models, and pioneering generative LLM-based approaches. This offers a valuable overview of the landscape for this new task.

4. The evaluation methodology is a strength. Beyond selection accuracy, the authors propose a robust set of metrics for the generative task, including question word/category agreement, BERTScore, and a creative downstream QA accuracy metric, which convincingly demonstrates the utility of the generated questions.

5. The paper is accompanied by a code repository and includes extensive details in the main text and appendices regarding model architectures, hyperparameters, and training procedures, ensuring the work is easily reproducible.

### Weaknesses
1. As the results show, the generative task is exceptionally difficult, and model performance, especially on new texts, is still limited. The generated questions are not yet on par with human-composed ones.

2.The paper successfully demonstrates that goals can be decoded, but offers less insight into how or why the models make their decisions from a cognitive perspective. The models remain somewhat black-box.


3. While generalization is tested, the significant performance drop in the "New Text & Participant" regime indicates that model robustness is still limited to the distribution of the training data.

### Questions
1.In the RoBERTEye-Fixations model, which achieved the best results, did you perform any ablation studies to understand which specific eye-movement features (e.g., fixation duration, saccade amplitude, regressions) were most critical for its performance, particularly on the challenging "Same Span" task?

2.For the generative DalEye-Llama model, performance drops most significantly in the "New Text" regime. Do you attribute this primarily to the model's difficulty in comprehending the new text content itself, or in associating the novel eye-movement patterns on that text with the question generation process?

3.The paper mentions promising applications in education and assistive technology. Given the current accuracy levels (e.g., ~49% for 3-way selection), what do you see as the minimum performance threshold for such systems to be reliably deployable in real-world scenarios? What are the key next steps to bridge this gap?

### Soundness
4

### Presentation
4

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
The paper introduces a cognitive-state decoding problem of recovering information-seeking goals (text questions) from a reader’s eye movements while reading a piece of text. The authors propose different task formulations for recovering the stimulus – (1) question selection and (2) question generation. They develop discriminative and generative models for the above tasks, that recover the question given the text and eye movement data. Results shown on the OneStop eye-tracking data showing discriminative models (RoBERTEye-Fixations) perform better than random while generative models (DalEye-Llama) shows promising performance on less challenging cases.

### Strengths
The dataset (OneStop) and problem setup is well suited to the objective of recovering reading goals. The evaluation regimes which include splitting data by new participant and new text is well conceived and creation of two tiers of difficulty are useful in comparing model performance in challenging settings.

The authors experiment with different types of baselines – heuristics, discriminative models based on adaptions of prior work and generative LLM models (DalEye-LLaVA, DalEye-Llama).

### Weaknesses
1. I would like to see what types of gaze features (eg: fixation durations, word revisits) are more useful for recovering the information seeking goals. Stronger experiments are required to investigate the feature attributions by gradually phasing out these features one by one from the eye movements data to train the models.

2. It is not clear in the paper if the question and the text span containing the corresponding answer have significant substring overlap. If so, the problem becomes more trivial where users just have to look for specific strings from the question in the text. A more realistic and challenging case to evaluate would be if the user would have to understand and infer the meaning from the passage if the language is phrase differently.

3. The authors should more comprehensively discuss about relevant literature in the field of eye movements conditioned on information seeking goals, for instance
- Synthesizing Human Gaze Feedback for Improved NLP Performance (EACL 2023) which generates gaze patterns conditioned on the reader’s intent / task
- GazeXplain: Learning to Predict Natural Language Explanations of Visual Scanpaths (ECCV 2024) which predicts scanpaths during performing visual question answering tasks or when instructed to search for a particular object in the image


Questions / Suggestions for Improvement:
- Show n-gram overlap between target question and corresponding text span and correlate with model correctness.
- More thorough experiments to analyse eye feature importances

### Questions
NA

### Soundness
2

### Presentation
2

### Contribution
2
