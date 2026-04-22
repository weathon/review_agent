# WavReward: Spoken Dialogue Models With Generalist Reward Evaluators

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 8, 4

## Abstract
End-to-end spoken dialogue models such as GPT-4o-audio have recently garnered significant attention in the speech domain. However, the evaluation of spoken dialogue models' conversational performance has largely been overlooked. This is primarily due to the intelligent chatbots convey a wealth of non-textual information which cannot be easily measured using text-based language models like ChatGPT. To address this gap, we propose WavReward, a reward feedback model based on audio language models that can evaluate both the IQ and EQ of spoken dialogue systems with speech input. Specifically, 1) based on audio language models, WavReward incorporates the deep reasoning process and the nonlinear reward mechanism for post-training. By utilizing multi-sample feedback via the reinforcement learning algorithm, we construct a specialized evaluator tailored to spoken dialogue models. 2) We introduce ChatReward-30K, a preference dataset used to train WavReward. ChatReward-30K includes both comprehension and generation aspects of spoken dialogue models. These scenarios span various tasks, such as text-based chats, nine acoustic attributes of instruction chats, and implicit chats. WavReward outperforms previous state-of-the-art evaluation models across multiple spoken dialogue scenarios, achieving a substantial improvement about Qwen2.5-Omni in objective accuracy from 53.4 to 91.5. In subjective A/B testing, WavReward also leads by a margin of 83. Comprehensive ablation studies confirm the necessity of each component of WavReward. All data and code will be publicly after the paper is accepted.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposed a generalist reward model for single-turn spoken dialogues. The authors proposed training the reward model with GRPO based on the IQ and EQ of the spoken response. In addition, they added CoT in the reward model and also an exponential curve design for calculating the reward for GRPO training. They also curated a dataset, ChatReward-30K, for training and evaluating reward models. For out-of-distribution evaluation, they curated the RealDialogue dataset, which records real human interaction with speech models. Throughout their ablation studies, they demonstrated the effectiveness of their training method over baselines with a large margin on their evaluation benchmark.

### Strengths
1. It’s the first paper to propose a "generalist" evaluator for spoken dialogue. I believe this contribution will enlighten further research in the field of spoken dialogue systems.
2. Their ablation shows the effectiveness of their proposed methods: CoT/multisamples/nonlinear reward.

### Weaknesses
1. The writing of this paper is hard for readers to follow. There are two models that are called reward models in this paper: the proposed WavReward and the reward model used to train WavReward. I strongly recommend that the author change the naming.
2. It’s a concern for me that the authors only invited 5 human experts to manually verify ChatReward-30K, which is an unreasonable workload for ensuring the quality of the dataset on such a large scale.. 
3. In natural human conversation, there exist multiple reasonable spoken responses (content and paralinguistic-wise) for a given query. I’m questioning whether it is realistic to train and evaluate such evaluators with a ground truth score from a dataset with only 5 human experts involved?

### Questions
1. Is there a typo in equation 2? There is no $W_{\theta}^{'}$ in the right-hand side of the equation.
2. If I’m understanding correctly, the authors are using GRPO for their reward model training. Why does the author not explicitly mention GRPO in their paper?
3. In Appendix F, the author mentions that their WavReward is initialized from a model named “Qwen-2.5-Omni-7B-Think”. However, I cannot find this model released on the websites/huggingface. Could the author provide some details on this?
4. I’m not fully understanding the sentence (Line 255 to 257) “Notably, the KL divergence loss $\mathcal{L}\_{KL}{\left( W_{\theta}, W_{\theta}^{ref} \right) }$ is not incorporated into the reward process of WavReward.” What do the authors mean by “reward process”?
5. How is the baseline model trained via supervised fine-tuning? What is the objective?
6. For the A/B testing experiment on the RealDialogue dataset, the authors mentioned that 5 human experts are involved in selecting optimal scores between different models. Are these 5 human experts the same group of people for the manual verification of ChatReward-30K?

### Soundness
2

### Presentation
2

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
This paper proposes WavReward, a novel model designed to evaluate the performance of spoken dialogue systems that can both understand and generate speech (e.g., GPT-4o-audio, Qwen2.5-Omni, Moshi). Existing evaluation methods mainly focus on text, ignoring non-verbal cues such as tone, emotion, and rhythm. As a result, they cannot accurately measure the true conversational and emotional quality of modern end-to-end speech models. WavReward fill this gap. Extensive experiments show the superior performance of WavReward.

### Strengths
1. It is an interesting and practical idea to train a reward model that evaluates generated speech based on the dialogue context.
2. The model outperforms existing baselines and SFT results. Experiments on RealDialogue demonstrate that the trained model not only fits well to the benchmark but also generalizes effectively to real-world conversations.

### Weaknesses
I think the paper is well-motivated and shows promising potential.

1. Since the thinking process lacks explicit ground truth and is learned through reinforcement learning, it is crucial to examine whether the reasoning process is meaningful and whether it provides useful insights into how the generated speech aligns with dialogue context. Adding human evaluation results to validate this aspect would greatly strengthen the paper. 
2. There is a large body of previous work (e.g., [1][2]) that employs Speech Language Models (SLMs) as speech quality evaluators and reasoners. Please include a discussion of these related studies for completeness.
3. Please disclose the source and background of human annotators to ensure transparency and reproducibility.
4. It would be helpful to include more analysis of the proposed dataset, such as inter-annotator agreement. In my own experiments, the variance of MOS scores tends to be high. I am curious whether the variance is even larger in this more complex setting where annotators must consider dialogue context.
5. Could you provide more qualitative examples of the implied chat scenarios? This part seems highly subjective, and there may not be a single “gold” answer for many cases.

[1] Wang, Siyin, et al. "Qualispeech: A speech quality assessment dataset with natural language reasoning and descriptions."

[2] Chen, Chen, et al. "Audio large language models can be descriptive speech quality evaluators."

### Questions
### Suggestions
1. When a citation is used as a noun in a sentence, consider using $\citet$ instead (line 286).

### Questions
1. Is there any example where $s_{1}$ is low? Does it refer to the semantic aspect of the speech?
2. In lines 376–377, how did you determine which samples in RealDialogue are labeled as positive and which as negative?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper focuses on the core challenges in evaluating end-to-end speech dialogue models. Traditional evaluation methods heavily rely on text transcripts, failing to capture and measure the rich paralinguistic information inherent in speech interactions. This limitation hinders comprehensive assessments of a model's overall capabilities. To address this, the authors propose WavReward—a universal reward evaluator based on audio language models—aimed at providing a unified and in-depth assessment of both a model's IQ and EQ.

WavReward's core architecture builds upon audio language models and is optimized through reinforcement learning, enabling direct processing of end-to-end speech dialogue data. To enhance evaluation accuracy and interpretability, the framework integrates CoT, prompting models to generate justification before assigning scores. Additionally, a novel nonlinear reward mechanism imposes exponential penalties on results exhibiting significant scoring deviations. Furthermore, the training process employs a positive-negative multi-sample feedback strategy. By comparing multiple superior and inferior responses to the same question, this approach significantly enhances the evaluation model's discriminative capabilities.

To support model training and validation, this work constructs ChatReward-30K, a large-scale, high-quality speech dialogue preference dataset covering multiple dimensions. WavReward significantly outperforms all baseline models in objective accuracy. More importantly, in real-world subjective A/B tests, its evaluation results demonstrate high consistency with human preferences, proving the method's effectiveness and practical value.

### Strengths
- How to scientifically, comprehensively, and efficiently evaluate speech dialogue models is a critical issue that demands urgent attention. This paper precisely addresses this pain point, making its research highly timely and significant.
- The ChatReward-30K dataset represents a significant contribution. The authors provide a detailed account of its construction process, covering multiple dimensions including content, acoustics, and implicit/explicit aspects. This fills a gap in the field by offering a high-quality, comprehensively annotated evaluation benchmark.
- The experiment compared multiple state-of-the-art audio-language models (including direct inference and supervised fine-tuning), making the results more compelling. Evaluation was conducted using both in-domain (ChatReward-30K-test) and out-of-domain (RealDialogue) data, demonstrating the model's generalization capability. This work combines objective accuracy metrics with subjective A/B testing to validate WavReward's superiority from multiple perspectives. The high consistency between results and human judgments indicates alignment between the model and human preferences.

### Weaknesses
- The ChatReward-30K dataset is primarily synthesized using TTS models. This process may introduce synthetic biases or artifacts (such as unnatural prosody), raising questions about its ability to generalize to real human conversations.
- The generated data is filtered by WER and SER. However, no systematic filtering was conducted for accents, pitch , and other factors. Consequently, samples with poor synthesis quality—such as those generated by TTS models that deviate from the specified accent in the instructions—may adversely affect both training and testing.

### Questions
- Could you elaborate on the specific implementation of the “think format reward” (Rf)? During training, how does the system automatically determine whether the CoT generated by the model is compliant, thereby awarding either 5 points or 0 points? Is this based on keyword matching or an inspection of the output structure?
- Is there a text-based LLM serving as an upper-bound baseline with the input of dialogue transcriptions (with paralinguistic information provided as textual descriptions)? It could help compare WavReward's theoretical upper limit.
- In Table 1, most baselines exhibit similar performance on ChatReward and RealDialogue, while WavReward significantly outperforms on ChatReward compared to RealDialogue. Does this indicate that WavReward risks overfitting on the in-domain test set, or could other factors be responsible?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces WavReward, a novel reward model designed to evaluate end-to-end spoken dialogue models by directly processing speech-to-speech dialogues. The authors correctly identify a significant gap in the evaluation of such models, as existing benchmarks rely on transcribing audio to text, thereby losing crucial paralinguistic information (e.g., emotion, tone, accent). WavReward is an audio language model trained via reinforcement learning, incorporating three key innovations: a Chain-of-Thought (CoT) reasoning process, a nonlinear reward function, and a multi-sample feedback mechanism. To support this, they also introduce ChatReward-30K, a comprehensive dataset for training and evaluating audio reward models, covering both explicit and implicit dialogue scenarios with human-expert scores. Experiments show that WavReward significantly outperforms strong baseline models, including GPT-4o-audio, in both in-domain and out-of-domain settings.

### Strengths
1. Well-Motivated Problem: The paper effectively highlights a critical, underexplored problem: the lack of evaluation methods that can assess the full spectrum of spoken dialogue models' capabilities, particularly the paralinguistic and emotional quotient (EQ), without converting speech to text. The motivation is clear and strong.

2. Comprehensive Solution (Model & Dataset): The work is holistic, proposing both a novel model (WavReward) and a supporting dataset (ChatReward-30K). This two-pronged approach is a significant contribution to the community.

3. Methodological Components: The integration of CoT reasoning, a nonlinear reward function, and multi-sample feedback is a sophisticated and well-justified approach. The ablation studies (Table 1) effectively validate the importance of each component.

4. Extensive and Convincing Evaluation:
The evaluation is thorough, testing on a wide range of acoustic attributes (age, accent, emotion, etc.) and dialogue types (content, explicit, implicit).
The use of a real-world, out-of-domain dataset (RealDialogue) is a strong point, demonstrating the model's robustness and practical applicability.
Including both objective metrics (accuracy) and subjective A/B tests with human experts greatly strengthens the validity of the claims. The high win rates (77-83%) against powerful baselines like GPT-4o-audio are impressive.

5. Strong Empirical Results: The performance gains over state-of-the-art baselines are substantial and clearly demonstrated. The leap from Qwen2.5-Omni's 53.4% to WavReward's 91.5% on objective accuracy is a powerful result.

### Weaknesses
1. Dataset Construction Details:
While the dataset construction process is described in stages, more details would be beneficial. For example, how were the "human experts" selected and calibrated to ensure scoring consistency?
The prompt templates in the appendix are helpful, but a more detailed discussion of the challenges in generating high-quality, diverse implicit dialogues would be valuable.

2. Computational Cost and Scalability:
The computational cost of WavReward's training and inference is not discussed. The multi-sample sampling and CoT reasoning likely introduce significant overhead compared to direct inference. A discussion of this trade-off between evaluation quality and cost is important for practical adoption.

3. Clarity of Writing and Terminology:
The writing can be improved for clarity. Some sentences are long and complex, making them difficult to parse (e.g., the first sentence of the abstract).
The term "positive-negative multi-sample sampling mechanism" is used in the abstract and text, but the method section describes it as selecting samples at different quality levels. The terminology should be made consistent.
The acronym "GT" is used before being defined.

### Questions
1. Baseline Fine-tuning: For the supervised fine-tuning baselines (Qwen2.5-Omni w/ Full-param & LoRA), what was the exact training objective? Was it a regression loss (MSE) between the predicted and ground truth scores?

2. Generalization: How do you envision WavReward generalizing to entirely new paralinguistic features or languages not present in ChatReward-30K? Is the model learning a general evaluation principle, or is it heavily tuned to the specific attributes in your dataset?

3. Inference Speed: What is the relative inference latency of WavReward compared to a baseline like Qwen2.5-Omni with direct inference? This is a practical concern for using such an evaluator in development pipelines.

### Soundness
3

### Presentation
3

### Contribution
3
