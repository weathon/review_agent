# LLM2Fx-Tools: Tool Calling for Music Post-Production

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 4

## Abstract
This paper introduces LLM2Fx-Tools, a multimodal tool-calling framework that generates executable sequences of audio effects (Fx-chain) for music post-production. LLM2Fx-Tools uses a large language model (LLM) to understand audio inputs, select audio effects types, determine their order, and estimate parameters, guided by chain-of-thought (CoT) planning. We also present LP-Fx, a new instruction-following dataset with structured CoT annotations and tool calls for audio effects modules. Experiments show that LLM2Fx-Tools can infer an Fx-chain and its parameters from pairs of unprocessed and processed audio, enabled by autoregressive sequence modeling, tool calling, and CoT reasoning. We further validate the system in a style transfer setting, where audio effects information is transferred from a reference source and applied to new content. Finally, LLM-as-a-judge evaluation demonstrates that our approach generates appropriate CoT reasoning and responses for music production queries. To our knowledge, this is the first work to apply LLM-based tool calling to audio effects modules, enabling interpretable and controllable music production.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new framework (including a model, dataset, and overall methodology) for music post-production based on a multimodal LLM. This is evaluated on inferring an Fx-chain, but also on style transfer and using the LLM as a judge.

### Strengths
* This is a relatively unexplored application domain in terms of using multimodal LLMs for music post-production tasks.
* The overall choice of models, the task definition, training process, dataset creation, and evaluation methodology are all appropriate and technically sound.

### Weaknesses
* My main source of criticism for this paper is that this work overall uses established AI methods for a new application. There is little to no AI innovation taking place, and I feel that this work would be more suitable for a venue specializing in audio production or audio engineering (e.g. AES conferences or conventions, ICASSP, or DAFx). I do not see any compelling evidence for inclusion in ICLR.

### Questions
As stated above, I fully agree with the design choices made by the authors in terms of methodology, problem setting, evaluation, and the new dataset based on MedleyDB. The paper is also well structured and well written, and I also appreciate the inclusion of a section on Limitations, which is not something always present in ICLR submissions. 

My only comment is the one stated above, on whether is ICLR the most suitable venue for a work which does not offer any innovation in AI, but rather uses established AI methods with some minor modifications as to support a research question and problem directly situated in the field of audio engineering. As such I might recommend this paper on being marginally out of scope of ICLR.

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
4

### Summary
This paper presents LLM2Fx-Tools, a tool-calling framework that generates sequences of audio effects (Fx-chains) for music post-production. The authors also introduce a new dataset, LP-Fx, to support this task. The topic is novel and of clear interest to the audio and music research community. The proposed system builds upon Fx-Encoder++ and fine-tunes Qwen-4B to achieve the goal of automatic Fx-chain generation.

### Strengths
- The proposed approach to Fx-chain estimation is novel. The integration of Chain-of-Thought (CoT) reasoning into the training framework is also interesting.

- The problem is clearly defined and well motivated.

- The methodology for dataset creation is clearly described and systematically organized.

### Weaknesses
- In Figure 1, the meaning of FxNorm is unclear.

- In Figure 2, why does e_{SEP} consist of two tokens?

- Below Equation (1), what is N? What is param_n?

- In Section 2.1, second paragraph, the authors mention “handle both tasks.” What exactly are the two tasks?

- In Section 2.1, the term “secondary task” is introduced but not clearly defined.

- In Section 2.2 (Audio Encoder), why was Fx-Encoder++ chosen over other possible encoders? How might different audio encoders influence system performance?

- The writing in Section 2.3 (Number Token Loss) and Equation (4) needs improvement for clarity. The statement “a key problem with Cross Entropy is that it treats all incorrect predictions equally” is vague—please elaborate on how this issue is addressed in your proposed loss. Overall, Subsection 2.3 and Equation (4) are difficult to follow.

### Questions
- Will the training dataset and training code be released for reproducibility in future work?

- Will the evaluation dataset and evaluation code also be made publicly available?

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
3

### Summary
The paper presents LLM2Fx-Tools, a novel tool-calling framework which for a given set of audio inputs, provides executable audio effects sequences (Fx-chain), with appropriate CoT reasoning and responses . The paper also introduces LP-FX, a new instruction following dataset with CoT annotations and tools calls for audio effects. The authors provide experimental validation for their approach along with a demo page for subjective verification.

### Strengths
Originality: The paper's key novelty lies in formulating Fx-chain estimation as a LLM-based tool call problem. The autoregressive modeling for LLMs is able to learn the sequential order of audio effect calls as opposed to systems only based on audio features. 

Quality: The paper has detailed experiments around the three evaluation tasks, reverse engineering to show the model can predict tool-chain for paired audios, blind style transfer to show the generalization capability to unseen audios, and natural language language generation to showcase interpretability. Across all the tasks,  LLM2FX-Tools results are strong as compared to the baselines. The authors conduct ablations to show the importance of optimization decisions (CoT, NTL, MST). 

Clarity: The paper is well written with clear notation and figures, with appendix covering all necessary details for dataset generation, evaluation and LLM prompting.

Significance: LLM2FX-Tools framework treats the audio effect modules as external non-differentiable tools, which makes the framework flexible to diverse real world scenarios.The authors also present a LP-FX dataset with CoT annotations and tool calls, which is beneficial for future research

### Weaknesses
For the reverse engineering task, the strongest baseline is Multi-task regression, which comes close even without relying on the ordering of Fx-chain, while the LLM is learning that information. The authors can consider adding a pairwise-ordering loss for the 9 audio effects for the multi-task baselines.

For the style transfer task, the style of the output appears to be mixed between the input and reference audio while listening subjectively to the demo examples. A comparison with differential audio effects style transfer baseline would be quite important to see (https://arxiv.org/pdf/2207.08759). The objective evaluation can benefit on a larger set than 100 test samples

As covered in the limitation section, the paper relies on Fx-normalization and Fx-removal preprocessing, while ideally, they should be modeling as part of the tool-calling framework. Experimental validation is limited to single instruments, datasets are relatively smaller in size with ~2k tracks.

### Questions
For equation 4, $N$ corresponds to the sequence length or training examples? Previous equation 3 has $t$, which is over the sequence, while for $t$ in equation 4 represents the upper range of the number token. It will be helpful if we can improve the notation a little here.

Please consider having a subjective evaluation and a stronger baseline like (https://arxiv.org/pdf/2207.08759) for the style transfer task. 

Gemini 2.5 Pro is used both for dataset generation judge in 3.2, and natural language evaluation judge for table 4. Could we use a different judge to remove the bias for this case?

### Soundness
2

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
3

### Summary
This paper applies existing LLM tool calling techniques to audio effects chain generation. The system uses chain-of-thought to predict effect sequences from audio. The authors create a 101K synthetic dataset LP-Fx generated by Gemini 2.5. In my opinion, the work is mostly an application of existing techniques to a new domain without significant technical innovation.

### Strengths
1. First work applying structured tool calling to audio effects chains
2. Comprehensive evaluation across multiple metrics

### Weaknesses
1. The paper misuses terminology. "Audio style transfer" has established meaning in audio processing literature (timbre/texture transformation). This work only does audio effects parameter transfer, which is much narrower. This creates confusion with existing work and is misleading.
2. Limited technical novelty. The method is standard multimodal LLM fine-tuning: audio encoder -> adapter -> LLM with LoRA. This is direct application of existing techniques without methodological contribution.
3. No human evaluation despite claims about "interpretability" and "controllable music production". All evaluation is automatic metrics or LLM-as-a-judge, which has known reliability issues.
4. Missing details: How does the model handle effects outside the 9 trained types? The paper claims "users can incorporate their own audio plugins" but provides no evidence.
5. The work is mostly experimental validation that LLM tool calling works for this task. The technical contribution is limited.

### Questions
Recommand using the term "audio effects (parameter) transfer" instead of "audio style transfer"

Gemini 2.5 Flash gets MAE 0.32 vs your 0.23, but it achieves 78% accuracy vs your 80%. Why does such a large model fail so badly on parameters? Does it indicate setup issue?

### Soundness
3

### Presentation
3

### Contribution
2
