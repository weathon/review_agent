# LatentQA: Teaching LLMs to Decode Activations Into Natural Language

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6

## Abstract
Top-down transparency typically analyzes language model activations using probes with scalar or single-token outputs, limiting the range of behaviors that can be captured. To alleviate this issue, we develop a more expressive probe that can directly output natural language, performing LatentQA: the task of answering open-ended questions about activations. A key difficulty in developing such a probe is collecting a dataset mapping activations to natural-language descriptions. In response, we propose an approach for generating a dataset of activations and associated question-answer pairs and develop a fine-tuning method for training a decoder LLM on this dataset. We then validate our decoder's fidelity by assessing its ability to read and control model activations. First, we evaluate the decoder on a number of supervised reading tasks with a known answer, such as uncovering hidden system prompts and relational knowledge extraction, and observe that it outperforms competitive probing baselines. Second, we demonstrate that the decoder is precise enough to steer the target model to exhibit behaviors unseen during training. Finally, we show that LatentQA scales well with increasing dataset and model size.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces LatentQA, a method for reading and steering LLM activations using natural language. Through Latent Interpretation Tuning (LIT), a decoder LLM is fine-tuned on pairs of activations and language labels to interpret and control model behaviors. LIT enables open-ended questioning about activations (e.g., detecting bias or goals) and can steer LLMs by applying gradients from natural language prompts. The approach outperforms prior probing methods in accuracy and bias reduction, showing promise for scalable, interpretable, and self-monitoring LLMs.

### Strengths
1. The paper proposes a novel framework that enables natural language interaction with LLM activations. 
2. By allowing both reading and steering of activations, the method bridges interpretability and alignment, demonstrating practical utility in bias detection and correction.
3. LIT significantly outperforms the baselines in accuracy and sample efficiency.

### Weaknesses
1. Relying on another LLM to generate QA pairs may introduce artificial or biased patterns, raising concerns about the authenticity of latent interpretation.
2. The current approach of linking activations to language and using them to steer the model is somewhat expensive.
3. All experiments are conducted exclusively on Llama-3-8B-Instruct, which raises concerns about the generality and robustness of the proposed method. Since the approach fundamentally depends on model activations and layer-specific representations, its effectiveness may vary significantly across different architectures, model sizes, or training paradigms (e.g., decoder-only vs. encoder-decoder models).

### Questions
1. I believe it would strengthen the paper to include comparisons with alternative methods for reducing bias in LLMs.
2. I believe this approach can be helpful for interpreting and understanding the model. However, when it comes to steering, the process of extracting activations and mapping them to natural language seems quite costly, and I am not fully convinced that this is the most efficient way to achieve control. What specific advantages does LatentQA offer for model steering compared to existing alignment techniques such as DPO?

### Soundness
3

### Presentation
3

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
This paper introduces LATENTQA, a novel task for probing Large Language Models (LLMs) by answering open-ended questions about their latent activations in natural language. Instead of traditional probes that output scalars or single tokens, this work proposes Latent Interpretation Tuning (LIT), a method to train a "decoder" LLM (a copy of the target model) to interpret the "target" LLM's activations.

The authors develop a scalable pipeline to create a pseudo-labeled dataset by using a powerful external LLM (o1-preview) to generate question-answer pairs corresponding to (prompt, completion, activation) tuples. The method is evaluated on two types of tasks: "reading" (e.g., extracting relational knowledge, uncovering hidden system prompts) and "control" (e.g., debiasing, steering model behavior). The results show that LIT outperforms baselines like linear probes and untrained patching methods (SelfIE/Patchscope).

### Strengths
1. **Novelty of the Task**: The core concept of LATENTQA is ambitious and novel. It moves transparency research beyond simple, low-bandwidth probes (linear, scalar) towards a much more expressive, high-bandwidth framework. The idea of "captioning" or "interrogating" a model's internal state using natural language is a compelling research direction. Treating activation itself as a modality for QA may provide foundation for further interpretation/intervention.

2. **Creative Experimental Design**: The task of uncovering a hidden system prompt solely from the activations of a user's message is a creative and strong demonstration of the method's potential. It shows that the activations may contain information about the model's "state" or "intent" beyond the text itself.

3. **Training Dataset Generation**: The paper proposes a clever and practical solution to the difficult problem of collecting a dataset for this task. The pipeline for generating (control, stimulus, completion) triples and then using another LLM to generate descriptive and reasoning-based QA pairs is a scalable approach.

### Weaknesses
1. **Unfair/Incomplete Baseline Comparisons**: This is the most significant weakness. The LIT method involves fine-tuning a decoder model, yet it is primarily compared against untrained methods (SelfIE, Patchscope) or much simpler linear probes training. It is not surprising that a fine-tuned LLM decoder outperforms these methods.

- For Control: A more convincing comparison would be against standard behavioral fine-tuning (SFT) or preference-tuning methods (DPO/RLHF) using the same (prompt, completion) data. How does the complex LIT-based gradient steering for debiasing compare to just fine-tuning the target LLM on unbiased text, or using rejection sampling?

- For Reading: How does the performance of identifying a hidden persona from activations compare to a baseline model that is simply fine-tuned on the (user_prompt, model_response) text pairs to predict the persona? This would isolate the true value of using the activations.

2. **Limited Generality (Model-Family Specific)**: The experiments are confined to the Llama-3 model family. Models in a same family share same architecture and very similar training data. Experimenting with only a single model family doesn’t show the method is general to all LLMs.

3. **Weak Motivation for Some Experiments**: The qualitative "control" experiments (e.t., "Golden Gate Claude" persona, eliciting harmful content) are interesting but not rigorous. As your notes point out, many of these behaviors (especially persona-following) can be achieved with simple prompting. The paper doesn't do enough to prove why activation-based steering is a more robust, efficient, or necessary method than standard SFT or even just prompting.

### Questions
1. **Comparison to SFT**: How does the performance of the "control" tasks (debiasing, steering) compare to a standard SFT baseline that fine-tunes the target LLM on the (prompt, completion) pairs generated for the LATENTQA dataset? This seems like the most direct and important comparison.

2. **Comparison to Text-Only Baseline**: For the "reading" tasks (like hidden persona identification), how does the LIT decoder compare to a baseline model trained only on the input-output text pairs? This would help quantify the actual information gain from using activations versus just observing behavior. Is the decoder actually “reading” the activations, or simply learns how the target model may behave given certain prompts?

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
The paper contributes:
- A method for decoding model's internal activations & steering its behaviour in natural language
- An evaluation setup for answering open-ended questions about model's internal activations
- A pipeline for creating training dataset for the method

The decoding method works by finetuning a separate decoder LM on [Activation]+[QA] pairs, where [Activation] come from the target model we want to interpret.
The steering method works by training the "decoder" model on QA pairs containing behaviour we want to elicit in the "target" model. The QAs are prepended with "target" model activations. The gradients flow through these activations to the weights of the "target" model, updating its weights accordingly.

The authors demonstrate effectiveness of both aspects of the method.

### Strengths
The paper is well written and illustrated. Evaluation is performed on multiple tasks, and demonstrates the effectiveness of the method, at least for the tested Llama model.

The impact of the steering method might be significant, as it allows for targeted modification of model's behaviour without training on big datasets of demonstrations.

### Weaknesses
The main weakness of the paper is the lack of evaluation of different models. It shows the effectiveness of the method on the selected Llama model, but it does not tell if the method generalizes to different settings. It would be beneficial to see results for different model families, for example Qwen3 models, or steering capability on the gpt-oss model which is known for strong safety/alignment training.

Moreover:
- For scaling experiments, the authors only show the test loss values. It does not tell what is its impact on the downstream performance.
- The steering aspect of the paper lacks details on the training procedure -- how many update steps are performed during training? How big is the dataset used?
- There's an issue with hyperlinks in the appendix B -- text is rendered as "???"

### Questions
1. Does the method require the same model for the "decoder" and "target" models? Or is it enough to have matching residual dimensions?
2. How many update steps and dataset size is required for the steering application?
3. Does the method generalize to other model families? E.g. Qwen3 models.
4. For scaling experiments it would be good to include evaluation metrics for different scales, at least for the checkpoints with the lowest test loss
5. Does the method require activations from the full prompt? It would be nice to see an ablation on the parts of the prompt sequence -- for example the first, middle, and last 10% of the prompt passed to the decoder.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces LatentQA, an approach that interprets and controls LLM activations by training a LLM to answer open-ended, natural language questions about its own activations. The method collects a pseudo-labeled dataset that maps LLM activations to question-answer pairs and fine-tunes a decoder to perform question answering directly on these activations. The system is evaluated for both interpretability tasks—uncovering hidden system prompts and extracting latent attributes—and control tasks, including debiasing, persona steering, and eliciting behaviors from LLMs. The approach demonstrates improvements over existing probing baselines and shows favorable scaling properties with respect to both dataset and model size.

### Strengths
- The paper addresses a significant limitation of current probe models by training decoders that can return rich, open-ended natural language answers based on LLM activations. This extends the scope of interpretability to capture nuanced behaviors and model states that scalar or single-token probes cannot express.
- The work presents a detailed procedure for dataset curation, including control and stimulus prompts, masking strategies, and data augmentation, resulting in a comprehensive training framework for probing activations.
- Experiments cover diverse interpretability and control settings with strong performance gains.
- The method’s ability to not only interpret but also modify model behavior through natural language-specified loss gradients is well-demonstrated empirically, enabling complex steering such as harmful or benign persona manipulation, with clear experimental validation.

### Weaknesses
1. **Insufficient mathematical formalization of core mechanisms:** The paper presents the core ideas clearly at a conceptual level but lacks rigorous mathematical exposition in several critical areas. The patching mechanism for transferring activations from the target LLM's layer k to the decoder LLM's layer ℓ is described operationally but not formally defined. There is no explicit mathematical specification of the patch operation—whether it involves replacement, addition, linear transformation, or another operation. Issues such as dimensional mismatch, normalization procedures, and potential information loss are not addressed. This ambiguity is particularly concerning given that improper patching could introduce spurious correlations or reduce the fidelity of the decoded information.
Similarly, while the control mechanism is described as computing gradients of the decoder's log-probability with respect to activations, the exact loss formulation, regularization terms, and optimization procedures are not formally specified. The paper does not address how the method avoids degenerate solutions, spurious gradients, or mode collapse in complex control scenarios. A formal equation defining the loss function and an explicit mapping from activation perturbations to model updates would strengthen reproducibility and theoretical grounding.


2. **Unclear justification for activation masking and data augmentation design:** The paper introduces activation masking to prevent the decoder from directly reading control token embeddings (Design Decision 1) and employs three types of data augmentation—control, stimulus, and stimulus+completion (Design Decision 2). While these design choices are motivated intuitively, the underlying hypotheses—such as information preservation under masking or mitigation of shortcut learning—lack empirical or theoretical validation. An ablation study isolating the impact of masking strategies and comparing different data augmentation schemes would provide stronger evidence for these design decisions. Without such analysis, it remains unclear whether these choices are essential or simply one possible implementation among many.



3. **Limited evaluation on diverse model architectures and families:** All experiments are conducted exclusively on the Llama-3 family of models, with the decoder always initialized as a copy of the target LLM. This raises questions about the generalizability of the approach across different model architectures, training paradigms, and parameter scales. It remains unclear whether the patching mechanism and decoding strategies are architecture-specific or represent general principles applicable to other model families, or models with different attention mechanisms. The restriction to same-architecture decoder-target pairs also limits understanding of whether cross-architecture decoding is feasible or whether architectural alignment is a fundamental requirement.
 
 
4. **Lack of robustness analysis and reliability guarantees:** The paper fails to provide a systematic analysis of the decoder's reliability and failure modes, which is critical given the proposed applications in auditing and safety monitoring. While the authors briefly acknowledge potential hallucination issues, there is no empirical evaluation of when and how the decoder might produce unfaithful interpretations. The circular logic that “steering success implies correct interpretation” might be wrong, as the decoder could control behavior even if it understands the underlying representations only partly or incorrectly. Without confidence measures or a systematic way to identify failure modes, we can’t tell if the system truly reflects the encoded information or if it just produces a plausible hallucination.

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3
