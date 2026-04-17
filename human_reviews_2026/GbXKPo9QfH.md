# LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
Large Language Model (LLM) pretraining, finetuning, and evaluation rely on input-space reconstruction and generative capabilities. Yet, it has been observed in vision that embedding-space training objectives, e.g., with Joint Embedding Predictive Architectures (JEPAs), are far superior to their input-space counterpart. That mismatch in how training is achieved between language and vision opens up a natural question: {\em can language training methods learn a few tricks from the vision ones?} The lack of JEPA-style LLM is a testimony of the challenge in designing such objectives for language. In this work, we propose a first step in that direction where we develop LLM-JEPA, a JEPA based solution for LLMs applicable both to finetuning and pretraining. Thus far, LLM-JEPA is able to outperform the standard LLM training objectives by a significant margin across models, all while being robust to overfiting. Those findings are observed across numerous datasets (NL-RX, GSM8K, Spider, RottenTomatoes) and various models from the Llama3, OpenELM, Gemma2 and Olmo families. Code: \url{https://github.com/galilai-group/llm-jepa}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper, LLM-JEPA, brings the Joint-Embedding Predictive Architecture (JEPA) that was originally introduced in the ViT domain to LLMs (specifically, text-code generation LLMs). In the vision domain, JEPA self-supervisedly to learn to predict the representations of some target image patches, given some context image patches. The underlying assumption is that all patches represent the same thing. In this paper, LLM-JEPA self-supervisedly learn to predict the representations of the code, given prompt text, and the underlying assumption is that the prompt and the code represent the same thing. LLM-JEPA is done simply by adding additional predictor tokens and a prediction loss. The authors show that this method improves standard LLM metrics significantly without overfitting.

### Strengths
-	Although the JEPA idea is not new, bringing it to the LLM domain is novel.
-	The authors acknowledge the underlying assumption of JEPA: they require multiple views of the *same* information as in text-code generation, but also go further to explore how it generalizes to QA tasks.
-	The authors applied Loss Dropout and observed dual wins in saving compute and improving accuracy.
-	The writing flows naturally (especially with the help of the natural questions), and is a pleasure to read. Having pseudo code, implementation (eg, Line 152) is a strong plus.

### Weaknesses
-	Line 161: recommend adding a simple diagram to illustrate how the Pred() is added to an LLM transformer. The description is a bit confusing: beyond the standard transformer, is there a separate prediction network? Are the predictors also generated in k auto-regressive loops or one-shot?
-	Recommend showing a visualization of the mask in Line 188. Readers can imagine this looks like two upper-triangular matrices along the diagonal, but adding a diagram makes it more intuitive.
-	Table 1 has no description in the main text. Does it intend to show failure cases?
-	(Minor) The citation style does not follow the ICLR 2026 template https://iclr.cc/Conferences/2026/AuthorGuide (i.e., xxx et al.).

### Questions
-	In Eq.1, recommend noting somewhere what XEnt() stands for. Typically, people just write CrossEntropy. Similarly, does NTP stand for next token prediction? Please note it somewhere.
-	In Line 161, is the [PRED] added after the concatenated text-code sequence or in between? (as mentioned above, a diagram would simplify things here).
-	In Line 166, could the authors elaborate on what it means to use the embedding of the last predictor token? Is it the hidden_state? 
-	Line 170 typo: “lies in obtained” -> “lies in obtaining”?
-	In Line 190, could the authors explain which two forward passes? What are the purposes of each two?
-	In Figure 3, the blue curve is when k=0, but in Line 166, the authors say in this case the predictor is trivial. Does being trivial mean JEPA is turned off, or mean it is most effective?
-	In Figure 3, right side, we observe that even the NTP finetuning loss is almost the same at stabilization (yellow/blue), but at evaluation time their accuracy in fact differs a lot (51 vs 71). This is interesting. Do the authors have a hypothesis for the reason?
-	Line 292: Are the wrong/missing/extra labels compared against the GT or any valid regex? The proposed method’s output at line 292 has two additional `{}`.
-	Line 373: using L2 norm degrades accuracy from 71% to 2%, which is surprising how sensitive it is to the distance metric. Is this number correct?

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
3

### Summary
This paper proposes LLM-JEPA, a novel training objective that brings Joint Embedding Predictive Architectures (JEPA) into large language models. By combining the standard autoregressive next-token prediction loss with a JEPA-style embedding prediction loss, the model improves representation quality while maintaining generative capability.

### Strengths
1. The work systematically introduces JEPA into LLM training for the first time, bridging a long-standing gap between vision and language representation learning.

2. Demonstrates consistent empirical improvements across diverse architectures, scales, and datasets.

3. Provides clear analysis of embedding structure (t-SNE, SVD), showing how JEPA regularizes the representation space.

4. Maintains generation ability while enhancing abstraction and generalization.

### Weaknesses
1. Training cost is substantially higher due to multiple forward passes (≈2× compute), as the authors acknowledge.

2. Hyperparameter tuning (λ, k) appears expensive and unstable across settings.

3. The approach depends on multi-view data (e.g., text–code pairs), limiting applicability to generic text-only corpora.

4. The paper could discuss more about how JEPA objectives affect downstream reasoning or interpretability beyond accuracy metrics.

5. It would be interesting to see whether JEPA acts more like regularization or representation alignment—the current discussion is mostly empirical.

### Questions
1. Could JEPA be extended to pure text tasks using data augmentation (e.g., paraphrase or Q–A pairs)?

2. How sensitive is the method to the choice of distance metric (cosine vs. L2)?

3. During large-scale pretraining, how do you plan to manage compute cost — would loss dropout scale effectively to billions of tokens?

4. Did you observe any degradation in generation diversity or fluency due to the embedding-space constraint?

5. Can the JEPA loss be interpreted as encouraging semantic disentanglement or latent structure emergence?

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
This paper proposes LLM-JEPA, a method that augments standard autoregressive LLM training (next-token prediction) with a JEPA-style embedding-space objective that forces embeddings of two views to align via a predictor network. The JEPA term is added to the generative cross-entropy loss with a balancing weight λ, and the predictor is implemented by appending small numbers of special [PRED] tokens that reuse model weights. The authors implement a custom attention mask to obtain separate view embeddings efficiently, introduce random JEPA-loss dropout to reduce compute overhead, and provide a thorough empirical study across multiple model families (Llama3, Gemma2, OpenELM, OLMo). They show consistent accuracy gains, representation improvements (t-SNE, SVD, near-linear mapping from text to code embeddings), and robustness to overfitting; they also provide ablations on loss choices, predictor placement, and loss dropout.

### Strengths
1. The formulation (LLLM + λ · JEPA) and practical predictor design (tied weights via [PRED] tokens) are simple, elegant, and directly applicable to existing transformer models without architectural surgery.
2. Results include full fine-tuning and LoRA, pretraining experiments, several datasets (including code and QA tasks), multiple model sizes, statistical significance testing (five seeds), and useful visualizations (t-SNE, SVD) that support the claimed representational benefits.

### Weaknesses
1. Most experiments are fine-tuning or pretraining on relatively small/targeted corpora. While dropout amortizes cost, it remains unclear how LLM-JEPA behaves (performance, wall-clock, memory) at large pretraining scales used for modern foundation models.
2. The paper presents empirical evidence (SVD, near-linear mapping), but lacks deeper theoretical analysis or causal ablations that would clarify when and why the embedding alignment improves downstream generation and reasoning

### Questions
1. The paper reports grid search results and suggests keeping λ·(1−α) constant with dropout, but could you provide concrete heuristics or automated tuning strategies? How sensitive are gains to λ when moving to very different view types or sequence lengths?
2. The method uses the last-token hidden state as the embedding and appends [PRED] tokens as a tied predictor. Have you tried averaging representations, CLS-like tokens, multilayer predictors, or decoupled predictors? How robust are the results to these choices?

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
This paper proposes a novel training objective $\mathcal{L}_{\mathrm{LLM-JEPA}}$ for LLM training, aiming at introducing the successful JEPA insights, i.e., directly predicting from embedding space to absorb abstraction knowledge, from vision into the LLM studies, which mainly rely on the reconstruction ability on token level. LLM-JEPA recognizes the naturally paired data (e.g., text & code or Q & A in this paper) as different views of the same thing. The objective is to predict from the embeddings of one view to the embeddings of another view. Experiments show the performances of LLM-JEPA on both the reconstruction ability and predictability.

### Strengths
- The idea of introducing JEPA from vision to language is novel. Compared to the representation learning in LLM, which mainly focuses on obtaining good embeddings, LLM-JEPA doesn’t neglect its generative capabilities, and experiments show that the additional JPEA loss helps both pretraining and finetuning.
- I like the analysis in the experiment part, including showing that next-token prediction loss cannot optimize the JPEA objective and visualizing the t-SNE and the singular value to verify the strengths of LLM-JEPA.

### Weaknesses
- Though efficient, I doubt the usage of this method, especially as its strong dependency on multi-view data, which may limit the application to real-world industrial LLMs. Do you have some blueprints of the future work for more universal scenarios (e.g., unsupervised pretraining via data augmentation on texts that don’t have paired views)?
- Another weakness, based on my understanding, is the additional training complexity and hyperparameter search cost. From Fig. 7 in appendix, there is no clear law of the best $(\lambda, k)$ combination, which is also mentioned in the limitation part.

### Questions
- For the predictor, you append $k$ `[PRED]` tokens to reuse the model’s weights. Can you elaborate more on the design intuition of this setting, especially when $k>0$? Why sometimes a larger $k$ would be better?
- In Table 3, I find that InfoNCE surprisingly performs much worse than metrics. Do you have some explanations on this? Does it just mean the predictability is more crucial than the contrastiveness in this task?
- In this paper, both the text and code encoders are initiated with a same LLM with same weights. In prior practice in JPEA [1], the target encoder is an EMA of the context encoder weights, which can enhance the training stability. Can the EMA design be applied to LLM-JPEA?

## References

[1] Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. CVPR 2023.

### Soundness
3

### Presentation
2

### Contribution
3
