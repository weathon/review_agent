# The Free Transformer

- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
We propose an extension of the decoder Transformer that conditions its
generative process on random latent variables. Those variables are
learned without supervision thanks to a variational
procedure. Experimental evaluations show that allowing such a
conditioning translates into substantial improvements on downstream
tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes to do variational training for decoder-only Transformers. Specifically, the authors introduce a simple way to add some encoder layers (Fig 2) such that the compute from the decoder can be largely reused. The authors also proposed to replace the KL term in the ELBO with a "KL governor", which assigns smaller variance to the posterior if the posterior mean is close to 0, maintaining the KL term at a constant. The authors trained a 1.5B and a 8B model and compared with decoder-only Transformers in a set of reasoning benchmarks.

### Strengths
1. As argued by the authors, variational training of latent-variable LLMs is an interesting and largely under-explored topic. 
2. The authors' proposal of latent inclusion and using KL governor to replace free bits in traditional VAEs are original. 
3. The improvement over baselines "on the generative tasks that put less emphasis on syntax, and measure performance either through the validity of generated code when it is executed (HumanEval+ and MBPP) or through exact matches of the expected response, which requires a complex reasoning (GSM8K)" is inspiring.

### Weaknesses
1. This paper seems to be written in a rush, with critical technical details missing. For example, in Fig. 2, the input to the encoder seems to be the KV from the middle layer of decoder, and the Q from "a trained token embedding $\zeta$ replicated to match the sequence length" (L211). However, how would these embeddings constitute "Full Self-Att" with those KVs from input tokens given that they are not even the same set of tokens? Do these $\zeta$ have positional embeddings? Another example is that when introducing the details of the "KL governor", which is the key contribution of this paper, the authors only vaguely mention "There is no analytical form for this σ(∥µ∥2), and we need to propagate the gradient through it. So, given the value of the hyperparameter κ we solve it numerically for values of ∥µ∥2 and fit a degree-eight polynomial to it." (L261-263) 

2. The empirical result is mixed, with minimum ablation missing. In the experiment section, the authors only reported the numbers on some reasoning benchmarks. Across all the free-bits value that the author swept over, none of the them outperforms baseline consistently. In fact, >5% degradation is very common in Table 1. IIUC, the baseline is just the autoregressive Transformer, without any RL-based thinking. In terms of ablation, the authors only briefly mentioned the proposed KL governor is more stable than directly optimizing KL (or its variant with free bits), without showing actual results. What if the whole training is deterministic and you still keep the clamp on $\mu$? 

3. Critical related work missing: Phan et al. are probably the first to discuss the relation between latent reasoning and variational inference, who definitely deserve their credits. Kong et al. also introduced variational training to LLMs, which seems to be the most relevant existing work.

Phan et al., Training chain-of-thought via latent-variable inference, NeurIPS'23

Kong et al., Latent Thought Models with Variational Bayes Inference-Time Computation, ICML'25

### Questions
The authors pretrained the proposed model, what is the perplexity? 

The reported mixed results seems to imply a zero-sum game (at a cost of >5% extra training compute), how would the authors postulate the generic usage of the proposed method?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a “Free Transformer,” a decoder-only transformer that is trained as a conditional VAE but keeps inference almost identical to a normal decoder. The author used two non-causal encoder layers to produce a tokenwise Gaussian posterior, and forces the KL between this posterior and a fixed standard normal prior to be a pre-chosen value. The experiment part provides 2 size of the models performance on several benchmarks.

### Strengths
I really appreciate the author for the simple and smart design, which includes minimal changes to the architecture. And according to the paper, the training and inference time doesn't affect a lot by the latent design. 

The model has a clear and explicit latent variable to control the generation, or probably perform some kind of reasoning in latent space. I think this is a great approach and "reasoning in the latent space" is a promising direction.

### Weaknesses
1. Model-related issue
* A big autoregressive decoder induces a complex, non-Gaussian posterior, but the paper uses only 2 non-causal layers to output a Gaussian posterior. This is a very narrow variational family and likely leaves an amortization gap. 
* I understand the author want to solve this posterior collapse issue using fixed KL. However, this approach must spend the same KL budget for all samples, so easy examples get unnecessary noise and hard examples cannot request more bits. This directly limits test-time reasoning on harder instances and is tightly coupled to the train–test latent mismatch, which contradict to the idea of this paper of adapting reasoning in different cases via latent variables. 
* Training always feeds structured, sequence-conditioned z at a fixed KL while the inference always feeds unconditioned N(0,I). In my understanding, in conditional generation, posterior should be used instead of prior. The paper does not show how much performance or diversity lost because of this. I hope the author can explain more on this part. 

2. Experimental concerns
* This part is my another major concern due to limited results and details/design included in the paper. 
* Since the author is studying a generative model, then density estimation and perplexity comparison should be included in the experiments. A basic validation perplexity of baseline GPT and other variational/diffusion based methods should be included. Because the method is based on ELBO, it is easy to see the comparison between different baselines in conditional and unconditional generation tasks. Without this, we cannot tell whether the method actually improves language modeling. 
* For the training data, the experiments specify only the token counts (47B, 200B) and the hardware, but not the underlying data distribution (sources, domain mix, code ratio, math ratio, filtering). Since the reported gains are largest on code, math, and CSQA, this is a confound. The authors should either (i) describe the training corpus in detail, or (ii) run ablations where the baseline and the Free Transformer are both trained on an explicitly defined subset (e.g. pure text, pure code) to show that the latent is the cause of the improvement, not the data mixture. 
* There is no concrete numbers to measure the training/inference efficiency. The paper claims “about 6–7% extra cost,” but it does not show per-token TFLOPs, tokens/sec, or memory usage compared to the baseline. Because the main selling point is “latent for free,” this should be quantified. 
* Also, more experiments like prior sampling and diversity results should also be added. The paper does not show this, so we cannot judge whether the learned latent space produces diverse, coherent generations. 
* More analysis on the behavior of this model should be studied. For example, I'm curious about which layers for adding z plays the most important change to the generation. Also, the author didn't show any results of changing z value could lead to different generation, especially the paper has a discussion of examples in the "motivation" section. These should be easily implemented in the ablation study.

### Questions
1. The encoder design is very simple, so I'm curious if the adding more layers to the encoder can bring better performance. From test-time scaling perspective, I'm expecting to see which factor of the design is important for reasoning capability. Is it still the model size?  If so, then the scaling behavior is the same as GPT based method. Are there any scaling in the latent space?
2. I'm really appreciate if the author can provides more details and experiments based on what I mentioned in the Weakness part. 
3. In real practice, how does the KL setting determined. I'm trying to understand if this is the bottleneck of this method. 
4. Is line 179 a typo?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes the "Free Transformer," a VAE-based decoder that conditions its generation on a latent variable $Z$. Its core novelty is the "KL Governor," a training mechanism that forces the $D_{KL}(Q(Z|S) || P(Z))$ to a fixed target value $\kappa$ rather than adding it as a loss penalty. This is intended to prevent posterior collapse and, by discarding the encoder at inference, adds no computational overhead. The authors report strong gains on generative tasks like HumanEval+ and GSM8K.

### Strengths
1. Motivation: The core idea is compelling: enabling models to use explicit latent "plans" rather than relying on purely "post-hoc" autoregressive token-level decisions.

2. Efficiency: The design is highly practical, incurring a small training overhead and zero inference-time cost.

3. Novelty: The "KL Governor" is a creative and new approach to managing the notoriously unstable VAE training objective.

4. Empirical Signals: The strong performance gains on complex coding and math reasoning tasks are significant and warrant attention.

### Weaknesses
The paper's central claim, while promising, rests on assumptions that would be significantly strengthened by further validation.

1. The Decoder is Not Constrained, and Its Usage of $Z$ is Unproven.

The paper's core thesis rests on the decoder using the latent $Z$. However, the proposed "KL Governor" only constrains the encoder to produce an informative $Z$; it does not place any direct constraint on the decoder. This leaves a critical question unanswered: it is unclear if the powerful autoregressive decoder truly learns to depend on $Z$ or if it simply relies on its own token-level context, effectively ignoring $Z$. This is a key aspect of the posterior collapse problem this paper aims to solve.

To validate this core claim, a critical ablation is necessary: an inference-time test with the latent variable disconnected (e.g., $Z=0$). If the decoder is dependent on $Z$, performance should collapse. Without this experiment, it is difficult to be certain that the reported gains are not just a side-effect of the VAE structure acting as a complex training-time regularizer, rather than the decoder actively using the latent "plan" at inference.

2. Inconsistent Performance and Unaddressed Regressions.

The performance gains appear to be highly task-specific. The paper highlights wins on generative tasks but does not address the notable performance regressions on several standard benchmarks (e.g., -8.46% on boolq and -5.21% on nq for the 8B model). A discussion of this trade-off—why the model might improve on reasoning while regressing on other tasks—is needed to fully interpret the results.

3. Unexplored Design Choices.

The paper introduces several specific design choices without justification or comparison.

(1) KL Governor vs. Alternatives: The "KL Governor" is novel, but the paper would be stronger if it were benchmarked against standard KL regularization methods like $\beta$-VAE or KL annealing, which also aim to prevent posterior collapse.

(2) Architectural Choice: The decision to inject $Z$ only at the middle layer is presented without an ablation or justification. It is unclear why this specific location was chosen over, for example, the input layer or all layers.

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes an LLM with one latent vector per token as sources of variation added to a middle layer of an otherwise-standard decoder-only transformer.  As in a VAE, at training time an encoder finds high probability samples of these variables conditioned on the observed text.  Several interesting architectural proposals improve the training efficiency and effectiveness of this model, including sharing the initial layers of the encoder with the initial layers of the decoder, and fixing the KL divergence rather than learning to minimise it.  Empirical evaluations on a large number of LLM benchmarks show promising results, suggesting that adding variance to internal representations of the transformer has added value over just adding variance to the token prediction step.

### Strengths
This paper contributes interesting insights into latent variable models for LLMs and a collection of effective methods for making the latent variables work as random choices about future outputs.  Simply adding randomness to hidden representations would not have the same effect, since during LM training these random choices would be independent of future outputs.

A central feature of this latent variable model is that there is one latent vector per token, which is novel with respect to the previous LLM models reviewed in this paper, but related to the work on nonparametric VAEs (Henderson and Fehr, ICLR 2023).  Combined with their method for fixing the KL divergence per token, this means that the number of bits conveyed by the latent variables grows linearly with the length of the generated text, which is essential for modelling the variability in language.

A large number of evaluations are run on reasonably-sized models trained from scratch.  These results succeed in showing the potential of this novel new direction for LLM architectures.

### Weaknesses
The paper seems to be work in progress.  There is very little discussion of the empirical results, and no ablation studies other than a standard LLM baseline.  The description of the model misses some key points (see below).

The experiments don't seem to be testing any hypothesis.  The paper reads like they have a cool idea, so lets see what happens. The conclusion is that something interesting happens, but it is not clear what.  This impression is reinforced by the lack of any ablation studies.  

As made clear in Figure 1, there is one latent vector per token.  But Figure 1 is never referenced in the text, and nothing in the specification of the model mentions this fact.  They even say that the prior is "standard Gaussian noise", but Gaussians are parametric distributions, while here the number of sampled values grows with the number of tokens in the text.  This is at best a misleading specification, and I would say incorrect.  In particular, the variable names violate standard naming conventions for vectors versus matrices and tensors.

### Questions
Please explain how Figure 1 is related to the specification in Section 3.  What are your variable naming conventions?

Please add the "batch size" to Figure 2.  The answer cannot be "we omit batch size", since this is an essential novelty of your approach over previous latent-variable LLMs.

### Soundness
2

### Presentation
3

### Contribution
3
