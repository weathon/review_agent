# LayerCake: Token-Aware Contrastive Decoding within Large Language Model Layers

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Large language models (LLMs) excel at natural language understanding and generation but remain vulnerable to factual errors, limiting their reliability in knowledge-intensive tasks. While decoding-time strategies provide a promising efficient solution without training, existing methods typically treat token-level and layer-level signals in isolation, overlooking the joint dynamics between them. In this work, we introduce a token-aware, layer-localized contrastive decoding method that aligns specific token types with their most influential transformer layers to improve factual generation. Through empirical attention analysis, we identify two key patterns: punctuation tokens receive dominant attention in early layers, while conceptual tokens govern semantic reasoning in intermediate layers. By selectively suppressing attention to these token types at their respective depths, we induce controlled factual degradation and derive contrastive signals to guide decoding. Our method requires no additional training or model modification, consistently improving factuality across multiple LLMs and various benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a new contrastive decoding technique (named "LayerCake") to improve the factuality of LLM generations. Extensive evaluations on benchmarks using various models (llama, mistral, qwen) show promising results of the technique with reasonable tradeoffs in the inference latency.

### Strengths
The experiments and evaluations are comprehensive and clearly convey the message of the paper. The writing and presentation of the methodology is also clear.

### Weaknesses
The method relies heavily on token type characterizations, additional hyper-parameters such as $\alpha, th_a, th_b, \beta$ and determining layers for stage 1,2 of interventions that are model specific. Since the authors have presented ablations but not guidance on how to select these values, (especially $\alpha$) in Eq(5), employing this technique can be cumbersome for unknown datasets or new models.

More importantly, this presents a situation where one has to tune a lot of hyper-parameters to obtain LLM outputs and verify if they are correct or not based on established benchmarks/verifiers. It would be beneficial if the authors discuss situations where verifiers are not available.

### Questions
1. What are the hyper-parameter values of $\alpha, th_a, th_b$ used for the results in Tables 1-6? There is an ablation in the Appendix but I couldn't find the $\alpha$ values. More importantly, is there a guidance for choosing them?

2. Regarding the comparison with SLED, I noticed that the Gemma-1B model performance reported in their paper is quite high (when compared to llama-3-8B) on TruthfulQA. Is it possible to have such an apples-apples comparison in this paper so that the readers have a clear picture of the effectiveness of LayerCake? I am asking because in the current paper, the authors show that the performance of Llama models on FACTOR with DOLA and SLED in Table 1 is lower than greedy, whereas SLED was shown to have better performance than greedy decoding for various gemma models. Clearing this confusion can be beneficial for future researchers.

3. For inference on an unknown dataset with long input sequences, how should one go about selecting the hyper-parameters? since the latency depends on $\beta$, what can be a guiding principle for choosing the optimal value?

4. Please fix the format of references (usage of "\ cite{}") throughout the text.

5. Figure 3(b) label: rename vallina -> vanilla

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes a variant of (intra-model) contrastive decoding, where the final next-token distribution is derived via contrast between the vanilla model output logits and the output logits of a lower-quality/perturbed model generation setting. In the method proposed here the low-quality token distribution is created by manipulating the attention within specific model layers, and this is done differentially for different categories of tokens (punctuation, concepts/content words, function words). The authors show some motivating analyses of attention behavior for different token types across different layers, and present experimental results that show how their decoding method leads to improved results for several models over multiple benchmarks, focusing on factuality, hallucinations, and QA.

### Strengths
* The general approach makes sense, that it is possible to identify more precise sources of "problematic" LLM behaviors, and use those to perform contrastive decoding that is more informed and targeted towards these areas (rather than a generic "weak model" vs. "expert model" scenario).
* The specific method of using attention interventions on specific token categories is novel as far as I'm aware.
* The results are presented over a wide array of benchmarks for different types of tasks, and show consistent and substantial gains across tasks and models.

### Weaknesses
1. The method relies on quite a lot of parameters and heuristics - I was not entirely convinced by the motivation for those, and at the same time the paper is somewhat unclear regarding the methodology for choosing them in practice. Specifically, there is $th_a$ and $th_b$, and $\alpha$; the choice of layers representing the "early" and "middle" stages of processing; a separate attention modification logic for punctuation and conceptual tokens; and the decision to perform contrast with each of those independently and then average the logits. Most of these decisions can be explained to some extent, but there is not really an overarching logic and we end up with a method that is quite complex, does not directly correspond to the analyses in the paper, and may require quite a bit of hyper-parameter optimization to work well (see Q1 below). Regarding the intuitions, several of the motivating statements are a bit vague and not sufficiently backed up by experiments/citations, like "such interventions can disrupt the model’s internal processes at different stages and thus serve as a mechanism to induce hallucinations" (l. 278), "C-type tokens are more likely to trigger hallucination issues" (l. 300), or "…enabling us to adaptively identify layers where the model begins focusing on semantic information" (l. 313).
2. The paper narrative is very focused on hallucinations and factuality, but there is not too much analysis that shows whether the gains are indeed related to hallucinations, or that the attention interventions relate specifically to model hallucination behaviors, as opposed to a general degradation/improvement in generation quality. One option in my view is to change the framing of the paper to be a bit more generic, that would also work better with parts like §5.6 that talk about generation quality and not hallucination. An alternative approach would be to keep the current narrative but add analyses and examples that show some specific connection to model hallucinations. I thought the analysis in §5.3 is good in this respect, ideally I would have liked to see more like that throughout the paper (e.g., analyses across datasets, examples of base model hallucinations and how they are resolved with the method etc.).
3. Regarding the scaling factor $\alpha$ (Eq. 2 / Eq. 5) I think something does not add up here with the formulation. In the original contrastive decoding formulation, we have $log(p_{orig}) - log(p_{mod})$, which is equivalent to $log(p_{orig}/p_{mod})$ - thus the score of each token reflects the _ratio_ between the probabilities $p_{orig}$ and $p_{mod}$ for this token. In my opinion multiplying just one side of the subtraction by a factor is not a straightforward way to scale the ratio/contrast, and moreover achieves the opposite effect from the one described in the text: $(1+\alpha)*log(p_{orig})$ is equivalent to $log(p_{orig}^{1+\alpha})$, and since $p<1$, an $\alpha$ larger than one actually gives *less* weight to $p_{orig}$ relative to standard CD, not more.

### Questions
Questions:
1. How were the hyperparameters $th_a$ and $th_b$ chosen? Was there a development/test split? Were the same parameter values used for the different benchmark datasets? Also re the layer interval choice - the values are specified in the paper but I am not sure how they were selected.
2. Do you have an explanation for the poor performance of the other approaches on Factor? The numbers here for DoLA and SLED here are very different from those reported in the SLED paper for the same models.

Additional comments/suggestions:
* Another relevant work re the attention on different tokens is Yu et al. 2024, "Unveiling and Harnessing Hidden Attention Sinks". Also some classic interpretability works deal with attention on punctuation, parts of speech, etc., e.g., Vig & Belinkov 2019. "Analyzing the Structure of Attention in a Transformer Language Model".
* Most of the citations are not in the right format (latex \cite{} should be switched to \citep{})
* Ideally colors should be consistent between Fig. 3a and 3b
* There is not enough spacing below the caption of Figure 3
* §5.6: it can be more clearly phrased that the GPT-5 nano results are only in the appendix

Typos:
- l. 150 over-allocating focus -> over-allocate focus
- l. 207 observe several trends and patterns emerge -> observe several trends and patterns that emerge
- Figure 3b Vallina -> Vanilla

### Soundness
2

### Presentation
3

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
The paper proposes LayerCake, a training-free decoding method that builds a contrastive next-token distribution by targeted attention interventions at specific layers and on specific token types. Empirically, the authors observe that punctuation-like “structural” tokens dominate early layers, while “conceptual” tokens carry semantics in mid layers. Experiments on LLaMA-2/3, Mistral, and Qwen across several QA benchmarks show consistent improvements prior work on contrastive-decoding like DoLa.

### Strengths
Constructing a contrastive signal by purposefully inducing erroneous predictions through targeted interventions is an elegant idea. The intervention design, which selectively suppresses attention to specific token types at their most influential layers, creates a meaningful contrastive distribution that exposes how factual reasoning emerges within the model. The link between token category (e.g. structural vs. conceptual) and layer range (early vs. mid vs. late) seems well-motivated, and the resulting perturbed distributions seem to provide a good way to reweight base logits toward more truthful outputs.

The method is seems simple to deploy. It requires no finetuning or architectural modifications, operates entirely at decode time, and generalizes across different model families. This practicality adds to its contribution. It can be adopted immediately with existing checkpoints and standard inference stacks.

Empirically, the work is supported by a broad evaluation over diverse QA-style benchmarks and several model scales. The improvements are consistent and are complemented by a series of interpretability analyses that link changes in attention allocation to model behavior. The “structural → semantic grounding → consolidation → final alignment” pattern offers an interpretable mental model for where factual reasoning resides within the network. This staged progression provides a solid rationale for why the specific intervention sites are effective.

### Weaknesses
Most evaluations focus on short-form, question-answering style tasks, which raises questions about the generality of the approach. It remains unclear whether the approach would perform equally well for more open-ended forms of text generation such as long-form summarization, creative writing, or code synthesis. These tasks involve richer discourse structures, longer contexts, and a more complex interplay of coherence and factuality than typical QA settings. The specific intervention strategy: emphasizing punctuation in early layers and conceptual tokens in mid layers; may implicitly rely on the structure of QA prompts. Extending the method to settings with different input-output formats or reasoning demands would significantly strengthen the generality claim.

The reliance on heuristic token classification and manually chosen thresholds introduces another limitation. Decisions such as which tokens to suppress and where to intervene across depth are defined through fixed rules rather than being learned or inferred from the model’s own behavior. This could limit robustness and portability across architectures or tokenizers. A more systematic sensitivity analysis, or ideally an adaptive mechanism that infers token-layer associations directly from attention statistics or model internals could be an interesting extension of the work.

From an engineering perspective, the approach introduces nontrivial inference overhead. The contrastive decoding step requires two forward passes per token, effectively doubling compute time. Practical deployment would probably require techniques like batched contrastive evaluation, layer caching, or partial reuse of intermediate activations. A more detailed discussion of such optimizations would help establish the approach’s feasibility for real-world use.

Finally, it would be valuable to disentangle the contribution of the interpretability-guided design from the general contrastive decoding framework. Comparing the proposed strategy against variants that apply random or uninformed interventions, e.g., such as suppressing attention at arbitrary layers or token subsets, would reveal whether the improvement truly arises from interpretability insights or merely from the presence of a contrastive perturbation.

### Questions
How well does the approach generalize beyond QA to settings like summarization, long-form reasoning, or code generation?

Could the token categories or layer boundaries be discovered adaptively, for example from attention statistics or intermediate states rather than fixed heuristics?

How sensitive are results to prompt style or language? Would the method still work if structural cues like punctuation were removed or heavily altered?

The authors can improve my opinion about the work by convincingly addressing my questions and weaknesses raised.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper is another contrastive decoding paper that resembles the idea of DoLa to improve factuality. They didn't contrast between layers, instead, they weaken attention to punctuation tokens in early layers and to conceptual tokens in middle layers to make a messed-up output distribution. And the do contrastive decoding with the original final layer output distribution and the messed-up output distribution. It seems works on models like LLaMA, Mistral, and Qwen, improving factual accuracy by around 3-6%.

### Strengths
- Experiment results on models like LLaMA, Mistral, and Qwen seems to improve factual accuracy on several benchmarks.

### Weaknesses
- The paper introduced so many hyperparams: α、tₕₐ、tₕ_b、β、layer ranges in the method, however, it's unclear how the authors find the hyperparams. The only clear thing is:

> We set β = 0.1 throughout the paper.

Otherwise, the authors mentioned that 

> thresholds tₕₐ, tₕ_b, and α are determined empirically

 but there is no explanation or details for it. What do you mean by "determined empirically"? It's possible that the author is adjusting the hyperparams based on each individual test set performance, which will be not allowed. If it's not the case, please provide the way that you select the hyperparams, for example, list the dev set performance and prove that you're choosing the best hyperparams on dev set and it transfers well to the test test. I suspect that there could be sensitivity in hyperparams between different tasks as the method is generally complex than the previous decoding method, especially when there are so many parameters to tune, so it may not be generalizing well.
- The method requires access to the attention weights and modifying the attention weights, which is not feasible for optimized attention layer like FlashAttention2, which is commonly used in inference engines. In practice it maybe not be very useful in real production. In contrast, methods like DoLa doesn't need access or modifying attention weights.
- The author should replace all \citet{} into \citep{} when citing refereneces, it's not standard to use \citet everywhere and it affects the reading experiences.

### Questions
Can you analytically show how you determine the thresholds tₕₐ, tₕ_b, and α? It's not valid to say vague things like "determined empirically" in a research paper. If it was tuned on the test set, it's not allowed. If it's tuned on the dev set, show me the numbers on the dev set and the details as a proof.

### Soundness
2

### Presentation
2

### Contribution
1
