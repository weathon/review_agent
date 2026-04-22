# Visual Autoregressive Models Beat Diffusion Models on Inference Time Scaling

- Avg Score: 4.40
- Decision: Reject
- Scores: 6, 2, 6, 6, 2

## Abstract
While inference-time scaling through search has revolutionized Large Language Models, translating these gains to image generation has proven difficult. Recent attempts to apply search strategies to continuous diffusion models show limited benefits, with simple random sampling often performing best. We demonstrate that the discrete, sequential nature of visual autoregressive models enables effective search for image generation. We show that beam search substantially improves text-to-image generation, enabling a 2B parameter autoregressive model to outperform a 12B parameter diffusion model across benchmarks. Systematic ablations show that this advantage comes from the discrete token space, which allows early pruning and computational reuse, and our verifier analysis highlights trade-offs between speed and reasoning capability. These findings suggest that model architecture, not just scale, is critical for inference-time optimization in visual generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper explores using more capable search algorithms, primarily beam search (and also GTO, Greedy Token Optimization), on autoregressive image generative models, focusing on Infinity. It analyzes several verifier choices (e.g., ImageReward, LLaVA) and shows that beam search outperforms naive random search. It also compares to the scaling of a continuous diffusion model (FLUX.1-dev).

### Strengths
1. Using stronger search (beam search/GTO) for autoregressive image generators is a sensible direction.

2. The idea feels general and likely applicable across architectures. This reads like an initial step toward a useful approach.

3. There are helpful ablations on diversity (generation temperature) and dynamic budget allocation, presented in the paper.

### Weaknesses
**Major:** 

1. The paper makes general claims about discrete-token AR, but all experiments are on Infinity. Other AR families (e.g., Janus) are not tested, limiting the scope and generality of the claims.

2. L.245 argues that random search has an undesirable logarithmic growth and motivates beam-like algorithms. Then it should be shown that beam actually scales differently, not just that it wins at some budgets.

3. To me, comparing Infinity (AR) to FLUX (continuous diffusion) is hard to interpret. NFE isn't a fair cross-architecture proxy, and the comparison mixes different models and different search algorithms. If we assume matched compute, how do the two paradigms scale under random search only? Right now it's hard to tell whether gains come from the model or the search.

**Minor:**

1. In Fig.1, FLUX is shown at very few points (only two). The curve could saturate below Infinity or keep rising. more NFE points are needed to understand FLUX's trend, whether it is scaling better or worse than Infinity.

2. Methods like beam search can get stuck in undesirable subtrees. A short limitations discussion for the proposed methods in the context of AR image generative models would help.


3. GenEval and T2I-CompBench++ use compositional but simple prompts. A benchmark with longer, more detailed prompts (e.g., DPG-Bench) would provide valuable insights.

### Questions
1.  Is "LLaVA + Random" in Tab. 4 the same as "LLaVA" in Tab. 3? Some category scores differ, making the comparison to ImageReward a bit difficult. If they are the same, are results averaged over multiple runs or single-shot?

2. In Tab. 4, beam search uses fewer NFEs than random, thus the scores are close. If you match NFEs, how big is the gap between the search algorithms on any of the benchmarks (e.g. T2I-CompBench++)?

3. Did the higher-temperature images show visible characteristics (lower quality or higher diversity)? Not required, but a small visualization of beam paths would help a reader. Also, since ImageReward underweights spatial correctness, why not using LLaVA here to test whether increased diversity helps spatial tasks (regarding the claim in L.417)?

Please also address the points raised in the weaknesses section.

### Soundness
3

### Presentation
4

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
This paper discusses the problem of Test-Time Scaling (TTS) for image generation, comparing Autoregressive (AR) models and Diffusion models. The authors compare three TTS strategies—random search, greedy token optimization, and beam search—along with different verifiers, evaluating their performance on TTS. The conclusion is that employing beam search with a composite verifier can enable AR-based TTS to surpass large-scale diffusion models.

### Strengths
1.  This paper addresses an interesting topic: the comparison between AR and Diffusion models for TTS, and the potential advantages of AR.
2.  The paper is generally clearly written and provides reasonably thorough experiments on TTS for AR models.

### Weaknesses
1.  The experiments are conducted exclusively on VAR (Infinity). However, VAR represents just one specific instance of autoregressive image generation; other representative paradigms include LlamaGen and MaskGIT. The authors need to provide experimental results across different autoregressive paradigms to robustly support their claims.
2.  The experiments lack sufficient baselines. VAR-TTS shares a similar objective with this work, yet the authors do not provide comparative results. Furthermore, only one Diffusion model and one specific TTS strategy are included as baselines. These results are too limited to definitively claim that AR outperforms Diffusion for TTS. To substantiate the authors' viewpoint, experiments are needed to demonstrate that TTS on AR models, particularly with the proposed beam search, generally outperforms Diffusion models.
3.  Regarding the reporting of experimental results, the authors should report the total per-image generation time, inclusive of verification, to serve as a practical reference metric, rather than reporting only the NFE.
4.  A common challenge in TTS for image generation is that the verifier rewards obtained during the generation process can be inaccurate, while full verification only after complete generation is time-consuming. The authors need to clarify how this issue is addressed (or acknowledged if not resolved) in their framework.
5.  Beyond experiments, the paper also lacks a detailed theoretical or principled analysis explaining why TTS on AR models would be superior to Diffusion models.

In summary, the claim that TTS on AR models surpasses Diffusion models is interesting but strong. While the problem is intriguing, the authors need to provide more comprehensive evidence to support this conclusion. If the authors' response adequately addresses these points, I would be inclined to increase my score.

ref:  TTS-VAR: A Test-Time Scaling Framework for Visual Auto-Regressive Generation

### Questions
Refer to the Weaknesses section.

### Soundness
1

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
2

### Summary
This paper proposes a new test-time search method, beam search, to enhance autoregressive T2I models. The paper demonstrates the effectiveness and scalability of the proposed approach.

### Strengths
1. The proposed test-time beam search can greatly improve the pretrained model's performance, demonstrated by comprehensive experiments.
2. The paper introduces multiple verifiers that consider different perspectives on generation quality.
3. The paper shows that three of the proposed verifiers exhibit logarithmic scaling.
4. The paper is well-written.

### Weaknesses
1. The methodological contribution is somewhat limited, but I believe the extensive experiments in this paper compensate for this limitation.
2. All the experiments for the proposed beam search are conducted using Infinity. It would be better if the authors could provide additional results with other autoregressive models, as this would better demonstrate the generalizability of the proposed method.

### Questions
1. To my understanding, the ablation study of w (parallel number) and c (candidate number) is reflected in the number of images and NFEs. Am I correct?
2. In Table 3, the “Ensemble” is not the best-performing approach. Have the authors tried different ways to ensemble the verifiers, such as adjusting their weights?

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
This paper dives into inference-time scaling through search for image generation. 
The key finding here is that while recent attempts to apply search to diffusion models have not been particularly successful, this *is* incredibly successful for autoregressive models. Here, the authors show that beam search significantly improves image generation, enabling their 2B autoregressive model to surpass a much larger 12B diffusion model on common benchmarks. 
This paper work suggests that architecture (or architectural compatibility with search?), rather than just scale, is critical for inference-time optimization in visual generation.

### Strengths
The paper's core claim is clear and well supported by experimental results, showing that a model's compatibility with search can decisively overcome a 6x parameter deficit against SOTA diffusion models. Analysis showcases trade-offs between various varifiers, and breaks out comparisons across different capabilities within aggregate benchmarks.

### Weaknesses
The main efficiency metric uses "Number of Function Evaluations" (NFEs), but the paper says that NFEs for an autoregressive model and a diffusion model "are not directly comparable in FLOPs" which is a potentially noteable caveat. An NFE for one of 13 generation steps is not the same efficiency as for the noising denoising steps in a diffusion model. A more direct efficiency comparison may be needed.
- The method relies on an external verifier model to guide the search, and this verifier can be a massive bottleneck - the paper shows the best verifier for complex reasoning (LLaVA-OneVision) is 36x slower and requires 9x more GPU memory than the 3 lightweight alternatives. This means the total inference cost (generation + verification) could be substantially higher than just running the 12B diffusion model, even if the generation NFEs are lower.

The autoregressive vision model used here, Infinity, was chosen because it is a state-of-the-art autoregressive model but also because it  'fundamentally differs from traditional autoregressive image generation' (L140). The scale-wise generation reduces the number of tokens generated and makes the model more appropriate for beam search than other AR models which are not compared here. This to me suggests some light reframing of the claims that VAR models beat etc and that it is fundamentally a property of AR models vs specific instanitations of them that permit such improvements through search, unless additional comparisons can be added.

### Questions
How general is this to all AR models? The AR model used, Infinity, seems particularly well-suited for this approach because it generates in 13 progressive scales. It would be great to make clearer through comparison or clearer discussion what benefits could hold for any discrete AR model vs are special properties of multi-scale generation. The paper's conclusion may be over-generalizing from a specific AR architecture.

The paper notes that optimizing for one verifier can hurt other metrics, for example optimizing for aesthetics can hurt prompt adherence. The paper explores different verifiers, but are there other ways to mitigate this? For example, guiding search by other scores or internal model probabilities in addition to just the external verifier?

Systematic ablations are mentioned in the abstract, but I may have missed ablations within the paper - could you please clarify?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper takes an existing autoregressive image generation model (which does next _scale_ prediction on images), and applies various inference time search techniques to it, most notably using beam search. The authors show how beam search outperforms other things like greedy decoding or rejection sampling. Then the authors compare this to existing inference time search techniques for diffusion models and show that much smaller AR model can beat a larger diffusion model.

### Strengths
1. The paper provides a valuable empirical comparison of different search strategies within the context of scale-wise autoregressive models. It evaluates random search, greedy decoding, and beam search, demonstrating clearly that beam search provides the best trade-off between computational cost and performance.

1. The paper also provides evidence that guided search can fix specific, challenging compositional errors. The figures (e.g., Figure 1 and Appendix A) provide examples where the baseline model fails on spatial relations ("giraffe on the right of a wallet"), object counts ("six keys"), or attribute binding ("a green rose and a blue tulip"), while the beam-search-guided model produces the correct image.

### Weaknesses
1. The related work section doesn't capture the full space of diffusion model inference time scaling. For instance, the authors mention "In contrast [to diffusion models], language models benefit consistently from ... reward-model guidance". This claim is not necessarily true, Black et. al. 2024, Fan et. al. 2024 both have shown that you can consistently benefit diffusion models with reward-model guidance.

1. The paper's central claim of superiority over diffusion models  rests heavily on a comparison against the findings of Ma et al. (2025) , which reported limited benefits for search in continuous spaces. This is potentially a "strawman" representation of inference-time scaling for diffusion, and perhaps not the strongest baseline. Further, training data differences between the Infinity model and the Flux model tested could have been a contributing factor to the result, and decorrelating this was not done.

1. Applying beam-search to an existing image generation model is not particularly non-obvious or challenging. While the empirical evaluations and results are indeed interesting, it feels like this paper is more suited at a workshop-level.

1. Without a direct comparison of FLOPs or at least wall-clock time for generating a single high-quality image, the claim that the 2B AR model is more efficient than the 12B diffusion model is unsubstantiated.

1. There may be some overclaiming in the title, I think the paper makes a specific claim about "hierarchical scale-wise AR models," not AR models as a class.

### Questions
1. Your central performance comparison is between the 2B Infinity model and the 12B FLUX.1-dev model. How did you account for potential differences in their respective training datasets?

1. Given that beam search is a standard, well-established algorithm for autoregressive sequences, what do you consider the primary novel technical contribution of this work, beyond the (albeit interesting) empirical finding that it works well on a hierarchical image model?

1. The title makes a very broad claim about "Visual Autoregressive Models" as a class. However, your method's tractability relies entirely on the Infinity model's specific "next-scale prediction," which has only 13 sequential decision points. How would your findings apply to traditional raster-scan AR models, where beam search would be computationally infeasible? Shouldn't the paper's claims be scoped more precisely to hierarchical or scale-wise AR models?

### Soundness
2

### Presentation
3

### Contribution
2
