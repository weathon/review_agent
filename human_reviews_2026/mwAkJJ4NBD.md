# Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Diffusion language models, especially masked discrete diffusion models, have achieved great success recently. While there are some theoretical and primary empirical results showing the advantages of latent reasoning with looped transformers or continuous CoT, continuous diffusion models typically underperform their discrete counterparts. In this paper, we argue that diffusion language models do not necessarily need to be in the discrete space. In particular, we prove that continuous diffusion models have stronger expressivity than discrete diffusions and looped transformers. We attribute the contradiction between the theoretical expressiveness and empirical performance to their practical trainability: while continuous diffusion provides intermediate supervision that looped transformers lack, they are harder to generate and decode tokens in the continuous representation space compared with discrete states. We therefore propose **C**oevolutionary **C**ontinuous **D**iscrete **D**iffusion (CCDD), which defines a joint multimodal diffusion process on the union of a continuous representation space and a discrete token space, leveraging a single model to simultaneously denoise in the joint space. By combining two modalities, CCDD is expressive with rich semantics in the latent space, as well as good trainability and sample quality with the help of explicit discrete tokens. We also propose effective architectures and advanced training/sampling techniques for CCDD, which reveals strong empirical performance in extensive language modeling experiments on real-world tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a hybrid framework that enhances the expressivity of discrete diffusion models (DDM) through an auxiliary loss head from a continuous diffusion model (CDM). Alongside the new CCDD framework, the authors also proposed several architectural improvements to bolster practical trainability. Furthermore, the authors rigorously analyzed the theoretical expressivity of various methods (CDM, DDM, Looped Transformer (LP)), which could yield valuable insights for future research.

### Strengths
The core concept of enhancing pure DDM training through auxiliary losses, specifically by integrating a continuous diffusion model within CCDD, is rather straightforward. The authors adeptly articulated the theoretical expressivity of various methods using rigorous mathematics and introduced novel network architectures (e.g., MMDiT, DiT with MOE) to improve CCDD's practical trainability. Ultimately, CCDD demonstrated good improvements in generation performance on the OpenWebText dataset.

### Weaknesses
1. The mathematical reasoning was somewhat challenging to follow, primarily due to two issues: 1) the definitions of some terms were not sufficiently clear. E.g., what do geometry and calibration refer to specifically? 2) the presentation could be enhanced by outlining the proof together with intuitive explanations. 

2. The paper has strengths in rigorous mathematical explanations and theoretical expressivity analysis. However, it lacks concrete evidence for some arguments, like the benefits of intermediate supervision in Section 3.2. Table 1 also contains typos (e.g., "Expressivity" and "Token-wise R^d") and vague comparisons (e.g., "low," "medium," and "high decoding ambiguity" need clear definitions and measures for consistency).

3. The technical aspects of increased training and inference costs were not adequately explored. In particular for inference, Algorithm 2 indicates that CCDD requires denoising with both DDM and CDM at each sampling step, yet CDM outputs appear to be discarded during inference.

4. The authors only evaluated CCDD on the OpenWebText benchmark, it’ll be desirable to extend the evaluation to other benchmarks to have a more comprehensive comparison with the baselines (e.g., MDLM, SEED in particular).

5. The authors provided some theoretical analysis of model expressivity. However, it is not clear that CCDD is more efficient or expressive than CDM and DDM in practice. This claim lacks sufficient evidence. Tables 2 and 3 compare model performance across different models, possibility with different network architectures (trainability). An additional ablation within CCDD can be more helpful. Specifically, with the same architecture, comparing pure DDM, pure CDM and CCDD. This will provide a clearer picture.

6. The model's expressivity might not be the most compelling argument, as increased theoretical expressivity doesn't always lead to better generation performance. For instance, even with fixed architectures and training algorithms (like CDM with consistent trainability), models with more parameters, while theoretically more expressive, are prone to overfitting. Instead, the real advantage of integrating Discrete Diffusion Models (DDM) with Continuous Diffusion Models (CDM), as seen in CCDD, likely lies in improving the representational efficiency of latent embeddings. This is because the CDM's auxiliary loss in CCDD can facilitate the learning of more efficient latent embeddings through continuous domain learning, which may mitigate information loss caused by logit quantization. Therefore, it seems more appropriate to focus on modeling efficiency rather than model expressivity.

### Questions
A few more questions, in addition to those raised in the Weakness section.
1. TC^0 and TC^1 classes are not defined throughout the text.

2. What is CCDD-MDiT in Table 2?

### Soundness
2

### Presentation
2

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
This paper compares different formalization of diffusion language models and draw connections to loop transformers, with a focus on the expressiveness. Through theoretical analysis, the author show that continuous diffusion language models can simulate loop transformers and being more expressive than discrete diffusion language models and autoregressive models. Motivated by this, the author propose to combine discrete diffusion and continuous diffusion to enjoy both expressiveness and trainablity.

### Strengths
1. The writing of this paper is clear and easy to follow. Showing theoretically convincing and intuitive discussions on the expressiveness comparison among discrete/continuous diffusion and loop transformers.
2. The discussion on expressiveness naturally motivate the method of enhancing discrete diffusions with continuous diffusions.

### Weaknesses
1. Despite fruitful theoretical discussions, the empirical evaluation is weak. The author only do some toy scales experiment (lm1b) and evaluate with metrics like perplexity.
2. Although claiming the advantage of implicit reasoning and more powerful than loop trnasformers, no empirical evidence are provided. The author should at least provide some experiments on conditional genration tasks like machine translation and reasoning tasks like sudoku as the author mention to support their claims. Perplexity does not reflect these improvement that the author claims.
3. Combining discrete and continuous diffusion is not a new idea, which has been studied in previous work like Diffuseq-v2. The author should include discussions on these related work.

### Questions
See above.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes Coevolutionary Continuous Discrete Diffusion (CCDD), a joint diffusion framework that denoises simultaneously in a continuous representation space and a discrete token space using a single time-conditioned model. The authors argue that continuous DLMs are strictly more expressive than discrete DLMs and can simulate looped transformers, yet underperform empirically due to larger decision spaces and decoding ambiguity; CCDD aims to marry continuous expressivity with discrete trainability. The method instantiates factored forward corruption (SDE on embeddings + CTMC on tokens) and factored reverse updates with cross-modal conditioning, plus practical techniques (architectures, CFG, asynchronous schedules). Experiments on LM1B/OWT show lower perplexity than re-implemented discrete baselines, especially when using Qwen3 contextualized embeddings; ablations relate cross-entropy (token decoding) and MSE (representation reconstruction) losses across embedding layers.

### Strengths
- Clear conceptual unification of continuous vs. discrete diffusion and looped transformers.
- Well-motivated trainability critique (large decision spaces; decoding combinatorics) and the joint design that conditions each modality on the other.
- Practical recipe (architectures, schedules, CFG) and nontrivial engineering to avoid leakage when computing ELBO/PPL.
- Competitive results on LM1B/OWT with careful tokenizer note and parameter accounting.

### Weaknesses
- Definitions are underspecified: “decision spaces,” “representation spaces,” and “combinatorial complexity in decoding” are invoked without precise formal definitions (Sec. 3.2). This weakens the causal link to trainability.

- The claim that CCDD “preserves full semantics in previous denoising steps” while masked DDMs “discard” them lacks a formal definition of “semantics” and empirical evidence isolating this factor.

- The ablation describing cross-entropy vs. MSE relationships is qualitative; the paper should define precisely which loss corresponds to which modeling component and how that supports the “combinatorial decoding” thesis.

### Questions
- Sec. 3.2 (lines 283–290): What are your formal mathematical definitions of the decision space and representation space used in your trainability analysis? Please also provide a theory- or evidence-backed comparison of CDMs vs. DDMs under these definitions (ideally with a lemma or controlled experiment).

- Sec. 3.2 (lines 295–298): How do you define “decoding featuring combinatorial complexity”? In the related experiments, what exactly do cross-entropy and MSE measure (reconstruction vs. likelihood surrogates), and how do they operationalize that combinatorial decoding difficulty? Please formalize the link.

- Sec. 4.1 (lines 348–351): You state that CCDD preserves “semantics in previous denoising steps” that masked DDMs “mostly discard.” What is the definition of “semantics” in DDM/CDM terms (e.g., information-theoretic quantity over histories)? If DDMs lack such a construct, is the “discarded” claim warranted? Provide evidence or rephrase.

- Sec. 5, Table 3: To assess trainability, please also report perplexity for both RoBERTa and Qwen3 embeddings. This is important for proving whether CCDD has better trainability, because if a language model with strong capabilities is used as a teacher, the trainability of any model will be high?

- Sec. 5, Table 3: Why is CCDD-MMDiT/Qwen3 worse than CCDD-MoEDiT/RoBERTa on OWT, given Qwen3 “provides more semantic information”? Please analyze whether capacity or other differences explain this.

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
4

### Summary
This paper proposes to address the gap between theoretical expressivity and practical trainability in diffusion language models (DLMs) and looped transformers (LTs). It first demonstrates theocratical justification that continuous diffusion models (CDMs) have stricter expressivity than discrete diffusion models (DDMs) (via non-atomic vs. finite-atomic marginals) and can simulate looped transformers (via PF–ODE numerical integration). The authors attribute CDMs’ poor empirical performance to challenges like large decision spaces and decoding ambiguity, while LTs suffer from OOD issues due to lacking intermediate supervision. To resolve this, they propose coevolutionary continuous discrete diffusion (CCDD), a joint diffusion process over discrete token space (CTMC-based) and continuous representation space (SDE-based). CCDD uses a single denoising model to leverage discrete tokens for trainability and continuous latents for expressivity. Experiments on LM1B and OpenWebText show CCDD reduces validation perplexity by over 25% compared to DDM baselines with the similar parameters budget.

### Strengths
1. The paper demonstrates notable originality by unifying three key language modeling paradigms (CDMs, DDMs, LTs) under a joint theoretical framework. This is a insightful combination of existing ideas that clarifies their relative expressivity and supplemented with empirical validation. 
2. CCDD addresses a critical practical limitation of CDMs (poor decoding) by integrating discrete tokens, while retaining CDMs’ potential of latent reasoning capabilities. This could advance DLMs toward broader use in complex tasks where AR LLMs struggle.
3. Clarity is strong, especially the table 1.

### Weaknesses
1. The experimental scope is limited - this is a major one. While results on LM1B/OWT (ppl) are compelling, the paper lacks evaluation on complex reasoning tasks (e.g., Sudoku, coding or even GSM8k) that it claims DLMs excel at. This makes it hard to verify if CCDD’s so-called latent reasoning (via continuous space) translates to better task performance beyond language modeling. 
2. Minor. The analysis of CCDD’s architecture variants (MDiT/MMDiT/MoEDiT) is underdeveloped: Table 2/3 only compare performance but not efficiency (e.g., FLOPs per step, inference time), which is critical for justifying architectural choices (e.g., why MoEDiT balances parameters and performance).
3. There is no comparison to loop transformers.

### Questions
1. Could you add experiments on complex reasoning tasks (e.g., Sudoku, code generation) to show how CCDD’s latent continuous space improves reasoning compared to DDMs/LTs? For example, do CCDD’s samples have more coherent intermediate steps in chain-of-thought (CoT) tasks?
2. Have you tested CCDD with random continuous embeddings (instead of Qwen3-Embedding)? A comparison would isolate the contribution of joint diffusion from the pretrained latent space’s knowledge.
3. In the asynchronous noise schedule, you set continuous space’s information decay slower than discrete space. Could you quantify how this choice impacts performance? For example, what happens if decay rates are reversed or synchronized?

### Soundness
2

### Presentation
3

### Contribution
3
