# SHIELD: Suppressing Hallucinations In LVLM Encoders via Bias and Vulnerability Defense

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 2

## Abstract
Large Vision-Language Models (LVLMs) excel in diverse cross-modal tasks. However, object hallucination, where models produce plausible but inaccurate object descriptions, remains a significant challenge. In contrast to previous work focusing on LLM components, this paper is the first to trace LVLM hallucinations to visual encoders and identifies three key issues: statistical bias, inherent bias, and vulnerability. To address these challenges, we propose SHIELD, a training-free framework that mitigates hallucinations through three strategies: re-weighting visual tokens to reduce statistical bias, introducing noise-derived tokens to counter inherent bias, and applying adversarial attacks with contrastive decoding to address vulnerability. Experiments demonstrate that SHIELD effectively mitigates object hallucinations across diverse benchmarks and LVLM families. Moreover, SHIELD achieves strong performance on the general LVLM benchmark, highlighting its broad applicability. Code is available at https://github.com/hukcc/SHIELD.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper identifies three encoder-level sources of object hallucination—statistical bias, inherent bias, and vulnerability—and proposes SHIELD, a training-free pipeline that integrates token re-weighting, token subtraction (noise-derived removal of erroneous features), and adversarially informed contrastive decoding to mitigate these issues. Extensive experiments on CHAIR / POPE / MME evaluations demonstrate consistent gains across several LVLM families.

### Strengths
1. Clear identification of encoder-level failure modes. The decomposition into statistical bias, inherent bias, and vulnerability is conceptually clean and well motivated by analyses (token peak-to-average ratios, dominant-object errors under noise, perturbation sensitivity). 

2. Strong token-level analysis. The paper’s quantitative and visualization evidence showing how individual visual tokens (peak norms / overemphasis) correlate with hallucination risk is compelling and valuable: it grounds the solution at the token representation level rather than treating hallucination as a black-box LLM issue. This token perspective is an important contribution for future encoder diagnostics. 

3. Modular, training-free design. SHIELD’s three modules are practical (can be combined or ablated) and the paper provides ablations showing complementary gains. Precomputable components (e.g., noise-derived estimates) make the approach more practical.

### Weaknesses
1.	Generative / open-ended hallucination benchmarks.
The current evaluation focuses mainly on binary or existence-based benchmarks (POPE, MME). To better demonstrate robustness under real generative scenarios, the paper should include results on open-ended hallucination benchmarks such as AMBER [1], MMHal-Bench [2], which assess caption-level fidelity and factual consistency rather than yes/no predictions.

2.	Comparison with token pruning or sparse-token methods.
It would be useful to compare SHIELD with token pruning or sparse-token selection approaches that also modify the visual token set before decoding.

3.	Ablation on precomputation and inference overhead.
Since the paper mentions precomputing noise-derived erroneous representations, reporting runtime and memory trade-offs between precomputation and online computation would make the efficiency claims more convincing.

4.	Broader hallucination metrics and human evaluation.
For open-ended captioning, automatic metrics can be unreliable. Including limited human or GPT-4–based evaluation [2] focusing on whether hallucinated objects are introduced or removed would enhance credibility and interpretability.

[1] Amber: An llm-free multi-dimensional benchmark for mllms hallucination evaluation, 2023.

[2] Aligning large multimodal models with factually augmented rlhf, ACL 2024.

### Questions
See weakness

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper identifies the visual encoder as a source of object hallucinations in Large Vision-Language Models (LVLMs), suggesting this stems from three issues: statistical bias, inherent bias, and vulnerability. The authors propose SHIELD, a training-free, post-hoc framework to mitigate these issues. SHIELD consists of three modules: (1) Token Re-weighting, which uses a naively generated caption to up-weight relevant visual tokens; (2) Token Subtraction, which estimates and removes a "bias" feature derived from noise inputs; and (3) Contrastive Decoding, which penalizes logits that are sensitive to a small adversarial perturbation on the input image. Experiments on several LVLMs (LLaVA, InstructBLIP, Qwen-VL) and hallucination benchmarks (CHAIR, POPE) show that SHIELD can reduce hallucination rates compared to other training-free methods.

### Strengths
- The paper's focus on diagnosing hallucination sources within the visual encoder is a valuable contribution. The categorization into statistical bias, inherent bias, and vulnerability provides a structured way to think about the problem, moving beyond the more common focus on the LLM component.

- The design of SHIELD is logical, with each of its three components clearly mapped to one of the three identified problems. This modularity makes the approach easy to understand.

- As a post-hoc, training-free method, SHIELD offers a practical advantage, as it can be applied to any pre-trained LVLM without requiring expensive re-training or fine-tuning.

- The experiments are thorough, covering multiple model families and benchmarks. The results demonstrate that SHIELD consistently improves over the vanilla models and outperforms other training-free baselines like VCD and OPERA in reducing object hallucinations, especially on benchmarks like POPE.

### Weaknesses
1. The primary weakness is the method's high computational cost at inference time. SHIELD requires multiple additional forward passes: one to generate the "naive caption," $K$ (e.g., 32) passes for the noise inputs, and another pass for the adversarially attacked image. This combined overhead likely makes the method prohibitively slow for any practical or real-time application, which severely limits its impact.

2. The method introduces several new and sensitive hyperparameters ($\alpha$ for subtraction, $\beta$ for contrastive decoding, $K$ for noise samples, and the attack parameters $l$ and step size). The paper provides very little ablation or sensitivity analysis for these. It's unclear how these were tuned or how performance would change on a different dataset or model, making the results potentially difficult to reproduce or generalize.

3. The idea of "inherent bias" being estimated by feeding pure noise to the encoder is a strong assumption. It is not convincingly demonstrated that the resulting feature vector is a meaningful representation of "bias" rather than just an artifact of the network's response to out-of-distribution data. Furthermore, using a large $K=32$ to get a stable estimate adds significantly to the computational burden.

4. The "Token Re-weighting" module fully depends on an initial caption from the vanilla LVLM, which is itself prone to hallucination. The paper's claim that this doesn't matter (because hallucinated words won't match visual tokens) seems overly optimistic and is not well-supported. A plausible hallucination could easily bias the re-weighting in the wrong direction, but this potential failure mode is not investigated.

### Questions
1. Can the authors provide a quantitative analysis of the inference-time latency (e.g., in seconds per image) and memory overhead of SHIELD compared to the vanilla model and the baselines (VCD, OPERA)? How much does each of the three modules (re-weighting, subtraction, contrastive decoding) contribute to this overhead?

2. The paper lacks a sensitivity analysis for the key hyperparameters $\alpha$, $\beta$, and $K$. How much does performance change when varying these? How were the final values chosen?

3. Regarding the "Token Re-weighting" module: What happens if the "naive caption" contains a plausible hallucination (e.g., seeing a "boat" in a wide shot of a lake)? Can you provide an analysis of this failure case?

4. Why was the specific PGD-style attack chosen for the "Address Vulnerability" module? How does the method perform if a different type of attack is used (e.g., FGSM) or if no attack is used at all (i.e., only using modules 1 and 2)?

5. For the CHAIR evaluation, the paper follows prior work using 500 randomly selected images. How is the balance between coverage (ensuring all present objects are in the ground-truth) and faithfulness (ensuring no non-existent objects are in the ground-truth) managed for this label set? This seems critical for a metric that punishes “hallucinated objects”.

6. For the GPT-4o assisted evaluation, why was a 0-10 human-like scoring system for “correctness” and “detailedness” chosen? Given recent work on automated evaluation (e.g., Valor-eval: Holistic coverage and faithfulness evaluation of large vision-language models. https://arxiv.org/pdf/2404.13874), what are the advantages of this 0-10 scale over more objective, feature-extraction-based or model-based evaluation methods?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper propose SHIELD, a training-free framework that mitigates hallucinations through three strategies: re-weighting visual tokens to reduce statistical bias, introducing noise-derived tokens to counter inherent bias, and applying adversarial attacks with contrastive decoding to address vulnerability.

### Strengths
1. This paper propose SHIELD, a training-free method that mitigates object hallucinations by reducing statistical bias via token re-weighting, alleviating inherent bias using token subtraction, and addressing vulnerability through contrastive decoding.
2. Comprehensive experiments validate SHIELD’s effectiveness in mitigating object hallucinations
across diverse benchmarks and multiple LVLMs.

### Weaknesses
1. This method is limited to object-level hallucination. Comprehensive multimodal hallucination includes attribute level, relation level etc, as well. 
2. Although a training free method, the paper should present comparison with SOTA hallucination mitigation methods as well, either traning-free or training needed.

### Questions
See Above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper claims that object hallucinations in LVLMs largely originate in the visual encoder and proposes a training‑free, three‑module wrapper SHIELD. The method includes token re-weighting to balance attention across visual tokens, token subtraction using noise-derived features, and contrastive decoding that contrasts outputs from natural and perturbed images. Authors presents correlational evidence for encoder statistical bias, inherent bias, and vulnerability and reports gains on CHAIR, POPE, and MME across LLaVA‑1.5, InstructBLIP, and Qwen‑VL.

### Strengths
- The paper presents a reasonable motivation and a clearly described method.
- The proposed method is training-free, making it easy to integrate without retraining overhead.
- Experiments on LLaVA‑1.5, InstructBLIP, and Qwen‑VL showed improvement in reducing hallucinations.

### Weaknesses
- Encoder-level causes of visual hallucination have been extensively studied in earlier publications and the proposed methods are largely adapted from existing ideas. For example, strategies like token re-weighting and contrastive decoding are incremental given VCD has proposed such contrastive approaches.
- Related to the first point, there is a lack of discussion on directly related works [1-5]. There have been a lot of existing work on training-free contrastive methods for hallucination reduction.
- Experiments are done with outdated models. The selected three LVLMs were the very early versions, despite substantial advancements in multimodal capabilities since then. For example, newer models such as Qwen2-VL, Qwen2.5-VL, and Qwen3-VL have been released. As a result, the current experimental setup appears outdated and limited in scope.
- Improvements in Table 2 and 3 are small.
- Many citation formats in this paper are incorrect. For example, line 322-323.

[1] HALC: Object Hallucination Reduction via Adaptive Focal-Contrast Decoding
[2] Brave: Broadening the Visual Encoding of Vision-Language Models
[3] Mitigating Object Hallucination in Large Vision-Language Models via Image-Grounded Guidance
[4] Reducing Hallucinations in Vision-Language Models via Latent Space Steering
[5] Contrastive Region Guidance: Improving Grounding in Vision-Language Models without Training

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2
