=== CALIBRATION EXAMPLE 15 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract**
The title "LLM AS A CLASSIFIER: LEVERAGING LARGE LANGUAGE MODELS FOR TEXT AND VISION CLASSIFICATION" clearly reflects the paper's contribution. The abstract makes strong, specific claims about accuracy (62.7% on MIntRec 2.0, outperforming GPT-4o and GPT-5) and latency improvements (8× lower tail latency). These claims are central to the paper's value proposition and are supported in the results section. The abstract effectively summarizes the three key contributions: a unified single-token framework, latency improvements, and strong empirical results.

**Introduction & Motivation**
The introduction effectively motivates the problem: classification with LLMs is slow due to multi-token generation, while encoders are fast but lack flexibility. The gap for latency-critical applications is well-articulated. The core insight—treating classification as constrained generation with single-token outputs—is clearly stated. The three claimed pillars (accuracy, latency, generality) and the three contributions are well-aligned with the presented work. One minor point: the claim of enabling "zero-shot classification" (p.1) is slightly overstated based on the method; the model is fine-tuned on a large corpus, and "zero-shot" here refers to adapting to new label mappings without retraining, not to entirely unseen tasks. This should be clarified to avoid confusion.

**Methodology**
The methodology is clear, reproducible, and logically sound. The problem formulation (Sec 3.1) is standard. The use of LoRA for parameter-efficient fine-tuning is appropriate. The design of atomic special tokens is the core innovation and is well-explained in Sec 3.3. The rationale for single-token outputs (eliminating multi-step decoding, avoiding ambiguity) is convincing.
*   **Randomized Label Assignments:** This is a clever design choice to prevent token memorization and is crucial for the claimed generality. However, the mechanism for "zero-shot adaptation" needs more detail. At inference, if the mapping between a new class and a control token is changed, how is the model informed of this new mapping? Presumably, the class description in the system prompt is updated, and the model must infer the correct token from the restricted set based on this description. This process should be explicitly described in the main method section, not just in the appendix (A.6).
*   **Loss Masking:** The description of loss masking (computing loss only on the final output token) is clear. However, it would be helpful to explicitly state that the model is still generating a full sequence (including the prompt and the special token) autoregressively, but the training loss is only applied to the final token position. This distinguishes it from a true "one-forward-pass" encoder model.
*   **Proof/Correctness:** The method does not rely on complex proofs; its correctness is empirical. The design choices are justified.

**Experiments & Results**
The experimental setup is extensive, covering multimodal (MIntRec 2.0) and multiple text classification benchmarks. The use of a combined training corpus is sensible.
*   **Baseline Comparisons (Critical Weakness):** The most significant concern is the fairness of comparisons with GPT-4o and GPT-5. The LaaC models are **fine-tuned** on a large corpus (28k examples) that likely includes data from similar domains (e.g., intent recognition). In contrast, GPT-4o/GPT-5 are evaluated in a **zero-shot** manner. This is an apples-to-oranges comparison for accuracy. A fairer comparison would involve few-shot prompting of the proprietary models or, conversely, comparing LaaC's zero-shot performance on entirely held-out tasks. The paper attempts to address this with the "Alternative Prompting Baselines" (Appendix A.4), showing GPT-4o doesn't reliably output single tokens, but this doesn't rectify the core issue of differential access to training data. The comparison with encoder models (MAG-BERT, MulT) is more appropriate as they are also fine-tuned on the target task.
*   **Latency Measurements:** Latency gains are impressive and are a core contribution. Reporting P50 and P95 is good. The batch size scaling analysis (Appendix A.7) is valuable. However, the latency comparison with GPT models is also potentially confounded by API overhead, network latency, and unknown internal batching. While the orders-of-magnitude difference are compelling, a stronger case would involve comparing against locally run, open-source LLMs of similar size using standard multi-token prompting.
*   **Ablation Studies:** The paper includes useful ablations: effect of label-set size (Sec 4.6), model scaling (Sec 4.7), and benefit of text+multimodal training data (Sec 4.8). The zero-shot permutation test (Appendix A.6) is excellent and directly supports the generality claim. However, a key ablation is missing: **What is the performance drop if we omit randomized label assignments during training?** This would quantify the importance of that specific design choice.
*   **Statistical Significance & Sample Size:** For text benchmarks, evaluation on "200 randomly sampled test examples" (Sec 4.1) is modest. Accuracy differences of 1-2% (e.g., 95.0% vs 94.0%) may not be statistically significant. Confidence intervals or significance tests would strengthen the claims.
*   **Results Presentation:** Table 2 is cut off in the provided text, making it hard to fully assess. Figure 3's y-axis labels are missing (likely a parsing artifact), reducing clarity. The Pareto-style analysis (Appendix C) is a nice holistic summary.

**Writing & Clarity**
Overall, the paper is well-written and logically structured. The method is clearly explained. Some sections suffer from PDF parsing artifacts (e.g., broken figure references, garbled words like `~~p~~ath`, missing axis labels in Fig 3), but these are not the authors' fault. The core narrative is easy to follow.

**Limitations & Broader Impact**
The conclusion and future work section (Sec 5) briefly mentions limitations: focus on text/vision, need for analysis on calibration/robustness/multilingual generalization. This is good but could be expanded.
*   **Major Limitations:** The paper should explicitly discuss the **training cost and data requirement** of LaaC. While inference is fast, the method requires curating a large, diverse fine-tuning corpus and performing parameter-efficient fine-tuning. This is a non-trivial upfront cost compared to prompt engineering for a powerful API model.
*   **Comparison Limitation:** As noted, the unfair comparison with zero-shot proprietary models should be acknowledged as a limitation of the current evaluation.
*   **Label Space Limitation:** The method is limited to the pre-defined set of 500 control tokens. Scaling to thousands of classes might require modifications.
*   **Broader Impact:** The positive impact (enabling efficient LLM deployment for classification) is clear. Potential negative impacts are minimal but could include energy consumption from training and the risk of automating decisions in sensitive domains. The authors disclose LLM usage appropriately (Appendix A.1).

### Overall Assessment

This paper presents a simple, clever, and empirically effective method (LaaC) for fast classification using decoder LLMs. The core idea of mapping classes to single, randomized special tokens is novel in its specific implementation and demonstrates clear, substantial benefits in inference latency while maintaining strong accuracy. The experiments are extensive and generally well-designed, though the **comparison against zero-shot proprietary LLMs is a major weakness that undermines some of the strongest claims**. If this issue is adequately addressed (e.g., by reframing comparisons or adding fairer baselines), the paper makes a solid contribution. The work is relevant to ICLR, demonstrating a practical advancement in making large generative models usable for discriminative tasks in latency-sensitive scenarios. With revisions to correct the baseline comparison and provide missing ablations, this could be a strong candidate for acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes LaaC, a framework that adapts decoder-based large language models (LLMs) for classification by introducing atomic special tokens for each class and applying parameter-efficient fine-tuning (LoRA). This formulation reduces classification to a single-token generation step, yielding deterministic, low-latency inference while preserving the generative capabilities of LLMs. The method is evaluated on both multimodal (MIntRec 2.0) and text-only benchmarks, demonstrating competitive accuracy and order-of-magnitude latency improvements over GPT-4o and encoder-based baselines.

### Strengths
1. **Clear and impactful problem formulation:** The paper effectively identifies the latency inefficiency of multi-token generation in prompt-based LLM classification and proposes a simple, intuitive solution via atomic label tokens. The design guarantees O(1) decoding steps, directly addressing a critical deployment bottleneck (Sections 1, 3.3).
2. **Comprehensive empirical validation:** Experiments span diverse settings (multimodal and text-only) and model scales (4B to 27B parameters). Results show fine-tuned Gemma-3-27B achieves 62.7% accuracy on MIntRec 2.0, outperforming GPT-4o (43.7%) and GPT-5 (51.8%) while being >10x faster (Table 1). On text benchmarks, LaaC matches GPT-4o accuracy with 8x lower tail latency (Table 2).
3. **Rigorous analysis and ablations:** The paper includes scaling laws (Fig. 3b,c), label-set size effects (Fig. 3a), batch-size latency studies (Appendix A.7), zero-shot generalization tests via token permutation (Appendix A.6), and comparisons with alternative prompting baselines (Appendix A.4). These analyses strengthen the claims of efficiency, scalability, and robustness.
4. **Practical implementation and release:** The method is model-agnostic, uses parameter-efficient LoRA, and the code is provided (anonymous link). Training details are well documented (Section 4.3, Appendix A.3), enhancing reproducibility.

### Weaknesses
1. **Limited novelty of core idea:** Using special tokens for classification is a well-established technique (e.g., [CLS] in BERT, class tokens in T5). The adaptation to decoder LLMs is a straightforward engineering solution, and the paper does not sufficiently distinguish its contribution from prior token-based classification methods in encoder-decoder or autoregressive models.
2. **Unfair comparison with proprietary models:** The superior accuracy over GPT-4o and GPT-5 is reported while LaaC models are fine-tuned on a substantial multi-task corpus (28k examples) that includes data from the evaluation domains (e.g., MIntRec in training). In contrast, GPT models are evaluated zero-shot. A more equitable comparison would involve fine-tuning open-source baselines with standard prompting or constrained decoding on the same corpus.
3. **Misleading "zero-shot" claims:** The paper emphasizes zero-shot adaptation by reassigning label tokens at inference. However, the model is first fine-tuned on a large multi-task mixture. When evaluating on held-out text benchmarks (SST-2, etc.), this is a form of transfer learning from a multi-task fine-tuned model, not zero-shot from a pretrained base LLM. The base Gemma/Mistral models perform poorly without this fine-tuning (Table 1, 2).
4. **Incomplete discussion of limitations:** Key deployment concerns are not addressed: (i) the upper bound of 500 classes (Section 3.2) and scalability to thousands of labels; (ii) calibration and confidence estimation for single-token outputs; (iii) handling of out-of-scope or ambiguous inputs; (iv) multilingual performance drops when training only on English (Table 8).

### Novelty & Significance
The novelty is moderate: the idea of atomic label tokens for single-step classification in decoder LLMs is simple and incremental, but the paper's value lies in its systematic engineering and thorough empirical demonstration across modalities. The significance is practical: it shows that decoder LLMs, with careful adaptation, can match specialized encoders in speed and accuracy while retaining generative flexibility. This could broaden the use of LLMs in latency-sensitive applications. The work meets ICLR's bar for a solid engineering contribution with comprehensive experiments.

### Suggestions for Improvement
1. **Reframe comparisons:** Add a baseline where an open-source LLM (e.g., Gemma) is fine-tuned using standard multi-token verbalizers on the same multi-task corpus, to isolate the benefit of single-token outputs versus the multi-task fine-tuning itself.
2. **Clarify the "zero-shot" claim:** Distinguish between (a) zero-shot from a pretrained base model and (b) zero-shot from a multi-task fine-tuned model. Re-evaluate the text benchmarks using the base model (without any fine-tuning) to establish a true zero-shot baseline.
3. **Expand limitations and future work:** Discuss the 500-class limit, strategies for larger label spaces (e.g., hierarchical tokens), calibration methods, and out-of-scope detection. Also, address the computational cost of fine-tuning 27B models despite using LoRA.
4. **Provide more dataset details:** Specify the exact number of examples per dataset in the training corpus, and confirm no data leakage between training and evaluation splits for text benchmarks (e.g., SST-2).
5. **Include encoder-decoder baselines:** Compare with T5-style models that also use special tokens for classification, to better situate the contribution within the landscape of token-based classification methods.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Comparison with modern, fine-tuned encoder baselines on text tasks.** The paper only compares to BERT/RoBERTa-base with 16-shot linear heads, which is not a competitive SOTA encoder baseline. To claim LaaC "matches or surpasses encoder baselines," it must be compared to properly fine-tuned DeBERTa-v3 or RoBERTa-large on full training sets for SST-2, AG News, etc. Without this, the accuracy claim is not substantiated.
2. **Controlled latency comparison with optimized encoder pipelines.** The latency advantage over encoders (e.g., MAG-BERT) is attributed largely to the Swin Transformer feature extractor bottleneck. A fair comparison requires benchmarking against an encoder pipeline that uses an equally efficient visual encoder (e.g., a ViT distilled for fast feature extraction) or includes the cost of running the VLM's own vision encoder in LaaC's total latency. The current comparison is apples-to-oranges.
3. **Ablation on the necessity of single-token outputs.** The core claim is that single-token outputs are key for latency. An ablation is needed where the same model is fine-tuned to output *multi-token* label strings (e.g., "positive") with constrained decoding, measuring the resulting latency/accuracy trade-off. Without this, it's unclear if the gains come from the token design or simply from fine-tuning.
4. **Evaluation on a true zero-shot task with unseen label semantics.** The "zero-shot" text evaluation uses known datasets (SST-2, etc.) where label meanings (e.g., "positive") are common. To prove generality, test on a novel classification schema with entirely new, composite labels not seen during training (e.g., "enthusiastically sarcastic").

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of what the model learns vs. memorizes.** The paper uses randomized token assignments to prevent memorization, but no analysis proves the model uses the semantic label descriptions from the prompt. A simple probe: shuffle label *descriptions* at test time (e.g., map the description for "positive" to the token for "negative") and see if accuracy collapses. This is critical to trust the "generality" claim.
2. **Calibration and confidence analysis.** For real-time applications, well-calibrated confidence scores are essential. The paper provides no analysis of whether the single-token probabilities are calibrated or if they are over/under-confident compared to base LLM prompting or encoder softmax outputs.
3. **Failure mode analysis on multimodal tasks.** The method underperforms base Mistral on AG News (83% vs 84%). Why? Is it due to domain mismatch, label ambiguity, or the single-token constraint? A qualitative analysis of errors on MIntRec and AG News is needed to understand the method's limitations.
4. **Sensitivity analysis of the LoRA configuration.** The paper uses a fixed LoRA setup (rank 8). A minimal sweep on rank and alpha for one model/dataset is needed to show the results are not fragile to these hyperparameters, which is important for the method's practical adoption.

### Visualizations & Case Studies
1. **Visualization of attention for multimodal decisions.** For MIntRec, show attention heatmaps from the final token to image patches and text tokens for correct and incorrect cases. This would validate that the model is actually doing multimodal reasoning and not ignoring one modality.
2. **Case studies comparing LaaC, prompting, and encoder outputs.** For a few challenging examples (e.g., ambiguous sentiment, complex MIntRec intents), show the full prompt, LaaC's single-token output, GPT-4o's verbose output, and the encoder's prediction. This would concretely illustrate the latency/verbosity trade-off and any accuracy differences.
3. **Error case gallery.** A dedicated figure showing 5-10 representative failure cases across modalities, with hypotheses for why LaaC failed (e.g., missed visual cue, semantic ambiguity). This is crucial for assessing deployment readiness.

### Obvious Next Steps
1. **End-to-end latency benchmark including vision encoding.** The latency tables omit the cost of running the VLM's vision encoder (e.g., SigLIP). For a fair comparison with encoder pipelines, the total latency from raw image/video input to class token must be reported and compared.
2. **Experiments on a broader suite of modern classification benchmarks.** The text benchmarks are standard but old. To be convincing for ICLR, include results on modern, challenging benchmarks like MMLU (as a multiple-choice classification task) or more diverse intent datasets (e.g., CLINC-150 full dataset, not just a subset).
3. **Investigate the trade-off: single-token efficiency vs. loss of explanatory capability.** The paper dismisses verbose outputs as a drawback, but in many applications, explanations are valuable. A discussion or experiment on whether the model can be extended to provide a short justification (as a second, optional generation step) is a necessary analysis of the method's limitations.

# Final Consolidated Review
## Summary
This paper introduces LaaC (LLM as a Classifier), a framework that adapts decoder-based large language models for fast classification by mapping classes to atomic special tokens and using parameter-efficient fine-tuning. This reduces classification to a deterministic single-token generation step, yielding substantial latency improvements while maintaining competitive accuracy across text and multimodal benchmarks.

## Strengths
- **Clear and practical problem formulation:** The paper effectively identifies the latency bottleneck of multi-token generation in LLM-based classification and proposes a simple, direct solution via single-token outputs, directly addressing a critical deployment need (Sections 1, 3.3).
- **Comprehensive and rigorous empirical evaluation:** Experiments span multimodal (MIntRec 2.0) and multiple text classification benchmarks, model scaling (4B to 27B), and include valuable analyses: label-set size effects, batch-size latency scaling, zero-shot generalization via token permutation, and comparisons with alternative prompting baselines (Tables 1-2, Figures 3, Appendices A.6, A.7).
- **Substantial and well-documented latency gains:** The method achieves order-of-magnitude latency reductions over GPT-4o (e.g., 8× lower tail latency) and consistent speedups over base LLMs, with deterministic O(1) decoding steps—a core contribution for real-time applications (Tables 1, 2, 5, 6).

## Weaknesses
- **Unfair accuracy comparison with proprietary LLMs:** The paper reports superior accuracy over GPT-4o and GPT-5 on MIntRec 2.0 (62.7% vs. 43.7%/51.8%), but LaaC models are fine-tuned on a 28k-example multi-task corpus that includes data from similar domains (e.g., MIntREC), while the GPT models are evaluated zero-shot. This conflates the benefits of fine-tuning with the benefits of the single-token design, undermining the claim that the *method* itself outperforms these models (Section 4.5.1, Table 1).
- **Insufficient comparison with strong encoder baselines on text tasks:** Text benchmark comparisons use weak encoder baselines (BERT/RoBERTa-base with 16-shot linear heads and LM-BFF). To substantiate claims of matching or surpassing encoder efficiency and accuracy, comparisons with modern, fully fine-tuned encoders (e.g., DeBERTa-v3, RoBERTa-large) on full training sets are necessary (Section 4.5.2, Table 2).
- **Overstated "zero-shot" claims and lack of clarity on generality:** The paper emphasizes "zero-shot classification" and "zero-shot adaptation," but the model is first multi-task fine-tuned on a large corpus. Evaluations on held-out text benchmarks (SST-2, etc.) demonstrate transfer from this multi-task fine-tuned model, not zero-shot ability from a pretrained base LLM. The base models perform poorly without fine-tuning (Tables 1, 2). This conflation misrepresents the actual capability.
- **Missing analysis of calibration and confidence estimation:** For deployment in latency-sensitive applications, reliable confidence scores are critical. The paper provides no analysis of whether the single-token output probabilities are well-calibrated compared to standard prompting or encoder softmax outputs, which is a key consideration for real-world use.
- **Incomplete latency comparison for multimodal tasks:** The latency advantage over encoder pipelines (MAG-BERT, MulT) is partly attributed to their Swin Transformer feature extraction bottleneck. A fairer comparison would either include the cost of the VLM's own vision encoder in LaaC's total latency or benchmark against an encoder pipeline using an equally efficient visual backbone, to isolate the benefit of the classification method itself (Section 4.5.1, Appendix A.5).

## Nice-to-Haves
- **Ablation on the necessity of single-token outputs:** A controlled experiment fine-tuning the same model to output multi-token label strings with constrained decoding would help isolate the latency/accuracy benefits attributable to the single-token design versus the multi-task fine-tuning.
- **Failure mode and qualitative error analysis:** A brief qualitative analysis of errors on challenging examples (e.g., from MIntRec or AG News) could illuminate limitations (e.g., modality neglect, label ambiguity) and strengthen the discussion.
- **Brief sensitivity analysis of LoRA hyperparameters:** A minimal sweep (e.g., rank) on one model/dataset would demonstrate robustness and provide practical guidance for adoption.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **"Proof/Correctness" concern:** The harsh critic notes the method lacks theoretical proofs, but this is an empirical engineering paper; demanding formal proofs is out of scope.
- **Statistical significance tests for small samples:** While confidence intervals could be informative, single-run evaluation on sampled test sets is common practice for large-scale benchmarks; this is not a core flaw.
- **Request for comparison with T5-style models:** The balanced reviewer suggests comparing with encoder-decoder models like T5, but the paper's focus is on adapting decoder-only LLMs; this is not a required baseline.
- **Formatting nitpicks (PDF artifacts):** Comments about missing axis labels or garbled words (e.g., `~~p~~ath`) are parser artifacts, not author errors.
- **Demand for user studies or multilingual expansion:** These are outside the paper's stated scope of efficiency and accuracy for text/vision classification.

## Novel Insights
The core novel insight is the combination of atomic label tokens with randomized assignments during training, which forces the model to rely on semantic label descriptions rather than memorizing token identities, enabling some degree of zero-shot remapping at inference. This design, coupled with single-token constrained generation, provides a practical pathway to achieve deterministic, low-latency classification while preserving the generative backbone of decoder LLMs—a distinct trade-off point between encoder efficiency and LLM flexibility.

## Suggestions
- **Reframe comparisons with proprietary LLMs:** Either compare LaaC's fine-tuned performance against few-shot prompted versions of GPT-4o/GPT-5 (with matched exemplar counts) or clearly state that the accuracy advantage is contingent on access to fine-tuning data, separating the method's benefits from the advantage of fine-tuning.
- **Add a strong encoder baseline for text tasks:** Include results from a modern encoder (e.g., DeBERTa-v3) fine-tuned on the full training sets of SST-2, AG News, etc., to properly contextualize LaaC's accuracy and latency trade-offs.
- **Clarify the "zero-shot" terminology:** Distinguish between (a) zero-shot from a pretrained base model and (b) zero-shot adaptation from a multi-task fine-tuned model. Consider using terms like "cross-task transfer" or "label-remapping generalization" for the latter.
- **Include a brief calibration analysis:** Report calibration metrics (e.g., ECE) for LaaC's single-token probabilities versus a baseline on one representative dataset, and discuss implications for deployment.
- **Provide end-to-end latency breakdown for multimodal tasks:** Report the total latency for LaaC including vision encoding time, and compare it with an encoder pipeline using a similarly efficient vision backbone for a fairer efficiency comparison.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 0.0]
Average score: 1.3
Binary outcome: Reject
