# WatchLog: From a Glimpse to Decision—Rapid Event Reasoning in Endpoint Detection and Response Logs with Multimodal LLMs

- Decision: Reject
- Scores: 4, 6, 6

## Abstract
Endpoint Detection and Response (EDR) systems are essential for detecting malicious activities on endpoint devices, yet existing approaches struggle to efficiently process ultra-long log sequences and provide interpretable reasonings for security analysts. This paper presents \textbf{WatchLog}, a framework that models raw logs as video-structured representations to enable efficient video-language modeling of endpoint behaviors. Specifically, each event is encoded into a key-value guided image, and the resulting images are temporally arranged into a video-structured sequence. A temporal cross-attention mechanism then performs pixel-wise temporal aggregation, producing compact sequence embeddings that preserve behavioral fidelity while reducing computational cost. We conduct two-stage pre-training followed by supervised fine-tuning to generate behavioral explanations grounded in the semantics of event sequences and final judgments. Experiments on our newly constructed EDR8M-20R dataset demonstrate that WatchLog achieves higher detection accuracy and recall than the state-of-the-art baselines, while also generating reliable reasoning traces and enabling more efficient inference. Furthermore, our real-world application of WatchLog has validated its efficiency, effectiveness, and strong generalization capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces WatchLog, a novel framework designed for malicious behavior detection and reasoning in Endpoint Detection and Response (EDR) logs. Traditional EDR systems face challenges in efficiently processing long, complex log sequences and providing interpretable reasoning for security analysts. WatchLog addresses these issues by transforming raw logs into video-structured representations, using a multimodal large language model (LLM) approach. It incorporates a three-stage framework where events are first encoded as images, then temporally aggregated into video-like sequences, and finally fine-tuned to generate both attack family labels and human-interpretable rationales. Experiments on the newly constructed EDR8M-20R dataset show that WatchLog achieves high detection accuracy and recall, while also producing reliable reasoning and significantly improving inference efficiency.

### Strengths
1. The approach of transforming raw EDR logs into video-structured representations is highly innovative. By encoding individual log events as images and aggregating them temporally, the model can efficiently process long, high-dimensional sequences and capture both event-level semantics and temporal dynamics.
2. WatchLog outperforms state-of-the-art baselines in both detection accuracy and reasoning quality. It achieves near-perfect binary accuracy (99.8%) and a recall of 100%, while also generating high-quality rationales that are coherent, consistent, and complete.
3. The introduction of a temporal cross-attention mechanism to model long sequences effectively is a significant strength. This mechanism helps preserve the temporal dependencies between log events while reducing the computational cost of processing long logs.

### Weaknesses
1. The transformation of logs into video-like structures and the use of large multimodal models introduce significant computational overhead. While the framework reduces redundancy through temporal aggregation, the overall cost could still be high, especially when processing large-scale, real-time logs in a production environment.
2. Although WatchLog demonstrates good performance on the EDR8M-20R dataset, which is large, there is still uncertainty regarding its scalability to even larger datasets or real-time systems where the log sequences are extremely long. The computational bottleneck remains an issue, particularly when handling ultra-long logs typical in enterprise environments.
3. While the paper compares WatchLog with several baselines, it lacks a detailed comparison with other advanced EDR systems. A more thorough analysis against other multimodal or transformer-based models in real-world EDR environments could provide a clearer understanding of the model's strengths and weaknesses.

### Questions
Please refer to the weaknesses.

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
```markdown
This paper presents WatchLog, a three-stage multimodal framework that converts structured EDR logs into event-level image-like tensors (via a kvEmbedding), stacks them into a video-like tensor, applies pixelwise temporal cross-attention to compress T events to t condensed frames (t ≪ T), and feeds the resulting compact representations to a multimodal LLM to produce an attack-family label plus a natural-language rationale. The authors introduce the EDR8M-20R dataset and report strong detection and rationale-quality results, claiming large gains over several traditional ML methods and LLM baselines. Training proceeds in three stages: image↔event pretraining, video↔log pretraining, and supervised fine-tuning.
```

### Strengths
```markdown
- Novel architectural framing: encoding structured key–value event data into a `c × h × w` tensor and applying spatial+temporal attention is an interesting and original way to reuse video–vision-language machinery for ultra-long log sequences.
- Clear staged training recipe that separates event grounding, temporal aggregation, and supervised reasoning — this is a sensible decomposition.
- Scalability focus: the temporal compression design and empirical profiling (GPU memory/latency) address an important practical challenge for long-context EDR analysis.
- Thorough ablation of some components (temporal compression length, alignment pre-training presence).
```

### Weaknesses
```markdown
1. Core conceptual justification: “image/video” framing may be an implementation of token compression rather than a genuine multimodal/visual inductive bias.
   - Concern: Mapping tokens into a ViT hidden-dimension tensor (`c × h × w`) and applying ViT/cross-attention could be equivalent to learned projection + pooling in the NLP domain. Without evidence of visual/structural priors (locality, patch structure) helping, the VLM framing risks being a different parameterization of token compression.
   - Required experiments/analyses:
     - Compare WatchLog to strong non-visual baselines that perform token/event compression with similar compute and parameter budgets:
       - Hierarchical transformer or attention pooling that compresses T events to t vectors, then feeds those to the same LLM.
       - Perceiver/Perceiver IO, Set Transformer, Linformer, BigBird/Longformer sparse-attention adaptations.
     - Ablate spatial layout:
       - Shuffle spatial patch arrangement (randomize patch order) and report performance; if performance is unchanged, spatial layout is not contributing a visual inductive bias.
     - Ablate initialization and pretraining:
       - Try initializing the ViT with visual pretraining (ImageNet) vs random init; if visual pretraining helps, this supports the “visual” claim.
     - `kvEmbedding` ablations:
       - Replace with mean, max, or standard attentive pooling; report impacts.
     - Diagnostics:
       - Visualize attention maps / pixelwise cross-attention across time to show whether spatial channels consistently encode field-level semantics.
     - Comparative scaling curves:
       - Plot performance vs compressed length t for both ViT pipeline and a comparable NLP-compression baseline to demonstrate different scaling behavior.
   - Rationale for asking these: these will clarify if the main contribution is truly exploiting vision-language inductive biases or simply proposing an effective token-compression pipeline implemented with ViT layers. If the latter, authors must reframe claims accordingly.

2. Fairness and interpretability of baseline comparisons — unusually high false alarm rates for traditional baselines on `EDR8M-20R`.
   - Concern: Table 2 shows traditional methods with very high false alarm rates (65–100%). The manuscript does not analyze why these methods degrade so much on `EDR8M-20R`, making it hard to tell if WatchLog’s advantage arises from algorithmic superiority or dataset/experimental artifacts.
   - Required clarifications/experiments:
     - For every baseline, state explicitly whether it was fine-tuned on the same training data or evaluated zero-shot. Provide per-baseline training details (data used, epochs, hyperparameters).
     - Provide ROC and precision–recall curves and AUCs for binary detection baselines, not only single operating points. Describe how decision thresholds were selected (e.g., tuned on validation).
     - Provide per-family confusion matrices for baselines and WatchLog; include representative failure cases (sample logs) where baselines issue false alarms.
     - Sanity checks:
       - Re-run at least one baseline on an established external dataset and show reproduced/expected performance to rule out implementation bugs.
       - If baselines succeed on external benchmarks but fail on `EDR8M-20R`, analyze dataset characteristics (class imbalance, templated artifacts, family similarity to benign events) that cause the drop.
     - Preprocessing parity:
       - Confirm that baselines receive the same input representation (same sampling of events, same field filtering, masking) and any sliding-window segmentation is reported and matched when applicable.

3. Use of GPT-family models for generating training rationales and for automated evaluation introduces circularity and bias.
   - Concern: The authors use an LLM to produce event descriptions, log stories, and SFT rationales (Appendix E) and use GPT-5 as the automated evaluator for rationale quality (Appendix G). This raises two issues:
     - Training and evaluation are potentially correlated: the model may learn to mimic GPT-produced style and therefore score well under a GPT-based evaluator. This is circular and can overestimate genuine interpretability for human analysts.
     - Reliance on GPT for ground-truth rationales needs documentation: were GPT-generated rationales human-verified? If so, by whom and with what inter-annotator agreement?
   - Required clarifications/experiments:
     - Report the fraction of GPT-generated labels/rationales that were human-verified/corrected. Provide details about the human annotator pool (number, qualifications, compensation) and inter-annotator agreement metrics.
     - Add a human expert evaluation (even a small-scale study, e.g., 50–100 samples) of rationale quality, and report agreement with GPT-5 scores.
     - Evaluate rationale quality with at least one automated metric independent of GPT (e.g., ROUGE/L, BERTScore against human-written rationales) and report results.
     - If human verification was not performed, acknowledge limitations and present plans or partial analyses.

4. Dataset release, ethics, and potential dual-use considerations are inadequately detailed.
   - Concern: The paper claims `EDR8M-20R` is publicly released and contains over 8M events and 172 malicious families. Handling malware traces and potentially sensitive logs requires careful redaction, legal review, and a release policy to mitigate misuse.
   - Required clarifications:
     - Provide the exact release plan: what will be shared (raw logs, masked logs, derived features, images), distribution license, access controls (open vs gated), and redaction/sanitization protocols (how PII, hostnames, file paths, sample hashes are sanitized).
     - Provide ethics/institutional approvals (if any) and steps taken to ensure legal compliance.
     - Discuss dual-use risk mitigation (e.g., rate-limited access, licensing with restrictions for offensive use, vetting requester institutions).
     - Clarify whether the “expert-verified” labels are produced entirely from virtualized attack scripts or include any captures from real enterprise environments.

5. Reproducibility and statistical reporting
   - Concern: Several headline numbers (e.g., 100% recall, 99.8% binary accuracy) are very high and may indicate leakage or evaluation artifacts.
   - Required clarifications:
     - Report standard deviations across multiple runs (3–5 seeds) for key metrics. Provide confidence intervals or p-values for comparisons to baselines.
     - Release a simple reproducibility checklist and a single table listing all hyperparameters, pretraining steps, number of examples used per step, and total compute used for each stage.

6. Robustness and threat model
   - Concern: The authors do not test how robust their image/video encoding is to simple manipulations that an adversary could produce (e.g., adding benign noise events, truncating or reordering events, obfuscating key–value fields).
   - Required experiments:
     - Simple evasion tests: add benign noise events, random field swaps, or reorder events while preserving counts; report detection and rationale quality changes.
     - Permutation test: permute temporal ordering to evaluate whether temporal cross-attention is leveraging order-sensitive signals.
     - Adversarial perturbation: assess whether small edits to critical fields drastically change outcomes.

Minor / presentation suggestions
- Add per-family precision/recall and a confusion matrix for the main test set in the main paper (not only in appendix).
- Combine dataset and preprocessing details into a single, clear subsection (number of hosts, number of runs, how train/test splits were constructed to avoid leakage).
- Provide several negative examples: cases where WatchLog mislabels or provides incorrect/misleading rationales, with analysis of why.
- Provide the exact prompts used for GPT tasks in an appendix (some prompts are present, but ensure full prompts and examples are included).
```

### Questions
```markdown
1. Dataset and release:
   - Will `EDR8M-20R` be publicly released as raw logs, processed images, or only labels and summary statistics? If not fully public, can a vetted subset be released for reproducibility?
2. GPT-produced rationales:
   - Were GPT-generated event descriptions/log stories/rationales human-verified? If yes, what fraction and what was the inter-annotator agreement?
3. Baseline protocol:
   - For each baseline in Table 2, list whether it was fine-tuned on your training data or evaluated zero-shot. Provide hyperparameters/epochs used.
4. Visual priors:
   - Did you try visual pretraining for the ViT backbone (e.g., ImageNet) and compare to random init? Does visual pretraining meaningfully change results?
5. Spatial arrangement sensitivity:
   - What happens if the patch layout (`M = h×w`) is permuted across training and testing? Does performance drop?
6. Robustness:
   - Have you measured how small, structured perturbations (noise events, reordering, masking key fields) impact detection and rationale quality?
```

### Soundness
2

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
3

### Summary
The paper transfer the tradition EDR problem to a video understanding task, with the help of MLLMs, they achieve more accurate and efficent detection.

### Strengths
1. It's the first work that converts raw logs from the EDR system into video-structured inputs and utilize MLLM for end2end reasoning.
2. To solve the complex temporal relationship in log data, they add a temporal cross-attention block into Qwen2VL after the replaced kvEmbedding layer, further squeeze the tmporal information than the original Qwen model.
3. A solid Three-Stage Progressive Training, ensure the image-event alignment,  cross-Event (temporal) dependencies capturing and standard reasoing learning (by SFT).
4. good performance on their own test set and the Carbon Black Cloud subset of ATLASv2.

### Weaknesses
1. Data is derived from virtualized enterprise Windows environments, predefined attack scripts, and Windows kernel callback collection. Performance on other platforms and in different scenarios requires further validation.
2. GPT is used throughout the process of generating training data, alignment, and evaluation, raising questions about objectivity and reproducibility.
3. If WatchLog's explicit temporal compression proves advantageous, comparative experiments should be conducted on standard video models to validate its superiority over conventional video backbones.
4. Only test on their own dataset and Carbon Black Cloud subset of ATLASv2. Although it performs well on its test dataset, its generalizability remains questionable.

Also please check questions for detailed weakness explaination.

### Questions
1. Pre-training first uses GPT to generate event descriptions for alignment, which injects synthetic text styles and biases into the training distribution. The rationales during the SFT phase are also generated by GPT-5, and the final Reasoning Quality assessment is also scored by GPT-5. **Does this affect objectivity and evaluation accuracy?** Could you report the Reasoning Quality score using another strong model like Gemini-2.5 pro.
2. You can already convert log data into visual data (images and videos). **Why not try smaller non-LLM models (e.g. Visual Classifier: image-ViT, video - VideoMAE/I3D) for direct detection?** This is now changed to a classification task, you can just train a classification head to test, even considering the parameter scale advantage of LLMs, there are now large encoders like InternVL2-ViT-26B available.
3. If your goal contains leveraging the reasoning capabilities of LLM, I believe **it is essential to present the reasoning performance results of large-scale VLMs**. Specifically, in Table 2, you should include the results for Deepseek 671B and Qwen3 235B, even if the zero-shot model's reasoning quality is not good enough, their results may also surpass the 7/8B SFT model.
4. Why did you evaluate only on the Carbon Black Cloud subset of ATLASv2 for external validation? Do you have results on other ATLASv2 subsets or on the full dataset, and **could you evaluate on more public benchmarks?** As it stands, the 66.7% binary accuracy is relatively weak as evidence.  
Your internal data contain LLM-generated textual supervision; on external public datasets that lack such text, does the model’s performance degrade?

### Soundness
4

### Presentation
3

### Contribution
3
