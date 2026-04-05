=== CALIBRATION EXAMPLE 15 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Does the title accurately reflect the contribution?**  
  Mostly yes: the paper is about using decoder-style LLMs as classifiers via single-token constrained generation, and it spans both text and vision tasks. However, the title is somewhat broader than the actual evidence: the paper mainly demonstrates intent/topic classification and a limited set of text benchmarks, not a general solution for “text and vision classification” across broad settings.

- **Does the abstract clearly state the problem, method, and key results?**  
  Yes. It identifies the latency problem with prompting/constrained decoding, proposes LaaC with atomic label tokens and PEFT, and reports headline numbers on MIntRec2.0 and text benchmarks.

- **Are any claims in the abstract unsupported by the paper?**  
  The strongest claim is that the method “matches GPT-4o in accuracy while achieving 8× lower tail latency” on text classification. The reported tables show this is not uniformly true across all text datasets: the paper presents a mixture of close matches, better results, and some weaker results depending on dataset/model pairing. Also, the abstract highlights a fine-tuned **Gemma-3-27B** result on MIntRec2.0 versus GPT-4o/GPT-5, but the main table for MIntRec2.0 reports **Mistral-3-24B (FT, LaaC)** at 49.34%, while the text discussion emphasizes Gemma-3-27B at 62.7%; the abstract should be clearer about which model underlies each claim and which results are central. The “more than an order of magnitude faster” comparison also depends on whether one uses GPT-4o/GPT-5 or the constrained-decoding baseline.

### Introduction & Motivation
- **Is the problem well-motivated? Is the gap in prior work clearly identified?**  
  Yes. The latency cost of free-form generation and token-by-token constrained decoding is a real issue, and the paper correctly contrasts this with encoder-based classifiers that are efficient but less flexible.

- **Are the contributions clearly stated and accurate?**  
  The high-level contributions are stated, but the claims are sometimes stronger than the evidence. In particular, the introduction suggests a unified framework that “supports zero-shot adaptation” via reassigning label tokens at inference time, but the experiments later show that performance under random token permutations drops substantially (Appendix A.6: mean accuracy 44.35% on MIntRec2.0), which is not strong evidence of zero-shot portability in the usual sense.

- **Does the introduction over-claim or under-sell?**  
  It over-claims in two places. First, the assertion of **O(1) latency** is not justified as stated: output generation is single-step, but total inference time still depends on model size, prompt length, multimodal preprocessing, and the final softmax over the label set. Second, the claim that the method “seamlessly adapts to new tasks without task-specific retraining” is not borne out by the training/evaluation setup, since the model is still fine-tuned on large balanced corpora and many reported gains are in-domain or near-domain, not truly task-agnostic zero-shot transfer.

### Method / Approach
- **Is the method clearly described and reproducible?**  
  The core mechanism is understandable: reserve atomic control tokens, train the model to emit exactly one label token, and use a restricted argmax over those tokens. That said, there is an important ambiguity: Section 3.2 says label mappings are randomized across training instances “to prevent memorization,” but the exact training pipeline and how the model learns stable class semantics under random remapping is not fully specified. In standard classification, a label should have a consistent mapping; randomizing mappings per instance seems like it could make the supervision contradictory unless the class description in the prompt is the real signal. This needs a clearer explanation and ideally an ablation showing why this does not destroy learnability.

- **Are key assumptions stated and justified?**  
  Several assumptions are implicit rather than justified:
  1. The label description in the prompt is sufficient for predicting the right control token, even when token-class mappings are randomized.
  2. A single output token is enough for all classification tasks considered.
  3. Restricting the output vocabulary to the control-token set is a fair way to compare against constrained decoding baselines.  
  These are plausible, but the paper does not fully justify them.

- **Are there logical gaps in the derivation or reasoning?**  
  Yes. The training objective in Section 3.3 is conceptually simple, but the description of the classifier head is unclear: it says logits are restricted to Ω and that the token embedding matrix and LM head remain trainable, but the final prediction is also treated as a cross-entropy over only the control token set. The paper should explicitly state whether the LM head is tied, whether the output projection is over the full vocabulary then masked, or whether it is replaced by a K-way classifier over control-token embeddings. As written, the architecture is under-specified.

- **Are there edge cases or failure modes not discussed?**  
  Yes, and they matter:
  - **Out-of-distribution or malformed inputs**: what happens if the prompt description conflicts with the label space or is incomplete?
  - **Large label spaces**: the paper claims a reserved pool of 500 tokens, but does not analyze how accuracy scales when K becomes large, beyond a few datasets.
  - **Ambiguous or overlapping labels**: single-token classification may be less robust when labels are semantically close or the task requires abstention.
  - **Multilingual settings**: Appendix B shows clear degradation on MTOP when training data are English-only, which indicates the method is not inherently language-agnostic.

- **For theoretical claims: are proofs correct and complete?**  
  There are no formal theoretical proofs, but the paper makes quasi-theoretical claims such as “guarantees O(1) latency” and “deterministic one-step inference.” Those are overstated unless carefully qualified to exclude prompt length, multimodal preprocessing, and softmax over K labels. The deterministic one-step decoding claim is only true conditional on the decoder being forced to emit one token, not as a statement about end-to-end system latency.

### Experiments & Results
- **Do the experiments actually test the paper’s claims?**  
  Partly. The MIntRec2.0 results do test whether single-token LaaC can be competitive with GPT-style prompting and some encoder baselines on a multimodal intent task. The text benchmarks test cross-domain classification performance and latency. However, the central “generality” claim is weaker than the experiments suggest because the model is still fine-tuned on the tasks’ general domain family, and several reported text results appear to compare against untuned or differently tuned baselines.

- **Are baselines appropriate and fairly compared?**  
  This is one of the paper’s weakest points.
  - The paper compares fine-tuned LaaC models against **GPT-4o/GPT-5** prompting baselines, but those proprietary models are not fine-tuned on the task.
  - For MIntRec2.0, it compares against MAG-BERT and MulT from the original benchmark, but those are not necessarily run under exactly the same end-to-end conditions as LaaC; the appendix acknowledges video feature extraction dominates latency, which complicates the fairness of “model latency” comparisons.
  - For text tasks, the paper says LaaC is “not fine-tuned on any of these datasets,” yet then it reports results for **fine-tuned** Gemma/Mistral models. This is confusing: if the model is fine-tuned on a broad corpus containing related tasks, the comparison is not truly zero-shot to those benchmarks.
  - The use of only **200 sampled test examples per text dataset** is a serious limitation for comparing to reported SOTA or even stable latency/accuracy estimates on large benchmarks like Amazon Reviews and DBpedia.

- **Are there missing ablations that would materially change conclusions?**  
  Yes, several.
  1. **Tokenization/label-token design ablation**: the paper argues atomic tokens are key, but there is no direct comparison against constrained decoding with short verbalizers or digit labels under the same model.
  2. **Randomized label assignment ablation**: this is central to the method’s purported generality, yet the paper does not show performance with fixed mappings versus random mappings in the main experiments.
  3. **Prompt-description ablation**: if the prompt’s class descriptions are essential, how much do they contribute versus the token mechanism?
  4. **Ablation on LoRA vs full fine-tuning**: since the method’s benefit is claimed to come from the classification formulation, it would help to know how much is due to PEFT choice.
  5. **Ablation on label-set size**: the paper discusses 2 vs 14 classes, but does not systematically vary K while controlling for dataset difficulty.

- **Are error bars / statistical significance reported?**  
  No. This is a notable issue for ICLR standards, especially because many comparisons are close and the evaluation subset for text tasks is only 200 examples. Latency also shows variability, but the paper reports only P50/P95, not uncertainty across runs.

- **Do the results support the claims made, or are they cherry-picked?**  
  Some results are supportive, but there are signs of selective emphasis:
  - The strongest headline result is the 62.7% MIntRec2.0 number for Gemma-3-27B, while the main table shown in Section 4.5.1 foregrounds Mistral-3-24B at 49.34%. The best result is discussed in text but not displayed in the table excerpt, making it harder to assess consistently.
  - Appendix B shows that on **Banking77**, LaaC underperforms GPT-4o and even the base Mistral-3-24B in accuracy, despite much lower latency. That is important evidence that the method is not uniformly superior, yet the main narrative emphasizes cases where LaaC wins.
  - Appendix A.6 shows only 44.35% under random token permutations, which weakens the zero-shot-adaptation claim, but this is framed positively as evidence that the model uses descriptions rather than token identity. The result actually indicates substantial performance loss under remapping, which should be discussed more honestly.

- **Are datasets and evaluation metrics appropriate?**  
  Mostly yes for accuracy and latency, but with caveats:
  - Accuracy is standard for classification.
  - Latency at batch size 1 is relevant for interactive systems, but a broader throughput/latency trade-off analysis would be needed for deployment claims.
  - Using only a sample of 200 test examples for several large text benchmarks is not ideal for a paper making broad generalization claims.
  - The multimodal benchmarks are appropriate, but the paper should more carefully separate model inference latency from upstream preprocessing costs.

### Writing & Clarity
- **Are there sections that are confusing or poorly explained?**  
  Yes, mainly in the method and experimental setup:
  - Section 3.2/3.3 leaves ambiguity about how random label-token shuffling works during training.
  - The “not fine-tuned on any of these datasets” statement in Section 4.5.2 conflicts with the fact that the models are fine-tuned on a 28k-example corpus that includes related intent and classification data.
  - The distinction between “base model,” “FT, LaaC,” and “zero-shot adaptation” is not consistently maintained.
  - The paper also uses “classification accuracy” and “zero-shot classification” in ways that may confuse readers about what is actually trained versus transferred.

- **Are figures and tables clear and informative?**  
  Generally yes in intent, but several tables would benefit from stronger methodological context:
  - Table 1 and Table 2 present accuracy/latency but do not make the training-data differences fully explicit in the table itself.
  - Figure 3 is useful for scaling trends, though the excerpted rendering is incomplete.
  - Appendix tables on latency are informative, but the comparison is still not fully apples-to-apples because preprocessing and benchmark-specific pipelines differ.

### Limitations & Broader Impact
- **Do the authors acknowledge the key limitations?**  
  Partially. The conclusion mentions audio as future work and notes the need for calibration, robustness, and multilingual generalization. That is good, but the limitations are underdeveloped relative to the strong claims elsewhere.

- **Are there fundamental limitations they missed?**  
  Yes:
  - The method still depends on large pretrained decoder/VLMs and substantial fine-tuning compute; it is not an inherently cheap classifier.
  - The claimed latency gains may shrink or disappear in pipelines where preprocessing or retrieval dominates, especially for multimodal tasks.
  - The approach may not be suitable for tasks needing calibrated probabilities, rejection, or structured uncertainty estimates.
  - The zero-shot token-permutation story is weaker than presented, because performance remains far from the standard setting.

- **Are there failure modes or negative societal impacts not discussed?**  
  The paper does not discuss whether the method could amplify deployment risks by making LLM-based classifiers appear simpler and more reliable than they are. There is also no discussion of fairness, domain shift, multilingual inequity, or the consequences of deploying high-confidence single-token classifiers in safety-sensitive classification settings. Given ICLR expectations, a brief broader-impact discussion would be appropriate.

### Overall Assessment
This paper presents a sensible and practically relevant idea: using atomic label tokens to turn decoder-style LLM classification into a single-step generation problem with clear latency benefits. The main empirical message—that this can work well on some multimodal and text benchmarks—is plausible and partially supported. However, for ICLR standards, the paper currently overstates several claims, especially around zero-shot generalization, O(1) latency, and broad superiority over baselines. The experimental design also leaves important questions unresolved: the randomized token mapping needs clearer justification, there are missing ablations, no uncertainty estimates are reported, and several comparisons are not fully fair or fully comparable. The contribution is interesting and potentially useful, but the paper would need a more careful framing and stronger experimental validation before it meets a strong ICLR acceptance bar.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes LaaC, a framework that reformulates classification with decoder-style LLMs/VLMs as a single-token constrained generation problem using atomic label tokens. The main claim is that by fine-tuning large decoder models with LoRA and reserving special output tokens, one can obtain both strong classification accuracy and substantially lower latency than prompting or constrained multi-token decoding, especially on multimodal intent recognition and several text classification benchmarks.

### Strengths
1. **Clear and practically relevant motivation: latency-aware classification with LLMs.**  
   The paper targets an important ICLR-relevant problem: using foundation models for classification under deployment constraints. The focus on tail latency (P95), batch-size scaling, and one-step decoding is concrete and relevant for real systems, not just offline accuracy.

2. **A simple and well-defined formulation.**  
   The core idea—map each class to an atomic special token and train only the final response token—is straightforward and easy to understand. The methodology section gives a relatively explicit training/inference recipe, including LoRA, tokenizer extension, and restricted argmax over the label token set.

3. **Empirical evidence of latency gains.**  
   The paper reports consistent latency reductions over prompting and constrained decoding, e.g. GPT-4o constrained decoding still being much slower than the proposed method on MIntRec 2.0, and the appendix showing benefits across batch sizes. This directly supports the central efficiency claim.

4. **Evaluation across text and multimodal settings.**  
   The paper does not restrict itself to one benchmark family: it includes multimodal intent recognition (MIntRec 2.0), standard text classification tasks, and additional appendix results for TweetTopic, Banking77, and MTOP. That breadth is a positive sign for potential applicability.

5. **Attention to deployment metrics beyond accuracy.**  
   Reporting P50 and P95 latency, and comparing against both proprietary API models and encoder baselines, aligns with ICLR’s growing interest in systems-aware ML and real-world utility.

### Weaknesses
1. **Novelty is moderate; the method is largely an engineering combination of known ideas.**  
   The paper combines atomic special tokens, constrained output spaces, LoRA fine-tuning, and classification-as-generation. Each component is sensible, but the conceptual leap beyond prior work on verbalizers, constrained decoding, and prompt-based classification appears incremental rather than fundamentally new. The paper needs a sharper articulation of what is scientifically new beyond “use a one-token label token.”

2. **The strongest results are not matched by equally strong baselines on the same footing.**  
   Several comparisons are potentially uneven. For instance, the paper compares fine-tuned LaaC models to GPT-4o/GPT-5 API models, which are not adapted to the task in the same way, and to encoder baselines drawn from prior work. ICLR reviewers will likely expect stronger, more carefully matched baselines, ideally including fine-tuned decoder-only models with standard classification heads or verbalizer-based adaptation under the same data regime.

3. **Accuracy claims are mixed and sometimes weaker than the narrative suggests.**  
   The paper highlights cases where LaaC beats GPT-4o or encoder baselines, but other reported numbers are less flattering. For example, on MIntRec 2.0 the main fine-tuned model reported in Table 1 is 49.34%, which is below the 60.6% encoder baseline cited in the text, and the appendix shows that performance under random token permutations drops substantially to about 44.35%. This suggests the method is not consistently superior across settings.

4. **The “zero-shot adaptation” claim is not convincingly supported.**  
   The paper says that reassigning label tokens at inference time enables zero-shot classification on new tasks, but the actual setup still relies on prompt descriptions and fine-tuned representations from English training corpora. Moreover, the random permutation experiment in the appendix shows a large accuracy drop relative to the main setting, indicating the claimed token-independence is limited.

5. **Reproducibility is incomplete.**  
   The paper provides many hyperparameters, but important details remain unclear or absent: exact prompt formats for all tasks, how multimodal inputs are serialized, how many seeds are used, whether reported numbers are averaged over runs, and how API latency was measured and normalized. The use of proprietary models also limits reproducibility of the headline comparisons.

6. **The experimental protocol raises questions about fairness and robustness.**  
   Text benchmarks are evaluated on only 200 randomly sampled test examples each, which is small for stable conclusions on datasets like Amazon Reviews and DBpedia. The training corpus mixes public datasets, synthetic data, and instruction-following data, which makes it hard to isolate which component drives performance gains.

7. **Significance is limited by dependence on large models and large labeled corpora.**  
   The paper’s best results come from a 27B model with substantial fine-tuning resources. This reduces the practical impact for many users who care about efficiency because they also care about memory, cost, and accessibility. The paper argues for efficiency at inference time, but training and model-hosting costs are still substantial.

### Novelty & Significance
**Novelty:** Moderate to low-moderate by ICLR standards. The atomic-token classification idea is useful, but it feels like a clean packaging of existing classification-via-generation and constrained-decoding ideas rather than a fundamentally new learning paradigm.  
**Clarity:** Generally good at the high level, but several claims are overstated relative to the evidence, especially around zero-shot adaptation and broad superiority. Some parts of the experimental narrative are also hard to reconcile across the main paper and appendix.  
**Reproducibility:** Moderate. The paper gives many training details and says code will be released, but evaluation details, seed reporting, API benchmarking methodology, and some dataset construction choices are not sufficiently precise for full replication.  
**Significance:** Potentially useful in practice, especially for latency-sensitive classification with decoder models. However, the paper’s scientific significance is limited by incremental novelty and by evidence that the gains are strongest only in certain setups.

### Suggestions for Improvement
1. **Tighten the novelty claim and position the method more honestly.**  
   Explicitly distinguish what is new relative to prior work on verbalizers, constrained decoding, and prompt-based classification. If the main contribution is a practical recipe, say so clearly.

2. **Add stronger and fairer baselines.**  
   Compare against fine-tuned decoder-only models using standard classification heads or standard label verbalizers under the same data and compute budget. Also include a stronger “same-model, different-head” baseline to isolate the effect of atomic tokens.

3. **Report results with multiple seeds and confidence intervals.**  
   This is especially important for the 200-example test subsets and for multimodal benchmarks where variance may be nontrivial. ICLR reviewers will expect statistical stability.

4. **Clarify the zero-shot adaptation claim.**  
   Provide an experiment where label-token remapping is performed on genuinely new tasks with no task-specific fine-tuning, and compare against standard prompting and constrained decoding under the same conditions.

5. **Separate accuracy gains from data-mixing effects.**  
   Since the training corpus includes public, synthetic, and instruction-following data, add ablations showing how much each component contributes. This would improve scientific interpretability.

6. **Make latency methodology fully transparent.**  
   Specify hardware, batching, caching, warm-up, prompt length, output-token settings, and whether timing includes preprocessing and multimodal feature extraction. For API models, clarify network and service variability handling.

7. **Address robustness and calibration.**  
   Since deployment claims hinge on reliability, add calibration metrics, out-of-distribution behavior, and failure cases, especially for out-of-scope intent detection and multilingual settings where performance drops are visible.

8. **Reduce overclaiming in the conclusion.**  
   The paper should avoid implying that LaaC broadly dominates encoder models and GPT models. A more balanced claim would be that it offers a strong accuracy-latency trade-off in some benchmark regimes, especially after substantial fine-tuning.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a direct comparison against strong, modern encoder-classifiers and task-tuned open LLM classifiers on the same training budgets. Right now the claim that decoder-style LaaC is competitive is undermined by weak or uneven baselines: the paper compares to GPT APIs and older encoder baselines, but not to strong recent text/multimodal classifiers fine-tuned with comparable data and compute.

2. Add ablations isolating what actually drives the gains: single-token output vs. LoRA vs. label-token randomization vs. extra training data. Without this, the paper cannot support the claim that the LaaC formulation itself, rather than more training or model scale, explains the results.

3. Evaluate on harder label-space settings, especially larger-class benchmarks and open-set/OOD classification with rejection. The paper claims scalability to larger label vocabularies and practical deployment, but the experiments only go up to 77 classes and do not test whether single-token decoding remains accurate and stable as class count grows.

4. Add a fair latency comparison that includes end-to-end preprocessing and identical decoding constraints for all methods. The current speed claims are not fully convincing because some baselines are compared under different interfaces, and latency numbers depend heavily on prompt length, batch size, and whether constrained decoding is enabled.

5. Add results on out-of-domain and multilingual classification with actual multilingual supervision or transfer. The paper explicitly claims generality, but the multilingual appendix shows degraded accuracy under English-only fine-tuning, which directly weakens the claim that LaaC is broadly practical beyond English.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify how much accuracy depends on the random label-token permutation strategy. The paper claims this prevents memorization and supports zero-shot remapping, but the observed drop under unseen permutations needs a clearer explanation of when the model is learning semantics versus exploiting token priors.

2. Analyze calibration, confidence, and abstention behavior. For deployment claims, a classifier that is fast but overconfident is not sufficient; without calibration or selective prediction analysis, it is unclear whether the model is usable in real real-time systems.

3. Explain why fine-tuned base models sometimes outperform the same architecture under LaaC on some datasets. The paper reports mixed results in places, so it needs a deeper error analysis showing whether the tokenization constraint introduces systematic mistakes or whether the gains are dataset-specific.

4. Measure performance as label space grows with controlled synthetic tasks. The claim that constant-time inference scales with labels is only partially supported; you need a controlled analysis showing accuracy, confusion, and latency as K increases independently of dataset difficulty.

5. Separate the contribution of multimodal vs. text supervision in the shared training corpus more carefully. The current evidence that combined training helps is too thin to justify the broader claim that a unified corpus improves general classification; it needs stronger controlled analysis across multiple held-out tasks.

### Visualizations & Case Studies
1. Add confusion matrices or per-class error breakdowns for MIntRec 2.0 and Banking77. These would show whether LaaC is truly learning intent semantics or just benefiting from coarse label regularities, especially for semantically close classes.

2. Show failure cases where the model chooses the wrong token under unseen permutations or multilingual inputs. This would expose whether “zero-shot adaptation” is robust or fragile to changes in label descriptions and language.

3. Plot latency distributions, not just P50/P95, for LaaC vs. constrained decoding vs. free-form prompting. The core claim is predictable low latency, so full distribution plots would reveal whether there are long-tail failures hidden by summary statistics.

4. Add examples demonstrating how the model behaves when label descriptions are ambiguous or misleading. Since the method relies on prompt-provided class descriptions, this is essential to judge whether the classifier is actually robust or overly prompt-sensitive.

### Obvious Next Steps
1. Test the method on more realistic deployment tasks with many classes, open-set rejection, and severe class imbalance. Those are the main settings where a single-token classifier would need to prove practical value for ICLR-level relevance.

2. Compare against a distilled encoder classifier and a task-tuned open-source VLM classifier under matched compute. That is the cleanest way to establish whether LaaC offers a real accuracy-latency trade-off advantage or just leverages bigger pretrained models.

3. Add a principled study of token-space design, including whether atomic control tokens can be learned without randomization and how many reserved tokens are actually needed. This directly affects the claimed simplicity and scalability of the framework.

4. Provide a stronger methodological argument for why one-step decoding should generalize beyond classification. As written, the paper suggests broader relevance, but it has not shown whether the approach extends to structured prediction or retrieval-style decision problems.

# Final Consolidated Review
## Summary
This paper proposes LaaC, a decoder-style classification framework that maps each class to an atomic special token and trains the model to emit exactly one token at inference time. The main appeal is reduced decoding latency while retaining the flexibility of large LLM/VLM backbones, and the paper reports strong results on some multimodal and text classification benchmarks. However, the evidence is uneven, and several of the strongest claims — especially around zero-shot adaptation, broad generality, and “O(1)” latency — are overstated relative to the actual experiments.

## Strengths
- **The core formulation is simple and practically relevant.** Reducing classification to a single restricted decoding step via atomic label tokens is easy to understand and directly targets an important deployment problem: token-by-token generation overhead.
- **The paper reports real latency gains, not just accuracy.** The MIntRec2.0 and appendix batch-size results consistently show lower P50/P95 latency for LaaC than prompting or standard autoregressive decoding, which supports the paper’s central efficiency motivation.
- **The evaluation spans both text and multimodal tasks.** The paper does not only test a narrow sentiment setting; it includes multimodal intent recognition plus several text benchmarks, which is a sensible attempt to probe generality.

## Weaknesses
- **The novelty is modest.** The method is largely a combination of known ideas: classification-as-generation, constrained decoding, reserved label tokens, and LoRA fine-tuning. The paper does not convincingly show a new learning principle so much as a useful engineering recipe.
- **Several headline claims are stronger than the evidence.** The paper repeatedly implies zero-shot adaptation and broad generality, but the model is still fine-tuned on large corpora, and the random token-permutation experiment drops to about 44% on MIntRec2.0 rather than demonstrating robust token-independent transfer. Likewise, “O(1) latency” is misleading as an end-to-end claim because model size, prompt length, and multimodal preprocessing still matter.
- **The comparison protocol is uneven.** The strongest LaaC results are compared against proprietary API models and older encoder baselines, but not against equally strong, task-tuned open decoder-classifiers under matched training budgets. In addition, the text benchmarks are evaluated on only 200 sampled test examples each, which is too small for stable broad claims on datasets like Amazon Reviews and DBpedia.
- **The method is under-specified in an important way.** The randomized mapping between labels and control tokens is central to the paper’s story, but the training/inference mechanics are not fully clarified, and there is no main-table ablation showing fixed vs. randomized mappings. This makes it hard to tell how much of the gain comes from the token formulation itself versus prompt descriptions and data scale.

## Nice-to-Haves
- A clearer ablation separating the effects of atomic tokens, LoRA, prompt descriptions, and extra training data would make the contribution much easier to interpret.
- Confidence intervals or multi-seed variance would strengthen the credibility of the smaller benchmark comparisons.
- A more explicit end-to-end latency accounting, including preprocessing and multimodal feature extraction, would make the deployment story cleaner.

## Novel Insights
The most interesting aspect of this paper is not that LLMs can classify, but that the authors try to make classification behave like a deterministic single-step routing problem inside a decoder model. That is a useful systems insight: if the output space is collapsed to atomic tokens, latency becomes much more predictable and the model interface becomes cleaner for downstream integration. The flip side is that the paper’s “generality” story is weaker than it sounds — the method seems to work best when it is still anchored by substantial fine-tuning and carefully engineered prompts, rather than as a truly zero-shot, model-agnostic classifier. The real contribution is thus a practical efficiency-oriented recipe, not a broad new paradigm.

## Potentially Missed Related Work
- **Standard decoder-only classification with verbalizers / constrained generation under matched fine-tuning** — highly relevant as a stronger same-family baseline that would isolate the gain from atomic tokens.
- **Recent task-tuned open-source LLM/VLM classifiers** — relevant because the paper’s strongest comparisons are against proprietary APIs and older encoder baselines, not modern matched alternatives.
- **Open-set / selective prediction work for classification** — relevant because the paper emphasizes deployment, but does not evaluate rejection, calibration, or uncertainty behavior.

## Suggestions
- Add a direct ablation comparing fixed label-token mappings, randomized mappings, and standard verbalizer-based decoding under the same base model and training budget.
- Report results on the full test sets, or at least provide confidence intervals for the sampled benchmark subsets.
- Include a stronger same-budget decoder baseline, ideally a task-tuned open-source LLM classifier with a standard classification head or verbalizer setup.
- Rephrase the claims around “zero-shot adaptation” and “O(1) latency” more carefully so they do not overstate what is actually demonstrated.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 0.0]
Average score: 1.3
Binary outcome: Reject
