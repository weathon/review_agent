# LLM as a Classifier: Leveraging Large Language Models for Text and Vision Classification

- Avg Score: 1.33
- Decision: Reject
- Scores: 2, 2, 0

## Abstract
Classification is a fundamental capability for AI systems, yet current large language model (LLM) approaches remain poorly suited for latency-critical applications. Prompting and constrained decoding produce verbose, multi-token outputs that require expensive token-by-token generation, while encoder-based models achieve faster inference at the cost of flexibility and generative capacity. We propose LaaC (LLM as a Classifier), a framework that formulates classification as constrained generation with single-token outputs. By introducing atomic label tokens and applying parameter-efficient fine-tuning, our method reduces classification to a deterministic one-step decoding problem. Experiments across text and multimodal benchmarks demonstrate both strong accuracy and consistently fast inference. On MIntRec 2.0, a fine-tuned Gemma-3-27B model attains 62.7\% accuracy, outperforming GPT-4o (43.7\%) and GPT-5 (51.8\%) while running more than an order of magnitude faster. On standard text classification benchmarks, our models match GPT-4o in accuracy while achieving 8 × lower tail latency. These results establish decoder-style LLMs as practical and scalable classifiers for real-time applications. Our code is available at https://anonymous.4open.science/r/LaaC_ICLR.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a somewhat new scheme for training an LLM classifier. The new scheme, which involves training a LoRA and using the LLM to emit exactly one token. The authors demonstrate the effectiveness of their method empirically, showing that it improves accuracy while also improving latency.

### Strengths
The biggest strength of the paper is in identifying a simple idea and experimenting with it. Their approach identifies and removes a major limitation from existing classification methods - primarily, it improves latency while also improving accuracy over the base model.

### Weaknesses
Unfortunately, the paper has various shortcomings: A. Insufficient substantiation of claims B. Insufficient baselines/comparison to related works. 
First, regarding insufficient substantiation of claims:
1. Th authors claim that latency is significantly improved, and latency is measured using vLLM as the inference engine - however, latency often changes with batch size (lower batch size having better latency), I'd love to see a measurement of latency across batch size.
2. Modern LLMs (GPT 4o etc) can often be pre-prompted to respond in a single character. Another alternative would be create a set of sequences which are [x, cls, cls-description] for any given input x, for all classes cls. Then we could infer on a batch containing all of these possible sequences and then pick the most-likely sequence? That would result in only needing 1 forward pass too. 

Next, regarding insufficient baselines:
1. The authors claim that their approach is related to other classification approaches such as PET, LM-BFF etc but that the authors' approach works better because it can better utilize bigger datasets. However, the authors do not empirically compare these baselines to their approach - I'd recommend comparing to at least one of these alternative methods.

### Questions
Q1. Regarding figure 3 - it seems like the 12B model is actually slightly better than 27B on SST-2, what is happening there?
Q2. Figure 3 is used to support the idea that the authors' approach is scalable, perhaps the right plot is Gap between LaaC and Base model as model size increases?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This work introduces LaaC (LLM as a Classifier), a framework that formulates classification as constrained single-token generation. The model is trained in a parameter-efficient manner using atomic labels tokens, and evaluations demonstrate high accuracy and low latency across both text and multimodal benchmarks.

### Strengths
- The design that collapses classification into a deterministic one-step decoding task significantly reduces latency.
- Demonstrates strong cross-domain performance without requiring fine-tuning on specific datasets

### Weaknesses
- Limited modality and language coverage, lacking support for audio and broader multilingual settings.
- Unclear justification for maintaining the model’s broader generative capabilities given its classification focus.
- Insufficient details on evaluation settings, particularly regarding prompt configurations when comparing LaaC with standard LLM baselines.

### Questions
- Since this work focuses on single-output classification, is the preservation of the model’s generative capabilities an intentional design choice?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper proposes a method for classification using typical decoder multimodal LLMs (such as Gemma-3 27B). The classification method constructs a prompt mapping from class names to special "control" tokens, followed by a classification prompt (Appendix A.2 for examples). The model then predicts a special control token corresponding to a class. The authors claim that this result is efficient because it involves only a single pass through the LLM (e.g. Gemma 4B and 27B), to generate a single control token. Models for this task are adapted with LoRA. The authors use this setup on a variety of multimodal and text-only tasks, showing competitive performance with encoder only models (e.g. MAG-BERT, Rahman et al. 2020). However, this compares a LoRA tuned 27B parameter model (62.72% accuracy) with a much smaller encoder BERT-based encoder model (60.48% accuracy). Notably, in the multimodal setting, Gemma 27B is the only model (out of other LLMs, Gemma 3-{4,27}B, Mistral-3 24B) that beats established baselines for this task. In the text only setting, results are further mixed (see below).

### Strengths
S1: This paper is fairly well-written, with clear diagrams, formatting, and content

S2: This paper evaluates multiple models on multiple datasets. However, there are many methodological shortcomings that I discuss below

### Weaknesses
This paper claims "significant latency improvements" and "strong empirical results" as their key contributions (Sec 1, lines 90 - 104). However, the paper does not show convincing latency improvements as they **do not benchmark the latency of the most performant baselines. Furthermore, their models rarely beat baselines and use non-standard evaluation setups. There are also multiple unsubstantiated claims in the paper (see below).

W1. This paper claims novelty for the method of tuning an LLM to predict a single control token rather than generating tokens. The authors acknowledge that contrained decoding and encoder-based methods like classification heads are also used for this task (Sec 2.1, 2.2). However, they do not use these as baselines.

W2. The authors state that their method "supports zero-shot adaptation" but do not show this. For example, they could repeat their experiments while shuffling the mapping between control tokens and task labels, hopefully showing very low variance in the accuracy.

W3. Does not show latency of the MAG-BERT and MulT baselines (table 1, 371-372), even though these outperform all but 1 of the largest LLM models. Examining the MAG-BERT paper shows that this method uses a BERT base sized model as the backbone ("As there exists M = 12 Encoder layers in our BERT model") - which suggests that MAG-BERT is around 110M parameters. **Since this is ~235 times smaller than the Gemma 27B model, we would expect very low latency.** For the text experiments, only 200 evaluation instances (line 299). Since the evaluations use an 80GB GPU (line 331), this may even fit in a single batch on that A100.

For emphasis: the encoder baseline achieves 60.58% accuracy and the authors do not report latency. The 24 billion parameter mistral model, with their method applied, achieves only 49.34%

W4. Since the paper uses a nonstandard evaluation setup, the results are not comparable to prior work:
> For text-only benchmarks, we evaluate on 200 randomly sampled test examples from each dataset to ensure consistent and efficient comparisons across models
Furthermore, it is unclear why subsampling 200 instances is necessary for "efficient comparisons".

W5. The authors select very old and saturated text benchmarks. Note above that the eval sets for text are sampled to 200 instances. However, the accuracies in Table 2 frequently differ by only 0.5% - meaning a **single test instance**. The accuracies are very high, indicating that these old (2013, 2015, lines 291 - 297) are saturated or possibly contaminated in the training data. For example, the SST-2 evaluations are all identical in accuracy (95.5%) except for the base gemma model (95.0%).

W6. The authors also emphasize that text encoding of the labels is more flexible, however the examples in the Appendix show only single class names rather than descriptors.

W7. In the text scenario, the model often decreases in performance even on the 200 sampled test set. **For example, their methods on AG News and DB Pedia both decrease in accuracy for both Gemma and Mistral models.**

W8. The authors also make claims about model latency for API models. They do not show that latencies are consistent, **and since they access by API, there could be arbitrary confounding variables in this measurement.** For example, internet latency between continents can often reach hundreds of ms, which would account for differences in table 2. Furthermore, this could depend on API load balancing, as this is unfortunately a black box to academic researchers.

W9 Further, the authors claim that Gemma is smaller than GPT-4o and GPT-5 (lines 346-347). **This claim requires evidence,** especially since GPT-5 and 4o differ substantially in costs and capabilities. For example, OLMO 2 32b has similar performance while being similar in size to the gemma model (https://allenai.org/blog/olmo2-32b).

W10. In general, making claims about latencies requires much more careful benchmarking than in this paper. There is no discussion of inference optimizations (only that VLLM is used), batching, no measurements around input latency (since input images could be expensive to encode), and no reported variance in inference time after repeated measurements.

### Questions
- Why are baseline latencies not reported?
- What are the exact parameter size differences between the baselines and the LLM models?
- How many LoRA parameters are tuned in the LLM models (what is the size)?
- Why is there no linear head baseline? This would be simpler to the LoRA approach while being closer to how smaller encoder models are used.
- Given that the 3 stated contributions are "accuracy, latency, and generality" (line 88), can you show a pareto improvement on these axes?

### Soundness
1

### Presentation
2

### Contribution
1
