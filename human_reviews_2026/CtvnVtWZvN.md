# Can We Predict Alignment Before Models Finish Thinking? Towards Monitoring Misaligned Reasoning Models

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Reasoning language models improve performance on complex tasks by generating long chains of thought (CoTs), but this process can also increase harmful outputs in adversarial settings. In this work, we ask whether the long CoTs can be leveraged for predictive safety monitoring: *do the reasoning traces provide early signals of final response alignment that could enable timely intervention?* We evaluate a range of monitoring methods using either CoT text or activations, including highly capable large language models, fine-tuned classifiers, and humans. First, we find that a simple linear probe trained on CoT activations significantly outperforms all text-based baselines in predicting whether a final response is safe or unsafe, with an average absolute increase of $13$ in F1 scores over the best-performing alternatives. CoT texts are often unfaithful and misleading, while model latents provide a more reliable predictive signal. Second, the probe can be applied to early CoT segments before the response is generated, showing that alignment signals appear before reasoning completes. Error analysis reveals that the performance gap between text classifiers and the linear probe largely stems from a subset of responses we call *performative* CoTs, where the reasoning consistently contradicts the final response as the CoT progresses. Our findings generalize across model sizes, families, and safety benchmarks, suggesting that lightweight probes could enable real-time safety monitoring and early intervention during generation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors examine whether it is possible to predict the alignment (refusal vs non-refusal) of a reasoning model's ultimate response from its reasoning trace. They compare text-based classifiers with linear probes on hidden activations, and find that the probes are superior. They also suggest that activation-based probes can predict the response outcome many reasoning steps ahead. They also conduct error analysis to show systematic cases where the probes beat text-based classifiers.

### Strengths
- The paper is well-written, clear, and easy to read.
- Thorough experiments, including multiple models, datasets, and training dataset sizes.

### Weaknesses
- Lacks justification of important technical details
  - The decision to train probes on the last layer of the last token position is an important design decision (which may have serious implications; see the Questions section), and lacks justification in the current manuscript.
  - There is also no discussion of sampling methodology. It is implicitly assumed that a partial reasoning chain deterministically leads either to refusal or non-refusal. In reality, reasoning model inference uses non-deterministic sampling, and so refusals are stochastic.
- Motivation is unclear
  - In my opinion, the motivation is unclear. Why should we care about predicting whether models refuse before they refuse? One of the predominant methods in LLM safety is to run classifiers of the *model output* ([Sharma et al., 2025](https://arxiv.org/abs/2501.18837)), and classify the model output directly.
- Limited to reasoning models trained via SFT
  - SotA reasoning models are trained via RL, and it's unclear whether analysis of SFT-trained reasoning models generalize to RL-trained reasoning models. In particular, it is unclear whether the results would hold on models that were trained to reason about whether or not to refuse a request, as in [Guan et al., 2024](https://arxiv.org/abs/2412.16339).

### Questions
- How is the sampling done?
  - For reasoning models, official documentation generally advises to not use temperature 0. If temperature 0 is used here, then whether a reasoning trace leads to a refusal or not is probabilistic, not deterministic. Have you thought about this? What temperature are you sampling with? Do you resample multiple times per rollout? I think the paper would be strengthened if you clarify and justify your sampling methodology.
- Why use the last layer of the last token position? Is this a fair design choice?
  - The probes are trained using activations from the last layer of the last token position. This is immediately preceding the unembedding, so this activation will contain information about the next token prediction. In this case, the next token prediction is very useful in order to determine whether the response will be a refusal or not (the first tokens of refusal responses are drawn from a narrow distribution of refusal phrases).
  - I fear that this gives an unfair advantage to the probing methods over the text-based classifiers - the probing methods effectively have access to the first token of the response.
  - One way to try and disentangle things / prove that the probe works beyond this last-token effect would be to give the first token of the response to the text classifiers, so as to try and give them an "even playing field".
  - In my opinion, this issue significantly weakens the main result, namely that activation-based probes outperform text-based classifiers (section 4).
- Do the results generalize to reasoning models that were actually trained via RL?
  - The models are limited to reasoning models that were trained via SFT on reasoning traces. Do you expect the results to generalize to reasoning models trained with RL? There are open-source models trained with RL available, such as Qwen 3, and I think it would be worth reproducing the experiments for that model family in order to test this question.
- Line 251: "Text-based classifiers rely on semantic cues in the CoT to infer model behavior. For example, if a CoT includes planning steps toward an illegal request, these classifiers will likely predict a harmful outcome."
  - Where is the evidence for this claim? How can we know how the text-based classifiers work? Is this speculation? If so, it'd be good to mention that this claim is speculative.
- Missing baselines for section 5
  - How do text-based classifiers perform in the "future" setting? Do probes still outperform text-based classifiers in this setting? Or does the advantage disappear?
- Section 6 - asymmetric error analysis
  - Did you study the opposite error analysis? E.g., are there cases where the probes systematically fail, but where the text-based classifiers do well?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates methods for predicting the safety alignment of language model responses, specifically focusing on reasoning language models (RLMs) that generate long chains of thought (CoTs). The core finding is that a simple linear probe trained on CoT activations significantly outperforms all text-based monitoring methods, including highly capable large language models (LLMs) and human annotators, in predicting whether a final response will be safe or unsafe. The linear probe demonstrates that alignment signals can be detected from early CoT segments, enabling potential real-time safety monitoring and intervention before the model finishes its full reasoning process. The research suggests that internal, latent representations provide a more reliable signal for safety monitoring across different RLM sizes and safety benchmarks than text-based analysis.

### Strengths
- The paper is easy to read and well structured. 
- This paper tackles a relevant problem that has gained significant attention recently. 
- The authors employ three relevant datasets for their experimental procedure, which seems well executed overall and the analysis of results is well conducted.

### Weaknesses
- There seems to be an important body of literature missed in this work's background. Real time safety alignment prediction is not novel, and it's been well understood that simple linear discriminators can perform well for this task (see references below).
- Related work focuses on Reasoning and Chain-of-Thought literature, while ignoring a large bulk of related work on controlled text generation. For example, how does this work sufficiently differ from [1] and [2] for it to be considered a worthwhile contribution? On the same note, recent work published at this conference, such as [3] also seem to incorporate a similar idea (in their case applied to domain certification, but one could argue it's a related concept).
- One of the key findings of the paper is the possibility of real-time monitoring. However, this has been known and explored in the past.
- Benchmark methods ignore some recent, popular baselines, such as LlamaGuard and (especially) WildGuard, which should be included in the experimental setting.
- Overall, it seems like this paper tackles a subset of a larger problem that has been studied for a while now, with findings that have been reported in a similar fashion in other papers. The main difference I see from this paper in comparison to the existing literature I am aware of is the analysis of model alignment in the presence of Chain-of-Thought. In my opinion, this distinction should be clear in the paper and the authors should explain why there is a need to extend this concept to CoT, since at present it looks to me like a trivial extension of existing work.

[1] Yang, K., & Klein, D. (2021). FUDGE: Controlled text generation with future discriminators. arXiv preprint arXiv:2104.05218.

[2] Fonseca, J., Bell, A., & Stoyanovich, J. (2025). Safeguarding large language models in real-time with tunable safety-performance trade-offs. arXiv preprint arXiv:2501.02018.

[3] Emde, C., Paren, A., Arvind, P., Kayser, M., Rainforth, T., Lukasiewicz, T., ... & Bibi, A. (2025). Shh, don't say that! Domain Certification in LLMs. arXiv preprint arXiv:2502.19320.

### Questions
- The concept "linear probes", in the context of machine learning, that I am aware of, comes from [4]. Is this what you are actually using? It's very unclear to me whether actual linear classifier probes are being used (and how), or whether this is just a simple Logistic Regression trained over 50 PCA components using the model logits. If what is being done is the latter, then previous work have also explored this concept [5].
- Why aren't the human evaluation results not reported in the main body of the paper as well? This comparison is posed as part of your contributions, yet it's only available in the appendix.
- How many human annotators were used?

[4] Alain, G., & Bengio, Y. (2016). Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644.

[5] Krause, B., Gotmare, A. D., McCann, B., Keskar, N. S., Joty, S., Socher, R., & Rajani, N. F. (2020). Gedi: Generative discriminator guided sequence generation. arXiv preprint arXiv:2009.06367.

### Soundness
4

### Presentation
4

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to monitor the harmfulness of LRMs' responses based on partial or full CoT reasoning procedure. The authors demonstrate that directly using the CoT texts might not be a proper solution, while conducting linear probing on the model activations is a better solution. Extensive experiments demonstrate the effectiveness of the proposed method.

### Strengths
- Monitoring the harmfulness of the LRMs' final responses based on the CoT procedure is interesting.
- Although simple, the authors compare multiple baselines and different settings of linear probing (e.g., future-trained and present-trained).

### Weaknesses
- About CoT monitoring methods:
  - I wonder what the differences are between the fine-tuned BERT classifier and the fine-tuned harmfulness classifier, since both are conducting binary classification.
  - I'm not sure about your settings. What do you try to predict? For each CoT index, do you try to predict the harmfulness of the final response without altering the original reasoning procedure, or will you interrupt and generate an instant response at each CoT step, and then do the prediction （as lines 78-79 say?
  -  For the fine-tuned BERT, fine-tuned harmfulness classifier, and the activation-based monitoring, how to deal with the sequence length dimension is unclear.
  - In Table 2, the probing accuracy of s1.1-7B on XSTest is an outlier. Any idea on the reasons?
- About early thinking:
  - After your analysis in Sec. 5, what are your final empirical suggestions? Like, you can keep detecting the harmfulness of the CoT procedure, and if there is an alarm triggered, what should you do?
- Overall, although simple, this is an interesting exploration of monitoring LLM CoTs. However, there are details that are still unclear. I would like to see the rebuttal for the final decisions.

### Questions
Check the Weakness part.

### Soundness
3

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
The paper studies whether the chain of thought of a LLM provides early insight into whether the eventual response of the model could be misaligned (unsafe). The chain of thought segment of the model is represented in two ways: in a textual manner, and as the activation (embedding) representation. The latter is shown to be an effective indicator of eventual response misalignment.

### Strengths
When access to a model’s activations is available, the paper demonstrates an important takeaway: the activations hold sufficient information to be predictive of eventual misalignment in long thinking or reasoning traces. This can facilitate setting up effective test-time safety guardrails

### Weaknesses
The analysis seems to have an unaccounted pathway for leakage of information, which influences the findings and takeaways. The *activations* at the final token position of the last layer for each partial CoT (Line 163) implicitly encode the **prompt** itself, in addition to the subsequent CoT. This leads to a few issues:
- This potentially explains the effectiveness of the linear probe: if the prompt itself is indicative of the final misalignment of the response, the CoT segment is not required for the prediction of misalignment. And if this embedding representation of the partial CoT does include information about the prompt (since the LLM has first parsed the prompt before the CoT and encoded it within its activations/parameters) that should suffice.
- This results in a confounding factor for evaluating the central claim of the paper: it is not *just* partial CoTs that are being tested for being predictive for eventual misalignment.
- This invalidates the comparison with text based monitoring, since the text of the CoT by itself does not also include the prompt.

It would be helpful if the authors validate and clarify this. I am open to re-assessment if the authors have followed a procedure that differs from the description in Line 162–163. 

If not, and assuming the updated claim is: activations of an LLM during its CoT processing are predictive of eventual misalignment, a drawback that arises is that such monitoring can only be applied to models whose activations are accessible, and thus not closed-source models.

### Questions
- In Line 193, what is the value of $d$? That is, how aggressive is the dimensionality reduction with PCA?
- How is “foresight” used for training? As per the task description, only “observed” must be used to predict eventual misalignment.
- It is surprising to see that the prediction accuracy is already very high with even 0-10% CoT (Figure 4). How is that the case? It would be helpful if the authors can provide a couple of examples of such cases. Is this a by-product of the prompt being encoded in activations used as input to the linear probe?

### Soundness
2

### Presentation
3

### Contribution
3
