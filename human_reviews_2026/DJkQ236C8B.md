# JALMBench: Benchmarking Jailbreak Vulnerabilities in Audio Language Models

- Decision: Accept (Poster)
- Scores: 6, 2, 6

## Abstract
Large Audio Language Models (LALMs) have made significant progress.
While increasingly deployed in real-world applications, LALMs face growing safety risks from jailbreak attacks that bypass safety alignment.
However, there remains a lack of an adversarial audio dataset and a unified framework specifically designed to evaluate and compare jailbreak attacks against them.
To address this gap, we introduce JALMBench, a comprehensive benchmark that assesses LALM safety against jailbreak attacks, comprising 11,316 text samples and 245,355 audio samples (>1,000 hours).
JALMBench supports 12 mainstream LALMs, 8 attack methods (4 text-transferred and 4 audio-originated), and 5 defenses.
We conduct in-depth analysis on attack efficiency, topic sensitivity, voice diversity, and model architecture.
Additionally, we explore mitigation strategies for the attacks at both the prompt and response levels.
Our systematic evaluation reveals that LALMs' safety is strongly influenced by modality and architectural choices: text-based safety alignment can partially transfer to audio inputs, and interleaved audio-text strategies enable more robust cross-modal generalization.
Existing general-purpose moderation methods only slightly improve security, highlighting the need for defense methods specifically designed for LALMs.
We hope our work can shed light on the design principles for building more robust LALMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors:

- create a benchmark for assessing the robustness of audio language models (ALMs) to text and audio jailbreaks
- benchmark the attack success rates (ASRs) of various jailbreaks against various models
- analyze ASRs along various dimensions of interest: efficiency, topics, languages
- analyze t-SNE embeddings to see how architecture affects embeddings of attacks

### Strengths
- Very solid, very thorough paper
- I thought Figure 1 is cute
- The models, attacks, and analyses seem sensible

### Weaknesses
Please begin with my line-by-line notes under "Questions".

Overall, I think this is a solid and thorough paper. I see two primary weaknesses that, if modified, would lead me to increase my score.

1. *All* of this uses a single judge model (gpt-4o-2024-11-20). I cannot find mention of another judge model (ideally, multiple other judge models) or any analysis of how accurate / reliable / trustworthy this particular judge model is. Thus, rather than understanding these results as describing the robustness of ALMs to jailbreaks, I must interpret the results as describing gpt-4o-2024-11-20's judgement of the robustness of ALMs to jailbreaks. This is especially important because the data that gpt-4o is judging is data designed to bamboozle (A)LMs, so we need to be certain that gpt-4o isn't similarly getting bamboozled! Fixing the bullet points under "Questions" and addressing this shortcoming would raise my score to an 8.

2. To me, I feel like the paper doesn't offer many (any?) conceptual insights or information to drive decision-making. It's more akin to: Here are tables and figures reporting robustness scores of ALMs and also looking at tSNE embeddings. That's reasonable, but what would make this paper impactful is if the authors could provide actionable recommendations for ALMs and robustness. Depending on the significant of the insights, if the authors fix the bullet points under "Questions", add more judges and analyze the judges, and provide meaningful insights, I could be persuaded to raise my score to a 10.

### Questions
## Title

- Succinct, clear, accurate! And easy to pronounce. I like the title.

## Abstract

- nit: since ALM stands for Audio Language Model, it’s a bit odd to call Language Models “Large”; otherwise, why are the ALMs not ALLMs? I feel like we’ve moved to Language Models as LMs. But feel free to disagree.
- The paper would be stronger if you could provide any insights or takeaways from evaluating models on JALMBench here in the abstract. I feel what separates mediocre benchmarks from great benchmarks is that great benchmarks don’t just provide scores but also drive insights and decision-making.

## Section 1 Introduction

- I like Figure 1! Very cute, very simple, very understandable. If I can make one minor suggestion, is “Dataset” a smaller size than “Statistics”? The two words appear to have different font sizes.
- Figure 1: I’m unclear on why there are two arrows from Attack to ALM, but only one arrow from Harmful Query to ALM and from ALM to Defense?
- Line 102: Numbers like “improve the average safety performance by 11.3%” mean little when the reader doesn’t have a strong prior over these numbers. It’d be helpful if you could add a sentence commenting on what this means practically. Does 11.3% mean that all attacks are blocked? That almost all attacks get through? etc.

## Section 2 Related Work

- No comments - looks good!

## Section 3 JALMBench

- nit: In general, I think methodology should be written in *past* tense. This is because the methodology is you (the authors) describing what you previously did.
- Line 163: “We manually curate and deduplicate” -> How did the authors actually validate that the dataset was properly deduplicated? Performing this task by hand seems tedious and error prone. Why not use something like n-grams or semhash (https://github.com/MinishLab/semhash) or other deduplication methods?
- Line 167: For comprehensiveness, it might be good to generate the speech using other methods/libraries e.g,. https://elevenlabs.io/. Otherwise, we don’t know whether the benchmark results are particular to Google TTS’s current model.
- Line 192-193: For BoN, with 600 independent variations of each harmful audio sample, how do you determine the ASR? Do you take a similar definition that the attack is successful if any of the variations are successful?

## Section 4 Evaluation

- Line 215: “We use GPT-4o … as the judge model”. How did you validate the correctness/accuracy/reliability of the judge model? How consistent are its judgements with other candidate judge models? Just like my above bullet point, we don’t otherwise know whether the benchmark results are merely GPT-4o’s judgements.
- Table 1: Without uncertainty quantification (e.g., 95% confidence intervals), it’s hard to know which (if any) of these values overlap
- Table 1: I personally find tables difficult to intuitively understand, and strongly prefer visualizations. Here’s one way you could convert this into a nice figure that communicates exactly the same information. I’ll use matplotlib+seaborn terminology, but feel free to use whatever library you prefer: Have two columns, one for text modality, one for audio modality. The y values should be the attacks (THarm, ICA, DI, DAN, PAP, etc.), the x values should be the ASRs, and each hue should be a different model. You can achieve this easily using seaborn’s catplot https://seaborn.pydata.org/generated/seaborn.catplot.html#seaborn.catplot or implement it yourself
- Lines 251-252: I think we need to be careful about comparing different methods if a method takes multiple attempts. Here, PAP is using 40 attacks and is considered successful if any attempt succeeds. If we want to make comparative statements about the efficacy of different methods, we should compare on a fair basis. I’m not sure what exactly that should look like, but the naive choice is that every method gets 40 tries and is considered successful if any attempt jailbreaks the model.
- Figure 2: Shouldn’t PAP be "PAP (N-40)" in the legend?
- Figure 2: For methods that take multiple attempts (PAP, BoN, maybe others), it might be useful to plot different “budgets”. How does BoN perform with N=10 attempts? 100 attempts?
- Lines 306-307;324-326: I’m not sure I agree with this characterization. As best as I can tell, these statements are about the average success per category. But I’m not sure that _averages_ are what we care about. If Method A and Method B aren’t successful jailbreaks on a topic, but Method C is, then an attacker can succeed. It’s not as if an attacker must use the “average” attack.
- Line 332: nit: there is a leading “,” by itself. It belongs on the line above.

### Soundness
3

### Presentation
3

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
The paper presents JALMBench, a large-scale benchmark designed to evaluate jailbreak vulnerabilities in Audio Language Models (ALMs). The authors compile a dataset of over 1,000 hours of audio and 11k text samples, covering 12 ALMs and 8 attack methods (4 text-transferred, 4 audio-originated), and 5 defenses. The benchmark measures attack success rate (ASR), efficiency, topic sensitivity, voice diversity, and architectural robustness. It also includes analyses of alignment gaps between modalities and evaluates prompt- and response-level mitigations.

### Strengths
1. **Comprehensiveness.** The scale and coverage of JALMBench may position the work as canonical testbed akin to JailbreakBench or AdvBench for text-based. It covers 12 ALMs, 8 distinct attacks, 5 defenses, and multidimensional analyses (efficiency, topic, voice, architecture).

2. **Reproducibility.** The paper provides an anonymous GitHub repository that unifies interfaces for ALMs and defenses, documenting generation pipelines that reflect high implementation quality.

3. **Empirical Findings.** a) Continuous-feature ALMs (e.g., SALMONN, LLaMA-Omni) exhibit cross-modal safety misalignment, whereas discrete tokenization (e.g., GLM-4-Voice) better preserves safety transfer. b) Audio-originated attacks (e.g., AdvWave) reach near ASR of 96%, showing ALMs are substantially less safe than text-only LLMs.

### Weaknesses
1. **Limited Novelty Beyond Benchmark Construction**. The paper primarily aggregates existing attack/defense methods without introducing fundamentally new algorithms or method. The main novelty only lies on integration and scale rather than the concept. Also, cite existing ALM jailbreaking attacks that shows overlap in both evaluation and method to provide the difference of this work compared to existing ones [1, 2]

2. **Overreliance on LLM-as-a-Judge**. The evaluation relies solely on GPT-4o judging outputs on a 5-point scale. Without human validation or inter-annotator calibration, this introduces potential bias and raises questions about consistency across attack modalities (text vs. audio). 

3. **Defense Evaluation Lacks Audio-Specific Techniques**. All defenses are adapted from text settings. The conclusion admits this limitation. However, this paper would be stronger if it proposed even a preliminary audio-native defenses (e.g., perturbation detection or phoneme-level filtering), considering that this is a Audio LM benchmark and that the paper's main novelty comes with the comprehensiveness.

4. **Ambiguities in Methodology**

- The paper does not specify how success thresholds (score greater than 4) were validated.

- Details on human-recorded data IRB approval are given, but the sample size (6 speakers) is too small to support the claims about "diversity"

-----
*References*

[1] Gupta, Isha, David Khachaturov, and Robert Mullins. "" I am bad": Interpreting Stealthy, Universal and Robust Audio Jailbreaks in Audio-Language Models." arXiv preprint arXiv:2502.00718 (2025).

[2] Roh, Jaechul, Virat Shejwalkar, and Amir Houmansadr. "Multilingual and multi-accent jailbreaking of audio llms." arXiv preprint arXiv:2504.01094 (2025).

### Questions
1. What is the reason behind choosing those specific ALMs? Because some of those models were already considered broken or not safely aligned (e.g., SALMONN and LLaMA-Omni)

2. Could the authors provide per-model refusal rates and false positives ("benign" classified as "unsafe" misclassification)?

3. Could the authors provide any metric or quantitative evaluation of transcription capability of the selected ALMs? As a reviewer, I am curious whether these models accurately understood the input and rather they are not answering non-sense.

4. What are the benign speech question answering capabilities of these chosen ALMs? 

5. What is the reason that for majority of the models text modality has a slightly / significantly lower ASRs compared to audio modality?

### Soundness
2

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
4

### Summary
The authors introduce what seems to genuinely be the first benchmark for audio language model robustness that fully takes into account, in particular, the variety of model architectures (as well as ASR variation across language & accents). They evaluate 12 state of the art audio models (some where audio is converted into text tokens, some where the audio gets passed as audio tokens simultaneously with the text input) against 8 attacks and 5 defenses.

### Strengths
I think overall the paper gives us valuable insight into the differences between discrete tokenization and continuous encoding approaches, though the defense evaluation could be strengthened with multi-objective analysis.

I'm convinced that the paper's contribution is overall original, and in particular it seems important to be able to compare across different architectures for audio models. The benchmark itself is pretty extensive on the fronts of both the models and the attack methods evaluated. The paper overall represents substantial engineering effort to generate all of the audio attacks. 

The empirical evidence of the t-SNE visualisation from fig 5 makes me quite confident in the author's conclusion that models trained on interleaved audio & text data are likely more robust against audio attacks. This seems like a valuable insight that model developers should take into account. 

The authors also provide useful insights in identifying the different axes on which attacks are more or less successful, especially in the variability of robustness across categories of harm (e.g. misinformation requests being much more successful than explicit hate speech), and across the diversity of voices (e.g. how attacks with non-US accents seem to have higher ASR). The section on attack efficiency (by query length in seconds) is also useful in understanding the variability of different types of attack.

The paper is overall well written and clearly presented, and the claims made are well-calibrated. For example, they rightly frame their defense work as more preliminary.

### Weaknesses
As above, I think the paper is overall quite useful. However, in the analysis & framing of the defense results, I'd be interested in the authors making some aspects a little clearer. In particular, while table 4 is helpful for understanding the raw ASR reduction from defense methods, I have to jump all the way to the appendix to get the full breakdown of ASR reduction vs capability retention. And, even then, the authors don't emphasise this tradeoff much in their analysis of the defenses, which makes it hard for me to know what the takeaways really are for defense methods. I also think a more impressive benchmark would have deeper things to say about other differences across defenses, like which attacks are particularly successful against which defenses/architectures and why, but perhaps this was out of scope. 

I understand and appreciate that the defense part of the paper is framed as more exploratory, but given that you have the data for this, I'd appreciate more depth of analysis on which defense methods are pareto-optimal across safety & utility. I think having something like a pareto plot and a few more comments here would help me draw more useful conclusions about the state of how to go about defending audio-llms.

Another significant area where I wish the paper had more depth was in the analysis of the differences between architectures. I appreciate that this is first large-scale analysis of how differences in architecture affect ALM robustness. Further, it's helpful that ground their hypothesis that discrete tokenization leads to more consistent safety in the resultant model by empirically visualising the last hidden layer’s representation in the backbone LLM with t-SNE. However, I think the authors could have gone into even more depth. First of all, it'd be helpful to get this visualisation for more models, rather than just one representative model for each architecture. Secondly, we only see inputs for one attack, PAP (admittedly the most successful). Overall, I'm left not totally confident that we understand where the gaps between different model architectures come from - especially in the differences between Quen2-audio and LLaMA-Omni (since the audio encoder are similar, which to my understanding was the hypothesis). It'd be helpful for the field to know why Qwen2-audio is more robust than LLaMA-Omni, and I'm not totally sure where the author's claim - that joint alignment objectives might be the reason - comes from.

Also, more minor, but I was a little surprised/confused by the multilingual attack results - it looks like ASR drops in table 10 for attacks in non-English languages. The authors claim this is due to "reduced model proficiency due to limited training data in non-English languages" - but I'm not sure if the authors really have the evidence to claim this. I'd expect it could be about as likely that the drop is due to imperfect translation, and I'd appreciate any extra detail that the authors could provide to support either hypothesis here.

### Questions
1. Can you provide more commentary on, and perhaps a Pareto plot for, the safety vs. utility for defense methods? Which defenses are Pareto-optimal? I think this would make me satisfied with Section 5.
2. Can you go into the architectural tradeoffs in more depth? What are the performance costs (latency, accuracy on benign tasks) of discrete vs. continuous architectures? Can you quantify the safety-performance tradeoff?
3. What is the baseline ASR for vanilla harmful requests (no jailbreak) across different languages? This would address my minor confusion/concern about the claims around translation of attacks.
4. More of a clarification, but can you provide more detail on the 6,000 GPU hour cost breakdown? What portion was dataset generation vs. evaluation?

### Soundness
3

### Presentation
2

### Contribution
3
