# Turning Speech Language Models into Multilingual Listeners

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Speech Language Models (SLMs) that understand spoken language questions and commands support only a few high-resource languages, limiting access to modern technology for millions of speakers worldwide. This gap in language coverage stems from the scarcity of multilingual speech-language instruction-tuning datasets. To address this issue, we present MULTISPEECHQA, a large-scale, synthetically generated and human-verified dataset comprising 9200 hours of more than 10.8 million spoken question-answer pairs in 23 typologically diverse languages, designed to improve the multilingual instruction-following capabilities of SLMs. Using MULTISPEECHQA, we also introduce MULTISPEECH-BENCH, a multi-task benchmark to evaluate SLM performance across 23 languages. We compare the performance of a strong cascading system to three leading open-weight SLMs on MULTISPEECH-BENCH and find that the cascading system outperforms all existing open-weight SLMs. We then demonstrate the effectiveness of MULTISPEECHQA by fine-tuning the best-performing open-weight SLM, Qwen 2.5-Omni, on our dataset, which substantially improves its performance and establishes new state-of-the-art results for open-weight models on our benchmark. Our findings show that high-quality synthetic datasets offer a scalable solution to improving the multilingual capabilities of SLMs, extending the benefits of natural spoken interactions to a wider range of language

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper tackles the scarcity of multilingual instruction-tuning data and evaluations for spoken QA by releasing MULTISPEECHQA—10.8M QA pairs (~9,200 hours) across 23 languages—built by translating VA-400K prompts/answers and synthesizing spoken questions (XTTS/Seamless/MMS), with native-speaker checks showing good comprehension but mixed naturalness. It also introduces MULTISPEECH-BENCH, which combines a human-corrected QA subset with CommonVoice (ASR) and CoVoST-2 (AST) to evaluate SQA/ASR/AST uniformly. Using LLM-as-a-judge, the authors find that among open-weight SLMs Qwen2.5-Omni is strongest on QA, while a strong cascading baseline (Whisper v3 → Aya 8B) remains competitive. LoRA-finetuning Qwen2.5-Omni on MULTISPEECHQA lifts its average QA win rate to 60.6% versus the base model, with ASR/AST staying roughly unchanged. A from-scratch SALMONN-style study shows training on all 23 languages beats using 10, while adding 20% AST data doesn’t yield consistent gains. Overall, the work argues that large-scale synthetic multilingual SQA can effectively post-train open-weight SLMs and provides datasets/benchmarks intended for community use.

### Strengths
1. Propose a speech dataset that has good coverage on 10.8M spoken QA pairs (~9,200 hours) across 23 languages, which is rare for SQA datasets.
2. Plans to release data and open-weight models; pipeline uses public MT/TTS. It is good for speech research community.
3.  Native-speaker ratings (≥2 raters per language, except Czech) show good content-understanding, diagnosing TTS naturalness as the main bottleneck.

### Weaknesses
1. QA win rates rely on a single LLM-as-a-judge (Command-A) without reported human calibration or bias checks.
2. BLEU-only for AST; lacks native-speaker evaluation and more comprehensive semantic metrics.
3. Core dataset is synthetic speech; limited evidence on robustness to real, noisy, accented speech.
4. A strong Whisper→LLM cascade remains hard to beat; ASR/AST see little improvement after fine-tuning.

### Questions
1. In Fig. 3 & 4, the win rates are evaluated by a single LLM (Command-A). Did you run any judge-bias checks to ensure its preferences have a positive correlation with human judgments? My suggestion is to run an experiment to verify the alignment between Command-A and human raters, rather than using Command-A simply because it covers these languages to evaluate translation quality. Otherwise, this setup may ignore the relationship between model preferences and human preferences. Thus, the resulting metric is not very convincing to me.

2. Although Qwen and Phi do not disclose their training data, they may have used real-world audio. MULTISPEECHQA uses synthetic speech—could this harm these models’ understanding of real speech? Beyond reducing compute, was LoRA also chosen to avoid such degradation? If we train a speech model from scratch and include MULTISPEECHQA as training data, would the resulting model show a gap in understanding real-world speech compared with models trained on real audio? I believe an ablation study is needed to substantiate the dataset’s quality and to assess any potential risks to speech language models pretraining.

3. Using WER for ASR is fine. But is BLEU too limited for AST? BLEU is an n-gram overlap metric and is less sensitive to paraphrases, multilingual/morphologically rich languages, and semantic equivalence. I don’t think it is a good evaluation metric. Would human evaluation by native speakers in these languages be better?

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
This paper introduces MULTISPEECHQA, a large synthetic multilingual spoken QA dataset spanning 23 languages. It is created by translating the Voice Assistant 400K text pairs and synthesizing the speech with text-to-speech systems. In addition, it presents MULTISPEECH-BENCH, a multilingual and multi-task benchmark that evaluates Spoken QA, ASR, and AST tasks. The authors benchmark several open-weight spoken language models and a strong cascade baseline, then fine-tune Qwen2.5-Omni on MULTISPEECHQA, which yields higher QA win rates. Overall, the paper aims to scale multilingual instruction-following for spoken language models through synthetic data generation and standardized evaluation.

### Strengths
- Expanding SLMs beyond high-resource languages is both timely and impactful, as the paper addresses a key challenge: data scarcity in multilingual spoken instruction-following.
- MULTISPEECH-BENCH unifies QA, ASR, and AST tasks across the same set of languages, providing a strong cascade baseline and evaluations with multiple open-weight SLMs, which makes it a valuable resource for the community.
- The authors at least measure synthetic quality (naturalness, content) and manually verify the QA test subset, which is a good practice even if the results expose weaknesses.

### Weaknesses
- The heavy reliance on an LLM-as-a-judge for multilingual QA without verifying its correlation with human judgments reduces trust, especially for typologically diverse languages and speaking styles. Conducting a small human evaluation on the QA test set (beyond TTS or translation quality) would help.
- Human edits were required for 72% of translations in the benchmark subset, and the TTS outputs show only moderate naturalness on average, with notably low scores in some languages. These issues raise concerns about the overall data quality.
- Since MULTISPEECHQA and MULTISPEECH-BENCH originate from the same data source, it is not surprising that the fine-tuned Qwen model achieved state-of-the-art results on MULTISPEECH-BENCH. Moreover, the lack of a clear separation between speaker sets in the training and test data could lead to speaker overlap, which may influence the evaluation results.
- As a benchmark paper, although the authors argue that existing benchmarks lack sufficient language coverage, they do not discuss any existing speech or audio language model benchmarks. The paper would benefit from a more thorough comparison between MULTISPEECH-BENCH and other available benchmarks in this area.

### Questions
In addition to the weaknesses mentioned above, I have the following questions and comments:
- In Table 3, ASR results are reported using WER. However, this metric may not be suitable for certain languages such as Japanese and Chinese, where CER is typically used. Could you clarify which metric was applied?
- Typically, a table’s caption is placed above the table, but Tables 1 and 2 do not follow this convention.
- Figure 2 appears to be missing a reference in the main text.
- There are several typos throughout the paper, including “speeech,” “eplicitly,” “whcih,” “AYA Expance 8B,” and “BLUE score.”

### Soundness
2

### Presentation
2

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
This paper attempts to address the lack of multilingual instruction-tuning data for Speech Language Models (SLMs), based on the fact that most existing SLMs focus on English or some other high-resource languages, limiting accessibility and global applications. It proposes a synthetic data pipeline to synthesize multilingual instruction-tuning data and open-source the synthesized datasets. The paper demonstrates that automated synthetic data pipelines can effectively scale multilingual capabilities of SLMs.

Main contributions:
1. MULTISPEECHQA - a large-scale, synthetic multilingual datasets with 10.8 million spoken question-answer pairs in 23 languages.
2. MULTISPEECH-BENCH - a multilingual benchmark suite for evaluating SLMs on SQA, ASR and AST.
3. Synthetic data pipeline for multilingual instruction-tuning employing machine translation models and multilingual TTS systems that produces high-quality speech-text pairs. The pipeline also validate the data quality by human evaluations.

### Strengths
Originality:
The paper introduces an approach for scaling multilingual SLMs through synthesizing and human-verifying data based off English datasets.  The originality of the paper lies in 

Quality:
1. The paper covers details of the dataset construction pipeline, and introduces quantitative human evaluation by native speakers on the naturalness and comprehension quality of the synthetic data of each language.
2. The authors conducted comprehensive benchmarking on the generated evaluation set, containing cascaded system baseline and open-source SLMs such as Qwen2-Audio, Phi-4-Multimodal, etc.

Clarity
The manuscript is well-organized and easy to follow.  Proper figures and charts make the paper more intuitive.

Significant
The release of MULTISPEECHQA and MULTISPEECH-BENCH fills gaps in multilingual SLMs training and evaluation, enabling the research community to develop more inclusive and accessible SLMs.

### Weaknesses
Potential limitations of data quality:
1. Given the scale of the synthetic data used in MULTISPEECHQA (10.8 million samples), the scale of the human validation is 2 small (20 samples per language).
2. Variance in TTS naturalness scores may indicate that some languages contain more noise in the dataset.

Evaluation setup:
1. LLM-as-a-judge: the judge LLM's multilingual capability needs to be calibrated (perhaps using other LLMs to test the correlation/stability, or even human validation).
2. Limited task diversity in evaluation: other SLM benchmarks (such as AIR-Bench[1]) contain more diverse tasks, it may be worth extending the bench suite to cover more (e.g. summarization, language ID, emotion recognition)
3. Due to the absence of commercial SLMs (e.g. GPT4o, gemini) in benchmark, the claim of SOTA results is thin.

There's lack of ablation study, it's not clear which components of MULTISPEECHQA datasets contribute the most to the gain.


[1] Yang, Qian, et al. "Air-bench: Benchmarking large audio-language models via generative comprehension." arXiv preprint arXiv:2402.07729 (2024).

### Questions
See suggestions in Weakness section.

### Soundness
3

### Presentation
3

### Contribution
3
