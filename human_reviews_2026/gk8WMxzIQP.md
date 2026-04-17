# Speech-to-LaTeX: New Models and Datasets for Converting Spoken Equations and Sentences

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Conversion of spoken mathematical expressions is a challenging task that involves transcribing speech into a strictly structured symbolic representation while addressing the ambiguity inherent in the pronunciation of equations. Although significant progress has been achieved in automatic speech recognition (ASR) and language models (LM), the problem of converting spoken mathematics into LaTeX remains underexplored. This task directly applies to educational and research domains, such as lecture transcription or note creation. Based on ASR post-correction, prior work requires 2 transcriptions, focuses only on isolated equations, has a limited test set, and provides neither training data nor multilingual coverage.  To address these issues, we present the first fully open-source large-scale dataset, comprising over 66,000 human-annotated audio samples of mathematical equations and sentences in English and Russian, drawn from diverse scientific domains. In addition to the ASR post-correction models and few-shot prompting, we apply audio language models, demonstrating comparable character error rate (CER) results on the MathSpeech benchmark (28\% vs. 30\%) for the equations conversion. In contrast, on the proposed S2L-equations benchmark, our models outperform the MathSpeech model by a substantial margin of more than 36 percentage points, even after accounting for LaTeX formatting artifacts (27\% vs. 64\%). We establish the first benchmark for mathematical sentence recognition (S2L-sentences) and achieve an equation CER of 40\%. This work lays the groundwork for future advances in multimodal AI, with a particular focus on mathematical content recognition.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces the first large-scale open-source dataset of spoken mathematical expressions and sentences in English and Russian, containing about 66,000 human-annotated audio samples from diverse scientific domains. It also provides evaluations of several speech-to-latex methods, including ASR post-correction, few-shot prompting, and audio-LLM integration, showing improvements over existing baselines such as MathSpeech.

Overall, this is a useful dataset paper with clear execution but limited novelty and a somewhat narrow scope for ICLR.

### Strengths
The work contributes a valuable resource for an understudied task and offers solid baselines that could support future research. The dataset appears well-organized and the experiments are clearly presented.

### Weaknesses
However, the topic is quite niche and seems better aligned with speech or audio processing venues such as Interspeech or ICASSP rather than ICLR. The scientific originality is limited, the main contribution is dataset creation and benchmarking rbut for a very nich domain.

### Questions
Some details need clarification, including which crowdsourcing platform was used for human annotation and the ratio of synthetic to natural speech samples in the dataset created. Providing these is important for the dataset publication.

### Soundness
3

### Presentation
3

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
In this paper, the author proposes a new benchmark dataset for the speech-to-LaTeX transcription task. Compared with the existing benchmark, MathSpeech, the proposed dataset includes a broader range of samples that better reflect real-world speech-to-text usage. It introduces more challenging cases, such as mixed text and mathematical formulas, making the benchmark more representative of practical scenarios. In addition, unlike MathSpeech, which relies on synthetic speech, this dataset is collected from real human speakers, further enhancing its realism and applicability. Using the proposed dataset, the author trains models with two different processing approaches, following established speech-to-LaTeX training pipelines, and benchmarks their performance to evaluate the effectiveness of current algorithms in this direction.

### Strengths
1. The benchmark dataset proposed in this paper covers a wider range of diverse, real-world scenarios, making it a valuable contribution to the research community.
2. The paper provides a comprehensive exploration of the dataset’s potential applications, including post-transcription correction and end-to-end multimodal fine-tuning.
3. The dataset-splitting strategy is well designed, and the experimental results effectively demonstrate the generalization ability of different models when adapting to various languages and previously unseen formula patterns.

### Weaknesses
1. The technical novelty of the paper is limited. The work mainly reimplements common approaches to fine-tuning models without proposing new model architectures or training algorithms to address the issues identified in the experiments.
2. The Character Error Rate (CER) remains relatively high in many cases, and the paper lacks an in-depth analysis of the causes behind these failures. A more thorough investigation into the unexpected phenomena observed in the results would strengthen the contribution.
3. Several parts of the paper are not well organized. For instance, some sections—such as the discussion of OCR methods in the related works—contain redundant content. Additionally, Table 2 is placed far from the results section, making it inconvenient for readers to reference while reviewing the experimental findings.

### Questions
1. For the few-shot learning experiments, were they conducted on a fine-tuned model or directly on the original pretrained model?
2. Could the authors provide more details about the data split ratios used in different experimental settings (e.g., training vs. testing)? For instance:
    - What is the ratio of training to testing equations in the Disjoint setting?
    - What are the training and testing sample sizes under the source-type split?
    - In the monolingual and bilingual experiments, do you use all available examples from one language plus a portion of the other language for training? If so, please specify the ratio used.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Speech-to-LaTeX (S2L)—the large-scale, open dataset for converting spoken math into LaTeX—addressing limits of prior work that used only synthetic audio, required two ASR passes, and handled only isolated equations. S2L includes 66k human and 571k TTS samples of equations and sentences in English/Russian. Two solutions are evaluated: (1) Whisper + Qwen post-correction, and (2) end-to-end audio-LLMs (e.g., SALMONN). On S2L-equations, Qwen reaches ~25–30% CER, and SALMONN achieves 17.5%, far outperforming MathSpeech (~64% on S2L). The paper also establishes the first benchmark for spoken mathematical sentences, where multimodal models remain strongest but accuracy drops due to contextual ambiguity.

### Strengths
- **Large, open S2L resource:** Releases a two-part dataset (S2L-equations, S2L-sentences) with multilingual coverage (English/Russian), mixing 66k human and 571k synthetic clips, collected from diverse sources and 33 annotators. This addresses the data bottleneck and standardizes evaluation.
- **Clear task framing & thorough splits:** Uses disjoint-formula splits, human vs. TTS source splits, and mono vs. bilingual training setups—plus KaTeX-based equation normalization—to probe generalization beyond memorization.
- **Strong baselines across paradigms:** Compares post-ASR pipelines (Qwen 0.5B/1.5B/Math/7B) with end-to-end Audio-LLMs (SALMONN, Flamingo-3, Gemma-3n, Qwen-Audio), giving a balanced picture of modular vs. E2E.
- **Transparent MathSpeech comparison:** After normalization, MathSpeech degrades to 64% CER on S2L-equations, while the proposed models remain strong; also competitive on the original MathSpeech benchmark.
- **Practical touches in evaluation:** High KaTeX compile rate (≈98–99.5%) suggests many errors are minor formatting differences rather than catastrophic syntax failures.

### Weaknesses
- **Compute/latency not discussed:** The best model (SALMONN-13B) is likely heavy; no throughput/latency/memory reporting limits practical takeaways for real-time use.
- **Ambiguity handling is under-analyzed:** The work acknowledges inherent ambiguity (“one over x plus two”) but gives limited breakdowns by ambiguity type or guidance on disambiguating conventions during annotation/evaluation.
- **Model behavior anomalies lack ablations:** 7B (LoRA-tuned, frozen base) underperforms fully fine-tuned 1.5B on equations; Qwen-Audio “fails completely.” Causes are hypothesized but not probed via targeted ablations.
- **Multilingual training instability:** Adding Russian sometimes harms English performance; analysis is brief (imbalance) without deeper diagnosis of cross-lingual interference.
- **Real-world robustness is unclear:** Data is largely clean/read speech; the gap to noisy, disfluent lecture audio (and multimodal cues, e.g., slides/pointing) isn’t evaluated.
- **Tokenization tweak had no effect:** Adding special LaTeX tokens didn’t help; this is noted but not further explored (e.g., tokenizer training, subword strategies).
- **Metric–semantics gap:** Heavy reliance on CER/TeXBLEU—even with normalization—can penalize semantically equivalent LaTeX or underplay mathematically serious mistakes; no semantic/visual-equivalence metric is reported.

### Questions
**Practicality & generalization**

- What are end-to-end latencies and memory footprints for the strongest systems, and can lighter E2E models approach SALMONN’s accuracy?
- How robust are models to real-world conditions (noise, disfluencies, accents, mic distance)? Any stress-tests or augmentation plans?
- Multilingual effects: can you quantify negative transfer vs. language imbalance and test mitigation (balanced sampling, language tags, adapters)?
- Fairness of comparisons: would training a MathSpeech-style architecture on S2L (or retraining Qwen on MathSpeech’s 6–8M) clarify the role of data vs. model size?

**Dataset & evaluation**

- How were inherently ambiguous utterances annotated (e.g., explicit “parentheses” vs. natural speech)? Any inter-annotator agreement stats and adjudication policies for these cases?
- Could you add a semantic check (e.g., render-and-compare, AST comparison) alongside CER/TeXBLEU to better reflect user-perceived correctness?
- Do you have a typology of common error modes (nested fractions, integrals, Greek variants, boundaries between text and math) for equations-in-sentences?
- Can you release the KaTeX normalization code and exact split scripts to facilitate strict reproducibility?

**Modeling choices & ablations**

- What ablations isolate why 7B (LoRA) < 1.5B (full FT) on equations? Is it LoRA rank, target layers, or optimizer/schedule?
- Why did Qwen-Audio fail? Can you share a brief post-mortem (feature pipeline, adapter alignment, loss setup)?
- Did adding 400k MathBridge-derived TTS samples change error composition (e.g., more bracket errors decreased), or only average CER?
- Given that Qwen-Math didn’t clearly beat Qwen on this task (inputs are NL), did math-specialized token priors help particular symbol families or structures?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses Speech-to-LaTeX for both isolated equations and full sentences, releasing a large bilingual dataset with human and TTS audio. It compares Whisper with LLM post-correction against end-to-end Audio-LLMs like SALMONN-13B, showing strong gains over prior works (MathSpeech).

### Strengths
1) This paper targets Speech-to-LaTeX, an underexplored but impactful task for education and research with large-scale S2L dataset.

2) It shows strong empirical results on S2L-equations (English). SALMONN achieves 17.5% CER, outperforming MathSpeech and Qwen; on S2L-sentences, SALMONN attains the best equation CER (39.7%).

### Weaknesses
1) Gap to real lecture conditions. Authors note the dataset does not capture paraphrases, incomplete expressions, or audio-visual coupling typical of classroom settings.

2) It is difficult to verify the reliability of the dataset presented in the paper. Although some sample data are available in the supplementary material, those samples are insufficient to establish whether the constructed dataset adequately covers diverse scenarios of spoken mathematical expressions.

3) Reproducibility risk for one baseline. Qwen-Audio “fails completely,” attributed to re-implementation differences—suggesting pipeline brittleness.

### Questions
1) Could you quantify how much equation normalization (and lowercasing) changes CER/TeXBLEU per model? Also, any experiments with structure-aware or render-equivalence metrics? Concrete numbers would help judge true semantic progress. 

2) Did you try constrained decoding (grammar or AST constraints) in the post-correction step to reduce bracket/scope errors? If so, how did it affect latency and CER?

3) This paper states, ‘We release the first large-scale, open-source dataset of spoken mathematical expressions and sentences.’ Is there an anonymized data pages related to this?

### Soundness
2

### Presentation
3

### Contribution
2
