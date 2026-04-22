# UniSS: Unified Expressive Speech-to-Speech Translation with Your Voice

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
The ultimate goal of expressive speech-to-speech translation (S2ST) is to accurately translate spoken content while preserving the speaker identity and emotional style. However, progress in this field is largely hindered by three key challenges: the scarcity of paired speech data that retains expressive styles, the complexity of multi-stage processing pipelines, and the limited transfer of translation capabilities from large language models (LLMs). In this work, we address these challenges by introducing UniSS, a novel single-stage framework for expressive S2ST. Our approach features carefully designed speech semantic and style modeling, enabling seamless integration with existing text-based LLM frameworks to develop a unified text-speech language model. To transfer translation capabilities from text to speech, we propose a cross-modal chain-of-thought prompting process that progressively aligns audio semantics with text and ensures style preservation in the decoded results. Furthermore, we construct and release a large-scale, high-quality expressive S2ST dataset, UniST, comprising 44.8k hours of data. Experimental results show that UniSS significantly outperforms previous methods in translation fidelity and speech quality while preserving voice, emotion, and duration consistency. Our work establishes a simpler and more effective paradigm for building the next generation of expressive S2ST systems. Audio samples are available at https://cmots.github.io/uniss-demo/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work introduces UniSS, a novel single-stage framework for expressive speech-to-speech translation (S2ST) that addresses key challenges such as the scarcity of expressive paired speech data, the complexity of multi-stage pipelines, and the limited transfer of translation capabilities from large language models. UniSS integrates speech semantics and style modeling through cross-modal chain-of-thought prompting, transitioning from text-only models to text-to-speech generation. Opensourced 44.8k hours of high-quality expressive S2ST data, UniSS achieves substantial improvements in translation fidelity and speech quality, while effectively preserving voice, emotion, and duration consistency compared to prior approaches.

### Strengths
- Introduces a single-stage autoregressive model that directly builds on a pre-trained textual LLM, simplifying prior multi-stage architectures for expressive speech-to-speech translation.
- Provides an open-source, reproducible pipeline for constructing expressive S2ST datasets, including UniST, a large-scale Chinese-English corpus preserving speaker identity and emotional style.
- Delivers a thorough evaluation combining objective metrics (Speech/Text-BLEU, A.PCP, SLC, UTMOS) with subjective MOS assessments of emotion, speaker similarity, and naturalness.

### Weaknesses
- Limited language coverage: Supports only Chinese–English translation.
- Ununified tokenizer: Using separate tokenizers for linguistic and semantic tokens raises concerns about vocabulary size and token consistency.
- Language model considerations: Employing a large LLM (Qwen2.5-1.5B-Instruct) for translation may be excessive compared to traditional MT models; more efficient parameter training could better leverage other model capacities.

### Questions
- Is the prompt used during multi-task training also required at inference for different tasks? If so, how would it impact the available context length?

### Soundness
3

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
5

### Summary
The paper presents **UniSS**, a unified single-stage speech-to-speech translation (S2ST) system designed to translate spoken content while preserving both speaker identity and expressive emotional style. UniSS eliminates multi-stage pipelines by leveraging a pretrained large language model (LLM) backbone, incorporating speech and text with a shared token vocabulary, and introducing cross-modal chain-of-thought (CoT) prompting that transfers text translation capabilities to the speech domain. Besides model proposals, the authors construct UniST, a large-scale expressive S2ST dataset (44.8k hours). Experimental results on several benchmarks demonstrate substantial improvements in translation fidelity, expressiveness, and runtime efficiency versus established baselines.

### Strengths
1. UniSS is a conceptually clean, single-model approach that collapses ASR, MT, and TTS into one autoregressive LLM with expanded vocabulary for both text and discrete speech tokens.
2. The construction of UniST (44.8k hours) as a high-quality, diverse Chinese-English corpus with explicit control over speaker, emotion, and timing, is a notable resource for the field.

### Weaknesses
1. The subjective evaluation does not include MLLMs such as GPT-4o, and the model’s performance is also inferior to Seed Live or Seamless-Ex on certain metrics.

2. I don’t think the so-called cross-modal prompting can be regarded as a form of Chain of Thought (CoT).


3. Despite its claim of being “unified,” UniSS relies on three distinct speech tokenizers (speaker, linguistic, and semantic), each originating from separate models. This design increases the vocabulary size (over 180k tokens) and may limit scalability, efficiency, or ease of deployment when supporting additional languages or larger vocabularies.

### Questions
1. In Phase 1, is the TTS task really useful? After all, your ultimate goal is translating source speech to target speech.

2. How reliable is the metric A.PCP?

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
The paper introduces UniSS, a single-stage autoregressive framework for expressive speech-to-speech translation (S2ST) that preserves speaker voice, emotion, and duration while achieving high translation fidelity. Built on the pre-trained Qwen2.5-1.5B-Instruct LLM, it uses a triple-tokenizer strategy (speaker, linguistic, semantic tokens) and cross-modal CoT prompting to transfer text translation capabilities to speech. Two modes (Quality and Performance) balance fidelity and efficiency. The authors release UniST, a 44.8k-hour Chinese-English expressive S2ST dataset. Experiments on CVSS-T and FLEURS show superior performance over baselines like SeamlessM4T, GPT-4o, and cascaded systems in metrics such as Speech-BLEU, prosody consistency, duration compliance, and speech quality. Subjective MOS evaluations confirm high emotion/speaker similarity and naturalness.

### Strengths
1. The paper addresses an important problem in speech-to-speech translation: preserving the speaker's voice, emotional style, and prosody across languages. This is crucial for applications like multilingual communication or media dubbing. The motivation is clear, and the cross-modal CoT approach effectively aligns text-based LLM capabilities with speech in a single model. This method is innovative and necessary, as ablations show that removing CoT leads to major drops in performance.

2. The authors release UniST, a 44.8k-hour dataset for expressive S2ST, which tackles the key issue of limited paired data with style preservation. The creation pipeline is well-documented, making it valuable for the research community.

3. The experiments cover a broad set of metrics, including semantic and acoustic part. Baselines are varied, spanning cascaded systems and end-to-end models, ensuring robust comparisons.

4. UniSS outperforms baselines on most metrics, such as emotion similarity and duration matching. The demopage also illustrates the model's success in maintaining expressive qualities.

5. The paper is well-structured and readable, with clear descriptions of the architecture, training strategy, and ablations that validate the approach.

### Weaknesses
1. The novelty is limited. The model's strength in maintaining speaker voice, emotion, and prosody largely relies on pre-existing components, such as the BiCodec tokenizer for speaker representations and SparkTTS for synthesizing expressive data. UniSS itself mainly integrates these by organizing a multiple token sequence then inputing it into a LLM, reducing its core innovation and novelty.

2. The "Single-Stage" claim is misleading.  While positioned as a streamlined single-stage model to avoid cascaded errors and latency, the Quality mode generates intermediate source text (T_src) and target text (T_tgt) before speech tokens, mirroring the sequential steps of a traditional 3-stage pipeline (ASR → MT → TTS). This undermines the claimed advantages and could lead to similar error accumulations and delays.

3. Training and evaluations are centered on English-Chinese pairs. Besides, the use of GLM tokenizer and SparkTTS also constrains the method for applying on other languages. This raises questions about scalability to diverse languages, accents, or noisy real-world scenarios, where data scarcity for expressiveness could amplify issues.

4. A significant issue is the unfair setting in evaluations, particularly for translation accuracy (BLEU). The UniST training data uses high-quality translations from the large Qwen2.5-72B model, while the 3-stage cascaded baseline relies on the much smaller pretrained NLLB-600M for machine translation. Why not use Qwen-2.5-1.5B-Instruct LLM in the cascaded system?

5. The framework's heavy reliance on text-based elements. For example, CoT prompting with intermediate transcripts makes it unsuitable for unwritten or low-resource languages without scripts. This contradicts goals in textless S2ST research and should be highlighted as a key limitation, especially given the focus on expressive translation.

### Questions
1. To make a fair comparison, it will be better if you test another cascaded baseline that uses the same Qwen-2.5-1.5B-Instruct model for MT and the same SparkTTS model for TTS. Besides, consider training the baselines on the UniST dataset. This would show if the UniSS framework is truly better, or if it just benefited from better components and training data.

2. How does the cross-modal CoT prompting specifically bridge text and speech modalities? Provide some studies of intermediate outputs (e.g., T_src, T_tgt) to explain whether UniSS addresses error propagation problems.

3. The motivation emphasizes unifying expressiveness, but much relies on existing tools (BiCodec, SparkTTS). What novel insights does UniSS offer for LLM-based S2ST, and how does it differ from prior CoT adaptations in multimodal tasks?

4. For training strategy, what is the individual contribution of each task to overall performance? Could you add ablations removing or weighting specific tasks to quantify their effects?

5. Since UniST is fully a synthetic dataset generated by SparkTTS, is it unable to effectively capture word-level semantic prosody mapping or relevance in speech translation? For instance, the transfer of semantic nuances such as emphasized stress. It is recommended to provide corresponding analysis and discuss the potential limitations.

### Soundness
3

### Presentation
3

### Contribution
3
