# VowelPrompt: Hearing Speech Emotions from Text via Vowel-level Prosodic Augmentation

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 8

## Abstract
Emotion recognition in speech presents a complex multimodal challenge, requiring comprehension of both linguistic content and vocal expressivity, particularly prosodic features such as fundamental frequency, intensity, and temporal dynamics. Although large language models (LLMs) have shown promise in reasoning over textual transcriptions for emotion recognition, they typically neglect fine-grained prosodic information, limiting their effectiveness and interpretability. In this work, we propose VowelPrompt, a linguistically grounded framework that augments LLM-based emotion recognition with interpretable, fine-grained vowel-level prosodic cues. Drawing on phonetic evidence that vowels serve as primary carriers of affective prosody, VowelPrompt extracts pitch-, energy-, and duration-based descriptors from time-aligned vowel segments, and converts these features into natural language descriptions for better interpretability.
Such a design enables LLMs to jointly reason over lexical semantics and fine-grained prosodic variation. Moreover, we adopt a two-stage adaptation procedure comprising supervised fine-tuning (SFT) followed by Reinforcement Learning with Verifiable Reward (RLVR), implemented via Group Relative Policy Optimization (GRPO), to enhance reasoning capability, enforce structured output adherence, and improve generalization across domains and speaker variations. Extensive evaluations across diverse benchmark datasets demonstrate that VowelPrompt consistently outperforms state-of-the-art emotion recognition methods under zero-shot, fine-tuned, cross-domain, and cross-linguistic conditions, while enabling the generation of interpretable explanations that are jointly grounded in contextual semantics and fine-grained prosodic structure.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper described a system that augments vowel-level prosody descriptors as additional information to the LLMs. It shows that with the vowel-level speech information, the LLMs performs better on emotion recognition in both zero-shot and supervised fine-tuning setup.

### Strengths
The paper is well written with good clarity, and easy to understand. It could contribute to the SER field as a baseline of fine-tuning LLMs with natural language descriptors of speech. I am personally interested in what granularity of speech information is needed for SER. This paper seems to suggests that going toward vowel-level will provide some improvement.

### Weaknesses
On the other hand, the idea of appending prosody descriptions itself is not particularly novel, e.g., SpeechCueLLM, which is also cited as a baseline by the authors. The novelty of the work lies in expanding it to the vowel-level.

There is one flaw that undermines the validity of the experiments that the work needs to revise before publication.
In Table 4, it is unclear to me how the other models are trained and tested. I assume only your model is fine-tuned with the thinking trajectory but not the other baselines, as I don't find how you generate the thinking trajectories for the baselines. In this case, I don't know if the other models will still output thinking trajectory after fine-tuning. And the performance gain you get, may simply be a difference of inference time scaling. And it can be possible that if you make the baseline models think/reason, they can also improve their performance to be potentially on par with your model.

The authors should add experiments either comparing the proposed model v.s. the baselines in either (both thinking enabled, or both thinking disabled). I am willing to raise the score if this issue is resolved.

Additionally, I don't see adding GRPO helps in Table 4, the authors should explain this. 

Typo: in the abstract: neglect "or"

### Questions
See the above weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper presents an approach to introduce an explainable and interpretable approach to perform emotion recognition from speech using  LLMs. The work introduces pitch, energy, and duration based descriptors from time-aligned vowel segments and converts these into natural language descriptors to fine tune existing LLMs. The authors hypothesize that the proposed descriptors would enable an LLM to reason using both semantic information in lexical representations as well as prosodic variations in acoustic information. 

Evaluation on multiple datasets demonstrate that the proposed approach helped to improve performance compared to prior art.

The motivation of the paper is well outlined, clearly citing relevant prior art and outlining the key contributions made in this paper. 
The proposed approach introduces prosodic descriptors based on their proposed approach, that are generated from standard benchmark datasets, it is not clear whether the authors intend to share the information with the community as that would help in both replication of the reported results and foster future research directions.

### Strengths
The paper introduces pitch, energy, and duration based descriptors from time-aligned vowel segments to generate emotion-salient prompts to improve emotion recognition performance using LLMs.

The process of generating the descriptors is well described and results demonstrate the promise of the proposed work. Evaluation from multiple datasets demonstrate the generalization of the findings.

### Weaknesses
The proposed approach introduces prosodic descriptors based on their proposed approach, that are generated from standard benchmark datasets, it is not clear whether the authors intend to share the information with the community as that would help in both replication of the reported results and foster future research directions. It is not clear whether the setup used to generate the descriptors, or the descriptors obtained from the five datasets (used in the paper) will be publicly shared.

There are some open questions regarding the introduction of the descriptors that need to be addressed: coarticulation and lenition in spontaneous speech can alter vowel durations where vowels can be deleted or influenced by neighboring phonemes. It is not clear how the proposed approach would address such situations, or such instances were ignored in the current work? It will also be interesting to explore how speech rate impacts the efficacy of using the proposed descriptors?

### Questions
(1) By vowel level, does it mean at the individual distinct vowel level or by vowel-groups, such as front and back vowels?

(2) For the cross-domain evaluations, the datasets had different number of emotion categories, how were the difference in categories accounted during the evaluation?

(3) When reporting the emotion recognition performance across the different datasets, were all emotion categories specified in those data sets (Table 2) used to obtain the evaluation metric?

(4) The descriptors presented in this work rely on the forced alignment information, in spontaneous speech due to coarticulation vowel boundaries may be uncertain, any thoughts on how such conditions may impact the proposed descriptors?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper targets a key limitation in speech emotion recognition (SER) with large language models (LLMs): difficulty leveraging fine-grained prosodic cues and lack of interpretability. The authors propose VowelPrompt, a framework that annotates and conditions LLMs on multi-dimensional, phoneme-level prosodic attributes with a focus on vowels. Experiments across five datasets (including IEMOCAP, MELD, CaFE) show consistent gains over text-only baselines and sentence-level prosody augmentation methods (e.g., SpeechCueLLM) in zero-shot, few-shot, supervised fine-tuning, cross-domain , and multilingual settings, while providing interpretable emotion reasoning.

### Strengths
1. Fine-grained modeling unit: The vowel-centric, phoneme-level prosodic prompting is a clear step beyond sentence-level prosody features, improving both recognition performance and interpretability. The focus on vowels is linguistically motivated and empirically supported.
2. Two-stage training design: The two-stage framework is logically structured, and the RL-based targeted optimization appears to contribute to robust cross-domain generalization.
3. Breadth of evaluation: The empirical study is comprehensive, covering zero-shot, few-shot, supervised fine-tuning, cross-domain transfer (IEMOCAP↔MELD), and multilingual scenarios (EN/FR/DE/mixed). The baselines include strong text-only and prosody-augmented LLM variants (e.g., SpeechCueLLM), making the comparisons meaningful.

### Weaknesses
1. The paper emphasizes the primacy of vowels for prosody, while citing evidence that consonants can convey complementary emotional cues (e.g., Bitouk et al., 2010). However, the method entirely excludes consonant segments. This design choice risks discarding potentially informative signals (e.g., frication intensity, voicing onsets, burst characteristics) that may be emotion-sensitive in certain languages and speaking styles. A controlled analysis is needed to justify the exclusion.
2. The baseline suite lacks strong non-LLM SER systems, despite a long line of work using acoustic-prosodic features and modern architectures (e.g., CNN/TDNN/Conformer or HuBERT/w2v2 features with phoneme-aligned prosody). Without such baselines, it is difficult to disentangle the benefit of VowelPrompt from the general advantage conferred by LLM priors and instruction tuning.
3. The framework assumes fine-grained (phoneme/vowel-level) labels and intermediate “reasoning” steps align with the final emotion decision. In practice, manual annotations and model-generated rationales may not be fully consistent. This raises concern that the LLM could optimize for correct final labels while producing intermediate explanations that are partially contradictory or post hoc. The paper should explicitly evaluate the faithfulness and consistency of the reasoning traces.
4. Table 4 suggests GRPO leads to consistent performance drops (negative optimization) under certain settings, while Table 5 shows clear cross-domain improvements. The manuscript needs a unified explanation reconciling these outcomes. For instance, is GRPO trading in-domain accuracy for distributional robustness, regularizing prosody usage, or mitigating spurious correlations? What hyperparameters or curricula drive this trade-off?
5. At inference time, the system still relies on MFA and explicitly labeled phoneme attributes before passing prompts to the LLM. It is therefore unclear why an LLM is preferable to a purpose-built classifier over the same discrete features (e.g., a gradient-boosted tree, MLP, or Transformer). If the LLM mainly converts discrete features to natural-language prompts, the incremental contribution appears limited. The paper should quantify the benefit of the LLM beyond a feature-to-label classifier and clarify where language priors or compositional reasoning matter.
5. The framework depends on accurate phoneme segmentation and prosodic feature extraction. It is unclear how errors in forced alignment or noisy conditions propagate to final performance. Robustness analyses (e.g., perturbing boundaries, adding noise, cross-accent conditions) are limited.

### Questions
1. Vowel vs. consonant contributions: Can you provide ablations comparing vowel-only, consonant-only, and all-phoneme prompting across languages and conditions? Are there language-specific effects where consonant cues (e.g., voicing contrasts, place/manner-dependent energy) help more?
2. Non-LLM baselines: How does VowelPrompt compare against a strong acoustic SER pipeline using phoneme-aligned prosody with CNN/Conformer or self-supervised speech features (HuBERT/w2v2) plus a classifier? Please include both in-domain and cross-domain comparisons.
3. Reasoning faithfulness: Do you measure the consistency between intermediate prosodic rationales and final predictions? For example, human judgments of faithfulness, agreement with attributions  or causal tests (masking/removing cited cues).
4. GRPO dynamics:  Can you explain or give learning curves and sensitivity analyses demonstrating when GRPO helps harm in-domain performance but helps cross-domain transfer? Also, if I understand correctly, the GRPO data is also from the source domain, right?
5. Value of LLMs: If the same phoneme-level attributes are available, what performance and interpretability gains remain when replacing the LLM with a standard classifier? Conversely, can the LLM operate without explicit phoneme labels (i.e., text + raw audio features) and still retain its advantages?
6. How robust is VowelPrompt to alignment errors and background noise? What is the performance degradation under realistic ASR/aligner noise or accented speech?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a novel Speech Emotion Recognition approach that augments speech transcripts with vowel-level prosodic descriptors (F0 level/slope/variation, intensity level/variation, duration) extracted via forced alignment. These descriptors are then used in an LLM-as-classifier framework, fine-tuned via supervised fine-tuning (SFT) and reinforcement learning (RLVR) from GPT-4o (LLM as oracle) for reasoning traces.

To account for cross-speaker and vowel variability, the authors first perform speaker- and vowel-type normalization, then quantile-bin the prosodic statistics and convert them into natural-language tokens, which are appended to the transcript so the LLM can reason over lexical and localized prosody. The method is evaluated on IEMOCAP, MELD, CaFE, EmoDB, and ASVP-ESD (mixed-lingual), showing consistent gains over transcript-only and sentence-level prosody prompts, with interpretable rationales.

### Strengths
The proposed method has several strengths, which can be grouped into two categories:

**Conceptual**: the method provides a privacy-oriented, interpretable formulation of SER by augmenting transcripts with symbolic, vowel-level prosody tokens (F0 level, intensity, duration), allowing emotion inference without raw audio at inference. This design is linguistically grounded (vowels are typically stable prosodic carriers), separates lexical content from paralinguistic cues, and supports closed-set classification, addressing both explainability and data-minimization goals.

**Implementation strengths**: the end-to-end pipeline is clear and modular: forced alignment, followed by speaker and vowel normalization, then quantile binning, conversion to natural-language tokens, and an LLM-as-classifier stage. Each step is inspectable and easily ablated. Training uses a practical two-stage recipe (supervised fine-tuning followed by RLVR/GRPO with verifiable rewards) and performs consistently across datasets and model families, improving over transcript-only and sentence-level prosody prompts. At deployment, the approach seems easy to integrate with existing text infrastructures.

### Weaknesses
Because the oracle used for RLVR traces (GPT-4o) is itself an LLM, there is a material risk that fine-tuned student LLMs learn spurious lexical or formatting heuristics unrelated to the intended prosodic mechanism. To establish that the model uses vowel-level prosodic tokens rather than incidental cues, please include controlled counterfactual ablations that preserve input statistics while breaking the hypothesized channel. Some examples to test:
- Transcript shuffle control: randomly permute word order, or replace content words with frequency-matched synonyms, while keeping the vowel-prosody tokens intact. Performance should remain near the original if prosody carries the signal and should drop sharply if lexical priors dominate.
- Prosody permutation control: permute the vowel-prosody descriptors across utterances within a mini-batch while keeping transcripts fixed. Performance should drop to near chance if the model relies on the prosodic channel.
- Matched-marginal placebo: replace prosody tokens with random draws from their empirical per-vowel distribution so that marginals are preserved but alignment is broken. This controls for token frequency or style leakage.
- Cross-swap (counterfactual consistency): for the same transcript, attach prosody from an utterance of a different emotion. Predicted labels should flip in the direction implied by the swapped prosody if the mechanism is causal.

Also, the paper does not analyze how the discrete labels are tokenized by each model’s tokenizer. If some labels map to a single token while others split into multi-token sequences, the model may exhibit unequal calibration and decoding bias unrelated to prosodic evidence. An ablation study can include:

- Tokenization audit per model: for each model used (for example, LLaMA-3-8B, Qwen-2-7B), report (i) tokenization of each label verbalizer (angry, sad, neutral, happy, excited), including the number of tokens and subword pieces, and (ii) empirical prior token likelihoods for these verbalizers under pseudo-neutral prompts, for example, a transcript without prosody information.
- Label set distribution: compute normalized label probabilities via the product of subtoken log-probabilities for multi-token labels and report calibration. This assesses whether the decoding process favors specific verbalizers independent of evidence.
- Label name permutation: assign the emotion categories to some random verbalizer strings to test sensitivity to label names. If the model captures the underlying categories rather than specific strings, performance should remain stable.

Answering some of these questions would strengthen the conceptual foundation of the paper and the interpretability of the suggested method.

### Questions
Some follow-up questions:

- Are `Speaker_{%d}` tokens used only as placeholders to distinguish multiple speakers within an utterance, or do they carry any stable identity across utterances and speakers?

- Related work organization: The “Vowel-Centric Prosody in Emotional Speech” paragraph and the first paragraph of Section 3.1 can be merged. This would reduce redundancy and improve flow by introducing the vowel-centric rationale once, then immediately presenting the operational relevance.

- Cross-linguistic normalization and scope: the manuscript states: ``We ensure cross-linguistic consistency and compatibility with multilingual phonetic analysis pipelines …`` and later ``To control for cross-lingual variation in prosodic realization, we further perform normalization at the language level. For each language, we compute global means and standard deviations for each prosodic feature and apply z-score normalization within that language``...  Please specify the language set to which these claims apply. Cross-linguistic prosody-emotion mappings differ across languages, especially in tone languages such as Vietnamese and Chinese where pitch contours are lexical. It would be helpful to limit claims to the languages actually evaluated (for example, English, French, German).

- Determinism of “parameter-free” processing and oracle traces: The manuscript states that the process is ``parameter-free, ensuring transparency and reproducibility,`` and that ``gold reasoning traces [are] automatically generated by a high-capacity text-only LLM such as GPT-4o.``... Are these oracle traces generated deterministically. Please report the decoding settings used for GPT-4o, including temperature, top-p, and any seed control. If temperature was set to zero, state this clearly. If not, either provide information about variance across repeated generations and whether multiple samples were filtered or selected, or elaborate on the "reproducibility" claim.

- RLVR objective and the role of reasoning traces: on the RLVR stage, how are the reasoning traces used. Are rationales part of the reward signal beyond format verification? Please clarify whether any rationale consistency checks are part of the verifiable objectives.

### Soundness
3

### Presentation
3

### Contribution
2
