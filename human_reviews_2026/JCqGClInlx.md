# Redefining Machine Simultaneous Interpretation: From Incremental Translation to Human-like Strategies

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 2

## Abstract
Simultaneous Machine Translation (SiMT) requires high-quality translations under strict real-time constraints, which traditional encoder-decoder policies with only READ/WRITE actions cannot fully address. We extend the action space of SiMT with four adaptive actions: SENTENCE_CUT, DROP, PARTIAL_SUMMARIZATION and PRONOMINALIZATION, which enable real-time restructuring, omission, and simplification while preserving semantic fidelity. We implement these actions in a decoder-only large language model (LLM) framework and construct training references through action-aware prompting. To evaluate both quality and latency, we further develop a latency-aware TTS pipeline that maps textual outputs to speech with realistic timing. Experiments on the ACL60/60 English-Chinese and English-German benchmarks show that our framework consistently improves semantic metrics (e.g., COMET-da and COMET-KIWI) and achieves lower delay (measured by Average Lagging) compared to reference translations and salami-based baselines. Notably, combining DROP and SENTENCE_CUT yields the best overall balance between fluency and latency. These results demonstrate that enriching the action space of LLM-based SiMT provides a promising direction for bridging the gap between human and machine interpretation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces new adaptive actions to address strict time constraints in simultaneous machine translation: sentence_cut, drop, partial_summarization and pronominalzation. The actions enable restructuring of input and improve latency through summarization and dropping redundant information. These actions are similar to human interpreter strategies and are designed to modulate translation quality and latency. The approach is compared against salami-based segmentation, and are explored along inference-based approach (TransLLaMA) and prompting strategies such as dynamic in-context learning and few-shot prompting. A text-to-speech pipeline is developed to evaluate latency of the approach and baselines. Overall, interpretations based on salami approach have higher surface overlap, scoring higher in BLEU, chrF, and TER. Action-adapted references score better on semantic metrics (COMET). Further ablations reveal that the combination of sentece_cut and drop significantly reduce latency, and achieve the best tradeoff with improvement on semantic metrics.

### Strengths
1. The adaptive actions are carefully chosen to address specific challenges in simultaneous interpretation. 
2. The evaluation pipeline is thorough and comprehensive.
3. The approach is tested with a range of inference and training methods.
4. Dynamic use of the actions in Table 3 shows significant improvement in quality and latency.
5. Good and clear analyses on the ablation studies reported in Table 2.

### Weaknesses
1. The writing occasionally lacks clarity. A few suggestions:
- It'd be great if there's an example demonstrating the difference between salami technique and sentence_cut
- The main results in Table 3 should emphasize flexibility/dynamicity of the action choices, and could be featured before Table 2.
- What model adaptation method is being used for the results in Table 2 and 3?
- Please clarify if multiple actions can be applied to the same input.
2. The method seems to work less well on En-De. 
3. There seems like a missed opportunity without exploration on the distribution of action being chosen by the model.

### Questions
1. All of action choices could ideally reduce latency with shortened output. When applied individually in Table 2, however, why is latency increased in most cases? Can you address this in the paper?
2. Can there be multiple decision points in the same input? How may this be applied in practice?
3. Are there premature decisions that make interpretation worse? Can you provide an error analysis on these examples?

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
This paper proposes expanding the set of adaptive actions available to simultaneous translation models by introducing SENTENCE_CUT, PARTIAL_SUMMARIZATION, DROP, and PRONOMINALIZATION. These actions allow the model to restructure sentences, omit less critical content, and simplify translations. By applying these adaptive actions, the models demonstrate increases in translation quality with a reduction in non-computationally aware latency.

### Strengths
•	The results section tests the approach across a wide variety of translation quality metrics and generally shows an improvement by applying the additional adaptive actions.

•	A solid ablation that identifies the influence of each of their proposed adaptive actions on downstream simultaneous translation quality and latency.

### Weaknesses
•	The approach feels more like a data preprocessing step that simplifies downstream translation rather than a fundamentally novel contribution. This raises concerns about fairness when comparing against baselines. 

•	It is unclear whether modifications introduced by SENTENCE_CUT, PARTIAL_SUMMARIZATION, DROP, and PRONOMINALIZATION are always desirable. How can we verify that these changes maintain equivalence or acceptability compared to the original content? 

•	The paper does not sufficiently justify the use of LLMs for simultaneous translation. Given their large parameter counts, wouldn’t they introduce latency that undermines real-time feasibility?

•	The full capitalization of adaptive actions is visually distracting.

•	Examples in Figure 2 are difficult to interpret without knowledge of the source language. Could the authors provide intermediate representations or choose an Indo-European language for better accessibility to English-speaking readers?

•	The work appears focused on text-to-text translation, yet includes components for speech-to-speech (e.g., TTS pipeline). The paper should either reframe as text-to-text or justify the need for speech-to-speech. If speech-to-speech is retained, a quality metric for speech output is necessary.

•	No computationally aware latency metric is provided to account for the additional time introduced by LLMs.

•	The paper does not compare against baseline encoder-decoder models.

•	Results are limited to a small number of datasets and language pairs.

•	Latency is measured only using Average Lagging (AL). Why not include LAAL, which accounts for overgeneration (a common issue with LLMs)? The justification provided (“it just serves as a reference of latency”) is insufficient.

•	Statistical rigor is lacking. Confidence intervals or averages over multiple runs would help demonstrate that improvements are not due to randomness.

### Questions
•	In Section 3.2, you mention four methods, but only three are described. Am I missing one?

•	Which language pair is used for the results in Table 1?

•	What do the ACL60/60 reference results in Table 2 represent?

•	Given the use of LLMs, is there a risk of information leakage that compromises the validity of simultaneous translation? Can you provide assurances that this is not affecting results?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper addresses the problem of trajectory synthesis for simultaneous machine translation (SiMT) . It extends the conventional *read/write* action space with four additional semantic actions:

(1) segmenting sentences into shorter semantic units,

(2) dropping non-informative words,

(3) summarizing partial prefixes, and

(4) replacing repetitive nouns with pronouns.

These enriched actions are used to construct training trajectories for adapting baseline models, including TransLlama and in-context learning (ICL) methods, which are then evaluated on the ACL 60/60 benchmark. The paper further leverages development-set statistics to directly prompt large language models to perform these enriched actions during SiMT inference.

### Strengths
The paper enriches the action space of SiMT to better emulate the behavior of human interpreters, introducing actions such as summarization, omission, and pronoun substitution. This is an intuitive and meaningful idea that bridges the gap between human interpretation strategies and machine translation policies, offering a more cognitively inspired approach to trajectory synthesis.

### Weaknesses
1. **Poor Writing and Presentation Quality:**
    
    - The font appears wrong.
    - In Figure 2, the actions are applied to the *reference translations*, yet Section 3.1 describes them as being applied to the *source text*, creating a contradiction.
    - The prompt design for ICL methods is not provided. Also, if *salami* references or other references are used as examples, this should be explicitly stated.
    - The meaning of “(a) reference translations” in line 254 is unclear; it seems to refer to using *TransLlama* style data synthesis with reference translations, but this should be clarified.
    - It is stated that model adaptation uses the ACL 60/60 dev set, but Tables 1 and 2 also report results on the dev set—does this imply training and evaluation on the same data, or is it a typo?
    - Line 264 “CozyVoice” should be “CosyVoice.”
    - In Section 3.4, it is implied that the proposed method is also used for direct SiMT inference, but this is never clearly described—this needs explicit explanation.
    - Tables 2 and 3 are oversized.
    - In Table 2 En-De, *TER* is a *lower-is-better* metric, yet the highest TER values are incorrectly highlighted.
    - The reported AL of the *reference translation* in Table 2 is conceptually unclear—how can the reference have latency below 1 s? Shouldn’t it be offline translation?
2. **Inappropriate Latency Metric:**
    
    The paper continues to use AL as the latency metric, despite known limitations. The community has largely moved toward Length-Adaptive Average Lagging (LAAL), which provides a more robust and fair comparison, as shown in recent IWSLT evaluations.
    
3. **Limited Effectiveness Across Language Pairs:**
    
    The proposed method shows improvement on **En–Zh**, but performs worse than *Salami* on **En–De** (Table 2). This suggests that the approach may not generalize well across languages. Additional experiments on more language directions are necessary to validate its overall effectiveness.

### Questions
Check weakness

### Soundness
2

### Presentation
1

### Contribution
2
