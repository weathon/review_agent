# Assembling the Mind's Mosaic: Towards EEG Semantic Intent Decoding

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 6

## Abstract
Enabling natural communication through brain–computer interfaces (BCIs) remains one of the most profound challenges in neuroscience and neurotechnology. While existing frameworks offer partial solutions, they are constrained by oversimplified semantic representations and a lack of interpretability. To overcome these limitations, we introduce **Semantic Intent Decoding(SID)**, a novel framework that translates neural activity into natural language by modeling meaning as a flexible set of compositional semantic units.
SID is built on three core principles: semantic compositionality, continuity and expandability of semantic space, and fidelity in reconstruction.
We present **BrainMosaic**, a deep learning architecture implementing SID. BrainMosaic decodes multiple semantic units from EEG/SEEG signals using set matching and then reconstructs coherent sentences through semantic-guided reconstruction. 
This approach moves beyond traditional pipelines that rely on fixed-class classification or unconstrained generation, enabling a more interpretable and expressive communication paradigm. Extensive experiments on multilingual EEG and clinical SEEG datasets demonstrate that SID and BrainMosaic offer substantial advantages over existing frameworks, paving the way for natural and effective BCI-mediated communication.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes Semantic Intent Decoding (SID) and a concrete model, BRAINMOSAIC, for translating EEG or SEEG into natural language by first decoding a variable set of semantic units, aligning them in a continuous text embedding space, and then reconstructing a sentence with an LLM constrained by those units. The pipeline comprises a set-matching Semantic Decomposer trained with a Hungarian loss, a Semantic Retriever that aligns slot embeddings to word and sentence embeddings and predicts global attributes, and a Semantic Decoder that prompts an LLM to generate text from retrieved units. The work targets semantic decoding during reading, listening, or imagined speech across Chinese and English datasets, plus a private SEEG dataset, and claims improvements over classification and unconstrained generation using embedding based metrics UMA, MUS, and SRS

### Strengths
1. The three principles (compositionality, continuity, fidelity) are well-motivated by linguistic and neuroscience evidence.
2. Interpretable pipeline: slots -> ranked retrieval -> prompted gen beats blackbox E2E.
3. Consistent empirical gains across multiple datasets and baselines with both concept level and sentence level metrics.
4. Comprehensive comparison with other relevant baselines.
5. Extensive supplementary material includes dataset details, baseline descriptions, and sensitivity analyses.

### Weaknesses
1. Lack of qualitative examples of reconstruction quality. Without concrete examples, readers cannot verify whether predicted semantic unit sets are genuinely interpretable or noisy/scattered, (b) how well do the quantitative metrics match real semantic correctness. This falls below the standards in neuro-decoding literature where it's common to show a few samples of the proposed model's input-output vs baseline
2. Lack of comparability to standard text metrics. While semantics-first metrics are appropriate here, it's also important to have surface level metrics common in the literature such as BLEU score, WER for surface overlap, and BERTScore for semantic similarity.

### Questions
1. For continuous corpora, how did you prevent leakage between train and test when the same long passage is segmented into sentences?
2. Can you report error bars and conduct significance tests comparing BrainMosaic to baselines?
3. How sensitive are results to the choice of embedding model? What happens with random embeddings as an ablation?
4. For multi-subject datasets, what is cross subject variance? Can models trained on one subject decode another subject’s neural activity? How good is the generalizability
5. Did you perform any electrode selection or analyze which channels/regions contribute most to semantic unit prediction?
6. Table 5A shows declining UMA with vocabulary expansion. At what vocabulary size does performance approach random? Can you characterize this scaling law?

### Soundness
3

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
4

### Summary
The paper introduces Semantic Intent Decoding (SID), a novel framework for brain-to-language translation that models communicative intent as a set of compositional semantic units rather than relying on fixed labels or unconstrained generation.

### Strengths
The paper proposes an intriguing motivation and introduces a novel perspective for brain-to-text decoding. By modeling intent as an unordered and variable set of semantic units, it moves beyond traditional fixed-label or sequential decoding paradigms, potentially offering a more brain-plausible representation of semantic processing.

### Weaknesses
1.Lack of robustness evaluation under input noise: the paper does not assess model performance under noisy or corrupted EEG inputs, which weakens the significance of the results reported in Tables 3, 4, and 5. Given the inherent noisiness of EEG signals, such evaluations are essential to validate the practical utility of the proposed method.

2.Insufficient justification for LLM-based sentence generation: the use of LLMs for sentence reconstruction raises concerns about data contamination, especially if the test set sentences or similar phrasings were present in the LLM’s pretraining corpus. The authors provide no strong evidence (e.g., n-gram overlap analysis or controlled LLM ablation) to rule out this possibility, which undermines the credibility of the generation results.

3.Missing comparison with standard generation metrics and SOTA methods: the paper does not compare with mainstream brain-to-text decoding methods using standard generation metrics such as WER, CER, BLEU, ROUGE, METEOR, or BERTScore. This omission makes it difficult to benchmark the proposed method against existing literature and assess its true advancement in the field.

### Questions
1.How was K (number of semantic slots) chosen per dataset? Was it tuned? How does performance vary with K?

2.Would the model scale to larger vocabularies? What are the computational bottlenecks?

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
The paper proposes to view EEG/SEEG-to-language decoding as predicting a variable-size, order-invariant set of semantic units (SID), rather than fixed-class decoding or fully free-form generation. It instantiates this with BRAINMOSAIC, including i) an EEG encoder + query slots, ii) a semantic retriever that aligns slots to an open-vocabulary unit bank, and iii) an LLM that does semantics-constrained generation from those units. Experiments on three public EEG datasets (Chinese + English) and one private SEEG dataset, using UMA, MUS, and sentence-level similarity, aim to show: (i) set-style decoding is reasonable, (ii) continuous semantic space helps scalability, (iii) constrained generation improves fidelity.

### Strengths
1. Interesting formulation: treating an utterance as a set of semantic units is a neat and fairly novel way to handle variable-length EEG semantics.

2. Well-structured method: the three design principles (compositionality, continuity/expandability, fidelity) map cleanly to three modules.

3. Reasonable experiment designs: the same idea is run on Chinese imagined speech, Chinese naturalistic reading, English reading, and a clinical SEEG case, which supports the claim that the approach is not tied to a single dataset.

### Weaknesses
1. **Train/validation split is underspecified:** the paper only mentions a unified 8:2 train–test split, but does not say whether this is by subject or by trial. Can the model guess which sentence it is by the sample length?

2. **No true random / text-prior baselines**: the main metrics (UMA, MUS, sentence similarity) are not compared against (i) picking the same number of units at random or (ii) a text-only/corpus-frequency prior. This makes it hard to see how much of the score actually comes from EEG, especially since the method later calls an LLM. The paper reports `MUS_exp` but doesn’t clearly define it.

3. Experiments don’t cleanly isolate the three research questions: the section is organized around “set is better,” “continuous space scales,” and “constrained generation is more faithful”. Several experiments change multiple components at once (set + LLM + thresholds), so it’s hard to attribute the improvements to the claimed factor.

### Questions
1.Is the 8:2 split done by subject or by trial on each dataset? Is there possibility of data leakage?
2.Please add complete random baselins for UMA/MUS/SRS. Please describe the MUS_exp more clearly.
3.Please add ablations where only one of the three design choices (set, continuous space, LLM-constrained decoding) is changed at a time, so we can see which part actually drives the gains?
4.Please distinguish clearly between ChineseEEG and ChineseEEG-2 as they are two distinct datasets.
Some references are missing. For example, there should be references for this statement: “Alternatively, a more recent direction seeks to enhance expressive capacity by mapping neural signals directly into the latent representation space of large language models (LLMs).”

### Soundness
3

### Presentation
4

### Contribution
3
