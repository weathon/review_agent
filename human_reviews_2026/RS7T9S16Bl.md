# Music Flamingo: Scaling Music Understanding in Audio Language Models

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
We introduce Music Flamingo, a novel large audio–language model, designed to advance music (including song) understanding in foundational audio models. While audio–language research has progressed rapidly, music remains challenging due to its dynamic, layered, and information-dense nature. Progress has been further limited by the difficulty of scaling open audio understanding models, primarily because of the scarcity of high-quality music data and annotations. As a result, prior models are restricted to producing short, high-level captions, answering only surface-level questions, and showing limited generalization across diverse musical cultures. To address these challenges, we curate MF-Skills, a large-scale dataset labeled through a multi-stage pipeline that yields rich captions and question–answer pairs covering harmony, structure, timbre, lyrics, and cultural context. We fine-tune an enhanced Audio Flamingo 3 backbone on MF-Skills and further strengthen multiple skills relevant to music understanding. To improve the model's reasoning abilities, we introduce a post-training recipe: we first cold-start with MF-Think, a novel chain-of-thought dataset grounded in music theory, followed by GRPO-based reinforcement learning with custom rewards. Music Flamingo achieves state-of-the-art results across 10+ benchmarks for music understanding and reasoning, establishing itself as a generalist and musically intelligent audio–language model. Beyond strong empirical results, Music Flamingo sets a new standard for advanced music understanding by demonstrating how models can move from surface-level recognition towards layered, human-like perception of songs. We believe this work provides both a benchmark and a foundation for the community to build the next generation of models that engage with music as meaningfully as humans do. Demo: https://musicflamingo.github.io

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Music Flamingo, a large audio–language model, designed to advance music understanding in foundational audio models. Music Flamingo is a fine-tuned model based on Audio Flamingo 3 backbone. This paper also introduces MF-Skills and MF-Think, two large-scale datasets containing music–caption and music–QA pairs designed to promote deliberate reasoning.

### Strengths
-	Music Flamingo achieves state-of-the-art results on 12 music understanding and reasoning benchmarks.
-	Low-level attributes are extracted using the MIR tool, and initial captions combined with these attributes are used to reduce hallucination, creating captions and QA data that include multifaceted information to construct MF-Skills data. Synthetic data is built with attention to hallucination by LLM, considering important characteristics in the music domain, and an effective approach has been proposed.

### Weaknesses
-	The ablation study is missing. Although improvements are made based on AudioFramingo 3, the model is trained in three stages: w/MF-skill, w/MF-think, and GRPO with custom rewards. Having evaluation results for each stage would help confirm each contribution.
-	Some details about the dataset creation are not described.

### Questions
- There seems no explanation of how and from where the 2M songs for MT-skills were collected. Additionally, is there a possibility that the data used for evaluation is included in the training data?
- I could not find an explanation for why the number of examples increases from 2M songs to 3M examples in MT-Skills. Does this mean that captions are added to all 2M songs, and some songs also have QA, thus resulting in 3M examples including pairs of music-caption and/or music-QA?
- Does AF3-St. 3 in Table 2 refer to the training data of the base model AudioFramingo3? Additionally, does 2.0 of MF-SFT mean adding MF-Skills twice? Are the other data also used in the training of AudioFramingo3 and then reused in fine-tuning for Music Framingo?
- It would be beneficial to elaborate on how quality filtering is performed.
- How were the categories in Figure 27 analyzed?
- Will the created dataset and source code/checkpoints be made publicly available?
- How were the inputs and outputs for ACC evaluation handled?
- In Figure 3, the term "MF-skills CoT" is used, but the caption seems to explain it as "MF-Think." It would be better to standardize the terminology.
- It seems Figures 4 and 27 are identical, but only Figure 27 is mentioned in the text. It would be better to refer to Figure 4 and note in its caption that an enlarged version is included as Figure 27 in the Appendix.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Music Flamingo, a large multimodal model extending the Flamingo architecture to handle music, text, and symbolic representations jointly. The model is trained on large-scale music–text–symbol triplets and evaluated across multiple tasks, including music captioning, retrieval, tagging, and question answering. The paper aims to demonstrate the benefits of scaling and unified modeling for general-purpose “music understanding.”

Overall, this is a solid and technically competent piece of work, with careful engineering, broad evaluation, and clear connections to the audio Flamingo. However, several conceptual and empirical gaps remain regarding the necessity of this architecture, its comparison to specialized systems, and its coverage of affective aspects of music understanding.

### Strengths
Comprehensive system design — The adaptation of the Flamingo multimodal framework to the music domain is well-motivated and technically executed. Training across multiple modalities (audio, symbolic, text) is a significant engineering contribution.

Unified modeling framework — Unlike prior task-specific MIR models, the same model handles captioning, retrieval, tagging, and QA, demonstrating promising few-shot and zero-shot generalization.

Dataset and reproducibility — The paper provides a substantial and well-curated multimodal dataset and detailed training setup, which is a strength for community reuse.

### Weaknesses
1. Necessity over specialized MIR pipelines

While the unified approach is appealing, it remains unclear why such a large, general model is necessary for tasks that existing MIR pipelines already solve effectively (e.g., onset detection, chord recognition, instrument tagging).
The paper could better justify what emergent capabilities arise from this unified model that cannot be achieved by composing established MIR tools.


2. Missing comparisons to domain-specialized music–language models

The paper positions Music Flamingo as a general music–language model but provides limited comparison to existing specialized systems -- I see it cited MuLLaMa but it seems there is no comparison?

3. Conceptual scope and framing

"Music understanding" has cognitive or perceptual components implied by that phrase, not only including facts like pitch, timbre, but also a focus on the feeling and emotional response. This is definitely a hard task and the seems not the focus here. That said, framing the contribution as scaling multimodal representations for music would be more accurate and modest, aligning better with the presented evidence.

### Questions
Would you include case studies or analyses showing cross-task generalization or contextual reasoning that would be infeasible for traditional MIR pipelines.

### Soundness
3

### Presentation
4

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
This paper introduces Music Flamingo (MF), an LLM-based model for music understading tasks. Authors created two large-scale datasets for music understanding tasks: MF-Skills, which contains 4M examples and MF-Think, which contains 300K CoT examples (derived from MF-Skills) for post-training MF to improve its reasoning abilities.  MF-Skills and MF-Think contain full-length music with detailed and multifaceted annotations, compared with previous music understanding datasets that often contain short and surface-level annotations.

Based on the two datasets, authors developed MF by continue training the Audio Flamingo 3 (AF3) model. MF achieved SOTA performances on several public music understanding benchmarks. Human evaluation with music experts show that MF seems better compared with other models.

### Strengths
- I think MF-Skills and MF-Think are very good resources for music understanding, where annotations and datasets are rare. The two  datasets could potentially benefit follow-up work of developing better models in this field.

- SOTA performance on several music understanding benchmarks.

- Adding a post-training stage to improve reasoning of the music understanding model is well motivated. Current music understanding   models tend to output surface-level outputs about a music. It is great to see adding post-training RL is also effective for music  understanding.  Authors also show that without post-training, performance drops on MMAU-Pro-Music and MuchoMusic.

### Weaknesses
- MF stresses a lot on music with vocal (sec 3.1), and motivates to add several ASR datasets. While this motivation is valid, but non-vocal   music seem less covered in the discussions.

- In the human evaluation from music experts (Tab4), it seems MF often attempts to provide deep details, but often inaccurate. This poses   some concerns on real-world use cases, beyond the good performance on benchmark numbers.

### Questions
- Appendix F: how generalizable is this user study? Have you tried with more music/songs, or music without vocals?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work proposes Music Flamingo, a large-scale audio-language model established from a trained Audio Flamingo 3 model, plus 4 post-training steps (3 supervised fine-tuning, 1 RL policy update phase). At the same time, this work proposed MF-Skill and MF-Think, two large-scale synthesized datasets curated for the post-training steps of Music Flamingo. The proposed model achieved top performance on 13 benchmarks (Table 1).

### Strengths
- Datasets and model proposed in this work are established on full-length songs rather than clips. This is important as some information can only be well determined by observing the full track (e.g. musical structure, mood, cultural context, etc.).

- Making useful incorporation of speech-centric datasets, such as including multilingual ASR and multi-talker ASR to improve the model capability on bridging texts(e.g. lyrics) and phonemes (speech, vocal). 

- The annotations included in MF-Skills incorporated multiple important aspects with not only tags, but also captions and QA pairs. In particular, temporal information is comparatively scarce in public datasets. 

- Incorporating a reinforced learning step in the entire pipeline. It is insightful to see a successful adaption of GRPO for LALM.

- Dataset and source code will be published for reproducibility.

### Weaknesses
- There's no ablation study on how each post training step improves the model, which is important to show the significance of introducing reinforcement learning. It's also a pity that there's no further discussion about the stability issue of GRPO.

- Although it's a positive sign that proposed work is aware of the dataset issue on cultural bias, according to Figure 4(the same as Figure 27?), the composition of MF-Skills is still western-centric. It can be understood that collecting dataset that covering everything is difficult, however, I believe it is still not sufficient to claim the current composition as "moving from western culture to diverse culture". For example, African, Middle east, India, East asia and Southeast asia music are still under-represented despite they all have huge audience population.

- The reference of GRPO seems to be missing: https://arxiv.org/pdf/2402.03300

### Questions
- Regarding quality filtering, what is the frontier model used to verify the captions? Also, what's the model used for re-writing the captions?

- How does the re-writing of MF-Think work? Is that done by repetitive re-generation followed by the verification from the said frontier model?

### Soundness
4

### Presentation
3

### Contribution
3
