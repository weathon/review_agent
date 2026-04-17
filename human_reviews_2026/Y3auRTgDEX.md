# CS-Dialogue: A 104-Hour Dataset of Spontaneous Mandarin-English Code-Switching Dialogues for Speech Recognition

- Decision: Reject
- Scores: 4, 2, 8, 8

## Abstract
Code-switching (CS), the alternation between two or more languages within a single conversation, presents significant challenges for automatic speech recognition (ASR) systems. Existing Mandarin-English code-switching datasets often suffer from limitations in size, spontaneity, and the lack of full-length dialogue recordings with transcriptions, hindering the development of robust ASR models for real-world conversational scenarios.  This paper introduces CS-Dialogue, a novel large-scale Mandarin-English code-switching speech dataset comprising 104 hours of spontaneous conversations from 200 speakers.  Unlike previous datasets, CS-Dialogue provides full-length dialogue recordings with complete transcriptions, capturing naturalistic code-switching patterns in continuous speech.  We describe the data collection and annotation processes, present detailed statistics of the dataset, and establish benchmark ASR performance using state-of-the-art models. Our experiments, using Transformer, Conformer, and Branchformer, demonstrate the challenges of code-switching ASR, and show that existing pre-trained models such as Whisper still have the space to improve. The CS-Dialogue dataset will be made freely available for all academic purposes.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a large, high-quality dataset of real conversational Mandarin-English code-switching, addressing the lack of authentic resources in this area. It provides detailed speaker and topic diversity, and demonstrates that training ASR models with this data significantly improves code-switching recognition.

### Strengths
1. This work presents a large-scale, spontaneous Mandarin-English code-switching speech dataset, filling a notable gap in the field. The paper invested substantial effort in collecting and processing real conversational data, which is of great value to the research community. The dataset addresses the scarcity of authentic code-switching conversational data and will facilitate further advancements in ZH-EN CS ASR.
2. The paper provides detailed analyses of the dataset’s composition, including speaker demographics (gender, age, region), topic distribution, and utterance statistics. Such thorough documentation enhances the dataset’s utility and transparency for future research.
3. Experiments on SenseVoice-small and other state-of-the-art models show that fine-tuning with this dataset can effectively improve code-switching ASR performance. The results validate the practical value of the resource.

### Weaknesses
1. **Insufficient Detail on Code-Switching Text Construction** The one of most critical aspect of a code-switching dataset is how the code-switched utterances are generated. The paper does not provide sufficient detail on whether the code-switching segments were produced spontaneously by speakers or guided by prompts. If speakers were given too much freedom, there is a risk that some utterances may be unnatural or overly contrived. For example, in Table A.2, some “sports” examples contain long English segments that are rarely observed in natural code-switching. The authors should clarify the construction process and discuss its impact on data authenticity.
2. **Limited Duration of Mixed (Code-Switching) Data** According to Figure 3, the mixed-language (code-switching) portion comprises only about 30 hours, less than one-third of the total dataset.
3. **Lack of Comparison with Synthetic or Segmented Data Approaches** Previous studies have shown that training with high-quality TTS-synthesized code-switching data or segmented real utterances can substantially improve code-switching ASR performance. However, this paper does not include experiments comparing the effectiveness of real conversational data versus synthetic or segmented data. Such comparisons would help highlight the unique advantages of the presented dataset.

### Questions
This work represents a major effort and a valuable resource for the community. I sincerely appreciate the authors’ dedication and the significant investment of time and funding. However, I feel that it falls short of the standards expected for a technical paper at a research conference. Beyond the resource construction and baseline experiments, the paper lacks deeper research insights and thoughtful analysis that would help the community better understand the challenges and opportunities in code-switching speech recognition. 

1. While the paper demonstrates the effectiveness of fine-tuning on SenseVoice-small, the test and training sets are from the same source. It would be valuable to know whether fine-tuning on CS-Dialogue leads to performance degradation on other open ASR benchmarks, indicating potential overfitting or lack of generalization.

2. Given the proven benefits of high-quality TTS-synthesized code-switching data, could the authors add experiments comparing models trained on real versus synthetic data to better illustrate the value of authentic conversational recordings?

3. The dataset contains a substantial amount of pure Mandarin and pure English data, which are already well-represented in existing corpora. What is the motivation for including these segments, and could their presence bias the test set, potentially inflating the observed improvements from fine-tuning?

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
5

### Summary
This study introduces the creation of a speech dataset for code-switching between English and Mandarin Chinese. It also presents results using different ASR frameworks to demonstrate the quality of the dataset. Although ASR corpora for code-switching between English and Chinese have been explored for at least the past decade, this dataset outperforms its competitors in terms of size (100 hours, compared to 10-20 hours for previous datasets and up to 63 hours for SEAME). While the low ASR performance warrants further discussion, this study represents significant effort and provides a valuable resource for the community.

### Strengths
1. The CS-Dialogue dataset will be made freely available for all academic purposes.

3. well-written

### Weaknesses
## Major Weaknesses:

1. No multi-seed runs reported; variance/standard deviation or confidence intervals are missing, so result stability is unclear.

2. Handling of English accents in transcription is unspecified; guidelines focus on Mandarin, leaving English variation policies unclear.

3. Possible regional/accent bias is not analyzed; no stratified evaluation by region/accent or related error analysis.

4. Inter-participant relationship is not reported (acquainted vs. strangers), which can affect dialogue style and code-switching behavior.

5. No cross-corpus generalization or zero-shot tests (e.g., TALCS, SEAME, DOTA-ME-CS); external validity remains unproven.


## Minor Weaknesses:

1. Authors state the "monolingual" blocks were not strictly enforced and CS was allowed, but there is no quantitative validation (e.g., comparison of switch-rate/CS types to naturally occurring corpora).

2. Recording setup and modality: Conducted via an online platform and audio-only retained, but it's unclear whether any sessions were co-located, how devices/network artifacts were normalized, or whether mixed setups occurred.

3. Generalization to unequal-status language pairs: Mentioned conceptually in related work, but no empirical evidence or analysis.

4. Breadth of CS dataset review: Coverage beyond Mandarin–English has improved but remains brief and not fully systematic.

5. Value of full-dialogue context: An intra-corpus ablation (previous turns 0 -> 3) shows gains, but there is no validation against non-dialogue corpora or a "utterance-only" baseline constructed for a like-for-like comparison.

### Questions
1. Licensing remains vague (“free for academic use”); no explicit open license or clarification on redistribution/commercial use.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces CS-Dialogue, a large-scale Mandarin-English code-switching dataset consisting of 104 hours of spontaneous conversations from 200 bilingual speakers. CS-Dialogue includes full-length, naturalistic dialogue recordings with comprehensive manual transcriptions. 

Benchmark ASR experiments with transformer-based models (Transformer, Conformer, Branchformer) and pre-trained models like Whisper and SenseVoice-Small demonstrate the dataset’s challenges in code-switching ASR and show substantial room for improvement.

### Strengths
1. Large-scale and spontaneous: 104 hours of fully transcribed dialogue data, larger and more natural than many existing CS datasets.

2. Full-length dialogues: Enables study of contextual code-switching patterns beyond isolated utterances.

3. Rigorously annotated: Manual transcription with word-for-word fidelity, preservation of disfluencies, accents, and exact pronunciations.

### Weaknesses
1. Limited language pair: Only Mandarin-English code-switching, excluding other important bilingual combinations.

2. Controlled environment: Recordings in quiet settings with smartphone microphones, limiting acoustic variability compared to wild scenarios.

3. Speaker bias: All speakers are native Chinese with strong English proficiency; no native English speakers code-switch to Mandarin.

### Questions
1. How does the annotation team handle ambiguous or unclear speech segments during transcription to ensure correctness?

2. What specific procedures were used during quality control to verify transcript accuracy systematically?

3. Are there error rates or inter-annotator agreement statistics reported for transcription quality evaluation?

4. How do frontier models like Gemini and GPT handle the acoustic and phonetic variations in code-switched speech?

### Soundness
3

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
4

### Summary
This paper presents a new dataset of Mandarin-English spontaneous conversational code-switched (CS) speech. It comprises full dialogues and appears to be very carefully transcribed. Although many Mandarin-English datasets are available, the authors note that this is the largest one to date that comprises fully spontaneous speech, and the only one to contain full dialogues.

Extensive baseline automatic speech recognition (ASR) experiments are carried out using pre-trained models, models trained only on the data, and fine-tuned models. Results confirm that CS remains challenging for ASR, even for very large foundation models.

### Strengths
This is a really thorough, well designed piece of work. Great care has been taken in the data collection, and the result is a corpus that will be of great use to many researchers.

Although some would say that a "dataset" paper does make a major contribution, I would argue that this is a particularly high-quality offering, both in terms of the usefulness of the data, the collection methodology, and the paper itself.

You did a great job of clearly differentiating your work from previous Chinese-English datasets.

### Weaknesses
I have listed some minor weaknesses below. The only significant one that I noted was that there was no comment on the way in which the way CS behaviour was enforced would have affect the type of switching observed in the data.

The participants were instructed to first speak Chinese, then mixed-language and then English. This design partly explains why the data is so rich in code-switching (compared to other datasets where utterances are mostly monolingual, so relatively few CS instances can be obtained) but it does worry me that it may make switching somewhat forced and unnatural. For example, it may remove the freedom of speakers to switch for topic-related or social reasons, resulting in (perhaps) syntax playing a more dominant role than in truly natural switching.  *See Questions for further discussion of this point*.

The authors should have provided on the instructions given to participants, and perhaps conducted validation that the slightly forced design did not influence their behaviour.

Were the Chinese and English monolingual portions of the dataset strictly monolingual? What if the participants strongly felt that it would be natural to switch in these sections, even though they had been instructed not to?

Because Chinese and English have somewhat equal status as major languages, I wonder whether any findings would generalise to language pairs where the two languages play unequal roles, or where switching would be more topic enforced? (For example, in many languages, speakers might be need to switch to English for technical vocabulary).

### Questions
The authors will recognise that I have reviewed this paper previously when submitted to the ACL ARR.  I previously gave the paper good scores, and since this version is very little changed from the previous one, I have for the most part left identical review comments for the new meta reviewers.  This includes the "weaknesses" above – though I have removed one from my previous review that I feel has been addressed.

I appreciate the new inclusion of CS-related works in other language pairs, although would have liked more discussion of these in the text.

I am including here my previous comments (in block quote) along with previous author rebuttal (in italics) with comments about whether the suggestions have been addressed in the new version.  The first of this features in the section above.  The others were just minor comments that I am just including here.

> The participants were instructed to first speak Chinese, then mixed-language and then English. This design partly explains why the data is so rich in code-switching (compared to other datasets where utterances are mostly monolingual, so relatively few CS instances can be obtained) but it does worry me that it may make switching somewhat forced and unnatural. For example, it may remove the freedom of speakers to switch for topic-related or social reasons, resulting in (perhaps) syntax playing a more dominant role than in truly natural switching. 

- _Participants were not strictly restricted to a single language within the designated monolingual segments and could code-switch naturally if it felt appropriate. Likewise, purely monolingual speech was permitted within the code-switching segment. The transcriptions faithfully capture the actual utterances spoken, ensuring that the dataset authentically reflects real conversational dynamics._

Where did you explain this in the revised manuscript?  I could not see it.

> When discussing SEAME (perhaps the most widely used English-Mandarin dataset) you imply that the only difference is that it is a paid dataset. Yours being free doesn't make it a contribution in itself. (The table shows that there are other differences – SEAME doesn't include full dialogues – and you should comment on this in the text).

- _SEAME: Thank you for the suggestion. We will revise our discussion of SEAME to emphasize key distinctions beyond accessibility, particularly noting that CS-Dialogue provides full-length dialogues with complete transcriptions, unlike SEAME._

I feel that you didn't do this, merely removing reference to SEAME being a paid dataset.

> Did the speakers know each other? If so, in what capacity? It's not made clear if they were physically in the same room, or using a virtual platform – and if so, was it audio-visual or audio-only?

- _Speaker Interaction & Setting: The dialogues were conducted via an audio-only virtual platform, and participants generally did not know each other beforehand._

You did make it more clear about the platform used, but I still did not see information about whether the participants were acquainted with each other before the data collection.

> You comment on accent accommodation in the transcriptions. I assume this was for Chinese only. Did you do anything to accommodate English accents, or where they considered to all be the same?

- _English Accents: You are correct. While our transcription protocol explicitly accommodated regional Chinese accents (Sec 3.2.1, point 4), English accents were not systematically differentiated during annotation._

I did not see any comment on this in the revised manuscript.

> The tokens per second speaking rate isn't very useful. Without an understanding of what exactly the tokens are and how they vary between languages. Perhaps phonemes per second might have been better - I assume that the speaking rates weren't really dramatically different between English and Chinese. If they were, it would cast doubt on the English proficiency of the speakers.

- _Speaking Rate Metric: We acknowledge the potential limitations of measuring speaking rate in tokens per second. We will consider alternative metrics, such as phonemes per second, to provide a more linguistically meaningful comparison._

This was not done – the speaking rate metric was simply removed, which is a shame.

- _Table 8 & 9 Wording: We will revise the captions to explicitly state that the results are "on the CS-Dialogue test set" to avoid ambiguity.
Results Breakdown by Language Type: While we already report WER and CER separately for English and Chinese segments (Sec 5.1 Metrics) alongside overall MER, we acknowledge that further breaking down results by EN/CN/Mixed segments within the test set could provide additional insights._

Thanks, you addressed these comments.

> why did you choose to fine-tune only Whisper, when it was one of the worst-performing foundation models?

- _Fine-Tuning Choice & Protocol: We selected Whisper for fine-tuning because it is one of the most widely used multilingual ASR foundation models. Details on the fine-tuning protocol, including hyperparameters (learning rate, epochs, batching), are provided in *Appendix D._

My question wasn't about why Whisper was chosen in the first place, but why was Whisper the _only_ model that was fine-tuned?  There was no discussion of this in the revised text.

### Soundness
3

### Presentation
3

### Contribution
3
