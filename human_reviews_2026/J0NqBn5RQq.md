# Learning Music Style For Piano Arrangement Through Cross-Modal Bootstrapping

- Decision: Reject
- Scores: 6, 6, 6

## Abstract
What is music style? Though often described using text labels such as "swing," "classical," or "emotional," the real style remains implicit and hidden in concrete music examples. In this paper, we introduce a cross-modal framework that learns implicit music styles from raw audio and applies the styles to symbolic music generation. Inspired by BLIP-2, our model leverages a Querying Transformer (Q-Former) to extract style representations from a large, pre-trained audio language model (LM), and further applies them to condition a symbolic LM for generating piano arrangements. We adopt a two-stage training strategy: contrastive learning to align auditory style with symbolic expression, followed by generative modelling to perform music arrangement. Our model generates piano performances jointly conditioned on a lead sheet (content) and a reference audio example (style), enabling controllable and stylistically faithful arrangement. Experiments demonstrate the effectiveness of our approach in piano cover generation, style transfer, and audio-to-MIDI retrieval, achieving substantial improvements in style-aware alignment and music quality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a cross-modal framework to learn and apply implicit music styles from raw audio to symbolic music generation. The model uses a Querying Transformer (Q-Former) to extract style representations from a pre-trained audio language model and conditions a symbolic language model to generate piano arrangements. A two-stage training strategy is adopted: (1) contrastive learning aligns auditory style with symbolic expression; (2) generative modeling performs arrangement. The system generates piano performances conditioned on both a lead sheet (content) and a reference audio example (style), enabling controllable, stylistically faithful results. Experiments show significant gains in piano cover generation, style transfer, and audio-to-MIDI retrieval.

### Strengths
The main strengths of the paper are:
- The proposed system obtains good results in style transfer, and this with audio input of diverse styles and instrumentations. 
- The experimental plan is solid and well highlights the merits of the proposed system. 
- The paper is clearly and nicely written
- The (online) demo is very convincing

### Weaknesses
The main weaknesses of the paper are:
- The novelty of the proposed system mainly relies on existing building blocks ;
- The overall proposed system is quite complex in the overall and it is not clear that it is better or more controllable than a much simpler system based on a style recognition from audio (or style retrieval/classification from audio) and then audio generation from lead-sheets (e.g. current music generation systems using chords sequence and a style label as inputs can provide excellent rendering in hundreds of styles).
- The design of the perceptual test is lacking some details.

### Questions
-	Perceptual tests: the participants participated online and had diverse musical backgrounds. More details could be given: Were they all musicians ? capable or reading lead sheets ? were the authors excluded from the survey ? how the quality of participant’s answers was controlled/monitored ?
-	Objective evaluation: in section 5.3, it is mentioned that “..Clamp3 compuutes similarity […] and all generated symbolic piano covers”.  Do you mean here all covers from all audio of the database with 1 cover per audio ? How many covers then (to have an idea of how good/bad is a rank of 16 or so on POP909) ?
-	The generative objective for the Q-former is not clear. It is said (p4) that the Q-former is trained on an input audio and lead sheet (but the lead lead is not input to the Q-former in stage 1 in figure 1 and figure 2… this is confusing and it is not really clear what is finally used as input information for stage 1.

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
3

### Summary
The paper proposes a cross-modal framework that learns implicit musical style from raw audio and applies it to symbolic music generation. A two-stage model is introduced: Stage 1 performs audio-symbolic representation learning via three complementary losses (contrastive, matching, and audio-grounded generative) to distill style information with a Q-Former.
Stage 2 conditions a symbolic LM (MuseCoco) on both the learned style embedding and a lead sheet for controllable piano arrangement. The system supports multiple tasks including piano-cover generation, style transfer, and audio-to-MIDI retrieval.

### Strengths
1. The paper defines a new setting — learning implicit style from audio for symbolic arrangement — that meets the need for fine-grained style control, distinguishing it from transcription or text-conditioned generation.
2. The two-stage design enables flexible applications (piano cover generation, style transfer, and audio-to-MIDI retrieval).
3. The three complementary training losses effectively achieve audio-symbolic representation learning; retrieval results confirm that the learned style representation is robust and generalizable.
4. The presentation is clear and visually well-organized.

### Weaknesses
1. While audio input allows more detailed style control, the current framework reduces usability — users must provide a reference audio rather than a simple text prompt. It would be more practical to retain MuseCoco’s original text-conditioning ability for flexible use.
2. The learned style representation mainly captures global characteristics and fails to model segment-level or bar-level dynamic style variations.
3. Objective evaluation is too limited. Using only CLaMP3 as a metric is insufficient to justify the quality and style consistency of generated outputs. Piano cover generation should be evaluated with richer metrics. In addition, the style transfer capability should be further validated through subjective listening evaluations, demonstrating perceptual differences and user preference across styles.
4. Minor: In Table 2, the Rank column should have a downward arrow (↓) instead of upward.

### Questions
See Weakness section above.

### Soundness
2

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
3

### Summary
The article introduces a novel cross-modal framework for generating expressive symbolic piano arrangements by learning implicit music style from raw audio. The primary objective is to generate a symbolic piano arrangement that is conditioned on two inputs: an audio example (for the style) and a lead sheet (for the melody and harmony content). This enables controllable and stylistically faithful arrangements that go beyond explicit text labels. The core of the framework is a Q-Former, inspired by BLIP-2, which acts as a lightweight bottleneck to bridge the gap between a frozen, pre-trained audio Language Model (LM) and a symbolic music LM. The Q-Former is designed to extract a representation of the implicit music style from the audio LM's hidden states. Similar to BLIP-2, a two-step sequential training strategy is used to "bootstrap" the system without retraining the large LM backbones. First, in a cross-modal representation learning, the Q-Former is trained using three complementary objectives (Contrastive Learning, Audio-Symbolic Matching, and Audio-Grounded Symbolic Generation) to align the auditory style representation with symbolic expression, while disentangling it from explicit music content. Then, in the second step, the symbolic LM (MuseCoco with a LORA adapter) is conditioned on the extracted style embedding from the Q-Former and the lead sheet content to generate the final piano arrangement.

### Strengths
- The paper tackles an ambitious and important problem in symbolic music generation: disentangling and transferring music style from a complex audio signal to a symbolic arrangement. By defining style as "hidden in concrete music examples" rather than relying on abstract text labels, the authors set a high bar, aiming for expressive performance nuances rather than just generic genre characteristics.

- The central technical contribution—using a Querying Transformer (Q-Former) to bridge a frozen Audio Language Model (LM) and a frozen Symbolic LM (MuseCoco)—is highly innovative and resource-efficient.

- The framework allows for an interesting form of controllable music generation. The final arrangement is jointly conditioned on two independent inputs: 1. The explicit lead sheet (melody and chords), and 2. The implicit style from a raw audio reference. This dual control is superior to systems that rely on simple text tags or require the arrangement to be fully contained within the style reference, making the model flexible for diverse creative applications.

- Audio examples reveal partially successful transfer of musical style (notably tempo, intensity contours for the main melody), which would probably partly qualify as implicit style and partly as arrangement.

- The framework's intermediary component, the Q-Former, is successfully evaluated in an experimental setup requiring the retrieval of the correct MIDI excerpt given audio. While the number of candidates remains small, it still indicates a successful extraction of the properties of symbolic music from audio examples.

### Weaknesses
- The paper does not define the term "musical style",  the introduction introduces the concept of implicit style. A clear definition of this term remains missing. Authors explain: *When musicians learn a style, instead of relying on abstract definitions like “romantic” or “jazz” alone, they absorb patterns from music examples that share common stylistic traits.* But jazz is not a *definition*, it is a *label* for a collection of examples (all music carrying the label given by humans) anyway. Then there is the interpretation style within the larger genre style. Each artist has their own style that they use to express their personal identity. Here, for piano performance, precise timing, arpeggios, velocity nuances, evolution (e.g., accentuation), and duration variations are certainly important. All this can be described explicitly without too much effort. All this should have been described more thoroughly.

- The evaluation appears weak. I don't quite understand what was measured in the objective evaluation. Authors write: *For each test audio, CLaMP3 computes the similarity between the audio and all generated symbolic piano covers. If the most similar symbolic candidate corresponds to the one generated from the audio input, we count it as a correct match.* If I see correctly, you have 100 audio examples, generate a cover for each of them, then you compare each generated cover with all audio, and you count as success if the original audio turns out to be the most similar one? If I see correctly, this does not allow comparison of the overall style quality between the various algorithms; it could be that one algorithm has very poor overall similarity but has one feature that it successfully copies from the lead sheet. Then this algorithm, even if overall producing bad results, would turn out best compared to the others, which are less coherent but overall more similar. Then, the subjective test does not evaluate style similarity but only musicality.

### Questions
Given the fact that the implicit musical style can be described at least partly with explicit parameters, I wonder whether it would not be possible to create some more objective metrics, as for example tempo, velocity contours over the notes, or similar simple metrics. I noted, for example, that the duration of the piano covers is quite different. A successful cover that copies the style should have keep the  duration, no? Or do you include the tempo from the audio into the lead sheet? Another possible metric could be the presence of all notes from the lead sheet.

### Soundness
2

### Presentation
2

### Contribution
3
