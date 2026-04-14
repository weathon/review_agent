# Representing speech through autoregressive prediction of cochlear tokens

- Decision: Reject
- Scores: 6, 6, 5, 5

## Abstract
We introduce a biologically-inspired model for encoding speech through an autoregressive prediction objective applied to input representations modeled after the human cochlea.
Our modeling framework is inspired by the human auditory processing hierarchy. The first stage of our framework transforms the raw audio waveform to a time-frequency representation inspired by the human cochlea, with an intermediary step that effectively discretizes the audio representations (cochlear tokens). The second stage of our model learns a simple, yet powerful, autoregressive sequence model over the discretized audio input.
We demonstrate that our model learns meaningful representations of phonemes and word identities, and state-of-the-art representations of lexical semantic similarity. In addition, our model shows competitive performance on several downstream audio tasks from the SUPERB benchmark. In addition to our model’s strong representational capabilities, we demonstrate our model's ability to generate continuations of audio at various temporal scales, which can be visualized in a cochleagram time-frequency space to provide insights into the model's predictions.
Our model provides a novel framework for speech representation learning, aiming to advance the development of more human-like models that flexibly and efficiently handles a range of speech-based tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces a two-stage framework for speech representation learning, drawing on biologically inspired cochlear representations and autoregressive modeling. The first stage, WavCoch, is a vector-quantized encoder designed to convert an STFT spectrogram into a cochleagram representation via MLP layers, a 14-bit Lookup Free Quantization (LFQ) layer, and convolutional layers. The output of the LFQ referred to as “cochlear tokens,” serves as the input for the second stage. The second stage, CochStream, is a GPT-style autoregressive transformer trained to predict the next cochlear token.

To evaluate the effectiveness of these representations, the authors assess CochStream on tasks such as linear phoneme and word probing, lexical semantic similarity, and several tasks from the SUPERB benchmark. CochStream shows excellent performance in lexical semantic similarity and demonstrates competitive results across other tasks, establishing it as a strong contender among state-of-the-art speech representation learning models.

### Strengths
1. **Originality**: The paper takes an interesting approach by introducing a cochlear-inspired representation (cochleagram) for speech modeling, which is relatively uncommon in the field. 

2. **Quality**: CochStream demonstrates excellent performance on certain tasks, particularly lexical-semantic similarity, suggesting that the model captures semantic features effectively. Additionally, the inclusion of various benchmarks and comparisons to other models is a positive aspect, as it provides a well-rounded view of how CochStream performs across different types of tasks, even if it doesn’t consistently outperform alternatives.

3. **Clarity**: The paper does a good job of describing the high-level design of the WavCoch and CochStream stages, making it reasonably easy to follow the process from quantization to autoregressive modeling. However, some clarity is lost due to inconsistencies in reported training details, which might affect reproducibility.

4. **Significance**: By aiming for a biologically-inspired design, the paper contributes to the broader conversation on incorporating more natural processing methods in machine learning. If refined, the framework could be a step toward more interpretable speech models and potentially inspire future work in biologically plausible approaches.

### Weaknesses
**Biological Inspiration**: While the cochleagram representation mimics some aspects of auditory processing, most other components, including the Lookup-Free Quantization (LFQ) and the Transformer, lack biological basis. This raises questions about the validity of the biological inspiration claim.

**Transformer-APC Comparison**: Since CochStream resembles a Transformer-based APC model but with cochlear tokens as input, a direct comparison with an APC trained on the same dataset would strengthen the claim that cochlear tokens are superior to filterbank features or other standard inputs. Without this, the benefit of cochlear tokens over more conventional features is less evident.

**Effectiveness of Separate vs. Integrated Discretization**: Unlike VQ-APC, which integrates discretization within its structure, CochStream uses a separate WavCoch stage. Yet, no experiments justify the choice of a separate discretization stage over an integrated approach. A direct comparison could validate whether the proposed method offers advantages.

**Unclear Advantage of Cochleagram Representation**: The benefit of using a cochleagram over other perceptually inspired representations, like the mel-spectrogram, is not fully demonstrated. Although the paper claims that cochleagram outputs improve interpretability, this interpretability could also apply to other representations. Moreover, the Transformer architecture remains largely a black box, limiting interpretability beyond the output of the WavCoch model.

**Efficiency Claims Not Verified**: While the paper claims efficiency as a strength of cochlear tokens, there is no quantitative analysis to support this. For instance, CochStream’s tokens are sampled at a higher rate (197Hz) than HuBERT’s (50Hz), and cochleagram computation is more costly than mel-spectrograms, albeit only at training time. These factors suggest that cochlear tokens may not be as efficient as implied.

**Inconsistent and Incomplete Training Details**: Key details about the training datasets and parameters are lacking or inconsistent, which raises concerns about reproducibility.

- No citation or details are provided about the 50,000-hour dataset, leaving ambiguity around its source. Using standard datasets like LibriSpeech or LibriLight for English speech tasks would enhance reproducibility.
- In Section 2.1.3, it’s mentioned that WavCoch was trained on 500 hours while CochStream was trained on 50,000 hours, yet CochStream base is reported as trained on 500 hours in the tables.
- Table 2 states CochStream base has 135 million parameters, which contradicts other descriptions in the paper, leading to further confusion.

**Unfair Comparisons**: There are no baselines that are trained on the same dataset, hence it is difficult to put the results into perspective. It remains unclear whether the performance improvements come from the dataset, the transformer configuration or the input representations.

**Improper References**: Several references are cited from the arxiv versions instead of the original published venues. Also, serveral citations lack any details about conference/journal where they were published.

### Questions
- A good ablation study is necessary to validate the claims of the paper. Following are some experiments that could help improve the paper:
  - Mel-spectrogram as intermediate representation instead of cochleagram to validate the choice of input representation
  - Cochleagram-based APC trained on the same dataset.
  - Comparison with VQ-APC

- Why were standard publicly available datasets for English speech not utilized for the experiments?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes a self-supervised learning (SSL) framework to learn generic speech representations using cochlear tokens derived from a biologically inspired waveform-to-cochleagram transformation, referred to as WavCoch. These cochlear tokens serve as input representations for an auto-regressive model, CochStream, which aims to emulate the human auditory processing stream.

### Strengths
The paper is well-written, with a clear and interesting motivation rooted in mimicking the human auditory system. The authors’ use of cochleograms as intermediate acoustic representations is compelling, and the performance of these biologically inspired features on downstream tasks is promising. The results show that hand-crafted, biologically motivated features can indeed achieve competitive performance, which could have significant implications for the design of speech processing systems.

### Weaknesses
While the paper achieves competitive performance on benchmark tasks, it relies on general-purpose, objective ML measures to validate the proposed features. This focus shifts away from the original biological motivation to a more standard ML performance evaluation. For researchers focused on ML, data-driven features generally remain more attractive due to better performance across tasks and the lack of a need for hand-crafting input features. To realign with the biological motivation, the paper would benefit from additional biologically relevant evaluations, such as subjective metrics from human listening tests, to better demonstrate the utility of these handcrafted features for tasks where biological relevance is key.

### Questions
Performance on Non-Content Tasks:
Given that WavCoch is designed based on human auditory processing, it would be expected to excel in tasks that are less content-focused. The promising emotion recognition results align with this expectation, but the lower performance on speaker identification is less encouraging. Could the authors consider discussing this disparity to clarify the strengths and limitations of the cochleogram-based representations for non-content-focused tasks.

Speech Continuation Capability:
In Section 3.3, the authors include examples to illustrate CochStream’s speech continuation ability. However, assessing this capability solely by visual inspection of cochleograms is challenging without quantitative metrics or synthesized speech outputs. As the framework is intended to mimic human auditory processing, could the authors incorporate a subjective evaluation from human annotators on synthesized outputs to strengthen the claims of biologically motivated capabilities?

Motivation for Auto-Regressive Training:
The paper currently lacks clarity on why an auto-regressive model is preferred over, for instance, a BERT-style masking training approach. Is the primary intent to enable speech continuation? This could be clarified. Providing further justification here would add clarity to the design decisions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces CochStream, a biologically-inspired model for representing speech by predicting sequences based on auditory representations akin to the human cochlea. The model operates in two stages. (1) WavCoch Encoding: The first stage transforms raw audio into a time-frequency cochleagram (a cochlea-inspired representation), effectively converting the continuous audio into discrete "cochlear tokens." (2) Autoregressive Prediction with CochStream: Using these tokens, CochStream—an autoregressive Transformer—predicts the next token in the sequence, creating a generative model that can continue audio sequences and interpret speech patterns.

### Strengths
- CochStream explores an audio representation generation method inspired by the human cochlea.
- CochStream was evaluated on the SUPERB benchmark for tasks such as speech recognition, intent classification, and speech separation.
- The model can visualize predictions in cochleagram form, offering interpretable insights into its speech representations.

### Weaknesses
- The baseline model seems a bit weak and should try to incorporate newer and more powerful models such as Whisper; additionally, CochStream claims to have the appearance of acoustic information, so that should be compared with models that aim at audio reconstruction such as Encodec, DAC, Soundstream;
- Based on the results in Tables 1 and 3, the performance improvement of CochStream over the baseline models appears limited. I would like the authors to further clarify the main advantages of CochStream compared to traditional approaches.
- The design of CochStream reminds me of recent popular single-codebook audio reconstruction approaches, such as Single-Codec and WavTokenizer. I noticed that CochStream has a vocabulary size of 16,384. Have the authors explored the impact of vocabulary size on performance and the vocabulary utilization rate? Is there any occurrence of collapse?
- CochStream seems to complete wav continuation based on a seed. I’m curious whether it has the ability to generate long-sequence audio and how effective it is.
- In fact, CochStream aims to explore audio representation from a biological perspective, but I believe that this simple autoregressive modeling approach cannot adequately simulate the complex processing mechanisms of biological systems.

### Questions
See details in Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents CochStream, a biologically-inspired two-stage framework for speech representation learning. The model leverages human cochlear-inspired audio processing to create a discrete "cochlear token" representation, which it then feeds into an autoregressive Transformer model to predict future tokens. This approach differs from traditional signal-reconstruction and contrastive models, aiming instead for a representation that captures the hierarchical structure of human auditory processing. Experimental results show that CochStream performs competitively on phoneme recognition, word decoding, and lexical semantics tasks, often surpassing existing baselines on the SUPERB benchmark.

### Strengths
1.	The model's biologically-inspired cochlear token framework is well-aligned with human auditory processing, making it a promising approach for more human-like and interpretable speech representations.

2.	The authors validate the model’s versatility across various tasks (phoneme/word decoding, lexical semantics) and benchmarks (e.g., SUPERB), demonstrating its competitive performance and interpretability advantages over existing models.

### Weaknesses
1.	The paper lacks formulas and a clear explanation of the cochlear representation, as well as a model architecture diagram. This makes it difficult to understand how the cochlear representation is converted into audio and how it compares to or provides advantages over the mel representation. A more thorough theoretical or visual explanation of the cochlear encoding process would enhance clarity.


2.	The experimental results on linear probing performance for phonemes and words on the TIMIT dataset are insufficient. Notably, CochStream-base is not compared with similarly parameterized models such as WavLM-base or wav2vec2-base, which limits the ability to assess CochStream's true effectiveness relative to models of similar scale. Additionally, CochStream-large demonstrates only average performance in Word Decoding. The model's embedding performance on various downstream tasks in the SUPERB benchmark also does not clearly showcase a significant advantage over existing models.

### Questions
1.	Why is section 2.1.4 OBTAINING COCHSTREAM EMBEDDINGS empty?

2.	Due to the lack of information in OBTAINING COCHSTREAM EMBEDDINGS, I am unsure how the representation predicted by this GPT-style autoregressive Transformer compares in speed to representations from models like HuBERT, which use masked prediction. Could the authors clarify this aspect?

### Soundness
2

### Presentation
2

### Contribution
2
