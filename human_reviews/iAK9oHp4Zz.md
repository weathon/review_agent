# MuPT: A Generative Symbolic Music Pretrained Transformer

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
In this paper, we explore the application of Large Language Models (LLMs) to the pre-training of music. While the prevalent use of MIDI in music modeling is well-established, our findings suggest that LLMs are inherently more compatible with ABC Notation, which aligns more closely with their design and strengths, thereby enhancing the model's performance in musical composition.
To address the challenges associated with misaligned measures from different tracks during generation, we propose the development of a $\underline{S}$ynchronized $\underline{M}$ulti-$\underline{T}$rack ABC Notation ($\textbf{SMT-ABC Notation}$), which aims to preserve coherence across multiple musical tracks. 
Our contributions include a series of models capable of handling up to 8192 tokens, covering 90\% of the symbolic music data in our training set. Furthermore, we explore the implications of the $\underline{S}$ymbolic $\underline{M}$usic $\underline{S}$caling Law ($\textbf{SMS Law}$) on model performance. The results indicate a promising research direction in music generation, offering extensive resources for further research through our open-source contributions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a Synchronized Multi-Track ABC to improve coherence across tracks in music generation, addressing the limitations of traditional MIDI formats. Combining with the LLM,  MuPT outperforms other models in terms of repetition rate and structure in multi-track music generation. The paper also investigates the Symbolic Music Scaling Law, showing improvements in performance as model size and training data scale up. However, the novelty of the model is limited, and the effect of the proposed SMT-ABC representation seems not very significant.

### Strengths
1. This paper introduces SMT-ABC, a novel representation that improves coherence across multi-track music generation.
2. The investigation into SMS Law offers valuable insights into the relationship between data repetition and model performance, providing practical guidelines for optimizing large-scale music generation models.

### Weaknesses
1. There are some mistakes like missing reference targets (e.g. in Section 5.1 ‘please refer to ??’).
2. Although the SMT-ABC representation is interesting by adding a special token ‘<|>’ and changing the sequence of the original ABC notations, the novelty of this paper seems limited, for the model architecture is mostly based on existing works.
3. Changing the sequence of the original ABC notation from bar-sequence to track-sequence may preserve coherence across multiple musical tracks. However, what about the coherence in a track? The contents inside a bar seem to be adjusted to the very back of the input sequence because SMT ABC aligns the tracks first. Besides, from Table 5, the improvement of SMT-ABC seems very slight compared with the original ABC.

### Questions
1. What is the model scale of SMT-ABC and Original-ABC in Figure 8 in Appendix B.2?
2. Since GPT-4 is not a model for symbolic music generation, what is the motivation to compare with GPT-4?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper "MuPT: A Generative Symbolic Music Pre-Trained Transformer" explores the use of Large Language Models (LLMs) for symbolic music generation, with a focus on leveraging ABC notation over the traditional MIDI format. The authors propose the Synchronized Multi-Track ABC Notation (SMT-ABC Notation) to improve coherence across musical tracks. They present models capable of handling up to 8192 tokens and introduce the Symbolic Music Scaling Law (SMS Law) to optimize the training of these models. Their MuPT model outperforms GPT-4 and ChatMusician in both objective metrics and subjective evaluations of music generation, and they have open-sourced their models to encourage community-led research.

### Strengths
The introduction of SMT-ABC Notation represents a novel approach to enhancing the coherence of multi-track symbolic music generation. The use of ABC notation as a foundation for LLMs in music generation, instead of the more commonly used MIDI, adds an original perspective to the field. The concept of the Symbolic Music Scaling Law (SMS Law) is also a significant contribution, offering new insights into the training dynamics of symbolic music models.
The paper is well-structured, with clear explanations of the SMT-ABC Notation and the SMS Law. 
The open-sourcing of the models and intermediate checkpoints is a valuable contribution, enabling further exploration and innovation in symbolic music modeling by the broader community.

### Weaknesses
1. The experimental design is questionable. SMT-ABC is the only methodological innovation in this paper, yet there is insufficient comparative experimentation to explore how SMT-ABC actually improves performance. Models like GPT-4 are not specifically designed for symbolic music generation, and it's predictable that the proposed model would perform better without even conducting experiments. Were the other models in the comparative experiments, such as GPT-4 and MMT, trained or fine-tuned on the SMT-ABC dataset? If not, it may not demonstrate the true effectiveness of the algorithmic innovation of SMT-ABC.

2. The Scaling Law experiments are detailed, but they primarily validate some emergent performance issues that have been verified in other domains. It does not clearly indicate whether the issues in the music domain can be resolved through the Scaling Law, which suggests that the detailed Scaling Law experiments may not be significantly relevant to the problems this paper aims to address.

3. (1) The training data lacks specific metadata, such as detailed categorization and style of music, which may limit the model's generalization ability in specific styles or types of music. (2) The paper requires adjustments for the alignment of different tracks, and the model may still have certain limitations in generating more complex musical structures, such as polyphonic music. (3) There is no in-depth analysis of the detailed differences between different model architectures, their respective strengths and weaknesses in handling symbolic music, which may make it difficult for readers to understand the specific improvements of MuPT over other models. (4) Subjective factors such as the musical background and preferences of listeners may affect the results of subjective evaluations, and more diverse assessments might provide more comprehensive conclusions. (5) Although experiments show that repeated data can improve model performance, it may lead to overfitting in some cases. Although the paper discusses early stopping and overfitting, more experimental support is needed for the effectiveness of these explanations.

### Questions
1. The paper's experimental design is flawed due to insufficient comparison with other models to validate SMT-ABC's performance improvement. It's unclear if models like GPT-4 and MMT were trained on the SMT-ABC dataset, questioning the true effectiveness of SMT-ABC's innovation.

2. The Scaling Law experiments, while detailed, do not clearly show their relevance to music domain issues, suggesting they may not be significantly pertinent to the paper's objectives.

3. (1) The training data lacks metadata, potentially limiting model generalization. (2) The model may struggle with complex musical structures like polyphony. (3) There's a lack of analysis on model architectures' strengths and weaknesses in symbolic music. (4) Subjective evaluations might be biased, necessitating more diverse assessments. (5) Repeated data may improve performance but could also cause overfitting; more experimental evidence is needed to support claims on overfitting prevention.

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
4

### Summary
This paper explores using large models for symbolic music pre-training, proposing ABC Notation as a more compatible format than the commonly used MIDI. To maintain track coherence in multi-track compositions, the authors introduce SMT-ABC Notation. Key contributions include models that handle up to 8192 tokens, covering 90% of their symbolic music dataset, and an examination of the Symbolic Music Scaling Law (SMS Law) on model performance. The experiments show better performance than GPT4.

### Strengths
1. Studying a foundation model for symbolic music is both interesting and meaningful work.

2. The team put significant effort into analyzing large datasets and training the foundation model.

3. The findings on scaling laws are interesting.

### Weaknesses
1. The research contribution is limited. To enhance the novelty of the approach, consider addressing research questions such as: What factors make ABC better than MIDI, and how does the uniqueness of ABC benefit large models in the feature space? Under what circumstances does the model perform better with ABC notation?

2. The comparison between ABC and MIDI is incomplete. The assumption that ABC is better than MIDI requires a detailed and careful analysis. For example, experiments exploring the following would help clarify the relative strengths and weaknesses of each format for large model training: under the same data size and model architecture, assess whether ABC consistently outperforms MIDI in terms of model efficiency and output quality. Note that MIDI can be represented in various formats, such as piano roll, matrix formats like Compound Word, or sequences like REMI. For Figure 1, further details would improve the visualization, such as comparing ABC directly with MIDI instead of only with the piano roll, as the piano roll omits much information found in MIDI. Indicate to readers what information is missing from both MIDI and ABC, what MIDI can represent (e.g., velocity, control information) that ABC cannot, and vice versa. Additionally, clarify the correspondence between ABC and MIDI, such as how "K: Bb" in ABC notation relates to MIDI, or what "|F2 z G|" represents in ABC and its MIDI equivalent in the figure.

3. The experimental design has flaws. When comparing music-related performance, the proposed model should be evaluated against symbolic music generation models rather than language models. Relevant symbolic music generation baselines include Music Transformer [1], Compound Word Transformer [2], and RWKV [3] on the same large symbolic music dataset.


4. typos such as Line 353.

### Questions
Considering the limitations of ABC notation—such as its inability to represent control information and note dynamics, and its less common use within the music community—these weaknesses could significantly impact the model’s potential for generating symbolic music for potential users (composers, musicians, etc). How do you foresee the future roles and applications of ABC and MIDI representations in music generation?

How many songs are included in the dataset, and what detailed information is provided for each song? How do the authors address intellectual property concerns?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents MuPT a pretrained music transformer for symbolic music generation. The paper is a study of the techniques from LLM literature applied to symbolic music data. The authors discuss a novel data representation for symbolic data: SMT-ABC to take issues related to misaligned multi-track data. The authors also present a scaling law appropriate for symbolic music where the dataset size is typically orders of magnitude smaller than text data that was used to establish the Chinchilla laws. The authors conduct subjective and objective evaluations of their model and compare with other similar models of symbolic music. They find that their model is heavily preferred compared to other models in human studies.

### Strengths
The study is conducted in a systematic fashion and covers a lot of the relevant topics that LLM-related literature should talk about, such as the relationship between dataset size, model parameters, etc with the quality of generated data.
The quality of the generated music shared in the supplementary material is decent.

### Weaknesses
The authors do not share any samples from the baseline models in supplementary materials which would have helped support the results in the human study.
The repetition metrics seem to not be clearly motivated. Why would the average repetition rate be a meaningful metric. Isn't the more important metric the position of the repeats?
Some writing issues. E.g. missing reference in line 353, missing word in line 505.

### Questions
Aside from the questions in the weaknesses section:
- Is there any study to check if the generations are copied from the training data?

### Soundness
3

### Presentation
3

### Contribution
3
