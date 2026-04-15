# Length-Induced Embedding Collapse in Transformer-based Models

- Decision: Reject
- Scores: 6, 6, 6, 5

## Abstract
Text embeddings enable various applications, but their performance deteriorates on longer texts. In this paper, we find that the performance degradation is due to a phenomenon called \textbf{Length Collapse}, where longer text embeddings collapse into a narrow space. This collapse results in a distributional inconsistency between embeddings of different text lengths, ultimately hurting the performance of downstream tasks. Theoretically, by considering the self-attention mechanism inherently functions as a low-pass filter, we prove that long sequences increase the attenuation rate of the low-pass filter effect of the self-attention mechanism. With layers going deeper, excessive low-pass filtering causes the token signals to retain only their Direct-Current (DC) component, which means the input token feature maps will collapse into a narrow space, especially in long texts. Based on the above analysis, we propose to mitigate the undesirable length collapse limitation by introducing a temperature in $\softmax(\cdot)$, which achieves a higher low-filter attenuation rate. The tuning-free method, called \textbf{TempScale}, can be plugged into multiple transformer-based embedding models. Empirically, we demonstrate that TempScale can improve existing embedding models especially on long text inputs, bringing up to \textbf{0.53\%} performance gains on 40 datasets from  Massive Text Embedding Benchmark (MTEB) and \textbf{0.82\%} performance gains on 4 datasets from LongEmbed, which specifically focuses on long context retrieval. The source code is available at \textcolor{blue}{\url{https://anonymous.4open.science/r/Length_Collapse-22D2}}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper describes the “length collapse” problem in text embedders. It extends previous work treating the self-attention mechanism in the Transformer as a low-pass filter by demonstrating that long sequences increase the attenuation rate of the low-pass filter effect of the self-attention mechanism. Then, the work introduces an additional temperature coefficient that results in a higher low-filter decay rate to mitigate the length collapse problem.

### Strengths
1. The paper is well organized and understandable, and the experiments are adequate.

2. This work attempts to relate the Transformer's self-attention module to the “length collapse” problem in embedding space, which is novel, as most of the related work has only investigated the related problems through empirical studies or manipulations on embedding space.

3. The proposed TempScale is orthogonal to existing methods, can be applied to various state-of-the-art models by plug-and-play, and is training-free.

### Weaknesses
1. The "transformer-based Models" in the title are too big because the analysis is only related to the self-attention mechanism, not to the other architectures in the Transformer.

2. In general, TempScale is a post-processing method for text embedding that solves the problem of too much similarity in long sequence embedding. Still, I don't see any comparison between TempScale and some post-processing baseline methods, e.g., Flow Function [1], and Whitening [2], to solve the similarity problem.

3. The improvement of TempScale on each model is inconsistent, and the main improvement does not come from the long text part (as shown in Figure 6), so more explanation and exploration are necessary.

4. The relevant work [3] and the important long-context embedder [4] have been overlooked.

[1] On the Sentence Embeddings from Pre-trained Language Models. EMNLP2020

[2] Whitening Sentence Representations for Better Semantics and Faster Retrieval. Arxiv

[3] Length is a Curse and a Blessing for Document-level Semantics. EMNLP2023

[4] mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval. EMNLP2024

### Questions
All embedding models studied by the authors have been fine-tuned based on contrastive learning; however, this technique seems to have been deliberately avoided in this work. Contrastive learning was first introduced to improve anisotropy in the embedding space of PLMs, and one manifestation of anisotropy is high similarity between random texts. In other words, without contrastive learning, the similarity between two short text embeddings, not only long texts, is also high.

Therefore, several questions need to be explained: 

(1) How to explain the change of embedding space before and after contrastive learning from the filter perspective; 

(2) Why there is still the problem of “length collapse” after contrastive learning, and where the contrastive learning learning is not good enough; 

(3) The introduction of long text in contrastive learning should make the model perform better on long text, so how can we understand this phenomenon from the filter perspective?

The contribution of this work is very limited if the authors' study is isolated and not linked to contrastive learning. In practice, one would be more inclined to introduce more long text for contrastive fine-tuning than to use a post-processing method such as TempScale, since clearly the former would be more effective.

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
This paper is concerned about text embeddings. This paper trickily discovers an interesting phenomenon where text embeddings of lengthy texts tend to cluster very close to each other and thus the performance of text embeddings is largely degraded. In this paper, this phenomenon is termed length collapse. This paper dedicates a lot efforts to connecting this phenomenon to theory through the lens of fourier transform. Base on the theoretical insight, this paper proposes a very simple, intuitive, yet effective method that imposes a temperature scale to the attention mechanism.

### Strengths
- The length collapse phenomenon is as far as I know not that new. However, this paper might be the very first one that shed theoretical insights on this phenomenon.
- The proposed tempscale is very intuitive and easy to understand.

### Weaknesses
- The experimental results though showcase that tempscale is performant, yet the performance gains over baselines are very marginal.
- I am wondering whether long-context llms still suffer this length collapse, and perhaps furthers discussions on this direction would be more exciting and expected.

### Questions
N/A

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
The paper utilizes Fourier Transform and low-pass filtering as the theoretical framework to analyze the phenomenon of length collapse in embedding models. Based on this analysis, the authors propose TempScale, which introduces a temperature parameter in the softmax function to mitigate the collapse phenomenon. The effectiveness of TempScale is demonstrated through its performance on MTEB and LongEmbed. Generally, I highly appreciate the theoretical framework adopted by the authors and believe that the identified length collapse phenomenon is noteworthy. However, I find the comparison of baseline methods in the experiment section somewhat lacking (see weaknesses). I am very much looking forward to seeing this issue addressed.

### Strengths
- The theoretical framework (Fourier Transform and low-pass filtering) is sound, and the identified length-induced embedding collapse phenomenon is noteworthy.

- The experimental section discusses the impact of the length collapse phenomenon on various tasks in MTEB and LongEmbed, which is commendable.

### Weaknesses
- The proposed method, $softmax(\frac{1}{\tau\sqrt{d}}QK^T)V$, lacks comparison to previous works (although not specifically proposed for embedding models), including: 
  1) $softmax(\frac{logn}{\sqrt{d}}QK^T)V$, as proposed in “Overcoming a Theoretical Limitation of Self-Attention” (ACL 2022); 
  2) $softmax(\frac{1}{\tau\sqrt{d}}QK^T)V$, $\tau=0.1ln(n)+1$, as proposed in “YaRN: Efficient Context Window Extension of Large Language Models” (ICLR 2024). 

  For both methods, $n$ refers to the input sequence length. I suggest that the authors include a discussion on the differences between TempScale and these two methods. The authors could provide a brief comparison of the theoretical motivations behind TempScale and the two mentioned methods; or include an ablation study comparing the performance of TempScale against these two methods on a subset of the MTEB or LongEmbed tasks; or discuss any potential advantages or limitations of TempScale compared to these existing approaches, particularly in the context of embedding models, etc. Have this issue addressed would positively influence my evaluation.

### Questions
None

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper addresses the ineffective performance of attention-based embedding models with long texts. It begins by highlighting that commonly used models struggle with longer inputs. Utilizing visualization techniques, such as t-SNE plots that demonstrate embeddings clustering near the origin, along with statistical evidence like high similarity among long-text embeddings, the authors illustrate a "Length Collapse" issue. This study analyzes the low-pass filtering effect of the attention mechanism, which contributes to this poor performance. Finally, the authors propose a solution by incorporating a temperature parameter in the softmax function to enhance the model's capacity for handling long texts.

### Strengths
1. The paper is logically coherent and well-written, ensuring a smooth reading experience. 
2. It presents a rigorous theoretical framework supported by substantial relevant research. 
3. Additionally, it provides a cohesive analysis of the "Length Collapse" problem, starting with the introduction of the issue, followed by preliminary analysis, in-depth theoretical exploration, and the proposal of a solution based on research findings.

### Weaknesses
1. The MTEB benchmark includes 56 evaluation datasets, but this paper evaluates only 36 of them.  
2. The paper attributes its poor performance in modeling long texts to the low-pass filtering effect of the attention mechanism, though I question whether this is the primary cause. My concerns include:  
   - In Figure 1b, the t-SNE visualization shows long text embeddings clustering near the origin, but this proximity is in the reduced-dimensional space and doesn't confirm their position in the original embedding space.  
   - If low-pass filtering significantly affects long-text modeling, it's puzzling that many large models can effectively handle texts of tens of thousands or even hundreds of thousands of tokens, especially since the long texts in this paper are only a few hundred tokens, which is minor compared to what current models manage.  
   - Figure 5 indicates that $\sigma_a$ values for text lengths between 100 and 500 are similar, yet Figure 1a reveals a significant performance drop in this range, further undermining the proposed analysis's relevance to the problem.  
   - Lastly, I wonder if the training corpus's text length contributes to Length Collapse. Figure 1c shows high similarity among long texts, which is often seen in embeddings before fine-tuning (e.g., embeddings from backbone models). Thus, a possible reason for the poor performance in modeling long texts could be insufficient training data containing longer texts.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2
