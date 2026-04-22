# FISHER: A Foundation Model for Multi-Modal Industrial Signal Comprehensive Representation

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 2

## Abstract
Industrial signal analysis has emerged as a critical problem for the industry. Due to severe heterogeneity within industrial signals, which we summarize as the M5 problem, previous works could only deal with small sub-problems by training specialized models, which lacks robustness and incurs huge burdens during development and deployment. However, we argue that the M5 problem can be dealt by scaling up, where dealing with the multi-sampling-rate is the first step. In this paper, we propose FISHER, a Foundation model for multi-modal Industrial Signal compreHEnsiveRepresentation. To support arbitrary sampling rates, FISHER considers the increment of sampling rate as the concatenation of sub-band information. Specifically, FISHER takes the STFT sub-band as the modeling unit and adopts a teacher-student SSL framework for pre-training. To evaluate the model performance, we also develop the RMIS benchmark, which consists of 19 datasets across four modalities. FISHER is compared with 15 SOTA speech/audio/music encoders, demonstrating versatile and outstanding capabilities with a general performance gain of at least 3.23\%. Meanwhile, FISHER possesses much more efficient scaling curves, where even FISHER-tiny with 5.5M parameters outperforms huge baseline encoders up to 2B. We further reveal that the key to success is adaptively utilizing the full signal bandwidth regardless of the sampling rate. Both FISHER and RMIS will be open-sourced.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes FISHER, a foundation model for multi-modal industrial signal representation, aiming to address the heterogeneity of industrial data, which the authors summarize as the M5 problem. The authors also introduce a new benchmark, RMIS, to evaluate general representation ability via KNN inference without fine-tuning.

### Strengths
1.	Industrial signal modeling is an underexplored domain compared to speech/audio; addressing the M5 heterogeneity problem with a scalable foundation model is meaningful.
2.	The idea of modeling higher sampling rates as concatenated sub-band information is simple yet effective.
3.	The proposed RMIS benchmark covers diverse tasks and modalities, potentially serving as a valuable testbed for future research.

### Weaknesses
1.	The core architecture and SSL setup (teacher-student distillation via EMA, ViT backbone) are nearly identical to existing work, raising concerns that the novelty lies mainly in the data preprocessing stage.
2.	The novelty of the approach is limited, since the method essentially reweights gradients in a way similar to prior balancing strategies, without a clearly distinguished contribution.
3.	Figures could better illustrate sub-band composition to enhance interpretability.

### Questions
1.	It seems that the M5 Industrial Signals are independently fed into the model, with each modality processed and trained separately, without cross-modal interaction during pretraining. Could the model potentially benefit from frequency-aware attention or cross-band interaction mechanisms, rather than pure concatenation?
2.	How are sub-band boundaries handled? Is there any information leakage or overlap between adjacent sub-bands?
3.	Does using a fixed STFT window and hop size across signals with different sampling rates risk under-resolving low-rate signals or over-smoothing high-rate ones?
4.	How was the $f_{base}$ (bandwidth unit) chosen, and how sensitive is the performance to this choice?
5.	Why was KNN inference used exclusively instead of fine-tuning or linear probing?

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
2

### Summary
This paper presents FISHER, a new foundation model aimed at creating a unified representation for industrial signals, a domain the authors characterize with the "M5 problem." The central and most compelling idea is to reframe the challenge of variable sampling rates by modeling signals as a composition of frequency sub-bands. This allows for a consistent processing architecture, which is trained via a teacher-student self-distillation scheme. To properly assess the model's capabilities, the authors have also constructed a new and comprehensive benchmark, RMIS. The experimental evaluation is strong.

### Strengths
The authors should be commended for tackling a problem of significant practical importance. Creating a single model for the diverse and heterogeneous world of industrial signals is a valuable goal, and this work makes a convincing step in that direction. The key strength of the paper, in my view, is the conceptual novelty of the sub-band approach. It's an elegant and intuitive way to handle the variable-length nature of signals sampled at different rates. This strong methodological contribution is backed by a very thorough empirical evaluation. The scale of the comparison, across the new RMIS benchmark and against 15 different encoders, lends significant weight to the authors' claims. Furthermore, the development of the RMIS benchmark itself stands out as a valuable contribution that will surely benefit the wider research community.

### Weaknesses
While the paper is promising, its clarity could be significantly improved in several areas, which currently hinders a full appreciation of the work. My primary concern is that the paper is not fully self-contained. For example, a key component of the training, the "mask cloning strategy," is mentioned without explanation, forcing readers to consult the external EAT paper to understand the methodology. Similarly, the crucial process of how signals yielding a variable number of sub-bands are actually batched together is left ambiguous. Beyond these clarity issues, some key design choices lack justification. It's unclear why the distillation loss is computed on unmasked patches, a deviation from common practice that warrants an explanation. The paper would also benefit from a deeper analysis of its own results. The strong performance of the BEATs model on anomaly detection is a fascinating finding, and the lack of discussion around why this might be the case is a missed opportunity for richer scientific insight.

### Questions
To help clarify some of the points above, I would appreciate your response to the following:
1. Could you please elaborate on the practical batching mechanism for inputs that have different sampling rates and thus produce a variable number of sub-bands?
2. Regarding the training process, what was the specific motivation for including the unmasked patches in the distillation loss, and what effect did you observe from this choice?
3. To make the paper more self-contained, would it be possible to provide a brief, high-level explanation of the "mask cloning strategy" and its specific role in your framework?
4. I would be very interested to hear your thoughts on the exceptional performance of the BEATs model in the anomaly detection tasks. Do you have a hypothesis for this result, and what might it suggest about the nature of those specific tasks?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposed the M5 problem, and presents FISHER, a foundation model that represents signals as STFT sub-bands, treats increments in sampling rate as concatenation of additional sub-band information, and uses a teacher–student self-supervised pretraining framework. To evaluate, the authors introduce RMIS benchmark spanning 19 datasets across four modalities. FISHER outperforms 15 state-of-the-art encoders by at least 3.23% on average, showing efficient scaling curves.

### Strengths
- Good problem formulation: clearly formalizing the M5 problem is a valuable contribution.

### Weaknesses
- Limited novelty: many modules in this paper can be found in references. For example, ViT, features emerging, and EMA are off-the-shelf modules. It’ll be great if the authors could explain more about what’s the unique contribution and novelty in this paper.

### Questions
- What’s the model sizes of baselines? In Appendix, authors described some but not all model sizes of baselines. To better understand the efficient scaling curves of FISHER, it'll be great if authors could display the number of parameters for all baselines. 
 - Should we train FISHER with pretrained weights instead of training from scratch? In AST (https://arxiv.org/abs/2104.01778), with the pretrained ViT weights, the performance of AST is further boosted.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper presents a foundation model for multi-modal signal representations including sound, vibration, voltage, current, temperature, and etc.

### Strengths
The paper introduces a foundation model for multi-modal signal representation. This is the first attempt to learn a multi-modal, multi-sampling-rate for various signal datasets.

### Weaknesses
My primary concern lies in the lack of novelty. I could not identify any clear novel contribution beyond the use of multi-modal training.

**[No Ablation Study]**
The paper does not include any ablation studies, making it difficult to understand why the proposed model outperforms others. It would be beneficial to provide ablation results for factors such as multi-modality, multi-scale datasets, model architecture, and sequence length.

**[Input Size]**
During training, FISHER uses a fixed 10-second segment, which limits the model’s usability. Could you clarify how the model handles longer input signals during evaluation? A 10-second limit seems too short for practical applications. 

I'm not an expert in this topic, specifically in industrial signal, so please lower the weight of my score accordingly.

### Questions
.

### Soundness
2

### Presentation
2

### Contribution
2
