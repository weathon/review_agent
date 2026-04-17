# OmniMouse: Scaling properties of multi-modal, multi-task Brain Models on 150B Neural Tokens

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Scaling data and artificial neural networks has transformed AI, driving breakthroughs in language and vision. Whether similar principles apply to modeling brain activity remains unclear. Here we leveraged a dataset of 3.1 million neurons from the visual cortex of 73 mice across 323 sessions, totaling more than 150 billion neural tokens recorded during natural movies, images and parametric stimuli, and behavior. We train multi-modal, multi-task models that support three regimes flexibly at test time: neural prediction, behavioral decoding, neural forecasting, or any combination of the three. OmniMouse achieves state-of-the-art performance, outperforming specialized baselines across nearly all evaluation regimes. We find that performance scales reliably with more data, but gains from increasing model size saturate. This inverts the standard AI scaling story: in language and computer vision, massive datasets make parameter scaling the primary driver of progress, whereas in brain modeling -- even in the mouse visual cortex, a relatively simple system -- models remain data-limited despite vast recordings. The observation of systematic scaling raises the possibility of phase transitions in neural modeling, where larger and richer datasets might unlock qualitatively new capabilities, paralleling the emergent properties seen in large language models. Code available at \url{https://github.com/enigma-brain/omnimouse}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents a model architecture and training methodology and uses it to analyze data and model scaling trends for neuroscience data. It discovers positive trend in data scaling, and a limited scaling on the model side (suggesting the need for more data). The overall message is an important contribution. My main concerns are with one of the baseline used in evaluation, and some concerns about phrasing.

### Strengths
- Analysis of both data scaling and model scaling trends is very valuable for the community. One of the first papers to do that.
- Evaluation strategy is well designed, and a good number of tasks are included.

### Weaknesses
I like the overall message of this paper, but I think the problems mentioned below need to be addressed before I can recommend acceptance.

**Main remaks**
- I would prefer the reasoning for upsampling the data was put in the main text (I see it is in the appendix right now.)
- (Opinionated) Just using number of neurons to compare dataset sizes doesn't seem like a good idea. This is reference to the line: "We used a dataset of over 3 million single-unit neuronal recordings – an order of magnitude larger than...". After all, I could have 1 second long recordings of 10 million neurons, and that dataset would not be considered big. To me, it seems a metric like "neuron-hours" would be better. i.e. including both the number of neurons and the recording durations to indicate data size.
- Section 3, Data utilization: "A key novelty is our ability to sample...". Can you explain how this is novel in context to existing work? As far as I know, POYO and POYO+ have used this kind of arbitrary continuous sampling, and it has been a part of the open-source [`torch_brain`](https://github.com/neuro-galaxy/torch_brain) package for quite some time.
- My biggest issue: When comparing the performance of behavior decoding, why not compare with something like POYO (individual behaviors), or POYO+ (multi-behavior)? These models (and many other recent methods) have proven to be much better than the CEBRA baseline. Comparing with something that is not a leading method for behavior decoding diminishes the results. Including stronger baselines would help convince the reader that the suggested expensive large-scale semi-supervised training does indeed lead to better performance, better than simple supervised learning using strong methods. I understand the paper is more focused on scaling, but it is important having a better reference of where purely supervised methods are in comparison.

**Nits**
- Fig 1. Missing y-axis labels
- Some places, such as paragraph 2 of Introduction should have em-dashes `---` instead of en-dashes
- Line 071 - comma after task - "single modality, task, or dataset"
- Typo: Fig 3 caption - "tookens"
- Line 249 - "shared linear" -> "shared linear layer"

### Questions
**Questions**
- It is unclear whether the dataset used for pretraining is public or private.
- Can you please expand on "restart training from intermediate checkpoints every 20k steps"? (From section 4, Training paragraph) Which intermediate checkpoints do you restart from? Why restart? Why not just continue with a step change in LR?
- In section 5, Forecasting: Why "40 frames of behavior" were used to condition the forecasting? Why not 30?

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
This paper studies whether the scaling laws in AI also apply to modeling neural activity. Using a dataset with 3.3 million neurons from the mouse visual cortex (78 mice, 323 sessions, and over 150 billion neural tokens), the authors trained multi-modal, multi-task transformer models ranging from 1M to 300M parameters. These models can perform neural prediction, behavioral decoding, and neural forecasting. Empirically, the authors found that performance scales with data quantity but saturates with increasing model size, indicating that current modeling efforts in neuroscience are data-limited.

### Strengths
This paper makes a timely contribution by systematically studying scaling laws in neurofoundation models using an unprecedentedly large dataset. The work is novel in introducing the first large-scale, multi-modal, multi-task transformer that unifies neural encoding, decoding, and forecasting with naturalistic video inputs. The paper is also well written and organized.

### Weaknesses
1. The paper is impressive in scale, but several aspects could be improved. The contribution is empirical rather than methodological. The proposed transformer largely builds on existing model designs (e.g., POYO+ and prior multi-modal fusion techniques), with limited architectural innovation beyond scaling and integration. Clarifying which components are novel vs. adapted from prior work would help better explain the contribution.

2. The paper would benefit from more qualitative analyses or visualizations to show what the model has learned.

3. Although the inclusion of naturalistic video inputs is interesting, the current analyses do not disentangle how much information comes from visual vs. behaviors (e.g., pupil location, size, running speed).

### Questions
1. The single-trial correlation metrics in Fig. 5 are quite low. Could the authors clarify the reason for this? For example, is it due to the use of naturalistic video stimuli rather than repeated trials? Also, why was correlation chosen as the primary evaluation metric, given that it only captures linear relationships and does not account for nonlinear dependencies or variance structure in the data?

2. The paper claims that scaling in neuroscience is limited by data rather than model size, unlike in AI. In Fig. 5, it seems that as data increases, model size should also grow; otherwise, performance may saturate. This implies that model scaling is still necessary when handling larger datasets. In AI, there is already abundant data, which may make increasing model size particularly helpful. I am curious if the authors have considered this factor.

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
3

### Summary
This paper presents OmniMouse, a multi-modal, multi-task transformer model trained to predict activity in the mouse visual cortex. The model is trained on a new, massive-scale dataset of 3.3 million neurons from 323 recording sessions, totaling over 150 billion neural tokens.

The primary contribution is a systematic study of scaling laws, training models from 1M to 300M parameters. The authors find that performance gains saturate with increasing model size but scale reliably and consistently with increasing dataset size. This leads to the central conclusion that current models of the visual cortex are data-limited, not parameter-limited, a finding that inverts the standard scaling narrative in mainstream AI. The largest model achieves new state-of-the-art performance on several predictive tasks, including neural forecasting, stimulus-driven prediction, and behavioral decoding.

### Strengths
1. **Scaling study for brain foundation model**: To my knowledge, this is the first work to systematically apply the scaling laws methodology to a single-neuron, multi-modal brain foundation model of this magnitude. The primary finding is that the model performance is data-limited rather than parameter-limited, which provides a clear directive for future progress, emphasizing the need for larger and more diverse datasets.
2. **Dataset and model contribution:** The assembly of a 150B+ token dataset from 3.3M neurons is a major contribution in itself. Furthermore, OmniMouse-300M outperforms strong, specialized baselines (e.g., IBL, CEBRA) across the Sensorium 2023 competition even with its core weights frozen (training only neuron/animal embeddings).
3. **Flexible and generalizable prediction:** The model's multi-task design, based on flexible masking of modalities, is effective. The paper shows that this design allows the model to learn representations that generalize to contextual variations not seen during training.

### Weaknesses
1. **Datasets scale misalignment in baseline comparison:**  The OmniMouse-300M model was trained on the full 323-session dataset, whereas all baselines were trained only on the smallest 8-mice data collection to reduce computational cost. It is unclear how much of OmniMouse's SOTA performance is due to its superior architecture versus simply having access to ~40x more data than the baselines it's compared against.
2. **Generalization bottleneck on per-neuron identity embeddings:** The use of per-neuron (and per-session/animal) identity embeddings limits the generalization of the model. As shown in Table 1, these neuronal parameters account for the vast majority of the model's total parameters (e.g., 779M $p_N$ vs. 348M $p_M$ for the "300M" model). The model cannot perform zero-shot prediction on a new, unseen neuron or animal; it must be fine-tuned. A true foundation model should learn a general representation of a neuron rather than memorizing 3.3 million specific instances.

### Questions
1. To provide a fair architectural comparison, what is the performance of a smaller OmniMouse model (e.g., OmniMouse-80M) when trained only on the 8-mouse dataset? How does that model compare to the baselines (IBL, CEBRA, etc.) trained on the same 8-mouse dataset? This would isolate the architectural contribution from the data-scaling contribution.
2. Have you explored neuron-agnostic tokenization strategies? For example, could a neuron be represented by an embedding derived from its anatomical coordinates, its functional tuning properties (e.g., a pre-computed receptive field), or its relative position to other neurons, rather than a unique learned ID?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors present a scaling analysis using a large neural model. The model is trained on multi modal data from head-fixed mice in a VR task consisting of visual stimulus, locomotion speed and pupil features recording. The neural data is calcium imaging confined to the visual cortex, for an overall tally of 3M neurons on 78 subjects. The model is based previous work POYO, with an additional hierarchical vision encoder to input the visual stimulus data. The authors present learning curves as a function of model size (measured in parameters) and show that performance saturates with model size, but does not saturate with input dataset size. The model performance is evaluated on a 2023 decoding benchmark with scores beating the baselines in all categories.

### Strengths
The scale analysis is a welcome contribution to the field.
- The discussion correctly assesses limitations of the current approach, namely the narrow brain region and behavior protocol.
- Despite the narrow brain region and behaviour type studied, the authors manage to show that their model is still data limited, which makes the claim that dataset size is a limiting factor even more compelling.

### Weaknesses
- The performance is compared to baselines that may not be state of the art models. For example in behaviour decoding tasks, to justify the claim of SOTA we may be more interested in others large transformer based ANNs results than in the contrastive learning CEBRA method
- No clear statement about sharing of code, models and private data for others to address the point above or even reproduce the proposed results, while the authors have made heavy use of public resources.


Minor comments:
- figure 1bc: label y-axis missing (loss)
- Code / model / data availability statement

### Questions
- What reasonable steps could be taken to reproduce those results from a third-party ?

### Soundness
2

### Presentation
3

### Contribution
3
