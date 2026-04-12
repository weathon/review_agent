## Human Reviewer 1

### Summary
This paper presents SSLAM, an SSL framework for audio that incorporates audio mixtures to better adapt to polyphonic environments. SSLAM leverages a two-stage training process: the first stage uses single-source audio, while the second incorporates partially mixed audio to simulate polyphonic scenarios. The paper introduces Source Retention Loss to preserve the distinct characteristics of each audio source in a mixture, making the model more robust in complex, multi-source audio settings. The authors evaluate SSLAM on a mix of monophonic and polyphonic datasets, reporting strong improvements over prior methods.

### Strengths
The paper addresses a relevant challenge in audio SSL, especially as polyphonic environments are common in real-world audio. The two-stage training approach combined with the Source Retention Loss appears effective, allowing the model to learn from mixed audio in a way that preserves source integrity. The evaluations include a range of polyphonic datasets, and the method generally performs well across them. The component-wise analysis of the model’s objectives is also informative, showing the impact of each training objective.

### Weaknesses
The paper does not provide an analysis of SSLAM’s generalization to unseen audio domains, such as speech or music distribution. It would be great to also observe if this technique is applicable to other domains’ downstream applications such as multi-speaker recognition or instrument recognition, where audio types could differ significantly from pre-training data.

### Questions
Since AudioSet is not fully monophonic, what is the intuition behind the mixing technique’s consistent performance improvement in polyphonic tasks? For example, in URBAN-SED, this improvement is not always observed, so further insights into why mixing boosts performance in some cases but not others would be helpful.

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper introduces **SSLAM (Self-Supervised Learning from Audio Mixtures)**, a novel self-supervised pre-training strategy for enhancing the performance of audio models on polyphonic soundscapes. Recognizing the limitation of current audio SSL methods, which are often benchmarked on predominantly monophonic datasets, SSLAM incorporates audio mixtures into the pre-training process. The approach utilizes a masked latent bootstrapping framework where a student model is trained on mixtures of audio spectrograms, created via an element-wise maximum operation inspired by the Ideal Binary Mask. Concurrently, a teacher model processes the individual audio sources separately, and its aggregated feature representations serve as targets for the student. A novel source retention loss is introduced to further encourage the model to learn and retain distinct features of each source within the mixture. Experiments on standard audio SSL benchmark datasets (AS-2M, AS-20K, ESC-50, KS1, KS2) demonstrate that SSLAM not only improves performance on polyphonic audio (achieving up to a 9.1% improvement on SPASS and setting a new SOTA mAP of 50.2 on AudioSet-2M) but also maintains or exceeds performance on monophonic datasets. Ablations study the contribution of individual components and the impact of mixing strategies and loss functions.

### Strengths
1. The paper tackles the under-explored issue of polyphonic sound processing in self-supervised audio learning. This is crucial because real-world audio scenes rarely consist of isolated sounds, and models trained primarily on monophonic data may struggle to generalize effectively in realistic scenarios.

2. SSLAM introduces a novel training strategy by incorporating audio mixtures and a source retention loss, both well-motivated by principles of auditory scene analysis (specifically, the Ideal Binary Mask concept). The partial mixing strategy further demonstrates a nuanced understanding of the balance between introducing new information and preserving existing audio characteristics.

3. The paper presents convincing empirical results, demonstrating state-of-the-art performance on both polyphonic datasets and standard (largely monophonic) audio SSL benchmarks. The ablation studies and analysis of mixing and aggregation strategies provide further insights into the effectiveness and robustness of the proposed method.

### Weaknesses
1. The core contribution of SSLAM, training with mixtures of mixtures, closely resembles the MixIT [A] approach for unsupervised sound separation. The novelty seemingly lies in applying this concept within a self-supervised representation learning framework. However, the paper does not sufficiently justify why this adaptation is novel or contributes significant new insights beyond the well-established principles of mixture invariant training.

2. A major weakness is the absence of a comparison with a straightforward baseline: pre-processing the mixed datasets with an unsupervised sound separation model like MixIT [A] to obtain separated sources, and then applying the self-distillation method. This would directly address the polyphony challenge and provide a more meaningful assessment of the value added by the proposed synthetic mixing strategy.

3. The reliance on a pre-trained model (here, an EMA teacher) trained on potentially millions of labeled audio clips undermines the claim of a fully self-supervised approach. While the student model is trained without labels, the teacher network embodies prior knowledge acquired through supervision. The paper should clearly acknowledge this dependency and avoid potentially misleading terminology. Please revise the manuscript accordingly to reflect that.

4. Figure 4, intended to visualize the mixing process, is of extremely low quality and uses an unconventional orientation (high frequencies displayed at the bottom). This hinders understanding and should be replaced with a lossless image with a correctly oriented frequency axis (low frequencies at the bottom).

5. Despite suggesting broader applicability, the evaluation primarily focuses on sound event detection. While performance gains are demonstrated on polyphonic SED datasets, the paper does not explore other downstream tasks like speech recognition or music analysis where polyphony is also prevalent. This limits the evidence for the generalizability of the learned representations.

[A] Wisdom, S., Tzinis, E., Erdogan, H., Weiss, R., Wilson, K. and Hershey, J., 2020. Unsupervised sound separation using mixture invariant training. Advances in neural information processing systems, 33, pp.3846-3857.

### Questions
The paper uses the term "self-supervised," yet the approach relies on a pre-trained teacher model that presumably required supervision. How many labeled audio clips were used to train the teacher model? Can the authors clarify the extent to which their method truly qualifies as self-supervised, given this dependency on a pre-trained, supervised component?  Would "teacher-student training" or a similar term be more accurate?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper proposes SSLAM, an approach to integrate audio mixing and self-supervised learning, an approach that stems from the fundamental concept that real-world audio data is, generally, polyphonic. The method claims that training on audio mixtures in masked latent bootstrapping framework improves performance for both monophonic and polyphonic soundscapes.

### Strengths
1. Clarity: paper is well written and apart from some sections, quite easy to read.
2. Soundness seems sufficient.

### Weaknesses
1. Small set of evaluated tasks: not enough diversity in evaluated downstream tasks. Only speech (keyword spotting) and in-domain audio classification tasks are evaluated. Instead of evaluating KS2 and KS1, either one would've sufficed. Dataset choice is also not motivated well enough.
2. Evaluation itself: no mean/std or confidence intervals are reported, which would be even more useful for polyphonic evaluations. Are the downstream results reported from a single test run? 
3. The overall objective function in the final SSLAM model is a composite of 5 losses. However, the losses don't seem to be motivated well enough. For instance, the global loss for the mixed version uses only the final layer outputs, which is not the case for your baseline.

### Questions
1. Why does SSLAM require both unmixed and mixed versions of the objectives?
2. Why is Table 1 missing the other variants you have evaluated: MB-UA (which I suppose corresponds to the "baseline" you talk about in Section 3.1), MB-PMA, MB-UA-PMA?
3. In table 2, MB-PMA works the best for IDMT DESED in linear evaluation and on SPASS-Market when fine-tuning, but it's not the one highlighted.


POST REBUTTAL
--------------------

Most of my questions have been addressed, so I am raising my recommendation to 6.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
In this paper, the authors propose a new self-supervised learning strategy for polyphonic soundscapes, called SSLAM. Unlike traditional masked latent bootstrapping, SSLAM leverages audio mixtures to enhance the self-supervised learning process at both global and local levels. The method demonstrates performance improvements on standard audio self-supervised learning benchmarks, as well as on polyphonic datasets.

### Strengths
1. The paper is generally well-written, with clear explanations and logical flow.
2. The paper addresses a fundamental challenge in audio classification—handling the polyphonic nature of real-world audio environments—and proposes a solution based on this insight.
3. SSLAM demonstrates competitive performance on standard audio classification benchmarks.

### Weaknesses
1. **Confusing Figure Presentation**: In Figure 1(C), it appears that there are two teacher models, but the description in the text suggests that there is only one teacher model handling two audio inputs separately, which creates some confusion. Additionally, Figure 2 shows layer averaging after the teacher model, whereas in Section 3.1.2, it is stated that only the last layer or all 12 layers are used for global or local loss, which seems inconsistent.
2. **Writing Errors**: There are some citation format issues in Tables 1, 7, and 8, as well as in several paragraphs. Additionally, a numerical error was found in Table 1, where the A-JEPA score for ESC-50 is listed as 06.3 instead of the correct value.

### Questions
1. **Audio Mixing**: How does your method compare to commonly used data augmentation techniques, such as Mixup? In Table 4, the performance improvement of mixed audio on AS-20K appears limited, and the baseline results with unmixed audio in stage 1 are already equal or stronger than those of prior works, as shown in Table 1. Could you clarify the specific advantages of your approach in this context?
2. **Layer Averaging**: In section 3.1.2, you mention that averaging over all 12 layers, followed by spatial pooling, could result in excessive information compression. This is why you chose to use only the output of the last layer for the global loss and all 12 layers for the local loss. Could you elaborate on how performance is affected when averaging all 12 layers, as is done in the baseline?
3. **Necessity of Unmixed Objective**: In Tables 2 and 3, the MB-UA-PMA configuration does not show significant improvements over MB-PMA. What would be the impact of using only mixed objectives and SRL? Is it necessary to include the unmixed objectives in this setup?
4. **Polyphonic Audio Datasets**: The datasets you selected for polyphonic scenario evaluation are primarily used for sound event detection, yet you treat them as multi-label classification tasks. Have you considered evaluating performance on sound event detection tasks with time-stamped data? For instance, how would the model perform on SED tasks if an RNN were added after the current model?

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
5