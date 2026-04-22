# Neural Brain Fields: A NeRF-Inspired Approach for Generating Nonexistent EEG Electrodes

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 8, 2

## Abstract
Electroencephalography (EEG) data present unique modeling challenges because recordings vary in length, exhibit very low signal to noise ratios, differ significantly across participants, drift over time within sessions, and are rarely available in large and clean datasets. Consequently, developing deep learning methods that can effectively process EEG signals remains an open and important research problem. To tackle this problem, this work presents a new method inspired by Neural Radiance Fields (NeRF). In computer vision, NeRF techniques train a neural network to memorize the appearance of a 3D scene and then uses its learned parameters to render and edit the scene from any viewpoint. We draw an analogy between the discrete images captured from different viewpoints used to learn a continuous 3D scene in NeRF, and EEG electrodes positioned at different locations on the scalp, which are used to infer the underlying representation of continuous neural activity. Building on this connection, we show that a neural network can be trained on a single EEG sample in a NeRF style manner to produce a fixed size and informative weight vector that encodes the entire signal. Moreover, via this representation we can render the EEG signal at previously unseen time steps and spatial electrode positions. We demonstrate that this approach enables continuous visualization of brain activity at any desired resolution, including ultra high resolution, and reconstruction of raw EEG signals. Finally, our empirical analysis shows that this method can effectively simulate nonexistent electrodes data in EEG recordings, allowing the reconstructed signal to be fed into standard EEG processing networks to improve performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a novel approach for modeling EEG data by drawing inspiration from Neural Radiance Fields (NeRF). Recognizing the challenges of EEG, such as low signal-to-noise ratio, temporal drift, inter-subject variability, and limited dataset size, the authors propose treating EEG electrodes analogously to camera viewpoints in NeRF, enabling the learning of a continuous representation of neural activity. The proposed network is trained in a NeRF-style manner on a single EEG sample to produce a compact weight vector encoding the entire signal, which can then be used to reconstruct EEG signals at unseen spatial or temporal points. This framework allows for continuous, high-resolution reconstruction of raw EEG signals and accurate reconstruction of missing electrode data, enhancing the performance of downstream EEG task performance.

### Strengths
1. The proposed use of implicit neural representations for EEG signal generation and missing-channel reconstruction is both novel and conceptually inspiring.
2. The reconstruction result visualizations are impressive, and the proposed model achieving superior performance compared to the baseline methods.

### Weaknesses
Major concerns:
1. The proposed approach is more akin to an implicit neural representation (INR) than to a true Neural Radiance Field (NeRF), as it lacks core NeRF components such as volumetric rendering and per-ray sampling strategies that enable view-dependent reconstruction.

2. Insufficient experimental validation. The experimental evaluation is limited to electrode-missing scenarios and does not explore other important cases, such as missing time points, combined spatiotemporal missing patterns, or varying sparsity levels. As a result, the generality and robustness of the proposed approach remain unverified. Though the idea of this paper is very interesting, a comprehensive evaluation would further largely strenghten the contributions and support the authors’ claims more convincingly.

3. The framework is limited in scalability and practicality. Each EEG sample requires a dedicated per-recording network trained on short (3s) segments with progressive fine-tuning. This setup limits scalability to large cohort of EEG datasets. Probably providing the time for fully finetune for of the model would be helpful.


Minor concerns:
1. The paper should discuss potential limitations when reconstructing signals at boundary electrodes. Similar to NeRF, the proposed model may struggle to extrapolate accurately in regions with no nearby training electrodes, leading to unreliable reconstruction near the scalp edges.

2. The evaluation is conducted on relatively small datasets, which limits the assessment of the model’s scalability and generalization. Including experiments on larger or more heterogeneous EEG datasets would further strengthen the paper’s claims.

3. Some important experimental details are missing or not presented clearly in the main text. For instance, information such as the number of subjects, channel configurations, or which electrodes were masked during evaluation is either hard to find (some of them are in the result section) or only appears in the appendix. These details are essential for understanding and reproducing the experiments and should be summarized clearly in the experiment section.

### Questions
1. Although not strictly necessary and somewhat beyond the current scope, it would be interesting to include comparisons with more recent generative models as additional baselines to further strengthen the contribution of this work.
2. How does the model handle reconstruction at boundary electrodes, where spatial extrapolation beyond observed positions is required? Are there any differences in performance between internal vs. boundary electrodes when reconstructing missing channels?

### Soundness
3

### Presentation
2

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
Inspired by NeRF, this paper models EEG signals as sparse samples from a continuous neural field. Leveraging a MLP architecture, the model reconstructs EEG signals at arbitrary spatial and temporal locations, and serves as an effective data augmentation strategy to enhance downstream task performance.

### Strengths
- The transfer of NeRF’s implicit field modeling paradigm to neuroscience is novel.
- Through comprehensive experiments and visualization results, this paper convincingly shows the effectiveness of its modeling approach, which can generate virtual electrode data and improve accuracy on three downstream speech decoding datasets.

### Weaknesses
- Current models require training a separate model for each EEG sample, making them impractical for real-world use. More efficient and generalizable solutions are needed.
- The analogy between NeRF and EEG data is conceptually appealing but physically questionable. NeRF samples exhibit spatial continuity and illumination consistency, with multi-view observations providing strong constraints. In contrast, EEG does not represent different views of an implicit field but rather a complex superposition of signals from multiple brain regions via volume conduction. Moreover, EEG/MEG sampling is highly sparse, making it uncertain whether the source current distribution can be effectively learned. The method proposed in the paper appears to be a simple regression model that merely completes voltage values in space.

### Questions
- The authors state in both the Abstract and Introduction that the model can predict EEG signals at arbitrary time points(e.g., L028, L081). However, this capability is not demonstrated or evaluated in the experiments. Could the authors clarify or provide supporting evidence?
- In Section 5.1, please specify which electrodes are used as “additional electrodes” in the experiments. Providing this information would improve reproducibility and interpretability.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work introduces a novel technique that leverages Neural Radiance Fields (NERF) to enhance the recording of electroencephalogram (EEG) data. By applying this approach, the authors aim to augment the spatial resolution of EEG recordings by increasing the number of electrode positions, thereby capturing a more comprehensive representation of brain activity. Additionally, the technique seeks to improve temporal sensing, allowing for more accurate monitoring of rapid neural dynamics.

### Strengths
1. By simultaneously enhancing both spatial and temporal data dimensions, the proposed method exhibits significant potential for improving the accuracy and applicability of EEG studies. This advancement is particularly relevant to fields such as clinical diagnostics, where precise interpretations can lead to better patient outcomes, and neuroscience research, which relies on detailed brain activity monitoring to uncover fundamental neural mechanisms.

2. The study demonstrates notable merit, introducing NeRF as an innovative approach to enhance EEG data quality. This concept stands out not only for its novelty but also for its potential to transform existing analytical frameworks in EEG research and applications.

3. The presentation of the results is clear and well-structured, emphasizing the advantages of the NeRF-based approach. The findings demonstrate that this method outperforms a recently developed alternative utilizing Variational Autoencoders (VAE), highlighting its superiority in processing and interpreting complex EEG data. This comparison adds significant weight to the argument for adopting NeRF techniques in future EEG analysis.

### Weaknesses
There is no major weakness evident of this work.

### Questions
Nil

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes NBF, a NeRF-like model that learns EEG/MEG as a continuous function and renders virtual electrodes. The approach reduces interpolation error and shows accuracy gains for speech decoding and emotion recognition. The ablation results support the stated architectural choices. The main concerns are computational cost and the need for broader validation across datasets and tasks.

### Strengths
The approach reduces interpolation error relative to classical interpolators and yields small to moderate gains on downstream tasks such as speech decoding and emotion recognition. Ablations support the importance of positional encoding, normalization choices, skip connections, and progressive initialization. The idea is interesting and potentially impactful.

### Weaknesses
The study has potential leakage risks due to per-subject training on sequential 3-second windows with progressive fine-tuning, and it is unclear if normalization, statistics, and hyperparameter search were confined strictly to training electrodes/windows in each split. Baseline coverage is limited, since there is no direct comparison to learning-based virtual-electrode super-resolution methods such as CNN upsampling or GAN approaches that the paper discusses. Electrode geometry and referencing are insufficiently specified, including whether 3-D digitizations or template montages are used, how references are set, and how montage alignment differs across datasets, all of which can materially influence interpolation difficulty and reported metrics.

### Questions
- Please specify the spatial referencing schemes used (e.g., average reference, linked mastoids, REST, etc) and analyze how each choice might bias the learned continuous field or interact with the positional encoding used by NBF. Discuss whether NBF can be trained or adapted in a reference-agnostic manner.

- Elaborate the normalization procedures. For z-score, clarify the axis and scope (per channel per trial, per subject, global) and whether statistics are computed on training data only. For min-max, state the range, drift handling, and leakage prevention across splits. 

- Discuss cross-subject generalizability. Can NBF align subjects with different montages or sampling grids. Describe any coordinate system, head model, or registration procedure used, and evaluate performance when training on one montage and testing on another.

- The related work and comparisons should include CNN-based virtual electrodes (e.g., Svantesson et al.) and GAN-based super-resolution methods that are already discussed in the text. Please add quantitative comparisons or justify their omission with clear constraints.


Minor corrections:
“NeRFMildenhall et al. (2021)” → “NeRF (Mildenhall et al., 2021)”
Remove the duplicate “EEGNet 2018a/2018b”
“(EasyChair, 2023")” → “(Torma & Szegletes, 2023)”

### Soundness
2

### Presentation
2

### Contribution
2
