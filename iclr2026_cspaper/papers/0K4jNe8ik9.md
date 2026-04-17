000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

## Abstract

As the global population ages and dementia cases rise, there is an urgent need for effective early diagnosis and monitoring of neurodegenerative diseases. Electroencephalogram (EEG)-based technologies are increasingly important due to their portability, affordability, and suitability for widespread screening compared to other neuroimaging methods. However, EEG signals present challenges such as low signal-to-noise ratio, high inter-subject variability, and limited labeled data, especially in elderly or dementia patients, which restricts the effectiveness of traditional supervised learning approaches. Leveraging the neurophysiological significance of the five EEG frequency bands (delta, theta, alpha, beta, gamma), this study introduces an innovative multi-head Simple Framework for Contrastive Learning of Visual Representations (SimCLR) architecture. The proposed Delta2Gamma (DGNet) model combines frequency-band specific representation learning, enabling more precise detection of subtle EEG changes linked to brain disorders like dementia. Our self-supervised learning (SSL) adaptive multiband heads model achieved a 31.5% relative performance improvement over training from scratch, and a 25.4% improvement over the single-head approach. To the best of our knowledge, our proposed method achieved state-of-the-art performance in multi-head approaches. The source code is available at GitHub by https://anonymous.4open.science/r/iclr2026-7FE2.

## 1 Introduction

At the turn of the 21st century, humanity is facing one of the greatest challenges in the history of public health. It is a complex crisis caused by the collision of two massive trends: the global aging of the population at an unprecedented rate and its inevitable corollary (Nichols et al., 2022), the explosion in the prevalence of dementia (Li et al., 2022). These two phenomena are more than just demographic shifts or individual diseases; they are a tsunami that is shaking the very foundations of socioeconomic structures and healthcare systems around the world. A clear understanding of the scale and urgency of the problem is a prerequisite for finding innovative solutions (Olivari et al., 2020). Traditional diagnostic methods for dementia, such as magnetic resonance imaging (MRI) and positron emission tomography (PET), have limitations in early detection due to their high costs and limited accessibility (Haidar et al., 2023; Juganavar et al., 2023). Additionally, these tests can only be performed at medical institutions operated by specialized healthcare professionals, making it difficult for the general public to regularly monitor their cognitive health in daily life (Kancharla, 2024). Early detection of mild cognitive impairment (MCI), the precursor stage of dementia, is particularly challenging (Sabbagh et al., 2020), and as a result, most patients are only diagnosed after their symptoms have significantly progressed (Cavedoni et al., 2020).

Despite the vast scale and severity of the dementia crisis, current clinical responses particularly in the diagnostic process are fundamentally inadequate to address the issue. The existing diagnostic paradigm relies on technologies that are not scalable, leading to a serious "diagnostic bottleneck" worldwide. Precisely analyzing the causes of this bottleneck is essential for setting the right course toward a solution. These technologies have critical limitations that prevent them from serving as Anonymous authors Paper under double-blind review

# Dgnet: Self-Supervised Delta2Gamma Multi- Band Eeg Representation Learning For De- Mentia Classification

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107

## 2 Proposed Method

frontline screening tools for large populations. As explicitly noted in the research literature, MRI and PET are fundamentally constrained by extremely high costs and limited accessibility. Because they require large, stationary, and expensive equipment, as well as highly specialized facilities and personnel, their benefits are inevitably restricted to only a small fraction of the population. As an alternative to overcoming the current diagnostic bottleneck, electroencephalography (EEG) offers a promising solution that combines scientific validity with practical advantages. EEG is a non-invasive method that can directly and quantitatively measure declines in brain function (Dabbabi et al., 2023), and it has the potential to overcome the limitations of existing technologies and establish a scalable diagnostic paradigm. This represents a fundamental distinction from functional magnetic resonance imaging (fMRI), which measures blood flow changes that serve as indirect proxies for neural activity (Yen et al., 2023). By directly capturing electrical signals generated by neurons, EEG most directly reflects how the neuropathological changes defining dementia impact brain function (Smailovic & Jelic, 2019). Consequently, EEG analysis can provide critical insights for understanding and diagnosing the underlying neurophysiological basis of cognitive decline (Yuan & Zhao, 2025). A substantial body of prior research has identified characteristic and quantifiable "spectral signatures" associated with Alzheimer's disease (AD) and other forms of dementia, as also supported by the technical analysis provided. The core of this signature can be summarized as an overall
"slowing" of brain oscillations. The specific biomarkers are as follows:
- **Increased power in low-frequency bands**: Patients with AD show a statistically significant increase in the power spectral density (PSD) of low-frequency bands, such as delta (δ, 0-4 Hz) and theta (θ, 4-8 Hz), compared to healthy controls (Moretti et al., 2004). Notably, the literature (Baik et al., 2022) states that profuse theta waves with age depict abnormal activity, emphasizing that excessive theta activity is indicative of pathological conditions.

- **Decreased power in high-frequency bands**: This slowing of brain activity is accompanied by a reduction in power in the higher-frequency bands (Benwell et al., 2020). This includes a decrease in alpha (α, 8-12 Hz) waves, which are associated with relaxed wakefulness, and beta (β, 12-30 Hz) waves, which are linked to active concentration. Most notably, there is a marked reduction in gamma (γ, 30∼ Hz) waves, which are essential for higher cognitive functions such as short-term memory (Kaminski et al., 2011). The literature explicitly ´ states, gamma waves decline with cognitive deterioration, clearly indicating that changes in gamma activity are a key marker of cognitive decline (Traikapi & Konstantinou, 2021).

This research perfectly aligns with the future direction of dementia research and treatment, which aims to integrate artificial intelligence, digital health tools, and personalized medicine. The development of home-based cognitive function tests or personalized assistive technologies demonstrates a clear trend toward shifting the focus of care from hospitals to patients. The EEG system proposed in this study will serve as a foundational technology for building this future medical ecosystem. The main contributions of this paper are summarized as follows:
1. **Frequency-band specific Encoding**: We propose Delta2Gamma (DGNet) architecture that decomposes EEG signals into five standard frequency bands (δ, θ, α, β, and γ) for processing, enabling the extraction of frequency-band specific representations.

2. **Multi-Band Head**: Each frequency band is processed by an independent CNN encoder and projection head, thereby preserving neural information unique to each band.

3. **Effective for dementia classification**: We evaluate a self-supervised learning model specifically tailored to the neurophysiological characteristics of EEG signals in the dementia classification task.

The proposed framework is illustrated in Figure 1. The overall training process consists of two stages: pre-training and linear evaluation. We perform contrastive learning on unlabeled EEG data. During this process, the model learns the general characteristics and patterns of EEG signals, and data augmentation enables it to acquire robust feature representations against various transformations. The following sections discuss these components in detail.

![2_image_0.png](2_image_0.png)

![2_image_1.png](2_image_1.png)

Figure 1: Overview of the entire learning process of our proposed Delta2Gamma (DGNet) model. (a) Pre-training phase based on self-supervised learning: Data augmentation is applied to the unlabeled EEG data. Then, using adaptive Normalized Temperature-scaled cross entropy (NT-Xent) contrastive loss with regularization, the encoder is trained to learn meaningful feature representations across multiple frequency bands (*δ, θ, α, β, γ*). (b) Linear evaluation phase: A classifier is added on top of the pre-trained encoder. The pre-trained encoder is frozen during this stage. Then, the entire model is intentionally retrained using labeled data for specific tasks such as classifying Alzheimer's disease (AD) and cognitively normal (CN) groups. The numbers on the linear layers indicate the dimensionality of each layer.

## 2.1 Architectures

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Self-Supervised learning (SSL) Simple Framework for Contrastive Learning of Visual Representations (SimCLR) is a self-supervised learning method developed by the Google Brain team (Chen et al., 2020). It offers an innovative approach for learning useful feature representations from large amounts of unlabeled data. The core idea of this method is contrastive learning, where different augmented versions of the same sample are encouraged to be close to each other in the feature space, while samples from different instances are pushed apart. Our implementation of SimCLR, as shown in Figure 1(a), is specifically tailored for EEG signals. This advanced adaptation applies the original image-based SimCLR framework to neural signal analysis. The model simultaneously processes multiple frequency bands of EEG signals (δ, θ, α, β, γ) and employs independent projection heads for each frequency band, enabling more fine-grained feature learning. This SimCLR-based self-supervised learning approach is an innovative methodology that overcomes the limitations of traditional supervised learning. In the medical field, for example, applications such as dementia diagnosis using EEG signals often face challenges in acquiring large amounts of labeled data, whereas unlabeled EEG data can be collected relatively easily. In such scenarios, SimCLR enables the learning of rich feature representations from unlabeled data, allowing high-performing classification models to be built even with only a small amount of labeled data. Multi-Band Our most innovative aspect lies in the processing of multiple frequency bands. EEG signals are composed of several frequency bands, each reflecting different brain activities and carrying unique neurological significance. We decompose the original signal into five frequency bands
(δ: 0.5-4 Hz, θ: 4-8 Hz, α: 8-13 Hz, β: 13-30 Hz, γ: 30-45 Hz) and perform independent feature extraction for each band. This approach separates EEG signals into five major frequency bands (δ, θ, α, β, γ) and extracts unique features from each band, as illustrated in Figure 2. The first stage involves a frequency band extractor that separates features corresponding to each frequency band from the original multi162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 channel EEG signal with shape [C, L], where C is the number of channels and L is the signal length. In our experiment, we set the value of C to 19, and L was defined as the number of seconds multiplied by the sampling rate of 500 Hz. This module consists of five parallel 1-dimensional depthwise convolution layers. Each convolution layer is configured with a kernel size of 7 and padding of 3, preserving the input signal length L, followed by batch normalization (Ioffe & Szegedy, 2015) and Rectified Linear Units (ReLU) activation (Agarap, 2018). With groups set to C, each input channel is processed independently by its own 1-dimensional convolution filter, minimizing inter-channel information mixing and allowing effective learning of the unique temporal patterns of each channel (Lawhern et al., 2018). The input to the model is a raw EEG signal with shape [*C, L*]. First, the signal is decomposed into five canonical frequency bands using bandpass filters. Each filtered signal retains the original shape, resulting in five parallel representations [5*, C, L*]. Each band is then independently processed by a 1-dimensional convolution encoder with reduced depth and downsampling operations, designed to efficiently extract temporal and spatial features while minimizing computational overhead. For each frequency band, the encoder consists of three convolutional blocks with increasing channel dimensions (32 → 64 → 128), interleaved with batch normalization, ReLU activation, and max pooling layers. To summarize temporal dynamics, a global average pooling layer is applied to each frequency-band specific representation. The output after passing through the encoder is [5*, C, L/*32], and the pooled outputs from all five bands are then utilized. The pooled outputs from all five bands are concatenated and passed through a feature fusion comprising a fully connected layer, batch normalization, ReLU activation. This results in a compact [5, 128-dimensional] embedding that captures multi-band EEG characteristics.

![3_image_1.png](3_image_1.png)

The projection head receives each of the 5×(128-dimensional) feature vectors output by the encoders. This design produces 5×(128-dimensional) feature vectors, one for each frequency band. These vectors encapsulate rich and diverse information from the multi-channel EEG signals and can be effectively used as input for further analysis or diagnostic models. The spectrogram visualization of embeddings from the encoder, which includes the pretrained feature extractor, across the five EEG bands is shown in Figure 3.

![3_image_0.png](3_image_0.png)

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 Downstream Task In this study, the classification model utilizes a pre-trained encoder on multi frequency band representations. Two approaches were considered for training: In the first approach, the encoder's parameters are kept frozen, and only the newly added classifier is trained. This process is illustrated in Figure 1b. This allows us to assess the quality of the features extracted by the pretrained encoder. In the second approach, known as linear evaluation, all parameters of the model including those of the encoder are updated during training to adapt to the specific classification task. The classifier, which is attached to the final layer of the pre-trained encoder, is a newly constructed neural network with a MLP architecture. It takes the feature vector output by the encoder as input and predicts the final class. The classifier consists of three linear layers: the first hidden layer contains 512 nodes, and the second hidden layer contains 256 nodes. After each hidden layer, a ReLU activation function (Agarap, 2018) is applied to introduce non-linearity, and batch normalization is used to stabilize the training process. To prevent overfitting and improve the generalization performance of the model, dropout (Srivastava et al., 2014) is applied to each hidden layer at rates of 0.3 and 0.2, respectively. The final output layer produces logits with a dimensionality equal to the number of target classes.

## 2.2 Data Augmentation

The core of self-supervised learning lies in extracting meaningful signals from the data itself through pretext tasks. In line with this philosophy, we apply various transformations to the original EEG signals to generate two views that are semantically identical but morphologically different. It generates two views of the same original signal that are semantically identical but morphologically different by applying various transformations. Specifically, techniques such as Gaussian noise addition, amplitude scaling, time domain masking, frequency domain masking, and channel dropout are used to create diverse perspectives from the same EEG signal, as illustrated in Figure 4. For each case, we apply the following augmentations: Gaussian noise with a standard deviation of 0.03, amplitude scaling by a random factor between 0.8 and 1.2, and random masking of 10% in both the time and frequency domains. Additionally, with a probability of 10%, we perform channel dropout on 10% of the total channels.

![4_image_0.png](4_image_0.png)

These augmented views are then used in an instance discrimination task. In this setup, two augmented views derived from the same original signal should have similar representations in the feature space (positive pair), while views derived from different original signals should have distinct representations (negative pairs). Through this contrastive learning process, the model learns to capture the essential characteristics of the signal and acquire feature representations that are robust to noise and transformations.

## 2.3 Training Objective

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 This section describes losses for training, including, adaptive Normalized Temperature-scaled cross entropy (NT-Xent) and regularization (Wang et al., 2024). The final loss is defined as ℓ =PB
b=1 ℓb, where ℓb denotes the loss of the b-th frequency band head as defined, B refers to the number of frequency band heads.

$$\ell_{i}=\sum_{b=1}^{B}\left(-\frac{1}{\tau_{i}^{(b)+}}\sin(\mathbf{z}_{i}^{(b)},\mathbf{z}_{i}^{(b)+})+\frac{1}{\tau_{(i,n^{+})}^{(b)-}}\max_{n=1,\cdots,N}\sin(\mathbf{z}_{i}^{(b)},\mathbf{z}_{(i,n)}^{(b)-})+\beta\,\Omega(\tau_{i}^{(b)+})-\beta\,\Omega(\tau_{(i,n^{+})}^{(b)-})\right)\tag{1}$$

In the above equation, the function sim(·, ·) denotes cosine similarity, which can correspond to either a positive pair (z
(b)
i, z
(b)+
i) or a negative pair (z
(b)
i, z
(b)− (i,n)
), n denotes the individual indices of negative samples (1, 2*, ..., N*), and N is the total number of negative samples for the i-th anchor. The τ
(b)+
iand τ
(b)−
(i,n)represent learnable adaptive positive and negative temperatures, respectively; the asterisk "∗" in τ b−
(i,n∗)
indicates the index n
∗ = arg max n=1,··· ,N
sim(z
(b)
i, z
(b)− (i,n)
). β ≥ 0 controls the temperature regularization imposed by Ω(·). We describe the Ω(·) function in more detail in section Regularization. NT-Xent Normalized Temperature-scaled cross entropy loss function, which is the core of Sim- CLR, provides a mathematical basis for contrastive learning. In the attached code, the multi-head implementation computes independent NT-Xent losses for each frequency band and combines them through a weighted average to obtain the final loss. Additionally, an adaptive temperature mechanism introduces a normalization term that accounts for the learning difficulty of each band, enabling more stable and effective training.

$$\ell_{i}=-\log{\frac{\exp\left(\sin(\mathbf{z}_{i},\mathbf{z}_{i}^{+})/\tau\right)}{\sum_{n=1}^{N}\exp\left(\sin(\mathbf{z}_{i},\mathbf{z}_{(i,n)}^{-})/\tau\right)}}$$

$$(2)$$

This sophisticated loss function aims to maximize the similarity between positive pairs while maximizing the distinction between negative pairs. Through this process, the model learns the intrinsic structure and patterns of the data. In the case of EEG signals, since each frequency band carries distinct neurological significance, performing independent contrastive learning for each band allows the model to capture more fine-grained and meaningful. Regularization Furthermore, we implement an advanced contrastive learning loss function that applies adaptive temperature to each frequency band head. While traditional SimCLR uses a fixed temperature parameter, this implementation dynamically adjusts the temperature based on the characteristics of each frequency band head. This approach considers the unique distribution properties and learning difficulty of different frequency bands, enabling more effective contrastive learning. The regularization is formulated as follows:

$$\Omega(\tau)=(d^{\prime}/2)\log(\tau)+1/\tau,$$
′/2) log(τ ) + 1/τ, (3)
where d is the feature dimension. The projection band head maps these d-dimensional vectors into d
′-dimensional vectors. This induces the temperature τ to move to τ = 2/d′.

$$({\mathfrak{I}})$$

## 3 Experimental Setup 324

325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 During the pre-training stage, the model was trained using the AdamW optimizer (Loshchilov &
Hutter, 2017) with a batch size of 64 and a learning rate of 1 × 10−4. The adaptive NT-XENT loss function was employed, with the adaptive temperature parameter adjusted between 0.05 and 0.5, and a temperature regularization parameter β of 0.01 applied. In the subsequent linear evaluation stage, Leave-One-Subject-Out (LOSO) cross-validation was used, and classification was performed with the pre-trained encoder weights kept frozen. The AdamW optimizer was used with a batch size of 32 and a learning rate of 1 × 10−4to minimize the cross-entropy loss. Both stages were trained for up to 100 epochs, with early stopping applied if no performance improvement was observed for 10 consecutive epochs. We trained the model on Intel Xeon-Silver 4410Y CPU and NVIDIA GeForce RTX 4090 GPU. In addition, a weight decay (Loshchilov & Hutter, 2017) of 1 × 10−5 was applied.

The scheduler uses cosine annealing with warm restarts (Loshchilov & Hutter, 2016).

## 3.1 Dataset

The dataset (Miltiadous et al., 2023b) used in this study consists of resting-state, eyes-closed EEG recordings from a total of 88 participants. Among them, 36 were diagnosed with Alzheimer's disease (AD) group, 23 with frontotemporal dementia (FTD) group, and 29 were cognitively normal (CN) group. Cognitive and neuropsychological status was assessed using the mini-mental state examination (MMSE), with scores ranging from 0 to 30, where lower scores indicate greater cognitive impairment. The median disease duration was 25 months, and no dementia-related comorbidities were reported in the AD group. The mean MMSE scores were 17.75 for the AD group, 22.17 for the FTD group, and 30 for the CN group. The mean ages were 66.4 years for the AD group, 63.6 years for the FTD group, and 67.9 years for the CN group. EEG recordings were collected by a team of experienced neurologists at the department of neurology, AHEPA University Hospital of Thessaloniki, using the Nihon Kohden EEG-2100 clinical system. 19 scalp electrodes (according to the 10-20 system) and two mastoid reference electrodes were used, with impedance maintained below 5 k. All recordings were performed with participants seated and eyes closed, at a sampling rate of 500 Hz and a sensitivity of 10 µV/mm. Both bipolar and referential montages (referenced to Cz) were included. The average recording duration was approximately 13.5 minutes for the AD group (5.1-21.3 min), 12 minutes for the FTD group (7.9-16.9 min), and 13.8 minutes for the CN group (12.5-16.5 min), resulting in a total of 485.5, 276.5, and 402 minutes of data for each group, respectively.

## 3.2 Preprocessing

First, when performing EEG source localization, it is recommended to calculate the average reference. Since referencing affects the measurement of signal amplitudes, it is an important step in EEG preprocessing. If a single electrode is used as a reference for the others, brain activity and noise from the reference electrode can be mixed into the signals. One approach is to compute the average potential across all channels and use this average as the reference. This way, the potential at each scalp location is compared to all other recorded sites, better reflecting the unique characteristics of each location. EEG data were preprocessed by applying a 6th order Butterworth bandpass filter in the 0.5 to 45 Hz range (Nour et al., 2024), which effectively preserves information relevant to distinguishing neural activity between AD and CN. Additionally, artifacts were removed using blind source separation with independent component analysis (ICA) (Comon, 1994; Makeig et al., 1995; Jung et al., 1997). All of the processing for the EEG was done with the python library MNE (Gramfort et al., 2013).

## 3.3 Segmentation

In recent AD research, the analysis of EEG characteristics related to sleep has emerged as an important trend. In particular, EEG data collected during the eyes-close state are widely used in early diagnosis studies of dementia, as this approach minimizes the influence of external stimuli and more clearly reflects intrinsic brain activity. In line with this trend, our study segmented EEG data acquired during the eyes-close state into 30-second intervals for analysis (Ye et al., 2023; Herzog et al., 2023; Park et al., 2025). This segmentation matches the standard epoch length used in sleep re378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

## 4 Results And Discussion 4.1 Comparison Of Classification Performance

To evaluate the performance of the proposed model, we compared it with major benchmark models in the field of EEG analysis. The details of each EEG benchmark model are provided in the appendix, and for the SSL models, fine-tuning was performed when pretrained weights were available. As shown in Table 1, the proposed model achieved 92.90% accuracy and 92.85% F1-score, significantly outperforming all comparison models. These results clearly demonstrate the superiority of our approach, indicating that the proposed adaptive multi-band head method is highly effective in learning and classifying the complex features of EEG signals.

| Model                                          | Backbone        | Acc (%)   | F1 (%)   |
|------------------------------------------------|-----------------|-----------|----------|
| ATCNet (Altaheri et al., 2022)                 | CNN & RNN & Att | 74        | 74       |
| BIOT (Yang et al., 2023)                       | Att             | 53        | 40       |
| CTNet (Zhao et al., 2024)                      | CNN & Att       | 74        | 73       |
| Deep4Net (Schirrmeister et al., 2017)          | CNN             | 49        | 49       |
| EEGConformer (Song et al., 2022)               | CNN & Att       | 57        | 54       |
| EEGInception (Santamaria-Vazquez et al., 2020) | CNN             | 39        | 37       |
| EEGNet (Lawhern et al., 2018)                  | CNN             | 46        | 45       |
| FBCNet (Mane et al., 2021)                     | CNN             | 48        | 38       |
| Labram (Jiang et al., 2024)                    | CNN & Att       | 54        | 38       |
| S-JEPA (Guetschel et al., 2024)                | CNN & Att       | 50        | 50       |
| SPARCNet (Jing et al., 2023)                   | CNN             | 54        | 53       |
| TIDNet (Kostas & Rudzicz, 2020)                | CNN             | 44        | 40       |
| Ours (adaptive 5 band heads)                   | CNN             | 93%       | 93%      |

Table 1: Comparison of the Alzheimer's disease (AD) and the cognitively normal (CN) classification performance between the proposed adaptive 5 band heads model and leading benchmark models in EEG analysis, including both supervised and self-supervised learning approaches. Here, CNN refers to convolutional neural networks, RNN to recurrent neural networks, and Att to attention mechanisms.

## 4.2 Loso Performance

To further validate the performance of the proposed model, we compared its results with those of previous studies on the task of classifying AD and CN subjects. For fair evaluation and to assess generalization performance, all models were evaluated using strict LOSO cross-validation. As shown search, thereby enhancing the reliability and comparability of our investigation into the relationship between dementia and sleep.

## 3.4 Leave-One-Subject-Out

In EEG research, the LOSO method (Kunjan et al., 2021) is a widely used cross-validation technique for rigorously evaluating the generalization performance of a dataset. In this approach, one subject is used as the validation set while all remaining participants are used as the training set. This process is repeated for each participant in the dataset. In other words, if the dataset consists of N subjects, the procedure is repeated N times, with each iteration using the data from one subject for testing and the data from the remaining (N − 1) subjects for training. The main purpose of this method is to account for the high inter-individual variability in EEG signals and to assess how well the model can predict data from new, unseen subjects. Because EEG data can vary greatly due to individual physiological characteristics, scalp conditions, noise, and other factors, a more stringent and realistic performance evaluation is required compared to simple k-fold cross-validation. The LOSO method addresses this by preventing data leakage between subjects and ensuring complete independence between the training and validation sets.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

## 5 Conclusion

in Table 2, the proposed adaptive 5 band heads model achieved 92.90% accuracy and an 92.85% F1-score. These results demonstrate that the proposed multi-frequency band-based self-supervised learning and linear evaluation approach is highly effective in overcoming inter-subject variability and learning features with excellent generalization performance.

Model Acc (%) F1 (%) Pre (%) Rec (%) kNN (Ntetska et al., 2025) 60.30 58.90 57.90 59.90 CNN (Stefanou et al., 2025) 79.45 77.60 76.32 76.06 Random Forest (Sarkar et al., 2025) 80.00 81.69 - 80.55 DICE-Net (Miltiadous et al., 2023a) 83.28 84.12 88.94 79.81 CNN (Vo et al., 2025) 84.62 86.11 - 86.11 MJANet (Sun et al., 2025) 85.23 86.37 88.12 84.69 Dual-Branch (Chen et al., 2023) 85.78 - - 83.22

Random Forest (Parihar & Swami, 2024) 88.90 - - -

BI-MCGNN (Zhang & Zhu, 2025) 91.25 ± 0.38 - - 93.32 ± **0.46** Ours (adaptive 5 band heads) 92.90 92.85 93.27 92.90

Table 2: Leave-One-Subject-Out (LOSO) performance comparison for Alzheimer's disease (AD) and the cognitively normal (CN) classification using the dataset (Miltiadous et al., 2023b).

## 4.3 Ablation Study

We analyze the impact of each component of the proposed model on performance, as summarized in Table 3. To compare with self-supervised learning, we trained the CNN model from scratch on the EEG dataset, and it achieved an accuracy of 63.35%. The experimental results show that using only a single projection head (single-head) led to a significant drop in accuracy to 73.52%, , while the 5 multi-head architecture achieved 79.55%, highlighting the importance of the multi-frequency band head architecture. Without data augmentation, we masked 15% of the EEG signal and trained the encoder model to reconstruct it using mean squared error (MSE) loss, achieving 78.58% accuracy. In addition, fixing the temperature (τ ) parameter at 0.1 and removing regularization resulted in decreased performance, with accuracies of 86.53% and 90.64%, respectively. These findings suggest that both the adaptive temperature adjustment and the regularization techniques in the proposed model contribute positively to the final performance.

Model Acc (%) F1 (%) Pre (%) Rec (%) AUC (%) Adaptive 5 band heads 92.90 92.85 93.27 92.90 96.77 w/o self-supervised learning 63.35 61.68 63.77 63.35 67.98 Single-head 73.52 72.33 82.88 73.52 64.89 w/o augmentation 78.58 78.05 79.74 78.58 78.34 Multi-head (5 heads) 79.55 79.38 82.77 79.55 78.04 constant temperature (τ = 0.1) 86.53 86.56 87.30 86.53 87.21 w/o regularization 90.64 90.59 91.72 89.64 91.33

Table 3: Ablation Study results for the proposed model on the dataset (Miltiadous et al., 2023b).

We propose a multi-head SimCLR-based framework for contrastive learning of EEG representations, leveraging independent CNN encoders and adaptive temperature parameters for each of the 5 frequency bands. Using Adaptive Multi-head Contrastive Learning (AMCL) strategy (Wang et al., 2024), we compute and aggregate contrastive losses for each band, resulting in superior representation learning and classification performance, especially with limited labels.

9 486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539

## References

Abien Fred Agarap. Deep learning using rectified linear units (relu). arXiv preprint arXiv:1803.08375, 2018.

H. Altaheri, G. Muhammad, and M. Alsulaiman. Physics-informed attention temporal convolutional network for eeg-based motor imagery classification. *IEEE Transactions on Industrial Informatics*, 19(2):2249–2258, 2022. doi: 10.1109/TII.2022.3140497.

K. Baik, J. H. Jung, S. H. Jeong, S. J. Chung, H. S. Yoo, P. H. Lee, others, and B. S. Ye. Implication of eeg theta/alpha and theta/beta ratio in alzheimer's and lewy body disease. *Scientific Reports*, 12(1):18706, 2022.

C. S. Benwell, P. Davila-Perez, P. J. Fried, R. N. Jones, T. G. Travison, E. Santarnecchi, others, and ´
M. M. Shafi. Eeg spectral power abnormalities and their relationship with cognitive dysfunction in patients with alzheimer's disease and type 2 diabetes. *Neurobiology of Aging*, 85:83–95, 2020.

S. Cavedoni, A. Chirico, E. Pedroli, P. Cipresso, and G. Riva. Digital biomarkers for the early detection of mild cognitive impairment: artificial intelligence meets virtual reality. Frontiers in Human Neuroscience, 14:245, 2020.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In Hal Daume III and Aarti Singh (eds.), ´
Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 1597–1607. PMLR, July 2020. URL https: //proceedings.mlr.press/v119/chen20j.html.

Y. Chen, H. Wang, D. Zhang, L. Zhang, and L. Tao. Multi-feature fusion learning for alzheimer's disease prediction using eeg signals in resting state. *Frontiers in Neuroscience*, 17:1272834, 2023.

P. Comon. Independent component analysis, a new concept? *Signal processing*, 36(3):287–314, 1994.

T. Dabbabi, L. Bouafif, and A. Cherif. A review of non invasive methods of brain activity measurements via eeg signals analysis. In *2023 IEEE International Conference on Advanced Systems and* Emergent Technologies (IC *ASET)*, pp. 01–06. IEEE, April 2023.

Alexandre Gramfort, Martin Luessi, Eric Larson, Denis A. Engemann, Daniel Strohmeier, Christian Brodbeck, Roman Goj, Mainak Jas, Teon Brooks, Lauri Parkkonen, and Matti S. Ham¨ al¨ ainen. ¨ MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7(267):1–13, 2013. doi: 10.3389/fnins.2013.00267.

Philipp Guetschel, Thomas Moreau, and Michael Tangermann. S-jepa: Towards seamless crossdataset transfer through dynamic spatial attention. *arXiv preprint arXiv:2403.11772*, 2024.

H. Haidar, R. E. Majzoub, S. Hajeer, and L. A. Abbas. Arterial spin labeling (asl-mri) versus fluorodeoxyglucose-pet (fdg-pet) in diagnosing dementia: a systematic review and meta-analysis. BMC Neurology, 23(1):385, 2023. doi: 10.1186/s12883-023-03334-0.

R. Herzog, S. Haghayegh, A. Iba´nez, and K. Hu. A novel biomarker of alzheimer's disease based ˜
on high-order interactions of low-density electroencephalography. *Sleep*, 46:A25–A25, 2023.

Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *International Conference on Machine Learning*, pp. 448–456. PMLR, 2015.

W. B. Jiang, L. M. Zhao, and B. L. Lu. Large brain model for learning generic representations with tremendous EEG data in BCI. *arXiv preprint arXiv:2405.18765*, 2024.

J. Jing, W. Ge, S. Hong, M. B. Fernandes, Z. Lin, C. Yang, and M. B. Westover. Development of expert-level classification of seizures and rhythmic and periodic patterns during EEG interpretation. *Neurology*, 100(17):e1750–e1762, 2023.

A. Juganavar, A. Joshi, and T. Shegekar. Navigating early alzheimer's diagnosis: a comprehensive review of diagnostic innovations. *Cureus*, 15(9), 2023.

Tzyy-Ping Jung, Colin Humphries, Te-Won Lee, Scott Makeig, Martin McKeown, Vicente Iragui, and Terrence J Sejnowski. Extended ica removes artifacts from electroencephalographic recordings. In *Advances in neural information processing systems*, volume 10, 1997.

J. Kaminski, A. Brzezicka, and A. Wr ´ obel. Short-term memory capacity (7±2) predicted by theta to ´
gamma cycle length ratio. *Neurobiology of Learning and Memory*, 95(1):19–23, 2011.

S. Kancharla. Integrating traditional machine and deep learning methods for enhanced alzheimer's detection from mri images. In *2024 IEEE International Conference on Information Reuse and* Integration for Data Science (IRI), pp. 290–295. IEEE, August 2024.

D. Kostas and F. Rudzicz. Thinker invariance: enabling deep neural networks for BCI across more people. *Journal of Neural Engineering*, 17(5):056008, 2020. doi: 10.1088/1741-2552/aba7c7.

S. Kunjan, T. S. Grummett, K. J. Pope, D. M. Powers, S. P. Fitzgibbon, T. Bastiampillai, and T. W.

Lewis. The necessity of leave one subject out (loso) cross validation for eeg disease diagnosis.

In Brain Informatics: 14th International Conference, BI 2021, Virtual Event, September 17–19, 2021, Proceedings 14, pp. 558–567. Springer International Publishing, 2021.

V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance. EEGNet:
a compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of* Neural Engineering, 15(5):056013, 2018. doi: 10.1088/1741-2552/aace8c.

X. Li, X. Feng, X. Sun, N. Hou, F. Han, and Y. Liu. Global, regional, and national burden of alzheimer's disease and other dementias, 1990–2019. *Frontiers in Aging Neuroscience*, 14: 937486, 2022. doi: 10.3389/fnagi.2022.937486.

Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983, 2016. URL https://arxiv.org/abs/1608.03983.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. *arXiv preprint* arXiv:1711.05101, 2017.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Scott Makeig, Adrian Bell, Tzyy-Ping Jung, and Terrence J Sejnowski. Independent component analysis of electroencephalographic data. In *Advances in neural information processing systems*, volume 8, 1995.

Ramesh Mane, Eng Tat Chew, Kian Meng Chua, Kai Keng Ang, Nicholas Robinson, Ashirbani P.

Vinod, and Cuntai Guan. Fbcnet: A multi-view convolutional neural network for brain-computer interface. *arXiv preprint arXiv:2104.01233*, 2021.

A. Miltiadous, E. Gionanidis, K. D. Tzimourta, N. Giannakeas, and A. T. Tzallas. Dice-net: a novel convolution-transformer architecture for alzheimer detection in eeg signals. *IEEE Access*, 11: 71840–71858, 2023a.

A. Miltiadous, K. D. Tzimourta, T. Afrantou, P. Ioannidis, N. Grigoriadis, D. G. Tsalikakis, others, and A. T. Tzallas. A dataset of scalp eeg recordings of alzheimer's disease, frontotemporal dementia and healthy subjects from routine eeg. *Data*, 8(6):95, 2023b.

D. V. Moretti, C. Babiloni, G. Binetti, E. Cassetta, G. Dal Forno, F. Ferreric, others, and P. M.

Rossini. Individual analysis of eeg frequency and band power in mild alzheimer's disease. Clinical Neurophysiology, 115(2):299–308, 2004.

E. Nichols, J. D. Steinmetz, S. E. Vollset, K. Fukutaki, J. Chalek, F. Abd-Allah, and X. Liu. Estimation of the global prevalence of dementia in 2019 and forecasted prevalence in 2050: an analysis for the global burden of disease study 2019. *The Lancet Public Health*, 7(2):e105–e125, 2022.

M. Nour, U. Senturk, and K. Polat. A novel hybrid model in the diagnosis and classification of alzheimer's disease using eeg signals: Deep ensemble learning (del) approach. *Biomedical Signal* Processing and Control, 89:105751, 2024.

A. Ntetska, A. Miltiadous, M. G. Tsipouras, K. D. Tzimourta, T. Afrantou, P. Ioannidis, and A. T.

Tzallas. A complementary dataset of scalp eeg recordings featuring participants with alzheimer's disease, frontotemporal dementia, and healthy controls, obtained from photostimulation eeg. Data, 10(5):64, 2025.

B. S. Olivari, M. E. French, and L. C. McGuire. The public health road map to respond to the growing dementia crisis. *Innovation in Aging*, 4(1):igz043, 2020.

A. Parihar and P. D. Swami. Analysis of eeg signals with the use of wavelet transform for accurate classification of alzheimer disease, frontotemporal dementia and healthy subjects using machine learning models. *Fusion: Practice & Applications*, 14(2), 2024.

C. Park, J. I. Byun, S. H. Choi, and W. C. Shin. Machine learning classifier solving the problem of sleep stage imbalance between overnight sleep. *Biomedical Engineering Letters*, pp. 1–11, 2025.

M. N. Sabbagh, M. Boada, S. Borson, M. Chilukuri, B. Dubois, J. Ingram, and H. Hampel. Early detection of mild cognitive impairment (mci) in primary care. *The Journal of Prevention of* Alzheimer's Disease, 7:165–170, 2020.

Edgar Santamaria-Vazquez, Victor Martinez-Cagigal, Fernando Vaquerizo-Villar, and Roberto Hornero. Eeg-inception: a novel deep convolutional neural network for assistive erp-based braincomputer interfaces. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 28 (12):2773–2782, 2020.

S. Sarkar, A. Chakraborty, A. Sinha, and S. K. Saha. Detection of alzheimer's disease using extreme band powers in eeg data. In *2025 IEEE Applied Sensing Conference (APSCON)*, pp. 331–334. IEEE, 2025.

R. T. Schirrmeister, J. T. Springenberg, L. D. J. Fiederer, M. Glasstetter, K. Eggensperger, M. Tangermann, F. Hutter, W. Burgard, and T. Ball. Deep learning with convolutional neural networks for eeg decoding and visualization. *Human Brain Mapping*, 38(11):5391–5420, 2017. doi: 10.1002/hbm.23730.

U. Smailovic and V. Jelic. Neurophysiological markers of alzheimer's disease: quantitative eeg approach. *Neurology and Therapy*, 8(Suppl 2):37–55, 2019.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Y. Song, Q. Zheng, B. Liu, and X. Gao. EEG conformer: Convolutional transformer for EEG decoding and visualization. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 31:710–719, 2022. doi: 10.1109/TNSRE.2022.3233000.

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov.

Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1):1929–1958, 2014.

K. Stefanou, K. D. Tzimourta, C. Bellos, G. Stergios, K. Markoglou, E. Gionanidis, others, and A. Miltiadous. A novel cnn-based framework for alzheimer's disease detection using eeg spectrogram representations. *Journal of Personalized Medicine*, 15(1):27, 2025.

Y. Sun, L. Feng, B. Xu, S. Jia, L. Duan, W. Ni, and Z. Jia. Enhanced alzheimer's detection with eeg source imaging and multi-branch joint attention. *Journal of Neural Engineering*, 22(3):036028, 2025.

A. Traikapi and N. Konstantinou. Gamma oscillations in alzheimer's disease and their potential therapeutic role. *Frontiers in Systems Neuroscience*, 15:782399, 2021.

T. Vo, A. K. Ibrahim, and H. Zhuang. A multimodal multi-stage deep learning model for the diagnosis of alzheimer's disease using eeg measurements. *Neurology International*, 17(6):91, 2025.

L. Wang, P. Koniusz, T. Gedeon, and L. Zheng. Adaptive multi-head contrastive learning. In European Conference on Computer Vision, pp. 404–421, Cham, September 2024. Springer Nature Switzerland.