# Myna: Masking-Based Contrastive Learning of Musical Representations

- Avg Score: 5.33
- Decision: Reject
- Scores: 2, 6, 8

## Abstract
In this paper, we present Myna, a simple yet effective approach for self-supervised musical representation learning. Built on a contrastive learning framework, Myna introduces two key innovations: (1) the use of a Vision Transformer (ViT) on mel-spectrograms as the backbone, replacing SampleCNN on raw audio; and (2) a simple yet novel data augmentation strategy—token masking—that masks 90% of spectrogram tokens (e.g., 16x16 patches). These innovations deliver both effectiveness and efficiency: (i) Token masking enables a significant increase in per-GPU batch size, from 48 or 120 in traditional contrastive methods (e.g., CLMR, MULE) to 4096. (ii) By avoiding traditional augmentations (e.g., pitch shifts), Myna retains pitch sensitivity, enhancing performance in tasks like key detection. (iii) The use of vertical patches (128x2 instead of 16x16) allows the model to better capture critical features for key detection. Our hybrid model, Myna-22M-Hybrid, processes both 16x16 and 128x2 patches, achieving state-of-the-art results. Trained on a single GPU, it outperforms MULE (62M) on average and rivals MERT-95M, which was trained on 16 and 64 GPUs, respectively. When scaled to 85M parameters, Myna achieves further improvements across all tasks and is competitive with models like MERT-330M, MusicFM, and MuQ despite being 3-7x smaller and trained with an order of magnitude fewer GPUs in less time. Additionally, it surpasses MERT-95M-public and MuQ$_{m4a}$, establishing itself as the best-performing model trained on publicly available data. We release our code and models to promote reproducibility and facilitate future research: https://github.com/ghost-signal/myna

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this paper, the authors proposes a new training method for the representation learning of music audio. The proposed method includes aggressive input masking that seems to allow avoiding pitch shifting as an augmentation step, making the model aware of pitch and key information. The experiment showed that even without finetuning on the downstream task's training set and despite its smaller size & training data, Myna outperforms many other methods.

### Strengths
- Good performance
- Parameter-efficient
- Trained on a public dataset only
- The proposed method is simple

### Weaknesses
- Limited novelty: Some core changes such as using ViT and masked autoencoder are already proposed in other, similar work including audio domain.
- Although the performance is strong, the margin is rather reasonable, not outstanding.

### Questions
- I don't think we should call the used audio processor as a "tokenizer", no matter how the word is over-subscribed in the community. It does not tokenize (making the input a discrete representation) at all, and it's even worse because some architectures indeed discretize the input audio.

### Soundness
4

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
This paper proposes MYNA, an efficient, masking-based contrastive learning framework for musical representation. It replaces traditional music augmentations with a high-rate token masking (90%) on mel-spectrograms. This heavy masking significantly reduces the number of input tokens, allowing for large batch sizes (4096) on a single GPU, and achieves an efficiency gain over prior contrastive methods. The authors also introduce a hybrid patching scheme (combining vertical and square patches) to capture complementary features (general purpose vs. pitch structure). The model is pretrained on the public AudioSet music subset. Myna achieves competitive performance with larger private models and establishes a new public-data SOTA.

### Strengths
1. The mask-only approach is simple and allows single-GPU large-batch training (batch size 4096), which translates to an 85x increase in efficiency over traditional contrastive methods like CLMR. The model achieves competitive average scores (68.6 for Myna-Hybrid) with MERT-95M, and surpasses public baselines like MERT-95M-public and MULE.

2. The hybrid patch design improves key detection (achieving SOTA among self-supervised methods) by integrating frequency-sensitive vertical patches. The method retains pitch sensitivity by avoiding traditional data augmentations (e.g., pitch shifts), which is beneficial for tasks like key detection.

### Weaknesses
1. Table 1 mixes public and private data baselines (e.g., MERT-330M) without transparently clarifying the training resource budgets.

2. The claim that "90% masking performs best" is not strongly supported by Figure 4. This is due to two issues: (a) Performance differences across high masking ratios looks marginal and lack verification of statistical significance; (b) The "average across all four benchmarks" curve can be mathematically unrigorous as it combines different metrics from different tasks.

3. The model's poor performance on EmoMusic is attributed to short clip length, a hypothesis that needs empirical verification.

### Questions
1. It would be helpful if Table 1 were explicitly partitioned to clearly distinguish models trained on public data from those trained on private or internal corpora.

2. Could you provide supplementary figures showing the performance curves across different masking ratios for each of the four downstream tasks (MTT, GiantSteps, GTZAN, and EmoMusic)?

### Soundness
3

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
3

### Summary
This paper focuses on music representation learning. It follows a contrastive learning framework, with its main contributions being the use of a Vision Transformer (ViT) as the backbone model and the application of token masking. Furthermore, considering the characteristics of music analysis, the authors extend the approach into a hybrid model that incorporates vertical filters to better capture the frequency-related features of spectrograms. Through this relatively simple training strategy, the proposed model achieves competitive performance on several downstream tasks compared to models that require more than 5× the training time and parameters. Overall, the paper is well written and presents a solid contribution to efficient representation learning for music.

### Strengths
The proposed use of ViT and token masking seems promising in music representation learning. 
The paper is easy-to-read and the illustration of the proposed method, experimental design, and results seems promising.

### Weaknesses
The proposed method seems to be only applicable to the clip-level MIR tasks. I wonder the authors opinion (maybe discussions) on how the proposed architecture can be applied towards frame-level tasks as well.

### Questions
I wonder the effect of the patch size variations on performances. For example, would 4x4, 96x2, 128x3, 32x32, hybrid of them, etc these kind of diverse patch size affects the performance?

### Soundness
3

### Presentation
3

### Contribution
3
