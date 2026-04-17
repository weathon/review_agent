# Cluster-Masked Scanning and Pretraining for Enhanced xLSTM Vision Performance

- Decision: Reject
- Scores: 6, 4, 4

## Abstract
While modern recurrent architectures like xLSTM show promise for vision tasks, their potential has been hindered by the challenge of effectively applying autoregressive pretraining---a cornerstone of NLP success---to 2D image data. This paper introduces MAL, a framework that unlocks autoregressive learning for vision-oriented xLSTMs. Our core innovation is a cluster-masked pretraining strategy, which reorganizes an image into a sequence of semantically meaningful local clusters. This approach creates a more structured input sequence uniquely suited to xLSTM's memory mechanisms. By combining this with our novel cluster scanning strategy which defines an optimal processing order, MAL effectively learns powerful visual representations by predicting entire image regions autoregressively. Our experiments show that this novel pretraining scheme allows MAL to significantly outperform traditional supervised models, fully leveraging the scaling potential of xLSTM and setting a new performance benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a cluster-masked pretraining strategy for autoregressive prediction of image regions (MAL). 
MAL reorganizes an image into a sequence of local clusters by forming groups of image patches and varying the scanning and prediction order.
MAL is accompanied by a two stage training procedure and an encoder decoder architecture, where the encoder is an xLSTM and the decoder an attention model. In the first stage the encoder-decoder architecture is used for pretraining to build strong features in the encoder. In the second stage the encoder is finetuned on respective downstream tasks.
On standard image benchmarks MAL shows strong performance, outperforming DeiT, Vision-Mamba and Vision-LSTM baselines.

### Strengths
- Extensive experiments on different backbone architectures like Mamba and Vision Transformers with attention.
- Strong performance of MAL compared to the baselines.

### Weaknesses
- While the experiments demonstrate superior performance of MAL, intuitively it is not fundamentally different from standard vision transformer based patching. Both masking and scanning techniques are not informed about the content of the images, and rather convert the image into sequences by patching and then arranging these patches into a sequence. MAL seems to be a more complex strategy for creating this sequence, but intuitively it is not clear why this results in better performance nor why the sequence is “semantically more meaningful” ?
Could the authors elaborate on the intuition why MAL shows better performance ?
- As described in the paper, MAL is complemented by a two stage process that trains an encoder-decoder architecture in the first stage and then finetunes  the encoder with linear heads in the second stage.
A natural question to ask is whether the cluster-masked pretraining strategy would also help direct ViT pretraining or whether the combination with this 2 stage training procedure and the encoder-decoder architecture is strictly necessary. Ablations on this would further strengthen the paper.

### Questions
- Have you also experimented with xLSTM or Mamba blocks in the Decoder?
- Did you ensure that the compute budget (e.g. in number of epochs) in the overall pre-training and finetuning is comparable to the budget of the baselines?

### Soundness
3

### Presentation
3

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
This paper proposes a modified autoregressive neural network on image patches built on x-LSTM, to learn image representations through a self-supervised reconstruction loss. Different from original x-LSTM, it employs a two-level patch-modeling strategy. The image is first divided into large patches, and the large patches are further divided into smaller patches. The LSTM forward pass first go through all smaller patches and then go across large patches. It also uses a masking strategy to enhance the model similar to masked auto-encoder (MAE). It achieves better performance than existing methods including transformer based (DeiT) and autoregressive models (Mamba, VMamba, VRWKV) on imagenet classification, detection and instance segmentation.

### Strengths
-	The overall idea is executed well. The model performance is strong among similar approaches.
-	The proposed cluster-masked strategy achieves better performance than the original MAE strategy (table 4)

### Weaknesses
-	The overall novelty of the proposed method is limited. The difference from original x-LSTM is the two-level scanning order, which cannot be deemed as a major contribution. Similar idea has also been used by existing literature such as: Autoregressive Pretraining with Mamba in Vision. arXiv 2024. The training /finetuning details are very similar to those of self-supervised learning literature.
-	The author has some misleading description about the formulation. The two-level patch formation is referred to as “cluster-based”, which is problematic since it does not involve any clustering algorithm. It could just be "patches" and "grouped patches".

### Questions
NA

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
4

### Summary
MAL proposes to enable effective autoregressive pretraining for vision xLSTMs by grouping adjacent patches into semantically coherent clusters, serializing these clusters via a proposed cluster-scanning order, and training xLSTM encoders to autoregressively predict cluster-level image regions; the method claims improved visual representations and downstream performance by leveraging cluster units to both reduce sequence length and better match xLSTM’s memory mechanisms.

### Strengths
- The idea of cluster-based serialization provides a clear intuition: grouping local regions can preserve spatial relationships better than flat patch sequences.
- The proposed framework is well-motivated and technically consistent, showing that autoregressive modeling can be made practical for xLSTM-based vision systems.
- The empirical results are solid, with comprehensive experiments and reasonable ablations that validate the design choices.

### Weaknesses
- The conceptual novelty of the cluster-masked strategy appears somewhat limited compared to existing masked or region-based pretraining methods.
- The cluster scanning order lacks theoretical grounding or strong empirical justification; the claimed “optimal” ordering seems heuristic.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3
