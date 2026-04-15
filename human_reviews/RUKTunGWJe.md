# INRSTEG: FLEXIBLE CROSS-MODAL LARGE CAPACITY STEGANOGRAPHY VIA IMPLICIT REPRESENTATIONS

- Decision: Reject
- Scores: 5, 3, 5, 5

## Abstract
We present INRSteg, an innovative lossless steganography framework based on a novel data form Implicit Neural Representations (INR) that is modal-agnostic. Our framework is considered for effectively hiding multiple data without altering the original INR ensuring high-quality stego data. The neural representations of secret data are first concatenated to have independent paths that do not overlap, then weight freezing techniques are applied to the diagonal blocks of the weight matrices for the concatenated network to preserve the weights of secret data while additional free weights in the off-diagonal blocks of weight matrices are fitted to the cover data. Our framework can perform unexplored cross-modal steganography for various modalities including image, audio, video, and 3D shapes, and it achieves state-of-the-art performance compared to previous intra-modal steganographic methods.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes cross-modal high-capacity steganography based on INRs. It occupies part of the weights of the stego INR with the INR containing the secret message and freezes it, and then uses the remaining weights of the stego INR to simulate the function of the cover INR, so as to hide the INR of the secret message while guaranteeing that the function of the stego INR is similar to that of the cover INR.

### Strengths
This article proposes a novel INR-based multimodal steganography framework.

### Weaknesses
1. Steganography pursues behavioral security, but the framework causes the size of the stego INR to be larger than the size of the normal cover INR, and an attacker may be able to detect the existence of INR steganography based on this anomalous behavior.

2. Security experiments: although this paper can resist traditional image steganalysis, considering that it is similar to neural network steganography, it should be supplemented with experiments on resisting neural network steganalysis.

3. Comparison experiments: Considering that multimodal data can be converted into binary streams, this paper should be supplemented with comparisons with binary stream steganography (e.g., chatgan, etc.).

### Questions
How robust is the framework? Can it resist network fine-tuning, pruning, and other operations?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes an innovative lossless cross-modal steganography framework based on implicit neural representations (INR). Extensive experiments demonstrate the superiority of the proposed method.

### Strengths
N/A

### Weaknesses
N/A

### Questions
(1)	The English writing should be improved to make this paper readable;
(2)	Is the meaning of “Cross-modal” same with that of “modal-agnostic”? This question should be explained comprehensively. 
(3)	I cannot clearly understand details of the proposed framework. I confused how to implement the lossless steganography.
(4)	The experimental results do not evaluate the embedding capacity of the proposed method. I don’t know what is the key reason why the proposed steganography can achieve large capacity.
(5)	Authors merely conduct image steganalysis to evaluate the security of the proposed method, which is not enough for the proposed “modal-agnostic” method.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a steganography framework for data represented as Implicit Neural Representations (INR). The method works as follows: multiple secret data are encoded with neural representations, the representations are concatenated without overlap and padded, the padded weights are treated as the only trainable weights and are trained to learn a neural representation for the cover data. Every neural representation is learned with a MLP. The weights in each layer of the final model can be optionally permuted by making use of a private key.

### Strengths
The strengths are as follows:
* The authors empirically show that you can hide different modality secret data in different modality cover data.
* The stego data and the recovered secret data both have low distortion. 
* Interesting analysis of the weight distribution in section 4.4.

### Weaknesses
The weaknesses are as follows:
* The quality of the steganography methods is measured by how much information can be stored in how much space. For instance, image steganography methods report bits per pixel to state how many bits of information can be hidden in each pixel. Similarly, it would be worthwhile to know what the size of the cover data, the secret data is, and the INRs are. It is also a limitation that the INR can be quite large. 
* There are multiple missing baselines like SteganoGAN [1] and LISO [2].
* The motivation for hiding data in INRs is not very clear.

[1] Zhang, K. A., Cuesta-Infante, A., Xu, L., & Veeramachaneni, K. (2019). SteganoGAN: High capacity image steganography with GANs. arXiv preprint arXiv:1901.03892.
[2] Chen, X., Kishore, V., & Weinberger, K. Q. (2022, September). Learning Iterative Neural Optimizers for Image Steganography. In The Eleventh International Conference on Learning Representations.

### Questions
* Why is hiding audio in audio worse than hiding audio in images?
* Did you train SiaStegNet and XuNet or use a pre-trained model? Do these models operate on INRs or the recovered cover data from the INR?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a framework to hide secret Implicit Neural Representations (INRs) into a cover INR, namely INRSteg.

INRSteg is capable of hiding multiple cross-modal data within the weight space of INR by concatenating multiple secret INRs and permutating the weights INR. When recovering the permutation, the secret INR can be retrieved. INRSteg shows significant improvement in distortion evaluation, capacity, and security in various experiments, including intra and cross-modal steganography, compared to previous steganography methods.

### Strengths
originality: the proposed steganography method for INR is simple and novel. 

quality: the paper is technically sound.

clarity: the paper is well-organized.

significance: the paper is somehow significant. This paper proposes a way to hide secret INRs in a cover INR, which is not only beneficial for hiding secret data but also for watermarking and copyright protection.

### Weaknesses
This paper describes how to encode secret INRs into a cover INR. However, this paper did not do a lot of protection/robustness analysis. For instance, what if the cover INR is being pruned during transmission? 

The cover INR would be very big if a lot of secret INRs were embedded. This may be suspicious to malicious attackers who wish to analyze this suspiciously big INR. Although after permutation the weight distribution seems nothing, how about we just sort the weights? The permutation of the weights will not affect the intermediate activations and the outputs, once sorting the weights, will there be any obvious changes? In summary, this paper lacks attack analysis, such as assuming the knowledge of attackers and how attackers can attack (to prevent the final owner from obtaining the correct secret data)/steal (to extract the secret).

### Questions
1. can you summarize the advantages/disadvantages of StegaNeRF [1] over the proposed method? Seems like StegaNeRF is also quite related to this proposed method and is one of the latest works in INR steganography.
2. What if the cover INR is being pruned during transmission? Can the secret INR still able to be retrieved?
3. I would be more interested in how robust this proposed method is against the attackers. Start from simple attacks like pruning/noises to the effort trying to reconstruct/retrieve the secret INRs.

[1] https://arxiv.org/abs/2212.01602

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good
