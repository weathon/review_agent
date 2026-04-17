# Test-Time Poisoned Sample Detection by Exploiting Shallow Malicious Matching in Backdoored CLIP

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
CLIP, known for its strong semantic matching capabilities derived from large-scale pretraining, has been shown to be vulnerable to backdoor attacks in prior work. In this work, we find that such attacks leave a detectable trace. This trace manifests as a divergence in how image features align with the CLIP's text manifold where semantically similar texts cluster. Specifically, benign images exhibit *deep benign matching*, where their features are close not only to the predicted text caption but also to the broader manifold of semantically equivalent variants of that caption. In contrast, poisoned images display *shallow malicious matching*, where their features shallowly align with the specific target caption but remain distant from its semantic neighborhood. Leveraging this insight, we propose **Subspace Detection**, a novel test-time poisoned image detection method against backdoored CLIP. First, for a test image, we approximate its corresponding local text manifold by constructing a low-dimensional subspace from semantically equivalent variants of its predicted text. Second, within this broad subspace, we probe a region-of-interest that maximally amplifies the separation between the two types of images: benign images remain close due to deep matching, while poisoned images deviate significantly due to shallow matching. Finally, we identify whether the test image is poisoned by measuring its deviation from this region; a large deviation indicates a poisoned image. Experimental results demonstrate that our method significantly outperforms existing detection methods against SoTA backdoor attacks and exhibits robust detection performance across multiple downstream datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper finds that images with a hidden backdoor ("poisoned" images) in a compromised CLIP model show a detectable flaw called "shallow malicious matching." While these images are forced to match a specific target text caption, they do not align with the broader meaning of that caption. In contrast, normal "benign" images show "deep benign matching," meaning their features are close to the whole group of text features that share the same meaning. Using this discovery, the authors propose "Subspace Detection," a method to spot poisoned images during testing. It works by checking if an image's feature is far away from the semantic neighborhood of its predicted text; a large distance indicates a poisoned image. Experiments show this method works better than existing defenses across different attacks and datasets.

### Strengths
1. Novelty. The paper's core insight is somewhat original. The concepts of "shallow malicious matching" versus "deep benign matching" offer a novel and intuitive lens through which to understand and detect backdoor attacks in vision-language models. The proposed "Subspace Detection" method, which leverages the geometry of the text manifold, is a creative and non-obvious application of this insight.

2. Clarity: The paper is well-written and structured. The central concept is introduced clearly in the abstract and introduction, and the illustrative Figure 1 effectively reinforces the explanation of "shallow" vs. "deep" matching. The description of the three-step "Subspace Detection" method is logical and easy to follow, making the technical contribution accessible even to readers not deeply versed in backdoor literature.

### Weaknesses
1. The paper should compare its method more directly with BDetCLIP. A clearer side-by-side comparison would help readers understand the specific improvements and differences.

2. Testing the method against more types of advanced multimodal backdoor attacks (beyond BadCLIP) would make the results stronger.

3. It would be good to test if the idea works on other multimodal models, not just CLIP. If it does, it would show that the finding is more general and important.

### Questions
What is the underlying cause of shallow malicious matching? Could you elaborate on the mechanism behind shallow malicious matching?

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
The paper proposes Subspace Detection, a test-time detection of poisoned images in backdoored CLIP models. The key observation is that clean images align deeply with the local semantic manifold of their class text representations, while poisoned images only align superficially to the target caption and deviate from semantically equivalent variants. Based on this, the method measures the image’s average distance to sampled text features for poison detection.

### Strengths
1. The identification of the contrast between shallow malicious matching and deep benign matching offers a useful perspective on how poisoned images behave in CLIP embedding space.

2. The proposed subspace-based detection framework is conceptually clear and methodologically coherent.

3. The method works at test-time and does not require model retraining or access to original training data, which increases practical applicability.

### Weaknesses
1. Not enough baseline for comparisons. Methods like [1] should be included in baseline.

2. The transformation ablation shows “more transforms help,” but the paper doesn’t analyze why some single transforms (e.g., description vs. font) differ so much, nor the minimal set needed to reach near-max performance. A short analysis would help the clarifications.

3. Thresholding is set via a clean reference set; sensitivity to the clean-set distribution and false-positive trade-offs is not fully explored.

4. Margins between clean and poisoned can be small, making the differences between shallow vs deep matching not as significant.

5. Lack of theoretical supports for the shallow malicious matching vs deep benign matching. 

6.The method is not evaluated against label-consistent backdoor attacks or cases where poisoned images match multiple semantically similar target prompts. It is unclear whether the shallow-vs-deep matching assumption holds in these scenarios.


[1] Niu, Yuwei, et al. "Test-Time Multimodal Backdoor Detection by Contrastive Prompting." Forty-second International Conference on Machine Learning.

### Questions
See weaknesses.

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
This paper introduces a test time backdoor image detection method for CLIP pretrained with backdoor samples. The proposed method is based on the hypothesis that backdoor samples are further away from the region of composed of semantic variants of the target text, comparing to benign samples. Thus the proposed detection constructs subspaces with variants of the target text and compare the distance between the test sample and text features sampled from the subspace, to the max of such distances computed from a set of benign samples.

### Strengths
1. The paper is easy to follow, and the proposed detection is well motivated by empirical evidence and prior works.
2. The proposed detection method performs significantly better than baselines considered with respect to selected metrics. (although BDetCLIP is not evaluated, see Weaknesses)
3. A variety backdoor attacks are evaluated. (although clean label attacks is not evaluated, see Weaknesses)

### Weaknesses
1. minor: line 25 "board" -> "broad".
2. While the paper cites BDetCLIP as a test time backdoor detection method, it does not compare with it as a baseline.
3. I appreciate that the paper included a time complexity section. However, the section does not compare the detection overhead with CLIP inference time. Additionally, the subspace detection.
4. Based on Table 4, Language alone performs better than Language + Description. The conclusion "This indicates performance improvement mainly arises from the synergy among transformations rather than a single transformation alone" does not appear very obvious.

### Questions
1. How is the shallow matching of backdoors impacted by the poison rate and training dynamic like the number of training epochs.
2. How does the proposed analysis detection perform on clean label attack? I think this evaluation is especially important, since it could affect the distribution of the backdoor samples semantically.
3. How important is the diversity of the reference set with respect to the number of distinct semantic concepts?
4. How important is the ratio between the number of text variants sampled from each augmentation category?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the vulnerability of backdoored CLIP models and identifies a phenomenon termed shallow malicious matching. Building on this insight, the authors propose Subspace Detection, a test-time poisoned sample detection method that constructs a low-dimensional subspace of the CLIP text manifold using semantically varied text transformations to distinguish benign from poisoned samples. Experiments demonstrate its effectiveness.

### Strengths
1. This paper is original in its formulation of the shallow malicious matching phenomenon and its introduction of a subspace-based detection framework for identifying test-time poisoned samples in multimodal models like CLIP.
2. This paper is well-structured, with clear motivation.

### Weaknesses
1. Lack related baselines. There are some works focusing on test-time detection, such as [1].
2. The motivation underlying the mixture of Gussian model is unclear. The author should give detailed formulation to explain.
3. In ablation study of time $L$, the performance does not reach the peak at $L=4$. More results on a larger value of $L$ should be reported.
4. The adopted distance metic, $d_2$, needs more consideration and discussion.



[1] Test-Time Multimodal Backdoor Detection by Contrastive Prompting.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
