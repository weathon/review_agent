## Human Reviewer 1

### Summary
This work presents a novel image generation model that utilizes channel-wise quantization to convert image features into discrete codes along the channel dimension, adopting a masked prediction paradigm for image generation. This approach offers efficient modeling of image structures and strong representational capacity, outperforming or matching state-of-the-art methods on the ImageNet benchmark. The authors also performed the text-to-image generation and demonstrated transferability to text-to-image generation on the COCO dataset. The contributions of this work include a simple yet effective visual tokenizer with 100% codebook usage and a generation framework based on channel-wise quantization for image generation tasks.

### Strengths
This paper is easy to follow and includes comprehensive experiments to demonstrate the effectiveness of the proposed method. It proposes novel channel-wise quantization, which offers efficient modeling of image structures and strong representational capacity. It also proposed a simple yet effective tokenizer with 100% codebook usage. These features enable the whole image generation framework to achieve superior or comparable performance to state-of-the-art methods across various image generation tasks.

### Weaknesses
1) This paper's biggest contribution is proposing a channel-wise quantization for image generation. However, channel-wise quantization is widely used in various applications, such as classification [a], LLM compression [b], and super-resolution [c].

[a] RDO-Q: Extremely Fine-Grained Channel-Wise Quantization via Rate-Distortion Optimization. ECCV 2022.

[b] OutlierTune: Efficient Channel-Wise Quantization for Large Language Models. Arxiv 2024.

[c] DAQ: Channel-Wise Distribution-Aware Quantization for Deep Image Super-Resolution Networks. WACV 2022.

2) Why not combine SD series for comparison? For example, SD1.5, SDXL and SD2.1.

3) More experimental results about high-resolution image generation should be provided to demonstrate the effectiveness of the proposed method.

4) The authors did not evaluate the text-image alignment between the text prompts and the synthesized images. 

5) It is not reasonable to develop a channel-wise tokenizer, which is limited to one specific image resolution. One general channel-wise tokenizer algorithm that could be suitable for all image resolutions is favored. Such limitation heavily limits the contribution of this work.

6) The authors only conducted experiments on MS-COCO and ImageNet datasets with the image resolution 256*256. 

7) Marginal performance. Compared with VAR, the proposed method only shows very marginal performance improvement (or even a slight performance drop) under the same experimental setting: similar network parameters and inference step from Table 2.

Minor issues:

1) The best results under the same setting in Table 2 should be bold. 

2) Figure 2 is a little blur and seems that some parts are screenshots of other images.

### Questions
What is the purpose of the image generation? 

Only evaluating the image quality of the synthesized images based on evaluation metrics like FID, SSIM and PSNR is not enough. Can the synthesized images promote the downstream visual perception performance such as classification?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 2

### Summary
In this paper, the authors propose a new image tokenization method called channel-wise quantization, which quantizes image feature along channel. Then, based on the learned tokenizer, the authors use masked-prediction paradigm similar to MaskGIT to generate images. Experiments show that the proposed method achieves competitive performances compared to other image tokenizers and generative models. Besides, the proposed method can reaches 100% codebook usage under different codebook size.

### Strengths
+ The writing is clear and easy to follow.
+ The idea of channel-wise quantization is novel and interesting.
+ The reconstruction ability (especially rFID and SSIM) is greatly improved compared to spatial tokenizers.

### Weaknesses
- The motivation of proposing channel-wise quantization is not very strong to me. While spatial tokenizers often suffer from low codebook usage and reduced code embedding dimension limits expressive ability, it's unclear how these issues lead to the design of channel-wise quantization.

- The paper lacks a detailed analysis of how the channel-wise tokenizer behaves differently from spatial tokenizers. For example, the authors claim that channel-wise tokens capture both global structures and local details, but there are no direct experiments to support this. It would be interesting if the authors could visualize the learned channel-wise tokens and discuss each channel's representation, so that we can have a deeper understanding of the channel-wise tokenizer.

- The authors attribute the 100% codebook usage to the nature of channel-wise quantization. However, I note that entropy regularization, which is known to be helpful for increasing codebook usage, is adopted in codebook learning. Additionally, the compared method LlamaGen did not use entropy regularization. Thus, I'm not sure if channel-wise quantization is the main factor behind high codebook usage.

### Questions
+ As mentioned in weaknesses part, will the channel-wise quantization still reach 100% codebook usage without entropy regularization?

+ What does a channel token typically represent? Is it possible to visualize the learned channel tokens?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper presents an alternative to standard VQ-VAE, which typically quantizes each spatial position as a token. Instead, this work quantizes image features along the channel dimension into discrete codes.

For comparison, in a standard VQ-VAE with a final feature dimension of C*H*W, there would be H*W tokens, whereas this work produces C tokens.

Advantages:
1. The paper is clearly written and easy to understand
2. The metrics appear reasonable

Disadvantages:
1. Channel-wise quantization lacks theoretical justification and Intuitive rationality.
   Without spatial-based quantization, the model training appears to lose its causal nature
2. As the model scales up, the number of tokens would need to increase, potentially making learning more difficult
3. The resulting tokenizer becomes incompatible when image resolution changes

I believe the motivation behind this idea is fundamentally flawed, which leads to several issues:
- Limited generalizability
- Sequence length problems
- Modeling methodology concerns

Therefore, I lean towards a negative assessment of this work.

### Strengths
Advantages:
1. The paper is clearly written and easy to understand
2. The metrics appear reasonable

### Weaknesses
Disadvantages:
1. Channel-wise quantization lacks theoretical justification and Intuitive rationality.
   Without spatial-based quantization, the model training appears to lose its causal nature
2. As the model scales up, the number of tokens would need to increase, potentially making learning more difficult
3. The resulting tokenizer becomes incompatible when image resolution changes

I believe the motivation behind this idea is fundamentally flawed, which leads to several issues:
- Limited generalizability
- Sequence length problems
- Modeling methodology concerns

Therefore, I lean towards a negative assessment of this work.

### Questions
see above

### Soundness
1

### Presentation
3

### Contribution
1

### Rating
3

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper provides an interesting perspective on image compression through learning a vector quantized autoencoder. Typically, we tokenize images spatially. That is, every (i,j) for i over h and j over w within an hxwxc latent space, each position is mapped to a token in a learned codebook. In contrast, this paper proposes to tokenize along the channel dimension of the latent space within the AE. This results in tokens that capture the global image along a latent channel dimension. Results are conducted on class conditional and text conditional image synthesis with various ablations and analysis.

### Strengths
- The paper tackles an interesting approach to tokenize along the channel dimension within a VQ AE for images.
- The paper is well-structured and written.
- The approach is easy to understand and to follow.
- Comparison against many relevant models (albeit not exactly fairly if I understood correctly; see below for details).

### Weaknesses
My main concerns are regarding fair evaluation wrt compression ratios, more analysis on various token dimensions and exploration of usage of channel wise tokenization in downstream diffusion and AR tasks.

- Since each token in channel space captures complete global information, configurations such as [4,64,64] result in only four tokens of size 64x64 each, rather than 64x64 tokens of size 4. This impacts discussions on code embedding size and codebook use, which are misleading because they affect the compression ratio. I suggest comparing against a fixed compression budget with various compression ratios rather than focusing solely on code embedding sizes (which does not give the full picture).
- Adding to the above, the token count in Table 5 is potentially misleading, as the models use different compression ratios. If I understand correctly, the comparison is between VQAN using 4x16x16 tokens (4x256) and the proposed model with 256x16x16 tokens (256x256).
- The ablation study for token dimensions is incomplete. Experiments within the range [8, 256] are needed to understand scaling behavior better.
- Tokenizing *global information* along the channel dimension implies a significant correlation among tokens, meaning each image is learned as a whole rather than in parts. Consequently, *C* tokens per image are kind of memorized, making it difficult to repurpose tokens for other images due to global encoding. This also explains the high overall usage of the codebook, as tokens are not easily reusable.
- Adding codebook size, embedding dimensions, and compression ratios to the tables would improve comparability.
- In Table 5, why is rFID significantly better while PSNR is worse, even though SSIM is better? I would expect a consistent trade-off between perceptual- and pixel-wise metrics.
- Exploring channel-wise quantization in autoregressive (AR) or diffusion tasks could be insightful. I assume it may not perform as well for AR tasks, as the AR function would need to predict the entire global image in one step rather than progressively building it up from parts.
- The claims regarding low similarity between channel tokens, efficient modeling of image structure, and strong representational capacity lack clear definition and verification.
- I have ignored going into details regarding quantitative results mainly because the issue of fair evaluation is not cleared yet. As of now, sometimes the model is better, sometimes worse, and it is unclear why and when one would want to choose this method over the spatial tokenization.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4