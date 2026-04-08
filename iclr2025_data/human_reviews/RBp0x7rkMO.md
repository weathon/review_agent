## Human Reviewer 1

### Summary
This paper proposed a patch-based approach for learning a discrete codebook for a set of mini strokes/shapes, and then using an autoregressive transformer that operates on this representation to perform "SVG" reconstruction and generation.

### Strengths
- The overall pipeline design looks okay to me. It make sense to use a codebook to represent a set of instruction token, and an autoregressive transformer is a good way to interact with such a set of tokens. The design choice for encoding the images and the text prompts also are reasonable.
- Results look reasonable on the set of data shown, with considerable discussions. 
- Code is available, and appears to be quite reproducible.

### Weaknesses
I am slightly out of touch with this class of problems recently, so I some of the stuff I mention below might not be that accurate or up to date. Would be happy to update that following clarifications. With that being said, I didn't get much out of this paper - both the problem studied and the technique used are far more well studied than what is depicted in this paper. I am aware of many approaches that are similar in terms of core ideas (autoregressive generation of 2D/3D content represented as a set of discrete instructions/parameters), many datasets that are far more complex than what is used in this paper, as well as multiple works that seem to provide better solutions to this problem (and not compared):

- Despite the claims in 2.1 about this work being different from other SVG generative models (and other CAD/sketch/3D content generation pipelines that follow similar idea of representing content with a set of tokenized instructions, none of which cited), this work is not doing anything that is that much different from these works. Use of vector quantization (e.g. SkexGen: CAD Construction Sequence Generation with Disentangled Codebooks) or grid based (quantized) features (e.g. ShapeFormer: Transformer-based Shape Completion via Sparse Representation) are both standard practices (just naming the first paper that came on top of my mind). In some sense, the paper is even a regression from some of these works, where the so called "SVG" representation used in this paper is not really SVG, but a patch wise curve/closed shape representation that lacks the structure of actual vector drawings.
- Despite the claim about lack of vector data, datasets like SketchGraph and DeepCAD offer more than enough data in vector/discrete formats, and datasets like Quick, Draw! (as cited briefly in this paper as well) already has a vector representation. Moreover, there are so many off the self approachs for vectorizing drawings (e.g. Deep Sketch Vectorization via Implicit Surface Extraction). With the vast abundance of such data, it just doesn't make sense to use the pseudo "SVG" representation in the paper, which has few of the benefits of actual vector drawings.
- Following the previous two points, the baselines used in the paper is both insufficient and inappropriate: of course the more wholistic representation used by Im2Vec  (which was also pretty ancient by the standard of current vision/graphics) won't have as much representation power as collection of patch-based strokes. However, it at least has a single, wholistic representation of the entire SVG, and thus is not only more flexible, but also allows various applications e.g. interpolation, editing, that is hard to achieve with the proposed method. Therefore, just showing that Im2Vec fails on some cases don't really make me buy into the usefulness of the proposed method: if the goal is to show "generative capabilities", then compare against many more recent SVG/sketch/CAD generation works; if the goal is to show that this works enables better raster-to-vector conversion and thus removes the need of curated dataset (of which there are many), then there's also many much more recent works to compare against; if the goal is to show that this patch-wise representation is a good representation, then more analysis should be done on how this is a good representation (and I don't think so, both from an intuitive level and by examining the various visual artifacts present in the qualitative examples) that is useful in practice (i.e. having at least some unique capacities of vector graphics, as opposed to just an image), then more analysis is needed (editing, interpolation, user studies, etc.).

### Questions
As mentioned in the weakness section, this paper really needs to be positioned more properly against (more recent) works on raster-to-vector, token-based 2D/3D generation, and the abundance of vector graphics/CAD datasets. I currently do not buy into the utility of this patch-based mini stroke/shape representation, that is far from a true SVG representation. Clarifications over these points will help sway me - after all, there is nothing that is fundamentally incorrect about the approach. I might also be wrong with some of my claims - am a bit out of touch with this topic, as mentioned, so please do point out anything that I missed.

### Soundness
3

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
The paper addresses the problem of vector graphics image generation and proposes a novel approach to generate vector graphics images conditioned on text by using raster image supervision. The vector graphics image format is SVG, which is built using cubic bezier curves.

Existing gaps that motivate this paper are two fold: (a) Lack of abundant SVG data for training generative models for vector graphics generation, and (b) the difficulty of incorporating visual attributes such as color and strokes.

To address these challenges, the paper introduces a method to train a generative vector graphics model using just the raster images, eliminating the need for ground truth SVG training data. The proposed methods is a two-stage approach. The first stage decomposes an input raster image into primitive patches and learns a discrete codebook that maps each patch to a code. In the second stage, an auto-
regressive text-conditioned transformer is used to learn joint distribution of textual description and codes. The loss functions used are: (1) Reconstruction loss to reconstruct primitive strokes and shapes, (2)  Geometry loss for better placement of Bezier curve points,
(3) Causal loss for auto-regressive transformer resulting from causal multi-head attention in the decoder. As probably understood, this is a strongly supervised method. To summarize, the input to the method during training is an RGB image and a textual description of that image, while during inference, the method only takes a textual of an image we want the learned model to generate. At the output, we have a generated vector graphics image in a SVG format. The representation of SVG is Cubic Bezier curves, with control points and a starting point. The underlying modeling tools are vector-quantized auto-encoder for image encoding with ResNet as image encoder, DiffVG as differentiable rasterizer, and BERT to encode text, and an auto-regressive Transformer to generate strokes and curves in an auto-regressive fashion. The datasets used for experiments are: MNIST, Fonts, FIGR-8. In terms of evaluations, the paper looks at the following metrics: MSE, FID score and CLIP score.For comparisons, the method compares against one prior work from zcvpr 2021, Im2Vec.

### Strengths
1) The idea of using raster image supervision is novel

2) Effectiveness of the proposed method is shown with extensive experiments and ablation
studies

3) Usefulness is shown on multiple use cases (a) text-conditioned generation, (b) text-
conditioned icon completion (This task was not possible with previous methods. The
proposed framework made it possible), and (c) ability to reconstruct vector images with
visual attributes such as color and stroke width.

Overall, the paper proposes a novel idea to use raster image supervision to train a generative
model for vector graphics images. It uses vector quantized autoencoder to learn a codebook for
primitive strokes+shapes, and then trains an auto-regressive transformer model to learn a joint distribution on
textual description and input images, as represented by codebook trained in the earlier stage. The
paper also puts a good effort in making extensive experiments, although comparisons could be made stronger (see next section for comments),  to demonstrate the usefulness of the proposed method.

### Weaknesses
1) The paper could have been written in a better way to make it easier to understand.
Please see the following section for improvement suggestions

**Writing improvement suggestions:**

- The equation on line 265 (right side) does not make sense. I guess there is a spelling
mistake. Please number the equations so that it is easier to refer to.

- It would be helpful to have one more diagram explaining the stage-1 in more detail.
Figure-2 shows the steps at a very abstract level and it does not include all the steps e.g.
input and output of the model is not shown, the Geometry Loss mentioned in line 265 is
missing in the figure.

- The “Related Work” section in its current form is too elaborate and can be cut down by
writing it succinctly

- The “Method” section includes proposed method as well as the experimental details
together making it harder to read and understand. The experimental details such use of
contour finding algorithm (on line 215), variants used for $\Phi_{points}$ (on line 234-
236), the number of trainable parameters (on line 269), etc., can be factored out and
written in a separate section named “Experimental Setting”


2) Only one baseline (Im2Vec) was used for experiments. How about comparing to DeepSVG (from NeurIPS 2020) and ICON SHOP (from SIGGRAPH Asia 2023)? This would have made the evaluations thorough and helped get a better understanding of where the proposed approach stands, and what other gaps remain.

### Questions
* Are the results shown in Tables 1,2,3 and Figure 4,5,6,7,8, on the train set or the test set?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper presents a method called GRIMOIRE for generating scalable vector graphics. The authors first train a visual shape auto-encoder that maps 2D images into a discrete codebook. Specifically, each images is divided into patches and then encoded and quantized into a set of discrete tokens, and then the decoder transforms the latent codes back into the latent space, followed by a neural network to predict the control points of the Bezier curves. By using the differentiable rasterization, the Bezier curves can be rendered back into 2D images for end-to-end training. Then, the second stage involves learning an auto-regressive transformer to model the joint distribution over text, positions of stroke tokens. The authors did the experiments on three different datasets, including MNIST, FONTS, and FIGR-8. Results demonstrated that the proposed auto-encoder can reconstruct the images well and the generator can generate good results conditioned on text prompts.

### Strengths
- The authors propose an image-supervised method to generate vector graphics, by fitting this task into a common used generation framework: vector-quantization auto-encoder with auto-regressive distribution learning. To assist auto-decoder learning, the authors also included novel components such as geometry losses.
- The authors have conducted detailed experiments to validate their method, both in terms of reconstruction and generation quality. The results also clearly showed that the proposed method can excel in the selected datasets compared with other baselines.
- The authors released the code and detailed instructions for great reproducibility.

### Weaknesses
- This paper focus on a relatively smaller problem: generating vector graphics on small datasets. The experiments are mostly done on simple datasets and I haven't seen more complex results. Compared with SDS-based methods, the proposed method are limited by the use of datasets and cannot produce more diverse results.
- Although the authors successfully applied the VQ-framework for generation, I think in terms of contribution, it's more like utilizing existing experience for some new tasks, so the technical contribution is not that significant for a higher score.

### Questions
- The description of auto-encoder part (Line 219 to Line 232) is a bit unclear for me, with several math equations omitted according to my understanding. First, the latent variable $z_i$ is of dim $d \times \xi$, then "each of the $\xi$ codes is projected to q dimensions", does this mean the results are $d \times q$ and there are $\xi$ codes for each patch? What is the shape of the codes $v_i$? How does the "transformation ensures that each unique combination of quantized values is mapped to a unique code"?
- How to make sure that strokes from different patches are connected to each other?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper focuses on the Scalable Vector Graphics (SVG) generation task. A text-guided SVG generative model, i.e., GRIMOIRE, is proposed to learn to draw vector graphics with only raster image supervision. The proposed model, which is formulated as the prediction of a series of individual shapes and positions, consists of two modules, one is a visual shape quantizer (VSQ), and the other is an auto-regressive transformer (ART). The effectiveness of the proposed method is demonstrated on both closed-filled shapes data and outline strokes data.

### Strengths
The whole pipeline considers the vector graphics generation into two stages. The first VSQ stage can be trained under raster supervision, solving the issue that existing works often require direct supervision from SVG data. The second ART stage makes it possible to apply the proposed method to support several applications, including text-to-SVG generation and SVG auto-completion. The experimental results demonstrate the effectiveness of the proposed method.

### Weaknesses
The main weaknesses are listed as follows.

1. The technical contribution of this paper is not very clear. Although the proposed method can be trained with only raster image supervision, the core modules, including DiffVG and VQ-VAE, come from existing works.

2. The proposed pipeline adopts different strategies for the MNIST digits and FIGR-8. I was wondering about the results using the same segmentation strategy, e.g., creating patches for both closed-filled shapes and outline strokes.

3. The current evaluation is not thorough. Either SVG reconstruction or text-to-SVG generation has been studied for several years. But this paper only compares the proposed method with one existing work, i.e., Im2vec, making the results less convincing.

4. For the ablation study, I cannot find any quantitative comparisons of the training losses for the proposed pipeline.

5. Although this paper states that some of the existing works are computationally expensive and impractical in real-world use cases, there are no detailed computational cost or running time analyses.

6. Minor issues. For the loss equation shown in L264-L266, the geometric loss should be added to the reconstruction loss, rather than itself.

### Questions
1. For closed paths, the current evaluations are only conducted on the MINST dataset. I was wondering about more comparisons on EMOJIS and ICONS datasets, similar to the experiments in Im2Vec.

2. How to determine the number of patches that should be used for closed-filled shapes?

3. How to determine the weight in the geometric loss?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4