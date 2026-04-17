# Pixel-Perfect Puppetry: Precision-Guided Enhancement for Face Image and Video Editing

- Decision: Accept (Poster)
- Scores: 6, 2, 4, 8

## Abstract
Preserving identity while precisely manipulating attributes is a central challenge
in face editing for both images and videos. Existing methods often introduce visual artifacts or fail to maintain temporal consistency. We present **FlowGuide**,
a unified framework that achieves fine-grained control over face editing in diffusion models. Our approach is founded on the local linearity of the UNet bottleneck’s latent space, which allows us to treat semantic attributes as corresponding
to specific linear subspaces, providing a mathematically sound basis for disentanglement. FlowGuide first identifies a set of orthogonal basis vectors that span
these semantic subspaces for both the original content and the target edit, a representation that efficiently captures the most salient features of each. We then
introduce a novel guidance mechanism that quantifies the geometric alignment
between these bases to dynamically steer the denoising trajectory at each step.
This approach offers superior control by ensuring edits are confined to the desired
attribute’s semantic axis while preserving orthogonal components related to identity. Extensive experiments demonstrate that FlowGuide achieves state-of-the-art
performance, producing high-quality edits with superior identity preservation and
temporal coherence. Our code is available at: https://github.com/yl4467/flow_edit.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a diffusion-based framework designed for identity-preserving face attribute edits in both images and videos. 
By leveraging the local linearity of the U-Net bottleneck, the authors identify semantic subspaces (for identity vs. target attributes) as linear directions. During denoising, a novel guidance mechanism dynamically steers the generation along the target attribute axis, while preserving orthogonal identity components.

### Strengths
1. The paper addresses an interesting and relatively under-explored problem with diffusion models, which is local attribute editing of videos.
2. The proposed solution is both novel and interesting.
3. The paper is well-written and experiments are reasonably solid, showing fair quantitative metrics and adequate comparisons with prior work.

### Weaknesses
1. There are almost no citations and comparisons with prior diffusion-based video editing methods, even ones that support or are designed for face attribute editing, e.g. [1-4].

2. The visual results are not entirely convincing, e.g. videos tend to be blurry, sunglasses (in both image and video examples) appear blended into the face rather than realistic, and sometimes the edits change additional attributes in addition to the intended ones.

[1] RAVE: Randomized Noise Shuffling for Fast and Consistent Video Editing with Diffusion Models

[2] V-LASIK: Consistent Glasses-Removal from Videos Using Synthetic Data 

[3] IP-FaceDiff: Identity-Preserving Facial Video Editing with Diffusion

[4] Qffusion: Controllable Portrait Video Editing via Quadrant-Grid Attention Learning

### Questions
1. Did the authors evaluate or compare their method against other recent diffusion-based video editing techniques?
2. A discussion of limitations (e.g., visual quality issues and attribute leakage, see 2. above) would strengthen the paper.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a method to edit face images and videos using the latent space of diffusion models. The authors find a basis in the latent space of the U-Net of the diffusion model to carry out editing with and then apply a binary mask to choose between a reconstruction noise direction $\epsilon^r$ and an editing noise direction $\epsilon^c$.

### Strengths
The results of the paper look good, and the method makes sense. The presentation is well-structured and the writing is good. The adherence to the original identity of the person is good, which is an important point.

### Weaknesses
One of my main criticisms is that the whole idea of section 3.3, finding the basis vectors using an SVD of the Jacobian, is directly the same as in the work of Park et al 2023. The main difference as far as I can tell, is that finding the bases is done in twoo different situations (reconstruction and editing), but apart from that, the technique is the same. I do not think this is necessarily something problematic, but the presentation of it is. While the authors indeed cite Park et al, they do not clarify that the technique is basically the same. So they should either say why it is different, or present this section more succinctly and not as a novel innovation.

Furthermore, the bases found for editing are only used to establish the binary mask to choose between reconstruction and editing! This feels like a bit of a let down, because usually when we look for a basis, it is to use these basis vectors to carry out the editing (because they encode useful semantic directions), not just to choose between editing and reconstruction. So it seems like a lot of work for a quite small gain and difference with respect to the work of Huberman et al. The difference boils down to using a binary choice between reconstruction and editing, rather than a linear combination. I feel that this is quite hidden, and the basis vectors are presented as having a much greater and direct importance than they actually have. I wonder if the authors tried using the basis vectors to edit but maybe they did not work as well as hoped.

Consequently, I found it difficult to isolate what the actual contribution is of the paper. As far as I can tell, there are two aspects:
- separating the editing direction into reconstruction and editing, which is a good idea, but is inspired by Huberman et al 2024. (as the authors acknowledge)
- Applying a binary make to choose either reconstruction or editing, and establishing this mask with a dynamic criterion, based on the basis vectors
Presented like this, the novelty is much smaller.

### Questions
The section 3.4 is a bit confusing, especially around lines 263-269. You say that the directions of $\epsilon^c" and $\epsilon^r$ should remain consistent with similarity. To me, this means that, the more similar the bases are, the more similar the directions should be, and vice versa. But, if I understand correctly, what equation 6 does is the following:
- if the difference between $\epsilon^c$ and $\epsilon^r$ is smaller than $\lambda$, then the final direction is simply the reconstruction direction
- if the difference between them is high, then you take the editing direction
This is different from just saying "they should remain consistent with the similarity", which is extremely vague and could mean many things. As it is, the choice makes sense, but I feel it is quite poorly explained. It is true at the end of the paragraph you say "editing process only targets the desired attributes while maintaining ...", but this should come earlier and be clearer. In particular, I would put a clear interpretation of equation 6. Also, "dynamic threshold" is not really clear I find, since in equation 3, you never mentioned a "threshold", indeed this is something you propose.

Finally, there is a method which works extremely well at the moment for prompt-based image editing, which is not cited here:
- FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space, 2025
While I agree that this is recent (however, you have included other methods from 2025), it does represent the state-of-the-art, and should at least be mentioned. It produces extremely impressive results, and so cannot be ignored.

Also, I did not understand why you present this as a video editing problem: everything is done image by image, so that there is no video element here (no enforcement of temporal consistency etc). I think you should present it just as an image editing method.

Smaller details

- Page 3, line 124: you use $X_0=\{x_1,\dots, x_n\}$ to represent initial input frames. I find that this is a bit confusing, since you then apply a neural network to this whole list. I understand that it represents a batch, and that everything can be done independently over the elements of the batch, but I would just say let $X_0$ represent a frame, and then that the frames will be processed independently. Furthermore, in section 3.3, you use $n$ to represent the number of vectors in the bases you find, and this cannot be the same $n$. So just drop the initial reference to all of the frames (unless somewhere you need to reference different frames, but I could not find such a point).
- Page 3, line 133: $\alpha_t$ not introduced. It is standard in diffusion models, but you still need to introduce it.
- Page 3, line 133: what is $\epsilon^r$, not introduced. We deduce that it is the output of the neural network, but still, not introduced. You specify that $\epsilon^c$ is the predicted noise with the condition, but this comes after $\epsilon^r$.
- Page 4, line 198: "Since the extraction of the latent basis is identical for both $X^r_T$ and $X^c_T$...". This is a bit misleading, it implies that the same basis is found, which is not the case. Rephrase to say that the process of finding the two bases is identical.
- Page 5, line 263: "To ensure that the directions of $\epsilon^c$ and $\epsilon^r$ remain consistent with the similarity $\varepsilon$. I find this a little confusion, it is not obvious how vectors can be consistent with a scalar (see above comment). 
- Page 5, equation 6 : you do not state what $\hat{\epsilon}$ is used for. We understand that it is the final editing direction (I guess), but this is not stated.

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
This paper tackles challenge in face editing: achieving precise attribute manipulation while preserving identity and temporal consistency. The method, **FlowGuide**, proposed the ​**​geometrically-grounded guidance mechanism​**​, which is built upon a key observation: the latent space in the UNet bottleneck exhibits a locally linear structure. By modeling attributes as linear subspaces in the UNet's latent space and using the geometric alignment between original and edited subspaces to guide the denoising process, it achieves precise, localized edits while preserving identity and temporal coherence.

### Strengths
1.  Leveraging the geometric structure (linear subspaces) of the latent space for control is elegant for disentanglement, moving beyond heuristic guidance methods. 
2. The method can be applied to both image and video editing tasks.
3. Comprehensive quantitative and qualitative results are provided.

### Weaknesses
1. This method rests on the assumption of ​**​local linearity​**​ in the UNet bottleneck's latent space  (Park et al., 2023; Kwon et al., 2022). However, the nature and bounds of this "locality" are not defined. How large is the linear region? Is it consistent across different inputs, and different denoising timesteps? The method's robustness depends on this assumption holding true universally.
2. The claim of disentanglement is challenging to prove. The qualitative results, while good, still show that edits can affect non-target regions. For example, in Figure 3, when adding a smile, changes in skin texture and face attributes are visible, which are not strictly part of the "smile" attribute but are correlated with it. 
3. Equation 1 uses $θ$ subscript for $q_θ$, but $q$ is supposed to be the fixed forward process, not parameterized.



**References**

[1] *Mingi Kwon,et al. Diffusion models already have a semantic latent space. ICLR, 2023.*

[2] *Yong-Hyun Park,et al. Understanding the latent space of diffusion models through the lens of riemannian geometry. NeurIPS 2023*

### Questions
1. How do the authors quantify the degree of local linearity? What happens when this assumption breaks down for large attribute changes?
2. How many basis vectors are used? What is the impact of this choice of $n$ on editing quality?
3. How crucial is the dataset-specific fine-tuning of the diffusion autoencoder to FlowGuide's video editing performance? Would the method work effectively on video in a true zero-shot setting with a generic pre-trained model?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents a unified framework called FlowGuide for face image and video editing based on a novel guidance mechanism to achieve precise attribute control in diffusion models. FlowGuide consists of two main key contributions: a Latent Basis Extraction (LBE) module and an Implicit Basis Guidance (IBG) mechanism. As compared to previous approaches, this algorithm can avoid introducing distortions to other facial features, identity, or background elements. Linear subspaces within the UNet bottleneck’s latent space for semantic attributes were obtained using SVD, and the LBE module identify orthogonal basis vectors in this subspace to isolate the identity from the attributes in the latent space. The Implicit Basis Guidance (IBG) mechanism computes the geometric alignment between these bases using cosine similarity to preserve the original identity. Two kinds of experiments were conducted: face image editing and face video editing. Those experiments showed the validity of the proposed algorithm well, supported by enhanced performance and some ablation studies.

### Strengths
This paper is well structured and well organized. It was quite easy to follow.
The proposed methods are well motivated and mathematically sound.
Two experiments show the validity of the proposed algorithm and its generality. The experiments are comprehensive using multiple datasets and the proposed algorithm is compared to recent methods.
The proposed method is more time-efficient than the previous SOTA.

### Weaknesses
I understand the page is quite limited, but I would suggest adding some discussions to make this paper more attractive.
Limitations, failure cases, and discussions on the reasons are not well presented.

I am afraid that the videos are rather simple. How long and how much can the proposed method keep valid for longer videos and for videos with larger movements and facial expressions/occlusions?

### Questions
Face editing while preserving the identity would have many possible applications. I am curious how deepfake detectors would respond to images/videos with your approach and other previous approaches.

### Soundness
4

### Presentation
4

### Contribution
3
