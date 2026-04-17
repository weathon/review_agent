# Universal Approximation with Softmax Attention

- Decision: Reject
- Scores: 6, 8, 2, 8

## Abstract
We prove that with linear transformations, both (i) two-layer self-attention and (ii) one-layer self-attention followed by a softmax function are universal approximators for continuous sequence-to-sequence functions on compact domains. 
Our main technique is a new interpolation-based method for analyzing attention’s internal mechanism.
This leads to our key insight: self-attention is able to approximate a generalized version of ReLU to arbitrary precision, and hence subsumes many known universal approximators.
Building on these, we show that two-layer multi-head attention or even one-layer multi-head attention followed by a softmax function suffices as a sequence-to-sequence universal approximator.
In contrast,  prior works rely on feed-forward networks to establish universal approximation in Transformers.
Furthermore, we extend our techniques to show that, 
(softmax-)attention-only layers are capable of approximating
gradient descent in-context.
We believe these techniques hold independent interest.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proves that even a single-layer transformer can be universal approximators for sequence-to-sequence functions, given certain architectural assumptions.

### Strengths
1. The paper does explore to what extent scaling the depth is needed for universal approximation in transformers, showing that one can in fact have constant depth of 1 and instead scale the sequence (and possibly hidden) dimensions instead. They also achieve that without using MLPs, solely by leveraging the attention mechanism.
2. They authors show that one can trade-off input length for the number of heads and extend their results to sequence-to-sequence modeling.

### Weaknesses
1. The fourth contribution point (“In-Context Approximation and Gradient Descent.”) is not at all discussed in the main text. I understand that it is difficult to fit a theoretical paper in the page limit but there is nevertheless the expectation that the main text is self-contained.
2. I didn’t see a clear note of the scaling properties of their construction, i.e., how do the model input length, hidden dimension and number of parameters scale with the precision?
3. Overall, the constructions were not particularly clearly explained. Still not entirely clear to me what the $G$, $\tilde{k}_i$ and $k_i$ are supposed to be. While Figure 1 was supposed to help, I can’t say I understood much of what it is supposed to illustrate. The paper can really benefit from rewriting Section 3 for clarity. 
4. The paper appears to overclaim its novelty a bit. The authors say that prior works have “strong assumptions on data or architecture” but do not explain what these assumptions are. Furthermore, they don’t appear to clearly explain what their own assumptions are (e.g., expanding the input into a much longer input using the initial linear projections and possibly having unbounded activations to achieve “pickiness” of the softmax attention). A clear discussion of the limitations of the paper is missing.

### Questions
In Definition 2.1 $W_O$ has dimensions $n \times n_o$ which indicates it is mixing information across the sequence elements, when in practice it should operate element-wise. Is this a typo or a required property?

Typos:
- Line 117: Missing $X$ for $W_Q^\top X$?
- Line 197: these segments serves -> these segments serve
- Line 392: Possibly $R$ should be blackboard font?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
1

### Summary
This paper establishes universal approximation results for softmax attention mechanisms. The authors prove that (i) two-layer self-attention or (ii) one-layer self-attention followed by a softmax function can universally approximate continuous sequence-to-sequence functions on compact domains. They proposed an interpolation-based method showing that attention can approximate generalized ReLU functions with O(1/n) precision for single-head and O(1/(nH)) for H-head attention. The results are extended to in-context learning, demonstrating that attention can approximate gradient descent.

### Strengths
1. This paper provides novel theoretical insights. It’s the first work to discuss universal approximation for attention without FFN. The proposed interpolation-based technique is a unique view for the expressiveness of attention. 
2. The theoretical foundation is solid, and the only assumption needed is the compactness. The detailed proofs and discussion of multiple cases makes the proof comprehensive and strong.

### Weaknesses
This paper is a theoretical paper which lack of real data experiments like language modeling to validate the predictions.

### Questions
How does the results change or extend to other attention like causal or masked attention?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper revolves around the proof of a Universal Approximation Property (UAP) for Attention. Crucially, it makes a point of not relying (for the most part) on the additional FFN layers one would normally find in a Transformer, thus departing from previous work, and highlighting the expressive power of the sole Attention layer. In a nutshell, the proof is built by showing that one single single-head attention layer, prepended by a linear operator, can represent a Truncated Linear Model (ie, a ramp) token-wise: that is, given an input X\in\mathbb{R}^{d x N}, attention can mimic Range(wT x_n + t). This result leverages an interpolation-based technique, which is at the core of the paper itself. Notably, this Truncated Linear Model is a basic building block which can be used for constructing piece-wise linear functions, which already hints at its approximation properties. In particular, its functional class includes many universal approximators, including ReLU: the proof then proceeds by leveraging the universality of ReLU approximators, and transferring it to Attention. The authors also provide some simplifications / recasting of the original proof: particularly, a similar result is recovered for a single multi-head attention layer (which simplifies the requirements on the prepending linear layer), and the final ReLU approximation can be done with either two attention layers, or one layer and a softmax nonlinearity.

### Strengths
- The approach of the paper is interesting: particularly, the first step of the proof and main contribution (the interpolation-based approximation of the Truncated Linear Model) is insightful
- Also the scope of the paper is relevant, as it claims to dramatically simplify the requirements for UAP for Transformers

### Weaknesses
- The paper would benefit from a strong revision to improve its structure and presentation
- Many relevant discussions and results (including mentioned core contributions) are relegated to the appendix
- Most importantly, while the overall proof strategy is clear, I found it difficult to verify its validity, also in light of the reduced clarity

### Questions
**The main proof**

There is a number of hidden details in the proof which I believe affect its overall validity, and should be spelled out more clearly, also to better highlight possible limitations in the approach proposed
- **Prepending linear layers** I believe this proof mainly aims at describing the properties of Attention, *as it appears inside a Transformers architecture* (otherwise its usefulness would be rather reduced). If this is the case, then the shape of the linear Layer A(X) used in Thm3.4, is not a valid one: by looking at Fig1 (and in general at its defined shape in L205, A:R^{d\times n}->R^{… \times p}), the layer expands the input tensor sequence-wise, which is not a valid operation in Transformers. In all fairness, you yourselves point this out in AppB and Remark G3, and highlight Thm3.9 (the multi-head version of UAP for attention) as the real “valid theorem” for the sake of proving UAP in a Transformer setting. While indeed the A in L1721 is a valid token-wise operator, due to its bias term it’s a rather unorthodox one—but I agree it can be seen as a special PE. In light of all of this, I understand why you decided to include Thm3.9 in the main (where at a first read it looked like a secondary extension to Thm3.4, which didn’t need to take up so much space, and could be put in the appendix), but if what I said above is correct, then it seems like Thm3.9 should be the *only* theorem we really care about. If not, then at the very least the limitations of Thm3.4 should be more clearly spelled out in the main text.
- **From token-wise to full sequence-to-sequence approximation** Step 2 in Sec3.3 is the one I’m struggling the most understanding. From Thm3.4/3.9 you managed to build a *token-wise* approximation to a ReLU, ie Att(xn)\sim ReLU(xn) \forall n=1…N. But here you need to represent a ReLU which depends on the *whole* input sequence at once. This requires at the very least to collapse information from the whole sequence into a single token. By skimming through AppG5, it seems like indeed this is the role of the interweaving linear layers A1, A2, but (as we established before) such sequence-wise linear operations are not “valid” in a Transformer setting (from G.42, it looks like A2 is expanding the sequence length). I admit due to time constraints I couldn’t go through the whole detailed proof in the appendix, so I might be missing something, in which case please do correct me. Nonetheless, I reckon Thm3.14 should provide the “valid” version, relying on the same trick of swapping single-head with multi-head (as in Thm3.4 vs Thm3.9), but then in L472 you need linear layers Eij that act as (tensor) one-hot selectors: these are full tensor operators (acting both sequence- and token-wise). I understand these are mainly technicalities, but again, if we’re trying to answer the question of “would a Transformer be able to implement this mechanism in practice?”, then the answer would remain “no”.
- **Limitations and comparison with previous work** In L92 and AppE you make a point of separating your result from previous work, highlighting in particular how, contrarily to [Yun et al., 19], you don’t require a (large) number of stacked layers to achieve UAP. However, in your approach you’re effectively swapping depth-complexity for head-complexity. Even from Thm3.9 it’s quick to realise how the number of attention heads necessary to achieve the desired level of accuracy directly grows with this accuracy. This is (potentially) worse than the logarithmic-power growth highlighted in [Takakura and Suzuki,23]. This is a limitation of your method, and should be highlighted. Moreover, from a practitioner’s point of view, rather than sheer number of layers / heads / feature components, a comparison of the total number of parameters required to assemble the architecture required by your approximation would be more useful.


**Clarity**

I have many reserves regarding the choices made in structuring the paper. As mentioned above, the focus should be on the most relevant Thm3.9 and Thm3.14, as they claim to pertain to the “real” Transformer architecture, rather than the “faulty” Thm3.4 single-head counterpart: including them, rather than simplifying the exposition, ends up confusions it and—most importantly—occupying relevant space in the main text. While I appreciate the attempt to provide high-level intuition behind the proofs—as in Fig1, and in the step-by-step explanations—the additional fleshed out technical details in the main don’t provide much more information, and could be directly incorporated in the step-by-step descriptions. The description of the core result is interrupted abruptly, without leaving room for a much-needed discussion of its applicability and limitations, nor the experimental results to help ground the theory, and relegating both to the appendix. Because of this, the overall impression is that this work has gone through a hastily revision, more concerned with meeting the page limit than with providing a cohesive, self-contained story, and the overall final result lacks in structure and clarity.
	

**The role of the Appendix**

Please don’t treat the Appendix as a free pass to extend the paper. This is ultimately unfair towards the other authors, who struggled to fit their content within the page limit. Particularly, I’ve never seen before a result being listed as core contribution and yet never mentioned in the main paper: if the application of your technique to in-context learning is a relevant contribution, as you report in Abstract and Introduction, then AppB should fit in the main.

	

**Minor**
- The Truncated Linear Function you introduce in Def3.1 is normally called a ramp function. Imho, this name is much more common than the one you provide
- L83: “Our theory only assumes the compactness of the target function’s domain”: more strictly, it relies on regularity of the target function (Lem3.11 requires f continuous, for example)
- L1091 “For any continuous sequence-to-sequence function, removing the factorial term n! in the denominator leaves the remaining term proportional to a power of n” If the goal of this comment is to highlight that this approximation requires a large number of layers, this is not a valid way to go on about it: the factorial term is itself proportional to a power of n (see Stirling’s approximation), and it helps counterbalancing the numerator. The main issue is the (1/\delta)^n term

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
1

### Summary
The paper shows that attention-only architectures, with only linear transformations, achieve sequence-to-sequence universal approximation on compact domains. 
The key technique is an interpolation-based construction in which the softmax serves as a near-argmax selector over a fixed grid of anchors. This converts self-attention into a piecewise-linear generalized ReLU approximator, establishing that attention alone suffices for universal approximation. 
The paper also shows that multi-head attention improves the approximation rate to O(1/H) and extends the method to approximate in-context gradient descent.

### Strengths
1. This work isolates the expressive power of attention without feed-forward layers, a minimalistic configuration not covered by previous proofs.
2. This work formalizes "anchor selection" within attention, turning softmax into a continuous selection mechanism that generalizes ReLU behavior (Theorems 3.4 & 3.9).
3. Synthetic experiments confirm the predicted O(1/p) and O(1/H) error rates and visualize one-hot-like attention heatmaps.
4. The theoretical analysis in this paper is exceptionally solid with minimal reliance on unverifiable assumptions. 
5. The boundary conditions for the softmax–hardmax approximation (Lemma F.1) are explicit, and the relationships among parameters p, H, and beta are carefully derived.
6.  The interpolation-anchor construction and separation of interpolation and softmax errors demonstrate a high level of precision. 
7. The appendix provides complete supporting derivations.

### Weaknesses
1. Achieving small error requires large p (number of anchors) or large beta (inverse temperature). Constants in the asymptotic rates are unspecified, making practical implications unclear.
2. Experiments use synthetic data only.

Comment:
The construction (Theorem 3.4) rely on a sequence-wise linear map A that expands columns, which is the less practical form.
 I understand that this structure is primarily motivated by interpretability, so I do not consider it a serious limitation.
although a token-wise version (Theorem 3.9) exists, this is the less practical form.

### Questions
I understand that this paper achieves attention-only universal approximation through a research idea fundamentally different from previous studies on universal approximation for LLMs. 
Although I am not sure how practically meaningful it is to prove universal approximation for an attention-only model, I believe the technical contribution is substantial. From that perspective, I would like to ask the following questions.

Q1.
I understand that demonstrating attention-only universal approximation is valuable because it reveals the intrinsic expressive power of the attention mechanism itself. However, if there are other intended meanings or motivations behind proving attention-only universal approximation, could you please elaborate on them?

Q2.
I understand well the general limitations of constructive universal approximation theorems, and this paper shares the same type of limitations as prior works, for example, being an existence proof that does not necessarily imply the model can be learned via gradient descent.  Also, in this paper, attention uses a temperature parameter to make softmax behave like a hard max; this approach is not unique to this work and appears in other studies as well. Nevertheless, hard-max operations are problematic when applying gradient-based optimization. Since some existing universal approximation theorems avoid using hard max, could you clarify the conceptual or technical significance of proposing a proof that explicitly depends on hard-max behavior?

Q3.
In the attention + feed-forward (FFN) configuration, attention acts as a context mapping, while the FFN plays the role of memory and recall. This decomposition is intuitive and is consistent with known interpretations that link the FFN to memory mechanisms like key-value memory, independent of universality. Demonstrating that attention alone can approximate ReLU networks clearly shows its expressive capacity. However, does your analysis uncover any new property or potential role of attention beyond its known expressive power? 
Could you explain what new conceptual insight about attention arises from your construction?

Q4.
Improving approximation accuracy in your construction requires substantial growth of the key parameters, namely the inverse temperature beta, the number of interpolation anchors p, and the number of heads H. This scaling seems far from what realistic Transformers typically employ.
Could you provide at least a rough quantitative estimate or empirical sense of how large these parameters would need to be to achieve a reasonable approximation accuracy (e.g., a target error epsilon)? Even an order-of-magnitude discussion would help clarify how far the theoretical regime is from practical parameter ranges.

### Soundness
3

### Presentation
3

### Contribution
3
