# Watermarks for Language Model via Probabilistic Automata

- Avg Score: 2.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2

## Abstract
A recent watermarking scheme for language models achieves distortion-free embedding and robustness to edit-distance attacks. However, it suffers from limited generation diversity and high detection overhead.
In parallel, recent research has focused on undetectability—a property ensuring that watermarks remain difficult for adversaries to detect and spoof.
In this work, we introduce a new class of watermarking schemes constructed through *probabilistic automata*.
We present two instantiations: (i) a practical scheme with exponential generation diversity and computational efficiency, and (ii) a theoretical construction with formal undetectability guarantees under cryptographic assumptions. Extensive experiments on LLaMA-3B and Mistral-7B validate the superior performance of our scheme in terms of robustness and efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper uses probabilistic automata for LLM watermarking.

### Strengths
They claim to have higher diversity than the watermark that simply biases all text towards a fixed set of polynomially many strings.
The experiments might be promising, I did not really look at them.

### Weaknesses
I cannot figure out what their scheme is actually doing. For the parts I was able to understand, there are serious flaws (see "Questions").

I do not understand how they can claim to achieve undetectability, when their detector appears to compute edit distance between the given text and a polynomial-sized list of strings from the key. These cannot both be true: If the generations have some correlation with a polynomial sized list of strings, then they are not pseudorandom.
Moreover, LPN or sparse parities are not mentioned anywhere in the description of their scheme. This seems like a big problem given that their entire claim of undetectability rests on LPN.

If there is actually something useful about the probabilistic automata perspective, I do not know what it is. The only paragraph I found that explains how the probabilistic automaton is used for watermarking is this:

"The automaton begins at an initial state q0 and terminates at a final state qf , progressing through |V|
layers that each encode a binary vector µi
. The first layer starts with: q0 → σ1,1, and each layer
proceeds through intermediate bitwise states: σi,j → σi,j+1, for 1 ≤ i ≤ |V|, 1 ≤ j < b, where σi,j
encodes the j-th bit of µi
. At σi,b, the automaton branches into two parallel Boolean paths: σi,b →
ιi,b+1, σi,b → ˆιi,b+1, which continue as: ιi,j → ιi,j+1, ιi,j → ˆιi,j+1, ˆιi,j → ιi,j+1, ˆιi,j → ˆιi,j+1,
where ιi,j = 0 and ˆιi,j = 1 represent bit encodings of µi
. Between layers, transitions connect the
terminal states of layer i to the initial states of layer i+1: ιi,c → σi+1,1, ˆιi,c → σi+1,1, for b ≤ j <
c, and the automaton concludes after the final layer with ι|V|,c, ˆι|V|,c → qf ."

### Questions
1. It appears that your undetectability proof is incorrect. You define undetectability in the natural way, following Christ et al., but then in your proof you say "Suppose there exists an algorithm A that KL-PAC-learns the class of such distributions..." and show a "decision-to-search" reduction, which is backwards! It is not difficult to give a correct search-to-decision reduction for sparse LPN, but it requires more care. The argument is slightly non-trivial, you have to re-randomize / use random self-reducibility.

It is possible that you are attempting to use Proposition 5 to bridge this issue. Proposition 5 is conspicuously missing a proof or a citation. That's for a very good reason: It's false! For instance, consider the case where F is a set of functions indexed by pseudorandom function (PRF) keys k which do the following: If the first bit of the input is 0, output 0; otherwise output PRF_k(x) where x is the remainder of the input.
Clearly this class is not efficiently PAC learnable, because on half of all inputs you only have a negligible advantage in guessing the output. But it's also clearly not a weak PRF, because it's trivially distinguishable from random.

Taking a step back, your construction is essentially that of Christ and Gunn, and your formulation in terms of weak PRFs is exactly the generalization given by Golowich and Moitra. In both of those papers, correct proofs of pseudorandomness are given from well-known assumptions. So you're using the exact method of those papers for the exact same purpose (constructing binary pseudorandom codes).
This raises the question: _Is there any reason why you state your results in terms of this particular scheme, rather than stating it more generally using an arbitrary pseudorandom code?_ That would make your method simpler to understand as well as more modular---then if new pseudorandom codes are found, they could be plugged in to the rest of your scheme to make it stronger.

2. There appears to be a fundamental misunderstanding about the meaning of "negligible." The paper correctly states that "A function is called negligible if it becomes asymptotically smaller than the inverse of any polynomial," but then immediately it says that "negligible functions can be expressed as negl(\lambda) = O(1/poly(\lambda))." There is a subtle but crucial distinction here: Negligible means decaying faster than _every polynomial_, whereas 1/poly(\lambda) (the O is unnecessary) means decaying faster than _some polynomial_. For instance, 1/n^2 is 1/poly(n) but not negl(n). This misunderstanding shows up again in equation (49) of page 21, where the paper says: "exp(−\Omega(log \lambda)) = negl(\lambda)", which is not true at all. In fact exp(−\Omega(log \lambda)) = 1/poly(\lambda).

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents a PA-based watermarking scheme for large language models (LLMs) that aims to achieve provable undetectability under computational hardness assumptions. Building on pseudorandomness and the sparse Learning Parity with Noise (LPN) problem, the authors claim their approach offers stronger theoretical guarantees than existing watermarking methods such as KGW, STA-M, and WEPA. The paper provides theoretical arguments about the connection between undetectability and LPN hardness, and presents experimental evaluations on detection accuracy and p-values.

### Strengths
Watermarking is a timely topic. Investigating the application of PA in watermarking is an interesting direction.

### Weaknesses
1. I am not convinced that a PA-based watermarking scheme can be computationally efficient, in theory or in practice. In theory, the PA scheme is still based on a randomized process, like other watermarking schemes. In practice, the detection efficiency (in Table 1) is much slower than the WEPA.
2. It is unclear how the theoretical guarantees translate into real-world scenarios. E.g., the hardness of sparse LPN is in the asymptotic of the input problem’s length, for watermarking, how large is this length? If it is 10-20, then even though the problem is hard (for large-size inputs), it may not be difficult to solve in practice (for small inputs), so the undetectability result may not be that useful.
3. The improvement in p-value is small when text length is large (Fig2-3), considering that p-value is already small, in 1e-2 to 1e-4 (Figure 4), further improving it by a small amount will not make a practical difference. In addition, when the text length is small, having a hardness-based theoretical guarantee on undetectability is not convincing. 
4. There is no evaluation of the utility of the generated text.

### Questions
See my weakness.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes WEPA, a family of text watermarking schemes built from probabilistic automata (PA). It argues this PA view subsumes cyclic key–sequence, “distortion-free” watermarks and enables two instantiations: (i) a practical, model-agnostic scheme that claims much higher generation diversity and near-linear detection, and (ii) a theoretical, undetectable construction via PNFA under sparse-LPN hardness. Experiments on LLaMA-3B and Mistral-7B (news-like C4 prompts) report strong edit-distance robustness and materially faster detection than the cyclic baseline, with comparable or better p-value curves; limits include no paraphrase robustness, reliance on a private key, and loose statistical bounds. The major issue is that this paper did not benchmark its method against many other SOTA watermarking methods.

### Strengths
1. The PA formalization is neat and unifying: it shows cyclic distortion-free watermarks (Kuditipudi et al.) are a special case of a PDFA and cleanly generalizes to PNFA, clarifying detectability vs. undetectability through learnability of automata. This perspective helps situate green-red lists and other decoder-based designs in one lens.
2. The practical WEPA design preserves the LM’s next-token distribution (exponential-min sampling), while increasing generation diversity from Θ(λ) to Ω(λ^d n) and cutting detector complexity from Θ(λ n k²) to Θ(λ n) via a dynamic-programming alignment against the PA’s support language. That directly addresses “deterministic outputs” and high detection cost in distortion-free baselines.

### Weaknesses
1. Evaluation breadth is narrow for a watermark meant to be robust “in practice.” There is no evaluation against paraphrase/semantic attacks, detector-aware adversaries, or order-agnostic sampling—gaps that matter given known fragility of text watermarks to paraphrase. Please consider discussing related papers (e.g., https://arxiv.org/abs/2410.13808).
2. Baselines omit key contemporaries the audience will expect: there’s no comparison to low entropy watermarking (https://arxiv.org/abs/2405.14604v3), nor to DiPMark (https://arxiv.org/abs/2310.07710) or SynthID (https://www.nature.com/articles/s41586-024-08025-4) for cross-modality discussion of detection error modes (even if image/video, they set operational expectations on robustness and thresholding). The “unbiased watermark” and green-red set are included, but the more recent methods are not included.
3. The deterministic output of unbiased watermark was observed and discussed, please cite previous studies (e.g. https://arxiv.org/abs/2406.02603).

### Questions
No more questions.

### Soundness
3

### Presentation
2

### Contribution
2
