Anonymous authors Paper under double-blind review

## Abstract

Sparse Autoencoders (SAEs) extract features from LLM internal activations, meant to correspond to interpretable concepts. A core SAE training hyperparameter is L0: how many SAE features should fire per token on average. Existing work compares SAE algorithms using sparsity–reconstruction tradeoff plots, implying L0 is a free parameter with no single correct value aside from its effect on reconstruction. In this work we study the effect of L0 on SAEs, and show that if L0 is not set correctly, the SAE fails to disentangle the underlying features of the LLM. If L0 is too low, the SAE will mix correlated features to improve reconstruction. If L0 is too high, the SAE finds degenerate solutions that also mix features. Further, we present a proxy metric that can help guide the search for the correct L0 for an SAE on a given training distribution. We show that our method finds the correct L0 in toy models and coincides with peak sparse probing performance in LLM SAEs. We find that most commonly used SAEs have an L0 that is too low. Our work shows that L0 must be set correctly to train SAEs with correct features.

## 1 Introduction

It is theorized that Large Language Models (LLMs) represent concepts as linear directions in representation space, known as the Linear Representation Hypothesis (LRH) (Elhage et al., 2022; Park et al., 2024). These concepts are nearly orthogonal linear directions, allowing the LLM to represent many more concepts than there are neurons, a phenomenon known as superposition (Elhage et al., 2022). However, superposition poses a challenge for interpretability, as neurons in the LLM are polysemantic, firing on many different concepts. Sparse autoencoders (SAEs) are meant to reverse superposition, and extract interpretable, monosemantic latent features (Cunningham et al., 2024; Bricken et al., 2023) using sparse dictionary learning (Olshausen & Field, 1997). SAEs have the advantage of being unsupervised, and can be scaled to millions of neurons in its hidden layer (hereafter called "latents"1). When training an SAE, practitioners must decide on the sparsity of SAE, measured in terms of L0, or how many latents activate on average for a given input. 2 L0 is typically considered a neutral design choice: most of the literature evaluates SAEs at a range of L0 values, referring to this as a "sparsity–reconstruction tradeoff" (Gao et al., 2024; Rajamanoharan et al., 2024). While most practitioners would expect that too high an L0 will break the SAE (afterall, it is called a *sparse* autoencoder), the implication of "sparsity–reconstruction tradeoff" plots is that any sufficiently low L0 is equally valid. However, recent work shows the same trend: low L0 SAEs perform worse on downstream tasks Kantamneni et al. (2025); Bussmann et al. (2025). What causes this degraded performance at low L0? In this work, we explore the effect of L0 on SAEs. We begin with toy model experiments using synthetic data, and show that if the L0 is too low, the SAE can "cheat" by mixing together components of correlated features, achieving better reconstruction compared to an SAE with correct, disentangled features. We consider this to be a manifestation of feature hedging (Chanin et al.,
1We use *latents* to prevent overloading the term *feature*, which we reserve for human-interpretable concepts the SAE may capture. This breaks from earlier usage which used *feature* for both (Elhage et al., 2022), but aligns with the terminology in (Lieberum et al., 2024) and makes the distinction more clear.

2TopK and BatchTopK SAEs (Gao et al., 2024; Bussmann et al., 2024) set the L0 (K) directly, whereas L1 and JumpReLU (Cunningham et al., 2024; Bricken et al., 2023; Rajamanoharan et al., 2024) adjust it via a coefficient in the loss. In any case, all SAE trainers must decide on the target L0.

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# Sparse But Wrong: Incorrect L0 Leads To In- Correct Features In Sparse Autoencoders

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 2025), where the SAE abuses feature correlations to compensate for insufficient resources to model the underlying features monsemantically. This mixing of correlated features into SAE latents affects both positively and negatively correlated features, meaning that in low L0 SAEs, nearly all latents are both less interpretable and more noisy than an SAE with a correctly set L0. Our findings also show that "sparsity–reconstruction tradeoff" plots, commonly used to assess SAE architectures, are not a sound method of evaluating SAEs. We demonstrate using toy model experiments that at low L0, an SAE with ground-truth correct latents achieves worse reconstruction than an SAE that mixes correlated features. Thus, if we had an SAE training method that resulted in perfect SAEs, "sparsity–reconstruction tradeoff" plots would cause us to reject that method. Finally, we develop a proxy metric based on projections between the SAE decoder and training activations that can detect if L0 is too low. We validate these findings on Gemma-2-2b (Team et al.,
2024), demonstrating that decoder patterns similar to what we observe in our toy model experiments also manifests in LLM SAEs. We further validate that the optimal L0 we find with our method in Gemma-2-2b matches peak performance on sparse probing tasks (Kantamneni et al., 2025). Our findings are of direct importance to anyone using SAEs in practice, showing that L0 must be set correctly for SAEs to learn correct features. Furthermore, our work implies that most SAEs used by researchers today have too low an L0.

## 2 Background

Sparse autoencoders (SAEs). An SAE decomposes an input activation x ∈ R
dinto a hidden state, a, consisting of h hidden neurons, called "latents". An SAE is composed of an encoder Wenc ∈ R
h×d, a decoder Wdec ∈ R
d×h, a decoder bias bdec ∈ R
d, and encoder bias benc ∈ R
h, and a nonlinearity σ, typically ReLU or a variant like JumpReLU (Rajamanoharan et al., 2024), TopK (Gao et al., 2024) or BatchTopK (Bussmann et al., 2024). The decoder is sometimes called the *dictionary*, in reference to sparse dictionary learning. We use both terms interchangeably.

$$\mathbf{a}=\sigma(\mathbf{W}_{\rm enc}(\mathbf{x}-\mathbf{b}_{\rm dec})+\mathbf{b}_{\rm enc})$$ $$\mathbf{\hat{x}}=\mathbf{W}_{\rm dec}\mathbf{a}+\mathbf{b}_{\rm dec}\tag{1}$$
$$(1)$$
$$(2)$$

In this work we focus on BatchTopK and JumpReLU SAEs as these are both considered SOTA
architectures. The JumpReLU activation is a modified ReLU with a threshold parameter τ > 0, so JumpReLUτ
(x) = x · 1x>τ . The BatchTopK activation function selects the top b × k activations across a batch of size b, allowing variance in the k selected per sample in the batch. After training, a BatchTopK SAE is converted to a JumpReLU SAE with a global τ . We follow the JumpReLU
training procedure outlined by Anthropic (Conerly et al., 2025).

SAEs are trained as follows, with an auxiliary loss Lp to revive dead latents with corresponding coefficient λp. JumpReLU SAEs also have a sparsity loss Ls and corresponding coefficient λs.

$${\mathcal{L}}=\|\mathbf{x}-{\hat{\mathbf{x}}}\|_{2}^{2}+\lambda_{s}{\mathcal{L}}_{s}+\lambda_{p}{\mathcal{L}}_{p}$$
2 + λsLs + λpLp (3)

![1_image_0.png](1_image_0.png)

The formulation of Ls and Lp for JumpReLU and BatchTopK SAEs is shown in Appendix A.1.

## 3 Toy Model Experiments

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 The Linear Representation Hypothesis (LRH) (Elhage et al., 2022; Park et al., 2024) states that LLMs represent concepts (alternatively referred to as "features") as (nearly) orthogonal linear directions in representation space. Thus, the hidden activations in an LLM are simply the sum all the firing feature vectors (a feature direction with a positive, non-zero magnitude) that are being represented. While an LLM can represent a potentially large number of concepts this way, in any given activation, only a small number of concepts are actively represented. For instance, if we inspect a hidden activation from within an LLM at the token " Canada", we may expect this activation to be a sum of feature vectors representing concepts like "country", "North America", "starts with C", "noun", etc... The job of a sparse autoencoder is to recover these "true feature" directions in its dictionary. In a real LLM, we do not have ground-truth knowledge of the "true features" the model is representing, so we do not know if the SAE has learned the correct features. Fortunately, it is easy to create a toy model setup that follows the requirements of the LRH while providing ground-truth knowledge of the underlying true features.

Our toy model has a set of feature embeddings F ∈ R
g×d, where d is the input dimension of our SAE, and g is the number of features. All features are orthogonal, so fi· fj = 0 for i ̸= j.

Each feature fi fires with probability pi, mean magnitude µi, and magnitude standard deviation σi. Feature activations follow a correlated Bernoulli process controlled by correlation matrix C, with final magnitudes given by mi = ai·ReLU(µi + σiϵi), where aiindicates whether feature i is active and ϵi ∼ N (0, 1). Training activations for an SAE, x ∈ R
d, are thus generated as x =Pn i=1 mifi In these toy model experiments, we mainly focus on BatchTopK SAEs (Bussmann et al., 2024) as this enables direct control of L0. Additionally, we validate our results with JumpReLU SAEs. We train SAEs on 15M synthetic samples with batch size 500 using SAELens (Bloom et al., 2024). Throughout this section we will use the following terminology: True L0 In toy models we have complete control over which features fire, so we know how many features are firing on average. We refer to this as the *true L0* of the toy model. Ground-truth SAE Since we know the ground-truth features in our toy models, we can construct an SAE that perfectly captures these features. We refer to this as the *ground-truth SAE*. This is an SAE where g = h, Wenc = F
T, Wdec = F, benc = 0, bdec = 0.

## 3.1 Low L0 Saes Mix Correlated And Anti-Correlated Features

We begin with a small toy model with 5 true features (g = 5) in an input space of d = 20. We set each pi = 0.4 such that on average 2 features are active per input, for a true L0 of 2. We begin with a simple correlation pattern between features, where f0 is positively correlated with every feature f1 through f4, but otherwise there are no other correlations. We then train an SAE with L0 = 2, matching the true L0 of the model, and an SAE with slightly lower value of L0 = 1.8 (BatchTopK SAEs permit setting fractional L0). For the L*0 = 1*.8 SAE, we initialize it to the ground-truth solution, ensuring that the result of training is due to gradient pressure rather than just being a local minimum. We show the toy model feature correlation matrix as well as decoder cosine similarity plots with the true features for both SAEs in Figure 2. When the SAE L0 matches the true L0, we see that the SAE perfectly learns the underlying true features. However, when SAE L0 is smaller than the true L0, the resulting SAE latents mix feature components together based on the correlation matrix. The latents tracking features f1 through f4 all mix in a *positive* component of f0, but they have no components of each other. Next, we invert the correlation, i.e. each feature f1 through f4 is negatively correlated with f0 instead, while keeping everything else unchanged. We show the correlation matrix and SAE decoder cosine similarity with true features plots in Figure 3.

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215

![3_image_0.png](3_image_0.png)

![3_image_1.png](3_image_1.png)

Now, we see the same pattern as with positive correlations except inverted. The latents tracking features f1 through f4 mix in a *negative* component of f0, but have no component of each other.

This pattern is problematic because it means that if our L0 is too low, every SAE latent will contain positive components of every positively correlated feature, and negative components of every negatively correlated feature in the model. Negative correlations are particularly bad, as negative correlations are prevalent throughout language. For instance, we may expect a nonsensical negative component of "Harry Potter" to appear in the latent for "French poetry", since Harry Potter has nothing to do with French poetry. This will result in highly polysemantic and noisy SAE latents. Extended toy model experiments are shown in Appendix A.3.

## 3.2 Larger Toy Model Experiments

Next, we scale up to a larger toy model with 50 true features (g = 50) in input space of d = 100.

We set p0 = 0.345 and linearly decrease to p49 = 0.05, so firing probability decreases with feature number. The true L0 of this model is 11. We randomly generate a correlation matrix, so the firings of each feature are correlated with other features. Feature correlations are shown in Appendix A.2.

We train SAEs with L0 values that are too small (L0 = 5), exactly correct (L0 = 11), and too large (L0 = 18). Results are shown in Figure 1. When the SAE L0 matches the true L0, the SAE exactly learns the true features. When SAE L0 is too low, the SAE mixes components of correlated features together, particularly breaking latents tracking high-frequency features. When L0 is too high, the SAE learns degenerate solutions that mix features together. The further SAE L0 is from the true L0, the worse the SAE. Interestingly, when L0 is too high the SAE still learns many correct latents, but when L0 is too low, every latent in the SAE is affected.

## 3.3 Mse Loss Incentivizes Low-L0 Saes To Mix Correlated Features

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 It is common practice to evaluate SAE architectures using a sparsity–reconstruction tradeoff plot (Cunningham et al., 2024; Gao et al., 2024; Rajamanoharan et al., 2024), where the assumption is that having better reconstruction at a given sparsity is inherently better, and indicates that the SAE is correct. Afterall, we train SAEs to reconstruct inputs, so surely an SAE that has better reconstruction must therefore be a better SAE than one that has lower reconstruction? Sadly, this is not the case. As we discussed in Section 3.3, when the L0 of the SAE is lower than optimal, the SAE can find ways to "cheat" by engaging in feature hedging (Chanin et al., 2025), and get a better MSE score by mixing components of correlated features together. This results in an SAE where the latents are not monosemantic, and do not track ground-truth features.

![4_image_0.png](4_image_0.png)

Figure 4: Sparsity (L0, lower is better) vs reconstruction (variance explained, higher is better) for learned SAEs and a ground-truth SAE. When L0 is less than the true L0 of the toy model (the dotted line), the trained SAE gets better reconstruction than the ground-truth SAE. Sparsity–reconstruction plots like this lead us to the incorrect conclusion that the ground-truth SAE is a worse SAE.

![4_image_1.png](4_image_1.png)

Figure 5: SAE decoder cosine similarity with true features for the SAEs from Figure 4 with L0=1 (left) and L0=5 (middle), compared with the ground-truth SAE (right). The trained SAEs score much better than the ground truth SAE on variance explained, despite their corrupted, polysemantic latents. We next explore the sparsity–reconstruction tradeoff by training SAEs on our toy model at various L0s. Since we know the ground-truth features in our toy model, we construct a ground-truth SAE that perfectly represents these features. We vary the L0 of the ground truth SAE while leaving the encoder and decoder fixed at the correct features. We plot the variance explained vs L0 in Figure 4

## 3.4 The Sparsity–Reconstruction Tradeoff

Why do SAEs with low L0 not learn the true features? We construct a ground-truth SAE and set L0 = 5, to match the low L0 SAE from Figure 1. We generate 100k synthetic training samples and calculate the Mean Square Error (MSE) of both these SAEs. The trained SAE with incorrect latents achieves a MSE of 2.73, while the ground-truth SAE achieves a much worse MSE of 4.88. Thus, MSE loss actively incentivizes low L0 SAEs to learn incorrect latents. for both SAEs. When the SAE L0 is lower than the true L0 of the toy model, the ground-truth SAE scores worse on reconstruction than the trained SAE! If we had an SAE training technique that gave us the ground truth correct SAE for a given LLM, sparsity–reconstruction plots would cause us to discard the correct SAE in favor of an incorrect SAE that mixes features together. We show the cosine similarity of the SAE decoder latents with the ground truth features for the SAEs learned with L0=1 and L0=2 compared with the ground-truth SAE in Figure 5. Both these SAEs outperform the ground-truth SAE on variance explained by over 2x despite learning horribly polysemantic latents bearing little resemblance to the underlying true features of the model.

## 3.5 Detecting The True L0 Using The Sae Decoder

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

## 3.6 Jumprelu Sae Experiments

Figure 1 reveals that the SAE decoder latents contain mixes of underlying features, both when the L0 is too high and also when it is too low. As the SAE approaches the correct L0, each SAE latent has fewer components of multiple true features mixed in, becoming more monosemantic. Thus, we expect that the closer the SAE is to the correct L0, the more latents should be orthogonal relative to each other, as there are fewer components of shared correlated features mixed into latents. If we are far from the correct L0, then SAE latents contain components of many underlying features, and thus we expect latents to have higher cosine similarity with each other.

We call this metric *decoder pairwise cosine similarity*, cdec, and define it as below:

$$c_{\mathrm{dec}}={\frac{1}{{\binom{h}{2}}}}\sum_{i=1}^{h-1}\sum_{j=i+1}^{h}|\cos(\mathbf{W}_{\mathrm{dec},i},\mathbf{W}_{\mathrm{dec},j})|$$

$$(4)$$
| cos(Wdec,i,Wdec,j )| (4)
where h2
=
h(h−1)
2is the total number of distinct pairs of latents in the SAE decoder.

If SAE decoder latents are mixing lots of positive and negative components of correlated and anticorrelated features, then each SAE latent should become less orthogonal to each other SAE latent, as many latents will likely mix together similar features. This should mean that the absolute value of the cosine similarity between arbitrary latents should also increase the worse this mixing becomes.

We calculate pairwise calculate similarity cdec for each of the BatchTopK SAEs we trained on toy models from Section 3.5. Results are shown in Figure 6. We see that pairwise cosine similarity is minimized at the true L0.

![5_image_0.png](5_image_0.png)

We explore alternative metrics in Appendix A.9. Further toy model experiments are shown in Appendix A.4. Pytorch code implementing cdec is provided in Appendix A.17. We provide formal theoretical justification for the cdec metric in Appendix A.6.

So far, we have only investigated BatchTopK SAES due to their ease of setting L0. We now validate that these same conclusions apply to JumpReLU SAEs. We train JumpReLU saes with a range of λs to control the sparsity of the SAEs. We show plots of λs vs L0 and decoder pairwise cosine 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

![6_image_0.png](6_image_0.png)

![6_image_1.png](6_image_1.png)

![6_image_2.png](6_image_2.png)

similarity vs L0 for these SAEs in Figure 7. We see that the cosine similarity vs L0 broadly follows the same pattern as we saw for BatchTopK SAEs, and is minimized at the correct L0.

Interestingly, we see that the L0 does not change linearly with λs, but instead "sticks" near the correct L0. This is a testament to Anthropic's JumpReLU SAE training method (Conerly et al., 2025), as a wide range of sparsity coefficients λs cause the SAE to naturally find the correct L0.

## 4 Llm Experiments

We train a series of BatchTopK SAEs (Bussmann et al., 2024) with h = 32768 on Gemma-2-2b
(Team et al., 2024) and Llama-3.2-1b (Dubey et al., 2024) varying L0 and calculate cdec. Each SAE is trained on 500M tokens from the Pile (Gao et al., 2020) using SAELens (Bloom et al., 2024). We also calculate k-sparse probing performance for these SAEs using the benchmark from Kantamneni et al. (2025), consisting of over 100 sparse probing tasks. Results are shown in Figure 8.

The Llama SAE cdec plot looks very similar to the toy model, with a clear minimum point. The Gemma-2-2b layer 5 SAEs also show a sharp increase in cdec at low L0 as we saw in toy models, but has a long shallow region with the global minimum actually appearing in that shallow region. In both cases, the "elbow" in the cdec plots just before the jump due to low L0 is around L0 200, and

![7_image_0.png](7_image_0.png)

this also corresponds to peak sparse probing performance. More plots and analysis of cdec curves are shown in Appendix A.15.

## 4.1 Jumprelu Vs Batchtopk Saes

We next explore how JumpReLU and BatchTopK SAEs compare with decoder pairwise cosine similarity plots. We train a suite of SAEs on 1B tokens on Gemma-2-2b layer 12. We plot cdec for a range N values as well as k-sparse probing results for JumpReLU and BatchTopK SAEs in Figure 9 (left).

JumpReLU and BatchTopK SAEs behave similarly at low L0, with the high cdec at low L0 corresponding to poor sparse-probing performance. However, we see notable differences at high L0.

The BatchTopK SAEs have a global cdec minimum around 200, but JumpReLU SAEs cdec minimum appears closer to 250-300. As we saw in Figure 8 as well, using the "elbow" of the plots just before cdec jumps due to low L0 seems to roughly correspond to peak k-sparse probing performance. For JumpReLU SAEs, we see that cdec rises much less than BatchTopK SAEs at high L0, and indeed, JumpReLU SAEs also perform much better than BatchTopK SAEs at sparse probing when L0 is high. We suspect this is due to JumpReLU SAEs being able to "stick" near the correct threshold per latent like we saw in our toy models section. We investigate the differences in learned SAEs between JumpReLU and BatchTopK further in Appendix A.16.

## 4.2 Can L0 Be Both Too Low And Too High Simultaneously?

In Figure 9 (right), we plot decoder projection histogram plots for BatchTopK SAEs on Gemma-22b layer 12 with L0 10, 200, 750, and 2000. These plots are created by projecting training inputs on the SAE decoder, creating a histogram of how strongly each latents projects onto the input.

We expect that the more SAE latents are mixing positive and negative components of underlying features, the more strongly they should project both positively and negatively on arbitrary training inputs. This should look like a narrow gaussian around 0 when there is little mixing, and a wider gaussian the more mixing there is. This is also the intuition behind the alternative metric discussed in Appendix A.9, and the theory behind this is formalized further in Appendix A.10. As expected, when L0 is very low (10) or very high (2000), we see a wide gaussian around 0, indicating that decoder latents are mixing correlated features together. At L0=200, we see a much more narrow distribution around 0, as we expect when near the correct L0. However, at L0=750, 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 we see an interesting phenomenon, where there is an even narrower distribution than at L0=200, but also a large hump starting at projection above 10 (more visible in the log plot). We suspect this indicates at L0=750, some latents become more monosemantic while other latents mix underlying features becoming less monosemantic. This likely means that the L0 is too high for some latents while simultaneously being too low for other latents. There is no reason why every latent has the same firing threshold, so there is likely a range of L0s where some latents are firing more than they ideally should while other latents are firing less than they ideally should. We also suspect this is part of why JumpReLU SAEs seem to perform much better at high L0, since JumpReLU SAEs can adjust firing threshold per-latent while BatchTopK SAEs cannot.

## 5 Related Work

Limitations of SAEs Early work on SAEs for interpretability highlight the problem of feature splitting (Bricken et al., 2023; Templeton et al., 2024), where a seemingly interpretable general feature splits into more specific features at narrower SAE widths. Chanin et al. (2025) explores feature hedging, showing SAEs mix correlated features into latents if the SAE is too narrow. We consider our work a version of feature hedging due to low L0. Till (2024) shows SAEs may increase sparsity by inventing features. Chanin et al. (2024) discuss the problem of feature absorption, where SAEs can improve their sparsity score by mixing hierarchical features together. Engels et al. (2024) investigates SAE errors and finds that SAE error may be pathological and non-linear. Engels et al. (2025) find that not all underlying LLM features themselves are linear, demonstrating circular embeddings of some concepts. Wu et al. (2025) and Kantamneni et al. (2025) both investigate empirical SAE performance, finding SAEs underperform relative to supervised baselines, but do not offer theoretical explanations as to why SAEs underperform. Picking SAE hyperparameters Related to our work is Minimum Description Lengths (MDL) SAEs (Ayonrinde et al., 2024), which attempt to find reasonable choices for SAE width and L0 based on information theory. However, MDL SAEs assume that there is no inherently "correct" decomposition for LLM activations and no "correct" L0, and therefore does not attempt to find the underlying true features. Our work takes the opposite approach, starting from simple toy models with linear features and showing that if L0 is not set correctly the SAE decoder becomes corrupted. Another SAE architecture which attempts to pick L0 heuristically is Approximate Feature Activation (AFA) SAEs (Lee et al., 2025). AFA SAEs selects L0 adaptively at each input by assuming underlying true features are maximally orthogonal and selecting features until the feature norm is close to the input norm. While the L0 is not set directly in AFA SAEs, there is an extra loss hyperparameter that may modulate the resulting L0. Choosing L0 in related fields In Independent Component Analysis (ICA), a related field, it has also been shown that selecting the correct number of independent components (equivalent the L0 in SAEs) is important to achieve successful disentanglement (Li et al., 2007; Yi et al., 2024). However, ICA differs from SAEs in that the ICA requires fewer features than the number of input dimensions, while SAEs typically use overcomplete dictionaries.

## 6 Discussion

While most practitioners of SAEs understand that having too high L0 is problematic, our work shows that having too low of L0 is perhaps even worse. Our work has several important implications for the field. First, the L0 used by most SAEs is lower than it ideally should be, as a cursory search of open source SAEs on Neuronpedia (Lin, 2023) shows L0 less than 100 is very common even for SAEs trained on large models (see Appendix A.13). We further show that the sparsity–reconstruction tradeoff, as commonly discussed by most SAE papers (Cunningham et al., 2024; Gao et al., 2024; Rajamanoharan et al., 2024), is misleading: when L0 is too low, an SAE with a correct dictionary achieves worse reconstruction than an incorrect SAE that mixes correlated features.

We presented a metric based on the correlation between the SAE decoder and input activations, cdec, that can give us hints about the correct L0 for a given SAE. However, we do not view this as a perfect guide. As we saw in our results, while low L0 SAEs consistently have very high cdec, the metric can sometime remain nearly flat for a wide range of L0. Still, we feel that this metric is a useful guide to avoid L0 that is clearly too low, and we hope this investigation into correlation-based SAE quality metrics can be built on further in future work. We are particularly excited about the possibility that we can learn more about the underlying correlational structure between underlying features by studying correlations in the SAE decoder. While our metric currently requires training a sweep over L0 to optimize, we are hopeful that it may be possible to optimize this metric automatically during training (steps towards this are discussed in Appendix A.11). Improving this further is left to future work.

## 7 Reproducibility Statement

Code for all toy model experiments and demonstration code for training and evaluating LLM SAEs is provided as part of the supplementary materials for this paper. We further provide details on toy model SAE training in Section 3 and Appendix A.2, and for LLM SAE training Section 4 and Appendix A.7.

## References

Kola Ayonrinde, Michael T Pearce, and Lee Sharkey. Interpretability as compression: Reconsidering sae explanations of neural activations with mdl-saes. *arXiv preprint arXiv:2410.11179*, 2024.

Joseph Bloom, Curt Tigges, Anthony Duong, and David Chanin. Saelens. https://github.

com/jbloomAus/SAELens, 2024.

Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, et al. Towards monosemanticity: Decomposing language models with dictionary learning. *Transformer Circuits Thread*, 2, 2023.

Bart Bussmann, Patrick Leask, and Neel Nanda. Batchtopk sparse autoencoders. arXiv preprint arXiv:2412.06410, 2024.

Bart Bussmann, Noa Nabeshima, Adam Karvonen, and Neel Nanda. Learning multi-level features with matryoshka sparse autoencoders. *arXiv preprint arXiv:2503.17547*, 2025.

David Chanin, James Wilken-Smith, Toma´s Dulka, Hardik Bhatnagar, and Joseph Bloom. A is ˇ
for absorption: Studying feature splitting and absorption in sparse autoencoders. arXiv preprint arXiv:2409.14507, 2024.

David Chanin, Toma´s Dulka, and Adri ˇ a Garriga-Alonso. Feature hedging: Correlated features break `
narrow sparse autoencoders. *arXiv preprint arXiv:2505.11756*, 2025.

Tom Conerly, Hoagy Cunningham, Adly Templeton, Jack Lindsey, Basil Hosmer, and Adam Jermyn. Dictionary learning optimization techniques. https: //transformer-circuits.pub/2025/january-update, 2025.

Hoagy Cunningham, Logan Riggs Smith, Aidan Ewart, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. In *The Twelfth International* Conference on Learning Representations, 2024. URL https://openreview.net/forum?

id=F76bwRSLeK.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael 486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Olivier Duchenne, Onur C¸ elebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaoqing Ellen Tan, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aaron Grattafiori, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alex Vaughan, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Franco, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, Danny Wyatt, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Firat Ozgenel, Francesco Caggioni, Francisco Guzman, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella ´ Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Govind Thattai, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Karthik Prasad, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kun Huang, Kunal Chawla, Kushal Lakhotia, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Maria Tsimpoukelli, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mo594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 hammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikolay Pavlovich Laptev, Ning Dong, Ning Zhang, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Rohan Maheswari, Russ Howes, Ruty Rinott, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Kohler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, V´ıtor Albiero, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaofang Wang, Xiaojian Wu, Xiaolan Wang, Xide Xia, Xilun Wu, Xinbo Gao, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yuchen Hao, Yundi Qian, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, and Zhiwei Zhao. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, et al. Toy models of superposition. *arXiv preprint arXiv:2209.10652*, 2022.

Joshua Engels, Logan Riggs, and Max Tegmark. Decomposing the dark matter of sparse autoencoders. *arXiv preprint arXiv:2410.14670*, 2024.

Joshua Engels, Eric J Michaud, Isaac Liao, Wes Gurnee, and Max Tegmark. Not all language model features are one-dimensionally linear. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=d63a4AM4hb.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. The Pile: An 800gb dataset of diverse text for language modeling. *arXiv preprint arXiv:2101.00027*, 2020.

Leo Gao, Tom Dupre la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya ´
Sutskever, Jan Leike, and Jeffrey Wu. Scaling and evaluating sparse autoencoders. *arXiv preprint* arXiv:2406.04093, 2024.

Subhash Kantamneni, Joshua Engels, Senthooran Rajamanoharan, Max Tegmark, and Neel Nanda.

Are sparse autoencoders useful? a case study in sparse probing. *arXiv preprint arXiv:2502.16681*, 2025.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Sewoong Lee, Adam Davies, Marc E Canby, and Julia Hockenmaier. Evaluating and designing sparse autoencoders by approximating quasi-orthogonality. *CoRR*, 2025.

Yi-Ou Li, Tulay Adalı, and Vince D Calhoun. Estimating the number of independent components for ¨
functional magnetic resonance imaging data. *Human brain mapping*, 28(11):1251–1266, 2007.

Tom Lieberum, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, Janos Kram ´ ar, Anca Dragan, Rohin Shah, and Neel Nanda. Gemma Scope: Open Sparse ´ Autoencoders Everywhere All At Once on Gemma 2, August 2024.

Johnny Lin. Neuronpedia: Interactive reference and tooling for analyzing neural networks, 2023.

URL https://www.neuronpedia.org. Software available from neuronpedia.org.