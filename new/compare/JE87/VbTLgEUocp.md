# Review

## Summary
This paper presents a method to encode a meaningful text into another meaningful text of the same length using a LLM. The encoded text is called stegotext and can be decoded back to the original text with the knowledge of the secret key. The method is very simple and efficient and can be implemented on a laptop in seconds. The paper also discusses the implications of this method, such as the radical decoupling of text from authorial intent and the challenges to AI safety.

## Soundness
3

## Presentation
4

## Contribution
3

## Strengths
1. The paper presents a simple and efficient method to hide a text inside another coherent looking text of the same length. This is achieved by generating the stegotext by selecting the rth most probable token instead of sampling from the probability distribution during the generation process. The original text can be recovered by selecting the token with the same rank in the stegotext.
2. The paper evaluates the method on Reddit posts and shows that the stegotexts are less probable than the original text for LLMs. It also discusses the security of the protocol and the implications of the method, such as the radical decoupling of text from authorial intent and the challenges to AI safety.

## Weaknesses
1. The paper does not compare the method with any existing methods. It would be helpful to compare the proposed method with other existing methods for text steganography or provide a discussion on why such a comparison is not applicable or necessary.
2. The paper does not provide a detailed analysis of the limitations of the method. While it mentions some limitations, such as the quality of the result depending on the LLM used, a more in-depth analysis of the potential drawbacks and scenarios where the method may not perform well would be beneficial.

## Questions
1. How does the method compare to other existing methods for text steganography? Are there any specific scenarios or contexts where this method is particularly suitable or unsuitable?
2. What are the limitations of the method in terms of the quality of the stegotext and the security of the protocol? Are there any specific cases where the stegotext is more likely to be detected or where the protocol is more vulnerable to attacks?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4