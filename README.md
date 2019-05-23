# info-aes
# The MADE Network

## Background/Motivation:

Have you ever wondered why you found a piece of art aesthetically pleasing? The principle of the aesthetic middle is an explanation of aesthetic preference that states that the most beautiful items are of moderate complexity. Simple artwork fails to hold viewers’ attention, while complex artwork exhausts viewers. Moderately complex art, however, is involved enough to spark curiosity, but is not overwhelming. The aesthetic middle may stem from feelings of reward that accompany correct predictions of what will be perceived. According to the theory of predictive processing, our brains predict what we will perceive and receive accuracy feedback from our senses. Because knowing about the world helps us secure resources and avoid threats, it is understandable how we could have evolved to experience reward when our predictions are satisfied. Not all predictions hold the same weight, though. It is more difficult and thus more rewarding to predict a surprising outcome than an expected one. Thus, the aesthetic middle: it is easy to make correct predictions about simple things, but making these predictions is not terribly rewarding; it is difficult to make correct predictions about complex things, so, though these predictions are highly rewarding, net reward is small; it is only moderately difficult to make rewarding correct predictions about moderately complex things, so total reward is large.

At present, surprisingness and complexity are imprecisely characterized in the cognitive science research literature. Existing studies of the principle of the aesthetic middle have relied on researchers’ unformalized discriminations of complexity. However, results of mathematical information theory can enable the calculation of values representing the predictability of stimuli. As these results are probabilistic, it is necessary to estimate the probability distributions characteristic of the domain of interest (eg. images of handwritten digits, images of faces, French impressionist paintings). We have implemented the present network in order to perform the required distribution estimation so that we may evaluate the complexity, here operationalized as information content, of images in order to further investigate the nature of this value’s relationship to aesthetic judgment.

The network is structured so that it learns to compute the likelihood of a binarized image relative to similar images. This likelihood is computed as the conditional probability that each of the image’s pixels have the values they do. By the chain rule for conditional probabilities, we know that we can calculate this value by ordering the pixels in the image and iteratively finding the conditional probability corresponding to each pixel using the probabilities corresponding only to the pixels of lower index. In order to instantiate this procedure, the network must satisfy what is called the autoregressive property: nodes corresponding to the pixels of the image must be indexed, and these nodes must only be allowed to receive input from nodes with indices lower than theirs. When this property is satisfied, the output of the network is the desired probability, and the binary cross-entropy loss becomes the information-theoretic quantity called Shannon entropy. Then, in training by minimizing the loss, the network effectively learns the probability distribution that yields the smallest information value for each image. In this manner, it finds the distribution that best fits the domain to which the training images belong, as desired.

## Data/Preprocessing

We used the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, a common neural network testing dataset consisting of 70,000 handwritten digits from 0-9. Each digit is a 28 x 28 image with pixel values ranging from 0 to 255, where 0 is black and 255 is white. Our model was built for binarized data, so we converted each pixel that was greater than zero to have a value of 1, which represented white. Any pixels that were already 0 were kept as such.

## The Model

For in-depth explanations of each individual function, please see the code, which has inline comments. The general structure of the MADE network is as follows:

- Input Layer x Mask

- Hidden Layer 1 x Mask

- Hidden Layer 2 x Mask

- Direct Connection / Output Layer x Mask

The MADE network functions similarly to a standard neural network with multiple hidden layers, with the distinction being that it also utilizes mask matrices in each layer. Masks are comprised of matrices filled with zeroes and ones. The masks are present in order to make the network conform to an autoregressive property, which in brief allows for each output pixel to be a conditional probability based only on the inputs from certain input pixels. Zeros in the mask matrices correspond directly with certain pixels or hidden units and effectively “zero out” their progression through the network. Each input pixel and hidden node is assigned an index and the placement of the zeros and ones in each respective mask is determined based on the relationship between node indices from layer to layer. For the input and hidden layers, each node in a hidden layer is compared to each index in the layer before it. If the index of a hidden layer is greater than or equal to an index in the layer before it, then the corresponding entry in the mask matrix for the previous layer is a one. Otherwise, it is a zero. The output layer is slightly different in that entries in the mask are only allowed to be ones if the index of the output layer is greater than the index in the previous layer. Again, this is all in order to satisfy the autoregressive property of the network and ensure that the output is a conditional probability. The direct connection simply consists of multiplying the input and its mask with the output layer. This has been found to help improve the quality of the network’s output.

In order to create this network, we heavily utilized the original [MADE paper](https://arxiv.org/pdf/1502.03509.pdf) as well as an extraordinarily useful [blog post](http://bjlkeng.github.io/posts/autoregressive-autoencoders/) by Brian Keng. For more in-depth detail on the intricacies of the MADE network, feel free to visit these resources.

## Results

Unfortunately, we were unable to successfully implement the MADE network on our own and as such could not produce any results. However, had the network been implemented correctly, once fully trained we would be able to generate new, unique images based off of conditional probabilities the network has learned.

#### New Image Generation

The most up-to-date image generation function in the repository is located within gen_imagev2.py. The image generation process is rather simple. First, we generate a random input vector to serve as the x input for the network. We then run x through a trained MADE network, which will output a set of conditional probabilities. Next, we take a look at the input pixel that has been assigned an index of one and use its assigned probability as a parameter to sample from a Bernoulli distribution. Based on the outcome of the Bernoulli sampling, we then change the value of that pixel to either zero or one and put the now changed vector back through the network. We continue this process with the pixel assigned an index of two, then three, and so on until the entire image has been generated. We are then left with a newly generated image!
