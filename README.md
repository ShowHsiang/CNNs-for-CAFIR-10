# CNNs-for-CAFIR-10
**Examples of some well-performing CNN models on the CAFIR-10 dataset, including accuracy, loss curves, and confusion matrices.**

**1.ResNet (Residual Networks):** ResNet, introduced by Kaiming He and colleagues in 2015, addresses the vanishing gradient problem that arises when training very deep neural networks. The key idea behind ResNet is the introduction of "residual blocks" and skip (or shortcut) connections.

   Key Features:

   Residual Blocks: In traditional networks, each layer tries to learn the desired underlying transformation from its inputs. In ResNet, each layer is designed to learn only 
   the residual (or difference) between its input and the desired output. Hence, it's called a "residual block". Mathematically, if H(x) is the desired underlying mapping, 
   the layers in the residual block try to learn the residual F(x) = H(x) - x.

   Skip Connections: These are connections that skip one or more layers and provide a direct path from earlier layers to later layers. Skip connections help in back- 
   propagating the gradient even to the very first layers, thus alleviating the vanishing gradient problem.

   Deep Architectures: With the introduction of residual blocks and skip connections, ResNet was able to train extremely deep networks, with architectures having 50, 101, or 
   even 152 layers.

   Advantages: Enables the training of much deeper networks without a significant increase in training time or risk of overfitting. Improved gradient flow through the 
   network due to skip connections. Generally gives better performance compared to its non-residual counterparts.

**2.MobileNet:** MobileNet, as the name suggests, is designed primarily for mobile and embedded vision applications. It's an efficient model that provides a good trade-off between computational cost (in terms of both memory and processing power) and performance.

   Key Features:

   Depthwise Separable Convolutions: Traditional convolutions are replaced by depthwise separable convolutions, which factorizes the convolution operation into two parts: a 
   depthwise convolution followed by a 1x1 pointwise convolution. This dramatically reduces the computational cost.

   Width Multiplier: It provides a trade-off between computational load and accuracy. By reducing the number of input/output channels (width), you can make your model faster 
   but less accurate.

   Resolution Multiplier: It allows setting the input resolution. Using a lower input resolution can further reduce computation at the cost of accuracy.

   There have been multiple versions of MobileNet, with MobileNetV2 introducing the inverted residual blocks and MobileNetV3 further improving the architecture based on a 
   combination of automated search and human expertise.

   Advantages: Significantly reduced number of parameters and computational cost compared to large models like VGG or ResNet, making it suitable for mobile devices. Despite 
   its small size, it offers competitive accuracy on many tasks.
   
**3.DenseNet:** DenseNet (Densely Connected Convolutional Networks) is a convolutional neural network (CNN) architecture introduced by Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger, and others in 2017. What distinguishes DenseNet from other popular convolutional networks is its dense connections between feature maps. While conventional CNNs, like VGG and ResNet, rely on skip connections or shortcuts between layers to facilitate the flow of gradients, the core proposition of DenseNet is to ensure every layer gets a direct connection to the feature information of all preceding layers. This is achieved by establishing dense connections.

   Key Features:
   Dense Connectivity: In DenseNet, the output of the l-th layer is the combined feature maps of all preceding layers. This means the l-th layer has 'l' inputs, which are 
   outputs from each of the preceding layers. All these inputs are concatenated together to form the input for the l-th layer.

   Growth Rate: The growth rate is a hyperparameter which denotes the number of feature maps produced by each layer. Since every layer receives feature maps from all 
   preceding layers, its total output is the sum of outputs from all preceding layers plus its own.
 
   Bottleneck Layers: To reduce the number of computations and parameters, DenseNet typically employs a 1x1 convolution before each convolution operation, referred to as the 
   bottleneck layer. It's responsible for reducing the number of feature maps.

   Transition Layers: Given that the continuous concatenation of feature maps would lead to an explosive increase in the number of feature maps, DenseNet introduces 
   transition layers to control the number of feature maps. Transition layers use a 1x1 convolution (to reduce the number of feature maps) followed by a 2x2 average pooling 
   to reduce the size of the feature maps.
   
   Advantages:
   Parameter Efficiency: Since all layers share feature maps from all preceding layers, DenseNet is more parameter-efficient than many other architectures.

   Improved Gradient Flow: The dense connections ensure better flow of gradients during both forward and backward propagation, which helps mitigate the vanishing gradient 
   problem.

   Encourages Feature Reuse: Given that all prior features are directly made available to every layer, DenseNet has the capability to reuse features across different parts 
   of the network.

   Better Regularization: The dense connections also provide an implicit form of regularization, allowing DenseNet to perform well with fewer data, thus reducing the risk of 
   overfitting.
   
**4.RepVGG** is a neural network architecture designed for both efficient deployment and effective training, introduced by researchers at Tencent. It stands out due to its simplicity and efficiency. The fundamental idea behind RepVGG is to have a structural difference between its training-time architecture and its inference-time architecture.

   Training-time Architecture: During training, RepVGG uses a multi-branch block. Each block contains three branches: a 3x3 convolution, a 1x1 convolution, and an identity 
   connection. Batch normalization and ReLU activation are applied after both convolutions. The 1x1 convolution and identity connection are only applied if the input and 
   output feature maps have the same dimensions.

   Inference-time Architecture: For deployment, the multi-branch block in the training-time architecture is converted into a single convolution layer. The three branches 
   from the training-time architecture are merged into a single kernel, making the architecture much simpler and efficient for inference.

   Key Features:

   Structural Re-parameterization: The conversion from the training-time architecture to the inference-time architecture is termed as "Structural Re-parameterization". The 
   convolutional kernels from the multiple branches are merged into a single convolutional kernel for inference.

   Simplicity: Unlike many other modern architectures that rely on sophisticated blocks or modules, the RepVGG architecture is quite simple, especially during inference, 
   making it easier to deploy.

   Performance: Despite its simplicity, RepVGG achieves competitive performance on benchmark datasets compared to other state-of-the-art architectures.

   Scalability: The architecture is scalable in terms of width and depth, allowing it to be adapted to various computational budgets or performance requirements.

   Advantages:
   Efficiency: The inference-time architecture is very efficient due to its simplicity, making it suitable for real-world deployment scenarios.

   Flexibility: The clear distinction between training-time and inference-time architectures provides flexibility. It allows for rich expressiveness during training while 
   retaining simplicity during inference.

   Competitive Accuracy: Despite being simple, RepVGG achieves competitive accuracy on benchmark tasks, making it an attractive choice for many computer vision tasks.

