[Summaries]
Attention not only tells where to focus, it also improves the representation of interests

Our goal is to increase representation power by using attention mechanism: focusing on important features and suppressing unnecessary ones.

To achieve this, we sequentially apply channel and spatial attention modules (as shown in Fig. 1),
so that each of the branches can learn ‘what’ and ‘where’ to attend in the channel and spatial axes respectively. As a result, our module efficiently helps the information flow within the network by learning which information to emphasize or suppress.

We visualize trained models using the grad-CAM [18] and observe that CBAMenhanced networks focus on target objects more properly than their baseline networks.

Fig. 1: The overview of CBAM. The module has two sequential sub-modules:
channel and spatial. The intermediate feature map is adaptively refined through our module (CBAM) at every convolutional block of deep networks.

Attention mechanism. It is well known that attention plays an important role in human perception [23,24,25]. One important property of a human visual system is that one does not attempt to process a whole scene at once. Instead, humans exploit a sequence of partial glimpses and selectively focus on salient parts in order to capture visual structure better [26].

3 Convolutional Block Attention Module
Given an intermediate feature map F ∈ R C×H×W as input, CBAM sequentially infers a 1D channel attention map Mc ∈ R C×1×1 and a 2D spatial attention map Ms ∈ R 1×H×W as illustrated in Fig. 1. The overall attention process can be summarized as: 
F ′ = Mc(F) ⊗ F, F ′′ = Ms(F ′ ) ⊗ F ′ , (1)
where ⊗ denotes element-wise multiplication. During multiplication, the attention values are broadcasted (copied) accordingly: channel attention values are broadcasted along the spatial dimension, and vice versa. F ′′ is the final refined output. Fig. 2 depicts the computation process of each attention map. The following describes the details of each attention module.

[My notes]
Intermediate feature map F will be passed into "channel attention modue" and "spatial attention module"
"channel attention modue" creates 1D channel attention map
"spatial attention module" creates 2D spatial attention map

channel_attention_map_1D=channel_attention_module(feature_map)
# @ c new_feature_map_CAM: new feature map after channel attention module
new_feature_map_CAM=elementwise_mul(channel_attention_map_1D,feature_map)

spatial_attention_map_2D=spatial_attention_module(new_feature_map)
# @ c new_feature_map_SAM: new feature map after spatial attention module
new_feature_map_SA=elementwise_mul(spatial_attention_map_2D,new_feature_map)

Note: use broadcasting in elementwise multiplication

[Summaries]
Channel attention module. We produce a channel attention map by exploiting the inter-channel relationship of features. 

As each channel of a feature map is considered as a feature detector [32], channel attention focuses on ‘what’ is meaningful given an input image. To compute the channel attention efficiently,
we squeeze the spatial dimension of the input feature map. For aggregating spatial information, average-pooling has been commonly adopted so far. 

Thus, we use both average-pooled and max-pooled features simultaneously

We first aggregate spatial information of a feature map by using both averagepooling and max-pooling operations, generating two different spatial context descriptors: 
F c avg and F c max, which denote average-pooled features and max-pooled features respectively.

Both descriptors are then forwarded to a shared network to produce our channel attention map Mc ∈ R C×1×1 . 

The shared network is composed of multi-layer perceptron (MLP) with one hidden layer. To reduce parameter overhead, the hidden activation size is set to R C/r×1×1 , where r is the reduction ratio. After the shared network is applied to each descriptor, we merge the output feature vectors using element-wise summation. In short, the channel attention is computed as:
Mc(F) = σ(MLP(AvgP ool(F)) + MLP(M axP ool(F))) = σ(W1(W0(F c avg)) + W1(W0(F c max))), (2)

where σ denotes the sigmoid function, W0 ∈ R C/r×C , and W1 ∈ R C×C/r. Note that the MLP weights, W0 and W1, are shared for both inputs and the ReLU activation function is followed by W0.

[My notes]
# @ Channel Attention Module
# @ c feat_max_in_CAM: feature after max pool in channel attention module
feat_max_in_CAM=max_pool(feature_map)
# @ c feat_avg_in_CAM: feature after average pool in channel attention module
feat_avg_in_CAM=avg_pool(feature_map)
# @ c MLP_shared_net: define MLP shared net
MLP_shared_net():
  FC1
  ReLU
  FC2
feat_max_MLP_in_CAM=MLP_shared_net(feat_max_in_CAM)
feat_avg_MLP_in_CAM=sigmoid(MLP_shared_net(feat_avg_in_CAM))
channel_attention_1D=elementwise_sum(feat_max_MLP_in_CAM,feat_avg_MLP_in_CAM)
feat_after_CAM=elementwise_mul(channel_attention_1D,feature_map)

If reduction ratio r is 10, and if channel of feature map is 1000
1000/10*1*1=100 will be hidden activation size

First layer W_0
1000/10*1000

[Summaries]
Spatial attention module. We generate a spatial attention map by utilizing the inter-spatial relationship of features. Different from the channel attention, the spatial attention focuses on ‘where’ is an informative part, which is complementary to the channel attention.

[My notes]
# @ Spatial Attention Module
feat_max_in_SAM=max_pool(feat_after_CAM)
feat_avg_in_SAM=avg_pool(feat_after_CAM)
feat_cat_in_SAM=concat(feat_max_in_SAM,feat_avg_in_SAM)
# @ Define conv net
conv_net():
  conv2d(kernel=(7,7))
  sigmoid()
feat_conv_in_SAM=conv_net(feat_cat_in_SAM)

[Summaries]
Arrangement of attention modules. Given an input image, two attention modules, channel and spatial, compute complementary attention, focusing on ‘what’ and ‘where’ respectively.
Considering this, two modules can be placed in a parallel or sequential manner. We found that the sequential arrangement gives a better result than a parallel arrangement. For the arrangement of the sequential process, our experimental result shows that the channel-first order is slightly better than the spatial-first. We will discuss experimental results on network engineering in Sec. 4.1.
