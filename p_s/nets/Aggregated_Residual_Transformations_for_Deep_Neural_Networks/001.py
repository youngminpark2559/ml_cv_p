# ======================================================================
# Aggregated Residual Transformations for Deep Neural Networks
# Table 1
# Figure 3 (a)

# input image, 
input_img

# output from conv1, feature map size is (112,112), number of feature map is 64
o_con1=conv1(input_img,k_size=(7,7),output_dim=64,stride=2)

# ---------------
# output from max pool, feature map size is (56,56), number of feature map is 64
o_mp=max_pool(o_con1,k_size=(3,3),output_dim=64,stride=2)

# conv2_1_1_1 is conv2, at first block (Table 1, x1), at first 3 layers group (Figrue 3 (a), most left 3 layers)
o_con2_1_1_1=conv2_1_1_1(o_mp,k_size=(1,1),output_dim=128,stride=2)
o_con2_1_1_2=conv2_1_1_2(o_con2_1_1_1,k_size=(3,3),output_dim=128,stride=2)
o_con2_1_1_3=conv2_1_1_3(o_con2_1_1_2,k_size=(1,1),output_dim=256,stride=2)
# ...
o_con2_1_32_1=conv2_1_32_1(o_mp,k_size=(1,1),output_dim=128,stride=2)
o_con2_1_32_2=conv2_1_32_2(o_con2_1_32_1,k_size=(3,3),output_dim=128,stride=2)
o_con2_1_32_3=conv2_1_32_3(o_con2_1_32_2,k_size=(1,1),output_dim=256,stride=2)

conv2_first_sum=o_con2_1_1_3+...+o_con2_1_32_3

# ---------------
o_con2_2_1_1=conv2_2_1_1(conv2_first_sum,k_size=(1,1),output_dim=128,stride=2)
o_con2_2_1_2=conv2_2_1_2(o_con2_2_1_1,k_size=(3,3),output_dim=128,stride=2)
o_con2_2_1_3=conv2_2_1_3(o_con2_2_1_2,k_size=(1,1),output_dim=256,stride=2)
# ...
o_con2_2_32_1=conv2_2_32_1(conv2_first_sum,k_size=(1,1),output_dim=128,stride=2)
o_con2_2_32_2=conv2_2_32_2(o_con2_2_32_1,k_size=(3,3),output_dim=128,stride=2)
o_con2_2_32_3=conv2_2_32_3(o_con2_2_32_2,k_size=(1,1),output_dim=256,stride=2)

conv2_second_sum=o_con2_2_1_3+...+o_con2_2_32_3
conv2_second=conv2_second_sum+conv2_first_sum

# ---------------
o_con2_3_1_1=conv2_3_1_1(conv2_second,k_size=(1,1),output_dim=128,stride=2)
o_con2_3_1_2=conv2_3_1_2(o_con2_3_1_1,k_size=(3,3),output_dim=128,stride=2)
o_con2_3_1_3=conv2_3_1_3(o_con2_3_1_2,k_size=(1,1),output_dim=256,stride=2)
# ...
o_con2_3_32_1=conv2_3_32_1(conv2_second,k_size=(1,1),output_dim=128,stride=2)
o_con2_3_32_2=conv2_3_32_2(o_con2_3_32_1,k_size=(3,3),output_dim=128,stride=2)
o_con2_3_32_3=conv2_3_32_3(o_con2_3_32_2,k_size=(1,1),output_dim=256,stride=2)


conv2_third_sum=o_con2_3_1_3+...+o_con2_3_32_3
conv2_third=conv2_third_sum+conv2_second

# ---------------
o_con3_1_1_1=conv3_1_1_1(conv2_third,k_size=(1,1),output_dim=256,stride=2)
o_con3_1_1_2=conv3_1_1_2(o_con3_1_1_1,k_size=(3,3),output_dim=256,stride=2)
o_con3_1_1_3=conv3_1_1_3(o_con3_1_1_2,k_size=(1,1),output_dim=512,stride=2)
# ...
o_con3_1_32_1=conv3_1_32_1(conv2_third,k_size=(1,1),output_dim=256,stride=2)
o_con3_1_32_2=conv3_1_32_2(o_con3_1_32_1,k_size=(3,3),output_dim=256,stride=2)
o_con3_1_32_3=conv3_1_32_3(o_con3_1_32_2,k_size=(1,1),output_dim=512,stride=2)

conv3_first_sum=o_con3_1_1_3+...+o_con3_1_32_3
conv3_first=conv2_third+conv3_first_sum

# ---------------
# ...

conv3_third=conv3_second+conv3_third_sum

# ---------------
o_con3_4_1_1=conv3_4_1_1(conv3_third,k_size=(1,1),output_dim=256,stride=2)
o_con3_4_1_2=conv3_4_1_2(o_con3_4_1_1,k_size=(3,3),output_dim=256,stride=2)
o_con3_4_1_3=conv3_4_1_3(o_con3_4_1_2,k_size=(1,1),output_dim=512,stride=2)
# ...
o_con3_4_32_1=conv3_4_32_1(conv3_third,k_size=(1,1),output_dim=256,stride=2)
o_con3_4_32_2=conv3_4_32_2(o_con3_4_32_1,k_size=(3,3),output_dim=256,stride=2)
o_con3_4_32_3=conv3_4_32_3(o_con3_4_32_2,k_size=(1,1),output_dim=512,stride=2)

conv3_forth_sum=o_con3_4_1_3+...+o_con3_4_32_3
conv3_forth=conv3_third+conv3_forth_sum

# ---------------
o_con4_1_1_1=conv4_1_1_1(conv3_forth,k_size=(1,1),output_dim=512,stride=2)
o_con4_1_1_2=conv4_1_1_2(o_con3_4_1_1,k_size=(3,3),output_dim=512,stride=2)
o_con4_1_1_3=conv4_1_1_3(o_con3_4_1_2,k_size=(1,1),output_dim=1024,stride=2)
# ...
o_con4_1_32_1=conv4_1_32_1(conv3_forth,k_size=(1,1),output_dim=512,stride=2)
o_con4_1_32_2=conv4_1_32_2(o_con3_4_32_1,k_size=(3,3),output_dim=512,stride=2)
o_con4_1_32_3=conv4_1_32_3(o_con3_4_32_2,k_size=(1,1),output_dim=1024,stride=2)

conv4_first_sum=o_con4_1_1_3+...+o_con4_1_32_3
conv4_first=conv3_forth+conv4_first_sum

# ---------------
# ...

conv4_fifth=conv4_forth+conv4_fifth_sum
# ---------------
o_con4_6_1_1=conv4_6_1_1(conv4_fifth,k_size=(1,1),output_dim=512,stride=2)
o_con4_6_1_2=conv4_6_1_2(o_con4_6_1_1,k_size=(3,3),output_dim=512,stride=2)
o_con4_6_1_3=conv4_6_1_3(o_con4_6_1_2,k_size=(1,1),output_dim=1024,stride=2)
# ...
o_con4_6_32_1=conv4_6_32_1(conv4_fifth,k_size=(1,1),output_dim=512,stride=2)
o_con4_6_32_2=conv4_6_32_2(o_con4_6_32_1,k_size=(3,3),output_dim=512,stride=2)
o_con4_6_32_3=conv4_6_32_3(o_con4_6_32_2,k_size=(1,1),output_dim=1024,stride=2)

conv4_sixth_sum=o_con4_6_1_3+...+o_con4_6_32_3
conv4_sixth=conv4_fifth+conv4_sixth_sum

# ---------------
o_con5_1_1_1=conv5_1_1_1(conv4_sixth,k_size=(1,1),output_dim=1024,stride=2)
o_con5_1_1_2=conv5_1_1_2(o_con5_1_1_1,k_size=(3,3),output_dim=1024,stride=2)
o_con5_1_1_3=conv5_1_1_3(o_con5_1_1_2,k_size=(1,1),output_dim=2048,stride=2)
# ...
o_con5_1_32_1=conv5_1_32_1(conv4_sixth,k_size=(1,1),output_dim=1024,stride=2)
o_con5_1_32_2=conv5_1_32_2(o_con5_1_32_1,k_size=(3,3),output_dim=1024,stride=2)
o_con5_1_32_3=conv5_1_32_3(o_con5_1_32_2,k_size=(1,1),output_dim=2048,stride=2)

conv5_first_sum=o_con5_1_1_3+...+o_con5_1_32_3
conv5_first=conv4_sixth+conv5_first_sum

# ---------------
# ...

conv5_second=conv5_first+conv5_second_sum

# ---------------
o_con5_3_1_1=conv5_3_1_1(conv5_second,k_size=(1,1),output_dim=1024,stride=2)
o_con5_3_1_2=conv5_3_1_2(o_con5_3_1_1,k_size=(3,3),output_dim=1024,stride=2)
o_con5_3_1_3=conv5_3_1_3(o_con5_3_1_2,k_size=(1,1),output_dim=2048,stride=2)
# ...
o_con5_3_32_1=conv5_3_32_1(conv5_second,k_size=(1,1),output_dim=1024,stride=2)
o_con5_3_32_2=conv5_3_32_2(o_con5_3_32_1,k_size=(3,3),output_dim=1024,stride=2)
o_con5_3_32_3=conv5_3_32_3(o_con5_3_32_2,k_size=(1,1),output_dim=2048,stride=2)

conv5_third_sum=o_con5_3_1_3+...+o_con5_3_32_3
conv5_third=conv5_second+conv5_third_sum

# ---------------
# global average pooling
conv5_third=g_a_p(conv5_third)
conv5_third=fc(conv5_third)
conv5_third=softmax(conv5_third)

index=arg_max(conv5_third)
test_dataset_index

loss=loss_function(index,test_dataset_index)
optimize(loss)
