# ======================================================================
# 4.1 Supervised losses (GCINTRINSIC)

# CGIntrinsics-supervised loss

# Scale factors are computed via least squares

# c c_r: scale factors for r
c_r

# c c_s: scale factors for s
c_s

# ---------------
# Eq 5

# c sf_L_si_MSE: sum for L_si_MSE
sf_L_si_MSE=0

for i in all_pixels:
    # c first_t: first term
    first_t=sq(gt_r[i]-c_r*pred_r[i])

    # c second_t: second term
    second_t=sq(gt_ss[i]-c_s*pred_s[i])

    sf_L_si_MSE+=first_t+second_t

L_si_MSE=sf_L_si_MSE/len(all_pixels)


# ---------------
# Eq 6

# c N_l: number of valid pixels at scale l
N[l]

# c N[1]: number of valid pixels at original image scale
# c N: number of pixels of original image
N = N[1]


# c sf_L_grad: sum for L_grad
sf_L_grad=0
for l in all_scales:
    for i in all_pixels:
        first_t=l1_norm(grad_gt_r[l][i]-c_r*grad_pred_r[l][i])

        second_t=l1_norm(grad_gt_s[l][i]-c_s*grad_pred_s[l][i])

        # c N[l]: number of valid pixel at scale l
        sf_L_grad+=(first_t+second_t)/N[l]

# ---------------
# c L_sup: loss of supervised learning with CGI dataset
# c L_si_MSE: scale-invariant mean-squared-error
# c L_grad: gradient domain multi-scale matching term
L_sup=L_si_MSE+L_grad


# ======================================================================
# Ordinal reflectance loss. (IIW)

# c L_ord_R: ordinal reflectance loss
L_ord_R

# c o_comp: one comparison
for o_comp in all_comparisons:
    if o_comp==0:
        # c w[o_comp]: confidence score of this one comparison
        w[o_comp]
        R_i=log(R[o_comp][i])
        R_j=log(R[o_comp][j])
        e_ij_R=w[o_comp]*sq(R_i-R_j)
        L_ord_R+=e_ij_R
    elif o_comp==1:
        # c w[o_comp]: confidence score of this one comparison
        w[o_comp]
        R_i=log(R[o_comp][i])
        R_j=log(R[o_comp][j])
        e_ij_R=w[o_comp]*sq(max(0,m-R_i+R_j))
        L_ord_R+=e_ij_R
    elif o_comp==-1:
        # c w[o_comp]: confidence score of this one comparison
        w[o_comp]
        R_i=log(R[o_comp][i])
        R_j=log(R[o_comp][j])
        e_ij_R=w[o_comp]*sq(max(0,m-R_j+R_i))
        L_ord_R+=e_ij_R

# ======================================================================
# SAW shading loss (SAW)

# Eq 8

# c N_c: pixels consisting constant shading region
N_c

sum_first_t=0
for i in N_c:
    sum_first_t+=sq(log(pred_s))

first_t=sum_first_t/len(N_c)

sum_second_t=0
for i in N_c:
    sum_second_t+=log(pred_s)

second_t=sq(sum_second_t)/sq(len(N_c))

# c L_constant_shading: loss of constant shading region
L_constant_shading=first_t-second_t


# Eq 9

# c N_sd: pixles consisting of each shadow boundary region
N_sd

s_first_L_shadow=0
for i in N_sd:
    s_first_L_shadow+=sq(log(pred_s[i])-log(I[i]))

# c fir_t_L_shadow: first term for L_shadow
fir_t_L_shadow=s_first_L_shadow/len(N_sd)

s_second_L_shadow=0
for i in N_sd:
    s_second_L_shadow+=log(pred_s[i])-log(I[i])

# c sec_t_L_shadow: second term for L_shadow
sec_t_L_shadow=sq(s_second_L_shadow)/sq(len(N_sd))


# c L_shadow: loss for each shadow boundary region (with N_{sd}) pixels
L_shadow=fir_t_L_shadow-sec_t_L_shadow


# ======================================================================
# 4.2 Smoothness losses 

# c L_rsmooth: loss of reflectance smoothness
L_rsmooth

# c L_ssmooth: loss of shading smoothness
L_ssmooth

# c N[l][i]: 8-connected neighborhood of pixel i, and scale l
N[l][i]

# c P[l][i]: spatial position of pixel i and scale l
P[l][i]

# c I[l][i]: image intensity of pixel i and scale l
I[l][i]

# c c[l][i]_1: first element of chromaticity at pixel i and scale l
c[l][i]_1

# c c[l][i]_2: second element of chromaticity at pixel i and scale l
c[l][i]_2

# c f[l][i]: feature vector of pixel i and scale l
f[l][i]=[P[l][i],I[l][i],c[l][i]_1,c[l][i]_2]


# c fir_t_v_l_i_j: first term for v_l_i_j
fir_t_v_l_i_j=f[l][i]-f[l][j].T

# c Sigma: covariance matrix defining distance between 2 feature vectors, f[l][i] and f[l][j]
Sigma

# c sec_t_v_l_i_j: first term for v_l_i_j
sec_t_v_l_i_j=f[l][i]-f[l][j]

# c v_l_i_j: reflectance weight
v_l_i_j=exp(-1/2*fir_t_v_l_i_j*(1/Sigma)*sec_t_v_l_i_j)


# ======================================================================
# 4.3 Reconstruction loss

# Eq 12

# c N: all pixels
N

sf_L_reconstruct=0
for i in range(N):
    sf_L_reconstruct+=sq(I[i]-pred_r[i]*pred_s[i])

# c L_reconstruct: reconstruction loss
L_reconstruct=sf_L_reconstruct/len(N)



L_CGI=L_sup+lambda_ord*L_ord_R+lambda_rec*L_reconstruct
L_IIW=lambda_ord*L_ord_R+lambda_rs*L_rsmooth+lambda_ss*L_ssmooth+L_reconstruct
L_SAW=lambda_S_NS*L_S_NS+lambda_rs*L_rsmooth+lambda_ss*L_ssmooth+L_reconstruct

entire_loss=L_CGI+lambda_IIW*L_IIW+lambda_SAW*L_SAW


# ======================================================================
# 4.4 Network architecture

# c o_ec1: output from encoder conv 1
o_ec1=encoder_conv_1(input_img,ksize=(4,4),stride=2)
o_ec1=BN(o_ec1)
o_ec1=L_ReLU(o_ec1)
o_ec2=encoder_conv_2(o_ec1,ksize=(4,4),stride=2)
o_ec2=BN(o_ec2)
o_ec2=L_ReLU(o_ec2)
o_ec3=encoder_conv_3(o_ec2,ksize=(4,4),stride=2)
o_ec3=BN(o_ec3)
o_ec3=L_ReLU(o_ec3)
o_ec4=encoder_conv_4(o_ec3,ksize=(4,4),stride=2)
o_ec4=BN(o_ec4)
o_ec4=L_ReLU(o_ec4)
o_ec5=encoder_conv_5(o_ec4,ksize=(4,4),stride=2)
o_ec5=BN(o_ec5)
o_ec5=L_ReLU(o_ec5)
o_ec6=encoder_conv_6(o_ec5,ksize=(4,4),stride=2)
o_ec6=BN(o_ec6)
o_ec6=L_ReLU(o_ec6)
o_ec7=encoder_conv_7(o_ec6,ksize=(4,4),stride=2)
o_ec7=BN(o_ec7)
o_ec7=L_ReLU(o_ec7)

# c o_ddc1: output from decoder deconv 1
o_ddc1=decoder_deconv_1(o_ec7,ksize=(4,4))
o_ddc1=BN(o_ddc1)
o_ddc1=ReLU(o_ddc1)
o_ddc2=decoder_deconv_2(o_ddc1+o_ec6,ksize=(4,4))
o_ddc2=BN()
o_ddc2=ReLU()
o_ddc3=decoder_deconv_3(o_ddc2+o_ec5,ksize=(4,4))
o_ddc3=BN()
o_ddc3=ReLU()
o_ddc4=decoder_deconv_4(o_ddc3+o_ec4,ksize=(4,4))
o_ddc4=BN()
o_ddc4=ReLU()
o_ddc5=decoder_deconv_5(o_ddc4+o_ec3,ksize=(4,4))
o_ddc5=BN()
o_ddc5=ReLU()
o_ddc6=decoder_deconv_6(o_ddc5+o_ec2,ksize=(4,4))
o_ddc6=BN()
o_ddc6=ReLU()
o_ddc7=decoder_deconv_7(o_ddc6,ksize=(4,4))
o_ddc7=BN()
o_ddc7=ReLU()

# c o_cf: output from conv final
o_cf=conv_final(o_ddc7,ksize=(1,1))
