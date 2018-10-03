# ======================================================================
# 2. Learning model of reflectance 

# c r[i]: reflectance estimate at pixel i
r[i]

# c R: set of all reflectance values in scene
R

if r_i<r_j:
    ref_r_i<ref_r_j
elif r_i=r_j:
    ref_r_i=ref_r_j
elif r_i>r_j:
    ref_r_i>ref_r_j


# ======================================================================
# 2.1. Relative reflectance classifier 

padded_img=pad_image_to_be_63_multiplier(input_img)
patch1,patch2=randomly_choose_2_patches(padded_img)
resized_img=resize(padded_img,(150,150))

pa1_cn=network1_sharing_weights(patch1)
pa2_cn=network1_sharing_weights(patch2)
r_img_cn=network2_small_img(resized_img)

class network1_sharing_weights():
    out=conv1(input_img,k_size=(3,3),n_filters=16,stride=2)
    out=ReLU(out)
    out=conv2(out,k_size=(3,3),n_filters=32,stride=2)
    out=ReLU(out)
    out=conv3(out,k_size=(3,3),n_filters=64,stride=2)
    out=ReLU(out)
    out=conv4(out,k_size=(7,7),n_filters=64,stride=1)
    out=ReLU(out)

class network2_small_img():
    out=conv1(input_img,k_size=(5,5),n_filters=32,stride=4)
    out=ReLU(out)
    out=conv2(out,k_size=(5,5),n_filters=32,stride=4)
    out=ReLU(out)
    out=conv3(out,k_size=(5,5),n_filters=32,stride=4)
    out=ReLU(out)
    out=conv4(out,k_size=(3,3),n_filters=64,stride=1)
    out=ReLU(out)


# c ssco: set of scores after CNN
ssco

eq_2=0

# c o_edge: one edge
for o_edge in set_of_sparse_edges_epsilon:
    # c o_comp: one comparison, 
    # 0 -> =
    # 1 -> i is darker
    # 2 -> i is brighter
    for o_comp in set_of_comparisons:
        # c p_i: pixel i
        p_i=o_edge[i]
        p_j=o_edge[j]
        
        # c rref_s: relative reflectance score
        rref_s=ssco[p_j,p_j,o_comp]

        if o_comp==0:
            # c mu_t: mu term
            # c pred_r: predicted reflectance
            mu_t=abs(pred_r[i]-pred_r[j])
        elif o_comp==1:
            mu_t=max(pred_r[i]-pred_r[j],0)
        elif o_comp==2:
            mu_t=max(pred_r[j]-pred_r[i],0)

        eq_2+=rref_s*mu_t

E_x=arg_x_when_minimize(eq_2)

# Not clear:
# Suppose you have 2 patches
# Each patch is is (63,63), in other words, each patch has 63*63 pixels
# You pass 2 patches into CNN which shares weights when processing 2 patches
# You resize input image into (150,150) image and also pass it CNN
# Then, how is (x1,y1,x2,y2) determined?
# Do I need to choose each pixel from each patch?
# Since JSON relative reflectance has comparison information,
# do I need to choose each pixel based on pixel locaions of JSON?
# Then, do I need to Fig 2 process iteratively for all pixels in each patch?
# p_1_1: patch 1 1
output1=[p_1_1,...,p_1_64]
output2=[p_2_1,...,p_2_64]
# s_1_1: small image 1 1
output3=[s_1_1,...,s_1_64]
# c pix_loca: pixel location
pix_loca=[x1,y1,x2,y2]
concat=[p_1_1,...,p_1_64,p_2_1,...,p_2_64,s_1_1,...,s_1_64,x1,y1,x2,y2]

out=FC1(concat,196,128)
out=FC2(out,128,128)
out=FC3(out,128,3)


# ======================================================================
# 3. Intrinsic image decomposition 

# c sum_un_lo: sum of unary losses, 
# unary term also constrains reflectance and shading to reconstruct original image
sum_un_lo=0
for i in all_pixels:
    # c u_lo: unary loss
    u_lo=unary_f(r[i],s[i])
    sum_un_lo+=u_lo

# c sum_pw_lo: sum of pairwise losses
# Most of heavy lifting of model is done by pairwise terms \psi^r and \psi^s 
# that enforce smoothness of reflectance and lighting respectively.
sum_pw_lo=0
for i in all_i_pixs_gt_j:
    # c u_lo: unary loss
    pw_r_lo=pairwise_f_r(r[i],s[i])
    pw_s_lo=pairwise_f_s(r[i],s[i])
    p_lo=pw_r_lo+pw_s_lo
    sum_pw_lo+=p_lo

e_func=sum_un_lo+sum_pw_lo
s,r=arg_s_r_when_minimize(e_func)


def pairwise_f_s(r_i,r_j):
    """
    Act
      Pairwise shading term is modeled 
      as fully connected smoothness prior
    """
    fir_t=(s[i]-s[j])**2
    # c p[i]: position of pixel i
    # c beta_1: parameter controlling spatial extent of prior
    sec_t=-beta1*(p[i]-p[j])**2
    # c ent_t: this prior captures intuition 
    # that shading varies smoothly over smooth surfaces.
    ent_t=fir_t*exp(sec_t)
    return ent_t


def pairwise_f_r(r_i,r_j):
    """
    Act
      pairwise reflectance term is modeled 
      as color sensitive regularizer 
      encouraging pixels with similar color value in original image 
      to take similar reflectance.
      This reflectance term is quite arbitrary,
      as original color values are usually not good sole predictor of reflectance.
      
      In rest of this section, 
      we will show how to replace this term with our data-driven pairwise reflectance prior.
      
      Overall energy E(s,r) is optimized using alternating optimization for s and r.
      Reflectance term r is optimized using mean-field inference algorithm of Krahenbuhl and Koltun [15],
      while shading term is optimized with iteratively reweighted least squares (IRLS)
    """
    fir_t=abs(r_i-r_j)
    sec_t=-beta2*(p[i]-p[j])**2
    # c I[i]: color value of pixel i
    # beta2, beta3: control spatial and color extent of prior
    thi_t=-beta3*(I[i]-I[j])**2
    ent_t=fir_t*exp(sec_t+thi_t)
    return ent_t


# ======================================================================
# 3.1. Data-driven reflectance prior 

def dd_pairwise_f_r(r_i,r_j):
    """
    Act
      We show how to incorporate our relative reflectance classifier 
      into mean-field inference for reflectance
      dd_pairwise_f_r: data-driven-prior-added term
      main difficulty here is to evaluate pairwise term densely over image.
    """
    eq_4=0
    # c o_edge: one edge
    for o_edge in set_of_sparse_edges_epsilon:
        # c o_comp: one comparison, 
        # 0 -> =
        # 1 -> i is darker
        # 2 -> i is brighter
        for o_comp in set_of_comparisons:
            # c p_i: pixel i
            p_i=o_edge[i]
            p_j=o_edge[j]
            
            # c rref_s: relative reflectance score
            rref_s=ssco[p_j,p_j,o_comp]

            if o_comp==0:
                # c mu_t: mu term
                # c pred_r: predicted reflectance
                mu_t=abs(pred_r[i]-pred_r[j])
            elif o_comp==1:
                mu_t=max(pred_r[i]-pred_r[j],0)
            elif o_comp==2:
                mu_t=max(pred_r[j]-pred_r[i],0)

            eq_4+=mu_t*rref_s
    
    return eq_4


# Not clear under Eq 4
# mean-field inference algorithm relies on efficient evaluation.
# Message passing
\tilde{Q}_i(r_i) = j P r_j \psi^r_{ij} (r_i,r_j)Q(r_i)

# ======================================================================
# 3.2. Nystrom approximation

# c w_o: matrix approximated by Nystron's method
w_o

# W: matrix which has all classifier outputs
W

multiplied=W*[Q_1(l),0,Q_2(l),0,Q_3(l),...]^T

extracted=extract_every_other_elements(multiplied)

# c sample: 2k sampled rows
C=randomly_sample_2k_rows(W)


# c d_pw_c_m: dense pairwise classifier matrix
# Nyström then approximates dense pairwise classifier matrix
# c D: K×K matrix corresponding to the dense pairwise classifier scores between all sampled points
# + refers to the pseudo-inverse
d_pw_c_m=C*D^+*C^T

# ======================================================================
# 4.1. Data augmentation

# symmetry and transitivity of comparisons. 
# DA generates pixel pairs that are spatially distant from each other 
# (in contrast to ones originally derived from edges of a Delauney triangulation [7]). 
# We create augmented training and test annotations as follows:
# 1. Remove low-quality comparisons with human confidence score < 0.5.
# 2. For each remaining pairwise comparison (r_i, r_j ), 
# augment annotation for (r_j, r_i) by either flipping (if r_i \ne r_j ) or keeping (if r_i = r_j ) sign.
# 3. For any unannotated pair of reflectances (r_i, r_j) 
# that share comparison with r_k, 
# we augment it using following rules: 
# 1) r_i = r_j, iff r_i = r_k and r_j = r_k for all connected r_k
# 2) r_i > r_j, iff r_i \ge r_k > r_j or r_i > r_k \ge r_j 
# 3) r_i < r_j, iff r_i < r_k \le r_j or r_i \le r_k < r_j
# If any pairwise comparisons are inconsistent we do not complete them.
# This step is done repetitively for each image until no further augmentation is possible.
# Our augmentation generates 22903366 comparisons in total, 
# out of which 18621626 are used for training and 4281740 for testing.
