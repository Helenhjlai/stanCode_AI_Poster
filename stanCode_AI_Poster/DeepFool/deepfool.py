import collections
import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
import torch.nn as nn
import torch.nn.functional as F
#from torch.autograd.gradcheck import zero_gradients

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")

    image_var = Variable(image[None, :, :, :], requires_grad=True)
    # output = net.forward(image_var)
    # print("output.shape", output.shape)
    # print("output:", output)
    # print("output.data:", output.data)
    # print("output.data.cpu():", output.data.cpu())
    # print("output.data.cpu().numpy():", output.data.cpu().numpy())
    # print("output.data.cpu().numpy().flatten():", output.data.cpu().numpy().flatten())
    output = net.forward(image_var)
    f_image = output.data.cpu().numpy().flatten() # dim = 1000,
    
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes] # ten indices of max value
    label = I[0]

    orig_prob = F.softmax(output, dim=1)
    orig_confidence = round(torch.max(orig_prob.data, 1)[0][0].item(), 4)
    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape) # 3 x 256 x 256
    # print("w shape:", w.shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x) # 1 x 1000
    fs_list = [fs[0,I[k]] for k in range(num_classes)] # ten max value of output
    k_i = label
    # print("fs[0, I[0]]", fs[0, I[0]])
    # print("fs[0, I[0]].backward()", fs[0, I[0]].backward())
    # print("fs[0, I[0]]", fs[0, I[0]])
    # print("fs_list:", fs_list)
    total_output_list = []
    while k_i == label and loop_i < max_iter:

        pert = np.inf # positive infinite
        # print("x.shape:", fs[0, I[0]].shape)
        fs[0, I[0]].backward(retain_graph=True)
        
        
        grad_orig = x.grad.data.cpu().numpy().copy() # 1x3x256x256 
        # print("grad_orig.shape", grad_orig.shape)
        
        for k in range(1, num_classes):
            outpuit_list = []
            # print("x.grad:", x.grad)
            zero_gradients(x) # make x.grad = 0
            # print("x.grad:", x.grad)
            # nn.Module.zero_grad(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()
            # print("cur_grad:", cur_grad)

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            # print(k, "loop, w_k:", w_k)
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()
            # np.linalg.norm(w_k.flatten()) -> length of vector w_k
            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())
        adv_prob = F.softmax(fs, dim=1)
        adv_confidence = round(torch.max(adv_prob.data, 1)[0][0].item(), 4)
        total_output_list.append(k_i)
        loop_i += 1

    r_tot = (1+overshoot)*r_tot
    pert_diff = pert_image - image

    return r_tot, loop_i, label, k_i, pert_image, total_output_list, orig_confidence, adv_confidence, pert_diff
