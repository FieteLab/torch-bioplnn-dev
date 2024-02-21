import os

import torch
import numpy as np
import scipy
import scipy.interpolate


def r_theta_mp(data, use_np=False):
    if use_np:
        tmp = np.exp(data[0] + 1j * data[1])-0.5
        return np.abs(tmp), np.angle(tmp)
    else:
        tmp = torch.exp(data[0] + 1j * data[1])-0.5
        return torch.abs(tmp), torch.angle(tmp)


def normalize_for_mp(indices, N_x=150, N_y=300, retina_radius=80, use_np=False):
    x, y = indices
    normalized_x = (1 - (x)/retina_radius)*2.4 - 0.6
    if use_np:
        normalized_y = ((y-N_y//2)/np.sqrt(retina_radius**2.))*3.5
    else:
        normalized_y = ((y-N_y//2)/torch.sqrt(retina_radius**2.))*3.5
    return normalized_x,normalized_y


def image2v1(image, retina_indices, image_top_corner=(4,4), N_x=150, N_y=300, retina_radius=80, mode='vector'):
    use_np = type(image) is np.ndarray
    image_x, image_y = image.shape[1:] # (C, H, W)
    img_ind = np.zeros((2,image_x,image_y))
    img_ind[0,:,:] = np.tile(0+np.arange(image_x),(image_y,1)).T /image_x*image_top_corner[0]
    img_ind[1,:,:] = np.tile(np.arange(image_y) - image_y//2,(image_x,1)) /image_y*image_top_corner[1]*2

    flat_img_ind = img_ind.reshape((2, image_x * image_y))

    normed_indices_retina = normalize_for_mp(retina_indices, N_x, N_y, retina_radius, use_np=True)
    r_indices, theta_indices = r_theta_mp(normed_indices_retina, use_np=True)

    v_field_x = r_indices * np.cos(theta_indices)
    v_field_y = r_indices * np.sin(theta_indices)

    if not use_np:
        device = image.device
        dtype = image.dtype
        if device.type != 'cpu':
            image = image.cpu()
        image = image.numpy()
    
    if len(image.shape) == 3:
        img_on_vfield = [scipy.interpolate.griddata(flat_img_ind.T,im.flatten(),np.array((v_field_x,v_field_y)).T) for im in image]
        img_on_vfield = np.stack(img_on_vfield, axis=-1)
    else:
        img_on_vfield = scipy.interpolate.griddata(flat_img_ind.T,image[0].flatten(),np.array((v_field_x,v_field_y)).T)


    if mode == 'image':
        retina_mask = np.zeros((N_x,N_y, img_on_vfield.shape[-1]))
        retina_mask[retina_indices[0], retina_indices[1]] = img_on_vfield
        xmin, ymin = retina_indices.min(1)
        xmax, ymax = retina_indices.max(1)
        img_on_vfield = retina_mask[xmin:xmax+1, ymin:ymax+1].transpose(2, 0, 1)

    if not use_np:
        img_on_vfield = torch.from_numpy(img_on_vfield).to(device).type(dtype)
    img_on_vfield = torch.nan_to_num(img_on_vfield)
    return img_on_vfield, (v_field_x,v_field_y)


def vfield_to_V1(image_on_vfield, retina_indices, N_x=150, N_y=300, retina_radius=80):
    v_field_x = retina_indices[0]
    v_field_y = retina_indices[1]
    use_np = type(image_on_vfield) is np.ndarray

    if use_np:
        tmp = np.log(v_field_x+1j*v_field_y+0.5)
        retina_mask = np.zeros((N_x,N_y))
    else:
        tmp = torch.log(v_field_x+1j*v_field_y+0.5)
        retina_mask = torch.zeros((N_x,N_y))

    retina_x = (1 - (tmp.real+0.6)/2.4)*retina_radius
    retina_y = (tmp.imag/3.5)*retina_radius+N_y//2

    x = np.round(retina_x).astype('int')
    y = np.round(retina_y).astype('int')
    retina_mask[x, y] = image_on_vfield

    return retina_mask


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class RunningMean(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = 0
        self.count = epsilon

    def update(self, x):
        batch_mean = x.mean(0).item()
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_count)

    def update_from_moments(self, batch_mean, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        new_count = batch_count + self.count

        self.mean = new_mean
        self.count = new_count


if __name__ == '__main__':
    image = np.zeros((3, 32,32))
    image = torch.zeros((3, 32, 32))
    path = '../connection'
    layer = 'V1_indices'
    data = np.load(os.path.join(path, f'{layer}.npy'))
    Ny = 300
    retina_indices =  data
    image_top_corner = (16,16)
    image_on_vfield, vfield = image2v1(image, retina_indices, image_top_corner,N_x=150,N_y=300,retina_radius=80)
    retina_inp = vfield_to_V1(image_on_vfield, vfield, retina_indices, N_x=150, N_y=300, retina_radius=80)
    breakpoint()
