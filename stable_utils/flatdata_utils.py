
import torch
from datasets import load_dataset
import numpy as np


import skimage
import skimage.transform



def demosaic_raw(meas, norm_stats=None):
    tform = skimage.transform.SimilarityTransform(rotation=0.00174)
    X = meas.numpy()[0,:,:]
    X = X/65535.0
    im1=np.zeros((512,640,4))
    im1[:,:,0]=X[0::2, 0::2]#b
    im1[:,:,1]=X[0::2, 1::2]#gb
    im1[:,:,2]=X[1::2, 0::2]#gr
    im1[:,:,3]=X[1::2, 1::2]#r
    im1=skimage.transform.warp(im1,tform)
    im=im1[6:506,10:630,:]
    rowMeans = im.mean(axis=1, keepdims=True)
    colMeans = im.mean(axis=0, keepdims=True)
    allMean = rowMeans.mean()
    im = im - rowMeans - colMeans + allMean
    im = im.astype('float32')
    meas = np.swapaxes(np.swapaxes(im,0,2),1,2)[None, ...]
    return meas[0,:,:,:]


def Edata_demosaic_raw2(meas, norm_stats=None):
    X = meas
    X = X / 65535.0
    H, W = X.shape
    im1 = np.zeros((H // 2, W // 2, 4))
    im1[:, :, 0] = X[0::2, 0::2]  # b
    im1[:, :, 1] = X[0::2, 1::2]  # gb
    im1[:, :, 2] = X[1::2, 0::2]  # gr
    im1[:, :, 3] = X[1::2, 1::2]  # r
    im = im1
    rowMeans = im.mean(axis=1, keepdims=True)
    colMeans = im.mean(axis=0, keepdims=True)
    allMean = rowMeans.mean()
    im = im - rowMeans - colMeans + allMean
    im = im.astype('float32')
    meas = np.transpose(im, axes=(2,0,1))
    return meas

def Edata_demosaic_raw_fullsize(meas, tform, i_black, real_cap=False):
    X = meas
    X = X / 4095.0

    if not real_cap:
        X = X - i_black
        X = X.clip(0)
    H, W = X.shape

    im1 = np.zeros((H // 2, W // 2, 4))
    im1[:, :, 0] = X[0::2, 0::2]  # b
    im1[:, :, 1] = X[0::2, 1::2]  # gb
    im1[:, :, 2] = X[1::2, 0::2]  # gr
    im1[:, :, 3] = X[1::2, 1::2]  # r

    im1 = skimage.transform.warp(im1, tform)
    im1 = im1[10:-10, 10:-10]
    im = im1
    rowMeans = im.mean(axis=1, keepdims=True)
    colMeans = im.mean(axis=0, keepdims=True)
    allMean = rowMeans.mean()
    im = im - rowMeans - colMeans + allMean
    im = im.astype('float32')
    meas = np.transpose(im, axes=(2,0,1))
    return meas


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
    }