import onnxruntime
import numpy as np
import nibabel as nib
import sys
from scipy.special import softmax
import heapq
import assets
import logging

def eliminateNoise(label, minArea=16):
    neighbors=[(-1,0),(1,0),(0,-1),(0,1)]
                
    seen=set()
    position=[]
    heapq.heapify(position)

    island=0
    newLabel=np.zeros(label.shape)
    i, j, k=label.shape
    for z in range(k):
        for x in range(i):
            for y in range(j):
                
                if (label[x,y,z]!=0) and (x,y,z) not in seen:
                    island+=1
                    area=0
                    curIsland=set()
                    seen.add((x,y,z))
                    curIsland.add((x,y,z))
                    heapq.heappush(position, (x,y,z))


                    while position:
                        cur=heapq.heappop(position)
                        area+=1


                        for neighbor in neighbors:

                            if cur[0]-neighbor[0]<0 or cur[0]-neighbor[0]>=i: continue
                            if cur[1]-neighbor[1]<0 or cur[1]-neighbor[1]>=j: continue
#                             if cur[2]-neighbor[2]<0 or cur[2]-neighbor[2]>=k: continue    

                            if label[cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]]==label[x,y,z] and (cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]) not in seen:
                                seen.add((cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]))
                                curIsland.add((cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]))
                                heapq.heappush(position, (cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]))



                    for (posX, posY, posZ) in curIsland: 
                        if area<minArea:
                            newLabel[posX, posY, posZ]=2
                        else:
                            newLabel[posX, posY, posZ]=label[x,y,z]


    return newLabel


def cutoff(label):

    neighbors=[(1,1,0),(0,1,0),(-1,1,0),(-1,0,0),(-1,-1,0),(0,-1,0),(1,-1,0),(1,0,0)]
    surpos = [3,3,3,3,3,3,3,3]
    i, j, k=label.shape

    for z in range(k):
        for x in range(i):
            for y in range(j):
                if label[x,y,z] ==2:
                    nei = []
                    for neighbor in neighbors:
                        if x-neighbor[0]<0 or x-neighbor[0]>=i: continue
                        if y-neighbor[1]<0 or y-neighbor[1]>=j: continue
                        nei.append(label[x-neighbor[0], y-neighbor[1],z-neighbor[2]])
                    if nei == surpos:
                        label[x,y,z] = 3
    return np.array(label)

def get_center(image, mask, i, j, k):

    sample = image[i-16:i+16+1,j-16:j+16+1,k-1:k+1+1]
    center = mask[i:i+1+1,j:j+1+1,k]
    
    return sample, center

def patch_generator(image, mask, batch_size=200):
    x,y,z = image.shape
    
    image2 = np.clip(image, -100, 200)
    image2 += 100
    image2 /= 300

    positions = []

    for k in range(1, z-1, 1):
        for i in range(17, x-17, 2):
            for j in range(17, y-17, 2):
                sample, center = get_center(image2, mask, i, j, k)
                if center.any():
                    positions.append((i,j,k))

    batches = [positions[i:i+batch_size] for i in range(0, len(positions), batch_size)]

    for b in batches:
        samples = []
        for pos in b:
            sample, _ = get_center(image2, mask, *pos)
            samples.append(sample)

        yield np.array(samples), b


def inference(image, mask, model=None, batch_size=200):
    """ Runs trained model on raw scan, yielding a segmented result.

    The raw scan provided is expected to be skull-stripped (no bone regions).
    The returned scan is segmented, with each voxel being one of 5 classes:

    Class   Description
    -----   -----------
      0     Background
      1     Skull
      2     Subarachnoid
      3     Ventricle
      4     Shunt


    Parameters
    ----------

    image:
        Raw scan to be segmented. Expected to be skull stripped.

    mask:
        Boolean mask indicating relevant regions of brain.

    model:
        Path to model to be run (has to be in *.onnx format).

    batch_size:
        Size of batches of patches to be fed to model.

    Returns
    -------

    nibabel.Nifti1Image:
        Segmented version of raw scan.
    """

    if model is None:
        model = str(assets.model)

    patches = patch_generator(image.get_fdata(), 
                              mask.get_fdata(), 
                              batch_size=batch_size) 
    # Generates batches of patches (of batch_size)

    reconstructed = np.zeros_like(image.get_fdata()) # TODO: Look into doing things in-place using given image (but make it option as it is destructive)
    ort_session = onnxruntime.InferenceSession(model)

    for p, idx in patches:

        # compute ONNX Runtime output prediction
        p = np.moveaxis(p, -1, 1)
        ort_inputs = {ort_session.get_inputs()[0].name: p.astype(np.float32)}
        ort_outs = ort_session.run(None, ort_inputs)[0]
        ort_outs = ort_outs.reshape((ort_outs.shape[0], 5, 2, 2))

        softmax_out = softmax(ort_outs, axis=1)
        pred = np.argmax(softmax_out, axis=1)
        #breakpoint()

        # logging.info(f"reconstructed.shape = {reconstructed.shape}")
        # logging.info(f"pred.shape = {pred.shape}")
        for k in range(pred.shape[0]):
            x, y, z = idx[k]
            reconstructed[x:x+1+1,y:y+1+1,z] = pred[k,0,...]

    # Noise Reduction
    result_noNoise = eliminateNoise(reconstructed, minArea=64)
    filldots = cutoff(result_noNoise)

    final = nib.Nifti1Image(filldots, affine=None, header=image.header)

    return final
