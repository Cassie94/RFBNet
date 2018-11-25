# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
import pdb
from IPython import embed

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    size = [float(xx.text) for x in ['width', 'height'] for xx in tree.iter(x)]
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        obj_struct['img_size'] = size
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        obj_struct['size'] = (obj_struct['bbox'][2] - obj_struct['bbox'][0]) * \
            (obj_struct['bbox'][3] - obj_struct['bbox'][1]) / (size[0] * size[1])
        objects.append(obj_struct)

    return objects



def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=[0.5, 0.7],
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5, 0.7)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    size_range = [.02, .4]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    npos_size = {}
    size_list = ['small', 'medium', 'large']
    for x in size_list:
        npos_size[x] = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        size = np.array([x['size'] for x in R])
        img_size = recs[imagename][0]['img_size']
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        npos_size[size_list[0]] += sum((size <= size_range[0]) & (~difficult))
        npos_size[size_list[1]] += sum((size > size_range[0]) & (size < size_range[1]) & (~difficult))
        npos_size[size_list[2]] += sum((size > size_range[1]) & (~difficult))
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det,
                                 'size': size,
                                 'img_size': img_size}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = {}
    for thres in ovthresh:
        tp[thres] = np.zeros(nd)
        fp[thres] = np.zeros(nd)
    # tp = np.zeros(nd)
    # fp = np.zeros(nd)
    obj_size = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        img_size = R['img_size']

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

                # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
            obj_size[d] = R['size'][jmax]
        else:
            obj_size[d] = (bb[2] - bb[0]) * (bb[3] - bb[1]) / (img_size[0] * img_size[1])

        for thres in ovthresh:
            if ovmax > thres:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[thres][d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[thres][d] = 1.
            else:
                fp[thres][d] = 1.

    size_index = np.piecewise(obj_size, [obj_size<=size_range[0], \
        (obj_size>size_range[0])*(obj_size<=size_range[1]), obj_size>size_range[1]], [1,2,3])
    # calculate rec,prec,ap for small/medium/large objects
    rec_thres, prec_thres, ap_thres = ({} for i in range(3))
    for thres in ovthresh:
        rec_thres[thres], prec_thres[thres], ap_thres[thres] =({} for i in range(3))
        fp_size, tp_size, rec_size, prec_size, ap_size = ({} for i in range(5))
        for x,xx in zip(size_list, [1,2,3]):
            fp_size[x] = np.cumsum(fp[size_index==xx])
            tp_size[x] = np.cumsum(tp[size_index==xx])
            rec_size[x] = tp_size[x] / float(npos_size[x])
            prec_size[x] = tp_size[x] / np.maximum(tp_size[x] + fp_size[x], np.finfo(np.float64).eps)
            ap_size[x] = voc_ap(rec_size[x], prec_size[x], use_07_metric)
        rec_thres[thres]['size'] = rec_size
        prec_size[thres]['size'] = prec_size
        ap_size[thres]['size'] = ap_size
        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        rec_thres[thres]['whole'] = rec
        prec_thres[thres]['whole'] = prec
        ap_thres[thres]['whole'] = ap
    # pdb.set_trace()
    # return rec, prec, ap, rec_size, prec_size, ap_size
    return rec_thres, prec_thres, ap_thres
