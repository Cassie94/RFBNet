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
    size_range = [.02, .2]
    size_list = ['small', 'medium', 'large']
    assert len(size_range) + 1 == len(size_list)
    obj_size_index = list(range(len(size_list)))
    iou_param = [(1.5,.65), (1.25, .8), (1, 1)]
    param_name_list = ['-'.join([str(xx) for xx in x]) for x in iou_param]

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
    # pdb.set_trace()
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    npos_size = {}
    det_index = {}

    for x in size_list:
        npos_size[x] = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        size = np.array([x['size'] for x in R])
        img_size = recs[imagename][0]['img_size']
        # det = [False] * len(R)
        npos = npos + sum(~difficult)
        npos_size[size_list[0]] += sum((size <= size_range[0]) & (~difficult))
        npos_size[size_list[1]] += sum((size > size_range[0]) & (size < size_range[1]) & (~difficult))
        npos_size[size_list[2]] += sum((size > size_range[1]) & (~difficult))
        det_index[imagename], nms_count, det = ( {} for i in range(3))
        max_score, max_overlap, temp_max_iou = ( {} for i in range(3))
        for param_name in param_name_list:
            for xx in [max_score, max_overlap, temp_max_iou]:
                xx[param_name] = [-1.] * len(R)
            det_index[imagename][param_name] = {}
            nms_count[param_name] = {}
            det[param_name] = [False] * len(R)
            for thres in ovthresh:
                det_index[imagename][param_name][thres] = [False] * len(R)
                nms_count[param_name][thres] = [0] * len(R)
        # max_score, max_overlap, temp_max_iou =([-1.] * len(R) for i in range(3))
        # nms_count = {}
        # for thres in ovthresh:
        #     det_index[imagename][thres] = [False] * len(R)
        #     nms_count[thres] = [0] * len(R)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det,
                                 'max_score': max_score,
                                 'temp_max_iou': temp_max_iou,
                                 'max_overlap': max_overlap,
                                 'nms_count': nms_count,
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
    sorted_scores = confidence[sorted_ind]
    # sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp, fp, obj_size = ({} for i in range(3))
    for param_name in param_name_list:
        obj_size[param_name] = np.zeros(nd)
        tp[param_name], fp[param_name] = ({} for i in range(2))
        for thres in ovthresh:
            tp[param_name][thres] = np.zeros(nd)
            fp[param_name][thres] = np.zeros(nd)
    # for thres in ovthresh:
    #     tp[thres] = np.zeros(nd)
    #     fp[thres] = np.zeros(nd)
    #
    # for param_name in param_name_list:
    #     obj_size[param_name] = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        det = det_index[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        gt_iou = R['max_overlap']
        gt_score = R['max_score']
        gt_temp_iou = R['temp_max_iou']
        gt_nms_count = R['nms_count']
        img_size = R['img_size']

        if BBGT.size == 0:
            for k,v in obj_size.items():
                v[d] = (bb[2] - bb[0]) * (bb[3] - bb[1]) / (img_size[0] * img_size[1])
            # obj_size[d] = (bb[2] - bb[0]) * (bb[3] - bb[1]) / (img_size[0] * img_size[1])
        else:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih
            # obj_size[d] = R['size'][jmax]

                # union for different iou_param
            for (alpha, beta), param_name in zip(iou_param, param_name_list):
                param_name = '-'.join([str(alpha), str(beta)])
                uni = (beta * (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       alpha * (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) + (1 - alpha - beta) * inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                obj_size[param_name][d] = R['size'][jmax]
                # Statistic the max_score or max_iou for each gt box
                if ovmax > 0.01:
                    if not R['det'][param_name][jmax]:
                        gt_iou[param_name][jmax] = ovmax
                        R['det'][param_name][jmax] = 1
                if ovmax > gt_temp_iou[param_name][jmax]:
                    gt_temp_iou[param_name][jmax] = ovmax
                    gt_score[param_name][jmax] = sorted_scores[d]

                for thres in ovthresh:
                    if ovmax > thres:
                        if not R['difficult'][jmax]:
                            if not det[param_name][thres][jmax]:
                                tp[param_name][thres][d] = 1.
                                det[param_name][thres][jmax] = 1
                            else:
                                fp[param_name][thres][d] = 1.
                                gt_nms_count[param_name][thres][jmax] += 1
                    else:
                        fp[param_name][thres][d] = 1.

    # Analysis the max_score, max_iou, size for each gt_box
    size_res, score_res, iou_res, nms_res = ({} for i in range(4))
    for param_name in param_name_list:
        for x in [size_res, score_res, iou_res]:
            x[param_name] = []
        nms_res[param_name] = {}
        for thres in ovthresh:
            nms_res[param_name][thres] = []

        for k,v in class_recs.items():
            if len(v['max_score'][param_name]) > 0:
                for x,xx in zip([score_res[param_name], iou_res[param_name], size_res[param_name]], \
                    ['max_score', 'max_overlap', 'size']):
                    x += list(v[param_name][xx])
                for kk,vv in v['nms_count'][param_name].items():
                    nms_res[kk] += vv

    # CALCULATE THE AP, RECALL, PRECISE FOR DIFFERENT SIZE OBJECTS.
    # calculate rec,prec,ap for small/medium/large objects
    rec_thres, prec_thres, ap_thres = ({} for i in range(3))
    for param_name in param_name_list:
        obj_param_size = obj_size[param_name]
        size_index = np.piecewise(obj_param_size, [obj_param_size<=size_range[0], \
            (obj_param_size>size_range[0])*(obj_param_size<=size_range[1]), \
            obj_param_size>size_range[1]], obj_size_index)
        for x in [rec_thres, prec_thres, ap_thres]:
            x[param_name] = {}
        for thres in ovthresh:
            for x in [rec_thres, prec_thres, ap_thres]:
                x[param_name][thres] = {}
            # rec_thres[param_name][thres], prec_thres[thres], ap_thres[thres] =({} for i in range(3))
            fp_size, tp_size, rec_size, prec_size, ap_size = ({} for i in range(5))
            for x,xx in zip(size_list, obj_size_index):
                fp_size[x] = np.cumsum(fp[param_name][thres][size_index==xx])
                tp_size[x] = np.cumsum(tp[param_name][thres][size_index==xx])
                rec_size[x] = tp_size[x] / float(npos_size[x])
                prec_size[x] = tp_size[x] / np.maximum(tp_size[x] + fp_size[x], np.finfo(np.float64).eps)
                ap_size[x] = voc_ap(rec_size[x], prec_size[x], use_07_metric)
            rec_thres[param_name][thres]['size'] = rec_size
            prec_thres[param_name][thres]['size'] = prec_size
            ap_thres[param_name][thres]['size'] = ap_size
            # compute precision recall for all the objects
            fp_whole = np.cumsum(fp[param_name][thres])
            tp_whole = np.cumsum(tp[param_name][thres])
            rec = tp_whole / float(npos)
                # avoid divide by zero in case the first detection matches a difficult
                # ground truth
            prec = tp_whole / np.maximum(tp_whole + fp_whole, np.finfo(np.float64).eps)
            ap = voc_ap(rec, prec, use_07_metric)
            rec_thres[param_name][thres]['whole'] = rec
            prec_thres[param_name][thres]['whole'] = prec
            ap_thres[param_name][thres]['whole'] = ap
    return rec_thres, prec_thres, ap_thres, size_res, score_res, iou_res,nms_res
