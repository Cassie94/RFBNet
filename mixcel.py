import numpy as np
import pandas as pd
import seaborn as sns
import sys,os,torch
import xml.etree.ElementTree as ET
home=os.environ["HOME"]
anno_dir = os.path.join(home, 'data/VOCdevkit/VOC2012/Annotations/')
# anno_dir = os.path.join(home, 'Downloads/voc/VOCdevkit/VOC2012/Annotations/')
size = ['width', 'height']
pts = ['xmin', 'ymin', 'xmax', 'ymax']
bndbox = []
for anno_file in os.listdir(anno_dir):
    anno = ET.parse(os.path.join(anno_dir,anno_file)).getroot()
    img_size = []
    for ii in size:
        for jj in anno.iter(ii):
            img_size.append(float(jj.text))
    for obj in anno.iter('object'):
        box = obj.find('bndbox')
        bbox = [float(box.find(x).text) for x in pts]
        bbox[0::2] = [ii/img_size[0] for ii in bbox[0::2]]
        bbox[1::2] = [ii/img_size[1] for ii in bbox[1::2]]
        bndbox.append(bbox)
gt = torch.from_numpy(np.asarray(bndbox))

from data import VOC_300
from layers.functions import PriorBox
cfg = VOC_300
priorbox = PriorBox(cfg)
priors = priorbox.forward()
# anchor = priors.numpy()

# min_xy = torch.max(gt[:,None,:2], priors[:,:2])
# max_xy = torch.min(gt[:,None,2:], priors[:,2:])
# inter = torch.clamp(max_xy - min-xy, min=0, max=1)

from utils.box_utils import jaccard
# iou2 = jaccard(gt.double(), priors.double())
# iou2_2_max = torch.max(jaccard(gt.double(), priors.double(),), 1)[0]

iou_static = {}
for alpha in [1,2,5]:
    iou_static[alpha]={}
    for beta in [.2,.5, 1]:
        max_iou, max_ratio =
        iou_static[alpha][beta] = torch.max(jaccard(gt.double(),
            priors.double(), alpha, beta), 1)[0]



# gt_area = (gt[:,2] - gt[:,0]) * (gt[:,3] - gt[:,1])
iou_static_dict = {"gt_area": (gt[:,2] - gt[:,0]) * (gt[:,3] - gt[:,1])}
for k1,v1 in iou_static.items():
    for k2,v2 in v1.items():
        iou_static_dict['{}_{}'.format(k1,k2)] = v2
iou_df = pd.DataFrame.from_dict(iou_static_dict)

bin = np.concatenate((np.arange(0,0.1-1e-5,0.01),np.arange(0.1,0.5-1e-5,0.05),np.arange(0.5,1+1e-5,0.1)))
labels = [np.mean(bin[i:i+2]).round(3) for i in range(bin.shape[0]-1)]
iou_df['size_index']=pd.cut(iou_df['gt_area'],bin, labels=labels)
iou_df['size_range']=pd.cut(iou_df['gt_area'],bin)



ax = sns.violinplot(x="size_index", y="2_0.5", data=iou_df);plt.show()
ax = sns.boxplot(x="size_index", y="2_0.5", data=iou_df)
ax = sns.swarmplot(x="size_index", y="2_0.5", data=iou_df, color=".25");plt.show()

import pandas as pd
import seaborn as sns
import numpy as np
import sys,os

# subplot
f, axes = plt.subplots(3, 3, figsize=(9,9), sharex=True)
sns.despine(left=True)
for i in range(1,10):
    sns.boxplot(x="size_index", y=iou_df.columns[i], data=iou_df, ax=axes[(i-1)//3,(i-1)%3])
plt.show()

iou_des_dict = {}

for ii in iou_df.columns[1:10]:
    iou_des_dict[ii] = iou_df.groupby('size_range')[ii].describe()
    iou_des_dict[ii].insert(loc=0, column='ratio',
        value=iou_des_dict[ii]['count'].div(iou_des_dict[ii]['count'].sum()))
    iou_des_dict[ii].insert(loc=0, column='cumsum',
        value=iou_des_dict[ii]['ratio'].cumsum())

df_50 = pd.DataFrame()
df_50['ratio']=iou_des_dict['1_1']['ratio']
df_50['cumsum']=iou_des_dict['1_1']['cumsum']
# key_list = ['1_0.5', '1_1', '2_0.5', '5_0.5']
key_list = [ '1_1', '2_0.5', '2_1']
ratio_list = ['25%','50%','75%']
for kk in ratio_list:
    for k in key_list:
        df_50[k+'_'+kk] = iou_des_dict[k][kk]
df_50.round(3)

import pandas as pd
ap_res = {
'ap_baseline' : [80.4, 51.5, 81.4, 87.1, 65.4, 33.5, 66.0, 78.0],
'ap_max_iou(1.2,.8)': [80.5, 52.9, 81.6, 86.4, 65.1, 34.9, 65.7, 77.1],
'ap_max_iou_thres(1.2,.8,.4/.5/.5)': [79.9, 54.7, 80.5, 86.6, 63.8, 37.4, 63.8, 76.0],
'ap_max_iou(2,.5)' : [79.9, 54.1, 81.0, 86.7, 63.0, 35.2, 62.5, 76.7],
'ap_max_iou_thres(1.5,.65,.4/.5/.5)': [79.7, 56.4, 80.2, 85.6, 62.3, 37.6, 62.4, 74.5],
'ap_thres(.35/.5/.5)' : [79.4, 54.4, 80.0, 86.5, 63.7, 38.1, 62.9, 76.9],
'ap_weight_iou(2, .5)' : [78.9, 53.2, 80.2, 84.5, 62.0, 34.9, 62.2, 74.6],
'ap_max_iou_thres(2,.5,.45/.45/.45)': [ 79.0, 54.2, 80.1, 86.3, 61.6, 35.2, 62.5, 75.1]
}
ap_df = pd.DataFrame.from_dict(ap_res, orient='index', \
    columns=['ap@.5', 'small@.5', 'medium@.5', 'large@.5', \
    'ap@.7', 'small@.7', 'medium@.7', 'large@.7'])
print(ap_df)

gt_dist.insert(loc=1,column='ratio',value=gt_dist['count'].div(gt_dist['count'].sum()))
gt_dist.insert(loc=2,column='cumsum',value=gt_dist['proportion'].cumsum())

iou2_df = pd.DataFrame.from_dict({"gt_area":gt_area, "max_iou": iou2_max})
gt_df = iou2_df
bin = np.concatenate((np.arange(0,0.1-1e-5,0.01),np.arange(0.1,0.5-1e-5,0.05),np.arange(0.5,1+1e-5,0.1)))
gt_df['obj_area'] = pd.cut(gt_df['gt_area'], bin)
gt_dist = gt_df.groupby('obj_area')['max_iou'].describe()
gt_dist.insert(loc=1,column='ratio',value=gt_dist['count'].div(gt_dist['count'].sum()))
gt_dist.insert(loc=2,column='cumsum',value=gt_dist['proportion'].cumsum())
gt_dist.to_pickle("iou5.pkl")



anchor_area = (anchor[:, 2]-anchor[:, 0]) * (anchor[:, 3]-anchor[:, 1])

union = area_a + area_b - inter
return inter / union  # [A,B]

# # slice the huge matrix and calculate by steps
# step = 2000
# slice_pt = list(range(step, gt.shape[0], step))
# slice_pt.append(gt.shape[0] - slice_pt[-1])
# inter_area_list = []
# for sub_gt in np.split(gt, slice_pt):
#     max_xy = np.minimum(sub_gt[:,None,2:], anchor[:,2:])
#     min_xy = np.maximum(sub_gt[:,None,:2], anchor[:,:2])
#     inter = np.clip(max_xy - min_xy, 0, 1)
#     inter_area_list.append(inter[:,:,0] * inter[:,:,1])
#
# def find_nearest(array, count):
#     array = np.asarray(array)
#     step = array.shape[0]
#     dist_list = []
#     for value in np.arange(0,1,1./count):
#         idx = (np.abs(array - value)).argmin()
#         dist_list.append((idx/step, array[idx]))
#     return dist_list

h,x=np.histogram(gt_area,bins=10000,normed=True)
dx = x[1] - x[0]
f1 = np.cumsum(h)*dx
find_nearest(f1, 10)

home=os.environ["HOME"]
gt_df=pd.read_pickle(os.path.join(home, "Downloads/gt_df.pkl"))
bin = np.concatenate((np.arange(0,0.1-1e-5,0.01),np.arange(0.1,0.5-1e-5,0.05),np.arange(0.5,1+1e-5,0.1)))
gt_df['bin'] = pd.cut(gt_df['gt_area'], bin)
gt_dist = gt_df.groupby('bin')['max_iou'].describe()
gt_dist.insert(loc=1,column='proportion',value=gt_dist['count'].div(gt_dist['count'].sum()))

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_trisurf(x1, x2, y, cmap=plt.cm.viridis, linewidth=0.2)
fig.colorbar( surf, shrink=0.5, aspect=5)
ax.view_init(30, 45)
ax.set_xlabel('Missing area')
ax.set_ylabel('Context area')
ax.set_zlabel('IoU-alpha(1)-beta(1)')
plt.show()
