import os
import numpy as np
import shutil
import random
from PIL import Image

root_dir = 'C:/Ph.d1/53. goggle Segmentation UNET/Gog_Beard_Mou/DatasetV2/JPEGImages'

dst_dir='C:/Ph.d1/53. goggle Segmentation UNET/Gog_Beard_Mou/DatasetV2/Non_Augment/80_10_10'
allFileNames=os.listdir(root_dir)
np.random.shuffle(allFileNames)

val_ratio = 0.10
test_ratio = 0.10


train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)* (1 - (val_ratio + test_ratio))),
                                                            int(len(allFileNames)* (1 - test_ratio))])











train_FileNames_path = [root_dir+'/'+ name for name in train_FileNames.tolist()]
val_FileNames_path = [root_dir+'/' + name for name in val_FileNames.tolist()]
test_FileNames_path = [root_dir+'/' + name for name in test_FileNames.tolist()]

train_mask_FileNames=[name.split('.')[0]+'.png'  for name in train_FileNames ]
test_mask_FileNames=[name.split('.')[0]+'.png'  for name in test_FileNames ]
val_mask_FileNames=[name.split('.')[0]+'.png'  for name in val_FileNames ]



mask_dir='C:/Ph.d1/53. goggle Segmentation UNET/Gog_Beard_Mou/DatasetV2/SegmentationClassPNG'

train_mask_FileNames_path = [mask_dir+'/'+ name for name in train_mask_FileNames]
val_mask_FileNames_path = [mask_dir+'/' + name for name in val_mask_FileNames]
test_mask_FileNames_path = [mask_dir+'/' + name for name in test_mask_FileNames]



# print(len(train_FileNames_path))
# print(len(train_mask_FileNames))

# print(len(val_FileNames))
# print(len(test_FileNames))


classes_dir=['train','test','valid','train_GT','test_GT','valid_GT','train_mask','valid_mask','test_mask']
for cls in classes_dir:
    os.makedirs(dst_dir +'/'+cls)
   



# Copy-pasting images
for name in train_FileNames_path:
    shutil.copy(name, dst_dir +'/'+'train/' )

for name in val_FileNames_path :
    shutil.copy(name, dst_dir +'/'+'valid/' )

for name in test_FileNames_path:
    shutil.copy(name, dst_dir +'/'+'test/')
   
   
print('***********DONE Spiltting ***********')
   
# Copy-pasting images
for name in train_mask_FileNames_path:
    shutil.copy(name, dst_dir +'/'+'train_GT/' )

for name in val_mask_FileNames_path:
    shutil.copy(name, dst_dir +'/'+'valid_GT/' )

for name in test_mask_FileNames_path:
    shutil.copy(name, dst_dir +'/'+'test_GT/')
   
   
print('***********DONE GROUND TRUTH***********')
   
   


for img in train_mask_FileNames_path:
    # print(img)
    mask_name=img.split('/')[6]
    # print(mask_name)
    mt=np.array(Image.open(img))
    mt1=Image.fromarray(mt)
    mt1.save(dst_dir+'/train_mask'+'/'+mask_name)
   
   
print('***********DONE TRAIN MASK GROUND TRUTH***********')
for img in test_mask_FileNames_path:
    # print(img)
    mask_name=img.split('/')[6]
    # print(mask_name)
    mt=np.array(Image.open(img))
    mt1=Image.fromarray(mt)
    mt1.save(dst_dir+'/test_mask'+'/'+mask_name)
   
   
print('***********DONE TEST MASK GROUND TRUTH***********')
   
   

for img in val_mask_FileNames_path:
    # print(img)
    mask_name=img.split('/')[6]
    # print(mask_name)
    mt=np.array(Image.open(img))
    mt1=Image.fromarray(mt)
    mt1.save(dst_dir+'/valid_mask'+'/'+mask_name)
   
   
   
print('***********DONE VALID MASK GROUND TRUTH***********')  
   
   

# dst='C:/Ph.d1/53. goggle Segmentation UNET/Gog_Beard_Mou/DatasetV2/Non_Augment/60_20_20'

# # classes_dir = ['train', 'test','valid','train_mask','test_mask','val_mask']

# classes_dir = ['images', 'masks'] #total labels

# for cls in classes_dir:
#     os.makedirs(dst +'train/' + cls)
#     os.makedirs(dst +'valid/' + cls)
#     os.makedirs(dst +'test/' + cls)
 


# all_maskfiles='C:/Ph.d1/53. goggle Segmentation UNET/Gog_Beard_Mou/DatasetV2/SegmentationClassPNG'













# for img in train_FileNames:
#     name=img.split('/')[9]
#     mask_name=name[:-4]+'.png'
#     print(name)
#     print(mask_name)
   
   
#     t=Image.open(img)
#     mt=Image.open(dst+'/'+'masks'+'/'+mask_name)
   
   
#     t.save(dst+'/90_10_10/train'+'/'+name)
#     mt.save(dst+'/90_10_10/train_mask'+'/'+mask_name)






   
# # # test_dir='C:/Ph.d1/53. goggle Segmentation UNET/Gog_Beard_Mou/DatasetV1/Augment/Dataset_1065/90_10_10/test'
# # # test_FileNames=os.listdir(test_dir)
# # # test_file_path=[]

# # # for img in test_FileNames:
# # #    # print(img)
# # #     path=os.path.join(test_dir+'/'+img)
# # #     test_file_path.append(path)





# # for img in test_file_path:
# #     name=img.split('/')[9]
# #     mask_name=name[:-4]+".png"
   
# #     # t=Image.open(img)
# #     mtest=Image.open(dst+'/'+'masks'+'/'+mask_name)
# #     print(mask_name)
   
   
# #     # t.save(dst+'/90_10_10/test'+'/'+name)
   
# #     mtest.save(dst+'/90_10_10/test_mask'+'/'+mask_name)
 
   
   
   
   
# # print(len(test_file_path))