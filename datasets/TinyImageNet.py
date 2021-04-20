import imageio
import numpy as np
import os

from collections import defaultdict
from torch.utils.data import Dataset

from PIL import Image

from datasets.datasets_utils import getItem, buildLabelIndex

def _add_channels(img, total_channels=3):
  while len(img.shape) < 3:  # third axis is the channels
    img = np.expand_dims(img, axis=-1)
  while(img.shape[-1]) < 3:
    img = np.concatenate([img, img[:, :, -1:]], axis=-1)
  return img

class TinyImageNetPaths:
  def __init__(self, root_dir):

    train_path = os.path.join(root_dir, 'train')
    val_path = os.path.join(root_dir, 'val')
    test_path = os.path.join(root_dir, 'test')

    wnids_path = os.path.join(root_dir, 'wnids.txt')
    words_path = os.path.join(root_dir, 'words.txt')

    self._make_paths(train_path, val_path, test_path,
                     wnids_path, words_path)

  def _make_paths(self, train_path, val_path, test_path,
                  wnids_path, words_path):
    self.ids = []
    with open(wnids_path, 'r') as idf:
      for nid in idf:
        nid = nid.strip()
        self.ids.append(nid)
    self.nid_to_words = defaultdict(list)
    with open(words_path, 'r') as wf:
      for line in wf:
        nid, labels = line.split('\t')
        labels = list(map(lambda x: x.strip(), labels.split(',')))
        self.nid_to_words[nid].extend(labels)

    self.paths = {
      'train': [],  # [img_path, id, nid, box]
      'val': [],  # [img_path, id, nid, box]
      'test': []  # img_path
    }

    # Get the test paths
    self.paths['test'] = list(map(lambda x: os.path.join(test_path, x),
                                      os.listdir(test_path)))
    # Get the validation paths and labels
    with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
      for line in valf:
        fname, nid, x0, y0, x1, y1 = line.split()
        fname = os.path.join(val_path, 'images', fname)
        bbox = int(x0), int(y0), int(x1), int(y1)
        label_id = self.ids.index(nid)
        self.paths['val'].append((fname, label_id, nid, bbox))

    # Get the training paths
    train_nids = os.listdir(train_path)
    for nid in train_nids:
      anno_path = os.path.join(train_path, nid, nid+'_boxes.txt')
      imgs_path = os.path.join(train_path, nid, 'images')
      label_id = self.ids.index(nid)
      with open(anno_path, 'r') as annof:
        for line in annof:
          fname, x0, y0, x1, y1 = line.split()
          fname = os.path.join(imgs_path, fname)
          bbox = int(x0), int(y0), int(x1), int(y1)
          self.paths['train'].append((fname, label_id, nid, bbox))


class TinyImageNetDataset(Dataset):
  def __init__(self, root_dir, mode='train', preload=True, load_transform=None,
               transform=None, max_samples=None, num_imgs_per_cat = None, training_mode='SSL'):
    tinp = TinyImageNetPaths(root_dir)
    
    self.mode = mode
    self.label_idx = 1  # from [image, id, nid, box]
    self.preload = preload
    self.transform = transform
    self.transform_results = dict()

    self.IMAGE_SHAPE = (64, 64, 3)

    self.img_data = []
    self.label_data = []

    self.max_samples = max_samples
    self.samples = tinp.paths[mode]
    self.samples_num = len(self.samples)

    self.training_mode = training_mode
        
    if self.max_samples is not None:
      self.samples_num = min(self.max_samples, self.samples_num)
      self.samples = np.random.permutation(self.samples)[:self.samples_num]

    if self.preload:
      self.img_data = np.zeros((self.samples_num,) + self.IMAGE_SHAPE,
                               dtype=np.float32)
      self.label_data = np.zeros((self.samples_num,), dtype=np.int)
      for idx in range(self.samples_num):
        s = self.samples[idx]
        img = imageio.imread(s[0])
        img = _add_channels(img)
        self.img_data[idx] = img
        self.label_data[idx] = s[self.label_idx]


      if (num_imgs_per_cat is not None) and self.mode=='train':
            self._keep_first_k_examples_per_category(num_imgs_per_cat)
            
      self.samples_num = len(self.label_data)      
      if load_transform:
        for lt in load_transform:
          result = lt(self.img_data, self.label_data)
          self.img_data, self.label_data = result[:2]
          if len(result) > 2:
            self.transform_results.update(result[2])
            
  def _keep_first_k_examples_per_category(self, num_imgs_per_cat):
    print('num_imgs_per_category {0}'.format(num_imgs_per_cat))
   
    labels = self.label_data
    data = self.img_data
    label2ind = buildLabelIndex(labels)
    all_indices = []
    for cat in label2ind.keys():
        label2ind[cat] = label2ind[cat][:num_imgs_per_cat]
        all_indices += label2ind[cat]
    all_indices = sorted(all_indices)
    data = data[all_indices]
    labels = [labels[idx] for idx in all_indices]
    self.label_data = labels
    self.img_data = data

    label2ind = buildLabelIndex(labels)
    for k, v in label2ind.items(): 
        assert(len(v)==num_imgs_per_cat) 
            


  def __len__(self):
    return self.samples_num

  def __getitem__(self, idx):
    X = Image.fromarray(self.img_data[idx].astype('uint8'))
    target = self.label_data[idx]
    
    return getItem(X, target, self.transform, self.training_mode)