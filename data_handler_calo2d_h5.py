from torch.utils.data import Dataset
import torch,time
import h5py,logging
import numpy as np
logger = logging.getLogger(__name__)


class BatchGenerator(Dataset):
   
   def __init__(self,filelist,net_opts):

      self.filelist = filelist
      self.net_opts = net_opts

      self.imgs_per_file = int(self.net_opts['evt_per_file'])

      self.fileindex = -1
      self.batchindex = -1
      self.raw = None
      self.truth = None

      self.img_width = int(self.net_opts['width'])
      self.img_height = int(self.net_opts['height'])
      self.channels = int(self.net_opts['channels'])
      self.grid_h = 8#self.net_opts['grid_h']
      self.grid_w = 180#self.net_opts['grid_w']
      self.batch_size = int(self.net_opts['batch'])
      self.num_classes = int(self.net_opts['classes'])

      self.total_images = self.imgs_per_file * len(self.filelist)
      self.total_batches = self.total_images // self.batch_size
      self.batches_per_file = self.imgs_per_file // self.batch_size

   def __len__(self):
      return self.total_batches


   def get_next_batch(self):

      if self.batchindex == -1 and self.fileindex == -1:
         self.batchindex = 0
         self.fileindex = 0
      else:


         self.batchindex += 1
         if self.batchindex >= self.batches_per_file:
            self.fileindex += 1
            self.batchindex = 0

         if self.fileindex >= len(self.filelist):
            raise Exception('reached end of filelist')

      return self.get_batch(self.fileindex,self.batchindex)

   def get_batch(self,file_index,batch_index):
      # logger.info('in getitem idx: %s', idx)
      start = time.time()
      if self.fileindex != file_index or self.raw is None:
         # set historical file index
         self.fileindex = file_index

         try:
            filename = self.filelist[self.fileindex]
         except IndexError:
            logger.exception('index %s is greater than number of files in list %s',self.fileindex,len(self.filelist))
            raise

         try:
            logger.info('opening file: %s',filename)
            if '.h5' in filename:
               hf = h5py.File(filename,'r')
               self.raw = hf['raw']
               self.truth = hf['truth']
            elif '.npz' in filename:
               nf = np.load(filename)
               self.raw = nf['raw']
               self.truth = nf['truth']

            a = time.time()
            self.raw   = self.convert_raw(self.raw)
            logger.info('convert_raw time: %s',time.time() - a)
            self.truth = self.convert_truth(self.truth)
         except:
            logger.exception('exception received when opening file %s',filename)
            raise

      self.batchindex = batch_index
      
      logger.info('batch time: %s',time.time() - start)
      return {'images': torch.from_numpy(self.raw[batch_index*self.batch_size:(batch_index+1)*self.batch_size]).double(),
              'truth':  torch.from_numpy(self.truth[batch_index*self.batch_size:(batch_index+1)*self.batch_size]).double()}


   def convert_truth(self,truth):

      pix_per_grid_w = self.img_width / self.grid_w
      pix_per_grid_h = self.img_height / self.grid_h

      new_truth = np.zeros((len(truth),2,self.grid_h,self.grid_w),dtype=np.int32)

      for img_num in range(len(truth)):

         img_truth = truth[img_num]

         for obj_num in range(len(img_truth)):
            obj_truth = img_truth[obj_num]

            obj_exists   = obj_truth[0]

            if obj_exists == 1:

               obj_center_x = obj_truth[1] / pix_per_grid_w
               obj_center_y = obj_truth[2] / pix_per_grid_h
               # obj_width    = obj_truth[3] / pix_per_grid_w
               # obj_height   = obj_truth[4] / pix_per_grid_h

               grid_x = int(np.floor(obj_center_x))
               grid_y = int(np.floor(obj_center_y))

               if grid_x >= self.grid_w:
                  raise Exception('grid_x %s is not less than grid_w %s' % (grid_x,self.grid_w))
               if grid_y >= self.grid_h:
                  raise Exception('grid_y %s is not less than grid_h %s' % (grid_y,self.grid_h))

               new_truth[img_num,0,grid_y,grid_x] = obj_exists
               new_truth[img_num,1,grid_y,grid_x] = np.argmax([np.sum(obj_truth[5:10]),np.sum(obj_truth[10:12])])

      return new_truth


   def convert_raw(self,raw):
      shape = raw.shape
      new_shape = [shape[0],2,shape[2],shape[3]]
      new_raw = np.zeros(new_shape)
      new_raw[:,0,...] = np.sum(raw[:,8:12,...],axis=1)
      new_raw[:,1,...] = np.sum(raw[:,12:16,...],axis=1)
      return new_raw[...,:-1]
