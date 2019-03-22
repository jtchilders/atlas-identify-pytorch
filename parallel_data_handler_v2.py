import torch
import logging,time
import numpy as np
logger = logging.getLogger(__name__)


class BatchGenerator:
   def __init__(self,filelist,evt_per_file,
                batch_size,img_shape,grid_shape,
                num_classes,use_random=True,
                max_queue=10):
      self.filelist     = np.array(filelist)
      self.evt_per_file = evt_per_file
      self.batch_size   = batch_size
      self.img_shape    = img_shape   # (channel,height,width)
      self.grid_shape   = grid_shape  # (h,w)
      self.num_classes  = num_classes

      self.total_images = self.evt_per_file * len(self.filelist)
      self.total_batches = self.total_images // self.batch_size
      self.batches_per_file = self.evt_per_file // self.batch_size

      self.use_random   = use_random

      self.man          = torch.multiprocessing.Manager()
      self.queue        = self.man.Queue(maxsize=max_queue)

   @staticmethod
   def report_done(result):
      logger.info('done: %s',result)

   def start_file_pool(self,num_procs=2):
      if self.use_random:
         np.random.shuffle(self.filelist)

      args = []

      for f in self.filelist:
         args.append((f,self.img_shape,self.grid_shape,self.queue))

      p = torch.multiprocessing.Pool(num_procs)
      self.results = p.map_async(file_proc,args)
      
   def __len__(self):
      return self.total_batches

   def batch_gen(self):
      if self.use_random:
         np.random.shuffle(self.filelist)

      raw_coords = None
      raw_features = None
      truth_batch = None

      for _ in range(self.total_batches):

         for i in range(self.batch_size):
            # logger.info('getting data from queue: %s',self.queue.qsize())
            input_data = self.queue.get()
            # logger.info('got data from queue: %s',self.queue.qsize())
            single_raw,single_truth = input_data

            raw_coords,raw_features = self.merge_raw(single_raw,raw_coords,raw_features,i)
            truth_batch = self.merge_truth(single_truth,truth_batch)
            # logger.info('truth_batch: %s',truth_batch.shape)

         yield {'images': [raw_coords,raw_features],'truth': truth_batch}

         raw_coords     = None
         raw_features   = None
         truth_batch    = None

   def merge_raw(self,raw,raw_coords,raw_features,image_counter):
      #raw = torch.from_numpy(raw)
      # create coords that includes image_count
      new_raw_coords = torch.zeros([len(raw[1]),3]).long()
      new_raw_coords[...,0:2] = torch.from_numpy(raw[1])
      new_raw_coords[...,2]   = torch.full((len(raw[1]),),image_counter)

      # convert features to torch tensor
      new_raw_features = torch.from_numpy(raw[0]).float()

      # merge new features to list

      if raw_coords is None:
         raw_coords = new_raw_coords.long()
         raw_features = new_raw_features.float()
      else:
         raw_coords = torch.cat([raw_coords,new_raw_coords])
         raw_features = torch.cat([raw_features,new_raw_features])

      return raw_coords,raw_features

   def merge_truth(self,truth,truth_batch):
      # convert truth to torch
      new_truth = torch.from_numpy(truth[np.newaxis,...]).double()

      if truth_batch is None:
         truth_batch = new_truth
      else:
         truth_batch = torch.cat([truth_batch,new_truth])

      return truth_batch


def file_proc(args):
   filename,img_shape,grid_shape,queue = args
   # logger.info('file_proc: %s',filename)
   filegen = FileGenerator(filename,img_shape,grid_shape)
   filegen.set_random_image_retrieval()
   for data in filegen.image_gen():
      # logger.warning('putting file data on queue: %s',queue.qsize())
      queue.put(data)

   return filename


class FileGenerator:
   def __init__(self,filename,img_shape,grid_shape):
      self.filename     = filename
      self.img_width    = img_shape[2]
      self.img_height   = img_shape[1]
      self.grid_w       = grid_shape[1]
      self.grid_h       = grid_shape[0]

      self.use_random = False

   def open_file(self):
      try:
         logger.info('opening file: %s',self.filename)
         nf = np.load(self.filename,allow_pickle=True)
         self.raw = nf['raw']
         truth = nf['truth']

         # a = time.time()
         self.truth = convert_truth(truth,self.img_width,self.img_height,self.grid_w,self.grid_h)
         # logger.info('convert_truth time: %s',time.time() - a)
      except:
         logger.exception('exception received when opening file %s',self.filename)
         raise

   def __getitem__(self,idx):
      if not hasattr(self,'raw'):
         self.open_file()
      assert(idx < len(self.raw))
      return (self.raw[idx],self.truth[idx])

   def set_random_image_retrieval(self,flag=True):
      self.use_random = flag

   def image_gen(self):
      if not hasattr(self,'raw'):
         self.open_file()

      index_list = np.arange(len(self.raw))

      if self.use_random: np.random.shuffle(index_list)
      for i,idx in enumerate(index_list):
         # logger.warning('getting index %s, %s of %s',idx,i,len(index_list))
         yield (self.raw[idx],self.truth[idx])
      # logger.warning('reached end of file %s',self.filename)


def convert_truth(intruth,img_width,img_height,grid_w,grid_h,new_channels=2):
   pix_per_grid_w = img_width / grid_w
   pix_per_grid_h = img_height / grid_h

   intruth_size = len(intruth)

   new_truth = np.zeros((intruth_size,new_channels,grid_h,grid_w),dtype=np.int32)

   for img_num in range(intruth_size):

      img_truth = intruth[img_num]

      for obj_num in range(len(img_truth)):
         obj_truth = img_truth[obj_num]

         obj_exists = obj_truth[0]

         if obj_exists == 1:

            obj_center_x = obj_truth[1] / pix_per_grid_w
            obj_center_y = obj_truth[2] / pix_per_grid_h
            # obj_width    = obj_truth[3] / pix_per_grid_w
            # obj_height   = obj_truth[4] / pix_per_grid_h

            grid_x = int(np.floor(obj_center_x))
            grid_y = int(np.floor(obj_center_y))

            if grid_x >= grid_w:
               raise Exception('grid_x %s is not less than grid_w %s' % (grid_x,grid_w))
            if grid_y >= grid_h:
               raise Exception('grid_y %s is not less than grid_h %s' % (grid_y,grid_h))

            new_truth[img_num,0,grid_y,grid_x] = obj_exists
            new_truth[img_num,1,grid_y,grid_x] = np.argmax([np.sum(obj_truth[5:10]),np.sum(obj_truth[10:12])])

   return new_truth
