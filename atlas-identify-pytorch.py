#!/usr/bin/env python
import argparse,logging,json,glob
import numpy as np
import cfg,time,loss_func
from data_handler_calo2d_h5 import BatchGenerator
from torch.utils.data import DataLoader
import torch
from model_calo2d_yolo import Net
from torch import optim
logger = logging.getLogger(__name__)



def main():
   ''' simple starter program that can be copied for use when starting a new script. '''
   logging_format = '%(asctime)s %(levelname)s:%(name)s:%(message)s'
   logging_datefmt = '%Y-%m-%d %H:%M:%S'
   logging.basicConfig(level=logging.INFO,format=logging_format,datefmt=logging_datefmt)

   parser = argparse.ArgumentParser(description='')
   parser.add_argument('-c','--config_file',help='input',required=True)
   parser.add_argument('--num_files','-n', default=-1, type=int,
                       help='limit the number of files to process. default is all')
   parser.add_argument('--model_save',default='model_saves',help='base name of saved model parameters for later loading')
   parser.add_argument('-i','--input_model_pars',help='if provided, the file will be used to fill the models state dict from a previous run.')


   parser.add_argument('--debug', dest='debug', default=False, action='store_true', help="Set Logger to DEBUG")
   parser.add_argument('--error', dest='error', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--warning', dest='warning', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--logfilename',dest='logfilename',default=None,help='if set, logging information will go to file')
   args = parser.parse_args()

   if args.debug and not args.error and not args.warning:
      # remove existing root handlers and reconfigure with DEBUG
      for h in logging.root.handlers:
         logging.root.removeHandler(h)
      logging.basicConfig(level=logging.DEBUG,
                          format=logging_format,
                          datefmt=logging_datefmt,
                          filename=args.logfilename)
      logger.setLevel(logging.DEBUG)
   elif not args.debug and args.error and not args.warning:
      # remove existing root handlers and reconfigure with ERROR
      for h in logging.root.handlers:
         logging.root.removeHandler(h)
      logging.basicConfig(level=logging.ERROR,
                          format=logging_format,
                          datefmt=logging_datefmt,
                          filename=args.logfilename)
      logger.setLevel(logging.ERROR)
   elif not args.debug and not args.error and args.warning:
      # remove existing root handlers and reconfigure with WARNING
      for h in logging.root.handlers:
         logging.root.removeHandler(h)
      logging.basicConfig(level=logging.WARNING,
                          format=logging_format,
                          datefmt=logging_datefmt,
                          filename=args.logfilename)
      logger.setLevel(logging.WARNING)
   else:
      # set to default of INFO
      for h in logging.root.handlers:
         logging.root.removeHandler(h)
      logging.basicConfig(level=logging.INFO,
                          format=logging_format,
                          datefmt=logging_datefmt,
                          filename=args.logfilename)

   blocks = cfg.parse_cfg(args.config_file)
   net = Net(blocks)
   net.double()

   if args.input_model_pars:
      net.load_state_dict(torch.load(args.input_model_pars))


   net_opts = blocks[0]

   trainlist,validlist = get_filelist(net_opts,args)

   trainds = BatchGenerator(trainlist,net_opts)
   batch_size = int(net_opts['batch'])
   validds = BatchGenerator(validlist,net_opts)

   cfg.print_cfg(blocks)


   optimizer = optim.SGD(net.parameters(),lr=float(net_opts['learning_rate']), momentum=float(net_opts['momentum']))

   accuracyCalc = loss_func.ClassOnlyAccuracy()
   
   batch_time_sum = 0
   batch_time_sum2 = 0
   batch_time_n = 0 
   
   for epoch in range(2):
      logger.info(' epoch %s',epoch)
      
      batch_counter = 0

      file_indices = np.array(range(len(trainlist)))
      np.random.shuffle(file_indices)

      
   

      for file_index in np.nditer(file_indices):

         for batch_index in range(trainds.batches_per_file):

            data = trainds.get_batch(file_index,batch_index)

            start = time.time()
            # logger.info('batch_counter: %s',batch_counter)
            inputs = data['images']
            targets = data['truth']

            # logger.info('retrieved data shape: %s',inputs.shape)
            # logger.info('retrieved truth shape: %s',targets.shape)


            optimizer.zero_grad()
            net.train()
            outputs = net(inputs)

            loss = net.loss(outputs,targets)

            forward = time.time()
            logger.info('forward pass: %6.2f',forward - start)

            loss.backward()
            optimizer.step()
            
            end = time.time()
            logger.info('backward pass: %6.2f',end - forward)

            timediff = end - start
            batch_time_sum += timediff
            batch_time_sum2 += timediff * timediff
            batch_time_n += 1

            # print statistics
            if True: #batch_counter % 1 == 0:
               mean_time = batch_time_sum/batch_time_n/batch_size
               logger.info('[%3d, %5d] loss: %.3f sec/image: %6.2f',epoch + 1, batch_counter + 1, loss.item(),mean_time)


            if batch_counter % 10 == 9:
               net.eval()


               for i in range(1):

                  data = validds.get_next_batch()
                  inputs = data['images']
                  targets = data['truth']
                  outputs = net(inputs)
                  acc = accuracyCalc.eval_acc(outputs,targets)

                  loss = net.loss(outputs,targets)
                  logger.info('valid loss: %10.3f accuracy: %10.3f',loss.item(),acc)


            batch_counter += 1


            if batch_counter % 10 == 9:
               torch.save(net.state_dict(),args.model_save + '_%05d_%05d.torch_model_state_dict' % (epoch,batch_counter))

   
   

def get_filelist(net_opts,args):
   # get file list
   logger.info('glob dir: %s',net_opts['inglob'])
   filelist = sorted(glob.glob(net_opts['inglob']))
   logger.info('found %s input files',len(filelist))
   if len(filelist) < 2:
      raise Exception('length of file list needs to be at least 2 to have train & validate samples')

   nfiles = len(filelist)
   if args.num_files > 0:
      nfiles = args.num_files

   
   train_file_index = int(float(net_opts['train_fraction']) * nfiles)
   first_file = filelist[0]
   np.random.shuffle(filelist)
   assert first_file != filelist[0]

   train_imgs = filelist[:train_file_index]
   valid_imgs = filelist[train_file_index:nfiles]
   logger.info('training index: %s',train_file_index)
   while len(valid_imgs) * int(net_opts['evt_per_file']) / int(net_opts['batch']) < 1.:
      logger.info('training index: %s',train_file_index)
      train_file_index -= 1
      train_imgs = filelist[:train_file_index]
      valid_imgs = filelist[train_file_index:nfiles]

   logger.info('training files: %s; validation files: %s',len(train_imgs),len(valid_imgs))

   return train_imgs,valid_imgs




if __name__ == "__main__":
   main()
