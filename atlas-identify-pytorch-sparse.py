#!/usr/bin/env python
import argparse,logging,glob,socket,json
import numpy as np
import time,loss_func
#from data_handler_calo2d_sparse_npz import BatchGenerator
from parallel_data_handler import BatchGenerator
import torch
from model_calo2d_yolo_sparse import Net2D
from torch import optim
from CalcMean import CalcMean
logger = logging.getLogger(__name__)


def main():
   ''' simple starter program that can be copied for use when starting a new script. '''

   parser = argparse.ArgumentParser(description='')
   parser.add_argument('-c','--config_file',help='configuration file in json format',required=True)
   parser.add_argument('--num_files','-n', default=-1, type=int,
                       help='limit the number of files to process. default is all')
   parser.add_argument('--model_save',default='model_saves',help='base name of saved model parameters for later loading')
   parser.add_argument('--nsave',default=100,type=int,help='frequency in batch number to save model')

   parser.add_argument('--nval',default=100,type=int,help='frequency to evaluate validation sample in batch numbers')
   parser.add_argument('--nval_tests',default=1,type=int,help='number batches to test per validation run')

   parser.add_argument('--status',default=20,type=int,help='frequency to print loss status in batch numbers')

   parser.add_argument('--batch',default=-1,type=int,help='set batch size, overrides file config')

   parser.add_argument('--random_seed',default=0,type=int,help='numpy random seed')

   parser.add_argument('-i','--input_model_pars',help='if provided, the file will be used to fill the models state dict from a previous run.')
   parser.add_argument('-e','--epochs',type=int,default=10,help='number of epochs')

   parser.add_argument('--horovod',default=False, action='store_true', help="Setup for distributed training")

   parser.add_argument('--debug', dest='debug', default=False, action='store_true', help="Set Logger to DEBUG")
   parser.add_argument('--error', dest='error', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--warning', dest='warning', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--logfilename',dest='logfilename',default=None,help='if set, logging information will go to file')
   args = parser.parse_args()

   logging_format = '%(asctime)s %(levelname)s:%(name)s:%(process)s:%(thread)s:%(message)s'
   logging_datefmt = '%Y-%m-%d %H:%M:%S'
   log_level = logging.INFO

   if args.debug and not args.error and not args.warning:
      log_level = logging.DEBUG
   elif not args.debug and args.error and not args.warning:
      log_level = logging.ERROR
   elif not args.debug and not args.error and args.warning:
      log_level = logging.WARNING

   rank = 0
   nranks = 1
   if args.horovod:
      print('importing horovod')
      import horovod.torch as hvd
      print('imported horovod')
      hvd.init()
      rank = hvd.rank()
      nranks = hvd.size()
      logging_format = '%(asctime)s %(levelname)s:' + '{:05d}'.format(rank) + ':%(name)s:%(process)s:%(thread)s:%(message)s'

   if rank > 0 and log_level == logging.INFO:
      log_level = logging.WARNING

   logging.basicConfig(level=log_level,
                       format=logging_format,
                       datefmt=logging_datefmt,
                       filename=args.logfilename)

   logger.info('rank %s of %s',rank,nranks)
   logger.info('hostname:           %s',socket.gethostname())

   logger.info('config file:        %s',args.config_file)
   logger.info('num files:          %s',args.num_files)
   logger.info('model_save:         %s',args.model_save)
   logger.info('random_seed:        %s',args.random_seed)
   logger.info('nsave:              %s',args.nsave)
   logger.info('nval:               %s',args.nval)
   logger.info('status:             %s',args.status)
   logger.info('input_model_pars:   %s',args.input_model_pars)
   logger.info('epochs:             %s',args.epochs)
   logger.info('horovod:            %s',args.horovod)
   logger.info('num_threads:        %s',torch.get_num_threads())

   np.random.seed(args.random_seed)

   config_file = json.load(open(args.config_file))
   config_file['rank'] = rank
   config_file['nranks'] = nranks
   
   input_shape = config_file['data_handling']['image_shape'][1:3]
   input_channels = config_file['data_handling']['image_shape'][0]
   logger.info(' input_shape: %s',config_file['data_handling']['image_shape'])
   net = Net2D(input_shape,input_channels)
   summary_string,output_shape,output_channels = summary(input_shape,input_channels,net)
   logger.info('model: \n%s',summary_string)

   if rank == 0 and args.input_model_pars:
      logger.info('loading model pars from file %s',args.input_model_pars)
      net.load_state_dict(torch.load(args.input_model_pars))

   if args.horovod:
      logger.info('hvd broadcast')
      hvd.broadcast_parameters(net.state_dict(),root_rank=0)

   if args.batch > 0:
      config_file['training']['batch_size'] = args.batch

   logger.info('getting filelists')
   trainlist,validlist = get_filelist(config_file,args)

   evt_per_file = config_file['data_handling']['evt_per_file']
   batch_size   = config_file['training']['batch_size']
   img_shape    = config_file['data_handling']['image_shape']
   grid_shape   = net.grid
   num_classes  = len(config_file['data_handling']['classes'])
   logger.info('evt_per_file:       %s',evt_per_file)
   logger.info('batch_size:         %s',batch_size)
   logger.info('img_shape:          %s',img_shape)
   logger.info('grid_shape:         %s',grid_shape)
   logger.info('num_classes:        %s',num_classes)

   logger.info('creating batch generators')
   trainds = BatchGenerator(trainlist,evt_per_file,
                            batch_size,img_shape,grid_shape,
                            num_classes,True,rank,nranks)
   validds = BatchGenerator(validlist,evt_per_file,
                            batch_size,img_shape,grid_shape,
                            num_classes)
   # validds.start_file_pool(1)
   validds_itr = validds.batch_gen()
   net_loss = loss_func.ClassOnlyLoss()

   optimizer = optim.SGD(net.parameters(),lr=config_file['training']['learning_rate'], momentum=config_file['training']['momentum'])
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer,1,0.5)
   if args.horovod:
      optimizer = hvd.DistributedOptimizer(optimizer,named_parameters=net.named_parameters())

   accuracyCalc = loss_func.ClassOnlyAccuracy(net.grid,input_shape)

   data_time = CalcMean()
   batch_time = CalcMean()
   forward_time = CalcMean()
   backward_time = CalcMean()

   for epoch in range(args.epochs):
      logger.info(' epoch %s',epoch)
      scheduler.step()
      # trainds.start_file_pool()

      for param_group in optimizer.param_groups:
         logging.info('learning rate: %s',param_group['lr'])

      net.train()
      batch_counter = 0
      for batch_data in trainds.batch_gen():
         logger.debug('got training batch %s',batch_counter)
         start_batch = time.time()
         inputs = batch_data['images']
         targets = batch_data['truth']

         logger.debug('inputs: %s,%s targets: %s',inputs[0].shape,inputs[1].shape,targets.shape)

         start_forward = time.time()

         optimizer.zero_grad()
         logger.debug('zeroed opt')
         outputs = net(inputs)
         logger.debug('got outputs')

         grid_id_loss,class_loss = net_loss(outputs,targets)
         logger.debug('got loss')
         loss = grid_id_loss + class_loss

         start_backward = time.time()
         loss.backward()
         optimizer.step()

         end = time.time()

         data_time.add_value(start_forward - start_batch)
         forward_time.add_value(start_backward - start_forward)
         backward_time.add_value(end - start_backward)
         batch_time.add_value(end - start_batch)

         batch_counter += 1

         # print statistics
         if rank == 0:
            if batch_counter % args.status == 0:
               mean_img_per_second = (forward_time.calc_mean() + backward_time.calc_mean()) / batch_size
               
               logger.info('[%3d of %3d, %5d of %5d] loss: %6.4f + %6.4f = %6.4f   sec/image: %6.2f   data time: %6.3f  forward time: %6.3f  backward time: %6.3f',epoch + 1,args.epochs,batch_counter,len(trainds),grid_id_loss.item(),class_loss.item(),loss.item(),mean_img_per_second,data_time.calc_mean(),forward_time.calc_mean(),backward_time.calc_mean())

            if batch_counter % args.nval == 0:
               logger.info('running validation')
               net.eval()

               # if validds.reached_end:
               #    logger.warning('restarting validation file pool.')
               #    validds.start_file_pool(1)

               for _ in range(args.nval_tests):

                  try:
                     batch_data = next(validds_itr)
                  except StopIteration:
                     validds_itr = validds.batch_gen()
                     batch_data = next(validds_itr)
                  
                  inputs = batch_data['images']
                  targets = batch_data['truth']
                  logger.debug('valid inputs: %s,%s targets: %s',inputs[0].shape,inputs[1].shape,targets.shape)

                  outputs = net(inputs)
                  true_positive_accuracy,true_or_false_positive_accuracy,filled_grids_accuracy = accuracyCalc.eval_acc(outputs,targets,inputs)

                  grid_id_loss,class_loss = net_loss(outputs,targets)
                  loss = grid_id_loss + class_loss

                  logger.info('valid loss: %6.4f + %6.4f = %6.4f true_positive_accuracy: %6.4f true_or_false_positive_accuracy: %6.4f  filled_grids_accuracy: %6.4f',grid_id_loss.item(),class_loss.item(),loss.item(),true_positive_accuracy,true_or_false_positive_accuracy,filled_grids_accuracy)
                  
               net.train()

            if batch_counter % args.nsave == 0:
               torch.save(net.state_dict(),args.model_save + '_%05d_%05d.torch_model_state_dict' % (epoch,batch_counter))

      # logger.info('result ready: %s',trainds.results.ready())
      # logger.info('result: %s',trainds.results.get())

'''
   for epoch in range(args.epochs):
      logger.info(' epoch %s',epoch)
      scheduler.step()
      batch_counter = 0

      file_indices = np.array(range(len(trainlist)))
      np.random.shuffle(file_indices)

      for file_index in np.nditer(file_indices):

         for batch_index in range(trainds.batches_per_file):

            data = trainds.get_batch(file_index, batch_index)

            start = time.time()
            # logger.info('batch_counter: %s',batch_counter)
            inputs = data['images']
            targets = data['truth']

            # logger.info('retrieved data shape: %s',inputs.shape)
            # logger.info('retrieved truth shape: %s',targets.shape)

            optimizer.zero_grad()
            net.train()
            outputs = net(inputs)

            grid_id_loss,class_loss = net_loss(outputs,targets)
            loss = grid_id_loss + class_loss

            # forward = time.time()
            # logger.info('forward pass: %6.2f',forward - start)

            loss.backward()
            optimizer.step()

            end = time.time()
            # logger.info('backward pass: %6.2f',end - forward)

            timediff = end - start
            batch_time_sum += timediff
            batch_time_sum2 += timediff * timediff
            batch_time_n += 1

            # print statistics
            if rank == 0:
               if (batch_counter + 1) % args.status == 0:
                  mean_time = batch_time_sum / batch_time_n / batch_size
                  logger.info('[%3d of %3d, %5d of %5d] loss: %6.4f + %6.4f = %6.4f   sec/image: %6.2f',epoch + 1,args.epochs,batch_counter + 1,len(trainds),grid_id_loss.item(),class_loss.item(),loss.item(),mean_time)

               if (batch_counter + 1) % args.nval == 0:
                  net.eval()

                  for i in range(1):

                     data = validds.get_next_batch()
                     inputs = data['images']
                     targets = data['truth']
                     outputs = net(inputs)
                     acc = accuracyCalc.eval_acc(outputs,targets)

                     grid_id_loss,class_loss = net_loss(outputs,targets)
                     loss = grid_id_loss + class_loss
                     logger.info('valid loss: %6.4f + %6.4f = %6.4f accuracy: %10.3f',grid_id_loss.item(),class_loss.item(),loss.item(),acc)

               if (batch_counter + 1) % args.nsave == 0:
                  torch.save(net.state_dict(),args.model_save + '_%05d_%05d.torch_model_state_dict' % (epoch,batch_counter))

            batch_counter += 1
      
      if rank == 0:
         torch.save(net.state_dict(),args.model_save + '_%05d_%05d.torch_model_state_dict' % (epoch,batch_counter))
'''
            

def get_filelist(config_file,args):
   # get file list
   logger.info('train glob dir: %s',config_file['data_handling']['train_glob'])
   logger.info('valid glob dir: %s',config_file['data_handling']['valid_glob'])
   train_filelist = sorted(glob.glob(config_file['data_handling']['train_glob']))
   valid_filelist = sorted(glob.glob(config_file['data_handling']['valid_glob']))
   logger.info('found %s training files, %s validation files',len(train_filelist),len(valid_filelist))

   if len(train_filelist) < 1 or len(valid_filelist) < 1:
      raise Exception('length of file list needs to be at least 1 for train (%s) and val (%s) samples',len(train_filelist),len(valid_filelist))

   logger.warning('first file: %s',train_filelist[0])

   return train_filelist,valid_filelist


def print_module(module,input_shape,input_channels,name=None,indent=0):

   output_string = ''
   output_channels = input_channels
   output_shape = input_shape

   output_string += '%10s' % ('>' * indent)
   if name:
      output_string += ' %20s' % name
   else:
      output_string += ' %20s' % module.__class__.__name__

   # convolutions change channels
   if 'submanifoldconv' in module.__class__.__name__.lower():
      output_string += ' %4d -> %4d ' % (module.nIn,module.nOut)
      output_channels = module.nOut
   elif 'conv' in module.__class__.__name__.lower():
      output_string += ' %4d -> %4d ' % (module.in_channels,module.out_channels)
      output_channels = module.out_channels
   elif 'pool' in module.__class__.__name__.lower():
      output_shape = [int(input_shape[i] / module.pool_size[i]) for i in range(len(input_shape))]
      output_string += ' %10s -> %10s ' % (input_shape, output_shape)
   elif 'batchnormleakyrelu' in module.__class__.__name__.lower():
      output_string += ' (%10s) ' % module.nPlanes
   elif 'batchnorm2d' in module.__class__.__name__.lower():
      output_string += ' (%10s) ' % module.num_features

   output_string += '\n'

   for name, child in module.named_children():
      string,output_shape,output_channels = print_module(child, output_shape, output_channels, name, indent + 1)
      output_string += string

   return output_string,output_shape,output_channels


def summary(input_shape,input_channels,model):

   return print_module(model,input_shape,input_channels)


if __name__ == "__main__":
   main()
