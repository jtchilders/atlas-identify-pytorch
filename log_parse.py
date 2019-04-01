#!/usr/bin/env python
import argparse,logging,json,time
import matplotlib.pyplot as plt
import numpy as np
logger = logging.getLogger(__name__)


def main():
   ''' simple starter program that can be copied for use when starting a new script. '''
   logging_format = '%(asctime)s %(levelname)s:%(name)s:%(message)s'
   logging_datefmt = '%Y-%m-%d %H:%M:%S'
   logging_level = logging.INFO
   
   parser = argparse.ArgumentParser(description='parse log file for loss and timing information throughout run')
   parser.add_argument('-i','--input',dest='input',help='log file produced by atlas-identify-pytorch-sparse.',required=True)
   parser.add_argument('-o','--output',dest='output',help='output json file')
   parser.add_argument('-p','--outputfig',dest='outputfig',help='output figure name.')
   parser.add_argument('-t','--sleep',dest='sleep',help='time between parsing.',default=10,type=int)
   parser.add_argument('-r','--repeat',dest='repeat',help='number of times to repeat',default=1000,type=int)

   parser.add_argument('--debug', dest='debug', default=False, action='store_true', help="Set Logger to DEBUG")
   parser.add_argument('--error', dest='error', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--warning', dest='warning', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--logfilename',dest='logfilename',default=None,help='if set, logging information will go to file')
   args = parser.parse_args()

   if args.debug and not args.error and not args.warning:
      logging_level = logging.DEBUG
   elif not args.debug and args.error and not args.warning:
      logging_level = logging.ERROR
   elif not args.debug and not args.error and args.warning:
      logging_level = logging.WARNING

   logging.basicConfig(level=logging_level,
                       format=logging_format,
                       datefmt=logging_datefmt,
                       filename=args.logfilename)

   if args.output is None:
      args.output = args.input + '.json'
   if args.outputfig is None:
      args.outputfig = args.outputfig + '.png'

   logger.info('parsing log file:   %s',args.input)
   logger.info('output json file:   %s',args.output)
   logger.info('output image file:  %s',args.outputfig)
   logger.info('sleep seconds:      %s',args.sleep)

   for _ in range(args.repeat):
      logger.info('parsing data')
      data = parse_file(args.input)

      json.dump(data,open(args.output,'w'),indent=4, sort_keys=True)

      plot_data(data,args.outputfig)

      logger.info('sleeping: %s',args.sleep)
      time.sleep(args.sleep)


def parse_file(filename):

   batch_vs_loss = []
   batch_vs_grid_loss = []
   batch_vs_class_loss = []

   valid_batch_vs_loss = []
   valid_batch_vs_grid_loss = []
   valid_batch_vs_class_loss = []

   true_positive_accuracy = []
   true_or_false_positive_accuracy = []
   filled_grids_accuracy = []

   with open(filename) as f:

      training_data = []
      valid_data = []
      batch_size = None
      nranks = 0
      for line in f:

         if '] loss:' in line:

            parts = line.split()
            data = {}
            index = 3
            data['epoch'] = int(parts[index])
            index += 2
            data['nepochs'] = int(parts[index][:-1])
            index += 1
            data['batch'] = int(parts[index])
            index += 2
            data['nbatches'] = int(parts[index][:-1])
            index += 2
            data['grid_loss'] = float(parts[index])
            index += 2
            data['class_loss'] = float(parts[index])
            index += 2
            data['loss'] = float(parts[index])
            index += 2
            data['sec_imgs'] = float(parts[index])
            index += 3
            data['data_time'] = float(parts[index])
            index += 3
            data['fwd_time'] = float(parts[index])
            index += 3
            data['bwd_time'] = float(parts[index])
            data['step'] = (data['epoch'] - 1) * data['nbatches'] + (data['batch'] - 1)

            batch_vs_loss.append([data['step'],data['loss']])
            batch_vs_grid_loss.append([data['step'],data['grid_loss']])
            batch_vs_class_loss.append([data['step'],data['class_loss']])

            training_data.append(data)
         elif ':valid loss:' in line:
            parts = line.split()
            data = {}
            data['epoch'] = training_data[-1]['epoch']
            data['nepochs'] = training_data[-1]['nepochs']
            data['batch'] = training_data[-1]['batch']
            data['nbatches'] = training_data[-1]['nbatches']
            data['step'] = training_data[-1]['step']

            index = 4
            data['grid_loss'] = float(parts[index])
            index += 2
            data['class_loss'] = float(parts[index])
            index += 2
            data['loss'] = float(parts[index])
            index += 2
            data['true_positive_accuracy'] = float(parts[index])
            index += 2
            data['true_or_false_positive_accuracy'] = float(parts[index])
            index += 2
            data['filled_grids_accuracy'] = float(parts[index])

            valid_batch_vs_loss.append([data['step'],data['loss']])
            valid_batch_vs_grid_loss.append([data['step'],data['grid_loss']])
            valid_batch_vs_class_loss.append([data['step'],data['class_loss']])
            if data['true_positive_accuracy'] > 0.:
               true_positive_accuracy.append([data['step'],data['true_positive_accuracy']])

            if data['true_or_false_positive_accuracy'] > 0.:
               true_or_false_positive_accuracy.append([data['step'],data['true_or_false_positive_accuracy']])

            if data['filled_grids_accuracy'] > 0.:
               filled_grids_accuracy.append([data['step'],data['filled_grids_accuracy']])

            valid_data.append(data)
         elif ':batch_size:' in line:
            parts = line.split()
            batch_size = int(parts[-1])
         elif ':rank ' in line and nranks == 0:
            parts = line.split()
            nranks = int(parts[-1])

   output = {'training':training_data,
             'valid':valid_data,
             'train_loss':batch_vs_loss,
             'train_grid_loss':batch_vs_grid_loss,
             'train_class_loss':batch_vs_class_loss,
             'valid_loss':valid_batch_vs_loss,
             'valid_grid_loss':valid_batch_vs_grid_loss,
             'valid_class_loss':valid_batch_vs_class_loss,
             'true_positive_accuracy':true_positive_accuracy,
             'true_or_false_positive_accuracy':true_or_false_positive_accuracy,
             'filled_grids_accuracy':filled_grids_accuracy,
             'batch_size': batch_size,
             'nranks': nranks,
            }

   logger.info('entries: %s',len(valid_batch_vs_loss))
   
   return output


def plot_data(data,outputfig):

   train_loss = np.array(data['train_loss'])
   train_grid_loss = np.array(data['train_grid_loss'])
   train_class_loss = np.array(data['train_class_loss'])

   valid_loss = np.array(data['valid_loss'])
   valid_grid_loss = np.array(data['valid_grid_loss'])
   valid_class_loss = np.array(data['valid_class_loss'])

   fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(15,15),dpi=80)

   ax1.plot(train_loss[...,0],train_loss[...,1],label='train loss')
   ax1.plot(valid_loss[...,0],valid_loss[...,1],label='valid loss')
   ax1.legend(loc='upper center', shadow=False, fontsize='x-large')
   ax1.grid(axis='y')
   # ax1.set_ylim([0,2])
   # ax1.set_yscale('log')

   ax2.plot(train_grid_loss[...,0],train_grid_loss[...,1],label='train grid loss')
   ax2.plot(valid_grid_loss[...,0],valid_grid_loss[...,1],label='valid grid loss')
   ax2.legend(loc='upper center', shadow=False, fontsize='x-large')
   ax2.grid(axis='y')
   # ax2.set_ylim([0,2])
   # ax2.set_yscale('log')

   ax3.plot(train_class_loss[...,0],train_class_loss[...,1],label='train class loss')
   ax3.plot(valid_class_loss[...,0],valid_class_loss[...,1],label='valid class loss')
   ax3.legend(loc='upper center', shadow=False, fontsize='x-large')
   ax3.grid(axis='y')
   # ax3.set_ylim([0,2])

   true_positive_accuracy = np.array(data['true_positive_accuracy'])
   true_or_false_positive_accuracy = np.array(data['true_or_false_positive_accuracy'])
   filled_grids_accuracy = np.array(data['filled_grids_accuracy'])
   
   ax4.plot(true_positive_accuracy[...,0],true_positive_accuracy[...,1],label='true pos')
   ax4.plot(true_or_false_positive_accuracy[...,0],true_or_false_positive_accuracy[...,1],label='true-false pos')
   ax4.plot(filled_grids_accuracy[...,0],filled_grids_accuracy[...,1],label='filled grid')
   ax4.legend(loc='center', shadow=False, fontsize='x-large')
   ax4.grid(axis='y')

   # logger.info('\n %s \n %s',dir(ax4),dir(fig))
   # logger.info('\n %s \n %s',ax4.lines,dir(ax4.lines[0]))

   plt.savefig(outputfig)


if __name__ == "__main__":
   main()
