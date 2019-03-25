#!/usr/bin/env python
import argparse,logging,json
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
   parser.add_argument('-o','--output',dest='output',help='output json file',required=True)
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
   
   batch_vs_loss = []
   batch_vs_grid_loss = []
   batch_vs_class_loss = []

   valid_batch_vs_loss = []
   valid_batch_vs_grid_loss = []
   valid_batch_vs_class_loss = []
   valid_batch_vs_accuracy = []

   with open(args.input) as f:

      training_data = []
      valid_data = []
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
            data['accuracy'] = float(parts[index])

            valid_batch_vs_loss.append([data['step'],data['loss']])
            valid_batch_vs_grid_loss.append([data['step'],data['grid_loss']])
            valid_batch_vs_class_loss.append([data['step'],data['class_loss']])
            if data['accuracy'] > 0.:
               valid_batch_vs_accuracy.append([data['step'],data['accuracy']])

            valid_data.append(data)

   output = {'training':training_data,'valid':valid_data}
   json.dump(output,open(args.output,'w'),indent=4, sort_keys=True)

   x = np.array(batch_vs_loss)
   y = np.array(batch_vs_grid_loss)
   z = np.array(batch_vs_class_loss)

   a = np.array(valid_batch_vs_loss)
   b = np.array(valid_batch_vs_grid_loss)
   c = np.array(valid_batch_vs_class_loss)

   fig,(ax1,ax2) = plt.subplots(2)

   ax1.plot(x[...,0],x[...,1],label='loss')
   ax1.plot(y[...,0],y[...,1],label='grid loss')
   ax1.plot(z[...,0],z[...,1],label='class loss')
   ax1.legend(loc='upper center', shadow=False, fontsize='x-large')
   ax1.grid(axis='y')
   ax1.set_yscale('log')

   ax2.plot(a[...,0],a[...,1],label='loss')
   ax2.plot(b[...,0],b[...,1],label='grid loss')
   ax2.plot(c[...,0],c[...,1],label='class loss')

   if len(valid_batch_vs_accuracy) > 0:
      d = np.array(valid_batch_vs_accuracy)
      ax2.plot(d[...,0],d[...,1],label='accuracy')
   ax2.legend(loc='upper center', shadow=False, fontsize='x-large')
   ax2.grid(axis='y')
   ax2.set_yscale('log')

   plt.show()


if __name__ == "__main__":
   main()
