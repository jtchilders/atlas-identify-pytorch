#!/usr/bin/env python
import argparse,logging,json,glob
import matplotlib.pyplot as plt
import numpy as np
logger = logging.getLogger(__name__)


def main():
   ''' simple starter program that can be copied for use when starting a new script. '''
   logging_format = '%(asctime)s %(levelname)s:%(name)s:%(message)s'
   logging_datefmt = '%Y-%m-%d %H:%M:%S'
   logging_level = logging.INFO
   
   parser = argparse.ArgumentParser(description='plot the loss function for a few runs')
   parser.add_argument('-g','--glob',dest='glob',help='glob to pick up json data files.',required=True)
   parser.add_argument('-p','--outputfig',dest='outputfig',help='output figure name.')

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

   logger.info(' glob:                 %s',args.glob)
   logger.info('output image file:     %s',args.outputfig)

   filelist = glob.glob(args.glob)

   logger.info('found %s files',len(filelist))

   rundata = []

   fig,(ax1,ax2) = plt.subplots(2,figsize=(15,15),dpi=80)

   for file in filelist:
      rundata.append(json.load(open(file)))
      # id = int(file.split('/')[-1].split('.')[0])

      batch_size = rundata[-1]['batch_size']
      nranks = rundata[-1]['nranks']
      train_loss = np.array(rundata[-1]['train_loss'])
      valid_loss = np.array(rundata[-1]['valid_loss'])

      imgs_per_step = batch_size * nranks

      ax1.plot(train_loss[...,0] * imgs_per_step,train_loss[...,1],label='bs %s ranks %s' % (batch_size,nranks))
      ax2.plot(valid_loss[...,0] * imgs_per_step,valid_loss[...,1],label='bs %s ranks %s' % (batch_size,nranks))


   ax1.legend(loc='upper center', shadow=False, fontsize='x-large')
   ax1.grid(axis='y')

   ax2.legend(loc='upper center', shadow=False, fontsize='x-large')
   ax2.grid(axis='y')


   plt.savefig(args.outputfig)



if __name__ == "__main__":
   main()
