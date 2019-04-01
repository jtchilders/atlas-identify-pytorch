#!/usr/bin/env python
import argparse,logging,glob
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
logger = logging.getLogger(__name__)


def main():
   ''' simple starter program that can be copied for use when starting a new script. '''
   logging_format = '%(asctime)s %(levelname)s:%(name)s:%(message)s'
   logging_datefmt = '%Y-%m-%d %H:%M:%S'
   logging_level = logging.INFO
   
   parser = argparse.ArgumentParser(description='')
   parser.add_argument('-i','--input',dest='input',help='input',required=True)
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
   
   fl = glob.glob(args.input)

   logger.info('found %s files',len(fl))

   for f in fl:
      data = np.load(f,allow_pickle=True)
      images = data['raw']
      truths = data['truth']

      for img_num in range(len(images)):

         image = images[img_num]
         coords = image[1]
         features = image[0]
         truth = truths[img_num]

         dense_data = np.zeros((256,5760,2))

         for point_num in range(len(coords)):
            eta = coords[point_num][1]
            phi = coords[point_num][0]
            
            dense_data[phi,eta,0] = features[point_num,0]
            dense_data[phi,eta,1] = features[point_num,1]

         rects = []
         
         
         logger.info('truth = \n %s',np.int32(truth))
         fig,(ax1,ax2) = plt.subplots(2,figsize=(15,15),dpi=80)
         a = ax1.imshow(dense_data[...,0],aspect='auto',cmap='YlOrBr')
         ax1.set_title('em')
         plt.colorbar(a,ax=ax1)

         for particle in truth:
            if particle[0] == 1:
               start_x = int(particle[1] - particle[3]/2.)
               start_y = int(particle[2] - particle[4]/2.)
               ax1.add_patch(patches.Rectangle((start_x,start_y),particle[3],particle[4],linewidth=1,edgecolor='r',facecolor='none'))

         b = ax2.imshow(dense_data[...,1],aspect='auto',cmap='YlOrBr')
         ax2.set_title('had')
         plt.colorbar(b,ax=ax2)

         for particle in truth:
            if particle[0] == 1:
               start_x = int(particle[1] - particle[3]/2.)
               start_y = int(particle[2] - particle[4]/2.)
               ax2.add_patch(patches.Rectangle((start_x,start_y),particle[3],particle[4],linewidth=1,edgecolor='r',facecolor='none'))

         plt.show()


   


if __name__ == "__main__":
   main()
