
import torch,logging
logger = logging.getLogger(__name__)


class ClassOnlyLoss(torch.nn.Module):

   def __init__(self):
      super(ClassOnlyLoss,self).__init__()
      self.num_classes = 2
      self.cut = 0.8

   def forward(self,output,target):

      # logger.info('output shape: %s target shape: %s',output.shape,target.shape)

      # output shape = (batch,4,grid_h,grid_w)
      # target shape = (batch,2,grid_h,grid_w)
      # flattened_output = torch.flatten(output,start_dim=1, end_dim=2)
      # flattened_target = torch.flatten(target,start_dim=1, end_dim=2)

      #cutout = torch.gt(output[...],self.cut).float()

      pred_grid_conf = output[:,0,...].float()
      true_grid_conf = target[:,0,...].float()

      # agreement of grid level object exits label
      # grid_id_loss = torch.sum( (true_grid_conf - pred_grid_conf) ** 2 )
      grid_id_loss = torch.nn.BCEWithLogitsLoss()(pred_grid_conf,true_grid_conf)

      # convert prediction to log probabilities

      pred_class_conf = torch.nn.functional.log_softmax(output[:,1:4,...],1)

      true_class = target[:,1,...].long()

      class_loss = torch.nn.CrossEntropyLoss(reduction='none')(pred_class_conf,true_class)

      class_loss = class_loss * true_grid_conf
      class_loss = torch.sum(torch.sum(class_loss,1),1)
      class_loss = torch.mean(class_loss)

      # categorical agreement
      # pred_grid_class = torch.nn.Softmax(dim=1)(output[:,1:3,...])
      # pred_grid_class = torch.argmax(pred_grid_class,dim=1)
      # truth_grid_class = target[:,1,...]
      # class_loss = torch.sum( (pred_grid_class.float() - truth_grid_class.float()) ** 2 * true_grid_conf.float())

      # logger.info('class_loss: %10.6f  grid_id_loss: %10.6f',class_loss.item(),grid_id_loss.item())
      # loss = grid_id_loss + class_loss

      return grid_id_loss,class_loss


class ClassOnlyAccuracy:
   def __init__(self):
      self.num_classes = 2
      self.cut = 0.8

   def eval_acc(self,output,target):

      # get entries with prediction above some threshold
      # and where they are equal to truth
      # logger.info('output = %s, target = %s',output.shape,target.shape)
      cutout = torch.gt(output.sigmoid(),self.cut).float()
      # logger.info('cut out = %s',cutout.shape)

      pred_grid_conf = cutout[:,0,...]
      # logger.info('pred_grid_conf = %s',pred_grid_conf.shape)
      true_grid_conf = target[:,0,...].float()
      # logger.info('true_grid_conf = %s %s %s %s',true_grid_conf.shape,true_grid_conf.min(),true_grid_conf.max(),true_grid_conf.sum())

      b = torch.eq(pred_grid_conf,true_grid_conf).float() * true_grid_conf
      # logger.info('b = %s %s %s %s',b.shape,b.min(),b.max(),b.sum())
      # logger.info('pred_grid_conf: %s  true_grid_conf: %s  b: %s',pred_grid_conf.sum(),true_grid_conf.sum(),b.sum())
      c = b.sum(dim=1).sum(dim=1)
      # logger.info('c = %s',c.shape)
      d = true_grid_conf.sum(dim=1).sum(dim=1)
      # logger.info('d = %s',d.shape)
      grid_acc = c / d
      # logger.info('grid_acc = %s',grid_acc.shape)

      return grid_acc
