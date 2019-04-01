
import torch,logging
import sparseconvnet as scn
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

      pred_grid_conf = output[:,0,...].float()
      true_grid_conf = target[:,0,...].float()

      # agreement of grid level object exits label
      # grid_id_loss = torch.sum( (true_grid_conf - pred_grid_conf) ** 2 )
      pos_weight = torch.Tensor(pred_grid_conf.size(1),pred_grid_conf.size(2)).fill_(pred_grid_conf.size(1)*pred_grid_conf.size(2))
      grid_id_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(pred_grid_conf,true_grid_conf)

      # convert prediction to log probabilities

      '''
      pred_class_conf = output[:,1:,...]

      true_class = target[:,1,...].long()

      class_loss = torch.nn.CrossEntropyLoss(reduction='none')(pred_class_conf,true_class)

      class_loss = class_loss * true_grid_conf
      class_loss = torch.sum(torch.sum(class_loss,1),1)
      class_loss = torch.mean(class_loss)
      '''

      # categorical agreement
      # pred_grid_class = torch.nn.Softmax(dim=1)(output[:,1:3,...])
      # pred_grid_class = torch.argmax(pred_grid_class,dim=1)
      # truth_grid_class = target[:,1,...]
      # class_loss = torch.sum( (pred_grid_class.float() - truth_grid_class.float()) ** 2 * true_grid_conf.float())

      # logger.info('class_loss: %10.6f  grid_id_loss: %10.6f',class_loss.item(),grid_id_loss.item())
      # loss = grid_id_loss + class_loss

      return grid_id_loss,torch.Tensor([0.])  # class_loss


class ClassOnlyAccuracy:
   def __init__(self,grid_size,input_shape):
      self.num_classes = 2
      self.cut = 0.5
      self.grid_size = grid_size
      self.input_shape = input_shape

      self.pool_size = [ int(self.input_shape[i] / self.grid_size[i]) for i in range(len(self.input_shape)) ]

   def eval_acc(self,output,target,inputs):

      nonzero_grids = self.get_occupied_grids(inputs)

      # get entries with prediction above some threshold
      # and where they are equal to truth
      # logger.info('acc output = %s %s %s %s, target = %s %s %s %s',output.shape,output.min(),output.mean(),output.max(),
                     # target.shape,target.min(),target.mean(),target.max())

      sigout = output.sigmoid()
      # logger.info('sigout = %s %s %s %s',sigout.shape,sigout.min(),sigout.mean(),sigout.max())

      cutout = torch.gt(sigout,self.cut).float()
      # logger.info('cutout = %s %s %s %s %s',cutout.shape,cutout.min(),cutout.mean(),cutout.max(),cutout.sum())

      pred_grid_conf = cutout[:,0,...]
      # logger.info('pred_grid_conf = %s %s %s %s %s',pred_grid_conf.shape,pred_grid_conf.min(),pred_grid_conf.mean(),pred_grid_conf.max(),pred_grid_conf.sum(dim=1).sum(dim=1))
      # logger.info('pred_grid_conf = %s',pred_grid_conf.shape)
      true_grid_conf = target[:,0,...].float()
      
      # logger.info('true_grid_conf = %s %s %s %s',
      #                               true_grid_conf.shape,
      #                               true_grid_conf.min(),
      #                               true_grid_conf.max(),
      #                               true_grid_conf.sum(dim=1).sum(dim=1))

      b = torch.eq(pred_grid_conf,true_grid_conf).float()
      # logger.info('b = %s %s %s %s',b.shape,b.min(),b.max(),b.sum())
      # logger.info('pred_grid_conf: %s  true_grid_conf: %s  b: %s',pred_grid_conf.sum(),true_grid_conf.sum(),b.sum())
      c = (b * true_grid_conf).sum(dim=1).sum(dim=1)
      # logger.info('c = %s',c.shape)
      d = true_grid_conf.sum(dim=1).sum(dim=1)
      # logger.info('d = %s',d.shape)
      true_positive_accuracy = c / d
      # logger.info('grid_acc = %s',grid_acc.shape)

      true_or_false_positive_accuracy = b.sum(dim=1).sum(dim=1) / b.size(1) / b.size(2)
      filled_grids_accuracy = (b * nonzero_grids).sum(dim=1).sum(dim=1) / nonzero_grids.sum(dim=1).sum(dim=1)

      return true_positive_accuracy.mean().item(),true_or_false_positive_accuracy.mean().item(),filled_grids_accuracy.mean().item()

   def get_occupied_grids(self,inputs):
      dense_inputs = scn.InputLayer(2,[256, 5760])(inputs)
      dense_inputs = scn.SparseToDense(2,2)(dense_inputs)
      dense_inputs = dense_inputs.sum(dim=1)
      gridified = torch.nn.MaxPool2d(self.pool_size)(dense_inputs)
      gridified = torch.gt(gridified,0.).float()
      # logger.info('gridified = %s %s',gridified.size(),gridified.sum())
      return gridified
