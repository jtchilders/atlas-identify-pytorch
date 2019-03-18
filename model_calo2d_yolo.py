import torch.nn as nn
import torch
import logging
import loss_func
logger = logging.getLogger(__name__)

class Net(nn.Module):
   def __init__(self,blocks):
      super(Net,self).__init__()

      self.blocks = blocks
      self.models = self.create_network(self.blocks) # merge conv, bn,leaky
      self.loss   = None


   def forward(self,x):
      ind = -2
      self.loss = None
      outputs = dict()
      for block in self.blocks:
         ind = ind + 1
         #if ind > 0:
         #    return x

         if block['type'] == 'net':
            continue
         elif ( block['type'] == 'convolutional' or block['type'] == 'maxpool' or 
                block['type'] == 'reorg' or block['type'] == 'avgpool' or 
               block['type'] == 'softmax' or block['type'] == 'connected'):
            x = self.models[ind](x)
            outputs[ind] = x
         elif block['type'] == 'route':
            layers = block['layers'].split(',')
            layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
            if len(layers) == 1:
               x = outputs[layers[0]]
               outputs[ind] = x
            elif len(layers) == 2:
               x1 = outputs[layers[0]]
               x2 = outputs[layers[1]]
               x = torch.cat((x1,x2),1)
               outputs[ind] = x
         elif block['type'] == 'shortcut':
               from_layer = int(block['from'])
               activation = block['activation']
               from_layer = from_layer if from_layer > 0 else from_layer + ind
               x1 = outputs[from_layer]
               x2 = outputs[ind-1]
               x  = x1 + x2
               if activation == 'leaky':
                  x = F.leaky_relu(x, 0.1, inplace=True)
               elif activation == 'relu':
                  x = F.relu(x, inplace=True)
               outputs[ind] = x
         elif block['type'] == 'region' or block['type'] == 'classonly':
            self.loss = self.models[ind]
            continue # can't do this here since need target
            if self.loss:
               self.loss = self.loss + self.models[ind](x)
            else:
               self.loss = self.models[ind](x)
            outputs[ind] = None
         elif block['type'] == 'cost':
            continue
         else:
            print('unknown type %s' % (block['type']))
      return x

   def create_network(self, blocks):
        models = nn.ModuleList()
    
        prev_filters = 3
        out_filters =[]
        conv_id = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = int((kernel_size-1)/2) if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                    #model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model = nn.MaxPool2d(pool_size, stride)
                else:
                    model = MaxPoolStride1()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(size_average=True)
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(size_average=True)
                out_filters.append(1)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                models.append(Reorg(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                elif len(layers) == 2:
                    assert(layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind-1]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'classonly':
                loss = loss_func.ClassOnlyLoss()
                loss.num_classes = int(block['classes'])
                out_filters.append(prev_filters)
                models.append(loss)
            else:
                print('unknown type %s' % (block['type']))
    
        return models



class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = int(stride)
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        hs_ = int(H/hs)
        ws_ = int(W/ws)
        x = x.view(B, C, hs_, hs, ws_, ws).transpose(3,4).contiguous()
        x = x.view(B, C, hs_*ws_, hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, hs_, ws_).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, hs_, ws_)
        return x

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x

# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x



