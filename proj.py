
# coding: utf-8

# In[ ]:


import numpy as np
import mxnet as mx
import time
import pandas as pd

import cv2

import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout


# In[ ]:


#import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')


# In[ ]:


data0 = pd.read_csv('fdata/fdata.csv', names=['name','state'])


# In[ ]:


#data0.head()


# In[ ]:


#data0['state'].unique()


# In[ ]:


num_class = len(data0['state'].unique())
ges_to_num = dict({(g,i) for i, g in enumerate(data0['state'].unique())})
num_to_ges = dict({(i,g) for i, g in enumerate(data0['state'].unique())})
#num_class, ges_to_num


# In[ ]:


data0 = data0.replace({'state':ges_to_num})


# In[ ]:


#data0.shape[0]


# In[ ]:


labels = np.empty((data0.shape[0]))

res_width, res_height = 200, 200
imgs = np.empty(shape=(data0.shape[0],1,res_width,res_height))
#imgs.shape, labels.shape


# In[ ]:


prefix = 'fdata/pic/'
outfix = 'fdata/bi_pic/'
for i, (im_name, state) in enumerate(data0.values):
    im_path = prefix + im_name
    print im_path
    img = cv2.imread(im_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.resize(gray,(200, 200), interpolation=cv2.INTER_CUBIC)

    imgs[i][0] = res
    labels[i] = state


# In[ ]:


train_data, train_label = imgs, labels
# test_data, test_label = imgs[23:], labels[2:]
train_data.shape, train_label.shape#, test_data.shape, test_label.shape


# In[ ]:


batch_size = 10
train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size, shuffle=True)
# eval_iter = mx.io.NDArrayIter(test_data, test_label, batch_size)


# In[ ]:


data = mx.sym.var('data')

conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20, name='conv1')
bn1 = mx.sym.BatchNorm(conv1, fix_gamma=True)
tanh1 = mx.sym.Activation(data=bn1, act_type='tanh')
pool1 = mx.sym.Pooling(data=tanh1, pool_type='max', kernel=(2,2), stride=(2,2))

conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50, name='conv2')
bn2 = mx.sym.BatchNorm(conv2, fix_gamma=True)
tanh2 = mx.sym.Activation(data=bn2, act_type='tanh')
pool2 = mx.sym.Pooling(data=tanh2, pool_type='max', kernel=(2,2), stride=(2,2))

flat = mx.sym.flatten(data=pool2)
fc1 = mx.sym.FullyConnected(data=flat, num_hidden=500)
tanh3 = mx.sym.Activation(data=fc1, act_type='tanh')

fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=num_class)

convnet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

mx.viz.plot_network(convnet)


# In[ ]:


model = mx.mod.Module(symbol=convnet, context=mx.gpu())


# In[ ]:


model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
model.init_params(initializer=mx.init.Uniform(scale=.1))
model.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate':0.1})

metric = mx.metric.Accuracy()


# In[ ]:


chk_prefix='models/chkpt'
for epoch in range(1200):
    train_iter.reset()
    metric.reset()
    
    st = time.time()
    for batch in train_iter:
        model.forward(data_batch=batch, is_train=True)
        model.update_metric(metric, batch.label)
        model.backward()
        model.update()
    
    if epoch % 50 == 0:
#         model_path = '{}_{}'.format(chk_prefix, epoch)
        model.save_checkpoint(chk_prefix, epoch)
        
    et = time.time()-st
    print('Epoch %d, Training %s, Time %.2f' % (epoch, metric.get(), et))





# model.score(train_iter, metric)
