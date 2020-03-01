import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
output_filters = {'0':[64],'1':[64,64,256],'2': [128,128,512],'3':[256,256,1024],'4': [512,512,2048]}
nb_identity_block = {'1': 2, '2': 3,'3': 5, '4': 2}

class ResNet50():
    def __init__(self):
        
        self.weights_list = []
        self.weights = dict()   
        self.weights['conv0_W'] = self.generate_Variable((7,7,3,64))
        self.weights['conv0_b'] = tf.Variable(tf.random.normal((64,),dtype = tf.float32),dtype = tf.float32)
        self.weights['conv0_offset'] = tf.Variable(tf.zeros((64,)),dtype = tf.float32)
        self.weights['conv0_scale']  = tf.Variable(tf.ones((64,)),dtype = tf.float32)
        self.weights_list.append(self.weights['conv0_W'])
        self.weights_list.append(self.weights['conv0_b'])
        self.weights_list.append(self.weights['conv0_offset'])
        self.weights_list.append(self.weights['conv0_scale'])
        
        
        for i in range(1,5):
            key = str(i)
            output_filters_sizes = output_filters.get(key)
            last_input_filter_size = output_filters.get(str( i - 1))[-1]
            self.weights[f"conv{i}_block0_W0"] = self.generate_Variable((1,1,last_input_filter_size,output_filters_sizes[-1]))
            self.weights[f"conv{i}_block0_b0"] = tf.Variable(tf.random.normal((output_filters_sizes[-1],),dtype = tf.float32),dtype = tf.float32)
            self.weights[f"conv{i}_block0_offset0"] = tf.Variable(tf.zeros((output_filters_sizes[-1],)),dtype= tf.float32)
            self.weights[f"conv{i}_block0_scale0"] = tf.Variable(tf.ones((output_filters_sizes[-1],)),dtype= tf.float32)
            
            self.weights[f"conv{i}_block0_W1"] = self.generate_Variable((1,1,last_input_filter_size,output_filters_sizes[0]))
            self.weights[f"conv{i}_block0_b1"] = tf.Variable(tf.random.normal((output_filters_sizes[0],),dtype = tf.float32),dtype = tf.float32)
            self.weights[f"conv{i}_block0_offset1"] = tf.Variable(tf.zeros((output_filters_sizes[0],)),dtype= tf.float32)
            self.weights[f"conv{i}_block0_scale1"] = tf.Variable(tf.ones((output_filters_sizes[0],)),dtype= tf.float32)

            self.weights[f"conv{i}_block0_W2"] = self.generate_Variable((3,3,output_filters_sizes[0],output_filters_sizes[1]))
            self.weights[f"conv{i}_block0_b2"] = tf.Variable(tf.random.normal((output_filters_sizes[1],),dtype = tf.float32),dtype = tf.float32)
            self.weights[f"conv{i}_block0_offset2"] = tf.Variable(tf.zeros((output_filters_sizes[1],)),dtype= tf.float32)
            self.weights[f"conv{i}_block0_scale2"] = tf.Variable(tf.ones((output_filters_sizes[1],)),dtype= tf.float32)

            self.weights[f"conv{i}_block0_W3"] = self.generate_Variable((1,1,output_filters_sizes[1],output_filters_sizes[2]))
            self.weights[f"conv{i}_block0_b3"] = tf.Variable(tf.random.normal((output_filters_sizes[2],),dtype = tf.float32),dtype = tf.float32)
            self.weights[f"conv{i}_block0_offset3"] = tf.Variable(tf.zeros((output_filters_sizes[2],)),dtype= tf.float32)
            self.weights[f"conv{i}_block0_scale3"] = tf.Variable(tf.ones((output_filters_sizes[2],)),dtype= tf.float32)
            
            nb_id_block = nb_identity_block.get(key)
            for j in range(1,nb_id_block + 1 ):
                self.weights[f"conv{i}_block{j}_W1"] = self.generate_Variable((1,1,output_filters_sizes[-1],output_filters_sizes[0]))
                self.weights[f"conv{i}_block{j}_b1"] = tf.Variable(tf.random.normal((output_filters_sizes[0],),dtype = tf.float32),dtype = tf.float32)
                self.weights[f"conv{i}_block{j}_offset1"] = tf.Variable(tf.zeros((output_filters_sizes[0],)),dtype= tf.float32)
                self.weights[f"conv{i}_block{j}_scale1"] = tf.Variable(tf.ones((output_filters_sizes[0],)),dtype= tf.float32)

                self.weights[f"conv{i}_block{j}_W2"] = self.generate_Variable((3,3,output_filters_sizes[0],output_filters_sizes[1]))
                self.weights[f"conv{i}_block{j}_b2"] = tf.Variable(tf.random.normal((output_filters_sizes[1],),dtype = tf.float32),dtype = tf.float32)
                self.weights[f"conv{i}_block{j}_offset2"] = tf.Variable(tf.zeros((output_filters_sizes[1],)),dtype= tf.float32)
                self.weights[f"conv{i}_block{j}_scale2"] = tf.Variable(tf.ones((output_filters_sizes[1],)),dtype= tf.float32)

                self.weights[f"conv{i}_block{j}_W3"] = self.generate_Variable((1,1,output_filters_sizes[1],output_filters_sizes[2]))
                self.weights[f"conv{i}_block{j}_b3"] = tf.Variable(tf.random.normal((output_filters_sizes[2],),dtype = tf.float32),dtype = tf.float32)
                self.weights[f"conv{i}_block{j}_offset3"] = tf.Variable(tf.zeros((output_filters_sizes[2],)),dtype= tf.float32)
                self.weights[f"conv{i}_block{j}_scale3"] = tf.Variable(tf.ones((output_filters_sizes[2],)),dtype= tf.float32)
                 
        self.weights['dense_W'] = self.generate_Variable((2048,1000))
        self.weights['dense_bias'] = tf.Variable(tf.random.normal((1000,),dtype= tf.float32),dtype = tf.float32)
        self.weights['dense_Wout'] = self.generate_Variable((1000,5))
        self.init_weight_list()
                
    def init_weight_list(self):
        for i in range(1,5):
            key = str(i)

            
            self.weights_list.append(self.weights[f"conv{i}_block0_W0"])
            self.weights_list.append(self.weights[f"conv{i}_block0_b0"])
            self.weights_list.append(self.weights[f"conv{i}_block0_offset0"])
            self.weights_list.append(self.weights[f"conv{i}_block0_scale0"])

            self.weights_list.append(self.weights[f"conv{i}_block0_W1"])
            self.weights_list.append(self.weights[f"conv{i}_block0_b1"])
            self.weights_list.append(self.weights[f"conv{i}_block0_offset1"])
            self.weights_list.append(self.weights[f"conv{i}_block0_scale1"])

            self.weights_list.append(self.weights[f"conv{i}_block0_W2"])
            self.weights_list.append(self.weights[f"conv{i}_block0_b2"])
            self.weights_list.append(self.weights[f"conv{i}_block0_offset2"])
            self.weights_list.append(self.weights[f"conv{i}_block0_scale2"])

            self.weights_list.append(self.weights[f"conv{i}_block0_W3"])
            self.weights_list.append(self.weights[f"conv{i}_block0_b3"])
            self.weights_list.append(self.weights[f"conv{i}_block0_offset3"])
            self.weights_list.append(self.weights[f"conv{i}_block0_scale3"])
                      
            nb_id_block = nb_identity_block.get(key)
            for j in range(1,nb_id_block + 1 ):
                self.weights_list.append(self.weights[f"conv{i}_block{j}_W1"])
                self.weights_list.append(self.weights[f"conv{i}_block{j}_b1"])
                self.weights_list.append(self.weights[f"conv{i}_block{j}_offset1"])
                self.weights_list.append(self.weights[f"conv{i}_block{j}_scale1"])
                
                self.weights_list.append(self.weights[f"conv{i}_block{j}_W2"])
                self.weights_list.append(self.weights[f"conv{i}_block{j}_b2"])
                self.weights_list.append(self.weights[f"conv{i}_block{j}_offset2"])
                self.weights_list.append(self.weights[f"conv{i}_block{j}_scale2"])

                self.weights_list.append(self.weights[f"conv{i}_block{j}_W3"])
                self.weights_list.append(self.weights[f"conv{i}_block{j}_b3"])
                self.weights_list.append(self.weights[f"conv{i}_block{j}_offset3"])
                self.weights_list.append(self.weights[f"conv{i}_block{j}_scale3"])
                
            
        self.weights_list.append(self.weights['dense_W'])
        self.weights_list.append(self.weights['dense_bias'])
        self.weights_list.append(self.weights['dense_Wout'])
                
            
    def forward(self,X):
        variance_epsilon = 1e-4
        
        conv0_W,conv0_b,conv0_offset,conv0_scale = self.weights['conv0_W'],self.weights['conv0_b'],self.weights['conv0_offset'],self.weights['conv0_scale']

        conv0_pad = tf.keras.layers.ZeroPadding2D((3,3))(X)
        
        conv0_conv = tf.nn.conv2d(conv0_pad,conv0_W,strides = [1,2,2,1],padding = 'VALID')
        conv0_conv = tf.nn.bias_add(conv0_conv,conv0_b)
        mean,variance = self.batch_norm(conv0_conv)
        conv0_bn = tf.nn.batch_normalization(conv0_conv,mean,variance,conv0_offset,conv0_scale,variance_epsilon)
        conv0_activation = tf.nn.relu(conv0_bn)

        pool0_pad = tf.keras.layers.ZeroPadding2D((1,1))(conv0_activation)

        max_pool0 = tf.nn.max_pool(pool0_pad,(3,3),strides = [1,2,2,1],padding = 'VALID')
        X_forward = max_pool0

        #Convolution blocks section
        for i in range(1,5):
            nb_id_block = nb_identity_block.get(str(i))
            X_forward = self.ConvBlock(X_forward,i)
            for j in range(1,nb_id_block + 1):
                X_forward = self.IdentityBlock(X_forward,i,j)

        #Global_Avg_Pooling

        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(X_forward) 

        #Dense Layer section 
        X_forward = tf.reshape(avg_pool,(X.shape[0],2048))
        X_forward = tf.matmul(avg_pool,self.weights['dense_W'])
        X_forward = tf.nn.bias_add(X_forward,self.weights['dense_bias'])
        X_forward = tf.nn.relu(X_forward)
        X_out = tf.matmul(X_forward,self.weights['dense_Wout'])
        X_out = tf.nn.softmax(X_out)
        return X_out

    def compile(self,optimizer = tf.optimizers.Adam(),loss = tf.losses.categorical_crossentropy,batch_size = 32,epochs = 20):
        self.optimizer = optimizer
        self.loss_function = loss
        self.batch_size = batch_size
        self.nb_epoch = epochs
    
    def fit(self,X,y):
        N = len(y)
        y = tf.keras.utils.to_categorical(y,5)
        dataset = tf.data.Dataset.from_tensor_slices((X,y))
        dataset = dataset.batch(self.batch_size)
        nb_batch =  N // self.batch_size
        print_time = int(nb_batch * 0.1 )
        print(f"print_time = {print_time}, nb_batch = {nb_batch}")
        for epoch in range(0,self.nb_epoch):
            accuracies = []
            losses = []
            tf.print(f"epoch {epoch} : ",end='')
            
            for i,(images,labels) in enumerate(dataset.take(nb_batch + 1)):
                with tf.GradientTape() as tape:
                    y_pred = self.forward(images)
                    loss = self.loss_fn(labels,y_pred)
                gradient = tape.gradient(loss,self.weights_list)
                self.optimizer.apply_gradients(zip(gradient,self.weights_list))
                accuracies.append(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred,1),tf.argmax(labels,1)),tf.float32)))
                losses.append(tf.reduce_mean(loss))
                if i % print_time == 0:
                    tf.print('.',end='')
            accuracy = tf.reduce_mean(accuracies)
            loss_mean = tf.reduce_mean(losses)
            print(f"\n epoch {epoch} ==> accuracy:{accuracy:.4f} loss:{loss_mean:.4f}")

    def loss_fn(self,y_true,y_pred):
        return self.loss_function(y_true,y_pred)

    def ConvBlock(self,X,stage_id):
        variance_epsilon = 1e-4
        key_W = "conv"+str(stage_id)+"_block0_W{}"

        key_b = "conv"+str(stage_id)+"_block0_b{}"
        key_offset = "conv"+str(stage_id)+"_block0_offset{}"
        key_scale = "conv"+str(stage_id)+"_block0_scale{}"
        
        conv_W0,conv_W1,conv_W2,conv_W3  = [self.weights[key_W.format(i)] for i in range(4)]
        conv_b0,conv_b1,conv_b2,conv_b3  = [self.weights[key_b.format(i)] for i in range(4)]

        offset0,offset1,offset2,offset3  = [self.weights[key_offset.format(i)] for i in range(4)]
        scale0,scale1,scale2,scale3  = [self.weights[key_scale.format(i)] for i in range(4)]
        
        conv1 = tf.nn.conv2d(X,conv_W1,strides = [1,2,2,1],padding ='VALID')
        conv1 = tf.nn.bias_add(conv1,conv_b1)
        mean,variance = self.batch_norm(conv1)
        conv1_bn = tf.nn.batch_normalization(conv1,mean=mean,variance = variance,offset = offset1 , scale = scale1,variance_epsilon = variance_epsilon)
        conv1_activation = tf.nn.relu(conv1_bn)
    
        conv2 = tf.nn.conv2d(conv1_activation,conv_W2,strides = [1,1,1,1],padding = 'SAME')
        conv2 = tf.nn.bias_add(conv2,conv_b2)
        mean,variance = self.batch_norm(conv2)
        conv2_bn = tf.nn.batch_normalization(conv2,mean = mean,variance = variance,offset = offset2,scale = scale2,variance_epsilon = variance_epsilon)
        conv2_activation = tf.nn.relu(conv2_bn)
        
        conv3 = tf.nn.conv2d(conv2_activation,conv_W3,strides = [1,1,1,1],padding = 'SAME')
        conv3 = tf.nn.bias_add(conv3,conv_b3)
        mean,variance = self.batch_norm(conv3)
        conv3_bn = tf.nn.batch_normalization(conv3,mean = mean,variance = variance,offset = offset3,scale = scale3,variance_epsilon = variance_epsilon)
        
        conv0  = tf.nn.conv2d(X,conv_W0,strides=[1,2,2,1],padding = 'VALID')
        conv0 = tf.nn.bias_add(conv0,conv_b0)
        mean,variance = self.batch_norm(conv0)
        conv0_bn = tf.nn.batch_normalization(conv0,mean,variance,offset = offset0,scale = scale0,variance_epsilon = variance_epsilon)
        
        conv_add = tf.add(conv3_bn,conv0_bn)
        conv_out = tf.nn.relu(conv_add)
        
        return conv_out  #,[conv_W0,conv_W1,conv_W2,conv_W3,conv_b0,conv_b1,conv_b2,conv_b3,offset0,offset1,offset2,offset3,scale0,scale1,scale2,scale3]
   
    def IdentityBlock(self,X,stage_id,block_id):
        
        variance_epsilon = 1e-4
        key_W = "conv"+str(stage_id)+"_block"+str(block_id)+"_W{}"

        key_b = "conv"+str(stage_id)+"_block"+str(block_id)+"_b{}"
        key_offset = "conv"+str(stage_id)+"_block"+str(block_id)+"_offset{}"
        key_scale = "conv"+str(stage_id)+"_block"+str(block_id)+"_scale{}"

        conv_W1,conv_W2,conv_W3 =   [self.weights[key_W.format(i)] for i in range(1,4)]
        conv_b1,conv_b2,conv_b3 =   [self.weights[key_b.format(i)] for i in range(1,4)]
        offset1,offset2,offset3 =   [self.weights[key_offset.format(i)] for i in range(1,4)]
        scale1,scale2,scale3    =   [self.weights[key_scale.format(i)] for i in range(1,4)]
        

        conv1 = tf.nn.conv2d(X,conv_W1,strides = [1,1,1,1],padding = 'SAME')
        conv1 = tf.nn.bias_add(conv1,conv_b1)
        mean,variance = self.batch_norm(conv1)
        conv1_bn = tf.nn.batch_normalization(conv1,mean,variance,offset1,scale1,variance_epsilon = variance_epsilon)
        conv1_activation = tf.nn.relu(conv1_bn)

        conv2 = tf.nn.conv2d(conv1_activation,conv_W2,strides = [1,1,1,1],padding = 'SAME')
        conv2 = tf.nn.bias_add(conv2,conv_b2)
        mean,variance = self.batch_norm(conv2)
        conv2_bn = tf.nn.batch_normalization(conv2,mean,variance,offset2,scale2,variance_epsilon = variance_epsilon)
        conv2_activation = tf.nn.relu(conv2_bn)

        conv3 = tf.nn.conv2d(conv2_activation,conv_W3,strides = [1,1,1,1],padding = 'SAME')
        conv3 = tf.nn.bias_add(conv3,conv_b3)
        mean,variance = self.batch_norm(conv3)
        conv3_bn = tf.nn.batch_normalization(conv3,mean,variance,offset3,scale3,variance_epsilon = variance_epsilon)

        conv_add = tf.add(conv3_bn,X)
        conv_out = tf.nn.relu(conv_add)
        return conv_out #,[conv_W1,conv_W2,conv_W3,conv_b1,conv_b2,conv_b3,offset1,offset2,offset3,scale1,scale2,scale3]
        
    def batch_norm(self,input_tensor):
        moments = tf.nn.moments(input_tensor,axes=[0, 1, 2])
        return moments    
    
    def glorot_uniform(self,shape):
        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
        fan_out = shape[1] if len(shape) == 2 else shape[0]
        return tf.random.uniform(shape,tf.cast(-tf.math.sqrt(6./(fan_in+fan_out)),tf.float32),tf.cast(tf.math.sqrt(6./(fan_in+fan_out)),tf.float32),dtype=tf.float32)
        
    def generate_Variable(self,shape):
        return tf.Variable(self.glorot_uniform(shape),dtype = tf.float32)

if __name__ == "__main__":
    
    X = tf.Variable(tf.random.uniform((500,224,224,3),dtype = tf.float32),dtype = tf.float32)
    y = np.random.randint(0,5,(500,),dtype = np.int64)
    model = ResNet50()
    model.compile(optimizer = tf.optimizers.Adam(),loss = tf.losses.categorical_crossentropy,batch_size= 32,epochs=50)
    model.fit(X,y)
    