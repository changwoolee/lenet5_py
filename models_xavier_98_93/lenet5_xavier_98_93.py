

####################################
# LeNet5 With Xavier Initialization#
####################################

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/temp/data/",one_hot=True)

train_epoch = 200
batch_size = 50
learning_rate = 0.0003

#define layers
def weight_variable(shape,n_inputs,n_outputs,_name):
	initial = xavier_init(n_inputs,n_outputs)
	return tf.get_variable(_name,shape=shape,initializer=initial)
def bias_variable(shape,_name):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial,name=_name)
def max_pool(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],
				strides=[1,2,2,1],padding='VALID')

def xavier_init(n_inputs, n_outputs, uniform=True):
	if uniform:
		# 6 was used in the paper.
		init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
		return tf.random_uniform_initializer(-init_range, init_range)
	else:
		# 3 gives us approximately the same limits as above since this repicks
		# values greater than 2 standard deviations from the mean.
		stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
		return tf.truncated_normal_initializer(stddev=stddev)



#input layer
X = tf.placeholder(tf.float32, shape=[None,784])
#output tag
y_ = tf.placeholder(tf.float32, shape=[None,10])
sess = tf.InteractiveSession()
#1st Layer(C1)
x_image = tf.reshape(X,[-1,28,28,1])
Wconv1 = weight_variable([5,5,1,6],25,6*25,"Wconv1")
bconv1 = bias_variable([6],"bconv1")
hconv1 = tf.nn.relu(tf.nn.conv2d(x_image,Wconv1,strides=[1,1,1,1],
				padding='SAME')+bconv1)
#2nd Layer(S2)
hpool2 = max_pool(hconv1)

#3rd Layer(C3)
Wconv3 = weight_variable([5,5,6,16],25*6,25*16,"Wconv3")
bconv3 = bias_variable([16],"bconv3")
hconv3 = tf.nn.relu(tf.nn.conv2d(hpool2,Wconv3,strides=[1,1,1,1],
				padding='VALID')+bconv3)

#4th Layer(S4)
hpool4 = max_pool(hconv3)

#5th Laeyr(C5)
Wconv5 = weight_variable([5,5,16,120],25*16,25*120,"Wconv5")
bconv5 = bias_variable([120],"bconv5")
hconv5 = tf.nn.relu(tf.nn.conv2d(hpool4,Wconv5,strides=[1,1,1,1],
				padding='VALID')+bconv5)

#FC Layer
fc_in = tf.reshape(hconv5,[-1,1*1*120])
Wfc1 = weight_variable([120,84],120,84,"Wfc1")
bfc1 = bias_variable([84],"bfc1")
hfc1 = tf.nn.relu(tf.matmul(fc_in,Wfc1)+bfc1)

Wfc2 = weight_variable([84,10],84,10,"Wfc2")
bfc2 = bias_variable([10],"bfc2")
y = tf.matmul(hfc1,Wfc2)+bfc2

#Weights saver
saver = tf.train.Saver()

#training
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,
				logits=y))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess.run(tf.global_variables_initializer())
total_batch = int(mnist.train.num_examples/batch_size)

past_valid_cost=0
valid_cost_rise=0
valid_cost=0
for epoch in range(train_epoch):
	avg_cost = 0.
	for i in range(total_batch):
		batch = mnist.train.next_batch(batch_size)
		_,c=sess.run([train_step,cost],feed_dict={X:batch[0],y_:batch[1]})
		avg_cost += c/total_batch
	print("Epoch : %04d cost=%.9f"%(epoch+1,avg_cost))
	past_valid_cost=valid_cost
	valid_cost = cost.eval(feed_dict={X:mnist.validation.images,y_:mnist.validation.labels})
	print("Cross-Validation : cost= %.9f"%valid_cost)
	if valid_cost > past_valid_cost:
		if valid_cost_rise==2:
			break
		else:
			valid_cost_rise+=1
#	else:
#		valid_cost_rise=0	


print("Test completed")
print("test accuracy %g"%accuracy.eval(feed_dict={X:mnist.test.images, y_:mnist.test.labels}))
save_path = saver.save(sess, "./models_xavier/models.ckpt")
print("Model saved in file : %s"% save_path)





