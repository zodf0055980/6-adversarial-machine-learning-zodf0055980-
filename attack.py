import sys
import numpy as np
from keras.applications import resnet
from keras.preprocessing import image
from keras.models import load_model
from keras.activations import relu, softmax
import keras.backend as K
import matplotlib.pyplot as plt

model = load_model('model.h5')
cls_list = ['cat', 'dog']
img_path = sys.argv[1]
img = image.load_img(img_path, target_size=(224,224))

# Create image array
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# initial prediction to generate target
preds = model.predict(x)[0]
top_inds = preds.argsort()[::-1][:5]
if(top_inds[0] > top_inds[1]) :
	target_class = 0
else:
	target_class = 1
print("attack to : %s" % cls_list[target_class])

# Get current session (assuming tf backend)
sess = K.get_session()

x_adv = x
x_noise = np.zeros_like(x)

# Set variables , can change epsilon
epochs = 400
epsilon = 0.18

for i in range(epochs): 
	# One hot encode the target class
	target = K.one_hot(target_class, 2)

	# Get the loss and gradient of the loss wrt the inputs
	loss = -1*K.categorical_crossentropy(target, model.output)
	grads = K.gradients(loss, model.input)

	# Get the sign of the gradient
	delta = K.sign(grads[0])
	x_noise = x_noise + delta

	# Perturb the image
	x_adv = x_adv + epsilon*delta

	# Get the new image and predictions
	x_adv = sess.run(x_adv, feed_dict={model.input:x})
	preds = model.predict(x_adv)
	adv_pred = np.asscalar(preds[0][target_class])
	print(i, preds[0][target_class])
	# if pred > 0.9 is success
	if adv_pred > 0.95 :
		image.save_img( img_path + "adv.png", x_adv[0])
		noise = x_adv - x
		image.save_img( img_path + "noise.png", noise[0])
		exit(0)