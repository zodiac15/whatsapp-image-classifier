import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
from os import listdir
from os.path import isfile, join
import shutil

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
images = []
path = []
counter = 0
mypath = "F:\phone\whatsapp\WhatsApp Images"
generic_path = "F:\phone\whatsapp\WhatsApp Images\generic"
non_generic_path = "F:\phone\whatsapp\WhatsApp Images\\non-generic"

print("WARNING!! HIGH MEMORY CONSUMPTION!")

files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for f in files:
	try:
		img = Image.open(f)
		images.append(img)
		path.append(f)
	except IOError:
		pass


#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
for image in images:
	image = ImageOps.fit(image, size, Image.ANTIALIAS)

	#turn the image into a numpy array
	image_array = np.asarray(image)

	# display the resized image
	#image.show()

	# Normalize the image
	normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

	# Load the image into the array
	data[0] = normalized_image_array

	# run the inference
	prediction = model.predict(data).tolist()
	score = prediction[0][0]

	if score>0.8:
		shutil.move(join(mypath,path[counter]),join(generic_path,path[counter]))
	else:
		shutil.move(join(mypath,path[counter]),join(non_generic_path,path[counter]))
	print("processed "+ path[counter])
	counter+=1

	#print(prediction)
print(counter)