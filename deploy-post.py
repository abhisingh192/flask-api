import os
import os,io
from flask import Flask, render_template, request,jsonify
from flask import Flask,request,render_template
from flask_restful import Resource,Api
import json

import pandas as pd
import glob 
import cv2
import sys


import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN


from keras.models import Sequential,model_from_json
from keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D,Dropout,Flatten,Activation
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
import numpy as np


li = []

    

app=Flask(__name__)
api=Api(app)
#@app.route("/")


class Index(Resource):      
    def get(self):
        return{
            "about":"Face Verifier"
            }
    


#APP_ROOT = os.path.dirname(os.path.abspath(__file__))


#@app.route("/upload", methods=['POST'])
class Upload(Resource):
    def post(self):
            
            K.clear_session()
            request_body = json.loads(request.data)
            FOLDER_PATH= str(request_body["FOLDER_PATH"])
            FILE_NAME1=str(request_body["FILE_NAME1"])
            FILE_NAME2=str(request_body["FILE_NAME2"])
            #target0 = os.path.join(APP_ROOT, 'images/')
            #if not os.path.isdir(target0):
                    #os.mkdir(target0)
            '''
            for file in request.files.getlist("file"):
                    filename = file.filename
                    img=os.path.join(target0,filename)
                    #print('helol')
                    destination = "/".join([target0, filename])
                    #print("first file",destination)
                
                    li.append(destination)
                    file.save(destination)

                    # load image from file'''
            
            #print(li[0])
            FILE_PATH1=os.path.join(FOLDER_PATH,FILE_NAME1)
            FILE_PATH2=os.path.join(FOLDER_PATH,FILE_NAME2)

            
            pixels0 = pyplot.imread(FILE_PATH1)
            # create the detector, using default weights
            detector0 = MTCNN()
            # detect faces in the image
            faces0 = detector0.detect_faces(pixels0)
            # display faces on the original image
            draw_image_with_boxes0(FILE_PATH1, faces0)
            #cv2.imwrite('bounded.jpg',)


            # load image from file
            pixels1 = pyplot.imread(FILE_PATH2)
            # create the detector, using default weights
            detector1 = MTCNN()
            # detect faces in the image
            faces1 = detector1.detect_faces(pixels1)
            # display faces on the original image
            draw_image_with_boxes1(FILE_PATH2, faces1)
            #cv2.imwrite('bounded.jpg',)



            print("loading vgg face")
            model = Sequential()
            model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
            model.add(Convolution2D(64, (3, 3), activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2,2), strides=(2,2)))
             
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(128, (3, 3), activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2,2), strides=(2,2)))
             
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(256, (3, 3), activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(256, (3, 3), activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(256, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2,2), strides=(2,2)))
             
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(512, (3, 3), activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(512, (3, 3), activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(512, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2,2), strides=(2,2)))
             
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(512, (3, 3), activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(512, (3, 3), activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(512, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2,2), strides=(2,2)))
             
            model.add(Convolution2D(4096, (7, 7), activation='relu'))
            model.add(Dropout(0.5))
            model.add(Convolution2D(4096, (1, 1), activation='relu'))
            model.add(Dropout(0.5))
            model.add(Convolution2D(2622, (1, 1)))
            model.add(Flatten())
            model.add(Activation('softmax'))

            model.load_weights('vgg_face_weights.h5')

            model.pop()


            epsilon = 0.45 #threshold, if less than threshold images are the same


            #verifyFace('cropped0.jpg','cropped1.jpg')
            img1 = 'cropped0.jpg'
            img2 = 'cropped1.jpg'

            img1_representation = model.predict(preprocess_image(img1))[0,:]
            img2_representation = model.predict(preprocess_image(img2))[0,:]

            cosine_similarity = findCosineDistance(img1_representation, img2_representation)
            
            flag = 0
            if(cosine_similarity < epsilon):
                #print("Verified! Same person")
                flag = 1
            #else:
                #print("Sorry, not the same person!")
            if flag:
                    status = "Verified! Same person"
            else:
                    status = "Sorry, not the same person!"

            #print("hello",Upload.status)

            K.clear_session()
            return {"status": status}

class Status(Resource):
    def get(self):
            def __init__(self):
                obj1 = Upload()
                obj2 = Status()
                status: Upload.status
                
            return{
                "status":Upload.status
                        }
    
    
                

api.add_resource(Index,'/')
api.add_resource(Status,'/status')
api.add_resource(Upload,'/upload')

def verifyFace(img1, img2):
    print("veifying the images", img1, img2)
    img1_representation = model.predict(preprocess_image(img1))[0,:]
    img2_representation = model.predict(preprocess_image(img2))[0,:]
 
    cosine_similarity = findCosineDistance(img1_representation, img2_representation)
    print("the cosine distance between the two images is", cosine_similarity)
    # euclidean_distance = findEuclideanDistance(img1_representation, img2_representation) 
    if(cosine_similarity < epsilon):
        print("Verified! Same person")
    else:
        print("Sorry, not the same person!")


# draw an image with detected objects
def draw_image_with_boxes0(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = pyplot.gca()
	# plot each box
	for result in result_list:
		# get coordinates
		x, y, width, height = result['box']
		
		cropped_img = data[y:y+height, x:x+height]
		gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
		cv2.imwrite('cropped0.jpg',gray)
		
		# create the shape
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
		break
	# show the plot
	pyplot.show()


def draw_image_with_boxes1(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = pyplot.gca()
	# plot each box
	for result in result_list:
		# get coordinates
		x, y, width, height = result['box']
		
		cropped_img = data[y:y+height, x:x+height]
		gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
		cv2.imwrite('cropped1.jpg',gray)
		
		# create the shape
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
		break
	# show the plot
	pyplot.show()

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# Finding the cosine distance to compare it to the threshold
def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

# Eucildean distance can also be used instead of cosine distance, threshold would change accordingly 
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def fun(img1, img2):
    img1_representation = model.predict(preprocess_image(img1))[0,:]
    img2_representation = model.predict(preprocess_image(img2))[0,:]
    error = 0
    for i in range(img1_representation.shape[0]):
        error += (img1_representation[i] - img2_representation[i])**2
    print(error)
    return error






if __name__ == '__main__':
    app.run(threaded=True, debug=True)
    

#print("hello", filename0)
