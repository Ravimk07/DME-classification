from keras import optimizers
from keras.models import Model
from keras.preprocessing import image
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils.layer_utils import print_summary
#from keras.applications.resnet50 import ResNet50

#from keras.deeplearningmodels.resnet50 import ResNet50
#from keras.deeplearningmodels.inception_v3 import InceptionV3
#from keras.deeplearningmodels.resnet152 import resnet152_model
from keras.deeplearningmodels.inception_resnet_v2 import InceptionResNetV2

from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

# from keras.applications.imagenet_utils import preprocess_input, decode_predictions

from sklearn import cross_validation

import os, sys

import h5py

import numpy as np

import matplotlib.pyplot as plt

# %matplotlib inline


# Create the model
base_model = InceptionResNetV2(include_top=False, weights=None)
model_name = 'InceptionResNetV2'

#base_model = ResNet50(include_top=False, weights=None)
#model_name = 'ResNet50'

#base_model = InceptionV3(include_top=True, weights=None)
#model_name = 'InceptionV3'

print(model_name)

# print_summary(base_model)


# If directory est
def check_dir(path):
    if os.path.exists(path):
        print("Dir already existing")

    elif sys.platform == 'win32':
        # print('OS: ', sys.platform)
        os.system('mkdir ' + path)

    else:
        os.system('mkdir -p ' + path)
        print('New Path created : ', path)
    return


print('OS: ', sys.platform)


# Set root to dataset location
root = 'C:\\Users\\admin\PycharmProjects\OCT_update\Experiment_' + model_name
os.system('mkdir ' + root)
#check_dir(root)
print('current root folder is : ', root)


# Define the routine the add a new last layer
def add_new_last_layer(base_model, nb_classes):
    print('Add last layer to the convnet..')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, activation='relu')(x)  # new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
    model = Model(inputs=base_model.input, outputs=predictions)  # Combine the network

    """
    from keras.models import Model
    >> The `Model` class adds training & evaluation routines to a `Container`.
    """
    return model


# Add the new last layer to the model
nb_classes = 2
model = add_new_last_layer(base_model, nb_classes)

learning_rate = 0.0001
decay_rate = learning_rate / 100
momentum = 0.8
# SGD = optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
SGD = optimizers.SGD(lr=learning_rate)

model.compile(optimizer=RMSprop(lr=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print_summary(model)


# Save the random weight
# model1.load_weights(path) #If you have pretrained weights
path = root + '\weight\\'
# path = '/home/deeplearningutp/PycharmProjects/OCT_Project/dataset/Cropped_BM3D/tmp/weight/InceptionResNetV2_rndm_weight .h5'
print('root: ', root)
print('model name: ', model_name)

check_dir(path)

path = path + model_name + '_rndm_weight.h5'
print('path: ', path)
model.save_weights(path)  # If you want to save random weights


# Load the data
hdf5_path = '.\Cropped_BM3D_4.hdf5'

h5f = h5py.File(hdf5_path, 'r')

input_img = h5f['img'][:]
input_labels = h5f['labels_OHE'][:]

h5f.close()


# Set Parameters of OCT-NET
batch_size = int(16)
nb_epochs = int(20)

# Train the model for each volume
path = root + '\weight\\'
check_dir(path)
path = path + model_name + '_rndm_weight.h5'
# path = '/home/deeplearningutp/PycharmProjects/OCT_Project/dataset/Cropped_BM3D/tmp/weight/InceptionResNetV2_rndm_weight .h5'

cvscores = []
bb = []
k = 0
volume = 1
nohup = 1

no_patients = 32
no_data = 4096
no_train_data = no_data - (no_data / no_patients)
no_test_data = no_data - no_train_data

cv = cross_validation.KFold(no_data, n_folds=no_patients, shuffle=False, random_state=None)

for train_index, test_index in cv:  # remaining

    if test_index[0] == k and test_index[0] < k + int(no_test_data):  # take out

        print("Training and Testing on the OCT Volume no: ", volume)

        pru = []
        aba = []

        for i in range(0, int(no_train_data)):
            at = input_img[train_index[i]]
            pru.append(at)

        pru2 = []
        aba2 = []

        for j in range(0, int(no_test_data)):
            at2 = input_img[test_index[j]]
            pru2.append(at2)

        X_tr, X_tes = np.asarray(pru), np.asarray(pru2)
        y_tr, y_tes = input_labels[train_index[0:int(no_train_data)]], input_labels[test_index[0:int(no_test_data)]]
        print(X_tr.shape, X_tes.shape, y_tr.shape, y_tes.shape)

        # create model
        model.load_weights(path)

        # Setup checkpoint, log and run the experiment
        CHKPT_PTH = root + '\checkpoint\\'
        check_dir(CHKPT_PTH)
        chkpt_pth = CHKPT_PTH + model_name + '_vol_' + str(volume) + '.hdf5'

        TB_LOG = root + '\logs_tb\\volume_' + str(volume)
        check_dir(TB_LOG)

        CSVLOG_PTH = root + '\logs_csv\\'
        check_dir(CSVLOG_PTH)
        CSV_FILENAME = CSVLOG_PTH + 'csv_logger_' + str(volume)

        csvlogger = CSVLogger(CSV_FILENAME,
                              separator=',',
                              append=False)
        """
        checkpointer = ModelCheckpoint(filepath=chkpt_pth,
                                       monitor='val_acc',
                                       verbose=1,
                                       save_best_only=True)
        """
        tensorboard = TensorBoard(log_dir=TB_LOG,
                                  histogram_freq=0,
                                  batch_size=batch_size,
                                  write_graph=True,
                                  write_grads=False,
                                  write_images=False,
                                  embeddings_freq=0,
                                  embeddings_layer_names=None,
                                  embeddings_metadata=None)

        history = model.fit([X_tr], [y_tr],
                            batch_size=batch_size,
                            nb_epoch=nb_epochs,
                            verbose=1,
                            validation_data=(X_tes, y_tes),
                            callbacks=[tensorboard, csvlogger])
        # loss_history = history_callback.history["loss"]


        # evaluate the model
        bb1 = model.predict([X_tes])
        scores = model.evaluate([X_tes], [y_tes], verbose=1)
        print('Test score:', scores[0])
        print('Test accuracy:', scores[1])
        print(history.history.keys())
        plt.figure()
        ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=4)

        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # summarize history for loss
        ax2 = plt.subplot2grid((4, 4), (1, 0), colspan=4)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        cvscores.append(scores[1] * 100)
        bb.append(bb1)

        # Next volume / patients
        volume += 1
        k += int(no_test_data)

    else:

        test_index[0]

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


