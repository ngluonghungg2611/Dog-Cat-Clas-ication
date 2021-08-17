from A_libraries import *
from B_FeatureImage import *
from C_PreData import *
from F_dftrain_val_test import *
path_train = "./train"
path_test = "./test1"
train_datagen = ImageDataGenerator(rotation_range=15, rescale=1./255, shear_range=0.1, zoom_range=0.2, horizontal_flip=True,
                               width_shift_range=0.1, height_shift_range=0.1)
train_generator = train_datagen.flow_from_dataframe(train_df, "./train/", x_col='filename', y_col='category',
                                                    target_size=Image_Size, class_mode='categorical',
                                                    batch_size=batch_size)

validate_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validate_datagen.flow_from_dataframe(validate_df, "./train/", x_col='filename', y_col='category',
                                                            target_size=Image_Size, class_mode='categorical',
                                                            batch_size=batch_size)

test_generator = train_datagen.flow_from_dataframe(train_df, "./test1/", x_col='filename', y_col='category',
                                                   target_size=Image_Size, class_mode='categorical', batch_size=batch_size)