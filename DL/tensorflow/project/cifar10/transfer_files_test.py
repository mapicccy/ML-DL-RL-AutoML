import os
import tensorflow as tf


# change pwd
os.chdir("C:\\Users\\77321\\Desktop\\网站图片")

with tf.Session() as sess:
    filenames = ['girl.PNG', 'icon.PNG']

    # establish filename queue
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False, num_epochs=4)

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    tf.local_variables_initializer().run()
    tf.train.start_queue_runners(sess)

    i = 0
    while True:
        i += 1
        image_data = sess.run(value)
        with open('read/test_%d.PNG' % i, 'wb') as fp:
            fp.write(image_data)
