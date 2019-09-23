import os
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import itertools
import threading
import time
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

file= input("H5 file name?:")
file1= input("PB file name:")
K.set_learning_phase(0)
K.clear_session()

done = False
def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rLOADING ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
t = threading.Thread(target=animate)
t.start()

model = load_model(file)

time.sleep(10)
done = True

print('\nModel.outputs:')
print(model.outputs)
print('Model.inputs:')
print(model.inputs)
frozen_graph = freeze_session(K.get_session(),output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, "model", file1, as_text=False)
print('********'+file1+' file saved in "model" folder********')