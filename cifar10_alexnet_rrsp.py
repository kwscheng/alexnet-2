from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import psutil
import random
import multiprocessing

import numpy as np
import tensorflow as tf
from nets import nets_factory

import cifar10
from tensorflow.python.client import timeline

#import threading2

from batchsizemanager import *

FLAGS = tf.app.flags.FLAGS


tf.logging.set_verbosity(tf.logging.INFO)

INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5
_WEIGHT_DECAY = 2e-4

def train():

	pid = os.getpid()
	pid_use = psutil.Process(pid)
	current_process = psutil.Process()
	
	global updated_batch_size_num
	global passed_info
	global shall_update
	ps_hosts = FLAGS.ps_hosts.split(',')
	worker_hosts = FLAGS.worker_hosts.split(',')
	print ('PS hosts are: %s' % ps_hosts)
	print ('Worker hosts are: %s' % worker_hosts)

	server = tf.train.Server({'ps': ps_hosts, 'worker': worker_hosts},
							 job_name = FLAGS.job_name,
							 task_index=FLAGS.task_id)

	batchSizeManager = BatchSizeManager(FLAGS.batch_size, len(worker_hosts))

	if FLAGS.job_name == 'ps':
		if FLAGS.task_id == 0:
			rpcServer = batchSizeManager.create_rpc_server(ps_hosts[0].split(':')[0])
			rpcServer.serve()
		server.join()

	rpcClient = batchSizeManager.create_rpc_client(ps_hosts[0].split(':')[0])
	is_chief = (FLAGS.task_id == 0)
	if is_chief:
		if tf.gfile.Exists(FLAGS.train_dir):
			tf.gfile.DeleteRecursively(FLAGS.train_dir)
		tf.gfile.MakeDirs(FLAGS.train_dir)

	device_setter = tf.train.replica_device_setter(ps_tasks=len(ps_hosts))
	with tf.device('/job:worker/task:%d' % FLAGS.task_id):
		partitioner=tf.fixed_size_partitioner(len(ps_hosts), axis=0)
		with tf.variable_scope('root', partitioner=partitioner):
			with tf.device(device_setter):
				global_step = tf.Variable(0, trainable=False)

				decay_steps = 50000*350.0/FLAGS.batch_size
				batch_size = tf.placeholder(dtype=tf.int32, shape=(), name='batch_size')
				images, labels = cifar10.distorted_inputs(batch_size,FLAGS.task_id)
				inputs = tf.reshape(images, [-1, _HEIGHT, _WIDTH, _DEPTH])
				labels = tf.one_hot(labels, 10, 1, 0)
				network_fn = nets_factory.get_network_fn('alexnet_v2',num_classes=10) 
				(logits,_) = network_fn(inputs)
				cross_entropy = tf.losses.sigmoid_cross_entropy(
				logits=logits, 
				multi_class_labels=labels)



				loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
					[tf.nn.l2_loss(v) for v in tf.trainable_variables()])

				train_op = cifar10.train(loss, global_step)

				correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)) 
				train_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

			# Decay the learning rate exponentially based on the number of steps.
				sv = tf.train.Supervisor(is_chief=is_chief,
									 logdir=FLAGS.train_dir,
					 init_op=tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()),
									 summary_op=None,
									 global_step=global_step,
									 saver=None,
					 recovery_wait_secs=1,
									 save_model_secs=60)

				tf.logging.info('%s Supervisor' % datetime.now())
				sess_config = tf.ConfigProto(allow_soft_placement=True,
											log_device_placement=FLAGS.log_device_placement)
				sess_config.gpu_options.allow_growth = True

				# Get a session.
				sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

				# Start the queue runners.
				queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
				sv.start_queue_runners(sess, queue_runners)

				"""Train CIFAR-10 for a number of steps."""
				batch_size_num = FLAGS.batch_size
				netio = psutil.net_io_counters(pernic=True)
				csv_file = open("./csv/alexnetrrsp_CPU_metrics_"+str(FLAGS.task_id)+".csv","w")
				csv_file.write("time,datetime,step,global_step,loss,accuracy,val_accuracy,examples_sec,sec_batch,duration,cpu,mem,net_usage\n")
				for step in range(FLAGS.max_steps):

					start_time = time.time()
					if FLAGS.job_name == 'worker' and FLAGS.task_id == 0 :
							current_process.cpu_affinity([random.randint(0,2)])
					elif FLAGS.job_name == 'worker' and FLAGS.task_id == 1 :
							current_process.cpu_affinity([random.randint(0,3)])
					elif FLAGS.job_name == 'worker' and FLAGS.task_id == 2 :
							current_process.cpu_affinity([random.randint(0,2)])
					elif FLAGS.job_name == 'worker' and FLAGS.task_id == 3 :
							current_process.cpu_affinity([random.randint(0,3)])
					elif FLAGS.job_name == 'worker' and FLAGS.task_id == 4 :
							current_process.cpu_affinity([random.randint(0,3)])

					run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
					run_metadata = tf.RunMetadata()
					NETWORK_INTERFACE = 'lo'

					net_usage = (netio[NETWORK_INTERFACE].bytes_sent + netio[NETWORK_INTERFACE].bytes_recv)/ (1024*1024)
					num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size_num
					decay_steps_num = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

					_, loss_value, gs = sess.run([train_op, loss, global_step], feed_dict={batch_size: batch_size_num},  options=run_options, run_metadata=run_metadata)
	#		if gs < 10:
	#	            with open('timeline.json', 'w') as f:
	#	                f.write(ctf)
	#		    tf.logging.info('write json')
					cpu_use=current_process.cpu_percent(interval=None)
					memoryUse = pid_use.memory_info()[0]/2.**20
					mem = psutil.virtual_memory()
			# call rrsp mechanism to coordinate the synchronization order and update the batch size

					train_accuracy = sess.run(train_acc, feed_dict={batch_size: batch_size_num})

					if step % 1 == 0:
						duration = time.time() - start_time
						num_examples_per_step = batch_size_num
						examples_per_sec = num_examples_per_step / duration
						sec_per_batch = float(duration)
						format_str = ("time: " + str(time.time()) +
								'; %s: step %d (global_step %d), loss = %.2f, accuracy = %.3f (%.1f examples/sec; %.3f sec/batch), duration = %.3f sec, cpu = %.3f, mem = %.3f MB, net usage= %.3f MB')
						csv_output = (str(time.time())+',%s,%d,%d,%.2f,%.3f,%.1f,%.3f,%.3f,%.3f,%.3f,%.3f')%(datetime.now(), step, gs, loss_value, train_accuracy, examples_per_sec, sec_per_batch, duration, cpu_use, memoryUse, net_usage)

						csv_file.write(csv_output+"\n")
						tf.logging.info(format_str % (datetime.now(), step, gs, loss_value, train_accuracy, examples_per_sec, sec_per_batch, duration, cpu_use, memoryUse, net_usage))
						batch_size_num = rpcClient.update_batch_size(FLAGS.task_id, int(duration),int(multiprocessing.cpu_count()),int(mem.available), step, batch_size_num) #task_index, last_batch_time, avail_cpu, avail_memory, step, batch_size
				csv_file.close()
def main(argv=None):
	cifar10.maybe_download_and_extract()
	train()

if __name__ == '__main__':
	tf.app.run()
