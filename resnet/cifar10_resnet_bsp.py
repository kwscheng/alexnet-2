from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
#from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import resnet_model


#from batchsizemanager import BatchSizeManager

import cifar10
import cifar10_input
from tensorflow.python.client import timeline


import psutil
import random

FLAGS = tf.app.flags.FLAGS


tf.logging.set_verbosity(tf.logging.INFO)

INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

updated_batch_size_num = 28
_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5
_WEIGHT_DECAY = 2e-4

def get_computation_time(step_stats, gs):
    tl = timeline.Timeline(step_stats)
#     [computation_time, communication_time, barrier_wait_time, processing_time] = tl.get_local_step_duration()
#     tf.logging.info('  gs: '+str(gs)+'; computation-phase1: '+str(computation_time) + '; communication-phase1: ' + str(communication_time))

#     [computation_time, communication_time, barrier_wait_time] = tl.get_local_step_duration('sync_token_q_Dequeue')
#     tf.logging.info('  gs: '+str(gs)+'; computation: '+str(computation_time) + '; communication: ' + str(communication_time) + '; barrier_wait: '+str(barrier_wait_time) + '; total processing time: '+ str(processing_time)+ '\n')
#     tf.logging.info('ccc-'+str(gs)+str(gs>10))
#    if gs == 130 or gs ==315 or gs==316:
#	tf.logging.info('ccc-start-'+str(gs))
#    	ctf = tl.generate_chrome_trace_format()
#	tf.logging.info('ccc-finish-generate-'+str(gs))
#        with open('timeline'+str(gs)+'.json', 'w') as f:
#            f.write(ctf)
#        tf.logging.info('write json')
    

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
    configP=tf.ConfigProto()
#    configP.intra_op_parallelism_threads=1
#    configP.inter_op_parallelism_threads=1

    server = tf.train.Server({'ps': ps_hosts, 'worker': worker_hosts},
                             job_name = FLAGS.job_name,
                             task_index=FLAGS.task_id,
			     config=configP)

#    batchSizeManager = BatchSizeManager(FLAGS.batch_size, len(worker_hosts))

    if FLAGS.job_name == 'ps':
#	rpcServer = batchSizeManager.create_rpc_server(ps_hosts[0].split(':')[0])
#        rpcServer.serve()
        server.join()

#    rpcClient = batchSizeManager.create_rpc_client(ps_hosts[0].split(':')[0])
    is_chief = (FLAGS.task_id == 0)
    if is_chief:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)

    device_setter = tf.train.replica_device_setter(ps_tasks=len(ps_hosts))
    with tf.device('/job:worker/task:%d' % FLAGS.task_id):
        with tf.device(device_setter):
            global_step = tf.Variable(0, trainable=False)
            decay_steps = 50000*350.0/FLAGS.batch_size
            batch_size = tf.placeholder(dtype=tf.int32, shape=(), name='batch_size')
            images, labels = cifar10.distorted_inputs(batch_size)
            val_images, val_labels = cifar10_input.get_validation_data(batch_size=batch_size/4)
#            print (str(tf.shape(images))+ str(tf.shape(labels)))
            re = tf.shape(images)[0]
            with tf.variable_scope('root', partitioner=tf.fixed_size_partitioner(len(ps_hosts), axis=0)):
                network = resnet_model.cifar10_resnet_v2_generator(FLAGS.resnet_size, _NUM_CLASSES)
            inputs = tf.reshape(images, [-1, _HEIGHT, _WIDTH, _DEPTH])
            val_inputs = tf.reshape(val_images, [-1, _HEIGHT, _WIDTH, _DEPTH])
#            labels = tf.reshape(labels, [-1, _NUM_CLASSES])
            print(labels.get_shape())
            labels = tf.one_hot(labels, 10, 1, 0)
            val_labels = tf.one_hot(val_labels, 10, 1, 0)
            print(labels.get_shape())
            logits = network(inputs, True)
            val_logits = network(val_inputs, False)
            print(logits.get_shape())
            cross_entropy = tf.losses.softmax_cross_entropy(
                logits=logits, 
                onehot_labels=labels)
            acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(labels,1),predictions=tf.argmax(logits,1))
            val_acc, val_op = tf.metrics.accuracy(labels=tf.argmax(val_labels,1),predictions=tf.argmax(val_logits,1))

#            logits = cifar10.inference(images, batch_size)

#            loss = cifar10.loss(logits, labels, batch_size)
            loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE * len(worker_hosts),
                                            global_step,
                                            decay_steps,
                                            LEARNING_RATE_DECAY_FACTOR,
                                            staircase=True)
            opt = tf.train.GradientDescentOptimizer(lr)

            # Track the moving averages of all trainable variables.
            exp_moving_averager = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
            variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())

            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=len(worker_hosts),
#                replica_id=FLAGS.task_id,
                total_num_replicas=len(worker_hosts),
                variable_averages=exp_moving_averager,
                variables_to_average=variables_to_average)

            # Compute gradients with respect to the loss.
#            grads0 = opt.compute_gradients(loss) 
#	    grads = list()
#	    for grad, var in grads0:
#		grads.append((tf.scalar_mul(tf.cast(batch_size/FLAGS.batch_size, tf.float32), grad), var))
            grads0 = opt.compute_gradients(loss) 
            grads = [(tf.scalar_mul(tf.cast(batch_size/FLAGS.batch_size, tf.float32), grad), var) for grad, var in grads0]
	    #grads = tf.map_fn(lambda x : (tf.scalar_mul(tf.cast(batch_size/FLAGS.batch_size, tf.float32), x[0]), x[1]), grads0)
	    #grads = tf.while_loop(lambda x : x, grads0)

#            grads = opt.compute_gradients(loss) 

            apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)

            with tf.control_dependencies([apply_gradients_op]):
                train_op = tf.identity(loss, name='train_op')

            chief_queue_runners = [opt.get_chief_queue_runner()]
            init_tokens_op = opt.get_init_tokens_op()

#            saver = tf.train.Saver()
            sv = tf.train.Supervisor(is_chief=is_chief,
                                     logdir=FLAGS.train_dir,
				     init_op=tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()),
                                     summary_op=None,
                                     global_step=global_step,
#                                     saver=saver,
                                     saver=None,
				     recovery_wait_secs=1,
                                     save_model_secs=60)

            tf.logging.info('%s Supervisor' % datetime.now())
            sess_config = tf.ConfigProto(allow_soft_placement=True,
					intra_op_parallelism_threads=1,
					inter_op_parallelism_threads=1,
   	                log_device_placement=FLAGS.log_device_placement)
            sess_config.gpu_options.allow_growth = True

   	    # Get a session.
            sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

            local_var_stream = [i for i in tf.local_variables()]
            print(local_var_stream)
            print("accuracy: ", sess.run(acc))
#	    sess.run(tf.global_variables_initializer())

            # Start the queue runners.
            queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
            sv.start_queue_runners(sess, queue_runners)

            sv.start_queue_runners(sess, chief_queue_runners)
            sess.run(init_tokens_op)

            """Train CIFAR-10 for a number of steps."""
#            available_cpu = psutil.cpu_percent(interval=None)

#            thread = threading2.Thread(target = local_update_batch_size, name = "update_batch_size_thread", args = (rpcClient, FLAGS.task_id,))
#            thread.start()
            
            time0 = time.time()
            batch_size_num = FLAGS.batch_size
            csv_file = open("../csv/resnet_CPU_metrics_"+str(FLAGS.task_id)+".csv","w")
            csv_file.write("time,datetime,step,global_step,loss,accuracy,val_accuracy,examples_sec,sec_batch,duration,cpu,mem,net_usage\n")
            for step in range(FLAGS.max_steps):

                start_time = time.time()

                if FLAGS.job_name == 'worker' and FLAGS.task_id == 0 :
                        current_process.cpu_affinity([random.randint(0,3)])
                elif FLAGS.job_name == 'worker' and FLAGS.task_id == 1 :
                        current_process.cpu_affinity([random.randint(0,3)])
                elif FLAGS.job_name == 'worker' and FLAGS.task_id == 2 :
                        current_process.cpu_affinity([random.randint(0,3)])
                elif FLAGS.job_name == 'worker' and FLAGS.task_id == 3 :
                        current_process.cpu_affinity([random.randint(0,3)])
                elif FLAGS.job_name == 'worker' and FLAGS.task_id == 4 :
                        current_process.cpu_affinity([random.randint(0,3)])

                current_cpu = current_process.cpu_num()

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                NETWORK_INTERFACE = 'lo'

                netio = psutil.net_io_counters(pernic=True)
                net_usage = (netio[NETWORK_INTERFACE].bytes_sent + netio[NETWORK_INTERFACE].bytes_recv)/ (1024*1024)

    #                batch_size_num = updated_batch_size_num
                if step <= 5:
                    batch_size_num = FLAGS.batch_size
                if step >= 0:
                    batch_size_num = 128
    #		    batch_size_num = 1100 + int(step/5)*10
    #		    batch_size_num = 3600 + int(step/5)*50

                num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size_num
                decay_steps_num = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

#                mgrads, images_, train_val, real, loss_value, gs = sess.run([grads, images, train_op, re, loss, global_step], feed_dict={batch_size: batch_size_num},  options=run_options, run_metadata=run_metadata)
                _, loss_value, gs = sess.run([train_op, loss, global_step], feed_dict={batch_size: batch_size_num},  options=run_options, run_metadata=run_metadata)
#                _, loss_value, gs = sess.run([train_op, loss, global_step], feed_dict={batch_size: batch_size_num}) 
                sess.run(acc_op, feed_dict={batch_size: batch_size_num})
                accuracy = sess.run(acc)
                sess.run(val_op, feed_dict={batch_size:32})
                val_accuracy = sess.run(val_acc)
                cpu_use=current_process.cpu_percent(interval=None)
                memoryUse = pid_use.memory_info()[0]/2.**20
                b = time.time()

    #    		tl = timeline.Timeline(run_metadata.step_stats)
    #		last_batch_time = tl.get_local_step_duration('sync_token_q_Dequeue')
                #thread = threading2.Thread(target=get_computation_time, name="get_computation_time",args=(run_metadata.step_stats,step,))
                #thread.start()

    #                available_cpu = 100-psutil.cpu_percent(interval=None)
    #                available_memory = psutil.virtual_memory()[1]/1000000
                c0 = time.time()

    #	        batch_size_num = rpcClient.update_batch_size(FLAGS.task_id, last_batch_time, available_cpu, available_memory, step, batch_size_num)



                if step % 1 == 0:
                    duration = time.time() - start_time
                    num_examples_per_step = batch_size_num
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

    ##                    tf.logging.info("time statistics - batch_process_time: " + str( last_batch_time)  + " - train_time: " + str(b-start_time) + " - get_batch_time: " + str(c0-b) + " - get_bs_time:  " + str(c-c0) + " - accum_time: " + str(c-time0))

                    format_str = ("time: " + str(time.time()) +
                            '; %s: step %d (global_step %d), loss = %.2f, accuracy = %.3f, val_accuracy = %.3f (%.1f examples/sec; %.3f sec/batch), duration = %.3f sec, cpu = %.3f, mem = %.3f MB, net usage= %.3f MB')
                    csv_output = (str(time.time())+',%s,%d,%d,%.2f,%.3f,%.3f,%.1f,%.3f,%.3f,%.3f,%.3f,%.3f')%(datetime.now(), step, gs, loss_value, accuracy, examples_per_sec, sec_per_batch, duration, cpu_use, memoryUse, net_usage)
                    csv_file.write(csv_output+"\n")         
                    tf.logging.info((format_str % (datetime.now(), step, gs, loss_value, accuracy, val_accuracy, examples_per_sec, sec_per_batch, duration, cpu_use, memoryUse, net_usage))+", current cpu: "+str(current_cpu))
    ##		    tf.logging.info("time: "+str(time.time()) + "; batch_size,"+str(batch_size_num)+"; last_batch_time," + str(last_batch_time) + '\n')
            csv_file.close()

def main(argv=None):
    cifar10.maybe_download_and_extract()
    train()

if __name__ == '__main__':
    tf.app.run()
