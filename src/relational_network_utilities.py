import sys
import time
import numpy as np
from global_settings import GlobalSettings
  
def print_debug(msg, is_end_line = True, is_print_console = True):
    logger = GlobalSettings.get_instance().logger
    
    if is_print_console:
        print msg,
    
    if logger != None:
        logger.write(msg+" ")
    
    if is_end_line:
        if is_print_console:
            print ''
        if logger != None:
            logger.write('\n')          
    
    if logger != None:
        logger.flush() 
    
    
def utility_train(networkrel, training_inputs, training_targets, test_inputs, test_targets):
    
    print_debug('Method Called: '+sys._getframe().f_code.co_name)
    
    global_settings = GlobalSettings.get_instance()
    
    
    num_epochs = global_settings.num_epochs
    lr_inital = global_settings.lr_inital
    lr_decay_factor = global_settings.lr_decay_factor
    decay_times = global_settings.decay_times
      
    print_debug("Start training with epoch {} initial rate  {:.9f} and decay factor {} and decay times {}".format(
        num_epochs, float(lr_inital), lr_decay_factor, decay_times))      
    
    print_debug('Inputs is {} batches, each = {}'.format(len(training_targets), training_inputs[0].shape))
    
    whole_training_start_time = time.time()
    training_losses = []
    
    train_loss = 0
    train_batches = 1
    val_loss = 0
    val_acc = 0
    val_batches = 1
    
    
    f = 1
    global_settings.lr.set_value( np.cast['float32'](lr_inital/f))
    
    for epoch in range(num_epochs):
        if epoch != 0 and epoch%lr_decay_factor == 0 and decay_times > 0:
            f = 10**(epoch//lr_decay_factor)
            global_settings.lr.set_value( np.cast['float32'](lr_inital/f))
            decay_times -= 1
            
        train_loss = 0
        train_batches = 0
        start_time = time.time()
        lr_cur = float(global_settings.lr.get_value())
        
        for idx in range(len(training_inputs)): # len = total # of batches 
            inputs = training_inputs[idx]       # (3, 4096, 12)    3 is the number of elements per batch
            targets = training_targets[idx]     # (3,) = array([7, 7, 6])
            
            if inputs.size == 0:
                continue                
            train_batches += 1            
            train_loss += networkrel.train_fn(inputs, targets)
            
        training_losses.append(train_loss / train_batches)
        if len(training_losses) > 10:
            training_losses = training_losses[1:]
            
        epoch_cond = (epoch+1 == 1 or epoch+1 == 5 or epoch+1 == num_epochs or (epoch != 0 and (epoch+1) % global_settings.test_factor == 0))
                        
        if epoch_cond:            
            # And a full pass over the validation data:
            val_loss = 0
            val_acc = 0
            val_batches = 0
            
            for idx in range(len(training_inputs)):
                inputs = training_inputs[idx]
                targets = training_targets[idx]
                err, acc, prediction = networkrel.val_fn(inputs, targets)
                val_acc += acc
                val_batches += 1
        
            print_debug("Epoch {:>3} of {}".format(epoch + 1, num_epochs), False)
            print_debug("dropout training loss: {:>10.7f}".format(train_loss / train_batches), False)
            print_debug("training loss: {:>10.7f}".format(val_loss / val_batches), False)
            print_debug("training accuracy: {:>5.1f} %".format(float(val_acc) / val_batches * 100.0), False)
            print_debug("learning rate: {:.9f}".format(lr_cur), False)
            print_debug("took {}".format(time_string(time.time() - start_time)))
            
            test_acc_val = utility_test(networkrel, test_inputs, test_targets, epoch+1 != num_epochs)
        
    print_debug("Total training time is {}\n".format(time_string(time.time() - whole_training_start_time)))



# This code handles temporal window
def utility_test(networkrel, test_inputs, test_targets = None, is_print_acc_only = False):    
    if not is_print_acc_only:
        print_debug('Method Called: '+sys._getframe().f_code.co_name)
        print_debug('Test is {} batches, Input shape = {}, Output shape = {}'.format(len(test_targets), test_inputs[0].shape, test_targets[0].shape))
    
    start_time = time.time()
    
    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    y_pred = []
    y_targets = []
    
    for idx in range(len(test_inputs)):        
        err, acc, window_prediction = networkrel.val_fn(test_inputs[idx], test_targets[idx])
        test_err += err      
        
        avg_soft_max = np.mean(window_prediction, axis=0)
        res = np.argmax(avg_soft_max)
        
        if res == test_targets[idx][0]:
            test_acc += 1
        
        test_batches += 1
        y_pred.append(np.array([res]))
        y_targets.append(np.array([test_targets[idx][0]]))
    
    acc_val = float(test_acc) / test_batches * 100.0
    msg = "\tTest loss: {:>10.7f}".format(test_err / test_batches)
    msg += "\tTest accuracy: {}/{} = {:>5.1f} %".format(int(test_acc), test_batches, acc_val)
    
    print_debug(msg)
    test_targets = y_targets

    if not is_print_acc_only:
        print_debug("Total testing time: {}\n".format(time_string(time.time() - start_time)))
        
    return acc_val

    
def time_string(time_dif):
    if time_dif < 60:
        return '{:.0f}s'.format(time_dif)
    
    if time_dif < 60*60:
        return '{:.1f}m'.format(time_dif/60.0)
    
    return '{:.2f}h'.format(time_dif/(60.0*60.0))

